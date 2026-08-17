//! The engine's conscience.
//!
//! Every non-trivial derivative in this crate is verified against central
//! finite differences, every fast path is verified against a naive reference,
//! and the streaming decoder is verified against the batched forward pass.
//! A machine-learning system you cannot falsify is not engineering, so the
//! tests are a first-class subcommand rather than an afterthought.

use crate::autograd::Graph;
use crate::bpe::Bpe;
use crate::ckpt;
use crate::data;
use crate::infer::{step as infer_step, LmState};
use crate::model::{Lm, LmConfig};
use crate::optim::AdamW;
use crate::rng::Rng;
use crate::scan::{scan_log_depth, scan_sequential};
use crate::sdm::{flip_bits, hamming, random_bits, Sdm};
use crate::tensor::{gemm_nn, gemm_nn_naive, gemm_nt, gemm_tn};

fn report(name: &str, pass: bool, detail: &str) -> bool {
    let tag = if pass { "PASS" } else { "FAIL" };
    println!("  [{}] {:<30} {}", tag, name, detail);
    pass
}

fn tiny_cfg(vocab: usize) -> LmConfig {
    LmConfig { vocab, d_model: 16, n_layer: 2, expand: 2, conv_k: 3, mlp_mult: 2, eps: 1e-5, tau_max: 8.0 }
}

// ---------------------------------------------------------------------------

fn test_gemm(threads: usize, rng: &mut Rng) -> bool {
    let (m, k, n) = (37usize, 29usize, 41usize);
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..a.len() {
        a[i] = rng.normal();
    }
    for i in 0..b.len() {
        b[i] = rng.normal();
    }
    let mut c_ref = vec![0.0f32; m * n];
    gemm_nn_naive(&a, &b, &mut c_ref, m, k, n);

    let mut c1 = vec![0.0f32; m * n];
    gemm_nn(&a, &b, &mut c1, m, k, n, threads);

    // b_t[n, k] so that gemm_nt(a, b_t) == a * b
    let mut b_t = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b[i * n + j];
        }
    }
    let mut c2 = vec![0.0f32; m * n];
    gemm_nt(&a, &b_t, &mut c2, m, k, n, threads);

    // a_t[k, m] so that gemm_tn(a_t, b) == a * b
    let mut a_t = vec![0.0f32; k * m];
    for i in 0..m {
        for j in 0..k {
            a_t[j * m + i] = a[i * k + j];
        }
    }
    let mut c3 = vec![0.0f32; m * n];
    gemm_tn(&a_t, &b, &mut c3, k, m, n, threads);

    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut e3 = 0.0f32;
    for i in 0..m * n {
        let d1 = (c1[i] - c_ref[i]).abs();
        let d2 = (c2[i] - c_ref[i]).abs();
        let d3 = (c3[i] - c_ref[i]).abs();
        if d1 > e1 {
            e1 = d1;
        }
        if d2 > e2 {
            e2 = d2;
        }
        if d3 > e3 {
            e3 = d3;
        }
    }
    let tol = 2e-3f32;
    let pass = e1 < tol && e2 < tol && e3 < tol;
    report("gemm nn/nt/tn vs naive", pass, &format!("max err {:.2e} / {:.2e} / {:.2e}", e1, e2, e3))
}

fn test_scan_equivalence(threads: usize, rng: &mut Rng) -> bool {
    let (batch, t, d) = (3usize, 129usize, 7usize);
    let mut a = vec![0.0f32; batch * t * d];
    let mut b = vec![0.0f32; batch * t * d];
    for i in 0..a.len() {
        a[i] = rng.f32_unit() * 0.98 + 0.01;
        b[i] = rng.normal();
    }
    let mut h1 = vec![0.0f32; batch * t * d];
    let mut h2 = vec![0.0f32; batch * t * d];
    scan_sequential(&a, &b, &mut h1, batch, t, d, threads);
    scan_log_depth(&a, &b, &mut h2, batch, t, d, threads);
    let mut e = 0.0f32;
    for i in 0..h1.len() {
        let dd = (h1[i] - h2[i]).abs();
        if dd > e {
            e = dd;
        }
    }
    // independent scalar reference on one channel
    let mut ref_err = 0.0f32;
    let ch = 3usize;
    let mut acc = 0.0f32;
    for i in 0..t {
        let idx = (0 * t + i) * d + ch;
        acc = a[idx] * acc + b[idx];
        let dd = (acc - h1[idx]).abs();
        if dd > ref_err {
            ref_err = dd;
        }
    }
    let pass = e < 1e-4 && ref_err < 1e-5;
    report(
        "parallel scan == recurrence",
        pass,
        &format!("log-depth vs seq {:.2e}, seq vs scalar {:.2e}", e, ref_err),
    )
}

fn test_scan_grad(threads: usize, rng: &mut Rng) -> bool {
    // loss = sum(w * scan(sigmoid(pa), pb)) -- exercises the reverse-time
    // adjoint recursion in isolation.
    let (batch, t, d) = (2usize, 6usize, 3usize);
    let n = batch * t * d;
    let mut g = Graph::new(threads);
    let mut pa_v = vec![0.0f32; n];
    let mut pb_v = vec![0.0f32; n];
    let mut w_v = vec![0.0f32; n];
    for i in 0..n {
        pa_v[i] = rng.normal();
        pb_v[i] = rng.normal();
        w_v[i] = rng.normal();
    }
    let pa = g.param("pa", vec![n], pa_v, false);
    let pb = g.param("pb", vec![n], pb_v, false);
    g.seal_params();
    let w = g.constant(vec![n], w_v);

    let build = |g: &mut Graph| -> usize {
        let a = g.sigmoid(pa);
        let h = g.scan(a, pb, batch, t, d);
        let m = g.mul(h, w);
        g.sum(m)
    };

    g.reset();
    g.zero_grad();
    let loss = build(&mut g);
    g.backward(loss);
    let ga = g.grad[pa].clone();
    let gb = g.grad[pb].clone();

    let eps = 1e-3f32;
    let mut max_rel = 0.0f32;
    for trial in 0..12 {
        let use_a = trial % 2 == 0;
        let id = if use_a { pa } else { pb };
        let i = rng.below(n);
        let orig = g.val[id][i];
        g.val[id][i] = orig + eps;
        g.reset();
        let l1 = build(&mut g);
        let up = g.scalar(l1);
        g.val[id][i] = orig - eps;
        g.reset();
        let l2 = build(&mut g);
        let dn = g.scalar(l2);
        g.val[id][i] = orig;
        let num = (up - dn) / (2.0 * eps);
        let ana = if use_a { ga[i] } else { gb[i] };
        let denom = {
            let m = if ana.abs() > num.abs() { ana.abs() } else { num.abs() };
            if m > 1e-2 {
                m
            } else {
                1e-2
            }
        };
        let rel = (ana - num).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let pass = max_rel < 2e-2;
    report("scan gradient (finite diff)", pass, &format!("max rel err {:.2e}", max_rel))
}

fn test_model_grad(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(23);
    let mut g = Graph::new(threads);
    let m = Lm::new(&mut g, rng, cfg);
    g.seal_params();
    let (batch, t) = (2usize, 5usize);
    let mut ids = vec![0u32; batch * t];
    let mut tgt = vec![0u32; batch * t];
    for i in 0..ids.len() {
        ids[i] = rng.below(cfg.vocab) as u32;
        tgt[i] = rng.below(cfg.vocab) as u32;
    }

    g.reset();
    g.zero_grad();
    let (_, loss) = m.loss(&mut g, &ids, &tgt, batch, t);
    g.backward(loss);
    let mut snap: Vec<Vec<f32>> = Vec::with_capacity(g.params.len());
    for p in 0..g.params.len() {
        snap.push(g.grad[g.params[p].id].clone());
    }

    let eps = 1e-3f32;
    let mut max_rel = 0.0f32;
    let mut worst = String::new();
    let n_check = 48usize;
    for _ in 0..n_check {
        let p = rng.below(g.params.len());
        let id = g.params[p].id;
        let i = rng.below(g.val[id].len());
        let ana = snap[p][i];
        let orig = g.val[id][i];
        g.val[id][i] = orig + eps;
        g.reset();
        let (_, l1) = m.loss(&mut g, &ids, &tgt, batch, t);
        let up = g.scalar(l1);
        g.val[id][i] = orig - eps;
        g.reset();
        let (_, l2) = m.loss(&mut g, &ids, &tgt, batch, t);
        let dn = g.scalar(l2);
        g.val[id][i] = orig;
        let num = (up - dn) / (2.0 * eps);
        let denom = {
            let mm = if ana.abs() > num.abs() { ana.abs() } else { num.abs() };
            if mm > 1e-2 {
                mm
            } else {
                1e-2
            }
        };
        let rel = (ana - num).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
            worst = g.params[p].name.clone();
        }
    }
    let pass = max_rel < 3e-2;
    report(
        "full model gradient",
        pass,
        &format!("{} coords, max rel err {:.2e} ({})", n_check, max_rel, worst),
    )
}

fn test_stream_matches_batch(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(19);
    let mut g = Graph::new(threads);
    let m = Lm::new(&mut g, rng, cfg);
    g.seal_params();
    let t = 12usize;
    let mut ids = vec![0u32; t];
    for i in 0..t {
        ids[i] = rng.below(cfg.vocab) as u32;
    }
    g.reset();
    g.no_grad = true;
    let logits = m.logits(&mut g, &ids, 1, t);
    let batched = g.val[logits].clone();
    g.no_grad = false;

    let mut st = LmState::new(&cfg);
    let mut e = 0.0f32;
    for i in 0..t {
        let row = infer_step(&g, &m, &mut st, ids[i]);
        for v in 0..cfg.vocab {
            let d = (row[v] - batched[i * cfg.vocab + v]).abs();
            if d > e {
                e = d;
            }
        }
    }
    let pass = e < 2e-3;
    report("streaming == batched", pass, &format!("{} steps, max logit diff {:.2e}", t, e))
}

fn test_optimizer(threads: usize) -> bool {
    let mut g = Graph::new(threads);
    let n = 8usize;
    let w = g.param("w", vec![n], vec![0.0f32; n], true);
    g.seal_params();
    let mut target = vec![0.0f32; n];
    for i in 0..n {
        target[i] = ((i as f32) - 3.5) * 0.3;
    }
    let mut opt = AdamW::new(&g, 0.0);
    let mut last = 0.0f32;
    for _ in 0..400 {
        g.reset();
        g.zero_grad();
        let loss = g.mse(w, &target);
        g.backward(loss);
        opt.step(&mut g, 0.05);
        last = g.scalar(loss);
    }
    let pass = last < 1e-5;
    report("AdamW convergence", pass, &format!("final mse {:.3e}", last))
}

fn test_bpe(rng: &mut Rng) -> bool {
    let (corpus, _) = data::load_or_synthesize("", 40_000, 7);
    let bpe = Bpe::train(&corpus, 400, false);
    let samples = [
        "the quiet engine folds the dense lattice .",
        "memo alpha = 512 ; query alpha = 512 .",
        "unicode: h\u{e9}llo \u{3b1}\u{3b2}\u{3b3} \u{6f22}\u{5b57} \u{1f680} done",
        "12 + 34 = 46 .",
    ];
    let mut ok = true;
    let mut ratio = 0.0f32;
    for s in samples.iter() {
        let ids = bpe.encode(s);
        let back = bpe.decode(&ids);
        if back != *s {
            ok = false;
        }
        ratio = (s.len() as f32) / (ids.len().max(1) as f32);
    }
    // random byte strings must survive too
    let mut rnd = String::new();
    for _ in 0..200 {
        rnd.push((32 + rng.below(90)) as u8 as char);
    }
    if bpe.decode(&bpe.encode(&rnd)) != rnd {
        ok = false;
    }
    report(
        "BPE roundtrip + unicode",
        ok,
        &format!("vocab {}, ~{:.2} bytes/token", bpe.vocab_size(), ratio),
    )
}

fn test_ckpt(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(17);
    let mut g1 = Graph::new(threads);
    let mut r1 = rng.fork();
    let _m1 = Lm::new(&mut g1, &mut r1, cfg);
    g1.seal_params();
    let path = "noetic_selftest.ckpt";
    let meta = vec![("vocab".to_string(), format!("{}", cfg.vocab))];
    if ckpt::save(path, &g1, &meta).is_err() {
        return report("checkpoint roundtrip", false, "save failed");
    }
    let mut g2 = Graph::new(threads);
    let mut r2 = Rng::new(999);
    let _m2 = Lm::new(&mut g2, &mut r2, cfg);
    g2.seal_params();
    let ck = match ckpt::load(path) {
        Ok(c) => c,
        Err(_) => return report("checkpoint roundtrip", false, "load failed"),
    };
    let (loaded, missing, mismatch) = ckpt::apply(&mut g2, &ck);
    let mut e = 0.0f32;
    for p in 0..g1.params.len() {
        let a = g1.params[p].id;
        let b = g2.params[p].id;
        for i in 0..g1.val[a].len() {
            let d = (g1.val[a][i] - g2.val[b][i]).abs();
            if d > e {
                e = d;
            }
        }
    }
    // corruption must be detected by the trailing CRC
    let mut detected = false;
    match std::fs::read(path) {
        Ok(mut raw) => {
            let mid = raw.len() / 2;
            raw[mid] ^= 0xFF;
            if std::fs::write(path, &raw).is_ok() {
                detected = ckpt::load(path).is_err();
            }
        }
        Err(_) => {}
    }
    let _ = std::fs::remove_file(path);
    let pass = e == 0.0 && missing == 0 && mismatch == 0 && detected;
    report(
        "checkpoint + CRC guard",
        pass,
        &format!("{} tensors bit-exact, corruption detected: {}", loaded, detected),
    )
}

fn test_sdm(rng: &mut Rng) -> bool {
    let bits = 256usize;
    let n_loc = 2048usize;
    let radius = Sdm::default_radius(bits);
    let mut mem = Sdm::new(bits, n_loc, radius, 42);
    let n_pat = 24usize;
    let mut pats: Vec<Vec<u64>> = Vec::new();
    for _ in 0..n_pat {
        let p = random_bits(rng, bits);
        mem.write(&p, &p);
        pats.push(p);
    }
    let mut worst = 0usize;
    let mut total = 0usize;
    for i in 0..n_pat {
        let cue = flip_bits(&pats[i], bits, 50, rng);
        let out = mem.read_iterated(&cue, 4);
        let d = hamming(&out, &pats[i]);
        total += d;
        if d > worst {
            worst = d;
        }
    }
    let avg = (total as f32) / (n_pat as f32);
    let pass = avg < 4.0;
    report(
        "SDM recall from noisy cue",
        pass,
        &format!("{} patterns, 50/{} bits flipped -> avg {:.2} bit errors (worst {})", n_pat, bits, avg, worst),
    )
}

fn test_rng() -> bool {
    let mut rng = Rng::new(0xDEAD_BEEF);
    let n = 200_000usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for _ in 0..n {
        let x = rng.normal() as f64;
        mean += x;
        m2 += x * x;
    }
    mean /= n as f64;
    let var = m2 / (n as f64) - mean * mean;
    let mut buckets = [0usize; 10];
    for _ in 0..n {
        buckets[rng.below(10)] += 1;
    }
    let expect = (n / 10) as f64;
    let mut chi = 0.0f64;
    for i in 0..10 {
        let d = (buckets[i] as f64) - expect;
        chi += d * d / expect;
    }
    let pass = mean.abs() < 0.02 && (var - 1.0).abs() < 0.05 && chi < 27.9;
    report(
        "RNG normal + uniformity",
        pass,
        &format!("mean {:.4}, var {:.4}, chi2(9) {:.1}", mean, var, chi),
    )
}

fn test_learning(threads: usize, rng: &mut Rng) -> bool {
    // Can it actually learn? Overfit a fixed batch and require the loss to
    // drop well below the uniform-prediction entropy.
    let cfg = tiny_cfg(32);
    let mut g = Graph::new(threads);
    let m = Lm::new(&mut g, rng, cfg);
    g.seal_params();
    let (batch, t) = (2usize, 16usize);
    let mut ids = vec![0u32; batch * t];
    let mut tgt = vec![0u32; batch * t];
    for i in 0..ids.len() {
        ids[i] = rng.below(cfg.vocab) as u32;
    }
    for i in 0..ids.len() {
        // deterministic function of the input: target = (id * 7 + 3) mod vocab
        tgt[i] = (((ids[i] as usize) * 7 + 3) % cfg.vocab) as u32;
    }
    let mut opt = AdamW::new(&g, 0.0);
    let start_ln = (cfg.vocab as f32).ln();
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for s in 0..120 {
        g.reset();
        g.zero_grad();
        let (_, loss) = m.loss(&mut g, &ids, &tgt, batch, t);
        g.backward(loss);
        g.clip_grad_norm(1.0);
        opt.step(&mut g, 0.02);
        last = g.scalar(loss);
        if s == 0 {
            first = last;
        }
    }
    let pass = last < 0.35 * start_ln && last < first;
    report(
        "learns a mapping (overfit)",
        pass,
        &format!("ln(V) {:.3} -> start {:.3} -> end {:.3}", start_ln, first, last),
    )
}

// ---------------------------------------------------------------------------

pub fn run_all(threads: usize, seed: u64) -> bool {
    println!("noetic selftest  (threads = {}, seed = {})", threads, seed);
    println!();
    let mut rng = Rng::new(seed);
    let mut ok = true;
    ok = test_gemm(threads, &mut rng) && ok;
    ok = test_scan_equivalence(threads, &mut rng) && ok;
    ok = test_scan_grad(threads, &mut rng) && ok;
    ok = test_model_grad(threads, &mut rng) && ok;
    ok = test_stream_matches_batch(threads, &mut rng) && ok;
    ok = test_optimizer(threads) && ok;
    ok = test_learning(threads, &mut rng) && ok;
    ok = test_bpe(&mut rng) && ok;
    ok = test_ckpt(threads, &mut rng) && ok;
    ok = test_sdm(&mut rng) && ok;
    ok = test_rng() && ok;
    println!();
    if ok {
        println!("all checks passed");
    } else {
        println!("FAILURES PRESENT");
    }
    ok
}
