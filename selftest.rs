//! The engine's conscience.
//!
//! Non-trivial derivatives are checked against central finite differences,
//! fast paths against independent references, and streaming decoding against
//! batched execution.

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

fn test_gemm(threads: usize, rng: &mut Rng) -> bool {
    let (m, k, n) = (37usize, 29usize, 41usize);
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for value in &mut a {
        *value = rng.normal();
    }
    for value in &mut b {
        *value = rng.normal();
    }
    let mut reference = vec![0.0f32; m * n];
    gemm_nn_naive(&a, &b, &mut reference, m, k, n);

    let mut c1 = vec![0.0f32; m * n];
    gemm_nn(&a, &b, &mut c1, m, k, n, threads);

    let mut b_t = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b[i * n + j];
        }
    }
    let mut c2 = vec![0.0f32; m * n];
    gemm_nt(&a, &b_t, &mut c2, m, k, n, threads);

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
        e1 = e1.max((c1[i] - reference[i]).abs());
        e2 = e2.max((c2[i] - reference[i]).abs());
        e3 = e3.max((c3[i] - reference[i]).abs());
    }
    let pass = e1 < 2e-3 && e2 < 2e-3 && e3 < 2e-3;
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
    let mut sequential = vec![0.0f32; batch * t * d];
    let mut parallel = vec![0.0f32; batch * t * d];
    scan_sequential(&a, &b, &mut sequential, batch, t, d, threads);
    scan_log_depth(&a, &b, &mut parallel, batch, t, d, threads);
    let mut scan_error = 0.0f32;
    for i in 0..sequential.len() {
        scan_error = scan_error.max((sequential[i] - parallel[i]).abs());
    }
    let mut reference_error = 0.0f32;
    let channel = 3usize;
    let mut acc = 0.0f32;
    for i in 0..t {
        let index = i * d + channel;
        acc = a[index] * acc + b[index];
        reference_error = reference_error.max((acc - sequential[index]).abs());
    }
    let pass = scan_error < 1e-4 && reference_error < 1e-5;
    report(
        "parallel scan == recurrence",
        pass,
        &format!("log-depth vs seq {:.2e}, seq vs scalar {:.2e}", scan_error, reference_error),
    )
}

fn test_scan_grad(threads: usize, rng: &mut Rng) -> bool {
    let (batch, t, d) = (2usize, 6usize, 3usize);
    let n = batch * t * d;
    let mut graph = Graph::new(threads);
    let mut a_values = vec![0.0f32; n];
    let mut b_values = vec![0.0f32; n];
    let mut weights = vec![0.0f32; n];
    for i in 0..n {
        a_values[i] = rng.normal();
        b_values[i] = rng.normal();
        weights[i] = rng.normal();
    }
    let pa = graph.param("pa", vec![n], a_values, false);
    let pb = graph.param("pb", vec![n], b_values, false);
    // Constants referenced by a reusable graph builder must be below the reset
    // watermark. The old test created this after seal_params(), so reset()
    // recycled its node id and accidentally differentiated a different graph.
    let weight = graph.constant(vec![n], weights);
    graph.seal_params();

    let build = |g: &mut Graph| -> usize {
        let a = g.sigmoid(pa);
        let h = g.scan(a, pb, batch, t, d);
        let product = g.mul(h, weight);
        g.sum(product)
    };

    graph.reset();
    graph.zero_grad();
    let loss = build(&mut graph);
    graph.backward(loss);
    let analytic_a = graph.grad[pa].clone();
    let analytic_b = graph.grad[pb].clone();

    let eps = 1e-3f32;
    let mut max_relative = 0.0f32;
    for trial in 0..12 {
        let use_a = trial % 2 == 0;
        let id = if use_a { pa } else { pb };
        let index = rng.below(n);
        let original = graph.val[id][index];
        graph.val[id][index] = original + eps;
        graph.reset();
        let upper_id = build(&mut graph);
        let upper = graph.scalar(upper_id);
        graph.val[id][index] = original - eps;
        graph.reset();
        let lower_id = build(&mut graph);
        let lower = graph.scalar(lower_id);
        graph.val[id][index] = original;
        let numeric = (upper - lower) / (2.0 * eps);
        let analytic = if use_a { analytic_a[index] } else { analytic_b[index] };
        let denominator = analytic.abs().max(numeric.abs()).max(1e-2);
        max_relative = max_relative.max((analytic - numeric).abs() / denominator);
    }
    let pass = max_relative < 2e-2;
    report("scan gradient (finite diff)", pass, &format!("max rel err {:.2e}", max_relative))
}

fn test_model_grad(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(23);
    let mut graph = Graph::new(threads);
    let model = Lm::new(&mut graph, rng, cfg);
    graph.seal_params();
    let (batch, t) = (2usize, 5usize);
    let mut ids = vec![0u32; batch * t];
    let mut targets = vec![0u32; batch * t];
    for i in 0..ids.len() {
        ids[i] = rng.below(cfg.vocab) as u32;
        targets[i] = rng.below(cfg.vocab) as u32;
    }

    graph.reset();
    graph.zero_grad();
    let (_, loss) = model.loss(&mut graph, &ids, &targets, batch, t);
    graph.backward(loss);
    let snapshot: Vec<Vec<f32>> = graph.params.iter().map(|param| graph.grad[param.id].clone()).collect();

    let eps = 1e-3f32;
    let mut max_relative = 0.0f32;
    let mut worst = String::new();
    let checks = 48usize;
    for _ in 0..checks {
        let p = rng.below(graph.params.len());
        let id = graph.params[p].id;
        let index = rng.below(graph.val[id].len());
        let analytic = snapshot[p][index];
        let original = graph.val[id][index];
        graph.val[id][index] = original + eps;
        graph.reset();
        let (_, upper_id) = model.loss(&mut graph, &ids, &targets, batch, t);
        let upper = graph.scalar(upper_id);
        graph.val[id][index] = original - eps;
        graph.reset();
        let (_, lower_id) = model.loss(&mut graph, &ids, &targets, batch, t);
        let lower = graph.scalar(lower_id);
        graph.val[id][index] = original;
        let numeric = (upper - lower) / (2.0 * eps);
        let denominator = analytic.abs().max(numeric.abs()).max(1e-2);
        let relative = (analytic - numeric).abs() / denominator;
        if relative > max_relative {
            max_relative = relative;
            worst = graph.params[p].name.clone();
        }
    }
    let pass = max_relative < 3e-2;
    report(
        "full model gradient",
        pass,
        &format!("{} coords, max rel err {:.2e} ({})", checks, max_relative, worst),
    )
}

fn test_stream_matches_batch(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(19);
    let mut graph = Graph::new(threads);
    let model = Lm::new(&mut graph, rng, cfg);
    graph.seal_params();
    let t = 12usize;
    let mut ids = vec![0u32; t];
    for id in &mut ids {
        *id = rng.below(cfg.vocab) as u32;
    }
    graph.reset();
    graph.no_grad = true;
    let logits = model.logits(&mut graph, &ids, 1, t);
    let batched = graph.val[logits].clone();
    graph.no_grad = false;

    let mut state = LmState::new(&cfg);
    let mut error = 0.0f32;
    for i in 0..t {
        let row = infer_step(&graph, &model, &mut state, ids[i]);
        for token in 0..cfg.vocab {
            error = error.max((row[token] - batched[i * cfg.vocab + token]).abs());
        }
    }
    let pass = error < 2e-3;
    report("streaming == batched", pass, &format!("{} steps, max logit diff {:.2e}", t, error))
}

fn test_optimizer(threads: usize) -> bool {
    let mut graph = Graph::new(threads);
    let n = 8usize;
    let weights = graph.param("w", vec![n], vec![0.0f32; n], true);
    graph.seal_params();
    let target: Vec<f32> = (0..n).map(|i| (i as f32 - 3.5) * 0.3).collect();
    let mut optimizer = AdamW::new(&graph, 0.0);
    let mut last = 0.0f32;
    for _ in 0..400 {
        graph.reset();
        graph.zero_grad();
        let loss = graph.mse(weights, &target);
        graph.backward(loss);
        optimizer.step(&mut graph, 0.05);
        last = graph.scalar(loss);
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
    for sample in samples {
        let ids = bpe.encode(sample);
        ok &= bpe.decode(&ids) == sample;
        ratio = sample.len() as f32 / ids.len().max(1) as f32;
    }

    let mut arbitrary = vec![0u8; 512];
    for byte in &mut arbitrary {
        *byte = rng.next_u32() as u8;
    }
    let ids = bpe.encode_bytes(&arbitrary);
    ok &= bpe.decode_bytes(&ids).as_deref() == Some(arbitrary.as_slice());
    ok &= bpe.decode_bytes(&[bpe.vocab_size() as u32]).is_none();

    report(
        "BPE roundtrip + raw bytes",
        ok,
        &format!("vocab {}, ~{:.2} bytes/token", bpe.vocab_size(), ratio),
    )
}

fn test_ckpt(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(17);
    let mut first = Graph::new(threads);
    let mut r1 = rng.fork();
    let _model1 = Lm::new(&mut first, &mut r1, cfg);
    first.seal_params();
    let path = "noetic_selftest.ckpt";
    let meta = vec![("vocab".to_string(), cfg.vocab.to_string())];
    if ckpt::save(path, &first, &meta).is_err() {
        return report("checkpoint roundtrip", false, "save failed");
    }

    let mut second = Graph::new(threads);
    let mut r2 = Rng::new(999);
    let _model2 = Lm::new(&mut second, &mut r2, cfg);
    second.seal_params();
    let checkpoint = match ckpt::load(path) {
        Ok(value) => value,
        Err(_) => return report("checkpoint roundtrip", false, "load failed"),
    };
    let (loaded, missing, mismatch) = ckpt::apply(&mut second, &checkpoint);
    let mut error = 0.0f32;
    for p in 0..first.params.len() {
        let a = first.params[p].id;
        let b = second.params[p].id;
        for i in 0..first.val[a].len() {
            error = error.max((first.val[a][i] - second.val[b][i]).abs());
        }
    }

    let mut detected = false;
    if let Ok(mut raw) = std::fs::read(path) {
        let middle = raw.len() / 2;
        raw[middle] ^= 0xFF;
        if std::fs::write(path, &raw).is_ok() {
            detected = ckpt::load(path).is_err();
        }
    }
    let _ = std::fs::remove_file(path);
    let pass = error == 0.0 && missing == 0 && mismatch == 0 && detected;
    report(
        "checkpoint + CRC guard",
        pass,
        &format!("{} tensors bit-exact, corruption detected: {}", loaded, detected),
    )
}

fn test_sdm(rng: &mut Rng) -> bool {
    let bits = 256usize;
    // More hard locations reduce seed sensitivity now that flip_bits correctly
    // applies exactly (rather than approximately) the requested corruption.
    let n_loc = 4096usize;
    let radius = Sdm::default_radius(bits);
    let mut memory = Sdm::new(bits, n_loc, radius, 42);
    let n_patterns = 24usize;
    let mut patterns = Vec::new();
    for _ in 0..n_patterns {
        let pattern = random_bits(rng, bits);
        memory.write(&pattern, &pattern);
        patterns.push(pattern);
    }
    let mut worst = 0usize;
    let mut total = 0usize;
    let mut exact_noise = true;
    for pattern in &patterns {
        let cue = flip_bits(pattern, bits, 50, rng);
        exact_noise &= hamming(&cue, pattern) == 50;
        let output = memory.read_iterated(&cue, 4);
        let distance = hamming(&output, pattern);
        total += distance;
        worst = worst.max(distance);
    }
    let average = total as f32 / n_patterns as f32;
    let pass = exact_noise && average < 4.0;
    report(
        "SDM recall from noisy cue",
        pass,
        &format!("{} patterns, exactly 50/{} bits flipped -> avg {:.2} errors (worst {})", n_patterns, bits, average, worst),
    )
}

fn test_rng() -> bool {
    let mut rng = Rng::new(0xDEAD_BEEF);
    let n = 200_000usize;
    let mut mean = 0.0f64;
    let mut second = 0.0f64;
    for _ in 0..n {
        let value = rng.normal() as f64;
        mean += value;
        second += value * value;
    }
    mean /= n as f64;
    let variance = second / n as f64 - mean * mean;
    let mut buckets = [0usize; 10];
    for _ in 0..n {
        buckets[rng.below(10)] += 1;
    }
    let expected = (n / 10) as f64;
    let mut chi_square = 0.0f64;
    for count in buckets {
        let delta = count as f64 - expected;
        chi_square += delta * delta / expected;
    }
    let pass = mean.abs() < 0.02 && (variance - 1.0).abs() < 0.05 && chi_square < 27.9;
    report(
        "RNG normal + uniformity",
        pass,
        &format!("mean {:.4}, var {:.4}, chi2(9) {:.1}", mean, variance, chi_square),
    )
}

fn test_learning(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(32);
    let mut graph = Graph::new(threads);
    let model = Lm::new(&mut graph, rng, cfg);
    graph.seal_params();
    let (batch, t) = (2usize, 16usize);
    let mut ids = vec![0u32; batch * t];
    for id in &mut ids {
        *id = rng.below(cfg.vocab) as u32;
    }
    let targets: Vec<u32> = ids
        .iter()
        .map(|&id| ((id as usize * 7 + 3) % cfg.vocab) as u32)
        .collect();
    let mut optimizer = AdamW::new(&graph, 0.0);
    let uniform_loss = (cfg.vocab as f32).ln();
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for step in 0..120 {
        graph.reset();
        graph.zero_grad();
        let (_, loss) = model.loss(&mut graph, &ids, &targets, batch, t);
        graph.backward(loss);
        graph.clip_grad_norm(1.0);
        optimizer.step(&mut graph, 0.02);
        last = graph.scalar(loss);
        if step == 0 {
            first = last;
        }
    }
    let pass = last < 0.35 * uniform_loss && last < first;
    report(
        "learns a mapping (overfit)",
        pass,
        &format!("ln(V) {:.3} -> start {:.3} -> end {:.3}", uniform_loss, first, last),
    )
}

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

#[cfg(test)]
mod tests {
    #[test]
    fn complete_engine_selftest() {
        assert!(super::run_all(2, 20250816));
    }
}
