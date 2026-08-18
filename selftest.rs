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
use crate::scan::{scan_chunked, scan_sequential};
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
    scan_chunked(&a, &b, &mut parallel, batch, t, d, threads);
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
        &format!("chunked vs seq {:.2e}, seq vs scalar {:.2e}", scan_error, reference_error),
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
    let pa = graph.param("pa", vec![batch * t, d], a_values, false);
    let pb = graph.param("pb", vec![batch * t, d], b_values, false);
    // Constants referenced by a reusable graph builder must be below the reset
    // watermark. The old test created this after seal_params(), so reset()
    // recycled its node id and accidentally differentiated a different graph.
    // The shape must match `scan`'s [batch * t, d] output exactly: elementwise
    // ops compare shapes, not just element counts.
    let weight = graph.constant(vec![batch * t, d], weights);
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

/// Finite-difference coverage for the tape ops the language model never builds
/// (`sub`, `scale`, `gelu`, and the NN-layout matrix product). Without this the
/// backward arms for four of the twenty-two differentiable ops are dead code.
fn test_aux_op_grads(threads: usize, rng: &mut Rng) -> bool {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let mut graph = Graph::new(threads);
    let left: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
    let right: Vec<f32> = (0..k * n).map(|_| rng.normal()).collect();
    let offset: Vec<f32> = (0..m * n).map(|_| rng.normal()).collect();
    let weights: Vec<f32> = (0..m * n).map(|_| rng.normal()).collect();
    let pa = graph.param("left", vec![m, k], left, false);
    let pb = graph.param("right", vec![k, n], right, false);
    let pc = graph.param("offset", vec![m, n], offset, false);
    let weight = graph.constant(vec![m, n], weights);
    graph.seal_params();

    let build = |g: &mut Graph| -> usize {
        let product = g.matmul_nn(pa, pb, m, k, n);
        let shifted = g.sub(product, pc);
        let scaled = g.scale(shifted, 0.75);
        let activated = g.gelu(scaled);
        let weighted = g.mul(activated, weight);
        g.sum(weighted)
    };

    graph.reset();
    graph.zero_grad();
    let loss = build(&mut graph);
    graph.backward(loss);
    let analytic: Vec<Vec<f32>> = vec![graph.grad[pa].clone(), graph.grad[pb].clone(), graph.grad[pc].clone()];

    let eps = 1e-3f32;
    let mut max_relative = 0.0f32;
    for (slot, id) in [pa, pb, pc].iter().copied().enumerate() {
        for index in 0..graph.val[id].len() {
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
            let value = analytic[slot][index];
            let denominator = value.abs().max(numeric.abs()).max(1e-2);
            max_relative = max_relative.max((value - numeric).abs() / denominator);
        }
    }
    let pass = max_relative < 2e-2;
    report("aux op gradients (sub/scale/gelu/nn)", pass, &format!("max rel err {:.2e}", max_relative))
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
    report("full model gradient", pass, &format!("{} coords, max rel err {:.2e} ({})", checks, max_relative, worst))
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

    report("BPE roundtrip + raw bytes", ok, &format!("vocab {}, ~{:.2} bytes/token", bpe.vocab_size(), ratio))
}

fn test_ckpt(threads: usize, rng: &mut Rng) -> bool {
    let cfg = tiny_cfg(17);
    let mut first = Graph::new(threads);
    let mut r1 = rng.fork();
    let _model1 = Lm::new(&mut first, &mut r1, cfg);
    first.seal_params();
    // Unique, temp-directory path: a fixed name in the working directory races
    // with a concurrent `cargo test` / `noetic selftest` and litters the repo.
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|elapsed| elapsed.as_nanos()).unwrap_or(0);
    let path_buf = std::env::temp_dir().join(format!("noetic-selftest-{}-{}.ckpt", std::process::id(), unique));
    let path = match path_buf.to_str() {
        Some(text) => text,
        None => return report("checkpoint roundtrip", false, "temporary path is not valid UTF-8"),
    };
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
    report("checkpoint + CRC guard", pass, &format!("{} tensors bit-exact, corruption detected: {}", loaded, detected))
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
    report("RNG normal + uniformity", pass, &format!("mean {:.4}, var {:.4}, chi2(9) {:.1}", mean, variance, chi_square))
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
    let targets: Vec<u32> = ids.iter().map(|&id| ((id as usize * 7 + 3) % cfg.vocab) as u32).collect();
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
    report("learns a mapping (overfit)", pass, &format!("ln(V) {:.3} -> start {:.3} -> end {:.3}", uniform_loss, first, last))
}

pub fn run_all(threads: usize, seed: u64) -> bool {
    println!("noetic selftest  (threads = {}, seed = {})", threads, seed);
    println!();
    let mut rng = Rng::new(seed);
    let mut ok = true;
    ok = test_gemm(threads, &mut rng) && ok;
    ok = test_scan_equivalence(threads, &mut rng) && ok;
    ok = test_scan_grad(threads, &mut rng) && ok;
    ok = test_aux_op_grads(threads, &mut rng) && ok;
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
    use super::*;
    use crate::autograd::{Nid, ScanPolicy};

    // One `#[test]` per check, so a regression names itself instead of hiding
    // behind a single monolithic "engine selftest" failure.
    const THREADS: usize = 2;
    const SEED: u64 = 20250816;

    fn rng() -> Rng {
        Rng::new(SEED)
    }

    #[test]
    fn gemm_matches_reference() {
        assert!(test_gemm(THREADS, &mut rng()));
    }

    #[test]
    fn scan_variants_agree() {
        assert!(test_scan_equivalence(THREADS, &mut rng()));
    }

    #[test]
    fn scan_gradient_matches_finite_differences() {
        assert!(test_scan_grad(THREADS, &mut rng()));
    }

    /// The chunked kernel is reachable through `Graph::scan` via `ScanPolicy`;
    /// forward values *and* gradients must match the sequential kernel, or the
    /// automatic policy would silently change results.
    #[test]
    fn scan_policies_agree_through_the_tape() {
        let (batch, t, d) = (2usize, 40usize, 3usize);
        let n = batch * t * d;
        let mut r = rng();
        let a_raw: Vec<f32> = (0..n).map(|_| r.uniform(0.05, 0.95)).collect();
        let b_raw: Vec<f32> = (0..n).map(|_| r.normal() * 0.5).collect();
        let weight: Vec<f32> = (0..n).map(|_| r.normal()).collect();

        let mut results = Vec::new();
        for policy in [ScanPolicy::Sequential, ScanPolicy::Chunked, ScanPolicy::Auto] {
            let mut g = Graph::new(THREADS);
            g.scan_policy = policy;
            let pa = g.param("a", vec![batch * t, d], a_raw.clone(), false);
            let pb = g.param("b", vec![batch * t, d], b_raw.clone(), false);
            g.seal_params();
            let h = g.scan(pa, pb, batch, t, d);
            let w = g.input(vec![batch * t, d], weight.clone());
            let prod = g.mul(h, w);
            let loss = g.sum(prod);
            g.zero_grad();
            g.backward(loss);
            results.push((g.val[h].clone(), g.grad[pa].clone(), g.grad[pb].clone()));
        }

        for (index, (h, ga, gb)) in results.iter().enumerate().skip(1) {
            let reference = &results[0];
            for i in 0..n {
                let tol = 2e-5;
                assert!((h[i] - reference.0[i]).abs() < tol, "policy {} forward mismatch at {}", index, i);
                assert!((ga[i] - reference.1[i]).abs() < tol, "policy {} grad a mismatch at {}", index, i);
                assert!((gb[i] - reference.2[i]).abs() < tol, "policy {} grad b mismatch at {}", index, i);
            }
        }
    }

    #[test]
    fn aux_op_gradients_match_finite_differences() {
        assert!(test_aux_op_grads(THREADS, &mut rng()));
    }

    /// Elementwise activations are perfectly conditioned for central
    /// differences, so their gradients must agree to ~1e-4 relative - not to
    /// the 3e-2 the whole-model check has to tolerate. Written after a
    /// deliberate 2% error in the SiLU derivative passed every other test in
    /// this repository, including the end-to-end verifier.
    #[test]
    fn elementwise_activation_gradients_are_tight() {
        // (name, forward through the tape)
        type Build = fn(&mut Graph, Nid) -> Nid;
        let ops: [(&str, Build); 5] = [
            ("silu", |g, x| g.silu(x)),
            ("gelu", |g, x| g.gelu(x)),
            ("sigmoid", |g, x| g.sigmoid(x)),
            ("tanh", |g, x| g.tanh(x)),
            ("one_minus", |g, x| g.one_minus(x)),
        ];
        let n = 15usize;
        // A spread that covers both saturating tails and the interesting middle.
        let base: Vec<f32> = (0..n).map(|i| -3.5 + 0.5 * (i as f32)).collect();
        let weight: Vec<f32> = (0..n).map(|i| 1.0 - 0.1 * (i as f32)).collect();

        for (name, build) in ops {
            let loss_of = |xs: &[f32]| -> f64 {
                let mut g = Graph::new(1);
                let x = g.param("x", vec![n], xs.to_vec(), false);
                g.seal_params();
                let y = build(&mut g, x);
                // Accumulate the weighted sum outside the tape in f64 so the
                // finite difference is limited by the op, not by summation.
                let mut total = 0.0f64;
                for i in 0..n {
                    total += (g.val[y][i] as f64) * (weight[i] as f64);
                }
                total
            };

            let mut g = Graph::new(1);
            let x = g.param("x", vec![n], base.clone(), false);
            g.seal_params();
            let y = build(&mut g, x);
            let w = g.input(vec![n], weight.clone());
            let prod = g.mul(y, w);
            let loss = g.sum(prod);
            g.zero_grad();
            g.backward(loss);
            let analytic = g.grad[x].clone();

            let eps = 1e-2f64;
            for i in 0..n {
                let mut plus = base.clone();
                let mut minus = base.clone();
                plus[i] = (base[i] as f64 + eps) as f32;
                minus[i] = (base[i] as f64 - eps) as f32;
                let numeric = (loss_of(&plus) - loss_of(&minus)) / (2.0 * eps);
                // Absolute, not relative: these gradients are O(0.1), so a
                // relative bound with a "+1" denominator would swallow a 3%
                // error. The measured noise floor on correct code is 2.3e-5,
                // so 2e-4 leaves an order of magnitude of headroom while still
                // catching a 2% slip in any derivative.
                let error = (numeric - analytic[i] as f64).abs();
                assert!(
                    error < 2e-4,
                    "{} gradient at x = {}: analytic {}, finite difference {}, abs err {:.2e}",
                    name,
                    base[i],
                    analytic[i],
                    numeric,
                    error
                );
            }
        }
    }

    #[test]
    fn model_gradient_matches_finite_differences() {
        assert!(test_model_grad(THREADS, &mut rng()));
    }

    #[test]
    fn streaming_decode_matches_batched_forward() {
        assert!(test_stream_matches_batch(THREADS, &mut rng()));
    }

    /// The happy-path parity check uses one shape. These are the shapes that
    /// break circular-buffer indexing: a degenerate kernel, a kernel longer
    /// than the sequence, a single-token sequence, and more than one layer.
    #[test]
    fn streaming_matches_batched_across_awkward_configs() {
        let cases: [(usize, usize, usize); 5] = [
            // (conv_k, n_layer, sequence length)
            (1, 1, 6),
            (7, 1, 3),
            (4, 3, 9),
            (4, 1, 1),
            (2, 2, 16),
        ];
        for (conv_k, n_layer, t) in cases {
            let mut r = rng();
            let cfg = LmConfig { conv_k, n_layer, ..tiny_cfg(17) };
            cfg.check().unwrap_or_else(|error| panic!("bad test config: {}", error));
            let mut graph = Graph::new(THREADS);
            let model = Lm::new(&mut graph, &mut r, cfg);
            graph.seal_params();
            let ids: Vec<u32> = (0..t).map(|_| r.below(cfg.vocab) as u32).collect();

            graph.reset();
            graph.no_grad = true;
            let logits = model.logits(&mut graph, &ids, 1, t);
            let batched = graph.val[logits].clone();
            graph.no_grad = false;

            let mut state = LmState::new(&cfg);
            let mut error = 0.0f32;
            for (i, &id) in ids.iter().enumerate() {
                let row = infer_step(&graph, &model, &mut state, id);
                for token in 0..cfg.vocab {
                    error = error.max((row[token] - batched[i * cfg.vocab + token]).abs());
                }
            }
            assert!(
                error < 2e-3,
                "streaming diverged from batched for conv_k {}, layers {}, t {}: max diff {:.3e}",
                conv_k,
                n_layer,
                t,
                error
            );
        }
    }

    /// A kernel wider than the sequence means every tap reads a clamped or
    /// zero-padded position; the backward pass must not credit those taps.
    #[test]
    fn dwconv_gradient_survives_kernel_longer_than_sequence() {
        let (batch, t, d, k) = (2usize, 3usize, 2usize, 6usize);
        let n = batch * t * d;
        let mut r = rng();
        let x_raw: Vec<f32> = (0..n).map(|_| r.normal()).collect();
        let w_raw: Vec<f32> = (0..k * d).map(|_| r.normal() * 0.5).collect();
        let b_raw: Vec<f32> = (0..d).map(|_| r.normal() * 0.1).collect();
        let weight: Vec<f32> = (0..n).map(|_| r.normal()).collect();

        let loss_of = |xs: &[f32], ws: &[f32], bs: &[f32]| -> f32 {
            let mut g = Graph::new(THREADS);
            let x = g.param("x", vec![batch * t, d], xs.to_vec(), false);
            let w = g.param("w", vec![k, d], ws.to_vec(), false);
            let bias = g.param("b", vec![d], bs.to_vec(), false);
            g.seal_params();
            let y = g.dwconv(x, w, bias, batch, t, d, k);
            let m = g.input(vec![batch * t, d], weight.clone());
            let prod = g.mul(y, m);
            let loss = g.sum(prod);
            g.scalar(loss)
        };

        let mut g = Graph::new(THREADS);
        let x = g.param("x", vec![batch * t, d], x_raw.clone(), false);
        let w = g.param("w", vec![k, d], w_raw.clone(), false);
        let bias = g.param("b", vec![d], b_raw.clone(), false);
        g.seal_params();
        let y = g.dwconv(x, w, bias, batch, t, d, k);
        let m = g.input(vec![batch * t, d], weight.clone());
        let prod = g.mul(y, m);
        let loss = g.sum(prod);
        g.zero_grad();
        g.backward(loss);
        let (gx, gw, gb) = (g.grad[x].clone(), g.grad[w].clone(), g.grad[bias].clone());

        let eps = 1e-3f32;
        let mut worst = 0.0f32;
        for i in 0..n {
            let mut plus = x_raw.clone();
            let mut minus = x_raw.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss_of(&plus, &w_raw, &b_raw) - loss_of(&minus, &w_raw, &b_raw)) / (2.0 * eps);
            worst = worst.max((fd - gx[i]).abs() / (1.0 + fd.abs()));
        }
        for i in 0..k * d {
            let mut plus = w_raw.clone();
            let mut minus = w_raw.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss_of(&x_raw, &plus, &b_raw) - loss_of(&x_raw, &minus, &b_raw)) / (2.0 * eps);
            worst = worst.max((fd - gw[i]).abs() / (1.0 + fd.abs()));
        }
        for i in 0..d {
            let mut plus = b_raw.clone();
            let mut minus = b_raw.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss_of(&x_raw, &w_raw, &plus) - loss_of(&x_raw, &w_raw, &minus)) / (2.0 * eps);
            worst = worst.max((fd - gb[i]).abs() / (1.0 + fd.abs()));
        }
        assert!(worst < 5e-3, "dwconv gradient with k > t is wrong: max rel err {:.3e}", worst);
    }

    #[test]
    fn adamw_converges() {
        assert!(test_optimizer(THREADS));
    }

    #[test]
    fn model_learns_a_fixed_mapping() {
        assert!(test_learning(THREADS, &mut rng()));
    }

    #[test]
    fn bpe_round_trips_text_and_raw_bytes() {
        assert!(test_bpe(&mut rng()));
    }

    #[test]
    fn checkpoints_round_trip_and_detect_corruption() {
        assert!(test_ckpt(THREADS, &mut rng()));
    }

    #[test]
    fn sparse_memory_recalls_from_a_noisy_cue() {
        assert!(test_sdm(&mut rng()));
    }

    #[test]
    fn rng_moments_and_uniformity_hold() {
        assert!(test_rng());
    }

    /// The samplers that nothing in the CLI path happens to call still have to
    /// be right, or they are a trap for library users.
    #[test]
    fn auxiliary_distributions_have_the_right_moments() {
        let n = 120_000usize;
        let mut r = Rng::new(0x5EED_1234);

        // Exponential(1): mean 1, variance 1, strictly positive.
        let mut mean = 0.0f64;
        let mut second = 0.0f64;
        for _ in 0..n {
            let value = r.exponential() as f64;
            assert!(value > 0.0, "exponential returned {}", value);
            mean += value;
            second += value * value;
        }
        mean /= n as f64;
        let variance = second / (n as f64) - mean * mean;
        assert!((mean - 1.0).abs() < 0.02, "exponential mean {:.4}", mean);
        assert!((variance - 1.0).abs() < 0.06, "exponential variance {:.4}", variance);

        // Gumbel(0,1): mean = Euler-Mascheroni, variance = pi^2/6.
        let mut mean = 0.0f64;
        let mut second = 0.0f64;
        for _ in 0..n {
            let value = r.gumbel() as f64;
            mean += value;
            second += value * value;
        }
        mean /= n as f64;
        let variance = second / (n as f64) - mean * mean;
        assert!((mean - 0.577_215_66).abs() < 0.03, "gumbel mean {:.4}", mean);
        assert!((variance - std::f64::consts::PI * std::f64::consts::PI / 6.0).abs() < 0.12, "gumbel variance {:.4}", variance);

        // The Gumbel-max trick must agree with softmax sampling frequencies.
        let logits = [1.0f32, 0.0, -1.0];
        let mut counts = [0usize; 3];
        let draws = 60_000usize;
        for _ in 0..draws {
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, &logit) in logits.iter().enumerate() {
                let perturbed = logit + r.gumbel();
                if perturbed > best_value {
                    best_value = perturbed;
                    best = index;
                }
            }
            counts[best] += 1;
        }
        let total: f32 = logits.iter().map(|l| l.exp()).sum();
        for (index, &logit) in logits.iter().enumerate() {
            let expected = logit.exp() / total;
            let observed = (counts[index] as f32) / (draws as f32);
            assert!(
                (observed - expected).abs() < 0.01,
                "gumbel-max frequency for {} was {:.4}, softmax says {:.4}",
                index,
                observed,
                expected
            );
        }
    }
}
