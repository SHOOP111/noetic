//! Command-line surface: argument parsing and the `selftest` / `bench` / `bpe`
//! / `train` / `gen` / `plan` / `mem` subcommands.

use crate::{ckpt, data, plan, scan, selftest, tensor};

use crate::autograd::Graph;
use crate::bpe::Bpe;
use crate::data::Batcher;
use crate::infer::{sample_token, Decoder, SampleCfg};
use crate::model::{Lm, LmConfig};
use crate::optim::{AdamW, Schedule};
use crate::plan::{action_name, Puzzle, PvNet, N_ACT};
use crate::rng::Rng;
use crate::sdm::{flip_bits, hamming, random_bits, Projection, Sdm};
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------
// argument parsing (hand-rolled: no clap)
// ---------------------------------------------------------------------------

struct Args {
    cmd: String,
    map: HashMap<String, String>,
    pos: Vec<String>,
}

fn invalid_argument(flag: &str, value: &str, expected: &str) -> ! {
    eprintln!("invalid value for --{}: {:?} (expected {})", flag, value, expected);
    std::process::exit(2);
}

impl Args {
    fn parse() -> Args {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut cmd = String::new();
        let mut map: HashMap<String, String> = HashMap::new();
        let mut pos: Vec<String> = Vec::new();
        let mut i = 0usize;
        while i < raw.len() {
            let a = raw[i].clone();
            if let Some(body) = a.strip_prefix("--") {
                let body = body.to_string();
                match body.find('=') {
                    Some(eq) => {
                        let k = body[..eq].to_string();
                        let v = body[eq + 1..].to_string();
                        map.insert(k, v);
                    }
                    None => {
                        if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                            map.insert(body, raw[i + 1].clone());
                            i += 1;
                        } else {
                            map.insert(body, "true".to_string());
                        }
                    }
                }
            } else if cmd.is_empty() {
                cmd = a;
            } else {
                pos.push(a);
            }
            i += 1;
        }
        Args { cmd, map, pos }
    }

    fn get_str(&self, k: &str, d: &str) -> String {
        match self.map.get(k) {
            Some(v) => v.clone(),
            None => d.to_string(),
        }
    }

    /// `--flag value` first, then the first bare positional argument, then the
    /// default: `noetic gen "a prompt"` beats retyping `--prompt`.
    fn get_str_or_positional(&self, k: &str, d: &str) -> String {
        match self.map.get(k) {
            Some(v) => v.clone(),
            None => match self.pos.first() {
                Some(v) => v.clone(),
                None => d.to_string(),
            },
        }
    }
    fn get_usize(&self, k: &str, d: usize) -> usize {
        match self.map.get(k) {
            Some(v) => match v.parse::<usize>() {
                Ok(x) => x,
                Err(_) => invalid_argument(k, v, "a non-negative integer"),
            },
            None => d,
        }
    }
    fn get_f32(&self, k: &str, d: f32) -> f32 {
        match self.map.get(k) {
            Some(v) => match v.parse::<f32>() {
                Ok(x) if x.is_finite() => x,
                Err(_) => invalid_argument(k, v, "a finite number"),
                Ok(_) => invalid_argument(k, v, "a finite number"),
            },
            None => d,
        }
    }
    fn get_bool(&self, k: &str, d: bool) -> bool {
        match self.map.get(k) {
            Some(v) => {
                let s = v.to_lowercase();
                match s.as_str() {
                    "true" | "1" | "yes" => true,
                    "false" | "0" | "no" => false,
                    _ => invalid_argument(k, v, "true/false, yes/no, or 1/0"),
                }
            }
            None => d,
        }
    }
}

fn threads_of(a: &Args) -> usize {
    let threads = a.get_usize("threads", tensor::n_threads_default());
    if threads == 0 {
        invalid_argument("threads", "0", "an integer greater than zero");
    }
    threads
}

fn require_positive(flag: &str, value: usize) -> usize {
    if value == 0 {
        invalid_argument(flag, "0", "an integer greater than zero");
    }
    value
}

fn require_nonnegative_finite(flag: &str, value: f32) -> f32 {
    if !value.is_finite() || value < 0.0 {
        invalid_argument(flag, &format!("{}", value), "a finite number greater than or equal to zero");
    }
    value
}

fn require_positive_finite(flag: &str, value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        invalid_argument(flag, &format!("{}", value), "a finite number greater than zero");
    }
    value
}

fn fmt_int(n: usize) -> String {
    let s = format!("{}", n);
    let b = s.as_bytes();
    let mut out = String::new();
    for i in 0..b.len() {
        if i > 0 && (b.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(b[i] as char);
    }
    out
}

fn help() {
    println!("noetic - pure-Rust AI engine (no dependencies, no transformer)");
    println!();
    println!("USAGE: noetic <command> [--flag value]");
    println!();
    println!("  selftest              gradient checks, kernel oracles, end-to-end checks");
    println!("  bench                 GEMM / scan / train-step / decode throughput");
    println!("  bpe                   train the byte-level BPE tokenizer and inspect it");
    println!("  train                 train the recurrent language model");
    println!("  gen                   stream text from a checkpoint");
    println!("  plan                  self-play MCTS reasoning on the 8-puzzle");
    println!("  mem                   sparse distributed memory demo");
    println!("  help                  this message");
    println!();
    println!("COMMON FLAGS");
    println!("  --threads N           worker threads (default: available parallelism)");
    println!("  --seed N              deterministic seed");
    println!();
    println!("train: --data PATH --bytes N --vocab N --tok PATH --out PATH --steps N");
    println!("       --batch N --ctx N --d N --layers N --expand N --convk N --mlp N");
    println!("       --lr F --wd F --clip F --taumax F --log N --retok");
    println!("       --resume PATH --valevery N --valbatches N");
    println!("gen:   --ckpt PATH --tok PATH --prompt TEXT --n N --temp F --topk N");
    println!("       --topp F --rep F --greedy");
    println!("plan:  --iters N --games N --sims N --scramble N --hidden N --lr F");
    println!("       --batch N --epochs N --replay N --gate N --eval N --maxsteps N");
    println!("mem:   --bits N --loc N --patterns N --noise N");
    println!();
    println!("EXAMPLES");
    println!("  cargo run --release -- selftest");
    println!("  cargo run --release -- train --steps 400 --d 128 --layers 2");
    println!("  cargo run --release -- train --resume noetic.ckpt --steps 400");
    println!("  cargo run --release -- gen --prompt \"memo alpha = \" --n 160");
    println!("  cargo run --release -- plan");
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

fn cmd_bench(a: &Args) {
    let threads = threads_of(a);
    let seed = a.get_usize("seed", 7) as u64;
    let mut rng = Rng::new(seed);
    println!("noetic bench  (threads = {})", threads);
    println!();

    // ---- GEMM ----
    let n = require_positive("n", a.get_usize("n", 256));
    let matrix_elements =
        n.checked_mul(n).unwrap_or_else(|| invalid_argument("n", &format!("{}", n), "a matrix width whose square fits in memory"));
    let mut x = vec![0.0f32; matrix_elements];
    let mut y = vec![0.0f32; matrix_elements];
    for i in 0..x.len() {
        x[i] = rng.normal();
        y[i] = rng.normal();
    }
    let mut z = vec![0.0f32; matrix_elements];
    let reps = require_positive("reps", a.get_usize("reps", 8));
    let t0 = Instant::now();
    for _ in 0..reps {
        tensor::gemm_nn(&x, &y, &mut z, n, n, n, threads);
    }
    let dt = t0.elapsed().as_secs_f64();
    let flops = 2.0 * (n as f64) * (n as f64) * (n as f64) * (reps as f64);
    println!("  gemm {}x{}x{}      {:>8.2} ms/call   {:>6.2} GFLOP/s", n, n, n, 1000.0 * dt / (reps as f64), flops / dt / 1e9);

    // ---- scan ----
    let (b, t, d) = (8usize, 512usize, 256usize);
    let mut sa = vec![0.0f32; b * t * d];
    let mut sb = vec![0.0f32; b * t * d];
    for i in 0..sa.len() {
        sa[i] = 0.5 + 0.4 * rng.f32_unit();
        sb[i] = rng.normal();
    }
    let mut h = vec![0.0f32; b * t * d];
    let t1 = Instant::now();
    for _ in 0..reps {
        scan::scan_sequential(&sa, &sb, &mut h, b, t, d, threads);
    }
    let d1 = t1.elapsed().as_secs_f64() / (reps as f64);
    let t2 = Instant::now();
    for _ in 0..reps {
        scan::scan_chunked(&sa, &sb, &mut h, b, t, d, threads);
    }
    let d2 = t2.elapsed().as_secs_f64() / (reps as f64);
    let elems = (b * t * d) as f64;
    println!("  scan seq  B{} T{} D{}  {:>8.2} ms      {:>6.1} M elem/s", b, t, d, 1000.0 * d1, elems / d1 / 1e6);
    println!("  scan time-chunked     {:>8.2} ms      {:>6.1} M elem/s", 1000.0 * d2, elems / d2 / 1e6);

    // Narrow batch is the case the time-chunked kernel exists for: with one
    // sequence there is nothing to spread across cores on the batch axis.
    let (nb, nt_len) = (1usize, 8192usize);
    let narrow = nb * nt_len * d;
    let mut na = vec![0.0f32; narrow];
    let mut nbv = vec![0.0f32; narrow];
    for i in 0..narrow {
        na[i] = 0.5 + 0.4 * rng.f32_unit();
        nbv[i] = rng.normal();
    }
    let mut nh = vec![0.0f32; narrow];
    let t5 = Instant::now();
    for _ in 0..reps {
        scan::scan_sequential(&na, &nbv, &mut nh, nb, nt_len, d, threads);
    }
    let d5 = t5.elapsed().as_secs_f64() / (reps as f64);
    let t6 = Instant::now();
    for _ in 0..reps {
        scan::scan_chunked(&na, &nbv, &mut nh, nb, nt_len, d, threads);
    }
    let d6 = t6.elapsed().as_secs_f64() / (reps as f64);
    println!(
        "  scan B{} T{} D{}: seq {:.2} ms vs chunked {:.2} ms  ({:.2}x)",
        nb,
        nt_len,
        d,
        1000.0 * d5,
        1000.0 * d6,
        d5 / d6.max(1e-12)
    );

    // Same comparison with a cache-resident working set, where the kernels are
    // not pinned to main-memory bandwidth.
    let (cb, ct, cd) = (1usize, 2048usize, 64usize);
    let cache_len = cb * ct * cd;
    let mut ca = vec![0.0f32; cache_len];
    let mut cbv = vec![0.0f32; cache_len];
    for i in 0..cache_len {
        ca[i] = 0.5 + 0.4 * rng.f32_unit();
        cbv[i] = rng.normal();
    }
    let mut ch = vec![0.0f32; cache_len];
    let inner_reps = 64usize;
    let t7 = Instant::now();
    for _ in 0..inner_reps {
        scan::scan_sequential(&ca, &cbv, &mut ch, cb, ct, cd, threads);
    }
    let d7 = t7.elapsed().as_secs_f64() / (inner_reps as f64);
    let t8 = Instant::now();
    for _ in 0..inner_reps {
        scan::scan_chunked(&ca, &cbv, &mut ch, cb, ct, cd, threads);
    }
    let d8 = t8.elapsed().as_secs_f64() / (inner_reps as f64);
    println!(
        "  scan B{} T{} D{}:  seq {:.3} ms vs chunked {:.3} ms  ({:.2}x)",
        cb,
        ct,
        cd,
        1000.0 * d7,
        1000.0 * d8,
        d7 / d8.max(1e-12)
    );

    // ---- training step ----
    let vocab = 512usize;
    let cfg = LmConfig {
        vocab,
        d_model: require_positive("d", a.get_usize("d", 128)),
        n_layer: require_positive("layers", a.get_usize("layers", 2)),
        expand: require_positive("expand", a.get_usize("expand", 2)),
        conv_k: 4,
        mlp_mult: 3,
        eps: 1e-5,
        tau_max: 128.0,
    };
    if let Err(message) = cfg.check() {
        eprintln!("invalid benchmark model configuration: {}", message);
        return;
    }
    let batch = require_positive("batch", a.get_usize("batch", 8));
    let ctx = require_positive("ctx", a.get_usize("ctx", 64));
    let mut g = Graph::new(threads);
    let m = Lm::new(&mut g, &mut rng, cfg);
    g.seal_params();
    let mut opt = AdamW::new(&g, 0.01);
    let train_elements = batch
        .checked_mul(ctx)
        .unwrap_or_else(|| invalid_argument("batch/ctx", "overflow", "dimensions whose product fits in memory"));
    let mut ids = vec![0u32; train_elements];
    let mut tgt = vec![0u32; train_elements];
    for i in 0..ids.len() {
        ids[i] = rng.below(vocab) as u32;
        tgt[i] = rng.below(vocab) as u32;
    }
    let steps = require_positive("steps", a.get_usize("steps", 5));
    let t3 = Instant::now();
    let mut nodes = 0usize;
    for _ in 0..steps {
        g.reset();
        g.zero_grad();
        let (_, loss) = m.loss(&mut g, &ids, &tgt, batch, ctx);
        g.backward(loss);
        opt.step(&mut g, 1e-4);
        nodes = g.nodes();
    }
    let d3 = t3.elapsed().as_secs_f64() / (steps as f64);
    println!();
    println!("  model  {} params, {} tape nodes/step", fmt_int(g.param_count()), fmt_int(nodes));
    println!("  train step B{} T{}     {:>8.1} ms      {:>6.0} tokens/s", batch, ctx, 1000.0 * d3, (batch * ctx) as f64 / d3);

    // ---- streaming decode ----
    let mut decoder = Decoder::new(&cfg);
    let warm = decoder.step(&g, &m, 1);
    let _ = warm.len();
    let n_dec = require_positive("decode", a.get_usize("decode", 64));
    let t4 = Instant::now();
    for i in 0..n_dec {
        let _ = decoder.step(&g, &m, (i % vocab) as u32);
    }
    let d4 = t4.elapsed().as_secs_f64();
    println!("  decode (O(1) state)   {:>8.2} ms/token {:>6.0} tokens/s", 1000.0 * d4 / (n_dec as f64), (n_dec as f64) / d4);
    println!();
    println!("  note: decode state is constant-size; cost per token does not grow");
    println!("        with context length (no KV cache exists in this design).");
}

// ---------------------------------------------------------------------------
// bpe
// ---------------------------------------------------------------------------

fn cmd_bpe(a: &Args) {
    let seed = a.get_usize("seed", 1) as u64;
    let path = a.get_str("data", "");
    let bytes = require_positive("bytes", a.get_usize("bytes", 400_000));
    let (text, synth) = data::load_or_synthesize(&path, bytes, seed);
    let vocab = a.get_usize("vocab", 512);
    if !(256..=1_000_256).contains(&vocab) {
        invalid_argument("vocab", &format!("{}", vocab), "an integer from 256 through 1000256");
    }
    println!("corpus: {} bytes ({})", fmt_int(text.len()), if synth { "synthetic" } else { path.as_str() });
    let t0 = Instant::now();
    let b = Bpe::train(&text, vocab, true);
    let dt = t0.elapsed().as_secs_f64();
    let ids = b.encode(&text);
    println!();
    println!("  vocab size      {}", b.vocab_size());
    println!("  merges learned  {}", b.merges.len());
    println!("  train time      {:.2} s", dt);
    println!(
        "  compression     {:.3} bytes/token  ({} -> {} tokens)",
        (text.len() as f32) / (ids.len().max(1) as f32),
        fmt_int(text.len()),
        fmt_int(ids.len())
    );
    let roundtrip = b.decode(&ids) == text;
    println!("  lossless        {}", roundtrip);
    println!();
    println!("  longest learned tokens:");
    let mut order: Vec<usize> = (256..b.token_bytes.len()).collect();
    order.sort_by(|x, y| b.token_bytes[*y].len().cmp(&b.token_bytes[*x].len()));
    let show = if order.len() < 12 { order.len() } else { 12 };
    for i in 0..show {
        let t = order[i];
        let s = String::from_utf8_lossy(&b.token_bytes[t]).to_string();
        println!("    {:>5}  {:?}", t, s);
    }
    let out = a.get_str("out", "noetic.tok");
    match b.save(&out) {
        Ok(_) => println!("\n  saved tokenizer -> {}", out),
        Err(e) => println!("\n  save failed: {}", e),
    }
    let sample = "the recursive engine remembers the hidden lattice .";
    let sids = b.encode(sample);
    print!("  tokenization of {:?}:\n    ", sample);
    for i in 0..sids.len() {
        print!("[{}]", String::from_utf8_lossy(&b.token_bytes[sids[i] as usize]));
    }
    println!();
}

// ---------------------------------------------------------------------------
// generation helper
// ---------------------------------------------------------------------------

fn flush_utf8_pending<W: Write>(writer: &mut W, pending: &mut Vec<u8>, final_chunk: bool) -> std::io::Result<()> {
    loop {
        match std::str::from_utf8(pending) {
            Ok(_) => {
                writer.write_all(pending)?;
                pending.clear();
                return Ok(());
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    writer.write_all(&pending[..valid])?;
                }
                match error.error_len() {
                    Some(invalid) => {
                        writer.write_all("�".as_bytes())?;
                        pending.drain(..valid + invalid);
                    }
                    None => {
                        pending.drain(..valid);
                        if final_chunk && !pending.is_empty() {
                            writer.write_all(String::from_utf8_lossy(pending).as_bytes())?;
                            pending.clear();
                        }
                        return Ok(());
                    }
                }
            }
        }
    }
}

fn generate(g: &Graph, m: &Lm, b: &Bpe, prompt: &str, n: usize, scfg: &SampleCfg, rng: &mut Rng, stream: bool) -> String {
    let cfg = m.cfg;
    assert_eq!(b.vocab_size(), cfg.vocab, "tokenizer vocabulary does not match the model");
    let mut decoder = Decoder::new(&cfg);
    let mut ids = b.encode(prompt);
    if ids.is_empty() {
        ids.push(10); // newline byte token
    }
    for &id in &ids {
        let _ = decoder.step(g, m, id);
    }

    // Sampling only consults the repetition window, so keep exactly that much
    // history rather than retaining an ever-growing prompt/output sequence.
    let history_limit = scfg.rep_window;
    let history_start = ids.len().saturating_sub(history_limit);
    let mut history = ids[history_start..].to_vec();
    let mut out_ids: Vec<u32> = Vec::with_capacity(n);
    let mut pending: Vec<u8> = Vec::new();
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();
    for generated in 0..n {
        let token = sample_token(decoder.logits_mut(), scfg, &history, rng);
        out_ids.push(token);
        if history_limit > 0 {
            if history.len() < history_limit {
                history.push(token);
            } else {
                history.rotate_left(1);
                history[history_limit - 1] = token;
            }
        }
        if stream {
            let token_bytes = &b.token_bytes[token as usize];
            pending.extend_from_slice(token_bytes);
            let _ = flush_utf8_pending(&mut writer, &mut pending, false);
            let _ = writer.flush();
        }
        if generated + 1 < n {
            let _ = decoder.step(g, m, token);
        }
    }
    if stream {
        let _ = flush_utf8_pending(&mut writer, &mut pending, true);
        let _ = writeln!(writer);
        let _ = writer.flush();
    }
    b.decode(&out_ids)
}

/// CLI sampling flags, defaulting to `SampleCfg::default_cfg()` so the library
/// and the command line cannot drift apart.
fn sample_cfg_from(a: &Args) -> SampleCfg {
    let d = SampleCfg::default_cfg();
    SampleCfg {
        temperature: a.get_f32("temp", d.temperature),
        top_k: a.get_usize("topk", d.top_k),
        top_p: a.get_f32("topp", d.top_p),
        rep_penalty: a.get_f32("rep", d.rep_penalty),
        rep_window: a.get_usize("repwin", d.rep_window),
        greedy: a.get_bool("greedy", d.greedy),
    }
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

/// Everything numeric the trainer needs, validated *before* any corpus is read
/// or any tokenizer is trained: a typo in `--steps` used to surface only after
/// a full BPE run had already completed.
struct TrainOptions {
    threads: usize,
    seed: u64,
    bytes: usize,
    vocab_target: usize,
    steps: usize,
    batch: usize,
    ctx: usize,
    d_model: usize,
    n_layer: usize,
    expand: usize,
    conv_k: usize,
    mlp_mult: usize,
    tau_max: f32,
    lr: f32,
    wd: f32,
    clip: f32,
    log_every: usize,
    val_every: usize,
    val_batches: usize,
}

impl TrainOptions {
    fn parse(a: &Args) -> TrainOptions {
        let steps = require_positive("steps", a.get_usize("steps", 300));
        let vocab_target = a.get_usize("vocab", 512);
        if !(256..=1_000_256).contains(&vocab_target) {
            invalid_argument("vocab", &format!("{}", vocab_target), "an integer from 256 through 1000256");
        }
        let tau_max = require_positive_finite("taumax", a.get_f32("taumax", 128.0));
        if tau_max < 1.0 {
            invalid_argument("taumax", &format!("{}", tau_max), "a finite number greater than or equal to one");
        }
        TrainOptions {
            threads: threads_of(a),
            seed: a.get_usize("seed", 1234) as u64,
            bytes: require_positive("bytes", a.get_usize("bytes", 400_000)),
            vocab_target,
            steps,
            batch: require_positive("batch", a.get_usize("batch", 8)),
            ctx: require_positive("ctx", a.get_usize("ctx", 64)),
            d_model: require_positive("d", a.get_usize("d", 128)),
            n_layer: require_positive("layers", a.get_usize("layers", 2)),
            expand: require_positive("expand", a.get_usize("expand", 2)),
            conv_k: require_positive("convk", a.get_usize("convk", 4)),
            mlp_mult: require_positive("mlp", a.get_usize("mlp", 3)),
            tau_max,
            lr: require_nonnegative_finite("lr", a.get_f32("lr", 3e-3)),
            wd: require_nonnegative_finite("wd", a.get_f32("wd", 0.01)),
            clip: require_nonnegative_finite("clip", a.get_f32("clip", 1.0)),
            log_every: a.get_usize("log", if steps > 40 { steps / 20 } else { 1 }).max(1),
            val_every: a.get_usize("valevery", if steps >= 50 { steps / 5 } else { steps }).max(1),
            val_batches: require_positive("valbatches", a.get_usize("valbatches", 8)),
        }
    }
}

/// Mean held-out loss over `batches` random crops, evaluated without a tape.
fn evaluate(g: &mut Graph, m: &Lm, batcher: &Batcher, batch: usize, ctx: usize, batches: usize, rng: &mut Rng) -> f32 {
    let previous = g.no_grad;
    g.no_grad = true;
    let mut total = 0.0f32;
    for _ in 0..batches {
        g.reset();
        let (x, y) = batcher.val_batch(batch, ctx, rng);
        let (_, loss) = m.loss(g, &x, &y, batch, ctx);
        total += g.scalar(loss);
    }
    g.no_grad = previous;
    g.reset();
    total / (batches.max(1) as f32)
}

fn config_from_meta(ck: &ckpt::Ckpt) -> std::io::Result<LmConfig> {
    Ok(LmConfig {
        vocab: ck.require_usize("vocab")?,
        d_model: ck.require_usize("d_model")?,
        n_layer: ck.require_usize("n_layer")?,
        expand: ck.require_usize("expand")?,
        conv_k: ck.require_usize("conv_k")?,
        mlp_mult: ck.require_usize("mlp_mult")?,
        eps: ck.optional_f32("eps", 1e-5)?,
        tau_max: ck.require_f32("tau_max")?,
    })
}

fn checkpoint_meta(cfg: &LmConfig, tok_path: &str, val_loss: f32, steps: usize, opt_steps: u64) -> Vec<(String, String)> {
    vec![
        ("vocab".to_string(), format!("{}", cfg.vocab)),
        ("d_model".to_string(), format!("{}", cfg.d_model)),
        ("n_layer".to_string(), format!("{}", cfg.n_layer)),
        ("expand".to_string(), format!("{}", cfg.expand)),
        ("conv_k".to_string(), format!("{}", cfg.conv_k)),
        ("mlp_mult".to_string(), format!("{}", cfg.mlp_mult)),
        ("eps".to_string(), format!("{}", cfg.eps)),
        ("tau_max".to_string(), format!("{}", cfg.tau_max)),
        ("tok".to_string(), tok_path.to_string()),
        ("val_loss".to_string(), format!("{}", val_loss)),
        ("steps".to_string(), format!("{}", steps)),
        ("opt_steps".to_string(), format!("{}", opt_steps)),
    ]
}

fn cmd_train(a: &Args) {
    let opt = TrainOptions::parse(a);
    let resume_path = a.get_str("resume", "");
    let out = a.get_str("out", "noetic.ckpt");
    let explicit_tok = a.get_str("tok", "");
    let prompt = a.get_str("prompt", "memo alpha = ");
    let scfg = sample_cfg_from(a);
    let sample_tokens = a.get_usize("n", 160);
    let mut rng = Rng::new(opt.seed);

    // ---- resume metadata, before spending time on data or tokenizers ----
    let resume = if resume_path.is_empty() {
        None
    } else {
        match ckpt::load(&resume_path) {
            Ok(loaded) => match config_from_meta(&loaded) {
                Ok(config) => {
                    println!(
                        "resume: {} (vocab {}, d {}, layers {}, {} steps done)",
                        resume_path,
                        config.vocab,
                        config.d_model,
                        config.n_layer,
                        loaded.meta_usize("steps", 0)
                    );
                    Some((loaded, config))
                }
                Err(error) => {
                    eprintln!("cannot resume from {}: {}", resume_path, error);
                    return;
                }
            },
            Err(error) => {
                eprintln!("cannot resume from {}: {}", resume_path, error);
                return;
            }
        }
    };

    let data_path = a.get_str("data", "");
    let (text, synth) = data::load_or_synthesize(&data_path, opt.bytes, opt.seed);
    println!("corpus: {} bytes ({})", fmt_int(text.len()), if synth { "synthetic" } else { data_path.as_str() });

    // ---- tokenizer: a resumed run must keep the one it was trained with ----
    let tok_path = match resume.as_ref().and_then(|(loaded, _)| loaded.meta.get("tok").cloned()) {
        Some(recorded) if explicit_tok.is_empty() => recorded,
        _ => {
            if explicit_tok.is_empty() {
                "noetic.tok".to_string()
            } else {
                explicit_tok
            }
        }
    };
    let reuse = std::path::Path::new(&tok_path).exists() && !a.get_bool("retok", false);
    let b = if reuse {
        match Bpe::load(&tok_path) {
            Ok(t) => {
                println!("tokenizer: loaded {} (vocab {})", tok_path, t.vocab_size());
                t
            }
            Err(e) => {
                if resume.is_some() {
                    eprintln!("cannot resume: tokenizer {} failed to load ({})", tok_path, e);
                    return;
                }
                println!("tokenizer: load failed ({}), retraining", e);
                let t = Bpe::train(&text, opt.vocab_target, true);
                let _ = t.save(&tok_path);
                t
            }
        }
    } else if resume.is_some() {
        eprintln!("cannot resume: tokenizer {} is missing", tok_path);
        return;
    } else {
        println!("tokenizer: training BPE to vocab {}", opt.vocab_target);
        let t = Bpe::train(&text, opt.vocab_target, true);
        match t.save(&tok_path) {
            Ok(_) => println!("tokenizer: saved {}", tok_path),
            Err(e) => println!("tokenizer: save failed ({})", e),
        }
        t
    };
    let tokens = b.encode(&text);
    println!("tokens: {} ({:.2} bytes/token)", fmt_int(tokens.len()), (text.len() as f32) / (tokens.len().max(1) as f32));
    if tokens.len() < 2 {
        eprintln!("training corpus must encode to at least two tokens");
        return;
    }
    let batcher = Batcher::new(tokens, 0.05);
    if batcher.split <= opt.ctx {
        invalid_argument("ctx", &format!("{}", opt.ctx), "less than the number of training tokens");
    }

    // ---- model ----
    let cfg = match resume.as_ref() {
        Some((_, config)) => *config,
        None => LmConfig {
            vocab: b.vocab_size(),
            d_model: opt.d_model,
            n_layer: opt.n_layer,
            expand: opt.expand,
            conv_k: opt.conv_k,
            mlp_mult: opt.mlp_mult,
            eps: 1e-5,
            tau_max: opt.tau_max,
        },
    };
    if cfg.vocab != b.vocab_size() {
        eprintln!("tokenizer/model vocabulary mismatch: tokenizer has {}, model expects {}", b.vocab_size(), cfg.vocab);
        return;
    }
    if let Err(message) = cfg.check() {
        eprintln!("invalid model configuration: {}", message);
        return;
    }
    let mut g = Graph::new(opt.threads);
    let m = Lm::new(&mut g, &mut rng, cfg);
    g.seal_params();
    let mut optimizer = AdamW::new(&g, opt.wd);
    let mut resumed_steps = 0usize;
    if let Some((loaded, _)) = resume.as_ref() {
        if let Err(error) = ckpt::apply_exact(&mut g, loaded) {
            eprintln!("cannot resume: {}", error);
            return;
        }
        match optimizer.load_state(&g, loaded, "opt_steps") {
            Ok(true) => println!("resume: restored parameters and AdamW moments"),
            Ok(false) => println!("resume: restored parameters (this checkpoint has no optimizer state)"),
            Err(error) => {
                eprintln!("cannot resume: {}", error);
                return;
            }
        }
        resumed_steps = loaded.meta_usize("steps", 0);
    }

    println!(
        "model: d={} layers={} inner={} mlp={} conv_k={} vocab={} -> {} params",
        cfg.d_model,
        cfg.n_layer,
        cfg.inner(),
        cfg.hidden(),
        cfg.conv_k,
        cfg.vocab,
        fmt_int(g.param_count())
    );
    println!(
        "train: {} steps, batch {} x ctx {} = {} tokens/step, threads {}",
        opt.steps,
        opt.batch,
        opt.ctx,
        fmt_int(opt.batch.checked_mul(opt.ctx).expect("tokens per step overflow")),
        opt.threads
    );
    println!();

    // A resumed run is already warm; re-warming would throw away progress.
    let warmup = if resumed_steps > 0 {
        0
    } else if opt.steps >= 40 {
        opt.steps / 20
    } else {
        2
    };
    let sched = Schedule { peak: opt.lr, min: opt.lr * 0.1, warmup, total: opt.steps };
    let mut ema = 0.0f32;
    let mut best_val = f32::INFINITY;
    let mut best_step = 0usize;
    let mut saved_any = false;
    let t_start = Instant::now();

    for step in 0..opt.steps {
        g.reset();
        g.zero_grad();
        let (x, y) = batcher.train_batch(opt.batch, opt.ctx, &mut rng);
        let (_, loss) = m.loss(&mut g, &x, &y, opt.batch, opt.ctx);
        g.backward(loss);
        let gn = g.clip_grad_norm(opt.clip);
        let cur_lr = sched.lr(step);
        optimizer.step(&mut g, cur_lr);
        let l = g.scalar(loss);
        ema = if step == 0 { l } else { 0.9 * ema + 0.1 * l };
        if step % opt.log_every == 0 || step + 1 == opt.steps {
            let el = t_start.elapsed().as_secs_f64().max(1e-9);
            let tps = ((step + 1) * opt.batch * opt.ctx) as f64 / el;
            println!(
                "  step {:>5}/{}  loss {:.4}  ema {:.4}  ppl {:>8.1}  |g| {:.3}  lr {:.2e}  {:.0} tok/s",
                step + 1,
                opt.steps,
                l,
                ema,
                ema.exp(),
                gn,
                cur_lr,
                tps
            );
        }

        // ---- periodic held-out evaluation + best-checkpoint tracking ----
        if (step + 1) % opt.val_every == 0 || step + 1 == opt.steps {
            let val = evaluate(&mut g, &m, &batcher, opt.batch, opt.ctx, opt.val_batches, &mut rng);
            let improved = val < best_val;
            if improved {
                best_val = val;
                best_step = step + 1;
            }
            println!(
                "    held-out {:.4} nats ({:.3} bits/token){}",
                val,
                val / std::f32::consts::LN_2,
                if improved { "  <- best, checkpointing" } else { "" }
            );
            if improved {
                let meta = checkpoint_meta(&cfg, &tok_path, val, resumed_steps + step + 1, optimizer.t);
                let aux = optimizer.state_tensors(&g);
                match ckpt::save_with_aux(&out, &g, &meta, &aux) {
                    Ok(_) => saved_any = true,
                    Err(e) => println!("    checkpoint save failed: {}", e),
                }
            }
        }
    }

    println!();
    if saved_any {
        println!(
            "best held-out {:.4} nats/token (ppl {:.1}) at step {} -> {} ({} tensors + optimizer state)",
            best_val,
            best_val.exp(),
            best_step,
            out,
            g.params.len()
        );
        println!("resume with:  noetic train --resume {} --steps N", out);
    } else {
        println!("no checkpoint was written");
    }

    // ---- sample ----
    println!();
    println!("sample (prompt {:?}):", prompt);
    print!("{}", prompt);
    let _ = std::io::stdout().flush();
    let _text = generate(&g, &m, &b, &prompt, sample_tokens, &scfg, &mut rng, true);
}

// ---------------------------------------------------------------------------
// gen
// ---------------------------------------------------------------------------

fn cmd_gen(a: &Args) {
    let threads = threads_of(a);
    let seed = a.get_usize("seed", 99) as u64;
    let mut rng = Rng::new(seed);
    let path = a.get_str("ckpt", "noetic.ckpt");
    let ck = match ckpt::load(&path) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            eprintln!("could not load {}: {}", path, error);
            eprintln!("train one first:  cargo run --release -- train --steps 400");
            return;
        }
    };
    let config_result: std::io::Result<LmConfig> = (|| {
        Ok(LmConfig {
            vocab: ck.require_usize("vocab")?,
            d_model: ck.require_usize("d_model")?,
            n_layer: ck.require_usize("n_layer")?,
            expand: ck.require_usize("expand")?,
            conv_k: ck.require_usize("conv_k")?,
            mlp_mult: ck.require_usize("mlp_mult")?,
            eps: ck.optional_f32("eps", 1e-5)?,
            tau_max: ck.require_f32("tau_max")?,
        })
    })();
    let cfg = match config_result {
        Ok(config) => config,
        Err(error) => {
            eprintln!("checkpoint has invalid model metadata: {}", error);
            return;
        }
    };
    if let Err(message) = cfg.check() {
        eprintln!("checkpoint model configuration is invalid: {}", message);
        return;
    }

    let default_tokenizer = ck.meta.get("tok").cloned().unwrap_or_else(|| "noetic.tok".to_string());
    let tok_path = a.get_str("tok", &default_tokenizer);
    let b = match Bpe::load(&tok_path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("could not load tokenizer {}: {}", tok_path, error);
            return;
        }
    };
    if b.vocab_size() != cfg.vocab {
        eprintln!("tokenizer/model vocabulary mismatch: tokenizer has {}, checkpoint expects {}", b.vocab_size(), cfg.vocab);
        return;
    }

    let mut g = Graph::new(threads);
    let m = Lm::new(&mut g, &mut rng, cfg);
    g.seal_params();
    let loaded = match ckpt::apply_exact(&mut g, &ck) {
        Ok(count) => count,
        Err(error) => {
            eprintln!("checkpoint is incompatible with the model: {}", error);
            return;
        }
    };
    let val_loss = match ck.optional_f32("val_loss", 0.0) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("warning: {}", error);
            0.0
        }
    };
    println!("loaded {} tensors from {}, val loss {:.4}", loaded, path, val_loss);
    let prompt = a.get_str_or_positional("prompt", "the ");
    let scfg = sample_cfg_from(a);
    println!();
    print!("{}", prompt);
    let _ = std::io::stdout().flush();
    let _ = generate(&g, &m, &b, &prompt, a.get_usize("n", 240), &scfg, &mut rng, true);
}

// ---------------------------------------------------------------------------
// plan (self-play search)
// ---------------------------------------------------------------------------

fn net_only_rollout(g: &Graph, net: &PvNet, start: &Puzzle, max_steps: usize) -> bool {
    let mut env = *start;
    let mut feat = vec![0.0f32; plan::FEAT];
    for _ in 0..max_steps {
        if env.is_solved() {
            return true;
        }
        env.features(&mut feat);
        let (p, _v) = net.eval_one(g, &feat);
        let mut best = usize::MAX;
        let mut bv = -1.0f32;
        for act in 0..N_ACT {
            if env.legal(act) && p[act] > bv {
                bv = p[act];
                best = act;
            }
        }
        if best == usize::MAX {
            return false;
        }
        env.step(best);
    }
    env.is_solved()
}

fn cmd_plan(a: &Args) {
    let threads = threads_of(a);
    let seed = a.get_usize("seed", 2024) as u64;
    let mut rng = Rng::new(seed);
    let hidden = require_positive("hidden", a.get_usize("hidden", 128));
    // Defaults that actually learn. The old ones (12 x 24 x 64, one pass over
    // fresh data, curriculum to 14) evaluated at 0% solved; see README.
    let iters = require_positive("iters", a.get_usize("iters", 30));
    let games = require_positive("games", a.get_usize("games", 32));
    let sims = require_positive("sims", a.get_usize("sims", 128));
    let batch = require_positive("batch", a.get_usize("batch", 64));
    let epochs = require_positive("epochs", a.get_usize("epochs", 2));
    let replay_capacity = require_positive("replay", a.get_usize("replay", 20_000));
    let lr = require_nonnegative_finite("lr", a.get_f32("lr", 2e-3));
    let max_scramble = a.get_usize("scramble", 12).max(4);
    let max_steps = require_positive("maxsteps", a.get_usize("maxsteps", 60));
    let temp_moves = a.get_usize("tempmoves", 8);
    let gate_boards = a.get_usize("gate", 8);

    let mut g = Graph::new(threads);
    let net = PvNet::new(&mut g, &mut rng, hidden);
    g.seal_params();
    let mut opt = AdamW::new(&g, 0.0);
    let mut replay = plan::ReplayBuffer::new(replay_capacity);
    println!("noetic plan - self-play MCTS on the 8-puzzle");
    println!(
        "policy/value net: {} -> {} -> {} -> (4 logits, 1 value), {} params",
        plan::FEAT,
        hidden,
        hidden,
        fmt_int(g.param_count())
    );
    println!(
        "{} iterations x {} games x {} simulations, {} epochs over a {} position replay buffer",
        iters,
        games,
        sims,
        epochs,
        fmt_int(replay_capacity)
    );
    println!("curriculum scramble 4..{}, acceptance gate on {} boards", max_scramble, gate_boards);
    println!();

    // Acceptance gate: an iteration that makes the solver measurably worse is
    // rolled back, so a bad batch of self-play cannot undo real progress.
    let mut best_snapshot: Vec<Vec<f32>> = g.params.iter().map(|p| g.val[p.id].clone()).collect();
    let mut best_gate = -1.0f32;
    let mut rejected = 0usize;

    let t0 = Instant::now();
    for it in 0..iters {
        let denom = if iters > 1 { iters - 1 } else { 1 };
        let scr = 4 + (it * (max_scramble.saturating_sub(4))) / denom;
        let st = plan::selfplay_iteration(
            &mut g,
            &net,
            &mut opt,
            &mut rng,
            &mut replay,
            games,
            sims,
            scr,
            batch,
            lr,
            max_steps,
            temp_moves,
            epochs,
        );

        let mut gate_note = String::new();
        if gate_boards > 0 {
            let mut gate_rng = Rng::new(seed ^ 0x9E37_79B9);
            let gate = plan::evaluate_solver(&net, &g, &mut gate_rng, gate_boards, scr, sims, max_steps);
            if gate + 1e-6 < best_gate {
                for (slot, param) in g.params.iter().enumerate() {
                    g.val[param.id].copy_from_slice(&best_snapshot[slot]);
                }
                rejected += 1;
                gate_note = format!("  gate {:>5.1}% (rejected, rolled back)", 100.0 * gate);
            } else {
                for (slot, param) in g.params.iter().enumerate() {
                    best_snapshot[slot].copy_from_slice(&g.val[param.id]);
                }
                best_gate = gate;
                gate_note = format!("  gate {:>5.1}%", 100.0 * gate);
            }
        }

        println!(
            "  iter {:>3}/{}  scramble {:>2}  solved {:>5.1}%  avg moves {:>5.1}  loss {:.4}  buffer {}{}",
            it + 1,
            iters,
            scr,
            100.0 * st.solve_rate,
            st.avg_len,
            st.loss,
            fmt_int(st.buffered),
            gate_note
        );
    }
    println!("  self-play time {:.1} s, {} iteration(s) rolled back by the gate", t0.elapsed().as_secs_f64(), rejected);

    // ---- evaluation: search vs raw policy ----
    let eval_n = require_positive("eval", a.get_usize("eval", 20));
    let eval_scramble = a.get_usize("evalscramble", max_scramble);
    let eval_sims =
        sims.checked_mul(2).unwrap_or_else(|| invalid_argument("sims", &format!("{}", sims), "a value that can be doubled"));
    let mut solved_search = 0usize;
    let mut solved_net = 0usize;
    let mut moves_sum = 0usize;
    let mut nodes_sum = 0usize;
    let mut showcase: Option<(Puzzle, Vec<usize>)> = None;
    for i in 0..eval_n {
        let mut env = Puzzle::solved();
        env.scramble(eval_scramble, &mut rng);
        let (ok, mv, nodes) = plan::solve(&env, &net, &g, &mut rng, eval_sims, max_steps);
        if ok {
            solved_search += 1;
            moves_sum += mv.len();
            if showcase.is_none() && i > 0 {
                showcase = Some((env, mv.clone()));
            }
        }
        nodes_sum += nodes;
        if net_only_rollout(&g, &net, &env, max_steps) {
            solved_net += 1;
        }
    }
    println!();
    println!("eval on {} fresh boards (scramble {}):", eval_n, eval_scramble);
    println!("  policy only, no search   {:>5.1}% solved", 100.0 * (solved_net as f32) / (eval_n as f32));
    println!(
        "  policy + MCTS ({} sims)  {:>5.1}% solved, avg {:.1} moves, {} nodes/board",
        eval_sims,
        100.0 * (solved_search as f32) / (eval_n as f32),
        if solved_search > 0 { (moves_sum as f32) / (solved_search as f32) } else { 0.0 },
        fmt_int(nodes_sum / eval_n.max(1))
    );
    if let Some((board, mv)) = showcase {
        println!();
        println!("example solve (manhattan {}):", board.manhattan());
        print!("{}", board.render());
        let mut names: Vec<&str> = Vec::new();
        for action in &mv {
            names.push(action_name(*action));
        }
        println!("  {} moves: {}", mv.len(), names.join(" "));
        let mut env = board;
        for action in &mv {
            env.step(*action);
        }
        print!("{}", env.render());
    }
}

// ---------------------------------------------------------------------------
// mem (sparse distributed memory)
// ---------------------------------------------------------------------------

fn cmd_mem(a: &Args) {
    let bits = require_positive("bits", a.get_usize("bits", 512));
    let loc = require_positive("loc", a.get_usize("loc", 4096));
    let n_pat = require_positive("patterns", a.get_usize("patterns", 50));
    let seed = a.get_usize("seed", 5) as u64;
    let mut rng = Rng::new(seed);
    let radius = a.get_usize("radius", Sdm::default_radius(bits));
    let mut mem = Sdm::new(bits, loc, radius, seed ^ 0x5EED);
    println!("noetic mem - Kanerva sparse distributed memory");
    println!("  address space 2^{},  {} hard locations,  activation radius {}", bits, loc, radius);
    let mut pats: Vec<Vec<u64>> = Vec::new();
    let mut act_sum = 0usize;
    for _ in 0..n_pat {
        let p = random_bits(&mut rng, bits);
        act_sum += mem.write(&p, &p);
        pats.push(p);
    }
    println!(
        "  wrote {} patterns, {:.1} locations activated per write ({:.2}% of memory)",
        n_pat,
        (act_sum as f32) / (n_pat as f32),
        100.0 * (act_sum as f32) / (n_pat as f32) / (loc as f32)
    );
    println!();
    println!("  autoassociative recall (1 read, then iterated clean-up):");
    println!("    cue noise      direct errors   iterated errors   exact recalls");
    let levels = [0usize, 5, 10, 15, 20, 25, 30, 35, 40];
    for li in 0..levels.len() {
        let pct = levels[li];
        let flips = bits * pct / 100;
        let mut d1 = 0usize;
        let mut d2 = 0usize;
        let mut exact = 0usize;
        for i in 0..n_pat {
            let cue = flip_bits(&pats[i], bits, flips, &mut rng);
            let r1 = mem.read(&cue);
            let r2 = mem.read_iterated(&cue, 6);
            d1 += hamming(&r1, &pats[i]);
            let e = hamming(&r2, &pats[i]);
            d2 += e;
            if e == 0 {
                exact += 1;
            }
        }
        println!(
            "    {:>3}% ({:>3} bits)  {:>8.2} bits    {:>8.2} bits     {:>3}/{}",
            pct,
            flips,
            (d1 as f32) / (n_pat as f32),
            (d2 as f32) / (n_pat as f32),
            exact,
            n_pat
        );
    }

    // key -> value binding
    println!();
    println!("  heteroassociative binding (recall a value from a corrupted key):");
    let mut mem2 = Sdm::new(bits, loc, radius, seed ^ 0xBEEF);
    let n_kv = require_positive("kv", a.get_usize("kv", 20));
    let mut keys: Vec<Vec<u64>> = Vec::new();
    let mut vals: Vec<Vec<u64>> = Vec::new();
    for _ in 0..n_kv {
        let k = random_bits(&mut rng, bits);
        let v = random_bits(&mut rng, bits);
        mem2.write(&k, &v);
        keys.push(k);
        vals.push(v);
    }
    for pct in [0usize, 10, 20, 30].iter() {
        let flips = bits * *pct / 100;
        let mut err = 0usize;
        let mut exact = 0usize;
        for i in 0..n_kv {
            let cue = flip_bits(&keys[i], bits, flips, &mut rng);
            let out = mem2.read(&cue);
            let e = hamming(&out, &vals[i]);
            err += e;
            if e == 0 {
                exact += 1;
            }
        }
        println!(
            "    key noise {:>3}%   avg {:>6.2} wrong bits of {}   exact {:>3}/{}",
            pct,
            (err as f32) / (n_kv as f32),
            bits,
            exact,
            n_kv
        );
    }

    // dense -> sparse binding via LSH projection
    println!();
    println!("  random-projection hashing (dense vector -> address):");
    let dim = 64usize;
    let proj = Projection::new(dim, bits, 1234);
    let mut base = vec![0.0f32; dim];
    for i in 0..dim {
        base[i] = rng.normal();
    }
    let a0 = proj.encode(&base);
    for noise in [0.0f32, 0.1, 0.3, 1.0, 3.0].iter() {
        let mut v = base.clone();
        for i in 0..dim {
            v[i] += rng.normal() * *noise;
        }
        let ai = proj.encode(&v);
        println!(
            "    perturbation sigma {:>4.1}  ->  hamming {:>4} / {} ({:.1}%)",
            noise,
            hamming(&a0, &ai),
            bits,
            100.0 * (hamming(&a0, &ai) as f32) / (bits as f32)
        );
    }
    println!();
    println!("  similar activations map to nearby addresses, so the network's");
    println!("  hidden state can index this memory directly - one-shot writes,");
    println!("  no gradients, no forgetting of earlier entries.");
}

// ---------------------------------------------------------------------------

/// CLI entry point. `main.rs` is a three-line shim over this so the whole
/// engine, including the command surface, stays importable as a library.
pub fn run() {
    let a = Args::parse();
    let cmd = if a.cmd.is_empty() { "help".to_string() } else { a.cmd.clone() };
    match cmd.as_str() {
        "selftest" => {
            let ok = selftest::run_all(threads_of(&a), a.get_usize("seed", 20250816) as u64);
            if !ok {
                std::process::exit(1);
            }
        }
        "bench" => cmd_bench(&a),
        "bpe" | "tok" => cmd_bpe(&a),
        "train" => cmd_train(&a),
        "gen" | "generate" => cmd_gen(&a),
        "plan" | "search" => cmd_plan(&a),
        "mem" | "sdm" => cmd_mem(&a),
        "help" | "-h" | "--help" => help(),
        other => {
            println!("unknown command: {}", other);
            println!();
            help();
            std::process::exit(2);
        }
    }
}
