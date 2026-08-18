# noetic

A from-scratch neural-computing playground in stable Rust, built with the standard library only.

Noetic combines a selective linear-recurrent language model, reverse-mode autodiff, byte-level BPE, optimizers, checkpointing, sparse distributed memory, and neural-guided Monte Carlo tree search. It intentionally uses neither transformers nor self-attention.

```toml
[dependencies]
# intentionally empty
```

> **Status:** experimental and educational. CI builds and tests the crate on Rust 1.70 and current stable, enforces `rustfmt` and a clippy `-D warnings` gate, runs the built-in self-test, and runs an end-to-end verifier that trains, resumes, generates and tokenizes with the compiled binary. It is still not a production ML framework or a substitute for audited numerical libraries.

## Why a linear recurrence?

The sequence core is the elementwise recurrence

```text
h_t = a_t ⊙ h_{t-1} + b_t
```

where the network derives `a_t` and `b_t` from the current token. Affine state transitions compose associatively:

```text
(a_L, b_L) then (a_R, b_R) = (a_R·a_L, a_R·b_L + b_R)
```

That permits either a sequential scan or a chunked parallel prefix scan over the time axis. Training work scales linearly with sequence length, while streaming generation keeps a fixed-size recurrent state instead of a growing KV cache.

The update uses `a_t = sigmoid(z_t)` and `b_t = (1 - a_t) ⊙ v_t`, making each recurrent update a convex blend when `v_t` is bounded.

## Architecture

```text
tokens -> tied embedding
            |
            +-> [ RMSNorm -> selective SSM -> residual
            |    RMSNorm -> SwiGLU       -> residual ] x n_layer
            |
            +-> RMSNorm -> tied vocabulary projection -> logits
```

Inside an SSM layer:

```text
u       = x W_in^T + b_in       # 3E channels
v,z,o   = split(u)
v       = SiLU(causal_depthwise_conv(v))
a       = sigmoid(z)
b       = (1-a) ⊙ v
h       = scan(a,b)
y       = h ⊙ SiLU(o)
out     = y W_out^T + b_out
```

The streaming decoder uses a circular convolution buffer, so it does not shift the full `K x E` history on every token.

## Components

| File | Responsibility |
|---|---|
| `tensor.rs` | Threaded `f32` GEMM variants, matrix-vector products, and vector primitives |
| `scan.rs` | Sequential scan, time-chunked parallel scan, and reverse-time adjoint |
| `autograd.rs` | Flat-arena reverse-mode tape and fused operators |
| `nn.rs` | Linear layers, RMSNorm, and SwiGLU |
| `model.rs` | Validated language-model configuration and selective SSM stack |
| `infer.rs` | Fixed-state streaming decode and sampling controls |
| `optim.rs` | AdamW, Lion, and warmup/cosine schedule |
| `bpe.rs` | Strictly serialized byte-level BPE with exact raw-byte round trips |
| `data.rs` | Synthetic corpus and bounds-checked random crop batching |
| `ckpt.rs` | Versioned, CRC-protected, shape-validated checkpoints |
| `sdm.rs` | Kanerva sparse distributed memory and random projection hashing |
| `plan.rs` | 8-puzzle environment, PUCT MCTS, and self-play policy/value learning |
| `rng.rs` | SplitMix64/xoshiro256++ and common distributions |
| `selftest.rs` | Kernel, gradient, learning, serialization, memory, and parity regressions |
| `verify_engine.py` | End-to-end verification of the compiled binary |
| `verify_structure.py` | Compiler-independent structural checks |
| `lib.rs` | Crate root: the module list and two documented lint exemptions |
| `cli.rs` | Argument parsing and every subcommand |

The Rust files intentionally live at the repository root; `Cargo.toml` declares `lib.rs` as the library target and `main.rs` (a three-line shim) as the binary. Because there is a library target, `tests/engine.rs` can exercise the crate exactly as an external consumer would.

## Requirements

- Rust 1.70 or newer
- No native libraries or third-party crates
- Python 3 only for the optional independent verifier scripts

## Quick start

```bash
cargo run --release -- selftest
cargo run --release -- bench
cargo run --release -- bpe --bytes 400000 --vocab 1024
cargo run --release -- train --steps 600 --d 192 --layers 3 --ctx 128 --taumax 128
cargo run --release -- gen --ckpt noetic.ckpt --prompt "memo alpha = " --temp 0.8 --topp 0.95
cargo run --release -- plan
cargo run --release -- mem --bits 512 --loc 4096 --patterns 50
```

Run `cargo run --release -- help` for the full command summary.

Rough costs on the reference machine below: `selftest` and `bench` take seconds, `bpe` on 3 MB about 0.2 s, `plan` with its defaults about 35 s, and the `train` line above (600 steps, 192 wide, 3 layers, 128 context) roughly 20-30 minutes. Start with `--steps 60 --d 96 --layers 2` for a one-minute smoke run.

### Recommended validation

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets                 # 31 unit + 6 integration tests
cargo run --release -- selftest --threads 2
python3 verify_structure.py
cargo build --release && python3 verify_engine.py target/release/noetic
```

CI runs exactly this list on every pull request. The build matrix covers the declared MSRV (`1.70.0`) and current stable Rust.

`verify_engine.py` deliberately does *not* re-implement any of the math. An earlier version transcribed the formulas into Python and checked the transcriptions, which meant it reported success during a period when `cargo test` did not even compile. It now drives the compiled binary and asserts only properties Rust cannot check about itself: that every reported error is inside a hard limit, that `--threads 1` and `--threads 4` produce byte-identical output, that training reduces a held-out loss and writes a resumable checkpoint, that greedy generation is reproducible across processes, and that BPE round-trips stay lossless.

## Training and generation

Train on a text file:

```bash
cargo run --release -- train \
  --data corpus.txt \
  --vocab 1024 \
  --d 192 \
  --layers 3 \
  --ctx 128 \
  --batch 12 \
  --steps 1500 \
  --out noetic.ckpt \
  --tok noetic.tok
```

If `--data` is omitted or cannot provide content, the CLI uses its synthetic curriculum. The curriculum mixes grammar, arithmetic, counting, and delayed memo/query examples.

Training evaluates a held-out split every `--valevery` steps (a fifth of the run by default) and only writes `--out` when that loss improves, so the file on disk is the best model seen rather than the last one. Checkpoints carry the AdamW moments and the step counter, so a run can be continued:

```bash
cargo run --release -- train --resume noetic.ckpt --steps 400
```

A resumed run reads the architecture and the tokenizer path out of the checkpoint, skips learning-rate warmup, and refuses to start if the recorded tokenizer is missing or the shapes do not match exactly. Argument validation happens before the corpus is read and before BPE training, so a typo in a flag fails immediately instead of after a few minutes of tokenizer work.

Generate from the resulting files:

```bash
cargo run --release -- gen \
  --ckpt noetic.ckpt \
  --tok noetic.tok \
  --prompt "memo gamma = " \
  --n 160 \
  --temp 0.9 \
  --topk 40 \
  --topp 0.95
```

Sampling supports temperature, top-k, nucleus filtering, greedy decoding, and a once-per-token repetition penalty over a bounded recent window. The CLI defaults come from `SampleCfg::default_cfg()`, so the library and the command line cannot drift apart.

## Self-play planning

`plan` trains a policy/value network on the 8-puzzle from nothing but its own search:

```bash
cargo run --release -- plan --iters 30 --games 32 --sims 128 --scramble 12 --epochs 2 --replay 20000
```

Each iteration plays `--games` games with PUCT MCTS, stores every position in a fixed-capacity replay buffer, fits the network for `--epochs` passes over that buffer, and then plays `--gate` fresh boards to decide whether to keep the update. An iteration that measurably worsens the solve rate is rolled back to the previous parameters, which is why a bad batch of self-play cannot undo earlier progress. Discarding the data after a single pass (the earlier behaviour) left the policy at 0% solved; keeping a replay buffer and gating updates is what produces the numbers in the table above.

## File-format safety

Checkpoint loading validates:

- magic and exact format version
- CRC-32 over the complete payload
- UTF-8 metadata and tensor names
- duplicate entries
- bounded rank and entry counts
- shape-product versus element-count consistency
- exact live tensor shapes during application
- trailing or truncated data

Optimizer state travels in the same file under a reserved `aux.` name prefix. Application still demands that the model tensors match the live graph exactly, so the reserved namespace adds resumability without weakening that check.

Checkpoint and tokenizer saves use write-then-rename behavior to avoid replacing a valid file with a partial write.

Tokenizer loading rejects malformed counts, unknown token references, duplicate merge pairs, extra fields, and trailing records. `encode_bytes` and `decode_bytes` provide exact round trips for arbitrary bytes; the text API retains UTF-8-aware pre-tokenization.

## Verification coverage

The built-in self-test covers:

1. three GEMM layouts against an independent `f64` reference
2. sequential versus chunked scan equivalence
3. scan gradients versus central finite differences
4. gradients of the auxiliary ops (`sub`, `scale`, `gelu`, `matmul_nn`) versus finite differences
5. full-model parameter gradients versus finite differences
6. batched versus streaming logits
7. AdamW convergence
8. end-to-end learning on a fixed mapping
9. Unicode and arbitrary-byte BPE round trips
10. checkpoint round trips plus corruption detection
11. sparse-memory recall from an exactly corrupted cue
12. RNG moment and uniformity checks

`cargo test` adds the cases a one-line summary cannot express, each as its own named test: streaming/batched parity across awkward configurations (`conv_k = 1`, a kernel wider than the sequence, three layers, a single-token sequence), depthwise-convolution gradients when the kernel is longer than the sequence, agreement between both scan policies through the tape, the scan adjoint against its closed form, activation recycling reaching a steady state, replay-buffer eviction order, search solving shallow boards with an untrained network, Gumbel-max sampling matching softmax frequencies, and checkpoint rejection on both shape mismatch and bit corruption.

## Performance notes

- Release builds enable `opt-level=3`, fat LTO, one codegen unit, symbol stripping, and abort-on-panic.
- GEMM and scan kernels use scoped standard-library threads over disjoint output regions.
- The autodiff graph stores nodes in parallel arenas and recycles activation buffers through a pool, so steady-state training steps stop allocating (a test watches the allocation counter to keep it that way).
- Streaming inference allocates scratch once per token and reuses it across layers; convolution history is circular.
- BPE training maintains an exact inverted pair index so stale word memberships do not accumulate.

### Measured, not claimed

Everything below comes from this repository at this commit, on a 4-core laptop CPU (`--threads 4`, release profile). Reproduce with the command shown; expect different absolute numbers on different hardware.

| What | Measurement | Command |
|---|---|---|
| GEMM 256x256x256 | 18.1 GFLOP/s | `bench` |
| Sequential scan, B8 T512 D256 | 643 M elem/s | `bench` |
| Training step, d=128 L=2 B8 T64 | 339 ms, 1.5 k tokens/s | `bench` |
| Streaming decode | 0.58 ms/token, 1.7 k tokens/s | `bench` |
| BPE, 3 MB to 1024 tokens | 0.14 s, 3.71 bytes/token, lossless | `bpe --bytes 3000000 --vocab 1024` |
| Held-out loss after 60 steps (d=96, L=2, ctx 64) | 1.62 nats = 2.34 bits/token, against a 6.24-nat uniform baseline | `train --steps 60 --d 96 --layers 2 --ctx 64 --vocab 512 --bytes 120000` |
| 8-puzzle, scramble 12, after 30 self-play iterations | 100% solved with search, 40% with the raw policy, 35 s of self-play | `plan` |
| Sparse memory, 50 patterns in 4096 locations | all 50 recalled exactly from a 10% corrupted cue, 45/50 at 20% | `mem --bits 512 --loc 4096 --patterns 50` |

Two results are worth stating plainly because they are negative:

- **The parallel scan is usually not a win here.** On this machine the time-chunked kernel is about 1.15x faster than the sequential one for cache-resident single-sequence shapes (B1 T2048 D64) and about 2x *slower* for DRAM-resident ones (B1 T8192 D256): both kernels are memory-bound, and chunking adds a pass over `a`. During training the batch axis already fills every core, so `ScanPolicy::Auto` keeps the sequential kernel unless the batch is too narrow to do that. The knob is for many-core single-sequence prefill, not a free speedup.
- **Search without learning is shallow.** With an untrained network, 400 simulations reliably solve 4-move scrambles, solve 6-move scrambles 5 times in 12, and never solve 8-move ones. The jump to 100% at scramble 12 comes from self-play training, not from the tree.

Always measure on the target machine with `cargo run --release -- bench`; no universal throughput number is claimed.

## Limitations

- CPU-only scalar Rust; no SIMD intrinsics, BLAS, GPU, distributed training, or mixed precision
- the parallel scan only helps when the batch axis cannot fill the cores; see the measured note above
- experimental checkpoint and tokenizer formats with no backward-compatibility promise beyond their explicit version
- basic hand-written CLI parsing rather than a full argument-schema library
- MCTS intentionally uses a simple tree without transposition-table reuse, and the planner ships a single toy environment (the 8-puzzle)
- model quality depends on data, scale, and training time; the architecture alone does not imply competitive language-model quality

## License

MIT. See `LICENSE`.
