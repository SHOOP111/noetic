# noetic

A from-scratch neural-computing playground in stable Rust, built with the standard library only.

Noetic combines a selective linear-recurrent language model, reverse-mode autodiff, byte-level BPE, optimizers, checkpointing, sparse distributed memory, and neural-guided Monte Carlo tree search. It intentionally uses neither transformers nor self-attention.

```toml
[dependencies]
# intentionally empty
```

> **Status:** experimental and educational. The crate is compiled and tested in CI on Rust 1.70 and the latest stable toolchain, but it is not a production ML framework or a substitute for audited numerical libraries.

## Why a linear recurrence?

The sequence core is the elementwise recurrence

```text
h_t = a_t ⊙ h_{t-1} + b_t
```

where the network derives `a_t` and `b_t` from the current token. Affine state transitions compose associatively:

```text
(a_L, b_L) then (a_R, b_R) = (a_R·a_L, a_R·b_L + b_R)
```

That permits either a sequential scan or a log-depth parallel prefix scan. Training work scales linearly with sequence length, while streaming generation keeps a fixed-size recurrent state instead of a growing KV cache.

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
| `scan.rs` | Sequential scan, log-depth scan, and reverse-time adjoint |
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
| `verify_math.py` | Independent numerical checks of core derivations |
| `verify_structure.py` | Compiler-independent structural checks |

The Rust files currently live at the repository root; `Cargo.toml` declares `main.rs` explicitly as the binary target.

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
cargo run --release -- plan --iters 12 --games 24 --sims 64 --scramble 14
cargo run --release -- mem --bits 512 --loc 4096 --patterns 50
```

Run `cargo run --release -- help` for the full command summary.

### Recommended validation

```bash
cargo check --all-targets
cargo test --all-targets
cargo run --release -- selftest --threads 2
python3 verify_math.py
python3 verify_structure.py
```

CI runs all of these on every pull request. The build matrix covers the declared MSRV (`1.70.0`) and current stable Rust.

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

Sampling supports temperature, top-k, nucleus filtering, greedy decoding, and a once-per-token repetition penalty over a bounded recent window.

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

Checkpoint and tokenizer saves use write-then-rename behavior to avoid replacing a valid file with a partial write.

Tokenizer loading rejects malformed counts, unknown token references, duplicate merge pairs, extra fields, and trailing records. `encode_bytes` and `decode_bytes` provide exact round trips for arbitrary bytes; the text API retains UTF-8-aware pre-tokenization.

## Verification coverage

The built-in self-test covers:

1. three GEMM layouts against an independent `f64` reference
2. sequential versus log-depth scan equivalence
3. scan gradients versus central finite differences
4. full-model parameter gradients versus finite differences
5. batched versus streaming logits
6. AdamW convergence
7. end-to-end learning on a fixed mapping
8. Unicode and arbitrary-byte BPE round trips
9. checkpoint round trips plus corruption detection
10. sparse-memory recall from an exactly corrupted cue
11. RNG moment and uniformity checks

`verify_math.py` independently checks the scan, scan adjoint, RMSNorm, softmax cross-entropy, distribution-target cross-entropy, depthwise convolution, SiLU derivative, decay initialization, and long-sequence state boundedness.

## Performance notes

- Release builds enable `opt-level=3`, fat LTO, one codegen unit, symbol stripping, and abort-on-panic.
- GEMM and scan kernels use scoped standard-library threads over disjoint output regions.
- The autodiff graph stores nodes in parallel arenas and truncates activations back to a watermark between steps.
- Streaming inference allocates scratch once per token and reuses it across layers; convolution history is circular.
- BPE training maintains an exact inverted pair index so stale word memberships do not accumulate.

Always measure on the target machine with `cargo run --release -- bench`; no universal throughput number is claimed.

## Limitations

- CPU-only scalar Rust; no SIMD intrinsics, BLAS, GPU, distributed training, or mixed precision
- experimental checkpoint and tokenizer formats with no backward-compatibility promise beyond their explicit version
- basic hand-written CLI parsing rather than a full argument-schema library
- MCTS intentionally uses a simple tree without transposition-table reuse
- model quality depends on data, scale, and training time; the architecture alone does not imply competitive language-model quality

## License

MIT. See `LICENSE`.
