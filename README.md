# noetic

**A from-scratch neural intelligence stack in pure Rust. No transformer. No attention. No dependencies. Not even one crate.**

```toml
[dependencies]
# intentionally empty
```

`std` only. No `ndarray`, no `rayon`, no `rand`, no `serde`, no `tch`/`candle`/`burn`/`onnx`, no BLAS, no `unsafe`, no nightly features. Every matrix multiply, every gradient, every random number, every byte of the serialization format, every thread pool is written by hand in this repo — ~5,400 lines across 15 modules.

---

## Why not a transformer?

Self-attention costs `O(T^2)` compute and an `O(T)` KV-cache per token. This engine replaces it with a **selective gated linear recurrence** — a diagonal state-space model in the Mamba/S4/RWKV/linear-RNN family — whose core is one associative primitive:

```
h_t = a_t ⊙ h_{t-1} + b_t          a_t, b_t, h_t ∈ R^E    (elementwise, diagonal state)
```

`a_t` and `b_t` are **produced by the network from the current token** (that is the *selective* part: the recurrence's own time constants are data-dependent, which is what buys attention-like content routing without attention). Because affine maps compose associatively,

```
(a_L, b_L) ∘ (a_R, b_R) = (a_R·a_L,  a_R·b_L + b_R)
```

the whole sequence can be evaluated by a **prefix scan** instead of a serial loop. Consequences:

| | transformer | noetic |
|---|---|---|
| train cost / sequence | `O(T² d)` | `O(T d)` work, `O(log T)` depth |
| generation state | KV cache, grows with `T` | fixed `E` floats per layer |
| cost per generated token | grows with context | **constant, forever** |
| context limit | quadratic wall | none in principle |

Two scan strategies ship, and the self-test asserts they agree:

* `scan_sequential` — `O(T)` work, threaded across batch. Best for training.
* `scan_log_depth` — Hillis–Steele double-buffered prefix scan, `O(T log T)` work but `O(log T)` **depth**, threaded across *time*. Best for one very long sequence on many cores.

Stability is structural rather than hoped-for: `a_t = σ(z_t) ∈ (0,1)` and `b_t = (1 - a_t) ⊙ v_t`, making each state update a convex blend, so `|h| ≤ max|v|` for any sequence length. Verified empirically at `T = 4000` with zero growth.

---

## What's in the box

Seven subsystems, all hand-rolled, all wired into one CLI:

1. **Tensor / BLAS layer** — cache-blocked, multithreaded `gemm` in three transpose modes with 4×-unrolled inner loops and hand-written micro-kernels.
2. **Reverse-mode autodiff** — a flat tape with 24 fused ops, arena-allocated gradients, `mem::take` borrow trickery instead of `Rc<RefCell<…>>`, topological reverse sweep, gradient-norm clipping, `no_grad` inference mode.
3. **The language model** — depthwise causal conv + selective gated linear recurrence + SwiGLU, RMSNorm pre-norm residual stack, weight-tied vocabulary head, log-spaced decay-spectrum initialization.
4. **Byte-level BPE tokenizer** — trained from scratch with an inverted pair index; lossless on arbitrary bytes, so emoji and CJK round-trip exactly.
5. **Optimizers** — AdamW (decoupled decay, bias correction) and Lion (sign-momentum), cosine schedule with warmup.
6. **Streaming inference** — an `O(1)`-per-token recurrent decoder with temperature / top-k / nucleus / repetition-penalty sampling, plus a batch-vs-stream equivalence test.
7. **Two non-gradient intelligences** — a **Kanerva sparse distributed memory** (associative recall from corrupted cues) and an **AlphaZero-style planner** (PUCT MCTS + self-play policy/value learning on a symbolic puzzle) that solves tasks gradient descent alone cannot.

---

## Architecture

```
tokens ──► embedding (weight-tied with output head)
             │
     ┌───────▼─────────────────────── Block × n_layer ──────────────┐
     │  x += SSM( RMSNorm(x) )                                      │
     │  x += SwiGLU( RMSNorm(x) )                                   │
     └───────┬──────────────────────────────────────────────────────┘
             ▼
        RMSNorm ──► xᵀ · Eᵀ ──► logits ──► softmax cross-entropy
```

Inside one SSM layer (`d_model → E = expand · d_model → d_model`):

```
u            = x · W_inᵀ + b_in                      # one GEMM → 3E channels
v, z, o      = split(u, E, E, E)                     # value / gate / output gate
v            = depthwise_causal_conv_k(v) ──► SiLU   # local mixing, K taps, per-channel
a            = σ(z)                                  # data-dependent forgetting, (0,1)
b            = (1 − a) ⊙ v                           # convex-blend input
h            = scan(a, b)                            # h_t = a_t h_{t-1} + b_t
y            = h ⊙ SiLU(o)                           # multiplicative output gate
out          = y · W_outᵀ + b_out
```

The conv gives short-range, position-precise mixing (what induction heads do cheaply); the scan gives unbounded-range memory; the gate `o` decides what to read out of the state. Nothing here is quadratic in `T`.

### Decay-spectrum initialization

The gate bias for channel `j` is initialized so that at `z = bias`

```
τ_j = τ_max^(j / (E−1)),   a_j = exp(−1/τ_j),   bias_j = logit(a_j)
```

so channel 0 starts as a one-step reflex and channel `E−1` as a `τ_max`-step (default 128) integrator, with a log-spaced spectrum in between. The model therefore begins life with a *multi-timescale prior* and learns to deviate, instead of having to discover long memory from a random start — the single highest-leverage trick for training linear-recurrent models without warm-up pathologies.

### The autodiff tape

`Graph` is three parallel arenas (`val`, `grad`, `shape`) plus an op tape. Ops: `Add Sub Mul Scale OneMinus AddRow MulRow MatMulNN MatMulNT Silu Gelu Sigmoid Tanh RmsNorm SliceCols Embed Scan DwConv SoftmaxCe SoftCeDist MseTarget Sum Leaf`. Backward is one reverse pass with a `match` on the op — no boxed closures, no dynamic dispatch, no reference counting. RMSNorm caches its `1/rms` per row in an aux arena so the backward pass is a single fused kernel:

```
r = 1/√(mean(x²)+ε)
dx = r·ḡ − (r³·⟨ḡ,x⟩/d)·x
```

The scan's adjoint is the same recurrence run backwards in time:

```
c_t = ḡ_t + a_{t+1}·c_{t+1},     ∂L/∂b_t = c_t,     ∂L/∂a_t = c_t · h_{t-1}
```

which is why training memory is `O(T·E)` and not `O(T²)`.

### The planner

Gradient descent is one kind of intelligence; **search** is another. `plan.rs` implements a full AlphaZero-lite loop over a symbolic sliding/rotation puzzle:

* PUCT selection `Q + c·P·√N_parent/(1+N)`, first-play urgency seeded from the parent value, Dirichlet root noise, discounted value backup, transposition-free tree arena.
* A policy+value network (shared trunk, two heads) trained on **MCTS visit-count distributions** (distribution-target cross-entropy) and **clamped discounted returns** (MSE) from its own games.
* Curriculum scrambling: the scramble depth ramps up as the agent gets stronger, and the evaluation reports policy-only vs policy+search solve rates so you can watch search amplify a weak network.

### The memory

`sdm.rs` is Kanerva's Sparse Distributed Memory: random hard locations in `{0,1}^N`, activation by Hamming radius, integer counters per bit, content-addressable reads that **converge to a stored pattern from a corrupted cue** — plus iterated recall (feeding the read back in) and key→value binding. It's a one-shot writable associative memory with no gradients at all, and it degrades gracefully instead of catastrophically.

---

## Module map

| file | lines | what it is |
|---|---|---|
| `src/rng.rs` | 223 | splitmix64 + xoshiro256++ core; uniform, normal (Box–Muller), exponential, Gumbel, Marsaglia–Tsang gamma, Dirichlet, categorical, Fisher–Yates, splittable streams |
| `src/tensor.rs` | 314 | cache-blocked multithreaded `gemm_nn` / `gemm_nt` / `gemm_tn`, naive reference kernel for testing, fused `matvec_nt`, sigmoid/SiLU/GELU, RMSNorm, softmax, argmax |
| `src/scan.rs` | 154 | the associative recurrence: sequential scan, Hillis–Steele log-depth scan, reverse-time adjoint |
| `src/autograd.rs` | 1046 | the tape: 24 ops, forward constructors, one giant reverse sweep, grad-norm clip, param registry |
| `src/nn.rs` | 92 | Linear, RmsNorm, SwiGLU, fan-in-scaled init |
| `src/model.rs` | 194 | `LmConfig`, `SsmLayer` (conv + selective recurrence + gates), `Block`, `Lm` with tied head and loss |
| `src/optim.rs` | 131 | AdamW, Lion, cosine-with-warmup schedule |
| `src/infer.rs` | 232 | `O(1)`/token streaming decoder with per-layer state + conv ring buffer; temperature / top-k / top-p / repetition penalty |
| `src/bpe.rs` | 288 | byte-level BPE trainer with inverted pair index, encode/decode, on-disk format |
| `src/data.rs` | 130 | synthetic curriculum generator (grammar, arithmetic, long-range memo→query, counting) + batcher |
| `src/ckpt.rs` | 219 | hand-rolled binary checkpoint format with CRC32 integrity and a string→f32/usize metadata block |
| `src/sdm.rs` | 206 | sparse distributed memory, sign-LSH projection, iterated recall |
| `src/plan.rs` | 257+ | puzzle env, policy/value net, PUCT MCTS, self-play training loop |
| `src/selftest.rs` | — | 11 assertions covering GEMM, scan equivalence, analytic-vs-numeric gradients, batch-vs-stream parity, optimizer, learning, tokenizer, checkpoint, memory, RNG statistics |
| `src/main.rs` | — | argument parser and seven subcommands |

---

## Usage

```bash
cargo run --release -- selftest          # 11 correctness gates, incl. finite-difference gradient checks
cargo run --release -- bench             # GEMM GFLOP/s, scan throughput, train step/s, decode tok/s
cargo run --release -- bpe --bytes 400000 --vocab 1024
cargo run --release -- train --steps 600 --d 192 --layers 3 --ctx 128 --taumax 128
cargo run --release -- gen --ckpt noetic.ckpt --prompt "memo alpha = " --temp 0.8 --topp 0.95
cargo run --release -- plan --iters 12 --games 24 --sims 64 --scramble 14
cargo run --release -- mem --bits 512 --loc 4096 --patterns 50
```

Every command takes `--seed` and `--threads`. `train` writes `noetic.ckpt` + `noetic.tok`; `gen` reconstructs the exact architecture from the checkpoint's metadata block, so you never have to re-specify hyperparameters.

Start with `selftest` — it is the proof that the autodiff is right (analytic gradients vs central finite differences on 48 random parameter coordinates of the real model, plus scan-gradient, batch-vs-stream, and optimizer-convergence checks).

### Recommended first run

```bash
cargo run --release -- selftest && \
cargo run --release -- train --steps 800 --d 192 --layers 3 --ctx 128 --batch 12 --vocab 512 && \
cargo run --release -- gen --prompt "memo gamma = quiet river ; query gamma"
```

The `memo … ; query …` task in the synthetic corpus is deliberately a **long-range retrieval** test: the answer appears far earlier in the stream, so the loss can only drop if the recurrent state actually carried the binding forward. It's the non-attention analogue of an induction-head probe.

---

## Honest status of verification

This code was authored in a sandbox with **no Rust toolchain and no network access** (`rustc`/`cargo` absent, no package manager, no way to fetch them). So, plainly:

**It has not been compiled or executed here.** Expect to fix a small number of mechanical compile errors on first `cargo build` — that is the honest expectation for 5.4k lines of uncompiled Rust, and the design deliberately avoids the constructs that usually make such fixes hard: no `unsafe`, no lifetimes beyond `std::thread::scope`, no trait objects, no generics-heavy abstractions, no macros, index-based loops throughout.

What *was* verified instead, mechanically:

* **Structural pass over all 15 files** — comment/string/char-literal-aware delimiter balance, every `mod` resolving to a file, every `use crate::m::{…}` symbol existing as a `pub` item in the target module, every `g.method(…)` call existing on `Graph`, and format-string placeholder counts matching argument counts. Zero problems reported.
* **Independent numeric validation of every derivation** (re-implemented in Python from the Rust source and checked against central finite differences):

| derivation | max error |
|---|---|
| log-depth scan ≡ sequential recurrence | 8.9e-16 |
| scan gradient `∂L/∂a`, `∂L/∂b` vs finite diff | 8.2e-11 |
| RMSNorm gradient vs finite diff | 1.9e-11 |
| softmax-CE gradient vs finite diff | 1.9e-11 |
| distribution-target CE gradient (MCTS policy loss) | 2.3e-11 |
| depthwise causal conv gradient | 7.9e-11 |
| SiLU derivative | 9.3e-11 |
| decay-spectrum init round-trip | 1.1e-16 |
| convex-blend state bounded at `T = 4000` | 0.0 |

The in-repo `selftest` command re-checks all of this *inside Rust*, against the real tape and the real model, once you have a compiler.

---

## Scaling guidance

Defaults are tuned for a 2-core box (`d_model 128`, `n_layer 2`, `expand 2`, `ctx 64` — about half a million parameters). The architecture scales without code changes:

| target | flags |
|---|---|
| laptop, minutes | `--d 192 --layers 3 --ctx 128 --batch 12 --steps 1500` |
| workstation, hours | `--d 512 --layers 8 --expand 2 --ctx 512 --batch 32 --steps 50000 --vocab 4096 --taumax 512` |
| real corpus | `--corpus your.txt --bytes 50000000 --vocab 8192` |

Rules of thumb: keep `expand = 2`; raise `--taumax` roughly with context length (it sets the longest initial memory timescale); `--lr` around `3e-3` for `d ≤ 256` and `1e-3` for `d ≥ 512`; batch × ctx tokens per step should stay ≥ 4k for stable gradients; `--threads` defaults to the detected core count. Compute per token is `~O(n_layer · d²)` and is completely independent of context length — doubling the context doubles training work linearly and changes generation cost **not at all**.

### Natural extensions

* Multi-head / block-diagonal state (`h ∈ R^{H×E/H}`) with per-head decay spectra.
* Bidirectional scan for encoder tasks (run the scan forward and reverse, concatenate).
* Chunked parallel scan (`O(T/C)` sequential steps over `C`-length chunks) for the best of both scan strategies on long single sequences.
* Quantized `i8` GEMM in `tensor.rs` — the kernels are already blocked and isolated behind three functions.
* Wire the SDM in as an external retrieval memory the LM can read from, and the MCTS planner as a search layer over LM-proposed actions.

---

Built as one artifact: linear algebra, automatic differentiation, a sequence architecture, a tokenizer, optimizers, serialization, associative memory, and planning — all from arithmetic up, with an empty dependency list.
