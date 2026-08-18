//! noetic - a from-scratch AI engine in pure Rust.
//!
//! Zero dependencies. No transformer, no attention, no autograd crate, no BLAS,
//! no tokenizer library, no serialization library. Everything below is built
//! from `std` and arithmetic:
//!
//!   rng.rs       splitmix64/xoshiro sampling: normal, gamma, Dirichlet, Gumbel
//!   tensor.rs    multithreaded GEMM kernels (nn / nt / tn) + matvec + vector ops
//!   scan.rs      sequential and time-chunked parallel associative scans
//!   autograd.rs  reverse-mode autodiff over a flat tape, 22 differentiable ops
//!   nn.rs        Linear / RMSNorm / SwiGLU
//!   model.rs     gated linear-recurrence (selective SSM) language model
//!   infer.rs     O(1)-per-token streaming decoder + sampling stack
//!   optim.rs     AdamW, Lion, cosine schedule, gradient clipping
//!   bpe.rs       byte-level BPE trainer/encoder/decoder
//!   ckpt.rs      binary checkpoint format with CRC-32 integrity
//!   sdm.rs       Kanerva sparse distributed memory (one-shot recall)
//!   plan.rs      PUCT MCTS + self-play policy/value learning
//!   selftest.rs  finite-difference gradient checks and reference oracles
//!   cli.rs       argument parsing and the subcommands
//!
//! Two clippy lints are silenced crate-wide, deliberately and narrowly:
//!
//! * `needless_range_loop` - numeric kernels index several arrays from one
//!   counter (`c[i*n+j] += a[i*k+p] * b[p*n+j]`). Iterator rewrites of those
//!   loops obscure the formula they implement and, for the strided cases, do
//!   not express it at all.
//! * `too_many_arguments` - GEMM-shaped functions take their dimensions and
//!   slices positionally, matching the BLAS convention the kernels mirror.
//!
//! Everything else is expected to build clean: CI runs
//! `cargo clippy --all-targets -- -D warnings`.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod autograd;
pub mod bpe;
pub mod ckpt;
pub mod cli;
pub mod data;
pub mod infer;
pub mod model;
pub mod nn;
pub mod optim;
pub mod plan;
pub mod rng;
pub mod scan;
pub mod sdm;
pub mod selftest;
pub mod tensor;
