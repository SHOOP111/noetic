//! The sequence model. **No attention anywhere.**
//!
//! Per layer:
//!
//! ```text
//!   x -> RMSNorm -> in_proj(d -> 3E) -> split(v | z | o)
//!                    v -> depthwise causal conv(k) -> SiLU
//!                    a = sigmoid(z)                    (per-channel decay)
//!                    b = (1 - a) * v                   (leaky integrator form)
//!                    h = scan(a, b)                    h_t = a_t h_{t-1} + b_t
//!                    y = h * SiLU(o)                   (output gating)
//!                 -> out_proj(E -> d) -> + residual
//!   x -> RMSNorm -> SwiGLU -> + residual
//! ```

use crate::autograd::{Graph, Nid};
use crate::nn::{init_std, Linear, RmsNorm, SwiGlu};
use crate::rng::Rng;

#[derive(Clone, Copy)]
pub struct LmConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layer: usize,
    pub expand: usize,
    pub conv_k: usize,
    pub mlp_mult: usize,
    pub eps: f32,
    pub tau_max: f32,
}

impl LmConfig {
    pub fn small(vocab: usize) -> LmConfig {
        LmConfig {
            vocab,
            d_model: 128,
            n_layer: 2,
            expand: 2,
            conv_k: 4,
            mlp_mult: 3,
            eps: 1e-5,
            tau_max: 128.0,
        }
    }

    pub fn validate(&self) {
        assert!(self.vocab > 0, "vocabulary size must be positive");
        assert!(self.d_model > 0, "d_model must be positive");
        assert!(self.n_layer > 0, "n_layer must be positive");
        assert!(self.expand > 0, "expand must be positive");
        assert!(self.conv_k > 0, "conv_k must be positive");
        assert!(self.mlp_mult > 0, "mlp_mult must be positive");
        assert!(self.eps.is_finite() && self.eps > 0.0, "RMSNorm epsilon must be finite and positive");
        assert!(self.tau_max.is_finite() && self.tau_max > 0.0, "tau_max must be finite and positive");
        let _ = self.inner();
        let _ = self.hidden();
        let _ = self.vocab.checked_mul(self.d_model).expect("embedding table is too large");
    }

    pub fn inner(&self) -> usize {
        self.d_model.checked_mul(self.expand).expect("model inner width overflow")
    }

    pub fn hidden(&self) -> usize {
        self.d_model.checked_mul(self.mlp_mult).expect("model hidden width overflow")
    }
}

pub struct SsmLayer {
    pub in_proj: Linear,
    pub conv_w: Nid,
    pub conv_b: Nid,
    pub out_proj: Linear,
    pub e: usize,
    pub k: usize,
}

impl SsmLayer {
    pub fn new(g: &mut Graph, rng: &mut Rng, name: &str, cfg: &LmConfig, depth_scale: f32) -> SsmLayer {
        let e = cfg.inner();
        let in_proj = Linear::new(g, rng, &format!("{}.in_proj", name), cfg.d_model, 3 * e, true, init_std(cfg.d_model));

        // a = exp(-1/tau) => z = logit(a), with tau log-spaced over
        // [1, tau_max]. This gives useful short and long memory at step zero.
        let bias_id = match in_proj.b {
            Some(id) => id,
            None => panic!("in_proj must have a bias"),
        };
        let tau_max = cfg.tau_max.max(2.0);
        for j in 0..e {
            let fraction = if e > 1 { (j as f32) / ((e - 1) as f32) } else { 0.0 };
            let tau = tau_max.powf(fraction);
            let decay = (-1.0f32 / tau).exp().clamp(0.001, 0.999_9);
            g.val[bias_id][e + j] = (decay / (1.0 - decay)).ln();
        }

        // Depthwise conv starts near a delta: identity at the current token,
        // with small causal taps behind it.
        let conv_len = cfg.conv_k.checked_mul(e).expect("convolution parameter size overflow");
        let mut conv_values = vec![0.0f32; conv_len];
        for j in 0..e {
            conv_values[j] = 1.0;
        }
        for q in 1..cfg.conv_k {
            for j in 0..e {
                conv_values[q * e + j] = rng.normal() * 0.1;
            }
        }
        let conv_w = g.param(&format!("{}.conv_w", name), vec![cfg.conv_k, e], conv_values, true);
        let conv_b = g.param(&format!("{}.conv_b", name), vec![e], vec![0.0f32; e], false);
        let out_proj = Linear::new(
            g,
            rng,
            &format!("{}.out_proj", name),
            e,
            cfg.d_model,
            true,
            init_std(e) * depth_scale,
        );
        SsmLayer { in_proj, conv_w, conv_b, out_proj, e, k: cfg.conv_k }
    }

    pub fn forward(&self, g: &mut Graph, x: Nid, batch: usize, t: usize) -> Nid {
        let rows = batch.checked_mul(t).expect("sequence row count overflow");
        let e = self.e;
        let u = self.in_proj.forward(g, x, rows);
        let v = g.slice_cols(u, rows, 3 * e, 0, e);
        let z = g.slice_cols(u, rows, 3 * e, e, e);
        let o = g.slice_cols(u, rows, 3 * e, 2 * e, e);
        let convolved = g.dwconv(v, self.conv_w, self.conv_b, batch, t, e, self.k);
        let value = g.silu(convolved);
        let decay = g.sigmoid(z);
        let update = g.one_minus(decay);
        let input = g.mul(update, value);
        let state = g.scan(decay, input, batch, t, e);
        let output_gate = g.silu(o);
        let gated = g.mul(state, output_gate);
        self.out_proj.forward(g, gated, rows)
    }
}

pub struct Block {
    pub norm1: RmsNorm,
    pub ssm: SsmLayer,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
}

pub struct Lm {
    pub cfg: LmConfig,
    pub emb: Nid,
    pub blocks: Vec<Block>,
    pub norm_f: RmsNorm,
}

impl Lm {
    pub fn new(g: &mut Graph, rng: &mut Rng, cfg: LmConfig) -> Lm {
        cfg.validate();
        let embedding_len = cfg.vocab.checked_mul(cfg.d_model).expect("embedding table is too large");
        let mut embedding = vec![0.0f32; embedding_len];
        for value in &mut embedding {
            *value = rng.normal() * 0.02;
        }
        let emb = g.param("emb", vec![cfg.vocab, cfg.d_model], embedding, true);

        // Residual-branch shrink keeps residual variance roughly constant as
        // depth grows.
        let depth_scale = 1.0f32 / (2.0 * cfg.n_layer as f32).sqrt();
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for layer in 0..cfg.n_layer {
            let name = format!("blk{}", layer);
            let norm1 = RmsNorm::new(g, &format!("{}.n1", name), cfg.d_model, cfg.eps);
            let ssm = SsmLayer::new(g, rng, &format!("{}.ssm", name), &cfg, depth_scale);
            let norm2 = RmsNorm::new(g, &format!("{}.n2", name), cfg.d_model, cfg.eps);
            let mlp = SwiGlu::new(g, rng, &format!("{}.mlp", name), cfg.d_model, cfg.hidden(), depth_scale);
            blocks.push(Block { norm1, ssm, norm2, mlp });
        }
        let norm_f = RmsNorm::new(g, "norm_f", cfg.d_model, cfg.eps);
        Lm { cfg, emb, blocks, norm_f }
    }

    /// ids: [batch * t] token ids -> logits [batch * t, vocab]
    pub fn logits(&self, g: &mut Graph, ids: &[u32], batch: usize, t: usize) -> Nid {
        assert!(batch > 0, "batch size must be positive");
        assert!(t > 0, "sequence length must be positive");
        let rows = batch.checked_mul(t).expect("sequence row count overflow");
        assert_eq!(ids.len(), rows, "logits: ids length");
        let mut x = g.embed(self.emb, self.cfg.d_model, ids);
        for block in &self.blocks {
            let h1 = block.norm1.forward(g, x, rows);
            let recurrent = block.ssm.forward(g, h1, batch, t);
            x = g.add(x, recurrent);
            let h2 = block.norm2.forward(g, x, rows);
            let feed_forward = block.mlp.forward(g, h2, rows);
            x = g.add(x, feed_forward);
        }
        let normalized = self.norm_f.forward(g, x, rows);
        // Weight tying: the embedding matrix is the output head.
        g.matmul_nt(normalized, self.emb, rows, self.cfg.d_model, self.cfg.vocab)
    }

    /// Returns (logits, mean cross-entropy loss in nats).
    pub fn loss(&self, g: &mut Graph, ids: &[u32], targets: &[u32], batch: usize, t: usize) -> (Nid, Nid) {
        let logits = self.logits(g, ids, batch, t);
        let rows = batch.checked_mul(t).expect("sequence row count overflow");
        let loss = g.softmax_ce(logits, rows, self.cfg.vocab, targets);
        (logits, loss)
    }
}
