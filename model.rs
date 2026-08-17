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
//!
//! Why this is not a toy RNN:
//!
//! * `a_t` is *input dependent* (selective state, like Mamba's S6) rather than
//!   a fixed matrix, so the layer can choose to remember or forget per token
//!   and per channel.
//! * The `(1-a)` coupling makes it a normalised leaky integrator (minGRU form):
//!   the state is a convex blend of history and input, so activations cannot
//!   blow up with sequence length.
//! * `z`'s bias is initialised to a *log-spaced spectrum of time constants*
//!   (tau in [1, tau_max]): channel 0 forgets in one step, the last channel
//!   integrates over ~tau_max steps. Multi-timescale memory at step 0 instead
//!   of hoping SGD discovers it.
//! * Diagonal state + associative scan => O(T) train, O(1) memory per token at
//!   inference. Unbounded context with a fixed-size state.

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
    pub fn inner(&self) -> usize {
        self.d_model * self.expand
    }
    pub fn hidden(&self) -> usize {
        self.d_model * self.mlp_mult
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

        // ---- decay spectrum init: bias of the z (gate) block ----
        // a = exp(-1/tau) => z = logit(a). tau log-spaced over [1, tau_max].
        let bid = match in_proj.b {
            Some(b) => b,
            None => panic!("in_proj must have a bias"),
        };
        let tau_max = if cfg.tau_max > 2.0 { cfg.tau_max } else { 2.0 };
        for j in 0..e {
            let frac = if e > 1 { (j as f32) / ((e - 1) as f32) } else { 0.0 };
            let tau = tau_max.powf(frac);
            let a = (-1.0f32 / tau).exp();
            let a = a.min(0.999_9).max(0.001);
            g.val[bid][e + j] = (a / (1.0 - a)).ln();
        }

        // depthwise conv initialised near a delta: identity at t, small taps behind
        let mut cw = vec![0.0f32; cfg.conv_k * e];
        for j in 0..e {
            cw[j] = 1.0;
        }
        for q in 1..cfg.conv_k {
            for j in 0..e {
                cw[q * e + j] = rng.normal() * 0.1;
            }
        }
        let conv_w = g.param(&format!("{}.conv_w", name), vec![cfg.conv_k, e], cw, false);
        let conv_b = g.param(&format!("{}.conv_b", name), vec![e], vec![0.0f32; e], false);
        let out_proj = Linear::new(g, rng, &format!("{}.out_proj", name), e, cfg.d_model, true, init_std(e) * depth_scale);
        SsmLayer { in_proj, conv_w, conv_b, out_proj, e, k: cfg.conv_k }
    }

    pub fn forward(&self, g: &mut Graph, x: Nid, batch: usize, t: usize) -> Nid {
        let rows = batch * t;
        let e = self.e;
        let u = self.in_proj.forward(g, x, rows);
        let v = g.slice_cols(u, rows, 3 * e, 0, e);
        let z = g.slice_cols(u, rows, 3 * e, e, e);
        let o = g.slice_cols(u, rows, 3 * e, 2 * e, e);
        let vc = g.dwconv(v, self.conv_w, self.conv_b, batch, t, e, self.k);
        let vs = g.silu(vc);
        let a = g.sigmoid(z);
        let om = g.one_minus(a);
        let b = g.mul(om, vs);
        let h = g.scan(a, b, batch, t, e);
        let og = g.silu(o);
        let y = g.mul(h, og);
        self.out_proj.forward(g, y, rows)
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
        let mut emb_v = vec![0.0f32; cfg.vocab * cfg.d_model];
        for i in 0..emb_v.len() {
            emb_v[i] = rng.normal() * 0.02;
        }
        let emb = g.param("emb", vec![cfg.vocab, cfg.d_model], emb_v, true);
        // residual-branch shrink: keeps the residual stream variance ~constant
        // as depth grows (GPT-2 trick, applies to any deep residual net).
        let depth_scale = 1.0f32 / ((2.0 * (cfg.n_layer as f32)).sqrt());
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let name = format!("blk{}", l);
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
        let rows = batch * t;
        assert_eq!(ids.len(), rows, "logits: ids length");
        let mut x = g.embed(self.emb, self.cfg.d_model, ids);
        for i in 0..self.blocks.len() {
            let h1 = self.blocks[i].norm1.forward(g, x, rows);
            let s = self.blocks[i].ssm.forward(g, h1, batch, t);
            x = g.add(x, s);
            let h2 = self.blocks[i].norm2.forward(g, x, rows);
            let m = self.blocks[i].mlp.forward(g, h2, rows);
            x = g.add(x, m);
        }
        let xf = self.norm_f.forward(g, x, rows);
        // weight tying: the embedding matrix is the output head
        g.matmul_nt(xf, self.emb, rows, self.cfg.d_model, self.cfg.vocab)
    }

    /// Returns (logits, mean cross-entropy loss in nats).
    pub fn loss(&self, g: &mut Graph, ids: &[u32], targets: &[u32], batch: usize, t: usize) -> (Nid, Nid) {
        let logits = self.logits(g, ids, batch, t);
        let rows = batch * t;
        let l = g.softmax_ce(logits, rows, self.cfg.vocab, targets);
        (logits, l)
    }
}
