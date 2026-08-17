//! O(1)-per-token streaming decoder.
//!
//! This is the payoff for refusing attention: generation carries a *fixed*
//! state (one h vector + a k-tap conv ring per layer) instead of a KV cache
//! that grows with context. Cost per token and memory per token are constant
//! no matter how long the conversation gets.
//!
//! The math here is the exact scalar-per-timestep specialisation of
//! `model.rs`; `selftest` checks that streaming N tokens equals the batched
//! forward pass over the same N tokens.

use crate::autograd::Graph;
use crate::model::{Lm, LmConfig};
use crate::rng::Rng;
use crate::tensor::{matvec_nt, rms_norm_vec, sigmoid, silu, softmax_inplace};

pub struct LayerState {
    /// conv[q*e + j] = post-projection value at time (t - q)
    pub conv: Vec<f32>,
    /// recurrent state
    pub h: Vec<f32>,
}

pub struct LmState {
    pub layers: Vec<LayerState>,
}

impl LmState {
    pub fn new(cfg: &LmConfig) -> LmState {
        let e = cfg.inner();
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(LayerState { conv: vec![0.0f32; cfg.conv_k * e], h: vec![0.0f32; e] });
        }
        LmState { layers }
    }

    pub fn reset(&mut self) {
        for l in 0..self.layers.len() {
            for x in self.layers[l].conv.iter_mut() {
                *x = 0.0;
            }
            for x in self.layers[l].h.iter_mut() {
                *x = 0.0;
            }
        }
    }
}

/// Advance the model by one token, mutating `st`. Returns logits over vocab.
pub fn step(g: &Graph, m: &Lm, st: &mut LmState, token: u32) -> Vec<f32> {
    let cfg = m.cfg;
    let d = cfg.d_model;
    let e = cfg.inner();
    let k = cfg.conv_k;
    let hid = cfg.hidden();
    let th = g.threads;

    let mut x = vec![0.0f32; d];
    let base = (token as usize) * d;
    for j in 0..d {
        x[j] = g.val[m.emb][base + j];
    }

    let mut nrm = vec![0.0f32; d];
    let mut u = vec![0.0f32; 3 * e];
    let mut vc = vec![0.0f32; e];
    let mut y = vec![0.0f32; e];
    let mut o = vec![0.0f32; d];
    let mut up = vec![0.0f32; 2 * hid];
    let mut gated = vec![0.0f32; hid];

    for l in 0..cfg.n_layer {
        let blk = &m.blocks[l];

        // ---- recurrent branch ----
        rms_norm_vec(&x, &g.val[blk.norm1.g], cfg.eps, &mut nrm);
        let wb = match blk.ssm.in_proj.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(&g.val[blk.ssm.in_proj.w], wb, &nrm, &mut u, 3 * e, d, th);

        // shift the depthwise conv ring buffer, newest at q = 0
        {
            let stl = &mut st.layers[l];
            let mut q = k;
            while q > 1 {
                q -= 1;
                for j in 0..e {
                    let prev = stl.conv[(q - 1) * e + j];
                    stl.conv[q * e + j] = prev;
                }
            }
            for j in 0..e {
                stl.conv[j] = u[j];
            }
        }

        {
            let cw = &g.val[blk.ssm.conv_w];
            let cb = &g.val[blk.ssm.conv_b];
            let stl = &st.layers[l];
            for j in 0..e {
                let mut s = cb[j];
                for q in 0..k {
                    s += cw[q * e + j] * stl.conv[q * e + j];
                }
                vc[j] = silu(s);
            }
        }

        {
            let stl = &mut st.layers[l];
            for j in 0..e {
                let a = sigmoid(u[e + j]);
                stl.h[j] = a * stl.h[j] + (1.0 - a) * vc[j];
                y[j] = stl.h[j] * silu(u[2 * e + j]);
            }
        }

        let ob = match blk.ssm.out_proj.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(&g.val[blk.ssm.out_proj.w], ob, &y, &mut o, d, e, th);
        for j in 0..d {
            x[j] += o[j];
        }

        // ---- feed-forward branch ----
        rms_norm_vec(&x, &g.val[blk.norm2.g], cfg.eps, &mut nrm);
        let ub = match blk.mlp.up.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(&g.val[blk.mlp.up.w], ub, &nrm, &mut up, 2 * hid, d, th);
        for j in 0..hid {
            gated[j] = silu(up[j]) * up[hid + j];
        }
        let db = match blk.mlp.down.b {
            Some(b) => Some(&g.val[b][..]),
            None => None,
        };
        matvec_nt(&g.val[blk.mlp.down.w], db, &gated, &mut o, d, hid, th);
        for j in 0..d {
            x[j] += o[j];
        }
    }

    rms_norm_vec(&x, &g.val[m.norm_f.g], cfg.eps, &mut nrm);
    let mut logits = vec![0.0f32; cfg.vocab];
    matvec_nt(&g.val[m.emb], None, &nrm, &mut logits, cfg.vocab, d, th);
    logits
}

/// Decoding controls. All of these are pure post-processing on the logit vector.
#[derive(Clone, Copy)]
pub struct SampleCfg {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub rep_window: usize,
    pub greedy: bool,
}

impl SampleCfg {
    pub fn default_cfg() -> SampleCfg {
        SampleCfg { temperature: 0.9, top_k: 40, top_p: 0.95, rep_penalty: 1.1, rep_window: 64, greedy: false }
    }
}

/// Repetition penalty -> temperature -> top-k -> nucleus -> categorical draw.
pub fn sample_token(logits: &mut Vec<f32>, cfg: &SampleCfg, history: &[u32], rng: &mut Rng) -> u32 {
    let v = logits.len();
    if cfg.rep_penalty > 1.0 && cfg.rep_window > 0 {
        let start = if history.len() > cfg.rep_window { history.len() - cfg.rep_window } else { 0 };
        for i in start..history.len() {
            let t = history[i] as usize;
            if t < v {
                if logits[t] > 0.0 {
                    logits[t] /= cfg.rep_penalty;
                } else {
                    logits[t] *= cfg.rep_penalty;
                }
            }
        }
    }
    if cfg.greedy {
        return crate::tensor::argmax(logits) as u32;
    }
    let temp = if cfg.temperature > 1e-4 { cfg.temperature } else { 1e-4 };
    for i in 0..v {
        logits[i] /= temp;
    }
    let mut probs = logits.clone();
    softmax_inplace(&mut probs);

    // order indices by descending probability (partial selection sort over the
    // truncation window only: we never need a full sort)
    let keep = if cfg.top_k == 0 || cfg.top_k > v { v } else { cfg.top_k };
    let mut idx: Vec<usize> = (0..v).collect();
    for i in 0..keep {
        let mut best = i;
        for j in (i + 1)..v {
            if probs[idx[j]] > probs[idx[best]] {
                best = j;
            }
        }
        let tmp = idx[i];
        idx[i] = idx[best];
        idx[best] = tmp;
    }

    let mut cum = 0.0f32;
    let mut n_keep = 0usize;
    for i in 0..keep {
        cum += probs[idx[i]];
        n_keep = i + 1;
        if cfg.top_p > 0.0 && cfg.top_p < 1.0 && cum >= cfg.top_p {
            break;
        }
    }

    let mut w = vec![0.0f32; n_keep];
    for i in 0..n_keep {
        w[i] = probs[idx[i]];
    }
    let pick = rng.categorical(&w);
    idx[pick] as u32
}
