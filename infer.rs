//! O(1)-per-token streaming decoder.
//!
//! Generation carries a fixed recurrent state and a circular convolution
//! buffer per layer. Cost and state size do not grow with context length.
//! `selftest` checks streaming logits against the batched forward pass.

use crate::autograd::Graph;
use crate::model::{Lm, LmConfig};
use crate::rng::Rng;
use crate::tensor::{matvec_nt, rms_norm_vec, sigmoid, silu, softmax_inplace};

pub struct LayerState {
    /// Circular buffer of post-projection values, laid out [conv_k, inner].
    pub conv: Vec<f32>,
    /// Slot containing the newest convolution input.
    pub conv_head: usize,
    /// Recurrent state.
    pub h: Vec<f32>,
}

pub struct LmState {
    pub layers: Vec<LayerState>,
}

impl LmState {
    pub fn new(cfg: &LmConfig) -> LmState {
        assert!(cfg.d_model > 0, "d_model must be positive");
        assert!(cfg.expand > 0, "expand must be positive");
        assert!(cfg.conv_k > 0, "conv_k must be positive");
        let e = cfg.inner();
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(LayerState {
                conv: vec![0.0f32; cfg.conv_k * e],
                conv_head: cfg.conv_k - 1,
                h: vec![0.0f32; e],
            });
        }
        LmState { layers }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.conv.fill(0.0);
            layer.h.fill(0.0);
            let k = layer.conv.len() / layer.h.len();
            layer.conv_head = k - 1;
        }
    }
}

/// Advance the model by one token, mutating `st`. Returns logits over vocab.
pub fn step(g: &Graph, m: &Lm, st: &mut LmState, token: u32) -> Vec<f32> {
    let cfg = m.cfg;
    let d = cfg.d_model;
    let e = cfg.inner();
    let k = cfg.conv_k;
    let hidden = cfg.hidden();
    let threads = g.threads;

    assert!((token as usize) < cfg.vocab, "streaming token is outside the vocabulary");
    assert_eq!(st.layers.len(), cfg.n_layer, "streaming state has the wrong layer count");

    let mut x = vec![0.0f32; d];
    let base = (token as usize) * d;
    x.copy_from_slice(&g.val[m.emb][base..base + d]);

    // Scratch is allocated once per token rather than once per layer.
    let mut norm = vec![0.0f32; d];
    let mut projection = vec![0.0f32; 3 * e];
    let mut conv_value = vec![0.0f32; e];
    let mut gated_state = vec![0.0f32; e];
    let mut residual = vec![0.0f32; d];
    let mut up = vec![0.0f32; 2 * hidden];
    let mut gated_mlp = vec![0.0f32; hidden];

    for layer_index in 0..cfg.n_layer {
        let block = &m.blocks[layer_index];
        let state = &mut st.layers[layer_index];
        assert_eq!(state.h.len(), e, "streaming recurrent state has the wrong width");
        assert_eq!(state.conv.len(), k * e, "streaming convolution state has the wrong shape");

        // ---- recurrent branch ----
        rms_norm_vec(&x, &g.val[block.norm1.g], cfg.eps, &mut norm);
        let in_bias = block.ssm.in_proj.b.map(|id| &g.val[id][..]);
        matvec_nt(
            &g.val[block.ssm.in_proj.w],
            in_bias,
            &norm,
            &mut projection,
            3 * e,
            d,
            threads,
        );

        // Store the newest value in a circular buffer. This replaces the old
        // O(k*e) per-token shift with an O(e) copy.
        state.conv_head = (state.conv_head + 1) % k;
        let current = state.conv_head * e;
        state.conv[current..current + e].copy_from_slice(&projection[..e]);

        let conv_weights = &g.val[block.ssm.conv_w];
        let conv_bias = &g.val[block.ssm.conv_b];
        for j in 0..e {
            let mut sum = conv_bias[j];
            for q in 0..k {
                let slot = (state.conv_head + k - q) % k;
                sum += conv_weights[q * e + j] * state.conv[slot * e + j];
            }
            conv_value[j] = silu(sum);
        }

        for j in 0..e {
            let decay = sigmoid(projection[e + j]);
            state.h[j] = decay * state.h[j] + (1.0 - decay) * conv_value[j];
            gated_state[j] = state.h[j] * silu(projection[2 * e + j]);
        }

        let out_bias = block.ssm.out_proj.b.map(|id| &g.val[id][..]);
        matvec_nt(
            &g.val[block.ssm.out_proj.w],
            out_bias,
            &gated_state,
            &mut residual,
            d,
            e,
            threads,
        );
        for j in 0..d {
            x[j] += residual[j];
        }

        // ---- feed-forward branch ----
        rms_norm_vec(&x, &g.val[block.norm2.g], cfg.eps, &mut norm);
        let up_bias = block.mlp.up.b.map(|id| &g.val[id][..]);
        matvec_nt(
            &g.val[block.mlp.up.w],
            up_bias,
            &norm,
            &mut up,
            2 * hidden,
            d,
            threads,
        );
        for j in 0..hidden {
            gated_mlp[j] = silu(up[j]) * up[hidden + j];
        }
        let down_bias = block.mlp.down.b.map(|id| &g.val[id][..]);
        matvec_nt(
            &g.val[block.mlp.down.w],
            down_bias,
            &gated_mlp,
            &mut residual,
            d,
            hidden,
            threads,
        );
        for j in 0..d {
            x[j] += residual[j];
        }
    }

    rms_norm_vec(&x, &g.val[m.norm_f.g], cfg.eps, &mut norm);
    let mut logits = vec![0.0f32; cfg.vocab];
    matvec_nt(&g.val[m.emb], None, &norm, &mut logits, cfg.vocab, d, threads);
    logits
}

/// Decoding controls. All are pure post-processing on the logit vector.
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
pub fn sample_token(logits: &mut [f32], cfg: &SampleCfg, history: &[u32], rng: &mut Rng) -> u32 {
    let vocab = logits.len();
    assert!(vocab > 0, "cannot sample from an empty vocabulary");

    // Penalize each recently seen token once. Applying the penalty once per
    // occurrence made repeated runs exponentially over-penalized.
    if cfg.rep_penalty.is_finite() && cfg.rep_penalty > 1.0 && cfg.rep_window > 0 {
        let start = history.len().saturating_sub(cfg.rep_window);
        for i in start..history.len() {
            let token = history[i] as usize;
            if token >= vocab || history[start..i].contains(&history[i]) {
                continue;
            }
            if logits[token] > 0.0 {
                logits[token] /= cfg.rep_penalty;
            } else {
                logits[token] *= cfg.rep_penalty;
            }
        }
    }
    if cfg.greedy {
        return crate::tensor::argmax(logits) as u32;
    }

    let temperature = if cfg.temperature.is_finite() && cfg.temperature > 1e-4 {
        cfg.temperature
    } else {
        1e-4
    };
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
    let mut probabilities = logits.to_vec();
    softmax_inplace(&mut probabilities);

    let keep = if cfg.top_k == 0 || cfg.top_k > vocab {
        vocab
    } else {
        cfg.top_k
    };
    let mut indices: Vec<usize> = (0..vocab).collect();
    indices.sort_unstable_by(|&left, &right| probabilities[right].total_cmp(&probabilities[left]));

    let top_p = if cfg.top_p.is_finite() { cfg.top_p } else { 1.0 };
    let mut cumulative = 0.0f32;
    let mut kept = 0usize;
    for &index in indices.iter().take(keep) {
        cumulative += probabilities[index];
        kept += 1;
        if top_p > 0.0 && top_p < 1.0 && cumulative >= top_p {
            break;
        }
    }

    let mut weights = Vec::with_capacity(kept);
    for &index in indices.iter().take(kept) {
        weights.push(probabilities[index]);
    }
    indices[rng.categorical(&weights)] as u32
}
