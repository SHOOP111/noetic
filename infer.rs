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
        cfg.validate();
        let e = cfg.inner();
        let conv_len = cfg.conv_k.checked_mul(e).expect("streaming convolution state is too large");
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(LayerState {
                conv: vec![0.0f32; conv_len],
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

struct DecoderScratch {
    x: Vec<f32>,
    norm: Vec<f32>,
    projection: Vec<f32>,
    conv_value: Vec<f32>,
    gated_state: Vec<f32>,
    residual: Vec<f32>,
    up: Vec<f32>,
    gated_mlp: Vec<f32>,
    logits: Vec<f32>,
}

impl DecoderScratch {
    fn new(cfg: &LmConfig) -> DecoderScratch {
        cfg.validate();
        let d = cfg.d_model;
        let e = cfg.inner();
        let hidden = cfg.hidden();
        DecoderScratch {
            x: vec![0.0; d],
            norm: vec![0.0; d],
            projection: vec![0.0; cfg.recurrent_projection()],
            conv_value: vec![0.0; e],
            gated_state: vec![0.0; e],
            residual: vec![0.0; d],
            up: vec![0.0; cfg.mlp_projection()],
            gated_mlp: vec![0.0; hidden],
            logits: vec![0.0; cfg.vocab],
        }
    }
}

/// Stateful streaming decoder that reuses every scratch buffer across tokens.
pub struct Decoder {
    pub state: LmState,
    scratch: DecoderScratch,
}

impl Decoder {
    pub fn new(cfg: &LmConfig) -> Decoder {
        Decoder { state: LmState::new(cfg), scratch: DecoderScratch::new(cfg) }
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn logits_mut(&mut self) -> &mut [f32] {
        &mut self.scratch.logits
    }

    /// Advance by one token and return the reusable logit buffer. The returned
    /// slice is overwritten by the next call.
    pub fn step<'a>(&'a mut self, g: &Graph, m: &Lm, token: u32) -> &'a mut [f32] {
        step_with_scratch(g, m, &mut self.state, &mut self.scratch, token);
        &mut self.scratch.logits
    }
}

fn step_with_scratch(g: &Graph, m: &Lm, st: &mut LmState, scratch: &mut DecoderScratch, token: u32) {
    let cfg = m.cfg;
    let d = cfg.d_model;
    let e = cfg.inner();
    let k = cfg.conv_k;
    let hidden = cfg.hidden();
    let recurrent_projection = cfg.recurrent_projection();
    let mlp_projection = cfg.mlp_projection();
    let threads = g.threads;

    assert!((token as usize) < cfg.vocab, "streaming token is outside the vocabulary");
    assert_eq!(st.layers.len(), cfg.n_layer, "streaming state has the wrong layer count");
    assert_eq!(scratch.x.len(), d, "decoder scratch has the wrong model width");
    assert_eq!(scratch.norm.len(), d, "decoder norm scratch has the wrong width");
    assert_eq!(scratch.projection.len(), recurrent_projection, "decoder projection scratch has the wrong width");
    assert_eq!(scratch.conv_value.len(), e, "decoder convolution scratch has the wrong width");
    assert_eq!(scratch.gated_state.len(), e, "decoder recurrent scratch has the wrong width");
    assert_eq!(scratch.residual.len(), d, "decoder residual scratch has the wrong width");
    assert_eq!(scratch.up.len(), mlp_projection, "decoder MLP scratch has the wrong width");
    assert_eq!(scratch.gated_mlp.len(), hidden, "decoder gated MLP scratch has the wrong width");
    assert_eq!(scratch.logits.len(), cfg.vocab, "decoder logit scratch has the wrong vocabulary");

    let DecoderScratch {
        x,
        norm,
        projection,
        conv_value,
        gated_state,
        residual,
        up,
        gated_mlp,
        logits,
    } = scratch;
    let base = (token as usize) * d;
    x.copy_from_slice(&g.val[m.emb][base..base + d]);

    for layer_index in 0..cfg.n_layer {
        let block = &m.blocks[layer_index];
        let state = &mut st.layers[layer_index];
        assert_eq!(state.h.len(), e, "streaming recurrent state has the wrong width");
        assert_eq!(
            state.conv.len(),
            k.checked_mul(e).expect("streaming convolution size overflow"),
            "streaming convolution state has the wrong shape"
        );

        // ---- recurrent branch ----
        rms_norm_vec(x.as_slice(), &g.val[block.norm1.g], cfg.eps, norm.as_mut_slice());
        let in_bias = block.ssm.in_proj.b.map(|id| &g.val[id][..]);
        matvec_nt(
            &g.val[block.ssm.in_proj.w],
            in_bias,
            norm.as_slice(),
            projection.as_mut_slice(),
            recurrent_projection,
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
            gated_state.as_slice(),
            residual.as_mut_slice(),
            d,
            e,
            threads,
        );
        for j in 0..d {
            x[j] += residual[j];
        }

        // ---- feed-forward branch ----
        rms_norm_vec(x.as_slice(), &g.val[block.norm2.g], cfg.eps, norm.as_mut_slice());
        let up_bias = block.mlp.up.b.map(|id| &g.val[id][..]);
        matvec_nt(
            &g.val[block.mlp.up.w],
            up_bias,
            norm.as_slice(),
            up.as_mut_slice(),
            mlp_projection,
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
            gated_mlp.as_slice(),
            residual.as_mut_slice(),
            d,
            hidden,
            threads,
        );
        for j in 0..d {
            x[j] += residual[j];
        }
    }

    rms_norm_vec(x.as_slice(), &g.val[m.norm_f.g], cfg.eps, norm.as_mut_slice());
    matvec_nt(
        &g.val[m.emb],
        None,
        norm.as_slice(),
        logits.as_mut_slice(),
        cfg.vocab,
        d,
        threads,
    );
}

/// Advance the model by one token, mutating `st`. Returns logits over vocab.
/// Prefer [`Decoder`] when processing multiple tokens so scratch is reused.
pub fn step(g: &Graph, m: &Lm, st: &mut LmState, token: u32) -> Vec<f32> {
    let mut scratch = DecoderScratch::new(&m.cfg);
    step_with_scratch(g, m, st, &mut scratch, token);
    scratch.logits
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
    softmax_inplace(logits);

    let keep = if cfg.top_k == 0 || cfg.top_k > vocab {
        vocab
    } else {
        cfg.top_k
    };
    let mut indices: Vec<usize> = (0..vocab).collect();
    if keep < vocab {
        indices.select_nth_unstable_by(keep, |left, right| {
            let order = logits[*right].total_cmp(&logits[*left]);
            if order == std::cmp::Ordering::Equal { left.cmp(right) } else { order }
        });
        indices.truncate(keep);
    }
    indices.sort_unstable_by(|left, right| {
        let order = logits[*right].total_cmp(&logits[*left]);
        if order == std::cmp::Ordering::Equal { left.cmp(right) } else { order }
    });

    // p <= 0 has the useful deterministic interpretation "keep the best one".
    let top_p = if cfg.top_p.is_finite() { cfg.top_p.clamp(0.0, 1.0) } else { 1.0 };
    let mut cumulative = 0.0f32;
    let mut kept = 0usize;
    for &index in &indices {
        cumulative += logits[index];
        kept += 1;
        if cumulative >= top_p {
            break;
        }
    }

    let candidates = &indices[..kept.max(1)];
    let total = candidates.iter().map(|&index| logits[index] as f64).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return candidates[rng.below(candidates.len())] as u32;
    }
    let mut draw = (rng.f32_unit() as f64) * total;
    for &index in candidates {
        draw -= logits[index] as f64;
        if draw <= 0.0 {
            return index as u32;
        }
    }
    candidates[candidates.len() - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_nucleus_probability_keeps_exactly_the_best_token() {
        let config = SampleCfg {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            rep_penalty: 1.0,
            rep_window: 0,
            greedy: false,
        };
        let mut logits = [f32::NAN, f32::INFINITY, f32::INFINITY, -2.0];
        let mut rng = Rng::new(7);
        assert_eq!(sample_token(&mut logits, &config, &[], &mut rng), 1);
    }
}
