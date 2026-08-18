//! Optimisers + LR schedule. Written directly against the parameter arena, so
//! a step is a flat pass over contiguous f32 buffers.

use crate::autograd::Graph;
use crate::ckpt::{Ckpt, AUX_PREFIX};

/// AdamW: Adam with *decoupled* weight decay (decay is applied to the weight,
/// not folded into the gradient, so it does not interact with the adaptive
/// denominator). Biases and norm gains are excluded via `Param::decay`.
pub struct AdamW {
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
    pub t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(g: &Graph, wd: f32) -> AdamW {
        assert!(wd.is_finite() && wd >= 0.0, "AdamW weight decay must be finite and non-negative");
        let mut m = Vec::with_capacity(g.params.len());
        let mut v = Vec::with_capacity(g.params.len());
        for p in 0..g.params.len() {
            let n = g.val[g.params[p].id].len();
            m.push(vec![0.0f32; n]);
            v.push(vec![0.0f32; n]);
        }
        AdamW { b1: 0.9, b2: 0.95, eps: 1e-8, wd, t: 0, m, v }
    }

    pub fn step(&mut self, g: &mut Graph, lr: f32) {
        assert!(lr.is_finite() && lr >= 0.0, "AdamW learning rate must be finite and non-negative");
        assert!(self.b1.is_finite() && self.b1 >= 0.0 && self.b1 < 1.0, "AdamW beta1 must be in [0, 1)");
        assert!(self.b2.is_finite() && self.b2 >= 0.0 && self.b2 < 1.0, "AdamW beta2 must be in [0, 1)");
        assert!(self.eps.is_finite() && self.eps > 0.0, "AdamW epsilon must be finite and positive");
        assert!(self.wd.is_finite() && self.wd >= 0.0, "AdamW weight decay must be finite and non-negative");
        assert_eq!(self.m.len(), g.params.len(), "AdamW parameter set changed after construction");
        assert_eq!(self.v.len(), g.params.len(), "AdamW parameter set changed after construction");
        self.t = self.t.checked_add(1).expect("AdamW step counter overflow");
        let tf = self.t as f32;
        let bc1 = 1.0 - self.b1.powf(tf);
        let bc2 = 1.0 - self.b2.powf(tf);
        let inv_bc1 = 1.0 / bc1;
        let inv_bc2 = 1.0 / bc2;
        for p in 0..g.params.len() {
            let id = g.params[p].id;
            let decay = g.params[p].decay;
            let n = g.val[id].len();
            assert_eq!(self.m[p].len(), n, "AdamW first-moment shape mismatch");
            assert_eq!(self.v[p].len(), n, "AdamW second-moment shape mismatch");
            for i in 0..n {
                let gr = g.grad[id][i];
                assert!(gr.is_finite(), "AdamW received a non-finite gradient");
                assert!(g.val[id][i].is_finite(), "AdamW received a non-finite parameter");
                let mi = self.b1 * self.m[p][i] + (1.0 - self.b1) * gr;
                let vi = self.b2 * self.v[p][i] + (1.0 - self.b2) * gr * gr;
                self.m[p][i] = mi;
                self.v[p][i] = vi;
                let mh = mi * inv_bc1;
                let vh = vi * inv_bc2;
                let mut upd = mh / (vh.sqrt() + self.eps);
                if decay && self.wd != 0.0 {
                    upd += self.wd * g.val[id][i];
                }
                g.val[id][i] -= lr * upd;
            }
        }
    }

    /// Moment buffers as checkpointable tensors. Without these a resumed run
    /// restarts Adam from zero momentum, which visibly dents the loss curve.
    pub fn state_tensors(&self, g: &Graph) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::with_capacity(self.m.len() * 2);
        for p in 0..g.params.len() {
            let name = &g.params[p].name;
            let len = self.m[p].len();
            out.push((format!("{}adamw.m.{}", AUX_PREFIX, name), vec![len], self.m[p].clone()));
            out.push((format!("{}adamw.v.{}", AUX_PREFIX, name), vec![len], self.v[p].clone()));
        }
        out
    }

    /// Restore moments saved by [`AdamW::state_tensors`]. Returns `false` when
    /// the checkpoint carries no optimizer state; errors when it carries state
    /// that does not fit the live parameter set.
    pub fn load_state(&mut self, g: &Graph, ckpt: &Ckpt, steps_key: &str) -> Result<bool, String> {
        let first = format!("{}adamw.m.{}", AUX_PREFIX, g.params[0].name);
        if !ckpt.tensors.contains_key(&first) {
            return Ok(false);
        }
        for p in 0..g.params.len() {
            let name = &g.params[p].name;
            let m_key = format!("{}adamw.m.{}", AUX_PREFIX, name);
            let v_key = format!("{}adamw.v.{}", AUX_PREFIX, name);
            let m = ckpt.tensors.get(&m_key).ok_or_else(|| format!("checkpoint is missing optimizer state '{}'", m_key))?;
            let v = ckpt.tensors.get(&v_key).ok_or_else(|| format!("checkpoint is missing optimizer state '{}'", v_key))?;
            if m.len() != self.m[p].len() || v.len() != self.v[p].len() {
                return Err(format!("optimizer state for '{}' has the wrong length", name));
            }
            self.m[p].copy_from_slice(m);
            self.v[p].copy_from_slice(v);
        }
        self.t = ckpt.meta.get(steps_key).and_then(|value| value.parse::<u64>().ok()).unwrap_or(0);
        Ok(true)
    }
}

/// Lion: sign-of-momentum updates. One state buffer instead of two, and the
/// update magnitude is exactly `lr` for every weight, which makes it very
/// robust on small/noisy batches. Kept as an alternative to AdamW.
pub struct Lion {
    pub b1: f32,
    pub b2: f32,
    pub wd: f32,
    m: Vec<Vec<f32>>,
}

impl Lion {
    pub fn new(g: &Graph, wd: f32) -> Lion {
        assert!(wd.is_finite() && wd >= 0.0, "Lion weight decay must be finite and non-negative");
        let mut m = Vec::with_capacity(g.params.len());
        for p in 0..g.params.len() {
            let n = g.val[g.params[p].id].len();
            m.push(vec![0.0f32; n]);
        }
        Lion { b1: 0.9, b2: 0.99, wd, m }
    }

    pub fn step(&mut self, g: &mut Graph, lr: f32) {
        assert!(lr.is_finite() && lr >= 0.0, "Lion learning rate must be finite and non-negative");
        assert!(self.b1.is_finite() && self.b1 >= 0.0 && self.b1 < 1.0, "Lion beta1 must be in [0, 1)");
        assert!(self.b2.is_finite() && self.b2 >= 0.0 && self.b2 < 1.0, "Lion beta2 must be in [0, 1)");
        assert!(self.wd.is_finite() && self.wd >= 0.0, "Lion weight decay must be finite and non-negative");
        assert_eq!(self.m.len(), g.params.len(), "Lion parameter set changed after construction");
        for p in 0..g.params.len() {
            let id = g.params[p].id;
            let decay = g.params[p].decay;
            let n = g.val[id].len();
            assert_eq!(self.m[p].len(), n, "Lion momentum shape mismatch");
            for i in 0..n {
                let gr = g.grad[id][i];
                assert!(gr.is_finite(), "Lion received a non-finite gradient");
                assert!(g.val[id][i].is_finite(), "Lion received a non-finite parameter");
                let c = self.b1 * self.m[p][i] + (1.0 - self.b1) * gr;
                let s = if c > 0.0 {
                    1.0f32
                } else if c < 0.0 {
                    -1.0f32
                } else {
                    0.0f32
                };
                let mut upd = s;
                if decay && self.wd != 0.0 {
                    upd += self.wd * g.val[id][i];
                }
                g.val[id][i] -= lr * upd;
                self.m[p][i] = self.b2 * self.m[p][i] + (1.0 - self.b2) * gr;
            }
        }
    }
}

/// Linear warmup then cosine decay to `min_lr`.
#[derive(Clone, Copy)]
pub struct Schedule {
    pub peak: f32,
    pub min: f32,
    pub warmup: usize,
    pub total: usize,
}

impl Schedule {
    pub fn lr(&self, step: usize) -> f32 {
        assert!(self.peak.is_finite() && self.peak >= 0.0, "schedule peak learning rate must be finite and non-negative");
        assert!(self.min.is_finite() && self.min >= 0.0, "schedule minimum learning rate must be finite and non-negative");
        assert!(self.total > 0, "schedule total step count must be positive");
        if self.warmup > 0 && step < self.warmup {
            return self.peak * ((step + 1) as f32) / (self.warmup as f32);
        }
        let span = if self.total > self.warmup { self.total - self.warmup } else { 1 };
        let frac = (((step - self.warmup) as f32) / (span as f32)).clamp(0.0, 1.0);
        let pi = std::f32::consts::PI;
        self.min + 0.5 * (self.peak - self.min) * (1.0 + (pi * frac).cos())
    }
}
