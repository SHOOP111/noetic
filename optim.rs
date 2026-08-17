//! Optimisers + LR schedule. Written directly against the parameter arena, so
//! a step is a flat pass over contiguous f32 buffers.

use crate::autograd::Graph;

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
        self.t += 1;
        let tf = self.t as f32;
        let bc1 = 1.0 - self.b1.powf(tf);
        let bc2 = 1.0 - self.b2.powf(tf);
        let inv_bc1 = 1.0 / bc1;
        let inv_bc2 = 1.0 / bc2;
        for p in 0..g.params.len() {
            let id = g.params[p].id;
            let decay = g.params[p].decay;
            let n = g.val[id].len();
            for i in 0..n {
                let gr = g.grad[id][i];
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
        let mut m = Vec::with_capacity(g.params.len());
        for p in 0..g.params.len() {
            let n = g.val[g.params[p].id].len();
            m.push(vec![0.0f32; n]);
        }
        Lion { b1: 0.9, b2: 0.99, wd, m }
    }

    pub fn step(&mut self, g: &mut Graph, lr: f32) {
        for p in 0..g.params.len() {
            let id = g.params[p].id;
            let decay = g.params[p].decay;
            let n = g.val[id].len();
            for i in 0..n {
                let gr = g.grad[id][i];
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
        if self.warmup > 0 && step < self.warmup {
            return self.peak * ((step + 1) as f32) / (self.warmup as f32);
        }
        let span = if self.total > self.warmup { self.total - self.warmup } else { 1 };
        let mut frac = ((step - self.warmup) as f32) / (span as f32);
        if frac < 0.0 {
            frac = 0.0;
        }
        if frac > 1.0 {
            frac = 1.0;
        }
        let pi = std::f32::consts::PI;
        self.min + 0.5 * (self.peak - self.min) * (1.0 + (pi * frac).cos())
    }
}
