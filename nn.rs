//! Layer primitives built on the tape. Every layer is a plain struct holding
//! parameter ids - no trait objects, no `Box<dyn Layer>`, so the compiler can
//! see straight through the whole forward pass.

use crate::autograd::{Graph, Nid};
use crate::rng::Rng;

pub fn fill_normal(rng: &mut Rng, n: usize, std: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    for i in 0..n {
        v[i] = rng.normal() * std;
    }
    v
}

/// Kaiming-style scale for a fan-in of `fan_in`.
pub fn init_std(fan_in: usize) -> f32 {
    (2.0f32 / (fan_in.max(1) as f32)).sqrt()
}

/// y = x W^T + b, weights stored [out, in].
pub struct Linear {
    pub w: Nid,
    pub b: Option<Nid>,
    pub din: usize,
    pub dout: usize,
}

impl Linear {
    pub fn new(g: &mut Graph, rng: &mut Rng, name: &str, din: usize, dout: usize, bias: bool, std: f32) -> Linear {
        let w = fill_normal(rng, dout * din, std);
        let wid = g.param(&format!("{}.w", name), vec![dout, din], w, true);
        let bid = if bias {
            Some(g.param(&format!("{}.b", name), vec![dout], vec![0.0f32; dout], false))
        } else {
            None
        };
        Linear { w: wid, b: bid, din, dout }
    }

    pub fn forward(&self, g: &mut Graph, x: Nid, rows: usize) -> Nid {
        let y = g.matmul_nt(x, self.w, rows, self.din, self.dout);
        match self.b {
            Some(b) => g.add_row(y, b),
            None => y,
        }
    }
}

/// RMSNorm + learned per-channel gain. Cheaper and more stable than LayerNorm
/// (no mean, no bias, scale-invariant to the residual stream's drift).
pub struct RmsNorm {
    pub g: Nid,
    pub d: usize,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(g: &mut Graph, name: &str, d: usize, eps: f32) -> RmsNorm {
        let gid = g.param(&format!("{}.gain", name), vec![d], vec![1.0f32; d], false);
        RmsNorm { g: gid, d, eps }
    }

    pub fn forward(&self, g: &mut Graph, x: Nid, rows: usize) -> Nid {
        let n = g.rms_norm(x, rows, self.d, self.eps);
        g.mul_row(n, self.g)
    }
}

/// SwiGLU feed-forward: one fused up-projection, split into value and gate.
pub struct SwiGlu {
    pub up: Linear,
    pub down: Linear,
    pub hidden: usize,
}

impl SwiGlu {
    pub fn new(g: &mut Graph, rng: &mut Rng, name: &str, d: usize, hidden: usize, depth_scale: f32) -> SwiGlu {
        let up = Linear::new(g, rng, &format!("{}.up", name), d, 2 * hidden, true, init_std(d));
        let down = Linear::new(g, rng, &format!("{}.down", name), hidden, d, true, init_std(hidden) * depth_scale);
        SwiGlu { up, down, hidden }
    }

    pub fn forward(&self, g: &mut Graph, x: Nid, rows: usize) -> Nid {
        let u = self.up.forward(g, x, rows);
        let a = g.slice_cols(u, rows, 2 * self.hidden, 0, self.hidden);
        let b = g.slice_cols(u, rows, 2 * self.hidden, self.hidden, self.hidden);
        let act = g.silu(a);
        let gated = g.mul(act, b);
        self.down.forward(g, gated, rows)
    }
}
