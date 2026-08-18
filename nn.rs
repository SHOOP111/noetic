//! Layer primitives built on the tape. Every layer is a plain struct holding
//! parameter ids - no trait objects, no `Box<dyn Layer>`, so the compiler can
//! see straight through the whole forward pass.

use crate::autograd::{Graph, Nid};
use crate::rng::Rng;

pub fn fill_normal(rng: &mut Rng, n: usize, std: f32) -> Vec<f32> {
    assert!(std.is_finite() && std >= 0.0, "initialization scale must be finite and non-negative");
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
        assert!(din > 0, "linear input width must be positive");
        assert!(dout > 0, "linear output width must be positive");
        let elements = dout.checked_mul(din).expect("linear weight size overflow");
        let w = fill_normal(rng, elements, std);
        let wid = g.param(&format!("{}.w", name), vec![dout, din], w, true);
        let bid = if bias { Some(g.param(&format!("{}.b", name), vec![dout], vec![0.0f32; dout], false)) } else { None };
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
        assert!(d > 0, "RMSNorm width must be positive");
        assert!(eps.is_finite() && eps > 0.0, "RMSNorm epsilon must be finite and positive");
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
        assert!(d > 0, "SwiGLU input width must be positive");
        assert!(hidden > 0, "SwiGLU hidden width must be positive");
        assert!(depth_scale.is_finite() && depth_scale >= 0.0, "residual depth scale must be finite and non-negative");
        let projected = hidden.checked_mul(2).expect("SwiGLU projection width overflow");
        let up = Linear::new(g, rng, &format!("{}.up", name), d, projected, true, init_std(d));
        let down = Linear::new(g, rng, &format!("{}.down", name), hidden, d, true, init_std(hidden) * depth_scale);
        SwiGlu { up, down, hidden }
    }

    pub fn forward(&self, g: &mut Graph, x: Nid, rows: usize) -> Nid {
        let u = self.up.forward(g, x, rows);
        let projected = self.hidden.checked_mul(2).expect("SwiGLU projection width overflow");
        let a = g.slice_cols(u, rows, projected, 0, self.hidden);
        let b = g.slice_cols(u, rows, projected, self.hidden, self.hidden);
        let act = g.silu(a);
        let gated = g.mul(act, b);
        self.down.forward(g, gated, rows)
    }
}
