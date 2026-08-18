//! Deterministic randomness, written from scratch.
//!
//! * `splitmix64`    - seed expansion
//! * `xoshiro256++`  - core stream, period 2^256 - 1
//! * Marsaglia polar - gaussians (with a cached spare)
//! * Marsaglia-Tsang - gamma variates -> Dirichlet (used by the planner)

/// SplitMix64: expands one u64 seed into the 256-bit xoshiro state.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
    spare: f32,
    has_spare: bool,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed ^ 0xA076_1D64_78BD_642F;
        let mut s = [0u64; 4];
        let mut i = 0usize;
        while i < 4 {
            s[i] = splitmix64(&mut z);
            i += 1;
        }
        // Guard against the (astronomically unlikely) all-zero state.
        if s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 {
            s[0] = 0x9E37_79B9_7F4A_7C15;
        }
        Rng { s, spare: 0.0, has_spare: false }
    }

    /// Split off an independent stream (for per-thread / per-worker RNGs).
    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in [0, 1) with 24 bits of mantissa entropy.
    #[inline]
    pub fn f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    #[inline]
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32_unit()
    }

    /// Unbiased integer in [0, n) via rejection sampling.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below requires a positive bound");
        if n == 1 {
            return 0;
        }
        let bound = n as u64;
        // Values below this threshold would make the accepted range size not
        // divisible by `bound`. This is Lemire's standard threshold formula.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return (r % bound) as usize;
            }
        }
    }

    /// Standard normal, Marsaglia polar method. One logarithm yields two draws.
    pub fn normal(&mut self) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        loop {
            let u = self.uniform(-1.0, 1.0);
            let v = self.uniform(-1.0, 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare = v * f;
                self.has_spare = true;
                return u * f;
            }
        }
    }

    /// Gumbel(0,1). `argmax(logits + gumbel)` == sampling from softmax(logits).
    pub fn gumbel(&mut self) -> f32 {
        let u = self.f32_unit().clamp(1e-7, 1.0 - 1e-7);
        -((-u.ln()).ln())
    }

    /// Exponential(1).
    pub fn exponential(&mut self) -> f32 {
        let u = self.f32_unit().max(1e-7);
        -u.ln()
    }

    /// Gamma(shape, 1) via Marsaglia-Tsang, with the `shape < 1` boost.
    pub fn gamma(&mut self, shape: f32) -> f32 {
        if !shape.is_finite() || shape <= 0.0 {
            return 0.0;
        }
        if shape < 1.0 {
            let g = self.gamma(shape + 1.0);
            let u = self.f32_unit().max(1e-7);
            return g * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.f32_unit().max(1e-7);
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v3;
            }
            if u.ln() < 0.5 * x2 + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }

    /// Symmetric Dirichlet(alpha, ..., alpha) of dimension `k`.
    pub fn dirichlet(&mut self, alpha: f32, k: usize) -> Vec<f32> {
        if k == 0 {
            return Vec::new();
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return vec![1.0 / (k as f32); k];
        }
        let mut out = vec![0.0f32; k];
        let mut sum = 0.0f64;
        for i in 0..k {
            let g = self.gamma(alpha);
            out[i] = g;
            sum += g as f64;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return vec![1.0 / (k as f32); k];
        }
        let inv = (1.0 / sum) as f32;
        for i in 0..k {
            out[i] *= inv;
        }
        out
    }

    /// Sample an index from unnormalised non-negative weights.
    pub fn categorical(&mut self, w: &[f32]) -> usize {
        if w.is_empty() {
            return 0;
        }
        let mut total = 0.0f64;
        for &weight in w {
            if weight.is_finite() && weight > 0.0 {
                total += weight as f64;
            }
        }
        if total <= 0.0 {
            return self.below(w.len());
        }
        let mut r = (self.f32_unit() as f64) * total;
        for (i, &weight) in w.iter().enumerate() {
            if weight.is_finite() && weight > 0.0 {
                r -= weight as f64;
                if r <= 0.0 {
                    return i;
                }
            }
        }
        // Floating-point fallback: last positive finite index.
        let mut last = 0usize;
        for (i, &weight) in w.iter().enumerate() {
            if weight.is_finite() && weight > 0.0 {
                last = i;
            }
        }
        last
    }

    /// In-place Fisher-Yates.
    pub fn shuffle_usize(&mut self, v: &mut [usize]) {
        let n = v.len();
        if n < 2 {
            return;
        }
        let mut i = n - 1;
        while i > 0 {
            let j = self.below(i + 1);
            v.swap(i, j);
            i -= 1;
        }
    }
}
