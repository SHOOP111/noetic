//! Kanerva Sparse Distributed Memory - a non-gradient, one-shot associative
//! store. Complementary to the network: the model learns slow statistics,
//! this writes fast episodic bindings in a single exposure.
//!
//! Mechanism:
//!
//! * `n_loc` random hard locations live in a `bits`-dimensional Hamming cube.
//! * A write activates every location within `radius` of the address and
//!   increments/decrements one saturating counter per bit.
//! * A read activates the same neighbourhood and thresholds the counter sums.
//!
//! Consequences that fall out of the geometry, not from training:
//!
//! * content-addressable recall from partial or corrupted cues,
//! * graceful degradation instead of catastrophic failure as it fills up,
//! * capacity ~ a few percent of `n_loc` patterns before interference wins.
//!
//! All operations are integer/bitwise: `count_ones` on u64 words.

use crate::rng::Rng;

pub fn words_for(bits: usize) -> usize {
    (bits + 63) / 64
}

pub fn get_bit(w: &[u64], i: usize) -> bool {
    (w[i / 64] >> (i % 64)) & 1 == 1
}

pub fn set_bit(w: &mut [u64], i: usize, v: bool) {
    let m = 1u64 << (i % 64);
    if v {
        w[i / 64] |= m;
    } else {
        w[i / 64] &= !m;
    }
}

pub fn hamming(a: &[u64], b: &[u64]) -> usize {
    let mut d = 0usize;
    for i in 0..a.len() {
        d += (a[i] ^ b[i]).count_ones() as usize;
    }
    d
}

pub fn random_bits(rng: &mut Rng, bits: usize) -> Vec<u64> {
    let nw = words_for(bits);
    let mut v = vec![0u64; nw];
    for i in 0..nw {
        v[i] = rng.next_u64();
    }
    // clear the tail so hamming distances stay meaningful
    let rem = bits % 64;
    if rem != 0 {
        let mask = (1u64 << rem) - 1;
        v[nw - 1] &= mask;
    }
    v
}

pub fn flip_bits(src: &[u64], bits: usize, n_flip: usize, rng: &mut Rng) -> Vec<u64> {
    let mut out = src.to_vec();
    for _ in 0..n_flip {
        let i = rng.below(bits);
        let cur = get_bit(&out, i);
        set_bit(&mut out, i, !cur);
    }
    out
}

/// Sign-random-projection: maps a real vector to a bit address such that
/// Hamming distance approximates angular distance (LSH). This is how dense
/// network activations get bound into the discrete memory.
pub struct Projection {
    pub dim: usize,
    pub bits: usize,
    w: Vec<f32>,
}

impl Projection {
    pub fn new(dim: usize, bits: usize, seed: u64) -> Projection {
        let mut rng = Rng::new(seed);
        let mut w = vec![0.0f32; dim * bits];
        for i in 0..w.len() {
            w[i] = rng.normal();
        }
        Projection { dim, bits, w }
    }

    pub fn encode(&self, x: &[f32]) -> Vec<u64> {
        let mut out = vec![0u64; words_for(self.bits)];
        for b in 0..self.bits {
            let row = &self.w[b * self.dim..b * self.dim + self.dim];
            let mut s = 0.0f32;
            for i in 0..self.dim.min(x.len()) {
                s += row[i] * x[i];
            }
            if s > 0.0 {
                set_bit(&mut out, b, true);
            }
        }
        out
    }
}

pub struct Sdm {
    pub bits: usize,
    pub n_loc: usize,
    pub radius: usize,
    words: usize,
    addr: Vec<u64>,
    counters: Vec<i16>,
    pub writes: u64,
}

impl Sdm {
    /// Default radius activates ~2% of locations: bits/2 - 2 sigma.
    pub fn default_radius(bits: usize) -> usize {
        let sigma = ((bits as f32) / 4.0).sqrt();
        let r = (bits as f32) / 2.0 - 2.0 * sigma;
        if r < 1.0 {
            1
        } else {
            r as usize
        }
    }

    pub fn new(bits: usize, n_loc: usize, radius: usize, seed: u64) -> Sdm {
        let words = words_for(bits);
        let mut rng = Rng::new(seed);
        let mut addr = vec![0u64; n_loc * words];
        for l in 0..n_loc {
            let a = random_bits(&mut rng, bits);
            for i in 0..words {
                addr[l * words + i] = a[i];
            }
        }
        Sdm { bits, n_loc, radius, words, addr, counters: vec![0i16; n_loc * bits], writes: 0 }
    }

    pub fn activate(&self, address: &[u64]) -> Vec<usize> {
        let mut out = Vec::new();
        for l in 0..self.n_loc {
            let a = &self.addr[l * self.words..l * self.words + self.words];
            if hamming(a, address) <= self.radius {
                out.push(l);
            }
        }
        out
    }

    pub fn write(&mut self, address: &[u64], data: &[u64]) -> usize {
        let act = self.activate(address);
        for ai in 0..act.len() {
            let l = act[ai];
            let base = l * self.bits;
            for b in 0..self.bits {
                let v = if get_bit(data, b) { 1i16 } else { -1i16 };
                let c = self.counters[base + b] + v;
                self.counters[base + b] = if c > 127 {
                    127
                } else if c < -127 {
                    -127
                } else {
                    c
                };
            }
        }
        self.writes += 1;
        act.len()
    }

    pub fn read(&self, address: &[u64]) -> Vec<u64> {
        let act = self.activate(address);
        let mut sums = vec![0i32; self.bits];
        for ai in 0..act.len() {
            let base = act[ai] * self.bits;
            for b in 0..self.bits {
                sums[b] += self.counters[base + b] as i32;
            }
        }
        let mut out = vec![0u64; self.words];
        for b in 0..self.bits {
            if sums[b] > 0 {
                set_bit(&mut out, b, true);
            }
        }
        out
    }

    /// Iterated read: SDM reads are contractive, so re-addressing with the
    /// retrieved pattern converges onto a stored attractor (Kanerva's
    /// convergence property). Autoassociative clean-up for free.
    pub fn read_iterated(&self, address: &[u64], steps: usize) -> Vec<u64> {
        let mut cur = address.to_vec();
        for _ in 0..steps {
            let next = self.read(&cur);
            if hamming(&next, &cur) == 0 {
                return next;
            }
            cur = next;
        }
        cur
    }
}
