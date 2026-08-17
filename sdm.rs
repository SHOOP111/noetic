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
//! All operations are integer/bitwise: `count_ones` on u64 words.

use crate::rng::Rng;

pub fn words_for(bits: usize) -> usize {
    bits / 64 + usize::from(bits % 64 != 0)
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
    assert_eq!(a.len(), b.len(), "hamming: address lengths differ");
    let mut d = 0usize;
    for i in 0..a.len() {
        d += (a[i] ^ b[i]).count_ones() as usize;
    }
    d
}

pub fn random_bits(rng: &mut Rng, bits: usize) -> Vec<u64> {
    let nw = words_for(bits);
    let mut v = vec![0u64; nw];
    for word in &mut v {
        *word = rng.next_u64();
    }
    // Clear the tail so distances only cover the declared address width.
    let rem = bits % 64;
    if rem != 0 {
        let mask = (1u64 << rem) - 1;
        v[nw - 1] &= mask;
    }
    v
}

/// Flip exactly `min(n_flip, bits)` distinct positions.
pub fn flip_bits(src: &[u64], bits: usize, n_flip: usize, rng: &mut Rng) -> Vec<u64> {
    assert_eq!(src.len(), words_for(bits), "flip_bits: bad address length");
    let mut out = src.to_vec();
    let n = n_flip.min(bits);
    if n == 0 {
        return out;
    }

    // Partial Fisher-Yates sampling gives distinct positions without a hash
    // table, so a request for N flipped bits really changes the Hamming
    // distance by N instead of occasionally toggling the same bit twice.
    let mut positions: Vec<usize> = (0..bits).collect();
    for i in 0..n {
        let j = i + rng.below(bits - i);
        positions.swap(i, j);
        let bit = positions[i];
        let cur = get_bit(&out, bit);
        set_bit(&mut out, bit, !cur);
    }
    out
}

/// Sign-random-projection: maps a real vector to a bit address such that
/// Hamming distance approximates angular distance (LSH).
pub struct Projection {
    pub dim: usize,
    pub bits: usize,
    w: Vec<f32>,
}

impl Projection {
    pub fn new(dim: usize, bits: usize, seed: u64) -> Projection {
        assert!(dim > 0, "projection dimension must be positive");
        assert!(bits > 0, "projection address width must be positive");
        let len = dim.checked_mul(bits).expect("projection is too large");
        let mut rng = Rng::new(seed);
        let mut w = vec![0.0f32; len];
        for value in &mut w {
            *value = rng.normal();
        }
        Projection { dim, bits, w }
    }

    pub fn encode(&self, x: &[f32]) -> Vec<u64> {
        assert_eq!(x.len(), self.dim, "projection input has the wrong dimension");
        let mut out = vec![0u64; words_for(self.bits)];
        for b in 0..self.bits {
            let row = &self.w[b * self.dim..b * self.dim + self.dim];
            let mut s = 0.0f32;
            for i in 0..self.dim {
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
    /// Default radius activates approximately 2% of random locations.
    pub fn default_radius(bits: usize) -> usize {
        if bits == 0 {
            return 0;
        }
        let sigma = ((bits as f32) / 4.0).sqrt();
        let r = (bits as f32) / 2.0 - 2.0 * sigma;
        if r < 1.0 {
            1.min(bits)
        } else {
            (r as usize).min(bits)
        }
    }

    pub fn new(bits: usize, n_loc: usize, radius: usize, seed: u64) -> Sdm {
        assert!(bits > 0, "SDM address width must be positive");
        assert!(n_loc > 0, "SDM must contain at least one hard location");
        assert!(radius <= bits, "SDM radius cannot exceed address width");
        let words = words_for(bits);
        let addr_len = n_loc.checked_mul(words).expect("SDM address arena is too large");
        let counter_len = n_loc.checked_mul(bits).expect("SDM counter arena is too large");
        let mut rng = Rng::new(seed);
        let mut addr = vec![0u64; addr_len];
        for l in 0..n_loc {
            let a = random_bits(&mut rng, bits);
            addr[l * words..(l + 1) * words].copy_from_slice(&a);
        }
        Sdm { bits, n_loc, radius, words, addr, counters: vec![0i16; counter_len], writes: 0 }
    }

    pub fn activate(&self, address: &[u64]) -> Vec<usize> {
        assert_eq!(address.len(), self.words, "SDM address has the wrong width");
        let mut out = Vec::new();
        for l in 0..self.n_loc {
            let a = &self.addr[l * self.words..(l + 1) * self.words];
            if hamming(a, address) <= self.radius {
                out.push(l);
            }
        }
        out
    }

    pub fn write(&mut self, address: &[u64], data: &[u64]) -> usize {
        assert_eq!(data.len(), self.words, "SDM data has the wrong width");
        let act = self.activate(address);
        for &l in &act {
            let base = l * self.bits;
            for b in 0..self.bits {
                let delta = if get_bit(data, b) { 1i32 } else { -1i32 };
                let next = (self.counters[base + b] as i32 + delta).clamp(-127, 127);
                self.counters[base + b] = next as i16;
            }
        }
        self.writes = self.writes.saturating_add(1);
        act.len()
    }

    pub fn read(&self, address: &[u64]) -> Vec<u64> {
        let act = self.activate(address);
        let mut sums = vec![0i32; self.bits];
        for &location in &act {
            let base = location * self.bits;
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

    /// Re-address repeatedly until the retrieved pattern stabilises.
    pub fn read_iterated(&self, address: &[u64], steps: usize) -> Vec<u64> {
        assert_eq!(address.len(), self.words, "SDM address has the wrong width");
        let mut cur = address.to_vec();
        for _ in 0..steps {
            let next = self.read(&cur);
            if next == cur {
                return next;
            }
            cur = next;
        }
        cur
    }
}
