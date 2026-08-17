//! Corpus loading, plus a self-contained synthetic corpus so the engine can
//! train with zero external data.
//!
//! The synthetic corpus deliberately mixes four skills so training curves are
//! informative rather than decorative:
//!
//! 1. grammar     - a small PCFG; tests local syntax
//! 2. arithmetic  - `a + b = c`; tests exact computation
//! 3. memo/query  - `memo <key> = <val> ... query <key> = <val>`; only solvable
//!                  by carrying information across many tokens, i.e. it
//!                  directly measures the recurrent state's memory
//! 4. counting    - ascending runs; tests positional bookkeeping

use crate::rng::Rng;

const NOUNS: [&str; 12] = [
    "engine", "kernel", "tensor", "agent", "signal", "lattice", "memory", "circuit", "river", "mountain", "machine", "garden",
];
const VERBS: [&str; 10] = [
    "builds", "observes", "folds", "remembers", "drives", "encodes", "resolves", "shapes", "tracks", "predicts",
];
const ADJS: [&str; 10] = [
    "quiet", "dense", "recursive", "sparse", "bright", "linear", "hidden", "stable", "ancient", "fast",
];
const KEYS: [&str; 8] = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"];

pub fn synthetic_corpus(target_bytes: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let mut s = String::with_capacity(target_bytes + 256);
    while s.len() < target_bytes {
        let kind = rng.below(4);
        if kind == 0 {
            let a = ADJS[rng.below(ADJS.len())];
            let n1 = NOUNS[rng.below(NOUNS.len())];
            let v = VERBS[rng.below(VERBS.len())];
            let a2 = ADJS[rng.below(ADJS.len())];
            let n2 = NOUNS[rng.below(NOUNS.len())];
            s.push_str(&format!("the {} {} {} the {} {} .\n", a, n1, v, a2, n2));
        } else if kind == 1 {
            let x = rng.below(50);
            let y = rng.below(50);
            if rng.below(2) == 0 {
                s.push_str(&format!("{} + {} = {} .\n", x, y, x + y));
            } else {
                let hi = if x > y { x } else { y };
                let lo = if x > y { y } else { x };
                s.push_str(&format!("{} - {} = {} .\n", hi, lo, hi - lo));
            }
        } else if kind == 2 {
            // long-range binding: the answer is only recoverable from state
            let k = KEYS[rng.below(KEYS.len())];
            let v = rng.below(1000);
            s.push_str(&format!("memo {} = {} ;", k, v));
            let filler = 1 + rng.below(3);
            for _ in 0..filler {
                let a = ADJS[rng.below(ADJS.len())];
                let n1 = NOUNS[rng.below(NOUNS.len())];
                s.push_str(&format!(" the {} {} waits ;", a, n1));
            }
            s.push_str(&format!(" query {} = {} .\n", k, v));
        } else {
            let start = rng.below(20);
            let len = 4 + rng.below(6);
            s.push_str("count :");
            for i in 0..len {
                s.push_str(&format!(" {}", start + i));
            }
            s.push_str(" .\n");
        }
    }
    s
}

/// Read `path`, or fall back to the synthetic corpus when it is missing/empty.
pub fn load_or_synthesize(path: &str, bytes: usize, seed: u64) -> (String, bool) {
    if !path.is_empty() {
        match std::fs::read(path) {
            Ok(raw) => {
                if !raw.is_empty() {
                    return (String::from_utf8_lossy(&raw).to_string(), false);
                }
            }
            Err(_) => {}
        }
    }
    (synthetic_corpus(bytes, seed), true)
}

/// Random-crop batcher over a flat token stream with a held-out tail.
pub struct Batcher {
    pub tokens: Vec<u32>,
    pub split: usize,
}

impl Batcher {
    pub fn new(tokens: Vec<u32>, val_frac: f32) -> Batcher {
        let n = tokens.len();
        let mut split = ((n as f32) * (1.0 - val_frac)) as usize;
        if split < 2 {
            split = n;
        }
        Batcher { tokens, split }
    }

    fn crop(&self, lo: usize, hi: usize, batch: usize, t: usize, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        let mut x = vec![0u32; batch * t];
        let mut y = vec![0u32; batch * t];
        let span = if hi > lo + t + 1 { hi - lo - t - 1 } else { 1 };
        for b in 0..batch {
            let s = lo + rng.below(span);
            for i in 0..t {
                let p = s + i;
                x[b * t + i] = self.tokens[p.min(self.tokens.len() - 2)];
                y[b * t + i] = self.tokens[(p + 1).min(self.tokens.len() - 1)];
            }
        }
        (x, y)
    }

    pub fn train_batch(&self, batch: usize, t: usize, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        self.crop(0, self.split, batch, t, rng)
    }

    pub fn val_batch(&self, batch: usize, t: usize, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        if self.split + t + 2 >= self.tokens.len() {
            return self.crop(0, self.split, batch, t, rng);
        }
        self.crop(self.split, self.tokens.len(), batch, t, rng)
    }
}
