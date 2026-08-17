//! Byte-level Byte-Pair Encoding, trained from scratch.
//!
//! * Tokens 0..255 are raw bytes, so *any* input encodes - no UNK, no
//!   normalisation, full Unicode by construction.
//! * Training keeps an inverted index `pair -> words containing it`, so each
//!   merge only touches the words it actually affects instead of rescanning
//!   the corpus (the difference between seconds and minutes).
//! * Encoding applies merges in learned rank order, which reproduces the
//!   training-time segmentation exactly.

use std::collections::{HashMap, HashSet};
use std::io::Write;

pub struct Bpe {
    /// merge i creates token (256 + i) from a pair of existing tokens
    pub merges: Vec<(u32, u32)>,
    pub rank: HashMap<(u32, u32), u32>,
    /// byte expansion of every token, for decoding
    pub token_bytes: Vec<Vec<u8>>,
}

impl Bpe {
    pub fn bytes_only() -> Bpe {
        let mut token_bytes = Vec::with_capacity(256);
        for b in 0..256usize {
            token_bytes.push(vec![b as u8]);
        }
        Bpe { merges: Vec::new(), rank: HashMap::new(), token_bytes }
    }

    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    fn push_merge(&mut self, a: u32, b: u32) {
        let r = self.merges.len() as u32;
        self.merges.push((a, b));
        self.rank.insert((a, b), r);
        let mut bytes = self.token_bytes[a as usize].clone();
        let tail = self.token_bytes[b as usize].clone();
        bytes.extend_from_slice(&tail);
        self.token_bytes.push(bytes);
    }

    /// Split text into pre-token chunks. Merges never cross a chunk boundary,
    /// which stops the tokeniser from gluing punctuation and words together.
    pub fn pretokenize(text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut cur_kind = 0u8; // 0 none, 1 alpha, 2 digit, 3 other
        for ch in text.chars() {
            let kind = if ch.is_alphabetic() {
                1u8
            } else if ch.is_ascii_digit() {
                2u8
            } else {
                3u8
            };
            let is_space = ch == ' ';
            if is_space {
                if !cur.is_empty() {
                    out.push(cur.clone());
                    cur.clear();
                }
                cur.push(ch);
                cur_kind = 0;
                continue;
            }
            if cur_kind == 0 && cur == " " {
                // leading space attaches to the following chunk
                cur.push(ch);
                cur_kind = kind;
                continue;
            }
            if kind != cur_kind || kind == 3 {
                if !cur.is_empty() {
                    out.push(cur.clone());
                    cur.clear();
                }
                cur_kind = kind;
            }
            cur.push(ch);
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    pub fn train(text: &str, target_vocab: usize, verbose: bool) -> Bpe {
        let mut bpe = Bpe::bytes_only();
        if target_vocab <= 256 {
            return bpe;
        }
        // word frequency table over pre-token chunks
        let mut freq: HashMap<String, u64> = HashMap::new();
        for c in Bpe::pretokenize(text) {
            let e = freq.entry(c).or_insert(0);
            *e += 1;
        }
        let mut words: Vec<(Vec<u32>, u64)> = Vec::with_capacity(freq.len());
        for (w, c) in freq.into_iter() {
            let mut sym: Vec<u32> = Vec::with_capacity(w.len());
            for b in w.as_bytes() {
                sym.push(*b as u32);
            }
            if sym.len() >= 2 {
                words.push((sym, c));
            }
        }

        let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        for wi in 0..words.len() {
            let c = words[wi].1 as i64;
            for p in 0..words[wi].0.len() - 1 {
                let key = (words[wi].0[p], words[wi].0[p + 1]);
                let e = counts.entry(key).or_insert(0);
                *e += c;
                where_.entry(key).or_insert_with(HashSet::new).insert(wi);
            }
        }

        while bpe.vocab_size() < target_vocab {
            // argmax over pair counts, deterministic tie-break
            let mut best_key = (0u32, 0u32);
            let mut best_cnt = 0i64;
            let mut found = false;
            for (k, v) in counts.iter() {
                if *v <= 0 {
                    continue;
                }
                if !found || *v > best_cnt || (*v == best_cnt && *k < best_key) {
                    best_cnt = *v;
                    best_key = *k;
                    found = true;
                }
            }
            if !found || best_cnt < 2 {
                break;
            }
            let new_id = bpe.vocab_size() as u32;
            bpe.push_merge(best_key.0, best_key.1);
            if verbose && bpe.merges.len() % 128 == 0 {
                println!(
                    "  merge {:>5}  count {:>8}  vocab {}",
                    bpe.merges.len(),
                    best_cnt,
                    bpe.vocab_size()
                );
            }

            let affected: Vec<usize> = match where_.get(&best_key) {
                Some(s) => s.iter().cloned().collect(),
                None => Vec::new(),
            };
            counts.insert(best_key, 0);
            for ai in 0..affected.len() {
                let wi = affected[ai];
                let cnt = words[wi].1 as i64;
                let old = words[wi].0.clone();
                if old.len() < 2 {
                    continue;
                }
                for p in 0..old.len() - 1 {
                    let key = (old[p], old[p + 1]);
                    if let Some(c) = counts.get_mut(&key) {
                        *c -= cnt;
                    }
                }
                let mut ns: Vec<u32> = Vec::with_capacity(old.len());
                let mut i = 0usize;
                while i < old.len() {
                    if i + 1 < old.len() && old[i] == best_key.0 && old[i + 1] == best_key.1 {
                        ns.push(new_id);
                        i += 2;
                    } else {
                        ns.push(old[i]);
                        i += 1;
                    }
                }
                words[wi].0 = ns;
                if words[wi].0.len() >= 2 {
                    for p in 0..words[wi].0.len() - 1 {
                        let key = (words[wi].0[p], words[wi].0[p + 1]);
                        let e = counts.entry(key).or_insert(0);
                        *e += cnt;
                        where_.entry(key).or_insert_with(HashSet::new).insert(wi);
                    }
                }
            }
        }
        bpe
    }

    /// Greedy lowest-rank-first merging inside one pre-token chunk.
    pub fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut sym: Vec<u32> = Vec::with_capacity(chunk.len());
        for b in chunk.as_bytes() {
            sym.push(*b as u32);
        }
        loop {
            if sym.len() < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            for i in 0..sym.len() - 1 {
                match self.rank.get(&(sym[i], sym[i + 1])) {
                    Some(r) => {
                        if *r < best_rank {
                            best_rank = *r;
                            best_i = i;
                        }
                    }
                    None => {}
                }
            }
            if best_i == usize::MAX {
                break;
            }
            sym[best_i] = 256 + best_rank;
            sym.remove(best_i + 1);
        }
        sym
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for c in Bpe::pretokenize(text) {
            let ids = self.encode_chunk(&c);
            out.extend_from_slice(&ids);
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for i in 0..ids.len() {
            let t = ids[i] as usize;
            if t < self.token_bytes.len() {
                bytes.extend_from_slice(&self.token_bytes[t]);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "noetic-bpe 1")?;
        writeln!(f, "{}", self.merges.len())?;
        for i in 0..self.merges.len() {
            writeln!(f, "{} {}", self.merges[i].0, self.merges[i].1)?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Bpe> {
        let text = std::fs::read_to_string(path)?;
        let mut bpe = Bpe::bytes_only();
        let mut lines = text.lines();
        let header = lines.next().unwrap_or("");
        if !header.starts_with("noetic-bpe") {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad tokenizer header"));
        }
        let _n = lines.next().unwrap_or("0");
        for line in lines {
            let mut it = line.split_whitespace();
            let a = it.next();
            let b = it.next();
            match (a, b) {
                (Some(x), Some(y)) => {
                    let xa: u32 = match x.parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let yb: u32 = match y.parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    bpe.push_merge(xa, yb);
                }
                _ => {}
            }
        }
        Ok(bpe)
    }
}
