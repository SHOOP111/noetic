//! Byte-level Byte-Pair Encoding, trained from scratch.
//!
//! * Tokens 0..255 are raw bytes, so arbitrary byte buffers have a lossless
//!   `encode_bytes` / `decode_bytes` path; UTF-8 text uses pre-token chunks.
//! * Training keeps an exact inverted index `pair -> words containing it`, so
//!   each merge only touches the words it currently affects.
//! * Encoding applies merges in learned rank order, reproducing the
//!   training-time segmentation.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_TOKENIZER_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SERIALIZED_MERGES: usize = 1_000_000;
const MAX_EXPANDED_TOKEN_BYTES: usize = 1024 * 1024;
const MAX_TOTAL_EXPANDED_BYTES: usize = 128 * 1024 * 1024;
static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub struct Bpe {
    /// Merge i creates token (256 + i) from a pair of existing tokens.
    pub merges: Vec<(u32, u32)>,
    pub rank: HashMap<(u32, u32), u32>,
    /// Byte expansion of every token, for decoding.
    pub token_bytes: Vec<Vec<u8>>,
}

fn invalid_data(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.to_string())
}

fn create_temp_file(path: &str) -> std::io::Result<(String, File)> {
    for _ in 0..128 {
        let nonce = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = format!("{}.tmp.{}.{}", path, std::process::id(), nonce);
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique tokenizer temporary file",
    ))
}

impl Bpe {
    pub fn bytes_only() -> Bpe {
        let mut token_bytes = Vec::with_capacity(256);
        for byte in 0..256usize {
            token_bytes.push(vec![byte as u8]);
        }
        Bpe { merges: Vec::new(), rank: HashMap::new(), token_bytes }
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    fn push_merge(&mut self, a: u32, b: u32) {
        let next_id = self.token_bytes.len();
        assert!((a as usize) < next_id && (b as usize) < next_id, "BPE merge references an unknown token");
        assert!(!self.rank.contains_key(&(a, b)), "duplicate BPE merge pair");
        let rank = self.merges.len() as u32;
        self.merges.push((a, b));
        self.rank.insert((a, b), rank);
        let left = &self.token_bytes[a as usize];
        let right = &self.token_bytes[b as usize];
        let length = left.len().checked_add(right.len()).expect("BPE token expansion overflow");
        let mut bytes = Vec::with_capacity(length);
        bytes.extend_from_slice(left);
        bytes.extend_from_slice(right);
        self.token_bytes.push(bytes);
    }

    /// Split text into pre-token chunks. Merges never cross a chunk boundary,
    /// which stops the tokenizer from gluing punctuation and words together.
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
            if ch == ' ' {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                cur.push(ch);
                cur_kind = 0;
                continue;
            }
            if cur_kind == 0 && cur == " " {
                // A leading ASCII space attaches to the following chunk.
                cur.push(ch);
                cur_kind = kind;
                continue;
            }
            if kind != cur_kind || kind == 3 {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
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
        let target_vocab = target_vocab.min(u32::MAX as usize);
        if target_vocab <= bpe.vocab_size() {
            return bpe;
        }

        // Word-frequency table over pre-token chunks.
        let mut frequencies: HashMap<String, u64> = HashMap::new();
        for chunk in Bpe::pretokenize(text) {
            *frequencies.entry(chunk).or_insert(0) += 1;
        }
        let mut words: Vec<(Vec<u32>, u64)> = Vec::with_capacity(frequencies.len());
        for (word, count) in frequencies {
            let symbols: Vec<u32> = word.as_bytes().iter().map(|&byte| byte as u32).collect();
            if symbols.len() >= 2 {
                words.push((symbols, count));
            }
        }

        let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut locations: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        for (word_index, (symbols, frequency)) in words.iter().enumerate() {
            let count = *frequency as i64;
            for pair in symbols.windows(2) {
                let key = (pair[0], pair[1]);
                *counts.entry(key).or_insert(0) += count;
                locations.entry(key).or_default().insert(word_index);
            }
        }

        while bpe.vocab_size() < target_vocab {
            // Deterministic argmax over pair counts.
            let mut best_key = (0u32, 0u32);
            let mut best_count = 0i64;
            let mut found = false;
            for (&key, &count) in &counts {
                if count <= 0 || bpe.rank.contains_key(&key) {
                    continue;
                }
                if !found || count > best_count || (count == best_count && key < best_key) {
                    best_count = count;
                    best_key = key;
                    found = true;
                }
            }
            if !found || best_count < 2 {
                break;
            }

            let new_id = bpe.vocab_size() as u32;
            bpe.push_merge(best_key.0, best_key.1);
            if verbose && bpe.merges.len() % 128 == 0 {
                println!(
                    "  merge {:>5}  count {:>8}  vocab {}",
                    bpe.merges.len(),
                    best_count,
                    bpe.vocab_size()
                );
            }

            // Remove the selected pair from the exact inverted index. Every
            // current occurrence is replaced below, so it cannot remain.
            let affected: Vec<usize> = locations
                .remove(&best_key)
                .map(|set| set.into_iter().collect())
                .unwrap_or_default();
            counts.insert(best_key, 0);

            for word_index in affected {
                let frequency = words[word_index].1 as i64;
                let old = std::mem::take(&mut words[word_index].0);
                if old.len() < 2 {
                    words[word_index].0 = old;
                    continue;
                }

                // Remove this word from every old pair's count and membership.
                let mut unique_old_pairs = HashSet::new();
                for pair in old.windows(2) {
                    let key = (pair[0], pair[1]);
                    *counts.entry(key).or_insert(0) -= frequency;
                    unique_old_pairs.insert(key);
                }
                for key in unique_old_pairs {
                    if let Some(set) = locations.get_mut(&key) {
                        set.remove(&word_index);
                    }
                }

                let mut merged = Vec::with_capacity(old.len());
                let mut i = 0usize;
                while i < old.len() {
                    if i + 1 < old.len() && old[i] == best_key.0 && old[i + 1] == best_key.1 {
                        merged.push(new_id);
                        i += 2;
                    } else {
                        merged.push(old[i]);
                        i += 1;
                    }
                }

                for pair in merged.windows(2) {
                    let key = (pair[0], pair[1]);
                    *counts.entry(key).or_insert(0) += frequency;
                    locations.entry(key).or_default().insert(word_index);
                }
                words[word_index].0 = merged;
            }
        }
        bpe
    }

    fn encode_raw(&self, bytes: &[u8]) -> Vec<u32> {
        let mut symbols: Vec<u32> = bytes.iter().map(|&byte| byte as u32).collect();
        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_index = usize::MAX;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.rank.get(&(symbols[i], symbols[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = i;
                    }
                }
            }
            if best_index == usize::MAX {
                break;
            }
            symbols[best_index] = 256 + best_rank;
            symbols.remove(best_index + 1);
        }
        symbols
    }

    /// Greedy lowest-rank-first merging inside one pre-token chunk.
    pub fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        self.encode_raw(chunk.as_bytes())
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for chunk in Bpe::pretokenize(text) {
            out.extend(self.encode_chunk(&chunk));
        }
        out
    }

    /// Lossless byte-buffer path. Unlike `encode`, this intentionally has no
    /// text pre-tokenization boundaries.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        self.encode_raw(bytes)
    }

    /// Decode exactly, returning `None` rather than silently dropping an
    /// out-of-range token id.
    pub fn decode_bytes(&self, ids: &[u32]) -> Option<Vec<u8>> {
        let total_bytes = ids.iter().try_fold(0usize, |total, &id| {
            let token = self.token_bytes.get(id as usize)?;
            total.checked_add(token.len())
        })?;
        let mut bytes = Vec::with_capacity(total_bytes);
        for &id in ids {
            let token = self.token_bytes.get(id as usize)?;
            bytes.extend_from_slice(token);
        }
        Some(bytes)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self.decode_bytes(ids) {
            Some(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            None => String::new(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let (tmp_path, file) = create_temp_file(path)?;
        let write_result = (|| -> std::io::Result<()> {
            let mut writer = BufWriter::new(file);
            writeln!(writer, "noetic-bpe 1")?;
            writeln!(writer, "{}", self.merges.len())?;
            for &(a, b) in &self.merges {
                writeln!(writer, "{} {}", a, b)?;
            }
            writer.flush()?;
            writer.get_ref().sync_all()?;
            drop(writer);
            std::fs::rename(&tmp_path, path)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        }
        write_result
    }

    pub fn load(path: &str) -> std::io::Result<Bpe> {
        let file_size = std::fs::metadata(path)?.len();
        if file_size > MAX_TOKENIZER_FILE_BYTES {
            return Err(invalid_data("tokenizer file exceeds the supported size limit"));
        }
        let text = std::fs::read_to_string(path)?;
        if text.len() as u64 > MAX_TOKENIZER_FILE_BYTES {
            return Err(invalid_data("tokenizer file changed while it was being read"));
        }
        let mut lines = text.lines();
        if lines.next() != Some("noetic-bpe 1") {
            return Err(invalid_data("unsupported tokenizer header or version"));
        }
        let merge_count: usize = lines
            .next()
            .ok_or_else(|| invalid_data("tokenizer merge count is missing"))?
            .parse()
            .map_err(|_| invalid_data("tokenizer merge count is invalid"))?;
        if merge_count > MAX_SERIALIZED_MERGES || merge_count > (u32::MAX as usize).saturating_sub(256) {
            return Err(invalid_data("tokenizer declares too many merges"));
        }

        let mut bpe = Bpe::bytes_only();
        let mut total_expanded_bytes = 256usize;
        for _ in 0..merge_count {
            let line = lines.next().ok_or_else(|| invalid_data("tokenizer merge list is truncated"))?;
            let mut fields = line.split_whitespace();
            let a: u32 = fields
                .next()
                .ok_or_else(|| invalid_data("tokenizer merge is missing its first token"))?
                .parse()
                .map_err(|_| invalid_data("tokenizer merge token is invalid"))?;
            let b: u32 = fields
                .next()
                .ok_or_else(|| invalid_data("tokenizer merge is missing its second token"))?
                .parse()
                .map_err(|_| invalid_data("tokenizer merge token is invalid"))?;
            if fields.next().is_some() {
                return Err(invalid_data("tokenizer merge has extra fields"));
            }
            let next_id = bpe.vocab_size();
            if (a as usize) >= next_id || (b as usize) >= next_id {
                return Err(invalid_data("tokenizer merge references an unknown token"));
            }
            if bpe.rank.contains_key(&(a, b)) {
                return Err(invalid_data("tokenizer contains a duplicate merge pair"));
            }
            let expanded_bytes = bpe.token_bytes[a as usize]
                .len()
                .checked_add(bpe.token_bytes[b as usize].len())
                .ok_or_else(|| invalid_data("tokenizer token expansion overflows"))?;
            if expanded_bytes > MAX_EXPANDED_TOKEN_BYTES {
                return Err(invalid_data("tokenizer contains an excessively large expanded token"));
            }
            total_expanded_bytes = total_expanded_bytes
                .checked_add(expanded_bytes)
                .ok_or_else(|| invalid_data("tokenizer expansion size overflows"))?;
            if total_expanded_bytes > MAX_TOTAL_EXPANDED_BYTES {
                return Err(invalid_data("tokenizer expanded vocabulary exceeds the supported size limit"));
            }
            bpe.push_merge(a, b);
        }
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(invalid_data("tokenizer has trailing data"));
        }
        Ok(bpe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_rejects_exponential_token_expansion() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("noetic-bpe-bomb-{}-{}.tok", std::process::id(), unique));
        let mut serialized = String::from("noetic-bpe 1\n22\n");
        for merge in 0..22u32 {
            let source = if merge == 0 { 97 } else { 255 + merge };
            serialized.push_str(&format!("{} {}\n", source, source));
        }
        std::fs::write(&path, serialized).unwrap();
        let result = Bpe::load(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }
}
