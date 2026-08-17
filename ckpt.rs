//! Checkpoint format, hand-rolled. No serde, no JSON, no protobuf.
//!
//! ```text
//! magic   8 bytes  "NOETIC\0\1"
//! u32              format version
//! u32              meta count, then (u32 len + bytes) key/value pairs
//! u32              tensor count, then per tensor:
//!                    u32 len + name bytes
//!                    u32 rank, u32 * rank dims
//!                    u32 numel, f32 little-endian payload
//! u32              CRC-32 (IEEE, reflected) of everything above
//! ```
//!
//! Loading is name-keyed, so a checkpoint stays valid when layers are added
//! elsewhere in the model. Malformed, truncated, duplicate, trailing, and
//! shape-inconsistent data is rejected before it can reach the live graph.

use crate::autograd::Graph;
use std::collections::{HashMap, HashSet};
use std::io::Write;

const MAGIC: &[u8; 8] = b"NOETIC\x00\x01";
const VERSION: u32 = 1;
const MAX_RANK: usize = 32;
const MAX_ENTRIES: usize = 1_000_000;

pub fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for i in 0..256usize {
        let mut c = i as u32;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        table[i] = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

fn checked_u32(value: usize, what: &str) -> std::io::Result<u32> {
    u32::try_from(value).map_err(|_| bad(&format!("{} exceeds the checkpoint format limit", what)))
}

fn put_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, value: &str, what: &str) -> std::io::Result<()> {
    put_u32(buf, checked_u32(value.len(), what)?);
    buf.extend_from_slice(value.as_bytes());
    Ok(())
}

pub fn save(path: &str, g: &Graph, meta: &[(String, String)]) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    put_u32(&mut buf, VERSION);
    put_u32(&mut buf, checked_u32(meta.len(), "metadata count")?);

    let mut meta_keys = HashSet::with_capacity(meta.len());
    for (key, value) in meta {
        if key.is_empty() {
            return Err(bad("checkpoint metadata keys cannot be empty"));
        }
        if !meta_keys.insert(key.as_str()) {
            return Err(bad("checkpoint metadata contains a duplicate key"));
        }
        put_str(&mut buf, key, "metadata key length")?;
        put_str(&mut buf, value, "metadata value length")?;
    }

    put_u32(&mut buf, checked_u32(g.params.len(), "tensor count")?);
    let mut tensor_names = HashSet::with_capacity(g.params.len());
    for param in &g.params {
        let id = param.id;
        if param.name.is_empty() {
            return Err(bad("checkpoint tensor names cannot be empty"));
        }
        if !tensor_names.insert(param.name.as_str()) {
            return Err(bad("checkpoint contains a duplicate tensor name"));
        }
        put_str(&mut buf, &param.name, "tensor name length")?;
        let shape = &g.shape[id];
        if shape.len() > MAX_RANK {
            return Err(bad("tensor rank exceeds the checkpoint format limit"));
        }
        put_u32(&mut buf, checked_u32(shape.len(), "tensor rank")?);
        for &dim in shape {
            put_u32(&mut buf, checked_u32(dim, "tensor dimension")?);
        }
        let values = &g.val[id];
        let expected = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        if expected != Some(values.len()) {
            return Err(bad("live tensor shape does not match its value count"));
        }
        put_u32(&mut buf, checked_u32(values.len(), "tensor element count")?);
        for &value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    let checksum = crc32(&buf);
    put_u32(&mut buf, checksum);

    // Write-then-rename prevents an interrupted save from replacing the last
    // valid checkpoint with a partial file.
    let tmp_path = format!("{}.tmp.{}", path, std::process::id());
    let write_result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(&buf)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    write_result
}

pub struct Ckpt {
    pub meta: HashMap<String, String>,
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl Ckpt {
    pub fn meta_usize(&self, key: &str, default: usize) -> usize {
        match self.meta.get(key) {
            Some(value) => value.parse::<usize>().unwrap_or(default),
            None => default,
        }
    }

    pub fn meta_f32(&self, key: &str, default: f32) -> f32 {
        match self.meta.get(key) {
            Some(value) => value.parse::<f32>().unwrap_or(default),
            None => default,
        }
    }
}

struct Cur<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cur<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, n: usize) -> std::io::Result<&'a [u8]> {
        let end = self.offset.checked_add(n).ok_or_else(|| bad("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(bad("checkpoint truncated"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> std::io::Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f32(&mut self) -> std::io::Result<f32> {
        let bytes = self.take(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn string(&mut self) -> std::io::Result<String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        let text = std::str::from_utf8(bytes).map_err(|_| bad("checkpoint contains invalid UTF-8"))?;
        Ok(text.to_string())
    }
}

pub fn load(path: &str) -> std::io::Result<Ckpt> {
    let raw = std::fs::read(path)?;
    if raw.len() < 24 {
        return Err(bad("checkpoint too small"));
    }
    let body = &raw[..raw.len() - 4];
    let checksum_bytes = &raw[raw.len() - 4..];
    let expected_checksum = u32::from_le_bytes([
        checksum_bytes[0],
        checksum_bytes[1],
        checksum_bytes[2],
        checksum_bytes[3],
    ]);
    if crc32(body) != expected_checksum {
        return Err(bad("checkpoint CRC mismatch (corrupt file)"));
    }

    let mut cursor = Cur { bytes: body, offset: 0 };
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(bad("not a noetic checkpoint"));
    }
    let version = cursor.u32()?;
    if version != VERSION {
        return Err(bad("unsupported checkpoint version"));
    }

    let meta_count = cursor.u32()? as usize;
    if meta_count > MAX_ENTRIES {
        return Err(bad("checkpoint contains too many metadata entries"));
    }
    let mut meta = HashMap::with_capacity(meta_count);
    for _ in 0..meta_count {
        let key = cursor.string()?;
        let value = cursor.string()?;
        if key.is_empty() || meta.insert(key, value).is_some() {
            return Err(bad("checkpoint metadata contains an empty or duplicate key"));
        }
    }

    let tensor_count = cursor.u32()? as usize;
    if tensor_count > MAX_ENTRIES {
        return Err(bad("checkpoint contains too many tensors"));
    }
    let mut tensors = HashMap::with_capacity(tensor_count);
    let mut shapes = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cursor.string()?;
        if name.is_empty() || tensors.contains_key(&name) {
            return Err(bad("checkpoint contains an empty or duplicate tensor name"));
        }
        let rank = cursor.u32()? as usize;
        if rank > MAX_RANK || rank > cursor.remaining() / 4 {
            return Err(bad("checkpoint tensor rank is invalid"));
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(cursor.u32()? as usize);
        }
        let numel = cursor.u32()? as usize;
        if numel > cursor.remaining() / 4 {
            return Err(bad("checkpoint tensor payload is truncated"));
        }
        let expected_numel = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        if expected_numel != Some(numel) {
            return Err(bad("checkpoint tensor shape does not match its element count"));
        }
        let mut values = Vec::with_capacity(numel);
        for _ in 0..numel {
            values.push(cursor.f32()?);
        }
        shapes.insert(name.clone(), shape);
        tensors.insert(name, values);
    }
    if cursor.remaining() != 0 {
        return Err(bad("checkpoint has trailing data"));
    }

    Ok(Ckpt { meta, tensors, shapes })
}

/// Copy checkpoint tensors into the live graph by parameter name.
/// Returns (loaded, missing, shape-mismatched).
pub fn apply(g: &mut Graph, ckpt: &Ckpt) -> (usize, usize, usize) {
    let mut loaded = 0usize;
    let mut missing = 0usize;
    let mut mismatch = 0usize;
    for param in &g.params {
        let id = param.id;
        match (ckpt.tensors.get(&param.name), ckpt.shapes.get(&param.name)) {
            (Some(values), Some(shape)) if shape == &g.shape[id] && values.len() == g.val[id].len() => {
                g.val[id].copy_from_slice(values);
                loaded += 1;
            }
            (Some(_), _) => mismatch += 1,
            (None, _) => missing += 1,
        }
    }
    (loaded, missing, mismatch)
}
