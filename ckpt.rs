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
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

const MAGIC: &[u8; 8] = b"NOETIC\x00\x01";
const VERSION: u32 = 1;
const MAX_RANK: usize = 32;
const MAX_ENTRIES: usize = 1_000_000;
const MAX_CHECKPOINT_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Reserved tensor-name prefix for non-model payloads (optimizer moments).
/// `apply_exact` ignores this namespace, so adding optimizer state to a file
/// does not weaken the "exactly the model tensors, nothing else" guarantee.
pub const AUX_PREFIX: &str = "aux.";
static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

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
    Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "could not allocate a unique checkpoint temporary file"))
}

fn serialized_size(g: &Graph, meta: &[(String, String)], extra: &[(String, Vec<usize>, Vec<f32>)]) -> std::io::Result<usize> {
    let mut bytes = (MAGIC.len() + 4 + 4 + 4 + 4) as u64;
    for (key, value) in meta {
        bytes = bytes
            .checked_add(8)
            .and_then(|size| size.checked_add(key.len() as u64))
            .and_then(|size| size.checked_add(value.len() as u64))
            .ok_or_else(|| bad("checkpoint serialized size overflows"))?;
    }
    for parameter in &g.params {
        let id = parameter.id;
        let shape_bytes = (g.shape[id].len() as u64).checked_mul(4).ok_or_else(|| bad("checkpoint shape size overflows"))?;
        let value_bytes = (g.val[id].len() as u64).checked_mul(4).ok_or_else(|| bad("checkpoint tensor size overflows"))?;
        bytes = bytes
            .checked_add(12)
            .and_then(|size| size.checked_add(parameter.name.len() as u64))
            .and_then(|size| size.checked_add(shape_bytes))
            .and_then(|size| size.checked_add(value_bytes))
            .ok_or_else(|| bad("checkpoint serialized size overflows"))?;
    }
    for (name, shape, values) in extra {
        let shape_bytes = (shape.len() as u64).checked_mul(4).ok_or_else(|| bad("checkpoint shape size overflows"))?;
        let value_bytes = (values.len() as u64).checked_mul(4).ok_or_else(|| bad("checkpoint tensor size overflows"))?;
        bytes = bytes
            .checked_add(12)
            .and_then(|size| size.checked_add(name.len() as u64))
            .and_then(|size| size.checked_add(shape_bytes))
            .and_then(|size| size.checked_add(value_bytes))
            .ok_or_else(|| bad("checkpoint serialized size overflows"))?;
    }
    if bytes > MAX_CHECKPOINT_FILE_BYTES || bytes > usize::MAX as u64 {
        return Err(bad("checkpoint exceeds the supported size limit"));
    }
    Ok(bytes as usize)
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

/// Save every model parameter. Equivalent to [`save_with_aux`] with no aux
/// tensors.
pub fn save(path: &str, g: &Graph, meta: &[(String, String)]) -> std::io::Result<()> {
    save_with_aux(path, g, meta, &[])
}

/// Save model parameters plus auxiliary tensors (for example optimizer moments,
/// which is what makes a training run resumable). Aux names must start with
/// [`AUX_PREFIX`]; model parameters must not.
pub fn save_with_aux(
    path: &str,
    g: &Graph,
    meta: &[(String, String)],
    extra: &[(String, Vec<usize>, Vec<f32>)],
) -> std::io::Result<()> {
    if meta.len() > MAX_ENTRIES || g.params.len().saturating_add(extra.len()) > MAX_ENTRIES {
        return Err(bad("checkpoint contains too many entries"));
    }
    let mut buf: Vec<u8> = Vec::with_capacity(serialized_size(g, meta, extra)?);
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

    let total = g.params.len().checked_add(extra.len()).ok_or_else(|| bad("checkpoint entry count overflows"))?;
    put_u32(&mut buf, checked_u32(total, "tensor count")?);
    let mut tensor_names = HashSet::with_capacity(total);
    for param in &g.params {
        let id = param.id;
        if param.name.is_empty() {
            return Err(bad("checkpoint tensor names cannot be empty"));
        }
        if param.name.starts_with(AUX_PREFIX) {
            return Err(bad("model parameters cannot use the reserved 'aux.' name prefix"));
        }
        if !tensor_names.insert(param.name.clone()) {
            return Err(bad("checkpoint contains a duplicate tensor name"));
        }
        write_tensor(&mut buf, &param.name, &g.shape[id], &g.val[id])?;
    }
    for (name, shape, values) in extra {
        if !name.starts_with(AUX_PREFIX) {
            return Err(bad("auxiliary checkpoint tensors must use the 'aux.' name prefix"));
        }
        if !tensor_names.insert(name.clone()) {
            return Err(bad("checkpoint contains a duplicate tensor name"));
        }
        write_tensor(&mut buf, name, shape, values)?;
    }

    let checksum = crc32(&buf);
    put_u32(&mut buf, checksum);

    // Write-then-rename prevents an interrupted save from replacing the last
    // valid checkpoint with a partial file.
    let (tmp_path, mut file) = create_temp_file(path)?;
    let write_result = (|| -> std::io::Result<()> {
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

fn write_tensor(buf: &mut Vec<u8>, name: &str, shape: &[usize], values: &[f32]) -> std::io::Result<()> {
    put_str(buf, name, "tensor name length")?;
    if shape.len() > MAX_RANK {
        return Err(bad("tensor rank exceeds the checkpoint format limit"));
    }
    put_u32(buf, checked_u32(shape.len(), "tensor rank")?);
    for &dim in shape {
        put_u32(buf, checked_u32(dim, "tensor dimension")?);
    }
    let expected = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
    if expected != Some(values.len()) {
        return Err(bad("live tensor shape does not match its value count"));
    }
    put_u32(buf, checked_u32(values.len(), "tensor element count")?);
    for &value in values {
        if !value.is_finite() {
            return Err(bad("checkpoint tensors must contain only finite values"));
        }
        buf.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

pub struct Ckpt {
    pub meta: HashMap<String, String>,
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl Ckpt {
    pub fn require_usize(&self, key: &str) -> std::io::Result<usize> {
        let value = self.meta.get(key).ok_or_else(|| bad(&format!("checkpoint metadata is missing '{}'", key)))?;
        value.parse::<usize>().map_err(|_| bad(&format!("checkpoint metadata '{}' is not a valid integer", key)))
    }

    pub fn require_f32(&self, key: &str) -> std::io::Result<f32> {
        let value = self.meta.get(key).ok_or_else(|| bad(&format!("checkpoint metadata is missing '{}'", key)))?;
        let parsed = value.parse::<f32>().map_err(|_| bad(&format!("checkpoint metadata '{}' is not a valid number", key)))?;
        if !parsed.is_finite() {
            return Err(bad(&format!("checkpoint metadata '{}' must be finite", key)));
        }
        Ok(parsed)
    }

    pub fn optional_f32(&self, key: &str, default: f32) -> std::io::Result<f32> {
        match self.meta.get(key) {
            Some(_) => self.require_f32(key),
            None => Ok(default),
        }
    }

    pub fn meta_usize(&self, key: &str, default: usize) -> usize {
        match self.meta.get(key) {
            Some(value) => value.parse::<usize>().unwrap_or(default),
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
    let file_size = std::fs::metadata(path)?.len();
    if file_size > MAX_CHECKPOINT_FILE_BYTES {
        return Err(bad("checkpoint file exceeds the supported size limit"));
    }
    let raw = std::fs::read(path)?;
    if raw.len() as u64 > MAX_CHECKPOINT_FILE_BYTES {
        return Err(bad("checkpoint file changed while it was being read"));
    }
    if raw.len() < 24 {
        return Err(bad("checkpoint too small"));
    }
    let body = &raw[..raw.len() - 4];
    let checksum_bytes = &raw[raw.len() - 4..];
    let expected_checksum = u32::from_le_bytes([checksum_bytes[0], checksum_bytes[1], checksum_bytes[2], checksum_bytes[3]]);
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
            let value = cursor.f32()?;
            if !value.is_finite() {
                return Err(bad("checkpoint tensor contains a non-finite value"));
            }
            values.push(value);
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

/// Validate a complete checkpoint against a live graph and apply it
/// transactionally. No parameter is changed unless every expected tensor is
/// present with the exact shape and there are no unexpected model tensors.
/// Tensors in the reserved [`AUX_PREFIX`] namespace (optimizer state) are
/// ignored here and applied by their own owner.
pub fn apply_exact(g: &mut Graph, ckpt: &Ckpt) -> std::io::Result<usize> {
    let model_tensors = ckpt.tensors.keys().filter(|name| !name.starts_with(AUX_PREFIX)).count();
    let model_shapes = ckpt.shapes.keys().filter(|name| !name.starts_with(AUX_PREFIX)).count();
    if model_tensors != g.params.len() || model_shapes != g.params.len() {
        return Err(bad("checkpoint tensor set does not exactly match the model"));
    }
    for parameter in &g.params {
        let id = parameter.id;
        let values =
            ckpt.tensors.get(&parameter.name).ok_or_else(|| bad(&format!("checkpoint is missing tensor '{}'", parameter.name)))?;
        let shape = ckpt
            .shapes
            .get(&parameter.name)
            .ok_or_else(|| bad(&format!("checkpoint is missing shape for '{}'", parameter.name)))?;
        if shape != &g.shape[id] || values.len() != g.val[id].len() {
            return Err(bad(&format!("checkpoint tensor '{}' has the wrong shape", parameter.name)));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(bad(&format!("checkpoint tensor '{}' contains a non-finite value", parameter.name)));
        }
    }
    for parameter in &g.params {
        let id = parameter.id;
        g.val[id].copy_from_slice(&ckpt.tensors[&parameter.name]);
    }
    Ok(g.params.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_apply_is_transactional() {
        let mut graph = Graph::new(1);
        let first = graph.param("first", vec![2], vec![1.0, 2.0], true);
        let second = graph.param("second", vec![1], vec![3.0], true);
        graph.seal_params();

        let mut tensors = HashMap::new();
        tensors.insert("first".to_string(), vec![10.0, 20.0]);
        tensors.insert("second".to_string(), vec![30.0]);
        let mut shapes = HashMap::new();
        shapes.insert("first".to_string(), vec![2]);
        shapes.insert("second".to_string(), vec![2]);
        let checkpoint = Ckpt { meta: HashMap::new(), tensors, shapes };

        assert!(apply_exact(&mut graph, &checkpoint).is_err());
        assert_eq!(graph.val[first], vec![1.0, 2.0]);
        assert_eq!(graph.val[second], vec![3.0]);
    }
}
