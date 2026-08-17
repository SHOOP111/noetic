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
//! elsewhere in the model, and shape mismatches are reported instead of
//! silently reinterpreted.

use crate::autograd::Graph;
use std::collections::HashMap;
use std::io::Write;

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
    for i in 0..data.len() {
        let idx = ((crc ^ (data[i] as u32)) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

pub fn save(path: &str, g: &Graph, meta: &[(String, String)]) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"NOETIC\x00\x01");
    put_u32(&mut buf, 1);
    put_u32(&mut buf, meta.len() as u32);
    for i in 0..meta.len() {
        put_str(&mut buf, &meta[i].0);
        put_str(&mut buf, &meta[i].1);
    }
    put_u32(&mut buf, g.params.len() as u32);
    for p in 0..g.params.len() {
        let id = g.params[p].id;
        put_str(&mut buf, &g.params[p].name);
        let sh = &g.shape[id];
        put_u32(&mut buf, sh.len() as u32);
        for i in 0..sh.len() {
            put_u32(&mut buf, sh[i] as u32);
        }
        let v = &g.val[id];
        put_u32(&mut buf, v.len() as u32);
        for i in 0..v.len() {
            buf.extend_from_slice(&v[i].to_le_bytes());
        }
    }
    let c = crc32(&buf);
    put_u32(&mut buf, c);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&buf)?;
    Ok(())
}

pub struct Ckpt {
    pub meta: HashMap<String, String>,
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl Ckpt {
    pub fn meta_usize(&self, key: &str, default: usize) -> usize {
        match self.meta.get(key) {
            Some(v) => match v.parse::<usize>() {
                Ok(x) => x,
                Err(_) => default,
            },
            None => default,
        }
    }
    pub fn meta_f32(&self, key: &str, default: f32) -> f32 {
        match self.meta.get(key) {
            Some(v) => match v.parse::<f32>() {
                Ok(x) => x,
                Err(_) => default,
            },
            None => default,
        }
    }
}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}

fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> std::io::Result<&'a [u8]> {
        if self.i + n > self.b.len() {
            return Err(bad("checkpoint truncated"));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> std::io::Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn f32(&mut self) -> std::io::Result<f32> {
        let s = self.take(4)?;
        Ok(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn string(&mut self) -> std::io::Result<String> {
        let n = self.u32()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).to_string())
    }
}

pub fn load(path: &str) -> std::io::Result<Ckpt> {
    let raw = std::fs::read(path)?;
    if raw.len() < 16 {
        return Err(bad("checkpoint too small"));
    }
    let body = &raw[..raw.len() - 4];
    let want = u32::from_le_bytes([
        raw[raw.len() - 4],
        raw[raw.len() - 3],
        raw[raw.len() - 2],
        raw[raw.len() - 1],
    ]);
    let got = crc32(body);
    if want != got {
        return Err(bad("checkpoint CRC mismatch (corrupt file)"));
    }
    let mut c = Cur { b: body, i: 0 };
    let magic = c.take(8)?;
    if magic != b"NOETIC\x00\x01" {
        return Err(bad("not a noetic checkpoint"));
    }
    let _version = c.u32()?;
    let n_meta = c.u32()? as usize;
    let mut meta = HashMap::new();
    for _ in 0..n_meta {
        let k = c.string()?;
        let v = c.string()?;
        meta.insert(k, v);
    }
    let n_t = c.u32()? as usize;
    let mut tensors = HashMap::new();
    let mut shapes = HashMap::new();
    for _ in 0..n_t {
        let name = c.string()?;
        let rank = c.u32()? as usize;
        let mut sh = Vec::with_capacity(rank);
        for _ in 0..rank {
            sh.push(c.u32()? as usize);
        }
        let n = c.u32()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(c.f32()?);
        }
        tensors.insert(name.clone(), v);
        shapes.insert(name, sh);
    }
    Ok(Ckpt { meta, tensors, shapes })
}

/// Copy checkpoint tensors into the live graph by parameter name.
/// Returns (loaded, missing, mismatched).
pub fn apply(g: &mut Graph, ck: &Ckpt) -> (usize, usize, usize) {
    let mut loaded = 0usize;
    let mut missing = 0usize;
    let mut mismatch = 0usize;
    for p in 0..g.params.len() {
        let id = g.params[p].id;
        let name = g.params[p].name.clone();
        match ck.tensors.get(&name) {
            Some(v) => {
                if v.len() == g.val[id].len() {
                    for i in 0..v.len() {
                        g.val[id][i] = v[i];
                    }
                    loaded += 1;
                } else {
                    mismatch += 1;
                }
            }
            None => {
                missing += 1;
            }
        }
    }
    (loaded, missing, mismatch)
}
