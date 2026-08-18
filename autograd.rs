//! Reverse-mode automatic differentiation on a flat arena tape.
//!
//! Design notes (why it looks like this):
//!
//! * Nodes are `usize` indices into parallel `Vec`s, not `Rc<RefCell<..>>`.
//!   No reference counting, no interior mutability, no lifetime plumbing,
//!   perfect cache locality, and the tape can be truncated in O(1).
//! * Every op is an enum variant, so `backward` is one `match` - no boxed
//!   closures, no dynamic dispatch, no allocation per node.
//! * Ops that need saved state (rms scale, softmax probabilities, targets)
//!   stash it in `aux` / `ids` on their own node.
//! * `reset()` truncates back to the parameter watermark: parameters and their
//!   gradient buffers survive, activations are freed. One allocation-free
//!   training step after warm-up.
//! * `no_grad` mode stops the tape from recording, so evaluation costs nothing
//!   extra in memory.

use crate::scan::{scan_adjoint, scan_chunked, scan_sequential};
use crate::tensor::{gemm_nn, gemm_nt, gemm_tn, sigmoid};

pub type Nid = usize;

/// Upper bound on recycled activation buffers held between steps.
const MAX_POOLED_BUFFERS: usize = 4096;

/// sqrt(2/pi) and the cubic coefficient of the tanh GELU approximation, at f32
/// precision so `tensor::gelu` and `Op::Gelu` agree exactly.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_CUBIC: f32 = 0.044_715;

#[inline]
fn checked_numel(shape: &[usize]) -> usize {
    shape.iter().try_fold(1usize, |product, &dimension| product.checked_mul(dimension)).expect("tensor shape product overflow")
}

#[inline]
fn checked_2(left: usize, right: usize, what: &str) -> usize {
    left.checked_mul(right).unwrap_or_else(|| panic!("{} size overflow", what))
}

#[inline]
fn checked_3(first: usize, second: usize, third: usize, what: &str) -> usize {
    first.checked_mul(second).and_then(|value| value.checked_mul(third)).unwrap_or_else(|| panic!("{} size overflow", what))
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Leaf,
    Add(Nid, Nid),
    Sub(Nid, Nid),
    Mul(Nid, Nid),
    /// x[rows,d] + bias[d]
    AddRow(Nid, Nid),
    /// x[rows,d] * gain[d]
    MulRow(Nid, Nid),
    Scale(Nid, f32),
    OneMinus(Nid),
    /// a[m,k] @ b[k,n]
    MatMulNN(Nid, Nid, usize, usize, usize),
    /// x[m,k] @ w[n,k]^T
    MatMulNT(Nid, Nid, usize, usize, usize),
    Silu(Nid),
    Gelu(Nid),
    Sigmoid(Nid),
    Tanh(Nid),
    /// x, rows, d, eps
    RmsNorm(Nid, usize, usize, f32),
    /// x, rows, d_total, offset, len
    SliceCols(Nid, usize, usize, usize, usize),
    /// table[vocab,d], d
    Embed(Nid, usize),
    /// a, b, batch, t, d
    Scan(Nid, Nid, usize, usize, usize),
    /// x, w[k,d], bias[d], batch, t, d, k  (depthwise causal conv)
    DwConv(Nid, Nid, Nid, usize, usize, usize, usize),
    /// logits[rows,vocab] + hard targets
    SoftmaxCe(Nid, usize, usize),
    /// logits[rows,k] + soft target distribution
    SoftCeDist(Nid, usize, usize),
    /// pred[rows] + targets
    MseTarget(Nid, usize),
    Sum(Nid),
}

pub struct Param {
    pub id: Nid,
    pub name: String,
    /// false for biases / norm gains: they are excluded from weight decay
    pub decay: bool,
}

pub struct Graph {
    pub val: Vec<Vec<f32>>,
    pub grad: Vec<Vec<f32>>,
    pub shape: Vec<Vec<usize>>,
    pub op: Vec<Op>,
    pub req: Vec<bool>,
    pub aux: Vec<Vec<f32>>,
    pub ids: Vec<Vec<u32>>,
    pub params: Vec<Param>,
    pub threads: usize,
    /// Recycled activation buffers. `reset()` returns every activation here
    /// instead of freeing it, so steady-state steps reuse memory.
    pool: Vec<Vec<f32>>,
    /// Fresh allocations served by `buf`. Steady-state steps should not grow it.
    allocations: u64,
    pub no_grad: bool,
    /// Scan implementation policy. See [`ScanPolicy`].
    pub scan_policy: ScanPolicy,
    watermark: usize,
    sealed: bool,
}

/// Which prefix-scan kernel [`Graph::scan`] uses.
///
/// `Auto` is the default and picks the time-chunked kernel only when the batch
/// axis is too narrow to keep the threads busy, because that is the only regime
/// where chunking has ever measured faster here (see `scan.rs` for numbers: on
/// a 4-core laptop it wins ~1.15x on cache-resident shapes and *loses* on
/// DRAM-resident ones, so this is a latency knob, not a throughput win).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanPolicy {
    Auto,
    Sequential,
    Chunked,
}

impl Graph {
    pub fn new(threads: usize) -> Graph {
        Graph {
            val: Vec::new(),
            grad: Vec::new(),
            shape: Vec::new(),
            op: Vec::new(),
            req: Vec::new(),
            aux: Vec::new(),
            ids: Vec::new(),
            params: Vec::new(),
            threads: threads.max(1),
            pool: Vec::new(),
            allocations: 0,
            no_grad: false,
            scan_policy: ScanPolicy::Auto,
            watermark: 0,
            sealed: false,
        }
    }

    pub fn nodes(&self) -> usize {
        self.val.len()
    }

    /// Number of buffers currently parked in the recycling pool.
    pub fn pooled_buffers(&self) -> usize {
        self.pool.len()
    }

    /// How many activation buffers have been allocated from scratch. Constant
    /// across steady-state training steps once the pool has warmed up.
    pub fn allocation_count(&self) -> u64 {
        self.allocations
    }

    /// A buffer of length `n` whose contents are initialised but unspecified,
    /// reusing pooled capacity when possible. Forward ops overwrite every
    /// element they produce, so re-zeroing a recycled buffer would just be a
    /// wasted pass over memory - `buf_zeroed` exists for the accumulating
    /// (backward) case.
    fn buf(&mut self, n: usize) -> Vec<f32> {
        let mut index = None;
        for (position, candidate) in self.pool.iter().enumerate().rev() {
            if candidate.capacity() >= n {
                index = Some(position);
                break;
            }
        }
        match index {
            Some(position) => {
                let mut buffer = self.pool.swap_remove(position);
                if buffer.len() < n {
                    buffer.resize(n, 0.0);
                } else {
                    buffer.truncate(n);
                }
                buffer
            }
            None => {
                self.allocations = self.allocations.saturating_add(1);
                vec![0.0f32; n]
            }
        }
    }

    /// A zeroed buffer of length `n`. Required wherever the consumer accumulates
    /// into the buffer instead of writing every element.
    fn buf_zeroed(&mut self, n: usize) -> Vec<f32> {
        let mut buffer = self.buf(n);
        for value in buffer.iter_mut() {
            *value = 0.0;
        }
        buffer
    }

    fn recycle(&mut self, buffer: Vec<f32>) {
        // Empty buffers carry no capacity worth keeping, and an unbounded pool
        // would defeat the point of freeing activations.
        if buffer.capacity() > 0 && self.pool.len() < MAX_POOLED_BUFFERS {
            self.pool.push(buffer);
        }
    }

    fn push(&mut self, shape: Vec<usize>, val: Vec<f32>, op: Op, req: bool) -> Nid {
        assert_eq!(checked_numel(&shape), val.len(), "tensor shape does not match its value count");
        self.val.push(val);
        self.grad.push(Vec::new());
        self.shape.push(shape);
        self.op.push(op);
        self.req.push(req && !self.no_grad);
        self.aux.push(Vec::new());
        self.ids.push(Vec::new());
        self.val.len() - 1
    }

    pub fn constant(&mut self, shape: Vec<usize>, val: Vec<f32>) -> Nid {
        self.push(shape, val, Op::Leaf, false)
    }

    pub fn input(&mut self, shape: Vec<usize>, val: Vec<f32>) -> Nid {
        self.push(shape, val, Op::Leaf, false)
    }

    /// A trainable leaf. Gradient buffer is allocated once and never freed.
    pub fn param(&mut self, name: &str, shape: Vec<usize>, val: Vec<f32>, decay: bool) -> Nid {
        assert!(!self.sealed, "cannot add parameters after Graph::seal_params");
        assert!(!name.is_empty(), "parameter name cannot be empty");
        assert!(self.params.iter().all(|parameter| parameter.name != name), "duplicate parameter name: {}", name);
        let n = val.len();
        let id = self.push(shape, val, Op::Leaf, true);
        self.req[id] = true; // parameters always require grad, even under no_grad
        self.grad[id] = vec![0.0f32; n];
        self.params.push(Param { id, name: name.to_string(), decay });
        id
    }

    /// Freeze the current tape length: `reset()` returns here.
    pub fn seal_params(&mut self) {
        assert!(!self.sealed, "Graph::seal_params may only be called once");
        assert!(self.op.iter().all(|op| matches!(op, Op::Leaf)), "seal parameters before building an operation tape");
        self.watermark = self.val.len();
        self.sealed = true;
    }

    /// Retire all activations, keep parameters + their gradients. Activation
    /// buffers move to the recycling pool rather than back to the allocator.
    pub fn reset(&mut self) {
        assert!(self.sealed, "call Graph::seal_params before Graph::reset");
        let w = self.watermark;
        let retired: Vec<Vec<f32>> = self.val.drain(w..).chain(self.grad.drain(w..)).chain(self.aux.drain(w..)).collect();
        for buffer in retired {
            self.recycle(buffer);
        }
        self.val.truncate(w);
        self.grad.truncate(w);
        self.shape.truncate(w);
        self.op.truncate(w);
        self.req.truncate(w);
        self.aux.truncate(w);
        self.ids.truncate(w);
    }

    pub fn zero_grad(&mut self) {
        for p in 0..self.params.len() {
            let id = self.params[p].id;
            let n = self.val[id].len();
            if self.grad[id].len() != n {
                self.grad[id] = vec![0.0f32; n];
            } else {
                for x in self.grad[id].iter_mut() {
                    *x = 0.0;
                }
            }
        }
    }

    pub fn scalar(&self, id: Nid) -> f32 {
        assert_eq!(self.val[id].len(), 1, "requested node is not a scalar");
        self.val[id][0]
    }

    pub fn numel(&self, id: Nid) -> usize {
        self.val[id].len()
    }

    // ================= forward ops =================

    pub fn add(&mut self, a: Nid, b: Nid) -> Nid {
        let n = self.val[a].len();
        assert_eq!(n, self.val[b].len(), "add: shape mismatch");
        assert_eq!(self.shape[a], self.shape[b], "add: shape mismatch");
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = self.val[a][i] + self.val[b][i];
        }
        let req = self.req[a] || self.req[b];
        let sh = self.shape[a].clone();
        self.push(sh, out, Op::Add(a, b), req)
    }

    pub fn sub(&mut self, a: Nid, b: Nid) -> Nid {
        let n = self.val[a].len();
        assert_eq!(n, self.val[b].len(), "sub: shape mismatch");
        assert_eq!(self.shape[a], self.shape[b], "sub: shape mismatch");
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = self.val[a][i] - self.val[b][i];
        }
        let req = self.req[a] || self.req[b];
        let sh = self.shape[a].clone();
        self.push(sh, out, Op::Sub(a, b), req)
    }

    pub fn mul(&mut self, a: Nid, b: Nid) -> Nid {
        let n = self.val[a].len();
        assert_eq!(n, self.val[b].len(), "mul: shape mismatch");
        assert_eq!(self.shape[a], self.shape[b], "mul: shape mismatch");
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = self.val[a][i] * self.val[b][i];
        }
        let req = self.req[a] || self.req[b];
        let sh = self.shape[a].clone();
        self.push(sh, out, Op::Mul(a, b), req)
    }

    pub fn scale(&mut self, a: Nid, k: f32) -> Nid {
        let n = self.val[a].len();
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = self.val[a][i] * k;
        }
        let req = self.req[a];
        let sh = self.shape[a].clone();
        self.push(sh, out, Op::Scale(a, k), req)
    }

    /// 1 - x, the complement gate of the recurrence.
    pub fn one_minus(&mut self, a: Nid) -> Nid {
        let n = self.val[a].len();
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = 1.0 - self.val[a][i];
        }
        let req = self.req[a];
        let sh = self.shape[a].clone();
        self.push(sh, out, Op::OneMinus(a), req)
    }

    pub fn add_row(&mut self, x: Nid, bias: Nid) -> Nid {
        let d = self.val[bias].len();
        let n = self.val[x].len();
        assert!(d > 0 && n % d == 0, "add_row: bad shapes");
        assert_eq!(self.shape[bias], vec![d], "add_row: bias must be one-dimensional");
        assert_eq!(self.shape[x].last().copied(), Some(d), "add_row: trailing dimension mismatch");
        let rows = n / d;
        let mut out = self.buf(n);
        for i in 0..rows {
            for j in 0..d {
                out[i * d + j] = self.val[x][i * d + j] + self.val[bias][j];
            }
        }
        let req = self.req[x] || self.req[bias];
        let sh = self.shape[x].clone();
        self.push(sh, out, Op::AddRow(x, bias), req)
    }

    pub fn mul_row(&mut self, x: Nid, gain: Nid) -> Nid {
        let d = self.val[gain].len();
        let n = self.val[x].len();
        assert!(d > 0 && n % d == 0, "mul_row: bad shapes");
        assert_eq!(self.shape[gain], vec![d], "mul_row: gain must be one-dimensional");
        assert_eq!(self.shape[x].last().copied(), Some(d), "mul_row: trailing dimension mismatch");
        let rows = n / d;
        let mut out = self.buf(n);
        for i in 0..rows {
            for j in 0..d {
                out[i * d + j] = self.val[x][i * d + j] * self.val[gain][j];
            }
        }
        let req = self.req[x] || self.req[gain];
        let sh = self.shape[x].clone();
        self.push(sh, out, Op::MulRow(x, gain), req)
    }

    pub fn matmul_nn(&mut self, a: Nid, b: Nid, m: usize, k: usize, n: usize) -> Nid {
        let mut out = self.buf(checked_2(m, n, "matrix product"));
        gemm_nn(&self.val[a], &self.val[b], &mut out, m, k, n, self.threads);
        let req = self.req[a] || self.req[b];
        self.push(vec![m, n], out, Op::MatMulNN(a, b, m, k, n), req)
    }

    /// x[m,k] @ w[n,k]^T -> [m,n]. Weights are stored [out,in].
    pub fn matmul_nt(&mut self, x: Nid, w: Nid, m: usize, k: usize, n: usize) -> Nid {
        let mut out = self.buf(checked_2(m, n, "matrix product"));
        gemm_nt(&self.val[x], &self.val[w], &mut out, m, k, n, self.threads);
        let req = self.req[x] || self.req[w];
        self.push(vec![m, n], out, Op::MatMulNT(x, w, m, k, n), req)
    }

    pub fn silu(&mut self, x: Nid) -> Nid {
        let n = self.val[x].len();
        let mut out = self.buf(n);
        for i in 0..n {
            let v = self.val[x][i];
            out[i] = v * sigmoid(v);
        }
        let req = self.req[x];
        let sh = self.shape[x].clone();
        self.push(sh, out, Op::Silu(x), req)
    }

    pub fn gelu(&mut self, x: Nid) -> Nid {
        let n = self.val[x].len();
        let mut out = self.buf(n);
        for i in 0..n {
            let v = self.val[x][i];
            let u = SQRT_2_OVER_PI * (v + GELU_CUBIC * v * v * v);
            out[i] = 0.5 * v * (1.0 + u.tanh());
        }
        let req = self.req[x];
        let sh = self.shape[x].clone();
        self.push(sh, out, Op::Gelu(x), req)
    }

    pub fn sigmoid(&mut self, x: Nid) -> Nid {
        let n = self.val[x].len();
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = sigmoid(self.val[x][i]);
        }
        let req = self.req[x];
        let sh = self.shape[x].clone();
        self.push(sh, out, Op::Sigmoid(x), req)
    }

    pub fn tanh(&mut self, x: Nid) -> Nid {
        let n = self.val[x].len();
        let mut out = self.buf(n);
        for i in 0..n {
            out[i] = self.val[x][i].tanh();
        }
        let req = self.req[x];
        let sh = self.shape[x].clone();
        self.push(sh, out, Op::Tanh(x), req)
    }

    /// Root-mean-square normalisation (no mean subtraction, no bias).
    /// aux stores the per-row scale 1/sqrt(mean(x^2)+eps).
    pub fn rms_norm(&mut self, x: Nid, rows: usize, d: usize, eps: f32) -> Nid {
        assert!(d > 0, "rms_norm: width must be positive");
        assert!(eps.is_finite() && eps > 0.0, "rms_norm: epsilon must be finite and positive");
        let n = checked_2(rows, d, "RMSNorm");
        assert_eq!(self.val[x].len(), n, "rms_norm: bad shape");
        let mut out = self.buf(n);
        let mut scale = self.buf(rows);
        for i in 0..rows {
            let mut ms = 0.0f32;
            for j in 0..d {
                let v = self.val[x][i * d + j];
                ms += v * v;
            }
            let r = 1.0 / (ms / (d as f32) + eps).sqrt();
            scale[i] = r;
            for j in 0..d {
                out[i * d + j] = self.val[x][i * d + j] * r;
            }
        }
        let req = self.req[x];
        let id = self.push(vec![rows, d], out, Op::RmsNorm(x, rows, d, eps), req);
        if req {
            self.aux[id] = scale;
        }
        id
    }

    /// Column slice: out[i, 0..len] = x[i, off..off+len]. Used to split fused
    /// projections into value / gate / decay streams for free.
    pub fn slice_cols(&mut self, x: Nid, rows: usize, dtot: usize, off: usize, len: usize) -> Nid {
        let end = off.checked_add(len).expect("slice_cols: range overflow");
        assert!(end <= dtot, "slice_cols: out of range");
        assert_eq!(self.val[x].len(), checked_2(rows, dtot, "column slice input"), "slice_cols: bad input shape");
        let out_len = checked_2(rows, len, "column slice output");
        let mut out = self.buf(out_len);
        for i in 0..rows {
            for j in 0..len {
                out[i * len + j] = self.val[x][i * dtot + off + j];
            }
        }
        let req = self.req[x];
        self.push(vec![rows, len], out, Op::SliceCols(x, rows, dtot, off, len), req)
    }

    pub fn embed(&mut self, table: Nid, d: usize, ids: &[u32]) -> Nid {
        assert!(d > 0, "embed: width must be positive");
        assert_eq!(self.val[table].len() % d, 0, "embed: table shape mismatch");
        let rows = ids.len();
        let vocab = self.val[table].len() / d;
        let mut out = self.buf(checked_2(rows, d, "embedding output"));
        for i in 0..rows {
            let t = ids[i] as usize;
            assert!(t < vocab, "embed: token out of range");
            for j in 0..d {
                out[i * d + j] = self.val[table][t * d + j];
            }
        }
        let req = self.req[table];
        let id = self.push(vec![rows, d], out, Op::Embed(table, d), req);
        self.ids[id] = ids.to_vec();
        id
    }

    /// h_t = a_t * h_{t-1} + b_t over [batch, t, d].
    pub fn scan(&mut self, a: Nid, b: Nid, batch: usize, t: usize, d: usize) -> Nid {
        let n = checked_3(batch, t, d, "scan");
        let rows = checked_2(batch, t, "scan rows");
        assert_eq!(self.val[a].len(), n, "scan: bad a");
        assert_eq!(self.val[b].len(), n, "scan: bad b");
        let mut h = self.buf(n);
        if self.chunked_scan_wanted(batch, t, d) {
            scan_chunked(&self.val[a], &self.val[b], &mut h, batch, t, d, self.threads);
        } else {
            scan_sequential(&self.val[a], &self.val[b], &mut h, batch, t, d, self.threads);
        }
        let req = self.req[a] || self.req[b];
        self.push(vec![rows, d], h, Op::Scan(a, b, batch, t, d), req)
    }

    /// Chunking pays only when the batch axis cannot fill the cores and the
    /// sequence is long enough to amortise the extra pass over `a`.
    fn chunked_scan_wanted(&self, batch: usize, t: usize, d: usize) -> bool {
        match self.scan_policy {
            ScanPolicy::Sequential => false,
            ScanPolicy::Chunked => true,
            ScanPolicy::Auto => self.threads > 1 && batch * 2 <= self.threads && t >= 512 && batch * t * d >= 1 << 16,
        }
    }

    /// Depthwise causal 1-D convolution: y[b,t,j] = bias[j] + sum_q w[q,j] x[b,t-q,j].
    /// Provides the short-range mixing the recurrence does not need to spend
    /// state capacity on (left-padded, so it never looks into the future).
    pub fn dwconv(&mut self, x: Nid, w: Nid, bias: Nid, batch: usize, t: usize, d: usize, k: usize) -> Nid {
        assert!(d > 0, "dwconv: width must be positive");
        assert!(k > 0, "dwconv: kernel width must be positive");
        let n = checked_3(batch, t, d, "depthwise convolution");
        let rows = checked_2(batch, t, "depthwise convolution rows");
        assert_eq!(self.val[x].len(), n, "dwconv: bad x");
        assert_eq!(self.val[w].len(), checked_2(k, d, "depthwise convolution weights"), "dwconv: bad w");
        assert_eq!(self.val[bias].len(), d, "dwconv: bad bias");
        let mut out = self.buf(n);
        for bi in 0..batch {
            let base = bi * t * d;
            for ti in 0..t {
                let o = base + ti * d;
                out[o..o + d].copy_from_slice(&self.val[bias][..d]);
                let qmax = if ti + 1 < k { ti + 1 } else { k };
                for q in 0..qmax {
                    let src = base + (ti - q) * d;
                    for j in 0..d {
                        out[o + j] += self.val[w][q * d + j] * self.val[x][src + j];
                    }
                }
            }
        }
        let req = self.req[x] || self.req[w] || self.req[bias];
        self.push(vec![rows, d], out, Op::DwConv(x, w, bias, batch, t, d, k), req)
    }

    /// Mean cross-entropy over rows. `u32::MAX` targets are ignored.
    /// aux stores softmax probabilities so backward is a single subtraction.
    pub fn softmax_ce(&mut self, logits: Nid, rows: usize, vocab: usize, targets: &[u32]) -> Nid {
        assert!(vocab > 0, "softmax_ce: vocabulary must be positive");
        let elements = checked_2(rows, vocab, "softmax cross-entropy");
        assert_eq!(self.val[logits].len(), elements, "softmax_ce: bad logits");
        assert_eq!(targets.len(), rows, "softmax_ce: bad targets");
        let mut probs = self.buf(elements);
        let mut loss = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..rows {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..vocab {
                let v = self.val[logits][i * vocab + j];
                assert!(v.is_finite(), "softmax_ce: logits must be finite");
                if v > mx {
                    mx = v;
                }
            }
            let mut s = 0.0f32;
            for j in 0..vocab {
                let e = (self.val[logits][i * vocab + j] - mx).exp();
                probs[i * vocab + j] = e;
                s += e;
            }
            let inv = 1.0 / s.max(1e-30);
            for j in 0..vocab {
                probs[i * vocab + j] *= inv;
            }
            let t = targets[i];
            if t != u32::MAX {
                let ti = t as usize;
                assert!(ti < vocab, "softmax_ce: target out of range");
                let p = probs[i * vocab + ti].max(1e-20);
                loss -= (p as f64).ln();
                cnt += 1;
            }
        }
        let denom = if cnt == 0 { 1.0f64 } else { cnt as f64 };
        let out = vec![(loss / denom) as f32];
        let req = self.req[logits];
        let id = self.push(vec![1], out, Op::SoftmaxCe(logits, rows, vocab), req);
        if req {
            self.aux[id] = probs;
            self.ids[id] = targets.to_vec();
        }
        id
    }

    /// Cross-entropy against a *distribution* target (planner policy head).
    pub fn soft_ce(&mut self, logits: Nid, rows: usize, k: usize, target: &[f32]) -> Nid {
        assert!(k > 0, "soft_ce: class count must be positive");
        let elements = checked_2(rows, k, "distribution cross-entropy");
        assert_eq!(self.val[logits].len(), elements, "soft_ce: bad logits");
        assert_eq!(target.len(), elements, "soft_ce: bad target");
        let mut probs = self.buf(elements);
        let mut loss = 0.0f64;
        for i in 0..rows {
            let mut mx = f32::NEG_INFINITY;
            let mut target_sum = 0.0f32;
            for j in 0..k {
                let v = self.val[logits][i * k + j];
                let target_value = target[i * k + j];
                assert!(v.is_finite(), "soft_ce: logits must be finite");
                assert!(target_value.is_finite() && target_value >= 0.0, "soft_ce: target weights must be finite and non-negative");
                target_sum += target_value;
                if v > mx {
                    mx = v;
                }
            }
            assert!(target_sum > 0.0, "soft_ce: every target row must have positive mass");
            let mut s = 0.0f32;
            for j in 0..k {
                let e = (self.val[logits][i * k + j] - mx).exp();
                probs[i * k + j] = e;
                s += e;
            }
            let inv = 1.0 / s.max(1e-30);
            for j in 0..k {
                probs[i * k + j] *= inv;
                let p = probs[i * k + j].max(1e-20);
                loss -= (target[i * k + j] as f64) * (p as f64).ln();
            }
        }
        let out = vec![(loss / (rows.max(1) as f64)) as f32];
        let req = self.req[logits];
        let id = self.push(vec![1], out, Op::SoftCeDist(logits, rows, k), req);
        if req {
            self.aux[id] = probs;
            let mut tv = Vec::with_capacity(target.len());
            for i in 0..target.len() {
                tv.push(target[i].to_bits());
            }
            self.ids[id] = tv;
        }
        id
    }

    /// Mean squared error against fixed targets (planner value head).
    pub fn mse(&mut self, pred: Nid, target: &[f32]) -> Nid {
        let n = self.val[pred].len();
        assert_eq!(n, target.len(), "mse: bad target");
        let mut loss = 0.0f64;
        for i in 0..n {
            assert!(self.val[pred][i].is_finite(), "mse: predictions must be finite");
            assert!(target[i].is_finite(), "mse: targets must be finite");
            let d = (self.val[pred][i] - target[i]) as f64;
            loss += d * d;
        }
        let out = vec![(loss / (n.max(1) as f64)) as f32];
        let req = self.req[pred];
        let id = self.push(vec![1], out, Op::MseTarget(pred, n), req);
        if req {
            self.aux[id] = target.to_vec();
        }
        id
    }

    pub fn sum(&mut self, x: Nid) -> Nid {
        let mut s = 0.0f64;
        for i in 0..self.val[x].len() {
            s += self.val[x][i] as f64;
        }
        let req = self.req[x];
        self.push(vec![1], vec![s as f32], Op::Sum(x), req)
    }
}

// ================= reverse pass =================

impl Graph {
    /// Accumulate d(loss)/d(node) for every node that requires grad.
    ///
    /// Single reverse sweep over the tape. Because the tape is append-only and
    /// every op only references *earlier* nodes, index order is already a
    /// topological order - no sorting, no visited set.
    ///
    /// Borrow strategy: the output gradient is `mem::take`n out of the arena
    /// for the duration of the arm, which lets us mutably borrow input
    /// gradients from the very same vector without any unsafe or cloning.
    pub fn backward(&mut self, loss: Nid) {
        assert!(loss < self.val.len(), "backward: loss node is out of range");
        assert_eq!(self.val[loss].len(), 1, "backward: loss node must be scalar");
        for i in 0..=loss {
            if self.req[i] {
                let n = self.val[i].len();
                if self.grad[i].len() != n {
                    let buffer = self.buf_zeroed(n);
                    let stale = std::mem::replace(&mut self.grad[i], buffer);
                    self.recycle(stale);
                }
            }
        }
        if !self.req[loss] {
            return;
        }
        self.grad[loss][0] = 1.0;

        let mut id = loss + 1;
        while id > 0 {
            id -= 1;
            if !self.req[id] || self.grad[id].is_empty() {
                continue;
            }
            let op = self.op[id];
            match op {
                Op::Leaf => {}

                Op::Add(a, b) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[a] {
                        let g = &mut self.grad[a];
                        for i in 0..go.len() {
                            g[i] += go[i];
                        }
                    }
                    if self.req[b] {
                        let g = &mut self.grad[b];
                        for i in 0..go.len() {
                            g[i] += go[i];
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Sub(a, b) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[a] {
                        let g = &mut self.grad[a];
                        for i in 0..go.len() {
                            g[i] += go[i];
                        }
                    }
                    if self.req[b] {
                        let g = &mut self.grad[b];
                        for i in 0..go.len() {
                            g[i] -= go[i];
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Mul(a, b) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[a] {
                        let v = &self.val[b];
                        let g = &mut self.grad[a];
                        for i in 0..go.len() {
                            g[i] += go[i] * v[i];
                        }
                    }
                    if self.req[b] {
                        let v = &self.val[a];
                        let g = &mut self.grad[b];
                        for i in 0..go.len() {
                            g[i] += go[i] * v[i];
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Scale(a, k) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[a] {
                        let g = &mut self.grad[a];
                        for i in 0..go.len() {
                            g[i] += go[i] * k;
                        }
                    }
                    self.grad[id] = go;
                }

                Op::OneMinus(a) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[a] {
                        let g = &mut self.grad[a];
                        for i in 0..go.len() {
                            g[i] -= go[i];
                        }
                    }
                    self.grad[id] = go;
                }

                Op::AddRow(x, bias) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    let d = self.val[bias].len();
                    let rows = go.len() / d;
                    if self.req[x] {
                        let g = &mut self.grad[x];
                        for i in 0..go.len() {
                            g[i] += go[i];
                        }
                    }
                    if self.req[bias] {
                        let g = &mut self.grad[bias];
                        for i in 0..rows {
                            for j in 0..d {
                                g[j] += go[i * d + j];
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::MulRow(x, gain) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    let d = self.val[gain].len();
                    let rows = go.len() / d;
                    if self.req[x] {
                        let w = &self.val[gain];
                        let g = &mut self.grad[x];
                        for i in 0..rows {
                            for j in 0..d {
                                g[i * d + j] += go[i * d + j] * w[j];
                            }
                        }
                    }
                    if self.req[gain] {
                        let v = &self.val[x];
                        let g = &mut self.grad[gain];
                        for i in 0..rows {
                            for j in 0..d {
                                g[j] += go[i * d + j] * v[i * d + j];
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::MatMulNN(a, b, m, k, n) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[a] {
                        // dA = dC @ B^T
                        let mut t = self.buf(checked_2(m, k, "MatMulNN input gradient"));
                        gemm_nt(&go, &self.val[b], &mut t, m, n, k, self.threads);
                        let g = &mut self.grad[a];
                        for i in 0..t.len() {
                            g[i] += t[i];
                        }
                        self.recycle(t);
                    }
                    if self.req[b] {
                        // dB = A^T @ dC
                        let mut t = self.buf(checked_2(k, n, "MatMulNN weight gradient"));
                        gemm_tn(&self.val[a], &go, &mut t, m, k, n, self.threads);
                        let g = &mut self.grad[b];
                        for i in 0..t.len() {
                            g[i] += t[i];
                        }
                        self.recycle(t);
                    }
                    self.grad[id] = go;
                }

                Op::MatMulNT(x, w, m, k, n) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        // dX = dY @ W
                        let mut t = self.buf(checked_2(m, k, "MatMulNT input gradient"));
                        gemm_nn(&go, &self.val[w], &mut t, m, n, k, self.threads);
                        let g = &mut self.grad[x];
                        for i in 0..t.len() {
                            g[i] += t[i];
                        }
                        self.recycle(t);
                    }
                    if self.req[w] {
                        // dW = dY^T @ X
                        let mut t = self.buf(checked_2(n, k, "MatMulNT weight gradient"));
                        gemm_tn(&go, &self.val[x], &mut t, m, n, k, self.threads);
                        let g = &mut self.grad[w];
                        for i in 0..t.len() {
                            g[i] += t[i];
                        }
                        self.recycle(t);
                    }
                    self.grad[id] = go;
                }

                Op::Silu(x) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let v = &self.val[x];
                        let g = &mut self.grad[x];
                        for i in 0..go.len() {
                            let s = sigmoid(v[i]);
                            g[i] += go[i] * s * (1.0 + v[i] * (1.0 - s));
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Gelu(x) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let v = &self.val[x];
                        let g = &mut self.grad[x];
                        let c = SQRT_2_OVER_PI;
                        for i in 0..go.len() {
                            let z = v[i];
                            let u = c * (z + GELU_CUBIC * z * z * z);
                            let th = u.tanh();
                            let du = c * (1.0 + 3.0 * GELU_CUBIC * z * z);
                            g[i] += go[i] * (0.5 * (1.0 + th) + 0.5 * z * (1.0 - th * th) * du);
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Sigmoid(x) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let y = &self.val[id];
                        let g = &mut self.grad[x];
                        for i in 0..go.len() {
                            g[i] += go[i] * y[i] * (1.0 - y[i]);
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Tanh(x) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let y = &self.val[id];
                        let g = &mut self.grad[x];
                        for i in 0..go.len() {
                            g[i] += go[i] * (1.0 - y[i] * y[i]);
                        }
                    }
                    self.grad[id] = go;
                }

                Op::RmsNorm(x, rows, d, _eps) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let scale = &self.aux[id];
                        let v = &self.val[x];
                        let inv_d = 1.0f32 / (d as f32);
                        let g = &mut self.grad[x];
                        for i in 0..rows {
                            let r = scale[i];
                            let mut dot = 0.0f32;
                            for j in 0..d {
                                dot += go[i * d + j] * v[i * d + j];
                            }
                            let k = r * r * r * dot * inv_d;
                            for j in 0..d {
                                g[i * d + j] += r * go[i * d + j] - k * v[i * d + j];
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::SliceCols(x, rows, dtot, off, len) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let g = &mut self.grad[x];
                        for i in 0..rows {
                            for j in 0..len {
                                g[i * dtot + off + j] += go[i * len + j];
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Embed(table, d) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[table] {
                        let toks = std::mem::take(&mut self.ids[id]);
                        let g = &mut self.grad[table];
                        for i in 0..toks.len() {
                            let t = toks[i] as usize;
                            for j in 0..d {
                                g[t * d + j] += go[i * d + j];
                            }
                        }
                        self.ids[id] = toks;
                    }
                    self.grad[id] = go;
                }

                Op::Scan(a, b, batch, t, d) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    let n = checked_3(batch, t, d, "scan adjoint");
                    let mut c = self.buf_zeroed(n);
                    scan_adjoint(&self.val[a], &go, &mut c, batch, t, d);
                    if self.req[b] {
                        let g = &mut self.grad[b];
                        for i in 0..n {
                            g[i] += c[i];
                        }
                    }
                    if self.req[a] {
                        // dL/da_t = c_t * h_{t-1}; h is this node's own value
                        let h = &self.val[id];
                        let g = &mut self.grad[a];
                        for bi in 0..batch {
                            let base = bi * t * d;
                            for ti in 1..t {
                                let cur = base + ti * d;
                                let prev = base + (ti - 1) * d;
                                for j in 0..d {
                                    g[cur + j] += c[cur + j] * h[prev + j];
                                }
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::DwConv(x, w, bias, batch, t, d, k) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let wv = &self.val[w];
                        let g = &mut self.grad[x];
                        for bi in 0..batch {
                            let base = bi * t * d;
                            for ti in 0..t {
                                let o = base + ti * d;
                                let qmax = if ti + 1 < k { ti + 1 } else { k };
                                for q in 0..qmax {
                                    let src = base + (ti - q) * d;
                                    for j in 0..d {
                                        g[src + j] += go[o + j] * wv[q * d + j];
                                    }
                                }
                            }
                        }
                    }
                    if self.req[w] {
                        let xv = &self.val[x];
                        let g = &mut self.grad[w];
                        for bi in 0..batch {
                            let base = bi * t * d;
                            for ti in 0..t {
                                let o = base + ti * d;
                                let qmax = if ti + 1 < k { ti + 1 } else { k };
                                for q in 0..qmax {
                                    let src = base + (ti - q) * d;
                                    for j in 0..d {
                                        g[q * d + j] += go[o + j] * xv[src + j];
                                    }
                                }
                            }
                        }
                    }
                    if self.req[bias] {
                        let g = &mut self.grad[bias];
                        let rows = checked_2(batch, t, "depthwise convolution gradient rows");
                        for i in 0..rows {
                            for j in 0..d {
                                g[j] += go[i * d + j];
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::SoftmaxCe(logits, rows, vocab) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[logits] {
                        let probs = &self.aux[id];
                        let targets = &self.ids[id];
                        let mut cnt = 0usize;
                        for i in 0..rows {
                            if targets[i] != u32::MAX {
                                cnt += 1;
                            }
                        }
                        let scale = go[0] / (cnt.max(1) as f32);
                        let g = &mut self.grad[logits];
                        for i in 0..rows {
                            if targets[i] == u32::MAX {
                                continue;
                            }
                            let ti = targets[i] as usize;
                            for j in 0..vocab {
                                let mut p = probs[i * vocab + j];
                                if j == ti {
                                    p -= 1.0;
                                }
                                g[i * vocab + j] += scale * p;
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::SoftCeDist(logits, rows, k) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[logits] {
                        let probs = &self.aux[id];
                        let tgt = &self.ids[id];
                        let scale = go[0] / (rows.max(1) as f32);
                        let g = &mut self.grad[logits];
                        for i in 0..rows {
                            let mut tsum = 0.0f32;
                            for j in 0..k {
                                tsum += f32::from_bits(tgt[i * k + j]);
                            }
                            for j in 0..k {
                                let tv = f32::from_bits(tgt[i * k + j]);
                                g[i * k + j] += scale * (tsum * probs[i * k + j] - tv);
                            }
                        }
                    }
                    self.grad[id] = go;
                }

                Op::MseTarget(pred, n) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[pred] {
                        let tgt = &self.aux[id];
                        let v = &self.val[pred];
                        let scale = 2.0 * go[0] / (n.max(1) as f32);
                        let g = &mut self.grad[pred];
                        for i in 0..n {
                            g[i] += scale * (v[i] - tgt[i]);
                        }
                    }
                    self.grad[id] = go;
                }

                Op::Sum(x) => {
                    let go = std::mem::take(&mut self.grad[id]);
                    if self.req[x] {
                        let g = &mut self.grad[x];
                        for i in 0..g.len() {
                            g[i] += go[0];
                        }
                    }
                    self.grad[id] = go;
                }
            }
        }
    }

    /// Global L2 norm of all parameter gradients.
    pub fn grad_norm(&self) -> f32 {
        let mut s = 0.0f64;
        for p in 0..self.params.len() {
            let g = &self.grad[self.params[p].id];
            for i in 0..g.len() {
                s += (g[i] as f64) * (g[i] as f64);
            }
        }
        (s.sqrt()) as f32
    }

    /// Rescale gradients so the global norm is at most `max_norm`.
    /// Returns the pre-clip norm (a useful training-health signal).
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        assert!(max_norm.is_finite() && max_norm >= 0.0, "gradient clip norm must be finite and non-negative");
        let norm = self.grad_norm();
        assert!(norm.is_finite(), "non-finite parameter gradient norm");
        if norm > max_norm && norm > 0.0 {
            let s = max_norm / norm;
            for p in 0..self.params.len() {
                let id = self.params[p].id;
                for x in self.grad[id].iter_mut() {
                    *x *= s;
                }
            }
        }
        norm
    }

    pub fn param_count(&self) -> usize {
        let mut n = 0usize;
        for p in 0..self.params.len() {
            n = n.checked_add(self.val[self.params[p].id].len()).expect("parameter count overflow");
        }
        n
    }
}

#[cfg(test)]
mod invariant_tests {
    use super::*;

    #[test]
    #[should_panic(expected = "call Graph::seal_params")]
    fn reset_requires_a_sealed_parameter_prefix() {
        let mut graph = Graph::new(1);
        let _ = graph.param("weight", vec![1], vec![0.0], true);
        graph.reset();
    }

    #[test]
    #[should_panic(expected = "cannot add parameters")]
    fn parameters_cannot_be_added_after_sealing() {
        let mut graph = Graph::new(1);
        graph.seal_params();
        let _ = graph.param("late", vec![1], vec![0.0], true);
    }

    #[test]
    fn activations_are_recycled_between_steps() {
        let mut graph = Graph::new(1);
        let weight = graph.param("w", vec![64], vec![0.5f32; 64], true);
        graph.seal_params();
        let mut allocations = Vec::new();
        for _ in 0..4 {
            graph.reset();
            graph.zero_grad();
            let activated = graph.silu(weight);
            let squared = graph.mul(activated, activated);
            let scaled = graph.scale(squared, 0.5);
            let loss = graph.sum(scaled);
            graph.backward(loss);
            allocations.push(graph.allocation_count());
        }
        assert!(allocations[0] > 0, "the first step has to allocate");
        assert_eq!(allocations[1], allocations[3], "steady-state steps must reuse pooled buffers ({:?})", allocations);
        graph.reset();
        assert!(graph.pooled_buffers() > 0, "reset must park activations in the pool instead of freeing them");
    }

    #[test]
    #[should_panic(expected = "tensor shape does not match")]
    fn leaf_shape_must_match_value_count() {
        let mut graph = Graph::new(1);
        let _ = graph.input(vec![2, 2], vec![0.0; 3]);
    }
}
