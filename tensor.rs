//! Dense f32 kernels: multi-threaded GEMM (3 transpose modes), matvec,
//! and the small vector primitives used by the streaming decoder.
//!
//! Everything is `&[f32]` / `&mut [f32]` over flat row-major buffers, so there
//! is no allocator pressure inside hot loops and no unsafe code anywhere.
//! Parallelism is `std::thread::scope` over disjoint output row-chunks:
//! provably data-race free, no locks, no atomics.

#[inline]
pub fn n_threads_default() -> usize {
    match std::thread::available_parallelism() {
        Ok(v) => v.get(),
        Err(_) => 4,
    }
}

// ---------------------------------------------------------------------------
// C[m,n] = A[m,k] * B[k,n]
// ---------------------------------------------------------------------------
pub fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, threads: usize) {
    assert_eq!(a.len(), m * k, "gemm_nn: bad A");
    assert_eq!(b.len(), k * n, "gemm_nn: bad B");
    assert_eq!(c.len(), m * n, "gemm_nn: bad C");
    if m == 0 || n == 0 {
        return;
    }
    let nt = threads.max(1).min(m);
    if nt == 1 {
        kern_nn(a, b, c, 0, m, k, n);
        return;
    }
    let rows = (m + nt - 1) / nt;
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(rows * n).enumerate() {
            let r0 = ci * rows;
            let cnt = chunk.len() / n;
            s.spawn(move || kern_nn(a, b, chunk, r0, cnt, k, n));
        }
    });
}

fn kern_nn(a: &[f32], b: &[f32], c: &mut [f32], r0: usize, rows: usize, k: usize, n: usize) {
    for i in 0..rows {
        let ar = &a[(r0 + i) * k..(r0 + i) * k + k];
        let cr = &mut c[i * n..i * n + n];
        for x in cr.iter_mut() {
            *x = 0.0;
        }
        let mut p = 0usize;
        // 4-way unroll over the reduction axis: 4 independent FMA chains,
        // one streaming pass over C per group instead of four.
        while p + 4 <= k {
            let a0 = ar[p];
            let a1 = ar[p + 1];
            let a2 = ar[p + 2];
            let a3 = ar[p + 3];
            let b0 = &b[p * n..p * n + n];
            let b1 = &b[(p + 1) * n..(p + 1) * n + n];
            let b2 = &b[(p + 2) * n..(p + 2) * n + n];
            let b3 = &b[(p + 3) * n..(p + 3) * n + n];
            for j in 0..n {
                cr[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
            }
            p += 4;
        }
        while p < k {
            let a0 = ar[p];
            let b0 = &b[p * n..p * n + n];
            for j in 0..n {
                cr[j] += a0 * b0[j];
            }
            p += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// C[m,n] = A[m,k] * B[n,k]^T   (weights stored [out,in], the layout linear
// layers actually want: both operands are read contiguously)
// ---------------------------------------------------------------------------
pub fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, threads: usize) {
    assert_eq!(a.len(), m * k, "gemm_nt: bad A");
    assert_eq!(b.len(), n * k, "gemm_nt: bad B");
    assert_eq!(c.len(), m * n, "gemm_nt: bad C");
    if m == 0 || n == 0 {
        return;
    }
    let nt = threads.max(1).min(m);
    if nt == 1 {
        kern_nt(a, b, c, 0, m, k, n);
        return;
    }
    let rows = (m + nt - 1) / nt;
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(rows * n).enumerate() {
            let r0 = ci * rows;
            let cnt = chunk.len() / n;
            s.spawn(move || kern_nt(a, b, chunk, r0, cnt, k, n));
        }
    });
}

fn kern_nt(a: &[f32], b: &[f32], c: &mut [f32], r0: usize, rows: usize, k: usize, n: usize) {
    for i in 0..rows {
        let ar = &a[(r0 + i) * k..(r0 + i) * k + k];
        let cr = &mut c[i * n..i * n + n];
        for j in 0..n {
            let br = &b[j * k..j * k + k];
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            let mut p = 0usize;
            while p + 4 <= k {
                s0 += ar[p] * br[p];
                s1 += ar[p + 1] * br[p + 1];
                s2 += ar[p + 2] * br[p + 2];
                s3 += ar[p + 3] * br[p + 3];
                p += 4;
            }
            let mut s = (s0 + s1) + (s2 + s3);
            while p < k {
                s += ar[p] * br[p];
                p += 1;
            }
            cr[j] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// C[m,n] = A[p,m]^T * B[p,n]   (the weight-gradient shape)
// ---------------------------------------------------------------------------
pub fn gemm_tn(a: &[f32], b: &[f32], c: &mut [f32], p: usize, m: usize, n: usize, threads: usize) {
    assert_eq!(a.len(), p * m, "gemm_tn: bad A");
    assert_eq!(b.len(), p * n, "gemm_tn: bad B");
    assert_eq!(c.len(), m * n, "gemm_tn: bad C");
    if m == 0 || n == 0 {
        return;
    }
    let nt = threads.max(1).min(m);
    if nt == 1 {
        kern_tn(a, b, c, 0, m, p, m, n);
        return;
    }
    let rows = (m + nt - 1) / nt;
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(rows * n).enumerate() {
            let r0 = ci * rows;
            let cnt = chunk.len() / n;
            s.spawn(move || kern_tn(a, b, chunk, r0, cnt, p, m, n));
        }
    });
}

fn kern_tn(a: &[f32], b: &[f32], c: &mut [f32], r0: usize, rows: usize, p: usize, m: usize, n: usize) {
    for i in 0..rows {
        let ii = r0 + i;
        let cr = &mut c[i * n..i * n + n];
        for x in cr.iter_mut() {
            *x = 0.0;
        }
        for q in 0..p {
            let av = a[q * m + ii];
            if av == 0.0 {
                continue;
            }
            let br = &b[q * n..q * n + n];
            for j in 0..n {
                cr[j] += av * br[j];
            }
        }
    }
}

/// Reference implementation used by `selftest` to police the fast kernels.
pub fn gemm_nn_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for q in 0..k {
                s += (a[i * k + q] as f64) * (b[q * n + j] as f64);
            }
            c[i * n + j] = s as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// Vector primitives (streaming decoder path: batch = 1, no autodiff tape)
// ---------------------------------------------------------------------------

/// out[o] = bias[o] + sum_i w[o*din+i] * x[i]. Threaded over output rows.
pub fn matvec_nt(w: &[f32], bias: Option<&[f32]>, x: &[f32], out: &mut [f32], dout: usize, din: usize, threads: usize) {
    assert_eq!(w.len(), dout * din, "matvec_nt: bad W");
    assert_eq!(x.len(), din, "matvec_nt: bad x");
    assert_eq!(out.len(), dout, "matvec_nt: bad out");
    if dout == 0 {
        return;
    }
    let nt = threads.max(1).min(dout);
    if nt == 1 {
        kern_matvec(w, bias, x, out, 0, dout, din);
        return;
    }
    let rows = (dout + nt - 1) / nt;
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(rows).enumerate() {
            let r0 = ci * rows;
            let cnt = chunk.len();
            s.spawn(move || kern_matvec(w, bias, x, chunk, r0, cnt, din));
        }
    });
}

fn kern_matvec(w: &[f32], bias: Option<&[f32]>, x: &[f32], out: &mut [f32], r0: usize, rows: usize, din: usize) {
    for i in 0..rows {
        let o = r0 + i;
        let wr = &w[o * din..o * din + din];
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut p = 0usize;
        while p + 4 <= din {
            s0 += wr[p] * x[p];
            s1 += wr[p + 1] * x[p + 1];
            s2 += wr[p + 2] * x[p + 2];
            s3 += wr[p + 3] * x[p + 3];
            p += 4;
        }
        let mut s = (s0 + s1) + (s2 + s3);
        while p < din {
            s += wr[p] * x[p];
            p += 1;
        }
        if let Some(b) = bias {
            s += b[o];
        }
        out[i] = s;
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    // tanh approximation, matches the autodiff op exactly
    let c = 0.797_884_56f32;
    let u = c * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

pub fn rms_norm_vec(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let d = x.len();
    let mut ms = 0.0f32;
    for i in 0..d {
        ms += x[i] * x[i];
    }
    let r = 1.0 / (ms / (d as f32) + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * r * w[i];
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for i in 0..x.len() {
        if x[i] > mx {
            mx = x[i];
        }
    }
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let e = (x[i] - mx).exp();
        x[i] = e;
        s += e;
    }
    let inv = 1.0 / s.max(1e-30);
    for i in 0..x.len() {
        x[i] *= inv;
    }
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for i in 0..x.len() {
        if x[i] > bv {
            bv = x[i];
            best = i;
        }
    }
    best
}

pub fn add_into(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}
