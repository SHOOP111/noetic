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

// Spawning scoped operating-system threads is relatively expensive. Keep
// small kernels on the caller thread and only split work when each worker has
// enough multiply-adds to amortize that cost.
const MIN_WORK_PER_THREAD: usize = 128 * 1024;

pub fn worker_count(requested: usize, jobs: usize, total_work: usize) -> usize {
    if jobs < 2 || total_work < MIN_WORK_PER_THREAD {
        return 1;
    }
    let useful_workers = (total_work / MIN_WORK_PER_THREAD).max(1);
    requested.max(1).min(jobs).min(useful_workers)
}

#[inline]
fn checked_product(left: usize, right: usize, what: &str) -> usize {
    left.checked_mul(right).unwrap_or_else(|| panic!("{} size overflow", what))
}

// ---------------------------------------------------------------------------
// C[m,n] = A[m,k] * B[k,n]
// ---------------------------------------------------------------------------
pub fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, threads: usize) {
    assert_eq!(a.len(), checked_product(m, k, "gemm_nn A"), "gemm_nn: bad A");
    assert_eq!(b.len(), checked_product(k, n, "gemm_nn B"), "gemm_nn: bad B");
    assert_eq!(c.len(), checked_product(m, n, "gemm_nn C"), "gemm_nn: bad C");
    if m == 0 || n == 0 {
        return;
    }
    let work = m.saturating_mul(k).saturating_mul(n);
    let nt = worker_count(threads, m, work);
    if nt == 1 {
        kern_nn(a, b, c, 0, m, k, n);
        return;
    }
    let rows = 1 + (m - 1) / nt;
    let chunk_len = checked_product(rows, n, "gemm_nn chunk");
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(chunk_len).enumerate() {
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
    assert_eq!(a.len(), checked_product(m, k, "gemm_nt A"), "gemm_nt: bad A");
    assert_eq!(b.len(), checked_product(n, k, "gemm_nt B"), "gemm_nt: bad B");
    assert_eq!(c.len(), checked_product(m, n, "gemm_nt C"), "gemm_nt: bad C");
    if m == 0 || n == 0 {
        return;
    }
    let work = m.saturating_mul(k).saturating_mul(n);
    let nt = worker_count(threads, m, work);
    if nt == 1 {
        kern_nt(a, b, c, 0, m, k, n);
        return;
    }
    let rows = 1 + (m - 1) / nt;
    let chunk_len = checked_product(rows, n, "gemm_nt chunk");
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(chunk_len).enumerate() {
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
    assert_eq!(a.len(), checked_product(p, m, "gemm_tn A"), "gemm_tn: bad A");
    assert_eq!(b.len(), checked_product(p, n, "gemm_tn B"), "gemm_tn: bad B");
    assert_eq!(c.len(), checked_product(m, n, "gemm_tn C"), "gemm_tn: bad C");
    if m == 0 || n == 0 {
        return;
    }
    let work = p.saturating_mul(m).saturating_mul(n);
    let nt = worker_count(threads, m, work);
    if nt == 1 {
        kern_tn(a, b, c, 0, m, p, m, n);
        return;
    }
    let rows = 1 + (m - 1) / nt;
    let chunk_len = checked_product(rows, n, "gemm_tn chunk");
    std::thread::scope(|s| {
        for (ci, chunk) in c.chunks_mut(chunk_len).enumerate() {
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
    assert_eq!(a.len(), checked_product(m, k, "naive gemm A"), "gemm_nn_naive: bad A");
    assert_eq!(b.len(), checked_product(k, n, "naive gemm B"), "gemm_nn_naive: bad B");
    assert_eq!(c.len(), checked_product(m, n, "naive gemm C"), "gemm_nn_naive: bad C");
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
    assert_eq!(w.len(), checked_product(dout, din, "matvec W"), "matvec_nt: bad W");
    assert_eq!(x.len(), din, "matvec_nt: bad x");
    assert_eq!(out.len(), dout, "matvec_nt: bad out");
    if let Some(values) = bias {
        assert_eq!(values.len(), dout, "matvec_nt: bad bias");
    }
    if dout == 0 {
        return;
    }
    let work = dout.saturating_mul(din);
    let nt = worker_count(threads, dout, work);
    if nt == 1 {
        kern_matvec(w, bias, x, out, 0, dout, din);
        return;
    }
    let rows = 1 + (dout - 1) / nt;
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
    assert!(d > 0, "rms_norm_vec: empty input");
    assert_eq!(w.len(), d, "rms_norm_vec: bad gain");
    assert_eq!(out.len(), d, "rms_norm_vec: bad output");
    assert!(eps.is_finite() && eps > 0.0, "rms_norm_vec: epsilon must be finite and positive");
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
    assert!(!x.is_empty(), "softmax_inplace: empty input");

    // Handle positive infinities explicitly. `inf - inf` is NaN, but the
    // limiting softmax is uniform over the entries tied at +infinity.
    let positive_infinities = x.iter().filter(|value| **value == f32::INFINITY).count();
    if positive_infinities > 0 {
        let probability = 1.0 / (positive_infinities as f32);
        for value in x.iter_mut() {
            *value = if *value == f32::INFINITY { probability } else { 0.0 };
        }
        return;
    }

    let mut mx = f32::NEG_INFINITY;
    for i in 0..x.len() {
        if x[i].is_finite() && x[i] > mx {
            mx = x[i];
        }
    }
    if mx == f32::NEG_INFINITY {
        x.fill(1.0 / (x.len() as f32));
        return;
    }
    let mut s = 0.0f32;
    for i in 0..x.len() {
        let e = if x[i].is_finite() { (x[i] - mx).exp() } else { 0.0 };
        x[i] = e;
        s += e;
    }
    if !s.is_finite() || s <= 0.0 {
        x.fill(1.0 / (x.len() as f32));
        return;
    }
    let inv = 1.0 / s;
    for i in 0..x.len() {
        x[i] *= inv;
    }
}

pub fn argmax(x: &[f32]) -> usize {
    assert!(!x.is_empty(), "argmax: empty input");
    let mut best = 0usize;
    let mut bv = if x[0].is_nan() { f32::NEG_INFINITY } else { x[0] };
    for i in 1..x.len() {
        if !x[i].is_nan() && x[i] > bv {
            bv = x[i];
            best = i;
        }
    }
    best
}

pub fn add_into(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "add_into: length mismatch");
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_handles_non_finite_inputs_deterministically() {
        let mut tied = [f32::INFINITY, 1.0, f32::INFINITY, f32::NAN];
        softmax_inplace(&mut tied);
        assert_eq!(tied, [0.5, 0.0, 0.5, 0.0]);

        let mut invalid = [f32::NAN, f32::NEG_INFINITY, f32::NAN];
        softmax_inplace(&mut invalid);
        for probability in invalid {
            assert!((probability - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn small_kernels_stay_on_the_caller_thread() {
        assert_eq!(worker_count(64, 64, MIN_WORK_PER_THREAD - 1), 1);
        assert_eq!(worker_count(8, 8, MIN_WORK_PER_THREAD * 8), 8);
    }
}
