//! The heart of the architecture: the **first-order linear recurrence**
//!
//!     h_t = a_t * h_{t-1} + b_t          (elementwise, diagonal state)
//!
//! This is what replaces self-attention. It is *associative*: composing two
//! affine maps gives another affine map,
//!
//!     (a_L, b_L) then (a_R, b_R)  ==  (a_R*a_L, a_R*b_L + b_R)
//!
//! so the whole sequence can be reduced with a prefix scan instead of a loop.
//! Two evaluation strategies are provided and `selftest` asserts they agree
//! bit-for-bit-ish (they differ only by float association order):
//!
//! * `scan_sequential`  - O(T) work, depth O(T), threaded across batch.
//!                        Optimal for training (batch is wide).
//! * `scan_log_depth`   - O(T log T) work, depth O(log T), threaded across
//!                        *time* via double buffering (Hillis-Steele).
//!                        Optimal for one very long sequence on many cores.
//!
//! Layout everywhere: index(b, t, j) = (b * T + t) * D + j, so the innermost
//! axis is contiguous and the per-timestep update is a straight vector op.

use crate::tensor::worker_count;

#[inline]
fn scan_len(batch: usize, t: usize, d: usize) -> usize {
    batch
        .checked_mul(t)
        .and_then(|value| value.checked_mul(d))
        .expect("scan dimensions overflow")
}

/// Inclusive scan, sequential in time, parallel over the batch axis.
pub fn scan_sequential(a: &[f32], b: &[f32], h: &mut [f32], batch: usize, t: usize, d: usize, threads: usize) {
    let n = scan_len(batch, t, d);
    assert_eq!(a.len(), n, "scan: bad a");
    assert_eq!(b.len(), n, "scan: bad b");
    assert_eq!(h.len(), n, "scan: bad h");
    if n == 0 {
        return;
    }
    let nt = worker_count(threads, batch, n);
    if nt == 1 {
        kern_scan_seq(a, b, h, 0, batch, t, d);
        return;
    }
    let per = 1 + (batch - 1) / nt;
    let chunk_len = per
        .checked_mul(t)
        .and_then(|value| value.checked_mul(d))
        .expect("scan chunk dimensions overflow");
    std::thread::scope(|s| {
        for (ci, chunk) in h.chunks_mut(chunk_len).enumerate() {
            let b0 = ci * per;
            let cnt = chunk.len() / (t * d);
            s.spawn(move || kern_scan_seq(a, b, chunk, b0, cnt, t, d));
        }
    });
}

fn kern_scan_seq(a: &[f32], b: &[f32], h: &mut [f32], b0: usize, batch: usize, t: usize, d: usize) {
    let mut carry = vec![0.0f32; d];
    for bi in 0..batch {
        let gbase = (b0 + bi) * t * d;
        let lbase = bi * t * d;
        for x in carry.iter_mut() {
            *x = 0.0;
        }
        for ti in 0..t {
            let g = gbase + ti * d;
            let l = lbase + ti * d;
            for j in 0..d {
                let v = a[g + j] * carry[j] + b[g + j];
                carry[j] = v;
                h[l + j] = v;
            }
        }
    }
}

/// Inclusive scan with O(log T) depth (Hillis-Steele, double buffered).
/// Threaded across the time axis; batch elements are processed in order.
pub fn scan_log_depth(a: &[f32], b: &[f32], h: &mut [f32], batch: usize, t: usize, d: usize, threads: usize) {
    let n = scan_len(batch, t, d);
    assert_eq!(a.len(), n, "scan: bad a");
    assert_eq!(b.len(), n, "scan: bad b");
    assert_eq!(h.len(), n, "scan: bad h");
    if n == 0 {
        return;
    }
    let span = t.checked_mul(d).expect("scan span overflow");
    let mut sa = vec![0.0f32; span];
    let mut sb = vec![0.0f32; span];
    let mut da = vec![0.0f32; span];
    let mut db = vec![0.0f32; span];
    for bi in 0..batch {
        let base = bi * span;
        sa.copy_from_slice(&a[base..base + span]);
        sb.copy_from_slice(&b[base..base + span]);
        let mut stride = 1usize;
        while stride < t {
            {
                let ra: &[f32] = &sa;
                let rb: &[f32] = &sb;
                let nt = worker_count(threads, t, span);
                let per = 1 + (t - 1) / nt;
                let chunk_len = per.checked_mul(d).expect("scan time chunk overflow");
                std::thread::scope(|s| {
                    let ia = da.chunks_mut(chunk_len);
                    let ib = db.chunks_mut(chunk_len);
                    for (ci, (xa, xb)) in ia.zip(ib).enumerate() {
                        let t0 = ci * per;
                        let cnt = xa.len() / d;
                        s.spawn(move || {
                            for i in 0..cnt {
                                let ti = t0 + i;
                                let cur = ti * d;
                                if ti >= stride {
                                    let prev = (ti - stride) * d;
                                    for j in 0..d {
                                        xb[i * d + j] = ra[cur + j] * rb[prev + j] + rb[cur + j];
                                        xa[i * d + j] = ra[cur + j] * ra[prev + j];
                                    }
                                } else {
                                    for j in 0..d {
                                        xb[i * d + j] = rb[cur + j];
                                        xa[i * d + j] = ra[cur + j];
                                    }
                                }
                            }
                        });
                    }
                });
            }
            std::mem::swap(&mut sa, &mut da);
            std::mem::swap(&mut sb, &mut db);
            stride = stride.saturating_mul(2);
        }
        h[base..base + span].copy_from_slice(&sb);
    }
}

/// Reverse-mode gradient of the recurrence.
///
/// Forward:  h_t = a_t h_{t-1} + b_t
/// Adjoint:  c_t = g_t + a_{t+1} c_{t+1}      (c_{T} = 0)
///           dL/db_t = c_t
///           dL/da_t = c_t * h_{t-1}          (h_{-1} = 0)
///
/// `c` is filled with the adjoint stream; callers scatter it into a/b grads.
pub fn scan_adjoint(a: &[f32], g: &[f32], c: &mut [f32], batch: usize, t: usize, d: usize) {
    let n = scan_len(batch, t, d);
    assert_eq!(a.len(), n, "scan adjoint: bad a");
    assert_eq!(g.len(), n, "scan adjoint: bad output gradient");
    assert_eq!(c.len(), n, "scan adjoint: bad destination");
    if n == 0 {
        return;
    }
    let mut carry = vec![0.0f32; d];
    for bi in 0..batch {
        let base = bi * t * d;
        for x in carry.iter_mut() {
            *x = 0.0;
        }
        let mut ti = t;
        while ti > 0 {
            ti -= 1;
            let idx = base + ti * d;
            for j in 0..d {
                let cv = g[idx + j] + carry[j];
                c[idx + j] = cv;
                carry[j] = a[idx + j] * cv;
            }
        }
    }
}
