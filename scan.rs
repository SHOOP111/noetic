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
//! * `scan_sequential` - O(T) work, depth O(T), threaded across batch. Optimal
//!   for training, where the batch axis is already wide.
//! * `scan_chunked` - 1.33x the traffic, depth O(T/threads + threads), threaded
//!   across *time*. For few long sequences; measure, both are memory-bound.
//!
//! Layout everywhere: index(b, t, j) = (b * T + t) * D + j, so the innermost
//! axis is contiguous and the per-timestep update is a straight vector op.

use crate::tensor::worker_count;

/// Decay products below this magnitude are flushed to zero: they cannot change
/// an f32 state value, and denormal multiplies cost orders of magnitude more
/// than normal ones on x86.
const DECAY_FLOOR: f32 = 1e-30;

#[inline]
fn scan_len(batch: usize, t: usize, d: usize) -> usize {
    batch.checked_mul(t).and_then(|value| value.checked_mul(d)).expect("scan dimensions overflow")
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
    let chunk_len = per.checked_mul(t).and_then(|value| value.checked_mul(d)).expect("scan chunk dimensions overflow");
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

/// Inclusive scan parallelised over the **time** axis, in three passes.
///
/// Hillis-Steele (the previous implementation) needs O(T log T) work and a
/// fresh thread scope per doubling step, which measured ~28x *slower* than the
/// sequential kernel. This is the standard work-efficient alternative:
///
/// 1. split time into one contiguous chunk per worker; each worker runs the
///    plain recurrence over its own chunk starting from zero and records the
///    chunk's composed decay `A = prod a_t`
/// 2. compose the chunk maps sequentially - `nt` tiny steps - to recover the
///    true state entering each chunk
/// 3. each worker folds its inherited carry back in:
///    `h_t += (prod_{r<=t} a_r) * carry_in`
///
/// Depth is O(T / nt + nt) at 1.33x the sequential traffic. Both kernels are
/// memory-bound, so measure before switching: on a 4-core laptop whose DRAM is
/// already saturated by one core this wins ~1.15x on cache-resident shapes
/// (B1 T2048 D64) and *loses* ~1.6x on DRAM-resident ones (B1 T8192 D256).
/// `bench` prints both comparisons.
pub fn scan_chunked(a: &[f32], b: &[f32], h: &mut [f32], batch: usize, t: usize, d: usize, threads: usize) {
    let n = scan_len(batch, t, d);
    assert_eq!(a.len(), n, "scan: bad a");
    assert_eq!(b.len(), n, "scan: bad b");
    assert_eq!(h.len(), n, "scan: bad h");
    if n == 0 {
        return;
    }
    let span = t.checked_mul(d).expect("scan span overflow");
    let nt = worker_count(threads, t, n);
    if nt == 1 || t < 2 {
        scan_sequential(a, b, h, batch, t, d, threads);
        return;
    }
    let per = 1 + (t - 1) / nt;
    let chunks = 1 + (t - 1) / per;
    let chunk_state_len = batch.checked_mul(chunks).and_then(|value| value.checked_mul(d)).expect("scan chunk state overflow");
    let mut chunk_decay = vec![0.0f32; chunk_state_len];
    let mut carry_in = vec![0.0f32; chunk_state_len];

    // ---- pass 1: independent local scans -------------------------------
    {
        let mut jobs: Vec<(usize, usize, &mut [f32], &mut [f32])> = Vec::with_capacity(batch * chunks);
        for ((bi, hb), db) in h.chunks_mut(span).enumerate().zip(chunk_decay.chunks_mut(chunks * d)) {
            for ((ci, hc), dc) in hb.chunks_mut(per * d).enumerate().zip(db.chunks_mut(d)) {
                jobs.push((bi, ci, hc, dc));
            }
        }
        let per_worker = 1 + (jobs.len() - 1) / nt;
        std::thread::scope(|s| {
            for group in jobs.chunks_mut(per_worker) {
                s.spawn(move || {
                    let mut carry = vec![0.0f32; d];
                    for (bi, ci, hc, dc) in group.iter_mut() {
                        let base = *bi * span + *ci * per * d;
                        let rows = hc.len() / d;
                        for value in carry.iter_mut() {
                            *value = 0.0;
                        }
                        for value in dc.iter_mut() {
                            *value = 1.0;
                        }
                        // Local scan + composed chunk decay in one traversal.
                        // A separate accumulator (rather than reading h[t-1])
                        // keeps the inner loop alias-free so it vectorizes.
                        // Products of |a| < 1 reach the denormal range fast, and
                        // denormal arithmetic is punishingly slow, so flush them
                        // to zero: a decay that small cannot move an f32 state.
                        let last_chunk = *ci + 1 == chunks;
                        for i in 0..rows {
                            let g = base + i * d;
                            for j in 0..d {
                                let value = a[g + j] * carry[j] + b[g + j];
                                carry[j] = value;
                                hc[i * d + j] = value;
                            }
                            if last_chunk {
                                // Nothing composes against the final chunk.
                                continue;
                            }
                            for j in 0..d {
                                let product = dc[j] * a[g + j];
                                dc[j] = if product.abs() < DECAY_FLOOR { 0.0 } else { product };
                            }
                        }
                    }
                });
            }
        });
    }

    // ---- pass 2: compose the chunk maps (sequential, `chunks` steps) ----
    for bi in 0..batch {
        for ci in 1..chunks {
            let previous = (bi * chunks + ci - 1) * d;
            let current = (bi * chunks + ci) * d;
            let last_row = bi * span + (ci * per - 1) * d;
            for j in 0..d {
                carry_in[current + j] = h[last_row + j] + chunk_decay[previous + j] * carry_in[previous + j];
            }
        }
    }

    // ---- pass 3: fold the inherited carry into every chunk --------------
    {
        let mut jobs: Vec<(usize, usize, &mut [f32], &[f32])> = Vec::with_capacity(batch * chunks);
        for (bi, hb) in h.chunks_mut(span).enumerate() {
            let carry_base = bi * chunks * d;
            let carry_rows = &carry_in[carry_base..carry_base + chunks * d];
            for ((ci, hc), cc) in hb.chunks_mut(per * d).enumerate().zip(carry_rows.chunks(d)) {
                if ci > 0 {
                    jobs.push((bi, ci, hc, cc));
                }
            }
        }
        if !jobs.is_empty() {
            let per_worker = 1 + (jobs.len() - 1) / nt;
            std::thread::scope(|s| {
                for group in jobs.chunks_mut(per_worker) {
                    s.spawn(move || {
                        let mut product = vec![0.0f32; d];
                        for (bi, ci, hc, cc) in group.iter_mut() {
                            let base = *bi * span + *ci * per * d;
                            let rows = hc.len() / d;
                            product.copy_from_slice(cc);
                            for i in 0..rows {
                                let g = base + i * d;
                                let mut alive = false;
                                for j in 0..d {
                                    let scaled = product[j] * a[g + j];
                                    let scaled = if scaled.abs() < DECAY_FLOOR { 0.0 } else { scaled };
                                    product[j] = scaled;
                                    hc[i * d + j] += scaled;
                                    alive |= scaled != 0.0;
                                }
                                // The inherited carry has fully decayed: every
                                // remaining row of this chunk is already correct.
                                if !alive {
                                    break;
                                }
                            }
                        }
                    });
                }
            });
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// The chunked scan must agree with the plain recurrence for shapes that do
    /// not divide evenly by the worker count, for d = 1, and for T = 1.
    #[test]
    fn chunked_scan_matches_sequential_for_awkward_shapes() {
        let mut rng = Rng::new(99);
        for (batch, t, d, threads) in
            [(1usize, 1usize, 1usize, 1usize), (3, 7, 5, 3), (2, 129, 3, 4), (5, 64, 1, 8), (1, 1024, 9, 4), (2, 1000, 4, 3)]
        {
            let n = batch * t * d;
            let a: Vec<f32> = (0..n).map(|_| rng.f32_unit() * 0.98 + 0.01).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
            let mut expected = vec![0.0f32; n];
            let mut actual = vec![0.0f32; n];
            scan_sequential(&a, &b, &mut expected, batch, t, d, threads);
            scan_chunked(&a, &b, &mut actual, batch, t, d, threads);
            let mut worst = 0.0f32;
            for i in 0..n {
                worst = worst.max((expected[i] - actual[i]).abs());
            }
            assert!(worst < 1e-4, "b{} t{} d{} threads{} -> {}", batch, t, d, threads, worst);
        }
    }

    /// c_t = sum_{s >= t} g_s * prod_{r = t+1..s} a_r, checked directly.
    #[test]
    fn adjoint_matches_the_closed_form() {
        let (batch, t, d) = (2usize, 6usize, 2usize);
        let n = batch * t * d;
        let mut rng = Rng::new(5);
        let a: Vec<f32> = (0..n).map(|_| rng.f32_unit()).collect();
        let g: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let mut c = vec![0.0f32; n];
        scan_adjoint(&a, &g, &mut c, batch, t, d);
        for bi in 0..batch {
            for j in 0..d {
                for ti in 0..t {
                    let mut expected = 0.0f32;
                    let mut product = 1.0f32;
                    for s in ti..t {
                        if s > ti {
                            product *= a[(bi * t + s) * d + j];
                        }
                        expected += g[(bi * t + s) * d + j] * product;
                    }
                    let actual = c[(bi * t + ti) * d + j];
                    assert!((expected - actual).abs() < 1e-4, "b{} t{} j{}: {} vs {}", bi, ti, j, expected, actual);
                }
            }
        }
    }
}
