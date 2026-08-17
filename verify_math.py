#!/usr/bin/env python3
"""Numeric validation of the math implemented in noetic's autograd/scan.

No Rust toolchain exists in this sandbox, so the *formulas* transcribed from
src/scan.rs and src/autograd.rs are re-implemented here line-for-line and
checked against central finite differences and independent references.

This validates the derivations; it does not compile the Rust.
"""
import math, random

random.seed(20250816)
fails = []


def report(name, err, tol):
    ok = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} err {err:.3e}  (tol {tol:.0e})")
    if not ok:
        fails.append(name)


def fd(f, x, i, eps=1e-5):
    x1 = list(x); x1[i] += eps
    x2 = list(x); x2[i] -= eps
    return (f(x1) - f(x2)) / (2 * eps)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


# ---------------------------------------------------------------------------
# 1. sequential recurrence vs double-buffered Hillis-Steele scan (scan.rs)
# ---------------------------------------------------------------------------
def scan_sequential(a, b, T, D):
    h = [0.0] * (T * D)
    carry = [0.0] * D
    for t in range(T):
        for j in range(D):
            carry[j] = a[t * D + j] * carry[j] + b[t * D + j]
            h[t * D + j] = carry[j]
    return h


def scan_log_depth(a, b, T, D):
    sa, sb = list(a), list(b)
    da, db = [0.0] * (T * D), [0.0] * (T * D)
    stride = 1
    while stride < T:
        for t in range(T):
            cur = t * D
            if t >= stride:
                prev = (t - stride) * D
                for j in range(D):
                    db[cur + j] = sa[cur + j] * sb[prev + j] + sb[cur + j]
                    da[cur + j] = sa[cur + j] * sa[prev + j]
            else:
                for j in range(D):
                    db[cur + j] = sb[cur + j]
                    da[cur + j] = sa[cur + j]
        sa, da = da, sa
        sb, db = db, sb
        stride <<= 1
    return sb


T, D = 67, 5
a = [random.uniform(0.01, 0.99) for _ in range(T * D)]
b = [random.gauss(0, 1) for _ in range(T * D)]
h1 = scan_sequential(a, b, T, D)
h2 = scan_log_depth(a, b, T, D)
report("log-depth scan == sequential recurrence", max(abs(x - y) for x, y in zip(h1, h2)), 1e-9)


# ---------------------------------------------------------------------------
# 2. scan adjoint (scan_adjoint + Op::Scan arm) vs finite differences
#    c_t = g_t + a_{t+1} c_{t+1};  dL/db_t = c_t;  dL/da_t = c_t * h_{t-1}
# ---------------------------------------------------------------------------
def scan_adjoint(a, g, T, D):
    c = [0.0] * (T * D)
    carry = [0.0] * D
    for t in range(T - 1, -1, -1):
        for j in range(D):
            cv = g[t * D + j] + carry[j]
            c[t * D + j] = cv
            carry[j] = a[t * D + j] * cv
    return c


T, D = 9, 3
av = [random.uniform(0.05, 0.95) for _ in range(T * D)]
bv = [random.gauss(0, 1) for _ in range(T * D)]
w = [random.gauss(0, 1) for _ in range(T * D)]


def loss_ab(flat):
    aa, bb = flat[: T * D], flat[T * D :]
    h = scan_sequential(aa, bb, T, D)
    return sum(hi * wi for hi, wi in zip(h, w))


flat = av + bv
h = scan_sequential(av, bv, T, D)
c = scan_adjoint(av, w, T, D)
grad_b = c[:]
grad_a = [0.0] * (T * D)
for t in range(1, T):
    for j in range(D):
        grad_a[t * D + j] = c[t * D + j] * h[(t - 1) * D + j]
ana = grad_a + grad_b
err = max(abs(ana[i] - fd(loss_ab, flat, i)) for i in range(len(flat)))
report("scan gradient d(a,b) vs finite diff", err, 1e-5)


# ---------------------------------------------------------------------------
# 3. RMSNorm backward (Op::RmsNorm arm)
#    r = 1/sqrt(mean(x^2)+eps);  y = r*x
#    dx = r*go - r^3 * dot(go,x)/d * x
# ---------------------------------------------------------------------------
d = 7
eps = 1e-5
x = [random.gauss(0, 1.7) for _ in range(d)]
go = [random.gauss(0, 1) for _ in range(d)]


def rms_loss(xx):
    ms = sum(v * v for v in xx) / d + eps
    r = 1.0 / math.sqrt(ms)
    return sum(r * xx[j] * go[j] for j in range(d))


ms = sum(v * v for v in x) / d + eps
r = 1.0 / math.sqrt(ms)
dot = sum(go[j] * x[j] for j in range(d))
k = r * r * r * dot / d
ana = [r * go[j] - k * x[j] for j in range(d)]
err = max(abs(ana[j] - fd(rms_loss, x, j)) for j in range(d))
report("RMSNorm gradient vs finite diff", err, 1e-5)


# ---------------------------------------------------------------------------
# 4. softmax cross-entropy (Op::SoftmaxCe arm): grad = (p - onehot)/rows
# ---------------------------------------------------------------------------
rows, V = 4, 6
logits = [random.gauss(0, 1) for _ in range(rows * V)]
targets = [random.randrange(V) for _ in range(rows)]


def ce_loss(lg):
    tot = 0.0
    for i in range(rows):
        row = lg[i * V : (i + 1) * V]
        mx = max(row)
        s = sum(math.exp(v - mx) for v in row)
        tot += -(row[targets[i]] - mx - math.log(s))
    return tot / rows


ana = [0.0] * (rows * V)
for i in range(rows):
    row = logits[i * V : (i + 1) * V]
    mx = max(row)
    ex = [math.exp(v - mx) for v in row]
    s = sum(ex)
    for j in range(V):
        p = ex[j] / s
        if j == targets[i]:
            p -= 1.0
        ana[i * V + j] = p / rows
err = max(abs(ana[i] - fd(ce_loss, logits, i)) for i in range(rows * V))
report("softmax-CE gradient vs finite diff", err, 1e-5)


# ---------------------------------------------------------------------------
# 5. distribution-target CE (Op::SoftCeDist arm): grad = (tsum*p - t)/rows
#    used for MCTS visit-count policy targets
# ---------------------------------------------------------------------------
rows, K = 5, 4
logits = [random.gauss(0, 1) for _ in range(rows * K)]
tgt = []
for i in range(rows):
    raw = [random.random() for _ in range(K)]
    s = sum(raw)
    tgt.extend(v / s for v in raw)


def soft_ce(lg):
    tot = 0.0
    for i in range(rows):
        row = lg[i * K : (i + 1) * K]
        mx = max(row)
        s = sum(math.exp(v - mx) for v in row)
        for j in range(K):
            lp = row[j] - mx - math.log(s)
            tot += -tgt[i * K + j] * lp
    return tot / rows


ana = [0.0] * (rows * K)
for i in range(rows):
    row = logits[i * K : (i + 1) * K]
    mx = max(row)
    ex = [math.exp(v - mx) for v in row]
    s = sum(ex)
    tsum = sum(tgt[i * K : (i + 1) * K])
    for j in range(K):
        p = ex[j] / s
        ana[i * K + j] = (tsum * p - tgt[i * K + j]) / rows
err = max(abs(ana[i] - fd(soft_ce, logits, i)) for i in range(rows * K))
report("distribution-target CE gradient", err, 1e-5)


# ---------------------------------------------------------------------------
# 6. causal depthwise conv backward (Op::DwConv arm)
# ---------------------------------------------------------------------------
T, D, K = 6, 3, 4
xv = [random.gauss(0, 1) for _ in range(T * D)]
wv = [random.gauss(0, 1) for _ in range(K * D)]
bias = [random.gauss(0, 1) for _ in range(D)]
go = [random.gauss(0, 1) for _ in range(T * D)]


def conv_fwd(xx, ww, bb):
    out = [0.0] * (T * D)
    for t in range(T):
        for j in range(D):
            s = bb[j]
            for q in range(min(t + 1, K)):
                s += ww[q * D + j] * xx[(t - q) * D + j]
            out[t * D + j] = s
    return out


def conv_loss(flat):
    xx = flat[: T * D]
    ww = flat[T * D : T * D + K * D]
    bb = flat[T * D + K * D :]
    out = conv_fwd(xx, ww, bb)
    return sum(o * g for o, g in zip(out, go))


gx = [0.0] * (T * D)
gw = [0.0] * (K * D)
gb = [0.0] * D
for t in range(T):
    for j in range(D):
        for q in range(min(t + 1, K)):
            gx[(t - q) * D + j] += go[t * D + j] * wv[q * D + j]
            gw[q * D + j] += go[t * D + j] * xv[(t - q) * D + j]
        gb[j] += go[t * D + j]
ana = gx + gw + gb
flat = xv + wv + bias
err = max(abs(ana[i] - fd(conv_loss, flat, i)) for i in range(len(flat)))
report("depthwise causal conv gradient", err, 1e-5)


# ---------------------------------------------------------------------------
# 7. SiLU derivative: s*(1 + v*(1-s))
# ---------------------------------------------------------------------------
err = 0.0
for _ in range(200):
    v = random.gauss(0, 3)
    s = sigmoid(v)
    ana = s * (1.0 + v * (1.0 - s))
    num = ((v + 1e-5) * sigmoid(v + 1e-5) - (v - 1e-5) * sigmoid(v - 1e-5)) / 2e-5
    err = max(err, abs(ana - num))
report("SiLU derivative", err, 1e-5)


# ---------------------------------------------------------------------------
# 8. decay-spectrum init (model.rs): z = logit(exp(-1/tau)) must reproduce tau
# ---------------------------------------------------------------------------
E, tau_max = 256, 128.0
err = 0.0
taus = []
for j in range(E):
    frac = j / (E - 1)
    tau = tau_max ** frac
    aa = math.exp(-1.0 / tau)
    aa = min(max(aa, 0.001), 0.9999)
    z = math.log(aa / (1.0 - aa))
    back = sigmoid(z)
    err = max(err, abs(back - aa))
    taus.append(-1.0 / math.log(back))
report("decay spectrum init round-trip", err, 1e-9)
print(f"         -> effective time constants span tau = {min(taus):.2f} .. {max(taus):.1f} steps")


# ---------------------------------------------------------------------------
# 9. state stability: with a in (0,1) and b = (1-a)*v the state is a convex
#    blend, so |h| <= max|v| for any sequence length (no blow-up).
# ---------------------------------------------------------------------------
T, D = 4000, 4
worst = 0.0
for _ in range(3):
    aa = [random.uniform(0.0, 1.0) for _ in range(T * D)]
    vv = [random.gauss(0, 1) for _ in range(T * D)]
    bb = [(1 - aa[i]) * vv[i] for i in range(T * D)]
    h = scan_sequential(aa, bb, T, D)
    worst = max(worst, max(abs(x) for x in h) / max(abs(x) for x in vv))
report("convex-blend state stays bounded (T=4000)", max(0.0, worst - 1.0), 1e-9)

print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all derivations verified")
