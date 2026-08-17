"""Shared machinery for the MeijerG(4,2,4,4,1) zeta(2)+zeta(3) trajectory search.

Multi-fidelity pipeline:
  Tier 1: kamidelta prescreen (cheap, no walk)
  Tier 2: sparse exact walk + local PSLQ identification of the z2+z3 relation
  Tier 3: exact finite-depth delta at increasing depths + extrapolation
  Tier 4: high-depth (>=3000) verification for finalists

No LIReC needed: mpmath.pslq identifies the (target, 1, c1, c2) relation locally.
"""
from __future__ import annotations

import math
import sys
import time

sys.set_int_max_str_digits(2_000_000_000)

import mpmath as mp
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf.meijer_g import MeijerG

a0, a1, a2, a3 = sp.symbols("a0:4")
b0, b1, b2, b3 = sp.symbols("b0:4")
n = sp.Symbol("n")

AXES = (a0, a1, a2, a3, b0, b1, b2, b3)

INITIAL_TUPLE = (1, 1, 1, 1, 3, 3, 2, 0)
BASELINE = (20, 18, 16, 15, 31, 29, 27, 1)
BASELINE_B = (2, -2, -6, -8, 24, 20, 16, -36)

_CMF = None


def get_cmf():
    global _CMF
    if _CMF is None:
        _CMF = MeijerG(4, 2, 4, 4, 1)
    return _CMF


def position(vec):
    return Position({s: int(v) for s, v in zip(AXES, vec)})


def make_tm(traj, initial=INITIAL_TUPLE):
    """Trajectory matrix M(n) for direction `traj` from `initial`."""
    return get_cmf().trajectory_matrix(position(traj), position(initial))


def warmup():
    """First trajectory_matrix call in a process is slow (sympy caches)."""
    t0 = time.time()
    make_tm(BASELINE)
    return time.time() - t0


# ---------------------------------------------------------------- Tier 1
def kamidelta_screen(tm, depth=100):
    """Cheap predicted-delta from asymptotic dynamics. Returns max branch."""
    try:
        preds = tm.kamidelta(depth) if depth else tm.kamidelta()
        vals = [float(p) for p in preds if p is not None and sp.im(sp.nsimplify(p, rational=False)) == 0]
        vals = [v for v in vals if math.isfinite(v)]
        return max(vals) if vals else float("-inf")
    except Exception:
        return float("-inf")


# ---------------------------------------------------------------- exact walks
def walk_states(tm, depths):
    """Exact rational state vectors (first column) at each depth (single pass)."""
    mats = tm.walk({n: 1}, list(depths), {n: 0})
    return [m.col(0) for m in mats]


def normalized(state):
    """(c1, c2) = (s1/s0, s2/s0) as exact sympy Rationals."""
    return sp.Rational(state[1], state[0]) if state[0] != 0 else None, \
        sp.Rational(state[2], state[0]) if state[0] != 0 else None


# ---------------------------------------------------------------- zeta library
ZETA_KEYS = (2, 3, 4, 5, 6, 7)
_ZETA_CACHE = {}  # k -> (dps, decimal_string)
_ZETA_CACHE_FILE = None


def _zeta_cache_path():
    import os
    global _ZETA_CACHE_FILE
    if _ZETA_CACHE_FILE is None:
        _ZETA_CACHE_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", ".zeta_cache.json")
    return _ZETA_CACHE_FILE


def _load_zeta_cache():
    import json, os
    path = _zeta_cache_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                for k, (dps, s) in json.load(f).items():
                    _ZETA_CACHE[int(k)] = (int(dps), s)
        except Exception:
            pass


def _save_zeta_cache():
    import json, os, tempfile
    path = _zeta_cache_path()
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as f:
            json.dump({str(k): v for k, v in _ZETA_CACHE.items()}, f)
        os.replace(tmp, path)
    except Exception:
        pass


def get_zeta(k, dps):
    """zeta(k) as an mpf valid at `dps` digits (computed under current context).
    Values are cached (memory + disk) since odd zetas are costly at high dps."""
    if not _ZETA_CACHE:
        _load_zeta_cache()
    cached = _ZETA_CACHE.get(k)
    if cached is None or cached[0] < dps:
        target_dps = int(dps * 1.25) + 50
        with mp.workdps(target_dps):
            if k == 2:
                val = mp.pi ** 2 / 6
            elif k == 3:
                val = mp.apery
            else:
                val = mp.zeta(k)
            _ZETA_CACHE[k] = (target_dps, mp.nstr(val, target_dps))
        _save_zeta_cache()
    return mp.mpf(_ZETA_CACHE[k][1])


def zeta_target(zcoeffs, dps):
    """sum z_i * zeta(i) as mpf under the current context."""
    total = mp.mpf(0)
    for k, z in zip(ZETA_KEYS, zcoeffs):
        if z:
            total += z * get_zeta(k, dps)
    return total


def relation_label(rel):
    """Human-readable target for a 9-int relation (z2..z7, r1, r2, r3)."""
    parts = [f"{z:+d}*z{k}" for k, z in zip(ZETA_KEYS, rel[:6]) if z]
    return " ".join(parts) if parts else "0"


# ---------------------------------------------------------------- PSLQ identify
def identify_relation(c1, c2, digits=250, maxcoeff=10**10):
    """Find integers (r0, r1, r2, r3), r0 != 0, with
        r0*(z2+z3) + r1 + r2*c1 + r3*c2 = 0
    Returns tuple or None.  c1, c2 exact rationals."""
    with mp.workdps(digits + 20):
        target = mp.pi ** 2 / 6 + mp.apery  # = zeta(2)+zeta(3), cached constants
        v1 = mp.mpf(c1.p) / mp.mpf(c1.q)
        v2 = mp.mpf(c2.p) / mp.mpf(c2.q)
        rel = mp.pslq([target, mp.mpf(1), v1, v2], maxcoeff=maxcoeff, maxsteps=10**6)
    if rel is None or rel[0] == 0:
        return None
    return tuple(int(r) for r in rel)


def relation_estimate(rel, c1, c2):
    """Exact rational estimate of z2+z3 implied by relation `rel`."""
    r0, r1, r2, r3 = rel
    return sp.Rational(-(r1 + r2 * c1 + r3 * c2), r0)


def identify_relation_zeta(c1, c2, digits=300, maxcoeff=10**8):
    """PSLQ over [zeta(2..7), 1, c1, c2].  Returns 9 ints
        (z2, z3, z4, z5, z6, z7, r1, r2, r3)
    with sum z_i*zeta(i) + r1 + r2*c1 + r3*c2 = 0 and some z_i nonzero,
    validated at 1.4x precision.  None if nothing found."""
    if c1 is None or c2 is None or c1 == 0 or c2 == 0:
        return None  # PSLQ requires nonzero entries
    with mp.workdps(digits + 30):
        basis = [get_zeta(k, digits + 30) for k in ZETA_KEYS]
        basis += [mp.mpf(1),
                  mp.mpf(c1.p) / mp.mpf(c1.q),
                  mp.mpf(c2.p) / mp.mpf(c2.q)]
        try:
            rel = mp.pslq(basis, maxcoeff=maxcoeff, maxsteps=10**6)
        except ValueError:
            return None
    if rel is None or all(z == 0 for z in rel[:6]):
        return None
    rel = tuple(int(r) for r in rel)
    # sign convention: first nonzero zeta coefficient positive
    first = next(z for z in rel[:6] if z != 0)
    if first < 0:
        rel = tuple(-r for r in rel)
    # high-precision validation against spurious PSLQ relations
    vdigits = int(digits * 1.4) + 50
    with mp.workdps(vdigits):
        res = zeta_target(rel[:6], vdigits) + rel[6] \
            + rel[7] * mp.mpf(c1.p) / mp.mpf(c1.q) \
            + rel[8] * mp.mpf(c2.p) / mp.mpf(c2.q)
        if abs(res) > mp.mpf(10) ** (-digits // 2):
            return None
    return rel


def identify_relation_z2z3(c1, c2, digits=300):
    """Relation REQUIRED to involve both zeta(2) and zeta(3) (9-int form).

    Tries the minimal zeta-library relation first; if it lacks either
    coefficient, falls back to the equal-coefficient z2+z3 PSLQ and lifts it
    to 9-int form.  Returns None when no such linear form exists."""
    rel = identify_relation_zeta(c1, c2, digits=digits)
    if rel is not None and rel[0] != 0 and rel[1] != 0:
        return rel
    legacy = identify_relation(c1, c2, digits=digits)
    if legacy is not None:
        r0, r1, r2, r3 = legacy
        if r0 < 0:
            r0, r1, r2, r3 = -r0, -r1, -r2, -r3
        return (r0, r0, 0, 0, 0, 0, r1, r2, r3)
    return None


def relation_estimate_zeta(rel, c1, c2):
    """Exact rational estimate of T = sum z_i*zeta(i) implied by `rel`."""
    r1, r2, r3 = rel[6], rel[7], rel[8]
    return sp.Rational(-(r1 + r2 * c1 + r3 * c2))


def delta_components_zeta(estimated, zcoeffs):
    """(delta, log_err, log_height) for reduced estimate p/q of
    T = sum z_i*zeta(i)."""
    p = int(sp.numer(estimated))
    q = int(sp.denom(estimated))
    H = max(abs(p), abs(q))
    digits_q = int(H.bit_length() * 0.30103) + 1
    prec = max(200, int(digits_q * 1.3) + 60)
    with mp.workdps(prec):
        T = zeta_target(zcoeffs, prec)
        est = mp.mpf(p) / mp.mpf(q)
        err = abs(T - est)
        log_H = float(mp.log(mp.mpf(H)))
        if err == 0 or q == 1:
            return float("inf"), float("-inf"), log_H
        log_err = float(mp.log(err))
        d = float(-1 - mp.log(err) / mp.log(mp.mpf(q)))
        return d, log_err, log_H


# ---------------------------------------------------------------- exact delta
def delta_exact(estimated):
    """Notebook delta: -1 - log|L - p/q| / log(q), L = z2+z3, exact p/q input."""
    q = sp.denom(estimated)
    digits_q = int(int(q).bit_length() * 0.30103) + 1
    prec = max(200, int(digits_q * 1.3) + 60)
    with mp.workdps(prec):
        L = mp.pi ** 2 / 6 + mp.apery  # cached constants
        est = mp.mpf(sp.numer(estimated).p if hasattr(sp.numer(estimated), 'p') else int(sp.numer(estimated))) / mp.mpf(int(q))
        err = abs(L - est)
        if err == 0:
            return float("inf")
        d = -1 - mp.log(err) / mp.log(mp.mpf(int(q)))
        return float(d)


def delta_components(estimated):
    """(delta, log_err, log_height) for reduced estimate p/q of L = z2+z3.

    Limsup framework:  rho_n = log_err / n,  eta_n = log_height / n,
    delta_n = -1 - rho_n/eta_n  (with height H = max(|p|, |q|))."""
    p = int(sp.numer(estimated))
    q = int(sp.denom(estimated))
    H = max(abs(p), abs(q))
    digits_q = int(H.bit_length() * 0.30103) + 1
    prec = max(200, int(digits_q * 1.3) + 60)
    with mp.workdps(prec):
        L = mp.pi ** 2 / 6 + mp.apery  # cached constants
        est = mp.mpf(p) / mp.mpf(q)
        err = abs(L - est)
        log_H = float(mp.log(mp.mpf(H)))
        if err == 0:
            return float("inf"), float("-inf"), log_H
        log_err = float(mp.log(err))
        d = float(-1 - mp.log(err) / mp.log(mp.mpf(q)))
        return d, log_err, log_H


def delta_at_depths(tm, rel, depths):
    """Exact finite-depth deltas for a fixed relation at several depths."""
    states = walk_states(tm, depths)
    out = []
    for st in states:
        c1, c2 = normalized(st)
        if c1 is None:
            out.append(float("nan"))
            continue
        est = relation_estimate(rel, c1, c2)
        out.append(delta_exact(est))
    return out


def extrapolate_delta(depths, deltas):
    """Fit delta_N ~ delta_inf + c/N (least squares); returns (delta_inf, resid)."""
    pts = [(d, v) for d, v in zip(depths, deltas) if math.isfinite(v)]
    if len(pts) < 2:
        return float("nan"), float("inf")
    xs = [1.0 / d for d, _ in pts]
    ys = [v for _, v in pts]
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = m * sxx - sx * sx
    if den == 0:
        return ys[-1], float("inf")
    slope = (m * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / m
    resid = math.sqrt(sum((intercept + slope * x - y) ** 2 for x, y in zip(xs, ys)) / m)
    return intercept, resid


# ---------------------------------------------------------------- colleague-pipeline helpers
def trajectory_scale(traj):
    """PadicInterpolation's cost driver: max coordinate and cross |x_i - y_j|.

    Their auto-policy sets max_break_denominator = modulus * this scale;
    profile depth (and the 74-min closed-form cost) grows with it."""
    x, y = traj[:4], traj[4:]
    cross = max(abs(a - b) for a in x for b in y)
    return max(max(abs(v) for v in traj), abs(sum(x) - sum(y)), cross)


def _vp(r, p):
    """p-adic valuation of a sympy Rational (min convention for gcd of rationals)."""
    num, den = int(sp.numer(r)), int(sp.denom(r))
    if num == 0:
        return None
    v = 0
    while num % p == 0:
        num //= p
        v += 1
    while den % p == 0:
        den //= p
        v -= 1
    return v


def gcd_diagnostics(tm, rel, N=400):
    """Pre-check for the colleague's valuation-profile reconstruction.

    Walks the trajectory matrix exactly, forms the linear-form pair
      v_n = -(r1*s0 + r2*s1 + r3*s2),  u_n = r0*s0,   d_n = gcd(v_n, u_n)
    and reports:
      gcd_rate      ~ lim log|d_n|/n            (tail estimate)
      raw_rate      ~ lim log|u_n|/n
      reduced_rate  = raw_rate - gcd_rate       (denominator growth G)
      profile_agreement in [0,1]: cross-prime consistency of v_p(d_n) vs n/p
        (proxy for whether their weak-periodicity reconstruction will pass)
      small_prime_slopes: v_p(d_N)/N for p in {2,3,5,7}
    """
    r0, r1, r2, r3 = rel
    P = sp.eye(3)
    d_vals = [None] * (N + 1)
    u_last = v_last = None
    for j in range(N):
        P = P * tm.subs({n: j})
        k = j + 1  # depth after j+1 factors M(0)..M(j), matching tm.walk
        s0, s1, s2 = P[0, 0], P[1, 0], P[2, 0]
        v_n = -(r1 * s0 + r2 * s1 + r3 * s2)
        u_n = r0 * s0
        d_vals[k] = (v_n, u_n)
        if k == N:
            u_last, v_last = u_n, v_n

    def vp_d(k, p):
        v_n, u_n = d_vals[k]
        a, b = _vp(v_n, p), _vp(u_n, p)
        if a is None or b is None:
            return None
        return min(a, b)

    # tail growth rates
    def log_abs(r):
        num, den = abs(int(sp.numer(r))), abs(int(sp.denom(r)))
        return (num.bit_length() - den.bit_length()) * math.log(2)

    vN, uN = d_vals[N]
    vH, uH = d_vals[N // 2]
    g_full = math.gcd(abs(int(sp.numer(vN))) * abs(int(sp.denom(uN))),
                      abs(int(sp.numer(uN))) * abs(int(sp.denom(vN))))
    gcd_rate = math.log(g_full) / N if g_full > 0 else float("nan")
    raw_rate = log_abs(uN) / N
    reduced_rate = raw_rate - gcd_rate

    # cross-prime profile agreement: v_p(d_n) at matched x = n/p
    primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    primes = [p for p in primes if 3 * p <= N]
    agree = total = 0
    xs = [i / 10 for i in range(2, 31)]  # x = n/p in (0.2, 3.0]
    for x in xs:
        vals = []
        for p in primes:
            k = round(x * p)
            if 1 <= k <= N:
                val = vp_d(k, p)
                if val is not None:
                    vals.append(val)
        if len(vals) >= 3:
            mode = max(set(vals), key=vals.count)
            agree += vals.count(mode)
            total += len(vals)
    profile_agreement = agree / total if total else float("nan")

    slopes = {}
    for p in (2, 3, 5, 7):
        val = vp_d(N, p)
        slopes[p] = (val / N) if val is not None else None

    return {
        "gcd_rate": gcd_rate,
        "raw_rate": raw_rate,
        "reduced_rate": reduced_rate,
        "profile_agreement": profile_agreement,
        "small_prime_slopes": slopes,
        "N": N,
    }


def gcd_diagnostics_task(args):
    """Pool-friendly wrapper: (traj, rel, N) -> diagnostics dict.
    rel may be legacy 4-int (r0,r1,r2,r3) or 9-int zeta relation."""
    traj, rel, N = args
    out = {"traj": list(traj)}
    if len(rel) == 9:
        rel = (1, rel[6], rel[7], rel[8])
    try:
        tm = make_tm(traj)
        out.update(gcd_diagnostics(tm, rel, N=N))
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"error: {type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------- high-depth verify
def full_verify(args):
    """((initial, traj), depths) -> record with high-depth exact deltas.
    Also accepts legacy (traj, depths).  Pool-friendly."""
    cand, depths = args
    if len(cand) == 2 and isinstance(cand[0], (tuple, list)):
        initial, traj = tuple(cand[0]), tuple(cand[1])
    else:
        initial, traj = INITIAL_TUPLE, tuple(cand)
    rec = {"initial": list(initial), "traj": list(traj), "depths": list(depths)}
    t0 = time.time()
    try:
        tm = make_tm(traj, initial=initial)
        states = walk_states(tm, depths)
        c1, c2 = normalized(states[0])
        rel = identify_relation_z2z3(c1, c2, digits=600)
        if rel is None:
            rec["status"] = "no_relation"
            return rec
        rec["relation"] = list(rel)
        rec["target"] = relation_label(rel)
        deltas = []
        for st in states:
            cc1, cc2 = normalized(st)
            est = relation_estimate_zeta(rel, cc1, cc2)
            deltas.append(delta_components_zeta(est, rel[:6])[0])
        rec["deltas"] = deltas
        dinf, resid = extrapolate_delta(depths, deltas)
        rec["delta_inf"] = dinf
        rec["resid"] = resid
        # legacy z2+z3 form (unproven target) tracked in parallel
        try:
            rel23 = identify_relation(c1, c2, digits=600)
            if rel23 is not None:
                rec["relation_z23"] = list(rel23)
                d23 = []
                for st in states:
                    cc1, cc2 = normalized(st)
                    d23.append(delta_exact(relation_estimate(rel23, cc1, cc2)))
                rec["deltas_z23"] = d23
                rec["delta_inf_z23"] = extrapolate_delta(depths, d23)[0]
        except Exception:
            pass
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {e}"
    rec["time"] = time.time() - t0
    return rec


# ---------------------------------------------------------------- float64 delta proxy
def lambdify_tm(tm):
    """Numeric M(n) evaluator (float64) for the Lyapunov proxy."""
    f = sp.lambdify(n, tm, modules="numpy")
    return f


def delta_proxy(tm, N=300, burn=20):
    """2-column QR Lyapunov proxy: delta_proxy = -lam2/lam1.
    This is E/G with G = raw (unreduced) denominator growth -> screening only."""
    import numpy as np
    f = lambdify_tm(tm)
    Q = np.eye(3, 2)
    log1 = log2 = 0.0
    steps = 0
    for k in range(1, N + 1):
        try:
            M = np.array(f(k), dtype=np.float64)
        except Exception:
            return float("nan"), float("nan"), float("nan")
        if not np.all(np.isfinite(M)):
            return float("nan"), float("nan"), float("nan")
        V = M.T @ Q
        r11 = max(np.linalg.norm(V[:, 0]), 1e-300)
        q1 = V[:, 0] / r11
        r12 = q1 @ V[:, 1]
        v2 = V[:, 1] - r12 * q1
        r22 = max(np.linalg.norm(v2), 1e-300)
        q2 = v2 / r22
        Q = np.stack([q1, q2], axis=1)
        if k > burn:
            log1 += math.log(r11)
            log2 += math.log(r22)
            steps += 1
    if steps == 0 or log1 == 0:
        return float("nan"), float("nan"), float("nan")
    lam1, lam2 = log1 / steps, log2 / steps
    return -lam2 / lam1, lam1, lam2
