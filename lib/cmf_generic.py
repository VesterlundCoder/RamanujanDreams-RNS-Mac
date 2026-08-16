"""
cmf_generic.py  (experimental_direct_companion)
==============
*** WARNING: NOT Ramanujan Dreams CMF trajectory semantics. ***
This module implements the DIRECT companion product P <- P*M(n).
It is RETIRED from the production verification path; the corrected
reference is lib/cmf_walk_corrected.py (theta companion + I+C/a axis
operators + unit-step trajectory decomposition).

Dimension-generic CMF machinery for hypergeometric pF(p-1) companion fields.
Generalizes enrich.py (hard-coded 6F5, DIM=6) to arbitrary order `dim = p`:

    6F5  -> dim=6,  nshift = 2*dim-1 = 11   (6 numerator + 5 denominator roots)
    8F7  -> dim=8,  nshift = 2*dim-1 = 15   (8 numerator + 7 denominator roots)

Convention (same gauge as the verified 6F5 pipeline):
    f_i = shift[i]       + n*dir[i]       + 1     (i = 0 .. dim-1)     numerator
    g_j = shift[dim+j]   + n*dir[dim+j]   + 2     (j = 0 .. dim-2)     denominator
    h_f = -z_num,  h_g = z_den            (integer-cleared: all entries are integers)
    ef  = esym(f)  (len dim+1, e0..e_dim)
    eg  = esym(g)  (len dim,   e0..e_{dim-1}) padded to len dim+1
    c[0]   = h_f * ef[dim]
    c[k]   = h_g * eg[dim-k] + h_f * ef[dim-k]    (k = 1 .. dim-1)
    M(n): dim x dim companion, sub-diagonal = 1, last column = c

Provides three engines that must agree (proof-ladder triangulation):
    spectral_dhat   : float64 QR Lyapunov,  dhat = -lam2/lam1           (Tier-1, cheap)
    arith_delta     : mpmath exact-ish double-depth delta               (Tier-2)
    independent_delta: pure-Python-int double-depth delta (no mpmath)   (V1, non-circular)
"""
from __future__ import annotations

import math

import numpy as np
import mpmath as mp

try:
    from spectral_delta import lyapunov_spectrum
except ImportError:  # spectral_delta only needed for Tier-1 spectral engines,
    lyapunov_spectrum = None  # not for the integer-exact PSLQ pipeline.


def nshift_for(dim: int) -> int:
    return 2 * dim - 1


# ── elementary symmetric polynomials (three independent implementations) ──
def _esym_float(vals):
    n = len(vals)
    e = np.zeros(n + 1)
    e[0] = 1.0
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def _esym_mp(vals):
    n = len(vals)
    e = [mp.mpf(0)] * (n + 1)
    e[0] = mp.mpf(1)
    for v in vals:
        for k in range(n, 0, -1):
            e[k] += v * e[k - 1]
    return e


def _esym_int(roots):
    """Coeffs of prod (t + r_i) by integer polynomial convolution -> [e0..ek]."""
    poly = [1]
    for r in roots:
        new = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new[i] += c
            new[i + 1] += c * r
        poly = new
    return poly  # [e0, e1, ..., ek]


# ── companion-matrix builders ──
def _roots(n, shift, dirv, dim):
    f = [shift[i] + n * dirv[i] + 1 for i in range(dim)]
    g = [shift[dim + j] + n * dirv[dim + j] + 2 for j in range(dim - 1)]
    return f, g


def build_M_float(n, shift, dirv, z_num, z_den, dim):
    h_f, h_g = float(-z_num), float(z_den)
    f, g = _roots(n, shift, dirv, dim)
    ef = _esym_float(f)
    eg = np.concatenate([_esym_float(g), [0.0]])
    c = np.zeros(dim)
    c[0] = h_f * ef[dim]
    for k in range(1, dim):
        c[k] = h_g * eg[dim - k] + h_f * ef[dim - k]
    M = np.zeros((dim, dim))
    for rr in range(1, dim):
        M[rr, rr - 1] = 1.0
    M[:, dim - 1] = c
    return M


def build_M_mp(n, shift, dirv, z_num, z_den, dim):
    h_f, h_g = mp.mpf(-z_num), mp.mpf(z_den)
    f, g = _roots(n, shift, dirv, dim)
    ef = _esym_mp([mp.mpf(x) for x in f])
    eg = _esym_mp([mp.mpf(x) for x in g]) + [mp.mpf(0)]
    c = [mp.mpf(0)] * dim
    c[0] = h_f * ef[dim]
    for k in range(1, dim):
        c[k] = h_g * eg[dim - k] + h_f * ef[dim - k]
    M = mp.zeros(dim)
    for rr in range(1, dim):
        M[rr, rr - 1] = mp.mpf(1)
    for rr in range(dim):
        M[rr, dim - 1] = c[rr]
    return M


def build_M_int(n, shift, dirv, z_num, z_den, dim):
    h_f, h_g = -z_num, z_den
    f, g = _roots(n, shift, dirv, dim)
    ef = _esym_int(f)
    eg = _esym_int(g) + [0]
    c = [0] * dim
    c[0] = h_f * ef[dim]
    for k in range(1, dim):
        c[k] = h_g * eg[dim - k] + h_f * ef[dim - k]
    M = [[0] * dim for _ in range(dim)]
    for rr in range(1, dim):
        M[rr][rr - 1] = 1
    for rr in range(dim):
        M[rr][dim - 1] = c[rr]
    return M


def matmul_int(A, B, dim):
    C = [[0] * dim for _ in range(dim)]
    for i in range(dim):
        Ai, Ci = A[i], C[i]
        for k in range(dim):
            a = Ai[k]
            if a:
                Bk = B[k]
                for j in range(dim):
                    Ci[j] += a * Bk[j]
    return C


def log_bigint(n):
    n = abs(int(n))
    if n == 0:
        return float("-inf")
    b = n.bit_length()
    shift = max(0, b - 53)
    return math.log(n >> shift) + shift * math.log(2.0)


# ── Tier-1: spectral proxy ──
def spectral_dhat(shift, dirv, z_num, z_den, N, dim):
    lam = lyapunov_spectrum(
        lambda n: build_M_float(n + 1, shift, dirv, z_num, z_den, dim), dim, N)
    if lam[0] == 0 or not np.isfinite(lam[0]):
        return float("nan"), lam
    return float(-lam[1] / lam[0]), lam


def spectral_ratios(shift, dirv, z_num, z_den, N, dim):
    """Full ladder of subdominant ratios r_k = -lambda_k / lambda_1, k=2..dim.

    The standard detector is r_2 = -lambda_2/lambda_1. When the observable
    (last column of the cocycle) does NOT couple to the 2nd Lyapunov mode --
    e.g. degenerate / block-structured trajectories where rows collapse -- the
    TRUE convergence is driven by a deeper mode lambda_k (k>2). This returns the
    whole ladder so a 'dominant driver elsewhere in the cocycle' can be found.

    Returns (ratios, lam) where ratios is a float array of length dim-1
    [r_2, r_3, ..., r_dim] (NaN entries where lambda_1 is degenerate).
    """
    lam = lyapunov_spectrum(
        lambda n: build_M_float(n + 1, shift, dirv, z_num, z_den, dim), dim, N)
    l1 = lam[0]
    if l1 == 0 or not np.isfinite(l1):
        return np.full(dim - 1, np.nan), lam
    ratios = np.array([-lam[k] / l1 for k in range(1, dim)], dtype=float)
    return ratios, lam


def spectral_dhat_robust(shift, dirv, z_num, z_den, N, dim,
                         degen_hi=1.05, min_thresh=None):
    """Primary detector with a multi-ratio fallback.

    Returns (dhat, driver_k, lam, ratios):
      * dhat      : the chosen detector value
      * driver_k  : Lyapunov index (2-based) whose ratio was chosen
      * ratios    : full ladder [r_2..r_dim]

    Logic: use r_2 unless it is 'non-meaningful' -- degenerate (|r_2|>=degen_hi)
    or below an optional floor -- in which case scan deeper ratios r_3, r_4, ...
    and pick the FIRST finite, non-degenerate one (the next dominant driver).
    This NEVER overrides a meaningful r_2; it only rescues otherwise-dead points.
    All selections are candidates to be confirmed by exact arithmetic.
    """
    ratios, lam = spectral_ratios(shift, dirv, z_num, z_den, N, dim)
    if not np.isfinite(ratios[0]):
        return float("nan"), 0, lam, ratios
    r2 = ratios[0]
    meaningful = np.isfinite(r2) and abs(r2) < degen_hi and \
        (min_thresh is None or r2 >= min_thresh)
    if meaningful:
        return float(r2), 2, lam, ratios
    for k in range(1, len(ratios)):           # ratios[k] == r_{k+2}
        rk = ratios[k]
        if np.isfinite(rk) and abs(rk) < degen_hi:
            return float(rk), k + 2, lam, ratios
    return float(r2), 2, lam, ratios          # nothing better: report r_2 as-is


# ── Tier-2: mpmath exact-ish double-depth delta ──
def arith_delta(shift, dirv, z_num, z_den, N, dim, lam1_hint=None):
    if lam1_hint is None or not np.isfinite(lam1_hint):
        lam1_hint = 8.0
    digits = abs(lam1_hint) * N / 2.302585
    mp.mp.dps = int(2.6 * digits) + 120
    P = mp.eye(dim)
    snapN = None
    for n in range(1, 2 * N + 1):
        P = P * build_M_mp(n, shift, dirv, z_num, z_den, dim)
        if n == N:
            snapN = mp.matrix(P)
    best = None
    last = dim - 1
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            qn, q2 = snapN[j, last], P[j, last]
            if qn == 0 or q2 == 0:
                continue
            err = abs(snapN[i, last] / qn - P[i, last] / q2)
            if err == 0:
                continue
            lq = mp.log(abs(qn))
            if lq == 0:
                continue
            d = float(-(1 + mp.log(err) / lq))
            if best is None or d > best:
                best = d
    return best


# ── V1: independent pure-integer double-depth delta ──
def independent_delta(shift, dirv, z_num, z_den, N, dim):
    P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    snapN = None
    for n in range(1, 2 * N + 1):
        P = matmul_int(P, build_M_int(n, shift, dirv, z_num, z_den, dim), dim)
        if n == N:
            snapN = [row[:] for row in P]
    best = None
    last = dim - 1
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            pn, qn = snapN[i][last], snapN[j][last]
            p2, q2 = P[i][last], P[j][last]
            if qn == 0 or q2 == 0:
                continue
            cross = pn * q2 - p2 * qn
            if cross == 0:
                continue
            log_err = log_bigint(cross) - log_bigint(qn) - log_bigint(q2)
            log_q = log_bigint(qn)
            if log_q == 0:
                continue
            d = -(1.0 + log_err / log_q)
            if best is None or d > best:
                best = d
    return best


# ── generic search-space pools (dimension-aware) ──
def dir_pool(dim):
    nsh = nshift_for(dim)
    pats = []
    for i in range(nsh):                       # single advancing root, +1
        d = [0] * nsh; d[i] = 1; pats.append(d)
    for i in range(dim, nsh):                  # single advancing denominator root, +2
        d = [0] * nsh; d[i] = 2; pats.append(d)
    # a few paired patterns (last num root + denom roots)
    for (i, j) in [(dim - 2, dim - 1), (dim, dim + 2), (dim + 2, dim + 3)]:
        if 0 <= i < nsh and 0 <= j < nsh:
            d = [0] * nsh; d[i] = 1; d[j] = 1; pats.append(d)
    return pats


def z_pool(z_max=1.0, z_min=0.1):
    """Rational z-values with z_min <= |z| < z_max (STRICT upper bound).
    |z| >= 1 makes the companion cocycle diverge (no convergent ratio), so the
    physically meaningful window is the open interval (-1, 1)."""
    from fractions import Fraction
    zs = set()
    qmax = int(math.ceil(2 * z_max))
    for q in sorted(set((1, 2, 3, 4, 5, 6, 10, 20)) | {qmax}):
        for p in range(-int(math.ceil(z_max * q)) - 1, int(math.ceil(z_max * q)) + 2):
            if p == 0:
                continue
            fr = Fraction(p, q)
            if z_min <= abs(float(fr)) < z_max:
                zs.add((fr.numerator, fr.denominator))
    return sorted(zs)


def default_seed(dim):
    """A small-shift seed; numerator shifts near 0, denominator shifts near -2."""
    nsh = nshift_for(dim)
    s = [0] * nsh
    for j in range(dim, nsh):
        s[j] = -2
    return s
