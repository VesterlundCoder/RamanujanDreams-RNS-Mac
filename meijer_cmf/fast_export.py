#!/usr/bin/env python
"""Fast trajectory-matrix generation for the MeijerG(4,2,4,4,1) GPU pipeline.

Replaces the per-candidate sympy path (make_tm ~1s + integer_poly_matrix
~5.6s) with python-flint fmpz_poly arithmetic:

- Each CMF axis matrix is (polynomial matrix N_ax) / (scalar polynomial).
  Scalar factors cancel in the projective limits c1, c2, so the trajectory
  matrix can be built as a product of numerator-only matrices.
- Backward steps use the polynomial adjugate: for 3x3, M^-1 = adj(M)/det(M)
  and det is scalar, so adj(N_ax) is the backward numerator.
- No denominators => no removable singularities, any unit-step order works.
- The raw product is (scalar polynomial) x M(n); the scalar vanishes at the
  removable-singularity steps, so it is stripped exactly with a flint
  polynomial GCD across the 9 entries, recovering the minimal degree ~67.

The per-axis term tables (numerator + adjugate) are built with sympy ONCE
and cached in axis_tables.pkl.

Usage (from meijer_search/):
  ./venv/bin/python rns/fast_export.py --selftest
  ./venv/bin/python rns/fast_export.py --pairs pairs.json --depth 1000 \
      --checkpoints 500,900,1000 --export rns/export_fast [--workers 8]
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from flint import fmpz_poly  # noqa: E402

TABLES_PATH = os.path.join(_HERE, "axis_tables.pkl")
M_DIM = 3
DEFAULT_INITIAL = (1, 1, 1, 1, 3, 3, 2, 0)
BASELINE = (1, -1, -3, -4, 12, 10, 8, -18)

_TABLES = None


# ------------------------------------------------------------ axis tables
def build_axis_tables():
    """Per axis: numerator matrix and its adjugate as term tables
    entry(i,j) -> [(int coeff, (e0..e7)), ...].  sympy, run once."""
    import sympy as sp
    import common as C

    def to_terms(expr):
        p = sp.Poly(sp.expand(expr), *C.AXES)
        return [(int(c), tuple(int(e) for e in mono))
                for mono, c in zip(p.monoms(), p.coeffs())]

    def adjugate(N):
        return N.adjugate()

    cmf = C.get_cmf()
    tables = []
    for ax in C.AXES:
        M = cmf.matrices[ax]
        den = sp.lcm([sp.fraction(sp.cancel(sp.together(e)))[1] for e in M])
        N = M.applyfunc(lambda e: sp.expand(sp.cancel(sp.together(e * den))))
        A = adjugate(N).applyfunc(sp.expand)
        fwd = [[to_terms(N[i, j]) for j in range(M_DIM)] for i in range(M_DIM)]
        bwd = [[to_terms(A[i, j]) for j in range(M_DIM)] for i in range(M_DIM)]
        tables.append({"fwd": fwd, "bwd": bwd})
    return tables


def get_tables():
    global _TABLES
    if _TABLES is None:
        if os.path.exists(TABLES_PATH):
            with open(TABLES_PATH, "rb") as f:
                _TABLES = pickle.load(f)
        else:
            _TABLES = build_axis_tables()
            with open(TABLES_PATH, "wb") as f:
                pickle.dump(_TABLES, f)
    return _TABLES


# ------------------------------------------------------------ fast build
def _hoist_axis(table, lin, axis):
    """Precompute, for one entry's term list, [(base_poly, e_axis)] where
    base = coeff * prod_{j != axis} lin_j^e_j.  Within one axis's unit
    steps only lin[axis] changes, so bases are loop-invariant."""
    out = []
    for coeff, exps in table:
        base = fmpz_poly([coeff])
        for j, e in enumerate(exps):
            if e and j != axis:
                lp = fmpz_poly([lin[j][0], lin[j][1]])
                for _ in range(e):
                    base *= lp
        out.append((base, exps[axis]))
    return out


def _entry_from_hoisted(hoisted, axis_pows):
    acc = fmpz_poly([0])
    for base, e in hoisted:
        acc += base * axis_pows[e] if e else base
    return acc


def _matmul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(M_DIM)),
                 fmpz_poly([0])) for j in range(M_DIM)]
            for i in range(M_DIM)]


def poly_matrix_fast(initial, traj):
    """Trajectory matrix (up to a scalar polynomial) as Horner coefficient
    lists [3][3], highest degree first, uniform length deg+1.

    Matches the walk convention P <- P * M(t), t = 0..N-1: unit-step
    product from position initial + t*traj to initial + (t+1)*traj."""
    tables = get_tables()
    offset = [0] * 8
    P = None
    for i in range(8):
        v = traj[i]
        if v == 0:
            continue
        table = tables[i]["fwd" if v > 0 else "bwd"]
        # coords j != i are constant during this axis's steps
        lin = [(initial[j] + offset[j], traj[j]) for j in range(8)]
        hoisted = [[_hoist_axis(table[i2][j2], lin, i) for j2 in range(M_DIM)]
                   for i2 in range(M_DIM)]
        max_e = max(e for row in hoisted for h in row for _, e in h)
        for _ in range(abs(v)):
            if v < 0:
                offset[i] -= 1
            la = fmpz_poly([initial[i] + offset[i], traj[i]])
            axis_pows = [fmpz_poly([1])]
            for _e in range(max_e):
                axis_pows.append(axis_pows[-1] * la)
            M = [[_entry_from_hoisted(hoisted[i2][j2], axis_pows)
                  for j2 in range(M_DIM)] for i2 in range(M_DIM)]
            P = M if P is None else _matmul(P, M)
            if v > 0:
                offset[i] += 1
    # The raw product is (scalar polynomial) x M(n); that scalar has integer
    # roots inside the walk range (the removable singularities), where the
    # uncancelled matrix would vanish identically.  Strip it exactly.
    g = None
    for i in range(M_DIM):
        for j in range(M_DIM):
            g = P[i][j] if g is None else g.gcd(P[i][j])
    if g.degree() > 0 or abs(int(g[0])) > 1:
        P = [[P[i][j] // g for j in range(M_DIM)] for i in range(M_DIM)]
    deg = max(P[i][j].degree() for i in range(M_DIM) for j in range(M_DIM))
    coeffs = []
    for i in range(M_DIM):
        row = []
        for j in range(M_DIM):
            cs = [int(P[i][j][d]) for d in range(deg + 1)]  # low->high
            row.append(list(reversed(cs)))                  # high->low
        coeffs.append(row)
    return coeffs, M_DIM, deg


# ------------------------------------------------------------ export
def _build_one(pair):
    initial, traj = pair
    t0 = time.time()
    coeffs, m, deg = poly_matrix_fast(tuple(initial), tuple(traj))
    return initial, traj, coeffs, m, deg, time.time() - t0


def export_fast(pairs, depth, checkpoints, out_dir, workers=1):
    from df64_walk import df64_split

    t0 = time.time()
    if workers > 1:
        with Pool(workers, initializer=get_tables) as pool:
            built = pool.map(_build_one, pairs, chunksize=4)
    else:
        built = [_build_one(p) for p in pairs]
    build_s = time.time() - t0

    m = built[0][3]
    deg1 = max(b[4] for b in built) + 1
    B, E = len(built), m * m
    mant = np.zeros((B, E, deg1, 2), dtype=np.float32)
    exps = np.zeros((B, E, deg1), dtype=np.int32)
    for bi, (_, _, coeffs, _, d, _) in enumerate(built):
        pad = deg1 - (d + 1)
        padded = [[([0] * pad) + e for e in row] for row in coeffs]
        mm, ee, _ = df64_split(padded, m, deg1)
        mant[bi] = mm.reshape(E, deg1, 2)
        exps[bi] = ee

    os.makedirs(out_dir, exist_ok=True)
    mant.tofile(os.path.join(out_dir, "coeffMant.bin"))
    exps.tofile(os.path.join(out_dir, "coeffExp.bin"))
    cfg = {"nTraj": B, "m": m, "deg1": deg1, "nSteps": depth,
           "nCheckpoints": len(checkpoints),
           "checkpointSteps": sorted(checkpoints),
           "pairs": [[list(b[0]), list(b[1])] for b in built]}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    total = time.time() - t0
    print(f"[export-fast] {out_dir}: B={B} m={m} deg1={deg1} "
          f"build={build_s:.2f}s total={total:.2f}s "
          f"({B/total:.1f} cand/s)", flush=True)


# ------------------------------------------------------------ selftest
def selftest():
    """Exact identity check: the fast product must equal the sympy
    trajectory matrix up to one scalar polynomial, i.e.
        P_fast[i][j] * P_sympy[0][0] == P_sympy[i][j] * P_fast[0][0]
    as integer polynomials, for every entry."""
    import common as C
    from rns_walk import integer_poly_matrix

    pairs = [(DEFAULT_INITIAL, BASELINE),        # has backward steps
             ((1, 1, 1, 1, 4, 4, 2, 0), (12, 11, 10, 9, 23, 22, 21, 0))]
    ok = True
    for initial, traj in pairs:
        t0 = time.time()
        fast, m, deg_f = poly_matrix_fast(initial, traj)
        fast_s = time.time() - t0
        t0 = time.time()
        tm = C.make_tm(tuple(traj), initial=tuple(initial))
        ref, _, deg_r = integer_poly_matrix(tm)
        sympy_s = time.time() - t0

        def poly(cs):  # high->low int list -> fmpz_poly
            return fmpz_poly(list(reversed(cs)))

        f00, r00 = poly(fast[0][0]), poly(ref[0][0])
        good = all(poly(fast[i][j]) * r00 == poly(ref[i][j]) * f00
                   for i in range(m) for j in range(m))
        ok &= good
        print(f"  traj={traj} deg_fast={deg_f} deg_sympy={deg_r} "
              f"fast={fast_s*1000:.0f}ms sympy={sympy_s:.1f}s "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("SELFTEST PASS" if ok else "SELFTEST FAIL", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs")
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--export", metavar="DIR")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    with open(args.pairs) as f:
        pairs = [(tuple(p[0]), tuple(p[1])) for p in json.load(f)]
    cps = ([int(x) for x in args.checkpoints.split(",") if x]
           if args.checkpoints else [args.depth])
    export_fast(pairs, args.depth, cps, args.export, workers=args.workers)


if __name__ == "__main__":
    main()
