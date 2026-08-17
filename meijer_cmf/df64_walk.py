#!/usr/bin/env python
"""Float walk with per-step normalization for the MeijerG(4,2,4,4,1) CMF.

CPU reference of the planned walk_meijer_df64.metal kernel (the df64 analog
of walk_6f5_df64.metal).  Design notes:

- Unit-step products of the 8 axis matrices do NOT work numerically: at
  many integer positions individual steps are singular and only cancel
  symbolically (ramanujantools removes these removable singularities in
  its symbolic diagonal decomposition).  The baseline pair is already
  singular at step 0.
- Instead the walk evaluates the degree-~67 integer polynomial trajectory
  matrix directly, with ALL coefficients pre-scaled by a single global
  power of two (projectively exact) so the dominant coefficient is ~1.
  Values then stay within float range: p(n) <= (deg+1) * n^deg ~ 2^680
  at n = 1000 (fits float64; the Metal kernel additionally renormalizes
  the Horner accumulator with a software exponent to fit float32 range).
- Every macro step the product matrix is rescaled by an exact power of
  two and the exponent accumulated, so G_raw (unreduced log-growth) comes
  for free alongside the limits:

    c1 = P[1,0]/P[0,0],  c2 = P[2,0]/P[0,0]   (~13 digits)

This is the fast no-primes screening stage; exact RNS/CRT stays reserved
for survivors.

Usage (from meijer_search/):
  ./venv/bin/python rns/df64_walk.py --validate --depth 100
  ./venv/bin/python rns/df64_walk.py --depth 1000 --checkpoints 500,900,1000
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common as C  # noqa: E402

DEFAULT_INITIAL = (1, 1, 1, 1, 3, 3, 2, 0)
BASELINE = (1, -1, -3, -4, 12, 10, 8, -18)

from rns_walk import integer_poly_matrix  # noqa: E402


def scaled_coeff_array(coeffs, m):
    """Integer Horner coefficients -> float64 array [E][deg1], all scaled
    by one global power of two so the largest coefficient is ~1.
    A single scalar on the whole matrix is projectively exact."""
    deg1 = len(coeffs[0][0])
    maxbits = max(abs(c).bit_length()
                  for row in coeffs for e in row for c in e)
    shift = maxbits - 1
    arr = np.zeros((m * m, deg1))
    for i in range(m):
        for j in range(m):
            for d, c in enumerate(coeffs[i][j]):
                if c:
                    e = c.bit_length()
                    keep = min(e, 53)
                    mant = (abs(c) >> (e - keep)) / (1 << keep)
                    val = math.ldexp(mant, e - shift)
                    arr[i * m + j, d] = val if c > 0 else -val
    return arr, deg1


def walk_df(initial, traj, depth, checkpoints):
    """Polynomial-matrix float64 walk with exact power-of-two rescaling.

    Returns {checkpoint: (c1, c2, log2_growth)} (growth up to the global
    coefficient scale, constant across checkpoints)."""
    tm = C.make_tm(tuple(traj), initial=tuple(initial))
    coeffs, m, _ = integer_poly_matrix(tm)
    arr, deg1 = scaled_coeff_array(coeffs, m)
    P = np.eye(m)
    exp_sum = 0
    out = {}
    cps = set(checkpoints)

    for t in range(depth):
        val = arr[:, 0].copy()
        for d in range(1, deg1):
            val = val * t + arr[:, d]
        M = val.reshape(m, m)
        P = P @ M
        mx = np.abs(P).max()
        if mx > 0.0 and math.isfinite(mx):
            e = math.frexp(mx)[1]
            P *= math.ldexp(1.0, -e)     # exact
            exp_sum += e
        if t + 1 in cps:
            if P[0, 0] != 0.0:
                out[t + 1] = (P[1, 0] / P[0, 0], P[2, 0] / P[0, 0],
                              exp_sum + math.log2(abs(P[0, 0])))
            else:
                out[t + 1] = None
    return out


def df64_split(coeffs, m, deg1):
    """Integer coefficients -> (float2 mantissa [E][deg1], int32 exp [E][deg1])
    with one global power-of-two scale (returned as shift).
    value = (hi + lo) * 2^exp, |hi+lo| in [0.5, 1)."""
    maxbits = max(abs(c).bit_length()
                  for row in coeffs for e in row for c in e) or 1
    mant = np.zeros((m * m, deg1, 2), dtype=np.float32)
    exps = np.zeros((m * m, deg1), dtype=np.int32)
    for i in range(m):
        for j in range(m):
            for d, c in enumerate(coeffs[i][j]):
                if c == 0:
                    continue
                bl = abs(c).bit_length()
                keep = min(bl, 53)
                x = math.copysign((abs(c) >> (bl - keep)) / (1 << keep), c)
                hi = np.float32(x)
                lo = np.float32(x - float(hi))
                mant[i * m + j, d] = (hi, lo)
                exps[i * m + j, d] = bl - maxbits
    return mant, exps, maxbits


def export_metal(pairs, depth, checkpoints, out_dir):
    """Tensors for main_meijer_df64.mm / walk_meijer_df64.metal.

    Layout: coeffMant.bin float32[B][E][deg1][2], coeffExp.bin int32[B][E][deg1],
    config.json.  One padded degree across the batch."""
    built = []
    for initial, traj in pairs:
        tm = C.make_tm(tuple(traj), initial=tuple(initial))
        coeffs, m, deg = integer_poly_matrix(tm)
        built.append((initial, traj, coeffs, m, deg))
    m = built[0][3]
    assert all(b[3] == m for b in built), "mixed matrix dimensions"
    deg1 = max(b[4] for b in built) + 1
    B, E = len(built), m * m

    mant = np.zeros((B, E, deg1, 2), dtype=np.float32)
    exps = np.zeros((B, E, deg1), dtype=np.int32)
    for bi, (_, _, coeffs, _, d) in enumerate(built):
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
           "pairs": [[list(i), list(t)] for i, t, *_ in built]}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[export] {out_dir}: B={B} m={m} deg1={deg1} "
          f"({mant.nbytes/1e6:.1f} MB mantissas)", flush=True)


def check_gpu(export_dir, out_bin):
    """Verify GPU output (main_meijer_df64.mm) against the CPU walk.

    Binary layout: for each checkpoint c, candidate b:
      float2 mat[E], int32 expRel[E], int32 growth."""
    with open(os.path.join(export_dir, "config.json")) as f:
        cfg = json.load(f)
    B, m, cps = cfg["nTraj"], cfg["m"], cfg["checkpointSteps"]
    E = m * m
    Ccp = len(cps)
    raw = np.fromfile(out_bin, dtype=np.uint8)
    rec = E * 8 + E * 4 + 4
    assert raw.size == Ccp * B * rec, f"size {raw.size} != {Ccp*B*rec}"
    ok = True
    for bi, (init, traj) in enumerate(cfg["pairs"]):
        cpu = walk_df(tuple(init), tuple(traj), cfg["nSteps"], cps)
        for ci, cp in enumerate(cps):
            off = (ci * B + bi) * rec
            mat = raw[off:off + E * 8].view(np.float32).reshape(E, 2)
            expr = raw[off + E * 8:off + E * 8 + E * 4].view(np.int32)
            vals = (mat[:, 0].astype(float) + mat[:, 1].astype(float)) \
                * np.exp2(expr.astype(float))
            g_c1 = vals[1 * m + 0] / vals[0]
            g_c2 = vals[2 * m + 0] / vals[0]
            c_c1, c_c2, _ = cpu[cp]
            e1 = abs(g_c1 - c_c1) / max(abs(c_c1), 1e-30)
            e2 = abs(g_c2 - c_c2) / max(abs(c_c2), 1e-30)
            good = e1 < 1e-9 and e2 < 1e-9
            ok &= good
            print(f"[{bi}] cp={cp} c1={g_c1:.13g} rel_err=({e1:.2e},{e2:.2e}) "
                  f"{'OK' if good else 'MISMATCH'}", flush=True)
    print("GPU==CPU" if ok else "GPU DIVERGES", flush=True)
    return ok


def run_pair(initial, traj, depth, checkpoints, validate):
    t0 = time.time()
    res = walk_df(initial, traj, depth, checkpoints)
    rec = {"initial": list(initial), "traj": list(traj), "depth": depth,
           "checkpoints": {}}
    for cp in sorted(res):
        if res[cp] is None:
            rec["checkpoints"][cp] = {"status": "degenerate"}
            continue
        c1, c2, log2g = res[cp]
        entry = {"status": "ok", "c1": c1, "c2": c2, "log2_growth": log2g}
        if validate:
            tm = C.make_tm(tuple(traj), initial=tuple(initial))
            ec1, ec2 = C.normalized(C.walk_states(tm, [cp])[0])
            entry["err_c1"] = abs(c1 - float(ec1))
            entry["err_c2"] = abs(c2 - float(ec2))
        rec["checkpoints"][cp] = entry
        msg = f"  [cp {cp}] c1={c1:.13g} c2={c2:.13g} log2G={log2g:.1f}"
        if validate:
            msg += f"  err=({entry['err_c1']:.2e}, {entry['err_c2']:.2e})"
        print(msg, flush=True)
    rec["time"] = time.time() - t0
    print(f"  ({rec['time']:.2f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="JSON file: [[initial, traj], ...]")
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", metavar="DIR",
                    help="write Metal host tensors instead of walking")
    ap.add_argument("--check-gpu", nargs=2, metavar=("DIR", "OUT_BIN"),
                    help="verify GPU output against the CPU walk")
    args = ap.parse_args()

    if args.check_gpu:
        sys.exit(0 if check_gpu(*args.check_gpu) else 1)

    if args.pairs:
        with open(args.pairs) as f:
            pairs = [(tuple(p[0]), tuple(p[1])) for p in json.load(f)]
    else:
        pairs = [(DEFAULT_INITIAL, BASELINE)]
    cps = ([int(x) for x in args.checkpoints.split(",") if x]
           if args.checkpoints else [args.depth])

    if args.export:
        export_metal(pairs, args.depth, cps, args.export)
        return

    results = []
    for initial, traj in pairs:
        print(f"[pair] init={initial} traj={traj}", flush=True)
        results.append(run_pair(initial, traj, args.depth, cps, args.validate))
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
