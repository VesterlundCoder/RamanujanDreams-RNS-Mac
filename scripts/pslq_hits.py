#!/usr/bin/env python3
"""Full PSLQ verification of library hits from match_limits.py.

CORRECTED-semantics edition. For every unique limit value:
  1. re-walk the trajectory with the corrected high-precision CMF walk
     (lib/cmf_walk_corrected.py, mpmath backend) at depth N and 2N,
  2. compute L for the SPECIFIC matched (i,j) pair (column rank-1),
  3. estimate convergence from |L_N - L_2N|,
  4. cross-check against the GPU float64 match (gate 1e-8),
  5. run the full PSLQ battery (pslq_companion.pslq_identify) on L_2N.

NOTE: cmf_generic.py (direct companion product) is NOT Ramanujan Dreams
CMF trajectory semantics and is no longer used in this production path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
PSLQ_PKG = "/Users/davidvesterlund/freedomcode/freedomcode/stringtheory/pslq_run/pslq_batch_package"
sys.path.insert(0, LIB if os.path.isdir(LIB) else PSLQ_PKG)

_CFG = {}


def init_worker(dps, n_depth, maxcoeff, maxsteps):
    import mpmath as mp
    mp.mp.dps = dps
    _CFG.update(dps=dps, N=n_depth, maxcoeff=maxcoeff, maxsteps=maxsteps)


def process_hit(rec: dict) -> dict:
    import mpmath as mp
    import pslq_companion as pc
    from cmf_walk_corrected import (apply_trajectory_step,
                                    rank_for_z, start_pos)

    shift = rec["shift"]
    dirv = rec["dir"]
    zn, zd = rec["z_num"], rec["z_den"]
    N = _CFG["N"]
    dps = _CFG["dps"]

    try:
        mp.mp.dps = dps
        z = mp.mpf(zn) / mp.mpf(zd)
        zero, one = mp.mpf(0), mp.mpf(1)
        rank = rank_for_z(Fraction(zn, zd))
        i, j, last = rec["i"], rec["j"], rank - 1
        out = dict(rec)

        # corrected high-precision walk, snapshot at N, continue to 2N
        pos = start_pos(shift)
        W = [[one if r == c else zero for c in range(rank)]
             for r in range(rank)]
        LN = None
        for n in range(1, 2 * N + 1):
            apply_trajectory_step(W, pos, dirv, z, rank, zero, one)
            # renormalize to keep magnitudes sane
            mx = max(abs(v) for row in W for v in row)
            if mx > 0:
                W = [[v / mx for v in row] for row in W]
            if n == N:
                if W[j][last] == 0:
                    out["status"] = "zero_denominator"
                    return out
                LN = W[i][last] / W[j][last]

        if W[j][last] == 0:
            out["status"] = "zero_denominator"
            return out
        L = W[i][last] / W[j][last]

        # convergence estimate from the two depths
        diff = abs(L - LN)
        out["conv_2N"] = float(mp.nstr(diff, 8)) if diff > 0 else 0.0
        out["L_exact_40d"] = mp.nstr(L, 40)

        # cross-check the GPU float64 match against the corrected limit
        rel = abs(float(L) - rec["L_float"]) / max(abs(float(L)), 1e-300)
        out["float_vs_exact_rel"] = rel
        if rel > 1e-8:
            out["status"] = "float_mismatch"
            return out
        if diff > mp.mpf(10) ** (-20):
            out["status"] = "not_converged"
            return out

        # reject rational limits (denominator <= 1e6 at high precision)
        fr = Fraction(float(L)).limit_denominator(1_000_000)
        if abs(L - mp.mpf(fr.numerator) / fr.denominator) < mp.mpf(10) ** (-30):
            out["status"] = "rational_limit"
            out["L_rational"] = f"{fr.numerator}/{fr.denominator}"
            return out

        out["pslq"] = pc.pslq_identify(L, maxcoeff=_CFG["maxcoeff"],
                                       maxsteps=_CFG["maxsteps"])
        out["status"] = "pslq_hit" if out["pslq"] else "pslq_no_relation"
        return out
    except Exception as e:  # noqa: BLE001
        return {**rec, "status": "error", "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hits_jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dps", type=int, default=120)
    ap.add_argument("--depth", type=int, default=500,
                    help="N for the double-depth walk (goes to 2N=1000)")
    ap.add_argument("--maxcoeff", type=int, default=100000)
    ap.add_argument("--maxsteps", type=int, default=20000)
    args = ap.parse_args()

    out_path = args.out or args.hits_jsonl.replace(".hits.jsonl", ".pslq.jsonl")
    if out_path == args.hits_jsonl:
        out_path = args.hits_jsonl + ".pslq.jsonl"

    # dedupe: one PSLQ per unique limit VALUE (12 significant digits).
    # Many trajectories share a limit; verify once, keep multiplicity.
    by_val: dict = {}
    with open(args.hits_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            key = (f"{rec['L_float']:.12g}", rec["z_num"], rec["z_den"])
            if key in by_val:
                by_val[key]["n_occurrences"] += 1
            else:
                rec["n_occurrences"] = 1
                by_val[key] = rec
    hits = list(by_val.values())

    print(f"{len(hits)} unique hits to verify (from {args.hits_jsonl})", flush=True)
    if not hits:
        open(out_path, "w").close()
        return

    t0 = time.time()
    n_confirmed = 0
    with open(out_path, "w") as out, Pool(
            args.workers, initializer=init_worker,
            initargs=(args.dps, args.depth, args.maxcoeff, args.maxsteps)) as pool:
        for k, res in enumerate(pool.imap_unordered(process_hit, hits), 1):
            out.write(json.dumps(res) + "\n")
            out.flush()
            if res.get("status") == "pslq_hit":
                n_confirmed += 1
                rel = res["pslq"][0] if res["pslq"] else {}
                print(f"  PSLQ HIT dir={res['dir']} pair=({res['i']},{res['j']}) "
                      f"lib={res['lib_name']} first_rel={rel}", flush=True)
            if k % 20 == 0 or k == len(hits):
                el = time.time() - t0
                print(f"  {k}/{len(hits)} verified ({k/el:.2f}/s), "
                      f"{n_confirmed} PSLQ-confirmed", flush=True)

    print(f"DONE: {n_confirmed}/{len(hits)} PSLQ-confirmed -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
