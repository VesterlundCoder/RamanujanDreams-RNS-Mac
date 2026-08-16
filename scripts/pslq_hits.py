#!/usr/bin/env python3
"""Full PSLQ verification of library hits from match_limits.py.

For every unique (shift, dir, z) hit: exact integer double-depth walk
(pslq_companion.delta_and_limit, depth 2N) to get the certified delta
and a high-precision limit, then the full PSLQ battery
(pslq_companion.pslq_identify) at the requested dps.

The float64 library match is also cross-checked against the exact limit
of the SPECIFIC matched (i,j) ratio.
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

DIM = 6

_CFG = {}


def init_worker(dps, n_depth, maxcoeff, maxsteps):
    import mpmath as mp
    mp.mp.dps = dps
    _CFG.update(dps=dps, N=n_depth, maxcoeff=maxcoeff, maxsteps=maxsteps)


def process_hit(rec: dict) -> dict:
    import mpmath as mp
    import cmf_generic as cg
    import pslq_companion as pc

    shift = rec["shift"]
    dirv = rec["dir"]
    zn, zd = rec["z_num"], rec["z_den"]
    N = _CFG["N"]

    try:
        # exact integer walk to depth 2N
        P = [[1 if i == j else 0 for j in range(DIM)] for i in range(DIM)]
        snapN = None
        for n in range(1, 2 * N + 1):
            P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, DIM), DIM)
            if n == N:
                snapN = [row[:] for row in P]

        i, j, last = rec["i"], rec["j"], DIM - 1
        pn, qn = snapN[i][last], snapN[j][last]
        p2, q2 = P[i][last], P[j][last]
        out = dict(rec)

        if qn == 0 or q2 == 0:
            out["status"] = "zero_denominator"
            return out

        L = mp.mpf(p2) / mp.mpf(q2)
        cross = pn * q2 - p2 * qn
        if cross == 0:
            # exact rational convergent: identical at both depths
            fr = Fraction(p2, q2)
            out["status"] = "exact_rational"
            out["L_exact"] = f"{fr.numerator}/{fr.denominator}"
            return out

        log_err = cg.log_bigint(cross) - cg.log_bigint(qn) - cg.log_bigint(q2)
        log_q = cg.log_bigint(qn)
        delta = -(1.0 + log_err / log_q) if log_q else None
        out["delta_exact"] = delta
        out["L_exact_40d"] = mp.nstr(L, 40)

        # cross-check the float64 match
        rel = abs(float(L) - rec["L_float"]) / max(abs(float(L)), 1e-300)
        out["float_vs_exact_rel"] = rel
        if rel > 1e-9:
            out["status"] = "float_mismatch"
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
