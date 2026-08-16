"""
pslq_companion.py
=================
Stage-2/3 for COMPANION-gauge survivors (the validated true-delta family).

For each survivor:
  1. EXACT arithmetic delta via pure-integer double-depth walk (cmf_generic).
  2. If delta > --min-delta: compute the limit L to high precision from the
     SAME exact-integer convergent (best row pair), then run EXTENDED PSLQ
     against a library of constants to identify closed forms.

PSLQ battery per limit L (mp.dps set high, maxcoeff bounded):
  - rational            : pslq([1, L])
  - single constant     : pslq([1, L, c])          for each c in the library
  - quadratic (alg deg2): pslq([1, L, L^2])
  - pi-polynomial       : pslq([1, L, pi, pi^2, pi^3])
  - zeta/pi mix         : pslq([1, L, pi^3, zeta3]) and ([1, L, pi^2, zeta3])

Reports delta stats + PSLQ hits every --report candidates; writes a results
jsonl and a summary json.

Usage:
  python3 pslq_companion.py --in 4f3_candidates/survivors_4f3_companion.jsonl \
      --dim 4 --out 4f3_candidates/pslq_4f3.jsonl --workers 8 \
      --min-delta 0.02 --N 90 --dps 160 --maxcoeff 100000 --report 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmf_generic as cg


# ── exact-integer convergent: delta + high-precision limit ──────────────────
def delta_and_limit(shift, dirv, zn, zd, dim, N):
    """Pure-integer double-depth walk. Returns (best_delta, L_mpf, pair)."""
    P = [[1 if i == j else 0 for j in range(dim)] for i in range(dim)]
    snapN = None
    for n in range(1, 2 * N + 1):
        P = cg.matmul_int(P, cg.build_M_int(n, shift, dirv, zn, zd, dim), dim)
        if n == N:
            snapN = [row[:] for row in P]
    last = dim - 1
    best_d = None
    best_L = None
    best_pair = None
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
            log_err = cg.log_bigint(cross) - cg.log_bigint(qn) - cg.log_bigint(q2)
            log_q = cg.log_bigint(qn)
            if log_q == 0:
                continue
            d = -(1.0 + log_err / log_q)
            if best_d is None or d > best_d:
                best_d = d
                best_L = (mp.mpf(p2) / mp.mpf(q2))     # deepest convergent (2N)
                best_pair = (i, j)
    return best_d, best_L, best_pair


# ── constant library (names -> value builder at current mp precision) ───────
# Cached per working precision (mp.mp.dps) so the ~40 constants — including the
# expensive gamma/zeta evaluations — are computed ONCE per worker process rather
# than rebuilt for every survivor's PSLQ battery.
_CONST_CACHE = {}


def const_library():
    key = mp.mp.dps
    cached = _CONST_CACHE.get(key)
    if cached is not None:
        return cached
    sq = mp.sqrt
    g13 = mp.gamma(mp.mpf(1) / 3)
    g14 = mp.gamma(mp.mpf(1) / 4)
    C = {
        # pi powers
        "pi": mp.pi,
        "pi^2": mp.pi ** 2,
        "pi^3": mp.pi ** 3,
        "pi^4": mp.pi ** 4,
        "pi^5": mp.pi ** 5,
        "pi^6": mp.pi ** 6,
        "pi^7": mp.pi ** 7,
        # odd zeta values (even ones are rational * pi^2k, already covered)
        "zeta3": mp.zeta(3),
        "zeta5": mp.zeta(5),
        "zeta7": mp.zeta(7),
        "zeta9": mp.zeta(9),
        "zeta3^2": mp.zeta(3) ** 2,
        # catalan / dirichlet-beta-like
        "catalan": mp.catalan,
        # logarithms of small primes
        "log2": mp.log(2),
        "log3": mp.log(3),
        "log5": mp.log(5),
        "log7": mp.log(7),
        # other transcendentals
        "e": mp.e,
        "euler_gamma": mp.euler,
        # gamma / period constants (lemniscatic & equianharmonic)
        "gamma_1_3": g13,
        "gamma_1_4": g14,
        "gamma_1_3^3": g13 ** 3,
        "gamma_1_4^2": g14 ** 2,
        "gamma_1_4^4": g14 ** 4,
        # algebraic
        "phi": (1 + sq(5)) / 2,
        "sqrt2": sq(2), "sqrt3": sq(3), "sqrt5": sq(5),
        "sqrt6": sq(6), "sqrt7": sq(7), "sqrt10": sq(10),
        "sqrt11": sq(11), "sqrt13": sq(13),
        # common products / ratios
        "pi*sqrt3": mp.pi * sq(3),
        "zeta3/pi^3": mp.zeta(3) / mp.pi ** 3,
        "zeta3/pi^2": mp.zeta(3) / mp.pi ** 2,
        "G/pi": mp.catalan / mp.pi,
    }
    _CONST_CACHE[key] = C
    return C


def pslq_identify(L, maxcoeff, maxsteps):
    """Run the PSLQ battery. Returns list of hit dicts (relation found)."""
    hits = []
    one = mp.mpf(1)

    def rel(vec, names):
        try:
            r = mp.pslq(vec, maxcoeff=maxcoeff, maxsteps=maxsteps)
        except Exception:
            return None
        if not r or all(c == 0 for c in r):
            return None
        # require L to actually participate (index 1 by convention)
        if len(r) > 1 and r[1] == 0 and names[1] == "L":
            return None
        return {"names": names, "coeffs": [int(c) for c in r]}

    C = const_library()
    # rational
    h = rel([one, L], ["1", "L"])
    if h:
        hits.append({"type": "rational", **h})
    # quadratic (algebraic degree 2)
    h = rel([one, L, L * L], ["1", "L", "L^2"])
    if h:
        hits.append({"type": "quadratic", **h})
    # cubic (algebraic degree 3)
    h = rel([one, L, L * L, L * L * L], ["1", "L", "L^2", "L^3"])
    if h:
        hits.append({"type": "cubic", **h})
    # single constants
    for name, c in C.items():
        h = rel([one, L, c], ["1", "L", name])
        if h:
            hits.append({"type": "linear_const", **h})
    # pi-polynomials
    h = rel([one, L, C["pi"], C["pi^2"], C["pi^3"]],
            ["1", "L", "pi", "pi^2", "pi^3"])
    if h:
        hits.append({"type": "pi_poly", **h})
    h = rel([one, L, C["pi"], C["pi^2"], C["pi^3"], C["pi^4"]],
            ["1", "L", "pi", "pi^2", "pi^3", "pi^4"])
    if h:
        hits.append({"type": "pi_poly4", **h})
    # odd-zeta ladder
    h = rel([one, L, C["zeta3"], C["zeta5"], C["zeta7"]],
            ["1", "L", "zeta3", "zeta5", "zeta7"])
    if h:
        hits.append({"type": "zeta_ladder", **h})
    # zeta/pi and gamma-period mixes
    for mix, nm in [([one, L, C["pi^3"], C["zeta3"]], ["1", "L", "pi^3", "zeta3"]),
                    ([one, L, C["pi^2"], C["zeta3"]], ["1", "L", "pi^2", "zeta3"]),
                    ([one, L, C["pi^5"], C["zeta5"]], ["1", "L", "pi^5", "zeta5"]),
                    ([one, L, C["pi^4"], C["zeta5"]], ["1", "L", "pi^4", "zeta5"]),
                    ([one, L, C["pi^7"], C["zeta7"]], ["1", "L", "pi^7", "zeta7"]),
                    ([one, L, C["pi^2"], C["zeta5"]], ["1", "L", "pi^2", "zeta5"]),
                    ([one, L, C["zeta3"], C["zeta5"]], ["1", "L", "zeta3", "zeta5"]),
                    ([one, L, C["pi^2"], C["catalan"]], ["1", "L", "pi^2", "catalan"]),
                    ([one, L, C["log2"], C["pi^2"]], ["1", "L", "log2", "pi^2"]),
                    ([one, L, C["gamma_1_3^3"], C["pi"]], ["1", "L", "gamma_1_3^3", "pi"]),
                    ([one, L, C["gamma_1_4^2"], C["pi"]], ["1", "L", "gamma_1_4^2", "pi"]),
                    ([one, L, C["gamma_1_4^4"], C["pi"]], ["1", "L", "gamma_1_4^4", "pi"])]:
        h = rel(mix, nm)
        if h:
            hits.append({"type": "mix", **h})
    return hits


def _one(args):
    rec, dim, N, dps, min_delta, maxcoeff, maxsteps = args
    mp.mp.dps = dps
    out = {"gid": rec.get("gid"), "r2": rec.get("r2"),
           "shift": rec["shift"], "dir": rec["dir"],
           "z_num": rec["z_num"], "z_den": rec["z_den"]}
    try:
        d, L, pair = delta_and_limit(rec["shift"], rec["dir"],
                                     rec["z_num"], rec["z_den"], dim, N)
        out["delta_exact"] = d
        if d is not None and d >= min_delta and L is not None:
            out["L"] = mp.nstr(L, 40)
            out["pslq"] = pslq_identify(L, maxcoeff, maxsteps)
        else:
            out["pslq"] = []
    except Exception as e:
        out["delta_exact"] = None
        out["pslq"] = []
        out["error"] = str(e)[:140]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--N", type=int, default=90, help="verify depth (2N walk)")
    ap.add_argument("--dps", type=int, default=160)
    ap.add_argument("--min-delta", type=float, default=0.02)
    ap.add_argument("--maxcoeff", type=int, default=100000)
    ap.add_argument("--maxsteps", type=int, default=20000)
    ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="cap #records (0=all)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    summary = args.summary or (os.path.splitext(args.out)[0] + "_summary.json")
    print(f"[pslq] {len(recs):,} companion survivors  dim={args.dim} "
          f"N={args.N} dps={args.dps} maxcoeff={args.maxcoeff} workers={args.workers}",
          flush=True)

    task = ((r, args.dim, args.N, args.dps, args.min_delta,
             args.maxcoeff, args.maxsteps) for r in recs)
    t0 = time.time()
    n_pos = 0
    n_hit = 0
    hit_types = {}
    notable = []
    outf = open(args.out, "w")
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_one, task, chunksize=8), 1):
            outf.write(json.dumps(res) + "\n")
            de = res.get("delta_exact")
            if de is not None and de >= args.min_delta:
                n_pos += 1
            for h in res.get("pslq", []):
                n_hit += 1
                hit_types[h["type"]] = hit_types.get(h["type"], 0) + 1
                if h["type"] not in ("rational",):
                    notable.append({"gid": res.get("gid"), "delta": de,
                                    "L": res.get("L"), **h})
            if i % args.report == 0:
                outf.flush()
                el = time.time() - t0
                print(f"  [{i:,}/{len(recs):,}] verified+={n_pos:,} pslq_hits={n_hit:,} "
                      f"types={hit_types}  {i/el:.1f} rec/s", flush=True)
    outf.close()
    el = time.time() - t0
    summ = {"input": args.inp, "dim": args.dim, "n": len(recs),
            "n_delta_positive": n_pos, "n_pslq_hits": n_hit,
            "hit_types": hit_types, "elapsed_s": el,
            "notable_hits": notable[:200]}
    with open(summary, "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[pslq] DONE {len(recs):,} in {el:.1f}s  verified_positive={n_pos:,} "
          f"pslq_hits={n_hit:,} types={hit_types}")
    print(f"  results -> {args.out}\n  summary -> {summary}")
    if notable:
        print(f"  notable (non-rational) hits: {len(notable)} (see summary)")


if __name__ == "__main__":
    main()
