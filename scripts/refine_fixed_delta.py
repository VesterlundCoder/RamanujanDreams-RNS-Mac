#!/usr/bin/env python3
"""Staged exact delta refinement for fixed-start GPU survivors.

The GPU stage only produces a convergence/height proxy. This script restores
exact arithmetic with ``fractions.Fraction`` and computes

    delta_N = -1 - log(|target - p/q|) / log(q)

from the reduced exact rational approximant p/q.

Exact depth is itself a funnel: by default 500 candidates are evaluated at
N=256, the best 100 continue to N=512, and only the best 25 continue to
N=1000. This avoids doing the most expensive bigint work on candidates that
are already uncompetitive at smaller exact depth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import mpmath as mp
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from cmf_walk_absolute import (  # noqa: E402
    exact_delta_from_fraction,
    projected_ratio,
    walk_projected_absolute,
)

_CFG = {}


def eval_target(expr: str, dps: int):
    return mp.mpf(str(sp.N(sp.sympify(expr), dps)))


def init_worker(profile, target_expr, dps, depth):
    mp.mp.dps = dps
    _CFG.clear()
    _CFG["profile"] = profile
    _CFG["target_expr"] = target_expr
    _CFG["dps"] = dps
    _CFG["depth"] = depth
    _CFG["target"] = eval_target(target_expr, dps)


def process_candidate(rec):
    p = _CFG["profile"]
    target = _CFG["target"]
    dps = _CFG["dps"]
    depth = _CFG["depth"]
    i, j = p["ratio_pair"]
    direction = rec["direction"]
    out = dict(rec)
    seq = list(out.get("delta_by_depth", []))

    try:
        _, rank, status, _, snaps = walk_projected_absolute(
            p["absolute_start"], direction, p["z_num"], p["z_den"],
            depth, i, j, field="fraction", snapshot_depths=(depth,),
        )
        out["exact_status"] = int(status)
        if status != 0:
            out["status"] = "exact_singular"
            return out

        approx = projected_ratio(snaps[depth], rank)
        delta = exact_delta_from_fraction(approx, target, dps=dps)
        approx_mp = mp.mpf(approx.numerator) / mp.mpf(approx.denominator)
        err = abs(target - approx_mp)
        entry = {
            "N": depth,
            "delta": None if delta is None else mp.nstr(delta, 20),
            "error": mp.nstr(err, 20),
            "denominator_digits": len(str(abs(approx.denominator))),
            "denominator_bits": int(abs(approx.denominator).bit_length()),
            "numerator_bits": int(abs(approx.numerator).bit_length()),
            "approximant_40d": mp.nstr(approx_mp, 40),
        }
        seq = [x for x in seq if int(x["N"]) != depth]
        seq.append(entry)
        seq.sort(key=lambda x: int(x["N"]))

        finite = [mp.mpf(x["delta"]) for x in seq if x["delta"] is not None]
        out["delta_by_depth"] = seq
        out["delta_exact"] = float(finite[-1]) if finite else None
        out["delta_max_observed"] = float(max(finite)) if finite else None
        if len(finite) >= 2:
            out["delta_tail_min"] = float(min(finite[-2:]))
            out["delta_slope_last"] = float(finite[-1] - finite[-2])
        else:
            out["delta_tail_min"] = out["delta_exact"]
            out["delta_slope_last"] = None
        out["positive_at_deepest"] = bool(finite and finite[-1] > 0)
        out["positive_tail"] = bool(len(finite) >= 2 and all(x > 0 for x in finite[-2:]))
        out["status"] = "exact_ok"
        return out
    except Exception as exc:  # noqa: BLE001
        out["status"] = "exact_error"
        out["error_message"] = str(exc)[:300]
        return out


def default_keeps(n0: int, nstage: int):
    keeps = [n0]
    while len(keeps) < nstage:
        keeps.append(max(25, math.ceil(keeps[-1] * 0.20)))
    # Never ask a later stage to increase the population.
    for i in range(1, len(keeps)):
        keeps[i] = min(keeps[i], keeps[i - 1])
    return keeps


def sort_exact(records):
    return sorted(records, key=lambda r: (
        r.get("delta_exact") is not None,
        r.get("delta_exact") if r.get("delta_exact") is not None else -math.inf,
        r.get("gpu_score", -math.inf),
    ), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates_jsonl")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-candidates", type=int, default=500)
    ap.add_argument("--depths", default="256,512,1000")
    ap.add_argument("--stage-keep", default=None,
                    help="survivors after each exact depth; default is ~20%% per stage")
    ap.add_argument("--dps", type=int, default=160)
    ap.add_argument("--target-expr", default=None,
                    help="override profile target expression")
    args = ap.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)
    target_expr = args.target_expr or profile.get("target_expr")
    if not target_expr:
        raise SystemExit(
            "exact delta requires a symbolic target expression; pass --target-expr "
            "or create the profile with identify_fixed_start.py --target-expr"
        )

    depths = sorted(set(int(x) for x in args.depths.split(",")))
    if not depths or depths[0] <= 0:
        raise SystemExit("--depths must contain positive integers")

    candidates = []
    with open(args.candidates_jsonl) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    candidates.sort(key=lambda r: r.get("gpu_score", -math.inf), reverse=True)
    candidates = candidates[:args.max_candidates]
    if not candidates:
        raise SystemExit("no candidates to refine")

    if args.stage_keep:
        keeps = [int(x) for x in args.stage_keep.split(",")]
        if len(keeps) != len(depths) or any(x <= 0 for x in keeps):
            raise SystemExit("--stage-keep must match --depths and contain positive integers")
    else:
        keeps = default_keeps(len(candidates), len(depths))
    keeps[0] = min(keeps[0], len(candidates))
    for i in range(1, len(keeps)):
        keeps[i] = min(keeps[i], keeps[i - 1])

    out_path = args.out or args.candidates_jsonl.replace(".jsonl", ".exact_delta.jsonl")
    if out_path == args.candidates_jsonl:
        out_path += ".exact_delta.jsonl"

    print(f"exact staged refinement: initial={len(candidates)}, depths={depths}, "
          f"keeps={keeps}, workers={args.workers}, target={target_expr}", flush=True)

    t0 = time.time()
    current = candidates
    stage_summary = []
    for stage_idx, (depth, keep) in enumerate(zip(depths, keeps), 1):
        print(f"[exact] stage {stage_idx}/{len(depths)} N={depth}: "
              f"evaluating {len(current)} candidates", flush=True)
        results = []
        s0 = time.time()
        with Pool(args.workers, initializer=init_worker,
                  initargs=(profile, target_expr, args.dps, depth)) as pool:
            for k, res in enumerate(pool.imap_unordered(process_candidate, current), 1):
                results.append(res)
                if k % 10 == 0 or k == len(current):
                    ok = sum(r.get("status") == "exact_ok" for r in results)
                    pos = sum(r.get("positive_at_deepest", False) for r in results)
                    el = time.time() - s0
                    print(f"  {k}/{len(current)} ({k/max(el,1e-9):.2f}/s) "
                          f"exact_ok={ok} positive={pos}", flush=True)

        stage_file = out_path + f".N{depth}.jsonl"
        ranked_all = sort_exact(results)
        with open(stage_file, "w") as f:
            for r in ranked_all:
                f.write(json.dumps(r) + "\n")

        viable = [r for r in ranked_all if r.get("status") == "exact_ok"]
        if not viable:
            raise RuntimeError(f"exact stage N={depth} produced no viable candidates")
        current = viable[:min(keep, len(viable))]
        stage_summary.append({
            "N": depth,
            "evaluated": len(results),
            "viable": len(viable),
            "retained": len(current),
            "positive": sum(r.get("positive_at_deepest", False) for r in viable),
            "best_delta": current[0].get("delta_exact"),
            "stage_file": stage_file,
        })
        print(f"[exact] N={depth}: retained {len(current)}/{len(viable)}, "
              f"best delta={current[0].get('delta_exact')}", flush=True)

    current = sorted(current, key=lambda r: (
        bool(r.get("positive_tail", False)),
        r.get("delta_exact") if r.get("delta_exact") is not None else -math.inf,
    ), reverse=True)

    with open(out_path, "w") as out:
        for r in current:
            out.write(json.dumps(r) + "\n")

    summary_path = out_path + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "target_expr": target_expr,
            "depths": depths,
            "keeps": keeps,
            "workers": args.workers,
            "wall_seconds": time.time() - t0,
            "stages": stage_summary,
            "final_survivors": len(current),
            "positive_at_deepest": sum(r.get("positive_at_deepest", False) for r in current),
            "positive_tail": sum(r.get("positive_tail", False) for r in current),
        }, f, indent=2)

    npos = sum(r.get("positive_at_deepest", False) for r in current)
    ntail = sum(r.get("positive_tail", False) for r in current)
    print(f"DONE: final={len(current)}, positive@deepest={npos}, "
          f"positive-tail={ntail} -> {out_path}")
    if current:
        best = current[0]
        print(f"best dir={best.get('direction')} delta={best.get('delta_exact')} "
              f"positive_tail={best.get('positive_tail')}")


if __name__ == "__main__":
    main()
