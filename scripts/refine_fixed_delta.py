#!/usr/bin/env python3
"""Exact/high-precision delta refinement for fixed-start GPU survivors.

The GPU stage only produces a convergence/height proxy. This script
re-walks the best candidates with exact Fraction arithmetic, extracts
the exact reduced rational approximant p/q at several depths, and
computes

    delta_N = -1 - log(|target - p/q|) / log(q).

Only this CPU stage reports ``delta_exact``. Multiple depths are retained
so a positive value can be distinguished from evidence toward positive
limsup-delta.
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


def init_worker(profile, depths, target_expr, dps):
    mp.mp.dps = dps
    _CFG["profile"] = profile
    _CFG["depths"] = depths
    _CFG["target_expr"] = target_expr
    _CFG["dps"] = dps
    _CFG["target"] = eval_target(target_expr, dps)


def process_candidate(rec):
    p = _CFG["profile"]
    depths = _CFG["depths"]
    target = _CFG["target"]
    dps = _CFG["dps"]
    i, j = p["ratio_pair"]
    direction = rec["direction"]

    out = dict(rec)
    try:
        _, rank, status, _, snaps = walk_projected_absolute(
            p["absolute_start"], direction, p["z_num"], p["z_den"],
            max(depths), i, j, field="fraction", snapshot_depths=depths,
        )
        out["exact_status"] = int(status)
        if status != 0:
            out["status"] = "exact_singular"
            return out

        seq = []
        for depth in depths:
            approx = projected_ratio(snaps[depth], rank)
            delta = exact_delta_from_fraction(approx, target, dps=dps)
            approx_mp = mp.mpf(approx.numerator) / mp.mpf(approx.denominator)
            err = abs(target - approx_mp)
            seq.append({
                "N": depth,
                "delta": None if delta is None else mp.nstr(delta, 20),
                "error": mp.nstr(err, 20),
                "denominator_digits": len(str(abs(approx.denominator))),
                "denominator_bits": int(abs(approx.denominator).bit_length()),
                "numerator_bits": int(abs(approx.numerator).bit_length()),
                "approximant_40d": mp.nstr(approx_mp, 40),
            })

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates_jsonl")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-candidates", type=int, default=500)
    ap.add_argument("--depths", default="256,512,1000")
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

    out_path = args.out or args.candidates_jsonl.replace(".jsonl", ".exact_delta.jsonl")
    if out_path == args.candidates_jsonl:
        out_path += ".exact_delta.jsonl"

    print(f"exact refinement: {len(candidates)} candidates, depths={depths}, "
          f"workers={args.workers}, target={target_expr}", flush=True)
    t0 = time.time()
    results = []
    with Pool(args.workers, initializer=init_worker,
              initargs=(profile, depths, target_expr, args.dps)) as pool:
        for k, res in enumerate(pool.imap_unordered(process_candidate, candidates), 1):
            results.append(res)
            if k % 10 == 0 or k == len(candidates):
                pos = sum(r.get("positive_at_deepest", False) for r in results)
                el = time.time() - t0
                print(f"  {k}/{len(candidates)} ({k/max(el,1e-9):.2f}/s), "
                      f"positive deepest={pos}", flush=True)

    # Most useful records first. Positive tail dominates; then deepest delta.
    results.sort(key=lambda r: (
        bool(r.get("positive_tail", False)),
        r.get("delta_exact") if r.get("delta_exact") is not None else -math.inf,
    ), reverse=True)

    with open(out_path, "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    npos = sum(r.get("positive_at_deepest", False) for r in results)
    ntail = sum(r.get("positive_tail", False) for r in results)
    print(f"DONE: positive@deepest={npos}, positive-tail={ntail} -> {out_path}")
    if results:
        best = results[0]
        print(f"best dir={best.get('direction')} delta={best.get('delta_exact')} "
              f"positive_tail={best.get('positive_tail')}")


if __name__ == "__main__":
    main()
