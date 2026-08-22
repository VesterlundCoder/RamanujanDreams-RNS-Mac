#!/usr/bin/env python3
"""Identify the limit attached to one absolute fixed start point.

This is phase A of the fixed-start architecture. It is intentionally run
once per start point, not once per trajectory. A reference direction is
walked at N and 2N with high precision, the limit is sent through the
existing PSLQ battery, and a reusable start profile is written to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mpmath as mp
import sympy as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))

from cmf_walk_absolute import projected_ratio, walk_projected_absolute  # noqa: E402
import pslq_companion as pc  # noqa: E402

NSHIFT = 11


def parse_vec(text: str):
    v = [int(x.strip()) for x in text.split(",")]
    if len(v) != NSHIFT:
        raise argparse.ArgumentTypeError("expected exactly 11 comma-separated integers")
    return v


def parse_pair(text: str):
    p = [int(x.strip()) for x in text.split(",")]
    if len(p) != 2:
        raise argparse.ArgumentTypeError("pair must be i,j")
    return tuple(p)


def eval_target(expr: str, dps: int):
    value = sp.N(sp.sympify(expr), dps)
    return mp.mpf(str(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, type=parse_vec,
                    help="ABSOLUTE CMF start vector")
    ap.add_argument("--dir", required=True, type=parse_vec,
                    help="one admissible reference direction")
    ap.add_argument("--pair", required=True, type=parse_pair,
                    help="projected matrix row pair i,j")
    ap.add_argument("--z-num", type=int, default=1)
    ap.add_argument("--z-den", type=int, default=2)
    ap.add_argument("--depth", type=int, default=1000,
                    help="first depth N; also evaluates 2N")
    ap.add_argument("--dps", type=int, default=120)
    ap.add_argument("--maxcoeff", type=int, default=100000)
    ap.add_argument("--maxsteps", type=int, default=20000)
    ap.add_argument("--target-expr", default=None,
                    help="optional known expression, e.g. '-9*zeta(3)/pi**2'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.z_den == 0 or args.depth <= 0:
        raise SystemExit("z-den must be nonzero and depth positive")

    mp.mp.dps = args.dps
    i, j = args.pair
    N = args.depth

    rows, rank, st, final_pos, snaps = walk_projected_absolute(
        args.start, args.dir, args.z_num, args.z_den, 2 * N,
        i, j, field="mpf", dps=args.dps, snapshot_depths=(N, 2 * N),
    )
    if st != 0:
        raise SystemExit(f"reference direction is singular/invalid: status={st}")

    LN = projected_ratio(snaps[N], rank)
    L2 = projected_ratio(snaps[2 * N], rank)
    conv = abs(L2 - LN)

    pslq = pc.pslq_identify(L2, maxcoeff=args.maxcoeff, maxsteps=args.maxsteps)
    target_expr = args.target_expr
    if target_expr:
        target = eval_target(target_expr, args.dps)
        target_err = abs(L2 - target)
        target_numeric = mp.nstr(target, args.dps)
    else:
        target_err = None
        # The numerical 2N value is sufficient as a screening target. For
        # exact delta work, provide the PSLQ-confirmed expression later.
        target_numeric = mp.nstr(L2, args.dps)

    profile = {
        "format": "FIXED_START_PROFILE_V1",
        "absolute_start": args.start,
        "reference_direction": args.dir,
        "z_num": args.z_num,
        "z_den": args.z_den,
        "rank": rank,
        "ratio_pair": [i, j],
        "depth_N": N,
        "dps": args.dps,
        "L_N": mp.nstr(LN, args.dps),
        "L_2N": mp.nstr(L2, args.dps),
        "convergence_abs": mp.nstr(conv, 30),
        "target_expr": target_expr,
        "target_numeric": target_numeric,
        "target_error_at_2N": None if target_err is None else mp.nstr(target_err, 30),
        "pslq": pslq,
        "final_position_2N": final_pos,
        "semantics": "absolute v0; first trajectory operator evaluated at n=0",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"rank={rank} pair=({i},{j})")
    print(f"L_N  = {mp.nstr(LN, 50)}")
    print(f"L_2N = {mp.nstr(L2, 50)}")
    print(f"|L_2N-L_N| = {mp.nstr(conv, 12)}")
    if target_expr:
        print(f"target={target_expr}  error={mp.nstr(target_err, 12)}")
    if pslq:
        print(f"PSLQ candidates: {len(pslq)}; first={pslq[0]}")
    else:
        print("PSLQ: no relation in current battery")
    print(f"profile -> {args.out}")


if __name__ == "__main__":
    main()
