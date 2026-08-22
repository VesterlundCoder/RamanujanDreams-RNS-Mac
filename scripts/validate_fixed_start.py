#!/usr/bin/env python3
"""Hard validation gate for the absolute fixed-start GPU path.

Checks:
  * N=0 in the exact oracle is the identity product AT the supplied v0
  * N=1 ends at v0+t, proving the first operator was applied from v0
  * projected GPU ratios match exact Fraction arithmetic at small depths
  * a known axis-pole path is flagged

The campaign runner should only be used after this prints
``FIXED-START SEMANTICS VERIFIED`` on the target Mac.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lib"))

from cmf_walk_absolute import projected_ratio, walk_projected_absolute  # noqa: E402

BIN = os.path.join(ROOT, "build", "dreams_fixed_start_df64")
NSHIFT = 11
HDR = 100
REC_DT = np.dtype([
    ("vals", "<f4", (4, 2)),
    ("exp", "<i4", (2,)),
    ("status", "<u4"),
])
TOL = 2e-8

BASE = [2] * 6 + [3] * 5
TESTS = [
    ("numerator +1", BASE, [1,0,0,0,0,0,0,0,0,0,0], -1, 1, (5,4)),
    ("denominator +1", BASE, [0,0,0,0,0,0,1,0,0,0,0], -1, 1, (5,4)),
    ("mixed positive", BASE, [1,0,2,0,0,1,0,1,0,0,1], -1, 1, (5,4)),
    ("nontrivial absolute start", [1,4,2,5,3,6,4,7,5,8,6],
     [1,0,1,0,1,0,1,0,1,0,1], 1, 2, (5,4)),
]
DEPTHS = [1, 2, 3, 5, 10, 20]


def csv(v):
    return ",".join(str(int(x)) for x in v)


def run_gpu(start, direction, zn, zd, pair, N):
    with tempfile.TemporaryDirectory() as td:
        dp = os.path.join(td, "dirs.bin")
        op = os.path.join(td, "out.bin")
        np.asarray(direction, dtype="<i4").tofile(dp)
        r = subprocess.run([
            BIN, dp, op, csv(start), str(pair[0]), str(pair[1]),
            str(zn), str(zd), str(N), "64",
        ], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"fixed-start GPU binary failed: {r.stderr[-1000:]}")
        with open(op, "rb") as f:
            if f.read(8) != b"FSDF6401":
                raise RuntimeError("bad fixed-start output magic")
            maxdim, rank, nshift, nsteps, cp0, cp1, rn, rd = struct.unpack("<8I", f.read(32))
            znum, zden = struct.unpack("<2i", f.read(8))
            start_h = list(struct.unpack("<11i", f.read(44)))
            (ntraj,) = struct.unpack("<Q", f.read(8))
            rec = np.fromfile(f, dtype=REC_DT, count=1)[0]
        assert maxdim == 6 and nshift == 11 and ntraj == 1
        assert start_h == start and (rn, rd) == pair and (znum, zden) == (zn, zd)
        x = rec["vals"][:, 0].astype(np.float64) + rec["vals"][:, 1].astype(np.float64)
        L0 = x[0] / x[1]
        L1 = x[2] / x[3]
        return rank, int(rec["status"]), cp0, cp1, float(L0), float(L1)


def main():
    if not os.path.exists(BIN):
        print(f"FATAL: {BIN} not built")
        return 2

    failures = 0

    # Semantic invariant independent of floating point/GPU.
    d = TESTS[2][2]
    rows0, rank0, st0, pos0, snap0 = walk_projected_absolute(
        BASE, d, -1, 1, 0, 5, 4, field="fraction", snapshot_depths=(0,)
    )
    expected0 = [[0] * rank0 for _ in range(2)]
    expected0[0][5] = 1
    expected0[1][4] = 1
    ok0 = st0 == 0 and pos0 == BASE and rows0 == expected0 and snap0[0] == expected0
    print(f"[FIXED PARITY] N=0 identity at exact v0: {'PASS' if ok0 else 'FAIL'}")
    failures += 0 if ok0 else 1

    rows1, rank1, st1, pos1, _ = walk_projected_absolute(
        BASE, d, -1, 1, 1, 5, 4, field="fraction"
    )
    expected_pos1 = [a + b for a, b in zip(BASE, d)]
    ok1 = st1 == 0 and pos1 == expected_pos1
    print(f"[FIXED PARITY] N=1 advances v0 -> v0+t: {'PASS' if ok1 else 'FAIL'}")
    failures += 0 if ok1 else 1

    for name, start, direction, zn, zd, pair in TESTS:
        for N in DEPTHS:
            rows, rank, st, _, snaps = walk_projected_absolute(
                start, direction, zn, zd, N, pair[0], pair[1],
                field="fraction", snapshot_depths=(max(1, N // 2), N),
            )
            rg, sg, cp0, cp1, L0g, L1g = run_gpu(start, direction, zn, zd, pair, N)
            if st != 0 or sg != 0 or rg != rank:
                print(f"[FIXED PARITY] {name:28s} N={N:3d} FAIL "
                      f"status cpu/gpu={st}/{sg} rank={rank}/{rg}")
                failures += 1
                continue
            L0 = float(projected_ratio(snaps[cp0], rank))
            L1 = float(projected_ratio(snaps[cp1], rank))
            e0 = abs(L0g - L0) / max(abs(L0), 1.0)
            e1 = abs(L1g - L1) / max(abs(L1), 1.0)
            ok = max(e0, e1) < TOL
            print(f"[FIXED PARITY] {name:28s} N={N:3d} "
                  f"{'PASS' if ok else 'FAIL'} relerr={max(e0,e1):.2e}")
            failures += 0 if ok else 1

    # BASE x0=2 with direction -1 reaches x0=1 after step 1; the next
    # negative unit step has denominator x0-1=0 and must be rejected.
    singular = [-1] + [0] * 10
    _, sg, _, _, _, _ = run_gpu(BASE, singular, -1, 1, (5,4), 2)
    oks = sg != 0
    print(f"[FIXED PARITY] known pole path flagged: {'PASS' if oks else 'FAIL'} status={sg}")
    failures += 0 if oks else 1

    print()
    if failures:
        print(f"FATAL: fixed-start parity failure ({failures} failed checks)")
        return 1
    print("FIXED-START SEMANTICS VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
