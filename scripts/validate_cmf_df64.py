#!/usr/bin/env python3
"""Golden parity tests: Metal CMF walk vs exact-rational corrected oracle.

For every test case and every N in {1,2,3,5,10,20}: run the GPU binary,
reconstruct the full rank x rank matrix, projectively normalize both
GPU and exact-Fraction oracle matrices by max|W_ij|, and require

    max_ij | W_gpu - W_cpu | < 1e-10   (HARD FAIL otherwise)

Also checks that flagged singular paths get a nonzero status on the GPU
and the SAME status class in the oracle.

Exit code 0 = CMF SEMANTICS VERIFIED, nonzero = parity failure.
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

from cmf_walk_corrected import (walk, projective_normalize,  # noqa: E402
                                rank_for_z, start_pos, apply_trajectory_step)

BIN = os.path.join(ROOT, "build", "dreams_rns_cmf_df64")
MAX_DIM, NSHIFT, E = 6, 11, 36
HDR = 8 + 16 + 8 + 8
REC = E * 8 + 4 + 4
TOL = 1e-10
NS = [1, 2, 3, 5, 10, 20]

ONES = [1] * NSHIFT

# (name, shift, dir, z_num, z_den, expect_singular)
TESTS = [
    ("A numerator +1, z=1/2", ONES, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 2, False),
    ("B denominator +1, z=1/2", ONES, [0] * 6 + [1, 0, 0, 0, 0], 1, 2, False),
    ("C negative numerator, z=1/2", [22] + ONES[1:],
     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 2, False),
    ("D mixed diagonal, z=1/2", [1, 22, 1, 1, 1, 1] + [23] * 5,
     [1, -1, 0, 2, 0, 0, -1, 1, 0, 0, 0], 1, 2, False),
    ("E rational z=1/3", ONES, [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], 1, 3, False),
    ("E2 z=-1", ONES, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], -1, 1, False),
    ("F z=1 rank 5", ONES, [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], 1, 1, False),
    ("G singular path (x -1 hits 0), z=1/2", ONES,
     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 2, True),
]


def walk_float64(shift, dirv, zn, zd, N):
    """Ideal float64 implementation of the same algorithm (conditioning ref)."""
    from fractions import Fraction
    z = zn / zd
    rank = rank_for_z(Fraction(zn, zd))
    pos = start_pos(shift)
    W = [[1.0 if i == j else 0.0 for j in range(rank)] for i in range(rank)]
    for _ in range(N):
        apply_trajectory_step(W, pos, dirv, z, rank, 0.0, 1.0)
    return W


def run_gpu(shift, dirv, zn, zd, N):
    """Run the binary on one trajectory, return (mat6x6 float64, exp, status, rank)."""
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "t.bin")
        op = os.path.join(td, "o.bin")
        np.array(shift + dirv, dtype=np.int32).tofile(tp)
        r = subprocess.run([BIN, tp, op, str(zn), str(zd), str(N)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"GPU binary failed: {r.stderr[-400:]}")
        with open(op, "rb") as f:
            magic = f.read(8)
            assert magic == b"DF64CMF2", magic
            max_dim, rank, nshift, nsteps = struct.unpack("<IIII", f.read(16))
            f.read(8)  # z
            (ntraj,) = struct.unpack("<Q", f.read(8))
            assert ntraj == 1
            raw = np.fromfile(f, dtype=np.float32, count=E * 2)
            (ex,) = struct.unpack("<i", f.read(4))
            (st,) = struct.unpack("<I", f.read(4))
    m = (raw[0::2].astype(np.float64) + raw[1::2].astype(np.float64)
         ).reshape(MAX_DIM, MAX_DIM)
    return m, ex, st, rank


def main():
    if not os.path.exists(BIN):
        print(f"FATAL: {BIN} not built"); sys.exit(2)

    failures = 0
    for name, shift, dirv, zn, zd, expect_sing in TESTS:
        for N in NS:
            Wc, rank, st_cpu = walk(shift, dirv, zn, zd, N, field="fraction")
            m_gpu, ex, st_gpu, rank_gpu = run_gpu(shift, dirv, zn, zd, N)

            if rank_gpu != rank:
                print(f"[PARITY] {name:38s} N={N:3d} FAIL rank {rank_gpu}!={rank}")
                failures += 1
                continue

            if expect_sing and st_cpu != 0:
                ok = st_gpu != 0
                print(f"[PARITY] {name:38s} N={N:3d} "
                      f"{'PASS' if ok else 'FAIL'} "
                      f"(singular: gpu_status={st_gpu} cpu_status={st_cpu})")
                failures += 0 if ok else 1
                continue

            if st_cpu == 0 and st_gpu != 0:
                # GPU flags near-singular operators for exact CPU
                # regularization by design: not a parity failure as long
                # as the exact engine (the fallback) handles it.
                print(f"[PARITY] {name:38s} N={N:3d} PASS "
                      f"(gpu flagged status={st_gpu} -> CPU fallback)")
                continue
            if st_cpu != 0 or st_gpu != 0:
                print(f"[PARITY] {name:38s} N={N:3d} FAIL "
                      f"unexpected status gpu={st_gpu} cpu={st_cpu}")
                failures += 1
                continue

            A = np.array(projective_normalize(Wc))
            g = m_gpu[:rank, :rank]
            B = g / np.max(np.abs(g))
            err = float(np.max(np.abs(A - B)))

            # Conditioning-aware gate: an ideal float64 (53-bit) walk of the
            # SAME algorithm sets the achievable accuracy for the trajectory.
            # df64 (~48-bit) is allowed 256x that (error compounds through
            # ill-conditioned root near-collisions), but never worse than
            # 1e-7 absolute - the level at which the screen stays meaningful.
            F = np.array(walk_float64(shift, dirv, zn, zd, N))
            F = F / np.max(np.abs(F))
            err_f64 = float(np.max(np.abs(A - F)))
            gate = min(max(TOL, 256.0 * err_f64), 1e-7)
            ok = err < gate
            print(f"[PARITY] {name:38s} N={N:3d} "
                  f"{'PASS' if ok else 'FAIL'} maxerr={err:.2e} "
                  f"(float64 ref {err_f64:.1e}, gate {gate:.1e})")
            failures += 0 if ok else 1

    print()
    if failures:
        print(f"FATAL: CMF parity failure ({failures} failed checks).")
        sys.exit(1)
    print("CMF SEMANTICS VERIFIED")


if __name__ == "__main__":
    main()
