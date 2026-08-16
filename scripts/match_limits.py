#!/usr/bin/env python3
"""CPU pipeline: match DF64MAT1 matrix limits against the constant library.

For every trajectory the stored 6x6 product matrix at N=1000 is rank-1
to ~14 digits when the walk converges. The limit candidates are the 30
ordered last-column ratios v[i]/v[j]. Convergence is certified from the
SAME matrix by cross-checking against column 4 (rank-1 test):
    conv = |r_col5 - r_col4| / |r_col5|
Only converged ratios (conv < conv-tol) are matched against the
~3.4M-entry float64 limit library (limit_library.py from the census
pipeline). Rational matches are dropped. Hits go to hits.jsonl for the
PSLQ stage.

Multiprocessing over record ranges; each worker mmaps the files.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
CENSUS = "/Users/davidvesterlund/freedomcode/freedomcode/stringtheory/6F5Sweeps/continuous_cmf/gpu_proxy"
sys.path.insert(0, LIB if os.path.isdir(LIB) else CENSUS)
from limit_library import build_library, is_rational_name  # noqa: E402

DIM = 6
E = 36
NSHIFT = 11
HDR = 8 + 12 + 8 + 8
REC = E * 8 + 4  # 292 bytes

REC_DT = np.dtype([("mat", "<f4", (E, 2)), ("exp", "<i4")])
TRAJ_DT = np.dtype([("shift", "<i4", (NSHIFT,)), ("dir", "<i4", (NSHIFT,))])

PAIRS = [(i, j) for i in range(DIM) for j in range(DIM) if i != j]  # 30

_G = {}


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"DF64MAT1", magic
        dim, nshift, nsteps = struct.unpack("<III", f.read(12))
        z_num, z_den = struct.unpack("<ii", f.read(8))
        (ntraj,) = struct.unpack("<Q", f.read(8))
    assert dim == DIM and nshift == NSHIFT
    return nsteps, z_num, z_den, ntraj


def init_worker(mat_path, traj_path, tol, conv_tol):
    vals, names = build_library()
    _G["vals"] = vals
    _G["names"] = names
    _G["mat"] = np.memmap(mat_path, dtype=REC_DT, mode="r", offset=HDR)
    _G["traj"] = np.memmap(traj_path, dtype=TRAJ_DT, mode="r")
    _G["tol"] = tol
    _G["conv_tol"] = conv_tol


def process_range(rng):
    lo, hi = rng
    mat = _G["mat"][lo:hi]
    vals, names = _G["vals"], _G["names"]
    tol, conv_tol = _G["tol"], _G["conv_tol"]

    m = (mat["mat"][:, :, 0].astype(np.float64) +
         mat["mat"][:, :, 1].astype(np.float64)).reshape(-1, DIM, DIM)

    hits = []
    n_conv = 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v5 = m[:, :, 5]  # last column
        v4 = m[:, :, 4]  # cross-check column
        ii = np.array([p[0] for p in PAIRS])
        jj = np.array([p[1] for p in PAIRS])
        r5 = v5[:, ii] / v5[:, jj]          # (B, 30)
        r4 = v4[:, ii] / v4[:, jj]
        conv = np.abs(r5 - r4) / np.maximum(np.abs(r5), 1e-300)

        ok = (np.isfinite(r5) & (conv < conv_tol) &
              (np.abs(r5) > 1e-6) & (np.abs(r5) < 1e6))
        n_conv = int(ok.sum())

        b_idx, p_idx = np.nonzero(ok)
        if len(b_idx):
            r = r5[b_idx, p_idx]
            pos = np.searchsorted(vals, r)
            for cand in (pos - 1, pos):
                c = np.clip(cand, 0, len(vals) - 1)
                lib = vals[c]
                rel = np.abs(r - lib) / np.maximum(np.abs(r), np.abs(lib))
                sel = rel < tol
                for t in np.nonzero(sel)[0]:
                    name = names[c[t]]
                    if is_rational_name(name.lstrip("-")):
                        continue
                    # reject limits that are themselves small rationals
                    # (composite library names like "-7/5*-20/17" are rational)
                    x = float(r[t])
                    fr = Fraction(x).limit_denominator(10_000)
                    if fr.denominator and abs(x - float(fr)) <= 1e-10 * abs(x):
                        continue
                    b = int(b_idx[t])
                    hits.append({
                        "traj_id": int(lo + b),
                        "i": int(ii[p_idx[t]]), "j": int(jj[p_idx[t]]),
                        "L_float": float(r[t]),
                        "lib_name": name,
                        "rel_err": float(rel[t]),
                        "conv": float(conv[b, p_idx[t]]),
                        "shift": _G["traj"][lo + b]["shift"].tolist(),
                        "dir": _G["traj"][lo + b]["dir"].tolist(),
                    })
    return len(mat), n_conv, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrices_bin")
    ap.add_argument("traj_bin")
    ap.add_argument("--out", default=None, help="hits jsonl (default <matrices>.hits.jsonl)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--block", type=int, default=1_000_000)
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--conv-tol", type=float, default=1e-10)
    args = ap.parse_args()

    nsteps, z_num, z_den, ntraj = read_header(args.matrices_bin)
    out_path = args.out or args.matrices_bin + ".hits.jsonl"
    print(f"matching {ntraj:,} trajectories (N={nsteps}, z={z_num}/{z_den}) "
          f"with {args.workers} workers", flush=True)

    # warm library cache before forking
    vals, _ = build_library()
    print(f"library: {len(vals):,} constants", flush=True)

    ranges = [(lo, min(lo + args.block, ntraj))
              for lo in range(0, ntraj, args.block)]

    t0 = time.time()
    done = 0
    n_conv_tot = 0
    n_hits = 0
    with open(out_path, "w") as out, Pool(
            args.workers, initializer=init_worker,
            initargs=(args.matrices_bin, args.traj_bin,
                      args.tol, args.conv_tol)) as pool:
        for nrec, n_conv, hits in pool.imap_unordered(process_range, ranges):
            done += nrec
            n_conv_tot += n_conv
            for h in hits:
                h["z_num"], h["z_den"], h["N"] = z_num, z_den, nsteps
                out.write(json.dumps(h) + "\n")
            n_hits += len(hits)
            el = time.time() - t0
            print(f"  {done:,}/{ntraj:,} ({done/el:,.0f} traj/s) "
                  f"converged-ratios={n_conv_tot:,} hits={n_hits}", flush=True)

    el = time.time() - t0
    print(f"DONE {ntraj:,} traj in {el:.1f}s = {ntraj/el:,.0f} traj/s; "
          f"{n_hits} library hits -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
