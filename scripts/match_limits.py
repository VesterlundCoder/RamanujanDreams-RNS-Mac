#!/usr/bin/env python3
"""CPU pipeline: match DF64CMF2 matrix limits against the constant library.

CORRECTED-semantics edition: reads the DF64CMF2 format produced by
dreams_rns_cmf_df64 (theta-companion CMF axis walk). Rank-aware
(rank 5 at z=1, else 6) and status-aware: only status==OK trajectories
enter the float64 screen; flagged trajectories (singular / needs
regularization) are written to a separate fallback queue for the exact
CPU engine.

The limit candidates are the rank*(rank-1) ordered last-column ratios
v[i]/v[j] (last = rank-1). Convergence is certified from the SAME matrix
by cross-checking against column rank-2 (rank-1 matrix test). Converged
ratios are matched against the constant library; rational limits are
rejected via continued fractions. Hits go to hits.jsonl for PSLQ.

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

MAX_DIM = 6
E = 36
NSHIFT = 11
HDR = 8 + 16 + 8 + 8
REC = E * 8 + 4 + 4  # 296 bytes

REC_DT = np.dtype([("mat", "<f4", (E, 2)), ("exp", "<i4"), ("status", "<u4")])
TRAJ_DT = np.dtype([("shift", "<i4", (NSHIFT,)), ("dir", "<i4", (NSHIFT,))])

_G = {}


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"DF64CMF2", ("not a corrected-semantics DF64CMF2 "
                                      f"file (magic={magic!r})")
        max_dim, rank, nshift, nsteps = struct.unpack("<IIII", f.read(16))
        z_num, z_den = struct.unpack("<ii", f.read(8))
        (ntraj,) = struct.unpack("<Q", f.read(8))
    assert max_dim == MAX_DIM and nshift == NSHIFT
    assert rank in (5, 6)
    return rank, nsteps, z_num, z_den, ntraj


def rational_mask(vals):
    """True for library entries that are rational or near-rational traps.

    Two classes are excluded from matching:
      1. exactly rational values (q<=5000 at 1e-12): composite names like
         '-1/12*-3/20' are rational; noisy rational limits flood them.
      2. entries within 1e-5 of a SMALL rational (q<=50): e.g.
         zeta(20)+7 = 8.00000095... is a trap for rational limits with
         ~1e-7 df64 walk drift.
    """
    mask = np.zeros(len(vals), dtype=bool)
    for k, v in enumerate(vals):
        if not np.isfinite(v) or v == 0:
            mask[k] = True
            continue
        fr = Fraction(float(v)).limit_denominator(5000)
        if fr.denominator and abs(v - float(fr)) <= 1e-12 * abs(v):
            mask[k] = True
            continue
        fr = Fraction(float(v)).limit_denominator(50)
        if fr.denominator and abs(v - float(fr)) <= 1e-5 * abs(v):
            mask[k] = True
    return mask


def init_worker(mat_path, traj_path, tol, conv_tol, rank):
    vals, names = build_library()
    _G["vals"] = vals
    _G["names"] = names
    _G["lib_rat"] = rational_mask(vals)
    _G["mat"] = np.memmap(mat_path, dtype=REC_DT, mode="r", offset=HDR)
    _G["traj"] = np.memmap(traj_path, dtype=TRAJ_DT, mode="r")
    _G["tol"] = tol
    _G["conv_tol"] = conv_tol
    _G["rank"] = rank
    _G["pairs"] = [(i, j) for i in range(rank) for j in range(rank) if i != j]


def process_range(rng):
    lo, hi = rng
    mat = _G["mat"][lo:hi]
    vals, names = _G["vals"], _G["names"]
    tol, conv_tol = _G["tol"], _G["conv_tol"]
    rank = _G["rank"]
    pairs = _G["pairs"]

    status = mat["status"]
    ok_mask = status == 0
    n_flagged = int((~ok_mask).sum())
    flagged_ids = (lo + np.nonzero(~ok_mask)[0]).tolist()

    m = (mat["mat"][:, :, 0].astype(np.float64) +
         mat["mat"][:, :, 1].astype(np.float64)).reshape(-1, MAX_DIM, MAX_DIM)
    m[~ok_mask] = np.nan  # exclude flagged trajectories from the screen

    hits = []
    n_conv = 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        v5 = m[:, :rank, rank - 1]  # last active column
        v4 = m[:, :rank, rank - 2]  # cross-check column
        ii = np.array([p[0] for p in pairs])
        jj = np.array([p[1] for p in pairs])
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
                    if _G["lib_rat"][c[t]]:
                        continue  # rational-valued library entry
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
    return len(mat), n_conv, hits, n_flagged, flagged_ids


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

    rank, nsteps, z_num, z_den, ntraj = read_header(args.matrices_bin)
    out_path = args.out or args.matrices_bin + ".hits.jsonl"
    fb_path = args.matrices_bin + ".fallback.jsonl"
    print(f"matching {ntraj:,} trajectories (N={nsteps}, z={z_num}/{z_den}, "
          f"rank={rank}) with {args.workers} workers", flush=True)

    # warm library cache before forking
    vals, _ = build_library()
    print(f"library: {len(vals):,} constants", flush=True)

    ranges = [(lo, min(lo + args.block, ntraj))
              for lo in range(0, ntraj, args.block)]

    t0 = time.time()
    done = 0
    n_conv_tot = 0
    n_hits = 0
    n_flagged_tot = 0
    traj_mm = np.memmap(args.traj_bin, dtype=TRAJ_DT, mode="r")
    with open(out_path, "w") as out, open(fb_path, "w") as fb, Pool(
            args.workers, initializer=init_worker,
            initargs=(args.matrices_bin, args.traj_bin,
                      args.tol, args.conv_tol, rank)) as pool:
        for nrec, n_conv, hits, n_flagged, flagged_ids in \
                pool.imap_unordered(process_range, ranges):
            done += nrec
            n_conv_tot += n_conv
            n_flagged_tot += n_flagged
            for h in hits:
                h["z_num"], h["z_den"], h["N"] = z_num, z_den, nsteps
                h["rank"] = rank
                out.write(json.dumps(h) + "\n")
            for tid in flagged_ids:
                fb.write(json.dumps({
                    "traj_id": tid,
                    "shift": traj_mm[tid]["shift"].tolist(),
                    "dir": traj_mm[tid]["dir"].tolist(),
                    "z_num": z_num, "z_den": z_den, "N": nsteps,
                }) + "\n")
            n_hits += len(hits)
            el = time.time() - t0
            print(f"  {done:,}/{ntraj:,} ({done/el:,.0f} traj/s) "
                  f"converged-ratios={n_conv_tot:,} hits={n_hits} "
                  f"flagged={n_flagged_tot:,}", flush=True)

    el = time.time() - t0
    print(f"DONE {ntraj:,} traj in {el:.1f}s = {ntraj/el:,.0f} traj/s; "
          f"{n_hits} library hits -> {out_path}; "
          f"{n_flagged_tot:,} flagged -> {fb_path}", flush=True)


if __name__ == "__main__":
    main()
