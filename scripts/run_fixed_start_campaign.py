#!/usr/bin/env python3
"""End-to-end fixed-start trajectory optimization campaign.

Architecture:
  1. one FIXED_START_PROFILE_V1 identifies the start/limit once
  2. CPU generates + canonicalizes + pole/shard-filters directions
  3. projected two-row GPU funnel at increasing depths
  4. each stage retains only the best directions by a convergence/height
     proxy (NOT claimed to be exact delta)
  5. CPU exact Fraction refinement computes real delta on final survivors

The GPU output is only 44 bytes/direction rather than a full 6x6 matrix.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
BIN = os.path.join(ROOT, "build", "dreams_fixed_start_df64")
NSHIFT = 11
HDR = 100
REC_DT = np.dtype([
    ("vals", "<f4", (4, 2)),
    ("exp", "<i4", (2,)),
    ("status", "<u4"),
])
DIR_DT = np.dtype(("<i4", (NSHIFT,)))


def run(cmd, **kw):
    print("[fixed] $ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True, **kw)


def csv(v):
    return ",".join(str(int(x)) for x in v)


def read_header(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"FSDF6401":
            raise ValueError(f"unexpected fixed-start output magic {magic!r}")
        hdr = struct.unpack("<8I", f.read(32))
        max_dim, rank, nshift, nsteps, cp0, cp1, row_num, row_den = hdr
        z_num, z_den = struct.unpack("<2i", f.read(8))
        start = list(struct.unpack("<11i", f.read(44)))
        (ntraj,) = struct.unpack("<Q", f.read(8))
    return {
        "max_dim": max_dim, "rank": rank, "nshift": nshift,
        "nsteps": nsteps, "cp0": cp0, "cp1": cp1,
        "row_num": row_num, "row_den": row_den,
        "z_num": z_num, "z_den": z_den,
        "start": start, "ntraj": ntraj,
    }


def score_output(path, target: float):
    h = read_header(path)
    rec = np.memmap(path, dtype=REC_DT, mode="r", offset=HDR,
                    shape=(h["ntraj"],))
    x = rec["vals"][:, :, 0].astype(np.float64) + rec["vals"][:, :, 1].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        L0 = x[:, 0] / x[:, 1]
        L1 = x[:, 2] / x[:, 3]
        conv = np.abs(L1 - L0) / np.maximum(np.abs(L1), 1e-300)
        target_rel = np.abs(L1 - target) / max(abs(target), 1e-300)

        # expSum tracks common projective magnitude growth. It is NOT the
        # exact rational denominator, so this is explicitly only a proxy.
        height_log = np.maximum(np.abs(rec["exp"][:, 1]).astype(np.float64) * math.log(2.0), 1.0)
        proxy = -1.0 - np.log(np.maximum(target_rel, 1e-300)) / height_log
        conv_penalty = 0.02 * np.log10(1.0 + conv / 1e-14)
        score = proxy - conv_penalty

    valid = ((rec["status"] == 0) & np.isfinite(L0) & np.isfinite(L1) &
             np.isfinite(score) & (x[:, 1] != 0) & (x[:, 3] != 0))
    score = np.where(valid, score, -np.inf)
    return h, rec, L0, L1, conv, target_rel, proxy, score, valid


def top_indices(score, valid, keep):
    idx = np.flatnonzero(valid)
    if len(idx) <= keep:
        return idx[np.argsort(score[idx])[::-1]]
    vals = score[idx]
    part = np.argpartition(vals, -keep)[-keep:]
    top = idx[part]
    return top[np.argsort(score[top])[::-1]]


def write_dirs(src_path, indices, dst_path):
    src = np.memmap(src_path, dtype=DIR_DT, mode="r")
    arr = np.asarray(src[indices], dtype=np.int32)
    arr.tofile(dst_path)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--profile", required=True,
                    help="JSON from identify_fixed_start.py")
    ap.add_argument("--directions", default=None,
                    help="optional pre-generated direction binary; otherwise generate")
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="directions to generate if --directions omitted")
    ap.add_argument("--dir-min", type=int, default=-20)
    ap.add_argument("--dir-max", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-dir", default=None,
                    help="11-value CSV; mutate locally instead of global random pool")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--shard-npz", default=None)
    ap.add_argument("--stages", default="128,256,512,1000")
    ap.add_argument("--keep", default="200000,50000,10000,2000",
                    help="absolute survivors retained after each stage")
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--exact-top", type=int, default=500)
    ap.add_argument("--exact-workers", type=int, default=14)
    ap.add_argument("--exact-depths", default="256,512,1000")
    ap.add_argument("--exact-dps", type=int, default=160)
    ap.add_argument("--keep-stage-binaries", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.profile) as f:
        profile = json.load(f)
    if profile.get("format") != "FIXED_START_PROFILE_V1":
        raise SystemExit("--profile is not FIXED_START_PROFILE_V1")

    stages = [int(x) for x in args.stages.split(",")]
    keeps = [int(x) for x in args.keep.split(",")]
    if len(stages) != len(keeps) or any(x <= 0 for x in stages + keeps):
        raise SystemExit("--stages and --keep must be equal-length positive integer lists")
    if stages != sorted(stages) or len(set(stages)) != len(stages):
        raise SystemExit("--stages must be strictly increasing")

    start = profile["absolute_start"]
    row_num, row_den = profile["ratio_pair"]
    z_num, z_den = profile["z_num"], profile["z_den"]
    target = float(profile["target_numeric"])

    if not os.path.exists(BIN):
        raise SystemExit(f"{BIN} not built; run cmake -S . -B build && cmake --build build")

    # Phase 1: direction pool.
    if args.directions:
        current = os.path.abspath(args.directions)
    else:
        current = os.path.join(args.outdir, "directions_initial.bin")
        cmd = [PY, os.path.join(HERE, "gen_fixed_directions.py"), current,
               "--start", csv(start), "--n", args.n,
               "--seed", args.seed, "--dir-min", args.dir_min,
               "--dir-max", args.dir_max,
               "--precheck-depth", max(stages)]
        if args.seed_dir:
            cmd += ["--seed-dir", args.seed_dir, "--radius", args.radius]
        if args.shard_npz:
            cmd += ["--shard-npz", args.shard_npz]
        run(cmd, cwd=ROOT)

    n0 = os.path.getsize(current) // (NSHIFT * 4)
    ids = np.arange(n0, dtype=np.int64)
    print(f"[fixed] initial admissible pool: {n0:,} directions", flush=True)

    manifest = {
        "format": "FIXED_START_CAMPAIGN_V1",
        "profile": os.path.abspath(args.profile),
        "absolute_start": start,
        "ratio_pair": [row_num, row_den],
        "z": [z_num, z_den],
        "target_numeric": profile["target_numeric"],
        "target_expr": profile.get("target_expr"),
        "initial_directions": n0,
        "stages": [],
    }
    man_path = os.path.join(args.outdir, "fixed_manifest.json")

    last_metrics = None
    t0 = time.time()
    for stage_no, (depth, keep) in enumerate(zip(stages, keeps), 1):
        out_bin = os.path.join(args.outdir, f"gpu_N{depth:04d}.bin")
        run([BIN, current, out_bin, csv(start), row_num, row_den,
             z_num, z_den, depth, args.chunk], cwd=ROOT)

        h, rec, L0, L1, conv, target_rel, proxy, score, valid = score_output(out_bin, target)
        if h["start"] != start or h["row_num"] != row_num or h["row_den"] != row_den:
            raise RuntimeError("GPU output metadata does not match fixed-start profile")
        n_valid = int(valid.sum())
        keep_eff = min(keep, n_valid)
        if keep_eff == 0:
            raise RuntimeError(f"stage N={depth}: no valid trajectories survived")
        idx = top_indices(score, valid, keep_eff)

        next_path = os.path.join(args.outdir, f"directions_after_N{depth:04d}.bin")
        arr = write_dirs(current, idx, next_path)
        ids = ids[idx]

        stage_meta = {
            "N": depth,
            "checkpoint0": h["cp0"],
            "checkpoint1": h["cp1"],
            "input": int(h["ntraj"]),
            "valid": n_valid,
            "retained": int(len(idx)),
            "best_gpu_score": float(score[idx[0]]),
            "best_proxy": float(proxy[idx[0]]),
            "best_target_rel": float(target_rel[idx[0]]),
            "best_convergence": float(conv[idx[0]]),
            "best_global_id": int(ids[0]),
            "best_direction": arr[0].tolist(),
        }
        manifest["stages"].append(stage_meta)
        with open(man_path + ".tmp", "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(man_path + ".tmp", man_path)

        print(f"[fixed] N={depth}: valid={n_valid:,}/{h['ntraj']:,}, "
              f"retained={len(idx):,}, best score={score[idx[0]]:.6f}, "
              f"target_rel={target_rel[idx[0]]:.3e}, conv={conv[idx[0]]:.3e}", flush=True)

        # Retain final-stage metrics for JSONL export before deleting mmap file.
        last_metrics = {
            "ids": ids.copy(),
            "dirs": arr.copy(),
            "score": np.asarray(score[idx]).copy(),
            "proxy": np.asarray(proxy[idx]).copy(),
            "L0": np.asarray(L0[idx]).copy(),
            "L1": np.asarray(L1[idx]).copy(),
            "conv": np.asarray(conv[idx]).copy(),
            "target_rel": np.asarray(target_rel[idx]).copy(),
            "exp": np.asarray(rec["exp"][idx]).copy(),
            "depth": depth,
            "cp0": h["cp0"],
        }

        prev = current
        current = next_path
        # Close memmap before optional unlink on macOS.
        del rec
        if not args.keep_stage_binaries:
            try:
                os.remove(out_bin)
            except OSError:
                pass
        # Direction survivor files are intentionally retained: they are small,
        # make every funnel decision auditable, and allow resuming/refinement.

    if last_metrics is None:
        raise RuntimeError("no stages executed")

    candidates_path = os.path.join(args.outdir, "GPU_SURVIVORS.jsonl")
    with open(candidates_path, "w") as out:
        m = last_metrics
        for k in range(len(m["ids"])):
            out.write(json.dumps({
                "global_id": int(m["ids"][k]),
                "direction": m["dirs"][k].tolist(),
                "N": int(m["depth"]),
                "checkpoint0": int(m["cp0"]),
                "L_checkpoint": float(m["L0"][k]),
                "L_final": float(m["L1"][k]),
                "target_rel": float(m["target_rel"][k]),
                "convergence": float(m["conv"][k]),
                "height_exp_checkpoint": int(m["exp"][k, 0]),
                "height_exp_final": int(m["exp"][k, 1]),
                "gpu_delta_proxy": float(m["proxy"][k]),
                "gpu_score": float(m["score"][k]),
            }) + "\n")

    manifest["gpu_survivors"] = len(last_metrics["ids"])
    manifest["gpu_survivors_file"] = candidates_path
    manifest["gpu_seconds_wall_total"] = time.time() - t0

    # Phase 3: real delta. This uses all requested CPU workers and exact
    # rational denominators, not the GPU exponent proxy.
    if args.exact_top > 0:
        if profile.get("target_expr"):
            exact_out = os.path.join(args.outdir, "EXACT_DELTA.jsonl")
            run([PY, os.path.join(HERE, "refine_fixed_delta.py"), candidates_path,
                 "--profile", args.profile, "--out", exact_out,
                 "--workers", args.exact_workers,
                 "--max-candidates", args.exact_top,
                 "--depths", args.exact_depths,
                 "--dps", args.exact_dps], cwd=ROOT)
            manifest["exact_delta_file"] = exact_out
        else:
            print("[fixed] profile has no target_expr; skipping exact delta. "
                  "Re-run identify_fixed_start.py with --target-expr once PSLQ is confirmed.",
                  flush=True)

    with open(man_path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(man_path + ".tmp", man_path)
    print(f"[fixed] DONE -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
