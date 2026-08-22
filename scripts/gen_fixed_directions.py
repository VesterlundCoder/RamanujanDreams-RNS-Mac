#!/usr/bin/env python3
"""Generate canonical directions for one absolute 6F5 CMF start point.

Output records contain ONLY int32 direction[11]. The absolute start is
campaign metadata and is never duplicated per trajectory.

Filtering performed before GPU work:
  * zero direction rejection
  * permutation canonicalization only among axes whose fixed start
    coordinates are equal (numerator and denominator blocks separately)
  * exact arithmetic pole prefilter through the requested depth
  * optional shard recession-cone filter A @ direction <= tol from NPZ

The final directions are sorted by (max|dir|, L1 norm) to reduce SIMD
branch divergence in the Metal unit-step loop.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

NSHIFT = 11
NX = 6


def parse_vec(text: str) -> np.ndarray:
    v = np.array([int(x.strip()) for x in text.split(",")], dtype=np.int32)
    if v.shape != (NSHIFT,):
        raise argparse.ArgumentTypeError("expected exactly 11 comma-separated integers")
    return v


def equal_start_groups(start: np.ndarray):
    groups = []
    for lo, hi in ((0, NX), (NX, NSHIFT)):
        block = start[lo:hi]
        for value in np.unique(block):
            idx = np.flatnonzero(block == value) + lo
            if len(idx) > 1:
                groups.append(idx)
    return groups


def canonicalize(a: np.ndarray, groups) -> np.ndarray:
    a = a.copy()
    for idx in groups:
        a[:, idx] = np.sort(a[:, idx], axis=1)
    return a


def pole_safe_mask(dirs: np.ndarray, start: np.ndarray, depth: int) -> np.ndarray:
    """Reject rays that necessarily hit an axis denominator zero.

    For a positive unit step the denominator is the current coordinate,
    so zero is forbidden. For a negative unit step it is current-1, so
    current coordinate 1 is forbidden. This test accounts for every
    intermediate unit step when |direction_i| > 1 and all macro steps
    through ``depth``.
    """
    d = dirs.astype(np.int64, copy=False)
    s = start.astype(np.int64)[None, :]
    safe = np.ones(len(d), dtype=bool)

    pos = d > 0
    upper = s + depth * d - 1
    hit_zero = pos & (s <= 0) & (upper >= 0)

    neg = d < 0
    lower = s + depth * d + 1
    hit_one = neg & (s >= 1) & (lower <= 1)

    safe &= ~(hit_zero | hit_one).any(axis=1)
    return safe


def shard_mask(dirs: np.ndarray, start: np.ndarray, shard_npz: str | None,
               tol: float) -> np.ndarray:
    if not shard_npz:
        return np.ones(len(dirs), dtype=bool)
    data = np.load(shard_npz)
    A = np.asarray(data["A"], dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != NSHIFT:
        raise ValueError("shard NPZ A must have shape (m,11)")
    if "b" in data:
        b = np.asarray(data["b"], dtype=np.float64)
        if b.shape != (A.shape[0],):
            raise ValueError("shard NPZ b must have shape (m,)")
        if not np.all(A @ start.astype(np.float64) < b + tol):
            raise ValueError("absolute start is not inside the supplied shard")
    # Closed recession cone: directions parallel to a facet are valid.
    return np.all(dirs.astype(np.float64) @ A.T <= tol, axis=1)


def candidate_batch(rng, batch: int, dmin: int, dmax: int,
                    seed_dir: np.ndarray | None, radius: int) -> np.ndarray:
    if seed_dir is None:
        return rng.integers(dmin, dmax + 1, size=(batch, NSHIFT), dtype=np.int32)
    off = rng.integers(-radius, radius + 1, size=(batch, NSHIFT), dtype=np.int32)
    a = seed_dir[None, :] + off
    return np.clip(a, dmin, dmax).astype(np.int32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output direction binary (11 int32 per record)")
    ap.add_argument("--start", required=True, type=parse_vec,
                    help="ABSOLUTE CMF start, 11 comma-separated integers")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dir-min", type=int, default=-20)
    ap.add_argument("--dir-max", type=int, default=20)
    ap.add_argument("--seed-dir", type=parse_vec, default=None,
                    help="optional known direction around which to mutate")
    ap.add_argument("--radius", type=int, default=3,
                    help="mutation radius when --seed-dir is supplied")
    ap.add_argument("--precheck-depth", type=int, default=1000,
                    help="reject guaranteed axis-pole hits through this depth")
    ap.add_argument("--shard-npz", default=None,
                    help="optional NPZ containing A and optional b; requires A@dir<=0")
    ap.add_argument("--shard-tol", type=float, default=1e-12)
    ap.add_argument("--batch", type=int, default=250_000)
    args = ap.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.dir_min > args.dir_max:
        raise SystemExit("--dir-min must be <= --dir-max")
    if args.precheck_depth <= 0:
        raise SystemExit("--precheck-depth must be positive")

    start = args.start
    groups = equal_start_groups(start)
    rng = np.random.default_rng(args.seed)

    # Tuple set is intentional: fixed-start campaigns usually generate a
    # finite optimization pool once, then reuse survivor subsets across
    # funnel stages. Scientific reproducibility is worth the memory cost.
    seen: set[tuple[int, ...]] = set()
    accepted = []
    generated = 0
    pole_rejected = 0
    shard_rejected = 0

    stall = 0
    while len(seen) < args.n:
        a = candidate_batch(rng, args.batch, args.dir_min, args.dir_max,
                            args.seed_dir, args.radius)
        generated += len(a)
        a = canonicalize(a, groups)
        a = a[a.any(axis=1)]
        if not len(a):
            stall += 1
            if stall > 20:
                raise RuntimeError("unable to generate nonzero directions")
            continue

        pm = pole_safe_mask(a, start, args.precheck_depth)
        pole_rejected += int((~pm).sum())
        a = a[pm]
        if not len(a):
            stall += 1
            if stall > 20:
                raise RuntimeError("candidate space appears exhausted by pole prefilter")
            continue

        sm = shard_mask(a, start, args.shard_npz, args.shard_tol)
        shard_rejected += int((~sm).sum())
        a = a[sm]

        before = len(seen)
        for row in a:
            key = tuple(int(v) for v in row)
            if key not in seen:
                seen.add(key)
                accepted.append(row.copy())
                if len(seen) >= args.n:
                    break
        stall = stall + 1 if len(seen) == before else 0
        if stall > 50:
            raise RuntimeError(
                "unique admissible direction space appears exhausted; widen bounds/radius"
            )

    arr = np.asarray(accepted[:args.n], dtype=np.int32)

    # Group similar unit-step costs together to reduce SIMD divergence.
    maxabs = np.max(np.abs(arr), axis=1)
    l1 = np.sum(np.abs(arr), axis=1, dtype=np.int64)
    order = np.lexsort((l1, maxabs))
    arr = arr[order]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    arr.astype("<i4", copy=False).tofile(args.out)

    meta = {
        "format": "FIXED_DIR_I32_V1",
        "absolute_start": start.tolist(),
        "n_directions": int(len(arr)),
        "dir_min": args.dir_min,
        "dir_max": args.dir_max,
        "seed": args.seed,
        "seed_dir": None if args.seed_dir is None else args.seed_dir.tolist(),
        "radius": args.radius if args.seed_dir is not None else None,
        "precheck_depth": args.precheck_depth,
        "symmetry_groups": [g.tolist() for g in groups],
        "generated_raw": generated,
        "pole_rejected": pole_rejected,
        "shard_rejected": shard_rejected,
        "shard_npz": args.shard_npz,
        "sorted_for_gpu_divergence": True,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(arr):,} unique admissible fixed-start directions -> {args.out}")
    print(f"raw={generated:,} pole_rejected={pole_rejected:,} shard_rejected={shard_rejected:,}")
    print(f"symmetry groups={meta['symmetry_groups']}")


if __name__ == "__main__":
    main()
