#!/usr/bin/env python3
"""Generate 6F5 trajectory binary for the Metal df64 walk.

Record layout: int32 shift[11] + int32 dir[11], concatenated.
Convention matches the census pipeline: initial point [1]*11,
direction components sampled from the census component set.
"""
import argparse
import numpy as np

NSHIFT = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dir-min", type=int, default=-20)
    ap.add_argument("--dir-max", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shift = np.ones((args.n, NSHIFT), dtype=np.int32)
    dirs = rng.integers(args.dir_min, args.dir_max + 1,
                        size=(args.n, NSHIFT), dtype=np.int32)
    # avoid all-zero directions
    zero = ~dirs.any(axis=1)
    while zero.any():
        dirs[zero] = rng.integers(args.dir_min, args.dir_max + 1,
                                  size=(int(zero.sum()), NSHIFT), dtype=np.int32)
        zero = ~dirs.any(axis=1)

    rec = np.concatenate([shift, dirs], axis=1)  # (n, 22)
    rec.astype("<i4").tofile(args.out)
    print(f"wrote {args.n} trajectories -> {args.out} "
          f"({rec.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
