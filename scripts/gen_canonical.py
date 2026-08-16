#!/usr/bin/env python3
"""Canonicalized 6F5 trajectory generator at scale.

The 6F5 companion matrix is invariant under permutations of the 6
numerator roots and the 5 denominator roots separately, so with fixed
shift=[1]*11 a trajectory is canonical iff dir[0:6] and dir[6:11] are
each non-decreasing. This enumerates canonical representatives ONLY:
no trajectory is ever generated twice, and the space is fully covered.

Enumeration scheme (deterministic + resumable):
  f-block: multisets of size 6 over [-20..20]  -> Nf = C(46,6) = 9,366,819
  g-block: multisets of size 5 over [-20..20]  -> Ng = C(45,5) = 1,221,759
  Total canonical space: Nf * Ng ~ 1.14e13 trajectories.

f-multisets are visited in a pseudo-random but bijective order
fi' = (fi*Af + Bf) mod Nf. One "slot" = one f-multiset x all Ng
g-multisets (1.22M trajectories). A chunk of ~100M = 82 slots.
Progress is tracked in <outdir>/manifest.json (next_slot cursor).

Output: <outdir>/traj_chunk_<k>.bin, records of int32 shift[11]+dir[11].
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np

K_F, K_G = 6, 5
NSHIFT = 11
DMIN, DMAX = -20, 20
NV = DMAX - DMIN + 1  # 41

NF = math.comb(NV + K_F - 1, K_F)  # 9,366,819
NG = math.comb(NV + K_G - 1, K_G)  # 1,221,759

AF = 6000011  # multiplier for the bijective slot shuffle
BF = 1234567


def _check_bijection():
    assert math.gcd(AF, NF) == 1, "AF must be coprime to NF"


def unrank_cwr(idx: int, k: int, v: int) -> list[int]:
    """Unrank a size-k multiset (non-decreasing tuple) over alphabet 0..v-1."""
    out = []
    lo = 0
    for pos in range(k):
        m = k - pos - 1  # remaining positions after this one
        for a in range(lo, v):
            cnt = math.comb((v - a) + m - 1, m)
            if idx < cnt:
                out.append(a)
                lo = a
                break
            idx -= cnt
        else:
            raise ValueError("unrank out of range")
    return out


def build_g_table() -> np.ndarray:
    """All Ng g-multisets in lex order, as int32 dir values."""
    g = np.array(
        list(itertools.combinations_with_replacement(range(NV), K_G)),
        dtype=np.int32,
    )
    return g + DMIN


def load_manifest(outdir: str) -> dict:
    p = os.path.join(outdir, "manifest.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"next_slot": 0, "Af": AF, "Bf": BF, "Nf": NF, "Ng": NG,
            "chunks": []}


def save_manifest(outdir: str, man: dict):
    p = os.path.join(outdir, "manifest.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--count", type=int, default=100_000_000,
                    help="approximate trajectories per chunk")
    ap.add_argument("--chunk-id", type=int, default=None,
                    help="override chunk id (default: len(manifest.chunks))")
    args = ap.parse_args()

    _check_bijection()
    os.makedirs(args.outdir, exist_ok=True)
    man = load_manifest(args.outdir)
    if man["Af"] != AF or man["Nf"] != NF:
        raise SystemExit("manifest was generated with different parameters")

    n_slots = max(1, round(args.count / NG))
    slot0 = man["next_slot"]
    if slot0 + n_slots > NF:
        n_slots = NF - slot0
        if n_slots <= 0:
            raise SystemExit("canonical space exhausted")

    chunk_id = args.chunk_id if args.chunk_id is not None else len(man["chunks"])
    out_path = os.path.join(args.outdir, f"traj_chunk_{chunk_id:05d}.bin")

    g_table = build_g_table()  # (NG, 5)
    shift = np.ones((NG, NSHIFT), dtype=np.int32)

    total = 0
    with open(out_path, "wb") as fh:
        for s in range(slot0, slot0 + n_slots):
            fi = (s * AF + BF) % NF
            f_tuple = np.array(unrank_cwr(fi, K_F, NV), dtype=np.int32) + DMIN

            rec = np.empty((NG, 2 * NSHIFT), dtype=np.int32)
            rec[:, :NSHIFT] = shift
            rec[:, NSHIFT:NSHIFT + K_F] = f_tuple
            rec[:, NSHIFT + K_F:] = g_table

            # drop the single all-zero direction if present
            if not f_tuple.any():
                keep = g_table.any(axis=1)
                rec = rec[keep]

            rec.tofile(fh)
            total += len(rec)

    man["next_slot"] = slot0 + n_slots
    man["chunks"].append({
        "chunk_id": chunk_id, "file": os.path.basename(out_path),
        "slot_start": slot0, "slot_end": slot0 + n_slots, "n_traj": total,
    })
    save_manifest(args.outdir, man)

    frac = 100.0 * man["next_slot"] / NF
    print(f"chunk {chunk_id}: {total:,} canonical trajectories -> {out_path}")
    print(f"cursor: slot {man['next_slot']:,}/{NF:,} ({frac:.4f}% of canonical space)")


if __name__ == "__main__":
    main()
