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

# Direction range. In CORRECTED Dreams semantics the start point from
# shift=[1]*11 is v0=(2,...,2;3,...,3): any NEGATIVE direction component
# drives x_i (or y_j) into the pole lattice within ~2 trajectory steps,
# so the deep fixed-start census uses non-negative directions.
DMIN_DEFAULT, DMAX_DEFAULT = 0, 20

BF = 1234567


def space_params(dmin: int, dmax: int):
    nv = dmax - dmin + 1
    nf = math.comb(nv + K_F - 1, K_F)
    ng = math.comb(nv + K_G - 1, K_G)
    af = 6000011
    while math.gcd(af, nf) != 1:
        af += 2
    return nv, nf, ng, af


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


def build_g_table(nv: int, dmin: int) -> np.ndarray:
    """All Ng g-multisets in lex order, as int32 dir values."""
    g = np.array(
        list(itertools.combinations_with_replacement(range(nv), K_G)),
        dtype=np.int32,
    )
    return g + dmin


def load_manifest(outdir: str, dmin: int, dmax: int) -> dict:
    p = os.path.join(outdir, "manifest.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    nv, nf, ng, af = space_params(dmin, dmax)
    return {"next_slot": 0, "Af": af, "Bf": BF, "Nf": nf, "Ng": ng,
            "dmin": dmin, "dmax": dmax, "chunks": []}


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
    ap.add_argument("--dmin", type=int, default=DMIN_DEFAULT)
    ap.add_argument("--dmax", type=int, default=DMAX_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    man = load_manifest(args.outdir, args.dmin, args.dmax)
    if man.get("dmin", args.dmin) != args.dmin or \
            man.get("dmax", args.dmax) != args.dmax:
        raise SystemExit("manifest was generated with a different dir range")
    NV, NF, NG, AF = space_params(man.get("dmin", args.dmin),
                                  man.get("dmax", args.dmax))
    if man["Af"] != AF or man["Nf"] != NF:
        raise SystemExit("manifest was generated with different parameters")
    DMIN = man.get("dmin", args.dmin)

    n_slots = max(1, round(args.count / NG))
    slot0 = man["next_slot"]
    if slot0 + n_slots > NF:
        n_slots = NF - slot0
        if n_slots <= 0:
            raise SystemExit("canonical space exhausted")

    chunk_id = args.chunk_id if args.chunk_id is not None else len(man["chunks"])
    out_path = os.path.join(args.outdir, f"traj_chunk_{chunk_id:05d}.bin")

    g_table = build_g_table(NV, DMIN)  # (NG, 5)
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
