#!/usr/bin/env python
"""RNS (residue number system) walk for the MeijerG(4,2,4,4,1) CMF.

Port of the DreamsRNS-Metal-Mac full-matrix RNS pipeline
(~/dreams_rns/DreamsRNS-Metal-Mac) to the Meijer G trajectory search:

  1. For each (initial, trajectory) pair the CMF trajectory matrix M(n) is a
     polynomial matrix in the step index n.  Entries are cleared to integer
     polynomial coefficients (a global scalar denominator cancels in every
     projective ratio, hence in c1 = s1/s0 and c2 = s2/s0).
  2. Coefficients are reduced modulo K 31-bit primes (auto-sized from a
     float64 growth proxy so Garner CRT capacity is guaranteed).
  3. The walk P <- P * M(n), n = 0..N-1, runs entirely in uint32/uint64
     modular arithmetic -- a bit-exact reference of walk_meijer_rns.metal
     (one lane = one (candidate, prime) pair; here lanes are vectorized
     over primes with numpy).
  4. Full modular matrices are snapshotted at every checkpoint; the CPU
     reconstructs exact integer matrices with Garner CRT, forms exact
     rational (c1, c2) and hands them to the existing PSLQ/delta stack.
  5. --validate cross-checks the CRT states against the exact sympy walk.
  6. --export writes the coefficient/prime tensors + config in the binary
     layout expected by the Metal host, for GPU runs at scale.

Usage examples (from meijer_search/):
  ./venv/bin/python rns/rns_walk.py --validate --depth 200
  ./venv/bin/python rns/rns_walk.py --depth 1000 --checkpoints 900,950,1000
  ./venv/bin/python rns/rns_walk.py --pairs pairs.json --depth 2000 \
      --export rns/export_meijer
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import n  # step symbol used by the CMF matrices  # noqa: E402

DEFAULT_INITIAL = (1, 1, 1, 1, 3, 3, 2, 0)
BASELINE = (1, -1, -3, -4, 12, 10, 8, -18)


# ------------------------------------------------------------ polynomialize
def integer_poly_matrix(tm):
    """Clear M(n) to integer polynomial coefficients.

    Returns (coeffs, m, deg): coeffs[i][j] is the high-to-low (Horner)
    integer coefficient list of entry (i, j), all padded to length deg+1.
    The cleared global scalar (lcm of denominators) is projectively
    irrelevant: it cancels in every matrix-element ratio.
    """
    m = tm.shape[0]
    entries = [sp.cancel(sp.together(tm[i, j])) for i in range(m) for j in range(m)]
    den = sp.Integer(1)
    for e in entries:
        den = sp.lcm(den, sp.fraction(e)[1])
    polys = []
    coeff_lcm = 1
    for e in entries:
        num, dd = sp.fraction(e)
        q = sp.expand(num * sp.cancel(den / dd))
        poly = sp.Poly(q, n)
        polys.append(poly)
        for c in poly.all_coeffs():
            coeff_lcm = sp.ilcm(coeff_lcm, sp.denom(c))
    deg = max(p.degree() for p in polys)
    coeffs = []
    for p in polys:
        cs = [int(c * coeff_lcm) for c in p.all_coeffs()]
        coeffs.append([0] * (deg + 1 - len(cs)) + cs)
    return [coeffs[i * m:(i + 1) * m] for i in range(m)], m, deg


# ------------------------------------------------------------ prime sizing
def gen_primes(k, below=2**31):
    """k largest primes below 2^31 (fit uint32; products fit uint64)."""
    out = []
    cand = below - 1
    while len(out) < k:
        if sp.isprime(cand):
            out.append(cand)
        cand -= 2
    return out


def required_primes(coeffs, m, deg, depth, margin=8):
    """Rigorous CRT capacity bound from the INTEGER coefficient sizes.

    |P_N|_max <= prod_{t=0}^{N-1} m * maxentry(M(t)) with
    maxentry(M(t)) <= (deg+1) * maxcoeff * max(t,1)^deg, so
    bits = sum_t [log2 m + log2(deg+1) + bits(maxcoeff) + deg*log2 max(t,1)].
    Guarantees signed-centered Garner CRT uniqueness (with one sign bit)."""
    maxcoeff_bits = max(abs(c).bit_length()
                        for row in coeffs for entry in row for c in entry)
    per_t_const = math.log2(m) + math.log2(deg + 1) + maxcoeff_bits
    bits = sum(per_t_const + deg * math.log2(max(t, 1)) for t in range(depth))
    return max(4, int(bits / 30.0) + margin)


# ------------------------------------------------------------ RNS reference walk
def encode_residues(coeffs, primes):
    """coeffs[i][j] (Horner ints) -> uint32 residue tensor [K][E][deg1]."""
    m = len(coeffs)
    deg1 = len(coeffs[0][0])
    K = len(primes)
    out = np.zeros((K, m * m, deg1), dtype=np.uint64)
    for k, p in enumerate(primes):
        for i in range(m):
            for j in range(m):
                for d, c in enumerate(coeffs[i][j]):
                    out[k, i * m + j, d] = c % p
    return out


def walk_rns(coeffR, primes, m, depth, checkpoints):
    """Vectorized-over-primes modular walk.  Bit-exact mirror of the Metal
    kernel: P <- P * M(t), t = 0..depth-1, snapshot at each checkpoint."""
    K = len(primes)
    pv = np.asarray(primes, dtype=np.uint64)
    P = np.zeros((K, m, m), dtype=np.uint64)
    P[:, range(m), range(m)] = 1
    snaps = {}
    cps = sorted(checkpoints)
    ci = 0
    deg1 = coeffR.shape[2]
    for t in range(depth):
        tmod = np.uint64(t) % pv                      # [K]
        val = coeffR[:, :, 0].copy()                  # [K][E]
        for j in range(1, deg1):
            val = (val * tmod[:, None] + coeffR[:, :, j]) % pv[:, None]
        M = val.reshape(K, m, m)
        newP = np.zeros_like(P)
        for z in range(m):
            newP = (newP + P[:, :, z][:, :, None] * M[:, z, :][:, None, :]) \
                % pv[:, None, None]
        P = newP
        while ci < len(cps) and cps[ci] == t + 1:
            snaps[cps[ci]] = P.copy()
            ci += 1
    return snaps


# ------------------------------------------------------------ CRT reconstruction
def garner_crt(residues, primes):
    """Signed-centered CRT lift of one integer from residues (Python ints)."""
    x = 0
    mod = 1
    for r, p in zip(residues, primes):
        t = ((int(r) - x) * pow(mod, -1, p)) % p
        x += t * mod
        mod *= p
    if x > mod // 2:
        x -= mod
    return x


def reconstruct_state(snap, primes, m):
    """First column of the product matrix as exact integers (s0, s1, ...)."""
    return [garner_crt([snap[k, i, 0] for k in range(len(primes))], primes)
            for i in range(m)]


def crt_capacity_ok(snap, primes, m):
    """Reconstruction is trusted only if dropping 2 primes gives the same
    integers (standard consistency test for unknown magnitudes)."""
    full = reconstruct_state(snap, primes, m)
    part = reconstruct_state(snap[:-2], primes[:-2], m)
    return full == part, full


# ------------------------------------------------------------ driver
def run_pair(initial, traj, depth, checkpoints, validate, quiet=False):
    t0 = time.time()
    tm = C.make_tm(traj, initial=initial)
    coeffs, m, deg = integer_poly_matrix(tm)
    K = required_primes(coeffs, m, deg, depth)
    primes = gen_primes(K)
    if not quiet:
        print(f"[pair] init={initial} traj={traj}: m={m}, deg={deg}, "
              f"K={K} primes, depth={depth}", flush=True)

    coeffR = encode_residues(coeffs, primes)
    snaps = walk_rns(coeffR, primes, m, depth, checkpoints)

    result = {"initial": list(initial), "traj": list(traj),
              "m": m, "deg": deg, "K": K, "depth": depth, "checkpoints": {}}
    for cp in sorted(snaps):
        ok, state = crt_capacity_ok(snaps[cp], primes, m)
        if not ok:
            result["checkpoints"][cp] = {"status": "crt_capacity_exceeded"}
            if not quiet:
                print(f"  [cp {cp}] CRT capacity exceeded", flush=True)
            continue
        if state[0] == 0:
            result["checkpoints"][cp] = {"status": "degenerate"}
            if not quiet:
                print(f"  [cp {cp}] degenerate state", flush=True)
            continue
        c1 = sp.Rational(state[1], state[0])
        c2 = sp.Rational(state[2], state[0])
        entry = {"status": "ok"}
        if validate:
            exact = C.walk_states(tm, [cp])[0]
            ec1, ec2 = C.normalized(exact)
            entry["exact_match"] = bool(c1 == ec1 and c2 == ec2)
        rel = C.identify_relation_z2z3(c1, c2, digits=300)
        if rel is not None:
            entry["relation"] = list(rel)
            entry["target"] = C.relation_label(rel)
            est = C.relation_estimate_zeta(rel, c1, c2)
            entry["delta"] = C.delta_components_zeta(est, rel[:6])[0]
        result["checkpoints"][cp] = entry
        if not quiet:
            msg = f"  [cp {cp}] " + entry.get("target", "no z2z3 relation")
            if "delta" in entry:
                msg += f"  delta={entry['delta']:+.7f}"
            if validate:
                msg += f"  exact_match={entry['exact_match']}"
            print(msg, flush=True)
    result["time"] = time.time() - t0
    return result


# ------------------------------------------------------------ Metal export
def export_metal(pairs, depth, checkpoints, out_dir):
    """Binary tensors for the walk_meijer_rns.metal host.

    Layout:
      config.json                 dims/prime count/checkpoints
      primes.bin                  uint32[K]
      coeffs.bin                  uint32[K][B][E][deg1]  (C order)
    All candidates share one padded degree and one prime set (sized for the
    worst-growing candidate) so a single dispatch covers the whole batch.
    """
    built = []
    for initial, traj in pairs:
        tm = C.make_tm(tuple(traj), initial=tuple(initial))
        coeffs, m, deg = integer_poly_matrix(tm)
        built.append((initial, traj, tm, coeffs, m, deg))
    m = built[0][4]
    assert all(b[4] == m for b in built), "mixed matrix dimensions"
    deg = max(b[5] for b in built)
    K = max(required_primes(b[3], b[4], b[5], depth) for b in built)
    primes = gen_primes(K)
    B, E, deg1 = len(built), m * m, deg + 1

    coeff_tensor = np.zeros((K, B, E, deg1), dtype=np.uint32)
    for bi, (_, _, _, coeffs, _, d) in enumerate(built):
        pad = deg1 - (d + 1)
        for k, p in enumerate(primes):
            for i in range(m):
                for j in range(m):
                    for di, cval in enumerate(coeffs[i][j]):
                        coeff_tensor[k, bi, i * m + j, pad + di] = cval % p

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "primes.bin"), "wb") as f:
        f.write(np.asarray(primes, dtype=np.uint32).tobytes())
    with open(os.path.join(out_dir, "coeffs.bin"), "wb") as f:
        f.write(coeff_tensor.tobytes())
    cfg = {"nTraj": B, "m": m, "deg1": deg1, "nSteps": depth, "K": K,
           "nCheckpoints": len(checkpoints),
           "checkpointSteps": sorted(checkpoints),
           "pairs": [[list(i), list(t)] for i, t, *_ in built],
           "kernel": "walk_meijer_rns",
           "snapshot_layout": "[C][K][B][E] uint32"}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[export] {out_dir}: B={B} m={m} deg1={deg1} K={K} "
          f"({K*B*E*deg1*4/1e6:.1f} MB coeffs)", flush=True)


def load_pairs(path):
    with open(path) as f:
        data = json.load(f)
    return [(tuple(p[0]), tuple(p[1])) for p in data]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", help="JSON file: [[initial, traj], ...]; "
                                    "default = baseline pair")
    ap.add_argument("--depth", type=int, default=500)
    ap.add_argument("--checkpoints", default="",
                    help="comma-separated; default = depth only")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check CRT states against the exact sympy walk")
    ap.add_argument("--export", metavar="DIR",
                    help="write Metal host tensors instead of walking on CPU")
    ap.add_argument("--out", default="", help="write results JSONL here")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs) if args.pairs \
        else [(DEFAULT_INITIAL, BASELINE)]
    cps = ([int(x) for x in args.checkpoints.split(",") if x]
           if args.checkpoints else [args.depth])
    assert max(cps) <= args.depth

    if args.export:
        export_metal(pairs, args.depth, cps, args.export)
        return

    results = []
    for initial, traj in pairs:
        results.append(run_pair(initial, traj, args.depth, cps, args.validate))
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[out] appended {len(results)} records to {args.out}")


if __name__ == "__main__":
    main()
