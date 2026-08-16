#!/usr/bin/env python3
"""Validate the Metal df64 walk against a float64 numpy reference.

Reference: P <- P @ M(n), n=1..N, with per-step max-abs normalization,
tracking log2 scale. Compares normalized matrix ratios and total log2
magnitude per element.
"""
import argparse
import struct
import numpy as np

DIM = 6
NSHIFT = 11
E = DIM * DIM
HDR = 8 + 4 * 3 + 4 * 2 + 8


def build_M(n, shift, dirv, z):
    f = [shift[i] + n * dirv[i] + 1.0 for i in range(DIM)]
    g = [shift[DIM + j] + n * dirv[DIM + j] + 2.0 for j in range(DIM - 1)]

    def esym(vals):
        e = np.zeros(len(vals) + 1)
        e[0] = 1.0
        for v in vals:
            for k in range(len(vals), 0, -1):
                e[k] += v * e[k - 1]
        return e

    ef = esym(f)
    eg = np.concatenate([esym(g), [0.0]])
    c = np.zeros(DIM)
    c[0] = -z * ef[DIM]
    for k in range(1, DIM):
        c[k] = eg[DIM - k] - z * ef[DIM - k]
    M = np.zeros((DIM, DIM))
    for r in range(1, DIM):
        M[r, r - 1] = 1.0
    M[:, DIM - 1] = c
    return M


def reference(shift, dirv, z, nsteps):
    P = np.eye(DIM)
    log2scale = 0.0
    for n in range(1, nsteps + 1):
        P = P @ build_M(n, shift, dirv, z)
        mx = np.abs(P).max()
        if mx > 0:
            ex = int(np.floor(np.log2(mx))) + 1  # frexp exponent
            P *= 2.0 ** (-ex)
            log2scale += ex
    return P, log2scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_bin")
    ap.add_argument("out_bin")
    ap.add_argument("--n-check", type=int, default=8)
    args = ap.parse_args()

    with open(args.out_bin, "rb") as fh:
        magic = fh.read(8)
        assert magic == b"DF64MAT1", magic
        dim, nshift, nsteps = struct.unpack("<III", fh.read(12))
        z_num, z_den = struct.unpack("<ii", fh.read(8))
        (ntraj,) = struct.unpack("<Q", fh.read(8))
        assert dim == DIM and nshift == NSHIFT
        recsz = E * 8 + 4
        raw = fh.read(args.n_check * recsz)

    traj = np.fromfile(args.traj_bin, dtype="<i4",
                       count=args.n_check * 2 * NSHIFT).reshape(-1, 2 * NSHIFT)
    z = z_num / z_den

    worst_ratio = 0.0
    worst_log2 = 0.0
    for t in range(min(args.n_check, ntraj)):
        rec = raw[t * recsz:(t + 1) * recsz]
        hilo = np.frombuffer(rec[:E * 8], dtype="<f4").reshape(E, 2)
        (expsum,) = struct.unpack("<i", rec[E * 8:])
        gpu = (hilo[:, 0].astype(np.float64) +
               hilo[:, 1].astype(np.float64)).reshape(DIM, DIM)

        shift = traj[t, :NSHIFT].astype(np.float64)
        dirv = traj[t, NSHIFT:].astype(np.float64)
        ref, log2ref = reference(shift, dirv, z, nsteps)

        # compare projective content: normalize both by their max-abs
        gn = gpu / np.abs(gpu).max()
        rn = ref / np.abs(ref).max()
        if np.sign(gn.flat[np.argmax(np.abs(gn))]) != np.sign(rn.flat[np.argmax(np.abs(rn))]):
            rn = -rn
        rel = np.abs(gn - rn).max() / np.abs(rn).max()
        worst_ratio = max(worst_ratio, rel)

        # compare total magnitude: expsum + log2(maxabs(gpu)) vs log2ref + log2(maxabs(ref))
        lg = expsum + np.log2(np.abs(gpu).max())
        lr = log2ref + np.log2(np.abs(ref).max())
        worst_log2 = max(worst_log2, abs(lg - lr))
        print(f"traj {t}: rel matrix err {rel:.3e}, log2 magnitude "
              f"gpu {lg:.6f} vs ref {lr:.6f}")

    print(f"\nworst relative matrix error: {worst_ratio:.3e}")
    print(f"worst log2 magnitude error:  {worst_log2:.3e}")
    ok = worst_ratio < 1e-9 and worst_log2 < 1e-3
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
