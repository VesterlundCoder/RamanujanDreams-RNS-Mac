#!/usr/bin/env python3
"""Campaign orchestrator: GPU walk + CPU limit-match + PSLQ, in chunks.

Per chunk:
  1. gen_canonical.py     -> canonical trajectories (resumable cursor)
  2. dreams_rns_df64      -> GPU df64 walk, matrices stored at N=1000
  3. match_limits.py      -> CPU library matching (runs in BACKGROUND
                             while the next chunk's GPU walk proceeds)
  4. pslq_hits.py         -> full PSLQ on library hits (background)

CPU stages overlap the next GPU chunk so the CPUs track GPU tempo.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
BIN = os.path.join(ROOT, "build", "dreams_rns_cmf_df64")


def run(cmd, **kw):
    print(f"[campaign] $ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def spawn(cmd, log_path):
    print(f"[campaign] & {' '.join(map(str, cmd))} > {log_path}", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)


def update_hits_summary(outdir):
    """Aggregate PSLQ-confirmed hits from all chunks into a live list."""
    confirmed = []
    for path in sorted(glob.glob(os.path.join(outdir, "*.pslq.jsonl"))):
        try:
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("status") == "pslq_hit":
                        rec["source"] = os.path.basename(path)
                        confirmed.append(rec)
        except Exception:
            continue

    jl = os.path.join(outdir, "PSLQ_CONFIRMED.jsonl")
    with open(jl + ".tmp", "w") as f:
        for rec in confirmed:
            f.write(json.dumps(rec) + "\n")
    os.replace(jl + ".tmp", jl)

    md = os.path.join(outdir, "PSLQ_HITS.md")
    with open(md + ".tmp", "w") as f:
        f.write("# PSLQ-Confirmed Hits (live)\n\n")
        f.write(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"Confirmed hits: **{len(confirmed)}**\n\n")
        if confirmed:
            f.write("| # | dir | z | pair | L (40d) | delta | lib match | "
                    "first PSLQ relation | mult |\n")
            f.write("|---|-----|---|------|---------|-------|-----------|"
                    "--------------------|------|\n")
            for k, r in enumerate(confirmed, 1):
                rel = r["pslq"][0] if r.get("pslq") else {}
                rel_s = " + ".join(
                    f"{c}*{n}" for c, n in zip(rel.get("coeffs", []),
                                               rel.get("names", [])) if c)
                f.write(f"| {k} | {r['dir']} | {r['z_num']}/{r['z_den']} | "
                        f"({r['i']},{r['j']}) | {r.get('L_exact_40d','')} | "
                        f"{r.get('delta_exact', float('nan')):.4f} | "
                        f"{r['lib_name']} | {rel_s} = 0 | "
                        f"{r.get('n_occurrences', 1)} |\n")
        else:
            f.write("No confirmed hits yet.\n")
    os.replace(md + ".tmp", md)
    return len(confirmed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--chunks", type=int, default=None,
                    help="max chunks (default: unlimited)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop starting new chunks after this many hours")
    ap.add_argument("--chunk-size", type=int, default=100_000_000)
    ap.add_argument("--z-num", type=int, default=1)
    ap.add_argument("--z-den", type=int, default=2)
    ap.add_argument("--nsteps", type=int, default=1000)
    ap.add_argument("--match-workers", type=int, default=14)
    ap.add_argument("--pslq-workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--delete-traj", action="store_true",
                    help="delete trajectory bin after matching (regenerable)")
    ap.add_argument("--delete-matrices", action="store_true",
                    help="delete matrix bin after matching (29GB/100M)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # MANDATORY parity gate: never start a campaign on unverified semantics
    print("[PARITY] running mandatory CMF golden tests...", flush=True)
    gate = subprocess.run([PY, os.path.join(HERE, "validate_cmf_df64.py")],
                          cwd=ROOT)
    if gate.returncode != 0:
        print("FATAL: CMF parity failure. Campaign aborted.", flush=True)
        sys.exit(1)
    print("[PARITY] CMF SEMANTICS VERIFIED", flush=True)

    bg = []  # (proc, stage, chunk_id, cleanup_paths)

    def reap(block=False):
        for ent in list(bg):
            proc, stage, cid, cleanup = ent
            if block:
                proc.wait()
            if proc.poll() is None:
                continue
            bg.remove(ent)
            if proc.returncode != 0:
                print(f"[campaign] WARNING {stage} chunk {cid} exited "
                      f"rc={proc.returncode}", flush=True)
                continue
            print(f"[campaign] {stage} chunk {cid} finished", flush=True)
            if stage == "pslq":
                n = update_hits_summary(args.outdir)
                print(f"[campaign] hits summary updated: {n} confirmed", flush=True)
            if stage == "match":
                # chain PSLQ
                mat = os.path.join(args.outdir, f"matrices_chunk_{cid:05d}.bin")
                hits = mat + ".hits.jsonl"
                if os.path.exists(hits) and os.path.getsize(hits) > 0:
                    p = spawn([PY, os.path.join(HERE, "pslq_hits.py"), hits,
                               "--workers", args.pslq_workers],
                              os.path.join(args.outdir, f"pslq_{cid:05d}.log"))
                    bg.append((p, "pslq", cid, []))
                else:
                    print(f"[campaign] chunk {cid}: no hits, skipping PSLQ", flush=True)
                for path in cleanup:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"[campaign] removed {path}", flush=True)

    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None
    man_path = os.path.join(args.outdir, "manifest.json")

    def manifest_len():
        if os.path.exists(man_path):
            with open(man_path) as f:
                return len(json.load(f)["chunks"])
        return 0

    def orphan_chunk():
        """A traj bin that was generated but never walked (previous run)."""
        for tp in sorted(glob.glob(os.path.join(args.outdir, "traj_chunk_*.bin"))):
            cid = int(os.path.basename(tp)[11:16])
            mp = os.path.join(args.outdir, f"matrices_chunk_{cid:05d}.bin")
            hp = mp + ".hits.jsonl"
            if not os.path.exists(mp) and not os.path.exists(hp):
                return cid
        return None

    gen_proc = None   # background generation of the NEXT chunk
    gen_cid = None
    chunks_done = 0
    while True:
        if args.chunks is not None and chunks_done >= args.chunks:
            break
        if deadline and time.time() > deadline:
            print("[campaign] time budget reached, stopping new chunks", flush=True)
            break

        # 1. obtain a trajectory chunk
        if gen_proc is not None:
            gen_proc.wait()
            if gen_proc.returncode != 0:
                print("[campaign] generation failed, aborting", flush=True)
                break
            cid = gen_cid
            gen_proc = None
        else:
            cid = orphan_chunk()
            if cid is None:
                cid = manifest_len()
                run([PY, os.path.join(HERE, "gen_canonical.py"), args.outdir,
                     "--count", args.chunk_size, "--chunk-id", cid])

        traj = os.path.join(args.outdir, f"traj_chunk_{cid:05d}.bin")
        mat = os.path.join(args.outdir, f"matrices_chunk_{cid:05d}.bin")

        # 2. pre-generate the NEXT chunk in the background (overlaps GPU)
        if (deadline is None or time.time() < deadline) and (
                args.chunks is None or chunks_done + 1 < args.chunks):
            nxt = manifest_len()
            gen_proc = spawn([PY, os.path.join(HERE, "gen_canonical.py"),
                              args.outdir, "--count", args.chunk_size,
                              "--chunk-id", nxt],
                             os.path.join(args.outdir, f"gen_{nxt:05d}.log"))
            gen_cid = nxt

        # 3. GPU walk (matrices stored at N=nsteps)
        run([BIN, traj, mat, args.z_num, args.z_den, args.nsteps], cwd=ROOT)

        # 4. CPU matching in background (overlaps next GPU chunk)
        cleanup = []
        if args.delete_traj:
            cleanup.append(traj)
        if args.delete_matrices:
            cleanup.append(mat)
        p = spawn([PY, os.path.join(HERE, "match_limits.py"), mat, traj,
                   "--workers", args.match_workers, "--tol", args.tol],
                  os.path.join(args.outdir, f"match_{cid:05d}.log"))
        bg.append((p, "match", cid, cleanup))
        chunks_done += 1
        reap()
        el = time.time() - t0
        print(f"[campaign] chunk {cid} dispatched "
              f"({chunks_done} chunks, {el/60:.1f} min elapsed)", flush=True)

    if gen_proc is not None:
        gen_proc.wait()

    print("[campaign] waiting for background CPU stages...", flush=True)
    while bg:
        reap(block=True)
    update_hits_summary(args.outdir)

    print(f"[campaign] ALL DONE: {chunks_done} chunks in "
          f"{(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
