#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re

p = argparse.ArgumentParser()
p.add_argument("run_dir")
a = p.parse_args()
run = pathlib.Path(a.run_dir)

cpu_re = re.compile(
    r"TOTAL\s+(\d+)\s+directions\s+\((\d+)\s+OK,\s+(\d+)\s+flagged\)\s+"
    r"in\s+([0-9.eE+-]+)\s+s\s+=>\s+([0-9.eE+-]+)\s+dir/s\s+wall;\s+"
    r"compute\s+([0-9.eE+-]+)\s+s\s+=>\s+([0-9.eE+-]+)\s+dir/s"
)
metal_re = re.compile(
    r"TOTAL\s+(\d+)\s+directions\s+\((\d+)\s+OK,\s+(\d+)\s+flagged\)\s+"
    r"in\s+([0-9.eE+-]+)\s+s\s+=>\s+([0-9.eE+-]+)\s+dir/s"
)

rows = []
for path in sorted(run.glob("*.log")):
    text = path.read_text(errors="replace")
    backend = path.stem
    m = cpu_re.search(text)
    if m:
        total, ok, flagged = map(int, m.group(1,2,3))
        wall, wall_rate, compute, compute_rate = map(float, m.group(4,5,6,7))
        rows.append(dict(backend=backend,total=total,ok=ok,flagged=flagged,
                         wall_seconds=wall,wall_dir_s=wall_rate,
                         compute_seconds=compute,compute_dir_s=compute_rate))
        continue
    m = metal_re.search(text)
    if m:
        total, ok, flagged = map(int, m.group(1,2,3))
        wall, wall_rate = map(float, m.group(4,5))
        rows.append(dict(backend=backend,total=total,ok=ok,flagged=flagged,
                         wall_seconds=wall,wall_dir_s=wall_rate,
                         compute_seconds="",compute_dir_s=""))

out = run / "summary.csv"
fields = ["backend","total","ok","flagged","wall_seconds","wall_dir_s","compute_seconds","compute_dir_s"]
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

if not rows:
    print("No TOTAL lines found in logs")
else:
    width = max(len(str(r["backend"])) for r in rows)
    print(f"{'backend':<{width}}  {'wall dir/s':>14}  {'compute dir/s':>14}  {'OK':>10}  {'flagged':>10}")
    for r in sorted(rows, key=lambda x: float(x["wall_dir_s"]), reverse=True):
        cr = r["compute_dir_s"]
        crs = f"{float(cr):.2f}" if cr != "" else "-"
        print(f"{r['backend']:<{width}}  {float(r['wall_dir_s']):14.2f}  {crs:>14}  {r['ok']:10d}  {r['flagged']:10d}")
    print(f"\nCSV: {out}")
