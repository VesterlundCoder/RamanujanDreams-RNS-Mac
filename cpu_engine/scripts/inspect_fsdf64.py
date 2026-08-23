#!/usr/bin/env python3
import argparse
import math
import struct
from collections import Counter

HEADER_FMT = "<8s8I2i11iQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
REC_FMT = "<8f2iI"
REC_SIZE = struct.calcsize(REC_FMT)

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--show", type=int, default=3)
a = p.parse_args()

with open(a.file, "rb") as f:
    raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise SystemExit("short header")
    h = struct.unpack(HEADER_FMT, raw)
    magic = h[0]
    if magic != b"FSDF6401":
        raise SystemExit(f"bad magic {magic!r}")
    max_dim, rank, nshift, nsteps, cp0, cp1, row_num, row_den = h[1:9]
    z_num, z_den = h[9:11]
    start = h[11:22]
    ntraj = h[22]
    print(f"magic={magic.decode()} rank={rank} nshift={nshift} nsteps={nsteps} cp=({cp0},{cp1}) pair=({row_num},{row_den}) z={z_num}/{z_den}")
    print(f"start={list(start)} ntraj={ntraj} record_bytes={REC_SIZE}")
    counts = Counter()
    shown = 0
    for i in range(ntraj):
        r = f.read(REC_SIZE)
        if len(r) != REC_SIZE:
            raise SystemExit(f"short record at {i}")
        vals = struct.unpack(REC_FMT, r)
        status = vals[-1]
        counts[status] += 1
        if shown < a.show:
            q = []
            for c in range(2):
                num = vals[c*4] + vals[c*4+1]
                den = vals[c*4+2] + vals[c*4+3]
                q.append(num / den if den else math.nan)
            print(f"[{i}] status={status} exp=({vals[8]},{vals[9]}) ratios=({q[0]:.17g},{q[1]:.17g})")
            shown += 1
    extra = f.read(1)
    if extra:
        print("warning: trailing bytes after expected records")
    print("status_counts=" + repr(dict(sorted(counts.items()))))
