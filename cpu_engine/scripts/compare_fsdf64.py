#!/usr/bin/env python3
import argparse
import math
import struct

HEADER_FMT = "<8s8I2i11iQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
REC_FMT = "<8f2iI"
REC_SIZE = struct.calcsize(REC_FMT)


def read_header(f):
    b = f.read(HEADER_SIZE)
    if len(b) != HEADER_SIZE:
        raise ValueError("short header")
    return struct.unpack(HEADER_FMT, b)

p = argparse.ArgumentParser(description="Compare Metal and CPU FSDF6401 output")
p.add_argument("a")
p.add_argument("b")
p.add_argument("--top", type=int, default=10)
a = p.parse_args()

with open(a.a, "rb") as fa, open(a.b, "rb") as fb:
    ha, hb = read_header(fa), read_header(fb)
    if ha != hb:
        print("HEADERS DIFFER")
        print("A:", ha)
        print("B:", hb)
        raise SystemExit(1)
    n = ha[-1]
    bitwise = 0
    status_mismatch = 0
    exp_mismatch = 0
    max_float_abs = 0.0
    max_ratio_abs = 0.0
    max_ratio_rel = 0.0
    worst = []

    for i in range(n):
        ba, bb = fa.read(REC_SIZE), fb.read(REC_SIZE)
        if len(ba) != REC_SIZE or len(bb) != REC_SIZE:
            raise SystemExit(f"short record at {i}")
        if ba == bb:
            bitwise += 1
        ra, rb = struct.unpack(REC_FMT, ba), struct.unpack(REC_FMT, bb)
        if ra[-1] != rb[-1]:
            status_mismatch += 1
        if ra[8:10] != rb[8:10]:
            exp_mismatch += 1
        for x, y in zip(ra[:8], rb[:8]):
            if math.isfinite(x) and math.isfinite(y):
                max_float_abs = max(max_float_abs, abs(x-y))
        if ra[-1] == 0 and rb[-1] == 0:
            for cp in range(2):
                na = ra[cp*4] + ra[cp*4+1]
                da = ra[cp*4+2] + ra[cp*4+3]
                nb = rb[cp*4] + rb[cp*4+1]
                db = rb[cp*4+2] + rb[cp*4+3]
                if da != 0.0 and db != 0.0:
                    qa, qb = na/da, nb/db
                    if math.isfinite(qa) and math.isfinite(qb):
                        ae = abs(qa-qb)
                        re = ae / max(abs(qa), abs(qb), 1e-300)
                        max_ratio_abs = max(max_ratio_abs, ae)
                        max_ratio_rel = max(max_ratio_rel, re)
                        worst.append((re, ae, i, cp, qa, qb))

    worst.sort(reverse=True)
    print(f"records={n}")
    print(f"bitwise_identical={bitwise}/{n} ({100.0*bitwise/max(n,1):.4f}%)")
    print(f"status_mismatch={status_mismatch}")
    print(f"exp_mismatch={exp_mismatch}")
    print(f"max_float_component_abs_diff={max_float_abs:.9g}")
    print(f"max_ratio_abs_diff={max_ratio_abs:.17g}")
    print(f"max_ratio_rel_diff={max_ratio_rel:.17g}")
    for re, ae, i, cp, qa, qb in worst[:a.top]:
        print(f"worst record={i} cp={cp} rel={re:.3e} abs={ae:.3e} A={qa:.17g} B={qb:.17g}")
