#!/usr/bin/env python3
import argparse
import random
import struct

NSHIFT = 11

p = argparse.ArgumentParser(description="Generate deterministic int32 direction[11] records")
p.add_argument("--count", type=int, required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--nonzero", type=int, default=2, help="nonzero coordinates per direction")
p.add_argument("--max-abs", type=int, default=1)
p.add_argument("--axes", default="0,1,2,3,4,5,6,7,8,9,10")
a = p.parse_args()

axes = [int(x) for x in a.axes.split(",") if x != ""]
if not axes or any(x < 0 or x >= NSHIFT for x in axes):
    raise SystemExit("--axes must be a comma-separated subset of 0..10")
if not (1 <= a.nonzero <= len(axes)):
    raise SystemExit("--nonzero must be between 1 and number of allowed axes")
if a.max_abs < 1:
    raise SystemExit("--max-abs must be >= 1")

rng = random.Random(a.seed)
with open(a.output, "wb") as f:
    for _ in range(a.count):
        d = [0] * NSHIFT
        for axis in rng.sample(axes, a.nonzero):
            mag = rng.randint(1, a.max_abs)
            d[axis] = mag if rng.getrandbits(1) else -mag
        f.write(struct.pack("<11i", *d))
print(f"wrote {a.count} directions to {a.output}")
