# Fixed-Start 6F5 Optimization Architecture

This path is for the second stage of a Ramanujan Dreams search:

1. choose one **absolute CMF start point** `v0`;
2. identify its limit once;
3. hold `v0` fixed;
4. search many admissible trajectory directions for the best delta/convergence;
5. compute real delta only for the small GPU survivor set.

The old global census remains available and unchanged.

## 1. Semantics: N=0 is a position index, not zero walk depth

For absolute start `v0` and direction `t`, a depth-N walk is

```text
W_N(v0,t) = T(v0,t) T(v0+t,t) ... T(v0+(N-1)t,t).
```

Therefore a run to `N=1000` still uses **1000 trajectory operators**. The
first operator is evaluated at the point called `n=0`, namely exactly `v0`.

Do not set the executable's `nSteps` to zero. Zero steps is the identity
product and contains no limit information. The fixed-start executable rejects
`nSteps=0` with an explanatory error.

## 2. Absolute start versus legacy shift

The fixed-start path accepts only **absolute CMF coordinates**:

```text
absolute_start = [x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4]
```

There is no implicit offset and no pre-step by the direction.

The retired/legacy Mac CMF trajectory record used a different convention:

```text
x_i(0) = legacy_shift_i + 1
y_j(0) = legacy_shift_{6+j} + 2
```

So, and only if a start is being migrated from an old Mac trajectory binary,
convert it once with

```text
absolute_start[0:6] = legacy_shift[0:6] + 1
absolute_start[6:11] = legacy_shift[6:11] + 2
```

If a coordinate vector already denotes the real Ramanujan Dreams start point,
use it **as-is**. Never apply the conversion twice.

## 3. Why the GPU propagates only two rows

For a fixed ratio pair `(i,j)`, the wanted approximation comes from two rows of
the product matrix. Every CMF update is a right multiplication

```text
W <- W A.
```

Rows evolve independently under right multiplication. Instead of maintaining
all 6 rows, initialize only

```text
e_i^T, e_j^T
```

and propagate those two rows. At rank 6 this reduces row-update work from six
rows to two rows, while giving exactly the same selected matrix ratio as the
full product.

The Metal kernel stores only the numerator and denominator last-column values
at `N/2` and `N`, plus the common power-of-two exponent and status. Output is
44 bytes per direction instead of 296 bytes per trajectory in `DF64CMF2`.

## 4. Build and mandatory validation

```bash
cmake -S . -B build
cmake --build build
python scripts/validate_fixed_start.py
```

A scientific run must print

```text
FIXED-START SEMANTICS VERIFIED
```

The validation checks:

- exact N=0 product is identity at the supplied absolute `v0`;
- after one macro step the position is `v0+t`;
- GPU projected ratios agree with exact `Fraction` arithmetic at small depths;
- a known pole trajectory is flagged.

`run_fixed_start_campaign.py` runs this gate automatically before every normal
campaign.

## 5. Phase A: identify a start point once

Choose one safe reference direction that converges from the start point.
Example shape:

```bash
python scripts/identify_fixed_start.py \
  --start 2,2,2,2,2,2,3,3,3,3,3 \
  --dir   1,0,0,0,0,0,0,1,0,0,0 \
  --pair 5,4 \
  --z-num -1 --z-den 1 \
  --depth 1000 --dps 160 \
  --target-expr '-9*zeta(3)/pi**2' \
  --out start_profile.json
```

The script evaluates N and 2N, runs the existing PSLQ battery, and stores a
`FIXED_START_PROFILE_V1` file. PSLQ is not repeated for every direction.

If the constant is not known yet, omit `--target-expr`. The 2N numerical value
can be used as the screening target. Once a PSLQ relation is confirmed, rerun
Phase A with its exact symbolic expression before computing exact delta.

## 6. Phase B: GPU funnel + exact CPU delta

A global direction search:

```bash
python scripts/run_fixed_start_campaign.py fixed_run \
  --profile start_profile.json \
  --n 1000000 \
  --dir-min -20 --dir-max 20 \
  --stages 128,256,512,1000 \
  --keep 200000,50000,10000,2000 \
  --exact-top 500 \
  --exact-workers 14 \
  --exact-depths 256,512,1000
```

A local optimization around a known direction:

```bash
python scripts/run_fixed_start_campaign.py fixed_local \
  --profile start_profile.json \
  --n 1000000 \
  --seed-dir 0,0,0,0,0,0,1,0,0,0,0 \
  --radius 4 \
  --dir-min -30 --dir-max 30 \
  --keep 200000,50000,10000,2000 \
  --exact-top 500 --exact-workers 14
```

### CPU preprocessing

`gen_fixed_directions.py`:

- rejects the zero vector;
- canonicalizes permutations only inside equal-start coordinate groups;
- rejects rays guaranteed to hit the axis denominator pole lattice through the
  maximum requested depth;
- optionally checks the shard recession condition `A @ t <= 0` from an NPZ;
- sorts directions by `max(abs(t))` and L1 cost to reduce Metal SIMD divergence.

For an optional shard file, store:

```python
np.savez("shard.npz", A=A, b=b)
```

`b` is optional for direction filtering. If present, the generator also checks
that `absolute_start` lies inside the supplied shard.

### GPU stages

Default depths are

```text
128 -> 256 -> 512 -> 1000.
```

Every stage starts from the same absolute `v0`. It is not a continuation from
the preceding stage's endpoint. Deeper stages rerun only the survivor
directions from the fixed start, which keeps semantics simple and auditable.

The GPU score combines:

- error to the already identified target;
- `N/2` versus `N` convergence;
- a projective exponent-growth proxy.

The resulting field is named `gpu_delta_proxy`. It is **not** exact delta.

### CPU exact stage

`refine_fixed_delta.py` re-walks the best directions using exact
`fractions.Fraction` arithmetic and computes the reduced approximant `p/q`.
For every requested depth,

```text
delta_N = -1 - log(|target - p/q|) / log(q).
```

Results include:

- `delta_by_depth`;
- `delta_exact` at the deepest tested N;
- `delta_tail_min`;
- `delta_slope_last`;
- `positive_at_deepest`;
- `positive_tail`.

A positive finite-depth delta is evidence, not a proof that
`limsup delta_N > 0`. The multi-depth sequence is retained specifically so the
best candidates can later be pushed to larger N.

## 7. Files produced

A campaign retains small, auditable survivor files:

```text
fixed_manifest.json
directions_initial.bin
directions_after_N0128.bin
directions_after_N0256.bin
directions_after_N0512.bin
directions_after_N1000.bin
GPU_SURVIVORS.jsonl
EXACT_DELTA.jsonl
```

Large per-stage GPU output binaries are deleted by default after survivor
selection. Use `--keep-stage-binaries` for debugging.

## 8. Hardware split on M5 Max 40-GPU / 18-CPU

Recommended division:

- **GPU**: the expensive CMF unit-axis walks at N=128/256/512/1000;
- **CPU**: direction generation, symmetry reduction, pole/shard filtering,
  survivor scoring, and exact Fraction delta refinement;
- exact refinement defaults to 14 workers, leaving several cores for the OS,
  I/O, and orchestration. Raise it only after benchmarking thermals and memory.

The CPU should not duplicate the bulk GPU walk. Its job is to make GPU work
scarce and useful, then restore exact arithmetic only for the tiny survivor
set.
