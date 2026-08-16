# RamanujanDreams-RNS-Mac

GPU-accelerated search for new mathematical constants from the 6F5
continued-fraction matrix field (CMF), built for Apple silicon.

One thread = one trajectory. The GPU multiplies 1000 companion matrices
per trajectory in **df64 double-single precision** (2×float32 with exact
FMA, ~48-bit mantissa ≈ float64-class) with an **exact power-of-two
normalization every step**, so nothing ever overflows and the scaling
introduces zero rounding error. The full 6×6 product matrix at N=1000 is
stored per trajectory together with the accumulated binary exponent.

The CPUs then run in parallel with the GPU: limits are extracted as
last-column ratios, matched against a ~223,000-entry constant library,
and every hit is re-derived with **exact big-integer arithmetic** and
pushed through a full **PSLQ battery** at 120 digits.

## Measured throughput (MacBook Pro, M5 Max, 40-core GPU, 18-core CPU)

| Stage | Rate | 100M-trajectory chunk |
|-------|------|----------------------|
| GPU df64 walk, N=1000 | 1.05–1.25M traj/s wall, **2.9–3.3M traj/s on-GPU** | 94 s |
| CPU limit matching (14 workers) | **1.7–1.8M traj/s** (1.4G converged ratios tested) | 58 s |
| PSLQ verification (12 workers) | ~23 candidate limits/s | seconds |
| Full campaign cadence (all stages overlapped) | ~40–50M traj/min sustained | ~2.1–2.5 min |

A 12-hour overnight campaign covers **~30 billion canonical
trajectories**, which is ~2.6 quadrillion raw trajectories after the
86,400× canonical symmetry reduction (see below).

Accuracy: the GPU df64 matrices agree with a numpy float64 reference to
≤ 6.5e-13 relative and with the exact big-integer walk to ~1e-13.

## Layout

```
metal/walk_6f5_df64.metal   GPU kernel: df64 companion walk + normalization
src/main_df64.mm            host: chunked I/O, dispatch, DF64MAT1 output
src/main.mm                 (optional) original exact-RNS/CRT edition, see docs/
lib/limit_library.py        ~223k-entry float64 constant library (+ .npz cache)
lib/cmf_generic.py          exact integer 6F5 companion matrices
lib/pslq_companion.py       exact delta/limit + PSLQ battery (mpmath)
scripts/gen_canonical.py    canonical trajectory enumerator (resumable)
scripts/gen_trajectories.py simple random trajectory generator
scripts/match_limits.py     CPU: library matching on stored matrices
scripts/pslq_hits.py        CPU: exact verification + PSLQ on hits
scripts/run_campaign.py     orchestrator: chunks of 100M/1B, overlapped stages
scripts/validate_df64.py    GPU vs numpy float64 validation
```

## Build

Requires macOS with Xcode command-line tools and CMake. Python side
needs `numpy` and `mpmath`.

```bash
cmake -S . -B build
cmake --build build          # -> build/dreams_rns_df64
```

(The boost-dependent exact-RNS binary is behind `-DBUILD_EXACT_RNS=ON`.)

## Quick start

```bash
# 1M random trajectories, walk to N=1000 at z=1, validate
python scripts/gen_trajectories.py /tmp/traj.bin --n 1000000
./build/dreams_rns_df64 /tmp/traj.bin /tmp/mat.bin 1 1 1000
python scripts/validate_df64.py /tmp/traj.bin /tmp/mat.bin

# CPU pipeline on the stored matrices
python scripts/match_limits.py /tmp/mat.bin /tmp/traj.bin --workers 14
python scripts/pslq_hits.py /tmp/mat.bin.hits.jsonl --workers 12
```

## Campaign mode

```bash
python scripts/run_campaign.py ~/campaign \
  --max-hours 12 --chunk-size 100000000 \
  --delete-traj --delete-matrices
```

Per chunk: canonical generation (background, overlaps the GPU) → GPU
walk → CPU matching (background, overlaps the next GPU chunk) → PSLQ on
hits (background). The live list of PSLQ-confirmed relations is kept in
`<outdir>/PSLQ_HITS.md` and `<outdir>/PSLQ_CONFIRMED.jsonl`; the
resumable cursor lives in `<outdir>/manifest.json` — rerunning the
command continues exactly where the last campaign stopped, never
repeating a trajectory.

## The mathematics of the funnel

**Canonicalization.** The 6F5 companion matrix depends only on the
multiset of the 6 numerator roots and the multiset of the 5 denominator
roots, so with fixed shift `[1]*11` any direction vector is equivalent
to one with both blocks sorted. Enumerating only canonical
representatives shrinks the search space 6!·5! = 86,400×. The canonical
space over direction components in [-20, 20] has
C(46,6)·C(45,5) ≈ 1.14e13 trajectories; f-multisets are visited in a
bijectively shuffled order so every chunk samples the space broadly.

**Convergence gate.** A converged walk has a rank-1 product matrix, so
the last-column ratio `P[i,5]/P[j,5]` must agree with `P[i,4]/P[j,4]`.
Ratios failing this cross-check at 1e-10 are discarded — this is a
delta/convergence measurement taken from a single stored checkpoint.

**Library match.** Converged ratios are binary-searched against the
sorted constant library at 1e-11 relative tolerance. Limits that are
themselves small rationals (denominator ≤ 1e4) are rejected via
continued fractions — many composite library entries are secretly
rational and would otherwise flood the hit list.

**Exact verification + PSLQ.** Each unique hit value is re-derived with
a pure big-integer companion walk to depth 1000 (giving the exact
convergent and irrationality-measure delta), cross-checked against the
float match at 1e-9, then run through the PSLQ battery (rational,
quadratic, cubic, single-constant, pi-polynomial relations, coefficients
up to 1e5) at 120 decimal digits. Only PSLQ-confirmed relations reach
the final list.

## Output format `DF64MAT1`

```
header : magic "DF64MAT1" | u32 dim=6 | u32 nshift=11 | u32 nSteps
         | i32 z_num | i32 z_den | u64 nTraj
record : 36 × (f32 hi, f32 lo)  |  i32 expSum        (292 B/trajectory)
```

Matrix element value = `(hi + lo) · 2^expSum`. Ratios `P[i]/P[j]` are
independent of the normalization; `expSum` recovers the exact magnitude
(growth is typically ~14,000 bits at N=1000).

## Why df64 instead of exact RNS/CRT?

Exact reconstruction of an N=1000 product needs ≥ 6·log2(1000!) ≈ 51,000
bits per matrix element, i.e. ~1,700–2,500 31-bit primes per trajectory
— a ~2000× compute and memory multiplier. The df64 walk gets ~14
significant digits for the price of one lane, which is exactly enough to
screen limits at 1e-11; exactness is then restored only for the rare
hits by the big-integer CPU stage. The original exact-RNS design is
preserved in `docs/EXACT_RNS_DESIGN.md` and `src/main.mm`.
