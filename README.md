# RamanujanDreams-RNS-Mac

GPU-accelerated search for new mathematical constants from the 6F5
continued-fraction matrix field (CMF), built for Apple silicon.

**CORRECTED CMF SEMANTICS EDITION.** The walk implements the true
Ramanujan Dreams trajectory operator (upstream `ramanujantools`
`Matrix.walk` semantics), not a direct companion product:

- theta operator `D(θ) = θ·Π(θ+y_j−1) − z·Π(θ+x_i)` with **monic**
  companion column `c_i = −d_i/d_rank` (leading coefficient `1−z`)
- **rank 5 at z=1**, rank 6 otherwise
- axis operators `I + C(v)/a` (and their inverses for negative steps),
  applied row-wise in O(r²) without any dense inversion
- every direction decomposed into **unit axis steps**
  (level max|t|→1, axis 10→0), start point `x_i=shift_i+1, y_j=shift_{6+j}+2`
- per-unit-step exact power-of-two normalization (zero rounding error)
- singular / near-singular paths are **flagged** (status codes) and
  routed to an exact CPU fallback queue — never silently wrong

Everything runs in **df64 double-single precision** (2×float32 with
exact FMA, ~48-bit mantissa), one thread per trajectory. The full
rank×rank product matrix at N=1000 plus exponent and status is stored
per trajectory (`DF64CMF2` format).

The CPUs run in parallel: limits are extracted as last-column ratios,
matched against a ~223,000-entry constant library (rational and
near-rational trap entries excluded), and every hit is re-derived with
the **corrected high-precision walk** (mpmath, 120 digits) and pushed
through a full **PSLQ battery**.

## Verification (hard gate, runs before every campaign)

`scripts/validate_cmf_df64.py` compares the full projective GPU matrix
against an **exact-rational oracle** (`lib/cmf_walk_corrected.py`,
`fractions.Fraction` arithmetic) for golden tests A–G at
N ∈ {1,2,3,5,10,20}: numerator ±1, denominator +1, mixed diagonal,
z ∈ {1/2, 1/3, −1}, z=1 (rank 5), and a singular path that must be
flagged identically by GPU and oracle. Typical parity ~1e-13; the gate
is conditioning-aware (an ideal float64 walk of the same algorithm sets
the achievable accuracy per trajectory). 48/48 checks pass.

## Measured throughput (MacBook Pro, M5 Max, 40-core GPU, 18-core CPU)

The corrected operator costs ~Σ|dir| unit steps per trajectory step
(vs 1 matrix multiply in the old direct product), so it is inherently
~100–200× more expensive:

| Stage | Rate (corrected walk) |
|-------|----------------------|
| GPU CMF walk, N=1000, dirs ∈ [0,20] | **~6.5k traj/s** (0.35 OK-fraction; flagged paths exit early) |
| CPU limit matching (14 workers) | ~70–130k traj/s |
| PSLQ verification (12 workers) | ~2 unique limits/min (mpmath depth-1000 corrected re-walk) |

For reference, the RETIRED direct-companion walk ran at 1.05–1.25M
traj/s wall — but computed the wrong operator. A 12-hour campaign now
covers ~250–300M canonical trajectories.

NOTE on the census space: with the fixed all-ones start
(v0 = 2,…,2;3,…,3) any negative direction component walks into the pole
lattice within ~2 steps, so the deep fixed-start census enumerates
**non-negative canonical directions** (dir ∈ [0,20]:
C(26,6)·C(25,5) ≈ 1.2e10 canonical trajectories). Negative components
require large starting shifts (with (shift,dir)-pair canonicalization) —
planned follow-up. z=1 (rank 5) is heavily lead-degenerate for mixed
direction sums; campaigns default to z=1/2.

## Layout

```
metal/walk_6f5_cmf_df64.metal  GPU kernel: CORRECTED CMF axis walk (df64)
src/main_cmf_df64.mm           host: DF64CMF2 output, rank, status field
lib/cmf_walk_corrected.py      corrected reference oracle (Fraction + mpmath)
lib/limit_library.py           ~223k-entry float64 constant library (+ cache)
lib/pslq_companion.py          PSLQ battery (mpmath)
lib/cmf_generic.py             RETIRED direct-companion walk (warning inside)
scripts/gen_canonical.py       canonical trajectory enumerator (resumable)
scripts/validate_cmf_df64.py   golden parity gate (GPU vs exact oracle)
scripts/match_limits.py        CPU: rank/status-aware library matching
scripts/pslq_hits.py           CPU: corrected high-precision walk + PSLQ
scripts/run_campaign.py        orchestrator with mandatory parity gate
metal/walk_6f5_df64.metal      legacy direct-product kernel (retired)
src/main.mm                    (optional) exact-RNS/CRT edition, see docs/
```

## Build

Requires macOS with Xcode command-line tools and CMake. Python side
needs `numpy` and `mpmath`.

```bash
cmake -S . -B build
cmake --build build          # -> build/dreams_rns_cmf_df64
```

(The boost-dependent exact-RNS binary is behind `-DBUILD_EXACT_RNS=ON`.)

## Quick start

```bash
# hard parity gate: GPU vs exact-rational oracle (must print
# "CMF SEMANTICS VERIFIED")
python scripts/validate_cmf_df64.py

# 1M canonical trajectories, corrected walk to N=1000 at z=1/2
python scripts/gen_canonical.py /tmp/run --count 1000000
./build/dreams_rns_cmf_df64 /tmp/run/traj_chunk_00000.bin \
    /tmp/run/matrices_chunk_00000.bin 1 2 1000

# CPU pipeline on the stored matrices
python scripts/match_limits.py /tmp/run/matrices_chunk_00000.bin \
    /tmp/run/traj_chunk_00000.bin --workers 14
python scripts/pslq_hits.py /tmp/run/matrices_chunk_00000.bin.hits.jsonl \
    --workers 12
```

## Campaign mode

```bash
python scripts/run_campaign.py ~/campaign_cmf \
  --max-hours 12 --chunk-size 25000000 \
  --delete-traj --delete-matrices
```

The campaign REFUSES to start unless the golden parity gate passes
(`FATAL: CMF parity failure. Campaign aborted.`). Per chunk: canonical
generation (background, overlaps the GPU) → GPU walk → CPU matching
(background, overlaps the next GPU chunk) → PSLQ on hits (background). The live list of PSLQ-confirmed relations is kept in
`<outdir>/PSLQ_HITS.md` and `<outdir>/PSLQ_CONFIRMED.jsonl`; the
resumable cursor lives in `<outdir>/manifest.json` — rerunning the
command continues exactly where the last campaign stopped, never
repeating a trajectory.

## The mathematics of the funnel

**Canonicalization.** The theta operator depends only on the multiset
of the 6 numerator roots and the multiset of the 5 denominator roots,
so with fixed shift `[1]*11` any direction vector is equivalent to one
with both blocks sorted (6!·5! = 86,400× reduction). When starting
points vary, the pairs `(shift_i, dir_i)` must be canonicalized as
multisets instead. f-multisets are visited in a bijectively shuffled
order so every chunk samples the space broadly; the cursor in
`manifest.json` guarantees full coverage without repeats.

**Convergence gate.** A converged walk has a rank-1 product matrix, so
the last-column ratio `W[i,r-1]/W[j,r-1]` must agree with column r-2.
Ratios failing this cross-check are discarded — a delta/convergence
measurement taken from a single stored checkpoint.

**Library match.** Converged ratios are binary-searched against the
sorted constant library. Library entries that are rational-valued or
within 1e-5 of a small rational (traps like `zeta(20)+7 ≈ 8.0000009`)
are excluded; limits that are themselves small rationals are rejected
via continued fractions.

**Corrected verification + PSLQ.** Each unique hit value is re-derived
with the corrected high-precision walk (mpmath, 120 digits) at depths N
and 2N, cross-checked against the GPU float match, rational limits
rejected exactly, then run through the PSLQ battery (rational,
quadratic, cubic, single-constant, pi-polynomial relations, coefficients
up to 1e5). Only PSLQ-confirmed relations reach the final list.

## Output format `DF64CMF2`

```
header : magic "DF64CMF2" | u32 max_dim=6 | u32 rank (5|6)
         | u32 nshift=11 | u32 nSteps | i32 z_num | i32 z_den | u64 nTraj
record : 36 × (f32 hi, f32 lo) | i32 expSum | u32 status   (296 B/traj)
status : 0=OK 1=ZERO_AXIS_DENOMINATOR 2=THETA_LEAD_DEGENERATE
         3=INVERSE_SINGULAR 4=NONFINITE 5=NEEDS_REGULARIZATION
```

Matrix element value = `(hi + lo) · 2^expSum`. Ratios are independent of
the normalization. The old `DF64MAT1` format (direct companion product)
is intentionally incompatible so stale data can never enter the
corrected PSLQ pipeline.

## Why df64 instead of exact RNS/CRT?

Exact reconstruction of an N=1000 product needs ≥ 6·log2(1000!) ≈ 51,000
bits per matrix element, i.e. ~1,700–2,500 31-bit primes per trajectory
— a ~2000× compute and memory multiplier. The df64 walk gets ~14
significant digits per unit step for the price of one lane — enough to
screen limits; exactness is then restored only for the rare hits by the
corrected high-precision CPU stage. The original exact-RNS design is
preserved in `docs/EXACT_RNS_DESIGN.md` and `src/main.mm`.
