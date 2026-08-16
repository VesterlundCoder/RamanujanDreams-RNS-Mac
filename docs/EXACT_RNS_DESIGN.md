# DreamsRNS Metal/Mac port - full-matrix RNS edition

Correctness-first Metal port of the ROCm RNS walk for Apple silicon, specialized for the 6F5 workflow with complete 10x10 matrix reconstruction.

## What changed in this edition

The GPU no longer returns only one hard-coded projective ratio such as `P[1]/P[0]`.

For every requested checkpoint it retains the complete modular matrix:

```text
snapshots[checkpoint][prime][trajectory][100]
```

For a 10x10 6F5 matrix, CPU CRT reconstructs all 100 exact matrix elements at every checkpoint. The ratio scanner can therefore test any matrix-element combination:

```text
P[num_row,num_col] / P[den_row,den_col]
```

By default the scanner uses ordered pairs, so both `A/B` and `B/A` are available. That gives at most 100*99 = 9900 non-trivial directed ratios per surviving trajectory. `unordered` mode reduces inverse duplicates to 4950 pairs.

## Pipeline

1. Stream trajectory records in chunks. Never materialize 3e9 trajectories at once.
2. `encode_trajectories_rns` converts both initial position and direction to residues for every prime.
3. `walk_6f5_rns` executes the existing bytecode semantics and performs N=1000 matrix products modulo every prime.
4. The GPU writes the entire 10x10 product matrix at 2-4 checkpoints, e.g. N=900,950,1000.
5. CPU reconstructs every matrix element independently with Garner CRT.
6. CPU forms arbitrary projective limits `P[i]/P[j]` from the reconstructed matrices.
7. For every pair it computes checkpoint-to-checkpoint absolute and relative deltas.
8. Only ratios passing the chosen convergence threshold are emitted to TSV.
9. `scripts/pslq_worker.py` performs PSLQ against the constant library while retaining the matrix coordinates that generated the candidate.

## Output columns

The ratio TSV contains:

```text
trajectory_id
num_idx num_row num_col
den_idx den_row den_col
checkpoint_0 limit_0_200d
checkpoint_1 limit_1_200d
...
delta_abs_last
delta_rel_last
delta_rel_max
```

The final `limit_*_200d` column is the N=1000 value when the last checkpoint is 1000.

### Delta definitions

For two consecutive checkpoint limits L_old and L_new:

```text
delta_abs = |L_new - L_old|
delta_rel = |L_new - L_old| / max(1, |L_new|)
```

`delta_rel_last` compares the last two checkpoints. `delta_rel_max` is the maximum relative change across all adjacent checkpoint pairs. The definitions live in `src/ratio_scan.hpp`, so another research-specific delta functional can be substituted without changing the Metal kernel.

## GPU memory model

The main snapshot buffer is:

```text
nCheckpoints * K * chunkSize * 100 * sizeof(uint32_t)
```

Example with 3 checkpoints and K=16:

```text
~19.2 KB per trajectory in the chunk
```

Therefore chunk size must be chosen from available unified memory. For the 3e9 trajectory campaign, chunking is mandatory.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Run from the project root so `metal/rns_6f5.metal` can be loaded at runtime.

## Intended command line

```bash
./build/dreams_rns_metal \
  trajectories.bin \
  DIM \
  K \
  CHUNK_SIZE \
  ratios.tsv \
  1e-40 \
  900,950,1000 \
  ordered
```

Arguments after the input path are:

```text
dim
K
chunk_size
output.tsv
maximum accepted relative delta
comma-separated checkpoints
ordered|unordered
```

The last checkpoint must be 1000, ensuring the final full matrix is retained.

## Important integration point

`src/main.mm` still deliberately fails fast until two repository-specific hooks are connected:

```cpp
load6F5ProgramOrThrow(...)
loadPrimesOrThrow(...)
```

They must be populated from the existing 6F5 CMF/bytecode compiler and the same RNS prime set used to create `ProgramHost.constTable`. This prevents a placeholder matrix or mismatched prime/constants table from producing numerically plausible but scientifically invalid PSLQ candidates.

The Metal kernel itself already preserves the complete matrix and is agnostic to which matrix-element ratios are examined later.

## Trajectory binary layout

For each trajectory:

```text
int32 shift[dim]
int32 dir[dim]
```

Records are concatenated for sequential chunked I/O.

## Precision warning

CRT reconstruction is unique only if the product of the RNS moduli is large enough for the actual integer represented by every reconstructed matrix element, or if a mathematically justified reconstruction bound is used.

This matters even more in full-matrix mode: the capacity condition must hold for all matrix elements that enter the ratio scan. N=1000 products can require substantially more than a small set of 31-bit primes.

## Scaling warning for 3e9 trajectories

Reconstructing 100 elements at multiple checkpoints and then evaluating up to 9900 ratios per trajectory is intentionally exhaustive but extremely expensive. The current implementation is a correctness/reference implementation.

For production scale, the recommended next optimization is a two-stage filter:

1. GPU computes cheap modular or approximate convergence indicators for candidate matrix pairs.
2. CPU performs full CRT only for trajectories/pairs that survive this prefilter.

This preserves the ability to discover limits from any matrix-element combination without paying full BigInt CRT + 9900-ratio cost on every one of 3e9 trajectories.

## Files

```text
metal/rns_6f5.metal       complete RNS matrix walk + checkpoint snapshots
src/main.mm               Metal host, chunking, full CRT reconstruction
src/crt.hpp               Garner CRT + 200-digit ratio conversion
src/ratio_scan.hpp        arbitrary element-pair limits and delta calculations
scripts/pslq_worker.py    PSLQ retaining matrix coordinates
include/metal_rns.hpp     shared host/kernel configuration structs
```
