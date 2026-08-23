# Development prompt: Dreams CPU Engine v0.2

You are upgrading the Apple-Silicon CPU backend in `VesterlundCoder/RamanujanDreams-RNS-Mac`, branch `cpu-engine-v0.1-m5-benchmarks`.

The goal is to improve **sustained end-to-end fixed-start 6F5 CMF throughput** without changing mathematical semantics, df64 arithmetic semantics, N=0 start semantics, status behavior or the `FSDF6401` output contract.

## Non-negotiable scientific invariants

1. `absolute_start` is the exact CMF coordinate `v0`.
2. The first trajectory operator is evaluated at `n=0`, exactly at `v0`.
3. A depth-1000 run applies 1000 macro trajectory operators.
4. Preserve rank handling, axis order and direction semantics exactly.
5. Preserve the two selected rows exactly as the corresponding rows of the full 6x6 right-product.
6. Preserve common power-of-two normalization across both rows.
7. Do not enable fast-math for the df64 CMF path.
8. Preserve all exceptional statuses and singularity thresholds.
9. CPU v0.2 must remain bitwise identical to CPU v0.1 and the Metal fixed-start output on the existing 100k parity corpus unless a numerical change is explicitly isolated, justified and approved.
10. Do not trade numerical correctness for speed.

## Measured v0.1 baseline on M5

End-to-end, 100k directions, depth 1000, `z=-1`, rank 6, pair `(5,4)`:

- CPU 1t: 14,885 wall dir/s; 14,973 compute dir/s
- CPU 6 P-core threads: 83,877 wall; 86,831 compute
- CPU 18t: 152,762 wall; 164,799 compute
- Metal: 264,809 wall; 277,007 compute

Sustained 1M:

- CPU 18t: 103,672 wall; 108,889 compute
- Metal: 412,265 wall; 450,860 compute

100k CPU/Metal parity: 100% byte-identical records.

6x6 FP32 microbenchmark:

- AoS scalar 1t: 16.5 Mmat/s
- AoS unrolled 1t: 23.4 Mmat/s
- AoS unrolled pthreads: 255.6 Mmat/s, 110.4 GFLOP/s
- SoA scalar pthreads: 53.8 Mmat/s
- SoA NEON4 pthreads: 142.6 Mmat/s, 61.6 GFLOP/s

RNS:

- `%` 1t: 4.5 Mmat/s
- `%` pthreads: 32.1 Mmat/s
- Barrett pthreads: 32.7 Mmat/s

These measurements imply that generic 6x6 multiplication is not the fixed-start bottleneck and that explicit SoA/NEON must not replace the winning AoS layout without new evidence.

## Phase 0: profile before rewriting

Instrument or sample-profile the existing v0.1 production path first. Attribute time to at least:

- theta polynomial construction / companion-column formation;
- forward row updates;
- inverse row updates and df64 division;
- power-of-two normalization;
- trajectory schedule/control-flow interpretation;
- pthread scheduling / chunk boundaries;
- output serialization.

Do not accept an optimization hypothesis solely from microbenchmarks. Every production change must be tied to a measured end-to-end hotspot.

## Primary optimization target: incremental theta state

The current `theta_companion_column()` reconstructs

`Px(theta) = product_i (theta + x_i)`

and

`Py(theta) = product_j (theta + y_j - 1)`

from scratch for every axis step, although an axis step changes only one coordinate. Implement experimental per-trajectory polynomial state so a single coordinate update can update only the relevant polynomial.

Benchmark at least two strategies.

### Strategy A: synthetic division + factor replacement

For `(theta+a) -> (theta+b)`:

1. synthetically divide the current polynomial by `(theta+a)` to obtain `Q`;
2. form `P_new = Q*(theta+b)`;
3. update only the affected x or y polynomial;
4. derive the companion coefficients from the updated polynomial pair.

For common `b=a+1`, investigate `P_new = P_old + Q`.

Important: these formulas are algebraically exact but may reorder df64 operations. Keep the existing slow theta builder as the reference. If an incremental variant is not bitwise identical, do not silently make it default. Report the divergence and its speed/numerical behavior separately.

### Strategy B: cached leave-one-out products

Maintain enough low-degree polynomial state to replace one linear factor without rebuilding all factors. Compare instruction count, cache footprint and end-to-end throughput against Strategy A and the v0.1 builder.

## Persistent hot trajectory state

Refactor experimental kernels around a compact state containing at least:

- `pos[11]`;
- selected `R[2][6]`;
- current x-polynomial coefficients;
- current y-polynomial coefficients;
- reusable companion data;
- normalization exponent;
- status.

Avoid rebuilding values unchanged between axis moves. Keep hot state stack/register/cache friendly.

## Predecode the trajectory macro-step schedule

The direction vector is constant during all 1000 macro steps, yet v0.1 repeatedly executes the nested `level`/`axis` tests. Build the ordered `(axis,sign)` operation list once per trajectory or direction pattern and replay exactly the same order for each macro step.

This is a priority because it can reduce branch/control overhead without altering floating-point operation order inside an axis update.

Provide at least:

- `generic_v01` reference schedule interpreter;
- `predecoded_schedule` implementation;
- a fast path for the common `max_abs == 1` case if present in production data.

## Direction-pattern grouping

Directions are discrete and may share identical active-axis/sign/level patterns. Add an optional preprocessing key and benchmark:

1. current ungrouped generic interpreter;
2. grouped trajectories using the same generic kernel;
3. a small number of specialized hot-pattern kernels only if profiling justifies them.

Do not build a large JIT/code-generation system in v0.2.

## Worker-pool and chunk lifetime

v0.1 creates/join pthread workers once per input chunk. Add an experimental persistent worker pool whose threads live across chunks. Benchmark whether removing repeated pthread creation and improving hot-cache state matters for 100k, 1M and larger sustained runs.

Preserve dynamic work stealing / atomic-grain behavior unless measured evidence supports another policy.

## P/E-core-aware scheduling

Report Apple-Silicon topology and test at least:

- 1 thread;
- 6 P-core threads on the benchmark M5 configuration;
- all 18 threads.

Investigate dynamic grain sizing and, if practical, different work-grain policies for fast and slow core classes. Do not depend on undocumented hard pinning. Any QoS/affinity experiment must remain optional.

## Sustained-throughput instrumentation

The 100k and 1M results differ substantially. Add a benchmark mode reporting fixed windows, e.g. every 50k or 100k completed directions:

- cumulative wall dir/s;
- window wall dir/s;
- cumulative compute dir/s;
- elapsed time;
- resident memory;
- chunk number and size;
- optional Apple power/thermal indicators if available through supported tooling.

Run at least 100k, 250k, 500k, 1M, 2M and 5M at multiple chunk sizes. Do not call the observed decline thermal throttling until measurements distinguish thermal effects from I/O, allocation, chunking and memory pressure.

## RNS v0.2 scope

Keep hardware `%` as the reference/default because Barrett was only about 2% faster. Do not spend the main optimization budget on another modular-reduction formula. Benchmark instead:

- trajectory-major vs prime-major traversal;
- cache blocking across `(trajectory,prime)`;
- persistent workers;
- reduced temporary memory traffic;
- batches chosen for cache locality.

Any RNS change must remain residue-identical to the reference.

## 2x6 projection constraints

The fixed-start optimizer intentionally propagates only the selected numerator and denominator rows. Preserve this. For selector `S` and full product `P_n`, `R_n=S P_n` obeys `R_{n+1}=R_n A_n`, so the selected rows are algebraically exact under right multiplication.

Do not generalize performance claims from this fixed-pair 2x6 backend to a full 6x6 discovery census. The 2x6 state cannot recover the four omitted rows, cannot choose a different row pair after the run and cannot provide general full-matrix invariants.

## Required full-matrix reference oracle

Add a deliberately slow df64 reference mode that propagates the complete 6x6 product with the same axis order and numerical primitives. On a representative regression subset, verify that the selected two rows from the full 6x6 state match the 2x6 projected state at every requested checkpoint.

Keep this structural regression test separate from performance timing. Also add a test demonstrating the information boundary: 2x6 may reproduce the fixed pair exactly but must not claim to recover non-selected rows, full-matrix invariants or arbitrary post-hoc row pairs.

## Required benchmark matrix

For every candidate optimization report at minimum:

- 1 thread;
- 6 P-core threads;
- 18 threads;
- 100k directions;
- 1M directions;
- wall dir/s;
- compute dir/s;
- speedup vs v0.1;
- CPU/Metal ratio;
- parity result.

Also include per-operation profiling or sampled profiles so each claimed improvement maps to a measured hotspot.

## Acceptance gates

An optimized path may become default only if:

1. status parity is 100% on the regression corpus;
2. exponent parity is 100%;
3. df64 component bitwise parity is 100%, preferably over the full 100k parity corpus;
4. N=0 semantics are unchanged;
5. full-6x6 vs selected-2x6 structural regression passes on the reference subset;
6. the change produces at least 5% reproducible end-to-end speedup on a relevant production corpus, or is an explicitly documented enabling change;
7. there is no material sustained-throughput regression at 1M.

If an algebraically valid optimization changes df64 rounding/order and fails bitwise parity, keep it experimental, quantify the numerical difference, and do not merge it into the default scientific path without explicit review.

## Stretch target

Try to move 18-thread sustained CPU throughput from about 104k dir/s toward 150k-200k dir/s without changing numerical behavior. Treat this as a stretch target, not a promised result. If reached, the sustained Metal/CPU ratio would move from roughly 4x toward roughly 2-3x.

## Deliverables

Produce:

- updated C sources under `cpu_engine/`;
- a persistent-pool and predecoded-schedule variant;
- unit/differential tests for any incremental polynomial replacement;
- the slow theta reference retained for testing;
- a slow full-6x6 df64 reference oracle for selected-row parity;
- parity regression against v0.1 and Metal;
- sustained benchmark CSV output;
- profiler notes with before/after hotspots;
- `BENCHMARK_V0_2.md` containing successful and failed optimization attempts;
- no silent changes to scientific semantics.
