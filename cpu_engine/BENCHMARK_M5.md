# M5 CPU Engine Benchmark Results

Date: 2026-08-23

Configuration: fixed-start 6F5 CMF, depth 1000, `z=-1`, rank 6, ratio pair `(5,4)`.

## End-to-end, 100k directions

| Backend | Threads | Wall dir/s | Compute dir/s | Speedup vs Python |
|---|---:|---:|---:|---:|
| Python Fraction | 1 | 9.4 | - | 1x |
| C CPU | 1 | 14,885 | 14,973 | 1,585x |
| C CPU | 6 P-cores | 83,877 | 86,831 | 8,923x |
| C CPU | 18 all cores | 152,762 | 164,799 | 16,251x |
| Metal GPU | GPU | 264,809 | 277,007 | 28,175x |

From 1 to 6 P-core threads, speedup is `5.64x`, corresponding to about 94% nominal parallel efficiency. From 1 to all 18 cores, wall speedup is `10.26x`. The latter must not be interpreted as homogeneous-core efficiency because the M5 CPU contains multiple performance classes. Relative to the 6-thread result, the additional 12 cores add 68,885 dir/s, about 5,740 dir/s per added core, roughly 41% of average P-core throughput.

## Sustained, 1M directions

| Backend | Wall dir/s | Compute dir/s |
|---|---:|---:|
| C CPU 18t | 103,672 | 108,889 |
| Metal GPU | 412,265 | 450,860 |

The sustained Metal/CPU wall-throughput ratio is about `3.98x`.

The CPU throughput decline between the 100k and 1M tests is observed, but its cause is not established. Thermal throttling and memory pressure are hypotheses. v0.2 should measure fixed time/work windows and vary chunk size before assigning causality.

## Numerical parity

For the 100k comparison:

- 100,000 / 100,000 records bitwise identical;
- 0 status mismatches;
- 0 exponent mismatches;
- 0 df64 component differences.

This is strong implementation parity between CPU and Metal for the tested workload. It is not, by itself, a proof of mathematical correctness; a high-precision or exact third oracle remains useful on a representative subset.

## 6x6 FP32 microbenchmark

1,000,000 matrices x 10 repeats:

| Backend | Mmat/s | GFLOP/s |
|---|---:|---:|
| `aos_scalar_1t` | 16.5 | 7.1 |
| `aos_unrolled_1t` | 23.4 | 10.1 |
| `aos_unrolled_pthreads` | 255.6 | 110.4 |
| `soa_scalar_pthreads` | 53.8 | 23.2 |
| `soa_neon4_pthreads` | 142.6 | 61.6 |

The AoS + fixed unrolling + pthread backend is about `1.79x` faster than the explicit SoA/NEON-over-trajectories backend. Do not force a GPU-like SoA layout into the production CMF kernel unless a later profile demonstrates a specific gain.

The fixed-start production kernel is not dominated by generic 6x6 multiplication. The main optimization targets are theta/companion construction, axis-dependent control flow, forward/inverse row transforms, normalization, scheduling and memory locality.

## RNS modular 6x6

200,000 pairs x 100 steps:

| Backend | Mmat/s | Gmodmul/s |
|---|---:|---:|
| `rns_div_1t` | 4.5 | 0.97 |
| `rns_div_pthreads` | 32.1 | 6.93 |
| `rns_barrett_pthreads` | 32.7 | 7.07 |

Barrett improves only about 2% over hardware `%` on this Apple-Silicon workload. The larger result is roughly 7x pthread scaling. RNS v0.2 should prioritize traversal order, cache locality and batching rather than another reduction trick.

## Current conclusions

1. The C CPU engine is a viable production screening backend, not only a correctness fallback.
2. Metal remains about 4x faster sustained on this particular fixed-start workload.
3. Trajectory-level pthread parallelism scales very well on the P-core cluster.
4. Explicit SoA/NEON is not automatically beneficial for tiny 6x6 products.
5. Generic 6x6 GEMM is not the main fixed-start bottleneck.
6. CPU and Metal fixed-start outputs are byte-identical for the 100k parity run.
7. df64 screening must not be described as exact reconstruction; exact finalists still belong on the RNS/CRT or multiprecision path.
