# Porting notes: Metal fixed-start df64 -> CPU C

The end-to-end CPU path intentionally mirrors `walk_6f5_fixed_start_df64`.

Matched behavior:

- `start[11]` is the absolute CMF coordinate `v0`;
- no implicit translation and no pre-step;
- the first macro operator is evaluated at `n=0`, exactly at `v0`;
- rank is 5 only when `z_num == z_den`, otherwise rank 6;
- two selected rows are propagated rather than the full 6x6 product;
- axis application order is level descending, then axis 10 down to 0;
- forward/inverse axis semantics match Metal;
- theta/companion construction, df64 operations, singularity thresholds and common power-of-two normalization use the same semantics;
- checkpoints are `max(1,nSteps/2)` and `nSteps`;
- output is layout-compatible `FSDF6401`.

## Floating-point parity

CPU and GPU are independent execution engines. Scientific correctness should ultimately be grounded by mathematical/exact or high-precision checks. For regression engineering, however, the measured 100k M5 corpus achieved 100% bitwise CPU/Metal parity, so v0.2 treats that as a strong gate unless a numerical change is explicitly isolated and justified.

The CMF target is compiled with `-fno-fast-math -ffp-contract=off`; explicit `fmaf()` calls inside df64 multiplication remain intentional.

## Why the 6x6 microbenchmark is separate

The fixed-start optimizer no longer performs a generic full 6x6 GEMM at each axis step. It propagates two selected rows because rows evolve independently under right multiplication. The standalone 6x6 benchmark asks a separate systems question about batched tiny-matrix CPU throughput.

## Exact RNS limitation

The public exact-RNS host on `main` still contains deliberate fail-fast stubs for the 6F5 bytecode and prime loaders. The included `rns_matmul6_bench` therefore isolates exact modular matrix arithmetic and must not be presented as an end-to-end exact-RNS comparison. Once the loaders are wired, the same interpreter/tables can be ported to a full CPU exact-RNS path.
