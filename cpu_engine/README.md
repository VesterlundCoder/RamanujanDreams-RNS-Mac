# Dreams CPU Engine v0.1.1

Standalone C CPU backend and benchmark suite for the fixed-start 6F5 CMF workflow on Apple Silicon.

The CPU path is a semantic port of `metal/walk_6f5_fixed_start_df64.metal`: absolute `v0`, first operator evaluated at `n=0`, selected two-row propagation, df64 arithmetic, common power-of-two normalization, two checkpoints and `FSDF6401` output.

## Build

```bash
cd cpu_engine
make -j
make test
```

The v0.1.1 branch includes the macOS compatibility fix for `sysconf(_SC_NPROCESSORS_ONLN)`: `<unistd.h>` is included and `_DARWIN_C_SOURCE` is enabled on Apple platforms.

Binaries:

- `build/dreams_fixed_start_cpu`
- `build/matmul6_bench`
- `build/rns_matmul6_bench`

## End-to-end example

```bash
build/dreams_fixed_start_cpu \
  directions.bin cpu.bin \
  "0,0,0,-2,-2,0,0,-1,-1,-2,-2" \
  5 4 -1 1 1000 262144 18 32
```

Arguments after the row pair are `z_num z_den nSteps chunk threads grain`.

Inspect output:

```bash
python3 scripts/inspect_fsdf64.py cpu.bin --show 5
```

## CPU vs Metal benchmark

From `cpu_engine/` inside this repository:

```bash
scripts/benchmark_cmf.sh \
  ../directions.bin \
  "0,0,0,-2,-2,0,0,-1,-1,-2,-2" \
  5 4 -1 1 1000
```

Override thread counts when desired:

```bash
THREAD_SET="1 6 18" scripts/benchmark_cmf.sh ...
```

The script writes logs, `FSDF6401` outputs, a CPU-vs-Metal numerical comparison and `summary.csv` under `results/`.

## Microbenchmarks

FP32 6x6:

```bash
build/matmul6_bench --n 1000000 --repeats 10 --threads 18 --csv results/matmul6.csv
```

Exact modular 6x6:

```bash
build/rns_matmul6_bench --pairs 200000 --steps 100 --threads 18 --csv results/rns.csv
```

## First M5 results

See [`BENCHMARK_M5.md`](BENCHMARK_M5.md). The main measured findings are:

- 14,885 wall dir/s on one CPU thread;
- 83,877 wall dir/s on 6 P-core threads;
- 152,762 wall dir/s on 18 threads for 100k directions;
- 103,672 wall dir/s sustained over 1M directions;
- Metal is about 4x faster than CPU-18 on the sustained 1M run;
- CPU and Metal produced 100% bitwise-identical `FSDF6401` records for the 100k parity test;
- AoS + unrolled + pthreads beat the explicit SoA/NEON 6x6 microkernel;
- the production fixed-start bottleneck is theta/control-flow work rather than generic 6x6 multiplication.

## 2x6 specialization

The fixed-start optimizer does not need to propagate a full 6x6 product when the numerator and denominator rows are known in advance. For right multiplication, each row evolves independently, so the selected 2x6 state is exactly the same pair of rows that a full 6x6 product would produce.

This is an exact specialization for the selected pair, not an approximation. It deliberately loses the other four rows and therefore cannot support a post-hoc full-matrix census or full-matrix invariants. See [`TWO_ROW_PROJECTION.md`](TWO_ROW_PROJECTION.md).

## v0.2

The next optimization program is in [`CPU_ENGINE_V0_2_PROMPT.md`](CPU_ENGINE_V0_2_PROMPT.md). Its priorities are profiling, predecoded direction schedules, incremental theta/polynomial state, persistent hot trajectory state, sustained-throughput instrumentation and conservative RNS locality experiments. Scientific and numerical semantics are regression-gated.

## Scientific roles

Recommended pipeline:

1. full matrix / sufficiently wide projection for discovery and broad ratio census;
2. 2x6 df64 CPU/Metal for fixed-start ratio optimization and screening;
3. RNS/CRT or multiprecision for exact finalists and PSLQ validation.

The df64 CPU backend is not an exact-reconstruction backend. Exact reconstruction still requires the RNS/CRT or multiprecision path.
