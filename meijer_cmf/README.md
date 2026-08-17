# MeijerG(4,2,4,4,1) CMF on Metal

GPU pipeline for the MeijerG(4,2,4,4,1) conservative matrix field
(zeta(2)+zeta(3) trajectory search). Same two-stage philosophy as the 6F5
pipeline in this repo: a cheap float screening walk on GPU for millions of
candidates, exact RNS/CRT arithmetic only for survivors.

Unlike 6F5 (companion matrix rebuilt from roots each step), the trajectory
matrix of a fixed (initial, direction) pair here is a 3x3 *polynomial*
matrix in the step index n, degree ~67, with integer coefficients of
hundreds of bits. That drives both kernel designs:

| file | stage | arithmetic |
|---|---|---|
| `walk_meijer_df64.metal` | screening | df64 (2xfloat32) mantissas + software exponents, exact power-of-two rescaling; ~14-digit limits + log2 growth, no primes |
| `walk_meijer_rns.metal` | certification | residue number system, one thread = (candidate, prime); bit-exact via Garner CRT on host |
| `df64_walk.py` | host tooling | validated CPU mirror of the df64 kernel, tensor exporter (`--export`), GPU verifier (`--check-gpu`) |
| `rns_walk.py` | host tooling | bit-exact CPU RNS mirror (validated against exact sympy walk), rigorous prime-count sizing, drop-2-primes consistency check, exporter |
| `common.py` | shared | CMF construction, trajectory matrices, PSLQ identification, delta machinery |

Design notes baked into the code:

- **No unit-step products in floats**: individual axis steps hit removable
  singularities that only cancel symbolically (the baseline pair is
  singular at step 0), so both kernels walk the polynomial trajectory
  matrix directly.
- **Coefficients overflow floats**: the df64 exporter ships each integer
  coefficient as df64 mantissa + int exponent under one global
  power-of-two scale per candidate (projectively exact); Horner
  accumulators renormalize their own exponents, so nothing leaves float32
  range at any depth.
- **`ext_add` zero guard**: an exact-zero accumulator has no meaningful
  exponent and must never win the magnitude comparison.
- **RNS capacity is rigorous**: prime counts come from coefficient
  bit-lengths and the product-norm bound (~5.3k 31-bit primes at N=200,
  ~20k at N=1000), plus a drop-2-primes reconstruction check.

## Host

`src/main_meijer_df64.mm` (target `dreams_meijer_df64`) compiles the df64
kernel at runtime with `MTLMathModeSafe` (exact IEEE fma, required by df64).

## Usage

```bash
pip install -r meijer_cmf/requirements.txt
cmake -S . -B build && cmake --build build --target dreams_meijer_df64 -j

# 1. export candidate tensors (sympy trajectory matrices -> Horner tables)
python meijer_cmf/df64_walk.py --pairs pairs.json --depth 1000 \
    --checkpoints 500,900,1000 --export export_run
# pairs.json: [[[a0..b3 initial], [8-dim direction]], ...]

# 2. GPU walk
./build/dreams_meijer_df64 export_run meijer_cmf/walk_meijer_df64.metal out.bin

# 3. verify GPU against the CPU mirror / consume checkpoints
python meijer_cmf/df64_walk.py --check-gpu export_run out.bin

# exact CPU reference & RNS certification for survivors
python meijer_cmf/df64_walk.py --validate --depth 1000 --checkpoints 500,1000
python meijer_cmf/rns_walk.py --validate --depth 200 --checkpoints 100,200
```

## Validation status (Apple M5 Max)

- df64 GPU == CPU mirror to ~1e-14 relative at N=1000 (12/12 checkpoints,
  incl. the baseline `+1680 z2 +1680 z3` pair and the `+432 z3` hit family).
- CPU mirror == exact sympy walk to ~1e-15 at N=1000.
- RNS CPU mirror == exact sympy walk bit-for-bit at N=100/200.
- Throughput: ~3,600 candidates/s at N=1000 (batch 1024, deg1=84).
  Pipeline bottleneck is the sympy exporter (~1 s/candidate/core).

Output record layout (per checkpoint, per candidate):
`float2 mat[9]`, `int32 expRel[9]`, `int32 growth` --
entry value = `(hi+lo) * 2^expRel`, product magnitude = `2^growth`,
limits `c1 = M[1][0]/M[0][0]`, `c2 = M[2][0]/M[0][0]`.

Canonical development copies live in
`stringtheory/meijer_search/` (`common.py`, `rns/`).
