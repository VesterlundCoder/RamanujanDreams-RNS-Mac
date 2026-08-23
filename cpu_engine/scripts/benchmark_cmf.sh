#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  cat >&2 <<'USAGE'
usage: scripts/benchmark_cmf.sh DIRECTIONS.bin START_CSV ROW_NUM ROW_DEN Z_NUM [Z_DEN=1] [NSTEPS=1000]

Environment:
  CPU_BIN       default: ./build/dreams_fixed_start_cpu
  METAL_BIN     default: ../build/dreams_fixed_start_df64
  CHUNK         default: 262144
  GRAIN         default: 32
  THREAD_SET    optional explicit list, e.g. "1 8 18"
  RUN_ID        optional result directory name

The Metal executable is optional. If it exists, the script runs the same input
through Metal and compares FSDF6401 outputs against the highest-thread CPU run.
USAGE
  exit 2
fi

DIRS="$1"
START="$2"
ROW_NUM="$3"
ROW_DEN="$4"
Z_NUM="$5"
Z_DEN="${6:-1}"
NSTEPS="${7:-1000}"
CPU_BIN="${CPU_BIN:-./build/dreams_fixed_start_cpu}"
METAL_BIN="${METAL_BIN:-../build/dreams_fixed_start_df64}"
CHUNK="${CHUNK:-262144}"
GRAIN="${GRAIN:-32}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="results/$RUN_ID"
mkdir -p "$RUN_DIR"

if [[ ! -x "$CPU_BIN" ]]; then
  echo "CPU binary not found: $CPU_BIN. Running make..." >&2
  make
fi
if [[ ! -f "$DIRS" ]]; then
  echo "directions file not found: $DIRS" >&2
  exit 1
fi

if [[ -n "${THREAD_SET:-}" ]]; then
  THREADS=( ${THREAD_SET} )
elif [[ "$(uname -s)" == "Darwin" ]]; then
  THREADS=(1)
  nlevels="$(sysctl -n hw.nperflevels 2>/dev/null || echo 0)"
  cumulative=0
  if [[ "$nlevels" =~ ^[0-9]+$ ]] && (( nlevels > 0 )); then
    for ((i=0; i<nlevels; ++i)); do
      c="$(sysctl -n "hw.perflevel${i}.logicalcpu" 2>/dev/null || echo 0)"
      if [[ "$c" =~ ^[0-9]+$ ]]; then
        cumulative=$((cumulative + c))
        (( cumulative > 0 )) && THREADS+=("$cumulative")
      fi
    done
  else
    THREADS+=("$(sysctl -n hw.logicalcpu)")
  fi
else
  THREADS=(1 "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)")
fi

DEDUP=()
for t in "${THREADS[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] || continue
  (( t > 0 )) || continue
  seen=0
  for x in "${DEDUP[@]}"; do [[ "$x" == "$t" ]] && seen=1; done
  (( seen == 0 )) && DEDUP+=("$t")
done
THREADS=("${DEDUP[@]}")

{
  echo "date=$(date "+%Y-%m-%dT%H:%M:%S%z")"
  echo "directions=$DIRS"
  echo "start=$START"
  echo "pair=$ROW_NUM,$ROW_DEN"
  echo "z=$Z_NUM/$Z_DEN"
  echo "nsteps=$NSTEPS"
  echo "chunk=$CHUNK"
  echo "grain=$GRAIN"
  echo "threads=${THREADS[*]}"
  uname -a
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    sysctl -a 2>/dev/null | grep '^hw\.perflevel' || true
  fi
} > "$RUN_DIR/metadata.txt"

last_cpu_out=""
for t in "${THREADS[@]}"; do
  echo
  echo "=== CPU threads=$t ==="
  out="$RUN_DIR/cpu_t${t}.bin"
  log="$RUN_DIR/cpu_t${t}.log"
  "$CPU_BIN" "$DIRS" "$out" "$START" "$ROW_NUM" "$ROW_DEN" \
      "$Z_NUM" "$Z_DEN" "$NSTEPS" "$CHUNK" "$t" "$GRAIN" \
      2> >(tee "$log" >&2)
  last_cpu_out="$out"
done

if [[ -x "$METAL_BIN" ]]; then
  echo
  echo "=== Metal ==="
  metal_out="$RUN_DIR/metal.bin"
  metal_log="$RUN_DIR/metal.log"
  "$METAL_BIN" "$DIRS" "$metal_out" "$START" "$ROW_NUM" "$ROW_DEN" \
      "$Z_NUM" "$Z_DEN" "$NSTEPS" "$CHUNK" \
      2> >(tee "$metal_log" >&2)

  echo
  echo "=== CPU vs Metal numerical comparison ==="
  python3 scripts/compare_fsdf64.py "$last_cpu_out" "$metal_out" \
      | tee "$RUN_DIR/cpu_vs_metal.txt"
else
  echo
  echo "Metal binary not found/executable at: $METAL_BIN" >&2
  echo "Set METAL_BIN=/path/to/dreams_fixed_start_df64 to include GPU comparison." >&2
fi

echo
python3 scripts/summarize_cmf_logs.py "$RUN_DIR"
echo "Run directory: $RUN_DIR"
