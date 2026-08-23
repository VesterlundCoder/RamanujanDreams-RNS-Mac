#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="results/$RUN_ID-micro"
mkdir -p "$RUN_DIR"

if [[ ! -x build/matmul6_bench || ! -x build/rns_matmul6_bench ]]; then
  make
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  ALL="$(sysctl -n hw.logicalcpu)"
else
  ALL="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
THREADS="${THREADS:-$ALL}"
MAT_N="${MAT_N:-500000}"
MAT_REPEATS="${MAT_REPEATS:-10}"
RNS_PAIRS="${RNS_PAIRS:-100000}"
RNS_STEPS="${RNS_STEPS:-20}"

scripts/system_info.sh | tee "$RUN_DIR/system.txt"

echo
echo "=== FP32 6x6 batch ==="
build/matmul6_bench --n "$MAT_N" --repeats "$MAT_REPEATS" --threads "$THREADS" \
  --csv "$RUN_DIR/matmul6.csv" | tee "$RUN_DIR/matmul6.log"

echo
echo "=== uint32 RNS 6x6 chain ==="
build/rns_matmul6_bench --pairs "$RNS_PAIRS" --steps "$RNS_STEPS" --threads "$THREADS" \
  --csv "$RUN_DIR/rns_matmul6.csv" | tee "$RUN_DIR/rns_matmul6.log"

echo
echo "Results: $RUN_DIR"
