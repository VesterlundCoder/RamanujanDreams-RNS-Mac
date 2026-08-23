#!/usr/bin/env bash
set -euo pipefail

echo "=== Dreams CPU Engine system info ==="
uname -a

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo
  echo "--- Apple / macOS CPU topology ---"
  for key in machdep.cpu.brand_string hw.memsize hw.logicalcpu hw.physicalcpu hw.nperflevels; do
    if sysctl -n "$key" >/dev/null 2>&1; then
      printf '%-30s %s\n' "$key" "$(sysctl -n "$key")"
    fi
  done

  nlevels="$(sysctl -n hw.nperflevels 2>/dev/null || echo 0)"
  if [[ "$nlevels" =~ ^[0-9]+$ ]] && (( nlevels > 0 )); then
    for ((i=0; i<nlevels; ++i)); do
      echo "performance level $i (lower index = faster core type):"
      for suffix in logicalcpu physicalcpu l1dcachesize l2cachesize l3cachesize cpusperl2 cpusperl3; do
        key="hw.perflevel${i}.${suffix}"
        if sysctl -n "$key" >/dev/null 2>&1; then
          printf '  %-28s %s\n' "$key" "$(sysctl -n "$key")"
        fi
      done
    done
  fi
else
  echo
  echo "--- CPU info ---"
  command -v lscpu >/dev/null 2>&1 && lscpu || true
fi

echo
echo "Compiler:"
${CC:-cc} --version | head -n 2 || true
