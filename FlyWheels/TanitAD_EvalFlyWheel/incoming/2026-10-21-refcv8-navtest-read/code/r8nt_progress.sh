#!/bin/bash
# r8nt_progress.sh [run_dir] -- one opaque line: rows per arm per split + markers seen + GPU util.  Grep for QQP...QQ only.
O=${1:-/home/nvidia/r8ns/navtest_read/full}
line="QQP"
for s in navtest warmup navhard; do
  for f in $O/bridge_$s/rows_*.jsonl; do [ -f "$f" ] && line="$line ${s}:$(basename $f .jsonl | sed 's/^rows_//')=$(wc -l < $f)"; done
done
m=$(grep -o 'ZZ[A-Z_a-z0-9]*' $O/../full.out 2>/dev/null | tr '\n' ',')
g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "$line markers=[$m] gpu=$g $(date +%H:%M:%S) QQ"
