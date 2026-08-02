#!/bin/bash
set -u
echo "=== caches intact ==="
for d in /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
         /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl; do
  echo "$d : $(ls "$d" | grep -c 'v2ep.pt$') payloads, geometry $(ls "$d"/_geometry.json 2>/dev/null | wc -l)"
done
echo "=== old name must be GONE ==="
ls -d /workspace/data/pai_wide120_v2png_train /workspace/v5eval/pai_wide120_v2png_train 2>&1 | head -2
echo "=== no stray processes ==="
ps -eo pid,cmd | grep -E 'bench_encoder|run_encshare|train_flagship' | grep -v grep | head -5
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== manifest on pod2 == repo? ==="
sha256sum /workspace/v5eval/stack/tanitad/data/parity_manifest.json
