#!/bin/bash
for p in 3608070 3608082 3608145 3608208 3608271; do kill -9 "$p" 2>/dev/null; done
sleep 4
ps -eo pid,cmd | grep -E 'bench_encoder|run_encshare' | grep -v grep
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== 128x576 pass1 result ==="
python3 - <<'PY'
import json
d = json.load(open('/workspace/v5eval/raw/encshare_p1_128x576.json'))
a = d['arms']['128x576']
print('full_step_s median', a['full_step_s']['median'], a['full_step_s']['min'], a['full_step_s']['max'])
print('encoder_only_s median', a['encoder_only_s']['median'])
print('share_standalone', a['encoder_share_of_step_standalone'])
print('peak_MiB', a['peak_mem_MiB'])
print('images/step', a['images_encoded_per_step'], 'tokens', a['n_tokens'])
PY
rm -f /workspace/v5eval/raw/encshare_p*_*.json
