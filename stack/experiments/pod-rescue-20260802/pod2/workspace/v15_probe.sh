#!/usr/bin/env bash
echo "=== v15/v16 experiment dirs ==="
ls /workspace/experiments/ | grep -iE 'v1[56]' || echo none
echo "=== contents of each match ==="
for d in /workspace/experiments/*v1[56]*; do
  [ -d "$d" ] || continue
  echo "--- $d ---"
  ls -la "$d"
done
echo "=== HF token presence (existence only) ==="
test -f "$HOME/.cache/huggingface/token" && echo TOKEN_CACHED_AT_HOME || echo NO_CACHED_TOKEN
ls -la "$HOME/.cache/huggingface/" 2>/dev/null | head
env | grep -iE 'HF_TOKEN|HUGGINGFACE|HF_HUB' | sed 's/=.*/=<redacted>/' || echo NO_HF_ENV
echo "=== huggingface_hub version ==="
/usr/bin/python3 -c 'import huggingface_hub as h; print("hfhub", h.__version__)' 2>&1 | tail -2
echo "=== v4 still running? compute-apps ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "=== v4 latest canary rows (the degradation) ==="
grep 'canary_ade@2s' /workspace/experiments/flagship-v4-30k/train.log 2>/dev/null | tail -6
echo "PROBE_DONE"
