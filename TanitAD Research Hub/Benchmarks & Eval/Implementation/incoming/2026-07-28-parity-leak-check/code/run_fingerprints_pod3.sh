#!/bin/bash
# pod3 leak-check fingerprinting driver. READ-ONLY on every cache.
set -u
PY=/workspace/venv/bin/python3
OUT=/workspace/leakcheck
mkdir -p "$OUT"
cd "$OUT"

echo "=== [1/4] KNOWN-POSITIVE CONTROL: leaky val f1b378f295ae (full) ==="
$PY /workspace/leakcheck_fingerprint.py \
  --cache /workspace/pai_epcache/physicalai-val-f1b378f295ae \
  --out "$OUT/fp_val_f1b378f295ae_full.jsonl" --mode full --workers 12

echo "=== [2/4] poses-only VIEW: clean val 600 (0c5f7dac3b11) ==="
$PY /workspace/leakcheck_fingerprint.py \
  --cache /workspace/s3parity/views/physicalai-val-0c5f7dac3b11 \
  --out "$OUT/fp_val_0c5f_view600_poses.jsonl" --mode poses --workers 12

echo "=== [3/4] poses-only VIEW: parity train 2376 ==="
$PY /workspace/leakcheck_fingerprint.py \
  --cache /workspace/s3parity/views/physicalai-train-e438721ae894 \
  --out "$OUT/fp_train_e4387_view_poses.jsonl" --mode poses --workers 12

echo "=== [4/4] THE BIG ONE: parity train e438721ae894 (full, 260 GB) ==="
$PY /workspace/leakcheck_fingerprint.py \
  --cache /workspace/pai_epcache/physicalai-train-e438721ae894 \
  --out "$OUT/fp_train_e4387_full.jsonl" --mode full --workers 12

echo "=== ALL DONE ==="
ls -la "$OUT"
