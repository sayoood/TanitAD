#!/usr/bin/env bash
set -uo pipefail
echo "=== HOST ==="; hostname
echo "=== venv python cuda ==="
PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
echo "=== disk dd REAL write test (2GB, oflag=direct) ==="
DD=/workspace/experiments/.v4_dd_test.bin
if dd if=/dev/zero of="$DD" bs=1M count=2048 oflag=direct 2>&1 | tail -1; then :; else dd if=/dev/zero of="$DD" bs=1M count=2048 2>&1 | tail -1; fi
ls -la "$DD" 2>/dev/null && echo "dd_write_OK_2GB" || echo "dd_write_FAILED"
rm -f "$DD"
echo "=== experiments dir usage ==="
du -sh /workspace/experiments 2>/dev/null | tail -1
echo "=== df context only (MooseFS hides quota; dd above is ground truth) ==="
df -h /workspace 2>/dev/null | tail -2
echo "ENVCHECK_DONE"
