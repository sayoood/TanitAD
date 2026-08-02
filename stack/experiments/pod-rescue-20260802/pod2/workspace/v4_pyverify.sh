#!/usr/bin/env bash
echo "=== /workspace/venv existence (2 probes) ==="
ls -ld /workspace/venv 2>/dev/null || echo "NO /workspace/venv dir"
ls -la /workspace/venv/bin/python* 2>/dev/null || echo "NO python under /workspace/venv/bin"
echo "=== /usr/bin/python3 REAL cuda compute ==="
PYTHONPATH=/workspace/TanitAD/stack /usr/bin/python3 - <<'PY'
import torch, sys, time
print("exe", sys.executable)
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
x = torch.randn(2048,2048, device='cuda')
t=time.time(); y=(x@x).sum().item()
print("matmul_finite", (y==y and abs(y)<1e30), "elapsed_s", round(time.time()-t,3))
print("dev", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
print("mem_alloc_MB", torch.cuda.memory_allocated()//1024//1024)
PY
echo "VERIFY_DONE"
