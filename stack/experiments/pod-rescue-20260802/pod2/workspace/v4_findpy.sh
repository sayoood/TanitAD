#!/usr/bin/env bash
echo "=== command -v ==="
command -v python python3 2>/dev/null
echo "=== ls /workspace/venv ==="
ls -la /workspace/venv/bin/ 2>/dev/null | head -20 || echo "no /workspace/venv/bin"
echo "=== conda envs ==="
ls -d /opt/conda/envs/*/ 2>/dev/null; ls -d /opt/conda/bin/python* 2>/dev/null
echo "=== find python interps (maxdepth 4, 25s box) ==="
timeout 25 find /workspace /opt /root -maxdepth 4 -type f \( -name 'python3' -o -name 'python' \) 2>/dev/null | sort -u | head -30
echo "=== torch+cuda test per candidate ==="
for p in /workspace/venv/bin/python /workspace/venv/bin/python3 /opt/venv/bin/python /opt/conda/bin/python /root/venv/bin/python /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$p" ]; then
    echo "--- $p ---"
    PYTHONPATH=/workspace/TanitAD/stack "$p" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")' 2>&1 | tail -2
  fi
done
echo "=== refc-small launch trace (what interpreter did this morning run use?) ==="
ls -d /workspace/experiments/refc-small* 2>/dev/null
grep -rEoh '/[^ ]*(venv|conda)[^ ]*/bin/python[0-9.]*' /workspace/experiments/refc-small*/*.log /workspace/experiments/refc-small*/*.sh 2>/dev/null | sort -u | head
echo "FINDPY_DONE"
