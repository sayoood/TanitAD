#!/bin/bash
# the A17 follow-up landing (tip 2ac0bfb + LANDING_READY_A17) vs the clean tip, in the GATE environment on Thor:
# tanitad-train venv, pytest from /home/nvidia/gate_fix_2224/pytest_pkgs, CUDA_VISIBLE_DEVICES empty, nice 19,
# offline and with NO HF token anywhere (huggingface_hub's offline error text dumps the request headers).
D="$1"
PY=/home/nvidia/venvs/tanitad-train/bin/python
AUD="/home/nvidia/bx_0412/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/code"
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
[ -d "$AUD" ] || { echo "NO AUDIT DIR" > "$D/tests.done"; exit 3; }
for V in a17 tip; do
  T="$D/tree_$V"
  LOG="$D/tests_$V.log"
  mkdir -p "$T" && tar -xf "$D/tip_2ac0bfb.tar" -C "$T" || { echo "EXTRACT FAILED" > "$LOG"; continue; }
  if [ "$V" = a17 ]; then
    tar -xf "$D/a17_files.tar" -C "$T" || { echo "EXTRACT FAILED" > "$LOG"; continue; }
    $PY "$D/verify_blobs.py" "$T" "$D/a17_blobs.txt" > "$LOG" 2>&1 || { echo "BLOB VERIFY FAILED" >> "$LOG"; continue; }
  fi
  cd "$T" || continue
  export PYTHONPATH="/home/nvidia/gate_fix_2224/pytest_pkgs:$T/stack:$T/taniteval:$T"
  export TANITAD_BOX_AUDIT_DIR="$AUD"
  nice -n 19 $PY -c "import tanitad,sys; print('tanitad from', tanitad.__file__); sys.exit(0 if tanitad.__file__.startswith('$T/') else 7)" >> "$LOG" 2>&1 || { echo "TANITAD NOT FROM TREE" >> "$LOG"; continue; }
  nice -n 19 $PY -m pytest -q --no-header -p no:cacheprovider -o addopts="" -rfEs $(cat "$D/union.txt") >> "$LOG" 2>&1
  echo "EXIT $?" >> "$LOG"
done
# token hygiene: COUNT hf_ token-shaped strings in the logs (the values are never printed); redact in place if any
$PY - "$D" > "$D/token_scan.txt" 2>&1 <<'PYEOF'
import re, sys
from pathlib import Path
pat = re.compile(rb"hf_[A-Za-z0-9]{20,}")
n = 0
for p in sorted(Path(sys.argv[1]).glob("tests_*.log")):
    b = p.read_bytes()
    k = len(pat.findall(b))
    if k:
        p.write_bytes(pat.sub(b"hf_REDACTED", b))
    print(p.name, k)
    n += k
print("TOTAL", n)
PYEOF
echo ALLDONE > "$D/tests.done"
