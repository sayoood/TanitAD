#!/usr/bin/env bash
# pod_clocktest_v8proof.sh -- lead's smoke of dscale_clockcheck.py on the v8proof pod (CPU only), with the same stand-in as pod_selftest_v8proof.sh:
# the corpus cache as the "union", the first half of its manifest as the "corpus list", the stand-in lead file. Not a result of the card.
# Usage: bash pod_clocktest_v8proof.sh > CLOCKTEST.txt 2>&1
set -u
D=/workspace/v8proof/dscale_lead; PY=/workspace/venv/bin/python; DATA=/workspace/v8proof/data
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
W=$D/clocktest_$(date -u +%H%M%S); mkdir -p "$W"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free=$(( (mm - an) / 1073741824 )); echo "container free ${free} GB"
[ "$free" -ge 20 ] || { echo "ZZ CLOCKTEST STOP memory"; exit 3; }
"$PY" - "$DATA/cache_pod/train/manifest.jsonl" "$W" <<'EOF'
import json, sys
ids = list(dict.fromkeys(json.loads(l)["clip"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()))
open(sys.argv[2] + "/half.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[: len(ids) // 2]) + "\n")
EOF
LEAD=${LEAD:-$D/standin/lead_train.npz}
CC=("$PY" "$D/dscale_clockcheck.py" --union-cache "$DATA/cache_pod" --bset-list "$W/half.txt" --lead-union "$LEAD")
"${CC[@]}" --out "$W/CLOCK.json"; echo "clock check rc $? (expected 0)"
"${CC[@]}" --plant-shift 1 --out "$W/CLOCK_PLANT1.json"; echo "clock check, lead rows moved by one token frame, rc $? (expected 2)"
"${CC[@]}" --plant-shift -2 --out "$W/CLOCK_PLANTm2.json"; echo "clock check, lead rows moved by minus two token frames, rc $? (expected 2)"
echo "ZZ CLOCKTEST DONE $W"
