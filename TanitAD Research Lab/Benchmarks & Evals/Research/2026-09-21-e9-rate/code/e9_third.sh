#!/usr/bin/env bash
# THIRD POINT — because two points fit a line EXACTLY and a fit that cannot fail is not evidence.
# Model from a(9,70s) and b(18,120s): fixed 20 s, marginal 5.556 s/win.
# PREDICTION for stride 10 (~36 windows), stated BEFORE the run: 20 + 36*5.556 = 220 s.
set -o pipefail
A3=C:/Users/Admin/tanitad-caches/a3-heldout-20260919/run
EPS=D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB
LAB=C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/s2_labels_v8_eval.jsonl.gz
OUT=C:/Users/Admin/qland/work/e9
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd /d/Projects/TanitAD || exit 9
t0=$(date +%s)
OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 "$PY" taniteval/tools/refcv3_arm.py \
  --ckpt "$A3/ckpt.pt" --config "$A3/config.json" \
  --episodes "$EPS" --labels "$LAB" --device cpu \
  --episodes-n 2 --window-stride 10 --lru 2 --n-boot 50 \
  --out "$OUT/e9_c.json" --dump-dir "$OUT/dump_c" > "$OUT/run_c.log" 2>&1
rc=$?; t1=$(date +%s)
n=$(grep -oE "n_win=[0-9]+" "$OUT/run_c.log" | tail -1 | cut -d= -f2)
echo "STRIDE=10 RC=$rc WALL=$((t1-t0)) NWIN=${n:-unknown}  PREDICTED=220s"
