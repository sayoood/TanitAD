#!/usr/bin/env bash
# E9 — convert the banked BOUND into a RATE by separating fixed cost from marginal cost.
#
# The banked figure is ONE point: `E9B_RC=0 wall=173s` over 9 windows = 19.2 s/window ALL-IN
# (2026-09-19). A single timing cannot tell model load + corpus build from per-window work, which
# is exactly why it is admissible only as an upper BOUND.
#
# ⭐ TWO POINTS, and the design keeps everything else fixed: same checkpoint, same episodes
# (--episodes-n 2), same device, same n-boot. ONLY --window-stride changes, so the episode load and
# model construction are identical and the difference is PURELY per-window.
#      marginal = (T_b - T_a) / (n_b - n_a)        fixed = T_a - n_a * marginal
# ⛔ Run A also RE-DERIVES the banked 173 s rather than inheriting it.
set -o pipefail
A3=C:/Users/Admin/tanitad-caches/a3-heldout-20260919/run
EPS=D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB
LAB=C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/s2_labels_v8_eval.jsonl.gz
OUT=C:/Users/Admin/qland/work/e9
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd /d/Projects/TanitAD || exit 9

run () {  # $1 = stride, $2 = tag
  local t0 t1
  t0=$(date +%s)
  OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 "$PY" taniteval/tools/refcv3_arm.py \
    --ckpt "$A3/ckpt.pt" --config "$A3/config.json" \
    --episodes "$EPS" --labels "$LAB" --device cpu \
    --episodes-n 2 --window-stride "$1" --lru 2 --n-boot 50 \
    --out "$OUT/e9_$2.json" --dump-dir "$OUT/dump_$2" > "$OUT/run_$2.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  local n
  n=$(grep -oE "n_win=[0-9]+" "$OUT/run_$2.log" | tail -1 | cut -d= -f2)
  [ -z "$n" ] && n=$(grep -oE "n=[ ]*[0-9]+" "$OUT/run_$2.log" | tail -1 | tr -d ' ' | cut -d= -f2)
  echo "STRIDE=$1 TAG=$2 RC=$rc WALL=$((t1-t0)) NWIN=${n:-unknown}"
}

echo "=== E9 two-point rate, CPU, same episodes, stride is the only variable ==="
run 40 a
run 20 b
