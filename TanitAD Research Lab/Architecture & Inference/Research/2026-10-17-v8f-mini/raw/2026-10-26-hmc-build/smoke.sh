#!/bin/bash
# SMOKE (NOT outcome-bearing): MC-H / STEEP / HACC (regression arm) s0, 24 windows, the pinned HACC-LAT tool (logged bucket: that tool has NO causal bucket), --dump-ticks.
# usage: smoke.sh   (writes C:/Users/Admin/hmc/smoke/*, wall times to smoke/wall_times.txt)
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
H=/c/Users/Admin/hmc; FR=/c/Users/Admin/hlat/frozen; FRm=C:/Users/Admin/hlat/frozen
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc; HACCA=E:/Projects/TanitAD-artifacts/v8f-mini-hacc
OUT=C:/Users/Admin/hmc/smoke
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FRm/stack;$FRm/taniteval;$FRm/stack/scripts"
TOOL_SHA=$(grep v8f_t1_eval.py /c/Users/Admin/hlat/PINS.sha256 | cut -c1-64)
cur() { sha256sum < $FR/stack/scripts/v8f_t1_eval.py | cut -c1-64; }
[ "$(cur)" = "$TOOL_SHA" ] || { echo "SMOKE-STOP tool sha"; exit 1; }
run() {  # run <ckpt> <tag>
  local t0=$(date +%s)
  "$PY" "$FRm/stack/scripts/v8f_t1_eval.py" run --ckpt "$1" --tag "$2" --tracks "$SC/sfresh/inputs/native_tracks.pt" --eval-cache "$SC/sfresh/cache" \
     --eval-grid C:/Users/Admin/hlat/inputs/eval_grid_windows.json --eval-join C:/Users/Admin/hlat/inputs/join_eval141.npz --out-dir "$OUT/run" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit logged --device cpu --max-windows 24 --dump-ticks "$OUT/ticks" > "$OUT/$2.out" 2>&1 < /dev/null
  local rc=$?; local t1=$(date +%s)
  [ "$(cur)" = "$TOOL_SHA" ] || { echo "SMOKE-STOP tool sha after $2"; exit 1; }
  echo "$2 rc=$rc wall=$((t1-t0))s" | tee -a "$OUT/wall_times.txt"
}
mkdir -p "$OUT/run" "$OUT/ticks"
: > "$OUT/wall_times.txt"
run $ART/M2b_s0_hmc_MCH.pt SMOKE_MCH_s0_m10
run $ART/M2b_s0_hmc_STEEP.pt SMOKE_STEEP_s0_m10
run $HACCA/M2b_s0_hacc.pt SMOKE_HACC_s0_m10
echo "SMOKE-DONE"
