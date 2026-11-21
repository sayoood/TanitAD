#!/bin/bash
# SMOKE (NOT outcome-bearing) on the MM's TOOL_TREE: MC-H s0, 24 windows, --limit causal --dump-ticks.   usage: smoke_tool.sh <TOOL_TREE>
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
TTu=$(cygpath -u "$1"); TTm=$(cygpath -m "$1")
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=C:/Users/Admin/hmc/smoke/tool
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack") PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
[ -f "$TTu/stack/scripts/v8f_t1_eval.py" ] || { echo "no tool at $TTu"; exit 1; }
mkdir -p $OUT/run $OUT/ticks
before=$(sha256sum < $TTu/stack/scripts/v8f_t1_eval.py | cut -c1-64)
for spec in "MCH SMOKE_TOOL_MCH_s0_m10" "STEEP SMOKE_TOOL_STEEP_s0_m10"; do set -- $spec
  t0=$(date +%s)
  "$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --ckpt E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/M2b_s0_hmc_$1.pt --tag $2 --tracks $SC/sfresh/inputs/native_tracks.pt --eval-cache $SC/sfresh/cache \
     --eval-grid C:/Users/Admin/hlat/inputs/eval_grid_windows.json --eval-join C:/Users/Admin/hlat/inputs/join_eval141.npz --out-dir $OUT/run \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --device cpu --max-windows 24 --dump-ticks $OUT/ticks > $OUT/$2.out 2>&1 < /dev/null
  echo "$2 rc=$? wall=$(( $(date +%s)-t0 ))s" | tee -a $OUT/wall_times.txt
done
[ "$(sha256sum < $TTu/stack/scripts/v8f_t1_eval.py | cut -c1-64)" = "$before" ] && echo "tool sha unchanged" || echo "TOOL SHA CHANGED"
