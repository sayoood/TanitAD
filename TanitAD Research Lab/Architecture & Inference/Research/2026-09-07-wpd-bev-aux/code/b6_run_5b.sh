#!/bin/bash
# WP-D section 5B -- the T1 four-family PLANNER arm for D0 (aux OFF) and D1 (aux ON).
#
# ⛔ D1 is loaded from its STRIPPED checkpoint. `refcv3_arm.py` has NO bev_aux
# handling at all, so the rebuilt model never constructs the head and D1's raw
# checkpoint would be REFUSED with 6 unexpected keys. Stripping is what makes the
# two arms load through an IDENTICAL code path -- which is the prereg's
# "removable with bit-identical planner output" claim used as a procedure.
#
# ⛔ DUMP FIRST, ANALYSE SECOND, ALWAYS. MEASURED here on the smoke: the rollout
# completed both episodes and the ANALYSIS then died on a missing sibling
# (`eval_four_families.py`) -- and the wrapper EXITED 0. Assert on the artifact.
# `--analyze-only` re-reads the banked dumps with ZERO GPU.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TOOL=/c/Users/Admin/wpd-run/taniteval/tools/refcv3_arm.py
EPS="C:\Users\Admin\tanitad-data\refav1-eval141\eps"
LAB="C:\Users\Admin\wpd-probe\labels\s2_labels_v7.2_eval.jsonl.gz"
LEAD="C:\Users\Admin\wpd-probe\leadblk\b1_eval_lead_block.npz"
NEP=${NEP:-40}
STRIDE=${STRIDE:-2}
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4

dump () {   # $1 arm  $2 ckpt  $3 config
  local out="C:\Users\Admin\wpd-probe\dumps\\$1"
  echo "[5b] === DUMP $1 $(date -u +%FT%TZ) ==="
  "$PY" -u "$TOOL" --ckpt "$2" --config "$3" --episodes "$EPS" --labels "$LAB" \
    --lead-block "$LEAD" --arm "$1" --grid 2s --device cuda \
    --episodes-n "$NEP" --window-stride "$STRIDE" \
    --dump-dir "$out" --dump-only --out "C:\Users\Admin\wpd-probe\raw\\$1.json"
  echo "[5b] dump $1 rc=$? files=$(ls /c/Users/Admin/wpd-probe/dumps/$1/ep*.npz 2>/dev/null | wc -l)"
}

analyze () {
  local out="C:\Users\Admin\wpd-probe\dumps\\$1"
  echo "[5b] === ANALYZE $1 $(date -u +%FT%TZ) ==="
  "$PY" -u "$TOOL" --analyze-only "$out" --arm "$1" --lead-block "$LEAD" \
    --n-boot 2000 --out "C:\Users\Admin\wpd-probe\raw\\$1.json"
  echo "[5b] analyze $1 rc=$? json=$(ls -la /c/Users/Admin/wpd-probe/raw/$1.json 2>/dev/null | wc -l)"
}

dump  wpdD0 "C:\Users\Admin\wpd-probe\ckpt\ckpt_D0.pt"         "C:\Users\Admin\wpd-probe\ckpt\config_D0.json"
dump  wpdD1 "C:\Users\Admin\wpd-probe\ckpt\ckpt_D1_planner.pt" "C:\Users\Admin\wpd-probe\ckpt\config_D1.json"
analyze wpdD0
analyze wpdD1
echo "[5b] CHAIN COMPLETE $(date -u +%FT%TZ)"
