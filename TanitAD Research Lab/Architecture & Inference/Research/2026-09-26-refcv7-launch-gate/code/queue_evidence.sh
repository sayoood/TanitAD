#!/usr/bin/env bash
# The gate's evidence queue on the dev box: ONE job at a time, each behind the 8 GB RAM floor
# (the gate's own --min-free-gb wait; a python psutil wait for the non-gate steps). Every step's
# verdict is the JSON it writes -- this script only sequences them.
#   WAIT_PID=<pid of a chain to finish first> ./queue_evidence.sh
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
L=C:/Users/Admin/lg0926
IN=$L/inputs
CODE=$L/work/pkg/code
export PYTHONIOENCODING=utf-8
# ⛔ MSYS rewrites POSIX-looking args (`/home/nvidia/...=D:/...`) for native python
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
KEY="--key-file $L/keys/rehearsal.key"
BASE="--cpu-only --omp 4 --min-free-gb 8 --ram-wait-s 21600"
PM=()
while IFS= read -r l; do [ -n "$l" ] && PM+=(--path-map "$l"); done < "$IN/pathmap.txt"
REAL=(--eval-loader C:/Users/Admin/ev6_battery/code/refcv6_loader.py --eval-kit D:/refcv6_eval_kit
      --eval-stamps D:/refcv6_eval_kit/ckpt_final/config.json
      --eval-remap-overrides "$IN/eval_remap_overrides.json"
      --clock-reference "$IN/q4c_grid_vs_egolog_ALLTRAIN.json"
      --clock-manifest train=C:/Users/Admin/qland/work/a16_switch/train_v2manifest.pt)

ram_wait() {   # the brief's floor, for steps that are not gate runs
  until "$PY" -c "import psutil,sys; sys.exit(0 if psutil.virtual_memory().available/2**30 >= 8 else 1)"; do
    sleep 60
  done
}
step() { echo "ZZQ-$1-$(date -u +%H%M%S)ZZ"; }

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
if [ -n "${WAIT_WINPID:-}" ]; then     # a native Windows pid (bash's kill -0 cannot see it)
  while "$PY" -c "import psutil,sys; sys.exit(0 if psutil.pid_exists($WAIT_WINPID) else 1)"; do
    sleep 30
  done
fi

if [ -z "${SKIP_TIP_MODEL:-}" ]; then
  step tip_model
  TREE=$L/c5d OUT=$L/g/tip_model bash "$CODE/run_tip_evidence.sh" G-HYG,G-DVB,G-EVAL > "$L/g/tip_model.out" 2>&1
fi
step tip_clock
TREE=$L/c5d OUT=$L/g/tip_clock bash "$CODE/run_tip_evidence.sh" G-CLOCK > "$L/g/tip_clock.out" 2>&1

step replay
ram_wait
"$PY" "$CODE/replay_refcv6_metrics.py" --tree "$L/c5d" --config D:/refcv6_eval_kit/ckpt_final/config.json \
  --metrics D:/refcv6_eval_kit/ckpt_final/metrics.jsonl --json "$L/g/replay_refcv6_metrics.json" \
  > "$L/g/replay.out" 2>&1

step arms_tiny_tip
"$PY" "$L/c5d/stack/scripts/launch_gate_arms.py" --tree "$L/c5d" --argv-file "$L/rig/argv_tiny.json" \
  --out-dir "$L/g/arms_tiny_tip" --arms no_cascade_passthrough,no_cascade_silent,undeclared_equalize,drop_fix3_field,missing_hygiene_module,missing_dvb_module,drivort_flag,no_fmap_s8_passthrough,ceiling_active_in_training,required_on_missing,tau_differs,tau_file_missing,map_class_weight_zero,map_drivable_only_logging,map_overfit_missing \
  --token-arms --alt-tree "$L/f5d" -- $BASE $KEY > "$L/g/arms_tiny_tip.out" 2>&1

step arms_tiny_fix
"$PY" "$L/f5d/stack/scripts/launch_gate_arms.py" --tree "$L/f5d" --argv-file "$L/rig/argv_tiny.json" \
  --out-dir "$L/g/arms_tiny_fix" --arms no_cascade_passthrough,no_cascade_silent,undeclared_equalize,drop_fix3_field,missing_hygiene_module,missing_dvb_module,drivort_flag,no_fmap_s8_passthrough,ceiling_active_in_training,required_on_missing,tau_differs,tau_file_missing,map_class_weight_zero,map_drivable_only_logging,map_overfit_missing \
  -- $BASE $KEY > "$L/g/arms_tiny_fix.out" 2>&1

step fix_model
TREE=$L/f5d OUT=$L/g/fix_model bash "$CODE/run_tip_evidence.sh" G-HYG,G-DVB,G-EVAL > "$L/g/fix_model.out" 2>&1
step fix_clock
TREE=$L/f5d OUT=$L/g/fix_clock bash "$CODE/run_tip_evidence.sh" G-CLOCK > "$L/g/fix_clock.out" 2>&1

step arms_real
"$PY" "$L/f5d/stack/scripts/launch_gate_arms.py" --tree "$L/f5d" --argv-file "$IN/argv_refcv6_r101_s0.json" \
  --profile refcv6 --out-dir "$L/g/arms_real_fix" \
  --arms legacy_label_clock,loader_skips_pin,drop_fix3_field,undeclared_equalize \
  -- $BASE $KEY "${PM[@]}" "${REAL[@]}" > "$L/g/arms_real_fix.out" 2>&1

step suite
ram_wait
"$PY" "$L/c5d/stack/scripts/launch_gate.py" run --profile refcv6 --checks G-SUITE \
  --tree "$L/t5d" --commit 5de93636043e3bafcbaa9027405bdf53013db140 \
  --argv-file "$IN/argv_refcv6_r101_s0.json" --out-dir "$L/g/suite_tip" \
  --git-dir C:/Users/Admin/tanitad-push/.git --baseline 5de93636043e3bafcbaa9027405bdf53013db140 \
  --suite-work C:/lgs $BASE $KEY > "$L/g/suite_tip.out" 2>&1
step done
