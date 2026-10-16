#!/bin/bash
# R40 follow-up: every refe/ and eval/ test NOT in the gate-lever landing sweep, FULL form, on tip b8a6a5f.  4 lanes.
set -u
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
B="C:/Users/Admin/AppData/Local/Temp/tipsweep/$PK"
OUT=C:/Users/Admin/AppData/Local/Temp/tipsweep_out; mkdir -p $OUT; : > $OUT/SUMMARY.txt
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
run() { local f="$1"; local d=$(dirname "$f"); local n=$(basename "$f" .py); local t0=$(date +%s)
  ( cd "$B/$d" && "$PY" "$(basename "$f")" ) > "$OUT/$n.log" 2>&1; local rc=$?
  echo "$f rc=$rc $(grep -oE 'ZZ[A-Z0-9_]*PASS[A-Z_]* [0-9]+' "$OUT/$n.log" | tail -1) skips=$(grep -c ZZTEST_SKIP "$OUT/$n.log") $(( $(date +%s) - t0 ))s" >> $OUT/SUMMARY.txt; }
L=(eval/test_aux_navtest_forward.py eval/test_aux_s1_readout.py eval/test_fam_direction.py eval/test_rl_val_eval.py eval/test_s1_read.py
   refe/test_aux_drivable_head.py refe/test_aux_label_store.py refe/test_d9_accept.py refe/test_dump_scores.py refe/test_field_loss.py
   refe/test_flock_rw.py refe/test_hf_egress.py refe/test_label_dir_refusal.py refe/test_p6_cpu_probe.py refe/test_p6_overlap.py
   refe/test_p6_planner_command.py refe/test_pdm_label_complete.py refe/test_rl_epo.py refe/test_rl_fan_dump.py refe/test_rl_partition.py
   refe/test_rl_pool_available.py refe/test_rl_reward.py refe/test_rl_terms_label.py refe/test_rl_tlane_label.py refe/test_sft4r_chain.py)
for lane in 0 1 2 3; do
  ( for i in "${!L[@]}"; do [ $((i % 4)) = $lane ] && run "${L[$i]}"; done ) &
done
wait
echo SWEEP_DONE >> $OUT/SUMMARY.txt
