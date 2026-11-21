#!/bin/sh
# 2026-10-04 (EvalFlyWheel): the unattended sequence after the step-5,000 G0 diagnosis, DETACHED.
#   nohup sh orchestrate_final.sh > /d/refcv7_eval_kit/chain/orchestrate_final.out 2>&1 &
#   1. the step-5,000 G0 diagnostic (run_g0_diag.sh; free-commit wait up to 12 h, then the GPU lock);
#   2. zero GPU: the POST-HOC A5 re-judge of step 5,000 (g0_rejudge_a5.py; SPEC A5 item 6, never a gate)
#      and a sanitized bank of both artifacts into the package (raw/g0diag_step5000/, raw/step5000/);
#   3. the milestone chain on the FINAL checkpoint, step 50,400 (chain_milestone.sh: commit wait -> lock ->
#      G0 under SPEC A5 with 24 seeds -> rolls -> families -> bars -> perception -> refcv6@38k rolls);
#   4. the same chain on step 30,000 (the trend; refcv6@38k baselines REUSED per SPEC A4).
# Markers ZZORCH...ZZ in /d/refcv7_eval_kit/chain/orchestrate_final.log; every decision reads an ARTIFACT.
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
LOG=/d/refcv7_eval_kit/chain/orchestrate_final.log
D5=/d/refcv7_eval_kit/battery/g0diag_step5000
echo "ZZORCHSTART $(date +%FT%T%z) pid $$" >> $LOG
# ---- 1. the diagnostic (run HERE, synchronously; it waits up to 12 h for free commit, then the lock) #
ARMS="s0,eps0,seed8,seed9,seed10,seed11,seed12,seed13,seed14,seed15,seed16,seed17,seed18,seed19,seed20,seed21,seed22,seed23,fp32_s0,loaderflags_s0,micro_alt_s0"
[ -f $D5/launcher.log ] && mv $D5/launcher.log $D5/launcher_attempt3_killed_$(date +%H%M%S).log
sh $PKG/code/run_g0_diag.sh 5000 "$ARMS" 43200 >> $D5/launcher.out 2>&1
echo "ZZORCHDIAGEND $(date +%FT%T%z) $(tail -1 $D5/launcher.log | cut -c1-80)" >> $LOG
# ---- 2. post-hoc A5 re-judge of step 5,000 + sanitized bank (zero GPU) -------------------------- #
if [ -s $D5/diag.json ]; then
  CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/g0_rejudge_a5.py" --g0 D:/refcv7_eval_kit/battery/step5000/g0.json \
      --diag D:/refcv7_eval_kit/battery/g0diag_step5000/diag.json \
      --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
      --out D:/refcv7_eval_kit/battery/step5000/g0_A5_posthoc.json >> $LOG 2>&1
  mkdir -p $PKG/raw/g0diag_step5000
  # bank only after a UUID scan of each file (scalars only are expected; a hit REFUSES the copy)
  for f in $D5/diag.json $D5/launcher.log $D5/diag.log $D5/diag_lock.json $D5/wait_commit.json \
           /d/refcv7_eval_kit/battery/step5000/g0_A5_posthoc.json; do
    [ -s "$f" ] || continue
    if grep -Eqi '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$f"; then
      echo "ZZORCHBANKREFUSED $f (UUID-shaped token)" >> $LOG
    else
      case "$f" in
        */step5000/*) cp "$f" $PKG/raw/step5000/ ;;
        *) cp "$f" $PKG/raw/g0diag_step5000/ ;;
      esac
    fi
  done
  echo "ZZORCHBANKED $(date +%FT%T%z)" >> $LOG
fi
# ---- 3. the FINAL checkpoint, step 50,400 ------------------------------------------------------ #
echo "ZZORCHCHAIN50400 $(date +%FT%T%z)" >> $LOG
sh $PKG/code/chain_milestone.sh 50400 >> /d/refcv7_eval_kit/chain/chain_50400.out 2>&1
echo "ZZORCHCHAIN50400END $(date +%FT%T%z) $(grep -o 'ZZ[A-Z0-9]*ZZ' /d/refcv7_eval_kit/chain/chain_50400.log | tail -1)" >> $LOG
# ---- 4. step 30,000 (trend) -------------------------------------------------------------------- #
echo "ZZORCHCHAIN30000 $(date +%FT%T%z)" >> $LOG
sh $PKG/code/chain_milestone.sh 30000 >> /d/refcv7_eval_kit/chain/chain_30000.out 2>&1
echo "ZZORCHCHAIN30000END $(date +%FT%T%z) $(grep -o 'ZZ[A-Z0-9]*ZZ' /d/refcv7_eval_kit/chain/chain_30000.log | tail -1)" >> $LOG
echo "ZZORCHEND $(date +%FT%T%z)" >> $LOG
