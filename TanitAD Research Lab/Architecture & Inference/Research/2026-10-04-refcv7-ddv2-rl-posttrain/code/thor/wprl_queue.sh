#!/bin/bash
# WP-RL stage 3 (ONLY after the Master Mind's PASS token): the four arms in the SPEC_RL run order,
# each exported, then the T0 secondary read (heldout) of BASE + every export at inference seeds 0, 1.
# The T1 PRIMARY (the refcv7 battery roll, os-only, S2 + S6) is launched separately per export.
source /home/nvidia/refcv7_post/rl/wprl_env.sh
DRV=stack/scripts/ddv2_rl_refcv7.py
G=$R/gate
mkdir -p $R/exports $R/eval
cd $TREE
for arm in rl_s0 rloff_s0 rlshuf_s0 rl_s1; do
  if [ -e $R/QUEUE_STOP ]; then echo "[queue] QUEUE_STOP -- stopping before $arm"; exit 0; fi
  $R/wprl_arm.sh $G/ARGV_$arm.json
  rc=$?
  if [ $rc -eq 3 ]; then echo "[queue] $arm REFUSED by the token check"; exit 3; fi
  if [ ! -s $R/arms/$arm/DONE.json ]; then echo "[queue] $arm not DONE -- stopping"; exit 4; fi
  $PY $R/wprl_check.py $R/arms/$arm --min-steps 1 --selftest --json $R/arms/$arm/check.json > /dev/null
  echo "[queue] $arm integrity rc=$? $(grep -o '"verdict": "[A-Z]*"' $R/arms/$arm/check.json | tail -1)"
  $PY $DRV export --run-dir $R/arms/$arm --out $R/exports/$arm.pt >> $R/exports/export.log 2>&1
done
for ck in base rl_s0 rloff_s0 rlshuf_s0 rl_s1; do
  if [ "$ck" = base ]; then P=$R/ckpt_50400.pt; else P=$R/exports/$ck.pt; fi
  [ -s $P ] || continue
  M=$(md5sum $P | cut -d' ' -f1)
  for s in 0 1; do
    out=$R/eval/heldout_${ck}_s$s.json
    [ -s $out ] && continue
    $PY $DRV heldout --ckpt $P --ckpt-md5 $M --infer-seed $s --stride 8 --windows $R/out/windows_eval.json \
        --inproc-lock $LOCK --out $out > $R/eval/heldout_${ck}_s$s.log 2>&1 < /dev/null
    echo "[queue] heldout $ck s$s rc=$? artifact=$( [ -s $out ] && echo yes || echo NO )"
  done
done
echo "[queue] ALLDONE $(date -u +%FT%TZ)"
