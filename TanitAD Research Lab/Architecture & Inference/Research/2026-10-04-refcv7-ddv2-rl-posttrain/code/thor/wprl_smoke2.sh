#!/bin/bash
# WP-RL stage-2 GPU smoke, v2: every GPU run in ONE process under ONE lock acquisition (the shared
# Thor lock is contended by the refcv8 R1 chain, whose 20 s gaps admit one acquisition at a time).
source /home/nvidia/refcv7_post/rl/wprl_env.sh
S=$R/smoke
cd $TREE
W=$R/out/windows_train.json
DRV=stack/scripts/ddv2_rl_refcv7.py
run () { local n=$1; shift; echo "[smoke2] $(date -u +%FT%TZ) start $n"; $PY $DRV "$@" > $S/$n.log 2>&1 < /dev/null; echo "[smoke2] $(date -u +%FT%TZ) end $n rc=$?"; }
rm -rf $S/rl_seg $S/rl_full $S/rloff $S/rlshuf $S/timing32
run smoke2 smoke --windows $W --workers 4 --inproc-lock $LOCK --out-dir $S
run rl_export export --run-dir $S/rl_full --out $S/rl_full/export.pt
run rl_check identity --export $S/rl_full/export.pt --expect different --out $S/rl_full/identity.json
for d in identity rl_seg rl_full rloff rlshuf timing32; do
  $PY $R/wprl_check.py $S/$d --selftest --json $S/$d/check.json > $S/check_$d.log 2>&1
  echo "[smoke2] check $d rc=$? $(grep -o '"verdict": "[A-Z]*"' $S/$d/check.json 2>/dev/null | tail -1)"
done
# the FORWARD identity: the identity export and the cold start read the same 1-in-40 eval windows
run heldout_base heldout --windows $R/out/windows_eval.json --stride 40 --infer-seed 0 --inproc-lock $LOCK --out $S/heldout_base_s0_stride40.json
M=$(md5sum $S/identity/export.pt | cut -d' ' -f1)
run heldout_ident heldout --ckpt $S/identity/export.pt --ckpt-md5 $M --windows $R/out/windows_eval.json --stride 40 --infer-seed 0 --inproc-lock $LOCK --out $S/heldout_identity_s0_stride40.json
echo "[smoke2] $(date -u +%FT%TZ) ALLDONE"
