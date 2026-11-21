#!/bin/bash
# CPU dry run of the train / resume / export / identity path (no GPU, no lock): catches loop bugs
source /home/nvidia/refcv7_post/rl/wprl_env.sh
export CUDA_VISIBLE_DEVICES=""
D=$R/dry
cd $TREE
W=$R/out/windows_train.json
DRV=stack/scripts/ddv2_rl_refcv7.py
run () { local n=$1; shift; echo "[dry] $(date -u +%FT%TZ) start $n"; $PY $DRV "$@" > $D/$n.log 2>&1 < /dev/null; echo "[dry] $(date -u +%FT%TZ) end $n rc=$?"; }
rm -rf $D/tr_seg $D/tr_full $D/tr_id
run tr_seg_a train --device cpu --arm rl --steps 2 --batch 2 --micro 1 --windows $W --out-dir $D/tr_seg --max-steps-this-segment 1 --ckpt-every 0 --segment-minutes 0 --workers 2
run tr_seg_b train --device cpu --arm rl --steps 2 --batch 2 --micro 1 --windows $W --out-dir $D/tr_seg --ckpt-every 0 --segment-minutes 0 --workers 2
run tr_export export --run-dir $D/tr_seg --out $D/tr_seg/export.pt
run tr_check identity --export $D/tr_seg/export.pt --expect different --out $D/tr_seg/identity.json
run tr_id train --device cpu --arm rloff --lr 0 --steps 1 --batch 2 --micro 2 --windows $W --out-dir $D/tr_id --ckpt-every 0 --segment-minutes 0 --workers 2
run tr_id_export export --run-dir $D/tr_id --out $D/tr_id/export.pt
run tr_id_check identity --export $D/tr_id/export.pt --expect identical --out $D/tr_id/identity.json
echo "[dry] ALLDONE"
