#!/bin/bash
# chain2_T.sh -- the TRAIN-DIAG pass again (the first attempt died in run_diag's map-loss wrapper: unexpected kwarg class_thresholds; fixed
# in perc8/code/run_diag.py), then the CPU refit.  Waits for chain_perc8.sh to be done so two GPU passes never overlap.
source /home/nvidia/r8ns/perc8/code/thor_env.sh
cd $P8/code
for i in $(seq 1 120); do grep -q CHAIN_DONE $P8/logs/chain.log && break; sleep 10; done
EMA=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
FL="flock -w 7200 /home/nvidia/refcv7_post/thor_gpu.lock"
stamp () { echo "[chain2] $(date +%H:%M:%S) $*"; }
rm -rf $P8/out/T_nosync; mkdir -p $P8/out/T_nosync
stamp start T_nosync
R8_NO_POSE_SYNC=1 R8_REMAP_FILE=refcv8_trdiag_remap_thor.json $FL $PY -X utf8 run_diag_r8.py --split train_diag --ckpt $EMA \
    --run-dir $P8/cache/run_dir_td --tag train_diag --out $P8/out/T_nosync --batch 8 --workers 4 --no-mf \
    > $P8/logs/T_nosync2.log 2>&1 < /dev/null
echo $? > $P8/status/T_nosync2.exit
stamp "end T_nosync rc=$(cat $P8/status/T_nosync2.exit) artifact=$( [ -s $P8/out/T_nosync/train_diag.json ] && echo yes || echo NO )"
if [ -s $P8/out/T_nosync/train_diag.packs.pkl ]; then
  $PY -X utf8 /home/nvidia/r8ns/code/tree/stack/scripts/refit_perception_thresholds.py --run refcv8-r101-s0-ema48852 \
    --checkpoint-step 48852 --checkpoint-md5 b4364029b68f29a33eeb4aebbbbd002b \
    --fit-set "TRAIN-DIAG: 139 TRAIN episodes x 8 windows of the refcv8 train4572 cache (sha12-sorted, first 139 with a 10 cm GT file); EMA-48,852 weights; unshifted pose clock" \
    --box-train-packs $P8/out/T_nosync/train_diag.packs.pkl --out-gates $P8/out/T_nosync/refcv8_det_presence_gates_train.json \
    --nms --out-nms $P8/out/T_nosync/refcv8_det_nms_train.json > $P8/logs/refit.log 2>&1 < /dev/null
  echo $? > $P8/status/refit.exit
  stamp "refit rc=$(cat $P8/status/refit.exit)"
fi
stamp CHAIN2_DONE
