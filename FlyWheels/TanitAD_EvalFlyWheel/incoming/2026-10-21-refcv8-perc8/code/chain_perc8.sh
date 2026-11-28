#!/bin/bash
# chain_perc8.sh -- the GPU passes, priority order.  Each step: exit code -> status/<tag>.exit (never read through a pipe); the ARTIFACT
# (perc.json / <tag>.json) is the evidence a pass ran, not the code.  Launched ONLY after the waiter saw Thor's GPU idle.
source /home/nvidia/r8ns/perc8/code/thor_env.sh
cd $P8/code
EMA=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
DEC=/home/nvidia/r8ns/navtrain_sel/wiseft/ckpt_mod2/ckpt_wiseft_mod2_DEC.pt
CFG=/home/nvidia/r8ns/ckpt/config.json
WIN=$P8/inputs/s2_windows.json
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock   # the Thor convention (refcv7_post/diag/code/run_chain.sh): one GPU pass at a time, taken for the pass only
FL="flock -w 7200 $LOCK"
mkdir -p $P8/out/A_nosync $P8/out/T_nosync $P8/out/W_dec_sub $P8/out/A_sync
stamp () { echo "[chain] $(date +%H:%M:%S) $*"; }
stamp "md5 ckpt: $(md5sum $EMA | cut -c1-32) DEC: $(md5sum $DEC | cut -c1-32)"
if [ ! -s $P8/out/A_nosync/perc.json ]; then
  stamp start A_nosync
  R8_NO_POSE_SYNC=1 $FL $PY -X utf8 perc_pass_r8.py --ckpt $EMA --config $CFG --windows-json $WIN --out $P8/out/A_nosync/perc \
      > $P8/logs/A_nosync.log 2>&1 < /dev/null
  echo $? > $P8/status/A_nosync.exit
  stamp "end A_nosync rc=$(cat $P8/status/A_nosync.exit) artifact=$( [ -s $P8/out/A_nosync/perc.json ] && echo yes || echo NO )"
fi
if [ ! -s $P8/out/T_nosync/train_diag.json ]; then
  stamp start T_nosync
  R8_NO_POSE_SYNC=1 R8_REMAP_FILE=refcv8_trdiag_remap_thor.json $FL $PY -X utf8 run_diag_r8.py --split train_diag --ckpt $EMA \
      --run-dir $P8/cache/run_dir_td --tag train_diag --out $P8/out/T_nosync --batch 8 --workers 4 --no-mf \
      > $P8/logs/T_nosync.log 2>&1 < /dev/null
  echo $? > $P8/status/T_nosync.exit
  stamp "end T_nosync rc=$(cat $P8/status/T_nosync.exit) artifact=$( [ -s $P8/out/T_nosync/train_diag.json ] && echo yes || echo NO )"
fi
  # the TRAIN refit is CPU-only: start it now, in the background, while the next GPU pass runs
  if [ -s $P8/out/T_nosync/train_diag.packs.pkl ]; then
    nohup $PY -X utf8 /home/nvidia/r8ns/code/tree/stack/scripts/refit_perception_thresholds.py --run refcv8-r101-s0-ema48852       --checkpoint-step 48852 --checkpoint-md5 b4364029b68f29a33eeb4aebbbbd002b       --fit-set "TRAIN-DIAG: 139 TRAIN episodes x 8 windows of the refcv8 train4572 cache (sha12-sorted, first 139 with a 10 cm GT file); EMA-48,852 weights; unshifted pose clock"       --box-train-packs $P8/out/T_nosync/train_diag.packs.pkl --out-gates $P8/out/T_nosync/refcv8_det_presence_gates_train.json       --nms --out-nms $P8/out/T_nosync/refcv8_det_nms_train.json > $P8/logs/refit.log 2>&1 < /dev/null &
    stamp "refit started in background (pid $!)"
  fi
if [ ! -s $P8/out/W_dec_sub/perc.json ]; then
  stamp start W_dec_sub
  R8_NO_POSE_SYNC=1 $FL $PY -X utf8 perc_pass_r8.py --ckpt $DEC --config $CFG --windows-json $P8/inputs/s2_every16.json --out $P8/out/W_dec_sub/perc \
      > $P8/logs/W_dec_sub.log 2>&1 < /dev/null
  echo $? > $P8/status/W_dec_sub.exit
  stamp "end W_dec_sub rc=$(cat $P8/status/W_dec_sub.exit) artifact=$( [ -s $P8/out/W_dec_sub/perc.json ] && echo yes || echo NO )"
fi
if [ -e $P8/status/RUN_SYNC_SUB ] && [ ! -s $P8/out/A_sync_sub/perc.json ]; then
  mkdir -p $P8/out/A_sync_sub
  stamp start A_sync_sub
  $FL $PY -X utf8 perc_pass_r8.py --ckpt $EMA --config $CFG --windows-json $P8/inputs/s2_every4.json --out $P8/out/A_sync_sub/perc       > $P8/logs/A_sync_sub.log 2>&1 < /dev/null
  echo $? > $P8/status/A_sync_sub.exit
  stamp "end A_sync_sub rc=$(cat $P8/status/A_sync_sub.exit) artifact=$( [ -s $P8/out/A_sync_sub/perc.json ] && echo yes || echo NO )"
fi
if [ -e $P8/status/RUN_A_SYNC ] && [ ! -s $P8/out/A_sync/perc.json ]; then
  stamp start A_sync
  $FL $PY -X utf8 perc_pass_r8.py --ckpt $EMA --config $CFG --windows-json $WIN --out $P8/out/A_sync/perc \
      > $P8/logs/A_sync.log 2>&1 < /dev/null
  echo $? > $P8/status/A_sync.exit
  stamp "end A_sync rc=$(cat $P8/status/A_sync.exit) artifact=$( [ -s $P8/out/A_sync/perc.json ] && echo yes || echo NO )"
fi
stamp CHAIN_DONE
