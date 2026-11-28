#!/bin/sh
# FIX-3 companion integration check on Thor (Master Mind GO, 2026-09-26 23:14): CPU only, OMP 4, fresh dir.
D=$(cd "$(dirname "$0")" && pwd)
PY=/home/nvidia/venvs/tanitad-train/bin/python
export REFCV6_KIT=/home/nvidia EQ_CONFIG=$D/config_pre.json EQ_REMAP_OVERRIDES=$D/remap_thor.json
export EQ_N_CLIPS=2 EQ_N_WIN=2 EQ_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
CK=/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt_30000.pt
L=$D/run.log
echo "ZZTHORSTARTZZ $(date +%FT%T)" >> $L
for t in pre post; do
  REFCV6_REPO=$D/tree_$t PYTHONPATH=$D/tree_$t/stack:$D/tree_$t/taniteval $PY -c "import sys; sys.path.insert(0, '$D/code'); import refcv6_loader as L; import tanitad; print('TREE $t tanitad', tanitad.__file__)" >> $L 2>&1
done
mkdir -p $D/out && cp $D/windows.PRIVATE.json $D/out/windows.PRIVATE.json
cd $D/code
for arm in R A B; do
  $PY eq_as_trained_check.py roll --arm $arm $D/out --pre-tree $D/tree_pre --post-tree $D/tree_post --ckpt $CK >> $D/check.log 2>&1
  echo "ZZTHORARM_${arm}ZZ exit=$? $(grep -o '\"status\": \"[A-Z_]*\"' $D/out/$arm/arm_record.json 2>/dev/null | head -1) $(date +%FT%T)" >> $L
done
$PY eq_as_trained_check.py compare $D/out >> $D/check.log 2>&1
echo "ZZTHORDONEZZ $(grep -o '\"verdict\": \"[A-Z]*\"' $D/out/eq_as_trained_record.json 2>/dev/null | head -1) $(date +%FT%T)" >> $L
