#!/bin/bash
R=/home/nvidia/refcv7_post/rl
PY=/home/nvidia/venvs/tanitad-train/bin/python
export REFCV6_REPO=$R/tree REFCV6_KIT=/home/nvidia PYTHONPATH=$R/tree/stack OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
cd $R/tree
echo "[w] $(date -u +%FT%TZ) tests"
$PY -m pytest -q stack/tests/test_ddv2_refcv7.py stack/tests/test_ddv2_rl.py stack/tests/test_ddv2_il.py stack/tests/test_pdm_proxy.py stack/tests/test_ddv2_refc_chain.py -p no:cacheprovider > $R/logs/tests_thor.log 2>&1 < /dev/null
echo "[w] tests rc=$?"
echo "[w] $(date -u +%FT%TZ) windows eval"
$PY stack/scripts/ddv2_rl_refcv7.py windows --split eval --device cpu --out $R/out/windows_eval.json > $R/logs/windows_eval.log 2>&1 < /dev/null
echo "[w] eval rc=$?"
echo "[w] $(date -u +%FT%TZ) windows train"
$PY stack/scripts/ddv2_rl_refcv7.py windows --split train --device cpu --out $R/out/windows_train.json > $R/logs/windows_train.log 2>&1 < /dev/null
echo "[w] train rc=$?"
echo "[w] $(date -u +%FT%TZ) end"
