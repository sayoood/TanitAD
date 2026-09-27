#!/bin/bash
# the ADDENDUM test file in the gate environment on Thor: the whole file at OMP 4 (the gate's reading), and the three
# TOY tests at OMP 1 and OMP 16. Runs after the TOY grid (progress16 ALLDONE). CPU, nice 19, offline, no HF token.
D=/home/nvidia/bx_1600
PY=/home/nvidia/venvs/tanitad-train/bin/python
T="$D/tree_toy"
AUD="/home/nvidia/bx_0412/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/code"
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TANITAD_BOX_AUDIT_DIR="$AUD"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
until grep -q ALLDONE "$D/toy/progress16.txt" 2>/dev/null; do sleep 20; done
cd "$T" || exit 2
export PYTHONPATH="/home/nvidia/gate_fix_2224/pytest_pkgs:$T/stack:$T/taniteval:$T"
for th in 4 1 16; do
  K=""; [ "$th" != 4 ] && K="-k TOY"
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th nice -n 19 $PY -m pytest -q --no-header -p no:cacheprovider -o addopts="" \
    -rfEs $K stack/tests/test_g_box_overfit.py > "$D/toy/pytest_toy_omp$th.log" 2>&1
  echo "EXIT $?" >> "$D/toy/pytest_toy_omp$th.log"
done
echo ALLDONE > "$D/toy/pytest_toy.done"
