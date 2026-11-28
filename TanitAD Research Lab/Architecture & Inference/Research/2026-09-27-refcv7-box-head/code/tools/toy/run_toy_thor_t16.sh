#!/bin/bash
# after run_toy_thor.sh: the tip + A17 tree at 16 intra-op threads (Thor has 14 cores), seeds 0-2 -- gate environment.
D=/home/nvidia/bx_1600
PY=/home/nvidia/venvs/tanitad-train/bin/python
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
until grep -q ALLDONE "$D/toy/progress.txt" 2>/dev/null; do sleep 20; done
T="$D/tree_a17"
export PYTHONPATH="/home/nvidia/gate_fix_2224/pytest_pkgs:$T/stack:$T/stack/scripts:$T/taniteval:$T"
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nice -n 19 $PY "$D/toy_repro.py" "$T" "$D/toy/thor_a17_t16.json" --seeds 0,1,2 \
  --threads 16 --log-every 100 > "$D/toy/thor_a17_t16.log" 2>&1
echo "a17_t16 exit $?" >> "$D/toy/progress16.txt"
echo ALLDONE >> "$D/toy/progress16.txt"
