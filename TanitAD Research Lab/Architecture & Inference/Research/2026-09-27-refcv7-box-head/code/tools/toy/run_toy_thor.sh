#!/bin/bash
# the TOY instrument on Thor in the GATE environment (tanitad-train venv + gate_fix_2224 pkgs, CPU, nice 19, offline,
# no HF token): the TIP tree (constant lr) and the tip + A17 tree, at 1 and 4 intra-op threads, seeds 0-2.
D=/home/nvidia/bx_1600
PY=/home/nvidia/venvs/tanitad-train/bin/python
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
mkdir -p "$D/toy"
run() {  # <tree name> <threads> <seeds> <tag> [--const-lr]
  T="$D/tree_$1"
  export PYTHONPATH="/home/nvidia/gate_fix_2224/pytest_pkgs:$T/stack:$T/stack/scripts:$T/taniteval:$T"
  OMP_NUM_THREADS=$2 MKL_NUM_THREADS=$2 nice -n 19 $PY "$D/toy_repro.py" "$T" "$D/toy/thor_$4.json" --seeds "$3" \
    --threads "$2" --log-every 100 $5 > "$D/toy/thor_$4.log" 2>&1
  echo "$4 exit $?" >> "$D/toy/progress.txt"
}
run tip 4 0 tip_t4_s0
run tip 1 0 tip_t1_s0
run a17 4 0,1,2 a17_t4
run a17 1 0,1,2 a17_t1
echo ALLDONE >> "$D/toy/progress.txt"
