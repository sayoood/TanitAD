# sourced by every perc8 launcher on Thor. Evidence: CLAUDE.md traps (PYTHONPATH required, OMP_NUM_THREADS=6, HF offline).
export R8_LAUNCH_TREE=/home/nvidia/r8ns/code/tree
export R8_EVAL_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/r8ns/code/tree/stack:/home/nvidia/r8ns/code/tree/taniteval
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export PY=/home/nvidia/venvs/tanitad-train/bin/python
export P8=/home/nvidia/r8ns/perc8
