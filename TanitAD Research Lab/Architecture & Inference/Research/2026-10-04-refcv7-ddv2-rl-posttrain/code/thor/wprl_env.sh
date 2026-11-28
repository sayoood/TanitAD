# sourced by every WP-RL Thor script
R=/home/nvidia/refcv7_post/rl
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
PY=/home/nvidia/venvs/tanitad-train/bin/python
TREE=$R/tree
export REFCV6_REPO=$TREE REFCV6_KIT=/home/nvidia PYTHONPATH=$TREE/stack OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
others () {  # processes OUTSIDE this session holding or waiting on the lock
  $PY -c "import sys; sys.path.insert(0,'$TREE/stack/scripts'); import ddv2_rl_refcv7 as m; print(len(m.other_lock_users('$LOCK')))"
}
