#!/bin/bash
# pod_level_checks.sh -- runs ON THE POD (nohup), CPU only. Card PREREG_V8F_PROOF_ADD37, note N6, controls v3 and v4 on the pod's own trees.
#   1. checks/v3_identity.py: the tree of the dose arms against the new tree with the level weight at 0 (4 training steps on the tiny synthetic world, CPU).
#   2. the two test files of the term, whole files, from the new tree.
# Starts only with 20 GB of the container memory limit free. No GPU job, no process test. Deletes nothing. Last line: ZZ POD-CHECKS DONE v3=<rc> pytest=<rc>.
set -u
W=/workspace/v8proof; T=$W/tree_actlevel; O=$W/actlevel/checks
cd $W/actlevel || exit 9
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
a=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat); m=$(cat /sys/fs/cgroup/memory.max); f=$(( (m - a) / 1073741824 ))
[ "$f" -ge 20 ] || { echo "[$(ts)] ZZ POD-CHECKS STOP: container memory free $f GB (need 20)"; exit 8; }
echo "[$(ts)] container memory free $f GB"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
/workspace/venv/bin/python checks/v3_identity.py $W/tree_actcons $T $O/v3_identity_pod.json /workspace/venv/bin/python > $O/v3_identity_pod.out 2>&1; r1=$?
echo "[$(ts)] v3 identity rc $r1: $(tail -n 1 $O/v3_identity_pod.out | cut -c1-300)"
mkdir -p $W/actlevel/work
cd $T || exit 9
PYTHONPATH=$T/stack:$T/taniteval:$T/stack/scripts /workspace/venv/bin/python -m pytest -q -p no:cacheprovider stack/tests/test_v8f_act_cons.py stack/tests/test_v8f_act_level.py --basetemp=$W/actlevel/work/pt > $O/pytest_pod.out 2>&1; r2=$?
echo "[$(ts)] pytest rc $r2: $(tail -n 1 $O/pytest_pod.out | cut -c1-200)"
echo "[$(ts)] ZZ POD-CHECKS DONE v3=$r1 pytest=$r2"
