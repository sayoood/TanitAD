#!/usr/bin/env bash
# pod_mut_q.sh <control-only|all|after|ID[,ID]> : ON THE POD, CPU only, one process (note PREREG_V8F_PROOF_ADD40_N7 tool checks). Copies the files of /workspace/v8proof/imagplan_q into the two small trees
# mut/tree_pristine/src and mut/tree_work/src (cp, nothing deleted), then runs mutlists/common/pod_mut_driver.py on spec_qplan.json. Refuses under 26 GB of free container memory.
set -u
W=/workspace/v8proof; D=$W/imagplan_q; R=$D/mut
M=$(cat /sys/fs/cgroup/memory.max); A=$(grep '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); F=$(( (M - A) / 1073741824 ))
[ "$F" -ge 26 ] || { echo "REFUSED: free container memory $F GB < 26 GB"; exit 2; }
mkdir -p $R/tree_pristine/src $R/tree_work/src $R/tmp
for f in imag_plan.py v8f_t1_eval_imagplan_q.py qhead_model.py test_imag_plan_q.py; do
  cp "$D/$f" $R/tree_pristine/src/$f; cp "$D/$f" $R/tree_work/src/$f
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  control-only) ARGS="--control-only"; OUT=$D/pod_out_control.json;;
  after) ARGS="--control-only"; OUT=$D/pod_out_qplan.json;;   # control after the mutations, into the record of the full run
  all) ARGS=""; OUT=$D/pod_out_qplan.json;;
  *) ARGS="--only $1"; OUT=$D/pod_out_only.json;;
esac
echo "free $F GB; spec sha $(sha256sum $D/spec_qplan.json | cut -c1-16)"
timeout 880 /workspace/venv/bin/python $W/mutlists/common/pod_mut_driver.py $D/spec_qplan.json $OUT $ARGS 2>&1 | tail -n 40 | cut -c1-600
echo "rc ${PIPESTATUS[0]}"
