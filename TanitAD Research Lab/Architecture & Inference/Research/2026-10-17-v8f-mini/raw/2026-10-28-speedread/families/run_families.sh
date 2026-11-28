#!/usr/bin/env bash
# Runs stack/scripts/v8f_paired_families.py (the run tree's f3971165 copy, unmodified) the way code/devbox_ops/run_speedcal.sh stage_pairfam does,
# on the SPEEDREAD dumps. Reads F:/TanitAD-scratch/speedread (never written); writes F:/TanitAD-scratch/speedread_result/families/. CPU only.
#   pairfam_read_s<s>   read - own   (the PRIMARY contrast)
#   pairfam_gt_s<s>     gt - own     (the T0 ceiling, a control only)
#   subctl_own_vs_banked_s<s>   the SUBSTITUTE control: SPEEDREAD's `own` dump vs the banked X_s dump, both against the banked eval json
# Resumable: a finished output (.json) is skipped.
set -u
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
E=F:/TanitAD-scratch/speedread
PFT=$E/tree_f3971165/stack/scripts/v8f_paired_families.py
O=F:/TanitAD-scratch/speedread_result/families
B2=C:/Users/Admin/box2_ship/logs
say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
run() {   # run <name> <args...>
  local n=$1; shift
  [ -f $O/$n.json ] && { say "$n: done earlier, skipped"; return 0; }
  say "$n: start"
  $PY -B -I $PFT "$@" --out $O/$n.json --md $O/$n.md > $O/$n.out 2>&1; say "$n: rc=$?"
}
for s in 0 1; do
  run pairfam_read_s$s --a $E/eval_s$s/eval_M2b-mixo_s$s-speedread-read.npz --b $E/eval_s$s/eval_M2b-mixo_s$s-speedread-own.npz --label-a read --label-b own
  run pairfam_gt_s$s --a $E/eval_s$s/eval_M2b-mixo_s$s-speedread-gt.npz --b $E/eval_s$s/eval_M2b-mixo_s$s-speedread-own.npz --label-a gt --label-b own
done
PB0=$B2/v8f_mixo/eval/eval_M2b-mixo_s0
PB1=$B2/v8f_kindx_s1/eval/eval_M2b-mixo_s1-kindx-own
run subctl_own_vs_banked_s0 --a $E/eval_s0/eval_M2b-mixo_s0-speedread-own.npz --a-json $PB0.json --b $PB0.npz --b-json $PB0.json --label-a own_via_speedread_driver --label-b banked_X_s0
run subctl_own_vs_banked_s1 --a $E/eval_s1/eval_M2b-mixo_s1-speedread-own.npz --a-json $PB1.json --b $PB1.npz --b-json $PB1.json --label-a own_via_speedread_driver --label-b banked_X_s1
say "ZZFAMILIES-DONE"
