#!/bin/bash
# hb_dump_n1dm.sh ARM VARIANT KAPPA -- one D1-HO T1-kin dump of an N1DM checkpoint on pod CPU; the N1DH dump script with the tree and runs renamed (family n1d: v8f_t1_eval_rbd as the inner tool, exitdist feat, the D channel npz).
# ARM N1DM_s0|N1DM_s1 (the real arms) | N1DM_s0_zuse|N1DM_s1_zuse (checkpoint copies with the K=3 branches zeroed: plan = head_o alone) | N1DM_s0_unif|N1DM_s1_unif (copies with route_head zeroed: p uniform 1/3, the constant mixture); VARIANT follow|zero (zero = --nav-rb-variant zero: the D CHANNEL zeroed, the zero twin)   KAPPA 0.2 (executor of record, v8f_t1_eval_hb.py --kappa-max 0.2) | none (old executor, beside)
# The wrapper is the N1DW overlap kit copy (hb/v8f_t1_eval_hb.py, sha256 74fd8445, equal to the git blob at bcf267ce); it is not in the code-of-record tree and is staged beside it by n1dm_stage.sh.
# CPU only, 2 threads, taskset 48-95, nice 10. Output /root/strat1/n1dm/out/hb<K>/t1[_zero]/ ; sentinels DONE_<ARM> / FAIL_<ARM>.
ARM=$1; VAR=${2:-follow}; K=${3:-0.2}
RUN=hb${K}; [ "$K" = none ] && RUN=hbnone
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
N=/root/strat1/n1dm; T=$N/tree
I=/root/strat1/navrbd_step0/out/inputs
RBD=/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz
SUF=""; [ "$VAR" != follow ] && SUF="_$VAR"
OUT=$N/out/$RUN/t1$SUF
mkdir -p $OUT
rm -f $OUT/DONE_$ARM $OUT/FAIL_$ARM
CK=$N/runs/$ARM/train/ckpt_last.pt
[ -s "$CK" ] || { echo "ZZ DUMP $ARM FAIL no checkpoint ZZ" > $OUT/FAIL_$ARM; exit 1; }
[ -s "$RBD" ] || { echo "ZZ DUMP $ARM FAIL no $RBD ZZ" > $OUT/FAIL_$ARM; exit 1; }
[ -s "$N/hb/v8f_t1_eval_hb.py" ] || { echo "ZZ DUMP $ARM FAIL no hb wrapper ZZ" > $OUT/FAIL_$ARM; exit 1; }
NAVSHA=$(sha256sum $I/navtext_nav_x.npz | cut -c1-64); RBSHA=$(sha256sum $RBD | cut -c1-64)
TAG=N1DM-HO-${ARM}${SUF}
export PYTHONPATH=$T/stack:$T/taniteval:$T/stack/scripts
taskset -c 48-95 nice -n 10 /root/venv/bin/python $N/hb/v8f_t1_eval_hb.py --kappa-max $K --inner v8f_t1_eval_rbd --hb-log $OUT/hb_$ARM.json \
  run --ckpt $CK --tag ${TAG}_m10_causal_heads \
  --tracks /root/dscale10k/in/labels/native_tracks_datascale10k.pt --eval-cache /root/strat1/navrbd_step0/cache_view --eval-grid $I/eval_grid_x_feasible.json --eval-join /root/dscale10k/merged/join_X.npz \
  --out-dir $OUT --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $OUT/ticks_$ARM --dump-heads --device cpu \
  --nav-text-npz $I/navtext_nav_x.npz --nav-text-sha256 $NAVSHA --dump-nav --nav-rb on --nav-rb-feat exitdist --nav-rb-npz $RBD --nav-rb-sha256 $RBSHA --nav-rb-variant $VAR \
  > $OUT/log_$ARM.txt 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "ZZ DUMP $ARM DONE ZZ" > $OUT/DONE_$ARM; else echo "ZZ DUMP $ARM FAIL rc=$rc ZZ" > $OUT/FAIL_$ARM; fi
exit $rc
