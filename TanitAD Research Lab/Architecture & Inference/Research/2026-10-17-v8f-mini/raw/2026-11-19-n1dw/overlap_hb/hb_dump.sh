#!/bin/bash
# hb_dump.sh FAMILY ARM VARIANT KAPPA -- one D1-HO T1-kin dump through the executor wrapper v8f_t1_eval_hb.py (card PREREG_N1DW_HB_OVERLAP.md).
# FAMILY n1r | n1d | n1dw. Every argument of the run is the family's own banked dump script (navrbd_step0/code/dump_ho.sh, navrbd/dump_rbd.sh,
# n1dw/dump_n1dw.sh) with three changes only: the wrapper prefix (--kappa-max KAPPA --inner <the family tool> --hb-log), the output root
# /root/strat1/n1dw/hb/out/<run>/<family>, and (n1dw) the checkpoint path hb/ckpts (the pod copies were cleared after banking).
# KAPPA 0.2 (the tier of record) or none (the old executor; the K-HB0 control). CPU, 2 threads, nice 10.
FAM=$1; ARM=$2; VAR=${3:-follow}; K=${4:-0.2}
RUN=hb${K}; [ "$K" = none ] && RUN=hbnone
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
HB=/root/strat1/n1dw/hb
I=/root/strat1/navrbd_step0/out/inputs
SUF=""; [ "$VAR" != follow ] && SUF="_$VAR"
OUT=$HB/out/$RUN/$FAM/t1$SUF
mkdir -p $OUT
rm -f $OUT/DONE_$ARM $OUT/FAIL_$ARM
NAVSHA=$(sha256sum $I/navtext_nav_x.npz | cut -c1-64)
case $FAM in
  n1r)  T=/root/strat1/navrb2/tree; INNER=v8f_t1_eval_rb; CK=/root/strat1/navrbd_step0/ckpt/$ARM/ckpt_last.pt
        RBD=$I/navtext_nav_rb_x.npz; FEAT=""; TAG=NAVRBD-HO-${ARM};;
  n1d)  T=/root/strat1/navrbd/tree; INNER=v8f_t1_eval_rbd; CK=/root/strat1/navrbd/runs/$ARM/train/ckpt_last.pt
        RBD=/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz; FEAT="--nav-rb-feat exitdist"; TAG=NAVRBD-HO-${ARM}${SUF};;
  n1dw) T=/root/strat1/n1dw/tree; INNER=v8f_t1_eval_rbd; CK=$HB/ckpts/$ARM/ckpt_last.pt
        RBD=/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz; FEAT="--nav-rb-feat exitdist"; TAG=N1DW-HO-${ARM}${SUF};;
  *) echo "ZZ DUMP $ARM FAIL unknown family $FAM ZZ" > $OUT/FAIL_$ARM; exit 1;;
esac
[ "$FAM" = n1r ] && [ "$VAR" != follow ] && { echo "ZZ DUMP $ARM FAIL n1r has follow only ZZ" > $OUT/FAIL_$ARM; exit 1; }
[ -s "$CK" ] || { echo "ZZ DUMP $ARM FAIL no checkpoint ZZ" > $OUT/FAIL_$ARM; exit 1; }
[ -s "$RBD" ] || { echo "ZZ DUMP $ARM FAIL no $RBD ZZ" > $OUT/FAIL_$ARM; exit 1; }
RBSHA=$(sha256sum $RBD | cut -c1-64)
export PYTHONPATH=$T/stack:$T/taniteval:$T/stack/scripts
taskset -c 48-95 nice -n 10 /root/venv/bin/python $HB/v8f_t1_eval_hb.py --kappa-max $K --inner $INNER --hb-log $OUT/hb_$ARM.json \
  run --ckpt $CK --tag ${TAG}_m10_causal_heads \
  --tracks /root/dscale10k/in/labels/native_tracks_datascale10k.pt --eval-cache /root/strat1/navrbd_step0/cache_view --eval-grid $I/eval_grid_x_feasible.json --eval-join /root/dscale10k/merged/join_X.npz \
  --out-dir $OUT --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $OUT/ticks_$ARM --dump-heads --device cpu \
  --nav-text-npz $I/navtext_nav_x.npz --nav-text-sha256 $NAVSHA --dump-nav --nav-rb on $FEAT --nav-rb-npz $RBD --nav-rb-sha256 $RBSHA --nav-rb-variant $VAR \
  > $OUT/log_$ARM.txt 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "ZZ DUMP $ARM DONE ZZ" > $OUT/DONE_$ARM; else echo "ZZ DUMP $ARM FAIL rc=$rc ZZ" > $OUT/FAIL_$ARM; fi
exit $rc
