#!/bin/bash
# dump_n1dw.sh ARM [VARIANT] -- D1-HO T1-kin dump of one N1DW checkpoint on pod CPU (NAVRBD dump_rbd.sh with the n1dw tree and runs). VARIANT follow (default) | zero (Z-USE beside).
# CPU only (CUDA_VISIBLE_DEVICES empty), 2 threads, taskset 48-95, nice 10. Output /root/strat1/n1dw/out/t1[_VARIANT]/ ; sentinels DONE_<ARM> / FAIL_<ARM>.
ARM=$1; VAR=${2:-follow}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
T=/root/strat1/n1dw/tree
export PYTHONPATH=$T/stack:$T/taniteval
I=/root/strat1/navrbd_step0/out/inputs
RBD=/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz
SUF=""; [ "$VAR" != follow ] && SUF="_$VAR"
OUT=/root/strat1/n1dw/out/t1$SUF
mkdir -p $OUT
rm -f $OUT/DONE_$ARM $OUT/FAIL_$ARM
CK=/root/strat1/n1dw/runs/$ARM/train/ckpt_last.pt
[ -s "$CK" ] || { echo "ZZ DUMP $ARM FAIL no checkpoint ZZ" > $OUT/FAIL_$ARM; exit 1; }
[ -s "$RBD" ] || { echo "ZZ DUMP $ARM FAIL no $RBD ZZ" > $OUT/FAIL_$ARM; exit 1; }
NAVSHA=$(sha256sum $I/navtext_nav_x.npz | cut -c1-64); RBSHA=$(sha256sum $RBD | cut -c1-64)
TAG=N1DW-HO-${ARM}${SUF}
taskset -c 48-95 nice -n 10 /root/venv/bin/python $T/stack/scripts/v8f_t1_eval_rbd.py run --ckpt $CK --tag ${TAG}_m10_causal_heads \
  --tracks /root/dscale10k/in/labels/native_tracks_datascale10k.pt --eval-cache /root/strat1/navrbd_step0/cache_view --eval-grid $I/eval_grid_x_feasible.json --eval-join /root/dscale10k/merged/join_X.npz \
  --out-dir $OUT --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $OUT/ticks_$ARM --dump-heads --device cpu \
  --nav-text-npz $I/navtext_nav_x.npz --nav-text-sha256 $NAVSHA --dump-nav --nav-rb on --nav-rb-feat exitdist --nav-rb-npz $RBD --nav-rb-sha256 $RBSHA --nav-rb-variant $VAR \
  > $OUT/log_$ARM.txt 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "ZZ DUMP $ARM DONE ZZ" > $OUT/DONE_$ARM; else echo "ZZ DUMP $ARM FAIL rc=$rc ZZ" > $OUT/FAIL_$ARM; fi
exit $rc
