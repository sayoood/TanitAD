#!/bin/bash
# NAVRBD step 0: T1-kin dump of one N1R checkpoint on the held-out windows. usage: dump_ho.sh ARM [MAXWIN]   (CPU, 2 threads, v8f_t1_eval_rb.py of the e71782a3 tree: the NAVRB2 read's own wrapper)
cd /root/strat1/navrbd_step0 || exit 1
ARM=$1; MAXW=${2:-0}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
T=/root/strat1/navrb2/tree
export PYTHONPATH=$T/stack:$T/taniteval
I=/root/strat1/navrbd_step0/out/inputs
TAG=NAVRBD-HO-${ARM}; OUT=/root/strat1/navrbd_step0/out/t1${MAXW/#0/}; [ "$MAXW" != 0 ] && OUT=/root/strat1/navrbd_step0/out/smoke
mkdir -p $OUT
rm -f $OUT/DONE_$ARM $OUT/FAIL_$ARM
EX=""; [ "$MAXW" != 0 ] && EX="--max-windows $MAXW"
NAVSHA=$(sha256sum $I/navtext_nav_x.npz | cut -c1-64); RBSHA=$(sha256sum $I/navtext_nav_rb_x.npz | cut -c1-64)
/root/venv/bin/python $T/stack/scripts/v8f_t1_eval_rb.py run --ckpt /root/strat1/navrbd_step0/ckpt/$ARM/ckpt_last.pt --tag ${TAG}_m10_causal_heads \
  --tracks /root/dscale10k/in/labels/native_tracks_datascale10k.pt --eval-cache /root/strat1/navrbd_step0/cache_view --eval-grid $I/eval_grid_x_feasible.json --eval-join /root/dscale10k/merged/join_X.npz \
  --out-dir $OUT --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $OUT/ticks_$ARM --dump-heads --device cpu \
  --nav-text-npz $I/navtext_nav_x.npz --nav-text-sha256 $NAVSHA --dump-nav --nav-rb on --nav-rb-npz $I/navtext_nav_rb_x.npz --nav-rb-sha256 $RBSHA --nav-rb-variant follow $EX \
  > $OUT/log_$ARM.txt 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "ZZ DUMP $ARM DONE ZZ" > $OUT/DONE_$ARM; else echo "ZZ DUMP $ARM FAIL rc=$rc ZZ" > $OUT/FAIL_$ARM; fi
