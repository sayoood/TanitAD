#!/bin/bash
# NAVRBD held-out T1-kin dump of one N1D / N1Dx checkpoint (CPU, 2 threads: the step-0 dumps' thread count, step0/code/dump_ho.sh).
# usage: dump_rbd.sh ARM [VARIANT] [MAXWIN]     ARM = N1D_s0 | N1D_s1 | N1Dx_s0 ; VARIANT follow (default) | real | zero | shuffle ; MAXWIN 0 = all windows (the step-0 dumps' setting)
# Same tool family, grid, join, tracks, text-nav and thread count as step 0; the tool is the NAVRBD wrapper v8f_t1_eval_rbd.py (15-column channel, c13 dead-reckoned), run from the 0c3b374f tree.
# Step-0 inputs are READ ONLY here. Output: /root/strat1/navrbd/out/t1[_VARIANT]/ ; sentinel files DONE_<ARM> / FAIL_<ARM> there.
# STAGED, NOT LAUNCHED (runs after the chain: dump_rbd_all.sh).
cd /root/strat1/navrbd || exit 1
ARM=$1; VAR=${2:-follow}; MAXW=${3:-0}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
T=/root/strat1/navrbd/tree
export PYTHONPATH=$T/stack:$T/taniteval
I=/root/strat1/navrbd_step0/out/inputs
RBD=/root/strat1/inputs/strat1/navtext_nav_rbd_x.npz
SUF=""; [ "$VAR" != follow ] && SUF="_$VAR"
OUT=/root/strat1/navrbd/out/t1$SUF; EX=""
[ "$MAXW" != 0 ] && { OUT=/root/strat1/navrbd/out/smoke$SUF; EX="--max-windows $MAXW"; }
mkdir -p $OUT
rm -f $OUT/DONE_$ARM $OUT/FAIL_$ARM
CK=/root/strat1/navrbd/runs/$ARM/train/ckpt_last.pt
[ -s "$CK" ] || { echo "ZZ DUMP $ARM FAIL no checkpoint ZZ" > $OUT/FAIL_$ARM; exit 1; }
[ -s "$RBD" ] || { echo "ZZ DUMP $ARM FAIL no $RBD ZZ" > $OUT/FAIL_$ARM; exit 1; }
NAVSHA=$(sha256sum $I/navtext_nav_x.npz | cut -c1-64); RBSHA=$(sha256sum $RBD | cut -c1-64)
TAG=NAVRBD-HO-${ARM}${SUF}
/root/venv/bin/python $T/stack/scripts/v8f_t1_eval_rbd.py run --ckpt $CK --tag ${TAG}_m10_causal_heads \
  --tracks /root/dscale10k/in/labels/native_tracks_datascale10k.pt --eval-cache /root/strat1/navrbd_step0/cache_view --eval-grid $I/eval_grid_x_feasible.json --eval-join /root/dscale10k/merged/join_X.npz \
  --out-dir $OUT --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $OUT/ticks_$ARM --dump-heads --device cpu \
  --nav-text-npz $I/navtext_nav_x.npz --nav-text-sha256 $NAVSHA --dump-nav --nav-rb on --nav-rb-feat exitdist --nav-rb-npz $RBD --nav-rb-sha256 $RBSHA --nav-rb-variant $VAR $EX \
  > $OUT/log_$ARM.txt 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "ZZ DUMP $ARM DONE ZZ" > $OUT/DONE_$ARM; else echo "ZZ DUMP $ARM FAIL rc=$rc ZZ" > $OUT/FAIL_$ARM; fi
