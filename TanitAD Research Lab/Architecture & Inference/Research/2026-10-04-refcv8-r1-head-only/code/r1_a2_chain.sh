#!/bin/bash
# SPEC_R1_A2 chain (REGISTERED sha256 23c407b6..., 2026-10-05T04:53:28Z). Starts only after the registered arms chain
# wrote ALLDONE. Heads are staged in /dev/shm/r1_heads/<arm>/heads.pt (RAM, not disk). Every GPU pass is its own flock
# acquisition (<= ~40 min); the ARTIFACT says a pass ran. Order (MM 2026-10-05): A2-1 train-fit -> A2-2 w choice on
# train-fit -> H5g / H5cg on eval -> A2-3 H1n -> score.
set -u
R=/home/nvidia/refcv8_r1
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack:/home/nvidia/refcv7_run/fec3a0dccf/taniteval
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
A=$R/arms
HD=/dev/shm/r1_heads
cd $R/code
free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
until grep -q ALLDONE $R/logs/arms_chain.out; do
  ps -eo args | grep -q "r1_arms_chai[n]" || { echo "[a2] arms chain not running and no ALLDONE -- stop"; exit 1; }
  sleep 60
done
for arm in H1 H2 H2s H3 H4 H5 H5c; do
  [ -s $HD/$arm/heads.pt ] || { echo "[a2] heads for $arm not staged in $HD -- stop"; exit 1; }
done
job () {   # tag artifact attempts bash-command
  local tag=$1 art=$2 n=$3 cmd=$4 i
  for i in $(seq 1 $n); do
    if [ -s "$art" ]; then echo "[a2] $tag banked"; return 0; fi
    if [ "$(free_gb)" -lt 20 ]; then echo "[a2] $tag REFUSED: $(free_gb) GB free"; return 1; fi
    echo "[a2] $(date -u +%H:%M:%S) start $tag attempt $i"
    flock $LOCK bash -c "$cmd" >> $R/logs/a2_$tag.log 2>&1 < /dev/null
    echo "[a2] $(date -u +%H:%M:%S) end $tag rc=$? artifact=$( [ -s "$art" ] && echo yes || echo NO )"
    sleep 20
  done
  [ -s "$art" ]
}
E="$PY r1_arms.py --phase eval --eval-split train --eval-seeds 0 --heads-dir $HD --out $A"
# ---- A2-1: train-fit, plain (H0..H4, H5c) ------------------------------------------------------------------- #
job tf1 $A/H2s/trainfit_s0.pt 3 "for a in H0 H1 H2 H2s; do $E --arm \$a || exit 1; done" || exit 1
job tf2 $A/H5c/trainfit_s0.pt 3 "for a in H3 H4 H5c; do $E --arm \$a || exit 1; done" || exit 1
# ---- A2-1 for H5 = A2-2's w grid on train-fit (with forcing; w = 1 is H5's own train-fit) ------------------- #
job tfw $A/H5g_w10/trainfit_s0.pt 3 "for w in 1 3 10; do $E --arm H5 --ctrl --guidance-w \$w --tag H5g_w\$w || exit 1; done" || exit 1
$PY r1_a2_choose_w.py $A $R/cache > $R/logs/a2_choose_w.log 2>&1
W=$(grep -oE "CHOSEN_W=[0-9]+" $R/logs/a2_choose_w.log | cut -d= -f2)
[ -n "$W" ] || { echo "[a2] no w chosen -- stop"; exit 1; }
echo "[a2] chosen w = $W"
$PY r1_trainfit_score.py --cache $R/cache --arms $A > $R/logs/a2_trainfit_score.log 2>&1
echo "[a2] trainfit score rc=$?"
# ---- A2-2 on eval: H5g and the control H5cg, w = chosen, both sampler seeds, with forcing --------------------- #
G="$PY r1_arms.py --phase eval --heads-dir $HD --out $A --ctrl --guidance-w $W"
job H5g0 $A/H5g/eval_s0.pt 3 "$G --arm H5 --tag H5g --eval-seeds 0" || exit 1
job H5g1 $A/H5g/eval_s1.pt 3 "$G --arm H5 --tag H5g --eval-seeds 1" || exit 1
job H5cg0 $A/H5cg/eval_s0.pt 3 "$G --arm H5c --tag H5cg --eval-seeds 0" || exit 1
job H5cg1 $A/H5cg/eval_s1.pt 3 "$G --arm H5c --tag H5cg --eval-seeds 1" || exit 1
# ---- A2-3: H1n ------------------------------------------------------------------------------------------------ #
job H1n_train $A/H1n/train_record.json 3 "$PY r1_arms.py --phase train --arm H1n --out $A" || exit 1
job H1n_eval $A/H1n/eval_s1.pt 3 "$PY r1_arms.py --phase eval --arm H1n --out $A --eval-seeds 0,1" || exit 1
$PY r1_score.py --cache $R/cache --arms $A > $R/logs/score_a2.log 2>&1
echo "[a2] score rc=$? ALLDONE $(date -u +%H:%M:%S)"
