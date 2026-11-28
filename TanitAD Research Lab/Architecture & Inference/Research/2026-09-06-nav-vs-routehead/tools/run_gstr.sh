#!/bin/bash
# D-NAVROUTE-1 §4 -- the g_str INTERVENTION arms, ONE surface (Thor), ONE ckpt.
#
# ⛔ EPISODE SUBSET IS MATCHED ACROSS ARMS: every arm passes the SAME
# --episodes-n, so the paired episode-cluster bootstrap compares like with like.
# The full 141-episode roll is ~3.9 h/arm on this box (MEASURED: ~100 s/episode),
# so the panel is run on a 45-episode subset and the reduced n is REPORTED.
#
# Arms, in priority order so a killed run still yields value:
#   1 REPL   baseline replicate  -> the rig's own run-to-run noise floor (MANDATORY:
#            a separated CI from one seed is necessary, not sufficient)
#   2 GZERO  --ablate gstr_zero  -> does removing the strategic goal move the plan?
#   3 BLIND  --ablate-frames     -> DELIBERATE REGRESSION; must degrade, else the
#                                   panel certifies nothing
#   4 GSHUF  --ablate gstr_shuffle --gstr-bank -> window-specific goal information
#
# ⚠️ 2-way concurrency only. MEASURED: 4-way is worse than serialising, and torch
# spawns ~113 threads per process (OMP_NUM_THREADS pins it).
cd /home/nvidia/navroute || exit 1
export OMP_NUM_THREADS=6
export PYTHONPATH=/home/nvidia/navpred/stack:/home/nvidia/navpred/taniteval:/home/nvidia/navpred/stack/scripts
export PYTHONIOENCODING=utf-8
PY=/home/nvidia/venvs/tanitad-train/bin/python
ARM=/home/nvidia/navpred/taniteval/tools/refcv3_arm.py
BANK=/home/nvidia/navpred/navflip_dump
NEP=45

run() {                     # $1 = tag, rest = extra flags
  tag=$1; shift
  $PY $ARM \
      --ckpt /home/nvidia/refcv4b/ckpt_40284_FINAL.pt \
      --config /home/nvidia/refcv4b/config.json \
      --episodes /home/nvidia/navpred/data_eval \
      --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
      --nav-source v72 --grid 2s --action-units steer --device cuda \
      --window-stride 5 --episodes-n $NEP \
      --lead-block /home/nvidia/refcv4b/b1_eval_lead_block.npz \
      --n-boot 2000 --seed 0 "$@" \
      --dump-dir /home/nvidia/navroute/dump_$tag \
      --out /home/nvidia/navroute/${tag}.json \
      > /home/nvidia/navroute/${tag}.log 2>&1
  # opaque marker: DISJOINT from any word a monitor greps for
  echo "QQ${tag}-$?-QQ" >> /home/nvidia/navroute/PROGRESS.txt
}

: > /home/nvidia/navroute/PROGRESS.txt
# batch 1 -- the decisive pair
run REPL &
run GZERO --ablate gstr_zero &
wait
echo "QQBATCH1-QQ" >> /home/nvidia/navroute/PROGRESS.txt
# batch 2 -- the validity control and the bonus arm
run BLIND --ablate-frames &
run GSHUF --ablate gstr_shuffle --gstr-bank $BANK --gstr-shuffle-seed 0 &
wait
echo "QQALLDONE-QQ" >> /home/nvidia/navroute/PROGRESS.txt
