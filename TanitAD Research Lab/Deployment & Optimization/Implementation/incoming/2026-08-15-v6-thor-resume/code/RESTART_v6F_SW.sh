#!/bin/bash
# RESTART RECIPE for the live v6F S-W run — GENERATED FROM /proc/25477 ITSELF.
#
# WHY THIS FILE EXISTS. The run has ~7.3 days left and NO SUPERVISOR MANIFEST.
# If the trainer dies at 04:00 the failure mode is not "it stays down" (a loop
# checks every ~30 min) — it is "someone retypes a 90-flag command under time
# pressure and gets one wrong". This is the argv VERBATIM, so no flag is
# reconstructed from memory or from a runbook.
#
# GENERATED %Y-%m-%dT%H:%M:%SZ from pid 25477 while it was RUNNING.
# ⛔ DOES NOT AUTO-RUN. Read the preconditions, then run it deliberately.
#
# PRECONDITIONS, each earned:
#   1. Confirm the old trainer is ACTUALLY GONE by EXPLICIT PID — never
#      pkill -f, which self-matches your own ssh command line.
#         kill -0 25477 2>/dev/null && echo STILL ALIVE  # must print nothing
#   2. --resume auto is already in the argv below: it picks up from ckpt.pt.
#      The last confirmed checkpoint step is in metrics.json / train_log.jsonl.
#      ⚠️ The stage-lineage guard added 2026-08-16 refuses a ckpt written by a
#      DIFFERENT stage, so a wrong-stage resume now fails loudly at startup
#      rather than silently adopting the wrong step.
#   3. Launch DETACHED with setsid + ssh -f. A nohup ... & inside a single ssh
#      HOLDS THE CHANNEL OPEN and looks exactly like a hang.
#   4. Verify from a SEPARATE connection, never from the launching one.
#
set -euo pipefail
export PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/stack/scripts
export OMP_NUM_THREADS=6
cd /home/nvidia
exec /home/nvidia/venvs/tanitad-train/bin/python \n     -u \n     /home/nvidia/TanitAD/stack/scripts/train_v6_staged.py \n     --stage \n     S-W \n     --out \n     /home/nvidia/experiments/v6F-SW-30k \n     --resume \n     auto \n     --in-channels \n     9 \n     --frame-h \n     256 \n     --frame-w \n     640 \n     --patch \n     16 \n     --enc-dim \n     768 \n     --enc-depth \n     12 \n     --enc-heads \n     12 \n     --grad-checkpoint \n     --readout-grid \n     4 \n     --readout-dim \n     128 \n     --pred-modern \n     --pred-dim \n     1024 \n     --pred-depth \n     12 \n     --pred-heads \n     16 \n     --window \n     6 \n     --horizons \n     1 \n     2 \n     4 \n     --d-tac \n     768 \n     --d-str \n     512 \n     --d-goal-embed \n     128 \n     --adapter-hidden \n     512 \n     --n-candidates \n     8 \n     --param-budget \n     350000000 \n     --f-hidden-tac \n     1024 \n     --f-hidden-str \n     1024 \n     --f-blocks \n     6 \n     --vit5-encoder \n     --n-registers \n     4 \n     --plan-steps \n     60 \n     --dt \n     0.1 \n     --a-max \n     4.0 \n     --kappa-max \n     0.2 \n     --uplink \n     stopgrad \n     --ema-decay \n     0.996 \n     --o1-k \n     10 \n     --w-o1-ctrl \n     1.0 \n     --w-o1-fact \n     1.0 \n     --w-o1-scene \n     0.3 \n     --dkappa \n     0.02 \n     --daccel \n     2.0 \n     --rand-dkappa-max \n     0.05 \n     --rand-daccel-max \n     3.0 \n     --w-o2 \n     1.0 \n     --o2-tau-s \n     2.0 \n     --w-o3 \n     1.0 \n     --o3-mode \n     action \n     --o3-blocks \n     2 \n     --o3-block-h \n     2 \n     --o3-block-w \n     2 \n     --o3-band-rows \n     0 \n     --o4-alpha \n     1.0 \n     --o4-floor \n     0.25 \n     --w-o5 \n     1.0 \n     --o5-k \n     60 \n     --o5-mode \n     uniform \n     --w-o6 \n     0.1 \n     --sigreg-slices \n     512 \n     --sigreg-free-dims \n     0 \n     --spectrum-every \n     200 \n     --w-t1 \n     1.0 \n     --w-s1 \n     1.0 \n     --v2-cache \n     /home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl \n     --v2-val-cache \n     /home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \n     --v2-lru \n     64 \n     --frame-hfov \n     120.0 \n     --projection \n     cylindrical \n     --require-parity \n     --eps-per-batch \n     4 \n     --max-horizon \n     60 \n     --steps \n     30000 \n     --batch \n     8 \n     --lr \n     0.0001 \n     --wd \n     0.05 \n     --clip \n     1.0 \n     --log-every \n     50 \n     --save-every \n     250 \n     --device \n     cuda \n     --seed \n     0 \n     --dry-steps \n     2 \n     --dry-batch \n     2 \n     --dry-k \n     12
