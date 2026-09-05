#!/bin/bash
# H-REFAV1-SURFACE-1 Stage B on Thor — B1 (pure-goal weights) + B2 (T0 oracle goal), ONE pass.
#
# Waits for the running chord arm (B3, the deliberate regression) to finish BY EXPLICIT PID —
# never `pgrep -f`, which self-matches the ssh command line — then launches on the SCORE half
# only (71 episodes, the pre-registered held-out split; weights were chosen on SELECT).
#
# --with-oracle-goal-arm rolls `cl_oraclegoal` (stamped T0 by the tool) beside `cl` in the SAME
# pass, so B1 and B2 cost one run between them.
set -u
R=/home/nvidia/refav1_ccos
WAIT_PID=${1:-0}
LOG=$R/stageB.log
exec >> "$LOG" 2>&1
echo "=== stageB launcher $(date -u +%FT%TZ) waiting on PID $WAIT_PID ==="
if [ "$WAIT_PID" != "0" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "PID $WAIT_PID gone at $(date -u +%FT%TZ)"
fi
sleep 60
# refuse to start beside another arm (the bracket keeps the pattern disjoint from the match)
N=$(ps -eo args | grep -c "refav1_ar[m].py")
echo "ZZOTHERARMS-${N}ZZ"
if [ "$N" != "0" ]; then echo "REFUSING: another arm is running"; exit 3; fi

PY=/home/nvidia/venvs/tanitad-train/bin/python
cd $R/repo || exit 4
export PYTHONPATH=$R/repo/stack
export OMP_NUM_THREADS=6
$PY taniteval/tools/refav1_arm.py \
  --ckpt /home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt \
  --config /home/nvidia/experiments/refav1-b1-v72-ep3-speed/config.json \
  --cache $R/score_cache \
  --episodes $R/score_eps \
  --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --nav /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --lead-block $R/b1_eval_lead_block.npz \
  --device cuda --window-stride 40 --no-navshuf --episodes-n 0 \
  --action-units kappa \
  --cost-metric ccos --cost-weights 0.0,0.0,64.29715042415070 \
  --with-oracle-goal-arm \
  --dump-dir $R/dump_B1_puregoal \
  --out $R/rec_B1_puregoal.json \
  --arm refav1-21109-B1-puregoal-SCORE
echo "B1_EXIT=$?" > $R/B1.EXIT
cat $R/B1.EXIT
date -u +%FT%TZ > $R/B1.DONE
echo "=== stageB done $(date -u +%FT%TZ) ==="
