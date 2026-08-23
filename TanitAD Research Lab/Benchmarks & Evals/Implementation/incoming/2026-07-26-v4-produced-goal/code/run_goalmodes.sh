#!/usr/bin/env bash
# The three goal-mode runs on flagship-v4-fromscratch @ step 15000 (EVAL POD).
#   oracle   -> must be BIT-IDENTICAL to the 2026-07-25 22:07 baseline
#   produced -> the deployable path (model's own goal_head)
#   neutral  -> the control that makes the gap readable
export PYTHONPATH=/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts
export TANITEVAL_STACK_OVERRIDE=/root/v4eval/stack
export CUBLAS_WORKSPACE_CONFIG=:4096:8
M=/workspace/models/flagship-v4-fromscratch-15k
OUT=/root/v4eval/results_goalmode
mkdir -p "$OUT"
cd /root/v4eval/stack/scripts || exit 1

echo "=== 15k run head_cfg ==="
python3 - <<'PY'
import json
c = json.load(open("/workspace/models/flagship-v4-fromscratch-15k/config.json"))
h = c.get("head_cfg", {})
print("  goal_dropout:", h.get("goal_dropout"), "| ego_dropout:", h.get("ego_dropout"))
print("  cond_*:", {k: v for k, v in h.items() if k.startswith("cond_")})
a = c.get("args", {})
print("  anchors_dense:", a.get("anchors_dense"))
PY

for MODE in oracle produced neutral; do
  echo
  echo "############################################################"
  echo "### --goal-mode $MODE   ($(date -u +%H:%M:%SZ))"
  echo "############################################################"
  python3 -u eval_flagship_v4.py \
    --ckpt          "$M"/ckpt_step15000.pt \
    --anchors-dense "$M"/flagship_v4_anchors_dense.pt \
    --head-config   "$M"/config.json \
    --val-cache     /root/valdata/physicalai-val-0c5f7dac3b11 \
    --goal-mode     "$MODE" \
    --key           "v4-15k-goal-$MODE" \
    --out           "$OUT/v4-15k-goal-$MODE.json" \
    --results-dir   "$OUT" \
    --episodes 40 --stride 8 --batch 16 --device cuda 2>&1 \
    | grep -vE "^  \[v4-eval\] (planner-path|canary)" | tail -60
  echo "EXIT_$MODE=$?"
done
echo "ALL_RUNS_DONE $(date -u +%H:%M:%SZ)"
