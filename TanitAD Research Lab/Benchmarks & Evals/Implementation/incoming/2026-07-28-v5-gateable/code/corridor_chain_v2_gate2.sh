#!/bin/bash
# Legs 5-7, re-run at a gate step the SMOKE log actually reaches.
#
# The first attempt registered `--gate-step 25` while the smoke's `train_log.jsonl`
# only carries `step 0` (--log-every defaults to 50 and the run was stopped at
# ~step 25). `check` therefore returned NOT_YET — correctly, and before reading
# the corridor at all, which is exactly what makes it useless as a demonstration
# of corridor CONSUMPTION. Re-registered at `--gate-step 0`.
#
# ⛔ These cards exist to exercise the plumbing. No verdict from them is quotable:
# the checkpoint is a 25-step smoke and its ADE is meaningless.
set -u
ROOT=/workspace/v5gate
STACK=$ROOT/stack
RUN=$ROOT/run/flagship-v5-SMOKE-176x624
RES=$ROOT/results
RAW=$ROOT/raw
KEY=v5smoke-176x624
export PYTHONPATH=$STACK:$STACK/scripts:/root/taniteval
cd "$STACK"

for K in 60; do
  rm -f "$ROOT/gates/$KEY-g0-K$K.card.json"
  python3 -u scripts/run_gate.py register \
    --run "$KEY" --gate-step 0 \
    --primary-metric ade_0_2s --primary-threshold 0.60 \
    --co-primary-threshold 0.35 --co-primary-horizon-K $K \
    --co-primary-junction-threshold 0.50 \
    --secondary "wm_canary_ade_2s<=0.55" --secondary "miss_2m<=0.10" \
    --lever-family encoder-geometry --restarts-used 0 \
    --card "$ROOT/gates/$KEY-g0-K$K.card.json" > /dev/null 2>&1
  echo "registered $KEY-g0-K$K"
done
python3 -u - <<'PY'
import json
from pathlib import Path
root = Path("/workspace/v5gate")
src = root / "gates" / "v5smoke-176x624-g0-K60.card.json"
dst = root / "gates" / "v5smoke-176x624-g0-K20-HANDWRITTEN.card.json"
c = json.loads(src.read_text())
c["co_primary_horizon_K"] = 20
c["_HANDWRITTEN"] = ("register REFUSES K<=20 (GATE_PROTOCOL 0.3). Hand-written "
                     "ONLY to prove `check` consumes a v2-produced corridor "
                     "block end to end. NOT an admissible gate card.")
dst.write_text(json.dumps(c, indent=2))
print("wrote", dst.name)
PY

echo
echo "=============== LEG 5: card K=60  vs  the K=20 artifact v2 can make ======"
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-g0-K60.card.json" --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" --corridor-json "$RES/corridor_$KEY.json" \
  --json "$RAW/gate_check_K60_with_K20_corridor.json" 2>&1 | tail -30
echo "LEG5_EXIT=$?"

echo
echo "=============== LEG 6: card K=60, NO corridor artifact at all ============"
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-g0-K60.card.json" --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --json "$RAW/gate_check_K60_no_corridor.json" 2>&1 | tail -30
echo "LEG6_EXIT=$?"

echo
echo "=============== LEG 7: GREEN — card K=20, artifact K=20 =================="
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-g0-K20-HANDWRITTEN.card.json" --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" --corridor-json "$RES/corridor_$KEY.json" \
  --json "$RAW/gate_check_K20_handwritten_GREEN.json" 2>&1 | tail -40
echo "LEG7_EXIT=$?"
echo "=============== DONE ==============="
