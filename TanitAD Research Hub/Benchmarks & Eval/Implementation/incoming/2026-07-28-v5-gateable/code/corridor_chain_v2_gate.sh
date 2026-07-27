#!/bin/bash
# Legs 5-7 of `corridor_chain_v2.sh` — the `run_gate.py check` legs.
# (Split out after the first run: `check` writes with `--json`, not `--out`.)
set -u
ROOT=/workspace/v5gate
STACK=$ROOT/stack
RUN=$ROOT/run/flagship-v5-SMOKE-176x624
RES=$ROOT/results
RAW=$ROOT/raw
KEY=v5smoke-176x624
export PYTHONPATH=$STACK:$STACK/scripts:/root/taniteval
cd "$STACK"

echo "=================== LEG 5: card K=60, artifact K=20 ==================="
echo "--- the ONLY corridor artifact a v2 open-loop eval can produce is K=20 ---"
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-K60.card.json" \
  --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --corridor-json "$RES/corridor_$KEY.json" \
  --json "$RAW/gate_check_K60_with_K20_corridor.json" 2>&1 | tail -25
echo "LEG5_EXIT=$?"

echo
echo "=================== LEG 6: card K=60, NO corridor artifact ============"
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-K60.card.json" \
  --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --json "$RAW/gate_check_K60_no_corridor.json" 2>&1 | tail -30
echo "LEG6_EXIT=$?"

echo
echo "=================== LEG 7: GREEN — card K=20 (HANDWRITTEN), artifact K=20"
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-K20-HANDWRITTEN.card.json" \
  --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --corridor-json "$RES/corridor_$KEY.json" \
  --json "$RAW/gate_check_K20_handwritten_GREEN.json" 2>&1 | tail -45
echo "LEG7_EXIT=$?"
echo "=================== DONE ==================="
ls -la "$RAW"
