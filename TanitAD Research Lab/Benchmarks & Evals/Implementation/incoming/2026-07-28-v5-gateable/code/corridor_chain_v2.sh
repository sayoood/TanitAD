#!/bin/bash
# ⭐ THE CO-PRIMARY CHAIN ON THE v2 CORPUS, END TO END, ON THE REAL POD2 CACHE.
#
#   eval_flagship_v4  ->  windows_<key>.pt (dense)  ->  gate_emitters corridor
#   ->  corridor_<key>.json  ->  run_gate.py register  ->  run_gate.py check
#
# Every leg is the SHIPPED command, not a re-implementation. The checkpoint is a
# 25-step v5-shaped SMOKE checkpoint at 176x624 — its ADE is meaningless and is
# never quoted; what is under test is whether the ARTIFACTS the gate consumes
# can be produced on a v2 cache at all.
#
# ⛔ Nothing here launches a run and nothing touches pod1.
set -u
ROOT=/workspace/v5gate
STACK=$ROOT/stack
RUN=$ROOT/run/flagship-v5-SMOKE-176x624
RES=$ROOT/results
RAW=$ROOT/raw
KEY=v5smoke-176x624
mkdir -p "$RES" "$RAW" "$ROOT/gates"
export PYTHONPATH=$STACK:$STACK/scripts:/root/taniteval
export OMP_NUM_THREADS=6
cd "$STACK"

echo "=================== LEG 1: MODE B eval on the v2 val cache ==================="
python3 -u scripts/eval_flagship_v4.py \
  --ckpt "$RUN/ckpt.pt" \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --require-parity --v2-lru 16 \
  --episodes "${EPISODES:-12}" --stride 8 --batch 8 --device cuda \
  --skip-bench \
  --results-dir "$RES" \
  --key "$KEY" --out "$RES/$KEY.json" 2>&1 | tail -40
echo "LEG1_EXIT=$?"
ls -la "$RES" || true

echo
echo "=================== LEG 2: the co-primary emitter ==================="
python3 -u scripts/gate_emitters.py corridor \
  --windows "$RES/windows_$KEY.pt" \
  --out-corridor "$RES/corridor_$KEY.json" 2>&1 | tail -40
echo "LEG2_EXIT=$?"

echo
echo "=================== LEG 3: register a card AT THE BLIND HORIZON =========="
echo "--- GATE_PROTOCOL 0.3 refuses K<=20; this must FAIL ---"
python3 -u scripts/run_gate.py register \
  --run "$KEY" --gate-step 25 \
  --primary-metric ade_0_2s --primary-threshold 0.60 \
  --co-primary-threshold 0.35 --co-primary-horizon-K 20 \
  --co-primary-junction-threshold 0.50 \
  --card "$ROOT/gates/$KEY-K20.card.json" 2>&1 | tail -20
echo "LEG3_EXIT_EXPECT_NONZERO=$?"

echo
echo "=================== LEG 4: register the PREP card's horizon K=60 =========="
rm -f "$ROOT/gates/$KEY-K60.card.json"
python3 -u scripts/run_gate.py register \
  --run "$KEY" --gate-step 25 \
  --primary-metric ade_0_2s --primary-threshold 0.60 \
  --co-primary-threshold 0.35 --co-primary-horizon-K 60 \
  --co-primary-junction-threshold 0.50 \
  --secondary "wm_canary_ade_2s<=0.55" \
  --secondary "miss_2m<=0.10" \
  --lever-family encoder-geometry --restarts-used 0 \
  --card "$ROOT/gates/$KEY-K60.card.json" 2>&1 | tail -25
echo "LEG4_EXIT=$?"

echo
echo "=================== LEG 5: check WITH the K=20 corridor artifact =========="
echo "--- the card is K=60; the only artifact v2 can produce is K=20 ---"
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-K60.card.json" \
  --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --corridor-json "$RES/corridor_$KEY.json" \
  --out "$RAW/gate_check_K60_with_K20_corridor.json" 2>&1 | tail -35
echo "LEG5_EXIT=$?"

echo
echo "=================== LEG 6: check with NO corridor artifact ==============="
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-K60.card.json" \
  --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --out "$RAW/gate_check_K60_no_corridor.json" 2>&1 | tail -35
echo "LEG6_EXIT=$?"

echo
echo "=================== LEG 7: the GREEN twin — a card AT the artifact's K ==="
echo "--- proves check CONSUMES a v2-produced corridor block when the K agrees;"
echo "--- the card is written by hand because register refuses K=20 by design ---"
python3 -u - <<'PY'
import json
import os
from pathlib import Path
root = Path("/workspace/v5gate")
src = root / "gates" / "v5smoke-176x624-K60.card.json"
dst = root / "gates" / "v5smoke-176x624-K20-HANDWRITTEN.card.json"
c = json.loads(src.read_text())
c["co_primary_horizon_K"] = 20
c["_HANDWRITTEN"] = ("register REFUSES K<=20 (GATE_PROTOCOL 0.3). This card is "
                     "written by hand ONLY to prove that `check` consumes a "
                     "v2-produced corridor block end to end. It is NOT an "
                     "admissible gate card and no verdict from it is quotable.")
dst.write_text(json.dumps(c, indent=2))
print("wrote", dst)
PY
python3 -u scripts/run_gate.py check \
  --card "$ROOT/gates/$KEY-K20-HANDWRITTEN.card.json" \
  --log "$RUN/train_log.jsonl" \
  --eval-json "$RES/$KEY.json" \
  --corridor-json "$RES/corridor_$KEY.json" \
  --out "$RAW/gate_check_K20_handwritten_GREEN.json" 2>&1 | tail -45
echo "LEG7_EXIT=$?"
echo "=================== DONE ==================="
