#!/usr/bin/env bash
# ⭐ LAUNCH o13p30k ON THOR — written BEFORE postrain30k finishes so the next arm
# starts in the SAME TURN, not after a round of composition.
#
# ⛔ EVERY STEP IS GATED. Three separate incidents in this programme came from
# launching onto a tree that was not what it looked like:
#   - a pod's stack drifted and a launch from it would have resurrected a fixed bug
#   - a one-file ship was grep-verified for its HEADLINE marker but not its
#     DEPENDENCY, and died instantly with KeyError('ep_idx')
#   - a supervisor RESURRECTED a finished run because no done-marker was written
# So: refuse if postrain30k is still alive, refuse if the swap md5 is wrong,
# refuse if either O13 marker is missing, refuse if the arm dir already exists.
set -u
STAGED=/home/nvidia/staging/train_v6_staged_O13.py
LIVE=/home/nvidia/TanitAD/stack/scripts/train_v6_staged.py
WANT_MD5=ed82d89f41a14e66c40aa0e3a64826d6
OUT=/home/nvidia/v7tiny/o13p30k
W_O13=0.1

# ---- GATE 1: the incumbent must be FINISHED, not merely quiet ----------------
n=$(ps -eo args | grep -c "[t]rain_v6_staged")
if [ "$n" != "0" ]; then
  echo "ZZREFUSE a trainer is still running ($n procs) ZZ"; exit 1
fi
if [ ! -f /home/nvidia/v7tiny/postrain30k/summary.json ]; then
  echo "ZZREFUSE postrain30k has no summary.json - it did not finish cleanly ZZ"
  exit 1
fi
echo "ZZGATE1 postrain30k done, no trainer running ZZ"

# ---- GATE 2: swap the trainer, verified by md5 -------------------------------
[ -f "$STAGED" ] || { echo "ZZREFUSE staged trainer missing ZZ"; exit 1; }
got=$(md5sum "$STAGED" | cut -d' ' -f1)
[ "$got" = "$WANT_MD5" ] || { echo "ZZREFUSE staged md5 $got != $WANT_MD5 ZZ"; exit 1; }
cp -n "$LIVE" /home/nvidia/staging/train_v6_staged_PRE_O13.py 2>/dev/null || true
cp "$STAGED" "$LIVE"
live=$(md5sum "$LIVE" | cut -d' ' -f1)
[ "$live" = "$WANT_MD5" ] || { echo "ZZREFUSE live md5 $live after copy ZZ"; exit 1; }
echo "ZZGATE2 trainer swapped, md5 $live ZZ"

# ---- GATE 3: BOTH markers, not just the headline one -------------------------
a=$(grep -c "o13_ego_dynamics_loss" "$LIVE")
b=$(grep -c 'future_poses": b.get' "$LIVE")
c=$(grep -c "w-o13-ego" "$LIVE")
if [ "$a" -lt 3 ] || [ "$b" -lt 1 ] || [ "$c" -lt 1 ]; then
  echo "ZZREFUSE markers fn=$a whitelist=$b cli=$c ZZ"; exit 1
fi
/home/nvidia/venvs/tanitad-train/bin/python -c \
  "import ast;ast.parse(open('$LIVE').read());print('ZZGATE3 parses ZZ')" || exit 1
echo "ZZGATE3 markers fn=$a whitelist=$b cli=$c ZZ"

# ---- GATE 4: never clobber an existing arm ----------------------------------
[ -e "$OUT" ] && { echo "ZZREFUSE $OUT already exists ZZ"; exit 1; }

# ---- LAUNCH: byte-identical to postrain30k except --out and the o13 knobs ----
cd /home/nvidia/TanitAD/stack || exit 1
PYTHONPATH=/home/nvidia/TanitAD/stack nohup \
  /home/nvidia/venvs/tanitad-train/bin/python scripts/train_v6_staged.py \
  --out "$OUT" --stage S-W \
  --v2-cache /home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --patch 16 --enc-dim 128 --enc-depth 3 --enc-heads 4 \
  --pred-dim 256 --pred-depth 3 --pred-heads 4 --readout-grid 4 --readout-grid-w 8 \
  --readout-dim 64 --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 8 --d-tac 128 \
  --d-str 64 --steps 30000 --batch 8 --v2-lru 64 --log-every 200 --save-every 2500 \
  --seed 0 --spectrum-accum 43 --sigreg-slices 512 --o5-form l1 \
  --sigreg-subspaces 32 --w-o5 1.0 --w-o6 0.1 --w-o1-ctrl 0 --w-o1-fact 0 \
  --w-o1-scene 0 --w-o2 0 --w-o3 0 \
  --init-from /home/nvidia/v7tiny/distill_init.pt \
  --w-o13-ego "$W_O13" --o13-k 4 \
  > /home/nvidia/v7tiny/o13p30k.log 2>&1 &
sleep 90
p=$(ps -eo args | grep -c "[t]rain_v6_staged")
echo "ZZLAUNCH procs=$p ZZ"
if [ -f "$OUT/train_log.jsonl" ]; then
  echo "ZZFIRSTROW $(head -c 200 "$OUT/train_log.jsonl" | tr -d '\n') ZZ"
else
  echo "ZZNOLOG yet - check $OUT.log ZZ"
  tail -5 /home/nvidia/v7tiny/o13p30k.log
fi
