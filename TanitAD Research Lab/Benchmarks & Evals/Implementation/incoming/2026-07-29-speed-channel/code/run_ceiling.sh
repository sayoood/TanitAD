#!/bin/bash
# Task #44 step 2 — the pre-registered speed-ceiling arm (PREREG_speed_ceiling.md, 0aba041).
# BOTH arms run HERE on the SAME 40-episode cache so the paired test cannot be crossing hosts.
set -u
S=/workspace/TanitAD/stack
TEV=/workspace/tev/taniteval
OUT=/workspace/ceiling
CK=/workspace/v4instr/v4fs_ckpt.pt
CFG=/workspace/v4instr/v4fs_config.json
VAL=/workspace/val40cache
ANCH=/workspace/v4run/flagship_v4_anchors_dense.pt
mkdir -p "$OUT"
cd "$S" || exit 1
for spec in "produced40::" "mixed40:vt_band,vt_speed:"; do
  key="${spec%%:*}"; rest="${spec#*:}"; chans="${rest%%:*}"
  echo "=== $key  oracle_channels='${chans}' ==="
  extra=""; [ -n "$chans" ] && extra="--oracle-channels $chans"
  PYTHONPATH="$S:$TEV:$S/scripts" OMP_NUM_THREADS=6 python3 -u scripts/eval_flagship_v4.py \
    --ckpt "$CK" --head-config "$CFG" --val-cache "$VAL" --anchors-dense "$ANCH" \
    --goal-mode produced $extra \
    --key "v4fs-$key" --out "$OUT/$key.json" --results-dir "$OUT" --device cuda \
    > "$OUT/$key.log" 2>&1
  echo "  rc=$?  $(grep -h '^\[driving\]' "$OUT/$key.log" | tail -1)"
done
echo "=== CEILING ARMS DONE ==="
