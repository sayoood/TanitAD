#!/usr/bin/env bash
# r10_chain.sh -- the R10-A oracle rolls on the dev box (RTX 4060), one arm after the other. Detach it; NEVER wait on it.
#   usage:  OUTROOT=E:/r10_pitch_out ARMS="static H C" [EXTRA="--every 40 --max-windows 12"] bash r10_chain.sh
#   markers (grep these, do not poll with model turns):  ZZR10ARM-<arm>-START/DONE/FAIL-<rc>ZZ and finally ZZR10CHAIN-DONEZZ in $OUTROOT/chain.log
#   a finished arm (its dump.json exists) is skipped, so re-running the same line RESUMES. Tree = the scratch tree this file lives in.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="${TREE:-$(cd "$HERE/../../../.." && pwd)}"
PY="${PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
OUTROOT="${OUTROOT:?set OUTROOT}"
ARMS="${ARMS:-static H C}"
EXTRA="${EXTRA:-}"
SPLIT="${SPLIT:-C:/Users/Admin/scratch_r10/ref/DEV_HOLD_SPLIT.json}"     # E0i's DEV_HOLD_SPLIT.json, sha256 ca5af020...8db0a (the repo copy under e0/)
W="$(cygpath -w "$TREE")"
export PYTHONPATH="$W\\stack;$W\\stack\\scripts;$W\\taniteval"
export REFCV7_KIT=E:/refcv6_eval_kit
export REFCV7_REMAP_OVERRIDES="${REMAP:-C:/Users/Admin/scratch_r10/refcv8_remap_devbox.json}"
export R9_S2_WINDOWS=E:/refcv8_eval_kit/windows/s2_windows.json
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p "$OUTROOT"
for ARM in $ARMS; do
  OUT="$OUTROOT/$ARM"
  if [ -f "$OUT/dump.json" ]; then echo "ZZR10ARM-${ARM}-SKIP-DONEZZ" | tee -a "$OUTROOT/chain.log"; continue; fi
  echo "ZZR10ARM-${ARM}-START $(date -u +%H:%M:%S)ZZ" | tee -a "$OUTROOT/chain.log"
  "$PY" -X utf8 "$HERE/r10_oracle_roll.py" --arm "$ARM" --table "$HERE/raw/pitch_table_eval139.npz" \
     --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json \
     --windows "$SPLIT" --split dev --out "$OUT" --pose-sync off --require-ema $EXTRA > "$OUTROOT/$ARM.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/dump.json" ]; then echo "ZZR10ARM-${ARM}-DONE $(date -u +%H:%M:%S)ZZ" | tee -a "$OUTROOT/chain.log"
  else echo "ZZR10ARM-${ARM}-FAIL-${rc}ZZ (see $OUTROOT/$ARM.log)" | tee -a "$OUTROOT/chain.log"; fi
done
echo "ZZR10CHAIN-DONEZZ $(date -u +%H:%M:%S)" | tee -a "$OUTROOT/chain.log"
