#!/bin/bash
# GATE 0 for the refcv4b landing. Runs POD-SIDE. Does NOT touch the GPU.
#
# ⛔ Everything here is a POSITIVE assertion. An empty or short result from any
# probe is INCONCLUSIVE, never a pass -- on this fleet a failed read and a
# genuine absence look identical.
#
# Emits ONE opaque marker per check as GG<name>=<value>GG so a client-side
# parser never matches the words this script itself contains.
set -uo pipefail
OUT=/workspace/experiments/refcv4b-b1-v72-40k
FINAL=$OUT/ckpt_40284_FINAL.pt
TARGET=40284

fail() { echo "GGGATE0=FAILGG $*"; exit 1; }

# --- 1. the done-marker ------------------------------------------------------
[ -s "$OUT/summary.json" ] || fail "no summary.json"
DONE=$(python3 -c '
import json, sys
d = json.load(open("'"$OUT"'/summary.json"))
print("1" if d.get("done") is True else "0")
print(d.get("final_step", -1))
' 2>/dev/null)
D1=$(echo "$DONE" | sed -n 1p); D2=$(echo "$DONE" | sed -n 2p)
[ "$D1" = "1" ] || fail "summary.json done != true ($(cat "$OUT/summary.json"))"
echo "GGDONE=1GG GGFINAL_STEP=${D2}GG"

# --- 2. the step actually in metrics.jsonl (independent of the marker) --------
MSTEP=$( { cat "$OUT/metrics.jsonl" 2>/dev/null || true; } | tail -n 400 | python3 -c '
import sys, json
s = 0
for line in sys.stdin:
    line = line.strip()
    if line:
        try: s = max(s, int(json.loads(line).get("step", 0)))
        except Exception: pass
print(s)' 2>/dev/null | tail -1)
case "$MSTEP" in ""|*[!0-9]*) fail "metrics step unreadable" ;; esac
[ "$MSTEP" -ge "$TARGET" ] || fail "metrics step $MSTEP < $TARGET"
echo "GGMETRICS_STEP=${MSTEP}GG"

# --- 3. supervisor and trainer must be GONE ----------------------------------
# character classes so grep cannot match its own argv
SUP=$( ps -eo args= | grep -c 'sup[_]refcv4b' )
TRN=$( ps -eo args= | grep -c 'refc[_]v3[_]train' )
echo "GGSUP=${SUP}GG GGTRN=${TRN}GG"
if [ "$SUP" -ne 0 ] || [ "$TRN" -ne 0 ]; then
  echo "GGGATE0=BLOCKEDGG supervisor/trainer still alive -- kill by EXPLICIT PID:"
  ps -eo pid=,args= | grep 'sup[_]refcv4b' | grep -v ' grep '
  ps -eo pid=,args= | grep 'refc[_]v3[_]train' | grep -v ' grep '
  exit 2
fi

# --- 4. freeze the rolling ckpt to an IMMUTABLE name and md5 it ---------------
# `MILESTONES` does not contain 40284, so the FINAL weights live in the ROLLING
# ckpt.pt -- a file whose NAME never changes while its contents do.
[ -s "$OUT/ckpt.pt" ] || fail "no ckpt.pt"
if [ ! -s "$FINAL" ]; then
  cp "$OUT/ckpt.pt" "$FINAL" || fail "copy failed"
fi
A=$(md5sum "$OUT/ckpt.pt" | cut -d' ' -f1)
B=$(md5sum "$FINAL"       | cut -d' ' -f1)
if [ ${#A} -ne 32 ] || [ ${#B} -ne 32 ]; then fail "md5 INCONCLUSIVE (${#A}/${#B})"; fi
[ "$A" = "$B" ] || fail "md5 MISMATCH rolling=$A final=$B"
echo "GGMD5=${B}GG GGSIZE=$(stat -c%s "$FINAL")GG"

# --- 5. the checkpoint's OWN step, read from the file ------------------------
CKSTEP=$(PYTHONPATH=/workspace/TanitAD/stack python3 -c '
import torch
ck = torch.load("'"$FINAL"'", map_location="cpu", weights_only=False, mmap=True)
sd = ck["model"]
assert len(sd) > 400, "state_dict too small: %d" % len(sd)   # CONTROL
print(int(ck["step"]))' 2>/dev/null | tail -1)
case "$CKSTEP" in ""|*[!0-9]*) fail "ckpt step unreadable" ;; esac
[ "$CKSTEP" -eq "$TARGET" ] || fail "ckpt step $CKSTEP != $TARGET"
echo "GGCKPT_STEP=${CKSTEP}GG"

echo "GGGATE0=PASSGG"
