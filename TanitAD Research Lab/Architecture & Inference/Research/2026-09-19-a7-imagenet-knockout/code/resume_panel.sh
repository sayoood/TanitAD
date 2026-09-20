#!/usr/bin/env bash
# RESUME the A7 -> P0 -> P0b chain after a PAUSE. One command, nothing to remember.
#
# PAUSED 2026-09-20 12:13 CEST on the PI's request ("I need the gpu for another more urgent
# task. Pause it only, so we can resume if it is free again"). A7-IN-s0 was at step 1,070/2,000.
#
# ⛔ WHY A REAL "PAUSE" WAS NOT POSSIBLE, stated so nobody retries it: a SUSPENDED process still
# HOLDS its VRAM. Freeing the card requires the trainer to EXIT. So the pause is a clean stop,
# and the resumability is the panel's own: an arm whose `a7_arm_check.json` says VALID is SKIPPED,
# and a partial arm directory is MOVED ASIDE (never deleted) and re-run from scratch.
#
# ⛔ AND WHY A7-IN-s0 IS NOT RESUMED FROM ITS step-1,000 CHECKPOINT, although it COULD be.
# `refc_v3_train.py` auto-resumes whenever `ckpt.pt` exists in `--out` (strict: model + optimizer
# + step). Tempting -- it would save ~1.4 h. It is WRONG for THIS arm: `--trunk-bn-recalib 256`
# runs ONCE, BEFORE step 1, by design. A resumed run would re-run the recalibration on a
# PARTIALLY-TRAINED model, which is a different experiment from the pre-registered one, and the
# arm is an ImageNet-vs-random knockout whose whole point is that nothing else differs. Correctness
# beats 1.4 h. (For P0/P0b, which carry no recalib flag, the objection would not apply -- but the
# panel moves partials aside regardless, which is what guarantees a clean arm.)
#
# WHAT IS PRESERVED: A7-IN-s0/train.log and its run/ (incl. ckpt.pt at step 1,000) are moved to
# A7-IN-s0.aborted-<ts>/ on resume, not deleted -- the log is the only record of why.
# WHAT IS LOST: A7-IN-s0's ~1.6 h of partial training. No completed arm existed, so nothing else.
#
# ⛔ GATE UNCHANGED at 4,300 MiB (PI, 2026-09-20). If the card is still busy, BOTH jobs simply
# wait -- that is the design working. Do not lower or raise it to force a start.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
REPO=/d/Projects/TanitAD
OUT=/c/Users/Admin/tanitad-caches/p-panel-20260920
A7OUT=/c/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919

echo "ZZRESUME-GPU-BEFORE $("$PY" /c/Users/Admin/qland/boxstat.py 2>/dev/null)ZZ"

nohup bash "$HERE/launch_a7_when_free.sh" >> "$A7OUT/launcher.log" 2>&1 &
echo "ZZRESUME-LAUNCHER-PID $! ZZ"

nohup "$PY" "$REPO/stack/scripts/p_runner.py" \
  --out "$OUT" --a7-dir "$A7OUT" \
  --run-tree C:/Users/Admin/tanitad-a7-run \
  --base-config C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/config.json \
  --trainer-md5 e7ac9ec33ff44568a5e931619087cec9 \
  --labels-md5 eefc38d1453bd1c73802d44d45affced \
  --authorise-arm P0-REPLICATE --authorise-arm P0B-REPLICATE \
  --stop-after P0B-REPLICATE --poll-s 120 --max-wait-h 72 \
  >> "$OUT/p_runner.log" 2>&1 &
echo "ZZRESUME-PRUNNER-PID $! ZZ"

sleep 20
# ⛔ assert they are REALLY there: this family of failures reports success and leaves nothing
# running. Count by EXE PATH so the check cannot match its own command line (the pgrep -f trap).
n=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.ExecutablePath -like 'C:\Users\Admin\venvs\tanitad*' -and \$_.CommandLine -like '*p_runner*' }).Count" 2>/dev/null | tr -d '\r')
echo "ZZRESUME-PRUNNER-ALIVE ${n:-?} ZZ"
echo "ZZRESUME-DONE $(date '+%Y-%m-%d %H:%M:%S')ZZ"
