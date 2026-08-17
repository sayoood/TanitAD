#!/bin/bash
# STEP 6 — acquire a free-tier T4 and, the moment it assigns, ship + drive the 86.
#
# ⛔ ACQUIRE AND RUN ARE ONE JOB ON PURPOSE. A free T4 is a scarce, time-budgeted
# window, and a loop that only ACQUIRED would burn that window sitting idle until
# someone noticed it had landed. `f2_drive86.py` is content-resumable, so re-running
# this script is always safe and never double-detects a clip.
#
# ⚠️ WHY IT RETRIES AT ALL: this account is entitled to T4 and to NOTHING ELSE
# (`--gpu L4` / `--gpu A100` are rejected for entitlement, MEASURED 2026-08-17), and
# T4 assignment was answering 503 Service Unavailable for the whole of that session.
# Capacity is transient; the correct response is to keep asking, not to change GPU.
#
# usage:  bash f6_acquire_and_run.sh [attempts] [sleep_s]
#   env:  REPO  SCRATCH  SESS  CHUNK  AUG120  V2DIR  BANK
set -u
REPO="${REPO:-G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD}"
SCRATCH="${SCRATCH:-$(dirname "$0")/../_work}"
SESS="${SESS:-tanitad-floor86}"
CHUNK="${CHUNK:-22}"
AUG120="${AUG120:-$SCRATCH/aug120}"
V2DIR="${V2DIR:-$SCRATCH/sam3_v2}"
BANK="${BANK:-$SCRATCH/sam3_86_v2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
C="${COLAB_EXE:-C:/Users/Admin/venvs/colab/Scripts/colab.exe}"
PY="${PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
ATTEMPTS="${1:-16}"
SLEEP_S="${2:-30}"

# ⚠️ BOTH ARE REQUIRED, NOT HYGIENE. PYTHONUTF8: colab-cli 0.6.0 opens the script
# with the locale codec (cp1252 here) and any file carrying a ⛔/⚠️ dies with
# UnicodeDecodeError before a line reaches the VM. MSYS_NO_PATHCONV: MSYS rewrites
# the remote `/content/…` into `C:/Program Files/Git/content/…` and the VM's
# contents API answers 500.
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONPATH="$REPO/colab/win_shims"

mkdir -p "$BANK"
SHIP_FLAG="--ship"
if "$C" ls 2>&1 | grep -q "$SESS"; then
  echo "ACQUIRE: session $SESS already live — skipping assignment"
  SHIP_FLAG="${SHIP:-}"          # kernel persists; re-ship only if asked
else
  got=0
  for i in $(seq 1 "$ATTEMPTS"); do
    out=$("$C" new -s "$SESS" --gpu T4 2>&1 | tr -d '\r')
    if echo "$out" | grep -q "READY"; then
      echo "ACQUIRE: T4 assigned on attempt $i"; got=1; break
    fi
    why=$(echo "$out" | grep -oE "Service Unavailable|not have quota or entitlement" | head -1)
    echo "ACQUIRE: attempt $i failed (${why:-unknown})"
    sleep "$SLEEP_S"
  done
  if [ "$got" != 1 ]; then
    echo "ACQUIRE_FAILED after $ATTEMPTS attempts — free-tier T4 capacity is 503."
    echo "  This is NOT a code fault: CPU sessions assign in seconds and L4/A100"
    echo "  are not entitled on this account. Retry later, or authorise units."
    exit 9
  fi
  SHIP_FLAG="--ship"             # a fresh VM always needs the import closure
fi

echo "RUN: driver ${SHIP_FLAG:-(no ship)} · chunk $CHUNK"
"$PY" "$HERE/f2_drive86.py" \
  --aug120 "$AUG120" --v2-dir "$V2DIR" --bank "$BANK" \
  --session "$SESS" --chunk "$CHUNK" --exec-timeout 2400 $SHIP_FLAG \
  --out "$HERE/../raw/f2_run86.json"
rc=$?
echo "DRIVER_RC=$rc"
# ⚠️ An unstopped Colab session burns units for 24 h. Stop it ONLY on success —
# on failure the kernel still holds the assets and a resume is far cheaper.
[ $rc = 0 ] && "$C" stop -s "$SESS" 2>&1 | tail -1
exit $rc
