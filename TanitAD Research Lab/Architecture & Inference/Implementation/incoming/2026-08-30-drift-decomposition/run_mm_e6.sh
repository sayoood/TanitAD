#!/usr/bin/env bash
# MM-E6 runner — one arm per invocation, RC captured BEFORE any pipe (MM-C3).
#
# ⛔ THE EXIT CODE THAT CANNOT GO RED. `cmd | tee log` reports tee's status, so a
# crashed probe exits 0 and a monitor calls it a pass. Every artifact here is
# written by `{ cmd; echo "RC=$?"; } > file`, and the pass criterion below also
# requires a NON-ZERO COUNT of arm blocks in the JSON — a green RC over an empty
# result is the failure this guards.
#
# ⚠️ The emitted marker is DISJOINT from anything the command line contains, so a
# grep of this log can never match its own echoed command (the polling-monitor
# self-match trap, measured three times in this programme).
set -u
WORK="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8e7cfa33-c625-47cf-88aa-711db80ac113/scratchpad"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ARM="${1:?usage: run_mm_e6.sh <arm> [tag]}"
TAG="${2:-$ARM}"

cd "$WORK" || exit 2
export CUDA_VISIBLE_DEVICES=""      # CPU-only: the 4060 is held by D-SAFE-CAL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONIOENCODING=utf-8
export MME6_ARMS="$ARM"
export MME6_OUT="$WORK/mm_e6_${TAG}.json"

{ "$PY" -u mm_e6_drift_decompose.py; echo "RC=$?"; } > "$WORK/mm_e6_${TAG}.log" 2>&1

RC=$(grep -c '^RC=0$' "$WORK/mm_e6_${TAG}.log")
N=$("$PY" - "$WORK/mm_e6_${TAG}.json" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    print(0); raise SystemExit
try:
    print(len(json.loads(p.read_text(encoding="utf-8")).get("arms", {})))
except Exception:
    print(0)
PYEOF
)
# PASS = clean exit AND a non-zero count of banked arm blocks.
if [ "$RC" = "1" ] && [ "${N:-0}" -gt 0 ]; then
  echo "ZZ${TAG}-OK-${N}ZZ"
else
  echo "ZZ${TAG}-BAD-rc${RC}-arms${N:-0}ZZ"
fi
