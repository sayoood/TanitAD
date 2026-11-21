#!/usr/bin/env bash
# run_estsave.sh -- pod: card PREREG_V8F_PROOF_ADD79, section 2: self-test of closeread_save.py, then fit and save the estimators N10 and L25 for reader seeds 0 and 1
# from the banked dumps of card ADD77 (control e0 inside the tool; nothing is written when it fails). CPU only. No delete, no kill.
# Markers START_ESTSAVE.txt / ESTSAVE_DONE.txt / ESTSAVE_STOP.txt in the closeread folder. Refuses to run twice.
set -u
W=/workspace/v8proof/closeread
PY=/workspace/venv/bin/python
cd "$W" || exit 2
( set -C; date -u +%Y-%m-%dT%H:%M:%SZ > START_ESTSAVE.txt ) 2>/dev/null || { echo "REFUSED: START_ESTSAVE.txt exists (a run was started)"; exit 2; }
stop() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > ESTSAVE_STOP.txt; exit 3; }
[ -e est ] && stop "OUTPUT-EXISTS est"
for f in tools/closeread_fit.py tools/closest.py tools/closeread_save.py read2/CLOSEREAD_RESULT.json dump/both/reader_s0_fit.npz dump/both/reader_s0_score.npz dump/both/reader_s1_fit.npz dump/both/reader_s1_score.npz; do
  [ -f "$f" ] || stop "MISSING $f"
done
export CLOSEREAD_DIR="$W/tools" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
timeout 3600 "$PY" tools/closest.py --selftest > estsave_selftest.log 2>&1
tail -n 1 estsave_selftest.log | grep -q "ZZ CLOSEST-SELFTEST rc 0" || stop "CLOSEST-SELFTEST-FAIL"
timeout 3600 "$PY" tools/closeread_save.py --selftest >> estsave_selftest.log 2>&1 || stop "SAVE-SELFTEST-FAIL"
timeout 5400 "$PY" tools/closeread_save.py --dump-dir dump/both --out-dir est --add78-result read2/CLOSEREAD_RESULT.json --seeds 0,1 > estsave.log 2>&1
rc=$?
[ $rc = 0 ] || stop "SAVE-RC $rc (rc 1 = control e0 did not pass; no estimator file was written)"
date -u +%Y-%m-%dT%H:%M:%SZ > ESTSAVE_DONE.txt
