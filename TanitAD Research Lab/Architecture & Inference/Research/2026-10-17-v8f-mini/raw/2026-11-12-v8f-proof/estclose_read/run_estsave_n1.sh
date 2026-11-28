#!/usr/bin/env bash
# run_estsave_n1.sh -- pod: card PREREG_V8F_PROOF_ADD79 with note N1: self-test of closeread_save_n1.py, then fit and save the estimators N10 and L25 for reader seeds 0 and 1.
# CPU only. No delete, no kill. Markers START_ESTSAVE2.txt / ESTSAVE2_DONE.txt / ESTSAVE2_STOP.txt. Refuses to run twice. The first run (run_estsave.sh) wrote no estimator file.
set -u
W=/workspace/v8proof/closeread
PY=/workspace/venv/bin/python
cd "$W" || exit 2
( set -C; date -u +%Y-%m-%dT%H:%M:%SZ > START_ESTSAVE2.txt ) 2>/dev/null || { echo "REFUSED: START_ESTSAVE2.txt exists (a run was started)"; exit 2; }
stop() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > ESTSAVE2_STOP.txt; exit 3; }
[ -e est ] && stop "OUTPUT-EXISTS est"
for f in tools/closeread_fit.py tools/closest.py tools/closeread_save_n1.py read2/CLOSEREAD_RESULT.json dump/both/reader_s0_fit.npz dump/both/reader_s0_score.npz dump/both/reader_s1_fit.npz dump/both/reader_s1_score.npz; do
  [ -f "$f" ] || stop "MISSING $f"
done
export CLOSEREAD_DIR="$W/tools" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
timeout 3600 "$PY" tools/closeread_save_n1.py --selftest > estsave2_selftest.log 2>&1 || stop "SAVE-SELFTEST-FAIL"
timeout 5400 "$PY" tools/closeread_save_n1.py --dump-dir dump/both --out-dir est --add78-result read2/CLOSEREAD_RESULT.json --seeds 0,1 > estsave2.log 2>&1
rc=$?
[ $rc = 0 ] || stop "SAVE-RC $rc (rc 1 = control e0 did not pass; no estimator file was written)"
date -u +%Y-%m-%dT%H:%M:%SZ > ESTSAVE2_DONE.txt
