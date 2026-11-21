#!/usr/bin/env bash
# run_closeread2.sh -- pod: card PREREG_V8F_PROOF_ADD78 (CLOSEREAD2): the self-test of closeread_fit_n2.py, then the fit and read on the banked dumps of card ADD77.
# CPU only, no GPU job, no dump. No delete, no kill. Markers START_CLOSEREAD2.txt / CLOSEREAD2_DONE.txt / CLOSEREAD2_STOP.txt. Refuses to run twice.
set -u
W=/workspace/v8proof/closeread
I=/workspace/v8proof/integ10k
PY=/workspace/venv/bin/python
RUN=I10RQ50KF5Hb20
cd "$W" || exit 2
( set -C; date -u +%Y-%m-%dT%H:%M:%SZ > START_CLOSEREAD2.txt ) 2>/dev/null || { echo "REFUSED: START_CLOSEREAD2.txt exists (a run was started)"; exit 2; }
stop() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > CLOSEREAD2_STOP.txt; exit 3; }
[ -e read2 ] && stop "OUTPUT-EXISTS read2"
for f in tools/closeread_fit.py tools/closeread_fit_n2.py episode_map.json dump/both/reader_s0_fit.npz dump/both/reader_s0_score.npz dump/both/reader_s1_fit.npz dump/both/reader_s1_score.npz dump/both/reader_s0_lists.json dump/both/reader_s1_lists.json; do
  [ -f "$f" ] || stop "MISSING $f"
done
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 timeout 3600 "$PY" tools/closeread_fit_n2.py --selftest2 > selftest2_pod.log 2>&1
tail -n 1 selftest2_pod.log | grep -q "ZZ CLOSEREAD2-SELFTEST rc 0" || stop "SELFTEST2-FAIL"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 timeout 5400 "$PY" tools/closeread_fit_n2.py --dump-dir dump/both --out-dir read2 --episode-map episode_map.json \
  --rulelog0 "$I/t1/${RUN}_s0/ticks/rule_log_${RUN}_s0.npz" --qmap0 "$I/qmap_sweep/QMAP_LR16_s0_q50.json" \
  --rulelog1 "$I/t1/${RUN}_s1/ticks/rule_log_${RUN}_s1.npz" --qmap1 "$I/qmap_sweep/QMAP_LR16_s1_q50.json" \
  --seeds 0,1 --boot 2000 > fit_read2.log 2>&1
rc=$?
[ $rc = 0 ] || stop "FIT-RC $rc (rc 1 = a control did not pass; no name was written)"
date -u +%Y-%m-%dT%H:%M:%SZ > CLOSEREAD2_DONE.txt
