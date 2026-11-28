#!/usr/bin/env bash
# podread_cropread.sh -- pod read of card PREREG_V8F_PROOF_ADD99 (CROPREAD). CPU only (CUDA_VISIBLE_DEVICES=-1 is set here). Tier T0.
# Usage: GAPRANGE=<GAPRANGE_RESULT.json on the pod> C6_OUT=<pytest output of the dev box on the pod> bash podread_cropread.sh controls
#        bash podread_cropread.sh names
# controls: needs DONE_FITQ.txt and no STOP_FITQ.txt; runs the selftest of read_cropread.py, then read_cropread.py --stage controls (writes read/CROPREAD_CONTROLS.json; prints one line per control).
# names:    reads that file; the reader refuses unless c1..c7 all passed (then nothing is opened); with p1 failed it opens arm D only. Writes read/CROPREAD_RESULT.json.
# Never reads twice into one file; deletes, moves and stops nothing. Last line: ZZ PODREAD-CROPREAD <stage> DONE rc <n>   or   ZZ PODREAD-CROPREAD STOP <why>.
set -u
STAGE=${1:-}
R=${CR_ROOT:-/workspace/v8proof/cropread}
K=$R/kit/fit
PY=${CR_PY:-/workspace/venv/bin/python}
OUT=$R/read
CT=${CR_CRTOOLS:-/workspace/v8proof/closeread/tools}
EPMAP=${CR_EPMAP:-/workspace/v8proof/closeread/episode_map.json}
PRESENT=${CR_PRESENT:-/workspace/v8proof/closeread/dump/both}
MINGB=${CR_MIN_FREE_GB:-8}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-CROPREAD STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge "$MINGB" ] || stop "MEMORY free ${free_gb} GB"; fi
for f in read_cropread.py kinclose2.py kinclose3.py kinclose4.py kinclose5.py kinclose6.py make_part_joins.py; do [ -f "$K/$f" ] || stop "MISSING-KIT-FILE $f"; done
for f in "$CT/closeread_fit.py" "$CT/closest.py" "$EPMAP" "$PRESENT/reader_s0_score.npz" "$PRESENT/reader_s1_score.npz"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
mkdir -p "$OUT"
if [ "$STAGE" = controls ]; then
  [ -f "$R/queue/DONE_FITQ.txt" ] || stop "FIT-QUEUE-NOT-DONE"
  [ -f "$R/queue/STOP_FITQ.txt" ] && stop "FIT-QUEUE-STOPPED"
  [ -e "$OUT/CROPREAD_CONTROLS.json" ] && stop "ALREADY-READ controls"
  [ -n "${GAPRANGE:-}" ] && [ -f "$GAPRANGE" ] || stop "GAPRANGE not set or missing (GAPRANGE=<path of GAPRANGE_RESULT.json>)"
  [ -n "${C6_OUT:-}" ] && [ -f "$C6_OUT" ] || stop "C6_OUT not set or missing (C6_OUT=<path of the pytest output>)"
  ( cd "$K" && CROPREAD_CRTOOLS="$CT" "$PY" "$K/read_cropread.py" --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$K" && "$PY" "$K/read_cropread.py" --stage controls --root "$R" --present-dump "$PRESENT" --episode-map "$EPMAP" --closeread-tools "$CT" --gaprange-result "$GAPRANGE" --c6-pytest-out "$C6_OUT" --boot 2000 > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^(c[1-7]|p1) (PASS|FAIL)|^ZZ READ-CROPREAD|Traceback|Error|REFUSED' "$OUT/read_controls.log" | cut -c1-300 | head -n 16
  echo "ZZ PODREAD-CROPREAD controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/CROPREAD_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/CROPREAD_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$K" && "$PY" "$K/read_cropread.py" --stage names --root "$R" --controls "$OUT/CROPREAD_CONTROLS.json" --present-dump "$PRESENT" --episode-map "$EPMAP" --closeread-tools "$CT" --boot 2000 > "$OUT/read_names.log" 2>&1 ); rc=$?
  cut -c1-300 "$OUT/read_names.log" | head -n 80
  echo "ZZ PODREAD-CROPREAD names DONE rc $rc $(ts)"
fi
