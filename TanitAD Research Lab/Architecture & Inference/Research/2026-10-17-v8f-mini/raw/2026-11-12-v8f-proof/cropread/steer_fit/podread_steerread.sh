#!/usr/bin/env bash
# podread_steerread.sh -- pod read of card PREREG_V8F_PROOF_ADD100 (STEERREAD). CPU only (CUDA_VISIBLE_DEVICES=-1 is set here). Tier T0.
# Usage: GAPRANGE=<GAPRANGE_RESULT.json on the pod> S7_OUT=<pytest output of the dev box on the pod> bash podread_steerread.sh controls
#        bash podread_steerread.sh names
# controls: needs DONE_HEADS.txt and no STOP_HEADS.txt in s_out/queue; runs the selftest of read_steerread.py, then read_steerread.py --stage controls (writes read/STEERREAD_CONTROLS.json,
#           one line per control s1..s8).
# names:    reads that file; the reader refuses unless s1..s8 all passed (then nothing is opened). Writes read/STEERREAD_RESULT.json.
# Never reads twice into one file; deletes, moves and stops nothing. Last line: ZZ PODREAD-STEERREAD <stage> DONE rc <n>   or   ZZ PODREAD-STEERREAD STOP <why>.
set -u
STAGE=${1:-}
S=${S_ROOT:-/workspace/v8proof/cropread/s_out}
R1=${CR_RUN1:-/workspace/v8proof/cropread/run1}
BOX=${CR_BOX:-/workspace/v8proof/cropread/enc_out/box}
K=${CR_KIT:-/workspace/v8proof/cropread/kit/steer_fit}
KF=${CR_KIT_FIT:-/workspace/v8proof/cropread/kit/fit}
PY=${CR_PY:-/workspace/venv/bin/python}
OUT=$S/read
CT=${CR_CRTOOLS:-/workspace/v8proof/closeread/tools}
EPMAP=${CR_EPMAP:-/workspace/v8proof/closeread/episode_map.json}
PRESENT=${CR_PRESENT:-/workspace/v8proof/closeread/dump/both}
MINGB=${CR_MIN_FREE_GB:-8}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-STEERREAD STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export KINCLOSE_CRTOOLS=/workspace/v8proof/closeread/tools      # without it the selftest does not find closeread_fit.py and closest.py on the pod
mm=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge "$MINGB" ] || stop "MEMORY free ${free_gb} GB"; fi
for f in "$K/read_steerread.py" "$K/steer_common.py" "$KF/read_cropread.py" "$KF/crop_fit.py" "$KF/kinclose2.py" "$KF/kinclose3.py" "$KF/kinclose4.py" "$KF/kinclose5.py" "$KF/kinclose6.py" "$KF/make_part_joins.py"; do [ -f "$f" ] || stop "MISSING-KIT-FILE $f"; done
for f in "$CT/closeread_fit.py" "$CT/closest.py" "$EPMAP" "$PRESENT/reader_s0_score.npz" "$PRESENT/reader_s1_score.npz"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
mkdir -p "$OUT"
if [ "$STAGE" = controls ]; then
  [ -f "$S/queue/DONE_HEADS.txt" ] || stop "HEADS-QUEUE-NOT-DONE"
  [ -f "$S/queue/STOP_HEADS.txt" ] && stop "HEADS-QUEUE-STOPPED"
  [ -e "$OUT/STEERREAD_CONTROLS.json" ] && stop "ALREADY-READ controls"
  [ -n "${GAPRANGE:-}" ] && [ -f "$GAPRANGE" ] || stop "GAPRANGE not set or missing (GAPRANGE=<path of GAPRANGE_RESULT.json>)"
  [ -n "${S7_OUT:-}" ] && [ -f "$S7_OUT" ] || stop "S7_OUT not set or missing (S7_OUT=<path of the pytest output>)"
  ( cd "$K" && "$PY" "$K/read_steerread.py" --selftest > "$OUT/selftest_pod.log" 2>&1 ); st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  ( cd "$K" && "$PY" "$K/read_steerread.py" --stage controls --root "$S" --run1 "$R1" --box-dir "$BOX" --present-dump "$PRESENT" --episode-map "$EPMAP" --closeread-tools "$CT" --gaprange-result "$GAPRANGE" --s7-pytest-out "$S7_OUT" --boot 2000 > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^s[1-8] (PASS|FAIL)|^ZZ READ-STEERREAD|Traceback|Error|REFUSED' "$OUT/read_controls.log" | cut -c1-300 | head -n 16
  echo "ZZ PODREAD-STEERREAD controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/STEERREAD_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/STEERREAD_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$K" && "$PY" "$K/read_steerread.py" --stage names --root "$S" --run1 "$R1" --box-dir "$BOX" --controls "$OUT/STEERREAD_CONTROLS.json" --present-dump "$PRESENT" --episode-map "$EPMAP" --closeread-tools "$CT" --boot 2000 > "$OUT/read_names.log" 2>&1 ); rc=$?
  cut -c1-300 "$OUT/read_names.log" | head -n 80
  echo "ZZ PODREAD-STEERREAD names DONE rc $rc $(ts)"
fi
