#!/usr/bin/env bash
# podread_labelunsafe10k.sh -- pod-side driver of card PREREG_V8F_PROOF_ADD135 (LABELUNSAFE10K). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout. It deletes, moves and stops nothing.
# Stages:
#   tests     runs the tool's test file with the pod pytest and writes /workspace/v8proof/labelunsafe/lead_tests/pod_pytest.out (refuses when that file exists; basetemp under tests_tmp/)
#   controls  needs lead_tests/dev_pytest.out (pushed by the lead) and lead_tests/pod_pytest.out; runs read_labelunsafe10k.py controls -> read_out/LABELUNSAFE10K_CONTROLS.json
#   names     runs read_labelunsafe10k.py names on that controls file -> read_out/LABELUNSAFE10K_RESULT.json and LABELUNSAFE10K_VERDICT.md (the tool refuses unless all five controls pass)
# Usage: bash podread_labelunsafe10k.sh tests|controls|names
# Prints the tool's lines (ctl u<k> PASS|FAIL <reason>, the last line of the stage) and: ZZ PODREAD-LABELUNSAFE10K <stage> DONE rc <n> <time>   (or ZZ PODREAD-LABELUNSAFE10K STOP <why>).
set -u
STAGE=${1:-}
V=/workspace/v8proof; GE=$V/labelunsafe; K=$GE/kit; OUT=$GE/read_out; LT=$GE/lead_tests; TT=$GE/tests_tmp
T1=$V/integ10k/t1
PY=/workspace/venv/bin/python
PYT=$V/pytest_env/bin/python
TOOL=$K/read_labelunsafe10k.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LABELUNSAFE10K STOP $*"; exit 3; }
case "$STAGE" in tests|controls|names) ;; *) stop "usage: tests|controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
[ -f "$TOOL" ] || stop "MISSING-KIT-FILE $TOOL"
if [ "$STAGE" = tests ]; then
  [ -f "$K/test_read_labelunsafe10k.py" ] || stop "MISSING-KIT-FILE test_read_labelunsafe10k.py"
  [ -e "$LT/pod_pytest.out" ] && stop "ALREADY-RUN tests ($LT/pod_pytest.out exists)"
  mkdir -p "$LT" "$TT"
  BT="$TT/pod_$(date -u +%Y%m%dT%H%M%SZ)"
  ( cd "$GE" && LABELUNSAFE_TEST_CWD="$GE" "$PYT" -m pytest "$K/test_read_labelunsafe10k.py" -q -p no:cacheprovider --basetemp "$BT" > "$LT/pod_pytest.out" 2>&1 ); rc=$?
  tail -n 1 "$LT/pod_pytest.out" | cut -c1-160
  echo "ZZ PODREAD-LABELUNSAFE10K tests DONE rc $rc $(ts)"
elif [ "$STAGE" = controls ]; then
  for f in "$LT/dev_pytest.out" "$LT/pod_pytest.out"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  [ -e "$OUT/LABELUNSAFE10K_CONTROLS.json" ] && stop "ALREADY-READ controls"
  mkdir -p "$OUT"
  ( cd "$GE" && "$PY" "$TOOL" controls --t1-root "$T1" --out-dir "$OUT" --tests-out "$LT/dev_pytest.out" --tests-out "$LT/pod_pytest.out" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^ctl u[0-4] (PASS|FAIL)|^ZZ READ-LABELUNSAFE10K|^REFUSED' "$OUT/read_controls.log" | cut -c1-200 | head -n 12
  echo "ZZ PODREAD-LABELUNSAFE10K controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/LABELUNSAFE10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/LABELUNSAFE10K_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$GE" && "$PY" "$TOOL" names --t1-root "$T1" --out-dir "$OUT" > "$OUT/read_names.out" 2>&1 ); rc=$?
  tail -n 3 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-LABELUNSAFE10K names DONE rc $rc $(ts)"
fi
