#!/usr/bin/env bash
# podread_labelunsafe10k_b.sh -- card PREREG_V8F_PROOF_ADD136 on the fork pod, CPU only. It deletes, moves and stops nothing.
#   tests     both test files with the pod pytest, file names in the output (no -q): lead_tests/pod_pytest.out and lead_tests/pod_pytest_b.out (refuses when one exists)
#   controls  read_labelunsafe10k_b.py controls -> read_out_b/LABELUNSAFE10K_CONTROLS.json (the tool refuses unless control u2 of card ADD135 failed with the other four passed)
#   names     read_labelunsafe10k_b.py names on that controls file
set -u
STAGE=${1:-}
V=/workspace/v8proof; GE=$V/labelunsafe; K=$GE/kit; KB=$GE/kit_b; OUTA=$GE/read_out; OUT=$GE/read_out_b; LT=$GE/lead_tests; TT=$GE/tests_tmp
T1=$V/integ10k/t1
PY=/workspace/venv/bin/python
PYT=$V/pytest_env/bin/python
TOOL=$KB/read_labelunsafe10k_b.py
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LABELUNSAFE10KB STOP $*"; exit 3; }
case "$STAGE" in tests|controls|names) ;; *) stop "usage: tests|controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
[ -f "$TOOL" ] || stop "MISSING-KIT-FILE $TOOL"
TESTS=(--tests-out "$LT/dev_pytest.out" --tests-out "$LT/pod_pytest.out" --tests-out "$LT/dev_pytest_b.out" --tests-out "$LT/pod_pytest_b.out")
if [ "$STAGE" = tests ]; then
  mkdir -p "$LT" "$TT" || stop "MKDIR"
  for t in a b; do
    f=$K/test_read_labelunsafe10k.py; o=$LT/pod_pytest.out
    [ "$t" = b ] && { f=$KB/test_read_labelunsafe10k_b.py; o=$LT/pod_pytest_b.out; }
    [ -e "$o" ] && stop "ALREADY-RUN tests ($o exists)"
    ( cd "$GE" && LABELUNSAFE_TEST_CWD="$GE" LABELUNSAFE_KIT="$K" "$PYT" -m pytest "$f" -p no:cacheprovider --basetemp "$TT/nq_$t" > "$o" 2>&1 )
    echo "tests $t rc $? : $(tail -n 1 "$o" | cut -c1-120)"
  done
elif [ "$STAGE" = controls ]; then
  [ -f "$OUTA/LABELUNSAFE10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE-OF-CARD-ADD135"
  [ -e "$OUT/LABELUNSAFE10K_CONTROLS.json" ] && stop "ALREADY-READ controls"
  mkdir -p "$OUT"
  ( cd "$GE" && "$PY" "$TOOL" controls --t1-root "$T1" --out-dir "$OUT" "${TESTS[@]}" --add135-controls "$OUTA/LABELUNSAFE10K_CONTROLS.json" > "$OUT/read_controls.log" 2>&1 ); rc=$?
  grep -E '^ctl u[0-4] (PASS|FAIL)|^ZZ READ-LABELUNSAFE10K|^REFUSED' "$OUT/read_controls.log" | cut -c1-200 | head -n 12
  echo "ZZ PODREAD-LABELUNSAFE10KB controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/LABELUNSAFE10K_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/LABELUNSAFE10K_RESULT.json" ] && stop "ALREADY-READ names"
  ( cd "$GE" && "$PY" "$TOOL" names --t1-root "$T1" --out-dir "$OUT" "${TESTS[@]}" --add135-controls "$OUTA/LABELUNSAFE10K_CONTROLS.json" > "$OUT/read_names.out" 2>&1 ); rc=$?
  tail -n 3 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-LABELUNSAFE10KB names DONE rc $rc $(ts)"
fi
