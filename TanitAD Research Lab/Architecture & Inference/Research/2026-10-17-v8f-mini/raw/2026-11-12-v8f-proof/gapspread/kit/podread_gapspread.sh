#!/usr/bin/env bash
# podread_gapspread.sh -- cards PREREG_V8F_PROOF_ADD132 and ADD133 on the fork pod, CPU only. It deletes, moves and stops nothing.
#   tests      both test files, outputs into lead_tests/ (outside the kit folder)
#   controls   the controls of card ADD132; when its control s3 alone decides against it, the controls of card ADD133
#   name-a     the name of card ADD132 (refused by the tool unless its controls file is all-pass)
#   name-b     the names of card ADD133 (refused by the tool unless its controls file is all-pass)
set -u
ROOT=/workspace/v8proof/gapspread
KIT=$ROOT/kit
LT=$ROOT/lead_tests
OUTA=$ROOT/read_out_a
OUTB=$ROOT/read_out_b
GF=/workspace/v8proof/cropscale/gapfiles
PY=/workspace/venv/bin/python
PYT=/workspace/v8proof/pytest_env/bin/python
ARGS=/workspace/v8proof/boxread/kit_f/read_args.txt
export BOXSIZE_READER=/workspace/v8proof/boxsize/kit/read/read_boxsize.py BOXREAD_READER=/workspace/v8proof/boxread/kit/read/read_boxread.py
export BOXFUSE_READER=/workspace/v8proof/boxread/kit_f/read_boxfuse.py JOINED_GAPFILES_TOOL=/workspace/v8proof/boxfuse10k/kit_gap/make_joined_gapfiles.py
export GAPKIT=/workspace/v8proof/boxfuse10k/kit_gap CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export TANITAD_STACK=/workspace/v8proof/tree/stack PYTHONPATH=/workspace/v8proof/tree/stack
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mm=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 ))
[ "$free_gb" -ge 20 ] || { echo "REFUSED: memory free ${free_gb} GB"; exit 2; }
COMMON=(--base "$GF/VGAPLE.npz" --fit "$GF/VGAPL_s2.npz" --fit "$GF/VGAPL_s3.npz" --fit "$GF/VGAPL_s4.npz" --rec106 "$GF/CROPSCALE_GAPFILES_RECORD.json" --rb-args "$ARGS")
TA=(--tests-out "$LT/dev_pytest_a.out" --tests-out "$LT/pod_pytest_a.out")
TB=("${TA[@]}" --tests-out "$LT/dev_pytest_b.out" --tests-out "$LT/pod_pytest_b.out")
case "${1:-}" in
  tests)
    mkdir -p "$LT" "$ROOT/tests_tmp" || exit 2
    cd "$KIT" || exit 2
    for t in a b; do
      f=test_read_gapspread_t0.py; [ "$t" = b ] && f=test_read_gapspreadb_t0.py
      [ -e "$LT/pod_pytest_$t.out" ] && { echo "REFUSED: $LT/pod_pytest_$t.out exists"; exit 2; }
      "$PYT" -m pytest "$f" -q -p no:cacheprovider --basetemp "$ROOT/tests_tmp/$t" > "$LT/pod_pytest_$t.out" 2>&1
      echo "tests $t rc $? : $(tail -n 1 "$LT/pod_pytest_$t.out")"
    done
    ;;
  controls)
    [ -e "$OUTA" ] && { echo "REFUSED: $OUTA exists"; exit 2; }
    "$PY" "$KIT/read_gapspread_t0.py" controls "${COMMON[@]}" --out-dir "$OUTA" "${TA[@]}" > "$ROOT/controls_a.out" 2>&1; rca=$?
    cat "$ROOT/controls_a.out"; echo "ZZ controls-a rc $rca $(ts)"
    if [ "$rca" -ne 0 ]; then
      "$PY" "$KIT/read_gapspreadb_t0.py" controls "${COMMON[@]}" --out-dir "$OUTB" "${TB[@]}" --add132-controls "$OUTA/GAPSPREAD_CONTROLS.json" > "$ROOT/controls_b.out" 2>&1; rcb=$?
      cat "$ROOT/controls_b.out"; echo "ZZ controls-b rc $rcb $(ts)"
    fi
    ;;
  name-a)
    "$PY" "$KIT/read_gapspread_t0.py" name "${COMMON[@]}" --out-dir "$OUTA" "${TA[@]}" > "$ROOT/name_a.out" 2>&1; rc=$?
    cat "$ROOT/name_a.out"; echo "ZZ name-a rc $rc $(ts)"
    ;;
  name-b)
    "$PY" "$KIT/read_gapspreadb_t0.py" name "${COMMON[@]}" --out-dir "$OUTB" "${TB[@]}" --add132-controls "$OUTA/GAPSPREAD_CONTROLS.json" > "$ROOT/name_b.out" 2>&1; rc=$?
    cat "$ROOT/name_b.out"; echo "ZZ name-b rc $rc $(ts)"
    ;;
  *) echo "usage: podread_gapspread.sh tests|controls|name-a|name-b"; exit 2 ;;
esac
