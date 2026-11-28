#!/bin/bash
# run_preserr_pod.sh -- pod-side read of card PREREG_V8F_PROOF_ADD116 (PRESERR-T0). CPU only, 2 threads; banked rule logs and the evaluation join. No rollout.
# Usage: bash run_preserr_pod.sh controls|names
# Stage controls needs the lead's test output $GE/lead_tests/lead_pytest_preserr.out; it prints only h<i> PASS|FAIL and one summary line.
# Stage names refuses unless h0 to h5 all pass. Never reads twice into one file. Deletes, moves and stops nothing.
set -u
STAGE=${1:-}
V=/workspace/v8proof; GE=$V/preserr; K=$GE/kit; OUT=$GE/read
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/venv/bin/python
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || { echo "ZZ RUN-PRESERR STOP memory free ${free_gb} GB"; exit 3; }; fi
mkdir -p "$OUT"
case "$STAGE" in
  controls)
    [ -f "$GE/lead_tests/lead_pytest_preserr.out" ] || { echo "ZZ RUN-PRESERR STOP no lead test output"; exit 3; }
    "$PY" "$K/preserr_t0.py" --stage controls --t1 "$V/integ10k/t1" --join "$V/hierrep/pins/join_eval141.npz" --tests "$GE/lead_tests/lead_pytest_preserr.out" \
        --out "$OUT/PRESERR_T0_CONTROLS.json" > "$OUT/controls.out" 2>&1; rc=$?
    grep -E '^h[0-5] (PASS|FAIL)|^ZZ PRESERR-T0|^REFUSED' "$OUT/controls.out" | cut -c1-160
    echo "ZZ RUN-PRESERR controls DONE rc $rc $(date -u +%H:%M:%SZ)";;
  names)
    "$PY" "$K/preserr_t0.py" --stage names --controls "$OUT/PRESERR_T0_CONTROLS.json" --out "$OUT/PRESERR_T0_RESULT.json" > "$OUT/names.out" 2>&1; rc=$?
    tail -n 1 "$OUT/names.out" | cut -c1-160
    echo "ZZ RUN-PRESERR names DONE rc $rc $(date -u +%H:%M:%SZ)";;
  *) echo "ZZ RUN-PRESERR STOP usage: controls|names"; exit 3;;
esac
