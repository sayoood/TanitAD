#!/bin/bash
# run_preshead_pod_r2.sh -- pod-side read of card PREREG_V8F_PROOF_ADD118 (PRESHEAD-T0). CPU only, 2 threads; the dumps and records of the fit queue, banked rule logs and the evaluation join.
# Root of the relaunched fit queue (preshead2; the first root stopped on the file-handle limit before any fit); the lead test outputs are read from the first root. No fit, no rollout. Usage: bash run_preshead_pod_r2.sh controls|names
# Stage controls needs the done marker of the fit queue and the lead's two test outputs; it prints only p<i> PASS|FAIL and one summary line.
# Stage names refuses unless p0 to p6 all pass. Never reads twice into one file. Deletes, moves and stops nothing.
set -u
STAGE=${1:-}
V=/workspace/v8proof; GP=$V/preshead2; K=$V/preshead/kit_read; OUT=$GP/read
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/venv/bin/python
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 20 ] || { echo "ZZ RUN-PRESHEAD STOP memory free ${free_gb} GB"; exit 3; }; fi
mkdir -p "$OUT"
case "$STAGE" in
  controls)
    [ -f "$GP/PRESHEAD_DONE.txt" ] || { echo "ZZ RUN-PRESHEAD STOP no done marker of the fit queue"; exit 3; }
    [ -f "$V/preshead/lead_tests/lead_pytest_read.out" ] || { echo "ZZ RUN-PRESHEAD STOP no lead test output of the reader"; exit 3; }
    "$PY" "$K/preshead_t0.py" --stage controls --preserr "$V/preserr/kit/preserr_t0.py" --t1 "$V/integ10k/t1" --join "$V/podread/data/join_eval141.npz" --root "$GP" \
        --ref-fit-record "$V/cropscale/fitroot/fits/D_H5_s2/FIT_RECORD.json" --tests-fit "$V/preshead/lead_tests/lead_pytest_fit.out" --tests-read "$V/preshead/lead_tests/lead_pytest_read.out" \
        --out "$OUT/PRESHEAD_T0_CONTROLS.json" > "$OUT/controls.out" 2>&1; rc=$?
    grep -E '^p[0-6] (PASS|FAIL)|^ZZ PRESHEAD-T0|^REFUSED' "$OUT/controls.out" | cut -c1-160
    echo "ZZ RUN-PRESHEAD controls DONE rc $rc $(date -u +%H:%M:%SZ)";;
  names)
    "$PY" "$K/preshead_t0.py" --stage names --controls "$OUT/PRESHEAD_T0_CONTROLS.json" --out "$OUT/PRESHEAD_T0_RESULT.json" > "$OUT/names.out" 2>&1; rc=$?
    tail -n 1 "$OUT/names.out" | cut -c1-160
    echo "ZZ RUN-PRESHEAD names DONE rc $rc $(date -u +%H:%M:%SZ)";;
  *) echo "ZZ RUN-PRESHEAD STOP usage: controls|names"; exit 3;;
esac
