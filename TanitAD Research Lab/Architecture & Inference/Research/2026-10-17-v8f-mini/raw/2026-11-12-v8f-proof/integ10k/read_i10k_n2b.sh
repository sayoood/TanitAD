#!/usr/bin/env bash
# read_i10k_n2b.sh : only the last step of read_i10k_n2.sh (the headway and family readers ended with rc 0), with read_i10k_n2b.py (k5 key path). Numpy only, under ops/pycap.sh.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; I=/f/TanitAD-scratch/v8proof/integ10k; RD=$I/read
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-800}
"$PY" "$I/read_i10k_n2b.py" --read-dir "$W/integ10k/read" --pull "$W/integ10k/pull" --k5 "$W/integ10k/K5_BANKED.json" --out "$W/integ10k/read/READ_I10K.json" > "$RD/read_i10k_b.log" 2>&1
echo "rc $?"
tail -n 70 "$RD/read_i10k_b.log" | cut -c1-430
echo "ZZ READ-I10K-N2B DONE $(date -u +%H:%M:%SZ)"
