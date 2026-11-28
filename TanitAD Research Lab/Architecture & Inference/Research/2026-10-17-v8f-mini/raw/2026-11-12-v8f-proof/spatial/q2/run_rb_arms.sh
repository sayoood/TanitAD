#!/usr/bin/env bash
# run_rb_arms.sh ARM...  -- the r-b read on the trained model (rb_trained.py) for each arm, serially, when free commit is at least 4 GB.
# Latents from the pixel job's run folder on C:, states from spatial/s1states. Deletes nothing. Last line ZZ RB DONE or ZZ RB FAIL <arms>.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; LAT=C:/Users/Admin/v8proof_c/latvis_gw; OUT=$W/spatial/q2/out
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
cd /f/TanitAD-scratch/v8proof/spatial/q2 || exit 1
mkdir -p "$OUT"
bad=""
for A in "$@"; do
  [ -f "$OUT/RESULT_RB_$A.json" ] && { echo "$A: already read"; continue; }
  [ -f "$LAT/LATENTS_GW_$A.npz" ] && [ -f "$W/spatial/s1states/${A}_eval.npz" ] || { echo "$A: latents or states absent"; bad="$bad $A"; continue; }
  k=0
  while true; do
    FREE=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -cd "0-9")
    [ -n "$FREE" ] && [ "$FREE" -ge 4 ] && break
    k=$((k+1)); [ $k -gt 180 ] && break
    ping -n 61 127.0.0.1 > /dev/null
  done
  "$PY" rb_trained.py run --arm "$A" --states "$W/spatial/s1states/${A}_eval.npz" --latents "$LAT/LATENTS_GW_$A.npz" --out-dir "$OUT" > "$OUT/stdout_rb_$A.log" 2>&1
  rc=$?
  echo "$A: rc $rc $(tail -n 1 "$OUT/stdout_rb_$A.log" | cut -c1-160)"
  [ $rc = 0 ] || bad="$bad $A(rc$rc)"
done
if [ -n "$bad" ]; then echo "ZZ RB FAIL$bad"; exit 1; fi
echo "ZZ RB DONE"
