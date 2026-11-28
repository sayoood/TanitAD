#!/bin/bash
# card ADD155: A1 LIN + A2-A4 per seed in parallel, then A1 MLP, then the card. Waits for the extraction done marker.
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=C:/Users/Admin/wua_work
cd /f/TanitAD-scratch/v8proof/wua
while [ ! -f $W/extract.done ]; do sleep 30; done
for s in 0 1 2; do grep -q npz_sha256 $W/ext/WUA_RH_s$s.json || { echo "EXTRACT_S${s}_MISSING" > $W/reads.done; exit 1; }; done
for s in 0 1 2; do
  $PY -I wua_a1.py --seed $s > $W/a1_s$s.log 2>&1 &
  $PY -I wua_imag.py --seed $s > $W/imag_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do $PY -I wua_a1.py --seed $s --mlp > $W/a1mlp_s$s.log 2>&1 & done
wait
$PY -I wua_card.py > $W/card.log 2>&1
echo "rc_card=$? $(ls $W/a1 $W/imag | tr '\n' ' ')" > $W/reads.done
