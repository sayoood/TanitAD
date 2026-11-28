#!/bin/bash
# card ADD155 AM3: the permutation jobs whose PERM json is missing. Per seed: the first job alone (it writes the grid-PCA cache), the rest two at a time
# (12 at once ran out of commit memory on 2026-11-28; commit free was 4 GB).
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=C:/Users/Admin/wua_work
cd /f/TanitAD-scratch/v8proof/wua
rm -f "$W/perm.done"
run1() { $PY -I wua_a1_perm.py --seed $1 --target $2 > "$W/perm_s$1_$2.log" 2>&1; }
for s in 0 1 2; do
  todo=(); for t in lead_gap ag1_pos ag2_pos lead_present; do [ -f "$W/a1/PERM_A1_s${s}_$t.json" ] || todo+=("$t"); done
  [ ${#todo[@]} -eq 0 ] && continue
  if [ ! -f "$W/a1/gridpca_s$s.npz" ]; then run1 $s ${todo[0]}; todo=("${todo[@]:1}"); fi
  i=0
  for t in "${todo[@]}"; do run1 $s $t & i=$((i+1)); [ $((i % 2)) = 0 ] && wait; done
  wait
done
n=$(ls $W/a1/PERM_A1_s*.json | wc -l)
$PY -I wua_card.py > $W/card.log 2>&1
echo "perm_json=$n rc_card=$?" > "$W/perm.done"
