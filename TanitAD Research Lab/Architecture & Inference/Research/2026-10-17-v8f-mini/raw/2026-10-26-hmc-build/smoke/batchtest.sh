#!/bin/bash
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
FRm=C:/Users/Admin/hlat/frozen
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FRm/stack;$FRm/taniteval;$FRm/stack/scripts"
OUT=C:/Users/Admin/hmc/smoke
t0=$(date +%s)
C:/Users/Admin/venvs/tanitad/Scripts/python.exe $FRm/stack/scripts/v8f_t1_eval.py run --ckpt E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/M2b_s0_hmc_MCH.pt --tag SMOKE_MCH_s0_m10_b12 --tracks $SC/sfresh/inputs/native_tracks.pt --eval-cache $SC/sfresh/cache --eval-grid C:/Users/Admin/hlat/inputs/eval_grid_windows.json --eval-join C:/Users/Admin/hlat/inputs/join_eval141.npz --out-dir $OUT/run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit logged --device cpu --max-windows 24 --batch 12 --no-analyze > $OUT/SMOKE_MCH_s0_m10_b12.out 2>&1 < /dev/null
echo "b12 rc=$? wall=$(( $(date +%s)-t0 ))s" >> $OUT/wall_times.txt
