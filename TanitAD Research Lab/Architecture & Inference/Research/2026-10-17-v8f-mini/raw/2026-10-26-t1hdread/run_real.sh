#!/usr/bin/env bash
# The ONE real read of PREREG_V8F_T1_HEADS.md: run only when C:/Users/Admin/t1hd_run/logs/chain.out has "ZZT1HD-DONE 4".
# usage: ./run_real.sh > read.log 2>&1      (extra args are passed through, e.g. --dev-fake-run ...)
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh
W=E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I C:/Users/Admin/t1hdread/t1hd_read.py --mode real --tool-root "$W" \
  --eval-cache $SC/cache --tracks $SC/inputs/native_tracks.pt --eval-grid "$W/$PK/eval_grid_windows.json" --eval-join "$W/$PK/join_eval141.npz" \
  --banked-dir C:/Users/Admin/t1hdread/banked --run-dir C:/Users/Admin/t1hd_run --out C:/Users/Admin/t1hdread/T1HD_READ.json --n-boot 2000 "$@"
