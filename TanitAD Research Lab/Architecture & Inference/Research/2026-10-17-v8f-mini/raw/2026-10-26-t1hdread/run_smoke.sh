#!/usr/bin/env bash
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh
W=E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I C:/Users/Admin/t1hdread/t1hd_read.py --mode smoke --tool-root "$W" \
  --eval-cache $SC/cache --tracks $SC/inputs/native_tracks.pt --eval-grid "$W/$PK/eval_grid_windows.json" --eval-join "$W/$PK/join_eval141.npz" \
  --banked-dir C:/Users/Admin/t1hdread/banked --out C:/Users/Admin/t1hdread/T1HD_SMOKE.json \
  --smoke-ticks C:/Users/Admin/t1heads/smoke/ticks/ticks_SM.npz --smoke-dump-dir C:/Users/Admin/t1heads/smoke/heads/dump_SM \
  --smoke-ref-dump-dir C:/Users/Admin/t1heads/smoke/off/dump_SM --smoke-seed s0 "$@"
