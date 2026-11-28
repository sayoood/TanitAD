#!/usr/bin/env bash
# PREREG_V8F_SPEEDCAL, the real arm (build agent, 2026-10-25 22:23). Dev-box CPU only, zero training. Runner/driver/reader are those of /c/Users/Admin/sc_work (tip 02e13a10 + the new SPEEDCAL files).
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 SPEEDCAL_OUT=C:/Users/Admin/sc_run SPEEDCAL_BANK="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/speedcal/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-speedcal/run"
bash "/c/Users/Admin/sc_work/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/devbox_ops/run_speedcal.sh" all
