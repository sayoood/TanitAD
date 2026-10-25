#!/usr/bin/env bash
# PREREG_V8F_SPEEDCAL: re-run the BANK stage only. The first bank failed on Windows' MAX_PATH (the deliverable tree is deep); the runner's bank now uses extended-length paths (runner sha256 in the line below).
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 SPEEDCAL_OUT=C:/Users/Admin/sc_run SPEEDCAL_LEAD_BLOCK=C:/Users/Admin/sc_run/inputs/b1_eval_lead_block.npz SPEEDCAL_BANK="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/speedcal/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-speedcal/run"
R="/c/Users/Admin/sc_work/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/devbox_ops/run_speedcal.sh"
echo "$(date '+%m-%d %H:%M:%S') == BANK again (runner sha256 $(sha256sum "$R" | cut -c1-16))"
bash "$R" bank || exit 3
echo "$(date '+%m-%d %H:%M:%S') ZZSPEEDCAL-DONE $(cat C:/Users/Admin/sc_run/OUTCOME.txt)"
