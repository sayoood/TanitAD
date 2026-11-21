#!/usr/bin/env bash
# PREREG_V8F_SPEEDCAL, RESUME of the real arm at the READ stage (build agent, 2026-10-25 23:1x).
# Why: run.out ended `ZZSPEEDCAL-STOP reader --check-inputs refused`: the reader found no lead block (the export of the repo has no raw/ banks).
# The fits, the 6+6 evals, the 2+2 baseline evals and the paired-families runs were COMPLETE and are not repeated (a re-fit would rewrite the fit json whose sha256 the cal evals record).
# The lead block is the repo's banked 2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz, copied unchanged (sha256 c0525943605cfd4a...; identical to E:/Projects/TanitAD's and to refav1_sweep's copy).
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 SPEEDCAL_OUT=C:/Users/Admin/sc_run SPEEDCAL_LEAD_BLOCK=C:/Users/Admin/sc_run/inputs/b1_eval_lead_block.npz SPEEDCAL_BANK="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/speedcal/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-speedcal/run"
R="/c/Users/Admin/sc_work/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/devbox_ops/run_speedcal.sh"
echo "$(date '+%m-%d %H:%M:%S') == RESUME at read (runner sha256 $(sha256sum "$R" | cut -c1-16))"
bash "$R" read || exit 3
bash "$R" bank || exit 3
echo "$(date '+%m-%d %H:%M:%S') ZZSPEEDCAL-DONE $(cat C:/Users/Admin/sc_run/OUTCOME.txt)"
