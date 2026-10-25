#!/usr/bin/env bash
# PREREG_V8F_SPEEDAVG, the real arm (MM, 2026-10-25 21:42): KINDX's seed-1 read is CARRIER (speed), so the condition is met.
# The runner is byte-identical to tip ea67b7a0's code/devbox_ops/run_speedavg.sh (sha256 754accfc59ff...); the driver is tip 3950bdbb's (the pair-check fix). Dev-box CPU only.
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 SPEEDAVG_OUT=C:/Users/Admin/sa_run
RN="/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/speedavg/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/devbox_ops/run_speedavg.sh"
bash "$RN" all
