# HANDOFF AFTER GRID 5 (Research Lab, 2026-10-20 ~03:55). Supersedes the "State" and "Exact next steps" of HANDOFF.md (landed at a8eeef6).

## State
- Grid 5 is FINISHED for seeds 0 and 1 and CRASHED for seed 2 (partial to c12) and for the fp32 control F12 (no eval). Cause per the MM: C: full, so the host commit limit was hit.
- Outcome (PREREG_GRID5 s6): G1 failed, G3/G2/G4 not met => "Otherwise" = INSTRUMENT DEFECT. BEST_120ep passes on seed 0 only, cos16_k8o8_wd1_120ep on seed 1 only. Full read: `RESULT_GRID5.md`.
- HANDOFF.md steps 2-4 are DONE (summary read, controls, probe 5b). Step 5 (the committed arm) is PARKED. Step 6: the draft's s6-s8 are NOT filled (G1 did not fire); a proposed s6 paragraph is in RESULT_GRID5 s8.
- Nothing is running. No lock is held by this package. No checkpoint is staged (`work/diag5/ckpt/*.pt` stay on the dev box).

## Exact next step (only after the unblock condition in RESULT_GRID5 s7)
1. Check `Get-PSDrive C` (>= 2 GB free) and take `sh5_gpulock` owner `sh5_diag`.
2. Write `PREREG_GRID6.md` + sha256 (visual-ctx dropout on ALL arms; seeds 0-2; the F12 fp32 control as its own cheap arm) BEFORE any number. Implement the dropout in a new module, smoke it on CPU (non-zero gradient plus an exact-0 control, and a RED test that refuses the no-dropout recipe), then detach and hand off.
3. Re-run F12 first if the GPU window is short: it is 12 ep x 2 seeds and settles the fp32 lever either way.

## Constraints (unchanged)
Never touch another owner's lock, `devbox_gpu.chain_active` or a pod. No `pkill -f`. Do not edit pinned files (`PREREG_GRID3/5.md.sha256`). CPU jobs: `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8`, at most 2 python processes. `PY` = `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore`. Read the repo only from the mirror tip.
