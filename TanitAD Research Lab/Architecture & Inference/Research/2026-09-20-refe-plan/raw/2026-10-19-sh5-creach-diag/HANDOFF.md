# HANDOFF: SH5 C-REACH diagnosis, grid 5 (Research Lab), written 2026-10-20

## State
- **Grids 3-4 are DONE with no pass.** Grid 3 is O3. They are written up in draft s4 and s6 and in `out/RESULT.md`. The per-cell table is `raw/grid34_cells.json`.
- **The first position-matched read is on seed 0 only:** W60 cos16_k6o8, 60 ep, F 0.846.
- Probe 5 (`raw/probe5_ckpt_seedsplit.json`) shows that every learner memorises TRAIN through the visual ctx. That is hypothesis H5.
- **Grid 5 is RUNNING**, detached: WMI cmd PID 39992, python PID 65288, `GPULOCKED` under owner `sh5_diag`.
  - It is pre-registered in `PREREG_GRID5.md`. Pins are in `PREREG_GRID5.md.sha256`, written before any grid-5 number.
  - Driver: `code/diag_pre5.py` (diag_pre3 plus the fp32 / AdamW options). The CPU smoke passed: `code/smoke_diag5_cpu.py` gives SMOKE5_OK.
  - Long runs: `work/grid5L.json` -> `work/diag5/`, log `work/diag5.log`. Runs G5_s1, then G5_s0, then G5_s2. Each is 120 ep (30 cycles), evaluated every 3 cycles. Cells: cos16_k8o8_120ep, cos16_k8o8_wd1_120ep, BEST_120ep, base_k8o8_120ep. G5_s2 runs only the first two.
  - Then, chained with `&`: `work/grid5F.json` -> `work/diag5f/`, log `work/diag5f.log`. Runs F12_s0 and F12_s1 with BEST_fp32_lr3e-3 and BEST_lr3e-3_rep.
  - ESTIMATED ~4 h after the lock: the log projects 1.48 h for G5_s1 alone.
- Checkpoints: `work/diag5/ckpt/`, `work/diag5f/ckpt/`. They live on the dev box only and are not staged.

## Done-marker / relaunch condition
- Done when `work/diag5f.log` has `ZZDIAG_EXIT`.
- Relaunch case 1: `work/diag5.log` has `ZZDIAG_EXIT` != 0 or a `Traceback`, and diag5f has not started. The `&` chain still runs F12 in this case.
- Relaunch case 2: `ZZDIAG_EXIT 75`, which means the lock never freed in 6 h.
- To relaunch, re-run `powershell -NoProfile -ExecutionPolicy Bypass -File work/launch_diag5.ps1`. That restarts BOTH grids, so first move the partial `work/diag5*` to `work/diag5_attemptN/`.
- ⚠ **C: had ~0.9 GB free at launch.** Grid-3 attempt 1 died of a host MemoryError with C: full. Check `Get-PSDrive C` if the job dies.

## Exact next steps (the fresh agent)
1. `cd par18/sh5diag; sha256sum -c PREREG_GRID5.md.sha256`. The comment line is expected to warn.
2. Read `work/diag5/diag_summary.json`, `work/diag5f/diag_summary.json`, and the curves in `work/diag5/diag_G5_s*.json`.
   - PASS_REG: `sys.path.insert(0,'code'); import diag_pre3 as D; D.passing_cells({k:v for k,v in S.items() if k in ('G5_s0','G5_s1')})`.
   - ROBUST also needs G5_s2.
   - The in-log `ZZDIAG3_PASSING` already includes s2 for the two candidate cells.
3. Replicate controls (PREREG_GRID5 s4). Report each |Δ| against the 0.030 floor.
   - BEST_120ep at c15 vs W60 BEST: s0 F 0.846. If it reads < 0.75, the transition is NOT reproducible.
   - base at c15 vs W60 base.
   - cos16_k8o8 at c3 vs grid-3.
   - F12 rep vs W12 BEST_lr3e-3.
4. Re-run probe 5 on the grid-5 checkpoints. Add the paths to `CKPTS` in a COPY such as `probe5b`; do not edit the pinned probe5. Use K and lever per cell. This gives the train F under no_plant, the H5 test for the wd cell.
5. Name the outcome (PREREG_GRID5 s6, in order: G1, G3, G2, G4, otherwise) and execute its committed next arm. RULE ZERO applies.
   - **G1:** copy `par18/sh5/code/*` -> `code/amend4/` (already an exact copy). Implement the picked recipe in the COPY:
     - the cos lever in `sh5_branch.CrossLayer`, as in `code/diag_attn.LeverLayer`;
     - FOURIER_K 8 if it is a K8 cell;
     - 120 epochs;
     - AdamW groups if it is the wd cell.
     - Tests: the lever is live with a non-zero tau gradient and an exact-0 control; a RED test that refuses the old recipe; query octaves == plant octaves.
     - Suite: `cd code/amend4; CUDA_VISIBLE_DEVICES=-1 SH5_TMP=C:/Users/Admin/par18/sh5diag/tmp PY run_tests.py --out ../../work/tests_summary_amend4.json` (~10 min, so detach it).
     - Then `code/confirm_pre.py --code code/amend4 --seeds 0,1 --out work/confirm` under the lock. Detach it, hand off, and END.
   - **G2:** the same cell(s) at 240 ep. Pre-register first (`PREREG_GRID6.md` + sha256), then detach.
   - **G3:** a softmin key-time pool on that cell. Pre-register, implement it in a new module, smoke it, then detach.
   - **G4 / otherwise:** INSTRUMENT DEFECT, with every lever's best number. The next lever is visual-ctx dropout for ALL arms. Implement it and pre-register it.
6. Fill draft s6/s7/s8 (s8 = the sha256 of every changed file). Run `PY code/stage_out.py`; it now stages `PREREG_GRID*` too. Update `out/RESULT.md`. Hand back in <= 300 words.

## Constraints (unchanged)
- GPU only through `sh5_gpulock`, owner `sh5_diag`. Never touch another owner's lock, `devbox_gpu.chain_active`, or a pod.
- No `pkill -f`. Kill by explicit PID only.
- Never run S, Snoprox or Sshuf, and never run `analyze` or open navtest.
- Do not touch l2fresh, l2stepb, podstage, unseen_s1b or D:/b9. D: was FULL (0 B free) at launch, so write nothing there.
- Do not edit pinned files (`PREREG_GRID3.md.sha256`, `PREREG_GRID5.md.sha256`). Make copies instead.
- Nothing is sealed or committed: stage in `out/`.
- `PY` = `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore`.
- CPU jobs run with `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8`, at most 2 at a time.
