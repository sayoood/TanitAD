# HANDOFF: SH5 C-REACH diagnosis (Research Lab, Opus), written 2026-10-19 ~23:20

## State
- Grids 1-2 DONE, verified from the summaries: **no recipe passes** (best final F 0.6008, ALLQ 0.6732). Table and reading are in the draft `PREREG_SH5_DAC_PROX_AMENDMENT4.draft.md` s6. Key point: zero_clear moves ALLQ by <= 0.0043 everywhere. Where R^2 reads the plant (K6/o8 lr 1e-3 s1: 0.644 -> 0.024 under zero_clear), **shuffle_clear keeps it at 0.630**. So the learned read is a TOKEN-LEVEL average, never position-matched. On s0 R^2 does not read the plant at all.
- Pre-launch CPU checks are MEASURED (`raw/levers_cpu_check.json`).
  - At init the plant logit range is ~1 (beta_eff 0.1-0.2). The ideal reader needs ~30 (beta >= 16).
  - temp at tau 1 is bit-identical to the sealed layer.
  - Logit reconstruction error is <= 4e-6.
  - beta estimator controls read 16.0001 / -0.07 / 0.0.
  - The tau gradient is exactly 0 at step 0 and > 0 after 3 steps.
  - The q/k scale was chosen by rule: c = 6.
  - The smoke test passes (`code/smoke_diag3_cpu.py`: SMOKE3_OK).
- **Pre-registered**: `PREREG_GRID3.md` (sha256 pins in `PREREG_GRID3.md.sha256`; re-hashed once for a typo fix BEFORE any grid-3 number). It defines PASS, the replicate controls, outcomes O1/O2/O3 and their committed next arms.
- **A RUNNING detached job** (WMI, cmd PID 11872, started ~23:05): `code/diag_pre3.py` grid3, then `&&` grid4. It uses the sealed `sh5_gpulock`, owner `sh5_diag`, job `sh5-creach-diag3`.
  - At launch the lock was held by `refcv8-g0-step40000` (chain_active), so the job waits. Max wait is 6 h per grid.
  - Grid 3 (`work/grid3.json` -> `work/diag3/`, log `work/diag3.log`) runs S12_s0 and S12_s1 at 12 ep. Each run has 8 cells: base / temp / cos16 / qk6 at K6/o8 and K8/o8, with ablations. Expect ~25 min once locked.
  - Grid 4 (`work/grid4.json` -> `work/diag4/`, log `work/diag4.log`) skips itself if a grid-3 cell passes. Otherwise it resolves BEST, the lever cell with the highest min-over-seeds F, and runs W12_s0/s1 {base_k8o8, BEST} x {lr 3e-3, p_sub 64} and W60_s0/s1 {base_k8o8, BEST} at 60 ep. Expect ~40 min.
  - Checkpoints of every learner are in `work/diag3/ckpt/`, `work/diag4/ckpt/`.
- Markers:
  - `ZZDIAGEVAL <run> <tag> cycle c: auc_all .. F .. r2 ..`
  - `ZZATTN <run> <tag> cycle c L<l>: beta_eff_max .. beta [4 heads] range_plant .. plant_mass .. self_mass .. top1 .. tau ..`
  - `ZZDIAGABL`, `ZZDIAGRUN_DONE`, `ZZDIAG3_PASSING [...]`, `ZZDIAG3_BEST {...}`, `ZZDIAG3_SKIPPED`
  - `ZZDIAG_EXIT <rc>` (75 = the GPU never freed in 6 h)

## RELAUNCH CONDITION (for the watcher)
Done when:
- `work/diag3.log` has `ZZDIAG_EXIT` with rc != 0, OR
- `work/diag4.log` has `ZZDIAG_EXIT`, OR
- either log has `Traceback`.

If rc is 75:
- From grid 3: re-run `powershell -NoProfile -ExecutionPolicy Bypass -File work/launch_diag3.ps1`.
- From grid 4 only: re-run the `$g4` line of that script alone, via the same WMI pattern.

## Exact next steps
1. `sha256sum -c PREREG_GRID3.md.sha256` (the comment line is expected to warn). Then read `work/diag3/diag_summary.json`, `work/diag4/diag_summary.json`, and the per-eval `attn` blocks in `work/diag3/diag_S12_s*.json`.
   - `sys.path.insert(0, 'code'); import diag_pre3 as D; D.passing_cells(json.load(...))` applies the PASS rule. The rule needs both seeds and requires zero_clear and shuffle_clear to collapse to <= 0.80.
2. Replicate controls first. The base_k6o8 cells must reproduce s0 0.6489 / s1 0.6732, and base_k8o8 s0 must reproduce 0.6467. Report each |Δ| as the GPU non-determinism floor.
   - Report the base cells' TRAINED beta_eff and plant logit range. That number answers "is sharpness the binding term": if base already has beta_eff >= 16, it is NOT.
3. Name the outcome per the prereg.
   - **O1: a cell passes.** Do old steps 3-5 with the cheapest passing cell. Lever order is qkinit < temp < cos, and K6 before K8.
     - Copy `par18/sh5/code/*` to `code/amend4/` (already an exact copy). Implement the lever in the COPY: `sh5_branch.CrossLayer` as in `code/diag_attn.LeverLayer`, or the q/k init scale. If K = 8, change `sh5_lib.FOURIER_K`.
     - Add tests: the lever is live with a non-zero tau gradient and an exact-0 control; a RED test that refuses the old recipe; a test that query octaves == plant octaves if changed.
     - Run the suite: `cd code/amend4; CUDA_VISIBLE_DEVICES=-1 SH5_TMP=C:/Users/Admin/par18/sh5diag/tmp PY run_tests.py --out ../../work/tests_summary_amend4.json` (~10 min, so detach it and hand off).
     - Then get C-REACH evidence on both seeds via the amended `_fit_locked` PRE, fold 0, under the lock: `code/confirm_pre.py --code code/amend4 --seeds 0,1 --out work/confirm`. Launch it like `launch_diag.ps1`.
   - **O2 or O3, and grid 4 also fails.**
     - O2 (sharpness learned: beta_eff >= 16 on both seeds, top-1 >= 0.5): the next lever is the readout, a softmin key-time pool on BEST.
     - O3: the next lever is fp32 attention (autocast off) on BEST at lr 3e-3. Then run a supervised attention target on the plant (a diagnostic only).
     - Run these as a grid5 the same way: pre-register first, then detach. If they are blocked or fail, report **INSTRUMENT DEFECT** with each lever's best F / ALLQ / R^2 / beta_eff, and name the next lever.
4. Write up.
   - Fill draft s4 (root cause, with its evidence class), s6 (grids 3-4 table plus the attention numbers), s7 and s8 (sha256 of every changed file).
   - Run `PY code/stage_out.py`. It now also stages `PREREG_GRID3.md` and its .sha256, plus `work/diag*/*.json` and logs. The .pt checkpoints are NOT staged: list them in the manifest as dev-box only.
   - Write `out/RESULT.md` with proposed register rows. Hand back in <= 300 words.

## Corrections to the MM's first glance (verified)
36 ep lr 1e-3 F reads 0.485-0.548 (not 0.50-0.56). R^2 0.60-0.68 (K6/o8) and 0.13-0.17 (K10/o10) is **seed 1 only**: seed 0 reads 0.437 and 0.412. The zero_clear R^2 collapse is also **seed 1 only**: on s0, 0.437 -> 0.433.

## Constraints (unchanged)
- GPU only through `sh5_gpulock` with owner `sh5_diag`. Never touch another owner's lock, `devbox_gpu.chain_active`, or a pod.
- No `pkill -f`. Kill your own processes by explicit PID only.
- Never run S, Snoprox or Sshuf, and never run `analyze`. Do not touch l2fresh, l2stepb, podstage, unseen_s1b or D:/b9. New D: files only in `DD/fits_local/`, and only small ones.
- Nothing sealed: the MM seals. Stage in `out/`, never commit.
- `PY` = `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore`. CPU jobs use `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8`, with at most 2 concurrent.
