# HANDOFF -- A-R8-CLAMP-2 compliant re-read (state at hand-off)

PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery`

## State
* Written and syntax-checked, **NOT YET EXECUTED** (blocked by the binding free-commit gate, see below):
  `code/compliant_reread_r8.py` (stages `read --tree T [--force]`, `strict --tree T`, `assemble`),
  `code/run_stage_compliant_r8.sh` (waits for FreeVirtualMemory >= 5 GB, runs one stage, logs to `raw/compliant_reread/logs/<stage>_<tree>.log`),
  `code/chain_compliant_r8.sh` (waits for the PRIMARY read to finish, asserts its ARTIFACT, then reads the other 8 trees, `strict`, `assemble`).
* LAUNCHED detached (nohup): `run_stage_compliant_r8.sh read step30000ema_unc` (polling the gate every 20 s) and `chain_compliant_r8.sh`.
  Gate evidence: free commit measured 2.85-5.1 GB over ~40 min (a 9 GB private python = the raw-30k GPU battery sits on the box); it has not yet been >= 5 GB at a poll.
* Design (details in the module docstring): mask = `~(gt_max_speed_6s > fed_ceiling)` from the dump extras (strict), one sha per tree/seed;
  the compliant read = `pair_read_r8.components/paired_cell` on the panel rows of the mask (components computed on the full panel then subset, so an all-True mask IS the banked read);
  capped = `clamp_posthoc_r8.apply_clamp` (vendored de4e15b) on `nav_true.traj`, substituted in memory exactly as `nr_matched_instrument.write_variant` does on disk.
* Controls gate every compliant number: C1 (all-True mask == banked PAIRED_READ cells AND == CLAMP_POSTHOC read_ON), C1v (all-window verdicts == banked), C2 (mask rolled 35), C3 (sha equal across seeds + re-derived from panel `decisions/gt_future_ext`).

## How to check / continue (do NOT re-run what exists)
1. `tail raw/compliant_reread/logs/chain.log` and `ls raw/compliant_reread/trees/` (the ARTIFACT is the evidence, not the log's exit lines).
2. If a tree json is missing, read `raw/compliant_reread/logs/read_<tree>.log` (the first run of this code has never executed: expect to fix a first-run bug, then `python compliant_reread_r8.py read --tree <t> --force` under the gate).
3. `python compliant_reread_r8.py strict --tree step30000ema_unc`, then `assemble` -> `COMPLIANT_REREAD.json/.md`. Env: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4, venv `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8`,
   PYTHONPATH `C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code`.
4. `step30000_unc`: run `read --tree step30000_unc` only after its `PAIRED_READ_step30000_unc.json` exists and is complete (the stage refuses otherwise, exit 3).
5. Hand-back needs: EMA-30k (capped, compliant set) E1/E3/NR + compliance row + controls + manifest.

## UPDATE 20:45 (supersedes "NOT YET EXECUTED" above)
The gate opened at 20:39:07 (5.50 GB) and the PRIMARY read ran first time: `trees/step30000ema_unc.json` WRITTEN, ALL_PASS=True (C1, C1v, C2, C3 incl. 4/4 independent re-derivations from panel decisions).
`chain_compliant_r8.sh` is now working through the other 8 trees (each ~3.5 min + gate wait), then `strict step30000ema_unc`, then `assemble`.
Check `logs/chain.log` and `ls trees/` -- the ARTIFACT is the evidence. If the chain died, resume the missing trees with `run_stage_compliant_r8.sh read <tree>`; `assemble` needs no GPU/torch.
Primary result: mask 4159/4754 windows (138/139 episodes) on S2, 3251/3668 on S6; capped == uncapped on the compliant set (0 binding windows are compliant).
