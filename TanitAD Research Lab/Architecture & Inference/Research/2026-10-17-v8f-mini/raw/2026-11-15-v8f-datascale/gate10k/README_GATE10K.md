# DSCALE10K block-1 launch gate (2026-11-16 / 17): what failed, what was repaired, what launched

## Outcome
* **All four stage-A tokens are minted and verified:** B_s0, U_s0, B_rep and R_s0 (`pod_out/gate/tokens/`; `pod_out/gate/verify_*.json`, each with 4 tampered variants RED).
* **Block 1 LAUNCHED 2026-11-16 23:41:37Z** on the LoRA pod, under the pod GPU queue lock: `flock /root/gpu_queue.lock bash /root/dscale10k/block1.sh`.
* It runs B_s0, U_s0 and B_rep (stage A, bank, stage B), then R_s0 (stage A).
* About 8.1 h ESTIMATED (`PILOT_REPORT.json` with the 4-shard bank factor 111/358, MEASURED).

## The first gate run FAILED (2026-11-16 22:59Z, `ZZ FINISH-GATE FAIL ZZ assemble_evidence`), on two causes
1. **Both stage-B selfchecks were REFUSED at K-B3.** The canonical eval-141 grid (`.../2026-10-17-v8f-mini/raw/eval_grid_windows.json`) was not in the pod study tree, because only `stack/` and `taniteval/` were exported.
   * Repair: copied from the STRAT1 tree and checked against sha256 659c3b65 (the a4357e79 blob; the file last changed in 4658d5c3 on 2026-10-17).
   * The file lies outside the checker's `TREE_DIRS`, so the tree manifest the tokens bind did not move.
   * After the repair: B_s0.B and U_s0.B selfchecks PASS (declared-vs-built PASS, dagger probe PASS). The mutated selfcheck is REFUSED for the swap: "the swapped kin3 of a selected row is not the bank record's own kin3".
2. **The resume smoke compared bitwise, and its fourth regression arm could never read RED.**
   * MEASURED (`mm/resume_floor.py`, `pod_out/gate/resume_floor.json`): two UNINTERRUPTED runs of the same argv differ in 238 of 276 tensors, max abs 1.95e-5. The trainer is not bitwise reproducible on this GPU.
   * Resumed vs straight differed by max abs 9.4e-7 in that measurement.
   * Repair (`mm/resume_smoke.py`; the original is `.pre_mm`):
     * The bar: resumed vs straight max abs <= 2 x the straight-vs-straight floor of the same run.
     * The 2 x was set after the floor read. It is an instrument repair, and its teeth are the defect arm.
     * New regression arm: a resume from the step-20 checkpoint with the Adam moments zeroed.
     * The compare function's sensitivity is checked on identical states (0) and a one-element change (1).
   * Re-run: floor 1.95e-5, resumed 2.14e-5 (1.10 x the floor), defect 1.54e-3 (79 x the floor). All 5 regression arms RED; PASS.

## Found while minting, and repaired
* **`verify_token.py` imported a stale copy of the checker.**
  * It imported `gate/dscale10k_check.py`, the script's own folder, which wins over PYTHONPATH. That copy is not the canonical `study/dscale10k_check.py`.
  * So every verify read FAIL on a checker sha256 mismatch, AND it exited 0, so `finish_gate.sh` read the FAIL as a pass.
  * Repair (`mm/verify_token.py`): it forces the study checker first, refuses any other, and exits 1 on FAIL. All three tokens then verify PASS.
  * The runner (`chk_arm.sh run`) always used the canonical checker, so the tokens themselves were sound.
* **R_s0 had no reach probe, no init record, and no union-set evidence.**
  * Reach probe on R's own release: every term finite (`pod_out/gate/probe2/R_s0.A.json`, problems []). This answers the G-LIVE condition of the R ruling: the own-`lead_gap` / donor-`lead_valid` mix yields no non-finite term.
  * `mm/init_r.py`: R_s0's G-HYG / C-INIT record (`init_r.json`, PASS, regression RED), kept separate so that `init.json`, which the other tokens' evidence cites, is unchanged.
  * `mm/assemble_evidence.py`: R_s0 carries the union-set reach kinds and the G-CLOCK evidence, as U_s0 does.
* **R ruling** (`mm/MM_RULING_R.md`): the R release stays as built. The trainer reads only `sid` / `row0` / `n_rows` / `k0` of the 14 clip columns.

## Files
* `pod_out/`: the gate evidence as pulled from the pod. Items, logs, selfchecks, tokens, probes, argv, dry records, the patched checker (`study/`, with `.pre_gate`), `block1.sh`, `chainlib.py`, `chk_arm.sh`, `bank4.sh`.
* `agent_local/`: the gate agent's local copies, its report and handoff, `R_COLUMNS.md`, and the two patches: `new/PATCH_dscale10k_check.diff` and `pod/PATCH_reach_probe_gen_device.diff` (the pilot's CPU-generator fix).
* `mm/`: the MM's repairs, each with the original kept as `.pre_mm`, plus the gate re-run, the R mint script and the waiters.
