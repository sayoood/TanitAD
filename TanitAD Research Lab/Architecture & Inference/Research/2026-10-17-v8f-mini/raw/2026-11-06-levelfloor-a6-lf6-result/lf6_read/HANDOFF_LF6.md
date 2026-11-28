# HANDOFF_LF6: addendum A6 (the bounded-lift floor F_B at kappa 1.0: LFB05 / LFB10 admissible, LFOB05 / LFOB10 ceiling). Built, validated, chain LAUNCHED 2026-11-06 07:03:49. Nothing committed, nothing landed.

Card `a6_prereg/PREREG_LEVELFLOOR_A6.md` sha fbf5de02 (frozen, unchanged). Packages (staged, not in git): `E:/Projects/TanitAD-artifacts/dagland/motion_c/lf6_build/` (tool, known answers, mutations) and `.../lf6_read/` (reader, precheck, chain).
Scratch `F:/TanitAD-scratch/motion_c/lf6_read/` (`tree/`, `out/` 24 rollouts, `work/`, `smoke/`, `pt/`) and `.../lf6_build/` (`smoke/`, `mut/`).

## State
* WMI `Win32_Process.Create` of `lf6_chain.cmd` (WMI pid 74620); the chain's own pid is in `raw/LF6.running` (4138712). Gates 0-2 and both pre-flights passed 07:04; first rollout RILFB10_s0 started 07:04:13.
* 24 rollouts, order: RILFB10 x4, RILFB05 x4 (heads, reindexed), RILFOB10 x4, RILFOB05 x4 (logged, CEILING), LFB10 x4, LFB05 x4 (`logged` scene). ESTIMATED ~3.75 min each on 12 threads (LF5 rate): rollouts done about 08:35, reader (ESTIMATED 15-25 min) about 09:00.
* Check: `tail -n 8 logs/chain.out` (lines `ZZLF6-*`); markers in `raw/`: `LF6_ROLLOUTS.done`, then `LF6.done` (RESULTS.md and raw/LF6_READ.json written) or `LF6.fail` (names the step). RESUME = launch `lf6_chain.cmd` again (finished rollouts are skipped). A reader failure leaves the 24 rollouts banked.
* Rerun the reader alone: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I code/lf6_read.py --mode production --out-root F:/TanitAD-scratch/motion_c/lf6_read/out --work-dir F:/TanitAD-scratch/motion_c/lf6_read/work --raw E:/Projects/TanitAD-artifacts/dagland/motion_c/lf6_read/raw --workers 4`
  (from `lf6_read/`, with `CUDA_VISIBLE_DEVICES=-1`).

## Built and validated (MEASURED; logs in the packages)
* Tool `lf6_build/tool/v8f_t1_eval_ri_lf6.py` sha 949c4b7a = LF5 tool 88b275a1 + `code/apply_lf6_patch.py` (regenerates byte for byte); `level_floor.py` unchanged (2781e23a). `--floor-lift-cap DELTA` (float >= 0, needs `--level-floor on`); absent = the LF5 tool.
  F_B lives in the tool (`bounded_progress`, `apply_floor_batch_bounded`); `lf_vf` stays the scalar kappa * v_hat (lead cap); new ticks fields `lf_dterm`, `lf_dterm_knots`, `lf_s`, `lf_sp`. The A5 defect is fixed: the ticks omit `floor_vhat_noise_seed` when it is None (strict `allow_pickle=False` load passed in the pre-flight).
* Known answers (smoke against smoke, 64 windows, bit for bit): K1 delta 0 == floor off PASS (heads bare, heads composite, logged bare; F never binds); K2 delta 1e9 == LF5 tool PASS (34-35 common fields, three smokes); K3 delta 0.5 partial PASS under the SAME-INPUT reading (see A6_NOTE_KA3_READING.md in lf6_build);
  bound s' - s <= delta t holds (max 4e-16 m), independent recomputation from the dumped plan_s agrees to 3e-8 m. Mutations M1, M2, M3 RED. Tool tests 12/12, 14 of 14 tool mutations RED. Reader tests 10/10, 19 of 19 reader mutations RED (incl. one-character changes of delta, kappa, smoothing, tolerance);
  reader ka-a5 PASS 12 of 12 (the banked A5 arrays in the A6 slots reproduce LF5_READ.json's LFO100 rules/label CEILING-HARMS and LF100 rules/label exactly; attribution against itself is exactly zero).
* **Deviation to rule on:** the card's wording for K3 ("bind rate strictly between") FAILS read at rollout level (each rollout on its own ticks: delta 0.5 binds 0.484 vs 0.427 at delta 1e9, heads bare); it holds at fixed inputs (0 < 0.484 < 0.490). First-run evidence: `lf6_build/raw/_old_KA_LF6_first_run_rollout_level_K3.json`. The card was not edited.
* The reader's production path (24-array extraction, LFB via lf_read.read_dose with doses B05/B10, LFOB via lfo_read.read_ceiling with the attribution base RILFO100, joint reading of card section 5) is exercised end to end only by ka-a5 and unit tests; its first production run is the chain's.

## Exact next step
On `raw/LF6.done`: read `lf6_read/RESULTS.md` (joint reading per card section 5, LFB labels, LFOB labels, attribution rows, instruments), then `python -I code/make_pins.py` in lf6_read and `code/make_pins_lf6.py` in lf6_build, package like `lf5_result_pkg`, and hand to the Master Mind to land with the claim-row update H-LEVELFLOOR-1 (A6).
A counted DOMINANT names kappa 1, its delta, says five heads-floor doses were read, and carries the adaptivity disclosure. If `LF6.fail` names the reader, rerun it with the command above.
