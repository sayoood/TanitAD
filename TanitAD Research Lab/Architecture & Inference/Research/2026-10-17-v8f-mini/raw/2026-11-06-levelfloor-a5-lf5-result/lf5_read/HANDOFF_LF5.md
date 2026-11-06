# HANDOFF_LF5: addendum A5 (the level floor at kappa 1.0: LFO100 / LFN100x10 / LFN100x20 / LF100). Built, validated, chain LAUNCHED 2026-11-06 05:01. Nothing committed, nothing landed.

Card `a5_prereg/PREREG_LEVELFLOOR_A5.md` sha 2aac1dbe (frozen, unchanged; time label: `A5_NOTE_TIME_LABEL.md`). Packages (staged, not in git): `E:/Projects/TanitAD-artifacts/dagland/motion_c/lf5_build/` (tool + known answers) and `.../lf5_read/` (reader, precheck, chain). Scratch `F:/TanitAD-scratch/motion_c/lf5_read/` (`tree/` read tree, `out/` 20 rollouts, `work/`, `smoke/`, `pt/`) and `.../lf5_build/` (`smoke/`, `mut/`).

## State
* Launched with WMI `Win32_Process.Create` of `lf5_read/lf5_chain.cmd` (WMI pid 53596); the chain's own pid is in `raw/LF5.running`. Gates 0-2 and the heads pre-flight passed at 05:01. 20 rollouts of ~4-5 min on 12 CPU threads (ESTIMATED ~90-100 min), then the reader (ESTIMATED 15-25 min).
* Order: RILF100 (s0, s1, C s0, C s1: heads, admissible), RILFO100 x4 (logged), RILFN100x10 x4, RILFN100x20 x4 (noise seed 20261106), LF100 x4 (`logged` scene), reader. All `reindexed` except LF100.
* Check with: `ls raw; tail -n 8 logs/chain.out` (lines `ZZLF5-*`). Markers: `raw/LF5.done` or `raw/LF5.fail` (names the failing step; a killed chain leaves one). RESUME = launch `lf5_chain.cmd` again (finished rollouts are skipped). A reader failure still leaves the 20 rollouts banked.

## Built and validated (MEASURED; logs in the packages)
* Tool `lf5_build/tool/v8f_t1_eval_ri_lf5.py` sha 88b275a1 = LFO tool 6744d3e4 + `code/apply_lf5_patch.py` (regenerates byte for byte). Flags `--floor-vhat-noise SIGMA` (only with `--floor-vhat-source logged`), `--floor-vhat-noise-seed N` (required for SIGMA > 0); v_hat = v_log * exp(SIGMA * eps), one `default_rng(N)` per rollout, one vector draw per replan call in replan order, taken whether or not a floor applies. Ticks gain `lf_vlog`, `lf_eps`; header records both.
* KA-N1 PASS (sigma 0 on the noise path == LFO tool at kappa 1.0, 35 common tick fields + analysis arms bit for bit; M1 mutation RED). KA-N2 PASS (bare and composite eps identical, 192 rows; M2 "draw only where bound" RED). KA-N3 PASS ((a) kappa 0.75 no noise == the LFO tool's own smoke bit for bit; (b) == banked RILFO075_s0 on the 46 of 64 windows whose model output is batch-identical; M3 RED). Tool tests 5/5, 8 of 8 tool mutations RED.
* Reader `code/lf5_read.py` (loads lfo_read.py and lf_read.py by pinned path); ka-a4 PASS 6 of 6 (banked RILFO075 arrays in the LFO100 slot reproduce LFO_READ.json P_O, G_O, F_O, ceiling of D and label CEILING-SHORT exactly). Reader tests 6/6, 11 of 11 mutations RED. The production path (rollout controls, 20-array extraction, LF100 via lf_read.read_dose with the dose table extended to "100") is exercised only by smoke ticks, not by a full read: the first production run is the chain's.

## Exact next step
On `raw/LF5.done`: read `lf5_read/RESULTS.md` (joint reading per card section 5, LF100 label, sigma*, spec read), then package `lf5_build` + `lf5_read` like `lfo_result_pkg` (`python -I code/make_pins.py` in lf5_read; `make_pins_lf5.py` in lf5_build) and hand to the Master Mind to land, with the A5 claim-row update H-LEVELFLOOR-1 (A5). A counted DOMINANT (LF100 D-HOLDS) names kappa 1.0 and says three doses were read. If `LF5.fail` names the reader, run `python -I code/lf5_read.py --mode production --out-root F:/TanitAD-scratch/motion_c/lf5_read/out --work-dir F:/TanitAD-scratch/motion_c/lf5_read/work` after fixing the named defect.

## Disclosed
* KA-N3(b) cannot be bit-identical on all 64 windows: the smoke's batch composition differs from the banked run's and the model's float32 output moves on 18 of 64 windows (plan_raw), with or without the patch; the comparison is on the clean windows and (a) is the strict one.
* s0 and s1 see the same noise draws (one generator seed, one window order): the noise arms carry one draw.
