# PREREAD_V8F_SPEEDSRC: on the adopted base DAG-OACC, in which situations is T's speed-cell error concentrated? (FROZEN 2026-11-10 by the Master Mind, before any number of the full read exists)

Status: FROZEN (drafted by a Sonnet agent, reviewed and frozen by the Master Mind; the one change at freeze is the episode wording of the Estimator). Zero training, CPU. Instrument `theory/speedsrc/speedsrc_read.py`, the T1-kin first-tick loop of `theory/consorc/cons_orc_read.py` (batch build of `theory/otband/read/topk_read.py`, tree `F:/TanitAD-scratch/otband_read/tree`). The sha256 of this card and of the script go in `PREREAD_V8F_SPEEDSRC.sha256`; every threshold below is fixed in both. Only the 64-window smoke (controls) has been run.

## Why
* `theory/consorc/VERDICT_TOPK_PAIR_CONSORC.md` / `out/CONSORC.json` (MEASURED, T1-kin first tick, eval-141): T's tactical code carries ~0.87-0.89 of the tactical headroom in its continuous constraint cells, and the speed cells dominate. Earlier reads: T's speed error is systematic across seeds and information-limited, not a head bias (H-V8F-SPEEDCAL-1).
* Nobody has asked WHERE the missing information is. The answer names what a T-side card would feed T (a lead-agent feature, a stop/light cue, a curvature cue, ...) or says there is no such place and the lever is perception capacity.

## Arms (checkpoints as banked, one environment, no training)
`PARITY_DAGOACC_s0`, `PARITY_DAGOACC_s1` (the adopted base; the question's object). `INIT_M2b_s0` / `_s1` may ride along (beside, no bar).

## Per window (eval-141, 2,399 windows per arm, first tick)
* T's own code `own` = `forward_topk` `code`; GT code `gt` = `V8FMini._code_gt(lab_t)`; speed cells = the five of `CODE_KINDS["speed"]` (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms`); a cell is PRESENT where `lab_t["cons"]` is not NaN (windows without `lab_t.has` are not read).
* Speed error per present cell: `own - gt` in normalised units and in m/s (`x cons_sd`; `cons_mu` cancels in an error and is added only to the stored GT / own values).
* GT-SPEED-ONLY tactical ADE gain: ADE over knots 11-30 (indices 10-29) of `forward_diag(own with the speed cells set to gt)` minus TOP1 (negative = a gain). Off-distribution for an O trained on whole codes (H-V8F-KIND-1); reported beside, no bar.
* Window rows are written to `out/speedsrc_<tag>.npz` (clip index, window start, episode id, errors, ADE rows, strata) before any analysis.

## Strata (labels / GT future only; label-side, not an input). Thresholds fixed here
GT future positions `g` (30 knots, 0.2 s, ego frame at t0). Step speed `v_i = |g_i - g_(i-1)| / 0.2` (g_(-1) = origin). `v2` = mean of knots 8-11 (1.6-2.4 s), `v_end` = mean of knots 26-29 (5.2-6.0 s), `dv = v_end - v2`.
| variable | levels (in precedence order where relevant) |
|---|---|
| (a) regime | STOP: `v_end < 0.5 m/s`; DECEL: `dv < -1 m/s`; ACCEL: `dv > +1 m/s`; CRUISE: else |
| (b) lead | LEAD_NONE (the `lead_gap_m` cell absent); LEAD_LT15: gap < 15 m; LEAD_15_40; LEAD_GE40: gap >= 40 m (`lead_gap_m` is the GT lead gap cell of `lab_t["cons"]`; the `lead_tg_s` cells are present on the same windows by construction of the label) |
| (c) tlstop | TL: GT goal has any of `TRAFFIC_LIGHT_REACT`, `_RED`, `_YELLOW`, `_GREEN` (goal_y = 1 and goal_w = 1); STOP_POINT: GT goal STOP_POINT and no TL; NO_TL_STOP: neither |
| (d) curvature ahead | max |kappa| over knots 10-27 (2.2-5.6 s), kappa from centred tangents over 2-step chords: `kappa_j = wrap(psi_(j+1) - psi_(j-1)) / |g_(j+1) - g_(j-1)|`; a sample needs all three chords > 0.8 m (else skipped; no valid sample = 0). CURV_LO < 0.005 1/m, CURV_MID 0.005-0.02, CURV_HI >= 0.02 |
| (e) speed at t0 | `p10f[:, -1, 0]` (the dump's `v0`): V0_LO < 4 m/s, V0_MID 4-12, V0_HI >= 12 |

**Stratum (c) vocabulary finding (from source):** the 22-way goal vocabulary DOES contain traffic-light and stop goals, so (c) is AVAILABLE and uses the GT goal. `GOAL22` = `stack/tanitad/data/v8f_mini_data.py:52-55` (same tuple at `stack/tanitad/data/v9_labels.py:33`): index 6 `STOP_POINT`, indices 16-19 `TRAFFIC_LIGHT_REACT`, `_RED`, `_YELLOW`, `_GREEN` (also `YIELD`, index 5, not used here); `STOP_I = 6` at `stack/tanitad/models/v8f_mini.py:46`. The GT goal vector is `lab_t["goal_y"]` with mask `goal_w`, the first 22 entries of `_code_gt` (`v8f_mini.py:894-902`). Caveat (INHERITED, not re-verified here): `D-TLIGHT-1` (CLAUDE.md) records that the v7.x traffic-light label may not reach training; that bears on what T can learn, not on whether the stratum exists. The smoke saw 6 TL and 1 STOP_POINT windows in 63.

## Estimator
`taniteval.ci.stat_cluster_bootstrap`, B = 2,000, seed 0, resampling the eval-141 episodes (each window's `eid` from its dump; 139 episodes carry labels in CONSORC), on windows with `lab_t.has` and >= 1 present speed cell. Per-window error `mse_w` = mean over the present speed cells of `(own - gt)^2` in (m/s)^2; per-window mass `mass_w` = their sum. Stratum share of mass = sum of `mass_w` in the stratum / total; window share = windows in the stratum / windows read. The stratum-minus-complement difference is the difference of two group means of `mse_w` (the complement is a different set of windows, so this is an unpaired two-group difference with the same resampled episodes in both groups). **It answers "another draw of episodes" only**; it does not answer "another training run" (the two seeds are the only look at that) nor "another inference run" (the planner is deterministic). 17 strata levels are tested with no multiplicity correction; the both-seeds requirement is the guard.

## Committed outcomes (DAG-OACC, both seeds must agree)
A stratum QUALIFIES at one seed when all hold: >= 50 windows; mass share >= 0.30; mass share / window share >= 2.0; the stratum-minus-complement `mse_w` difference is positive with its 95 % interval excluding 0.
* **CONCENTRATED:** at least one stratum qualifies at BOTH seeds. The trigger names the information source a T-side card would feed T: regime -> the longitudinal intent / look-ahead cue; lead -> lead-agent state (gap, closing speed); tlstop -> a traffic-light / stop-line cue; curvature -> a road-geometry cue; speed level -> a speed-dependent scale of the target. A T-side card on the named source is drafted the same day, the stratum with the largest mass share first. If strata of several variables qualify, all are named (they overlap by construction) and the card feeds the one with the larger mass share first.
* **DIFFUSE:** no stratum qualifies at both seeds. The error is not located in a situation class; the trigger is perception capacity: the LoRA arm L, resolution, data scale.
* A stratum qualifying at one seed only is SEEDS-DISAGREE for that stratum (reported, no trigger).
* **Beside, no bar:** the GT-SPEED-ONLY tactical ADE gain by stratum (point, interval); per-cell speed MSE; the same table on M2b if run.
* Tier stamp: T1-kin first tick, oracle diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25), reported as paused, never as a pass.

## Controls (the read is INVALID if one fails)
1. Each checkpoint's sha256 equals its bank manifest where one exists.
2. TOP1 equals the arm's dump `ol` at tick 0 (max |diff| <= 1e-4 m); the window set and the GT future `g` equal the dump's; `v0` equals the dump's `v0`.
3. IDENT equals TOP1 (max |diff| <= 1e-5 m); OWN-ALL (`gt` with all 60 indices set to `own`) equals IDENT (max |diff| <= 1e-6 m).
4. Memory: 4 GB free before an arm, 3 GB before a chunk; `OMP_NUM_THREADS=2`, CPU only, one python read at a time.

## Log
* 2026-11-10 DRAFT written; smoke only (64 windows, controls on the model side; the dump-side controls need the full read).
* 2026-11-10 FROZEN by the Master Mind after review (script's qualify rule, verdict and episode resampling checked against this card). Only change: "the 40 val episodes" -> the eval-141 episodes. The trigger exists because H-V8F-CONSORC-1 read CONS-CARRY, kind speed, at both seeds.
