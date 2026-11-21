# VERDICT SLOWWM: CELLS-DISAGREE (NO-EFFECT everywhere, SLOW-COSTS on LATERAL in three cells) -> NO-EFFECT's next step

Card `PREREG_V8F_SLOWWM.md` (sha256 `7c2775e3...94ca`), `AMEND_SLOWWM_1.md` (sha256 `5d2270d4...1549`, frozen before any basis or run). Read `RESULT_SLOWWM.json` / `.md` (deciding space `traj`: LONGITUDINAL = `speed_err`, LATERAL = `heading_err`), generated 2026-11-14 00:06 by `report_slowwm.py`. Beside: `RESULT_SLOWWM_action.*` (`--space action`), which gives the same cell pattern except PLANTEST c1 = NO-EFFECT only.

**Tier T0 (ladder planner harness, ORACLE goal, frozen readers and planner, world model only): a world-model diagnostic, never driving performance.** Estimator: EC-BOOT, paired episode-cluster bootstrap, B 2,000. It answers "another draw of episodes" only; "another training run" is the floor |m(SLOW_rep) − m(SLOW_s0)|, from one replicate pair. Evidence class MEASURED. Controls a-d: PASS (`CONTROLS_READ_SLOWWM.json`, banked record `be6b4f0d`, `all_pass` True).

Plan arrays (117 files, the read's input, numeric only): PRIVATE on HF `Sayood/tanitad-v8f-plantest3-runs` `slowwm/slowwm_plan_runs.tar`, rev `6b49a4da`, sha256 `6cda600e...32e1`, equal after a re-download and as the remote LFS sha256 (`HF_BANK_SLOWWM.json`, with each file's sha256).

## Outcome as committed
* Cells (A2: {PLANTEST, PLANTEST2} x {c1, c3}): PLANTEST c1 MIXED {NO-EFFECT, SLOW-COSTS}; PLANTEST c3 MIXED; PLANTEST2 c1 NO-EFFECT; PLANTEST2 c3 MIXED. No outcome is shared by all four, so the read is **CELLS-DISAGREE** (A3/A4). The least favourable cell's next step applies: `next_step_of` = SLOW-COSTS, whose next step the card sets to NO-EFFECT's.
* SLOW-COSTS comes from LATERAL alone. In PLANTEST c1, c3 and PLANTEST2 c3, SLOW's margin is below R35H's by more than the floor at both seeds. There SLOW still beats PCA-B. LONGITUDINAL holds NO-EFFECT in every cell.
* Committed next step (card, NO-EFFECT row): **"The z1 basis is not the planner's limit. The predicted-goal path (HWM-PLAN G4 / G5) and the T-side speed levers get the next box-2 time; L4 is deprioritised."**

## The numbers that carry it (PLANTEST, context 1; full tables in `RESULT_SLOWWM.md`)
m = mean(HOLD) − mean(G-TRUE), positive = planning beats holding. Seeds s0 / s1:

| arm | LONGITUDINAL m | LATERAL m |
|---|---|---|
| R35H (record) | −0.6486 / −0.7489 | −0.0672 / −0.0831 |
| SLOW | −0.9384 / −0.8067 | −0.1033 / −0.0988 |
| PCA-B | −0.7761 / −0.8197 | −0.1127 / −0.1141 |
| SLOW_rep (s2) | −0.6936 | −0.0980 |

Floors: LONGITUDINAL 0.2448, LATERAL 0.0053. Every m interval is wholly below 0. G-TRUE − G-RAND is negative in every row of every cell: the true goal is used, planning toward it beats a random goal.

## Master Mind context
* **No arm plans better than holding, the world model of record included.** All 7 arms x 4 cells x 2 families read NO-GAIN, with G-TRUE worse than HOLD and better than G-RAND. That is the pattern of every earlier planner read on this stream (PLANTEST, PLANTEST2, PLANTEST3, PT3N, PT3M: NO-GAIN). Changing z1's basis to a whitened temporal-CCA basis (SLOW) or a whitened PCA basis (PCA-B) does not move it. SLOW makes the lateral margin worse. The card's own reading follows: the basis is not the limit.
* **IA-0 (beside, no bar) agrees in direction:** at the 2 s band end SLOW's latent skill over its own Copy-Last (0.129-0.144) is the lowest of the three bases (PCA-B 0.184-0.185, R35H 0.259-0.267). Each skill is in its own z1 space, so this is descriptive (`IA0_README.md`).
* **Caveat on SLOW-COSTS:** the LATERAL floors come from one replicate pair and are small (0.0003 to 0.0075). The SLOW-minus-R35H gaps that trigger the cost (about 0.016 to 0.036 at c1) would still clear a floor several times larger. With one replicate pair, the training-run question is answered once (H-ESTIM-SEED-1).
* **SLOW's cov(u), reported beside as `AMEND_SLOWWM_2` B asks** (record `basis_check3/CHECK3.json` in the repo beside this file, schema `slowwm-check3-v2`, pulled 2026-11-14 from box 2 `D:\tanitad\slowwm\basis\`, sha256 `49fc02d2...47a9`, equal on both ends; the per-split records `CHECK3.eval.json` / `CHECK3.train.json` beside it): the card's identity cov(u) = I − ridge VᵀV holds to 7.4e-8 (tol 1e-3, ridge 1.13e-4, k 256); PCA-B's cov(u) = I holds to 5.6e-8. SLOW's deviation from I is 1.0e-4 to 0.0154 on the diagonal (per direction, 256 values in the record) and at most 0.0016 off it, so SLOW's input scale is at most 1.6 % below PCA-B's in any direction. That is far smaller than the margin gaps above. `AMEND_SLOWWM_2` A (check 3 per split) holds: both splits pass, the shuffled-time basis fails as it must.
* **Families:** LONGITUDINAL and LATERAL read; TACTICAL n = 0, NOT COMPUTABLE (the harness returns action plans, no decision output); STRATEGIC PAUSED (PI item 25).

## Next lever (Rule Zero)
HWM-PLAN G4/G5, the predicted-goal path on the frozen ladder world models (`theory/hwm_plan/`; zero world-model training, a head-only level-3 goal head). **Blocked on:** its card is a DRAFT (`PREREG_V8F_HWMPLAN_DRAFT.md`, agent draft 2026-11-08, "the Master Mind freezes it"). Planner code, tests and smokes exist (`TEST_HWMPLAN.done`, `SMOKE_HWMPLAN.done`, `SMOKE_BANDS.done`). Next action: the MM reviews the draft against this read and PLANTEST3 and freezes it, then runs it. The T-side speed levers are the other committed recipient of box-2 time.
