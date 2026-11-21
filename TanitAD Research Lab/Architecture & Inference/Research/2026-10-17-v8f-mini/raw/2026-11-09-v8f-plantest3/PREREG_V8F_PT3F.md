# PREREG_V8F_PT3F: does a reader fitted on every trained clip bring READ-END under HOLD? (Master Mind, 2026-11-09, written before any PT3F code or run)

## Why
`plantest3/mlp4/VERDICT_PT3M.md` applied its card's row: no PLANNABLE row, ORACLE-END beats HOLD and READ-END does not, in both families, so the reader's precision is the limit; the card commits the N4565 readers of the full-n probe next. READ-END needs no world model and no planner, so it bounds what a planner with this reader could do, on CPU, in under an hour. Tier: **T0, an oracle-goal ceiling, never a driving result.**

## Objects
PT3M's SCORE windows (71 episodes, 994 windows), M2b s0 and s1, HOLD, the ORACLE-END and READ-END formulas in `pt3m_lib.end_plan`, measured current speed and yaw rate at the start. The fit set is the full-n probe's: 4,512 train clips plus the 53 TUNE inner-fit episodes (4,565 episodes), with DECN's early-stopping episodes and the 17 inner-held-out episodes. Input: four consecutive z1 latents, newest first. No SCORE window enters any fit or any choice.

## Readers
* **Speed, two forms at N4565:** ridge on four frames (`pt3m_lib.ridge_weights`), and DECN's MLP on four frames at learning rate 1e-3.
* **Yaw rate at N4565:** DECN's MLP on four frames at learning rate 1e-3.
* **The speed form used for the deciding row is fixed per training seed by the held-out R2 on the 17 inner-held-out episodes:** the higher of ridge and MLP. The rows of `decinfo/decn/fulln/runs/R0_*_N4565.json` already show this: MLP at s0 (0.778 against 0.755), ridge at s1 (0.696 against 0.688). The run recomputes the choice from its own fit and control 1 ties it to those rows.
* MLP architecture, optimiser and stopping as DECN (two hidden layers of 256, GELU, dropout 0.2, AdamW, weight decay 1e-2, batch 512, patience 12, cap 60, seed 0).

## Arms (READ-END on the SCORE windows, reader on the encoded true future at knots 12, 11, 10, 9)
* **F-DECIDE:** N4565, speed by the rule above, yaw rate MLP at 1e-3. The only deciding arm.
* **F-OTHER (beside):** N4565, the other speed form.
* **N1053-LR1E3 (beside):** PT3M's fit set, its speed ridge, the yaw-rate MLP at 1e-3 in place of the frozen rule's 1e-4. The row `VERDICT_PT3M.md` carries forward.
* **PT3M (tie):** PT3M's saved reader through this runner.

## Read (paired episode-cluster bootstrap over the 71 SCORE episodes, B 2,000, seed 0, both training seeds required)
Deciding, per family: action MAE against the 12 logged actions (LONGITUDINAL acceleration, LATERAL yaw rate), READ-END minus HOLD. Beside, no bar: ORACLE-END; the absolute read error at knot 12 in each target; acceleration from the difference of two reads (knots 12 and 3); ADE, speed error and heading error at 2.4 s. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).

## Committed outcomes (per family, F-DECIDE, both seeds)
* **READS:** READ-END minus HOLD is separated-better at both seeds.
* **NOT-READ:** it is not separated-better at both seeds.
* **SEEDS-DISAGREE:** separated-better at one seed only. It does not count as READS.

## What each outcome triggers (committed now, Rule Zero)
* **READS in a family:** FLAT1 DEC-END under the F-DECIDE readers, PT3M's protocol with only the reader changed, on that family, under an amendment to this card written before the run.
* **NOT-READ or SEEDS-DISAGREE in a family:** the reader lever on z1 at this scale is spent for that family. No planner run follows from this card. Next for that family: the harness read on the ladder world models when they finish, and the measured-speed world-model card (`egowm/PREREG_V8F_EGOWM_DRAFT.md`), frozen after the ladder's T0 read and run after the band retrain (PI, 2026-11-09).
* **Every outcome:** the latent shards under `F:/TanitAD-scratch/decn_scratch/` are deleted when this card's read is landed, unless a READS row needs them.

## Controls (the read is INVALID if one fails)
1. **Reader tie to the full-n probe:** on the 17 held-out episodes, the speed ridge, speed MLP and yaw-rate MLP at N4565 each read within 0.01 of the matching four-frame row of `R0_<tag>_N4565.json`, both seeds.
2. **Tie to PT3M:** PT3M's saved reader through this runner gives the READ-END acceleration MAE and yaw-rate MAE of PT3M's `ceil_<tag>.npz` within 1e-9, window by window.
3. **HOLD and ORACLE-END tie:** identical to PT3M's on every window (at most 1e-9).
4. **Split:** `split_guard_n` on the real ids before each fit; a mutation of the guard goes red; no SCORE id in the fit set.
5. **Clips:** 4,512 unique train clips, all shards present.
6. **Isolation:** PT3M's `tune_choice.json` and `ceil_*.npz` are unchanged by this run (sha256 before and after; every write goes under `plantest3/ceilf/`).

## Log
* 2026-11-09 15:15:42 frozen: prefix 4731 bytes, sha256 6c826c3938f941b8d6f16d37f53a03eb76eda0449d96c495dd2dbd9f3ef2dfe9 (everything above this section). The PT3F code was drafted alongside this card and has not been run; no PT3F result existed at this time.
