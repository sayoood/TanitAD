# PREREG_V8F_PT3M: does a four-frame nonlinear readout inside the cost change PT3N's NO-GAIN? (Master Mind, 2026-11-09, written before any PT3M code or run)

## Why
`plantest3/n1053/VERDICT_PT3N.md` read NO-GAIN on FLAT1 DEC-END with the ridge decoder fitted on N1053, with DECODER-WEAK raised and KIN-CEILING not beating HOLD. That card commits this next arm: a nonlinear multi-frame readout in the cost, fitted at N1053. DECN read at N1053 on the 17 inner-held-out TUNE episodes (T0, no interval): yaw rate MLP m = 4 0.570 / 0.441 against ridge m = 4 -0.058 / +0.109; speed ridge m = 4 0.695 / 0.686 against MLP m = 4 0.650 / 0.623. Tier: **T0, an oracle-goal diagnostic, never a driving result.**

## Objects (one change from PT3N: the reader in the cost)
The PT3N harness and windows unchanged: M2b s0 and s1 world models, zero training of the world model, 1,976 windows, TUNE and SCORE halves (salt `plantest2-v1`), 71 SCORE episodes / 994 windows, G-TRUE, G-RAND, HOLD, the TRAIN action prior, the SEED + REG(lambda) solver, the metric code, the N1053 fit set (53 TUNE inner-fit episodes plus DECN's 1,000 train clips), the 17 inner-held-out episodes.

**The reader.** Input: four consecutive level-1 latents, newest first (t, t-1, t-2, t-3), concatenated, as DECN. Per target, one reader, fixed now from DECN's held-out values above (no SCORE window has been read): **speed: ridge on four frames** (closed form, 1e-3 times the sample count); **yaw rate: DECN's MLP on four frames** (two hidden layers of 256, GELU, dropout 0.2, AdamW, weight decay 1e-2, batch 512, patience 12, cap 60 epochs, seed 0, early stopping on DECN's early-stopping episodes inside the fit set). Learning rate of the MLP chosen per training seed from {1e-3, 3e-4, 1e-4} by the early-stopping loss (DECN's MLP stopped at 13 to 20 epochs at 1e-3, so the first epochs were already the best); the held-out episodes are not used for this choice. Outputs in fit-standard units (speed and yaw rate each divided by its standard deviation over the fit samples), as PT3N.

**The cost.** DEC-END on the last four knots: `sum_j (r_j(roll at knots 12, 11, 10, 9) - r_j(goal at knots 12, 11, 10, 9))^2`. G-TRUE: the encoded true-future latents at those knots; G-RAND: the donor's. The reader carries no gradient of its own parameters during planning.

## Arms
FLAT1 DEC-END, plain and SHRINK(c), c as PLANTEST3 (0.496 / 0.526). TD3 and DEC-PATH are not run.

## Tuning and order
Lambda chosen on TUNE windows under G-TRUE per (form, seed, family) from {0, 0.1, 1}, as PLANTEST3 and PT3N. Then SCORE: G-TRUE and G-RAND at the chosen lambdas, planner seed 0; G-TRUE planner seed 1 as replicate. CPU only, at most 2 workers, RAM guard 6 GB.

## Read (paired episode-cluster bootstrap over the SCORE episodes, B 2,000, seed 0, both training seeds required)
Deciding, per family: action MAE as PLANTEST3 (LONGITUDINAL acceleration, LATERAL yaw rate, 12 logged actions). Beside, no bar: speed, heading and ADE at 2.4 s; distance of the plan from HOLD; chosen lambda; the planner-seed replicate; PT3M minus PT3N on the same windows, per form, seed and family; and the three ceilings below. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).

## Ceilings (no world model, no planner; SCORE; MAE against the 12 logged actions)
* **KIN-CEILING** under this reader: knots 3 to 11 from the reader on the encoded true future (acceleration from speed differences, yaw rate read directly); knots 0 to 2 stay at HOLD, because the four-frame reader needs k >= 3.
* **ORACLE-END:** what the END goal could express even if it were known exactly. Constant acceleration `(v_12 - v_0) / 2.4 s`, and a yaw-rate ramp from the measured current value to the true value at knot 12.
* **READ-END:** the same two formulas with `v_12` and the yaw rate at knot 12 taken from the reader on the encoded true-future frames, and the measured current speed and yaw rate at the start (R-EGO: current measured ego state is admissible).

## Committed outcomes (per form and family, both seeds; PLANTEST3's table unchanged)
* **PLANNABLE:** G-TRUE minus HOLD is separated-better and G-TRUE minus G-RAND is separated-better, at both training seeds.
* **GOAL-BLIND:** G-TRUE minus G-RAND is not separated.
* **NO-GAIN:** G-TRUE minus G-RAND is separated-better, and G-TRUE minus HOLD is not separated-better.
* **SEEDS-DISAGREE:** the two training seeds carry different labels; it does not count as PLANNABLE.
* **HOLD-COLLAPSE (flag):** the plan's mean absolute distance from HOLD is below 10 % of the prior std; the row does not count as planning.
* **DECODER-WEAK (flag on the whole read):** held-out R2 of this reader on encoded latents below 0.70 for speed or below 0.50 for yaw rate at either seed.

## What each outcome triggers (committed now, Rule Zero)
* **PLANNABLE without HOLD-COLLAPSE in either family:** this cost and reader become the HWM-PLAN level-1 cost and are rerun on the ladder world models.
* **No such row, ORACLE-END does not beat HOLD (separated-better, both seeds) in a family:** an end-point goal cannot express that family at this horizon, whatever the reader; next for that family is a per-knot goal (DEC-PATH under this reader), with the tactical goals of R-T2O as the target design.
* **No such row, ORACLE-END beats HOLD and READ-END does not, in a family:** the reader's precision is the limit; next are the N4565 readers of the full-n probe and the measured-speed world-model card (`egowm/PREREG_V8F_EGOWM_DRAFT.md`).
* **No such row, READ-END beats HOLD in a family:** the goal is readable and the plan does not reach it, so the world model's action response is the limit; next is the action-sensitivity lever on the ladder arms and the measured-speed world-model card.
* **GOAL-BLIND:** action sensitivity first.
* **Every outcome:** the harness is read on the ladder world models when they finish; PLANTEST and PLANTEST2 stay required gates.

## Controls (the read is INVALID if one fails)
1. **Reader tie to DECN:** on the 17 held-out episodes, the speed ridge reads within 0.01 of DECN's ridge m = 4 row at N1053, and the yaw-rate MLP at learning rate 1e-3 within 0.01 of DECN's MLP m = 4 row, both seeds.
2. **Reachability:** on real latents and goals, the gradient of the cost with respect to the plan is non-zero in both action channels under the full rollout, and exactly 0.0 under SHRINK(0), through this cost class and from this runner.
3. **Harness tie:** on two real SCORE batches, this cost class holding the N1053 single-frame ridge as its reader gives the same cost as PT3N's DEC-END (relative difference at most 1e-4 on every candidate) and the same gradient (relative difference of the norm at most 1e-3).
4. **Split:** no train clip is an eval clip, and the fit reads no SCORE or inner-held-out window (DECN's `split_guard_n` on ids; a mutation goes red).
5. **Clips:** the 1,000 shards are DECN's; their count and order are asserted.
6. **Isolation:** PLANTEST3's and PT3N's `tune_choice.json` are unchanged by this run (sha256 before and after; every write goes under `plantest3/mlp4/`).

## Log
* 2026-11-09 12:04:08 frozen: prefix 7105 bytes, sha256 70797004302bf7c302450fb651625d4a1ce2bb6e3b2869b3cfdfe17e1f1d0eec (everything above this section). No PT3M code or run existed at this time.
