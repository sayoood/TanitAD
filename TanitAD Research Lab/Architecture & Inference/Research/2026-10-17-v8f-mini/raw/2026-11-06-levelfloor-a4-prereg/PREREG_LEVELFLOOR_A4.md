# Addendum A4 to PREREG_LEVELFLOOR_A3: arm LFO075, the level floor with the level estimate replaced by the logged speed, a CEILING diagnostic (written 2026-11-06 03:4x after the LF read, before any LFO rollout)

**Why.** LF read P-FAILS-M1-MOVED at both doses (`raw/2026-11-05-levelfloor-lf-result/lf_read/RESULTS.md`). At κ 0.75 the floor binds on
0.455 / 0.462 of the M1 crawl ticks and closes 0.395 / 0.392 of the excess, and the guards fail broadly (ADE6 +0.3683 / +0.2172 m). The
floor speed misses the logged speed by 3.894 / 3.979 m/s (MAE) and exceeds it on 0.191 / 0.174 of the ticks with a floor. The A3 card
reads this as "the level estimator is the bottleneck", and its levers (the A40, T2 re-perception, a stronger estimator on more data) need
resources that are the PI's to allocate.

That reading is an inference. Nothing measured so far separates its two parts:
* the estimator's error, which a better estimator removes;
* harm from the operator itself (the ramp, the resampling along O's path), which no estimator removes. At κ 0.50 the floor exceeds the
  logged speed on only 0.066 / 0.055 of the ticks, yet ADE6 is +0.1089 m at s0 and cross-track MAE is separated-worse at both seeds.

This card measures what F gives with a perfect level estimate, at the same dose and on the same planner, so the estimator's value is
MEASURED before any resource is asked for it. Zero training, dev-box CPU only, no box-2 job. PI go of 2026-11-05 as in A3. Claim row:
H-LEVELFLOOR-1 (A4).

⛔ **The arm is NOT admissible at inference, by design.** Its level input is the human's logged route speed, a privileged channel. No
outcome of this card is a driving result, a capability claim or a counted DOMINANT. It ranks levers and nothing else, and every report
of it carries the label CEILING.

## 1. Hypothesis
If the level estimate were exact, the floor at κ 0.75 would close at least half of the M1 excess inside the CL guards. The harm read at
LF075 would then be the estimator's, and a better estimator is the lever worth its compute.

## 2. The arm
* **LFO075_s{0,1}.** Everything is LF075_s{s}'s, with one difference: the level input.
  * **Planner:** C0_s{s}, as in LF.
  * **Lead cap:** π and g from LLR_s{s}'s heads, as in LF.
  * **Dose:** κ = 0.75.
  * **Operator:** F steps 1 to 5, the 3 m/s band and every constant, unchanged from the A3 card.
* **The level input:** v_hat := v_log, the logged route speed at the tick. It is the reader's own definition (`lf_read.py`,
  `logged_speed`): the path length of knots 1 to 6 of the tick's logged route in the ego frame, divided by 1.0 s. Where the tick has no
  complete logged route (`gt_ok` False), there is no floor.
  * It is computed in the tool at the replan, before F, from the same `gt_in_frame` the tick dump uses, so the instrument and the input
    are the same quantity.
* **Attribution:** LFO075 − LF075 isolates the level estimate's error at a fixed operator, dose, planner and lead cap.
* **Placement, as in LF:** bare arm X = O's plan → S (sg72) → F; composite XC = S → F → kbox (A3_RULING_1).
* **Known answers before any rollout:**
  * with the level source set to the heads, the new tool reproduces the banked LF075 tick arrays on one rollout bit for bit;
  * with the floor off, it reproduces RIC0 exactly;
  * on a synthetic tick, the oracle input equals `logged_speed` of the same `gt` to 1e-9 m/s.
* **Nothing here was chosen on eval-141.** The dose is LF075's own, and the level definition is the reader's own, both fixed before the
  LF read.

## 3. Reads (T1-kin on the RI tool + F; paired episode-cluster bootstrap B = 2000, seed 0, eval-141)
* **Rollouts:** RILFO075_s{0,1} (bare) and RILFO075C_s{0,1} (composite), on `reindexed` only. Four rollouts.
  * `logged` is not read. Its role in LF was the LEAK-BORNE rule, i.e. whether O's frames at the human's clock carry the lift. Here the
    level input is the human's speed by construction, so that rule has no question left to answer. This is disclosed.
* **Bases:** the banked arrays, unchanged: RIC0, RIC0C, C0, C0REP, P10, RILF075 and RILF075C.
* **Instruments**, as A3 section 3: bind rate overall and inside the M1 ticks, cap rate, mean lift where bound, the negative chord lift
  (A3_RULING_2). Beside them: v_f against the logged speed, which here reads (1 − κ) · v_log by construction and is a check, not a result.
* **Reach control:** as A3. If F binds on fewer than 1 % of the M1 ticks at either seed, the read is F-UNREACHED.

## 4. Rules (the A3 rules with LF := LFO, deciding scene `reindexed`)
* **P_O:** (RIC0 − RILFO)[M1] / (RIC0 − P10)[M1] >= 0.5 at both seeds, with RILFO − RIC0 on M1 separated-better at both.
* **G_O:** on RILFO − RIC0, ADE6 point <= +0.0604 m and no non-crawl row of the 19 separated-worse, at both seeds.
* **F_O:** |RILFO − RIC0|[M1] > |C0REP − C0|[M1] at each seed.
* **Attribution (reported, never deciding):** RILFO − RILF075 on all 19 rows, at both seeds, separated or not.
* **The composite** RILFOC − P10, read A (D1, D2, D3 over four families): reported as the CEILING of D. Never counted.

## 5. Outcomes and the next arm, committed now (Rule Zero)
* **P_O and G_O hold at both seeds: CEILING-CLEARS.**
  * The operator works given a good level, so the estimator is the MEASURED bottleneck. Its value is the ceiling read here.
  * This goes to the PI as the evidence for the estimator levers: the A40 motion-aware encoder, or a stronger estimator on more data.
  * The next CPU arm is a degradation sweep (addendum A5): the oracle input with a priori multiplicative noise. It sets the estimator
    accuracy at which P and G stop holding, i.e. the spec those levers must meet.
* **P_O holds, G_O fails: CEILING-HARMS.**
  * Even an exact level harms, so the harm is the operator's and no estimator removes it. The failing rows name it.
  * The estimator levers are NOT signalled.
  * The next arm changes F, not its input: an operator variant in its own addendum, written after the failing rows are read and before
    its rollouts.
* **P_O fails (at either seed): CEILING-SHORT.**
  * An exact level at κ 0.75 does not close half the crawl, so a floor on progress is not the form that closes M1 at this dose.
  * The estimator levers are NOT signalled for this form.
  * What remains is reported to the PI: T2 re-perception (option (c)), or the operator at κ 1 in its own addendum.
* **F-UNREACHED:** the instrument is checked first.

## 6. Variance named
* **Episodes:** the paired episode-cluster bootstrap.
* **Training:** O is C0's own checkpoint in every contrast with C0, and the lead cap comes from one LLR run per seed (not measured).
* **Inference:** deterministic; the oracle input carries no draw.

## 7. Freeze
This file's sha256 is recorded in `a4_prereg/FROZEN_SHA.txt` before any LFO rollout.
