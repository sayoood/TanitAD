# Addendum A5 to PREREG_LEVELFLOOR_A4: the level floor at κ 1, read with the logged speed, with the logged speed under a priori noise, and with the heads (written 2026-11-06 04:5x after the A4 read, before any κ 1 rollout)

**Why.** A4 read CEILING-SHORT (`raw/2026-11-06-levelfloor-a4-lfo-result/lfo_read/RESULTS.md`, landed cc129bed). With an exact level at
κ 0.75, the floor closes 0.126 / 0.132 of the M1 excess, against 0.395 / 0.392 with the heads, and it removes most of LF075's harm
(attribution RILFO − RILF075: ADE6 −0.3916 / −0.2418). LF075's closure came largely from the estimate overshooting the human's speed.

The descriptive geometry of the A4 rollouts, computed after the read (`beside/M1_GEOMETRY.md`), explains the shortfall. On 0.848 / 0.838
of the M1 ticks, 0.75 · v_log < 5.2 m/s, so a κ 0.75 floor cannot lift the target above the M1 threshold. At κ 1 that share is 0.0.

The A4 card committed the operator at κ 1 as its next arm for this outcome (section 5). This card reads κ 1 three ways:
* with the exact level (the ceiling);
* with the exact level under a priori noise, to set the estimator accuracy at which the floor stops working;
* with the trained heads (the admissible arm).

Zero training, dev-box CPU only, no box-2 job. PI go of 2026-11-05 as in A3. Claim row: H-LEVELFLOOR-1 (A5).

## 1. Hypotheses
* **[H1]** With the exact level at κ 1, the floor closes at least half the M1 excess inside the CL guards.
* **[H2]** The closure survives some estimator error, and the noise level at which P or G first fails is the accuracy spec an estimator
  must meet.

## 2. The arms (κ = 1.0 in every arm; everything else is LF075's)
Common to every arm: planner C0_s{s}; lead cap π and g from LLR_s{s}'s heads; operator F, the 3 m/s band and every constant unchanged
from the A3 card; placement as in LF (bare = S → F; composite = S → F → kbox).

* **LFO100_s{0,1}** (CEILING): v_hat := v_log, exactly as in A4.
* **LFN100x10_s{0,1}** and **LFN100x20_s{0,1}** (CEILING):
  * v_hat := v_log · exp(σ ε), with σ = 0.10 and σ = 0.20, fixed a priori and not fitted.
  * ε ~ N(0, 1) is drawn independently per replan by one generator per rollout, seeded 20261106.
  * One draw is taken at every replan, in replan order, whether or not a floor applies, so the bare and composite rollouts of a seed
    see the same draws.
* **LF100_s{0,1}** (ADMISSIBLE, vision-only at inference, exactly as LF): v_hat from LLR's level head. This is the only arm that can
  count.

**Known answers before any κ 1 rollout:**
* with σ = 0, the noise path reproduces LFO100 on a 64-window smoke bit for bit;
* the draws are identical between a bare and a composite smoke of one seed;
* the LFO tool with `--floor-kappa 0.75` and no noise still reproduces the banked RILFO075_s0 on the 64-window smoke.

## 3. Reads (T1-kin on the LFO tool; paired episode-cluster bootstrap B = 2000, seed 0, eval-141)
* **Rollouts:**
  * every arm and seed, bare and composite, on `reindexed` (deciding): 16 rollouts;
  * LF100 also on `logged`, bare and composite, for its LEAK-BORNE rule: 4 rollouts.
  * 20 in all.
* **Bases:** the banked arrays, unchanged: RIC0, RIC0C, C0, C0C, C0REP, P10, RILF075, RILFO075.
* **Instruments**, as A4: bind rate overall and in the M1 ticks, cap rate, mean lift where bound, negative chord lift, no-route share.
  Beside them, for the noise arms, the realised MAE of v_hat against v_log. For LF100, the heads' MAE and overshoot share against v_log.

## 4. Rules (deciding scene `reindexed`)
* **The ceiling arms LFO100, LFN100x10, LFN100x20:** P_O, G_O and F_O exactly as A4 (with LFO := the arm). Each arm gets one label:
  CEILING-CLEARS, CEILING-HARMS, CEILING-SHORT or F-UNREACHED. The composite − P10 is reported as the CEILING of D, never counted.
* **The admissible arm LF100:** the A3 rules exactly (P, G, F, D read A over four families, LEAK-BORNE on `logged`, reach control),
  with LF := LF100.
* **Multiplicity, named:** the heads-floor family has now read three doses (κ 0.50, 0.75, 1.0). A counted DOMINANT names its dose and
  says that three were read.
* **The spec read (reported, never deciding).**
  * σ* is the largest σ in {0, 0.10, 0.20} at which P_O and G_O both hold at both seeds.
  * It is set beside the heads' realised relative error on the same frames.

## 5. Outcomes and the next arm, committed now (Rule Zero)
* **LF100: D holds at both seeds, with P and G, and the arm is not LEAK-BORNE: counted DOMINANT.**
  * Signal the PI for scale (data and size), with the video evidence.
  * It names κ 1 and states that three doses were read.
* **LFO100 CEILING-CLEARS and LF100 not DOMINANT: the form works given a good level, and σ* is the spec.**
  * If σ* >= 0.10, the estimator levers (the A40, a stronger estimator on more data) go to the PI with the ceiling, σ* and the heads'
    measured error as the case.
  * The next CPU arm is a temporal filter of the heads' level in its own addendum.
  * If σ* = 0, the form needs an exact level. That is reported as such, and no estimator lever is signalled on this evidence.
* **LFO100 CEILING-HARMS:** matching the human's speed along O's path harms on the named rows, so the operator is the problem. The next
  arm changes F in its own addendum, and no estimator lever is signalled.
* **LFO100 CEILING-SHORT:** even the human's exact speed, reached by the 1.0 m/s² ramp along O's path, does not close half the crawl, so
  the progress-floor form is exhausted. What remains is reported to the PI: T2 re-perception (option (c)) and the A40.
* **F-UNREACHED:** the instrument is checked first.

## 6. Variance named
* **Episodes:** the paired episode-cluster bootstrap.
* **Training:** O is C0's own checkpoint in every contrast with C0, and the heads come from one LLR run per seed (not measured).
* **Inference:** LFO100 and LF100 are deterministic. The noise arms carry one noise draw (one generator seed), so a difference between
  noise levels smaller than a second draw would show is not resolved here. This is disclosed, and the spec read is reported, never
  deciding.

## 7. Freeze
This file's sha256 is recorded in `a5_prereg/FROZEN_SHA.txt` before any κ 1 rollout or tool change.
