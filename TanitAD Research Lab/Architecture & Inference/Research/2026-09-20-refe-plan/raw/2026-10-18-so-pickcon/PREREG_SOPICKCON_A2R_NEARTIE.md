# PREREG_SOPICKCON A2R: the near-tie-robust redesign of SO-CARVE A2's control 3, and the A2 tracking read on the near-tie-free sets

Written 2026-10-18, BEFORE any margin, runner-up-mass or control number of this addendum was computed. sha256 in `PREREG_SOPICKCON_A2R_NEARTIE.sha256`. It changes nothing in SO-CARVE's PREREG_SOCARVE*.md or `tempsweep_a2.json`; it REPLACES control 3 only and adds one robustness read.

## 0. What I had seen before writing this (disclosure)
* `tempsweep_a2.json` (all 3,000 sets): L_T at T = 0.1 / 0.05 / 0.02 / 0.01 against L_P1 at u250 / u500 / u1000, outcome A2-ALIGNED(T* = 0.02) PROVISIONAL because control 3 FAILED as written.
* `logs/a2_control3_diag.log` (post hoc): at u1000, T = 1e-4 reads L - L_P1 = -0.0174 (0 picks differ from the argmax) while T = 1e-6 / 1e-8 / 1e-10 read exactly 0.0; the largest non-top softmax mass over sets is 0.74 (B) / 0.82 (A) at T = 1e-4, 0.49 / 0.38 at 1e-6, 0.11 / 0.0 at 1e-8. So numerical near-ties exist; their number and their labels I have NOT looked at.
* I have computed NO margin, no runner-up mass and no coverage figure. Nothing below is fitted to a number of this addendum.

## 1. The defect being repaired
SO-CARVE A2 control 3: "T = 1e-4 must reproduce L_P1 to 1e-6". It cannot, by construction, on sets whose top two selection scores differ by less than a few T in log space: there the softmax puts non-negligible mass on a runner-up that the argmax ignores, so E_pi[TTC] differs from the pick's TTC. That is a property of the data (near-ties) and of the tolerance, not of the machinery. The control must be restricted to sets on which the neglected mass is provably below the tolerance, and that restriction must be derived, not tuned.

## 2. Quantities (all CPU numpy, the SAME banked states and loaders as `so_tempsweep_a2.py`)
States: R0 u250 / u500 / u1000, arms A and B, full RL-val (3,000 sets / 75 logs), `scores_update{u}_{A,B}.npz`, true PDM TTC labels `y`.
Selection score of arm X on set s: `a = AT.sel_v1_lane(AT.RV.sigmoid(logits))` (the base's own rule, exactly as `pol()` in `so_tempsweep_a2.py`), log-score `l_j = log(a_j + AT.EPS)`, the quantity the policy divides by T.
Runner-up mass ratio at the control temperature T_c = 1e-4 (the temperature of the original control 3):
  `r_{X,s} = sum_{j != argmax} exp((l_j - l_max) / T_c)`   (an exact tie at the top gives r >= 1).
The softmax mass neglected by the argmax is r/(1+r) <= r, and since 0 <= y <= 1, the per-set error of the control is `|d_s(T_c) - d_s(P1)| <= 100 (r_{A,s} + r_{B,s})` points.

## 3. The near-tie-free set S* (the exclusion rule, DERIVED)
`S* = { s : r_{A,s} <= rho and r_{B,s} <= rho }`, `rho = 5e-9`.
Derivation (tolerance-driven, not data-driven): the original control tolerance is 1e-6 points on the MEAN; a retained set then contributes at most 100 (rho + rho) = 1e-6, hence the mean error over ANY retained subset is <= 1e-6. rho is a function of the tolerance only; no label, no L and no coverage figure enters it. For information I report the equivalent top-2 margin `eps_eq = T_c ln(63 / rho) = 2.3257e-3` (a set whose top-2 log-score margin is >= eps_eq has r <= rho; the r-rule retains at least those sets). M = 64 proposals per set is asserted in the script.
Coverage floor: |S*| / 3,000 >= 0.50 at every state. Below it the control and the tracking read are reported as UNDERPOWERED, not as pass or fail.

## 4. Controls (A2R INVALID if any fails; each is computed before the tracking read is looked at)
1. T = 0.1 over ALL 3,000 sets reproduces `tempsweep_a2.json` `L_T` at every state (|diff| <= 1e-9).
2. B := A gives exactly 0.0 for every grid T on S*.
3'. (replaces control 3) T_c = 1e-4 on S*: `|L_{T_c}(S*) - L_P1(S*)| <= 1e-6` at all three states.
4. Planted-array self-test of the rule (synthetic, no banked data): a set with an exact top-2 tie and different labels is EXCLUDED from S*, the all-sets control FAILS on it and the S* control PASSES; a set with a 1e-2 log margin is RETAINED. A rule that cannot tell the two apart is invalid.

## 5. The tracking read on S* and its committed outcomes
For T in {0.1, 0.05, 0.02, 0.01}: `L_T(S*)` against `L_P1(S*)`, difference with the paired log-cluster bootstrap (10,000 resamples, seed 20260927, the log multiset of A2: `VI75 = default_rng(20260927).integers(0, 75, (10000, 75))`, estimator "another draw of logs"; the resample is of the 75 logs with the retained sets of each). ALIGNED_S*(T) iff |L_T(S*) - L_P1(S*)| <= 0.25 points at ALL THREE states (0.25 = half the g5 bound, as in A2).
Outcome, exactly one, committed in advance:
* **A2R-CONFIRMED**: controls 1, 2, 3', 4 pass, coverage floor met, ALIGNED_S*(0.02). T* = 0.02 tracks the pick on the near-tie-free sets too. The A2 PROVISIONAL tag is lifted for T* = 0.02 (the original all-sets outcome A2-ALIGNED(0.02) then stands as VALID under control 3'). The next arm's constraint temperature stays 0.02.
* **A2R-SHIFTED(T')**: controls and coverage pass, 0.02 is NOT aligned on S*, some grid T' is; T' = the largest aligned. The all-sets alignment of 0.02 was carried by near-ties; the next arm's T must be re-set to T'.
* **A2R-NONE**: controls and coverage pass, no grid T is aligned on S*. The tempered expectation does not track the pick once near-ties are removed; the next lever is the fallback named in A2 s.4 (a hinge on the pick's TTC through a straight-through or sampled estimator).
* **A2R-CONTROL-FAILS**: control 3' fails on S* although the rule guarantees <= 1e-6 there: the machinery or the bound is wrong. A2 stays PROVISIONAL; the diagnosis is the next step. No tracking verdict is drawn.
* **A2R-UNDERPOWERED**: the coverage floor is not met at some state. A2 stays PROVISIONAL.
Not a lever-effect claim (one R0 run per state, no replicate; estimator = another draw of logs only). Four metric families: only the LONGITUDINAL TTC quantity is touched here; lateral and tactical have no instrument at this tier, STRATEGIC is PAUSED (PI ruling 2026-09-19).
