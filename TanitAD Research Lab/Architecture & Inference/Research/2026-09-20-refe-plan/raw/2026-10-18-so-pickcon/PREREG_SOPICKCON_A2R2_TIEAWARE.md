# PREREG_SOPICKCON A2R2: the tie-aware pick -- the follow-up that A2R's UNDERPOWERED outcome leaves behind

Written 2026-10-18, AFTER `a2r_neartie.json` (outcome A2R-UNDERPOWERED), BEFORE any number of this addendum. sha256 in `PREREG_SOPICKCON_A2R2_TIEAWARE.sha256`. Rule Zero (CLAUDE.md): a failed arm must leave the NEXT arm behind, not only its verdict.

## 0. What I had seen (disclosure; this addendum is POST-HOC-MOTIVATED by A2R)
* A2R, as committed: control 3' PASS at all three states (|diff| <= 2.2e-10 on S*), but coverage of S* = 0.458 / 0.449 / 0.439 < the 0.50 floor, so the registered outcome is **A2R-UNDERPOWERED**; no tracking verdict is drawn from S*.
* Why the floor was missed: near-ties are the NORM for this scorer. 1,426 / 1,433 / 1,483 of the 3,000 sets have a top-2 log-score margin below eps_eq = 2.3257e-3 in arm A (1,247 / 1,357 / 1,315 in arm B); exact top ties: 0.
* Descriptive only (not a verdict, S* is under-covered): on S* the T = 0.02 difference to the pick is -0.070 / +0.028 / -0.367 (u250 / u500 / u1000), T = 0.01: -0.048 / +0.130 / -0.193; on all sets T = 0.02 is -0.052 / +0.022 / -0.225.
So I know that excluding near-ties costs more than half the data and that the pick is ill-defined on exactly those sets. I have NOT computed anything of the quantity below.

## 1. The idea
The bar S reads the pick, `argmax`, a one-draw tie-break. On a set whose top candidates are within eps_eq of each other, WHICH candidate is the pick is decided by numerical noise, so the pick's loss there is an arbitrary draw from the tied candidates' labels. Excluding those sets (A2R) is not affordable; instead define the pick's loss in a way that does not depend on the tie-break: the **tie-aware pick** averages the label over the candidate set `Q_s(tau) = { j : l_max - l_j <= tau }` (the candidates a tie-break at resolution tau could return), with `l = log(sel_score + AT.EPS)` as in A2 / A2R:
`d_s^tau = 100 ( mean_{j in Q^B_s(tau)} y_j - mean_{j in Q^A_s(tau)} y_j )`, `L_P1^tau = -mean_s d_s^tau` over ALL 3,000 sets (same bootstrap, labels, states as A2).
tau = 0 is the ordinary pick (Q = the argmax; there is no exact tie in the banked states), so `L_P1^0 = L_P1`.

## 2. Controls (A2R2 INVALID if any fails)
1. T = 0.1 over all sets reproduces `tempsweep_a2.json` `L_T` (|diff| <= 1e-9) at every state.
2. B := A gives exactly 0.0 for L_P1^tau at every tau of the grid, and for every grid T.
3. tau = 0 reproduces `L_P1` of `tempsweep_a2.json` (|diff| <= 1e-12) at every state.
4. Planted arrays (no banked data): a 2-candidate near-tie with labels 0 / 1 reads d = 50 points of tie-aware loss at tau >= the gap and the argmax label at tau = 0; a clear set reads the same value at every tau below its margin.

## 3. Quantities and the committed outcomes
tau_0 = eps_eq = T_c ln(63 / rho) = 2.3257e-3 (fixed by A2R's derivation; NOT chosen here). Grid T in {0.1, 0.05, 0.02, 0.01} (A2's). For each T and state: `diff = L_T(all) - L_P1^{tau_0}(all)` with the paired log-cluster bootstrap of A2 (10,000, seed 20260927, 75 logs). ALIGNED_tie(T) iff |diff| <= 0.25 points at ALL THREE states.
* **A2R2-ALIGNED(T\*)**: T\* = the largest aligned grid T. If T\* = 0.02 the constraint temperature of the next arm is supported against the tie-aware pick; the margin to 0.25 at the thinnest state is reported.
* **A2R2-SHIFTED(T')**: 0.02 not aligned, T' (< 0.02 or > 0.02) is; T' = the largest aligned.
* **A2R2-NONE**: no grid T aligned; the next lever is the fallback of A2 s.4 (a hinge on the pick's TTC through a straight-through or sampled estimator).
Descriptive, NOT part of the verdict: the sensitivity of `L_P1^tau` to tau in {0, 1e-4, 1e-3, tau_0, 1e-2} (how much of the bar's own quantity is tie-break noise), and the alignment of 0.02 and 0.01 at each of those tau.
Not a lever-effect claim (R0, one run per state; estimator = another draw of logs). LONGITUDINAL TTC only; lateral and tactical not computable at this tier; STRATEGIC PAUSED (PI ruling 2026-09-19).
