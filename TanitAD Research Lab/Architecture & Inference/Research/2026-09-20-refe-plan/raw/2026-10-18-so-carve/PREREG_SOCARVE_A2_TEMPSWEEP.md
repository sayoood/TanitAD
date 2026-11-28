# PREREG_SOCARVE addendum A2: which smooth quantity tracks the bar's quantity (the pick)? The temperature of the constraint's policy, on the full RL-val R0 dumps

Written 2026-10-18 ~04:58Z, AFTER the main read and addendum A1 (`fullval_a1.json`), BEFORE any number of this addendum. sha256 in `PREREG_SOCARVE_A2_TEMPSWEEP.sha256`. Changes nothing in the main prereg or A1.

## 1. Why (Rule Zero: A1 redirected the next lever)
A1 (full RL-val, 3,000 sets / 75 logs, R0 unconstrained, live dumps, MEASURED) read, at u1000, the dual's quantity P2 (E_pi[TTC] at T = 0.1) as L = +0.223 [-0.15, +0.62] but the bar's quantity P1 (the argmax pick) as +1.000 [+0.30, +1.73]; u500 0.230 vs 0.633; u250 0.851 vs 1.233. So the SAFEOBJ constraint acts on a quantity that, on val, is far smaller than the one the bar S reads. I have seen these T = 0.1 numbers; I have NOT computed any other temperature.
The cheapest zero-training test of the pick-aligned-constraint lever named in PREREG_SOCARVE s.11: does an E_pi[TTC] at a LOWER policy temperature track the pick on val?

## 2. Quantities
Same data, labels, loaders, estimator and seed as A1 (`so_fullval_a1.py`; VI75 = default_rng(20260927).integers(0, 75, (10000, 75))). For T in {0.1, 0.05, 0.02, 0.01}: `pi_T = softmax(log(sel(sigmoid(sx)) + 1e-6) / T)`, `d_s(T) = 100 (sum_i pi_T^B y^TTC_i - sum_i pi_T^A y^TTC_i)`, `L_T = -mean d_s` on the 3,000 sets; target `L_P1` (pick). Difference `L_T - L_P1` with the paired log bootstrap interval. R0 states u250, u500, u1000.

## 3. Controls (A2 INVALID if any fails)
1. T = 0.1 reproduces A1's P2 TTC `L_va_full` at every state (|diff| <= 1e-9).
2. B := A gives exactly 0.0 for every T.
3. T = 1e-4 reproduces L_P1 (|diff| <= 1e-6; the softmax becomes the argmax; ties are the only exception and are counted).

## 4. Outcome (committed)
* **PICK-ALIGNED(T)** iff |L_T - L_P1| <= 0.25 points at ALL THREE states (0.25 = half the g5 bound).
* **A2-ALIGNED(T\*)**: T* = the largest grid T that is PICK-ALIGNED. Next lever: constrain g at T* (the constraint term's own temperature; one trainer flag with a G-DVB entry, an A40 arm, a PI decision), with the carve as its feed question.
* **A2-NONE**: no grid T is PICK-ALIGNED. Next lever: the constraint must act on the pick itself (a hinge on the pick's TTC through a straight-through or sampled estimator), not a tempered expectation.
* Gradient variance at low T is NOT tested here (a low-T policy concentrates its mass on few proposals); it is named as the cost of A2-ALIGNED.

## 5. Limits
R0 (unconstrained), one run, val side only (the harness arrays store T = 0.1 and the pick, not the logits, so no train-side L_T). Not a lever-effect claim. Four families: as the main prereg s.10.
