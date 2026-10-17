# PREREG_V8F_MINI — the PI's three-layer v8F on frozen DINOv3-L features (track M), and data vs epochs

**Committed:** 2026-10-17 (Europe/Berlin), BEFORE any data. It is governed by `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13 (the PI rulings of 2026-10-17) and extends `PREREG_HJ_STAGE0.md`. RULE ZERO applies: a missed bar is reported FAILED and the named next lever runs.

## Common settings

* **Grid:** the refav1 trunk-probe grid: 141 v7.2 EVAL episodes, **2,399 windows**, horizons 2 s and 6 s, T1-equivalent open loop. P10's banked numbers live on this grid (2 s 0.2229 / 6 s 2.9257 m).
* **Features:** the pooled frozen DINOv3 ViT-L/16 cache from S4 (4×10 primary, 8×20 secondary). TRAIN clips are disjoint from the eval clips by id **and by recording session**.
* **Estimator:** the paired episode-cluster bootstrap, B = 2,000. Every row names the variance it answers.
* **Seeds:** **training seeds 0 and 1** for every arm. The seed floor is F = \|arm_s0 − arm_s1\|.
* **Inputs, never labels:** t0 ego dynamics (the P10 features), max target speed (v8 sidecar), and nav command + distance (v9).
* **Metric families:**
  * the four families in full; **STRATEGIC is un-paused** (PI ruling of 2026-10-17);
  * plus **CONSISTENCY**;
  * plus the "acts" check: mean |a| and the zero-accel fraction.

## Arms

| arm | what |
|---|---|
| `M1` | S (1 Hz) → T (2 Hz) → O (10 Hz) on the scene features. O emits a 6 s residual on P10. Sample-and-hold between layers; scheduled sampling p_GT 1.0 → 0.2; consistency penalties on |
| `M1-navonly` | M1 with **no scene features** in S and T. This is the strategic and tactical "vision helps" control, and the PI's literal nav + distance-only strategic layer |
| `M1-navshuf` | M1 evaluated with the nav command + distance taken from another episode (fixed permutation, seed 0). This is the nav-use control |
| `M1-nocons` | M1 with the consistency penalties at 0. It isolates what they buy |
| `M2a` / `M2b` | M1 + an H-JEPA WM auxiliary (z¹ / z² / z³ on O / T / S cycles), **without / with IDM** (β 0.08, γ 100) |

## Hypotheses and bars

**H-MINI-K-1.** `best(M1, M2a, M2b) − P10`: ADE at 6 s < 0, separated, at both seeds, with \|Δ\| > 2F. Also read against `damp50` and `kd_x`. Event strata are reported: GT turn onset, stop, lead braking.
* PASS ⇒ AR-1, trajectory part.
* FAIL ⇒ plan §13.6, row 1.

**H-MINI-CONS-1.** CONSISTENCY(a) of the best arm ≤ P10's own rate on the same windows, separated; self-consistency (b) ≥ 0.90.

Violation rules, frozen now:
* **SPEED:** the planned max speed is more than 0.5 m/s above min(limit, SPEED_BAND v_hi).
* **STOP:** the planned speed is more than 1.0 m/s at the stop distance, or the plan passes the stop point by more than 2 m.
* **TURN:** the sign of the planned heading change differs from the goal side, while a TURN goal is active.
* **LEAD:** the minimum planned time gap is more than 0.2 s below the constraint.

`M1-nocons − M1` is reported; it is expected to be worse.

**H-MINI-STRAT-1.** Strategic checkpoint position error at 30 / 50 / 80 m.
* **Bar:**
  * M1 beats the **nav-only kinematic baseline**, separated at 50 m and 80 m;
  * `M1-navshuf` is **worse** than M1, separated (nav is used);
  * M1 beats `M1-navonly` (the scene adds strategic value).
* **The nav-only kinematic baseline:**
  * extrapolate the P10 6 s path to 30 / 50 / 80 m of arc;
  * when nav announces a TURN at distance d, go straight to d, then follow an R = 12 m arc to the nav side.
* **If M1 does not beat `M1-navonly`:** ship the nav-only strategic layer (the PI's literal spec), and record that the scene adds nothing at the strategic level at this scale.

**H-MINI-TAC-1.** v9 tactical goal **macro-F1** over the labelled cells, with IGNORE masked.
* **Bar:** M1 beats the majority class and `M1-navonly`, separated.
* **Reporting:** the per-class confusion matrix is reported. So is TURN timing error against the label's onset.

**H-DATA-SCALE-1** (plan D2). M1's recipe (or M2's, if it wins H-MINI-K-1) at three settings:
* **(a)** 25 / 50 / 100 % of the TRAIN clips, at matched optimiser steps S, where S = one epoch at 100 %;
* **(b)** 100 % at 2S steps;
* **(c)** 100 % at 4S steps.

Every setting is run at 2 seeds.

* **Two gains are compared:**
  * Δ_data = ADE₆(50 %, S) − ADE₆(100 %, S): doubling unique data at fixed compute;
  * Δ_epoch = ADE₆(100 %, S) − ADE₆(100 %, 2S): doubling compute at fixed data.
* **The outcomes:**

| result | meaning ⇒ next |
|---|---|
| Δ_data > Δ_epoch + 2F | **data is the lever** ⇒ D3 (the egomotion-mined extension) goes first |
| Δ_epoch > Δ_data + 2F | **epochs are the lever** ⇒ train longer on the 26 h first; D3 is deferred |
| both within 2F of 0 | neither binds at this scale ⇒ the architecture and the trunk (LoRA) are the lever |

* **Reporting:** the 25 % point and the 4S point are read for curvature and reported, but they do not decide.

## Amendment 1 (2026-10-17, before any M-arm data): the P10 reference is refit in the correct steer unit

* The banked refav1-grid P10 (2 s 0.2229 / 6 s 2.9257 m, "Common settings" above) was fit as a residual on a `kd_x` built with the steer angle integrated as a curvature (RETRACTION_LOG R46).
* On that grid the correctly-integrated echo `ha0_ext` reads 2.838 m, which already beats it.
* ⇒ **M1's residual base, and the P10 every H-MINI bar is read against, is P10 refit on the correct-unit `kd_x`** (`taniteval/taniteval/floors_kin.py`, `action_units="steer"`). Its 2 s / 6 s numbers are reported beside the old ones.
* **`ha0_ext` (correct unit) is added as a reported floor.**
* This makes every bar stricter, never looser. No arm has been trained.

## Amendment 2 (2026-10-17, evening, before any M-arm data): reference numbers, label release, and two rulings

Source: the M1-build package `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/` (`RESULT.md`, `raw/p10/p10_fit.json`, `LABEL_JOIN_REPORT.md`). No arm has been trained.

1. **The P10 reference, refit in the correct steer unit** (Amendment 1), on the 2,399-window grid:
   * 6 s ADE **2.5744 m** [2.327, 2.829];
   * 2 s ADE (first 10 knots) **0.2187 m**;
   * 6 s FDE 8.024 m.
   * The same script in the legacy unit gives 2.9281 m, against the banked 2.9257, so the pipeline reproduces.
   * The correct-unit `kd_x` reads 2.9396 m. S1 independently read 2.9396 for the same floor (`…/2026-10-17-p10-floors-refcv/raw/refav1_units_check.json`).
   * ⇒ **H-MINI-K-1 is read against 2.5744 m, not 2.9257.** The bar is harder by 0.35 m.
   * Artifacts: `raw/p10/p10_prior.pt` (the residual base) and `raw/p10/p10_eval_pred6.npy` (the comparison).
2. **Label release.** Train on the v9 **4,572-clip** release (md5 `83e30bd3…`), which labels 100 % of train windows. The f63ece41 release leaves 4.4 % unlabelled.
   * Eval: **139 of 141 episodes** carry v9 labels (2,365 of 2,399 windows).
   * TACTICAL, STRATEGIC and CONSISTENCY are read on those 139. The trajectory bar (K-1) is read on all 141.
   * The 2 missing episodes are a named gap, not a silent drop.
3. **CONSISTENCY floor, reported beside every reading.** The frozen SPEED rule flags **6.3 %** of windows on the human's own path, because the max-speed sidecar holds one limit per clip. The GT-plan control reads **7.9 %** overall. P10 reads **32.4 % [27.6, 36.9]**.
   * The rules and the bar are **unchanged**: the best arm must read ≤ P10's rate, separated, and self-consistency must be ≥ 0.90.
   * The GT control is the rule's own noise floor, so no arm is expected to go below it.
4. **TAC-1 ruling, made before any data:**
   * **primary = goal-token macro-F1** over the v9 tactical goal vocabulary, with IGNORE masked. This is the PI's "tactical goals" as a layer output;
   * **reported, not decisive:** lateral and longitudinal macro-F1, plus the confusion matrices.
5. **Strategic floor numbers.** The nav-only kinematic checkpoint baseline reads **2.01 / 4.82 / 11.15 m** at 30 / 50 / 80 m of arc.
