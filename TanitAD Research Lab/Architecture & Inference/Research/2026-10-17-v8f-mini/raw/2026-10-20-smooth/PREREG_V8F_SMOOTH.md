# PREREG_V8F_SMOOTH: is v8F-mini's path roughness free high-frequency residual noise? (2026-10-20, zero training)

**Written and hashed BEFORE any smoothed number exists.** The seed-0 roughness it tests is `H-V8F-LON-1` (`raw/2026-10-20-lon-rough/NOTE.md`):
* mean |accel| of M2b and M1 is about 2× the human's;
* paired M1 − P10 is worse on accel, yaw rate and curvature, and better on heading and cross-track.

No smoothed path, and no number derived from one, existed when this file was hashed.

## The operator (fixed now, no tuning, no selection on any eval)

* The model emits `pred = P10 + residual`: a 12-term cosine basis over 60 steps, ramped to vanish at t0 (`stack/tanitad/models/v8f_mini.py:454-457`, `:536`).
* On the eval-141 30-step paths, the residual is `r = pred − P10`, where P10 = P10-yaw-4512's eval predictions (`…/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_eval_pred6.npy`).
* **S(pred) = P10 + SG(r)**: a Savitzky–Golay filter along time, per coordinate, with **window 7 steps (1.4 s), polynomial order 2, `mode="interp"`** (`scipy.signal.savgol_filter`).
* SG(7, 2) is fixed a priori: the shortest odd window that spans the period of the basis's highest term at this grid (k = 11, about 1.1 s).
* **Dose (DESCRIPTIVE only, decides nothing):** window 11, order 2.

## Readings (eval-141, per arm M1 and M2b, per training seed k = 0, 1; paired over the 141 episodes)

Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, through `ra1_hjepa_eval.paired_family_block` and the ADE pairing. It answers *another draw of EPISODES* only.

* **D (dynamics):** S(arm) − arm on **accel MAE** and on **yaw-rate MAE** are each < 0 and separated, at BOTH seeds.
* **P (no position cost):** S(arm) − arm on **ADE@6s** has upper CI ≤ **+0.01 m** at BOTH seeds.
* **SMOOTH-PASS(arm) = D and P.** Each arm is read on its own; no arm is selected.
* **Control C (validity):**
  * S applied to the HUMAN path with the same base, S(GT) = P10 + SG(GT − P10), must stay within **0.05 m ADE@6s** of the human path.
  * If it does not, SG(7, 2) distorts a human-smooth path, and every reading is **INVALID**.
* The other families are reported beside the readings and decide nothing:
  * speed, along-track, heading, curvature, cross-track;
  * consistency (a), recomputed only if the instrument allows it on S's paths; otherwise stated as not read.

## Outcomes, committed now

* **SMOOTH-PASS** ⇒ the roughness is high-frequency residual noise that a fixed post-filter removes at no position cost.
  * S becomes a CANDIDATE post-processor. It is NOT adopted into any K-1 / WM-1 claim without its own confirm on a fresh read.
  * The training lever named is a smooth output basis: fewer high cosine terms, or a penalty on them. That is the PI's go.
* **D PASS, P FAIL** ⇒ the high-frequency content carries position signal. Smoothing is a trade, not free, and the lever is training-side only.
* **D FAIL** ⇒ the roughness is not in the residual's high band.
  * The next diagnostic is the speed-profile SHAPE per window: is the error low-frequency, i.e. the wrong decision?
  * That ties to `H-V8F-LON-1`'s longitudinal lever.
* **C FAIL** ⇒ INVALID, and nothing is read. The next arm is a narrower operator, registered anew.

## Instrument

* `code/v8f_smooth_read.py`, with `stack/tests/test_v8f_smooth_read.py`. Required identities:
  * a zero-residual arm reads exactly 0 on every delta;
  * a residual that is a quadratic in time is unchanged by SG(·, 2);
  * added white noise is reduced.
* The ground truth is rebuilt from the repo's eval inputs, and each run's saved ade6 must equal ||pred − gt|| (as `v8f_mini_full_report.py` does).
* Seed 0 is read when this file lands. Seed 1 is read when the v8F-mini chain delivers `eval_*_s1.npz`. The bars apply only at BOTH seeds; a seed-0 read alone is INTERIM.
