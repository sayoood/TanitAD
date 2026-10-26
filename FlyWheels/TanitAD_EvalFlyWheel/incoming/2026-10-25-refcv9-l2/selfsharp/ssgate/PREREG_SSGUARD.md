# PREREG_SSGUARD -- keep the native PATH, take the sharpened SPEED PROFILE (Master Mind, 2026-10-26 ~03:09 CET)

Sealed BEFORE any guarded-arm number exists. 0 GPU, 0 training: composed per window from the two banked rolls.

## 0. What is known
* SSGATE (`PREREG_SSGATE.md` 6e5d4235, `result/SSGATE_RESULT.json`) read **FAIL (G_strict)**: P1 MD 0.4695 -> 0.4319
  (-0.0376 [-0.0704, -0.0080], separated), P2 holds; G fails on **LAT_heading_mae_deg** +0.0737 [+0.0300, +0.1266] and
  **LAT_yaw_rate_mae_radps_valid** -- both rows the privileged ORACLE force ALSO degrades (the channel). Cross-track is no longer
  worse; LON rows improve. Per margin decile the heading cost is present in EVERY decile (+0.04 .. +0.62 deg): it is not a
  wrong-choice cost, it rides on the sharpen itself.
* Mechanism read: forcing the lon class changes the decoded PATH as well as the speed profile (lat and lon are coupled in the
  decode -- the programme's known "one softmax mixes lat+lon" defect). The committed SSGATE FAIL_G lever is "a softer sharpen
  with a LATERAL GUARD". The guard alone is 0 GPU, so it runs first.
* NOT computed, therefore unseen: any guarded plan.

## 1. Arms
* **GUARD-ALL (primary):** for every window, `P = at_progress(P_base, arc(P_ss))` -- the native (BASE-os) plan's polyline, sampled at
  the SELFSHARP plan's per-instant cumulative arc length (`RT.polyline_geometry` / `RT.eval_polyline`, linear extrapolation along the
  last segment beyond the end -- the shipped RETIME stage's own machinery). Deployable: two decodes of the same model (native and
  self-sharpened lon logits), no label. Then the shipped stage, as every arm.
* **GUARD-CF (secondary, read ONLY if GUARD-ALL fails a bar):** GUARD on windows with margin >= tau_k, BASE-os elsewhere, tau_k by
  SSGATE's exact cross-fit rule (same folds, candidates, admissibility, pick).

## 2. Population, estimator, tier
As SSGATE: S6 compliant, stage on, paired episode-cluster bootstrap over EPISODES B 2000 seed 0 (`l2_score.compare`); one training
seed, one inference seed (0); T1 self-action open loop on logged eval139 frames; S2 rows PENDING.

## 3. Controls (a failed control voids the bars)
* **K-ID:** `at_progress(P_base, arc(P_base))` reproduces BASE-os's MD / FD (0.46948 / 0.13955) to 5 dp; **K-SSID:**
  `at_progress(P_ss, arc(P_ss))` reproduces SELFSHARP's (0.38967 / 0.15032).
* **K-ALL / K-NONE / K-IND** as SSGATE.

## 4. Bars and outcomes (on GUARD-ALL vs BASE-os; then GUARD-CF only if GUARD-ALL fails)
* **P1** MD separated BELOW. **P2** FD NOT separated above. **G-strict** no S6 row (`PR.METRICS` + `ade_m`) separated worse.
* **PASS (either arm)** -> the lat/lon-FACTORED decode (native path x sharpened progress) is the lever. Next: the SELFSHARP roll at
  inference seed 1 and the re-read at seed 1 (replicate against the inference floor), then refcv9's decode design (lon class drives
  the speed profile only) and the PI for SHIP-R8.
* **FAIL P1** -> the gain lived in the path, not the speed profile: the next lever is the temperature roll (softer sharpen),
  pre-registered first.
* **FAIL G** -> the cost is not the path change either: name the rows; the temperature roll is the next lever.
* Descriptive beside: GUARD-ALL vs SELFSHARP on the LON rows (how much of the speed-profile gain the path swap keeps).
