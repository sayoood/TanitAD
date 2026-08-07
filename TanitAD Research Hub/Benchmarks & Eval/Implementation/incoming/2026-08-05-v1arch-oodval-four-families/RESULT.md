# v1arch on PhysicalAI's own OOD-val — the first COMPLETE four-family block

**Arm:** `flagship-v1arch-v2bal-30k` (step 29999) — the **v1 ARCHITECTURE** on the v2bal 9000-clip
pool. Every `v2_*` lever in its config is `false`, so architecture is held constant and only the
data varies. That is what makes the PI's question — *the effect of more and better-distributed
data* — attributable.

**Corpus:** `physicalai-oodval-6f4b94e4c7ce-q90` — PhysicalAI-AV's **own official eval split**
(290 clips, `reasoning/ood_reasoning.parquet`), **zero overlap** with the arm's training pool, JPEG
q90 round-tripped to match the format the arm trained on. **6,382 windows · 290 episodes.**

**Estimator:** episode-cluster bootstrap over the 290 clips, n_boot 2000 (`taniteval/ci.py`).
⛔ `overlapping_holdout_se` is not used anywhere in this block.

**`_complete: true`** — all four families populated, distance-keeping included. This is the first
eval in the programme where nothing is UNAVAILABLE.

---

## The headline is NOT the ADE

| | value | 95 % CI |
|---|---|---|
| `ade_mean_4wp_m` | 0.5752 | [0.5370, 0.6142] |
| `fde_2s_m` | 1.4018 | [1.3040, 1.5010] |

⚠️ **Not comparable to any canonical-val number** until the other arms are scored on this corpus.
The OOD split is PhysicalAI's own out-of-distribution selection, not a random holdout.

⚠️ Two harnesses on the same corpus disagree slightly: `eval_flagship_v4.py`'s MODE-A canary gave
**0.5705**, this gives **0.5752** (0.8 %). Recorded, not smoothed over.

---

## ⭐ LONGITUDINAL — the arm drives systematically TOO FAST

| metric | value | reading |
|---|---|---|
| `speed_mae_mps` | 0.8491 | |
| **`speed_bias_mps`** | **+0.4840** | ⛔ **not noise — a systematic over-speed** |
| `along_mae_m` | 0.4089 [0.3820, 0.4370] | |
| `along_bias_m` | +0.1929 | ahead of the human on average |
| **`along_final_bias_m`** | **+0.9433** | ⛔ **~1 m ahead of the human at 2 s** |
| `accel_mae_mps2` | 2.3686 | |
| `ego_progress.progress_ratio_mean` | **1.0795** | drives **8 % further** than the human |
| `ego_progress.under_progress_rate` | 0.2650 | n = 5,796 |

**This is exactly what the binding rule said ADE would hide.** The lateral error is 5.5 cm; the
longitudinal error is a metre, and it has a *sign*. An arm can win ADE while setting the wrong speed,
and this one does — it is consistently early/fast rather than randomly wrong.

⭐ **And it is not a few outliers dragging a mean — it is most windows.** MEASURED over all 6,382:

| | |
|---|---|
| prediction **ahead** of the human at 2.0 s (along-track) | **71.95 %** of windows |
| median overshoot at 2.0 s | **+0.7152 m** (mean +0.9433 m) |
| predicted mean speed **faster** than the human's | **75.51 %** of windows |

A mean bias of +0.484 m/s is consistent with a symmetric distribution plus a heavy tail; a
**72 / 76 % rate** is not. The arm has a systematic over-speed prior, and any fix has to move the
central tendency, not clip a tail.

### Distance-keeping (n = 2,846 windows with a lead)

| | value |
|---|---|
| `mean_headway_min_m` | 25.5263 |
| `mean_time_gap_min_s` | 5.7552 (n = 2,517) |
| `mean_min_ttc_s` | 14.7310 (n_closing = 2,214; **632 censored at the 30 s cap**) |

⭐ **Stratified, because a pooled number averages regimes that do not resemble each other:**

| ego speed | n | lead rate | headway (m) [CI] | time-gap (s) | min-TTC (s) |
|---|---|---|---|---|---|
| 0–1 m/s | 846 | 0.50 | 19.49 [14.70, 25.16] | 26.62 | 22.28 |
| 1–3 | 610 | 0.53 | 21.27 [18.09, 24.63] | 11.60 | 17.32 |
| 3–6 | 1427 | 0.53 | 24.00 [21.51, 26.62] | 5.36 | 15.14 |
| 6–10 | 1792 | 0.48 | 28.22 [25.57, 31.03] | 3.67 | 12.16 |
| 10–15 | 867 | 0.32 | 32.08 [27.83, 36.74] | 2.73 | 10.38 |
| **15+** | 752 | 0.28 | 30.01 [24.89, 36.53] | **1.43** | 10.37 |

Time-gap falls monotonically from 26.6 s to **1.43 s** as speed rises. At 15+ m/s a 1.43 s gap is
inside the range where a human driver would be called a tailgater — and it is the regime where the
+0.484 m/s over-speed bias costs the most. **The two longitudinal findings point at the same defect.**

Window states: **LEAD 3,002 (47.0 %) · NO_LEAD 2,752 (43.1 %) · NO_LABEL 628 (9.8 %)**. NO_LABEL is
excluded, never counted as free flow.

---

### ⛔ ADE IS ~0.64-CORRELATED WITH SPEED — any ADE ranking of episodes is partly a speed ranking

MEASURED on the same 6,382 windows: **Pearson r(`v0`, per-window ADE) = 0.6408**. ADE is a
displacement over a fixed 2 s horizon, so it scales with distance travelled — ~38 m at 19 m/s
against ~4 m at 2 m/s, which makes an identical *relative* error ~9× larger in metres.

The effect is not subtle at the tails. Taking the 12 best and 12 worst episodes **by ADE**:

| set | n windows | mean `v0` | stopped (< 0.5 m/s) | mean ADE |
|---|---|---|---|---|
| best 12 | 266 | **2.34 m/s** | **32.7 %** | 0.173 m |
| corpus | 6,382 | 7.54 m/s | 11.8 % | 0.575 m |
| worst 12 | 264 | **19.19 m/s** | **0.0 %** | 1.454 m |

⇒ **A "worst episodes" list selected on ADE is close to a "fastest episodes" list.** A third of the
best-12 windows are the vehicle standing still. Nothing above is invalidated — the four families are
corpus-wide and the LONGITUDINAL over-speed finding is a *rate over all windows* (71.95 % / 75.51 %),
immune to this — but **any per-episode ADE comparison, and any ranked highlight reel, must be
speed-matched or read as a speed contrast.**

⚠️ Same family as the pooled distance-keeping number this block already stratifies: a statistic
averaged over regimes that do not resemble each other. **WORK ITEM: a speed-matched episode ranking.**

---

## LATERAL — tight, and not the problem

| metric | value | CI |
|---|---|---|
| `heading_mae_deg` | 0.8060 (pooled over steps) | 0.9169 [0.7496, 1.1229] (per-window) |
| `yaw_rate_mae_degps` | 1.6565 | |
| `curvature_mae_1pm` | 0.007615 | 0.0109 [0.0070, 0.0158] (per-window) |
| `curvature_bias_1pm` | −0.000126 | essentially unbiased |
| `cross_mae_m` | 0.0552 | [0.0500, 0.0611] |
| `cross_bias_m` | −0.0138 | |
| `cross_final_mae_m` | 0.1920 | |

⚠️ **The two heading/curvature columns are DIFFERENT QUANTITIES, not a discrepancy.** The family
scalar pools over all valid steps; the CI component is a per-window masked mean averaged over
windows, so windows contribute equally regardless of how many valid steps they have. Both are
correct; quoting one as the other is not. (`excluded_below_min_ds` = 15,639 steps.)

---

## TACTICAL — the manoeuvre head is honest, the SEAM is not

| | value |
|---|---|
| `maneuver_vs_trajectory_kappa` | **0.6033** → verdict **SUBSTANTIAL** |
| `maneuver_vs_trajectory_agreement` | 0.8881 [0.8740, 0.9021] |
| `grounded_op_rollout_ade_2s` | 0.5752 |
| `ungrounded_tactical_head_ade_2s` | 2.1948 (**3.8× worse**) |
| **`seams_beneficial_of_3`** | **0** |
| `seam_verdict` | ⛔ **"FALSIFIED — top-down conditioning does NOT measurably carry downstream performance at this checkpoint"** |

The declared manoeuvre does match the path actually driven (κ 0.60 is substantial agreement, not a
decorative label). **But none of the three hierarchy seams is beneficial.** On a clean corpus, with
no leak, the programme's central thesis does not show at this checkpoint.

---

## ⛔ STRATEGIC — the route head has NO vision-only skill, confirmed on clean data

| | value |
|---|---|
| `route_acc_nav` (**privileged**) | **1.0000** |
| `route_acc_follow` (**the deployable read**) | **0.8031** |
| `majority_straight_rate` | **0.8031** |
| `follow_pred_distribution` | **`{left: 0, straight: 1737, right: 0}`** |
| `route_acc_zeronav` | 0.2602 (below chance 0.3333) |
| `delta_nav_vs_follow` | +0.1969 [0.1538, 0.2432], separated |
| n valid | 1,737 |

**Vision-only route accuracy equals the always-predict-straight baseline to four decimal places,
because the head predicts "straight" on 1,737 of 1,737 windows and never once predicts a turn.**
`route_acc_nav = 1.0000` is an **echo**: the nav command is derived from the ego's own future, so
feeding it in hands the model the answer. This reproduces v1's known defect on a corpus that shares
no clips with training — so it is the model, not the leak.

⛔ **`nonav_route_beats_majority` is VOID BY CONSTRUCTION** (GATE_PROTOCOL §0.7) and the flag travels
in the JSON. ⛔ The **option-set** strategic path is impossible here: PhysicalAI ships **no map**
("we do not include open maps data"), so this label cannot tell whether a branch even existed.

---

## The JPEG-format confound: real in pixels, negligible in every family

Paired on the same 6,382 windows (raw uint8 vs the q90 round-trip the arm trained on; max |pixel
delta| **185**):

| metric | q90 | raw | Δ |
|---|---|---|---|
| `ade_mean_4wp_m` | 0.5752 | 0.5767 | −0.0015 |
| `fde_2s_m` | 1.4018 | 1.4052 | −0.0034 |
| `LON_speed_mae_mps` | 0.8491 | 0.8498 | −0.0007 |
| `LON_along_mae_m` | 0.4089 | 0.4101 | −0.0012 |
| `LAT_cross_mae_m` | 0.0552 | 0.0553 | −0.0001 |
| `LAT_heading_mae_deg` | 0.9169 | 0.9405 | −0.0236 |
| `LAT_curvature_mae_1pm` | 0.0109 | 0.0101 | +0.0008 |

Strategic and tactical are identical on both (`route_acc_follow` 0.8031 = majority on both;
`seams_beneficial_of_3` = 0 on both; κ 0.6033 vs 0.6058).

**q90 is the headline arm** because it is format-faithful, but no conclusion here depends on that.

---

## What this says, in order of how much it should change

1. **The defect is longitudinal and it has a sign.** +0.484 m/s, +0.94 m at 2 s, 1.08× progress, and
   a 1.43 s time-gap at highway speed. Four independent instruments, one story.
2. **The hierarchy seam is falsified at this checkpoint** — 0 of 3 beneficial, on clean data.
3. **There is no vision-only strategic route capability at all** — a constant predictor, exactly.
4. **The lateral controller is fine** and is not where the next effort belongs.
5. **Compression is not a confound** at the metric level.

## Provenance

* `raw/v1arch_oodval_q90_4fam_LEAD.json` — the complete block (`_complete: true`)
* `raw/v1arch_oodval_raw_4fam.json` — the raw-pixel control arm
* `raw/lead_oodval.pt.report.json` — lead-track coverage and drops
* tools: `taniteval/tools/eval_four_families.py`, `taniteval/tools/build_lead_block.py`
* protocol: `Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md`
* on pod4: `/workspace/evalout/` (windows dumps, lead block, logs)
