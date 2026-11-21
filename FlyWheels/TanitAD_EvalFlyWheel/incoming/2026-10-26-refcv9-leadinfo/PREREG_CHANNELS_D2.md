# PREREG_CHANNELS_D2 -- curve geometry and crosswalk-ahead as decel information channels (DIAGNOSTIC, Master Mind, 2026-10-26 ~08:00 CET)

Sealed before any feature, fit or AUC of this diagnostic exists. Same status as PREREG_LEADINFO_D (8f02a6dd): a DIAGNOSTIC that decides where
the next information lever goes; NO bar for any deliverable; nothing from it fits, tunes or chooses anything later scored on eval139.

## 0. Why (disclosed)
LEADINFO_D read **LEAD-INFO ABSENT** (`result/LEADINFO_D.json` 4cefbf83): dAUC(B - A) +0.0293 [-0.0027, +0.0618]; lead presence 40.0 % in
GT-decel vs 38.9 % in GT-cruise windows. Its committed next channel, the traffic-light state, is BLOCKED at window resolution: the only GT
light labels are CLIP-level CoT goals (eval: 28 of 147 clips in `s2_labels_v8_eval.jsonl.gz`); no window-level light state exists. The next
untested channels are (i) the curve the planner's own path is about to take and (ii) a crosswalk ahead in the map.

## 1. Data, target, estimator -- IDENTICAL to PREREG_LEADINFO_D s1 and s3
(S6 compliant eval139 windows, `l2_common.lon_indicators` populations, y = 1[GT decel], feature set A = LON6 softmax + v0, logistic L2 C = 1,
5-fold episode-grouped cross-fit fold seed 0, AUC of out-of-fold P(decel), episode-RESAMPLING bootstrap B 2000 seed 0 with OOF predictions
fixed.) Coverage per channel is reported; windows a channel cannot cover are dropped from THAT channel's comparison only, and its A is re-fit
on the same windows.

## 2. Channels (each = A + its block; each with its own shuffled control = the block permuted across EPISODES, seed 0)
* **CURVE (deployable: the planner's own native S6 plan, `P_os8`):** max |curvature| over the plan polyline (three-point, segments < 0.5 m
  merged), the lateral-acceleration demand v0^2 * kappa_max, the curve-limited speed sqrt(2.0 / kappa_max) and the required decel
  max(0, v0 - that speed), the end-minus-start heading change, and the along-path distance to the max-curvature point.
* **XWALK-OWN (deployable: refcv8's own map, MAPSEL rasters `F:/refcv9_mapsel/rast_s0`, `code` plane under the declared prior_corrected
  rule):** share of crosswalk cells, and the nearest crosswalk distance, inside the corridor |y| <= 1.8 m, 4 <= x <= 40 m (the never-supervised
  x < 4 m zone excluded); same for the drivable-edge class.
* **XWALK-GT (upper bound: the `gt` plane of the same rasters):** the same features from the GT map.

## 3. Reading per channel (fixed now; the control rule corrects LEADINFO_D's, disclosed)
* **PRESENT** iff dAUC(channel - A) >= +0.03 with CI lo > 0 AND the shuffled control's dAUC point estimate <= +0.01. (LEADINFO_D required the
  shuffled CI to contain 0; a shuffled block can HURT out-of-fold AUC by a small overfitting penalty -- it read -0.0060 [-0.0110, -0.0014] --
  which says nothing about leakage. A shuffled block that HELPS would; that is what the control is for.)
* Next: CURVE PRESENT -> a deployable curve feature into the fresh-split gate (HSGATE-v2 / SELCOR-v2), pre-registered first, 0 GPU.
  XWALK-GT PRESENT but XWALK-OWN not -> the map fine-tune (PREREG_MAPFT) gains a second consumer; read XWALK-OWN again on MAPFT's map.
  All ABSENT -> the window-level decel decision is not recoverable from these explicit channels: the refcv9 lever is TRAINING the lon decision
  (a clean-split lon head with richer supervision), named in PLAN_REFCV9 with this evidence.
* Tier: DIAGNOSTIC on logged eval139 labels and the planner's own logged outputs; GT channels are UPPER BOUNDS, never inference inputs.
