# refcv8 heading drift -- localisation VERDICT (diagnosis only; CPU; existing dumps)

Everything below is MEASURED from the battery panels in this package's trees (paths in the JSON `trees`), tier **T1 open loop on logged eval139 frames (not driving performance)**.
Heading = `LAT_heading_mae_deg` (degrees, per window, mean over plan/GT-valid chord segments). Delta = new minus ref, **paired episode-cluster bootstrap** (`taniteval/ci.py`, B 2000, seed 0, 95 %), windows finite on both sides, 139 episodes. Unless stated, the number is the **2-inference-seed mean of the per-window metric** (S2 = the 0-2 s surface, 4,510 windows with a heading). CONTROL: my overall S2/S6 heading cells reproduce the battery's own `PAIRED_READ_step10500ema_unc.json` (n, delta, CI) at both seeds: `control_vs_battery.all_reproduced = true`.
The estimator answers "another draw of EPISODES". It does **not** answer "another training run" (CLAUDE.md H-ESTIM-SEED-1: one training run, no replicate arm); the s0-vs-s1 floor below is the only inference-run evidence. Bins are fixed constants from GT / measured v0 / the clip's nav token only (never tuned on a delta); 14 bins are tested per comparison, so a Bonferroni flag is reported beside each 95 % interval.

## One-paragraph answer

The drift is **not uniform and not a noise-level wobble: it is a progressive under-rotation in tight turns at junction speeds.** It lives in GT sharp-turn windows (|yaw rate| >= 0.15 rad/s: 8.1 % of windows, 42 episodes) and at v0 5-10 m/s (30.8 % of windows, 84 episodes); straight driving, gentle curves, high speed and the nav class do not carry it. In the sharp-turn windows the plan's chord heading lags the GT turn **multiplicatively** (refcv7 -9.2 %, EMA-5k -13.0 %, EMA-10.5k -15.5 % of the GT heading at the last 2-s chord), and the lag is **worse in the raw weights** (RAW-5k -23.4 %) -- the EMA drift is the scored average catching up with a raw path that is already worse than refcv7. The 2-s goal-error drift is **not** in the same windows.

## Q1 -- where is the gap (S2, heading MAE deg; delta [95 % CI]; n windows / episodes)

| bin | share of windows | EMA-10.5k - refcv7 | EMA-10.5k - EMA-5k | EMA-5k - refcv7 | RAW-5k - refcv7 |
|---|---|---|---|---|---|
| ALL | 100 % | +0.0464 [0.0007, 0.0919] SEP (4510/139) | +0.0320 [-0.0004, 0.0618] | +0.0145 [-0.0144, 0.0436] | +0.3144 [0.2188, 0.4222] SEP |
| GT sharp turn >= 0.15 rad/s | 8.1 % (367/42) | **+0.3379 [0.0225, 0.7386] SEP**, 59 % of the delta | **+0.2618 [0.0926, 0.5038] SEP**, 67 % (Bonf-SEP) | +0.0761 [-0.1596, 0.3724] | +1.1086 [0.5223, 1.8133] SEP |
| GT straight < 0.02 | 64 % | +0.0287 [-0.0053, 0.0648] | +0.0155 [-0.0056, 0.0392] | +0.0133 [-0.0083, 0.0393] | +0.2286 [0.1524, 0.3151] SEP |
| GT gentle / medium turn | 16 % / 11 % | -0.0199 / +0.0350, both n.s. | n.s. | n.s. | -- |
| v0 5-10 m/s | 30.8 % (1390/84) | **+0.1692 [0.0780, 0.2667] SEP**, 112 % of the delta (Bonf-SEP) | **+0.1166 [0.0488, 0.1898] SEP**, 112 % (Bonf-SEP) | +0.0526 [0.0049, 0.1051] SEP | +0.7232 [0.5236, 0.9449] SEP |
| v0 15-25 m/s | 19 % (856/35) | -0.0404 [-0.0783, -0.0071] SEP (better) | -0.0152 [-0.0289, -0.0039] SEP (better) | n.s. | -0.0045 n.s. |
| v0 < 2, 2-5, 10-15, >= 25 | -- | all n.s. | v0 <2: +0.1120 [0.0186, 0.2277] SEP, the rest n.s. | n.s. | -- |
| nav follow / left / right | 64 / 9 / 27 % | +0.0496 [0.0071, 0.0963] SEP / +0.0173 n.s. / +0.0484 n.s. (excess-vs-rest ~0 for all three) | follow +0.0424 SEP | n.s. | -- |

* **Concentrated, not uniform.** The overall +0.046 is **3.3 %** of refcv7's 1.41 deg. Sharp turns carry 59 % (67 % of the 5k -> 10.5k step) from 8 % of the windows; v0 5-10 m/s carries >100 %. Excess-vs-rest, sharp turn vs refcv7 is *not* Bonferroni-separated ([-0.107, 0.938]); for 10.5k vs 5k it is (+0.2501 [0.0266, 0.6211]); v0 5-10 is Bonferroni-separated in both.
* **Bulk, not tail, inside the sharp bin; tail-carried outside it.** Sharp bin: median per-window delta +0.120, 10 %-trimmed mean +0.158, 56 % of windows worse (vs 10.5k - 5k: median +0.073, 62 % worse). Over ALL windows the median is +0.004 and the trimmed mean +0.007: the overall mean is carried by a heavy tail (top 1 % of windows sum to 304 deg against a total of 209).
* **By episode.** 60 % of episodes have a positive mean delta; the 5 largest carry 68 % of the S2 sum (ep013, ep111, ep012, ep023, ep075: all curved-path episodes, mean v0 6-11 m/s, 3 of 5 nav=follow); without them the overall delta is +0.015. S6 (6 s) is more extreme: top 5 episodes carry **92 %** of the sum (ep111 27 %, ep080 23 %, ep013 17 %, ep085 15 %, ep012 10 %) -- S6 is a weaker surface for this question.
* **By horizon (turn-aligned, curved windows).** The lag grows along the 2-s horizon: sharp windows, mean turn-aligned signed heading error per chord (deg): seg1 -0.36 -> seg4 -3.36 (EMA-10.5k); refcv7 +0.01 -> -1.84; EMA-5k -0.27 -> -2.74. At every segment the order is refcv7 > EMA-5k > EMA-10.5k (monotone).
* **S6 (0-6 s)** agrees in direction and location (overall +0.145 [-0.034, 0.319] n.s.; sharp +1.409 [0.190, 2.885] SEP; v0 5-10 +0.404 [0.067, 0.813] SEP; both seeds same sign), but see the 92 % episode concentration.

## Q2 -- bias or spread? (sharp-turn windows, S2; turn-aligned sign: negative = plan rotates LESS into the turn than GT)

* **Systematic under-rotation that GROWS with training.** Pooled turn-aligned bias: refcv7 -1.054 deg, EMA-5k -1.620, EMA-10.5k -1.951; shift vs refcv7 -0.90 (window-level -0.891 [-1.405, -0.490] SEP); 10.5k vs 5k -0.331 pooled (window-level -0.335 [-0.643, -0.118] SEP).
* **It is multiplicative (curvature shrinkage), not a constant offset.** Through-origin slope of the heading error on the GT chord heading (last 2-s chord, mean GT heading 25.4 deg, 364 windows): refcv7 **-0.092** [-0.177, -0.026], EMA-5k **-0.130** [-0.232, -0.054], EMA-10.5k **-0.155** [-0.257, -0.076]; slope minus refcv7: -0.038 [-0.065, -0.018] SEP and **-0.063 [-0.098, -0.034] SEP**. (S6, last chord: -0.257 / -0.256 / -0.289, difference n.s.: every model under-rotates ~25 % at 6 s, so the drift is a 2-s-surface phenomenon here.)
* **Spread rises too.** Sharp bin, vs refcv7: delta MSE +9.75 deg^2 = +2.70 (bias^2) + 7.05 (spread; SD 5.96 vs 5.33); removing the mean shift leaves 85 % of the MAE rise. Part of this "spread" is the same multiplicative lag acting on a heterogeneous GT heading, so the two are not separable with this data.
* **Straight windows:** no systematic signed bias (pooled shift -0.10; window-level -0.143 [-0.352, -0.029], i.e. not toward a turn direction), heading rise n.s. Nav classes: no differential. Nav is still used: benefit of nav for heading (MAE(os_navzero) - MAE(os), seed 0 only; seed 1 has no nav arms) is unchanged 5k -> 10.5k (overall +0.0048 [-0.057, 0.065]; sharp -0.190 [-0.508, 0.083]).

## Q3 -- does the 2-s goal-error drift line up?  Only partly testable; it does NOT line up with these windows

* The dumps carry **no goal vector and no goal label**: only `goal_dist_sel` = distance from the *selected plan's 2-s point* to the model's `g_tac` 2-s goal point (`refc_v3.py:2119-2131`, SEAM_SLOT 3 = 2 s). The training metric `goal2s_err_m` is `g_tac(2 s)` against the `goal_tac` label (`refc_v3_train.py:4783-4787`). **Missing for the real test:** per-window `goal_point_tac` (the vector) and `goal_tac` (label). The proxy equals the training quantity only to within the plan's own 2-s error (FDE ~0.6 m).
* Training EMA series (`ema_eval_goal2s_err_m`, 8 fixed batches of the same 139 episodes): 3.549 (3.0k, min) -> 3.599 (5k) -> 3.727 (10.5k) -> 3.771 (14.5k) -> 3.774 (15.0k) -> 3.759 (16.0k); refcv7 final raw 3.5875. (The brief's 3.61 -> 3.70 -> 3.77 is within this envelope; I read 3.599 at 5k.)
* Dump proxy, same direction: EMA-5k 3.754 -> EMA-10.5k 3.805 m (seed 0: +0.0504 [-0.0065, 0.1123] n.s.); RAW-5k 4.078.
* **Not the same windows:** the proxy rises in the *straight* bin (+0.10 m, 4.13 -> 4.23) and *falls* slightly in sharp turns (-0.04 m); per-window corr(delta heading, delta goal-dist) is -0.057 / -0.006 (S2 seeds 0/1) and -0.014 / +0.012 (S6).
* Selection: 31 % of windows change the selected anchor 5k -> 10.5k; flipped windows carry most of the per-window heading change (+0.078 [-0.015, 0.165] vs unflipped +0.008) but not separated; the selector agrees with the oracle anchor **more** often at 10.5k (+0.028 [0.012, 0.043] SEP) while heading gets worse -- selection quality is not the lever. Windows where 5k found the oracle anchor and 10.5k did not: heading +0.20 (sharp: +1.30, n 25, borderline).

## Q4 -- inference floor, and what clears it (S2; |delta| / max inference floor; 2x rule)

* Inference floor |seed0 - seed1| of the mean heading MAE: overall EMA-10.5k 0.0031, EMA-5k 0.0080, refcv7 0.0110; sharp bin 0.0286 / 0.0710 / 0.1080; v0 5-10 0.0111 / 0.0164 / 0.0006.
* EMA-10.5k vs refcv7: overall 4.2x, **sharp 3.1x** (pooled separated, but separated at seed 1 only), **v0 5-10 15.2x (separated at both seeds)**, v0 15-25 17.8x (better), nav=follow 13.9x (both seeds). EMA-10.5k vs EMA-5k: **sharp 3.7x (both seeds), v0 5-10 7.1x (both seeds)**; overall 4.0x but not separated.
* So the drift clears the *inference* floor in the sharp-turn and v0 5-10 bins. **It cannot be shown to clear the TRAINING-run floor**: one training run, no replicate arm, so a training-seed effect of the same size cannot be excluded (H-ESTIM-SEED-1). What argues for a systematic origin is an ordered four-point trajectory in the same bins (refcv7 < EMA-5k < EMA-10.5k in under-rotation at every horizon segment) and the larger same-signature effect in the raw weights -- indicative, not proof.

## HYPOTHESIS (mechanism) -- H-HEAD-WASHOUT-1

**The scored EMA is converging, as the warm-start memory washes out of it, onto a raw weight path that under-rotates in tight turns far more than refcv7.** Evidence (MEASURED): RAW-5k (`step5000_unc`) is already **+0.314 [0.219, 0.422]** worse than refcv7 on heading (+23 %), sharp +1.109, v0 5-10 +0.723 (same bins as the EMA drift, 21x the EMA-5k delta at the same step), shrinkage slope -0.234 vs EMA-5k -0.128 vs refcv7 -0.091 (raw-vs-EMA run); the goal proxy shows the same pattern (RAW-5k 4.078 vs EMA-5k 3.754) and selection is less oracle-aligned (-0.067). With EMA decay 0.9998 (HANDOFF; time constant ~5,000 steps) and **assuming the EMA starts at the warm-start weights** (not checked here), the warm-start share is ~0.37 at step 5,000, ~0.12 at 10,500, ~0.05 at 15,000 -- the EMA drift grows exactly while that share decays. Mechanism of the raw degradation itself is unexplained here (candidates, untested: high-LR re-optimisation of the warm-started trunk under the new per-frame nav / road-type speed inputs; a position-only trajectory loss that prices a 3-4 deg late-chord heading error at centimetres, so curvature shrinkage is nearly free in the loss).
* Why this matters for the PI ruling: the EMA-scored bars are currently *flattered* by the warm-start share; the same amount of raw-path degradation will surface in the EMA as that share goes to ~0 unless the raw path recovers (LR decay).
* **Falsifier / prediction (washout-only model, crude; linear mixing is an assumption).** Using EMA-10.5k's slope -0.155 and share 0.12, the implied path slope is ~-0.164 and the EMA-15k sharp last-chord slope should be ~**-0.160** (slope minus refcv7 ~ **-0.07**, i.e. almost flat vs 10.5k), the sharp-bin heading delta vs refcv7 ~ +0.37 (S2, 2-seed). **H is refuted as the whole story** if the 15k slope-minus-refcv7 CI lies entirely below -0.10 (the raw path itself keeps degrading) or if the drift appears in bins other than sharp-turn / v0 5-10; **H is supported** if it stays near -0.07 +- 0.03 in the same bins.
* **Cheapest experiment (CPU, no training):** re-run this script on the step-15,000 EMA dumps when they land (command below; ~3.5 min, ~0.5 GB). The earlier points are repeatable arguments, so the table shows the whole refcv7 -> 5k -> 10.5k -> 15k trajectory. **One more GPU-free-after-the-fact discriminator:** the trainer also writes the raw `ckpt_15000.pt`; a raw-15k pass (same battery) separates "raw path degrades further" from "raw path recovers with LR decay" -- recommended, needs the GPU slot, not run here.

```
python raw/heading_drift/heading_drift.py --new-root E:/refcv8_eval_kit/battery/step15000ema_unc --label-new EMA-15k \
  --earlier EMA-10.5k=E:/refcv8_eval_kit/battery/step10500ema_unc EMA-5k=C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc \
  --out-dir raw/heading_drift --tag HEADING_DRIFT_step15000ema     # OMP_NUM_THREADS=4; interpreter C:/Users/Admin/venvs/tanitad/Scripts/python.exe
```

## Limits (read before quoting)

Single training run, 2 inference seeds, no replicate arm; the bootstrap holds the models fixed. 14 bins x several comparisons: some 95 %-separated bins are expected by chance (Bonferroni flags are in the tables). Sharp-turn vs refcv7 is separated pooled and at seed 1 only. The washout reading assumes (a) EMA initialised at the warm-start weights and (b) approximately linear mixing -- both unverified. goal proxy is not the training metric. S6 is dominated by 5 episodes. `nav` arms exist for seed 0 only. No number here is a bar verdict.

---

(Generated tables follow: script output, `heading_drift.py`.)


# refcv8 heading-drift localisation: EMA-10.5k vs refcv7-50,400 and vs earlier refcv8 points

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: paired episode-cluster bootstrap (taniteval/ci.py, B 2000, seed 0), models FIXED; answers 'another draw of EPISODES' only -- NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence
* generated 2026-10-17T01:19:23; trees: `EMA-10.5k` = E:/refcv8_eval_kit/battery/step10500ema_unc; `EMA-5k` = C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc
* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s [0.02, 0.06, 0.15]; speed = v0 m/s [2.0, 5.0, 10.0, 15.0, 25.0]
* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **True**

## S2 ([0.5, 1.0, 1.5, 2.0] s) -- 4754 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2908, "gentle 0.02-0.06": 723, "turn 0.06-0.15": 513, "sharp >=0.15": 368, "undef (GT crawling)": 242}, "speed": {"v0 <2": 465, "v0 2-5": 516, "v0 5-10": 1390, "v0 10-15": 1192, "v0 15-25": 856, "v0 >=25": 335}, "nav": {"nav=follow": 3011, "nav=left": 445, "nav=right": 1298}}

### S2 -- EMA-10.5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4510 | 139 | 1.4542 | 1.4077 | 0.0464 [0.0007, 0.0919] | **SEP** |  |
| turn: straight <0.02 | 2898 | 132 | 1.1884 | 1.1597 | 0.0287 [-0.0053, 0.0648] |  | 0.643 / 0.398 / -0.0495 [-0.2079, 0.0914] |
| turn: gentle 0.02-0.06 | 720 | 98 | 0.9896 | 1.0094 | -0.0199 [-0.1232, 0.0677] |  | 0.160 / -0.068 / -0.0789 [-0.2462, 0.0545] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9557 | 1.9207 | 0.0350 [-0.0851, 0.1512] |  | 0.113 / 0.085 / -0.0129 [-0.2081, 0.1573] |
| turn: sharp >=0.15 | 367 | 42 | 3.3249 | 2.9870 | 0.3379 [0.0225, 0.7386] | **SEP** | 0.081 / 0.593 / 0.3174 [-0.1074, 0.9382] |
| turn: undef (GT crawling) (low n) | 17 | 13 | 11.0545 | 11.1380 | -0.0835 [-0.2200, 0.0355] |  | 0.004 / -0.007 / -0.1304 [-0.3509, 0.0868] |
| speed: v0 <2 | 221 | 37 | 13.3571 | 13.2326 | 0.1246 [-0.0429, 0.3078] |  | 0.049 / 0.132 / 0.0822 [-0.1801, 0.4226] |
| speed: v0 2-5 | 516 | 58 | 1.8093 | 1.8694 | -0.0601 [-0.3546, 0.1978] |  | 0.114 / -0.148 / -0.1203 [-0.5812, 0.2910] |
| speed: v0 5-10 | 1390 | 84 | 1.1784 | 1.0092 | 0.1692 [0.0780, 0.2667] | **SEP** | 0.308 / 1.124 / 0.1775 [0.0250, 0.3232] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.5878 | 0.5744 | 0.0133 [-0.0258, 0.0479] |  | 0.264 / 0.076 / -0.0449 [-0.1492, 0.0684] |
| speed: v0 15-25 | 856 | 35 | 0.3330 | 0.3734 | -0.0404 [-0.0783, -0.0071] | **SEP** | 0.190 / -0.165 / -0.1072 [-0.1937, -0.0138] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1466 | 0.1576 | -0.0110 [-0.0406, 0.0079] |  | 0.074 / -0.018 / -0.0620 [-0.1392, 0.0225] |
| nav: nav=follow | 2880 | 88 | 0.6478 | 0.5982 | 0.0496 [0.0071, 0.0963] | **SEP** | 0.639 / 0.683 / 0.0089 [-0.1520, 0.1794] |
| nav: nav=left (low n) | 403 | 13 | 6.9292 | 6.9119 | 0.0173 [-0.0821, 0.1172] |  | 0.089 / 0.033 / -0.0320 [-0.2038, 0.1507] |
| nav: nav=right | 1227 | 38 | 1.5486 | 1.5002 | 0.0484 [-0.0904, 0.1709] |  | 0.272 / 0.284 / 0.0027 [-0.2033, 0.2134] |

episode concentration: 0.597 of episodes have mean delta > 0; top-5 episodes carry 0.682 of the summed delta; delta excluding the top 5 = 0.0153 (overall 0.0464).

### S2 -- EMA-10.5k - EMA-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4507 | 139 | 1.4549 | 1.4229 | 0.0320 [-0.0004, 0.0618] |  |  |
| turn: straight <0.02 | 2897 | 132 | 1.1886 | 1.1731 | 0.0155 [-0.0056, 0.0392] |  | 0.643 / 0.311 / -0.0462 [-0.1514, 0.0478] |
| turn: gentle 0.02-0.06 | 720 | 98 | 0.9896 | 1.0251 | -0.0355 [-0.1478, 0.0531] |  | 0.160 / -0.177 / -0.0804 [-0.2601, 0.0488] |
| turn: turn 0.06-0.15 | 508 | 72 | 1.9557 | 1.9002 | 0.0555 [-0.0094, 0.1237] |  | 0.113 / 0.195 / 0.0264 [-0.0702, 0.1364] |
| turn: sharp >=0.15 | 367 | 42 | 3.3249 | 3.0631 | 0.2618 [0.0926, 0.5038] | **SEP** | 0.081 / 0.666 / 0.2501 [0.0266, 0.6211] BONF-SEP |
| turn: undef (GT crawling) (low n) | 15 | 13 | 12.5088 | 12.4628 | 0.0459 [-0.0200, 0.1377] |  | 0.003 / 0.005 / 0.0140 [-0.0913, 0.1959] |
| speed: v0 <2 | 218 | 37 | 13.5368 | 13.4247 | 0.1120 [0.0186, 0.2277] | **SEP** | 0.048 / 0.169 / 0.0841 [-0.0456, 0.3034] |
| speed: v0 2-5 | 516 | 58 | 1.8093 | 1.9040 | -0.0947 [-0.3147, 0.0646] |  | 0.114 / -0.339 / -0.1431 [-0.5391, 0.0747] |
| speed: v0 5-10 | 1390 | 84 | 1.1784 | 1.0618 | 0.1166 [0.0488, 0.1898] | **SEP** | 0.308 / 1.123 / 0.1222 [0.0109, 0.2528] BONF-SEP |
| speed: v0 10-15 | 1192 | 64 | 0.5878 | 0.5717 | 0.0160 [-0.0025, 0.0372] |  | 0.264 / 0.133 / -0.0217 [-0.0898, 0.0598] |
| speed: v0 15-25 | 856 | 35 | 0.3330 | 0.3482 | -0.0152 [-0.0289, -0.0039] | **SEP** | 0.190 / -0.090 / -0.0583 [-0.1139, -0.0001] BONF-SEP |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1466 | 0.1448 | 0.0019 [-0.0079, 0.0095] |  | 0.074 / 0.004 / -0.0326 [-0.0805, 0.0202] |
| nav: nav=follow | 2877 | 88 | 0.6481 | 0.6057 | 0.0424 [0.0138, 0.0774] | **SEP** | 0.638 / 0.846 / 0.0288 [-0.0649, 0.1436] |
| nav: nav=left (low n) | 403 | 13 | 6.9292 | 6.9022 | 0.0270 [-0.0233, 0.0816] |  | 0.089 / 0.075 / -0.0055 [-0.0945, 0.0941] |
| nav: nav=right | 1227 | 38 | 1.5486 | 1.5394 | 0.0092 [-0.0927, 0.0862] |  | 0.272 / 0.079 / -0.0313 [-0.1766, 0.0844] |

episode concentration: 0.597 of episodes have mean delta > 0; top-5 episodes carry 0.614 of the summed delta; delta excluding the top 5 = 0.0128 (overall 0.0320).

### S2 -- EMA-5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 4518 | 139 | 1.4616 | 1.4471 | 0.0145 [-0.0144, 0.0436] |  |  |
| turn: straight <0.02 | 2899 | 132 | 1.1726 | 1.1593 | 0.0133 [-0.0083, 0.0393] |  | 0.642 / 0.587 / -0.0035 [-0.1431, 0.1046] |
| turn: gentle 0.02-0.06 | 721 | 98 | 1.0252 | 1.0095 | 0.0157 [-0.0228, 0.0604] |  | 0.160 / 0.172 / 0.0013 [-0.0681, 0.0768] |
| turn: turn 0.06-0.15 | 509 | 72 | 1.9091 | 1.9295 | -0.0204 [-0.1283, 0.0816] |  | 0.113 / -0.158 / -0.0393 [-0.2345, 0.1168] |
| turn: sharp >=0.15 | 367 | 42 | 3.0631 | 2.9870 | 0.0761 [-0.1596, 0.3724] |  | 0.081 / 0.426 / 0.0671 [-0.2501, 0.5536] |
| turn: undef (GT crawling) (low n) | 22 | 18 | 16.7858 | 16.8657 | -0.0799 [-0.2458, 0.0381] |  | 0.005 / -0.027 / -0.0949 [-0.3562, 0.0649] |
| speed: v0 <2 | 229 | 38 | 13.6121 | 13.5957 | 0.0164 [-0.1129, 0.1237] |  | 0.051 / 0.057 / 0.0020 [-0.2251, 0.1838] |
| speed: v0 2-5 | 516 | 58 | 1.9040 | 1.8694 | 0.0345 [-0.1385, 0.2320] |  | 0.114 / 0.271 / 0.0226 [-0.2288, 0.3376] |
| speed: v0 5-10 | 1390 | 84 | 1.0618 | 1.0092 | 0.0526 [0.0049, 0.1051] | **SEP** | 0.308 / 1.115 / 0.0550 [-0.0315, 0.1490] |
| speed: v0 10-15 | 1192 | 64 | 0.5717 | 0.5744 | -0.0027 [-0.0357, 0.0224] |  | 0.264 / -0.049 / -0.0234 [-0.0943, 0.0530] |
| speed: v0 15-25 | 856 | 35 | 0.3482 | 0.3734 | -0.0252 [-0.0541, 0.0038] |  | 0.189 / -0.329 / -0.0490 [-0.1132, 0.0137] |
| speed: v0 >=25 (low n) | 335 | 13 | 0.1448 | 0.1576 | -0.0129 [-0.0345, 0.0044] |  | 0.074 / -0.066 / -0.0296 [-0.0889, 0.0235] |
| nav: nav=follow | 2879 | 88 | 0.6053 | 0.5982 | 0.0072 [-0.0185, 0.0332] |  | 0.637 / 0.315 / -0.0202 [-0.1407, 0.0900] |
| nav: nav=left (low n) | 406 | 13 | 7.3105 | 7.3200 | -0.0095 [-0.1057, 0.0757] |  | 0.090 / -0.059 / -0.0264 [-0.2023, 0.1061] |
| nav: nav=right | 1233 | 38 | 1.5351 | 1.4955 | 0.0396 [-0.0403, 0.1212] |  | 0.273 / 0.744 / 0.0345 [-0.0851, 0.1788] |

episode concentration: 0.525 of episodes have mean delta > 0; top-5 episodes carry 1.358 of the summed delta; delta excluding the top 5 = -0.0054 (overall 0.0145).

### S2 -- EMA-10.5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4474 | 139 | 0.6806 | 0.6282 | 0.0524 [0.0007, 0.1396] | **SEP** | 0.0234 [-0.0301, 0.1128] |
| seg1 | straight | 2879 | 132 | 0.6389 | 0.5697 | 0.0692 [-0.0006, 0.2091] |  | 0.0373 [-0.0353, 0.1818] |
| seg1 | curved | 1586 | 101 | 0.6450 | 0.6229 | 0.0222 [-0.0075, 0.0530] |  | -0.0016 [-0.0495, 0.0422] |
| seg2 | all | 4469 | 139 | 1.0096 | 0.9390 | 0.0707 [0.0002, 0.1658] | **SEP** | -0.1572 [-0.4207, 0.0080] |
| seg2 | straight | 2879 | 132 | 0.7417 | 0.6575 | 0.0842 [-0.0001, 0.2379] |  | -0.2550 [-0.6860, -0.0406] SEP |
| seg2 | curved | 1590 | 101 | 1.4948 | 1.4487 | 0.0461 [-0.0314, 0.1277] |  | 0.0197 [-0.1008, 0.1320] |
| seg3 | all | 4462 | 139 | 1.1352 | 1.2044 | -0.0692 [-0.3275, 0.0890] |  | -0.1455 [-0.4217, 0.0405] |
| seg3 | straight | 2875 | 132 | 0.6189 | 0.7742 | -0.1553 [-0.5963, 0.0691] |  | -0.2679 [-0.7136, -0.0402] SEP |
| seg3 | curved | 1587 | 101 | 2.0705 | 1.9836 | 0.0869 [-0.0395, 0.2205] |  | 0.0762 [-0.0841, 0.2274] |
| seg4 | all | 4445 | 139 | 1.5630 | 1.4925 | 0.0705 [-0.0133, 0.1548] |  | 0.1056 [-0.0403, 0.3086] |
| seg4 | straight | 2860 | 131 | 0.8717 | 0.8399 | 0.0318 [-0.0255, 0.0950] |  | 0.0747 [-0.1019, 0.3737] |
| seg4 | curved | 1577 | 101 | 2.8215 | 2.6798 | 0.1418 [-0.0521, 0.3470] |  | 0.1629 [-0.0671, 0.3930] |

### S2 -- EMA-10.5k - EMA-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4472 | 139 | 0.6807 | 0.5114 | 0.1693 [0.0021, 0.4982] | **SEP** | 0.0869 [-0.0005, 0.2535] |
| seg1 | straight | 2878 | 132 | 0.6389 | 0.3847 | 0.2542 [-0.0001, 0.8137] |  | 0.1320 [0.0010, 0.4130] SEP |
| seg1 | curved | 1586 | 101 | 0.6450 | 0.6293 | 0.0157 [-0.0005, 0.0318] |  | 0.0057 [-0.0155, 0.0294] |
| seg2 | all | 4467 | 139 | 1.0098 | 0.8593 | 0.1505 [0.0101, 0.4030] | **SEP** | 0.0513 [-0.0092, 0.1421] |
| seg2 | straight | 2878 | 132 | 0.7418 | 0.5358 | 0.2060 [0.0056, 0.6287] | **SEP** | 0.0654 [-0.0141, 0.2154] |
| seg2 | curved | 1589 | 101 | 1.4953 | 1.4454 | 0.0499 [-0.0005, 0.0995] |  | 0.0256 [-0.0354, 0.0942] |
| seg3 | all | 4463 | 139 | 1.1349 | 1.1787 | -0.0438 [-0.2167, 0.0652] |  | -0.0561 [-0.2405, 0.0645] |
| seg3 | straight | 2876 | 132 | 0.6186 | 0.7216 | -0.1030 [-0.3957, 0.0455] |  | -0.1336 [-0.4318, 0.0177] |
| seg3 | curved | 1587 | 101 | 2.0705 | 2.0070 | 0.0635 [-0.0247, 0.1485] |  | 0.0843 [-0.0191, 0.1969] |
| seg4 | all | 4443 | 139 | 1.5636 | 1.6399 | -0.0763 [-0.3381, 0.0884] |  | 0.1404 [0.0549, 0.2468] SEP |
| seg4 | straight | 2859 | 131 | 0.8719 | 1.0494 | -0.1776 [-0.5978, 0.0393] |  | 0.1013 [0.0072, 0.2552] SEP |
| seg4 | curved | 1577 | 101 | 2.8215 | 2.7146 | 0.1069 [-0.0261, 0.2359] |  | 0.2119 [0.0520, 0.3871] SEP |

### S2 -- EMA-5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 4480 | 139 | 0.5111 | 0.6679 | -0.1568 [-0.4850, 0.0109] |  | -0.0228 [-0.0357, -0.0086] SEP |
| seg1 | straight | 2879 | 132 | 0.3846 | 0.5697 | -0.1851 [-0.6013, 0.0075] |  | -0.0944 [-0.2362, -0.0254] SEP |
| seg1 | curved | 1587 | 101 | 0.6295 | 0.6227 | 0.0068 [-0.0152, 0.0299] |  | -0.0070 [-0.0415, 0.0236] |
| seg2 | all | 4475 | 139 | 0.8589 | 0.9381 | -0.0792 [-0.2477, 0.0180] |  | -0.2088 [-0.5428, -0.0223] SEP |
| seg2 | straight | 2884 | 132 | 0.5354 | 0.6569 | -0.1215 [-0.4025, 0.0168] |  | -0.3195 [-0.8844, -0.0550] SEP |
| seg2 | curved | 1591 | 101 | 1.4454 | 1.4478 | -0.0025 [-0.0596, 0.0590] |  | -0.0082 [-0.0938, 0.0671] |
| seg3 | all | 4465 | 139 | 1.1796 | 1.2049 | -0.0253 [-0.1154, 0.0381] |  | -0.0891 [-0.1850, -0.0153] SEP |
| seg3 | straight | 2876 | 132 | 0.7216 | 0.7739 | -0.0523 [-0.2033, 0.0350] |  | -0.1339 [-0.2883, -0.0496] SEP |
| seg3 | curved | 1589 | 101 | 2.0085 | 1.9850 | 0.0235 [-0.0644, 0.1180] |  | -0.0081 [-0.1186, 0.0975] |
| seg4 | all | 4450 | 139 | 1.6406 | 1.4938 | 0.1468 [-0.0147, 0.4187] |  | -0.0346 [-0.1222, 0.0707] |
| seg4 | straight | 2863 | 131 | 1.0484 | 0.8392 | 0.2092 [-0.0094, 0.6385] |  | -0.0263 [-0.1235, 0.1296] |
| seg4 | curved | 1579 | 101 | 2.7195 | 2.6846 | 0.0349 [-0.1000, 0.1790] |  | -0.0491 [-0.2181, 0.1149] |

### S2 -- EMA-10.5k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 439 | 47 | 5.9079 | 5.9675 | -0.0596 [-0.1966, 0.0494] |  |  |
| straight <0.02 | v0 5-15 | 1519 | 93 | 0.4460 | 0.3674 | 0.0786 [0.0286, 0.1387] | **SEP** |  |
| straight <0.02 | v0 >=15 | 940 | 43 | 0.1841 | 0.1947 | -0.0106 [-0.0237, 0.0012] |  |  |
| gentle 0.02-0.06 | v0 <5 | 111 | 29 | 1.4703 | 1.7693 | -0.2989 [-0.9229, 0.0691] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 461 | 74 | 1.0380 | 0.9711 | 0.0669 [-0.0069, 0.1516] |  |  |
| gentle 0.02-0.06 | v0 >=15 | 148 | 20 | 0.4782 | 0.5590 | -0.0808 [-0.1633, -0.0052] | **SEP** |  |
| turn 0.06-0.15 | v0 <5 (low n) | 78 | 25 | 5.0447 | 5.1476 | -0.1029 [-0.6403, 0.3242] |  |  |
| turn 0.06-0.15 | v0 5-15 | 343 | 54 | 1.5353 | 1.4275 | 0.1078 [-0.0068, 0.2339] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 87 | 11 | 0.8436 | 0.9719 | -0.1284 [-0.2847, 0.0418] |  |  |
| sharp >=0.15 | v0 <5 (low n) | 92 | 16 | 5.9492 | 5.2396 | 0.7097 [-0.0465, 1.7757] |  |  |
| sharp >=0.15 | v0 5-15 | 259 | 33 | 2.5326 | 2.2857 | 0.2469 [-0.0514, 0.6587] |  |  |
| sharp >=0.15 | v0 >=15 (low n) | 16 | 3 | 1.0602 | 1.3866 | -0.3264 [-0.5535, 0.0718] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 17 | 13 | 11.0545 | 11.1380 | -0.0835 [-0.2200, 0.0355] |  |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S2 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA-10.5k s0-s1 [CI] | EMA-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | 0.0031 [-0.0109, 0.0180] | 0.0080 [-0.0053, 0.0227] | 0.0110 [-0.0050, 0.0277] |
| turn|straight <0.02 | 0.0028 [-0.0175, 0.0223] | -0.0025 [-0.0152, 0.0107] | 0.0019 [-0.0151, 0.0193] |
| turn|gentle 0.02-0.06 | 0.0039 [-0.0207, 0.0359] | 0.0222 [-0.0046, 0.0560] | -0.0161 [-0.0555, 0.0143] |
| turn|turn 0.06-0.15 | -0.0099 [-0.0410, 0.0178] | 0.0029 [-0.0637, 0.0729] | 0.0331 [-0.0137, 0.0927] |
| turn|sharp >=0.15 | 0.0286 [-0.0320, 0.0952] | 0.0710 [-0.0077, 0.1690] | 0.1080 [-0.0316, 0.2884] |
| turn|undef (GT crawling) | -0.1140 [-0.2743, 0.0250] | -0.0061 [-0.1265, 0.1043] | -0.0237 [-0.0956, 0.0287] |
| speed|v0 <2 | -0.0346 [-0.0718, 0.0181] | 0.0507 [-0.0497, 0.1881] | 0.0573 [-0.0532, 0.2184] |
| speed|v0 2-5 | 0.0219 [-0.0389, 0.1008] | 0.0044 [-0.0592, 0.0734] | 0.0719 [-0.0183, 0.1911] |
| speed|v0 5-10 | 0.0111 [-0.0255, 0.0414] | 0.0164 [-0.0160, 0.0478] | 0.0006 [-0.0317, 0.0234] |
| speed|v0 10-15 | -0.0039 [-0.0181, 0.0088] | -0.0014 [-0.0124, 0.0091] | -0.0001 [-0.0071, 0.0073] |
| speed|v0 15-25 | 0.0012 [-0.0064, 0.0076] | 0.0006 [-0.0037, 0.0050] | -0.0023 [-0.0076, 0.0027] |
| speed|v0 >=25 | -0.0036 [-0.0103, 0.0025] | 0.0018 [-0.0030, 0.0066] | -0.0005 [-0.0049, 0.0035] |
| nav|nav=follow | -0.0035 [-0.0139, 0.0086] | 0.0030 [-0.0075, 0.0161] | -0.0036 [-0.0122, 0.0045] |
| nav|nav=left | -0.0000 [-0.0180, 0.0186] | 0.0540 [-0.0031, 0.1154] | 0.0608 [-0.0792, 0.2273] |
| nav|nav=right | 0.0197 [-0.0260, 0.0663] | 0.0044 [-0.0318, 0.0433] | 0.0283 [0.0035, 0.0597] SEP |

### S2 -- EMA-10.5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0464 | 0.0428 / 0.0500 | False / True / True | 0.0031 / 0.0110 | 4.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|straight <0.02 | 0.0287 | 0.0292 / 0.0283 | False / False / False | 0.0028 / 0.0019 | 10.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0199 | -0.0101 / -0.0296 | False / False / False | 0.0039 / 0.0161 | 1.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0350 | 0.0154 / 0.0545 | False / False / False | 0.0099 / 0.0331 | 1.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.3379 | 0.2982 / 0.3777 | False / True / True | 0.0286 / 0.1080 | 3.1 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | -0.0835 | -0.0960 / -0.0711 | False / False / False | 0.1140 / 0.0237 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1246 | 0.0818 / 0.1674 | False / False / False | 0.0346 / 0.0573 | 2.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.0601 | -0.0852 / -0.0351 | False / False / False | 0.0219 / 0.0719 | 0.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1692 | 0.1744 / 0.1640 | True / True / True | 0.0111 / 0.0006 | 15.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0133 | 0.0114 / 0.0152 | False / False / False | 0.0039 / 0.0001 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0404 | -0.0387 / -0.0422 | True / True / True | 0.0012 / 0.0023 | 17.8 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | -0.0110 | -0.0125 / -0.0094 | False / False / False | 0.0036 / 0.0005 | 3.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0496 | 0.0497 / 0.0495 | True / True / True | 0.0035 / 0.0036 | 13.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0173 | -0.0111 / 0.0457 | False / False / False | 0.0000 / 0.0608 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0484 | 0.0441 / 0.0527 | False / False / False | 0.0197 / 0.0283 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA-10.5k - EMA-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0320 | 0.0295 / 0.0346 | False / False / False | 0.0031 / 0.0080 | 4.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0155 | 0.0180 / 0.0131 | False / False / False | 0.0028 / 0.0025 | 5.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0355 | -0.0449 / -0.0262 | False / False / False | 0.0039 / 0.0222 | 1.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.0555 | 0.0490 / 0.0619 | False / False / False | 0.0099 / 0.0029 | 5.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.2618 | 0.2406 / 0.2830 | True / True / True | 0.0286 / 0.0710 | 3.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| turn|undef (GT crawling) | 0.0459 | -0.0010 / 0.0966 | False / True / False | 0.1140 / 0.0061 | 0.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.1120 | 0.0658 / 0.1581 | False / True / True | 0.0346 / 0.0507 | 2.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 2-5 | -0.0947 | -0.0859 / -0.1034 | False / False / False | 0.0219 / 0.0044 | 4.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1166 | 0.1139 / 0.1192 | True / True / True | 0.0111 / 0.0164 | 7.1 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.0160 | 0.0148 / 0.0173 | False / False / False | 0.0039 / 0.0014 | 4.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0152 | -0.0149 / -0.0155 | True / True / True | 0.0012 / 0.0006 | 12.7 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 >=25 | 0.0019 | -0.0009 / 0.0046 | False / False / False | 0.0036 / 0.0018 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0424 | 0.0392 / 0.0457 | True / True / True | 0.0035 / 0.0030 | 12.2 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | 0.0270 | -0.0003 / 0.0543 | False / True / False | 0.0000 / 0.0540 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0092 | 0.0165 / 0.0020 | False / False / False | 0.0197 / 0.0044 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- EMA-5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0145 | 0.0133 / 0.0157 | False / False / False | 0.0080 / 0.0110 | 1.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0133 | 0.0112 / 0.0155 | False / False / False | 0.0025 / 0.0019 | 5.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | 0.0157 | 0.0348 / -0.0035 | False / False / False | 0.0222 / 0.0161 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0204 | -0.0337 / -0.0070 | False / False / False | 0.0029 / 0.0331 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.0761 | 0.0576 / 0.0947 | False / False / False | 0.0710 / 0.1080 | 0.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.0799 | -0.0575 / -0.1099 | False / False / False | 0.0061 / 0.0237 | 3.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | 0.0164 | 0.0174 / 0.0157 | False / False / False | 0.0507 / 0.0573 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.0345 | 0.0008 / 0.0683 | False / False / False | 0.0044 / 0.0719 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.0526 | 0.0605 / 0.0448 | True / False / True | 0.0164 / 0.0006 | 3.2 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| speed|v0 10-15 | -0.0027 | -0.0033 / -0.0021 | False / False / False | 0.0014 / 0.0001 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | -0.0252 | -0.0238 / -0.0267 | False / False / False | 0.0006 / 0.0023 | 11.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0129 | -0.0117 / -0.0140 | False / False / False | 0.0018 / 0.0005 | 7.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0072 | 0.0106 / 0.0038 | False / False / False | 0.0030 / 0.0036 | 2.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | -0.0095 | -0.0111 / -0.0080 | False / False / False | 0.0540 / 0.0608 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.0396 | 0.0278 / 0.0514 | False / False / False | 0.0044 / 0.0283 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |

### S2 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1586): EMA-10.5k bias -0.049, refcv7-50400 bias +0.082, EMA-5k bias -0.009; EMA-10.5k MAE 0.645, refcv7-50400 MAE 0.623, EMA-5k MAE 0.629
* seg2 (n 1589): EMA-10.5k bias -0.410, refcv7-50400 bias -0.097, EMA-5k bias -0.283; EMA-10.5k MAE 1.495, refcv7-50400 MAE 1.448, EMA-5k MAE 1.445
* seg3 (n 1587): EMA-10.5k bias -1.009, refcv7-50400 bias -0.618, EMA-5k bias -0.799; EMA-10.5k MAE 2.071, refcv7-50400 MAE 1.984, EMA-5k MAE 2.007
* seg4 (n 1577): EMA-10.5k bias -1.360, refcv7-50400 bias -0.714, EMA-5k bias -1.018; EMA-10.5k MAE 2.822, refcv7-50400 MAE 2.680, EMA-5k MAE 2.715
**sharp (>= 0.15)**

* seg1 (n 365): EMA-10.5k bias -0.356, refcv7-50400 bias +0.011, EMA-5k bias -0.266; EMA-10.5k MAE 0.917, refcv7-50400 MAE 0.915, EMA-5k MAE 0.879
* seg2 (n 366): EMA-10.5k bias -1.337, refcv7-50400 bias -0.528, EMA-5k bias -1.088; EMA-10.5k MAE 2.543, refcv7-50400 MAE 2.413, EMA-5k MAE 2.367
* seg3 (n 366): EMA-10.5k bias -2.755, refcv7-50400 bias -1.862, EMA-5k bias -2.392; EMA-10.5k MAE 4.096, refcv7-50400 MAE 3.643, EMA-5k MAE 3.789
* seg4 (n 364): EMA-10.5k bias -3.360, refcv7-50400 bias -1.837, EMA-5k bias -2.737; EMA-10.5k MAE 5.564, refcv7-50400 MAE 4.878, EMA-5k MAE 5.109
**curved & v0 5-10**

* seg1 (n 627): EMA-10.5k bias -0.184, refcv7-50400 bias -0.038, EMA-5k bias -0.134; EMA-10.5k MAE 0.585, refcv7-50400 MAE 0.535, EMA-5k MAE 0.553
* seg2 (n 627): EMA-10.5k bias -0.641, refcv7-50400 bias -0.282, EMA-5k bias -0.477; EMA-10.5k MAE 1.512, refcv7-50400 MAE 1.363, EMA-5k MAE 1.377
* seg3 (n 627): EMA-10.5k bias -1.317, refcv7-50400 bias -0.871, EMA-5k bias -1.035; EMA-10.5k MAE 2.239, refcv7-50400 MAE 2.019, EMA-5k MAE 2.036
* seg4 (n 627): EMA-10.5k bias -1.867, refcv7-50400 bias -1.194, EMA-5k bias -1.432; EMA-10.5k MAE 3.061, refcv7-50400 MAE 2.730, EMA-5k MAE 2.759
**sharp & v0 5-10**

* seg1 (n 164): EMA-10.5k bias -0.249, refcv7-50400 bias +0.103, EMA-5k bias -0.151; EMA-10.5k MAE 0.824, refcv7-50400 MAE 0.857, EMA-5k MAE 0.787
* seg2 (n 164): EMA-10.5k bias -1.209, refcv7-50400 bias -0.422, EMA-5k bias -0.929; EMA-10.5k MAE 2.344, refcv7-50400 MAE 2.243, EMA-5k MAE 2.117
* seg3 (n 164): EMA-10.5k bias -2.919, refcv7-50400 bias -2.032, EMA-5k bias -2.456; EMA-10.5k MAE 3.897, refcv7-50400 MAE 3.465, EMA-5k MAE 3.412
* seg4 (n 164): EMA-10.5k bias -3.856, refcv7-50400 bias -2.403, EMA-5k bias -3.048; EMA-10.5k MAE 5.463, refcv7-50400 MAE 4.635, EMA-5k MAE 4.706

### S2 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1577 / 1577 | 10.2 | EMA-10.5k -0.1456 [-0.2323, -0.0786]; refcv7-50400 -0.0844 [-0.1557, -0.0274]; EMA-5k -0.1194 [-0.2052, -0.0541] | EMA-10.5k -0.0613 [-0.0911, -0.0369] SEP; EMA-5k -0.0350 [-0.0582, -0.0171] SEP |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1593 / 4753 | 7.2 | EMA-10.5k -0.2037 [-0.3257, -0.1048]; refcv7-50400 -0.1465 [-0.2654, -0.0521]; EMA-5k -0.1804 [-0.3037, -0.0810] | EMA-10.5k -0.0572 [-0.0843, -0.0351] SEP; EMA-5k -0.0338 [-0.0542, -0.0185] SEP |
| sharp (>= 0.15) | last segment | 364 / 364 | 25.4 | EMA-10.5k -0.1548 [-0.2571, -0.0759]; refcv7-50400 -0.0922 [-0.1765, -0.0257]; EMA-5k -0.1303 [-0.2322, -0.0535] | EMA-10.5k -0.0626 [-0.0980, -0.0339] SEP; EMA-5k -0.0382 [-0.0648, -0.0175] SEP |
| sharp (>= 0.15) | segments 2..K | 367 / 1096 | 17.9 | EMA-10.5k -0.1654 [-0.2676, -0.0867]; refcv7-50400 -0.1041 [-0.1873, -0.0361]; EMA-5k -0.1424 [-0.2435, -0.0658] | EMA-10.5k -0.0613 [-0.0938, -0.0340] SEP; EMA-5k -0.0383 [-0.0626, -0.0194] SEP |
| curved & v0 5-10 | last segment | 627 / 627 | 10.7 | EMA-10.5k -0.1479 [-0.2329, -0.0826]; refcv7-50400 -0.0925 [-0.1767, -0.0360]; EMA-5k -0.1128 [-0.1915, -0.0594] | EMA-10.5k -0.0553 [-0.1025, -0.0150] SEP; EMA-5k -0.0202 [-0.0429, +0.0055] |
| curved & v0 5-10 | segments 2..K | 627 / 1881 | 7.6 | EMA-10.5k -0.1560 [-0.2400, -0.0922]; refcv7-50400 -0.0993 [-0.1849, -0.0421]; EMA-5k -0.1233 [-0.2007, -0.0703] | EMA-10.5k -0.0568 [-0.1007, -0.0192] SEP; EMA-5k -0.0240 [-0.0453, +0.0004] |
| sharp & v0 5-10 | last segment | 164 / 164 | 26.6 | EMA-10.5k -0.1439 [-0.2364, -0.0718]; refcv7-50400 -0.0912 [-0.1852, -0.0297]; EMA-5k -0.1116 [-0.1998, -0.0515] | EMA-10.5k -0.0527 [-0.1039, -0.0100] SEP; EMA-5k -0.0203 [-0.0436, +0.0069] |
| sharp & v0 5-10 | segments 2..K | 164 / 492 | 18.8 | EMA-10.5k -0.1483 [-0.2400, -0.0764]; refcv7-50400 -0.0955 [-0.1896, -0.0316]; EMA-5k -0.1186 [-0.2069, -0.0577] | EMA-10.5k -0.0528 [-0.1023, -0.0107] SEP; EMA-5k -0.0231 [-0.0459, +0.0041] |

### S2 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA-10.5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4510 | 0.0464 | 0.00375 | 0.00724 | 0.512 / 0.488 / 0.000 | 304.1 / 209.3 |
| turn|straight <0.02 | 2898 | 0.0287 | 0.00124 | -0.00057 | 0.507 / 0.493 / 0.000 | 140.5 / 83.2 |
| turn|gentle 0.02-0.06 | 720 | -0.0199 | 0.00615 | 0.01147 | 0.511 / 0.489 / 0.000 | 30.0 / -14.3 |
| turn|turn 0.06-0.15 | 508 | 0.0350 | 0.01513 | 0.02973 | 0.514 / 0.486 / 0.000 | 30.4 / 17.8 |
| turn|sharp >=0.15 | 367 | 0.3379 | 0.11961 | 0.15752 | 0.561 / 0.439 / 0.000 | 54.2 / 124.0 |
| turn|undef (GT crawling) | 17 | -0.0835 | -0.03532 | -0.03805 | 0.412 / 0.588 / 0.000 | 0.3 / -1.4 |
| speed|v0 <2 | 221 | 0.1246 | -0.00662 | 0.02402 | 0.493 / 0.507 / 0.000 | 14.9 / 27.5 |
| speed|v0 2-5 | 516 | -0.0601 | 0.01604 | 0.02236 | 0.525 / 0.475 / 0.000 | 57.5 / -31.0 |
| speed|v0 5-10 | 1390 | 0.1692 | 0.02043 | 0.04247 | 0.544 / 0.456 / 0.000 | 115.1 / 235.2 |
| speed|v0 10-15 | 1192 | 0.0133 | 0.00539 | 0.00412 | 0.519 / 0.481 / 0.000 | 25.3 / 15.9 |
| speed|v0 15-25 | 856 | -0.0404 | -0.00795 | -0.01784 | 0.452 / 0.548 / 0.000 | 7.0 / -34.6 |
| speed|v0 >=25 | 335 | -0.0110 | 0.00040 | -0.00042 | 0.501 / 0.499 / 0.000 | 0.7 / -3.7 |
| nav|nav=follow | 2880 | 0.0496 | -0.00076 | -0.00291 | 0.495 / 0.505 / 0.000 | 171.4 / 143.0 |
| nav|nav=left | 403 | 0.0173 | 0.02063 | 0.02719 | 0.563 / 0.437 / 0.000 | 18.7 / 7.0 |
| nav|nav=right | 1227 | 0.0484 | 0.01116 | 0.03215 | 0.535 / 0.465 / 0.000 | 104.0 / 59.4 |
**EMA-10.5k - EMA-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4507 | 0.0320 | 0.00206 | 0.00845 | 0.515 / 0.485 / 0.000 | 197.6 / 144.3 |
| turn|straight <0.02 | 2897 | 0.0155 | -0.00012 | 0.00129 | 0.499 / 0.501 / 0.000 | 94.7 / 44.9 |
| turn|gentle 0.02-0.06 | 720 | -0.0355 | 0.00293 | 0.00734 | 0.512 / 0.487 / 0.000 | 26.4 / -25.6 |
| turn|turn 0.06-0.15 | 508 | 0.0555 | 0.01479 | 0.03496 | 0.535 / 0.465 / 0.000 | 20.5 / 28.2 |
| turn|sharp >=0.15 | 367 | 0.2618 | 0.07272 | 0.15679 | 0.619 / 0.381 / 0.000 | 30.9 / 96.1 |
| turn|undef (GT crawling) | 15 | 0.0459 | 0.00482 | 0.00427 | 0.600 / 0.400 / 0.000 | 0.5 / 0.7 |
| speed|v0 <2 | 218 | 0.1120 | 0.00810 | 0.03594 | 0.555 / 0.445 / 0.000 | 7.6 / 24.4 |
| speed|v0 2-5 | 516 | -0.0947 | 0.00518 | 0.03326 | 0.512 / 0.488 / 0.000 | 23.1 / -48.9 |
| speed|v0 5-10 | 1390 | 0.1166 | 0.01224 | 0.02884 | 0.545 / 0.455 / 0.000 | 98.1 / 162.0 |
| speed|v0 10-15 | 1192 | 0.0160 | 0.00164 | 0.00216 | 0.515 / 0.485 / 0.000 | 22.6 / 19.1 |
| speed|v0 15-25 | 856 | -0.0152 | -0.00350 | -0.00460 | 0.466 / 0.534 / 0.000 | 3.9 / -13.0 |
| speed|v0 >=25 | 335 | 0.0019 | -0.00004 | 0.00231 | 0.499 / 0.501 / 0.000 | 0.7 / 0.6 |
| nav|nav=follow | 2877 | 0.0424 | 0.00064 | 0.00328 | 0.504 / 0.496 / 0.000 | 123.5 / 122.1 |
| nav|nav=left | 403 | 0.0270 | 0.00625 | 0.01731 | 0.529 / 0.471 / 0.000 | 10.4 / 10.9 |
| nav|nav=right | 1227 | 0.0092 | 0.00798 | 0.02273 | 0.538 / 0.462 / 0.000 | 57.7 / 11.3 |
**EMA-5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 4518 | 0.0145 | 0.00098 | 0.00031 | 0.504 / 0.496 / 0.000 | 203.4 / 65.6 |
| turn|straight <0.02 | 2899 | 0.0133 | 0.00000 | -0.00155 | 0.500 / 0.500 / 0.000 | 88.7 / 38.5 |
| turn|gentle 0.02-0.06 | 721 | 0.0157 | -0.00091 | 0.00216 | 0.497 / 0.503 / 0.000 | 21.8 / 11.3 |
| turn|turn 0.06-0.15 | 509 | -0.0204 | 0.00607 | 0.00126 | 0.513 / 0.487 / 0.000 | 21.0 / -10.4 |
| turn|sharp >=0.15 | 367 | 0.0761 | 0.05080 | 0.02528 | 0.537 / 0.463 / 0.000 | 38.3 / 27.9 |
| turn|undef (GT crawling) | 22 | -0.0799 | -0.01009 | -0.00502 | 0.455 / 0.545 / 0.000 | 0.3 / -1.8 |
| speed|v0 <2 | 229 | 0.0164 | -0.00388 | 0.00269 | 0.480 / 0.520 / 0.000 | 10.1 / 3.8 |
| speed|v0 2-5 | 516 | 0.0345 | 0.00515 | 0.01925 | 0.508 / 0.492 / 0.000 | 44.3 / 17.8 |
| speed|v0 5-10 | 1390 | 0.0526 | 0.00582 | 0.01205 | 0.519 / 0.481 / 0.000 | 79.0 / 73.2 |
| speed|v0 10-15 | 1192 | -0.0027 | 0.00432 | 0.00316 | 0.523 / 0.477 / 0.000 | 16.0 / -3.2 |
| speed|v0 15-25 | 856 | -0.0252 | -0.00653 | -0.01192 | 0.464 / 0.536 / 0.000 | 6.7 / -21.6 |
| speed|v0 >=25 | 335 | -0.0129 | -0.00237 | -0.00258 | 0.484 / 0.516 / 0.000 | 0.4 / -4.3 |
| nav|nav=follow | 2879 | 0.0072 | -0.00233 | -0.00610 | 0.490 / 0.510 / 0.000 | 86.8 / 20.7 |
| nav|nav=left | 406 | -0.0095 | 0.01469 | 0.01876 | 0.559 / 0.441 / 0.000 | 15.9 / -3.9 |
| nav|nav=right | 1233 | 0.0396 | 0.00632 | 0.01408 | 0.517 / 0.483 / 0.000 | 87.5 / 48.8 |

### S2 -- the episodes carrying the summed delta (panel file index only)

**EMA-10.5k - refcv7-50400** (sum 209.3 over 139 episodes)

* ep013: n 34, share 0.175, mean delta 1.0755, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep111: n 34, share 0.154, mean delta 0.9475, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.136, mean delta 0.8349, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep023: n 34, share 0.109, mean delta 0.6723, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep075: n 34, share 0.109, mean delta 0.6682, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep121: n 35, share 0.089, mean delta 0.5295, sharp-window frac 0.20, curved frac 0.74, mean v0 6.1, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
**EMA-10.5k - EMA-5k** (sum 144.3 over 139 episodes)

* ep111: n 34, share 0.196, mean delta 0.8303, sharp-window frac 0.26, curved frac 0.53, mean v0 10.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep012: n 34, share 0.124, mean delta 0.5248, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep023: n 34, share 0.099, mean delta 0.4199, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep108: n 23, share 0.098, mean delta 0.6169, sharp-window frac 0.00, curved frac 0.57, mean v0 9.3, nav {'nav=follow': 23, 'nav=left': 0, 'nav=right': 0}
* ep066: n 27, share 0.097, mean delta 0.5192, sharp-window frac 0.00, curved frac 0.00, mean v0 3.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep024: n 35, share 0.091, mean delta 0.3736, sharp-window frac 0.00, curved frac 0.00, mean v0 5.6, nav {'nav=follow': 35, 'nav=left': 0, 'nav=right': 0}
**EMA-5k - refcv7-50400** (sum 65.6 over 139 episodes)

* ep013: n 34, share 0.527, mean delta 1.0183, sharp-window frac 0.18, curved frac 0.29, mean v0 7.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep075: n 34, share 0.342, mean delta 0.6610, sharp-window frac 0.00, curved frac 0.38, mean v0 7.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}
* ep067: n 34, share 0.165, mean delta 0.3191, sharp-window frac 0.32, curved frac 0.65, mean v0 4.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep021: n 34, share 0.162, mean delta 0.3134, sharp-window frac 0.15, curved frac 0.53, mean v0 5.4, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep012: n 34, share 0.161, mean delta 0.3100, sharp-window frac 0.32, curved frac 0.56, mean v0 7.2, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 34}
* ep023: n 34, share 0.131, mean delta 0.2524, sharp-window frac 0.18, curved frac 0.94, mean v0 6.9, nav {'nav=follow': 34, 'nav=left': 0, 'nav=right': 0}

### S2 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA-10.5k vs EMA-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1611 / 0.0068 [-0.0034, 0.0183] | 73 / 0.1166 [-0.0452, 0.2927] |
| only new agrees | 401 / -0.1063 [-0.2009, -0.0349] | 28 / -0.0248 [-0.2931, 0.2443] |
| only earlier agrees | 275 / 0.2046 [0.0675, 0.4360] | 25 / 1.3040 [-0.0258, 3.5168] |
| neither agrees | 2221 / 0.0488 [-0.0013, 0.0953] | 241 / 0.1986 [0.0273, 0.3924] |
**EMA-10.5k vs EMA-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1590 / 0.0067 [-0.0038, 0.0180] | 80 / 0.0791 [-0.0737, 0.2285] |
| only new agrees | 411 / -0.0871 [-0.1657, -0.0215] | 23 / 0.0981 [-0.2711, 0.5890] |
| only earlier agrees | 281 / 0.2541 [0.0743, 0.4964] | 21 / 1.5007 [0.0038, 3.7828] |
| neither agrees | 2226 / 0.0492 [-0.0161, 0.1046] | 243 / 0.2624 [0.0988, 0.4636] |

### S2 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA-10.5k vs EMA-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 4483 | 0.0414 [-0.0150, 0.0973] | 0.0377 [-0.0311, 0.1106] | 0.0048 [-0.0565, 0.0645] |
| turn|straight <0.02 | 2887 | 0.0330 [-0.0111, 0.0910] | 0.0066 [-0.0273, 0.0472] | 0.0265 [-0.0169, 0.0751] |
| turn|gentle 0.02-0.06 | 718 | 0.0278 [-0.0292, 0.0880] | -0.0222 [-0.1579, 0.1103] | 0.0503 [-0.0586, 0.1762] |
| turn|turn 0.06-0.15 | 501 | 0.0665 [-0.0712, 0.2335] | 0.1061 [-0.0326, 0.3074] | -0.0396 [-0.2664, 0.1599] |
| turn|sharp >=0.15 | 365 | 0.1006 [-0.4657, 0.5041] | 0.3019 [-0.2941, 0.7995] | -0.1896 [-0.5081, 0.0826] |
| turn|undef (GT crawling) | 12 | 0.0236 [-0.1437, 0.1779] | 0.1364 [-0.0364, 0.3356] | -0.1660 [-0.4217, 0.0154] |
| speed|v0 <2 | 194 | -0.1405 [-0.3612, 0.0558] | 0.2537 [0.0988, 0.4898] | -0.3942 [-0.7780, -0.1071] SEP |
| speed|v0 2-5 | 516 | 0.1001 [-0.2674, 0.4425] | 0.0959 [-0.3776, 0.5625] | 0.0042 [-0.4369, 0.3854] |
| speed|v0 5-10 | 1390 | 0.1209 [0.0333, 0.2160] | 0.0594 [-0.0384, 0.1665] | 0.0614 [-0.0156, 0.1430] |
| speed|v0 10-15 | 1192 | -0.0034 [-0.0302, 0.0272] | -0.0114 [-0.0289, 0.0049] | 0.0080 [-0.0187, 0.0380] |
| speed|v0 15-25 | 856 | -0.0054 [-0.0197, 0.0059] | -0.0039 [-0.0198, 0.0083] | -0.0015 [-0.0088, 0.0059] |
| speed|v0 >=25 | 335 | 0.0057 [-0.0033, 0.0194] | -0.0007 [-0.0094, 0.0106] | 0.0065 [0.0005, 0.0132] SEP |
**EMA-10.5k vs EMA-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S2 -- selection / goal proxy (new vs earlier, per seed)

* EMA-10.5k vs EMA-5k|seed0: anchor flip rate 0.306; heading delta on flipped windows 0.0776 [-0.0148, 0.1646] (n 1407) vs unflipped 0.0076 [-0.0013, 0.0174]; goal_dist_sel proxy new-earlier 0.0504 [-0.0065, 0.1123]; corr(window dHeading, window dGoalDist) -0.057
* EMA-10.5k vs EMA-5k|seed1: anchor flip rate 0.309; heading delta on flipped windows 0.0881 [-0.0177, 0.1827] (n 1422) vs unflipped 0.0099 [0.0002, 0.0207]; goal_dist_sel proxy new-earlier 0.0540 [-0.0034, 0.1152]; corr(window dHeading, window dGoalDist) -0.006

## S6 ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s) -- 3668 windows / 139 episodes; heading = LAT_heading_mae_deg

bin census (windows): {"turn": {"straight <0.02": 2141, "gentle 0.02-0.06": 681, "turn 0.06-0.15": 442, "sharp >=0.15": 317, "undef (GT crawling)": 87}, "speed": {"v0 <2": 330, "v0 2-5": 422, "v0 5-10": 1040, "v0 10-15": 952, "v0 15-25": 679, "v0 >=25": 245}, "nav": {"nav=follow": 2324, "nav=left": 342, "nav=right": 1002}}

### S6 -- EMA-10.5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3525 | 139 | 3.3919 | 3.2465 | 0.1454 [-0.0342, 0.3191] |  |  |
| turn: straight <0.02 | 2098 | 122 | 1.5346 | 1.4595 | 0.0751 [0.0110, 0.1398] | **SEP** | 0.595 / 0.307 / -0.1737 [-0.8037, 0.5115] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.7017 | 2.8900 | -0.1883 [-0.7293, 0.1880] |  | 0.191 / -0.248 / -0.4128 [-1.3495, 0.1290] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4718 | 6.3633 | 0.1085 [-0.2301, 0.4297] |  | 0.123 / 0.092 / -0.0421 [-0.6004, 0.4532] |
| turn: sharp >=0.15 | 304 | 33 | 12.8508 | 11.4418 | 1.4090 [0.1902, 2.8853] | **SEP** | 0.086 / 0.836 / 1.3829 [-0.4234, 3.6670] |
| turn: undef (GT crawling) (low n) | 13 | 11 | 14.7134 | 14.2040 | 0.5095 [0.0545, 1.2103] | **SEP** | 0.004 / 0.013 / 0.3654 [-0.3154, 1.6921] |
| speed: v0 <2 | 187 | 24 | 13.6258 | 13.8079 | -0.1821 [-0.9210, 0.2642] |  | 0.053 / -0.066 / -0.3458 [-1.9259, 0.3882] |
| speed: v0 2-5 | 422 | 52 | 4.0290 | 4.1501 | -0.1211 [-1.1064, 0.6765] |  | 0.120 / -0.100 / -0.3027 [-2.0481, 0.8554] |
| speed: v0 5-10 | 1040 | 79 | 3.8977 | 3.4941 | 0.4036 [0.0670, 0.8126] | **SEP** | 0.295 / 0.819 / 0.3663 [-0.1342, 1.0121] |
| speed: v0 10-15 | 952 | 62 | 2.6434 | 2.4870 | 0.1564 [-0.0500, 0.3872] |  | 0.270 / 0.290 / 0.0151 [-0.4361, 0.5478] |
| speed: v0 15-25 | 679 | 34 | 1.5135 | 1.4456 | 0.0679 [-0.0136, 0.1623] |  | 0.193 / 0.090 / -0.0960 [-0.4224, 0.2553] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4497 | 0.5193 | -0.0696 [-0.1632, 0.0083] |  | 0.070 / -0.033 / -0.2310 [-0.5296, 0.0996] |
| nav: nav=follow | 2249 | 88 | 2.1601 | 1.9783 | 0.1818 [0.0345, 0.3566] | **SEP** | 0.638 / 0.798 / 0.1005 [-0.5135, 0.7141] |
| nav: nav=left (low n) | 320 | 13 | 9.7087 | 9.7553 | -0.0467 [-0.4725, 0.3121] |  | 0.091 / -0.029 / -0.2113 [-0.9843, 0.3613] |
| nav: nav=right | 956 | 38 | 4.1752 | 4.0510 | 0.1241 [-0.4050, 0.5937] |  | 0.271 / 0.232 / -0.0292 [-0.8284, 0.7446] |

episode concentration: 0.612 of episodes have mean delta > 0; top-5 episodes carry 0.918 of the summed delta; delta excluding the top 5 = 0.0124 (overall 0.1454).

### S6 -- EMA-10.5k - EMA-5k, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3523 | 139 | 3.3922 | 3.3520 | 0.0403 [-0.0931, 0.1664] |  |  |
| turn: straight <0.02 | 2097 | 122 | 1.5347 | 1.5272 | 0.0075 [-0.0628, 0.0728] |  | 0.595 / 0.111 / -0.0810 [-0.4864, 0.3670] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.7017 | 2.8890 | -0.1873 [-0.6001, 0.0735] |  | 0.192 / -0.891 / -0.2815 [-1.0752, 0.0949] |
| turn: turn 0.06-0.15 | 435 | 59 | 6.4718 | 6.5123 | -0.0404 [-0.2489, 0.1476] |  | 0.123 / -0.124 / -0.0921 [-0.4582, 0.2396] |
| turn: sharp >=0.15 | 304 | 33 | 12.8508 | 11.9709 | 0.8799 [-0.0008, 1.9691] |  | 0.086 / 1.885 / 0.9189 [-0.3436, 2.6810] |
| turn: undef (GT crawling) (low n) | 12 | 10 | 15.5815 | 15.3580 | 0.2235 [0.0173, 0.4221] | **SEP** | 0.003 / 0.019 / 0.1838 [-0.2877, 0.5780] |
| speed: v0 <2 | 185 | 24 | 13.7436 | 13.8909 | -0.1473 [-0.6623, 0.2878] |  | 0.053 / -0.192 / -0.1979 [-1.2615, 0.5641] |
| speed: v0 2-5 | 422 | 52 | 4.0290 | 4.3226 | -0.2936 [-0.9960, 0.1548] |  | 0.120 / -0.873 / -0.3793 [-1.6455, 0.1941] |
| speed: v0 5-10 | 1040 | 79 | 3.8977 | 3.6560 | 0.2417 [-0.0491, 0.5895] |  | 0.295 / 1.772 / 0.2858 [-0.1489, 0.8163] |
| speed: v0 10-15 | 952 | 62 | 2.6434 | 2.6253 | 0.0181 [-0.0823, 0.1251] |  | 0.270 / 0.122 / -0.0303 [-0.3341, 0.3180] |
| speed: v0 15-25 | 679 | 34 | 1.5135 | 1.4734 | 0.0401 [-0.0138, 0.0955] |  | 0.193 / 0.192 / -0.0003 [-0.2216, 0.2455] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4497 | 0.4614 | -0.0116 [-0.0513, 0.0315] |  | 0.070 / -0.020 / -0.0558 [-0.2633, 0.1527] |
| nav: nav=follow | 2247 | 88 | 2.1596 | 2.0725 | 0.0871 [0.0081, 0.1863] | **SEP** | 0.638 / 1.380 / 0.1293 [-0.3279, 0.6254] |
| nav: nav=left (low n) | 320 | 13 | 9.7087 | 9.8381 | -0.1295 [-0.3683, 0.0727] |  | 0.091 / -0.292 / -0.1867 [-0.6794, 0.1644] |
| nav: nav=right | 956 | 38 | 4.1752 | 4.1882 | -0.0130 [-0.4589, 0.3642] |  | 0.271 / -0.088 / -0.0731 [-0.7163, 0.5233] |

episode concentration: 0.547 of episodes have mean delta > 0; top-5 episodes carry 2.158 of the summed delta; delta excluding the top 5 = -0.0484 (overall 0.0403).

### S6 -- EMA-5k - refcv7-50400, 2-seed mean of the per-window metric

| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |
|---|---|---|---|---|---|---|---|
| ALL | 3541 | 139 | 3.3694 | 3.2663 | 0.1031 [0.0009, 0.2105] | **SEP** |  |
| turn: straight <0.02 | 2103 | 122 | 1.5255 | 1.4585 | 0.0670 [-0.0014, 0.1482] |  | 0.594 / 0.386 / -0.0888 [-0.4844, 0.2439] |
| turn: gentle 0.02-0.06 | 675 | 77 | 2.8890 | 2.8900 | -0.0011 [-0.1769, 0.1642] |  | 0.191 / -0.002 / -0.1287 [-0.4357, 0.1678] |
| turn: turn 0.06-0.15 | 437 | 59 | 6.4953 | 6.3490 | 0.1463 [-0.1561, 0.4543] |  | 0.123 / 0.175 / 0.0493 [-0.3595, 0.5159] |
| turn: sharp >=0.15 | 307 | 33 | 12.1191 | 11.5771 | 0.5421 [-0.2105, 1.4704] |  | 0.087 / 0.456 / 0.4807 [-0.5988, 1.9760] |
| turn: undef (GT crawling) (low n) | 19 | 10 | 11.2453 | 11.5385 | -0.2931 [-0.7032, 0.1586] |  | 0.005 / -0.015 / -0.3983 [-1.0973, 0.2968] |
| speed: v0 <2 | 203 | 25 | 13.2602 | 13.3215 | -0.0613 [-0.3400, 0.1491] |  | 0.057 / -0.034 / -0.1744 [-0.6486, 0.1491] |
| speed: v0 2-5 | 422 | 52 | 4.3226 | 4.1501 | 0.1725 [-0.2426, 0.7556] |  | 0.119 / 0.199 / 0.0788 [-0.5196, 0.9445] |
| speed: v0 5-10 | 1040 | 79 | 3.6560 | 3.4941 | 0.1619 [-0.0457, 0.3709] |  | 0.294 / 0.461 / 0.0832 [-0.2652, 0.4450] |
| speed: v0 10-15 | 952 | 62 | 2.6253 | 2.4870 | 0.1383 [-0.0437, 0.3427] |  | 0.269 / 0.361 / 0.0481 [-0.2840, 0.4412] |
| speed: v0 15-25 | 679 | 34 | 1.4734 | 1.4456 | 0.0278 [-0.0664, 0.1377] |  | 0.192 / 0.052 / -0.0931 [-0.3396, 0.1481] |
| speed: v0 >=25 (low n) | 245 | 11 | 0.4614 | 0.5193 | -0.0579 [-0.1214, -0.0014] | **SEP** | 0.069 / -0.039 / -0.1730 [-0.3808, -0.0069] BONF-SEP |
| nav: nav=follow | 2251 | 88 | 2.0698 | 1.9774 | 0.0924 [-0.0068, 0.1992] |  | 0.636 / 0.570 / -0.0294 [-0.4576, 0.3213] |
| nav: nav=left (low n) | 322 | 13 | 10.0026 | 9.9171 | 0.0855 [-0.3436, 0.5053] |  | 0.091 / 0.075 / -0.0193 [-0.7236, 0.6515] |
| nav: nav=right | 968 | 38 | 4.1849 | 4.0511 | 0.1338 [-0.1056, 0.4025] |  | 0.273 / 0.355 / 0.0423 [-0.3196, 0.5710] |

episode concentration: 0.540 of episodes have mean delta > 0; top-5 episodes carry 0.827 of the summed delta; delta excluding the top 5 = 0.0185 (overall 0.1031).

### S6 -- EMA-10.5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3490 | 139 | 0.9343 | 0.7643 | 0.1701 [-0.0084, 0.4937] |  | -0.0829 [-0.2010, 0.0021] |
| seg1 | straight | 2071 | 122 | 0.6464 | 0.5434 | 0.1030 [0.0051, 0.2954] | **SEP** | 0.0509 [-0.0525, 0.2477] |
| seg1 | curved | 1407 | 88 | 1.2333 | 0.9651 | 0.2683 [-0.0398, 0.8534] |  | -0.2794 [-0.8722, 0.0520] |
| seg2 | all | 3482 | 139 | 1.2975 | 1.4585 | -0.1610 [-0.6012, 0.0998] |  | 0.0034 [-0.0935, 0.1095] |
| seg2 | straight | 2075 | 122 | 0.5973 | 0.8394 | -0.2420 [-0.8143, 0.0476] |  | -0.1235 [-0.3273, -0.0031] SEP |
| seg2 | curved | 1407 | 88 | 2.3300 | 2.3716 | -0.0416 [-0.3516, 0.2112] |  | 0.1906 [-0.0976, 0.5140] |
| seg3 | all | 3448 | 139 | 2.1609 | 2.0566 | 0.1043 [-0.0512, 0.2585] |  | 0.4564 [0.0436, 1.1346] SEP |
| seg3 | straight | 2057 | 121 | 1.0093 | 0.9782 | 0.0310 [-0.0150, 0.0813] |  | 0.4034 [0.0132, 1.1768] SEP |
| seg3 | curved | 1391 | 88 | 3.8638 | 3.6512 | 0.2126 [-0.1694, 0.5849] |  | 0.5347 [0.0185, 1.1532] SEP |
| seg4 | all | 3411 | 139 | 3.2565 | 2.9618 | 0.2947 [0.0349, 0.5561] | **SEP** | 0.3572 [0.1465, 0.6058] SEP |
| seg4 | straight | 2041 | 121 | 1.3280 | 1.1386 | 0.1894 [0.0464, 0.4136] | **SEP** | 0.2319 [0.0866, 0.4568] SEP |
| seg4 | curved | 1370 | 87 | 6.1294 | 5.6779 | 0.4515 [-0.1076, 1.0330] |  | 0.5439 [0.0571, 1.0927] SEP |
| seg5 | all | 3353 | 139 | 4.5244 | 4.2807 | 0.2437 [-0.0579, 0.5502] |  | 0.5181 [0.2458, 0.8257] SEP |
| seg5 | straight | 2015 | 121 | 1.3294 | 1.3087 | 0.0207 [-0.2005, 0.1828] |  | 0.2873 [0.1177, 0.5292] SEP |
| seg5 | curved | 1338 | 88 | 9.3361 | 8.7564 | 0.5797 [-0.1034, 1.2836] |  | 0.8658 [0.2385, 1.5201] SEP |
| seg6 | all | 3311 | 138 | 6.1911 | 5.9044 | 0.2867 [-0.0765, 0.6294] |  | 0.5107 [0.1943, 0.8534] SEP |
| seg6 | straight | 2000 | 120 | 1.6929 | 1.7283 | -0.0354 [-0.2800, 0.1497] |  | 0.3152 [0.1066, 0.5747] SEP |
| seg6 | curved | 1310 | 88 | 13.0599 | 12.2845 | 0.7754 [-0.0459, 1.6004] |  | 0.8065 [0.0673, 1.5491] SEP |

### S6 -- EMA-10.5k - EMA-5k: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3490 | 139 | 0.9343 | 0.7145 | 0.2198 [-0.0005, 0.6411] |  | 0.1166 [-0.0004, 0.3331] |
| seg1 | straight | 2071 | 122 | 0.6464 | 0.3744 | 0.2720 [0.0019, 0.8444] | **SEP** | 0.2718 [-0.0012, 0.8457] |
| seg1 | curved | 1407 | 88 | 1.2333 | 1.0908 | 0.1425 [-0.0182, 0.4291] |  | -0.1097 [-0.4035, 0.0562] |
| seg2 | all | 3481 | 139 | 1.2977 | 1.5324 | -0.2347 [-0.7738, 0.0626] |  | 0.0235 [-0.1076, 0.1262] |
| seg2 | straight | 2074 | 122 | 0.5975 | 1.0223 | -0.4249 [-1.3747, 0.0319] |  | -0.0452 [-0.2510, 0.0708] |
| seg2 | curved | 1407 | 88 | 2.3300 | 2.2844 | 0.0456 [-0.0846, 0.1555] |  | 0.1246 [0.0001, 0.2700] SEP |
| seg3 | all | 3448 | 139 | 2.1607 | 2.1829 | -0.0222 [-0.1826, 0.1178] |  | 0.1705 [0.0113, 0.3213] SEP |
| seg3 | straight | 2057 | 121 | 1.0091 | 1.1118 | -0.1027 [-0.3173, 0.0195] |  | 0.0514 [-0.1581, 0.1865] |
| seg3 | curved | 1391 | 88 | 3.8638 | 3.7669 | 0.0969 [-0.1645, 0.3420] |  | 0.3466 [0.1139, 0.6284] SEP |
| seg4 | all | 3396 | 139 | 3.2190 | 3.0601 | 0.1589 [-0.0750, 0.4307] |  | 0.2548 [-0.0296, 0.5025] |
| seg4 | straight | 2038 | 121 | 1.3233 | 1.1263 | 0.1970 [-0.0342, 0.6050] |  | 0.0362 [-0.3748, 0.2798] |
| seg4 | curved | 1358 | 87 | 6.0638 | 5.9622 | 0.1017 [-0.2986, 0.4932] |  | 0.5828 [0.1959, 1.0263] SEP |
| seg5 | all | 3346 | 139 | 4.5164 | 4.4235 | 0.0929 [-0.1390, 0.3115] |  | 0.3522 [0.0270, 0.6460] SEP |
| seg5 | straight | 2013 | 121 | 1.3212 | 1.3035 | 0.0177 [-0.1061, 0.1215] |  | 0.1224 [-0.2940, 0.3897] |
| seg5 | curved | 1333 | 88 | 9.3415 | 9.1350 | 0.2064 [-0.3037, 0.7102] |  | 0.6994 [0.1819, 1.2767] SEP |
| seg6 | all | 3300 | 138 | 6.1976 | 6.0715 | 0.1261 [-0.1015, 0.3709] |  | 0.4436 [0.1842, 0.7321] SEP |
| seg6 | straight | 1997 | 120 | 1.6831 | 1.7033 | -0.0202 [-0.1674, 0.1017] |  | 0.3082 [0.1560, 0.4616] SEP |
| seg6 | curved | 1303 | 88 | 13.1165 | 12.7663 | 0.3502 [-0.2007, 0.9065] |  | 0.6511 [0.0271, 1.3143] SEP |

### S6 -- EMA-5k - refcv7-50400: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift

| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |
|---|---|---|---|---|---|---|---|---|
| seg1 | all | 3499 | 139 | 0.7139 | 0.8147 | -0.1007 [-0.3129, 0.0156] |  | -0.1473 [-0.3616, -0.0235] SEP |
| seg1 | straight | 2072 | 122 | 0.3745 | 0.5433 | -0.1689 [-0.5493, 0.0131] |  | -0.2208 [-0.6070, -0.0363] SEP |
| seg1 | curved | 1410 | 88 | 1.0897 | 1.0917 | -0.0020 [-0.0479, 0.0462] |  | -0.0416 [-0.1049, 0.0210] |
| seg2 | all | 3485 | 139 | 1.5335 | 1.4599 | 0.0737 [-0.0206, 0.2012] |  | -0.0197 [-0.1160, 0.1046] |
| seg2 | straight | 2076 | 122 | 1.0214 | 0.8386 | 0.1827 [-0.0095, 0.5655] |  | -0.0779 [-0.1200, -0.0426] SEP |
| seg2 | curved | 1409 | 88 | 2.2882 | 2.3752 | -0.0870 [-0.3698, 0.1342] |  | 0.0659 [-0.1761, 0.3528] |
| seg3 | all | 3458 | 139 | 2.1877 | 2.0612 | 0.1266 [0.0013, 0.2739] | **SEP** | 0.2843 [-0.1610, 1.0533] |
| seg3 | straight | 2062 | 121 | 1.1110 | 0.9771 | 0.1339 [0.0068, 0.3535] | **SEP** | 0.3505 [-0.1424, 1.2954] |
| seg3 | curved | 1396 | 88 | 3.7782 | 3.6624 | 0.1158 [-0.1063, 0.3638] |  | 0.1864 [-0.2853, 0.7581] |
| seg4 | all | 3412 | 139 | 3.0729 | 2.9680 | 0.1048 [-0.0503, 0.2594] |  | 0.1403 [-0.1373, 0.5042] |
| seg4 | straight | 2048 | 121 | 1.1405 | 1.1380 | 0.0025 [-0.2122, 0.1648] |  | 0.1852 [-0.1518, 0.7673] |
| seg4 | curved | 1364 | 87 | 5.9742 | 5.7157 | 0.2585 [-0.0311, 0.5955] |  | 0.0729 [-0.3150, 0.4498] |
| seg5 | all | 3363 | 139 | 4.4549 | 4.3053 | 0.1496 [-0.0426, 0.3543] |  | 0.1626 [-0.1800, 0.5764] |
| seg5 | straight | 2019 | 121 | 1.3218 | 1.3072 | 0.0146 [-0.2184, 0.2112] |  | 0.1545 [-0.2090, 0.7653] |
| seg5 | curved | 1344 | 88 | 9.1615 | 8.8091 | 0.3524 [-0.0267, 0.7608] |  | 0.1747 [-0.4080, 0.7129] |
| seg6 | all | 3325 | 138 | 6.0944 | 5.9267 | 0.1677 [-0.0589, 0.3944] |  | 0.0769 [-0.2255, 0.3965] |
| seg6 | straight | 2006 | 120 | 1.7029 | 1.7264 | -0.0235 [-0.2615, 0.1769] |  | 0.0006 [-0.2025, 0.2650] |
| seg6 | curved | 1317 | 88 | 12.7719 | 12.3079 | 0.4640 [0.0002, 0.9878] | **SEP** | 0.1885 [-0.5509, 0.8637] |

### S6 -- EMA-10.5k - refcv7-50400: turn x speed cross

| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |
|---|---|---|---|---|---|---|---|
| straight <0.02 | v0 <5 | 364 | 39 | 5.2804 | 5.1725 | 0.1079 [-0.0128, 0.2616] |  |  |
| straight <0.02 | v0 5-15 | 1050 | 84 | 0.8962 | 0.7632 | 0.1330 [0.0229, 0.2452] | **SEP** |  |
| straight <0.02 | v0 >=15 | 684 | 36 | 0.5211 | 0.5523 | -0.0313 [-0.0778, 0.0139] |  |  |
| gentle 0.02-0.06 | v0 <5 (low n) | 98 | 22 | 4.0855 | 5.5283 | -1.4428 [-4.8081, 0.2240] |  |  |
| gentle 0.02-0.06 | v0 5-15 | 424 | 59 | 2.5106 | 2.5310 | -0.0203 [-0.2527, 0.2711] |  |  |
| gentle 0.02-0.06 | v0 >=15 (low n) | 153 | 14 | 2.3448 | 2.1952 | 0.1496 [-0.0180, 0.3548] |  |  |
| turn 0.06-0.15 | v0 <5 (low n) | 64 | 20 | 10.5305 | 11.2985 | -0.7681 [-1.6810, -0.0519] | **SEP** |  |
| turn 0.06-0.15 | v0 5-15 | 293 | 43 | 6.0360 | 5.8272 | 0.2088 [-0.1938, 0.5790] |  |  |
| turn 0.06-0.15 | v0 >=15 (low n) | 78 | 10 | 4.7789 | 4.3281 | 0.4508 [0.0556, 0.8923] | **SEP** |  |
| sharp >=0.15 | v0 <5 (low n) | 70 | 16 | 15.1518 | 14.3016 | 0.8502 [-1.5780, 4.5818] |  |  |
| sharp >=0.15 | v0 5-15 | 225 | 26 | 12.4270 | 10.7541 | 1.6730 [0.2117, 3.2906] | **SEP** |  |
| sharp >=0.15 | v0 >=15 (low n) | 9 | 3 | 5.5475 | 6.3909 | -0.8433 [-1.7868, 1.9949] |  |  |
| undef (GT crawling) | v0 <5 (low n) | 13 | 11 | 14.7134 | 14.2040 | 0.5095 [0.0545, 1.2103] | **SEP** |  |
| undef (GT crawling) | v0 5-15 | 0 | 0 | | | | | |
| undef (GT crawling) | v0 >=15 | 0 | 0 | | | | | |

### S6 -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)

| bin | EMA-10.5k s0-s1 [CI] | EMA-5k s0-s1 [CI] | refcv7-50400 s0-s1 [CI] |
|---|---|---|---|
| overall | -0.0064 [-0.0498, 0.0457] | -0.0153 [-0.0751, 0.0424] | 0.0128 [-0.0357, 0.0621] |
| turn|straight <0.02 | 0.0000 [-0.0297, 0.0321] | -0.0024 [-0.0263, 0.0190] | -0.0216 [-0.0640, 0.0031] |
| turn|gentle 0.02-0.06 | 0.0243 [-0.1031, 0.2032] | 0.0493 [-0.0553, 0.1670] | 0.0713 [-0.0127, 0.1814] |
| turn|turn 0.06-0.15 | -0.0237 [-0.1328, 0.0888] | 0.0108 [-0.1388, 0.1615] | 0.0611 [-0.0773, 0.2153] |
| turn|sharp >=0.15 | -0.1128 [-0.4015, 0.1588] | -0.2362 [-0.7709, 0.2315] | 0.2004 [-0.1315, 0.6371] |
| turn|undef (GT crawling) | 0.4538 [0.0969, 0.9207] SEP | -0.7635 [-1.9674, 0.4258] | -1.9278 [-4.4732, 0.0225] |
| speed|v0 <2 | -0.0574 [-0.2403, 0.1549] | 0.2812 [0.0192, 0.7088] SEP | 0.0328 [-0.3336, 0.3737] |
| speed|v0 2-5 | 0.0116 [-0.2115, 0.3108] | -0.1594 [-0.3925, 0.0077] | 0.2009 [-0.0453, 0.5276] |
| speed|v0 5-10 | 0.0409 [-0.0569, 0.1368] | -0.0367 [-0.1984, 0.0987] | -0.0243 [-0.1262, 0.0532] |
| speed|v0 10-15 | -0.0616 [-0.1130, -0.0170] SEP | -0.0091 [-0.0758, 0.0548] | -0.0268 [-0.0598, 0.0033] |
| speed|v0 15-25 | -0.0007 [-0.0271, 0.0258] | 0.0031 [-0.0236, 0.0284] | 0.0055 [-0.0167, 0.0282] |
| speed|v0 >=25 | 0.0003 [-0.0086, 0.0111] | 0.0016 [-0.0069, 0.0122] | 0.0008 [-0.0108, 0.0140] |
| nav|nav=follow | -0.0193 [-0.0557, 0.0184] | 0.0235 [-0.0170, 0.0654] | 0.0080 [-0.0141, 0.0313] |
| nav|nav=left | -0.0334 [-0.1654, 0.1030] | 0.0874 [-0.0438, 0.2535] | 0.0140 [-0.3152, 0.2886] |
| nav|nav=right | 0.0332 [-0.1061, 0.1944] | -0.1398 [-0.3317, 0.0343] | 0.0235 [-0.1175, 0.1775] |

### S6 -- EMA-10.5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1454 | 0.1323 / 0.1822 | False / True / False | 0.0064 / 0.0128 | 11.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0751 | 0.0878 / 0.0624 | True / False / True | 0.0000 / 0.0216 | 3.5 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|gentle 0.02-0.06 | -0.1883 | -0.2123 / -0.1643 | False / False / False | 0.0243 / 0.0713 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.1085 | 0.0745 / 0.1424 | False / False / False | 0.0237 / 0.0611 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 1.4090 | 1.2607 / 1.8208 | False / True / True | 0.1128 / 0.2004 | 7.0 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|undef (GT crawling) | 0.5095 | 0.7430 / 0.2759 | True / False / True | 0.4538 / 1.9278 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.1821 | -0.2976 / 0.3771 | False / False / False | 0.0574 / 0.0328 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.1211 | -0.2157 / -0.0264 | False / False / False | 0.0116 / 0.2009 | 0.6 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.4036 | 0.4362 / 0.3710 | True / True / True | 0.0409 / 0.0243 | 9.9 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| speed|v0 10-15 | 0.1564 | 0.1390 / 0.1738 | False / False / False | 0.0616 / 0.0268 | 2.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0679 | 0.0648 / 0.0710 | False / False / False | 0.0007 / 0.0055 | 12.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0696 | -0.0698 / -0.0693 | False / False / False | 0.0003 / 0.0008 | 83.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.1818 | 0.1672 / 0.1963 | True / True / True | 0.0193 / 0.0080 | 9.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=left | -0.0467 | -0.0598 / 0.2267 | False / False / False | 0.0334 / 0.0140 | 1.4 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1241 | 0.1143 / 0.1340 | False / False / False | 0.0332 / 0.0235 | 3.7 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA-10.5k - EMA-5k: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.0403 | 0.0429 / 0.0469 | False / False / False | 0.0064 / 0.0153 | 2.6 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|straight <0.02 | 0.0075 | 0.0093 / 0.0061 | False / False / False | 0.0000 / 0.0024 | 3.2 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.1873 | -0.1998 / -0.1747 | False / False / False | 0.0243 / 0.0493 | 3.8 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | -0.0404 | -0.0571 / -0.0238 | False / False / False | 0.0237 / 0.0108 | 1.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.8799 | 0.9459 / 0.9127 | False / True / False | 0.1128 / 0.2362 | 3.7 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | 0.2235 | 0.2484 / 0.1985 | False / True / True | 0.4538 / 0.7635 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.1473 | -0.3648 / 0.2471 | False / False / False | 0.0574 / 0.2812 | 0.5 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | -0.2936 | -0.2081 / -0.3791 | False / False / False | 0.0116 / 0.1594 | 1.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.2417 | 0.2806 / 0.2029 | False / False / False | 0.0409 / 0.0367 | 5.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.0181 | -0.0081 / 0.0444 | False / False / False | 0.0616 / 0.0091 | 0.3 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0401 | 0.0381 / 0.0420 | False / False / False | 0.0007 / 0.0031 | 12.8 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0116 | -0.0123 / -0.0109 | False / False / False | 0.0003 / 0.0016 | 7.1 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=follow | 0.0871 | 0.0657 / 0.1089 | False / True / True | 0.0193 / 0.0235 | 3.7 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| nav|nav=left | -0.1295 | -0.1961 / 0.0374 | False / False / False | 0.0334 / 0.0874 | 1.5 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | -0.0130 | 0.0696 / -0.0957 | False / False / False | 0.0332 / 0.1398 | 0.1 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- EMA-5k - refcv7-50400: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)

| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |
|---|---|---|---|---|---|---|
| overall | 0.1031 | 0.0901 / 0.1207 | False / True / True | 0.0153 / 0.0128 | 6.7 | CLEARS 2x floor + pooled-separated, not separated at each seed |
| turn|straight <0.02 | 0.0670 | 0.0788 / 0.0542 | True / False / False | 0.0024 / 0.0216 | 3.1 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|gentle 0.02-0.06 | -0.0011 | -0.0125 / 0.0104 | False / False / False | 0.0493 / 0.0713 | 0.0 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|turn 0.06-0.15 | 0.1463 | 0.1163 / 0.1621 | False / False / False | 0.0108 / 0.0611 | 2.4 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|sharp >=0.15 | 0.5421 | 0.3322 / 0.8313 | False / True / False | 0.2362 / 0.2004 | 2.3 | NOT PROVEN (not separated and/or within 2x floor) |
| turn|undef (GT crawling) | -0.2931 | 0.4706 / -1.0569 | False / False / False | 0.7635 / 1.9278 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 <2 | -0.0613 | 0.0806 / -0.1180 | False / False / False | 0.2812 / 0.0328 | 0.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 2-5 | 0.1725 | -0.0077 / 0.3526 | False / False / False | 0.1594 / 0.2009 | 0.9 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 5-10 | 0.1619 | 0.1557 / 0.1681 | False / False / False | 0.0367 / 0.0243 | 4.4 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 10-15 | 0.1383 | 0.1471 / 0.1294 | False / False / False | 0.0091 / 0.0268 | 5.2 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 15-25 | 0.0278 | 0.0267 / 0.0290 | False / False / False | 0.0031 / 0.0055 | 5.1 | NOT PROVEN (not separated and/or within 2x floor) |
| speed|v0 >=25 | -0.0579 | -0.0575 / -0.0583 | True / True / True | 0.0016 / 0.0008 | 35.4 | CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds) |
| nav|nav=follow | 0.0924 | 0.0991 / 0.0851 | False / False / False | 0.0235 / 0.0080 | 3.9 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=left | 0.0855 | 0.1340 / 0.1149 | False / False / False | 0.0874 / 0.0140 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |
| nav|nav=right | 0.1338 | 0.0545 / 0.2055 | False / False / False | 0.1398 / 0.0235 | 1.0 | NOT PROVEN (not separated and/or within 2x floor) |

### S6 -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms

**curved (GT |yaw| >= 0.02)**

* seg1 (n 1407): EMA-10.5k bias -0.157, refcv7-50400 bias +0.166, EMA-5k bias -0.008; EMA-10.5k MAE 1.233, refcv7-50400 MAE 0.965, EMA-5k MAE 1.091
* seg2 (n 1407): EMA-10.5k bias -0.324, refcv7-50400 bias -0.233, EMA-5k bias -0.235; EMA-10.5k MAE 2.330, refcv7-50400 MAE 2.372, EMA-5k MAE 2.284
* seg3 (n 1391): EMA-10.5k bias -1.263, refcv7-50400 bias -1.102, EMA-5k bias -1.032; EMA-10.5k MAE 3.864, refcv7-50400 MAE 3.651, EMA-5k MAE 3.767
* seg4 (n 1358): EMA-10.5k bias -3.112, refcv7-50400 bias -2.491, EMA-5k bias -2.716; EMA-10.5k MAE 6.064, refcv7-50400 MAE 5.706, EMA-5k MAE 5.962
* seg5 (n 1333): EMA-10.5k bias -4.988, refcv7-50400 bias -4.438, EMA-5k bias -4.489; EMA-10.5k MAE 9.341, refcv7-50400 MAE 8.768, EMA-5k MAE 9.135
* seg6 (n 1303): EMA-10.5k bias -7.001, refcv7-50400 bias -6.811, EMA-5k bias -6.543; EMA-10.5k MAE 13.117, refcv7-50400 MAE 12.298, EMA-5k MAE 12.766
**sharp (>= 0.15)**

* seg1 (n 302): EMA-10.5k bias -0.728, refcv7-50400 bias +0.176, EMA-5k bias -0.028; EMA-10.5k MAE 1.982, refcv7-50400 MAE 1.443, EMA-5k MAE 1.366
* seg2 (n 299): EMA-10.5k bias -1.675, refcv7-50400 bias -0.535, EMA-5k bias -1.152; EMA-10.5k MAE 4.180, refcv7-50400 MAE 3.812, EMA-5k MAE 3.827
* seg3 (n 296): EMA-10.5k bias -4.812, refcv7-50400 bias -2.461, EMA-5k bias -3.561; EMA-10.5k MAE 7.870, refcv7-50400 MAE 6.639, EMA-5k MAE 7.049
* seg4 (n 294): EMA-10.5k bias -9.625, refcv7-50400 bias -6.361, EMA-5k bias -7.513; EMA-10.5k MAE 13.892, refcv7-50400 MAE 11.604, EMA-5k MAE 12.708
* seg5 (n 290): EMA-10.5k bias -14.669, refcv7-50400 bias -11.457, EMA-5k bias -12.092; EMA-10.5k MAE 21.716, refcv7-50400 MAE 18.726, EMA-5k MAE 19.956
* seg6 (n 285): EMA-10.5k bias -19.190, refcv7-50400 bias -16.901, EMA-5k bias -16.958; EMA-10.5k MAE 28.371, refcv7-50400 MAE 24.758, EMA-5k MAE 26.030
**curved & v0 5-10**

* seg1 (n 547): EMA-10.5k bias +0.018, refcv7-50400 bias +0.049, EMA-5k bias +0.052; EMA-10.5k MAE 0.959, refcv7-50400 MAE 0.896, EMA-5k MAE 0.900
* seg2 (n 547): EMA-10.5k bias -0.639, refcv7-50400 bias -0.516, EMA-5k bias -0.472; EMA-10.5k MAE 2.509, refcv7-50400 MAE 2.231, EMA-5k MAE 2.292
* seg3 (n 544): EMA-10.5k bias -2.006, refcv7-50400 bias -1.675, EMA-5k bias -1.543; EMA-10.5k MAE 4.411, refcv7-50400 MAE 3.806, EMA-5k MAE 3.914
* seg4 (n 522): EMA-10.5k bias -4.316, refcv7-50400 bias -3.616, EMA-5k bias -3.393; EMA-10.5k MAE 6.972, refcv7-50400 MAE 6.311, EMA-5k MAE 6.328
* seg5 (n 504): EMA-10.5k bias -6.565, refcv7-50400 bias -5.918, EMA-5k bias -5.392; EMA-10.5k MAE 10.515, refcv7-50400 MAE 9.460, EMA-5k MAE 9.519
* seg6 (n 479): EMA-10.5k bias -8.453, refcv7-50400 bias -8.259, EMA-5k bias -7.349; EMA-10.5k MAE 14.218, refcv7-50400 MAE 12.653, EMA-5k MAE 12.920
**sharp & v0 5-10**

* seg1 (n 139): EMA-10.5k bias -0.094, refcv7-50400 bias +0.062, EMA-5k bias +0.031; EMA-10.5k MAE 1.139, refcv7-50400 MAE 1.251, EMA-5k MAE 1.109
* seg2 (n 139): EMA-10.5k bias -2.182, refcv7-50400 bias -1.267, EMA-5k bias -1.331; EMA-10.5k MAE 4.040, refcv7-50400 MAE 3.564, EMA-5k MAE 3.402
* seg3 (n 138): EMA-10.5k bias -6.570, refcv7-50400 bias -4.253, EMA-5k bias -4.386; EMA-10.5k MAE 8.727, refcv7-50400 MAE 6.990, EMA-5k MAE 6.905
* seg4 (n 137): EMA-10.5k bias -12.488, refcv7-50400 bias -9.127, EMA-5k bias -8.767; EMA-10.5k MAE 15.569, refcv7-50400 MAE 12.522, EMA-5k MAE 12.675
* seg5 (n 136): EMA-10.5k bias -17.835, refcv7-50400 bias -14.401, EMA-5k bias -13.095; EMA-10.5k MAE 24.383, refcv7-50400 MAE 19.814, EMA-5k MAE 20.088
* seg6 (n 131): EMA-10.5k bias -20.659, refcv7-50400 bias -18.177, EMA-5k bias -15.844; EMA-10.5k MAE 30.890, refcv7-50400 MAE 25.484, EMA-5k MAE 25.890

### S6 -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh

| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |
|---|---|---|---|---|
| curved (GT |yaw| >= 0.02) | last segment | 1303 / 1303 | 26.2 | EMA-10.5k -0.2597 [-0.3547, -0.1759]; refcv7-50400 -0.2421 [-0.3109, -0.1764]; EMA-5k -0.2332 [-0.3144, -0.1616] | EMA-10.5k -0.0176 [-0.0622, +0.0161]; EMA-5k +0.0088 [-0.0214, +0.0347] |
| curved (GT |yaw| >= 0.02) | segments 2..K | 1414 / 6792 | 16.4 | EMA-10.5k -0.1880 [-0.2850, -0.1049]; refcv7-50400 -0.1753 [-0.2392, -0.1183]; EMA-5k -0.1566 [-0.2418, -0.0826] | EMA-10.5k -0.0127 [-0.0661, +0.0461]; EMA-5k +0.0187 [-0.0242, +0.0715] |
| sharp (>= 0.15) | last segment | 285 / 285 | 63.1 | EMA-10.5k -0.2886 [-0.4125, -0.1910]; refcv7-50400 -0.2573 [-0.3403, -0.1817]; EMA-5k -0.2559 [-0.3590, -0.1726] | EMA-10.5k -0.0313 [-0.0923, +0.0111]; EMA-5k +0.0014 [-0.0375, +0.0302] |
| sharp (>= 0.15) | segments 2..K | 304 / 1464 | 39.9 | EMA-10.5k -0.2418 [-0.3628, -0.1462]; refcv7-50400 -0.1895 [-0.2723, -0.1201]; EMA-5k -0.2014 [-0.3057, -0.1206] | EMA-10.5k -0.0523 [-0.1177, -0.0097] SEP; EMA-5k -0.0119 [-0.0520, +0.0169] |
| curved & v0 5-10 | last segment | 479 / 479 | 28.5 | EMA-10.5k -0.2900 [-0.4253, -0.1602]; refcv7-50400 -0.2623 [-0.3827, -0.1674]; EMA-5k -0.2382 [-0.3570, -0.1365] | EMA-10.5k -0.0277 [-0.1088, +0.0436]; EMA-5k +0.0241 [-0.0134, +0.0606] |
| curved & v0 5-10 | segments 2..K | 547 / 2596 | 17.9 | EMA-10.5k -0.2442 [-0.3651, -0.1351]; refcv7-50400 -0.2015 [-0.3145, -0.1179]; EMA-5k -0.1887 [-0.2967, -0.1057] | EMA-10.5k -0.0427 [-0.1180, +0.0200]; EMA-5k +0.0128 [-0.0204, +0.0440] |
| sharp & v0 5-10 | last segment | 131 / 131 | 67.2 | EMA-10.5k -0.3066 [-0.4629, -0.1616]; refcv7-50400 -0.2613 [-0.3923, -0.1569]; EMA-5k -0.2406 [-0.3752, -0.1275] | EMA-10.5k -0.0453 [-0.1487, +0.0374]; EMA-5k +0.0207 [-0.0237, +0.0616] |
| sharp & v0 5-10 | segments 2..K | 139 / 681 | 43.9 | EMA-10.5k -0.2683 [-0.4156, -0.1377]; refcv7-50400 -0.2068 [-0.3341, -0.1122]; EMA-5k -0.1981 [-0.3248, -0.0999] | EMA-10.5k -0.0615 [-0.1593, +0.0138]; EMA-5k +0.0088 [-0.0328, +0.0474] |

### S6 -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)

**EMA-10.5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3525 | 0.1454 | 0.02354 | 0.03780 | 0.543 / 0.457 / 0.000 | 697.7 / 512.5 |
| turn|straight <0.02 | 2098 | 0.0751 | 0.02532 | 0.03289 | 0.564 / 0.436 / 0.000 | 139.0 / 157.5 |
| turn|gentle 0.02-0.06 | 675 | -0.1883 | -0.00741 | -0.03732 | 0.480 / 0.520 / 0.000 | 63.9 / -127.1 |
| turn|turn 0.06-0.15 | 435 | 0.1085 | 0.08640 | 0.13987 | 0.540 / 0.460 / 0.000 | 47.7 / 47.2 |
| turn|sharp >=0.15 | 304 | 1.4090 | 0.12766 | 0.49622 | 0.536 / 0.464 / 0.000 | 165.8 / 428.3 |
| turn|undef (GT crawling) | 13 | 0.5095 | 0.28851 | 0.25924 | 0.692 / 0.308 / 0.000 | 4.1 / 6.6 |
| speed|v0 <2 | 187 | -0.1821 | -0.02632 | 0.00017 | 0.492 / 0.508 / 0.000 | 24.2 / -34.0 |
| speed|v0 2-5 | 422 | -0.1211 | 0.06449 | 0.11784 | 0.562 / 0.438 / 0.000 | 102.9 / -51.1 |
| speed|v0 5-10 | 1040 | 0.4036 | 0.03626 | 0.05879 | 0.542 / 0.458 / 0.000 | 346.9 / 419.7 |
| speed|v0 10-15 | 952 | 0.1564 | 0.04183 | 0.04775 | 0.577 / 0.423 / 0.000 | 107.5 / 148.9 |
| speed|v0 15-25 | 679 | 0.0679 | 0.01194 | 0.01205 | 0.539 / 0.461 / 0.000 | 37.2 / 46.1 |
| speed|v0 >=25 | 245 | -0.0696 | -0.02818 | -0.03919 | 0.433 / 0.567 / 0.000 | 1.9 / -17.0 |
| nav|nav=follow | 2249 | 0.1818 | 0.01702 | 0.02416 | 0.532 / 0.468 / 0.000 | 350.2 / 408.8 |
| nav|nav=left | 320 | -0.0467 | 0.06156 | 0.10276 | 0.591 / 0.409 / 0.000 | 53.0 / -14.9 |
| nav|nav=right | 956 | 0.1241 | 0.03795 | 0.04970 | 0.553 / 0.447 / 0.000 | 281.0 / 118.7 |
**EMA-10.5k - EMA-5k**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3523 | 0.0403 | 0.01224 | 0.01964 | 0.523 / 0.477 / 0.000 | 512.6 / 141.9 |
| turn|straight <0.02 | 2097 | 0.0075 | 0.01154 | 0.01460 | 0.530 / 0.470 / 0.000 | 96.1 / 15.7 |
| turn|gentle 0.02-0.06 | 675 | -0.1873 | -0.00091 | 0.00918 | 0.496 / 0.504 / 0.000 | 32.3 / -126.4 |
| turn|turn 0.06-0.15 | 435 | -0.0404 | 0.03250 | 0.03921 | 0.526 / 0.474 / 0.000 | 31.8 / -17.6 |
| turn|sharp >=0.15 | 304 | 0.8799 | 0.05301 | 0.13464 | 0.520 / 0.480 / 0.000 | 132.4 / 267.5 |
| turn|undef (GT crawling) | 12 | 0.2235 | 0.26869 | 0.26175 | 0.833 / 0.167 / 0.000 | 0.9 / 2.7 |
| speed|v0 <2 | 185 | -0.1473 | -0.00691 | 0.00402 | 0.481 / 0.519 / 0.000 | 20.6 / -27.2 |
| speed|v0 2-5 | 422 | -0.2936 | 0.04320 | 0.04533 | 0.536 / 0.464 / 0.000 | 36.6 / -123.9 |
| speed|v0 5-10 | 1040 | 0.2417 | 0.01487 | 0.03173 | 0.517 / 0.483 / 0.000 | 286.9 / 251.4 |
| speed|v0 10-15 | 952 | 0.0181 | 0.01251 | 0.01421 | 0.526 / 0.474 / 0.000 | 78.4 / 17.3 |
| speed|v0 15-25 | 679 | 0.0401 | 0.01732 | 0.01834 | 0.548 / 0.452 / 0.000 | 20.9 / 27.2 |
| speed|v0 >=25 | 245 | -0.0116 | -0.00616 | -0.01148 | 0.482 / 0.518 / 0.000 | 1.6 / -2.8 |
| nav|nav=follow | 2247 | 0.0871 | 0.01005 | 0.01739 | 0.523 / 0.477 / 0.000 | 241.4 / 195.7 |
| nav|nav=left | 320 | -0.1295 | 0.02120 | 0.01144 | 0.531 / 0.469 / 0.000 | 35.0 / -41.4 |
| nav|nav=right | 956 | -0.0130 | 0.01439 | 0.02583 | 0.522 / 0.478 / 0.000 | 209.6 / -12.5 |
**EMA-5k - refcv7-50400**

| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |
|---|---|---|---|---|---|---|
| ALL | 3541 | 0.1031 | 0.01419 | 0.00679 | 0.528 / 0.472 / 0.000 | 517.2 / 365.0 |
| turn|straight <0.02 | 2103 | 0.0670 | 0.02291 | 0.01520 | 0.552 / 0.448 / 0.000 | 171.4 / 141.0 |
| turn|gentle 0.02-0.06 | 675 | -0.0011 | -0.02399 | -0.04162 | 0.474 / 0.526 / 0.000 | 61.9 / -0.7 |
| turn|turn 0.06-0.15 | 437 | 0.1463 | 0.02104 | 0.07108 | 0.519 / 0.481 / 0.000 | 48.6 / 63.9 |
| turn|sharp >=0.15 | 307 | 0.5421 | -0.03805 | 0.25309 | 0.492 / 0.508 / 0.000 | 122.1 / 166.4 |
| turn|undef (GT crawling) | 19 | -0.2931 | 0.00301 | -0.01160 | 0.579 / 0.421 / 0.000 | 1.0 / -5.6 |
| speed|v0 <2 | 203 | -0.0613 | -0.01959 | -0.03344 | 0.473 / 0.527 / 0.000 | 17.3 / -12.4 |
| speed|v0 2-5 | 422 | 0.1725 | 0.04604 | 0.06773 | 0.550 / 0.450 / 0.000 | 106.0 / 72.8 |
| speed|v0 5-10 | 1040 | 0.1619 | 0.02298 | 0.01830 | 0.530 / 0.470 / 0.000 | 211.6 / 168.4 |
| speed|v0 10-15 | 952 | 0.1383 | 0.03129 | 0.02136 | 0.579 / 0.421 / 0.000 | 101.0 / 131.6 |
| speed|v0 15-25 | 679 | 0.0278 | -0.00051 | -0.00854 | 0.499 / 0.501 / 0.000 | 37.7 / 18.9 |
| speed|v0 >=25 | 245 | -0.0579 | -0.02692 | -0.03108 | 0.408 / 0.592 / 0.000 | 1.4 / -14.2 |
| nav|nav=follow | 2251 | 0.0924 | 0.00301 | -0.00746 | 0.506 / 0.494 / 0.000 | 242.1 / 207.9 |
| nav|nav=left | 322 | 0.0855 | 0.03608 | 0.07483 | 0.584 / 0.416 / 0.000 | 50.6 / 27.5 |
| nav|nav=right | 968 | 0.1338 | 0.03718 | 0.02930 | 0.560 / 0.440 / 0.000 | 211.4 / 129.6 |

### S6 -- the episodes carrying the summed delta (panel file index only)

**EMA-10.5k - refcv7-50400** (sum 512.5 over 139 episodes)

* ep111: n 26, share 0.266, mean delta 5.2389, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep080: n 26, share 0.226, mean delta 4.4648, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep013: n 27, share 0.174, mean delta 3.3096, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.149, mean delta 2.9453, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep012: n 26, share 0.102, mean delta 2.0024, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep036: n 26, share 0.086, mean delta 1.7029, sharp-window frac 0.27, curved frac 0.65, mean v0 12.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
**EMA-10.5k - EMA-5k** (sum 141.9 over 139 episodes)

* ep080: n 26, share 0.713, mean delta 3.8901, sharp-window frac 0.38, curved frac 0.54, mean v0 10.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep111: n 26, share 0.548, mean delta 2.9906, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep122: n 26, share 0.377, mean delta 2.0588, sharp-window frac 0.38, curved frac 0.73, mean v0 8.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep012: n 26, share 0.265, mean delta 1.4474, sharp-window frac 0.46, curved frac 0.77, mean v0 7.0, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 26}
* ep036: n 26, share 0.254, mean delta 1.3864, sharp-window frac 0.27, curved frac 0.65, mean v0 12.4, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep028: n 27, share 0.182, mean delta 0.9568, sharp-window frac 0.33, curved frac 0.96, mean v0 5.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
**EMA-5k - refcv7-50400** (sum 365.0 over 139 episodes)

* ep013: n 27, share 0.248, mean delta 3.3550, sharp-window frac 0.11, curved frac 0.26, mean v0 7.3, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep085: n 26, share 0.164, mean delta 2.3032, sharp-window frac 0.85, curved frac 1.00, mean v0 11.5, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep111: n 26, share 0.160, mean delta 2.2483, sharp-window frac 0.42, curved frac 0.92, mean v0 11.9, nav {'nav=follow': 26, 'nav=left': 0, 'nav=right': 0}
* ep059: n 27, share 0.133, mean delta 1.8028, sharp-window frac 0.00, curved frac 0.00, mean v0 5.8, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep000: n 27, share 0.121, mean delta 1.6405, sharp-window frac 0.33, curved frac 0.67, mean v0 7.5, nav {'nav=follow': 0, 'nav=left': 0, 'nav=right': 27}
* ep100: n 18, share 0.102, mean delta 2.0699, sharp-window frac 0.50, curved frac 1.00, mean v0 5.2, nav {'nav=follow': 0, 'nav=left': 18, 'nav=right': 0}

### S6 -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed

**EMA-10.5k vs EMA-5k|seed0**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1276 / 0.0164 [-0.0056, 0.0377] | 57 / 0.0986 [-0.1037, 0.3620] |
| only new agrees | 323 / -0.5359 [-0.9633, -0.2225] | 21 / -2.0365 [-3.5125, -0.7732] |
| only earlier agrees | 211 / 1.0982 [0.3839, 2.2392] | 22 / 7.6557 [1.7164, 16.3948] |
| neither agrees | 1714 / 0.0418 [-0.1691, 0.2381] | 205 / 0.7669 [-0.3888, 2.0474] |
**EMA-10.5k vs EMA-5k|seed1**

| category | all: n / delta [CI] | sharp: n / delta [CI] |
|---|---|---|
| both agree with oracle | 1259 / 0.0076 [-0.0169, 0.0318] | 66 / -0.0741 [-0.3005, 0.1613] |
| only new agrees | 323 / -0.5375 [-0.9456, -0.2123] | 18 / -3.3222 [-6.4370, -1.0565] |
| only earlier agrees | 232 / 1.3298 [0.5599, 2.3641] | 19 / 10.8412 [2.9139, 18.7797] |
| neither agrees | 1711 / 0.0122 [-0.2118, 0.1927] | 202 / 0.6786 [-0.1736, 1.4573] |

### S6 -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)

**EMA-10.5k vs EMA-5k|seed0**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|
| overall | 3484 | 0.1721 [-0.0134, 0.3537] | 0.2206 [-0.0735, 0.5536] | -0.0619 [-0.2937, 0.1326] |
| turn|straight <0.02 | 2076 | -0.0055 [-0.0696, 0.0632] | -0.0575 [-0.1372, 0.0084] | 0.0520 [-0.0211, 0.1402] |
| turn|gentle 0.02-0.06 | 674 | -0.0765 [-0.3387, 0.1613] | -0.3451 [-0.8539, 0.0238] | 0.2694 [0.0007, 0.6365] SEP |
| turn|turn 0.06-0.15 | 425 | 0.0570 [-0.3001, 0.3767] | 0.1441 [-0.1844, 0.4765] | -0.0874 [-0.4008, 0.1801] |
| turn|sharp >=0.15 | 302 | 2.1150 [0.5477, 3.8549] | 3.4834 [0.8893, 6.7820] | -1.5521 [-3.8008, 0.1138] |
| turn|undef (GT crawling) | 7 | -0.0435 [-0.7224, 0.6428] | -0.0513 [-0.3960, 0.1713] | 0.1196 [-0.6255, 0.8160] |
| speed|v0 <2 | 146 | 0.3626 [-0.1288, 1.0876] | 0.0184 [-0.5053, 0.4379] | 0.0837 [-0.5190, 0.5840] |
| speed|v0 2-5 | 422 | 0.1757 [-0.7928, 1.0442] | 0.7573 [-0.9143, 2.6853] | -0.5816 [-2.0723, 0.5667] |
| speed|v0 5-10 | 1040 | 0.4750 [0.1547, 0.8209] | 0.5257 [0.0316, 1.0835] | -0.0507 [-0.4037, 0.2712] |
| speed|v0 10-15 | 952 | -0.0139 [-0.1278, 0.1037] | -0.0878 [-0.1902, 0.0053] | 0.0739 [-0.0451, 0.2170] |
| speed|v0 15-25 | 679 | -0.0215 [-0.0708, 0.0191] | -0.0227 [-0.0751, 0.0173] | 0.0012 [-0.0361, 0.0380] |
| speed|v0 >=25 | 245 | 0.0246 [-0.0128, 0.0928] | 0.0277 [-0.0188, 0.1225] | -0.0031 [-0.0399, 0.0298] |
**EMA-10.5k vs EMA-5k|seed1**

| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |
|---|---|---|---|---|

### S6 -- selection / goal proxy (new vs earlier, per seed)

* EMA-10.5k vs EMA-5k|seed0: anchor flip rate 0.306; heading delta on flipped windows 0.1082 [-0.3055, 0.5046] (n 1090) vs unflipped 0.0137 [-0.0083, 0.0360]; goal_dist_sel proxy new-earlier 0.0504 [-0.0085, 0.1125]; corr(window dHeading, window dGoalDist) -0.014
* EMA-10.5k vs EMA-5k|seed1: anchor flip rate 0.312; heading delta on flipped windows 0.1229 [-0.2977, 0.5031] (n 1115) vs unflipped 0.0117 [-0.0132, 0.0344]; goal_dist_sel proxy new-earlier 0.0521 [-0.0078, 0.1130]; corr(window dHeading, window dGoalDist) 0.012
