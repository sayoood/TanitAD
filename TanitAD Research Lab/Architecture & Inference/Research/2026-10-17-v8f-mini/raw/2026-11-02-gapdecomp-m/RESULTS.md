# RESULTS: H-GAPDECOMP-M-1, class M of the gap against P10 split by GT path shape and speed (zero training, dev-box CPU)

**MM TO LAND (integration):** copy this package (about 1.0 MB) to `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-02-gapdecomp-m/` and add one claims-register row for H-GAPDECOMP-M-1 (the package does not). Also record, from the same turn, that the hypothesis written into the landed split's next-lever note ("the lateral excess in M lives in the turning windows") is **eliminated** here for the two lateral rows (TURN holds +0.100 / +0.084 of yaw-rate and +0.062 / +0.077 of curvature with 0.074 of the windows; enrichment 0.85 to 1.35; Table 3).

Every number is **MEASURED (ours)** unless marked **ESTIMATED** or **POST-HOC**; artifacts `raw/GAPDECOMP_M.json` (the frozen card's analysis) and `raw/POSTHOC.json`, produced by `gapdecomp_m.py` / `posthoc.py` from the landed per-window arrays (`dagland/gapdecomp/raw/arrays`, imported and sha-pinned, not copied) and the signed plan arrays of `extract_signed.py`. Card: `PREREG_FROZEN.md`, sha256 `1c5dbaf0...d4745`, frozen 2026-11-02T14:19:53+0100 before any subclass number; nothing in it changed after data (`gapdecomp_m.py` refuses to run on a changed card; pytest pins that).
**Tier.** T1-kin (own executed state, logged cameras; a self-action open loop). **Estimator.** Paired episode-cluster bootstrap, B = 2,000, seed 0, the landed draw stream, subclass statistics recomputed per draw. It answers **another draw of episodes only**; it is blind to training variance (`H-ESTIM-SEED-1`): each arm is one training run per seed, so s0 against s1 is the only run-to-run evidence here and a verdict needs both seeds. The planner is deterministic (no inference seed). SOUP is a second arm on the same windows and the same P10, not an independent replication of DAG-OACC.
**Controls.** KA1, KA2 (60 / 60, max deviation 5.3e-15 against the landed class M), KA3, KA4, KA5 (8 pytest tests), KA6 all PASS (Table 0).

## Headline

**Does the class-M gap against P10 sit in one identifiable part of M? Partly: three of the five rows are LOCALISED to the 3 to 8 m/s windows (`speed=3-8`, 413 windows / 68 episodes, 20.3 % of M) at both seeds for DAG-OACC, and the same three for SOUP. The curvature row and ADE6 are DIFFUSE as committed. Path shape localises nothing.**

| row (DAG-OACC vs P10, class M) | share of Delta_M in `speed=3-8`, s0 [95 %] | s1 [95 %] | verdict |
|---|---|---|---|
| yaw-rate MAE | +0.653 [+0.510, +0.828] | +0.684 [+0.516, +0.888] | **LOCALISED** (w 0.204) |
| L2 curvature MAE (GT>=5.2) | +0.934 [+0.805, +0.984] | +0.915 [+0.765, +0.983] | **LOCALISED** (w 0.169) |
| M1 crawl share | +0.792 [+0.613, +0.910] | +0.751 [+0.568, +0.879] | **LOCALISED** (w 0.169) |
| curvature MAE | +0.473 [+0.312, +0.684] | +0.549 [+0.387, +0.760] | **DIFFUSE** (s0 point 0.473 misses the committed 0.5; s1 qualifies) |
| ADE6 (a gain: Delta_M -1.24 / -1.18) | no eligible candidate reaches 0.5; best `cross=STRAIGHT/>15` +0.350 / +0.362 (w 0.247) | | **DIFFUSE** (`speed=>15`, w 0.304 just above the 0.30 cap, holds +0.451 / +0.473, also below 0.5) |

* **Selection scope.** 8 eligible candidates per row; the named subclass is the argmax-share eligible candidate in 1.000 / 1.000 of the draws (yaw-rate, L2) and 1.000 / 0.998 (M1) for DAG-OACC. The interval is per candidate and does not carry the choice among the 8; no multiplicity adjustment (card).
* **SOUP (same rule, stated separately):** yaw-rate +0.677 [+0.558, +0.805] / +0.635 [+0.474, +0.843], L2 +0.893 [+0.700, +0.957] / +0.795 [+0.528, +0.937], M1 +0.741 [+0.580, +0.858] / +0.683 [+0.498, +0.822]: LOCALISED in `speed=3-8`; curvature and ADE6 DIFFUSE.
* **Shape is not the axis.** No shape level qualifies on any row. On yaw-rate and curvature STRAIGHT holds +0.69 to +0.79 of Delta_M with 0.73 of the windows (enrichment 0.94 to 1.09), TURN +0.06 to +0.10 with 0.074 (0.85 to 1.35), GENTLE +0.11 to +0.23 with 0.197. TURN is over-represented on M1 (+0.214 / +0.140, enrichment 2.93 / 1.91) and on the ADE6 gain (+0.199 / +0.198, enrichment 2.7), but holds a fifth or less of either.
* **Over-representation, not ownership.** `speed=3-8` is 2.3 to 5.5 times its window share on the localised and curvature rows (Table 3), and `speed=0.5-3` (81 windows, 18 episodes: below the 20-episode floor, so it cannot be named) carries 5.3 / 4.3 times (yaw-rate) and 11.0 / 9.2 times (curvature) its window share.
* **Beside, plan against GT in `speed=3-8` (MEASURED, Table 4; plan minus GT).** DAG-OACC plan speed 5.871 / 5.713 m/s against GT 5.473: error **+0.398 / +0.241**; P10 7.072, error **+1.599**. So the plan is not slower than GT on average; P10 overshoots. Plan mean |kappa| **0.1845 / 0.2621** against GT 0.0135 and P10 0.0623; the **signed** curvature error is +0.0037 / -0.0020 (P10 +0.0013), i.e. the excess is magnitude without a sign: jitter, not a steady bias. Sign of the speed error: plan faster than GT at both seeds.
* **Beside, D2' (ESTIMATED, decomposition arithmetic, Table 5).** With the `speed=3-8` gap set to 0, **D2' fails** for DAG-OACC at both seeds (s0: curvature, L2, M1 stay separated-worse and yaw-rate clears; s1: all four stay) and for SOUP (all four, both seeds). Not a retrained model: it says a P10-level 3 to 8 m/s behaviour alone would not clear D2.

### POST-HOC (not in the card; decides nothing; Table 6)

* **The pooled low-speed view of M, v0 < 8 m/s (the 0.5-3 and 3-8 buckets together)**, 24.1 % of M windows, carries DAG-OACC's yaw-rate **+0.850 / +0.844**, **curvature +0.880 [+0.786, +0.950] / +0.893 [+0.815, +0.952]** (enrichment 3.7), L2 +0.934 / +0.915, M1 +0.792 / +0.751 of Delta_M; ADE6 only +0.219 / +0.208 (enrichment 0.9, the ADE6 gain is not a low-speed effect). This pooling was chosen after seeing the 0.5-3 bucket fall below the floor, among many possible poolings, so it is a hypothesis to pre-register, not a verdict.
* **Mechanism descriptors of `speed=3-8`.** Per-window mean |kappa|: DAG-OACC median 0.0339 / 0.0387 against P10 0.0140, p90 0.695 / 0.888 against 0.151, and the top 10 % of windows carry 0.550 / 0.493 of the sum; **9.0 % / 11.4 % of DAG-OACC windows have a plan step below 0.5 m/s, against 0.7 % for P10**; the median slowest plan step is 4.21 / 3.95 m/s against 5.60. In the rest of M the median |kappa| is 0.0024 / 0.0026 against 0.0022. Reading (**HYPOTHESIS**, untested): the 3 to 8 m/s defect is an intermittent slow-dip tail in the plan (about one window in ten), where heading noise over a short step becomes large curvature and yaw-rate and where the plan falls below the 5.2 m/s line that M1 and L2 count; the mean speed hides it.
* **D2 arithmetic with more windows zeroed (ESTIMATED).** Zeroing the trigger windows (L + S) and `speed=3-8` together leaves yaw-rate, curvature, L2 and M1 separated-worse at both seeds, for both arms; **zeroing every window with v0 < 8 m/s (861 of 2,399) still leaves the same four rows separated-worse** (for example DAG-OACC s0: yaw-rate +0.776, curvature +0.0052, L2 +0.0007, M1 +0.0047, all separated). The residual is the 8 to 15 m/s bucket. The gap is concentrated at low speed, not confined to it.

## Next lever (rule zero: a diagnosis that does not stop at the finding)

The M gap on the three crawl-linked rows lives in the 3 to 8 m/s windows, as an intermittent slow-dip tail in the plan, and removing that bucket's gap alone does not clear D2, so two things remain: **(1)** the dips themselves (about 10 % of these windows) and **(2)** a smaller residual in the 8 to 15 m/s bucket (per-window Delta_sub s0 / s1: yaw-rate +2.48 / +3.27 deg/s, curvature +0.0136 / +0.0174, M1 +0.0106 / +0.0156). Cheapest discriminating experiments, both zero-training and CPU, needing no compute, spend or PI decision (not run here: this deliverable is the one registered split):
* **Dip-floor oracle on the dumps.** In the 3 to 8 m/s windows replace each DAG-OACC plan step below a floor by the previous step's speed and heading (a pure post-processing of the executed track) and recompute the four rows against P10. Committed outcomes to register: if yaw-rate, curvature, L2 and M1 clear their separated-worse state, the defect is the dip tail and the lever is a plan-side speed floor or a training-side weighting of slow windows; if curvature and yaw-rate stay worse, the lateral jitter is not only the dip and the lateral head at low speed is the next lever.
* **Pre-register the pooled v0 < 8 m/s curvature hypothesis** (this card committed DIFFUSE for curvature on a 0.027 miss at s0; the pooled view is the post-hoc suggestion above).
No lever is blocked: everything above is CPU on banked arrays and dumps.

## Reproduction

```
python -I extract_signed.py --arm {CTRL,SOUP} --seed {0,1} --out raw/signed/signed_<arm>_s<seed>.npz     # about 10 s each
python -I gapdecomp_m.py --signed raw/signed --out raw/GAPDECOMP_M.json
python -I posthoc.py --signed raw/signed --out raw/POSTHOC.json
python -I make_tables.py
python -I -m pytest -q -p no:cacheprovider test_gapdecomp_m.py
```
(interpreter `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, 2 threads; `gapdecomp_m.py` re-runs identically: JSON equal apart from the clock fields.)

## Controls and tables (generated by `make_tables.py` from `raw/GAPDECOMP_M.json` and `raw/POSTHOC.json`; no number typed by hand)

### Table 0. Known-answer controls (run before any verdict was read)

| control | result |
|---|---|
| KA1 partitions cover M once; M = 2,032 windows / 140 episodes = the landed class M; named cross cells at the floor | PASS |
| KA2 subclass sums reproduce the landed class-M n, Delta_M, lo, hi, contribution and its interval, 60 checks (5 rows x 2 arms x 2 seeds x 3 partitions) | PASS; max abs deviation 5.3e-15 (tolerance 1e-9) |
| KA3 (X, X) reads exactly 0.0 in every subclass, interval [0, 0], 60 checks | PASS |
| KA4 shares of each partition sum to 1 at the point and in every draw, 60 checks | PASS; max deviation 2.2e-16 (point), 5.6e-16 (draws) |
| KA5 planted effect LOCALISED / uniform effect DIFFUSE / zero effect NOT DEFINED / floor refuses to name (pytest, 8 tests) | PASS (`raw/pytest.log`) |
| KA6 an empty subclass reproduces the landed Delta of all 19 rows and the landed `D2_worse_rows`, 4 arm-seeds | PASS; max abs deviation 0.0e+00 |

### Table 1. The subclasses of M (windows / episodes on the 2,032 M windows)

| partition | cell | windows | episodes | nameable |
|---|---|---|---|---|
| cross | cross=ALL/other | 183 | 33 | no (pooled remainder) |
| cross | cross=GENTLE/3-8 | 98 | 29 | yes |
| cross | cross=GENTLE/8-15 | 184 | 37 | yes |
| cross | cross=STRAIGHT/3-8 | 272 | 57 | yes |
| cross | cross=STRAIGHT/8-15 | 645 | 74 | yes |
| cross | cross=STRAIGHT/>15 | 501 | 39 | yes |
| cross | cross=TURN/other | 149 | 26 | yes |
| shape | shape=GENTLE | 400 | 68 | yes |
| shape | shape=STRAIGHT | 1483 | 132 | yes |
| shape | shape=TURN | 149 | 26 | yes |
| speed | speed=0.5-3 | 81 | 18 | yes |
| speed | speed=3-8 | 413 | 68 | yes |
| speed | speed=8-15 | 923 | 85 | yes |
| speed | speed=>15 | 615 | 44 | yes |

Windows of M with no pair-valid GT step (STRAIGHT by rule): 3. |dpsi| over M, quantiles: p10 0.2 deg, p50 2.9 deg, p90 36.2 deg, p99 89.6 deg.

### Table 2. Verdict per row (PREREG_FROZEN rule), DAG-OACC headline and SOUP

Cell: share of the row's Delta_M carried by the named subclass (point [95 % interval]); w = its share of the row's valid M windows; eligible = candidates with support, nameable and w <= 0.30; argmax = fraction of draws in which the named subclass has the highest share among the eligible.

| row | arm | Delta_M s0 | Delta_M s1 | verdict | named | share s0 | share s1 | w s0 / s1 | eligible s0 / s1 | argmax s0 / s1 |
|---|---|---|---|---|---|---|---|---|---|---|
| ADE6 | DAG-OACC | -1.24 [-1.45, -1.03] | -1.18 [-1.41, -0.9461] | **DIFFUSE** | - | - | - | - | 8 / 8 | - |
| yaw-rate MAE | DAG-OACC | +6.05 [+3.72, +8.77] | +7.89 [+5.03, +11.33] | **LOCALISED** | speed|speed=3-8 | +0.653 [+0.510, +0.828] | +0.684 [+0.516, +0.888] | 0.204 / 0.204 | 8 / 8 | 1.000 / 1.000 |
| curvature MAE | DAG-OACC | +0.0503 [+0.0308, +0.0726] | +0.0722 [+0.0471, +0.1022] | **DIFFUSE** | - | - | - | - | 8 / 8 | - |
| L2 curvature MAE (GT>=5.2) | DAG-OACC | +0.0115 [+0.0042, +0.0221] | +0.0205 [+0.0086, +0.0365] | **LOCALISED** | speed|speed=3-8 | +0.934 [+0.805, +0.984] | +0.915 [+0.765, +0.983] | 0.169 / 0.169 | 8 / 8 | 1.000 / 1.000 |
| M1 crawl share | DAG-OACC | +0.0254 [+0.0151, +0.0381] | +0.0313 [+0.0190, +0.0469] | **LOCALISED** | speed|speed=3-8 | +0.792 [+0.613, +0.910] | +0.751 [+0.568, +0.879] | 0.169 / 0.169 | 8 / 8 | 1.000 / 0.998 |
| ADE6 | SOUP | -1.20 [-1.44, -0.9741] | -1.19 [-1.42, -0.9659] | **DIFFUSE** | - | - | - | - | 8 / 8 | - |
| yaw-rate MAE | SOUP | +9.74 [+6.72, +13.14] | +8.16 [+5.17, +11.52] | **LOCALISED** | speed|speed=3-8 | +0.677 [+0.558, +0.805] | +0.635 [+0.474, +0.843] | 0.204 / 0.204 | 8 / 8 | 1.000 / 1.000 |
| curvature MAE | SOUP | +0.0805 [+0.0541, +0.1105] | +0.0708 [+0.0437, +0.1027] | **DIFFUSE** | - | - | - | - | 8 / 8 | - |
| L2 curvature MAE (GT>=5.2) | SOUP | +0.0246 [+0.0098, +0.0468] | +0.0156 [+0.0070, +0.0271] | **LOCALISED** | speed|speed=3-8 | +0.893 [+0.700, +0.957] | +0.795 [+0.528, +0.937] | 0.169 / 0.169 | 8 / 8 | 1.000 / 0.989 |
| M1 crawl share | SOUP | +0.0369 [+0.0240, +0.0519] | +0.0325 [+0.0200, +0.0473] | **LOCALISED** | speed|speed=3-8 | +0.741 [+0.580, +0.858] | +0.683 [+0.498, +0.822] | 0.169 / 0.169 | 8 / 8 | 1.000 / 0.995 |

### Table 3. DAG-OACC, every candidate against its share of Delta_M (shape, speed and cross partitions)

q = qualifies at that seed (eligible, share >= 0.5, interval lower > 0.3). enr = share / w.

**ADE6**

| subclass | windows (episodes) | w | Delta_sub s0 | share s0 [interval] | Delta_sub s1 | share s1 [interval] | enr s0 / s1 | q s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| shape=GENTLE | 400 (68) | 0.197 | -1.41 | +0.224 [+0.153, +0.299] | -1.34 | +0.225 [+0.146, +0.311] | 1.14 / 1.14 | - / - |
| shape=STRAIGHT | 1483 (132) | 0.730 | -0.9764 | +0.577 [+0.469, +0.679] | -0.9305 | +0.577 [+0.465, +0.689] | 0.79 / 0.79 | - / - |
| shape=TURN | 149 (26) | 0.073 | -3.35 | +0.199 [+0.110, +0.297] | -3.18 | +0.198 [+0.101, +0.301] | 2.71 / 2.70 | - / - |
| speed=0.5-3 | 81 (18) | 0.040 | -2.77 | +0.089 [+0.037, +0.152] | -2.83 | +0.096 [+0.040, +0.167] | 2.24 / 2.41 | - / - (below floor) |
| speed=3-8 | 413 (68) | 0.203 | -0.7855 | +0.129 [+0.066, +0.195] | -0.6474 | +0.112 [+0.035, +0.187] | 0.64 / 0.55 | - / - |
| speed=8-15 | 923 (85) | 0.454 | -0.8988 | +0.331 [+0.248, +0.422] | -0.8274 | +0.319 [+0.227, +0.417] | 0.73 / 0.70 | - / - |
| speed=>15 | 615 (44) | 0.303 | -1.84 | +0.451 [+0.337, +0.559] | -1.84 | +0.473 [+0.357, +0.588] | 1.49 / 1.56 | - / - |
| cross=ALL/other | 183 (33) | 0.090 | -2.33 | +0.170 [+0.104, +0.245] | -2.44 | +0.186 [+0.116, +0.266] | 1.89 / 2.07 | - / - |
| cross=GENTLE/3-8 | 98 (29) | 0.048 | -1.00 | +0.039 [+0.009, +0.077] | -0.5368 | +0.022 [-0.017, +0.064] | 0.81 / 0.46 | - / - |
| cross=GENTLE/8-15 | 184 (37) | 0.091 | -1.17 | +0.085 [+0.046, +0.130] | -1.19 | +0.091 [+0.049, +0.144] | 0.94 / 1.01 | - / - |
| cross=STRAIGHT/3-8 | 272 (57) | 0.134 | -0.3720 | +0.040 [-0.003, +0.082] | -0.4374 | +0.050 [+0.004, +0.094] | 0.30 / 0.37 | - / - |
| cross=STRAIGHT/8-15 | 645 (74) | 0.317 | -0.4519 | +0.116 [+0.071, +0.172] | -0.3342 | +0.090 [+0.037, +0.151] | 0.37 / 0.28 | - / - |
| cross=STRAIGHT/>15 | 501 (39) | 0.247 | -1.75 | +0.350 [+0.232, +0.466] | -1.73 | +0.362 [+0.248, +0.479] | 1.42 / 1.47 | - / - |
| cross=TURN/other | 149 (26) | 0.073 | -3.35 | +0.199 [+0.110, +0.297] | -3.18 | +0.198 [+0.101, +0.301] | 2.71 / 2.70 | - / - |

**yaw-rate MAE**

| subclass | windows (episodes) | w | Delta_sub s0 | share s0 [interval] | Delta_sub s1 | share s1 [interval] | enr s0 / s1 | q s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| shape=GENTLE | 400 (68) | 0.197 | +3.30 | +0.108 [+0.036, +0.199] | +9.13 | +0.228 [+0.128, +0.340] | 0.55 / 1.16 | - / - |
| shape=STRAIGHT | 1477 (132) | 0.729 | +6.58 | +0.793 [+0.647, +0.913] | +7.45 | +0.688 [+0.557, +0.816] | 1.09 / 0.94 | - / - |
| shape=TURN | 149 (26) | 0.074 | +8.19 | +0.100 [+0.012, +0.214] | +8.97 | +0.084 [+0.008, +0.182] | 1.35 / 1.14 | - / - |
| speed=0.5-3 | 75 (18) | 0.037 | +32.24 | +0.197 [+0.036, +0.364] | +33.67 | +0.160 [-0.014, +0.335] | 5.33 / 4.27 | - / - (below floor) |
| speed=3-8 | 413 (68) | 0.204 | +19.38 | +0.653 [+0.510, +0.828] | +26.49 | +0.684 [+0.516, +0.888] | 3.20 / 3.36 | q / q |
| speed=8-15 | 923 (85) | 0.456 | +2.48 | +0.187 [+0.078, +0.296] | +3.27 | +0.189 [+0.089, +0.305] | 0.41 / 0.41 | - / - |
| speed=>15 | 615 (44) | 0.304 | -0.7385 | -0.037 [-0.068, -0.021] | -0.8491 | -0.033 [-0.059, -0.019] | -0.12 / -0.11 | - / - |
| cross=ALL/other | 177 (33) | 0.087 | +12.18 | +0.176 [+0.024, +0.330] | +13.02 | +0.145 [-0.023, +0.307] | 2.01 / 1.65 | - / - |
| cross=GENTLE/3-8 | 98 (29) | 0.048 | +10.50 | +0.084 [+0.031, +0.158] | +28.38 | +0.174 [+0.085, +0.280] | 1.74 / 3.60 | - / - |
| cross=GENTLE/8-15 | 184 (37) | 0.091 | +2.63 | +0.039 [+0.009, +0.091] | +3.76 | +0.043 [+0.006, +0.105] | 0.43 / 0.48 | - / - |
| cross=STRAIGHT/3-8 | 272 (57) | 0.134 | +21.86 | +0.485 [+0.335, +0.652] | +25.36 | +0.431 [+0.304, +0.577] | 3.61 / 3.21 | - / - |
| cross=STRAIGHT/8-15 | 645 (74) | 0.318 | +2.67 | +0.140 [+0.044, +0.253] | +3.61 | +0.146 [+0.059, +0.255] | 0.44 / 0.46 | - / - |
| cross=STRAIGHT/>15 | 501 (39) | 0.247 | -0.5944 | -0.024 [-0.046, -0.013] | -0.7133 | -0.022 [-0.040, -0.013] | -0.10 / -0.09 | - / - |
| cross=TURN/other | 149 (26) | 0.074 | +8.19 | +0.100 [+0.012, +0.214] | +8.97 | +0.084 [+0.008, +0.182] | 1.35 / 1.14 | - / - |

**curvature MAE**

| subclass | windows (episodes) | w | Delta_sub s0 | share s0 [interval] | Delta_sub s1 | share s1 [interval] | enr s0 / s1 | q s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| shape=GENTLE | 400 (68) | 0.197 | +0.0387 | +0.152 [+0.076, +0.238] | +0.0841 | +0.230 [+0.126, +0.350] | 0.77 / 1.16 | - / - |
| shape=STRAIGHT | 1477 (132) | 0.729 | +0.0543 | +0.786 [+0.674, +0.880] | +0.0687 | +0.694 [+0.543, +0.812] | 1.08 / 0.95 | - / - |
| shape=TURN | 149 (26) | 0.074 | +0.0426 | +0.062 [+0.021, +0.127] | +0.0752 | +0.077 [+0.026, +0.160] | 0.85 / 1.04 | - / - |
| speed=0.5-3 | 75 (18) | 0.037 | +0.5535 | +0.407 [+0.176, +0.594] | +0.6622 | +0.344 [+0.119, +0.525] | 11.00 / 9.17 | - / - (below floor) |
| speed=3-8 | 413 (68) | 0.204 | +0.1167 | +0.473 [+0.312, +0.684] | +0.1947 | +0.549 [+0.387, +0.760] | 2.32 / 2.70 | - / q |
| speed=8-15 | 923 (85) | 0.456 | +0.0136 | +0.123 [+0.052, +0.217] | +0.0174 | +0.110 [+0.050, +0.188] | 0.27 / 0.24 | - / - |
| speed=>15 | 615 (44) | 0.304 | -0.0005 | -0.003 [-0.006, -0.002] | -0.0006 | -0.003 [-0.005, -0.001] | -0.01 / -0.01 | - / - |
| cross=ALL/other | 177 (33) | 0.087 | +0.2319 | +0.402 [+0.174, +0.586] | +0.2799 | +0.340 [+0.118, +0.521] | 4.61 / 3.88 | - / - |
| cross=GENTLE/3-8 | 98 (29) | 0.048 | +0.0829 | +0.080 [+0.032, +0.149] | +0.2393 | +0.160 [+0.075, +0.272] | 1.65 / 3.31 | - / - |
| cross=GENTLE/8-15 | 184 (37) | 0.091 | +0.0135 | +0.024 [+0.005, +0.059] | +0.0182 | +0.023 [+0.004, +0.055] | 0.27 / 0.25 | - / - |
| cross=STRAIGHT/3-8 | 272 (57) | 0.134 | +0.1272 | +0.339 [+0.203, +0.510] | +0.1715 | +0.319 [+0.209, +0.464] | 2.53 / 2.37 | - / - |
| cross=STRAIGHT/8-15 | 645 (74) | 0.318 | +0.0148 | +0.094 [+0.027, +0.186] | +0.0188 | +0.083 [+0.030, +0.157] | 0.29 / 0.26 | - / - |
| cross=STRAIGHT/>15 | 501 (39) | 0.247 | -0.0003 | -0.002 [-0.003, -0.001] | -0.0004 | -0.001 [-0.003, -0.001] | -0.01 / -0.01 | - / - |
| cross=TURN/other | 149 (26) | 0.074 | +0.0426 | +0.062 [+0.021, +0.127] | +0.0752 | +0.077 [+0.026, +0.160] | 0.85 / 1.04 | - / - |

**L2 curvature MAE (GT>=5.2)**

| subclass | windows (episodes) | w | Delta_sub s0 | share s0 [interval] | Delta_sub s1 | share s1 [interval] | enr s0 / s1 | q s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| shape=GENTLE | 365 (62) | 0.197 | +0.0117 | +0.199 [+0.071, +0.468] | +0.0316 | +0.304 [+0.124, +0.514] | 1.01 / 1.54 | - / - |
| shape=STRAIGHT | 1351 (123) | 0.730 | +0.0120 | +0.760 [+0.458, +0.898] | +0.0178 | +0.635 [+0.400, +0.823] | 1.04 / 0.87 | - / - |
| shape=TURN | 135 (23) | 0.073 | +0.0064 | +0.040 [-0.006, +0.127] | +0.0171 | +0.061 [-0.006, +0.187] | 0.55 / 0.84 | - / - |
| speed=0.5-3 | 0 (0) | 0.000 | n/a | n/a | n/a | n/a | n/a | no valid pair |
| speed=3-8 | 313 (65) | 0.169 | +0.0638 | +0.934 [+0.805, +0.984] | +0.1110 | +0.915 [+0.765, +0.983] | 5.53 / 5.41 | q / q |
| speed=8-15 | 923 (85) | 0.499 | +0.0019 | +0.081 [+0.029, +0.231] | +0.0039 | +0.095 [+0.027, +0.258] | 0.16 / 0.19 | - / - |
| speed=>15 | 615 (44) | 0.332 | -0.0005 | -0.015 [-0.044, -0.006] | -0.0006 | -0.010 [-0.026, -0.005] | -0.05 / -0.03 | - / - |
| cross=ALL/other | 104 (15) | 0.056 | -0.0010 | -0.005 [-0.016, -0.001] | -0.0011 | -0.003 [-0.009, -0.001] | -0.08 / -0.05 | - / - (below floor) |
| cross=GENTLE/3-8 | 77 (26) | 0.042 | +0.0468 | +0.168 [+0.036, +0.420] | +0.1384 | +0.281 [+0.098, +0.484] | 4.05 / 6.75 | - / - |
| cross=GENTLE/8-15 | 184 (37) | 0.099 | +0.0041 | +0.036 [+0.012, +0.106] | +0.0054 | +0.026 [+0.007, +0.072] | 0.36 / 0.26 | - / - |
| cross=STRAIGHT/3-8 | 205 (52) | 0.111 | +0.0755 | +0.724 [+0.380, +0.870] | +0.1058 | +0.572 [+0.297, +0.759] | 6.54 / 5.16 | q / - |
| cross=STRAIGHT/8-15 | 645 (74) | 0.348 | +0.0015 | +0.044 [+0.014, +0.131] | +0.0041 | +0.069 [+0.017, +0.205] | 0.13 / 0.20 | - / - |
| cross=STRAIGHT/>15 | 501 (39) | 0.271 | -0.0003 | -0.008 [-0.023, -0.003] | -0.0004 | -0.006 [-0.015, -0.003] | -0.03 / -0.02 | - / - |
| cross=TURN/other | 135 (23) | 0.073 | +0.0064 | +0.040 [-0.006, +0.127] | +0.0171 | +0.061 [-0.006, +0.187] | 0.55 / 0.84 | - / - |

**M1 crawl share**

| subclass | windows (episodes) | w | Delta_sub s0 | share s0 [interval] | Delta_sub s1 | share s1 [interval] | enr s0 / s1 | q s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| shape=GENTLE | 365 (62) | 0.197 | +0.0388 | +0.301 [+0.187, +0.442] | +0.0593 | +0.374 [+0.270, +0.501] | 1.52 / 1.90 | - / - |
| shape=STRAIGHT | 1351 (123) | 0.730 | +0.0169 | +0.486 [+0.283, +0.661] | +0.0208 | +0.486 [+0.319, +0.630] | 0.67 / 0.67 | - / - |
| shape=TURN | 135 (23) | 0.073 | +0.0745 | +0.214 [+0.089, +0.365] | +0.0598 | +0.140 [+0.044, +0.254] | 2.93 / 1.91 | - / - |
| speed=0.5-3 | 0 (0) | 0.000 | n/a | n/a | n/a | n/a | n/a | no valid pair |
| speed=3-8 | 313 (65) | 0.169 | +0.1191 | +0.792 [+0.613, +0.910] | +0.1389 | +0.751 [+0.568, +0.879] | 4.68 / 4.44 | q / q |
| speed=8-15 | 923 (85) | 0.499 | +0.0106 | +0.208 [+0.090, +0.387] | +0.0156 | +0.249 [+0.121, +0.432] | 0.42 / 0.50 | - / - |
| speed=>15 | 615 (44) | 0.332 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0000 | +0.000 [+0.000, +0.000] | 0.00 / 0.00 | - / - |
| cross=ALL/other | 104 (15) | 0.056 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0000 | +0.000 [+0.000, +0.000] | 0.00 / 0.00 | - / - (below floor) |
| cross=GENTLE/3-8 | 77 (26) | 0.042 | +0.1434 | +0.235 [+0.117, +0.373] | +0.2113 | +0.281 [+0.173, +0.396] | 5.64 / 6.76 | - / - |
| cross=GENTLE/8-15 | 184 (37) | 0.099 | +0.0169 | +0.066 [+0.026, +0.127] | +0.0292 | +0.093 [+0.036, +0.178] | 0.66 / 0.93 | - / - |
| cross=STRAIGHT/3-8 | 205 (52) | 0.111 | +0.0982 | +0.428 [+0.218, +0.614] | +0.1078 | +0.382 [+0.188, +0.539] | 3.86 / 3.45 | - / - |
| cross=STRAIGHT/8-15 | 645 (74) | 0.348 | +0.0042 | +0.058 [+0.014, +0.129] | +0.0094 | +0.104 [+0.043, +0.198] | 0.17 / 0.30 | - / - |
| cross=STRAIGHT/>15 | 501 (39) | 0.271 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0000 | +0.000 [+0.000, +0.000] | 0.00 / 0.00 | - / - |
| cross=TURN/other | 135 (23) | 0.073 | +0.0745 | +0.214 [+0.089, +0.365] | +0.0598 | +0.140 [+0.044, +0.254] | 2.93 / 1.91 | - / - |

### Table 4. Beside (MEASURED, descriptive): plan against GT in speed|speed=3-8 (localised); plan minus GT; window means

| arm / seed | who | plan speed | GT speed | speed err | plan mean abs kappa (1/m) | GT mean abs kappa | abs-kappa diff | signed kappa err | n |
|---|---|---|---|---|---|---|---|---|---|
| DAG-OACC s0 | DAG-OACC | 5.871 | 5.473 | +0.398 | 0.1845 | 0.0135 | +0.1710 | +0.0037 | 413 |
| DAG-OACC s0 | P10 | 7.072 | 5.473 | +1.599 | 0.0623 | 0.0135 | +0.0488 | +0.0013 | 413 |
| DAG-OACC s1 | DAG-OACC | 5.713 | 5.473 | +0.241 | 0.2621 | 0.0134 | +0.2487 | -0.0020 | 413 |
| DAG-OACC s1 | P10 | 7.072 | 5.473 | +1.599 | 0.0623 | 0.0135 | +0.0488 | +0.0013 | 413 |
| SOUP s0 | SOUP | 5.629 | 5.473 | +0.156 | 0.2912 | 0.0135 | +0.2777 | +0.0303 | 413 |
| SOUP s0 | P10 | 7.072 | 5.473 | +1.599 | 0.0623 | 0.0135 | +0.0488 | +0.0013 | 413 |
| SOUP s1 | SOUP | 5.862 | 5.473 | +0.390 | 0.2366 | 0.0134 | +0.2232 | -0.0047 | 413 |
| SOUP s1 | P10 | 7.072 | 5.473 | +1.599 | 0.0623 | 0.0135 | +0.0488 | +0.0013 | 413 |

### Table 5. Beside (ESTIMATED, decomposition arithmetic): D2' with the subclass's gap set to 0 (all 19 rows)

| subclass | arm | seed | rows separated-worse with the subclass zeroed | D2' | rows separated-worse before (landed) |
|---|---|---|---|---|---|
| speed|speed=3-8 | DAG-OACC | s0 | curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share |
| speed|speed=3-8 | DAG-OACC | s1 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share |
| speed|speed=3-8 | DAG-OACC | both seeds | | fails | |
| speed|speed=3-8 | SOUP | s0 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share |
| speed|speed=3-8 | SOUP | s1 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share |
| speed|speed=3-8 | SOUP | both seeds | | fails | |

### Table 6. POST-HOC (not in the card; decides nothing)

**6a. The pooled low-speed view of M (v0 < 8 m/s: the 0.5-3 and 3-8 buckets together), share of Delta_M**

| row | arm | w s0 | share s0 [interval] | share s1 [interval] | enr s0 / s1 |
|---|---|---|---|---|---|
| ADE6 | DAG-OACC | 0.243 | +0.219 [+0.136, +0.306] | +0.208 [+0.117, +0.308] | 0.90 / 0.85 |
| yaw-rate MAE | DAG-OACC | 0.241 | +0.850 [+0.735, +0.967] | +0.844 [+0.729, +0.949] | 3.53 / 3.50 |
| curvature MAE | DAG-OACC | 0.241 | +0.880 [+0.786, +0.950] | +0.893 [+0.815, +0.952] | 3.65 / 3.70 |
| L2 curvature MAE (GT>=5.2) | DAG-OACC | 0.169 | +0.934 [+0.805, +0.984] | +0.915 [+0.765, +0.983] | 5.53 / 5.41 |
| M1 crawl share | DAG-OACC | 0.169 | +0.792 [+0.613, +0.910] | +0.751 [+0.568, +0.879] | 4.68 / 4.44 |
| ADE6 | SOUP | 0.243 | +0.196 [+0.101, +0.295] | +0.202 [+0.112, +0.304] | 0.81 / 0.83 |
| yaw-rate MAE | SOUP | 0.241 | +0.833 [+0.741, +0.917] | +0.810 [+0.685, +0.922] | 3.45 / 3.36 |
| curvature MAE | SOUP | 0.241 | +0.874 [+0.798, +0.933] | +0.864 [+0.760, +0.935] | 3.62 / 3.58 |
| L2 curvature MAE (GT>=5.2) | SOUP | 0.169 | +0.893 [+0.700, +0.957] | +0.795 [+0.528, +0.937] | 5.30 / 4.71 |
| M1 crawl share | SOUP | 0.169 | +0.741 [+0.580, +0.858] | +0.683 [+0.498, +0.822] | 4.39 / 4.05 |

**6b. D2 arithmetic with more windows zeroed (ESTIMATED)**

| zeroed | arm | seed | rows separated-worse | D2 |
|---|---|---|---|---|
| trigger L+S and the 3-8 m/s windows of M | DAG-OACC | s0 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| trigger L+S and the 3-8 m/s windows of M | DAG-OACC | s1 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| trigger L+S and the 3-8 m/s windows of M | SOUP | s0 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| trigger L+S and the 3-8 m/s windows of M | SOUP | s1 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| every window with v0 < 8 m/s (861) | DAG-OACC | s0 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| every window with v0 < 8 m/s (861) | DAG-OACC | s1 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| every window with v0 < 8 m/s (861) | SOUP | s0 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |
| every window with v0 < 8 m/s (861) | SOUP | s1 | yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share | fails |

**6c. Mechanism descriptors of the 3-8 m/s windows of M** (GT step speed in [5.2, 6.2): 0.120; below 5.2: 0.449)

| arm / seed | who | median per-window mean abs kappa | p90 | share of the sum in the top 10 % of windows | windows with a plan step < 0.5 m/s | median of the plan's slowest step (m/s) | rest of M: median abs kappa / stall share |
|---|---|---|---|---|---|---|---|
| DAG-OACC s0 | DAG-OACC | 0.0339 (mean 0.1845) | 0.6946 | 0.550 | 0.090 | 4.21 | 0.0024 / 0.029 |
| DAG-OACC s0 | P10 | 0.0140 (mean 0.0623) | 0.1509 | 0.698 | 0.007 | 5.60 | 0.0022 / 0.016 |
| DAG-OACC s1 | DAG-OACC | 0.0387 (mean 0.2621) | 0.8882 | 0.493 | 0.114 | 3.95 | 0.0026 / 0.030 |
| DAG-OACC s1 | P10 | 0.0140 (mean 0.0623) | 0.1509 | 0.698 | 0.007 | 5.60 | 0.0022 / 0.016 |
| SOUP s0 | SOUP | 0.0664 (mean 0.2912) | 0.9180 | 0.457 | 0.109 | 3.21 | 0.0026 / 0.032 |
| SOUP s0 | P10 | 0.0140 (mean 0.0623) | 0.1509 | 0.698 | 0.007 | 5.60 | 0.0022 / 0.016 |
| SOUP s1 | SOUP | 0.0397 (mean 0.2366) | 0.7776 | 0.488 | 0.104 | 3.88 | 0.0025 / 0.032 |
| SOUP s1 | P10 | 0.0140 (mean 0.0623) | 0.1509 | 0.698 | 0.007 | 5.60 | 0.0022 / 0.016 |
