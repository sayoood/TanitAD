# RESULT: H-V8F-WEATHERHEAD-1, form (a), read 2026-11-17

**Outcome (pre-registered rules, `wh_read.py` `decide`): weather = WH-DETECTS + WH-INPUT-FEATURES.** The other variables read country POOL-GAIN, and surface, illumination and period_of_day POOL-REPLACES. Driving reads NO-COST-BY-IDENTITY.

**Evidence class and tier.** MEASURED (ours). T0, per-frame perception output at the window's NOW frame. Zero training of any driving parameter. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, 2,399 windows over 141 episodes (eval cache of record). It answers "another draw of EPISODES" only. The fit replicate H_s0_rep gives F_fit, the run-to-run floor of the head fit.

## Instrument stage

Every required check passes: K0, K2, K3, K4p, K5, K6, R1, K-ID, T-ISO (a)-(d), T-VIS, REACH-live, REACH-census, DISJOINT and G-EVAL (`instrument.json`, `WH_READ.json`).

The first read on 2026-11-17 was INSTRUMENT-FAIL on K-ID: max |diff| 1.144e-5 and 1.335e-5 against atol 1e-5. The cause was the window batch size. `v8f_mini_eval.py` batches 64 windows (`:304`, default `:672`), while the fit script's identity pass used 256 (`wh_lib.py:48,415`), so the CPU matmul reduction order differed.

At batch 64 the two paths are bit-identical (max |diff| 0.0 on both arms). A mutation that swaps two logit columns reads 18.1 / 18.97 and fails. The fix is on the identity side only (`kidfix/`, `wh_fit.py kid --batch`, default 64). No bar, tolerance, fit or dump changed. Only the K-ID stage and the read re-ran. Details: `kidfix/BUILD_KIDFIX.md`.

## Weather

| | H_s0 | H_s1 |
|---|---|---|
| accuracy | 0.9379 | 0.9408 |
| constant (clear) | 0.8574 | 0.8574 |
| accuracy - constant | +0.0805 [+0.0237, +0.1338] | +0.0834 [+0.0296, +0.1359] |
| balanced accuracy | 0.835 | 0.809 |
| recall clear / fog / rain / snow | 0.957 / 0.750 / 0.706 / 0.927 | 0.967 / 0.750 / 0.596 / 0.921 |

- **Controls.** The re-initialised head K4p does not detect weather at either seed (balanced accuracy 0.299 / 0.254).
- **Trained vs random adapter (H - RA)** is not separated: -0.0058 [-0.0179, +0.0038] and +0.0100 [-0.0029, +0.0254]. Hence WH-INPUT-FEATURES. The committed wording: *weather is readable from the model's vision input features (cached DINOv3 4x10 tokens over 4 frames) by the shipped pooled head; S1's training is not shown to add to it.*
- **Beside, no bar:**
  - H - FEAT: -0.0175 [-0.0317, -0.0054] at s0 and -0.0042 [-0.0196, +0.0092] at s1.
  - H - C0A: +0.0033 / +0.0013, not separated.
  - H - probe: +0.0021 at both seeds, not separated.
  - Replicate: H_s0_rep's weather accuracy equals H_s0's (0.9379), so F_fit(weather) = 0.0. That is one replicate pair, a weak floor. Balanced accuracy still moves 0.835 -> 0.812 on the rare classes.
- **Label ceiling, beside.** The weather label agrees with the PI's frames 36/40 = 90.0 % [76.9, 96.0] (INHERITED, `H-CTX-LABELS-V1`), with no per-class check. The head's 0.94 against the label is above that agreement. Further accuracy against this label cannot be told from label noise.

## The other variables (accuracy - constant, both seeds separated)

| variable | H_s0 | H_s1 | outcome |
|---|---|---|---|
| country | +0.638 [+0.577, +0.701] | +0.628 [+0.565, +0.690] | POOL-GAIN; beyond the 0.0779 rig floor (K6) |
| surface | +0.133 [+0.072, +0.193] | +0.135 [+0.076, +0.194] | POOL-REPLACES |
| illumination | +0.366 [+0.294, +0.444] | +0.373 [+0.302, +0.450] | POOL-REPLACES |
| period_of_day | +0.322 [+0.238, +0.405] | +0.308 [+0.221, +0.391] | POOL-REPLACES |

Pooled head minus the shipped `ctx_logits` (H - shipped):

- country: +0.236 [+0.173, +0.297] and +0.183 [+0.114, +0.251].
- period_of_day: +0.085 [+0.015, +0.155] at s0; not separated at s1.
- surface and illumination: not separated at s0, with illumination +0.016 [+0.002, +0.037] at s1. No variable reads separated-worse than the shipped head.

## The four driving families

- **LONGITUDINAL, LATERAL, TACTICAL:** a structural zero. The 24 driving cells read exactly 0.0 [0, 0].
- **STRATEGIC (reported in full):** the 6 route-setting cells and S_CKPT also read exactly 0.0 [0, 0].
- **Why:** these zeros are an identity, not an estimate. The planner outputs are bit-identical to S1, with 32 arrays compared per seed and no differences. T-ISO (a)-(d) show the head does not reach the planner.

## Next

The planned sequence (wh_read `next` field):

1. Weather is a reported model output from `ctx_pool_head.weather`.
2. The label lever, Q3, approved by the PI on 2026-11-17: about 40 frames drawn blind and stratified on fog, rain and snow. The sheet is prepared next.
3. The re-fit on the adopted driving checkpoint.
4. The derived-goals card, with its own goal-input admissibility check.

Form (b) is not triggered, because weather is detected; no pod time is spent.
