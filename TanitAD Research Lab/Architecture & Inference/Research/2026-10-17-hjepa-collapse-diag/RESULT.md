# RESULT -- H-HJ-COLLAPSE-1 (RA-0): is refav1's action-deafness H-JEPA's slow-feature / advection collapse?

**STATUS (2026-10-17, final): MEASURED, 40/40 episodes, 80 PREREG windows.** All numbers below are filled from
`raw/readout.json` (rendered in `raw/RESULT_numbers.md`); nothing was recomputed in this write-up.

**HEADLINE.** PREREG outcome on the POINT estimate (as committed, not moved): **O-COLLAPSE.** Operative steer-sensitivity at
k = 10 (2.0 s) = **0.0043 [0.0026, 0.0063]** (< 0.05; the CI clears the threshold by 8x), speed-channel sensitivity
**0.0590 [0.0474, 0.0716]**, v/steer = **13.79** (>= 5). **Pre-registered next lever: the IDM lever (alpha3 / RA-1 arm A1).**
BUT the mechanism evidence is mixed (section 6.1, post-hoc, labelled): the OPERATIVE level shows no slow-feature collapse
(rho_tok 0.132, displacement 42 % of ||z_0|| at k = 10); the TACTICAL level does look collapsed in rho (0.0161) yet is accel-sensitive
(0.198). So "O-COLLAPSE" is the PREREG's operational label (steer-deaf, speed/accel-responsive), not a demonstration of H-JEPA's
encoder-collapse mechanism, and the copycat/history-action-dropout lever should run alongside A1 (RESULT section 6 caveat 3).

Stream S2 of `Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`. Pre-registration:
`Project Steering/PREREG_HJ_STAGE0.md` (H-HJ-COLLAPSE-1). Primary: H-JEPA, arXiv 2610.06805 (banked:
`TanitAD Research Lab/Library/papers/2610.06805_*.pdf`; App. L Def. 1 / Eq. 10, Table 12). Tier: **T0**
(teacher-forced recorded actions; a WM diagnostic, not driving performance). Estimator: **episode-cluster
bootstrap, B = 2000, seed 0** -- it answers "would another draw of EPISODES say this?"; it does NOT answer
training-seed or inference-seed variance (one checkpoint, deterministic forward, no sampling).

## 1. What was measured (all MEASURED, ours; artifacts in `raw/`)

| item | value | evidence |
|---|---|---|
| checkpoint | `refav1-b1-v72-ep3-speed`, `ckpt['step']` = **21109** (md5 `1189bc02...` per PREREG; md5 NOT re-hashed here -- 2.1 GB) | `raw/meta.json` |
| strict load into repo `refa_v1.py` | `missing_keys: []`, `unexpected_keys: []` (asserted in code, fitted standardizer required) | `raw/meta.json` |
| code identity | `tanitad` imported from `E:\Projects\TanitAD\stack` (asserted at startup) | `raw/meta.json` `tanitad_file` |
| probe == model | probe rollouts reproduce `model.forward()`'s `op_pred[:, :10]` and `tac_pred` with **max abs diff 0.0** on episode 00 window t=3 | `raw/tripwire.json` |
| data | first 40 sorted clip ids of the 141 v7.2 EVAL episodes (cache stems are the clip ids, asserted); 2,681 loader windows (W=4, K_loader=30, 0.2 s grid); labels/nav join 40/40, md5 `aa12c948...` | `raw/meta.json`, `run3.log` |
| PREREG windows | stride 40 = `(t-3) % 40 == 0` -> exactly **2 windows per episode (t=3, t=43), 80 total** | `raw/meta.json` |
| donor | fixed derangement of the 40 episodes (`numpy default_rng(0).permutation`, fixed points repaired), donor window = donor's S_40 window nearest in t | `raw/meta.json` `donor_perm` |
| precision | fp32 on CPU, 6 threads, `torch.inference_mode`. Training was bf16 autocast on GPU; **UNVERIFIED**: fp32-CPU vs the GPU eval on these exact windows (the probe is relative within one numeric path, so the swap ratios do not depend on it, but the absolute displacements might shift in the 3rd digit) | -- |

Note: `prov['trainable_parameters']` prints 0 because `load_model` freezes the model; the strict load, not that
count, is the identity check.

## 2. Definitions (fixed before any number was seen; the full text is the docstring of `code/hj_collapse_diag.py`)

* **z_t** = `adapter(std(feats))[:, -1]` of the 4-frame window ending at t (window-causal, as the model sees it);
  rollout start `last` = `_last_state(field)` (= `field[:, -1]`, `motion_inject` is False).
* **rho** (paper Eq. 10) = E_episode[tr Var_t(z | episode)] / tr Var(z), unbiased (ddof=1) within each episode and
  pooled. Levels: token-mean 1024-d and the full 640x1024 field, for: raw standardised DINOv3 features
  (REFERENCE), adapter latent (all 2,681 windows; and the stride-10 subset), operative predictions k in {1,5,10}
  (0.2/1.0/2.0 s), tactical start field and tactical predictions k in {1,5,10} (0.6/3.0/6.0 s). Predictions are
  on stride-10 windows (7 per episode, 280 total).
* **sens_k(c)** = mean over windows of ||zhat_k(swap c) - zhat_k|| / ||zhat_k - z_0||, L2 over the full flattened
  field; zhat = the operative (or tactical) predictor rolled from `last` with the window's OWN `intent` held fixed
  (so the swap isolates the predictor's action pathway, H-JEPA's definition).
* **Channel swaps.** The predictor input per step is `(a_j, steer_j, v_j/30)`, `v_j = v0 + sum_{i<j} a_i*0.2`
  (`augment_actions`, `refa_v1.py:1688`). Own window A and donor window D are both 3-wide augmented inputs:
  `all` = D; `a` = A with channel 0 := D (v channel NOT re-integrated); `steer` = A with channel 1 := D;
  `v` = A with channel 2 := D (the donor's whole speed profile v0_d + cumulative a_d); `v0` = own controls with the
  donor's v0 re-integrated (a constant offset on channel 2); `self` = A (control). PREREG columns: all, a, steer,
  v. Loader channel 1 is the road-wheel steer ANGLE fed as `kappa` (`REFAV1_ARM.md` section 3b) -- the model is fed
  exactly what it was trained on.
* **Tactical** = `tactical.rollout(tac_field, acts[:, ::3][:, :10], intent)` (the `forward()` striding), same swaps.
* **Pre-registered rule** (PREREG + one uncovered cell resolved before the data): steer-sens(op, k=10) < 0.05 AND
  v-sens >= 5x steer-sens -> **O-COLLAPSE**; steer-sens >= 0.10 -> **O-NOT**; 0.05-0.10 -> inconclusive;
  steer < 0.05 but v < 5x steer (uncovered by the PREREG) -> "inconclusive (uncovered cell)", both levers run.
  Decision on the POINT estimate (mean of per-window ratios, `v` = the channel swap); CI reported beside it.

### Additions made AFTER the 1-episode shake-down and BEFORE episodes 01-39 ran (exploratory; none enters the decision)
dose-response probes (own input + constant +0.05 rad steer; +1 m/s^2 accel); `||z_0||`; the same sens ratio measured after the model's
linear decoder `to_enc` (the space the loss reads); the T0 WM-vs-persistence MSE (arm `wm_diagnostic_T0`
definition). Variants `a_reint` and `d_steer15` exist only for the episode-00 shake-down and are not reported.
Reason for all of them: the shake-down showed the donor steer swap is tiny (rms 0.005 rad) and that the rollout
displacement is ~43 % of ||z_0|| per step, so the PREREG ratio's denominator carries a large action-independent
term. These are context, flagged as post-hoc.

## 3. Controls -- ALL READ THEIR KNOWN VALUES (MEASURED; `raw/readout.json["controls"]`, `raw/RESULT_numbers.md` "Controls")
| control | expected | read |
|---|---|---|
| tripwire: probe rollouts vs `model.forward()` (ep 00, t = 3) | max abs diff 0 | op **0.0**, tac **0.0** (`raw/tripwire.json`) |
| self-swap, real model, tactical | numerator exactly 0 | 80 windows, bit-identical, max abs field diff **0.0** |
| self-swap, real model, operative | numerator exactly 0 | **40 of 80 windows** (by design `SELF_OP_EVERY = 4` with the counter reset per episode = each episode's first S_40 window; CPU saving), bit-identical, max abs diff **0.0** |
| identity-predictor stub | numerator exactly 0 | `num_max_abs` **0**, denominator also 0 -> ratio undefined (80/80 windows), both levels |
| identity + fixed-drift stub | exactly 0.0, non-degenerate denominator | **0.0** with `den_min` 161.9 (op) / 51.0 (tac), all channels |
| linear-action stub (op, k = 10) | > 0 | self 0.0; all **1.543** [1.237, 1.932]; a 1.371; v 0.481; steer **0.0735** [0.0420, 0.1109] (same swaps as the model) |

The linear stub's channel weights are the stub's own, not a calibrated "healthy" model: it shows the probe CAN read a non-zero on these swaps and
is NOT a threshold calibration.

## 4. Results (MEASURED, T0 = teacher-forced recorded actions = WM diagnostic, NOT driving performance)
Estimator for every interval: **episode-cluster bootstrap, B = 2000, seed 0, 40 episodes** -- it answers "would another draw of EPISODES
say this?" ONLY. One checkpoint (step 21109), deterministic forward: training-seed and inference-seed variance are NOT answered
(`H-ESTIM-SEED-1`). Artifact: `raw/readout.json` (full) / `raw/RESULT_numbers.md` (rendered). n = 80 windows (2 per episode) unless stated.

### 4.1 PREREG decision (operative predictor, k = 10 = 2.0 s)
| quantity | value [95 % CI] |
|---|---|
| steer-sens | **0.0043 [0.0026, 0.0063]** |
| v-sens (speed-channel swap) | **0.0590 [0.0474, 0.0716]** |
| v / steer (ratio of the two means) | **13.79** (crude lower bound from the marginal CIs: 0.0474 / 0.0063 = 7.5; not a bootstrap CI of the ratio) |
| rule | steer < 0.05 AND v >= 5 x steer -> O-COLLAPSE |
| **PREREG outcome (point estimate)** | **O-COLLAPSE.** `steer_ci_straddles_thresholds: []` -- the outcome does not depend on where in the CI the truth lies |

**Named next lever (PREREG_HJ_STAGE0.md, H-HJ-COLLAPSE-1):** O-COLLAPSE => the **IDM lever (alpha3, RA-1 arm A1)** is the primary next step.
(The other branches, for the record: O-NOT => action-inferable-from-history / copycat `2010.14876`, history-action dropout on the tiny rig;
0.05-0.10 => run both. Neither fired.) Context for A1's SENS bar (steer-sens >= 0.10 at k = 10): the baseline is 0.0043, so the bar sits
**23x above the baseline** (DERIVED: 0.10 / 0.00428).

### 4.2 sens_k(c), operative level (mean of per-window ratios [CI]; k = 1/5/10 = 0.2/1.0/2.0 s)
| swap | k=1 | k=5 | k=10 |
|---|---|---|---|
| self (control) | 0.0000 | 0.0000 | 0.0000 |
| all | 0.0476 [0.0363, 0.0602] | 0.0663 [0.0542, 0.0804] | 0.0742 [0.0625, 0.0869] |
| a | 0.0202 [0.0126, 0.0319] | 0.0305 [0.0221, 0.0403] | 0.0384 [0.0306, 0.0474] |
| **steer** | 0.0041 [0.0020, 0.0066] | 0.0055 [0.0031, 0.0083] | **0.0043 [0.0026, 0.0063]** |
| v | 0.0376 [0.0280, 0.0489] | 0.0530 [0.0411, 0.0665] | 0.0590 [0.0474, 0.0716] |

Median of the per-window ratios at k = 10: steer 0.0007, a 0.0323, v 0.0467, all 0.0687 (the steer mean is carried by a few large-steer windows).
Even the full-action swap (0.0742) is below 0.10 at the operative level.

### 4.3 Tactical level (k = 1/5/10 = 0.6/3.0/6.0 s)
| swap | k=1 | k=5 | k=10 |
|---|---|---|---|
| all | 0.2488 [0.1967, 0.3088] | 0.2540 [0.2111, 0.3021] | 0.2134 [0.1695, 0.2643] |
| **a** | 0.2472 [0.1943, 0.3092] | 0.2457 [0.2037, 0.2914] | **0.1983 [0.1602, 0.2418]** |
| **steer** | 0.0071 [0.0034, 0.0116] | 0.0062 [0.0033, 0.0097] | **0.0045 [0.0028, 0.0066]** |
| v | 0.1008 [0.0734, 0.1361] | 0.0924 [0.0721, 0.1150] | 0.0903 [0.0709, 0.1110] |

**The tactical level IS sensitive to `a` (0.198) but NOT to steer (0.0045)** -- the same steer-deafness as the operative level (0.0043),
so it is not confined to one rollout. Caveat on reading across levels: the tactical ratio's denominator is small (mean ||zhat_k - z_0|| =
17.9 at k = 10 vs 355.9 operative; 0.061 vs 0.418 of ||z_0||), so tactical and operative ratios are not on a common scale. The PREREG
decision uses the operative level only; this tactical row is reported, not decisive.

### 4.4 rho (paper Eq. 10, within-episode variance share), operative and tactical, with the frozen-feature reference
| quantity | rho [CI] | n_vec |
|---|---|---|
| **`feat_tok`** (raw standardised DINOv3, token-mean 1024-d; REFERENCE) | **0.1669 [0.1494, 0.1934]** | 2681 |
| **`feat_full`** (raw standardised DINOv3, full 640 x 1024 field) | **0.4813 [0.4565, 0.5203]** | 2681 |
| `adp_tok` / `adp_full` (adapter latent z_t) | 0.1067 [0.0950, 0.1261] / 0.3530 [0.3324, 0.3885] | 2681 |
| `op1_tok` / `op10_tok` (operative prediction, 0.2 s / 2.0 s) | 0.1259 [0.1115, 0.1495] / 0.1319 [0.1174, 0.1555] | 280 |
| `op10_full` | 0.3899 [0.3664, 0.4296] | 280 |
| `tac0_tok` = `tac0_full` (tactical start field) | 0.0161 [0.0119, 0.0239] | 280 |
| `tacp10_tok` = `tacp10_full` (tactical prediction, 6.0 s) | 0.0165 [0.0122, 0.0241] | 280 |

Paper reference (App. L Table 12, 5 fps, DROID; PUBLISHED, banked `2610.06805`): collapsed rho 0.008-0.016; IDM-on healthy 0.12;
time-repulsion 0.17-0.92. **INDICATIVE ONLY: different encoder, corpus and latent shape.** Operative predictions (0.126-0.132 token-mean) sit at the
paper's healthy value, not in its collapsed band; the tactical field (0.0161) sits numerically inside the collapsed band.
`tac0_tok` equals `tac0_full` to 4 d.p. -- the 64 tactical tokens move together (cause not inspected: UNVERIFIED).

### 4.5 Rollout displacement (variant `all`, mean ||zhat_k - z_0|| / ||z_0||)
Operative: **0.361 / 0.380 / 0.418** at k = 1/5/10 (||z_0|| = 851.0; den 306.9 / 323.2 / 355.9). Tactical: 0.0062 / 0.0307 / 0.0609 (||z_0|| = 319.5).
The operative predictor is far from the identity at every horizon (it moves the field by ~36-42 % of its norm).

### 4.6 EXPLORATORY (not in the PREREG decision rule; added after the 1-episode shake-down and before episodes 01-39, section 2)
**(a) Post-decoder sensitivity** -- the same ratio after the model's linear decoder `to_enc`, i.e. in the space the loss reads (operative):
| swap | k=1 | k=5 | k=10 |
|---|---|---|---|
| all | 0.0617 [0.0475, 0.0777] | 0.1524 [0.1288, 0.1788] | 0.2131 [0.1816, 0.2473] |
| a | 0.0263 [0.0177, 0.0392] | 0.0981 [0.0785, 0.1200] | 0.1514 [0.1259, 0.1802] |
| steer | 0.0048 [0.0024, 0.0079] | 0.0084 [0.0046, 0.0127] | 0.0097 [0.0054, 0.0148] |
| v | 0.0474 [0.0355, 0.0612] | 0.1089 [0.0861, 0.1330] | 0.1472 [0.1182, 0.1776] |

In the loss-readable space `a` and `v` clear 0.10 at k = 10 (0.151, 0.147; DERIVED amplification 3.9x and 2.5x vs their pre-decoder 0.0384 and 0.0590)
while steer stays at 0.0097 (2.3x), 15.6x below `a`. The deafness is steer-specific, not a general action-deafness.

**(b) Dose-response** (own input + a constant offset; operative k = 10): **+0.05 rad steer -> 0.0052 [0.0050, 0.0054]; +1 m/s^2 accel ->
0.0448 [0.0407, 0.0499]**. Tactical: 0.0073 [0.0069, 0.0079] vs 0.3240 [0.3075, 0.3405]. The doses are not commensurate, so this is context, not a ratio claim.

**(c) Swap-magnitude terciles** (operative k = 10; `raw/readout.json["exploratory_terciles_op_k10"]`, n = 27/26/27; means only, no CI):
steer low/mid/high perturbation rms 0.0025 / 0.0090 / 0.125 rad -> sens 0.0002 / 0.0008 / 0.0117 (DERIVED slope 0.087 / 0.087 / 0.094 per rad = roughly
linear). The donor steer swap is mostly tiny (median window ratio 0.0007); the response scales with the input.

**(d) T0 WM vs persistence** (frozen target space, teacher-forced, mean MSE over 280 S_10 windows, no CI computed): model 0.299 / 0.378 / 0.423 vs
constant-latent 0.261 / 0.489 / 0.593 at k = 1/5/10. The model is worse than persistence at k = 1 and better at k = 5 and 10.

## 5. UNVERIFIED / limits (final list)
* fp32-CPU vs GPU numerics (section 1).
* ckpt md5 not re-hashed in this run (size/step/strict-load checked).
* No ground-truth counterfactual: a "right" steer sensitivity is not known; only ratios between channels and
  doses of the model's own response are measured.
* One checkpoint, one deterministic forward: seed floors (training/inference) are not answered by this interval.
* 40 of the 141 EVAL episodes (first 40 sorted clip ids) and only 2 windows per episode (t = 3 and t = 43, the PREREG stride) -- a pre-registered
  sample, not a corpus-wide estimate. The rho rows use 2,681 (adapter/feature) or 280 (predictions) windows from the same 40 episodes.
* The operative self-swap control ran on 40 of 80 windows (section 3); the tactical one on all 80.
* `tac0_tok == tac0_full` (identical rho) has no verified cause (section 4.4).
* rho and the sens ratios are compared with the paper's Table 12 values only as INDICATIVE references (different encoder, corpus, latent shape).

## 6. Post-hoc caveats to carry into the narrative (known BEFORE the full run; do not let them change the rule's outcome)
1. **The mechanism differs from H-JEPA's.** In H-JEPA the encoder is trained and may collapse onto episode-
   constant features, leaving an identity predictor. In refav1 the targets are the FROZEN DINOv3 features
   (`target_space: frozen`, the loss reads `to_enc(op_pred)` vs `std(future)`), so Proposition 1's escape
   (the encoder choosing background-only features) is structurally closed for the frozen part, and the
   shake-down shows the predictor is far from the identity (rollout displacement ~350-400 vs ||z_0|| ~ 850).
   The decisive mechanism evidence is therefore rho and the displacement, NOT only the sens ratio.
2. The PREREG ratio is divided by ||zhat_k - z_0||, which includes large action-independent drift; even the
   `all` swap can sit below 0.05 while the model clearly responds to a and v. Report all channels.
3. The O-COLLAPSE label (if the rule fires) is a statement about the PREREG's operational criterion
   (steer-deaf, speed-responsive). If rho and the displacement contradict the collapse mechanism, say so
   explicitly and run the copycat lever as well (RULE ZERO: a refutation of the mechanism is a waypoint).

### 6.1 How the caveats read against the data (POST-HOC, labelled; none of this moves the PREREG outcome above)
1. **Caveat 1 (mechanism differs) is borne out at the operative level.** rho_tok of the operative predictions is 0.126-0.132 (paper "healthy" 0.12,
   "collapsed" 0.008-0.016; indicative only) and the rollout moves the field by 36-42 % of ||z_0||. By the paper's definition a slow-feature collapse onto
   episode-constant features with an identity predictor would show rho near 0.01 and displacement near 0 (expectation, not measured by us); the operative level shows neither.
   The frozen target space (`target_space: frozen`) closes the encoder-collapse escape for the frozen part.
2. **Caveat 2 (denominator) is partly borne out.** The `all` swap reads 0.0742 at the operative level (not below 0.05 as the caveat allowed, but below
   the 0.10 SENS bar), while the post-decoder table (4.6a) shows `a` and `v` at ~0.15 in the space the loss reads, steer at 0.0097.
3. **Caveat 3 fires, partially.** What the data support is the OPERATIONAL statement "the operative and tactical predictors are steer-deaf
   (0.0043 / 0.0045) while accel/speed-responsive (a 0.038 / 0.198; v 0.059 / 0.090)". They do NOT support "H-JEPA's encoder/predictor
   collapse is the cause" at the operative level. The tactical level is numerically collapsed in rho (0.0161, in the paper's band) and moves little
   (0.061 of ||z_0|| at k = 10) yet responds to `a`, so a tactical slow-feature component is plausible but UNTESTED (cause of `tac0_tok == tac0_full`
   not inspected).
4. **Consequence (RECOMMENDATION, not a PREREG outcome):** run the PREREG lever (IDM, A1) AND the copycat lever
   (action-inferable-from-history, `2010.14876`; history-action dropout on the tiny rig), because rho and displacement do not corroborate
   the collapse mechanism. The cheapest discriminator: the A1 SENS bar (steer-sens >= 0.10 at k = 10, 23x above the baseline 0.0043)
   read together with the copycat arm on the same instrument.
5. **Input scale.** Donor-swap steer perturbations are mostly tiny (median window ratio 0.0007; tercile slope ~0.09 per rad, 4.6c), and the model's
   steer response is roughly linear in the perturbation. Part of the low steer ratio is therefore the swap's size, not only the model's gain; the
   PREREG ratio mixes both. The linear-action stub (section 3) is not a calibration for this.
