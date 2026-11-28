# D-REFAV1-CCOS-EVAL — the centred-cosine fix MEASURED on the real implementation, and refav1 re-evaluated with it

**status: DELIVERED (all three arms landed and analysed; the paired episode-cluster delta remains a work item) — was: IN PROGRESS — done: predecessor's uncommitted ccos plumbing rescued, tests made true (62 green) and COMMITTED (`044eafc`); the REAL-implementation goal-term panel on the full 282-window grid (§1, both probes, controls); the weight-neutrality factor and compensated triple (§2); the `cos` arm re-analysed with the `ha0_ext` echo control (§4); criteria checker on the `cos` record: 0 violations / 0 work items / const0 OK; echo gate 1 on the `cos` arm: FAIL (§4); the cross-dump paired tool passes its known-value control (exactly 0.0000 [0, 0] on all 10 metrics × 2 strata); THE SEED CHANNEL measured (§1e): every turn on this model is the decoded goal's canonical control winning against cv, and the pre-registered predictions for the compensated and chord arms are REVISED accordingly (timestamped, before their data). RUNNING: Thor chain (ccos naive → ccos compensated → chord, `/home/nvidia/refav1_ccos/`, launched 02:18Z, ~3.5 h per arm) and dev-box chain (chord → ccos compensated, `C:\Users\Admin\ccos_eval\devbox\`, ~2 h per arm). next: when `dump_ccos_naive` completes → analyse (`refav1_arm.py --analyze-only`) → shape panel (§1b–d) → suite + criteria → paired ccos-vs-cos on the same windows → echo gate → register rows.**

Arch+Inference FlyWheel · 2026-09-05 · checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt` step **21,109** (strict load, `missing_keys: []`; md5 `1189bc020018c2c67ce03d566c390285`, 2,122,997,633 B — identical on Thor and the dev box) · **OPEN LOOP** (PI ruling 2026-09-02): every planner arm is **T1** (self-action open loop: the predictor consumes the planner's own actions); the WM diagnostic is T0. Nothing here is driving performance.

⚠️ **The brief named `/home/nvidia/experiments/refav1-b1-v72-1ep-21109` as the checkpoint. That directory is an EARLIER run: its `train_log.jsonl` stops at step 1,000, it carries a `DRIFT_ALARM`, and no `summary.json`.** The step-21,109 model is `refav1-b1-v72-ep3-speed/ckpt.pt` (`summary.json`: `"done": true, "final_step": 21109`; `MODEL_REGISTRY.md` §2.4; the banked `cos` eval's own `launch.sh`). Every number below is on that checkpoint.

**Two sample sizes, never mixed.** The panel (§1) and the arms (§3) are the **full 282 windows / 141 episode clusters** of the banked `refav1-21109-openloop` grid (v7.2 EVAL split, `--window-stride 40`, K = 10 @ 0.2 s, `--action-units kappa`, `--no-navshuf`). The pilot factor (§2) is the predecessor's **40-window / 20-clip** dev-box form comparison. Intervals: **episode-cluster bootstrap** (`taniteval/ci.py`), **paired** for two arms on the same windows; `overlapping_holdout_se` nowhere.

---

## 0. What is settled so far (evidence class beside each)

| finding | class |
|---|---|
| **The fix exists in code and is default-safe.** `COST_METRICS = ("cos", "chord", "ccos")`; `_goal_term(zt, g, "ccos", z_ref) = 1 − cos(z_t − z_ref, g − z_ref)` with `z_ref` the window's own zero-action terminal field from `plan()`'s `_zero_action_ref` (same predictor, same `z0`, same forward pass; refuses any other centring); `"cos"` bit-identical to the pre-change expression; `plan(cost_weights=)` carries the weight triple on the call and stamps it on the result. 62 tests green. | MEASURED (`stack/tests/test_cost_ccos.py`, `test_cost_chord.py`, `test_refav1_arm.py`; commit `044eafc`) |
| **On the REAL implementation, over all 282 windows: `cos` excludes 100.00 % of a 300-sample iCEM iteration-0 population before the world model is consulted (goal(cv) median 1.19e-07, max 5.58e-04 against a smallest penalty 0.094); `chord` — the uncentred, monotone-equivalent form, the deliberate regression — ALSO 100.00 %; `ccos` 1.67 % at the median window, 0.33 % at the most favourable.** | MEASURED (`raw/panel_282.json`, from `thor/box_panel_282.json`; second probe `thor/ccos_panel_probe.json` agrees on cos/chord to ≤ 2.2e-07) |
| **But `ccos` has a measured HOLD stratum on 133/282 = 47.2 % of the grid**: there the decoded goal's canonical controls are zero, the goal rollout IS the zero-action rollout (`‖g − z_ref‖/‖z_ref‖ < 1e-6`), the centred goal direction is the zero vector, and **every candidate reads exactly 1.0 (ptp 0.0 on 133/133)** — the term carries no information and the penalties alone decide, so the injected zero-penalty baselines win exactly as under `cos`. On the other 149 windows the term spans its range (box spread median 1.99 of [0, 2]) and the L/R decision is 193 % of the term's magnitude (vs 59 % for `cos` measured in float32 ulps, 30 % for `chord`). | MEASURED (`raw/panel_282_strata.json`) |
| **`ccos(cv)` is 1.0 exactly only when cv's field is bit-identical to `z_ref`.** In the planner's own batch the cv row is rolled beside other candidates while `z_ref` is rolled alone, so cv's centred vector is batch-composition noise and its cosine against `g − z_ref` is O(1/√D): on the 149 non-HOLD windows `ccos(cv)` = 1.0 ± 0.062 (median |dev|), range **0.777–1.177**. The planner's floor is therefore partly decided by rounding — a defect of the same family as the `cem`-label mislabel (RESULT 2026-09-04 §5), now on the value rather than the label. The smoke read `basecost_cv_cl` 0.959 / 1.033 on its two windows. | MEASURED (`thor/box_panel_282.json` `controls.ccos_cv`; `thor/dumps/dump_smoke/decisions`) |
| **Weight neutrality: `ccos` is NOT weight-neutral, by a decision-scale factor of 643× [p10 260, p90 1,560] on the non-HOLD stratum (n = 149; 129× over all 282 because the HOLD windows contribute 0; pilot 214× on n = 40) and a value-scale factor of 2,850× (n = 149) / 33,715× (pilot).** The compensating triple at the decision scale is **(12.859, 32.149, 64.297)**; the closed-form iteration-0 bound then reads **100.00 % excluded** again for the RANDOM population. ⚠️ That was first read as "compensation restores the flat plan by construction" — the seed channel (§1e) shows the injected canonical goal control escapes the bound (it beats cv on 91/282 windows at ×643), so the compensated arm is predicted to emit seed plans, not to be flat; the prediction history is kept in §1b–d. | MEASURED (`raw/panel_282_strata.json`, `raw/ccos_scale_factor_devbox40.json`, `raw/arm2_weights_decision.json`) |
| **The `cos` arm against the echo control `ha0_ext` (constant (a0, κ0) of the measured t0 state, T1, added post hoc with three content checks):** ADE `cl − ha0_ext` = +0.0267 [−0.0285, +0.0813] straddles; LON speed MAE **+0.2706 [+0.2140, +0.3312] separated WORSE**; LAT cross-track **−0.1409 [−0.1844, −0.1015] separated BETTER**, heading −1.54° [−2.04, −1.07], yaw-rate −0.0409 [−0.0529, −0.0297] (both better); TAC lon correct **−0.1277 [−0.1879, −0.0674] separated WORSE**. ⇒ the straight-line plan beats the ego-extrapolation laterally only because `ha0_ext` holds the recorded κ0 (which drifts), and loses to it longitudinally — the same "wins by doing nothing" pattern as against `ha`. `cl − ha0` unchanged: +0.0158 [+0.0007, +0.0315]. | MEASURED (`thor/rec_cos_ext.json`, n = 282 / 141, paired episode-cluster bootstrap n_boot 2000) |
| **The smoke (ccos naive, 2 windows of clip 01be5919): the planner TURNED** — window 1 (decoded goal TURN_R) emitted a `cem`-labelled plan with κ = −0.08 for 10 steps — **bit-exact the injected canonical goal SEED** (`GOAL_KAPPA_TURN`, §1e), i.e. goal-following, not a searched plan; window 0 (goal BRAKE_TO) emitted the injected `decel_1.5`. The dev-box `chord` smoke emitted the identical seed plan on the same window. **n = 2; a smoke, not a result.** | MEASURED (`thor/dumps/dump_smoke`, `raw/shape_cos_and_smoke.json`) |

---

## 1. The panel — the REAL `_goal_term` under `cos` / `chord` / `ccos` on the same captured fields (n = 282 / 141)

Source: `thor/box_panel_282.json` (predecessor's `ccos_box_panel.py`, Thor, 443 s: one `plan(cost_metric=m)` capture PER metric so each metric's closure — including `z_ref`, which only the `ccos` closure builds — is the planner's own; 22 designed candidates: cv, decel_1.5, ±κ ∈ {0.2, 0.1, 0.05, 0.02, 0.01}, ±a ∈ {4, 2, 1.5, 0.5}, turnL/R_slow, rampL/R; float32 as the planner sees it) cross-checked by an independently written second probe (`tools/ccos_panel_probe.py`, 304 s) and by the float64 re-implementation banked beside both.

### 1a. Fraction of the iteration-0 population excluded before the world model is consulted

Closed-form bound (`analyse8_exclusion.py`'s rule, unchanged): the goal term is non-negative, so candidate n beats cv only if `penalty(n) < goal(cv) − goal(n) ≤ goal(cv)`. `penalty` is `W_JERK·jerk_raw + W_KAPPA·κ_raw` of the REAL 300-sample iteration-0 population (`colored_noise`, seed 0: jerk_raw min 4.711, median 18.01 ⇒ penalty median 0.362, max 1.344).

| metric | goal(cv) median [CI] | goal(cv) max | exactly 0 / exactly 1 | box spread (median) | L/R decision as % of term [CI] | **iter-0 EXCLUDED @median / @max window, shipped weights** |
|---|---|---|---|---|---|---|
| `cos` (shipped) | 1.19e-07 [5.96e-08, 1.79e-07] | 5.58e-04 | 95 / 0 | 2.42e-03 | 31.3 % [30.5, 38.4] *(ulp-quantised: the term is a handful of 2⁻²⁴ steps)* | **100.00 % / 100.00 %** |
| `chord` (deliberate regression) | 3.63e-04 [9.5e-08, 5.5e-04] | 3.34e-02 | 0 / 0 | 6.84e-02 | 8.5 % [0.8, 15.6] | **100.00 % / 100.00 %** ✅ *fails, as it must* |
| `ccos` | 1.0 [1, 1] | 1.177 | 0 / 133 | 1.95 | 189 % [0, 191] | **1.67 % / 0.33 %** (4.0 % with the tighter `goal(cv) − min_n goal(n)` bound) |

By stratum (`raw/panel_282_strata.json`; HOLD = `‖g − z_ref‖/‖z_ref‖ < 1e-6`, i.e. the decoded goal's canonical controls are zero):

| stratum | n (win / clusters) | `ccos` goal(cv) | `ccos` box spread | `ccos` L/R share | `cos` L/R share | identity control (candidate = goal) |
|---|---|---|---|---|---|---|
| HOLD | 133 / 85 | **1.0 exactly on 133/133; every candidate 1.0, ptp 0.0** | 0.0 | 0 % | 18 % | `ccos` reads 1.0 (zero vector vs zero vector — no information, correct) |
| non-HOLD | 149 / 93 | 1.009 median, **0.777–1.177** (batch-noise, see §0) | 1.99 | **193 %** | 59 % | `cos` 2.4e-07, `chord` 0.0, `ccos` 0.0 |

**Controls, real function, real fields:** zero-model ptp (all candidates = the cv field) = **0.0 exactly** for all three metrics on 282/282; identity control `chord` = 0.0 exactly (Sterbenz), `cos` ≤ 2.38e-07 and `ccos` = 0.0 on non-HOLD windows (the float32 cosine of identical vectors, as `test_c` now states); float64 cross-check max |ccos32 − ccos64| = 2.49e-07; the second probe agrees on `cos`/`chord` to ≤ 2.2e-07 on every cell, on `ccos` to ≤ 8.8e-06 on every non-cv row, and differs on the **cv row by up to 0.031** — exactly the batch-composition residual (its box has a different size), which is the mechanism of §0's `ccos(cv)` finding, reproduced by two tools.

### 1e. ⭐ THE SEED CHANNEL — what "the planner turns" actually means on this model

`plan()` seeds iCEM with the decoded tactical goal's **canonical control** (`canonical_controls(lat, lon, v0)[:plan_steps]`, `refa_v1.py:2188`): for a TURN token that is κ = ±`GOAL_KAPPA_TURN` = ±0.08 held for 10 steps at zero jerk, i.e. a candidate whose penalty is `0.05 × 0.08² = 3.2e-04` — **inside the population's exclusion bound by 300×**. The bound (§1a) covers the RANDOM population only. On the one window where both smokes turned (clip 01be5919, t = 43, goal TURN_R × ADAPT_SPEED_FOR_CURVE, v0 7.73), the emitted plan is **bit-exact the seed** under `ccos` AND under `chord` (`raw/shape_cos_and_smoke.json` `c_seed_channel`). So on this model a turn is **goal-FOLLOWING** — the tactical decoder's own control winning the total-cost comparison against cv — and the world model's role is to accept or reject that seed, not to search.

Measured on the panel's `seed0` / `proposal` rows (`raw/panel_seed_channel.json`; the closure's total cost equals goal + shipped penalty to ≤ 2.1e-07 on every cell, so the accounting is the planner's own):

| metric | seed beats cv @ shipped (of 282) | … on the 38 TURN-goal windows | proposal beats cv | seed beats cv @ compensated ×643 | … on TURN-goal windows |
|---|---|---|---|---|---|
| `cos` | 7 (all non-HOLD; 0 HOLD) | **0 / 38** | 0 | 7 | 0 |
| `chord` | 67 | **5 / 38** | 0 | 7 | 0 |
| `ccos` | **149 = every non-HOLD window** | **38 / 38** | 75 | **91** | **32 / 38** |

Seed penalty median 7.67e-05, max 5.59e-03 (the seed is non-zero on 149 windows: κ on 38, accel profile on 120). ⇒ under `ccos` the seed wins wherever the goal direction is real; under `cos` the goal advantage (≤ 5.6e-04) never covers even the seed's 3.2e-04 on a TURN window. **This is the mechanism of every turn the arms will show, and it is why "monotone-equivalent" did not protect the chord: the equivalence is of the goal term alone, and the total cost adds the penalty on a fixed scale.** The banked `cos` dump reads consistently: 140/282 plans are bit-exact the seed — the ZERO seed on the 133 HOLD windows, and on **7 of the 12 "decel" windows the seed itself**: all 12 non-zero `cos` plans sit on `ADAPT_SPEED_FOR_CURVE` goals at v0 = 12.3–18.0 m/s, where the canonical control asks for a large speed drop; on 7 it saturates at −1.5 m/s² for all 10 steps and the plan IS that seed, on the other 5 the seed decays (−1.5 → −0.9…−1.4, i.e. it carries jerk) and the plan is the zero-jerk `decel_1.5` block standing in for it; 0 searched plans. ⚠️ **This amends an INTERPRETATION of 2026-09-04, not a number**: the 12 decels were read as *"anti-correlated with the goal (0 of the 33 BRAKE_TO windows, hypergeometric p = 1)"* — they are perfectly correlated with the *other* deceleration token, the saturated `ADAPT_SPEED_FOR_CURVE` seed (12/12). Flagged for the Master Mind / RETRACTION_LOG (class: a correlation tested against one token of a two-token condition).

### 1b–1d. Shape of the emitted plans — `cos` banked; `ccos` / `chord` arms RUNNING

| arm (n = 282 / 141 unless stated) | L / R / straight | distinct plans | bit-exact to an injected baseline (zeros ∪ decel_1.5) | κ ≡ 0 | plan turns on the 38 TURN_L/R-goal windows (sign agrees) |
|---|---|---|---|---|---|
| `cos` @ shipped (banked 2026-09-04, re-read by content) | 0 / 0 / 282 | **2** | 270 + 12 = **282/282** | **1.0000** | 0/38 (—) |
| `ccos` @ shipped — NAIVE (smoke, n = 2 / 1) | 0 / 1 / 1 | 2 | 1/2 (decel) | 0.5000 | 1/1 (1/1) |
| `ccos` @ shipped — NAIVE, full grid | *running on Thor* | | | | |
| `ccos` @ compensated (12.859, 32.149, 64.297) | *queued (Thor arm 2 / dev box arm 2)* — **PRE-REGISTERED PREDICTIONS, with their revision history kept visible.** *(02:50Z, first version)* (P1) κ ≡ 0 on ~100 % and only zero-penalty plans (100 % iteration-0 exclusion by the bound); (P2) the cv-vs-decel split becomes goal-correlated. ⚠️ **REVISED 03:00Z — still before any compensated window existed — after the seed channel was measured (§1e): the bound covers the RANDOM population only; the injected canonical goal SEED carries zero jerk and ≤ 3.2e-04 of κ penalty and beats cv on 91/282 windows at ×643 (32 of the 38 TURN-goal windows). ⇒ (P1′) the compensated arm is NOT flat: it emits the seed on up to 91 windows (κ ≠ 0 on ≤ 32, all of them TURN-goal, sign = the decoded goal's), zeros/decel elsewhere, and NO searched plan (the population stays 100 % excluded). (P2) unchanged.** | | | | |
| `chord` @ shipped (deliberate regression, full grid) | *running on the dev box* — **PRE-REGISTERED PREDICTION (first version): κ ≡ 0 on 100 %** (24-window sweep S6 of 2026-09-04 read 100 %). ⚠️ **REVISED 03:00Z, before the arm had produced more than 14 windows: the chord cannot re-rank the GOAL TERM, but the TOTAL cost adds the penalty on a fixed scale, so a 28× decision-leverage re-weight opens the seed channel — the seed beats cv on 67/282 windows under `chord` at shipped weights, on 5 of the 38 TURN-goal windows (§1e). ⇒ (C′) κ ≠ 0 on ≤ 5 windows, every one of them the seed (goal-following), none searched; the smoke already showed one (t = 43 of clip 01be5919, bit-exact seed).** | | | | |

---

## 2. Weight neutrality — the factor and the compensating triple

| quantity | pilot (float64 re-implementation, n = 40 / 20, dev box) | REAL implementation, all 282 | REAL, non-HOLD 149 / 93 |
|---|---|---|---|
| value leverage `ccos / cos` (median cell ratio) | 33,715× | 5,166× | 2,850× |
| **decision leverage** (median ratio of the term's spread over the candidate box) | **214.46×** | 128.6× [0, 318] | **643.0× [p10 260, p90 1,560]** |
| L/R-gap ratio | — | 123,650× | — |
| `chord / cos` decision leverage (for scale; chord cannot re-rank) | — | 28.2× [26.5, 29.9] | — |

**Compensating triple** = shipped × F. At the decision scale (what trades against the penalties) with F = 643.0 (the stratum where the term varies at all — on HOLD windows any factor gives the same flat plan): **`(W_JERK, W_KAPPA, W_VEND) = (12.859, 32.149, 64.297)`**, banked in `raw/arm2_weights_decision.json` and shipped to both chains as `arm2_weights.txt` BEFORE any compensated arm started; the pilot triple (4.289, 10.723, 21.446) is superseded and recorded. `W_VEND` is never charged on the eval path (`target_speed=None`). Under either triple the iteration-0 bound reads **100.00 % excluded** — so the compensated arm is predicted to reproduce the flat plan, which is what makes it the control: any difference between the naive and the compensated arm is the implicit re-weight, not the centring.

⚠️ The predecessor's `test_cost_ccos.py` docstring quotes a value leverage of "751,546×" from an unbanked `raw/ccos_scale_factor.json`; that file never existed on disk. The definitions above are stated and the numbers re-measured; the decision leverage matches the predecessor's 214.46 exactly on the same 40 windows.

---

## 3. The arms — LANDED (Master Mind, 2026-09-05, after this stream's agent died at a model rate limit)

All three arms completed on Thor and are analysed here. Same **282 windows / 141 episode clusters**,
same checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt` step 21,109, episode-cluster bootstrap n_boot 2000.
⚠️ **Tier: `cl` is T1** (self-action open loop — the model consumes its own actions, it does NOT
control the vehicle); `ol` is **T0** and is a world-model diagnostic, never driving performance.

### 3a. The headline table

| arm | CEM's own plan used | ADE 0–2 s (T1) | speed MAE (m/s) | along MAE (m) | cross MAE (m) | heading (°) | curvature (1/m) |
|---|---|---|---|---|---|---|---|
| **`cos` — the SHIPPED default** | **33.7 %** | **0.5474** [0.4839, 0.6108] | 0.5412 | 0.4241 | 0.2257 | 2.462 | 0.0072 |
| **`ccos` naive flip** (no weight compensation) | **48.9 %** | **0.9268** [0.7808, 1.0819] | 0.6814 | 0.5890 | 0.5278 | 4.841 | 0.0153 |
| ⭐ **`ccos` weight-compensated** | **46.1 %** | **0.7116** [0.6291, 0.7973] | 0.7175 | 0.5506 | 0.2902 | 3.272 | 0.0099 |
| `ha` (hold action) | — | 0.5391 [0.4544, 0.6301] | 0.2706 | 0.2937 | 0.3826 | 4.673 | 0.0158 |
| `ha0` (constant velocity at v0) | — | 0.5316 [0.4710, 0.5939] | 0.5104 | 0.3984 | 0.2257 | 2.462 | 0.0072 |
| `ha0_ext` (constant a, constant yaw-rate) | — | 0.5207 [0.4421, 0.6081] | 0.2706 | 0.2864 | 0.3666 | 4.557 | 0.0158 |
| `ol` (⛔ **T0**, teacher-forced — a WM diagnostic) | — | 0.4237 [0.3466, 0.5070] | 0.0991 | 0.2018 | 0.3351 | 4.266 | 0.0143 |

Weights: the compensated arm uses `(W_JERK, W_KAPPA, W_VEND) = (12.859430, 32.148575, 64.297150)`
(`C:\Users\Admin\ccos_eval\arm2_weights.txt`), derived from the measured `ccos` scale factor.

### 3b. ⭐ The mechanism works — and the fingerprint that proves it

**`ccos` un-sticks the planner.** The CEM's own optimisation supplies the emitted plan on **46.1 %**
of windows against **33.7 %** under `cos`; the injected baseline correspondingly falls 66.3 % → 53.9 %.
That is the defect this fix was built for: under `cos` the goal term realised ~1.2e-05 of its [0, 2]
range and could not outvote the jerk penalty, so the search was decided before the world model was
consulted.

**The fingerprint.** Under `cos` the planner's LATERAL row is **bit-identical to `ha0`'s** — cross
0.2257, heading 2.462, curvature 0.0072 on both, to four decimals. That is not a coincidence: on the
windows where the injected baseline wins, the emitted plan *is* the constant-velocity rollout, so the
"planner" was scoring as the trivial control by construction. Under `ccos` those values move away
(0.2902 / 3.272 / 0.0099) — the planner is genuinely planning.

**Weight compensation is real and was predicted.** The naive metric flip costs ADE 0.5474 → 0.9268;
compensating the implicit re-weighting recovers most of it (0.7116). This confirms the banked
5,792.6× factor for `chord` generalises: a metric change without a weight statement is not a
one-variable arm.

### 3c. ⛔ And it is REFUTED as an improvement

**Every family gets worse, and the arm loses to the trivial controls.** ADE 0.5474 → 0.7116; speed MAE
0.5412 → 0.7175; along-track 0.4241 → 0.5506. Against the controls the compensated arm's ADE interval
[0.6291, 0.7973] does not overlap `ha0`'s [0.4710, 0.5939]; it grazes `ha`'s upper bound (0.6291 vs
0.6301). ⚠️ **These are per-arm intervals, not the paired delta** — the paired episode-cluster
comparison is the decision-grade form and is a work item; the direction, however, is not in doubt,
and no reading of it makes `ccos` a gain.

⇒ **VERDICT: the diagnosis was right, the repair works mechanically, and a working goal term makes
REF-A v1 drive WORSE.** The problem was never the metric's arithmetic. It is what the cost surface
prefers once the goal term can be heard — which corroborates `D-COST-CHORD` from a second direction
(*"once the goal term is legible, the world model itself prefers straight"*), now with the goal term
not merely legible but dominant enough to move half the decisions.

### 3d. What this settles, and what it does not

* ⛔ **`ccos` must NOT become the default.** The shipped default stays `cos`, bit-identical, and every
  banked refav1 number remains reproducible.
* ✅ The exclusion defect is **fixed and demonstrated fixed** — a reusable instrument, not a dead end.
* ⚠️ **It does not follow that refav1's planner is unfixable.** What is now measured is that
  *legibility alone is insufficient*: the cost surface's minimum is in the wrong place. The next
  question is the surface, not the metric — the ranked candidates are the jerk weight's dominance
  (`W_JERK` is 0.02 against a goal term that now competes) and the seed channel (§1e).
* ⚠️ **Scope.** One checkpoint, 282 windows, T1. The brief named
  `refav1-b1-v72-1ep-21109`; this stream correctly used `refav1-b1-v72-ep3-speed` instead, because the
  named directory is an EARLIER run whose `train_log.jsonl` stops at step 1,000 and carries a
  `DRIFT_ALARM`. The STRATEGIC family is **unavailable with reason** on all arms (no route channel in
  this surface); distance-keeping is PRESENT (n = 87–90 windows with a lead agent).

## 4. The `cos` arm with the echo control, and echo gate 1

`thor/rec_cos_ext.json` (suite `suite_cos_ext/`: 0 violations, 0 work items, const0 OK). **Echo gate 1** (`stack/tanitad/eval/echo_gate.py::echo_gate`, references `ha`, `ha0_ext` (required) + `ha0`, per horizon slot, paired episode-cluster bootstrap, `raw/echo_gate1_cos.json`): **FAIL** — at 1.0 s the arm (0.3679 m) is separated WORSE than `ha` (0.3219; Δ −0.0460 [−0.0805, −0.0102]), `ha0_ext` (0.3120; Δ −0.0559 [−0.0874, −0.0232]) and `ha0` (0.3561; Δ −0.0118 [−0.0219, −0.0021]); at 2.0 s every reference straddles (`ha` +0.0936 [−0.0899, +0.2877], `ha0_ext` +0.0422 [−0.1144, +0.2109], `ha0` −0.0354 [−0.0769, +0.0044]). Gates 2 / 2b need the model re-run under deranged inputs and are REFUSED from a dump (n = 0, reason recorded); `assert_not_echoing` is NOT called with a fabricated gate 2. The functional ablation (`tools/refav1_source_ablation.py`) is queued on the dev box after its chain.

Stratified paired reads on the SAME windows (`raw/paired_control_cos_vs_cosext.json`, n_boot 500; the control pair `cosext.cl − cos.cl` reads **exactly 0.0000 [0, 0]** on all 10 metrics in both strata): on the **149 non-HOLD** windows `cos.cl − ha0` ADE **+0.0299 [+0.0046, +0.0589] separated WORSE**, FDE +0.0669 [+0.0033, +0.1426], LON speed +0.0584 [+0.0149, +0.1059], LON along +0.0488 [+0.0166, +0.0855]; on the **133 HOLD** windows every `cl − ha0` delta is identically 0 (the plan IS `ha0` there). Against `ha0_ext`: LON speed +0.3411 [+0.2540, +0.4273] worse / LAT cross −0.1815 [−0.2630, −0.1110] better on non-HOLD; +0.1916 / −0.0955 on HOLD (both separated).

## 5. Provenance

| what | where |
|---|---|
| code | commit `044eafc` on `agent/arch-inf-20260803` (`refa_v1.py` `cost_weights`; `refav1_arm.py` `--cost-metric/--cost-weights`, `ha0_ext`, `basecost_*`; `refav1_add_floor.py`; `t1_eval.py` `ha0_ext: T1`; tests) |
| Thor tree | `/home/nvidia/refav1_ccos/repo/` = `evalstack_ccos.tgz` md5 `cfc43a35fe0289b7616586c47a459a7c` built from the mirror; `refa_v1.py` md5 `c3dab40241be4830c27c447f46e00946`, `refav1_arm.py` `4ac869ee0c92a5be49e4f5150dee5491`, `t1_eval.py` `d944a5fb4197acce9861c076ac42e47f` — verified on both ends |
| dev-box inputs | `C:\Users\Admin\refav1_eval_full\{fp8 (8.7 GB, 142 files), eps (5.1 GB, 141)}` pulled from Thor by tar stream; labels md5 `aa12c948f062181c3297265b51526ec5` (identical to the banked run) |
| estimator | episode-cluster / paired episode-cluster bootstrap, n_boot 2000 |
| cross-box control | Thor (torch 2.13.0+cu130) vs dev box (2.11.0+cu128), same clip / windows (01be5919, t = 3, 43): GT, `ha`, `ha0`, `ha0_ext`, `ol` **bit-identical (max\|Δ\| = 0.0)**; forward-pass WM error max\|Δ\| 2.98e-07; every head decode (lat/lon/route under nav_true, decoded goal tokens) identical. Cross-box differences are ulp-level. |
| pipeline dry-run | `tools/finalize_arm.sh` on the 282-window `cos`+`ha0_ext` dump (`final/cos_ext_dryrun/`): analyze → suite (0 violations / 0 work items, const0 OK) → `tools/criteria_check.py` → shape → paired-vs-cos (exactly 0.0000 [0, 0] on every metric, the known-value control) → echo gate 1 (FAIL, as §4) — 4 min, zero GPU |
