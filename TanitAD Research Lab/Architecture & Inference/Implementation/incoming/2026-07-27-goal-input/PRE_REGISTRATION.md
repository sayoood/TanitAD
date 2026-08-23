# PRE-REGISTRATION — is the goal input a capability or an oracle artifact?

**Stream:** `2026-07-27-goal-input` · **Host:** dev box, CPU only · **Written:** 2026-07-27,
**before any number in this stream was computed.** Nothing below was edited after the first
measurement; corrections and surprises are recorded in `GOAL_INPUT.md`, never here.

**Estimator (binding):** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`,
**B = 2000**, resampling unit = **episode** (40 episodes, 881 windows).
`overlapping_holdout_se` is **never** called — it biases the point estimate as well as the
interval (`CLAUDE.md`). Not separated at n = 40 ⇒ **UNPOWERED, NOT REFUTED**.

---

## 0. What produced this, and the objection that governs the whole design

`…/incoming/2026-07-27-fan-conditioning/FAN_CONDITIONING.md` §7 measured that handing the
selector the **true 2 s goal position** — 2 of the 8 target scalars — moves the realised pick
**0.4714 → 0.2009 [0.1689, 0.2351]**, paired **−0.2705**, separated: **88.0 %** of the fan's
headroom, clearing both pre-registered bars, replicated on 3 fans.

⛔ **The objection that must be answered before anything else:** the 2 s goal position is
(nearly) the thing we are trying to predict. *"Give the selector the answer and it picks
better"* may be a tautology. **The claim has content only if predicting a 2-D endpoint is
genuinely EASIER than picking among 256 eight-dimensional trajectories.** That is S1 below,
and it is priority 1 because nothing else is interpretable without it.

---

## 1. Inputs, and the leakage audit that fixes the admissible feature set

All inputs are artifacts **already committed to this repo**. No pod is contacted, no
checkpoint is loaded, no episode is opened. Parity (`physicalai-train-e438721ae894`,
skip-hash `f09e44db`) is untouched; the dev box's non-parity cache `14231cd29c74` is never read.

| artifact | supplies |
|---|---|
| `taniteval/results/fan_refc-{xl,base}-30k.pt` | the fan, `sel`, `gt`, `cv`, `v0`, `logits` |
| `…/2026-07-22-refc-small-30k/fan_refc-small-30k.pt` | replication fan |
| `…/2026-07-27-t3-and-lambda-tau/raw/eh2_cache.pt` | v4's **latent-derived** `lat`(8) / `lon`(7) / `dist`(8), `refined_pre`(256), `prior`(256), on the **same 881 windows** |

### 1.1 ⛔ ORACLE FIELDS — pre-declared inadmissible as head inputs

Verified in source before the feature set was fixed:

| field | why it is future-derived | source |
|---|---|---|
| `head_deg` | `\|net heading change\|` over the **future** `K_MAX` steps | `stack/scripts/driving_diagnostic.py:139-142` |
| `a_gt` | ground-truth acceleration | fan dump |
| `v_target`, `vt_valid`, `vt_lookahead` | `lake.vtarget` = 85th pct of **future** speed, 10–20 s ahead | `stack/scripts/goal_modes.py` |
| `gt` (any waypoint) | the target | — |
| `route`, `route_graded` | `route_from_future_v3`, ≤25 s **forward** | `stack/scripts/goal_modes.py` |

⚠️ `head_deg` looks like an ego-state scalar and is not one. Any head that reads it is
measuring an oracle. **This is stated in advance so that a good number cannot be rationalised
later.**

### 1.2 ✅ ADMISSIBLE — observation-only, verified in source

| set | dims | content | provenance check |
|---|---:|---|---|
| `F_lat` | 23 | `lat`+`lon`+`dist` = projections of v4's world-model latent `state_last` | `bar_a_selector.py:299` `lat = head.lat_head(stl)`; `state_last = world.encode_window(frames)` — frames only |
| `F_ego` | 9 | `v0` + the 4 CV waypoints (8) | `baseline_waypoints` uses `poses[last]`/`poses[last-1]` **only** |
| `F_ans` | 4 | the selector's OWN answer: as-trained pick endpoint (2) + logit-softmax-weighted fan endpoint (2) | derived from `fan`,`logits`,`sel` — all deploy-time |
| `F_conf` | ≤32 | PCA of `refined_pre`(256) ⧺ `prior`(256), fitted **on the train fold only** | v4 selector confidences, latent-derived |

`F_ans` is included **deliberately**: it lets the head copy the selector, so the tautology test
cannot be won by withholding information from the head.

⚠️ The `eh2_cache` features come from **v4**'s latent while the fan comes from **REF-C**. This
is a *cross-arm* read and is declared as a threat to validity, not hidden: it is the only
latent staged on this host. `neutral` (no goal channel) is the primary variant; `produced`
(the model's own predicted goal scalars) is a robustness arm. Both are observation-only.

---

## 2. S0 — GATE. No new number is quoted until every committed one reproduces

Bit-level reproduction, from raw, before anything else runs. `_all_ok` must be `true`:

| quantity | committed |
|---|---:|
| REF-C-XL as-trained `ade_0_2s` | 0.4714 |
| REF-C-XL `oracle_in_fan` | 0.1640 |
| `R_goal2s` realised | 0.2009 |
| `R_goal2s` − as-trained (paired) | −0.2705 |
| goal − no-goal (paired) | −0.6149 |
| REF-C-base / small `oracle_in_fan` | 0.1914 / 0.2213 |
| window identity `eh2_cache` ↔ REF-C fan | `v0` identical, `max_abs_diff = 0` |

**If the gate fails the stream stops and reports the failure.**

---

## 3. S1 — ⛔ THE TAUTOLOGY TEST (priority 1)

### 3.1 The axis

Both quantities are expressed as **2 s endpoint L2 error in metres**:

```
e(p̂) = ‖ p̂(2s) − p_GT(2s) ‖₂          [metres]
```

- `e_sel` — the as-trained selector's achieved endpoint error: `‖fan[w, sel_w, −1] − gt[w, −1]‖`.
  **This is the trajectory-picking error expressed on the goal axis.**
- `e_head` — a learned goal head's **out-of-fold** endpoint error.

Secondary axis, reported alongside: `ade_0_2s` (mean-over-4-waypoint L2), the program headline.

### 3.2 The arms (all endpoint error; 5 **episode-disjoint** folds, seed fixed)

| arm | what it is | training |
|---|---|---|
| `H0_const` | train-fold mean endpoint | learn-nothing floor |
| `H1_cv` | the constant-velocity endpoint | zero-training kinematic |
| `H2_sel` | **the as-trained selector's own endpoint** | zero-training — **the bar** |
| `H3_ridge_lat` | ridge on `F_lat`+`F_ego` | OOF |
| `H4_ridge_all` | ridge on `F_lat`+`F_ego`+`F_ans`+`F_conf` | OOF |
| `H5_mlp` | 2-layer MLP, same features | OOF |
| `H6_gbr` | gradient boosting, same features | OOF |
| `H7_insample` | best of the above, fitted **on the evaluation windows** | **the bound** — discharges "the head is undertrained / n=881 is thin" in advance |
| `NC_shuffle` | features permuted across windows | must land at `H0_const` |

### 3.3 Decision rule — and the failing values, stated in advance

Let `Δ_taut = e_head − e_sel`, paired episode-cluster bootstrap.

- **T-PASS** — `Δ_taut < 0`, **separated** ⇒ predicting the endpoint IS easier than picking the
  trajectory; the decomposition has content. Report the ratio `e_head / e_sel`.
- **T-REFUTE** — `Δ_taut ≥ 0`, or separated-worse ⇒ **the decomposition buys nothing and the
  88 % is an artifact of handing over ground truth.** This is a complete answer and it is
  reported with equal force.
- **T-UNPOWERED** — not separated either way at n = 40; state the n at which it would separate.

⚠️ **T-PASS is NECESSARY, NOT SUFFICIENT.** A regression head minimises L2 by shrinking toward
the conditional mean; a *shrunk* goal can have lower endpoint error and still be a **worse
reference** for a nearest-candidate rule. The operative test is S3.

**Can each rule return a failing value?** ✅ Yes, and it is cheap to see:
`H0_const`'s error is the marginal spread of the 2 s displacement (tens of metres), which is
far above any plausible `e_sel`; if the latent carries no endpoint information every learned
arm collapses onto `H0_const` and **T-REFUTE returns**. Conversely `H7_insample` can drive
`e_head` toward 0, so **T-PASS is reachable**. Neither outcome is structurally excluded.

---

## 4. S2 — THE GOAL-ERROR CURVE (priority 2): how accurate must a goal be?

The engineering requirement every downstream option is measured against.

### 4.1 Construction

The reference is built exactly as `fanc_goal.py` builds it, with the goal perturbed:

```
ĝ_w   = g_w + δ_w                       g_w = gt[w, −1]   (the true 2 s endpoint)
R_w   = ĝ_w[None,:] * frac              frac = [0.25, 0.5, 0.75, 1.0]
pick  = argmin_c  mean_i ‖ fan[w,c,i] − R_w[i] ‖      (mean-over-waypoint L2 — the SAME
                                                        metric that scores the pick)
realised(δ) = mean_w  ade_0_2s( fan[w, pick_w] , gt[w] )
```

### 4.2 Noise families

| family | δ | parameter |
|---|---|---|
| `ISO(σ)` | `σ·N(0,I₂)` | **σ = per-axis SD, metres**; radial RMS = `σ√2`, mean radial = `σ√(π/2)` |
| `LONG(σ)` | `σ·N(0,1)` on component **0** (along-track) only | same |
| `LAT(σ)` | `σ·N(0,1)` on component **1** (cross-track) only | same |
| `SHRINK(α)` | `ĝ = ḡ_train + α(g − ḡ_train)` | the **realistic regression-head error model** (biased, not isotropic) |

σ grid: `0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0` m.
α grid: `1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0`.
**16 noise seeds per cell**; the curve is the seed-mean, and the paired CI is reported at the
cells bracketing 50 % and 0 % recovery.

⚠️ **Units, not correlations** (retraction class `CORRELATION-WITHOUT-SLOPE`): every point on
this curve is quoted as *metres of goal error → metres of `ade_0_2s`*, and the requirement is
published in **metres of radial goal error**, never as an r or an R².

### 4.3 Read-outs, pre-registered

```
recovery(δ) = ( A0 − realised(δ) ) / ( A0 − realised(0) )      A0 = as-trained = 0.4714
```
- **σ₅₀** — radial goal error at which `recovery = 0.50`. *(the spec if half the prize is enough)*
- **σ₀** — radial goal error at which `recovery = 0`, i.e. `realised = A0`. *(break-even)*
- **the negative region** — `recovery < 0`, a goal that is **worse than no goal**. Reported with
  its σ and its paired interval. ⚠️ **Not assumed; measured.** `R_cv` already realises 0.8158
  against A0 = 0.4714, so a bad reference is known to be able to hurt.
- **the interaction** — `ISO` vs `LONG` vs `LAT` at matched radial error, because the oracle
  effect is a pure interaction (−31.4 % / −5.8 % / +88.0 %).

**Failing values:** the curve is vacuous if `recovery` is flat in σ. It cannot be — `recovery(0)
= 1` by construction and the pure-noise control (§6) must land ≤ 0. Monotone decrease is a
**prediction**, and a non-monotone curve is a bug signal, not a finding.

---

## 5. S3 — THE LEARNED HEAD, PLACED ON THE CURVE (priority 3)

`R_head` = the best OOF head's predicted goal, fed through **exactly** the §4.1 pipeline.

Reported:
1. `e_head` (metres, OOF) — its position on the §4 curve.
2. `realised(R_head)` and `recovery` — **measured directly, not read off the curve.**
3. **The gap:** `e_head − σ₅₀` and `e_head − σ₀`, in metres, with intervals. ⚠️ This is the
   whole question: **requirement from S2 vs achievement from S3.**
4. `R_head_insample` — the bound, so "more data would fix it" is answered in advance.
5. **Curve-vs-actual consistency:** if the head's measured recovery differs materially from the
   `ISO` curve at its own radial error, the difference is the **bias structure** of a real
   regressor and is reported as a finding, not smoothed away.

---

## 6. Controls — required in BOTH directions

| control | must return |
|---|---|
| **Fidelity A** | `δ = 0` reproduces `0.2009` **exactly** (bit-level) |
| **Fidelity B** | `pick_nearest_to(gt_full)` reproduces `oracle_in_fan = 0.1640` exactly |
| **Fidelity C** | metric identity: the proximity metric and the scoring metric are the **same** mean-over-waypoint L2 *(this control caught a real 0.3655 m bug in the parent stream — `PROXIMITY-METRIC ≠ SCORING-METRIC`)* |
| **NC1 pure noise** | a goal drawn from the marginal endpoint distribution, independent of the window ⇒ `recovery ≤ 0` |
| **NC2 shuffled goal** | another window's true endpoint ⇒ `recovery ≤ 0` |
| **NC3 shuffled features** | head features permuted ⇒ `e_head → H0_const` |
| **Monotonicity** | `recovery(σ)` non-increasing in σ |

**Assume the scoring code has a bug until these say otherwise.** Three streams in two days
shipped stable, plausible, wrong numbers from exactly this class.

---

## 7. The pre-registered verdicts

| verdict | condition | what it licenses |
|---|---|---|
| **CONFIRM** | S1 = T-PASS **and** S3's `R_head` recovers a **separated** share of the −0.2705 out-of-fold | the hierarchy has a **measured mechanism**; **v5 carries a goal input**. State the recovered fraction. |
| **PARTIAL** | the S2 requirement is real (σ₀ finite, curve steep) but `e_head > σ₀` | report the **gap in metres** — it becomes the spec for whatever must supply the goal |
| **REFUTE** | S1 = T-REFUTE, **or** a realistic goal error erases the benefit (`e_head ≥ σ₀`, and `R_head` not separated-better) | **the 88 % is an oracle artifact. Say so plainly. Do not re-scope.** |

## 8. What this stream will NOT do

- It will **not** re-probe PhysicalAI-AV for a route/map/goal signal. Settled at five
  independent probes; the card says verbatim *"we do not include open maps data"*, and
  `egomotion` carries no lat/lon so map-matching is impossible. §S4 costs the **alternatives**.
- It will **not** retrain the selector. `pick_nearest_to` is the only realised-pick family
  evaluable on a counterfactual reference without a GPU-week; this limitation is stated in
  advance, as it was in the parent stream, not discovered late.
- It will **not** claim anything past **2 s**, and every number is an **ADE** number — blind to
  collision and TTC.
- It will **not** touch pod1 (training) or pod2 (arm panel).

## 9. Priority order (a killed agent must still yield value)

**S0 gate → S1 tautology → S2 curve → S3 head → S4 sources.** Each result is written into
`GOAL_INPUT.md` as it lands; nothing is held for a final synthesis.
