# GOAL INPUT — is the 88 % a capability, or an oracle artifact?

**Stream:** `2026-07-27-goal-input` · **Host:** dev box, CPU only · **Date:** 2026-07-27
**Pre-registration:** `PRE_REGISTRATION.md`, written **before any number below was computed.**
**Estimator:** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**,
unit = **episode** (40 episodes / 881 windows). `overlapping_holdout_se` is never called.

---

## 0. HEADLINE — **VERDICT: REFUTE**, and the redirect is measured

> ### ⛔ The 88 % is an oracle artifact. Predicting the 2 s endpoint is **not** easier than picking the trajectory — it is the *same problem*, to three decimal places.
>
> **MEASURED · DECISION-GRADE** for the tautology test; **MEASURED · CONFIRMED** (replicated ×3)
> for the requirement-vs-achievement gap.
>
> | | 2 s endpoint L2 error, metres |
> |---|---:|
> | the as-trained selector's own pick (`e_sel`) | **1.0061** [0.827, 1.193] |
> | the best out-of-fold learned goal head (`e_head`) | **1.0028** [0.846, 1.169] |
> | **paired difference** | **−0.0034 m [−0.1169, +0.1232]**, **not separated** |
>
> The head is **0.33 %** better than the thing it was supposed to inform. Separating that
> difference would take **≈ 4.99 × 10⁴ episodes**. This is not an underpowered measurement of a
> real effect; it is a **null**.
>
> ### And the operative test agrees, on the metric that decides.
>
> | | radial goal error (RMS, m) | realised `ade_0_2s` | recovery of the −0.2705 | sep |
> |---|---:|---:|---:|---|
> | **requirement** — break-even (`σ₀`, isotropic) | **0.955** | 0.4714 | 0 % | — |
> | **requirement** — break-even, *biased regressor* (`SHRINK`) | **0.721** | 0.4714 | 0 % | — |
> | **achievement** — best OOF head | **1.330** | **0.4996** | **−10.4 %** | ❌ |
> | achievement — fan-independent latent-only head | 1.463 | 0.5178 | **−17.1 %** | ✅ **worse** |
>
> ⇒ **A realisable goal is past break-even. It does not recover a share of the −0.2705; on the
> deployed 256-anchor fan it makes the pick worse**, separated for the head a strategic brain
> would actually have. **Both pre-registered REFUTE clauses fire.**

### ⭐ The redirect, and it is the most useful thing in this stream

**The 88 % is carried almost entirely by ONE coordinate, and it is not the one a map supplies.**
With the *learned* value on the other axis held fixed (i.e. a realistic supplier, not an oracle):

| what is oracle | what is learned | realised | **recovery** | sep |
|---|---|---:|---:|---|
| **along-track** (how far) | cross-track | **0.2450** | **+83.7 %** | ✅ |
| **cross-track** (which way) | along-track | 0.4635 | **+2.9 %** | ❌ |

> **A route / map / lane-graph signal supplies lateral topology. On the 2 s selection surface this
> program has been optimising, the lateral axis is worth ≈ 3 %, not 88 %.** The binding unknown is
> **how far the car will travel in the next 2 seconds** — a *longitudinal intent* quantity.
> **The spec, in units a supplier can be built against: the 2-s-mean speed must be predicted to
> ≈ 0.22 m/s for half the prize and ≈ 0.41 m/s to break even. The best head achieves 0.58 m/s.**

| pre-registered outcome | returned |
|---|---|
| **CONFIRM** (T-PASS **and** a separated share recovered OOF) | ❌ — T-PASS did not fire; recovery is **negative** |
| **PARTIAL** (requirement real, features cannot meet it) | ⚠️ **the requirement IS real and is published below** — but the head is not merely short of it, it is **past break-even**, which is the REFUTE clause |
| **REFUTE** (head no better than the selector, **or** realistic goal error erases the benefit) | ✅ **this one, on BOTH clauses** |

---

## 1. S0 — GATE. Every committed number reproduced from raw before a new one was quoted

`raw/gi_gate.json`, `_all_ok = true`.

| quantity | committed | reproduced |
|---|---:|---:|
| REF-C-XL as-trained `ade_0_2s` | 0.4714 | **0.4714** |
| REF-C-XL `oracle_in_fan` | 0.1640 | **0.1640** |
| `R_goal2s` realised | 0.2009 | **0.2009** |
| `R_goal2s` − as-trained (paired) | −0.2705 | **−0.2705** |
| goal − no-goal (paired) | −0.6149 | **−0.6149** |
| REF-C-base / small `oracle_in_fan` | 0.1914 / 0.2213 | **0.1914 / 0.2213** |

**Fidelity A** (δ = 0 reproduces 0.2009) ✅ · **Fidelity B** (`pick_nearest_to(GT)` == `oracle_in_fan`,
`max_abs_diff = 0.0` **exactly, per window**) ✅ · **Fidelity C** (the proximity metric is the *same*
mean-over-waypoint L2 as the scoring metric — matching in flat 8-dim L2 instead returns
**0.1658 vs 0.1640**, a **+0.0018 m** penalty) ✅.

⚠️ **On Fidelity C's power, honestly:** on *this* configuration the metric-mismatch penalty is
small (+0.0018 m), far below the **0.3655 m** it cost the parent stream. **Fidelity C alone would
not have caught that bug here** — the load-bearing control is **Fidelity B**, which is a *per-window
exact identity* (`max_abs_diff = 0.0`) rather than a mean-vs-mean comparison, and cannot be passed
by a wrong `argmin`. Both are run; B is the one to trust.

**Window identity, established not assumed.** v4's `eh2_cache` (the latent the goal head reads) and
the REF-C fan are the same 881 windows: `max |v0_eh2 − v0_fan| = 0.0` exactly, n = 881 both.
And `lat`/`lon`/`dist` are **goal-independent** — the `produced` and `neutral` variants are
bit-identical (`max_abs = 0.0`), confirming they are pure projections of `state_last`.

### 1.1 ⛔ A leakage trap, found in source before the feature set was fixed

`head_deg` in every fan dump is **`|net heading change|` over the FUTURE `K_MAX` steps**
(`stack/scripts/driving_diagnostic.py:139-142`). It reads like an ego-state scalar and is an
**oracle**. Along with `a_gt`, `v_target`/`vt_valid`/`vt_lookahead` (85th percentile of *future*
speed, 10–20 s ahead) and `route`/`route_graded` (≤25 s forward), it is declared inadmissible in
`PRE_REGISTRATION.md §1.1` and enforced at runtime by `gi_common.assert_no_oracle`.
**Any goal head that reads `head_deg` is measuring an oracle and will produce a beautiful,
stable, wrong number.**

---

## 2. S1 — ⛔ THE TAUTOLOGY TEST. Axis: 2 s endpoint L2, metres

`raw/gi_tautology.json`. 5 **episode-disjoint** folds; ridge `alpha` chosen by an inner
GroupKFold over **train-fold episodes only**, so no episode is ever in both fit and score.

**Admissible inputs** (observation-only, verified in source): `F_lat` (23) = `lat`⧺`lon`⧺`dist`,
the projections of v4's world-model latent `state_last = world.encode_window(frames)`; `F_ego` (9)
= `v0` ⧺ the 4 constant-velocity waypoints; `F_ans` (4) = **the selector's own answer** (its picked
endpoint + the logit-weighted fan endpoint); `F_conf` = PCA₃₂ of `refined_pre`⧺`prior` (512),
fitted **on the train fold only**. `F_ans` is included deliberately, so the tautology test cannot
be won by withholding information from the head.

| arm | `e` (m) | CI | vs `e_sel` | sep |
|---|---:|---|---:|---|
| `H0_const` (train-fold mean endpoint) | 15.4536 | [12.240, 19.114] | +14.4475 | ✅ worse |
| `H1_cv` (constant velocity) | 1.7406 | [1.278, 2.202] | +0.7344 | ✅ worse |
| **`H2_sel` — the as-trained selector, the bar** | **1.0061** | [0.827, 1.193] | — | — |
| `H_ridge_lat_ego_raw` (latent + ego only) | 1.1613 | [1.003, 1.331] | +0.1552 | ❌ |
| `H_ridge_lat_ego_resid_sel` | 1.0570 | [0.883, 1.230] | +0.0508 | ❌ |
| **`H_ridge_all_raw` — best OOF** | **1.0028** | [0.846, 1.169] | **−0.0034** | ❌ |
| `H_mlp_*` (best 1.5052) / `H_gbr_*` (best 1.1293) | ≥1.13 | — | ≥ +0.12 | ✅ worse |
| `H7_insample` (same model, fitted on the eval windows) | 0.6072 | [0.513, 0.709] | −0.3989 | ✅ |
| `NC3_shuffled_feats` | 15.4238 | [12.242, 19.088] | — | = `H0_const` ✅ |

> **`e_head / e_sel` = 0.9967.** Δ = **−0.0034 m [−0.1169, +0.1232]**, `p(Δ>0) = 0.4705`.

**Formally this is `T-UNPOWERED` by the letter of the pre-registration** (Δ < 0 but the interval
spans zero). ⚠️ **The honest reading is a NULL, and the distinction matters**: `UNPOWERED, NOT
REFUTED` is the correct label only when the point estimate is materially non-zero. Here it is
**0.33 % of `e_sel`**, and the n required to separate it is **≈ 49,910 episodes** — three orders of
magnitude past the 600-episode ceiling that exists. *(Half-widths scale as ≈ n^−½, consistent with
`MODEL_REGISTRY §1.2a`'s measured ×2.8–3.9 over a ×15 increase in n.)*

**Both readings license the same conclusion**, and the operative test in §4 removes the ambiguity
entirely: the head does not sit at a materially better place on the endpoint axis than the
selector already occupies.

### 2.1 What the numbers say about *where* the difficulty is

The 2 s displacement has **SD 18.73 m along-track and 2.01 m cross-track** (mean displacement
25.51 m). A learn-nothing head sits at 15.45 m. The selector sits at 1.006 m — **4 % of the mean
displacement**. ⇒ **The selector is already an extremely good endpoint predictor.** That is why
there is nothing for a goal head to add: *the goal decomposition asks a model to do accurately, in
one shot, the very thing the pick already does.*

### 2.2 ⚠️ What `H7_insample` does and does not bound

`H7_insample` fits **the same ridge model class on the evaluation windows themselves** and reaches
0.6072 m. It is **an optimistic bound on this model class, not on the feature set** — a
high-capacity learner would memorise to ≈ 0. Its content is: *even with the generalisation penalty
removed entirely, a linear read of these features stops at 0.61 m*, which (§4) recovers only 61 %
and requires fitting on the test set. **"More data would fix it" is answered: the gap is not
sample size.**

---

## 3. S2 — THE GOAL-ERROR CURVE. The requirement, in metres

`raw/gi_sweep.json`. 16 noise seeds per cell; the per-window quantity entering the bootstrap is the
**seed-mean** realised ADE. `recovery = (0.4714 − realised) / 0.2705`.
**σ is the per-axis SD in metres; the curve is indexed by radial RMS = the RMS of ‖δ‖.**
*(Quoting slopes and metres, never correlations — retraction class `CORRELATION-WITHOUT-SLOPE`.)*

### 3.1 `ISO` — isotropic goal error

| σ (m/axis) | radial RMS (m) | realised | **recovery** | Δ vs as-trained | sep |
|---:|---:|---:|---:|---:|---|
| 0.00 | 0.000 | 0.2009 | **+100.0 %** | −0.2705 [−0.3323, −0.2128] | ✅ better |
| 0.10 | 0.141 | 0.2098 | +96.7 % | −0.2616 | ✅ better |
| 0.20 | 0.281 | 0.2335 | +87.9 % | −0.2379 | ✅ better |
| 0.30 | 0.422 | 0.2722 | +73.7 % | −0.1993 | ✅ better |
| **0.50** | **0.703** | 0.3698 | **+37.6 %** | −0.1016 [−0.1660, −0.0379] | ✅ better |
| **0.75** | **1.054** | 0.5114 | **−14.8 %** | +0.0400 [−0.0291, +0.1071] | ❌ |
| **1.00** | **1.405** | 0.6645 | **−71.4 %** | +0.1930 [+0.1231, +0.2611] | ✅ **worse** |
| 1.50 | 2.108 | 0.9701 | −184.3 % | +0.4986 | ✅ worse |
| 2.00 | 2.810 | 1.2638 | −292.9 % | +0.7924 | ✅ worse |
| 3.00 | 4.215 | 1.8233 | −499.7 % | +1.3519 | ✅ worse |
| 5.00 | 7.025 | 2.8166 | −866.8 % | +2.3451 | ✅ worse |
| 20.0 | 28.100 | 8.8460 | −3095.5 % | +8.3746 | ✅ worse |

> **σ₅₀ = 0.606 m · σ₀ (break-even) = 0.955 m radial RMS.** Monotone ✅.

### 3.2 ⚠️ **YES, IT GOES NEGATIVE — a confidently wrong goal is worse than no goal**

Measured, not assumed. Recovery crosses zero at **0.955 m** radial RMS and is **separated-worse by
1.405 m** (Δ = +0.1930 [+0.1231, +0.2611]). At the far end the damage is **+8.37 m of `ade_0_2s`**
— a 3095 % *negative* recovery. **This is decision-relevant: a strategic brain that emits a
confident bad goal is not neutral, it is destructive**, and a supplier must be gated on accuracy
before it is wired in.

### 3.3 Per-axis, and ⭐ the family that matters — a biased regressor is **harsher**

| family | σ₅₀ (radial RMS, m) | σ₀ (radial RMS, m) | min recovery |
|---|---:|---:|---:|
| `ISO` (unbiased isotropic noise) | 0.606 | 0.955 | −3095 % |
| `LONG` (along-track only, cross exact) | **0.538** | **0.881** | −3051 % |
| `LAT` (cross-track only, along exact) | 0.749 | 1.218 | −720 % |
| ⭐ **`SHRINK`** (`ĝ = ḡ + α(g−ḡ)` — a **regression head's** error) | **0.425** | **0.721** | −2961 % |

> ⭐ **`SHRINK` is the realistic family and it is the strictest.** At matched radial error a
> *biased* goal (shrunk toward the mean, which is exactly what an L2-trained regressor produces)
> breaks even at **0.721 m** versus **0.955 m** for unbiased noise — **a 25 % tighter spec.**
> A goal head that minimises endpoint L2 is optimising the wrong loss for this consumer.
>
> Note also that **`LAT` is the gentlest family** (σ₀ = 1.218 m vs 0.881 m for `LONG`): even here,
> before any conditioning, the lateral coordinate is the more forgiving one.

### 3.4 Both-directions validation

| control | required | measured |
|---|---|---|
| Fidelity A (δ = 0) | reproduces 0.2009 | **0.2009** ✅ |
| Fidelity B (`pick_nearest_to(GT)`) | == `oracle_in_fan` | `max_abs = 0.0` ✅ |
| Fidelity C (proximity metric == scoring metric) | flat-8-L2 must be worse | +0.0664 m ✅ (control has power) |
| **NC1 pure noise** (goal from the marginal, window-independent) | recovery ≤ 0 | **−4425 %** ✅ |
| **NC2 shuffled goal** (another window's true endpoint) | recovery ≤ 0 | **−4261 %** ✅ |
| NC3 shuffled features | head → `H0_const` | 15.4238 vs 15.4536 ✅ |
| Monotonicity | non-increasing in σ | ✅ all four families |

**The instrument can return "the goal is the lever": it does, at every σ below 0.955 m, separated.
It returns the opposite for every realisable goal we can build.** The REFUTE is a measurement, not
an absence of power.

---

## 4. S3 — REQUIREMENT vs ACHIEVEMENT. This is the whole question

`raw/gi_place.json`. Every learned goal is fed through **exactly** the §3 pipeline, so the head arm
and the noise curve are the same instrument. Recovery is **measured**, never read off the curve.

| arm | radial RMS (m) | along RMS | cross RMS | realised | **recovery** | Δ vs A0 | sep |
|---|---:|---:|---:|---:|---:|---:|---|
| `R_goal2s_ORACLE` | 0.000 | 0.000 | 0.000 | 0.2009 | **+100.0 %** | −0.2705 [−0.3323, −0.2128] | ✅ better |
| **`R_head_oof` (`H_ridge_all_raw`)** | **1.330** | 1.151 | 0.666 | **0.4996** | **−10.4 %** | +0.0282 [−0.0339, +0.0992] | ❌ |
| **`R_head_oof_lat_ego`** (no selector answer) | **1.463** | 1.310 | 0.651 | 0.5178 | **−17.1 %** | +0.0464 [+0.0164, +0.0792] | ✅ **worse** |
| `R_head_insample` (bound) | 0.842 | 0.631 | 0.557 | 0.3054 | +61.4 % | −0.1661 [−0.2162, −0.1185] | ✅ better |
| `R_selend` (**circularity control**) | 1.453 | 1.305 | 0.638 | 0.4775 | −2.2 % | +0.0060 [+0.0012, +0.0128] | ✅ worse |
| `R_cvend` | 2.646 | 1.451 | 2.213 | 0.8158 | −127.3 % | +0.3443 | ✅ worse |
| `R_head_deshrunk` *(exploratory)* | 1.328 | 1.145 | 0.673 | 0.4790 | −2.8 % | +0.0075 [−0.0463, +0.0658] | ❌ |

**The gap, in metres — the number the PARTIAL branch was reserved for:**

| | value |
|---|---:|
| requirement, break-even (`ISO` σ₀) | **0.955 m** |
| requirement, break-even (`SHRINK` σ₀ — the realistic family) | **0.721 m** |
| requirement, 50 % recovery (`ISO` σ₅₀) | **0.606 m** |
| **achieved, best OOF head** | **1.330 m** |
| **gap past break-even** | **+0.375 m** (**1.39×** too inaccurate; **1.85×** vs `SHRINK`) |
| **gap to 50 % recovery** | **+0.724 m** (**2.20×** too inaccurate) |

**Two controls that make the verdict unarguable:**

1. ⭐ **`R_selend` — the pure circularity control.** Feed the selector *its own endpoint* back as a
   goal: **0.4775, separated-WORSE than doing nothing** (+0.0060 [+0.0012, +0.0128]). The
   re-quantisation step itself has a small, real cost. So a goal head must not merely match the
   selector — it must beat it by enough to pay that toll first.
2. **`R_head_oof_lat_ego` — what a strategic brain actually has.** A head that never sees the
   tactical selector's answer is **separated-WORSE** (+0.0464 [+0.0164, +0.0792]). ⛔ **The
   fan-independent, deployable form of this idea is measured to damage the pick.**

⚠️ **The best head's +0.0282 is not separated at n = 40 — and it would separate at ≈ 222
episodes.** A **600-episode clean val build already exists** (order-preserving prefix of the 40,
parity holds — `2026-07-26-0757-program-report §2.3`). **The honest statement is therefore: the
damage is probably real and is currently unresolved, and the cheapest way to resolve it is the
600-episode build, not a new experiment.** It is *not* "unpowered, therefore maybe positive" —
the point estimate is on the damaging side on the two strongest fans.

### 4.1 ⚠️ Curve-vs-actual: the RMS index is **conservative**, and that is a finding

At radial RMS 1.330 m the `ISO` curve predicts **−59.2 %**; the head measures **−10.4 %**. The head
does substantially *better* than isotropic noise of the same RMS because its error is
**heavy-tailed** (RMS 1.330 vs mean 1.003, ratio 1.33 against a Gaussian's 1.13): most windows are
accurate and a few are badly wrong, and the realised penalty is concave in the tail.

⇒ **Do not grade a future goal supplier by placing its RMS on this curve — it will look worse than
it is.** Grade it by running it through the rule, as done here. *(Recorded because the obvious next
mistake is exactly this substitution.)* **The measured number, −10.4 %, is the operative one, and
it is still negative.**

### 4.2 Replication on all three fans

`raw/gi_replicate.json`, fan-independent latent-only head (`H_ridge_lat_ego_raw`, radial RMS 1.434):

| fan | anchors | A0 | `R_goal2s` | σ₅₀ | σ₀ | head realised | **recovery** | Δ vs A0 | sep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| REF-C-**XL** | 256 | 0.4714 | 0.2009 | 0.606 | 0.955 | 0.5020 | **−11.3 %** | +0.0305 | ❌ |
| REF-C-base | 128 | 0.4728 | 0.2257 | 0.674 | 1.074 | 0.4771 | −1.7 % | +0.0043 | ❌ |
| REF-C-small | 64 | 0.5261 | 0.2479 | 0.799 | 1.291 | 0.4757 | +18.1 % | −0.0504 | ❌ |

> **On no fan is a realisable goal separated-better than no goal.** There is a clean, interpretable
> gradient: **the coarser the proposal vocabulary, the more a rough goal helps** (σ₀ relaxes
> 0.955 → 1.291 as anchors fall 256 → 64). ⇒ **The goal-input idea is a small-vocabulary
> compensator, and the deployed fan is not a small vocabulary.**

---

## 5. ⭐ THE REDIRECT — the 88 % is longitudinal, and a map does not supply it

`raw/gi_place.json`. Hold one coordinate at the **learned** (realistic) value and make the other
exact. This is the decomposition that matters, because no supplier delivers an exact axis
alongside *nothing* on the other.

| oracle axis | other axis | realised | **recovery** | Δ vs A0 | sep |
|---|---|---:|---:|---:|---|
| **along-track** | learned cross (0.666 m RMS) | **0.2450** | **+83.7 %** | −0.2264 [−0.2852, −0.1717] | ✅ **better** |
| **cross-track** | learned along (1.151 m RMS) | 0.4635 | **+2.9 %** | −0.0080 [−0.0742, +0.0656] | ❌ |

**The conditional spec — the actual engineering requirement:**

| axis, conditioned on the other being *learned* | σ₅₀ | σ₀ (break-even) | ceiling |
|---|---:|---:|---:|
| **along-track** (given learned cross-track) | **0.439 m** | **0.813 m** | +83.7 % |
| cross-track (given learned along-track) | *never reaches 50 %* | **0.210 m** | **+2.9 %** |

> ⭐ **In supplier units: the 2 s along-track displacement is `2 s × mean speed`, so**
>
> | | 2-s-mean-speed accuracy required |
> |---|---:|
> | for **50 %** of the prize | **0.219 m/s** |
> | to **break even** | **0.406 m/s** |
> | **achieved by the best head** | **0.576 m/s** |
>
> **The supplier must be 1.42× more accurate to stop doing harm, and 2.62× more accurate to be
> worth half the oracle.**

**Why this reframes the whole sources question.** A route, a map, a lane graph, a junction
annotation or a turn command tells you **which way**. This instrument says the *which way* axis is
worth **+2.9 %** once a small ridge on the latent has supplied it, and that even a **perfect**
cross-track buys only that. **The 88 % lives on the *how far* axis** — target speed, lead-vehicle
constraint, whether the car is about to brake.

⚠️ **Honesty conditions on that reading, stated with the claim:**
- This is a **2 s, open-loop, ADE** measurement on a corpus whose cross-track SD is **2.01 m**
  against an along-track SD of **18.73 m**. At 2 s, lateral choice barely differentiates
  trajectories **by construction**. It does **not** say routes are unimportant for driving; it says
  **routes are not the lever for the 2 s selection problem this program has been optimising.**
- At longer horizons, at junctions, or on a metric that can see collisions, the lateral axis may
  dominate. **This instrument cannot see any of that** and does not claim to.

**Relationship to the parent stream's decomposition.** `FAN_CONDITIONING.md §7` reports
`R_goal_LONG_only` −31.4 % and `R_goal_LAT_only` −5.8 % and concludes *"a pure interaction… a goal
must be 2-D."* That is measured against a **degenerate** other coordinate. With a *realistic* other
coordinate the interaction resolves: **the along-track axis carries +83.7 % on its own.** These are
compatible measurements of different quantities, and this one is the decision-relevant one.
⚠️ **UNVERIFIED — I could not reproduce the parent's decomposition:** `raw/fanc_goal_decomp.json`
has **no producing script staged anywhere in the repo** (probed twice: `grep --include=*.py` over
the tree and a repo-wide ripgrep for `LONG_only`/`goal_decomp` — the only two hits are the JSON and
the prose). Its manifest attributes it to `code/fanc_goal.py`, which contains only `R_cv` /
`R_goal2s` / `R_goalfull`. **Escalated in §8.**

---

## 6. S4 — where could a real goal come from, and what would each cost

⛔ **PhysicalAI-AV carries no map, lane graph, junction annotation, roundabout label, traffic-light
feature or route/goal signal** — settled at five independent probes; the card says verbatim *"we do
not include open maps data"*, and `egomotion` has no lat/lon so map-matching our traces is
impossible. **Not re-probed here, per the brief.**

**But §5 changes which of the options are even aimed at the right axis.** Costs below are
`INHERITED` from the cited sibling streams unless marked.

| option | supplies | axis it serves | cost | what it licenses |
|---|---|---|---|---|
| **AlpaSim map / route** — lane graph readable, 11,877 lanes, 12,030 successor edges, 0 dangling, `target_branch` computable, ego-to-lane match 0.9827 `INHERITED` (`2026-07-26-0757 §2.2`) | route, lane succession | ⚠️ **lateral — the +2.9 % axis** | needs **~103 scenes vs 51** ⇒ a download; AlpaSim runs ~0.8–1.0× real-time on the A40, renderer-bound `INHERITED` | a *strategic-topology* capability. **On this instrument it does not buy the 88 %.** |
| ⭐ **AlpaSim `traffic_light.parquet`** — present on **51/51** scenes `INHERITED` (same source) | signal state ahead | ✅ **longitudinal** | same download | **stop/go intent** — a first-class input to the *how far* axis. Currently framed as a map asset; it is a **speed-intent** asset. |
| ⭐ **PhysicalAI `obstacle.offline` agent tracks** — 3D tracks on **97.44 %** of the corpus, mean **53.6** tracked agents/window, only **0.03 %** of windows empty; `gap` / `ttc_min` already computed on 3,360 windows `INHERITED` (`2026-07-27-percandidate-labels`) | lead-vehicle gap, closing speed, TTC | ✅ **longitudinal — the +83.7 % axis** | **no download, no new corpus.** Needs an aligned per-window feature dump on the canonical 881 (and later 600-episode) val windows — the existing dump is on a different, alias-keyed window set | **the cheapest discriminating experiment in this stream** — see §7 |
| **An external mapped corpus** (nuScenes / Argoverse / Waymo-style, with HD maps and routes) | full route + map | ⚠️ lateral | a new ingest **and a parity break** — a different corpus cannot be compared to `e438721ae894` arms | a *different* program. **Refuse to fold into the parity line.** |
| **A predicted goal from a strategic brain** | both axes | — | **this is what §4 tested** | ⛔ **measured to be past break-even on the axis that matters.** Not licensed today. |

> ⛔ **The corpus is NOT the binding constraint on the axis that carries the value.** The parent
> stream's conclusion — *"the goal input must come from AlpaSim or an external corpus… the binding
> constraint on the whole hierarchy"* — is aimed at the **lateral** axis, which this stream
> measures at **+2.9 %**. **The longitudinal information is already in PhysicalAI-AV**, in the
> 32-of-36 features our ingest does not read.
>
> ⚠️ **HYPOTHESIS, with its test named:** that lead-vehicle gap/TTC features close the 0.576 →
> 0.406 m/s gap is **not measured**. It is the next experiment (§7), it is cheap, and it can fail.

⛔ **One thing AlpaSim cannot supply:** its agents **do not react to the ego** (effect bounded at
[−0.21, +0.14] m against a 4.5 m noise floor, null in 4/4 near-ego strata) `INHERITED`. A
longitudinal-intent model learned there would learn car-following against non-reactive traffic.
**Use AlpaSim for signals and topology, not for interaction.**

---

## 7. WHAT THIS LICENSES, AND WHAT IT DOES NOT

### 7.1 Settled — the verdict

1. ⛔ **v5 must NOT carry a learned 2-D goal input on the strength of the 88 %.** The 88 % is an
   oracle bound; every realisable form of it measured here is at or past break-even, and the
   fan-independent form is separated-**worse**.
2. **The goal decomposition is close to a tautology on this surface.** `e_head / e_sel = 0.9967`,
   n-to-separate ≈ 5 × 10⁴ episodes. **The hierarchy's cheapest justification is retired.**
3. **The requirement is now published and is the spec for any future supplier:** 2 s endpoint to
   **≤ 0.955 m** radial RMS to break even, **≤ 0.721 m** if the estimator is a biased regressor,
   **≤ 0.606 m** for half the prize — or, on the axis that matters, **2-s-mean speed to ≤ 0.406 /
   0.219 m/s**.
4. **A confidently wrong goal is destructive, not neutral** (up to +8.37 m of `ade_0_2s`). Any goal
   channel must be gated on measured accuracy before it is wired in.
5. ⭐ **The value is longitudinal, not lateral** (+83.7 % vs +2.9 %) — **and the corpus already
   carries the longitudinal signal.**

### 7.2 What I refuse to conclude

- **NOT** "the hierarchy is wrong." This stream refutes **one argument for it** — the cheapest one.
  A three-planner hierarchy may still be justified on horizon, on safety metrics, on closed-loop
  behaviour or on compute allocation. **None of those is tested here.**
- **NOT** "goal conditioning cannot work." It works, measurably, **below 0.955 m of goal error**.
  What is refuted is that *the features we have* can get there, and that a map would help.
- **NOT** that routes are unimportant. **At 2 s, ADE, open-loop, on a corpus with 2.01 m of lateral
  spread**, the lateral axis is worth 2.9 %. That is a scoped claim and it is scoped in §5.
- **NOT** anything past **2 s**, and every number is an **ADE** number — blind to collision and TTC.
  *(The per-candidate-labels stream measured that the **ADE-optimal** candidate collides **4.7×**
  more often than the rule-optimal one. A goal that improves ADE is not thereby safe.)*
- **NOT** that a *better-trained* head would clear the bar. `H7_insample` removes the entire
  generalisation penalty and still reaches only 61 % — and it requires fitting on the test set.

### 7.3 The next experiment — cheap, discriminating, and unowned

**E-GOAL-1 — does lead-vehicle context close the longitudinal gap?**
Add `gap`, `ttc_min`, closing speed and lead-agent kinematics (from `obstacle.offline`, already
proven present on 97.44 % of the corpus) to `F_lat`+`F_ego`, refit the same OOF ridge, and place
the resulting **along-track** RMS against the pre-registered **0.813 m break-even / 0.439 m
half-prize**. Pre-register both outcomes.
- **Cost:** one aligned per-window agent-track dump on the canonical 881 windows, then ~10 min of
  dev-box CPU using this stream's code unchanged.
- **It can fail**, and a failure is informative: it would say the 2 s longitudinal residual is
  *ego intent*, not *traffic constraint*, and no external signal supplies it.

**E-GOAL-2 — resolve the damage at n = 600.** The best head's +0.0282 m separates at ≈ 222
episodes. The 600-episode clean val build exists and is a parity-preserving prefix. **Cost: a 66 GB
move after pod2 finishes.** This converts "probably harmful, unresolved" into a decision-grade sign.

---

## 8. ESCALATIONS — these must not sit in a file

1. 🔴 **`Project Steering/Gates/flagship-v5-retrain.PREP.md` item 6 and `V5_PLAN.md` §8's
   *"⭐ What DOES move it: the goal input — 88.0 % of the fan's headroom"* must be amended.**
   The section already carries the oracle caveat; what it does not carry is that **the realisable
   version is measured to be at or past break-even, and separated-worse in its deployable form.**
   Both documents are being edited by sibling agents in the current index — **this needs an edit by
   the owner, not a note in an incoming folder.** *(An orthogonality instrument sat unmerged for 10
   days on exactly this failure mode.)*
2. 🔴 **The "the corpus is the blocker / we need AlpaSim or an external corpus for a goal signal"
   line — in `FAN_CONDITIONING.md §7`, `V5_PLAN.md` §8 and `CLAUDE.md`'s operating-standard rule 2
   — is aimed at the LATERAL axis, which measures +2.9 %.** The axis worth +83.7 % is longitudinal
   and its signal is **already in PhysicalAI-AV**. This changes what is on the critical path.
3. 🟠 **`raw/fanc_goal_decomp.json` has no producing script staged** (two independent probes). Its
   numbers — the widely-quoted *"−31.4 % / −5.8 % / +88.0 %, a pure interaction"* — are currently
   **unreproducible from the repo** and have already propagated into `V5_PLAN.md`. This is
   Operating-Standard rule 3 (*an artifact on one disk is not done*) applied to a producer rather
   than an output. **Ask the parent stream to stage the script.**
4. 🟠 **For `RETRACTION_LOG.md` — root-cause classes:**
   - **`ORACLE-SHAPED-AS-EGO-STATE`** — `head_deg` is `|net heading change|` over the **future**
     `K_MAX` steps and sits in every fan dump beside `v0` and `speed`, which are observations.
     A head that reads it produces a stable, plausible, wrong number. **The check is one line: read
     the producing function, not the field name.** *(Caught here before any fit; declared in
     `PRE_REGISTRATION.md §1.1`.)*
   - **`BOUND-QUOTED-AS-CAPABILITY`** — *"the goal input is worth 88 % of the headroom"* was
     correctly stamped as an oracle by its author and **still propagated into two steering
     documents as a direction.** The stamp is not enough; **an oracle result needs its realisable
     counterpart measured before it can enter a plan.**
   - **`RMS-PLACED-ON-A-NOISE-CURVE`** — grading a real estimator by putting its RMS on an
     isotropic-noise curve over-predicted the damage by **5.7×** here (−59.2 % predicted vs −10.4 %
     measured), because real error is heavy-tailed. **Run the estimator through the rule.**

---

## 9. THREATS TO VALIDITY I COULD NOT REMOVE

| threat | direction | mitigation / status |
|---|---|---|
| ⚠️ **Cross-arm latent.** The goal head reads **v4**'s latent (`eh2_cache`) while the fan is **REF-C**'s. It is the only latent staged on this host. | unknown; plausibly **against** the head | window identity is exact (`max\|Δv0\| = 0`); the head is also given `F_ans`, REF-C's own answer, so it can always copy the arm that produced the fan — and it lands at 0.9967× its error |
| The as-trained selector **cannot** be retrained on a counterfactual reference | unknown | **stated in advance** (`PRE_REGISTRATION.md §8`), not discovered late; `pick_nearest_to` is the only realised-pick family evaluable without a GPU-week |
| `H7_insample` bounds the **model class**, not the feature set | favours "more capacity would help" | stated explicitly in §2.2; a high-capacity in-sample learner would trivially reach 0 |
| Ridge `alpha` chosen by inner GroupKFold on the train fold | negligible | no scored episode ever enters a fit |
| 40 episodes | widens every interval | every non-separated headline carries its **n-to-separate**; the 600-episode build exists and is a parity-preserving prefix |
| **ADE only** — no collision, no TTC | unknown, possibly large | §7.2; the ADE-optimal candidate is measured to collide **4.7×** more than the rule-optimal one `INHERITED` |
| Every number is a **2 s** number | unknown | §7.2 |

**Evidence classes.** Everything in §§1–5 is `MEASURED (ours)` with its artifact path.
§6 rows marked `INHERITED` are sibling-stream numbers **not re-verified here**. §6's
lead-vehicle proposal and §7.3 are `HYPOTHESIS` with a named test. **Tier of the headline:
DECISION-GRADE** for the tautology null (bit-level gate, both-directions validated, three
independent controls) and **CONFIRMED** for the requirement-vs-achievement gap (replicated on 3
fans; single corpus, single latent source).

---

## 10. DELIVERABLE MANIFEST

**Every artifact is in the repo and staged. Nothing lives in only one place. Nothing was committed
or pushed.**

| artifact | where | what |
|---|---|---|
| `GOAL_INPUT.md` | `repo:…/incoming/2026-07-27-goal-input/` | this document |
| `PRE_REGISTRATION.md` | same | bars, feature whitelist, leakage audit, failing-value proofs — **written before any fit** |
| `code/gi_common.py` | same | loaders, oracle whitelist + `assert_no_oracle`, metric primitives, folds, estimators |
| `code/gi_gate.py` | same | S0 gate + fidelity A/B/C + window identity |
| `code/gi_tautology.py` | same | **S1, the decisive instrument** — the head grid, NC3, the in-sample bound |
| `code/gi_sweep.py` | same | S2 goal-error curve, 4 families, NC1/NC2 |
| `code/gi_place.py` | same | S3 requirement-vs-achievement, the axis decomposition, the conditional spec, power statements |
| `code/gi_replicate.py` | same | S3b replication on 3 REF-C fans |
| `raw/gi_gate.json` | same | gate + endpoint axis raw |
| `raw/gi_tautology.json` | same | every head arm, per-axis errors, OOF predictions |
| `raw/gi_head_preds.npz` | same | per-window OOF goal predictions, so any interval is recomputable |
| `raw/gi_sweep.json` | same | full σ grids × 4 families, both negative controls |
| `raw/gi_place.json` | same | head arms, conditional axis specs, power statements |
| `raw/gi_replicate.json` | same | 3-fan replication incl. per-fan ISO curves |

**Inputs consumed** (all pre-existing repo artifacts; **no pod contacted, no checkpoint loaded, no
episode opened, parity `e438721ae894` untouched, the dev box's non-parity cache `14231cd29c74`
never read**): `taniteval/results/fan_refc-{xl,base}-30k.pt`,
`…/2026-07-22-refc-small-30k/fan_refc-small-30k.pt`,
`…/2026-07-27-t3-and-lambda-tau/raw/eh2_cache.pt`.

🔒 No clip UUIDs or raw PhysicalAI content appear in any artifact; episodes are opaque integers
already present in committed dumps.

**Total compute: ~11 minutes of dev-box CPU. No GPU. Zero pod load** — pod1 (training), pod2 (arm
panel), pod3 and the eval pod were never contacted.

**Suite green:** `taniteval` **559 passed** in 234 s (2026-07-27, re-run at the end of this
stream). This stream added **no files** to `taniteval/` or `stack/`; all new code lives in the hub
folder above.
