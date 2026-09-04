# D-REFAV1-COST-SCALE — the refav1 planner cost, in units; and what a re-weight can and cannot buy

**Arch+Inference FlyWheel · 2026-09-04 · checkpoint `refav1-b1-v72-ep3-speed` step 21,109 (strict
load, `missing_keys: []`) · NO RETRAINING, NO WEIGHT CHANGE — every number here is an inference-time
re-evaluation.**

**Two sample sizes, stated per table and never mixed.** The COST ANATOMY (§1–§5, §8) is the **full
282 windows / 141 episode clusters** of the banked `refav1-21109-openloop` grid. The **iCEM SWEEP**
(§6) re-runs the planner itself, at 55–80 s per plan, on a **stratified 24 windows / 24 episode
clusters** (every 12th window of that same grid), and `S0_shipped` is **bit-identical to the banked
`cl` path on 24/24, max \|diff\| = 0 m**. The FORM comparison (§4b–§4d) is **40 windows / 20 clips**
on the dev-box RTX 4060, same checkpoint.

⛔ Everything here is **OPEN LOOP** (PI ruling 2026-09-02). The planner feeds its own predictor.

---

## 0. Headline

| | |
|---|---|
| ⭐ **THE DEFECT IS A THEOREM ABOUT THE COST, NOT A PROPERTY OF THE WEIGHTS OR THE CHECKPOINT.** | To beat `cv` a candidate must satisfy `penalty < goal(cv) − goal(n) ≤ goal(cv)`, because the goal term is non-negative. MEASURED: `goal32(cv)` median **1.19e-07**, max over all 282 windows **5.58e-04**; the SMALLEST `jerk_raw` in a 300-sample iCEM population is **4.711 (m/s³)²**, costing `0.02 × 4.711 = 0.094`. ⇒ **100.00 % of the iteration-0 population is excluded on the median window AND on the most favourable window in the entire eval set, before the world model is consulted at all.** The only zero-penalty candidates in existence are `cv` / `hold_v0` / `decel_1.5` — and `colored_noise` subtracts the time-mean (`refa_v1_plan.py:157`) so a *sustained* acceleration is unreachable by the proposal (max \|mean_t a\| = **1.8e-07**). **⇒ output = {a=0, κ=0} ∪ {a=−1.5, κ=0}. That is exactly the 2 distinct plans observed on 282/282.** |
| ⛔ **CORRECTION 1 — the binding penalty is JERK, not κ².** | On the population actually searched, median penalty **0.362**, of which **`W_JERK·jerk²` is 99.5 %**. The prior "99.5 % is the κ² penalty" is true of a box of *constant-control* candidates — every one of which has zero jerk by construction. **Confirmed by the iCEM sweep (§6): `W_KAPPA = 0` with `W_JERK` shipped is still flat; `W_JERK = 0` with `W_KAPPA` shipped is not.** |
| ⛔ **CORRECTION 2 — `1.63e-10` does not hold at step 21,109.** | That is `D-REFAV1-COST-SURFACE`'s `incumbent` number; its own companion checkpoint read **5.82e-08**, a 357× spread inside that study's own table. At step 21,109 the goal term's range along the curvature axis is **median 1.99e-06** (mean 1.26e-05, p90 3.16e-05, max 1.26e-04) — **1.2 × 10⁴ × larger**. |
| ⛔ **CORRECTION 3 — float32 is not what annihilates the curvature signal.** | The κ sub-box spans **33.3 ulp**; on **0.00 %** of windows does it fit inside one ulp. What float32 costs is *resolution*: the argmin over the 10 curvature candidates is **TIED on 45.4 %** of windows (float64: 0.0 %) and disagrees with float64 on **21.6 %**. |
| ⭐⭐ **CORRECTION 4 — the world-model term is SIGNAL, correctly signed and goal-conditioned.** | Paired, magnitude-matched contrasts, episode-cluster intervals: **TURN_L lift +0.50221 [+0.38348, +0.57927] SEPARATED**, **TURN_R lift −0.38725 [−0.49764, −0.26686] SEPARATED**, against a **LANE_KEEP control reading −0.01254 [−0.02581, +0.00079] — straddling zero.** Longitudinally ACCELERATE **+0.577**, BRAKE_TO **−0.826**, CRUISE **−0.0035** (control ≈ 0). |
| ⛔ **AND THE 12 NON-FLAT PLANS ARE NOT SKILL.** | The 12 `decel_1.5` plans landed on **0 of the 33 `BRAKE_TO` windows** and on 12/249 (4.8 %) of the windows whose goal did *not* ask to brake. One-sided hypergeometric **p = 1**. The one axis on which the planner is not flat is **anti-correlated with the goal**. |
| ⭐ **VERDICT: RE-FORM, not re-balance** (§6, §7). ⛔ **NO setting in the 7-point weight sweep makes the planner plan** — zeroing `W_KAPPA` leaves κ ≡ 0 on 87.5 %, the proportionate re-balance leaves it on 100.0 %, and the settings that DO turn are SEPARATED WORSE than the constant-velocity floor on ADE, both lateral components and the tactical manoeuvre class. | The gap is created by the **squaring** in `1 − cos` acting on an **uncentred** field. Under the centred direction form the SHIPPED weights are already in the right regime: a-priori iCEM exclusion falls **100 % → 1.67 %** and the L/R decision goes from **1.4 % → 41 %** of the term's own magnitude. Scale-normalisation alone does **not** work, and that is proved, not asserted (§4c). |

---

## 1. What the cost actually is (source)

`stack/tanitad/refs/refa_v1.py:2165-2181`. The eval's call (`taniteval/tools/refav1_arm.py:735`)
passes no `target_speed`, so **`W_VEND` is never charged**:

```
c  =  (1 − cos(z_K , g))                       #  DIMENSIONLESS, bounded in [0, 2]
   +  W_JERK  · mean_t( ((a_{t+1} − a_t)/dt)² ) #  (m/s³)²,  W_JERK  = 0.02
   +  W_KAPPA · mean_t( κ_t² )                  #  (1/m)²,   W_KAPPA = 0.05
```

`z_K` is the **terminal** field only (`rollout(..., last_only=True)`) — the goal term is **not** a
mean over a large K. `g` is the tactical brain's own imagined goal rolled through the same
predictor under the decoded manoeuvre's canonical controls.

**Three quantities in three different units are added with no normalisation.** Their balance is set
entirely by two empirical constants, and the empirically required values are 10²–10⁴ from shipped.

### 1a. ⭐ THE UNITS ANSWER IN ONE SENTENCE

> **The goal term is a [0, 2]-BOUNDED quantity that only ever realises ~1.2 × 10⁻⁵ of its range
> (median 2.455e-05 of 2), while the penalties are UNBOUNDED quantities that routinely realise 0.36
> and can reach 32.**

Put both penalties on the same [0, 1] scale using the normalisers that already exist in
`PlanConfig` — `κ_max = 0.2` and `jerk_max = 2·a_max/dt = 40 m/s³`:

| penalty | shipped weight | value at FULL deflection | as a multiple of the goal term's ENTIRE range (2) | normalised weight `w = W · (max)²` |
|---|---|---|---|---|
| `W_JERK · jerk²` | 0.02 | `0.02 × 40² =` **32** | **× 16** | **32** |
| `W_KAPPA · κ²` | 0.05 | `0.05 × 0.2² =` **0.002** | × 0.001 | **0.002** |

⇒ **the shipped cost weights jerk 16,000× more heavily than curvature once both are on the same
scale**, and the jerk penalty alone can reach **16× the goal term's entire range**. The κ penalty is
by contrast small against that range — it binds only because the term realises 10⁻⁵ of the range
rather than O(1). **Two different errors, in opposite directions, on the same line of code.**

---

## 2. Why the model term is 10⁻⁵ and the penalties are 10⁻¹

The brief's four hypotheses, answered: it is **a cosine already in [−1, 1]**; it is **not**
multiplied by anything tiny; it is **not** a mean over a large K; it is **not** an unnormalised
chord in metres; and it is **not** a vanishing signal (§3). Two compounding, measured reasons.

### 2a. The compared fields are UNCENTRED and nearly parallel

MEASURED over 282 windows × 26 candidates (`raw/cost_anatomy_full.json`):

| quantity | value |
|---|---|
| `D` = `tac_queries × d_state` | 64 × 1024 = **65,536** |
| ‖z_K‖ | median **315.1** [167.1, 491.9] · ‖g‖ median **316.1** |
| candidate deviation ‖x_i − x̄‖ / ‖x̄‖ | median **7.19e-03** |
| cos(x_i , x_cv) across the box | median **0.999999** |
| token-mean (DC) fraction of ‖z_K‖ | median **0.999999** |

⇒ **99.3 % of the terminal field is candidate-independent common mode.**

⚠️ Incidental but should not sit unrecorded: the last row says the 64 tactical query tokens are
**essentially identical to one another** — the field is close to rank-1 along the query axis. Not
the cause of this defect; a separate finding.

**And the action-response is strongly ANISOTROPIC** (`raw/cost_forms_devbox.json`, ‖x_i − x_cv‖ as a
fraction of ‖x_cv‖):

| candidate | field response |
|---|---|
| `acc+4.0` | **7.07 %** |
| `decel_1.5` | 2.64 % |
| `acc+0.5` | 0.90 % |
| the imagined goal's own controls (`seed0`, where non-zero) | 0.99 % |
| `kap+0.2` | 0.156 % |
| **`kap+0.1`** | **0.078 %** |

⇒ the predictor's **lateral** response is ~**90×** weaker than its longitudinal response at
comparable aggressiveness. Real, but the metric is what turns "weak" into "invisible".

### 2b. `1 − cos` SQUARES that deviation

`1 − cos(x, y) = ½‖x̂ − ŷ‖²` exactly — MEASURED `max|(1−cos) − chord²/2| = 4e-16` over all 7,332
(window, candidate) cells.

```
chord  = ‖x̂ − ŷ‖              median 7.007e-03
1−cos  = chord² / 2            median 2.455e-05      ( = (7.007e-03)²/2 ✅ )
```

⇒ **the goal term's magnitude is (relative field deviation)² / 2.** The squaring is where three
orders of magnitude are lost, and it is the single most repairable thing in the expression.

### 2c. What the penalties charge — on the population that is actually searched

⛔ The measurement the earlier work did not make, and it inverts its ordering. **No GPU** —
`colored_noise` and `_clip` are pure torch. iCEM iteration 0, `n_samples=300`, `beta=2.5`,
`init_var=1.0`, `mean` init zeros (`refa_v1_plan.py:211`):

| term | median | p10 | p90 | max |
|---|---|---|---|---|
| `jerk_raw` (m/s³)² | **18.01** | 11.03 | 34.24 | 67.12 |
| `kappa_raw` (1/m)² | 0.03649 | 0.03289 | 0.04 | 0.04 |
| `W_JERK · jerk_raw` | **0.3601** | 0.2207 | 0.6848 | 1.342 |
| `W_KAPPA · kappa_raw` | 0.001825 | 0.001645 | 0.002 | 0.002 |
| **total penalty** | **0.362** | 0.2225 | 0.6868 | 1.344 |
| the goal term over the whole box | **2.455e-05** | — | p99 4.27e-03 | 0.3268 |

> **A typical iCEM candidate pays 1.474 × 10⁴ × its entire goal term in penalty** (51.7× under the
> chord). **99.5 % of that penalty is JERK.** `cv` / `hold_v0` / `decel_1.5` pay **exactly 0**
> (verified at `v0 ∈ {0, 1.91, 10.28}`).

### 2d. The closed-form exclusion bound

Because the goal term is non-negative, `penalty(n) < goal(cv)` is **necessary** for any candidate to
beat `cv`. That removes the model from the question entirely:

| setting | metric | W_JERK | W_KAPPA | iteration-0 population excluded @ median window | @ the most favourable window in the eval set |
|---|---|---|---|---|---|
| **S0 shipped** | cos | 0.02 | 0.05 | **100.00 %** | **100.00 %** |
| S3 | cos | 0 | 0 | 0.00 % | 0.00 % |
| S4 | cos | 0 | 0.05 | 100.00 % | 100.00 % *(κ-only ⇒ escapable by variance shrinkage)* |
| S5 | cos | 0.02 | 0 | **100.00 %** | **100.00 %** |
| S6 | chord | 0.02 | 0.05 | **100.00 %** | **100.00 %** |
| S7 | chord | 2e-5 | 5e-5 | 50.00 % | 0.00 % |
| S8 | cos | 1e-4 | 2e-4 | **100.00 %** | 99.67 % |

> At the shipped setting a candidate needs `jerk_raw < 5.96e-06 (m/s³)²`; the smallest in a
> 300-sample population is **4.711**, i.e. **7.9 × 10⁵ ×** too large.

⚠️ **This is an ITERATION-0 bound.** iCEM shrinks the variance, so a penalty on **one** channel can
be escaped later by driving that channel toward zero while the other stays free — which is why S4
(κ-only) escapes and S0/S5/S6/S8 (jerk alive) do not. A penalty on the **accel** channel is
inescapable, because a constant (zero-jerk) acceleration is unreachable by a zero-time-mean
proposal. The sweep (§6) is the empirical authority; the bound predicted every one of its outcomes.

### 2e. The float32 quantum — what it does and does not do

Near `cos = 1` the float32 step is `2⁻²⁴ = 5.9604645e-08`.

| MEASURED | |
|---|---|
| shipped `goal32` values that are exact integer multiples of the ulp | **100.0000 %** |
| cells where `1−cos` evaluates to exactly 0.0 | **10.31 %** |
| cells where it is **negative** (`cos > 1`, impossible in exact arithmetic) | **0.68 %** |
| the banked eval's winning plan costs | 113/282 exactly **0**, 26 at **−2 ulp**, 13 at **+1 ulp**, 9 at **+2 ulp** (118 distinct values) |
| κ sub-box spread, float64 | median **1.986e-06 = 33.3 ulp** |
| windows where the whole κ sub-box fits inside ONE ulp | **0.00 %** |
| argmin over the 10 curvature candidates **TIED** in float32 | **45.4 %** (float64: 0.0 %) |
| argmin agrees float32 vs float64 | **78.4 %** |

⇒ a **resolution** defect, not an annihilation — and it is the mechanism behind §5.

---

## 3. Tipping weights — is a re-weight a WEIGHT or a DELETION?

⚠️ **A selection warning first.** "The best of 10 curvature candidates beats `cv` on 67.7 % of
windows" is a **best-of-10 selection on the data it scores**; under exchangeability the minimum of
10 beats a single reference **10/11 = 90.9 %** of the time, so 67.7 % is **below chance** and says
the goal term *prefers straight*. It is inadmissible as evidence and is not used below.

The admissible version fixes the candidate **before** looking at the cost: **sign** from the decoded
manoeuvre, **magnitude** fixed a priori at |κ| = 0.1 (nearest box point to `GOAL_KAPPA_TURN = 0.08`).
`LANE_KEEP` is the control.

| metric | stratum | n (win/clust) | correct turn cheaper than `cv` | tipping `W_KAPPA` (median) | shipped ÷ tipping |
|---|---|---|---|---|---|
| `1−cos` f64 | TURN_L | 7 / 4 | **7/7** | 4.63e-06 | 1.08 × 10⁴ |
| `1−cos` f64 | TURN_R | 31 / 20 | **23/31** | 3.21e-06 | 1.56 × 10⁴ |
| `1−cos` f64 | **pooled manoeuvre** | **38 / 24** | **30/38 = 78.9 %** | **3.40e-06** | **1.47 × 10⁴** |
| `1−cos` f64 | **LANE_KEEP (CONTROL)** | 244 / 127 | **0 / 244** ✅ | — | — |
| `chord` | TURN_L | 7 / 4 | 7/7 | **0.01051** | **4.76** |
| `chord` | TURN_R | 31 / 20 | 23/31 | **0.008784** | **5.69** |
| `chord` | LANE_KEEP (CONTROL) | 244 / 127 | 46 / 244 | 9.66e-05 | 518 |

**The deletion test** (rule pre-registered by `D-REFAV1-COST-REPAIR`, applied verbatim: `w` is a
deletion iff `w·κ_max² < q(metric)`, `q(cos) = 2⁻²⁴`; threshold `w < 1.49e-06`):

> The pooled tipping weight **3.40e-06** is **2.28× ABOVE** the threshold and **0 of 30** per-window
> tipping weights are deletions. ⇒ **At step 21,109, M5 ("a deletion, not a weight") no longer
> holds.** But the headroom is **2.3×** with **45 % of windows already tied** — a weight in name and
> a coin-flip in practice.

**`W_JERK` is 203× too large** by the same construction: break-even for the planner's own
imagined-goal seed is **9.85e-05** against the shipped **0.02** (the seed carries non-zero jerk on
39.7 % of windows and non-zero curvature on 13.5 %).

---

## 4. The controls, and which FORM of the goal term is commensurable

### 4a. The model term IS goal-conditioned (the control that makes §3 admissible)

Within-window contrast between two candidates of **equal |κ| and opposite sign**, so the field's
common mode cancels. Estimator **episode-cluster bootstrap** (`taniteval/ci.py`, `n_boot = 4000`).
⛔ `overlapping_holdout_se` is not used anywhere in this document.

**LATERAL** (|κ| = 0.1, chord; > 0 = the model prefers LEFT):

| stratum | n (win/clust) | mean | 95 % interval | lift vs the LANE_KEEP control |
|---|---|---|---|---|
| **LANE_KEEP (CONTROL)** | 244 / 127 | **−0.01254** | [−0.02581, **+0.00079**] | — *(straddles 0 ✅)* |
| TURN_L | 7 / 4 | +0.48913 | [+0.37718, +0.55836] | **+0.50221 [+0.38348, +0.57927] SEPARATED** |
| TURN_R | 31 / 20 | −0.39803 | [−0.50848, −0.27649] | **−0.38725 [−0.49764, −0.26686] SEPARATED** |

**LONGITUDINAL** (|a| = 0.5, chord; > 0 = prefers ACCELERATE):

| decoded `lon` | n (win/clust) | mean | 95 % interval |
|---|---|---|---|
| `ACCELERATE` | 55 / 45 | **+0.57695** | [+0.57586, +0.57782] |
| `CRUISE` *(natural control)* | 116 / 73 | **−0.00354** | [−0.00440, −0.00260] |
| `ADAPT_SPEED_FOR_CURVE` | 78 / 55 | −0.24778 | [−0.32956, −0.17084] |
| `BRAKE_TO` | 33 / 25 | **−0.82648** | [−0.87245, −0.76013] |

**Every sign correct; every control reads its known value.**

⚠️ **Stated honestly:** the coarse *sign-of-the-box-gradient* version reads 81.6 % agreement — which
is **exactly** what an "always RIGHT" constant predictor scores here (31 of 38 manoeuvre windows are
TURN_R). That headline is uninformative and is **not** the evidence above.

### 4b. Eight forms of the goal term, on the SAME rollouts

`raw/cost_forms_devbox.json` — n = 40 windows, 20-episode slice, dev-box RTX 4060, **same step
21,109 checkpoint** (byte size 2,122,997,633 identical to Thor's; labels md5
`aa12c948f062181c3297265b51526ec5` identical to the banked run).

| form | definition | tipping `W_KAPPA` | shipped ÷ tip | **iCEM excluded @ median** | **L/R decision as % of the term** |
|---|---|---|---|---|---|
| **`cos` (SHIPPED)** | `1 − cos(x, y)` | 3.93e-04 | 127 | **100.00 %** | **1.369 %** |
| `chord` | `‖x̂ − ŷ‖` | 0.0726 | 0.69 | **100.00 %** | 0.685 % |
| `rel_l2` | `‖x − y‖ / ‖y‖` | 0.0736 | 0.68 | **100.00 %** | 0.811 % |
| `abs_l2` | `‖x − y‖` | 24.2 | 0.0021 | **100.00 %** | 0.811 % |
| `boxnorm` | `‖x − y‖ / median_i‖x_i − x_cv‖` | 47.4 | 0.0011 | **100.00 %** | 0.811 % |
| `goalnorm` | `‖x − y‖ / (‖y − x_cv‖ + 1e-3‖x_cv‖)` | 10.3 | 0.0049 | **100.00 %** | 0.811 % |
| **`ccos`** | `1 − cos(x − x_cv, y − x_cv)` | 20.7 | 0.0024 | **1.67 %** | **41.29 %** |
| **`cchord`** | `‖(x−x_cv)^ − (y−x_cv)^‖` | 74.1 | 0.00068 | **1.67 %** | **20.87 %** |

`x_cv` is the terminal field under **zero** controls — already rolled as the `cv` baseline, so
centring costs **no extra rollout and no privileged information**.

⚠️ **Read the two right-hand columns as different questions, and read the tipping column with care.**
*"iCEM excluded"* is **commensurability** — the ABSOLUTE size of the decision against the ABSOLUTE
penalties, and it is the column that decides whether a form can work at all. *"L/R decision as % of
the term"* is **conditioning** — whether the decision is a tiny perturbation of a large constant,
which is what sets the float precision needed and how robust the ranking is. They rank the forms the
same way here, but they are not the same statement: `cos` has a *larger* relative share than `chord`
purely because `1−cos = chord²/2` doubles any relative change, while its absolute decision is
**185× smaller**. ⛔ And the `tipping W_KAPPA` column here is a **best-of-2-sign statistic on a mixed
40-window stratum**, computed only to RANK the forms against each other — the admissible,
sign-fixed, controlled weight statement is §3, and no weight should be set from this column.

### 4c. Why scale-normalisation alone does NOT work — proved, not asserted

`‖(x − x_cv) − (g − x_cv)‖ ≡ ‖x − g‖`. **Centring changes an L2 DISTANCE by exactly nothing.** That
is why `abs_l2`, `rel_l2`, `boxnorm` and `goalnorm` all read the **identical 0.811 %** L/R decision
share — they differ only by a per-window constant. The common mode lives in the **numerator**, and
only a **direction** comparison in centred space removes it (`ccos` 41.3 %, `cchord` 20.9 %).

⇒ **"Normalise the model term" is not enough. It has to be centred, and the comparison has to be of
DIRECTIONS of action-induced change.**

### 4d. The price of the centred forms — stated before recommending them

| MEASURED (n = 40) | |
|---|---|
| ‖x_i − x_cv‖ / ‖x_cv‖ over the box | median **0.150 %** |
| ‖g − x_cv‖ / ‖x_cv‖ when the goal's canonical controls are **zero** (25/40) | **4.18e-08** — i.e. float32 rounding only, which is **correct**: the goal *is* to hold |
| ‖g − x_cv‖ / ‖x_cv‖ when the goal asks for something (15/40) | **0.58 %** |
| `ccos(cv)` | **exactly 1.0 on 40/40** — the do-nothing candidate scores the worst possible value **by definition** (its centred vector is the zero vector), not by measurement |

⇒ a centred form **must** carry an explicit HOLD branch (gate on ‖g − x_cv‖ and fall back to a
distance when the goal is "hold"), or it trades one degeneracy for another on the majority stratum
(LANE_KEEP is 244/282 = 86.5 % of the grid). **This is a pre-registration item, not something to
ship from this document.**

---

## 5. `baseline_won_frac` — re-derived from CONTENT, and the mechanism

Independently re-derived from the banked decision sidecars (282 windows / 141 clusters).

| | |
|---|---|
| κ identically 0 | **282 / 282 = 100.00 %** |
| acceleration constant in time | **282 / 282 = 100.00 %** |
| DISTINCT plans (bit-exact over the `[10, 2]` block) | **2** — `a = 0` (270), `a = −1.5` (12) |
| plan is bit-exactly an injected baseline, **BY CONTENT** | **282 / 282 = 1.0000** |
| `baseline_won_frac` **as reported (by LABEL)** | **0.6631** |
| label histogram | `cem` 95 (33.69 %), `baseline:hold_v0` 180, `baseline:decel_1.5` 7 |
| cross-tab of the 95 `cem`-labelled plans | **90 bit-exactly zeros, 5 bit-exactly the decel block, 0 anything else** |

**The mechanism, now MEASURED not inferred.** The baselines compete *inside* the CEM loop
(`refa_v1_plan.py:243-244`), so `best_cem` is frequently a baseline row; the floor loop relabels on
`c <= best_cost` (`:277-281`). But `base_costs` comes from a **second, separate** `cost_fn(base_stack)`
call (`:264`) whose batch composition differs, and the goal term is a float32 reduction over 65,536
dimensions — so the *identical* control block scores differently between the two evaluations. The
sweep banks both: for `cem`-labelled plans whose controls **are** a baseline, `res.cost −
min(baseline_costs)` is **−1.192e-07 = exactly −2.00 float32 ulp** (S0, S5, S8) and −0.02 ulp under
`chord`. Two ulp is enough to fail `<=`; and the second repair (`refa_v1.py:2203`) is gated on
`res.source.startswith("baseline:")`, so it never fires either.

⇒ **Both repairs are gated on the plan's LABEL; neither looks at its CONTENT. `plan_source` is not
evidence about what produced a plan.**

---

## 6. The sweep — does any setting make the planner actually PLAN?

**ANSWER: NO — not one of the seven.** `raw/cost_sweep_main.json`, the SHIPPED iCEM planner re-run at
inference over **n = 24 windows / 24 episode clusters** (every 12th of the same 282-window eval grid),
2.56 h on Thor, no retraining and no weight file touched — `W_JERK` / `W_KAPPA` are module globals
read by `_cost_chunk` at plan time and `cost_metric` is already a `plan()` argument.

> ⭐ **REPRODUCTION CONTROL — `S0_shipped` is bit-identical to the BANKED `cl` path on 24/24 windows,
> max \|diff\| = 0 m.** The harness reproduces the record before it varies it; without this every row
> below would be inadmissible.

### 6a. The shape probes — the ONLY probes that detect planning

| setting | metric | `W_JERK` | `W_KAPPA` | **κ ≡ 0** | **distinct plans** | `== cv` | `== decel` | non-trivial | trivial by **CONTENT** | `baseline_won_frac` (**label**) |
|---|---|---|---|---|---|---|---|---|---|---|
| **S0 shipped** | cos | 0.02 | 0.05 | **100.0 %** | **2** | 23 | 1 | 0 | **1.0000** | 0.5000 |
| S5 κ-weight **OFF** | cos | 0.02 | **0** | **87.5 %** | 4 | 20 | 1 | 3 | 0.8750 | 0.4583 |
| **S8 proportionate re-balance** | cos | 1e-4 | 2e-4 | **100.0 %** | 4 | 19 | 1 | 4 | 0.8333 | 0.4167 |
| S6 chord @ shipped weights | chord | 0.02 | 0.05 | **100.0 %** | 3 | 19 | 1 | 4 | 0.8333 | 0.1667 |
| S4 jerk-weight **OFF** | cos | **0** | 0.05 | 70.8 % | 10 | 10 | 1 | 13 | 0.4583 | 0.2500 |
| S7 chord + weights ÷1000 | chord | 2e-5 | 5e-5 | 29.2 % | 19 | 6 | 1 | 17 | 0.2917 | 0.0417 |
| S3 **both weights zero** (extreme control) | cos | **0** | **0** | 12.5 % | 23 | 2 | 1 | 21 | 0.1250 | 0.0833 |

**Three things this table says.**

1. ⛔ **`W_KAPPA` IS NOT THE LEVER — and the extreme control proves it.** Setting the curvature weight
   to **exactly zero** (S5) leaves κ ≡ 0 on **87.5 %** of windows and yields **4** distinct plans. The
   **proportionate re-balance** (S8 — *both* weights at their measured break-even, a 200× and 250×
   reduction) leaves κ ≡ 0 on **100.0 %**. ⇒ **a re-weight, however generous, does not make the
   planner turn.**
2. ⭐ **`W_JERK` is the binding term.** The only settings that turn are the ones that remove or nearly
   remove it: S4 (`W_JERK = 0`) → 70.8 %, S7 (÷1000) → 29.2 %, S3 (both zero) → 12.5 %. **Exactly what
   the closed-form iteration-0 bound (§2d) predicted, setting by setting, before the sweep ran.**
3. ⛔ **`baseline_won_frac` is a false-positive generator at EVERY setting.** S0 reads **0.5000** by
   label against a CONTENT truth of **1.0000**, and **12 of 12** `cem`-labelled plans are bit-exactly
   an injected baseline. Never trust that field.

⚠️ **And note S6 and S8: κ ≡ 0 on 100 % while `trivial by CONTENT` is only 0.8333.** They emit
constant-acceleration plans that are neither `cv` nor `decel_1.5` — so "non-trivial" ticks up while
the planner still **never turns on any window**. **A setting that leaves κ ≡ 0 has changed nothing,
whatever the other columns say.**

### 6b. The mislabel mechanism, MEASURED at every setting

For every window labelled `cem` whose controls **are** a baseline, `res.cost < min(baseline_costs)` on
**100 %** of them, by a margin of:

| setting | n | median `cost − min(baseline_costs)` |
|---|---|---|
| S0 / S3 / S4 / S8 (`cos`) | 12 / 1 / 5 / 10 | **−5.96e-08 = −1.00 float32 ulp** |
| S5 (`cos`) | 10 | −8.94e-08 = **−1.50 ulp** |
| S6 / S7 (`chord`) | 16 / 6 | −3.53e-09 = −0.06 ulp |

⇒ the two evaluations of the *same* control block differ by one to two ulp, which is enough to fail
`c <= best_cost` (`refa_v1_plan.py:279`). Under `chord` the margin drops to 0.06 ulp — the better-conditioned
metric shrinks the mislabel too, though it does not remove it.

### 6c. The four families — for the settings that DO plan

Absolute values, episode-cluster bootstrap, **n = 24 windows / 24 clusters**; `ha0` (constant
velocity) is the binding floor, `ha` (held action) and `ol` (recorded actions replayed) beside it.

| metric | **ha0** | **ha** | ol | cl (banked) | S3 both-0 | S4 jerk-0 | S5 κ-0 | S6 chord | S7 chord÷1000 | S8 rebal |
|---|---|---|---|---|---|---|---|---|---|---|
| **ADE** `ade_m` | **0.4435** | 0.6817 | 0.4753 | 0.4844 | **2.9692** | 0.5087 | 0.4999 | 0.4966 | **1.9476** | 0.5065 |
| **ADE** `fde_m` | **1.1012** | 1.8927 | 1.2030 | 1.2128 | 6.5210 | 1.2704 | 1.2515 | 1.2585 | 4.2694 | 1.2755 |
| **LON** speed MAE m/s | 0.3946 | 0.1959 | 0.0815 | 0.4508 | 0.4713 | 0.4562 | 0.4508 | 0.4679 | 0.4700 | 0.4793 |
| **LON** along MAE m | 0.3237 | 0.4085 | 0.2379 | 0.3649 | 1.3643 | 0.3673 | 0.3607 | 0.3739 | 0.7486 | 0.3857 |
| **LON** accel MAE m/s² | 0.3795 | 0.2440 | 0.0671 | 0.4420 | 0.6005 | 0.4992 | 0.4420 | 0.4751 | 0.6230 | 0.4789 |
| **LAT** cross MAE m | 0.2247 | 0.4759 | 0.3475 | 0.2247 | 2.4799 | 0.2341 | 0.2527 | 0.2247 | 1.6383 | 0.2247 |
| **LAT** heading MAE ° | 1.9939 | 13.1262 | 11.2154 | 1.9939 | 17.8307 | 2.0201 | 2.6629 | 1.9939 | 11.2954 | 1.9939 |
| **LAT** yaw-rate MAE rad/s | 0.0305 | 0.1099 | 0.0618 | 0.0305 | 0.6093 | 0.0392 | 0.0431 | 0.0305 | 0.3837 | 0.0305 |
| **TAC** traj lat correct | 0.8333 | 0.7083 | 0.7917 | 0.8333 | 0.3750 | 0.8333 | 0.7917 | 0.8333 | 0.4583 | 0.8333 |
| **TAC** traj lon correct | 0.7500 | 0.8750 | 1.0000 | 0.7083 | 0.7083 | 0.7083 | 0.7083 | 0.7083 | 0.7083 | 0.6667 |

⚠️ **STRATEGIC family: REFUSED, with the reason and the n.** The strategic decision/route half is not
computable on this record — the eval grid is planned with `nav_cmd` supplied and no route label
travels with the window, so there is no strategic target to score against. It is a **work item**, not
a pass. *(The TACTICAL family IS present, in both halves: the trajectory-derived manoeuvre class
above, and the decoded-goal histogram in §8.)*

**Paired deltas vs the `ha0` floor** (paired episode-cluster bootstrap, same windows, n = 24 / 24):

| arm − `ha0` | ADE `ade_m` | LAT cross MAE | LAT yaw-rate MAE | TAC lat correct |
|---|---|---|---|---|
| cl (banked) | +0.0409 [+0.0000, +0.1226] straddles | **+0.0000 zero-width** | **+0.0000 zero-width** | **+0.0000 zero-width** |
| **S3 both-0** | **+2.5257 [+1.3876, +3.9010] SEP** | **+2.2551 [+1.2408, +3.4170] SEP** | **+0.5788 [+0.3733, +0.8316] SEP** | **−0.4583 [−0.6667, −0.2500] SEP** |
| **S7 chord÷1000** | **+1.5041 [+0.5964, +2.6443] SEP** | **+1.4135 [+0.5229, +2.4965] SEP** | **+0.3532 [+0.2190, +0.4939] SEP** | **−0.3750 [−0.5833, −0.1250] SEP** |
| **S4 jerk-0** | +0.0652 [−0.0228, +0.1718] straddles | +0.0093 [−0.0249, +0.0466] straddles | **+0.0087 [+0.0002, +0.0191] SEP (WORSE)** | **+0.0000 zero-width** |
| S5 κ-0 | +0.0564 [−0.0000, +0.1537] straddles | +0.0280 [+0.0000, +0.0840] straddles | +0.0126 [+0.0000, +0.0372] straddles | −0.0417 [−0.1250, +0.0000] straddles |
| S6 chord | +0.0531 [−0.0065, +0.1481] straddles | **+0.0000 zero-width** | **+0.0000 zero-width** | **+0.0000 zero-width** |
| S8 rebal | +0.0630 [−0.0044, +0.1580] straddles | **+0.0000 zero-width** | **+0.0000 zero-width** | **+0.0000 zero-width** |

⭐ **THE VERDICT OF THE SWEEP, IN ONE LINE.** Every setting that makes the planner turn (S3, S7) is
**SEPARATED WORSE than the constant-velocity floor on ADE, on both lateral components, AND on the
tactical manoeuvre class** — it turns, and it turns the wrong way. The one setting that turns without
blowing up (S4) buys **nothing**: ADE and cross-track straddle zero, yaw-rate is **separated worse**,
and the tactical lateral class is unchanged at **exactly 0.0000**. And every setting that stays at the
floor (S5, S6, S8) has κ ≡ 0 on 87.5–100 % — it never turns at all.

⛔ **This is precisely why ADE was not allowed to be the probe.** S4's ADE delta straddles zero, which
a careless reader would call "no harm, and now it plans". It does **not** plan: its lateral
manoeuvre-class accuracy is identical to a straight line's, to the last bit.

---

## 7. Verdict — re-balance or re-form?

### ⭐ RE-FORM. A re-balance is arithmetically possible and structurally fragile, and it is not where the defect is.

**7.1 — A re-weight does NOT make the planner plan, and the sweep settles it.** Setting `W_KAPPA` to
**exactly zero** leaves κ ≡ 0 on **87.5 %** of windows; the **proportionate re-balance** — *both*
weights at their measured break-even, a 200× and 250× reduction — leaves κ ≡ 0 on **100.0 %**, with
every lateral and tactical delta vs `ha0` **exactly 0.0000 with a zero-width interval**. ⇒ **the
earlier reading that `W_KAPPA` is the lever is wrong**; a repair aimed at it is aimed at 0.5 % of the
penalty. The closed-form bound (§2d) predicted this setting by setting before the sweep ran.

**7.2 — And where a re-weight DOES make it move, it moves it the wrong way.** The settings that turn
(S3, S7) are **SEPARATED WORSE than the constant-velocity floor on ADE, on cross-track, on yaw-rate
AND on the tactical manoeuvre class** (S3 − ha0: ADE **+2.5257 [+1.3876, +3.9010]**, TAC lat correct
**−0.4583 [−0.6667, −0.2500]**). The one setting that turns without blowing up (S4) buys nothing: ADE
and cross-track straddle zero, yaw-rate is separated **worse**, and the tactical lateral class is
unchanged at **exactly 0.0000**.

**7.3 — A joint re-balance is a weight, not a deletion — but with 2.3× of headroom.** At step 21,109
the tipping `W_KAPPA` is **3.40e-06**, which is **2.28× ABOVE** the pre-registered deletion threshold
(0 of 30 per-window tipping weights are deletions). So `M5` ("a deletion, not a weight") **no longer
holds at this checkpoint**. But 2.3× of headroom while **45.4 %** of windows already have a tied
float32 argmin means the re-balanced cost would decide by rounding on nearly half the grid — and the
sweep shows what that buys in practice: **nothing.**

**7.4 — The gap is created by the FORM, and the form fix needs no new weights.** `1 − cos` squares a
relative field deviation of 7.0e-03 into 2.5e-05, and the field it is squaring is **99.3 % common
mode**. Centring on the zero-action terminal field — which the planner **already rolls** as the `cv`
baseline, so it costs no extra rollout and uses no privileged information — takes the a-priori iCEM
exclusion from **100 % → 1.67 %** and the L/R decision share from **1.4 % → 41.3 %**, **at the
SHIPPED weights**, where `W_JERK = 0.02` sits at **0.36× its cap** and `W_KAPPA = 0.05` is if
anything too small.

**7.5 — And "normalise the model term" is NOT enough — proved, not argued.**
`‖(x − x_cv) − (g − x_cv)‖ ≡ ‖x − g‖`, so centring changes an L2 **distance** by exactly nothing.
Measured: `abs_l2`, `rel_l2`, `boxnorm` and `goalnorm` all read the **identical 0.811 %** decision
share and all leave **100.00 %** of the population excluded. **The common mode lives in the
numerator; only a DIRECTION comparison in centred space removes it.**

**7.6 — The penalties should also be made RELATIVE, and that is a separate, independent fix.** They
charge `(m/s³)²` and `(1/m)²` against a dimensionless [0, 2] term. Dividing by `jerk_max² = 1600` and
`κ_max² = 0.04` makes both dimensionless in [0, 1] and turns the weights into trade-off ratios that
can be reasoned about instead of tuned. In those units the shipped pair is **`w_jerk = 32`,
`w_kappa = 0.002`** — a 16,000:1 imbalance between two comfort penalties (§1a).

### ⛔ WHAT MUST NOT BE DONE

1. **Do not ship any of this from this document.** The centred form has a MEASURED degeneracy on the
   HOLD stratum — `ccos(cv) = 1.0` exactly on 40/40 by definition, and `‖g − x_cv‖/‖x_cv‖ = 4.18e-08`
   (float32 noise) whenever the goal is "hold", which is **86.5 %** of the grid. It needs an explicit
   HOLD branch or it trades one degeneracy for another. **`TanitAD_ValidateAIDesign` ladder, with a
   pre-registered spec, a deliberate-regression arm (the uncentred form), and the controls that
   already exist here: LANE_KEEP must read the no-information value, and a constant-only candidate
   must read a known value.**
2. ⛔ **Do not read an ADE improvement as a working planner.** MEASURED in the sweep: removing both
   penalties (S3) makes the planner emit a different plan on every window — and its ADE is **worse
   than the `ha0` floor**. Removing the penalties makes the planner *move*; it does not make it
   *drive*. The admissible evidence that a setting plans is the pair (κ ≡ 0 fraction, distinct plans)
   **plus** the goal-conditioning controls of §4a — never ADE, and never `plan_source`.
3. ⛔ **Do not quote `1.63e-10` for this checkpoint**, and do not quote any cost-surface statistic
   without naming the **checkpoint** it was measured on and the **population** it was computed over
   (§9, and the proposed retraction in `PROPOSED_REGISTER_ROWS.md` §4).

### THE CHEAPEST DISCRIMINATING NEXT EXPERIMENT

**Zero GPU-days, one flag.** `cost_metric` is already a `plan()` argument and `COST_METRICS` is
already a validated tuple, so a `"ccos"` entry is a ~6-line addition to `_goal_term` plus its HOLD
branch. The whole panel is then the harness already banked here
(`tools/cost_sweep.py` + `tools/analyse_sweep.py`), re-run over the same 282 windows: **~40 s/plan ×
282 × (form × weights) on Thor, no retraining, and the shape probes decide it in one table.**

---

## 8. The honest scope of the defect

⚠️ "the plan is a trivial baseline on 282/282" is TRUE and is **not** the same as "the plan is WRONG
on 282/282". On a `LANE_KEEP × CRUISE` goal the canonical controls **are** zero, so the imagined goal
field **is** the zero-action rollout and `cv` is the correct answer. Sizing a repair needs the other
number.

| the decoded goal's own canonical controls | n / 282 |
|---|---|
| exactly the ZERO control (⇒ `cv` is CORRECT) | **133 = 47.2 %** |
| want non-zero **acceleration** | **120 = 42.6 %** |
| want non-zero **curvature** | **38 = 13.5 %** |
| **non-trivial in either channel** | **149 = 52.8 %** |

⇒ **the planner emitted a trivial plan on 149/149 of the windows where its own goal asked for
something.** On the 38 curvature-wanting windows it emitted κ ≡ 0 on **38/38**.

**And the 12 non-flat plans are anti-correlated with the goal:**

| goal (lat × lon) | n | plan = zeros | plan = decel_1.5 |
|---|---|---|---|
| LANE_KEEP × CRUISE | 116 | 116 | 0 |
| LANE_KEEP × ACCELERATE | 55 | 55 | 0 |
| LANE_KEEP × ADAPT_SPEED_FOR_CURVE | 40 | 30 | **10** |
| **LANE_KEEP × BRAKE_TO** | **33** | **33** | **0** |
| TURN_R × ADAPT_SPEED_FOR_CURVE | 31 | 29 | **2** |
| TURN_L × ADAPT_SPEED_FOR_CURVE | 7 | 7 | 0 |

> **0 of the 33 `BRAKE_TO` windows received a brake**; all 12 decels landed on
> `ADAPT_SPEED_FOR_CURVE`. One-sided hypergeometric **p = 1**. ⇒ the one axis on which the planner is
> not flat is not skill.

---

## 9. Provenance

| what | where |
|---|---|
| model | `tanitad-thor-wifi:/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt`, step **21,109**, 2,122,997,633 B, mtime 2026-09-04 02:55:03 (**unchanged since before the banked eval ran**; `summary.json` `"done": true, "final_step": 21109`), strict load `missing_keys: []` / `unexpected_keys: []`, `trainable_parameters: 0` — identical to the banked manifest |
| corpus | v7.2 EVAL split, 141 clips, `--window-stride 40` → the **same 282 windows** as `refav1-21109-openloop`; window origins verified equal on **282/282 by CONTENT** against `full_dump/*.npz` `ws`, and `v0` max\|diff\| = 0 |
| plan config | shipped: `n_samples 300, n_iters 30, n_elites 30, init_var 1.0, horizon 10, dt 0.2, a_max 4.0, kappa_max 0.2, beta 2.5, decay 1.25, elite_memory 0.3, seed 0, inject_baselines True` |
| call conventions | `cost_time_grid="dense"`, `goal_time_grid="full"`, `model_action_units="kappa"`, `target_speed=None` — the shipped defaults, i.e. the conventions every banked refav1 number was produced under |
| estimator | episode-cluster / paired episode-cluster bootstrap (`taniteval/ci.py`), `n_boot = 4000`, `dp=8`. ⛔ `overlapping_holdout_se` nowhere |
| compute | Thor (`nvidia@thor6`) and the dev-box RTX 4060, `OMP_NUM_THREADS=6`. **The A40 was not touched** (`refcv4-b1-v72-40k` training). Anatomy probe 266 s / 282 windows; forms probe 32 s / 40 windows; sweep as recorded in its JSON. ⚠️ A sibling agent's `refav1_arm.py` ran concurrently on Thor for part of the sweep, which roughly doubled its wall-clock (80 s/plan vs the banked 36 s). **Timings only — the plan is deterministic (`seed=0`, `no_grad`, fixed batch shapes).** |
