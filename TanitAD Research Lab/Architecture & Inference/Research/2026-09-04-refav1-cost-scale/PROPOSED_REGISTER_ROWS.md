# PROPOSED register rows — D-REFAV1-COST-SCALE (2026-09-04, Arch+Inference FlyWheel)

⛔ **PROPOSED, not applied.** `Project Steering/GOALS_AND_CLAIMS.md` and
`Project Steering/RETRACTION_LOG.md` are high-contention shared files; per the precedent set by
`…/2026-09-03-refav1-cost-surface/PROPOSED_REGISTER_ROWS.md`, this FlyWheel proposes the text and
the **Master Mind applies it**. Escalated in the report headline, not left in a README.

Evidence for every row: `…/2026-09-04-refav1-cost-scale/RESULT.md`, `raw/cost_anatomy_full.json`
(282 windows), `raw/cost_forms_devbox.json` (40 windows), `raw/cost_sweep_main.json` (the iCEM
sweep), tools in `tools/`.

---

## 1. NEW ROW — `D-REFAV1-COST-SCALE`

| claim | verdict | evidence |
|---|---|---|
| ⭐⭐ **THE FLAT PLAN IS A THEOREM ABOUT THE COST — provable without the world model, and therefore not fixable by training, by a checkpoint, or by a weight.** The goal term is non-negative, so a candidate beats `cv` only if `penalty(n) < goal(cv)`. MEASURED at step 21,109 over the full 141-clip grid: `goal32(cv)` median **1.19e-07**, max over all 282 windows **5.58e-04**; the smallest `jerk_raw` in a 300-sample iCEM population is **4.711 (m/s³)²**, costing `0.02·4.711 = 0.094`. ⇒ **100.00 % of the iteration-0 population is excluded on the median window AND on the most favourable window in the entire eval set, before the world model is consulted.** `cv`/`hold_v0`/`decel_1.5` are the only zero-penalty candidates, and `colored_noise` subtracts the time-mean (`refa_v1_plan.py:157`, max \|mean_t a\| = **1.8e-07**) so a *sustained* acceleration is unreachable by the proposal. ⇒ output = `{a=0,κ=0} ∪ {a=−1.5,κ=0}` — **exactly** the 2 distinct plans on 282/282. ⛔ **AND THE BINDING PENALTY IS JERK, NOT κ²**: over the searched population the median penalty is **0.362**, of which `W_JERK·jerk²` is **99.5 %**; confirmed in the sweep, where `W_KAPPA = 0` with jerk shipped is STILL flat and `W_JERK = 0` with `W_KAPPA` shipped is not. ⭐ **AND THE WORLD-MODEL TERM IS SIGNAL, NOT A DEGENERACY**: paired magnitude-matched contrasts with episode-cluster intervals give TURN_L lift **+0.50221 [+0.38348, +0.57927] SEPARATED**, TURN_R **−0.38725 [−0.49764, −0.26686] SEPARATED**, against a LANE_KEEP control at **−0.01254 [−0.02581, +0.00079]** (straddles 0); longitudinally ACCELERATE **+0.577**, BRAKE_TO **−0.826**, CRUISE **−0.0035**. **A working, correctly-signed, goal-conditioned predictor is being outvoted by two unnormalised penalties.** | **SUPPORTED (MEASURED 2026-09-04; n = 282 windows / 141 clusters, same grid and same ckpt as `refav1-21109-openloop`; Thor + dev-box RTX 4060; NO retraining, NO weight change — inference-time re-evaluation)** | `RESULT.md` §0, §2c–2d, §4a |

---

## 2. NEW ROW — `D-REFAV1-COST-FORM` (the repair direction)

| claim | verdict | evidence |
|---|---|---|
| ⭐⭐ **RE-FORM, NOT RE-BALANCE — and "normalise the model term" is NOT enough, which is PROVED rather than argued.** `‖(x − x_cv) − (g − x_cv)‖ ≡ ‖x − g‖`, so centring changes an L2 **distance** by exactly nothing; measured, `abs_l2`, `rel_l2`, `boxnorm` and `goalnorm` all read the **identical 0.811 %** L/R decision share and all leave **100.00 %** of the iCEM population a-priori excluded. Only a **DIRECTION comparison in centred space** removes the common mode: `1 − cos(z_K − z_K⁰, g − z_K⁰)` takes the a-priori exclusion **100 % → 1.67 %** and the L/R decision share **1.4 % → 41.3 %**, with `z_K⁰` = the terminal field under zero controls, which the planner **already rolls** as the `cv` baseline (no extra rollout, no privileged information). Under that form the SHIPPED `W_JERK = 0.02` is **0.36× its cap** (i.e. already fine) and `W_KAPPA = 0.05` is if anything **too small**. ⛔ **THE PRICE, STATED BEFORE THE RECOMMENDATION:** `ccos(cv) = 1.0` exactly on 40/40 — the do-nothing candidate scores the worst possible value **by definition**, and on a HOLD goal the goal's own centred vector is float32-noise (‖g − z_K⁰‖/‖z_K⁰‖ = **4.18e-08**), so the comparison degenerates on the majority stratum (LANE_KEEP = 244/282 = 86.5 %). **A centred form must carry an explicit HOLD branch or it trades one degeneracy for another.** ⇒ this is a **PRE-REGISTRATION item** (`TanitAD_ValidateAIDesign` ladder: deliberate-regression arm = the uncentred form; controls = LANE_KEEP must read the no-information value and a constant-only candidate must read a known value), **not a change to ship from this document.** | **PROPOSED — the FORM comparison is SUPPORTED (MEASURED, n = 40 windows / 20 clips, dev-box RTX 4060, same step-21,109 ckpt: byte size 2,122,997,633 identical to Thor's, labels md5 `aa12c948f062181c3297265b51526ec5` identical to the banked run); the REPAIR itself is UNVALIDATED and gated on a pre-registration** | `RESULT.md` §4b–4d, §7 |

---

## 3. AMENDMENT — `D-REFAV1-EPOCH-PLAN-VOID`

The row's **verdict stands** (the planner IS a trivial injected baseline on 282/282; I re-derived
κ ≡ 0 on 282/282, accel constant on 282/282, 2 distinct plans, 282/282 trivial by CONTENT, and the
95-window `cem`/content cross-tab, all independently). **Three items in its MECHANISM sentence need
amending:**

1. ⛔ **Replace** *"the world-model term varies by 1.63e-10 along κ"*. That number is
   `D-REFAV1-COST-SURFACE`'s **`incumbent`** measurement; the same study's companion checkpoint read
   **5.82e-08** for the same quantity. **At step 21,109 it is median 1.99e-06** (mean 1.26e-05,
   p90 3.16e-05, max 1.26e-04) — **1.2 × 10⁴ × larger**. Quoting the `incumbent` figure for this
   checkpoint is a **scope error**, and it matters: it is the difference between the pre-registered
   *"a deletion, not a weight"* verdict and a tipping `W_KAPPA` of **3.40e-06** that sits **2.28×
   ABOVE** the deletion threshold (0 of 30 per-window tipping weights are deletions).
2. ⛔ **Replace** *"against a float32 `1−cos` step of 5.96e-08"* used as if float32 destroyed the
   signal. It does not: the κ sub-box spans **33.3 ulp** and on **0.00 %** of windows fits inside one
   ulp. What float32 costs is **resolution** — the argmin over 10 curvature candidates is **TIED on
   45.4 %** of windows (float64: 0.0 %) and disagrees with float64 on **21.6 %**. That tie rate is
   the mechanism behind the row's own `baseline_won_frac` finding (below).
3. ⭐ **Add the measured mislabel mechanism**, which the row correctly flagged but left unexplained:
   for `cem`-labelled plans whose controls **are** a baseline, `res.cost − min(baseline_costs)` is
   **exactly −2.00 float32 ulp**. `base_costs` is produced by a **second, separate**
   `cost_fn(base_stack)` call (`refa_v1_plan.py:264`) whose batch composition differs from the
   in-loop evaluation, and the goal term is a float32 reduction over 65,536 dimensions — so the
   *identical* control block scores two ulp apart between the two evaluations, which is enough to
   fail the `c <= best_cost` relabel.

⭐ **And add the honest SCOPE**, which the row does not carry and which a repair must be sized
against: the decoded goal's own canonical controls are **exactly zero on 133/282 = 47.2 %** of
windows (where `cv` is the CORRECT plan), want non-zero acceleration on **120/282 = 42.6 %**, and
non-zero curvature on **38/282 = 13.5 %**. ⇒ **the planner emitted a trivial plan on 149/149 of the
windows where its own goal asked for something**, and κ ≡ 0 on **38/38** of the curvature-wanting
ones. ⛔ **The 12 non-flat plans are anti-correlated with the goal**: they landed on **0 of the 33
`BRAKE_TO` windows** and on 12/249 (4.8 %) of the windows whose goal did not ask to brake
(one-sided hypergeometric **p = 1**).

---

## 4. PROPOSED `RETRACTION_LOG.md` entry

| field | text |
|---|---|
| **date** | 2026-09-04 |
| **retracted** | *"the world model contributes 1.63e-10 along the curvature axis"* quoted for the step-21,109 checkpoint (`MODEL_REGISTRY.md:2141`, `GOALS_AND_CLAIMS.md` `D-REFAV1-EPOCH-PLAN-VOID`, `PREREG_TACTICAL_DECODER.md:190,276`), and the reading that float32 *annihilates* the curvature signal. |
| **corrected to** | On step 21,109 the goal term's curvature-axis range is **median 1.99e-06** over 282 windows — 1.2 × 10⁴ × the quoted figure. The κ sub-box spans **33.3 ulp**; float32 costs **resolution** (45.4 % tied argmins), not the signal. |
| **root-cause CLASS** | ⭐ **A TRUE MEASUREMENT QUOTED OUTSIDE ITS SCOPE** — the `df` / Thor `free` / cgroup `usage_in_bytes` / `step_s` / epcache-artifact family, with the object being a **checkpoint**. The source study measured the quantity on TWO checkpoints and published **both** (1.63e-10 and 5.82e-08, a 357× spread **inside its own table**); the downstream quote took one of them and dropped the qualifier. ⇒ **A quantity whose own source shows a 357× spread across checkpoints may never be quoted for a THIRD checkpoint without re-measuring it.** The registry rule "state the fit window / the estimator" needs its sibling here: **state the CHECKPOINT, and re-measure before reusing.** |
| **second, subtler half** | *"99.5 % of the cost variation is the κ² penalty"* is TRUE **of a candidate box of constant-control candidates — every one of which has zero jerk by construction** — and **INVERTED** on the population iCEM actually searches, where **99.5 % of the penalty is JERK**. Same class again: a correct measurement whose **population** was not carried with it. ⇒ **A cost-surface statistic must name the population it was computed over, and a claim about what the SEARCH does must be computed over the SEARCH's own population.** |
| **cost** | The scope error made the defect look unfixable by any weight (*"a deletion, not a weight"*), and the population error pointed the repair at `W_KAPPA` when the binding term is `W_JERK`. Both were caught only by re-measuring on the live checkpoint and on the real proposal distribution. |
