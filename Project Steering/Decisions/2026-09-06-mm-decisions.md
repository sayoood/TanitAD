# TanitAD — Master Mind decisions, 2026-09-06

*(continues the M-numbering from `2026-09-05-mm-decisions.md`, which ended at M52)*

## M53. ⭐⭐⭐ A PERFECT GOAL MAKES refav1 TWICE AS BAD — the defect is the SEARCH, not the goal-setting layers

### 1. The measurement

MEASURED, zero GPU, from the banked `rec_oracle_s0.json` (`…/2026-09-05-refav1-cost-geometry/raw/arms/`),
n = 40 windows / 8 episodes. `cl_oraclegoal` is, in the record's own words, *"as `cl` with the TRUE
future field as goal_field (future information => T0; search-vs-goal attribution)"*.

| arm | tier | ADE m | curv MAE | speed MAE | turn_L | turn_R |
|---|---|---|---|---|---|---|
| `ol` — kinematic contract (recorded actions) | T0 | **0.8052** | 0.08261 | 0.0906 | 0.7273 | 1.0000 |
| `ha0_ext` — THE ECHO CONTROL | T1 | 0.8772 | 0.07730 | 0.3058 | **0.7273** | **0.7500** |
| `cl` — the planner | T1 | 1.3272 | 0.05537 | 0.7155 | 0.3636 | 0.7500 |
| **`cl_oraclegoal`** — planner + TRUE future as goal | T0 | ⛔ **2.6899** | 0.12032 | 0.8766 | 0.1818 | **0.0000** |
| `cl_oracleseed` — de-confounded oracle | T0 | ⛔ **2.5452** | 0.11867 | 0.9001 | 0.1818 | 0.0000 |

### 2. ⛔⛔ The inference, and it redirects the programme's refav1 effort

**Handing the planner a PERFECT goal makes it 2.03x worse than its own normal operation** and 3.34x
worse than the contract arm. ⭐ The de-confounded variant (`cl_oracleseed`, the head's own canonical
seed retained in the pool) reads **2.5452** — so this is **not a seeding artifact**.

⇒ **If the gap lived in the GOAL — i.e. in the tactical/strategic layers choosing bad targets — a
perfect target would close it.** It does the opposite. ⇒ ⛔ **THE DEFECT IS IN THE SEARCH AND ITS COST
COUPLING**, and **improving goal-setting cannot help until the cost geometry is fixed.**
⭐ This is independent confirmation, from an arm nobody ran for this purpose tonight, of the earlier
localisation of refav1's defect to the **cost geometry** — and it is the direct answer to the PI's
goal 5 (*"identify the gaps whether in the world model or in the planner"*): **the planner.**

⚠️ **Tier honesty:** `cl_oraclegoal`/`cl_oracleseed` are **T0** (they consume future information). This
is therefore an **ATTRIBUTION DIAGNOSTIC, NOT a capability claim** — and the cross-tier comparison is
legitimate here **only** because that arm exists for exactly this attribution. ⛔ Neither number may be
quoted as driving performance, and neither belongs in a T1 table.

### 3. ⭐ Two further facts from the same table

* **`ol` = 0.8052 m is effectively this rig's PARAMETERISATION FLOOR** — the ADE you get by integrating
  the *recorded* actions. `lonshift` reads **0.7868**, i.e. **ADE headroom is essentially exhausted.**
  ⇒ ⛔ **Further ADE optimisation on this rig is not where the remaining driving quality lives**; the
  residual is **lateral and tactical**. *(An arm slightly below `ol` is possible because ADE scores the
  integrated path against recorded POSES, and the recorded controls accumulate their own integration
  error — but it does mean ADE is now measuring the parameterisation, not the policy.)*
* ⛔ **THE ECHO CONTROL TURNS BETTER THAN THE PLANNER.** `ha0_ext` (hold the initial `(a0, kappa0)`)
  reads **0.7273 / 0.7500**; the best planner arm reads **0.3636 / 0.7500**. **The planner turns worse
  than not planning.** ⇒ the sharpest statement yet of the lateral problem, and fully consistent with
  the constraint-vs-penalty frontier (`M48`, and the `kamm07` row buying 77 % of the penalty's ADE gain
  at zero turning cost).

### 4. ⛔ My own read error, for the second time tonight — same class as M51

I reported that `oracle_s0`, `ccosh_w000` and `ccos_argmax` were **bit-identical on every column** and
flagged it as a possible defect (an `M40` shared-denominator signature). **It was my read.** The
manifests differ only by `arms`/`tiers`/`arm_meaning`: `oracle_s0` **ADDS** `cl_oraclegoal` and
`cl_oracleseed`, and its `cl` **is** `ccos_argmax`'s `cl` by construction. ⇒ the identical rows are
**correct**, and **the oracle content was never in the column I was reading.**
⚠️ `ccosh_w000` likewise differs in `cost.metric` (`ccosh` vs `ccos`) at weight zero — a deliberate
no-op control behaving exactly as designed.
⇒ ⛔ **A NAME IS NOT PROVENANCE, AND NEITHER IS A COLUMN YOU HAPPENED TO READ.** `M51` was the same
error with the GT arm (`ol`, not the run named `oracle_s0`); this is the same error with the oracle
arm. **Read `arms` / `arm_meaning` FIRST and enumerate what a record actually contains, before
comparing anything across records.**

### 5. What this changes, concretely

1. ⭐ **The refav1 plan's A3 (a goal-conditioned lateral cost) is promoted** — it is a *cost* fix, and
   the cost is now the proven defect.
2. ⛔ **Deprioritise goal-setting work on refav1** until the search can exploit a good goal. Any
   improvement to the tactical/strategic goal heads is currently unmeasurable through this planner.
3. ⭐ **A new cheap arm is implied and should be pre-registered:** *why* does a true-future goal hurt?
   The likely mechanism is that the goal enters as a **direction** cost (`ccos` = `1 - cos(z, g)`)
   whose geometry does not match the rollout, so a "correct" target is unreachable and the search
   chases it. **Measure the realised cost of the oracle goal against the shipped goal** — if the oracle
   goal scores *worse under our own cost*, the cost is refuted directly, with no GPU.

## M54. ⭐⭐⭐ THE SEARCH IS EXONERATED — THE COST IS THE DEFECT. Two independent streams, same verdict.

### 1. The measurement M53 asked for, run in the same session

MEASURED, zero GPU, `dump_oracle_s0/decisions/ep000..ep007.npz`, **8 episodes / 40 windows** (both
counts asserted against their controls):

| arm | falls back to a baseline | CEM won | `finecost_cv` (landscape control) | `plan_neval` |
|---|---|---|---|---|
| `cl` — shipped goal | **10 / 40** | 0.750 | 1.06419 | 2100 |
| **`cl_oraclegoal`** — TRUE future as goal | **0 / 40** | **1.000** | 1.05611 | 2099 |
| `cl_oracleseed` — de-confounded | **0 / 40** | **1.000** | 1.05611 | 2100 |

⇒ ⛔⛔ **Given a PERFECT goal the CEM search beats every injected baseline on EVERY window** — against
falling back on a quarter of them in normal operation — **with an essentially identical evaluation
budget and a near-identically scaled cost landscape** (the `finecost_cv` control, 1.064 vs 1.056).
**And it drives 2.03x worse** (`M53`: ADE 2.6899 vs 1.3272).

⇒ ⭐⭐ **THE SEARCH OPTIMISES BETTER AND DRIVES WORSE. That is the definition of a MISSPECIFIED
OBJECTIVE.** `M53` concluded *"the search and its cost coupling"*; this narrows it and **exonerates
half of it: the search is healthy; the COST is the defect.**

⚠️ **Caveat kept explicit rather than buried:** `finecost_plan` (0.82033 vs 0.63541) is **NOT directly
comparable across these arms** — the goal term is evaluated against different targets, so it is a
lower value of a *different* function. ⛔ **The admissible signals are the BASELINE-FALLBACK COUNT and
the matched `finecost_cv` control**, both of which are computed the same way in every arm. The
per-window `fine/cv` ratio printed in the probe is also inadmissible (mean-of-ratios with small
denominators) and is not quoted.

### 2. ⭐⭐ INDEPENDENT CONFIRMATION from a stream that was answering a different question

The turn-asymmetry stream found a **second assignment site** it had previously missed
(`refa_v1.py:2571-2599`): the decoded goal's **own canonical control is appended to `seed_pool`
unconditionally on every window**, with the source's own comment explaining why — *"without this the
planner returned `hold_v0` on 24/24 windows against a TURN goal… `colored_noise` is zero-mean, so a
sustained curvature is unreachable unless some candidate carries it."*

MEASURED (realised `|kappa|` exactly 0.080000; `GOAL_KAPPA_TURN` imported from source):

| arm | exact-rung windows | `TURN_L` | `TURN_R` |
|---|---|---|---|
| `ccos_argmax` (`W_KAPPA` 0) | **21/22** | **8/9** | **13/13** |
| `ccos_seed1` (replicate) | **21/22** | **8/9** | **13/13** |
| `wk15` (`W_KAPPA` 15.11) | 9/22 | ⛔ **0/9** | 9/13 |

⇒ ⛔ **"The search failed to FIND the left-turn candidate" is dead.** On all nine left-goal windows the
full `+0.08` candidate was **present by construction and LOST ON COST.**

⇒ ⭐⭐⭐ **Two streams, different data, different questions, one verdict: THE COST IS THE DEFECT.**
That convergence is worth more than either result alone, because neither was designed to test the
other. ⭐ It also re-reads `M48`'s ladder finding: the ladder did not help the search *find* anything —
it added **cheaper** candidates to a comparison that was already being lost, which is exactly why it
removed the RIGHT turns too.

### 3. What this changes

1. ⭐⭐ **Promote the cost work to the top of refav1.** The goal-conditioned lateral cost (plan item A3)
   is no longer one option among several — it targets the proven defect. ⛔ **Deprioritise every
   goal-setting improvement**: an arm that improves the goal is currently unmeasurable through a
   planner that drives worse when handed a perfect one.
2. ⭐ **The κ penalty is refuted as a lateral lever twice over** — by the frontier (`kamm07` buys 77 %
   of its ADE gain at zero turning cost) and now by mechanism (it makes the correct, injected candidate
   lose). **The constraint replaces it.**
3. ⚠️ **`ol` = 0.8052 m is the parameterisation floor and `lonshift` is already at 0.7868** (`M53` §3)
   ⇒ **ADE is now measuring the parameterisation, not the policy.** Judge refav1 on the LATERAL and
   TACTICAL families from here; an ADE-led arm will mislead.

### 4. ⛔ A Master-Mind scheduling ruling, made rather than deferred

The longitudinal stream escalated that **`lonshift_s1`** — the inference-seed replicate that converts
the package's largest separated win into a **quotable** one — has been starved: its gate correctly caps
the 8 GB dev-box at 2 concurrent arms (a third is a measured OOM risk that would also endanger the
sibling's arms), and the sibling refav1 stream kept both slots with back-to-back exploratory arms.

⇒ **RULING: `lonshift_s1` takes the next free dev-box slot.** ⭐ **Validating a landed result outranks
launching another exploratory arm** — our own estimator doctrine says a one-seed win is *necessary,
not sufficient*, and refav1's planner **samples**, so the inference-seed replicate is the specific
control that makes the +0.2263 [−0.3173, −0.1411] longitudinal win admissible. The gate is behaving
correctly and must NOT be loosened; this is a priority decision, not a capacity one.

## M55. ⭐ THE STRATEGIC HEAD DOES USE NAV — and refav1's eval slice cannot score whether it uses it CORRECTLY

### 1. Two probes of mine withdrawn before they became results

* ⛔ **Withdrawn:** *"world-model error vs tactical-decision correctness"*, which split 40 windows into
  **3 correct / 37 wrong** and returned p = 0.70. **The grouping variable was invalid.** `lat_label` is
  **`-100` (PyTorch `ignore_index`) on 32 of 40 windows** — I compared a decoder prediction against an
  ignore token. ⭐ **Caught by implausibility, not by the p-value:** the tactical family's own recalls
  (0.7143 / 0.3636 / 0.7500) imply ~25 correct, not 3. **A p-value computed on a broken grouping is not
  a negative result; it is nothing.**
* ⛔ **Withdrawn:** *"the route head is an exact nav echo"*, raised because `route_pred_nav_true` and
  `nav_cmd` have **identical histograms** ({0:10, 1:10, 2:20}). **Per-window they agree on only 20/40.**
  ⭐ The `M40` rule — *an implausibly exact agreement points at a shared denominator, not a shared
  effect* — fired correctly and **REFUTED** the hypothesis instead of confirming it. Identical marginals
  are not identity.

### 2. ⭐ The real result: nav-responsiveness, MEASURED

`dump_ccos_argmax`, 40 windows, control `nav_cmd` non-constant ({0:10, 1:10, 2:20}):

| probe | result |
|---|---|
| `route_pred` **changes** when nav is SHUFFLED | **23/40** |
| `route_pred` **changes** when nav is ZEROED | **30/40** |
| `route_pred == nav_cmd`: true / shuffled / zero | 20 / 13 / 10 of 40 |
| `lat_pred` changes under shuffle / zero | 15/40 · 22/40 |

⇒ ⭐ **The strategic head genuinely CONSUMES the nav command** — its output moves on 23 of 40 windows
when the command is corrupted and 30 of 40 when it is removed — **and it is NOT an echo** (20/40, not
40/40). ⛔ **This is NOT the flagship-v1 defect** (whose route head was an exact bijection of its own
nav input, 369/369 and 81/81, scoring 1.0000). **That is the first half of the PI's goal 5
(*"follow nav command"*) answered affirmatively, on evidence.**

### 3. ⛔ The gap: whether it follows nav CORRECTLY is UNSCORABLE on this slice

`route_label` is **`-100` on 32 of 40 windows**; so are `lat_label` and `lon_label`. ⇒ **the STRATEGIC
family is scorable on n <= 8 here**, and every strategic claim about refav1 currently rests on eight
windows. ⚠️ **This is also why the tactical family uses a TRAJECTORY-DERIVED labeller**
(`refc_tactical.factor_from_kinematics`) rather than these labels — a workaround whose existence should
have told us the supervised labels were mostly absent, and did not, because nobody read their values.

⇒ ⭐⭐ **refav1's eval slice has exactly the disease the refcv5 stream cured tonight** — parity label
coverage **7.917 % -> 100.000 %** (2,400/2,400) built with the trainer's own arithmetic, zero GPU,
13.5 s. **Apply the same build to refav1's eval episodes.** It is cheap, it is already implemented, and
until it lands no strategic number from this rig is admissible beyond n = 8.

### 4. Consequences

1. ⭐ **Goal 5 splits cleanly now.** *Lateral/longitudinal driving* — the defect is the **COST** (`M54`,
   two independent streams). *Nav-following* — the head **does** consume nav; its **accuracy** is
   unmeasured for want of labels, not for want of capability.
2. ⛔ **No strategic claim from refav1 may be quoted without its n**, and on this slice that n is <= 8.
3. ⭐ **A 0-GPU work item with a known recipe:** rebuild the refav1 eval slice's v7.2 labels to full
   parity coverage, then re-read the strategic family. **This is the cheapest remaining lever on goal 5.**

## M56. ⛔⛔ RETRACTION — "the first zero-violation arm in the programme" DIES AT A SECOND INFERENCE SEED

### 1. What I told the PI, and what the replicate says

I reported `combined`'s `kamm_over_rate = 0.0000` as **the programme's first zero-violation arm**,
bracketed by controls, free in ADE. **MEASURED at `--plan-seed 1`: `kamm_over` = 0.0741** (2 of 27
windows), **`peak_g` max 0.702 — OVER the mu = 0.7 circle.** ⇒ **FAIL against a criterion committed
before that arm existed.** ⛔ **The zero-violation headline is WITHDRAWN.**

⇒ `best`'s own `kamm_over 0.0000` and `peak_g` 0.082/0.332 — which the stream had reported as
*"smoother than the human"* — are **single-seed and unreplicated**, and **the one zero in this family
that WAS replicated did not hold.** The admissible form is now: *"reduces the violation rate to a
single-seed 0.0000, unreplicated."*

### 2. ⭐ The doctrine predicted this exactly, and the test was commissioned FOR it

`CLAUDE.md`'s third-variance rule: **refav1's planner SAMPLES (iCEM), so the same checkpoint evaluated
twice does not give the same answer**, and a separated interval from a one-seed arm is **necessary, not
sufficient**. I stated that risk when I reported the zero, and commissioned the inference-seed
replicate for precisely this reason. ⇒ **The process worked. The claim did not.** That is the correct
order of events, and it is the argument for budgeting a replicate into every arm from the start rather
than adding one after a headline.

⚠️ **Second instance in one turn** of the sufficiency rule catching this same package — the first was
the wrong-floor withdrawal (a paired floor applied to a per-class recall).

### 3. ⭐ What SURVIVES the refutation — scope it, do not over-retract

* ⭐ **The constraint-vs-penalty principle (`M48`) STANDS.** It rests on **turn recalls**, and those were
  **bit-identical across the seed pair**. The frontier reading — the Kamm **constraint** buys 77 % of the
  **penalty's** ADE gain at zero turning cost — is unaffected by a safety-rate that moves with the seed.
* ⭐ **The cost-is-the-defect finding (`M54`) STANDS** — it rests on baseline-fallback counts and the
  oracle-goal arms, different data entirely, and is corroborated by a second stream.
* ⭐ **`M55`'s nav-responsiveness STANDS** (23/40 shuffled, 30/40 zeroed).
* ⛔ **What falls is exactly one thing: the SAFETY claim.** No arm in this programme has a replicated
  zero friction-circle violation rate.

### 4. `best` landed — the tightest parity yet, and it is PARITY, not a win

`best` = `ccos` + `W_KAPPA 15.11245` + `--kamm-mu 0.7`, **no ladder** (removed on `M48`'s prediction):

| metric | `ccos_argmax` | `wk15` | `kamm07` | **`best`** | floor |
|---|---|---|---|---|---|
| **ADE m** | 1.3272 | 0.8934 | 0.9927 | **0.8838** | 0.0607 |
| LAT curvature MAE | 0.05537 | 0.03098 | 0.04846 | 0.03128 | 0.00066 |
| LAT heading MAE | 23.4578 | 15.2704 | 21.0683 | 15.2975 | 1.1458 |

**ADE 0.8838 is the best of all ten arms** — and against `ha0_ext`'s 0.8772 that is **+0.0066, 9.2x
BELOW the seed floor** ⇒ ⛔ **PARITY, NOT A WIN, and it is reported as such.** ⭐ But it is **parity
while ACTING**: 19 distinct realised curvatures (against `wk15_ladder`'s 2), `turn_right` recall 0.50,
and better than the floor on the turn stratum (0.9742 vs 1.1521). It takes `wk15`'s whole lateral
family at no cost and **gives back** part of the longitudinal damage (speed −0.0168, accel −0.0139,
both separated).

### 5. ⭐⭐ And the longitudinal blocker is now NAMED, and it is a SECOND, GOAL-SIDE defect

`best` still reads LON speed MAE **0.7751 against `ha0_ext`'s 0.3058**, and the cause is measured:
⛔ **the decoded LON token commands `a == 0` on 29 of 40 windows** — while the ground-truth arm (`ol`)
is at constant speed on **0 of 40** (`M51`). **Untouched by every lever in this package.**

⇒ ⭐ **This does NOT contradict `M54`; it complements it. There are TWO defects, on different paths:**
* **LATERAL** — the **COST** is misspecified (`M54`): the search optimises better with a perfect goal
  and drives worse; the correct turn candidate is injected and loses on cost.
* **LONGITUDINAL** — the **GOAL TOKEN** is wrong: the decoder emits "maintain" on 29/40 windows where
  the vehicle accelerates. ⭐ **That is why `lonshift` (`a0_shift`) produced a 29.8x-floor win** — it
  changes what that token yields, and it drove `frac a == 0` from 0.475 to **0.000, exactly onto ground
  truth.**
⇒ ⛔ **Do not apply `M54`'s "deprioritise goal-setting" to the LONGITUDINAL path.** `M54`'s scope is the
lateral/full goal field probed by `cl_oraclegoal`; the LON token is a separate, and demonstrably
tractable, target.

### 6. Consequences

1. ⛔ **No safety claim from refav1 may be quoted.** Every zero in that family is single-seed; the one
   that was replicated failed.
2. ⭐ **A seed replicate is now MANDATORY in every refav1 arm's SPEC, budgeted at launch** — not added
   after a headline. Two claims died to this rule in one turn.
3. ⭐ **`best` + a seed replicate** is the next arm, and **`kamm07 + a0_shift`** (plan item A1 — the
   constraint base plus the longitudinal fix, no penalty) remains unrun and is now the more interesting
   of the two, because it attacks both named defects at once.
