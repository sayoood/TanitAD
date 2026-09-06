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

## M57. ⭐⭐ THE GOAL HEAD IS ITSELF LEFT-BLIND — and the eval slice cannot resolve it. A CORPUS request, not a compute request.

### 1. The head, decoded on CPU before any planner output existed

The decoded goal token is an **input** to the planner (`lat_head(intent)`, identical across all seven
banked arms **and across plan seeds**), so it can be read without running anything. 75 wide-panel
windows, 6 episodes:

| goal head ALONE (no planner, no cost) | value |
|---|---|
| goal recall **LEFT** | **0.3000** [0.0000, 0.6333] (n = 30) |
| goal recall **RIGHT** | **0.6667** [0.3125, 0.9375] (n = 30) |
| pooled R − L | +0.3667 [−0.3433, +0.8525] — **not separated** |
| **wrong-direction** rate | GT-left **9/30** vs GT-right **2/30** = **4.50x** |

⇒ ⛔ **Since at `W_KAPPA = 0` the plan IS the goal's canonical seed on 21/22 turn-goal windows
(`M54` §2), THE PLAN'S TURN RECALL IS BOUNDED ABOVE BY THE HEAD'S.** A plan-recall gap is therefore
**partly INHERITED**, and *"the penalty costs left turns first"* cannot be read off the plan's gap
alone. ⭐ **Only the INCREMENT over the head is attributable to the cost** — and the stream registered
that rule **before** any `cl` output existed.

⚠️ **This qualifies `M54`, and I state the qualification rather than let it stand unmarked.**
⭐ **What SURVIVES:** `M54`'s core contrast is **penalty-ON vs penalty-OFF against the SAME head**
(`ccos_argmax` 8/9 left vs `wk15` 0/9 left). The head's asymmetry is **common to both arms**, so it
cannot explain that delta — the cost still destroys what the head does get right. ⛔ **What must be
withdrawn is any claim that the LEFT-TURN DEFICIT IS ENTIRELY THE COST'S.** It is at least two
defects stacked: a left-blind head, and a cost that discards the correct candidate the head supplies.

### 2. ⛔⛔ The cluster ceiling — and it is the CORPUS, not the panel size

| episode | goal recall R | L | R − L |
|---|---|---|---|
| 0 | 0.8333 | 0.0000 | +0.8333 |
| 2 | 1.0000 | 0.0000 | +1.0000 |
| **6** | 0.0000 | 1.0000 | ⛔ **−1.0000** |
| 7 | 1.0000 | 0.0000 | +1.0000 |
| mean | — | — | **+0.4583 [−0.5000, +1.0000]** — not separated |

⇒ ⛔ **On this eval slice, left/right turn behaviour varies MORE BETWEEN EPISODES than between
DIRECTIONS.** Three episodes say right, one says left, and a bootstrap over four clusters cannot
choose. ⭐ Raising n per direction from 11/8 to **30/30 did NOT raise the cluster count** — and the
decision estimator (episode-cluster bootstrap) consumes **clusters**, not windows.

⇒ ⭐⭐ **SETTLING THE TURN ASYMMETRY NEEDS MORE EPISODES, NOT MORE COMPUTE. This is a PI-level corpus
request** — and it was registered **before** the result, so it cannot read as an excuse.
⚠️ **It does not license reading a null as a result:** outcome C means *"this eval slice cannot resolve
an effect of this size"*, **never** *"there is no asymmetry."*

### 3. A structural unattributability, and an omission the stream registered against itself

Decoded `TURN_L` occurs only in episodes {1, 6} and `TURN_R` only in {0, 2, 4, 7} — ⛔ **no episode
carries both.** Not a sampling artefact: **the head only ever decodes `TURN_L` in the left-heavy
episodes.** ⇒ the *retention* statistic is **unattributable on the wide panel too** and decides
nothing. The *recall* statistic is unaffected (GT-defined strata; 4 episodes carry both).
⭐ The stream also registered, unprompted, that its within-episode contrast has **4 clusters, below the
>=5 it had set for the pooled strata and never set for this one** — so a failure to separate there is a
**power limit, not a negative.**

### 4. Consequence

⭐ **Goal 5's lateral half now has three named, separable targets**, in order of measured size:
1. **the COST** — discards the correct injected candidate (`M54`);
2. **the GOAL HEAD** — left recall 0.3000 vs right 0.6667, wrong-direction 4.50x;
3. **the EVAL SLICE** — cannot resolve direction effects at all; **more episodes required.**
⛔ **No lateral-asymmetry claim may be quoted from this slice until (3) is fixed**, and any that is
must report the head's recall **beside** the plan's on the same windows.

## M58. ⭐⭐⭐ THE 2x2x2 FACTORIAL SETTLES IT — `W_KAPPA` IS THE TURN-KILLER, AND THE CAP IS INERT BEHIND IT. Carry the constraint, drop the penalty.

### 1. The factorial splits perfectly — no interpretation required

| condition | `turn_left` recall |
|---|---|
| **`W_KAPPA` = 0** | **0.3636 in ALL FOUR cells** |
| **`W_KAPPA` on** | ⛔ **0.0000 in ALL FOUR cells** |

⇒ ⛔ **`W_KAPPA` is the turn-killer, factorially, with the cap and the ladder varied across it.** No
other lever moves the class. This is the strongest form the constraint-vs-penalty finding (`M48`) could
take, and it arrived from a designed factorial rather than a frontier read.

⭐ **And the cap is INERT BEHIND the penalty:** `best − wk15` ADE **−0.0096 [−0.0415, +0.0278]** (ns),
TAC lateral **+0.0000 [0, 0]**; `bestlad − wk15_ladder` +0.0010. **Because `W_KAPPA` has already driven
`max|kappa|` far inside the friction circle, there is nothing left for the cap to forbid.**

⇒ ⭐⭐ **RULING: carry the CAP forward, drop `W_KAPPA`.** The cap buys **67 % of the ADE**, more of the
safety, **at zero turn cost and no longitudinal cost**. The penalty buys the remaining third by
deleting a manoeuvre class.

### 2. ⭐⭐ The mechanism, stated exactly: the decision is CORRECT, the execution is SUPPRESSED

MEASURED: under `W_KAPPA` the **decoded goal mix is IDENTICAL** (18 / 9 / 13) while **`TURN_L` mean
`kappa^2` falls 98.0 %.** ⇒ ⛔ **`W_KAPPA` does not change the decision — it refuses to EXECUTE it.**

⇒ ⭐ That is the same object `M54` identified from the cost side (the correct candidate is injected and
**loses on cost**) and `M57` bounded from the head side (the head is left-blind upstream). **Three
streams, three routes, one mechanism: the tactical decision survives and the cost discards it.**

### 3. ⛔ A superlative of the predecessor's, refuted — and the root cause is new

*"the only zero-violation arm"* was **FALSE: four arms read 0.0000**, and `wk15` reached it **by not
turning.** ⛔ **Root cause: a number quoted from a STALE GENERATION of a REGENERATED artifact**
(`feas_audit.txt` superseded by `feas_audit_all.txt`) — and the refuting file was written **19 minutes
BEFORE** the report that contradicted it. ⇒ ⭐ **CLASS: *a control set is not a census.* Checking three
arms and finding one zero does not establish uniqueness; enumerate the population.** Adjacent to the
stale-read family but distinct: here both files were readable and correct, and the wrong one was read.

### 4. ⭐ A gate design worth reusing: the VACUITY gate

`bestlad`'s pre-committed gates read **S PASS · M FAIL · T FAIL · A FAIL**, and the **M** gate is the
instructive one: `max|kappa|` **0.0049 < 0.02 ⇒ the zero-violation result is VACUOUS.** ⇒ ⭐⭐ **A safety
zero only means something if the arm APPROACHES the limit.** An arm that never turns cannot violate a
curvature constraint, and a gate that does not test for that will certify it as safe.
⇒ **Adopt: every safety-rate gate ships with a magnitude gate beside it.** This is the general form of
`M46`'s authority control, and it generalises past friction to every budget-shaped constraint.

### 5. Three more, all consequential

* ⭐ **The seed replicate reads `separated` on 4 of 8 rows with ZERO levers moved** — `H-ESTIM-SEED-1`
  measured on refav1 itself, with **separation and magnitude disagreeing in BOTH directions.**
  Confirms `M56`: a replicate is mandatory at launch, not after a headline.
* ⛔ **STRATEGIC and distance-keeping read UNAVAILABLE, n = 0** (no route channel; no lead track) —
  the same hole `M55` measured from the label side. **Two of the four binding families are structurally
  unreported on refav1's rig**, and that is a work item, not an omission.
* ⛔ **refav1's records are `UNKNOWN_SCOPE` to `criteria_check.py`** ⇒ **the programme's only T1 driving
  artifacts are INVISIBLE to the completeness machinery.** The machinery has therefore never audited
  the arms that matter most.

### 6. The next arm, named and unrun

⭐ **`cap + ladder + a0_shift`, `W_KAPPA = 0`** — the constraint (which preserves turns), the ladder
(which a constraint can use, and a penalty cannot — `M48`), and the longitudinal token fix (`M56` §5).
**It attacks all three named defects at once and carries no lever that deletes a manoeuvre class.**
⇒ This supersedes plan item A1 and is the highest-priority refav1 arm.

## M59. ⛔⛔ TWO RETRACTIONS THAT BOTH POINT THE SAME WAY — refcv5 AND the RL plan are FURTHER ALONG than I told the PI

### 1. RETRACTED — "refcv5's WP-4 sampler and WP-6 agent seam are not in the model"

I reported to the PI that `refc.py` contained `control_head` / `sampler` / `cross_agent` **0 / 0 / 0**
against a 54-`def` control, that 14 tests were red, and that **"this is the refcv3 defect repeating —
skeleton without mechanism."**

**MEASURED just now, with a passing control (59 `def` in the same read):**

| symbol | occurrences in `stack/tanitad/refs/refc.py` |
|---|---|
| `sampler` | **49** |
| `control_head` | **12** |
| `cross_agent` | **11** |
| `agent_gate` | **4** |

And the provenance is unambiguous: **`a5dbfbb`, 2026-09-05 14:57:05 +0200**, subject
***"refcv5 WP-4: the diffusion mechanism, in CONTROL space"*** — **committed hours BEFORE tonight's
session and before the report that said it was absent.**

⇒ ⛔ **The absence claim was FALSE, and I relayed it to the PI without verifying it.** Class: `M45`
exactly — **an INHERITED claim reported as MEASURED** — compounded by `M50`, *a search result trusted
without proving its channel*. ⚠️ The originating report also claimed **"7 of 7 local `refc.py` copies
read zero"**, which cannot be reconciled with a live file reading 49; **that report's grep, not the
repository, is what needs explaining.**

⭐ **What is NOT retracted:** whether the 14 tests pass is a separate question — tests can fail against
a module that exists. The wiring stream owns that and will report it. ⛔ **But "refcv5 has no
mechanism" is withdrawn**, and with it the claim that refcv5 was repeating refcv3's defect.

### 2. RETRACTED — my own mechanism argument in `M52`

`M52` said: *"DD-v2 post-trains a diffusion policy's DENOISING trajectory. refcv3 has no denoiser to
post-train."* **PUBLISHED-PRIMARY (banked `2512.07745` + released code at `1cd12a1`) refutes the
premise:** the released RL stage sets **`std_dev_t_add = 0.0`** and forms
`prev_sample = mean * eps_mul + 0 * eps_add` — ⇒ **it never uses the denoising chain's randomness at
all.** The only stochasticity is **two multiplicative scalars per trajectory** at a 4 % floor, and the
trained parameters are **`_trajectory_head` only** (everything else frozen and `.eval()`).

⇒ ⭐⭐ **A DETERMINISTIC FAN CARRIES THE PUBLISHED MECHANISM FAITHFULLY.** Therefore *"no denoiser"*
**does not block RL post-training on refcv4b**, and my reason for saying the refcv3 null carried no
information was **the wrong reason**.

⭐ **`M52`'s CONCLUSION stands on a better argument**, which the same work supplies: the refcv3 arm was
uninformative because **the reward had no headroom** — `sel_contact` and `top8_contact` both **0.0,
UNDETECTABLE-DOWNWARD** — and because the estimator's advantage is **identically zero in 92 % of
windows** (`D-RL-COLL-SPARSE-1`). **A reward aimed at a term already at its floor cannot show a gain
under any method.** That was always the load-bearing half; I led with the wrong half.

⇒ ⭐ **Both retractions point the same way: the PI was right to push, and we are further along than I
said. RL post-training can run on refcv4b as it is.**

### 3. ⛔ `G-REWARD` CANNOT BE PASSED AS WRITTEN — and the ceiling, not the reward, is what is wrong

`D-REFCV5-PLAN-6` makes `G-REWARD` a **hard precondition** (*"a reward that fails it may not train
anything"*), and **nobody had ever scored it.** Scored now, 0 GPU, **6,089 lead windows / 73 episodes**,
channel control exact (`err = 0.000e+00`):

* over **2,400 admissible weightings the MINIMUM is 0.3840** against a **<= 0.30** ceiling ⇒ **the gate
  is unreachable by construction**;
* the safety terms fire on **1.03 %** and **0.13 %** of windows while **progress and comfort fire on
  95.7 %** ⇒ ⛔ **the gate's POPULATION is wrong, not its threshold**;
* independently corroborated by the advantage being identically zero in 92 % of windows;
* **`comfort` is the entire defect and `feasibility` is inert** — removing feasibility alone moves
  0.7249 -> 0.7247 (a no-op); removing comfort delivers the whole move to 0.4419. ⭐ **A
  constant-velocity path has zero jerk and zero lateral acceleration, so it scores comfort EXACTLY
  1.0** — the same *"win a metric by declining to act"* pattern as `M46`/`M58`'s vacuity gate, now in
  the reward.

⇒ ⛔ **MASTER-MIND RULING: `G-REWARD` is RE-SPECIFIED to score on the SIGNAL-BEARING population
(lead-present windows), not on all windows.** A gate whose terms fire on ~1 % of its population is
measuring the population, not the reward. ⛔ **Until that re-specification lands, "G-REWARD fails" may
NOT be quoted as evidence that the reward is bad** — that is the agent's own caution and it is correct.
⭐ **`robust_contact` is built and scored** on the residual a projection provably cannot discharge: it
widens the human's advantage **1.89x -> 2.53x**, and on the signal-bearing population reads
**0.0635 [0.0000, 0.3158]** against the default's **0.8413**. ⚠️ **n = 63 windows / 6 episodes, CI
straddles the ceiling, and the restriction is post-hoc ⇒ necessary, not sufficient.**

### 4. A silent launch-blocker removed, and it is worth naming

The RL adapter passed **5 of the 8 parameters `forward` accepts**. refcv4b trains **with
`--ego-state-inject`**, so omitting `ego_state` **returns a well-formed fan from a DIFFERENTLY
CONDITIONED policy, with nothing raising.** ⇒ ⭐ **the arm would have run, produced numbers, and
measured a different model** — the most dangerous shape of defect this programme has, because its
output is indistinguishable from a result.

## M60. ⛔⛔ TWO INDEPENDENT REVIEWS CONVERGED ON THE SAME FALSE CONCLUSION — refcv5's architecture is PRESENT, and a shared stale mirror is why both said otherwise

### 1. The contradiction, and the measurement that settles it

Two agents, working separately, both concluded refcv5 was **skeleton-without-mechanism**:
* the readiness stream: `control_head` / `sampler` / `cross_agent` = **0 / 0 / 0**, *"7 of 7 local
  `refc.py` copies read zero"*;
* the independent review: **`DecoderConfig(sampler='ddim')` -> `TypeError`**, 15 red tests, and the
  headline *"if refcv5 started training tomorrow it would train a model with no sampler and no agent
  head."*

**MEASURED on this repository, with controls:**

| probe | result |
|---|---|
| `DecoderConfig(sampler='ddim')` | ⭐ **OK — constructs** |
| `DecoderConfig` sampler-ish fields | `sampler`, `sampler_space`, `sampler_train_t_max`, `sampler_infer_t`, `sampler_steps`, `sampler_groups`, `control_norm`, `cross_agent` |
| `refc.py` at **HEAD** | `sampler` 49 · `control_head` 12 · `cross_agent` 11 (control: 59 `def`) |
| `refc.py` at **`a5dbfbb`** (2026-09-05 **14:57**) | `sampler` **46** · `control_head` **8** · `cross_agent` **11** (control: 55 `def`) |
| `a5dbfbb:refc.py` lines 396-407 | `sampler: str = "none"`, `sampler_train_t_max`, `sampler_infer_t`, `sampler_steps`, `sampler_groups`, `sampler_space` — **all declared** |

The WP series landed **yesterday afternoon**: `4b36c96` 14:48 (WP-6 agent-token seam) -> `a5dbfbb`
14:57 (WP-4, *"the diffusion mechanism, in CONTROL space"*) -> `3fd2291` 15:10 (WP-6/WP-7 selector +
trainer) -> `68bc48c` 19:40 (feasibility-aware decode).

⇒ ⛔ **`DecoderConfig(sampler='ddim')` COULD NOT have raised `TypeError` from `a5dbfbb` onward.**
**Both reports are measuring something that is not this repository.**

### 2. ⭐⭐ The class, and it is the night's recurring one at AGENT scale

`M50` established that a known-flaky channel becomes an **alibi generator** — and that *"three probes
agreed on a wrong answer because they share one failure mode"* is exactly what *"take multiple
samples"* fails to protect against. ⭐ **Here the shared channel is not a mount read but a WORKING
COPY.** The banked hazard is explicit: **`tanitad-wt` is re-synced FROM the repo and silently drops
edits AND deletes repo-absent files mid-run**, and this session has already seen `scratchpad/pkg/`
overwritten between two siblings and a `RESULT.md` silently reverted to a stale 12,817-byte version.

⇒ ⛔⛔ **TWO INDEPENDENT REVIEWS ARE NOT INDEPENDENT IF THEY READ THE SAME TREE.** Agreement between
reviewers is evidence only when their **channels** differ, not merely their reasoning.
⇒ ⭐ **RULE: any agent asserting that code is ABSENT must first print the file's provenance — the
resolved path and the commit its tree came from — and assert a positive control from HEAD** (e.g.
`git show HEAD:<path> | grep -c <symbol>`). A count from an unnamed working copy is inadmissible.

⚠️ **I am not exempt: `M59` retracted the first report on symbol counts alone**, which is *"a count is
not a capability"* — the same error one level up. It reached the right answer for an insufficient
reason, and only the **reachability test** (`DecoderConfig(sampler='ddim')` constructing) and the
**`a5dbfbb` provenance** actually establish it.

### 3. ⛔ SCOPE THE CORRECTION — most of the review SURVIVES and is valuable

Only the *"no mechanism"* headline falls. **These are independent of the mirror question and stand
until re-measured:**

* ⭐⭐ **`n = 2,400` IS A CLIP COUNT, and only 23.30 % of WINDOWS are supervised** (76.70 % carry
  `IGNORE_ID`), measured on the real parity cache. **The label build's "100 % coverage" is 100 % of
  CLIPS.** ⇒ **This is the same defect `M55` measured on refav1 from the other side** — `lat_label`
  `-100` on **32 of 40 windows = 80 %**, against this rig's 76.70 %. **Two rigs, one defect**, and it
  is the single most consequential item in the review.
* ⛔ **The eval-side label join has NO COVERAGE FLOOR** — the 0.50 floor exists only on the train side.
* ⛔ **The STRATEGIC defect gate CANNOT FIRE, mutation-proven:** `refcv3_arm.py:2633` writes at
  `rec["refcv3"]["_defects"]` while `run_hierarchy_panel.py:71` reads `rec["_defects"]`. A record
  carrying a real defect returns **`ok=True`**. All three guards over it are blind — two are
  source-string matches, and one is a test whose **NAME** asserts *"at the top level"* while its
  **assertion does not**.
* ⛔ **`assert_matches_diffusers` can only SKIP or FAIL** — it compares float64 against float32 at
  `atol=1e-10` (float32 eps 1.192e-07; observed error 1.689e-07 = 1.42x eps). **The only independent
  check on WP-4's schedule math is incapable of passing.**
* ⛔ **`criteria_check --all`: 493 silent omissions, 61 work items, 7 unstamped tiers; STRATEGIC
  `nav_compliance` and both its controls read 0 present / 38 MISSING — never once produced.**
* ⛔ **The turn bound is 8.2x worse than stated**: **10.24 % of the turn class (n = 293)**, not 1.25 %
  of the corpus — **on the class whose recall already collapsed to 0.0000.**
* ⭐ **The 2,400 vs 2,376 question is RESOLVED, and there is no re-selection:** `3000 discovered −
  600 val = 2,400 CLIPS`; `− 24 raw-build skips = 2,376 raw EPISODES`. The v2/w120 cache built all
  2,400 and is a separately registered corpus (`skip_count 0`), digest recomputed **MATCH**; 24/24
  skipped clips named and labelled. **Parity is intact.** One sentence is owed in cross-arm tables:
  refcv5 trains 2,400, raw-epcache arms trained 2,376 — a **1.00 %** disjoint difference.

### 4. What this changes for goal 7

⭐ **The GPU is not the only blocker, and neither is the architecture.** The binding items are now:
1. **the supervision denominator** — 23.30 % of windows, not 100 %; the record must say so, and the
   eval-side join needs the same coverage floor the train side has;
2. **the STRATEGIC defect gate**, which cannot fire on a family that has never been produced;
3. **`assert_matches_diffusers`**, the one independent check on the sampler's math, which cannot pass.
⛔ **None of these needs a GPU, and none of them is "the model has no sampler."**

## M61. ⛔⛔ A FOURTH VARIANCE — THE SEED FLOOR IS RIG-DEPENDENT. Every "x floor" multiple carries its GPU or it is inadmissible.

### 1. The measurement

**MEASURED:** the **same baseline pair** — same arms, same flags — reads a seed floor of **−0.1000 on
one GPU and +0.0000 on the other**. And the in-rig floors on Thor are **~7x SMALLER** than the dev-box
floors that had been used to judge the same arms.

⇒ ⛔ **A floor measured on one rig, applied on another, is a scope error** — and it is **directional in
the dangerous way here: the too-LARGE floor SUPPRESSES real effects.** The stream retracted its own
*"not established"* on **curvature** and **tactical-lat** on exactly this basis: both costs are **real**
and had been dismissed against a floor from the wrong hardware.

### 2. ⭐ The estimator family now has FOUR members, and each answers a DIFFERENT question

| # | variance | the question it answers | how you measure it |
|---|---|---|---|
| 1 | **episode draw** | *would another draw of EPISODES say this?* | the paired episode-cluster bootstrap — **the ONLY one it answers** |
| 2 | **training run** (`H-ESTIM-SEED-1`) | *would another TRAINING RUN say this?* | a replicate arm, same flags |
| 3 | **inference run** | *would another INFERENCE RUN say this?* | a `--plan-seed` replicate — mandatory wherever the planner **samples** |
| 4 | ⭐ **the RIG** (this entry) | *would another GPU say this?* | the floor **re-measured on the rig the arm ran on** |

⇒ ⛔ **NAME WHICH ONE YOUR INTERVAL ANSWERED**, and ⛔ **quote every `x floor` multiple WITH ITS RIG.**
A multiple is a ratio of two numbers from **two possibly different machines**, and until tonight nobody
was checking that they matched.

⚠️ **Same shape as every scope trap this programme has banked** — `df` on a pod, `free` on Thor, cgroup
`usage_in_bytes`, `step_s`, the cylindrical FOV, `anchors.pt`'s units, `M52`'s model scope — *a true
measurement quoted outside the thing it was measured on.* **Here the scope is the HARDWARE.**

### 3. ⛔ Blast radius — bounded, but it must be checked, not assumed

Every claim of the form *"N x its seed floor"* is exposed **iff** the delta and the floor came from
different rigs. ⭐ **The remedy is cheap and mechanical: `raw/seed_floor.txt` already prints the
per-metric table — it now needs a RIG column, and every quoted multiple must name it.**
⚠️ **This compounds the already-banked rule that a floor belongs to the STATISTIC, not the family**
(a paired floor applied to a per-class recall withdrew a finding earlier tonight). ⇒ **A floor is now
identified by THREE things: the metric, the statistic, and the rig.**

### 4. Two further results from the same stream, both clean

* ⭐ **`T_lonseam` is exactly inert on all ELEVEN metrics — confirmed on a SECOND GPU.** `W_JERK = 0.0`
  makes the jerk-seam lever **arithmetically dead**; P4's first named successor is blocked by
  construction, and **`goal_reach_s` is now the live lever.** *(This is the independent replication of
  the inert-arm error banked earlier: a lever multiplied by a zero coefficient is not a null about the
  lever.)*
* ⭐ **The pre-registered "D2 ~ D1" outcome FIRES:** `a_shift` vs `a_sustain` **straddles zero** on LON
  speed and separates on **1 metric of 11 at only 2.1x the seed floor**. ⇒ ⛔ **the expressivity claim
  that D2 dominates DOES NOT TRANSFER TO THE PLANNER** — an oracle-level advantage that the search does
  not realise. **Reported as the pre-registered null, not reframed.**

### 5. Where refav1 stands, per family, honestly

**refav1 does NOT beat `ha0_ext` outright at T1.** But **`a_shift` is a real, REPLICATED,
seed-floor-clearing LONGITUDINAL win that halves the programme's stated gap** — and that is the first
replicated win in this package, against a night in which two headline zeros died to replication.

## M62. ⛔ I REVERSE MY OWN M54 SCHEDULING RULING — the `ccos` arms answer an OPEN question; `lonshift_s1` confirms a REPORTED one

### 1. The ruling, and why it cannot be delivered as written

`M54` §4 ruled that **`lonshift_s1` takes the next free dev-box slot.** MEASURED tonight, that ruling
would require an **intervention**, not merely a priority:

* `ta_queue3.py`'s `PLAN` holds **four** arms — `ta_wk15_s0`, `ta_wk15_s1`, `ta_ccos_s0`, `ta_ccos_s1`;
* its log shows it relaunching **one second** after a slot freed (`01:33:20 FINISHED ta_wk15_s0` ->
  `01:33:21 LAUNCHED ta_ccos_s0`);
* ⇒ **`queueLON4.sh` can never win a race against it**, even though `lonshift_s1` is correctly first in
  its own list (line 69, verified).

⇒ Delivering the ruling would mean **killing a sibling's live queue** — which risks cascading into two
running arms, on a box whose 2-arm cap exists because a third is a **measured OOM risk**.

### 2. ⭐ And the new information inverts the priority anyway

I had assumed the remaining turn-panel arms were another exploratory round. **They are not.** The two
`ccos` arms run at **`W_KAPPA` = 0**, and `M57` established that at `W_KAPPA` = 0 **the plan IS the
goal's canonical seed on 21/22 turn-goal windows.** ⇒ ⭐⭐ **The `ccos` arms ARE the head-baseline that
`M57`'s registered increment analysis requires** — the analysis whose whole point is

```
plan gap  −  head gap  =  INCREMENT      (the only part attributable to the COST)
```

**Without them, the cost's contribution to the left-turn deficit cannot be computed at all**, and that
is the central open question of goal 5's lateral half (`M54`, `M57`, `M58`).

⇒ ⛔ **`lonshift_s1` CONFIRMS a result already reported WITH its single-seed caveat. The `ccos` arms
ANSWER a question that is open.** ⭐ **An open question outranks confirming a reported one.**

### 3. The reversal, recorded rather than dropped

⭐ **RULING REVERSED: the turn panel keeps both slots until its four arms complete (~2.5 h);
`lonshift_s1` takes the first slot after that.** No process is killed and no gate is loosened.

⚠️ **`M54` §4's REASONING was not wrong** — validating a landed result *does* outrank another
exploratory arm; **its PREMISE was wrong**, because the competing arms were not exploratory. ⇒ ⭐ **The
error to avoid was ruling on a stream's priority without reading what it was actually going to run** —
the same shape as reading a name instead of the field that declares it (`M51`, `M53`) and a count
instead of a capability (`M60`). **I classified a queue by its owner rather than by its contents.**

⛔ **A reversed ruling must be RECORDED.** A ruling silently abandoned leaves the register asserting a
priority that nothing is executing, and the next reader cannot tell a decision from a drift.

## M63. ⭐ THE HIERARCHY PANEL'S DEFECT GATE COULD NOT FIRE — fixed reader-side, and my own mutation proof was under-mutated

### 1. The defect, confirmed at source

* **writer** — `taniteval/tools/refcv3_arm.py:2633`: `ref.setdefault("_defects", []).append(...)`,
  where `ref` is `rec["refcv3"]`;
* **reader** — `taniteval/tools/run_hierarchy_panel.py:71`: `rec.get("_defects")`.

⛔ **The reader looks at the top level; the writer files one level down.** And the writer's OWN comment
claims otherwise — *"recorded ON the block AND collected at the top level, so a driver can exit
non-zero instead of publishing a family-shaped hole."* ⇒ ⭐ **the code contradicts its own documented
intent, which is why three separate guards over it were all blind** (two source-string matches and a
test whose NAME asserts "at the top level" while its assertion does not).

⇒ **A record carrying a real defect returned `ok=True`**, and the panel published a family-shaped hole
as a result. **This is how the STRATEGIC family stayed silently absent from every refcv3 eval.**

### 2. Fixed on the READER, deliberately

`collect_defects()` scans the top level **and one level down**. ⭐ **Reader-side because it rescues the
records that ALREADY EXIST** — a writer-side fix would only help future runs, and every banked refcv3
record files its defects under `refcv3`. ⚠️ **Urgency was real, not theoretical: refcv4b's landing
panel runs this driver in ~6 hours.**

**Pinned by `stack/tests/test_hierarchy_panel_defect_gate.py` — 5 passed.**

### 3. ⭐⭐ The lesson is about the PROOF, not the fix: my mutation was INCOMPLETE

I reverted the **call** (`defects = collect_defects(rec)` -> `rec.get("_defects")`) and left the helper
**defined**. The proof then reported only **1 of 3** cases discriminating, and its own `>= 2` threshold
**failed the run** — which is what exposed it.

⛔ **The "both levels" case is a poor discriminator for `record_ok` anyway**: that record also carries a
top-level `_defects`, which the OLD reader already caught, so it reads `ok=False` on both trees. It
discriminates only through `collect_defects` being **absent** — which an under-mutation restores.

⇒ ⭐ **CLASS: A MUTATION THAT DOES NOT FULLY REINTRODUCE THE DEFECT UNDERSTATES THE TEST'S POWER.**
This is the exact mirror of the rule it serves — *a test that cannot fail on the defect it names is
decoration* — and it is the same failure one level up: **the mutation must restore the pre-fix tree,
not merely the pre-fix line.** After removing the helper too: **2 of 3 discriminate**, with the
**clean-record control passing on BOTH**, so the tests cannot be satisfied by a gate that always
returns `False`.

⭐ **What made this catchable was a threshold that could fail.** A proof script that merely *printed* a
table would have shown "1 of 3" and been read as success.

### 4. Still open, and NOT fixed here

⛔ The **writer's** comment remains wrong and its placement remains misleading — `refcv3_arm.py` is
being edited by sibling streams, so it is left alone deliberately rather than raced. ⚠️ **The reader
fix makes the gate correct regardless**, but the comment should be corrected when that file is quiet.

## M64. ⭐⭐⭐ OUTCOME A — THE TURN ASYMMETRY IS REAL, AND THE DEFECT IS NAMED: refav1's planner emits ZERO left turns

### 1. The verdict, on both inference seeds, every power target met

The power target was derived **before looking**: a recall on `n` trials moves in steps of `1/n`, so
`2/n <= 0.0750` ⇒ **n >= 27 per direction, >= 5 clusters.** ⚠️ **The banked panel's 11/8 resolved only
to 0.0909 / 0.1250 — COARSER THAN THE FLOOR ITSELF**, which is why it could never have decided this.

Achieved **30 left / 30 right, 6 clusters each, granularity 0.0333**, parity untouched (windows
stratified inside the same 8 episodes).

| arm | recall LEFT | recall RIGHT | R − L |
|---|---|---|---|
| `ta_wk15_s0` | ⛔ **0.0000** | 0.4333 | **+0.4333 [+0.1667, +0.6539] SEPARATED** |
| `ta_wk15_s1` | ⛔ **0.0000** | 0.5000 | **+0.5000 [+0.1874, +0.7407] SEPARATED** |
| `ha0_ext` (floor) | 0.7333 | 0.7000 | −0.0333 |

Panel's own seed floor `0.1935` (the CI's **reach**, per the stream's own hardening of its rule).
**Both gaps clear it by 2.24x and 2.58x, signs agree.** ⭐ And it survives the confound that voided the
banked panel: the **within-episode** contrast over the 4 episodes carrying both directions reads
**+0.4250** and **+0.5083**, both separated, while the floors read **−0.1917** and **+0.0000** — **the
plan's gap runs AGAINST its own floor.**

### 2. ⛔⛔ The named defect

**refav1's planner emits ZERO left turns.** Not *fewer* — **zero**, on **30 GT-left windows across 6
episodes, at BOTH seeds**, while emitting **13–15 right turns** on the matched stratum. **A structural
zero with full cluster support.**

⭐ **And it is the COST, not the goal head.** Restricted to windows where the head decoded the
**correct** turn token: GT-left **0/9 at both seeds** (median realised kappa +0.021/+0.025, full
magnitude **0/9**); GT-right **13/20** and **15/20** (median −0.080, full on 18–19/20). In episodes 1
and 6 the head decodes left correctly (**4/5** and **5/5**) and the plan still produces **0/5 in both.**
⇒ Combined with the full `+0.08` candidate being **handed** to the search and losing (`M54` §2), **the
loss is the cost comparison.**

⛔ **The next lever is NOT a direction-aware seed pool — refuted: that candidate is already there.**

### 3. ⛔ What is still OPEN, and is not being claimed

**That the curvature PENALTY causes the asymmetry.** The increment over the goal-head baseline is
**+0.0667 / +0.1333**, both **BELOW** the 0.1935 floor. The head-correct cell says the cost is doing
the work; the pooled arithmetic cannot separate cost from head. **`ta_ccos_s0/s1` (`W_KAPPA = 0`, same
75 windows) adjudicates it and is running.** ⭐ That is exactly why `M62` reversed the scheduling
ruling — those arms are the head-baseline, not another exploratory round.

### 4. ⭐ M57's CORPUS REQUEST IS NARROWER THAN I ESCALATED — correcting myself to the PI

`M57` told the PI that settling this **needs more EPISODES, not more compute**, because raising n did
not raise the cluster count. **That is true of the WITHIN-EPISODE contrast (4 clusters) and NOT of the
per-direction recall**, which reached **6 clusters by stratifying windows inside the SAME 8 episodes**
and **separated at both seeds.**

⇒ ⛔ **The corpus ask I escalated is real but bounded**: it constrains the *within-episode* attribution,
not the headline question, **and the headline question is now answered without it.** ⭐ **A power
computation done before looking is what converted an "unresolvable" slice into a decided one** — the
banked panel failed not because the corpus was too small but because **nobody had checked that its
granularity was coarser than the floor it was being judged against.**

### 5. Three cross-stream results, all from independent convergence

* ⭐ **The inert-seam diagnosis reproduced on a SECOND GPU** — `T_lonseam` reads `+0.0000 [0,0]` on all
  eleven metrics with a same-breath non-zero control. **Now two-rig.**
* ⛔ **A pre-registered outcome fired AGAINST its own author's attribution**: `lonshift − lonvocab`
  straddles zero on LON speed (−0.0549 [−0.1248, +0.0068]), along and ADE, separating on **1 of 11** at
  2.1x the in-rig floor. ⇒ **`D-REFAV1-LON-ADDRESSABLE-62` is REFUTED at the arm level**; the
  expressivity table's *"D2 dominates on every column"* **does not transfer to the planner.** Both
  remain real levers (D1 36.0 %, D2 47.4 % of the gap) — D2 weakly better, not distinguishably so.
* ⭐ **`M61`'s rig-dependence confirmed from the other side, and the author accepted the correction:**
  a `TAC_traj_lat_correct −0.1000` reported as *not attributable* on the dev box **is** attributable on
  Thor, whose in-rig floor is ~7x tighter. *"A seed floor carries its rig and its arm, never a
  programme constant."*

### 6. ⛔ A time-critical cross-stream refutation, delivered the right way

One stream named **`goal_reach_s` as the live next lever**; another had already **refuted it 0-GPU**:
tau = 1.0 gives LON speed **+0.1552** against D2's **+0.1517**, tau = 0.6 gives **+0.1825** and loses
the floor win entirely — with a **tau = 2.0 control reproducing D2 to four decimals**, so the negative
is a measurement rather than a broken rig. ⭐ **It put the refutation in the COMMIT SUBJECT LINE
(`e432341`), because a `git log` subject is what another agent actually reads.** That is the
escalate-integration rule executed properly — not a note buried in a document nobody re-opens.

## M65. ⭐ THE PRICING INSTRUMENT SAID *YES* — and the `lonshift_s1` ruling is MOOT, closed by evidence rather than abandoned

### 1. ⭐⭐ An instrument that only ever says "no" is not discriminating

The same 0-GPU cost-pricing that **declined D4** (~4.6 % of the gap, below the seed floor) and
**refuted D5** (worse on every headline column, with a tau = 2.0 control reproducing D2 to four
decimals) has now **approved** an arm. Priced from the banked plans exactly as `_cost_chunk` computes
them:

| arm | jerk term | **seam delta** | kappa term | goal term | seam / (goal+kappa) |
|---|---|---|---|---|---|
| `wk15` | 3.552e-02 | 7.403e-02 | 2.314e-02 | 2.646e-01 | **0.494** |
| `lonshift` | 2.606e-02 | 2.786e-02 | 4.067e-03 | 6.975e-03 | **4.05** |

⇒ The seam is **half** the size of everything it trades against on `wk15` and **4x larger** on
`lonshift` — **first-order, so the pair earns its GPU.**

⭐ **The check that made this non-trivial:** the obvious fix was a pair at the shipped
`W_JERK = 0.02`, but `W_KAPPA` in the same triple is **756x larger**, so the pair could have been
*technically live and practically inert* — **the same inert-arm trap one notch weaker, costing two more
GPU hours to discover.** Pricing it first is what distinguished the two cases.

⭐ **And the outcome was committed BEFORE the numbers:** at ratio 4.05 the seam could **dominate** D2's
objective in `loncomb3` and undo the vocabulary win. ⛔ **If `loncomb3` is worse than `lonshift`, that
IS the finding — not a reason to retune the weight.**

### 2. The `lonshift_s1` ruling is MOOT, and that is recorded rather than dropped

`M54` §4 ruled `lonshift_s1` takes the next free slot; `M62` reversed it because the competing `ccos`
arms are the head-baseline the cost attribution needs. **Both are now overtaken by evidence:** the
stream itself **moved `lonshift_s1` to LAST**, because Thor measured `a_shift`'s own seed floor and it
**replicates across two GPUs to 0.002 m/s.** ⇒ the replicate it was competing for is **no longer the
binding uncertainty**; it is kept only because `M61` established floors are rig-dependent.

⇒ ⭐ **Neither ruling binds anything now, and the queue was re-ranked on evidence that arrived after
both** — `lonvocab` dropped (Thor answered D1-vs-D2), `seambase`/`seamon` promoted to first as the only
queued experiment testing a lever nobody has tested. ⛔ **Recorded because `M62`'s own rule requires
it:** a ruling silently abandoned leaves the register asserting a priority nothing is executing, and a
later reader cannot tell a decision from a drift. ⭐ **This one was not abandoned — it was retired by
measurement, which is the outcome a ruling should have.**

### 3. The transferable point

⛔ **A pricing rule that has only ever refused arms is indistinguishable from a rule that refuses
everything.** Its first APPROVAL is what makes its earlier refusals evidence rather than caution — and
it arrived from the same instrument, on the same rig, with the same arithmetic. **Report an
instrument's first pass as carefully as its refusals.**

## M66. ⛔⛔ THE FLOOR WE MEASURE EVERYTHING AGAINST IS NOT DRIVABLE — `ha0_ext` violates the friction circle on 18.5 % of windows

### 1. The finding, and it reframes every refav1 comparison in the programme

**MEASURED** while rendering the arms reel: **`ha0_ext` is outside the friction circle on 18.5 % of
windows, peaking at 1.436 g.** It holds a **noisy MEASURED curvature** from t0, and that is exactly why
it beats every planner arm on ADE (**0.8772 m**).

⇒ ⛔⛔ **`ha0_ext` is a HARD-TO-BEAT REFERENCE, NOT A FEASIBLE ONE.** Every *"parity with `ha0_ext`"*
statement this programme has made — including tonight's headline that `best` reaches **+0.0066, 9.2x
below the seed floor** — is **parity with a trajectory a real car could not execute on nearly a fifth
of the windows.**

⇒ ⭐ **This does NOT make the comparisons wrong; it makes them incomplete.** `ha0_ext` remains the right
**echo control** — its whole purpose is to show what "just hold the initial action" achieves, and that
is a legitimate and demanding bar. ⛔ **But it is not a target**, and an arm that matched it exactly
would inherit an infeasible plan on 18.5 % of windows. ⇒ **Every `ha0_ext` comparison must now carry
its feasibility rate**, exactly as every interval carries its estimator.

⚠️ ⭐ **And it explains a puzzle from `M53`:** `ol` (the T0 kinematic contract) reads ADE **0.8052** and
`ha0_ext` **0.8772** — both far better than any planner. Part of that margin is bought with
**physically inadmissible curvature**, which no feasibility-respecting planner can or should match.

### 2. The zero-violation superlative, enumerated properly — third confirmation

⛔ **Six arms read `kamm_over_rate` 0.0 %**, not one: `combined`, `wk15`, `best`, `wk151`,
`wk15_ladder`, `cos_wk`. ⭐ **What is unique to `combined` is zero violations WHILE KEEPING CURVATURE**
(max|kappa| **0.1505**, and the same 27.5 % straight-plan rate as `ccos_argmax`) — the others reach zero
by not turning, `cos_wk` at **100 % straight**.

⇒ This is the **third independent confirmation** of `M58` §3's retraction, now with the **full
population** rather than a control set. ⭐ *A control set is not a census* — and the census is what
turns "the only arm" into the correct and much more interesting claim: **the only arm that is both safe
and still curving.** ⛔ And `M56` still stands over all of it: `combined`'s zero **dies at seed 1**.

### 3. ⚠️ A "correction" I am NOT accepting, because it is scoped to a different panel

The reel's stream noted *"no arm in the 13-arm panel has max|kappa| 0.0267"*. **True of ITS panel, and
it does not touch the figure I reported.** **VERIFIED at source:** `raw/lonshift_cannot_turn.md:25`
gives `max|kappa|` over 27 windows as **0.0267 1/m** for **`lonshift`** — an arm from the longitudinal
package that **is not in the reel's panel at all.**
⇒ ⭐ *A true statement about one population, read as a statement about another* — the night's most
frequent error, and it is worth recording that it also runs in this direction: **an agent's correction
must be scoped before it is accepted.**

### 4. ⭐⭐ What the video shows that no table could

**The planner's failure is not a slightly-worse number — it is a car leaving the road.** On clip
`dbad28c2` at 14.97 m/s the road is **dead straight**; the human, the floor, `wk15` and `best` all go
straight; **`ccos_argmax` swings hard right into the trees — ADE 9.113 m against `best`'s 0.139 m, at
3.262 g lateral.** ⛔ **A "1.3272 vs 0.8838" ADE column cannot show that one of those arms is a crash.**

⭐ And on `24fee8a5`'s left turn **the ordering inverts, visibly**: the physics-capped `combined` tracks
the human round the ramp while the curvature-charged `best` goes almost straight. ⇒ **The arms are not
better or worse than one another — they FAIL IN DIFFERENT DIRECTIONS, and the cost geometry picks
which.** That is `M54`/`M58`'s conclusion made watchable.

### 5. The GT control passed, and both halves were required

Recorded actions fed through the planner's **own** integrator over all 40 windows: **STEER reads ADE
0.1552 m** (median 0.1512) against the legacy **KAPPA at 0.8727 m** — **5.62x separation**, 8.57x on
turning windows. ⭐ **Both halves are required: a passing arm alone cannot distinguish a right
integrator from a loose tolerance.**

⭐ **Three defects were caught by decoding the output back rather than by exit codes** — a **3.9 %
vertical stretch on every frame while every exit code read 0** (ffmpeg's image demuxer silently
rescales to the first frame); the **ground truth drawn underneath the floor line** and therefore
invisible on straight windows; and the camera overlay **escaping its pane** into the arms table.
⇒ **A render that reports success is not a render that is correct**, and the only check that found
these was reading the artifact back.

## M67. ⛔ A CROSS-STREAM GPU COLLISION IS LIVE — Thor is armed to run a lever that was refuted 0-GPU four hours earlier

### 1. The collision

* **Thor stream:** *"The `goal_reach_s` panel is shipped, verified and syntax-checked on Thor, ready to
  launch the moment the rig frees."*
* **Dev-box stream, commit `e432341` (2026-09-06 01:47), SUBJECT LINE:**
  ***"STOP -- goal_reach_s is ALREADY REFUTED 0-GPU (do not spend the arm)"*** — with the refutation
  written into `GOALS_AND_CLAIMS.md` (7 insertions).

**The refutation, MEASURED 0-GPU:** tau = 1.0 gives LON speed **+0.1552** against D2's **+0.1517**;
tau = 0.6 gives **+0.1825** and **loses the floor win entirely**; and a **tau = 2.0 control reproduces
D2 to four decimals**, so the negative is a measurement rather than a broken rig. It helps only the 6
clip-bound windows, as a cruder D4.

### 2. ⭐ Does `M61`'s rig-dependence rescue the Thor run? — I checked, and NO

`M61` established seed floors are **rig-dependent**, and Thor's are **~7x tighter**. ⚠️ That is a real
reason to ask whether a dev-box negative transfers. **It does not rescue this one:** the refutation is
a **DELTA in the wrong direction** (+0.1552 vs +0.1517 — *worse*), not a null. ⇒ **A tighter floor makes
a wrong-direction delta MORE clearly separated, not less** — it would confirm the negative with more
confidence, at the cost of a GPU arm.

⚠️ **The one caveat I cannot resolve from here:** if the Thor panel composes `goal_reach_s` with
something else rather than substituting it for D2, it is a different experiment and the refutation does
not bind. **That is the stream's to determine — and it is exactly the question `e432341` should prompt
it to ask.**

### 3. ⭐ The escalation mechanism worked; the question is whether it is READ

⭐ The refuting stream did **everything right**: it put the STOP in the **commit subject line** because
*"a `git log` subject is what another agent actually reads"*, and it wrote the finding into
**`GOALS_AND_CLAIMS.md`**, which the operating standard requires a fresh context to read **before
acting**. ⇒ **The channel exists and is correct. This entry exists so the next loop tick checks whether
it was used** — and, if Thor launches anyway, so the failure is attributed to the READ and not to the
escalation.

⛔ **Master-Mind position: do not unilaterally block another stream's queue.** The register carries the
refutation; the stream owns the launch decision and may have a composition the refutation does not
cover. ⭐ **What I own is making sure the collision is visible** — no agent can see a sibling in real
time, and the orchestrator is the only party who can.

### 4. ⚠️ And a third stream has hit the committer's silent-success path

A refcv5 wiring commit subject reads: ***"log the commit tool that exited 0, printed nothing, and
committed nothing."*** ⇒ **third independent occurrence tonight**, after my own stale-cached-read empty
commit and the earlier pipeline misattribution.

⭐ **The mechanism is now identified and it is NOT the pipeline** (that was `D-SHELL-PIPEFAIL-1`, and it
was my error): `mktree_commit.py`'s `main()` has an **early `return 0` on
`"nothing to commit (tree identical to HEAD)"`** — a **success exit that commits nothing**. It is
correct when the content really is identical, ⛔ **but on a caching mount a STALE READ makes the
rebuilt tree equal HEAD's while the file on disk has new content** — and the message reads reassuring.
⇒ ⭐ **This is why the content-marker-inside-the-blob check is not optional**: it is the only check that
separates "nothing to commit" from "I read the wrong bytes." ⚠️ A distinct exit code for that path
would be the durable fix; it is deliberately **not** applied while agents are committing through the
file every few minutes.

## M68. ⭐⭐⭐ refcv5 IS TRAINING-READY — and M59/M60 were half-wrong: the leaves existed, the TRUNK did not

### 1. ⛔ The correction I owe, third-order and final

**M59** retracted the *"WP-4/WP-6 are missing"* report as FALSE. **M60** went further and blamed a
stale mirror for *"two independent reviews converging on a false conclusion."* **Both were half-wrong,
and the wiring stream's account is the authoritative one:**

| | truth |
|---|---|
| the `0/0/0` count | ⭐ **CORRECT — but scoped to `refc.py`** |
| the implementations | **existed in HEAD, committed and UNREFERENCED**: `refc_sampler.py` (314 lines / 14 defs, `a5dbfbb`) and `refc_agents.py` (820 lines / 20 defs) |
| the trainer seam | **already complete** (`_pin_refcv5_seams`, `_seam_stamp`, all flags, 4 refusals) |
| what was actually missing | ⛔ **the TRUNK** — `refc.py`'s `CrossAttnLayer` / `AnchoredDiffusionDecoder` / `RefCModel` and `refc_v3.py`'s `RefCV3Model.forward` |

⇒ ⭐ **It WAS an escalation-to-merge, exactly as the report framed it — and nothing was rewritten.**
⛔ **My error: I measured the CONFIG SURFACE in `refc.py` (fields, flags, ~46 mentions at `a5dbfbb`)
and concluded "the mechanism is there."** A dataclass field that constructs is not a module that runs.
⇒ **"A count is not a capability" — which I wrote in `M60` §2 about someone else's error, and then
committed myself in the same entry.** The reachability test I ran (`DecoderConfig(sampler='ddim')`
constructing) probed the **config**, not the **model**, and I read it as settling the model.

⚠️ **What `M60` still gets right** and should not be discarded: two reviews are not independent if they
read the same tree, and an absence claim must print its resolved path and provenance. ⛔ **What is
withdrawn is its verdict that the reports were false and its stale-mirror explanation.**

### 2. ⭐⭐ THE HEADLINE: refcv5 is wired, guarded, tested — the ONLY blocker is a GPU

* **20 red tests, not 14.** A **third** pre-existing spec (`test_refc_agents.py`, 6 tests) was found
  only by running the full suite — ⭐ and the stream **conformed its own API to the existing names**
  (`agent_tokens`/`agents`, not its `agent_tok`), **never the reverse. No test weakened.**
  **14 failed / 23 passed → 81 passed / 0 skipped**; **215 passed** across the 41-module blast radius.
* ⭐⭐ **The `roll_bank` bit-identity is EXACT**: 117 anchors x 8 slots x 5 speeds x 2 coords =
  **9,360 float32 scalars**, from **standstill (0.0 m/s) to 36 m/s**, `torch.equal` **True**, max |diff|
  **0.000e+00**, **0 differing bit patterns, in BOTH unit systems.**
* ⭐⭐⭐ **The mechanism is REAL and refcv3's pathology does not reproduce.** refcv3's ranking was
  *"unchanged on 201/201 windows"*; adding a DDIM rung here **moves the output on 128/128 rows.**
* **Full suite 35 failed / 6359 passed — all 42 pre-existing** (HEAD baseline on the same 13 files =
  40, identical). **The change fixes 6 and introduces 0.**

⇒ ⭐ **GOAL 3 IS COMPLETE. GOAL 7 is gated only on the A40 freeing (~08:05 UTC).** The two remaining
preconditions are corpus flags the trainer already **refuses** without: `--agents head` needs
`--agent-join` (⭐ **`--agents oracle` is the first rung and needs no detector**), and `--sampler ddim`
needs an `--anchor-file` carrying `controls` + declared `control_units`.

### 3. ⭐ Four guards fixed, each shown capable of failing

* **False provenance CLOSED, and the dataclass field alone was not the fix** — the stamp is built from
  the **config** (intent) while only the **model** is fact. `assert_seams_are_built(model, stamp)`
  checks **bidirectionally** before `config.json` is written. **4 reintroduced defects rejected, 2
  honest controls pass.**
* ⛔ **`assert_matches_diffusers` had NEVER RUN** — diffusers was never installed, so its impossible
  `atol=1e-10` was masked by a SKIP. Measured: ours-f32 vs diffusers **0.000e+00 (bit-equal)**; against
  an exact f64 reference **ours 6.939e-18, diffusers 1.689e-07**. Replacement is **stricter**; skip
  count now **0**. ⇒ ⭐ *`M60` listed this as "can only SKIP or FAIL" — the truth is worse: it had never
  executed at all.*
* ⛔ **Its own deliberate-regression arm COULD NOT HAVE FAILED** — the metre arm divided by
  `metre_sigma_m`, **31.7x too gentle**, so the DD-literal arm would have *passed* the gate it exists to
  fail. Fixed: control **0.53 m/s², 0.0000 over mu=0.7** vs metre **5.29 m/s², 0.3333 over** — matching
  the design plan's predicted ~5.8. ⭐ **A deliberate regression that passes is worse than no control.**
* **`CRITERIA_REGISTRY` v2.7.0 -> v2.8.0 adds `hyg.inference_seed`** + regression arm + control
  (`tools/tests` **110 passed**). ⭐ **The WP-4 sampler draws fresh noise at eval BY DESIGN, so refcv5
  is the refc line's first STOCHASTIC planner** — the registry could not express that requirement
  before (7 probes read 0 against a control of 46). ⇒ **`M56`/`M61`'s inference-seed doctrine is now
  machine-enforced for refcv5 from its first arm**, rather than learned after a headline dies.

### 4. ⚠️ A config decision that must be made, not a bug

**`SelectionConfig.refined` defaults `False`**, so **the fan is SAMPLED but RANKED BY A HEAD THAT NEVER
SAW THE SAMPLE.** Stamped via new telemetry `sampler_ranks_the_fan`. ⛔ **An arm that wants the sampler
to reach selection must pass `--sel-refined`** — otherwise refcv5 ships a diffusion sampler whose output
the selector ignores, which is refcv3's defect wearing a working denoiser.

### 5. Two process failures worth carrying

* ⛔ **`robocopy /MIR` reported SUCCESS and silently did not copy the changed file** — the tests then
  reproduced the old failures verbatim. *Same family as every "reported success" trap tonight.*
* ⛔ **`mktree_commit.py` exited 0, printed nothing, and committed nothing** during a G: outage —
  caught **only** because 10 of 11 content markers read MISSING against controls reading 1–95. ⇒ **the
  content-marker check is what stands between us and silent data loss**, and this is its third
  independent save tonight.

## M69. ⭐⭐⭐ ROW 3 MEASURED (the head is NOT dead) — and the TURN GOAL ITSELF IS WRONG, which qualifies my own M48/M58 ruling

### 1. Row 3: the ~0-motion readout is NOT a dead emission head

⭐ **The "blocker" dissolved on a second probe — for the fourth time tonight.** The gate stream reported
Row 3 *"blocked solely on the path to a banked v7-tiny checkpoint"*; **Thor holds 54 of them**, in
directories named after the board's own arms. *(`CLAUDE.md`: absence found at ONE location is not
absence.)*

MEASURED, zero GPU, `champ30k/ckpt.pt`, **step 30000**:

| layer | shape | norm | std | absmax |
|---|---|---|---|---|
| `step_readout_op.net.3` (final emission) | (3, 512) | 1.00138 | **0.025558** | 0.044183 |
| `step_readout_op.net.1` | (512, 4096) | 13.06625 | 0.009023 | **0.015625** |
| `step_readout_op.net.0` (LayerNorm) | (4096,) | 64.0 | **0.000000** | 1.000000 |
| **CONTROL** `readout.proj` | (128, 128) | 6.11773 | **0.047796** | 0.311320 |

**Ratio final-emission std / control std = 0.5347.** ⇒ ⛔ **The emission head is within 2x of a working
head — it is NOT dead, collapsed, or near-zero.** Combined with the gate stream's independent finding
that **the T1 eval path carries no multiplicative constant anywhere** (`StepDisplacementReadout` is a
bare MLP; `rollout_decode` only accumulates; `dense_speed_profile` is `norm(dp)/dt`), **Row 3's cause is
neither the harness NOR the head's own weights.**

⇒ ⭐ **That is real progress and it redirects the search**: the ~0 motion must come from **what the head
is FED, or what it was TRAINED AGAINST** — not from a scale defect. ⛔ **And it retires my own claim to
the PI that "if it is a scale error, every T1 number is recoverable by re-analysis with zero
retraining."** It is not a scale error.

⚠️ **Two caveats stated rather than buried:**
* `net.1`'s absmax is **exactly 0.015625 = 2^-6** on **both** weight and bias — a **quantisation
  signature**. If this checkpoint is quantised, the std ratio is **not directly comparable** to an
  unquantised control. ⇒ a second probe was owed before quoting 0.5347 as decisive.
  ⭐ **RESOLVED — the owed probe was run in the same session and the checkpoint is NOT quantised.**
  MEASURED: `net.3.weight` is float32 with **100.0 % unique values** (1,536/1,536) and **no grid**
  (max off-grid 0.4995); the control `readout.proj.weight` reads **100.0 % unique**, off-grid 0.5000
  — **identical behaviour**. `net.1.weight` is 94.0 % unique over 2.1 M values, which is float32
  collision at scale, not a grid. ⇒ the `2^-6` absmax is a **clipping or init bound**, not a
  quantisation step. ⭐ **The 0.5347 ratio IS comparable, and Row 3's finding stands: the emission
  head is NOT dead.**
* ⭐ `net.0` (LayerNorm, 4096) sits at **EXACTLY initialisation after 30,000 steps** — weight all 1.0
  (std **0.000000**), bias all 0.0. Even tiny gradients would break an exact tie across 4,096 values.
  ⇒ **either frozen by design or receiving NO GRADIENT**, and that is worth explaining before any v7f
  launch.

### 2. ⛔⛔ THE TURN GOAL IS WRONG — obeying it is worse than driving straight

MEASURED (paired attribution by goal token, grid control passed on all 5 pairs):
**`GOAL_KAPPA_TURN = 0.08` is R 12.5 m**, on a corpus that curves at **R 100–1000 m.**

| TURN_L windows | ADE |
|---|---|
| obeying the command (kappa 0.08) | **1.5043** |
| partially under-turning | 1.2429 |
| under-turning most | **1.1959** |
| **`ha0` straight-line floor** | **1.4630 — beats the faithfully-turning planner** |

⇒ ⭐⭐ **The planner is not wrong to under-turn; it is obeying a command roughly 10x too sharp.** And
**78–89 % of every lever's ADE gain is earned on `LANE_KEEP` windows** — the turn windows were never
where the ADE lived.

⇒ ⛔ **This reframes `M54`/`M58`/`M64`.** "The cost discards the correct candidate" is true — **but the
"correct" candidate is a 12.5 m-radius turn that is itself wrong for this corpus.** ⇒ **A
goal-conditioned cost alone recovers RECALL and WORSENS turn-window ADE, by arithmetic** (predicted
0.9777). ⭐ **That is not the fix failing; it is the fix correctly obeying a wrong goal.** The stream
withdrew half its own prediction **before** its arms landed, which is the only reason this is legible.

### 3. ⛔ My M48/M58 ruling is QUALIFIED — `kamm07`'s "free" turning was a NON-FIRING

I ruled *"carry the CAP, drop `W_KAPPA`"* because the cap buys 67–77 % of the ADE **at zero turning
cost**. ⚠️ **MEASURED: that zero is a NON-FIRING.** At **v0 ~ 2.78 m/s**, **0 of 13** TURN_R windows
could bind the friction circle at all. ⛔ **At 20 m/s the same circle cuts 0.08 to 0.0172 — it would
under-turn HARDER than the penalty.**

⇒ ⛔ **Every turning claim in that frontier is LOW-SPEED**, and the ruling must be restated as: *the cap
is preferable to the penalty **on this low-speed slice**, and its behaviour at road speed is
**unmeasured and predicted to be worse**.* ⭐ Same class as `M61` (a floor carries its rig) and `M52`
(a result carries its model): **a lever carries the OPERATING POINT it was measured at.**

### 4. What survives, and the next lever

⭐ **The under-turn is now quantified rather than thresholded:** at `W_KAPPA = 15.11245` the plan emits
curvature on **100 %** of TURN_L-goal windows — at **0.02066** against the commanded **0.08000**, a
**3.9x under-turn**. Recall only *thresholds* it. And it is **asymmetric under a symmetric kappa^2
term** (TURN_R unchanged at 0.08000) ⇒ **refav1's turn asymmetry is a COST-BALANCE fact, not a head
fact.**

⭐ A second independent route agrees: the weight at which a full turn loses to doing nothing is
computable at median **153.4**, while recall dies by **15.11** — a **10x gap** that confirms the
under-turn mechanism from the cost columns alone.

⇒ **`A4a_gk_kt02`** (goal-conditioned cost **+** `--goal-kappa-turn 0.02`, a constant so it stays T1
and never an oracle chooser) is **running on Thor**, with `A4b_kt02` as its attribution arm and the
refutation branch committed: **if it fails, the defect is upstream of the cost and the next lever is
`--lat-logit-bias`, not another cost term.**

### 5. ⚠️ `M67`'s collision resolved — and NOT in my favour. Recorded as it happened.

`M67` flagged that Thor was armed to run `goal_reach_s` against a sibling's **`STOP -- ALREADY
REFUTED 0-GPU`** commit subject, and said *"this entry exists so the next tick checks whether it was
READ."* **It launched.**

⭐ **And on inspection the launch is DEFENSIBLE, so my flag is not vindicated and I say so.** What went
to the GPU is **`T_grsCTL` / `T_grs1` / `T_grs8` / `T_grs1_s1`** — a **4-arm panel with the CONTROL
FIRST and a SEED REPLICATE**, plus a **zero-GPU unit test proving default-path safety** (`None`/`2.0`
bit-identical to omitted; `1.0` doubles the opening acceleration exactly as `a_i = (v_t - v)/reach`
predicts). ⇒ **That is a stronger test than the parameter sweep it duplicates**, and it carries the
right reading instruction: ⛔ *read `T_grsCTL` first — if it does not reproduce `T_lonshift`, every
other arm in the panel is void.*

⇒ ⭐ **The transferable point is about MY role, not theirs:** an orchestrator seeing two streams should
**surface** a collision, and must then accept that the owning stream may have a better-designed
experiment than the one being duplicated. ⛔ **A cross-stream STOP is evidence, not authority** — and
`M67`'s own caveat (*"if the Thor panel composes rather than substitutes, the refutation does not
bind"*) turned out to be the operative clause. **The GPU spend is real; so is the fact that the
resulting panel answers the question better than the refutation did.**

## M70. ⛔⛔ THE SHARED COMMITTER'S NAME-PRESERVATION GUARDS WERE DELETED — and their tests still read like coverage

### 1. The finding

MEASURED by positive assertion, each with a same-breath control (`def main` = 1):

| commit | `assert_names_preserved` | `read_tree_entries` |
|---|---|---|
| `e685d90` | **3** | **6** |
| `2c47dc9` — *"the committer that could land a commit tonight when mm_commit could not"* | **0** | **0** |
| **`HEAD`** | **0** | **0** |

⇒ ⛔ **A rewrite deleted the guards that refuse a LOST TREE ENTRY and a CR-MANGLED NAME** — precisely
the failure modes `CLAUDE.md` documents at length, in which **a commit's subject sits in the log while
its content is gone from HEAD**. The six `test_mktree_commit.py` failures dismissed as *"pre-existing"*
all raise `AttributeError: … has no attribute 'read_tree_entries'` — **one cause, on the tool every
agent commits with.**

⇒ ⭐⭐ **This is the likely mechanism behind tonight's three independent "exited 0, printed nothing,
committed nothing" events** — including one of mine.

⭐ **The sharpest part: a guard that has never been shown to fail proves nothing; a guard DELETED WHILE
ITS TESTS STILL NAME IT is worse, because the test file still reads like coverage.** A red suite that
everyone has learned to call "pre-existing" is indistinguishable from a guarded one until someone reads
the failure text.

⚠️ **Scope, stated honestly:** `git log -- stack/scripts/mktree_commit.py` returned **empty** on this
mount while the file demonstrably has history, so that table is **what the positive probes found, not a
complete history.**

### 2. ⛔ MASTER-MIND RULING: do NOT repair the shared committer tonight

⭐ **The agent was right to record and not repair.** Rewriting the shared commit path while ~8 agents
commit concurrently is **exactly how a sibling's work disappears** — the failure mode the guards exist
to prevent. Repairing it under contention would risk causing the harm it protects against.

**RULING, in three parts:**
1. ⛔ **No edit to `mktree_commit.py` while agents are live.** Queue the repair for a quiet window.
2. ⭐ **The compensating control is ALREADY IN FORCE and is sufficient**: every commit must be verified
   by a **content marker INSIDE the committed blob**, with a same-breath control that must read
   non-zero. **That check caught all three silent-failure events tonight**, and it is in every live
   brief. ⇒ **We are not unprotected; we are protected by a different layer than we thought.**
3. ⭐ **When the repair happens, the CAS must be KEPT, not reverted.** Recovery source is
   `git show e685d90:stack/scripts/mktree_commit.py`; the current compare-and-swap is what makes a lost
   race FAIL instead of clobbering, and `e685d90` predates it. ⛔ **A straight revert would trade one
   integrity bug for a worse one.**

### 3. ⭐ The transferable lesson, and it is about how we read test failures

⛔ **"Pre-existing failures" is a category that hides deletions.** Six red tests were carried for an
unknown period as background noise; their failure text named a **missing function**, which is the
signature of removed code rather than of a flaky test. ⇒ ⭐ **A red test whose error is
`AttributeError: no attribute <name>` is a DELETION until proven otherwise — read the message, not the
count.** Every stream tonight that reported a suite result reported it as *"N failed, all
pre-existing"*; **none had read what the N were.**

⚠️ This is the same family as `M63`'s under-mutated proof and the reel's *"exit code 0 while every frame
was stretched 3.9 %"*: **an artifact that reports its own health in a form nobody parses.**

## M71. ⭐ THE AT-INIT NORM LAYERS ARE THE FINGERPRINT OF v7-tiny's ZEROED PLANNER OBJECTIVES — Row 3's last thread closed

### 1. The measurement, with its control

`M69` flagged that `step_readout_op.net.0` (LayerNorm, 4096) sat at **exactly** initialisation after
30,000 steps and asked whether it was frozen by design or gradient-starved. The discriminator is
whether **other** norm layers did the same. MEASURED, zero GPU, `champ30k/ckpt.pt`, control **39 1-D
`.weight` tensors found**:

| group | count | examples |
|---|---|---|
| ⛔ **EXACTLY at init** (std **0.000000**, mean 1.0) | **25 / 39** | `vocab_str`, `vocab_tac`, `vocab_a_str`, `vocab_a_lat`, `vocab_a_lon`, `cond_op.vocab.norm`, `cond_tac.vocab.norm`, `adapter_tac.3`, `predictor_tac.blocks.{0,1,2}.0`, **`step_readout_op.net.0`** |
| ⭐ **MOVED from init** | **14 / 39** | **every one is `encoder.blocks.N.norm{1,2}`** — std 0.0111–0.0195 |

⇒ ⭐⭐ **The split is clean and structural: the ENCODER's norms trained; the VOCABULARY, CONDITIONING,
TACTICAL-PREDICTOR and READOUT norms did not.** Not one layer — an entire half of the model.

### 2. ⭐ It is a CONSISTENCY CONFIRMATION, not a new defect

The v7f status board already established that **v7-tiny ran with the planner objectives at ZERO**
(19.3 M params = 6.5 % of budget). ⇒ **The modules those objectives would have trained are exactly the
25 at-init norms.** The fingerprint matches the known configuration, which is the outcome you want from
a probe like this: **it corroborates the board rather than adding a mystery.**

⚠️ **One residual that is genuinely odd and is NOT explained by the zeroed objectives:**
`step_readout_op.net.1` and `.3` **DID train** (std 0.0090 and 0.0256) while `.0` — the LayerNorm
*between them and the input* — did not. Gradient reaching `.1` must pass through `.0`. ⇒ **the most
likely account is an optimizer PARAM-GROUP split** (norm affine params excluded outside the encoder),
not a dead path.
⭐⭐ **REFUTED BY SOURCE, in the same session — and the question is now SHARPER, not answered.**
`train_v6_staged.py:8165` is `torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)` — **a SINGLE
param group**, no `no_decay` split, no norm-affine exclusion (control: 151 `def` in the same read).
The only freezes are `--freeze-encoder` (`:6574`) and `--freeze-readout` (`:6581`), **both
flag-gated** — and `champ30k`'s encoder norms **MOVED**, so `freeze_encoder` was OFF.
⇒ ⛔ **Neither an optimizer group nor either available freeze explains 25 norm weights sitting at
EXACTLY 1.0 through 30,000 AdamW steps.** ⭐ And with a non-zero `weight_decay`, a parameter that is
IN the optimizer drifts even at zero gradient — decoupled decay shrinks it. **Staying bit-exactly at
1.0 therefore implies those params are NOT IN `trainable` at all**, by a route not in the two blocks
above (`trainable = [p for p in stack.parameters() if p.requires_grad]`, `:5638`/`:6596`).
⇒ ⛔ **OPEN, and it is now MORE important, because v7f inherits this trainer:** find what clears
`requires_grad` on those modules, or show `wd == 0` makes the inference unsound. **This is owed
before any v7f launch** — because if it is *not* a deliberate group, then every non-encoder norm
in v7f will also fail to adapt.

### 3. What this does to Row 3

⭐ **Row 3's chain is now complete and consistent:**
1. the eval **harness** carries no multiplicative constant (gate stream, independent);
2. the emission head's **weights are not dead** — std ratio **0.5347** against a working head, and the
   checkpoint is **not quantised** (100.0 % unique, no grid, control identical);
3. its input **norm never trained**, along with 24 others, **because the objectives that would have
   trained them were off.**

⇒ ⛔ **The ~0-motion readout is a property of what v7-tiny was TRAINED to do, not of a broken head or a
broken harness.** ⭐ **And that is exactly why it cannot be read as a v7f defect:** the board's first
row already says all six problems are measured on an arm whose planner objectives were zero. **Row 3
does not gate v7f; it gates any claim made FROM v7-tiny.**

⇒ ⭐ **The cheapest remaining v7f question is therefore not Row 3 at all** — it is whether the
param-group split is deliberate, which is a source read, not an experiment.

## M72. ⭐⭐ refcv4b BEATS refcv3 BY 35 % SEPARATED — and its entire win lives in the ego pathway, which is a PI ruling

### 1. The preliminary result (CPU, 171 windows / 20 episodes, `ckpt_30000` vs refcv3's final 40,284)

⭐ **Both models are on ONE surface, proven before the comparison:** the model-free arms read
**bit-identically** across both runs (`ha` 0.2860, `ha0` 0.6542, `ha0_ext` 0.2769), and refcv3's
checkpoint md5 matches the registry.

| | refcv3 @40,284 | **refcv4b @30,000** |
|---|---|---|
| `os` ADE | 0.4707 m | **0.3055 m** |
| **paired** | **−0.1652 [−0.2318, −0.1090] SEPARATED, −35.1 %** | |
| LONGITUDINAL kappa | 0.2037 | **0.5782** (brake recall 0.3571→0.6429, accel 0.3704→0.6667) |
| LATERAL kappa | 0.7554 | 0.7039 |

⭐ **And it is 10,284 steps LESS trained.** **Void gate passes:** `frames_blind` → 1.2198 m separated,
both kappas collapse, and `ha`/`ha0`/`ha0_ext`/GT return bit-identical ⇒ **the panel is admissible.**

⛔ **But it still LOSES to a trivial hold-action control: 0.3055 against `ha`'s 0.2860.** Reported as
written. *(Same shape as refav1: better than its predecessor, not yet better than doing nothing
clever.)*

### 2. ⛔⛔ THE ENTIRE ADVANTAGE IS EGO-MEDIATED — and that is a PI ruling, not a design choice

**Axis attribution, two ablations on the same checkpoint and windows:**
* **vision-only** keeps lateral kappa **0.6738 (96 % of full)** and longitudinal **0.0653 (11 %)**;
* **ego-only** keeps **neither**.

⇒ ⛔ **refcv4b's LATERAL competence is vision-derived; its LONGITUDINAL competence — which is the whole
of its advantage over refcv3 — DOES NOT EXIST without the measured ego block.**

⭐ **Ranked by measured effect, the ego pathway is 0.8082 m separated — the LARGEST lever in the panel**,
**3–240x every other lever measured**, and it is precisely the one the vision-only rule constrains.
⚠️ The PI has already ruled that **measured `v0` at t0 is admissible** (velocity at cycle time,
2026-09-02). ⛔ **What is NOT ruled is WHICH ego channels beyond that are admissible at inference — and
that ruling selects between 0.3055 m and 1.1137 m**, a **3.6x** swing in the headline number.
⇒ **Escalated as a PI decision. It is not mine to take, and no refcv4b headline should be quoted
without stating which ego channels it assumes.**

⇒ ⭐ **The highest-value change this proves for refcv5** (goal 2): make the **longitudinal** competence
**vision-derived or PREDICTED rather than ego-mediated.** That is where the entire win lives, where the
entire deployment exposure lives, and it dominates every other lever by 3–240x.

### 3. ⛔⛔ `sel_refined` SEPARATES THE WRONG WAY — correcting a note I relayed hours ago

`M68` §4 relayed the wiring stream's config note: *"an arm wanting the sampler to reach selection must
pass `--sel-refined`."* **MEASURED: turning it on is 0.0259 m SEPARATED WORSE.**

**Mechanism, and it is diagnostic:** 30 % of picks flip; the **lateral decision is byte-identical**;
longitudinal *accuracy* rises while **both minority recalls fall** ⇒ ⛔ **a MAJORITY-CLASS TRAP.**
**The head was never trained to rank.**

⇒ ⭐⭐ **WP-7 SELECTOR TRAINING IS THE LOAD-BEARING PART, NOT THE WIRING.** Wiring the sampler into
selection without training the ranker makes the model *worse* — ⛔ **so `--sel-refined` must NOT be set
on refcv5's first arm on the strength of `M68` §4 alone.** *(Same family as `M48`: giving a search a
new option is not the same as giving it a reason to prefer the right one.)*

⚠️ `h19_off` reads 0.0034 m, **not separated** — sub-1 % of ADE, low value either way. ⭐ Its
zero-coefficient check was run **first**: the weights are live, so this is a real null and not an inert
arm (`M65`'s lesson applied prospectively).

### 4. Three instrument facts worth carrying

* ⛔ **The pod was missing 137 of 224 `taniteval` files, including `nav_compliance.py`** — the module
  whose absence hid the STRATEGIC family for its entire life (`M55`). Synced, 210/210 md5-identical.
  ⭐ Model code deliberately **not** synced: the repo's `refc.py` carries a sibling's refcv5 wiring.
* ⛔ **A `refcv3_arm` record is `UNKNOWN_SCOPE` to the criteria checker** — the eval would have
  **escaped its own completeness check.** Chaining `openloop_suite` gives **0 violations** against the
  incumbent record's **3**.
* ⛔ **`GOALS_AND_CLAIMS.md` is BLANK-LINE DOUBLED in its committed bulk (4,438 of 7,806 lines),
  breaking its tables.** Not repaired — not safe to reflow a file a sibling is appending to. **Queued
  for a quiet window**, same ruling as `M70`'s committer repair.

### 5. State

The run is at **step 33,350 / 40,284**, pace 3.98 s/step ⇒ **ETA ~08:20 UTC**. Goal 1's headline
artifacts — the **4,823-window** four-family read and the reel (**15 clips / 2,571 frames / 257.1 s =
4.95x the refcv3 reel**) — are gated on wall-clock and nothing else; the landing sequence is
preflighted to a single changed argument. ⭐ The watcher exits on **completion, failure AND channel
death**, and its failure detector was **proved to read non-zero** rather than assumed.

## M73. ⭐⭐ THE EGO-CHANNEL ESCALATION WAS A FACT, NOT A DECISION — no channel reads the future

### 1. I escalated without a default, and then found the answer in source

`M72` escalated *"which ego channels are admissible at inference"* as a PI ruling worth a **3.6x swing**
(0.3055 m vs 1.1137 m) — ⛔ **and I sent it up without a recommended default, which our own
program-report standard forbids.** Going back for one resolved most of the question.

**`refc_v3_train.py:2966` stamps the derivation into every run's `config.json`:**

> *"t0 = last OBSERVED frame; `a_long` = corpus ax (`actions[:,-1,1]`); `curvature` = tan(steer)/2.9
> (`actions[:,-1,0]`); `yaw_rate` = v0*curvature. **NO finite differencing, NO future read.**"*

| channel | what it is | future-derived? |
|---|---|---|
| `v0` | speed at t0 | **no** — already PI-ruled admissible (2026-09-02) |
| `a_long` | the **observed** longitudinal action at t0 | **no** |
| `curvature` | `tan(steer)/2.9` — the **observed** steering angle at t0 | **no** |
| `yaw_rate` | `v0 * curvature` — a product of two t0 quantities | **no** |
| `keep` | the validity mask | **no** |

⇒ ⭐⭐ **ALL FIVE are onboard PROPRIOCEPTION at t0.** Steering-wheel angle and longitudinal
acceleration are exactly what a real vehicle's own sensors report at cycle time. ⛔ **None of them
reads the future, and the trainer says so explicitly in the artifact rather than in someone's memory** —
which is the `M18`/`anchors.pt` units lesson applied correctly, for once, by the code that wrote it.

### 2. ⭐ What that does to the escalation

⛔ **The LEAK question is ANSWERED: there is none.** The vision-only rule's own test — *"does an input
at inference contain something the thing being measured also produces?"* — reads **no** on all five.
These are t0 measurements, not future-derived labels, and they are not the situation classifier's
output either.

⇒ **What remains is a much narrower POLICY question, and it is genuinely the PI's:** the binding rule
says *"for inference only vision"*, and the 2026-09-02 ruling carved out `v0`. **Whether the carve-out
extends to three more t0 proprioceptive channels is a decision about the deployable spec, not about
evidence.**

⭐ **RECOMMENDED DEFAULT: admit all five.** They are onboard-measurable at cycle time, carry no future
information, and rest on the **same principle** that admitted `v0` — a real car knows its own speed,
steering angle and longitudinal acceleration at the instant it plans. ⇒ On that default the headline is
**0.3055 m** and refcv4b's 35 % win stands as a deployable number.

⚠️ **One caveat that must travel with it:** `curvature = tan(steer)/2.9` uses a **CONSTANT WHEELBASE**
(`wheelbase_mode: const2p9`). That is a **modelling assumption, not a measurement** — harmless for a
single vehicle, but it hard-codes 2.9 m into a channel we are about to declare deployable.

### 3. ⭐ The transferable lesson, and it is about me

⛔ **I escalated a question I had not tried to answer.** The provenance was **stamped in the config the
run itself writes**, three lines of source away, and finding it converted *"a PI ruling worth a 3.6x
swing"* into *"a fact plus a narrow policy call with a recommended default."*
⇒ ⭐ **Before escalating, read the artifact that would settle it.** An escalation that could have been a
source read spends the PI's attention on work the orchestrator owed — and it is the mirror of the
night's other recurring error, **a blocker that dissolves on a second probe** (four times tonight).
⚠️ **`M72`'s escalation is not withdrawn — it is narrowed**, and it now carries the default our own
standard required in the first place.

## M74. ⛔⛔⛔ THE TURN GATE IS UNREACHABLE AND THE GROUND TRUTH FAILS IT — M64's named defect is a metric artefact, and the answer INVERTS

### 1. The measurement that breaks the night's headline

The eval's v1 lateral gate is **`|dyaw| > 0.15` rad**. Scored through it on the TURN_L-goal windows:

| arm | passes the turn gate |
|---|---|
| ⛔ **the HUMAN's own recorded driving** | **3 of 9** (median dyaw **0.0431**) |
| `ccos_argmax` | **6 of 9 — TWICE AS OFTEN AS THE HUMAN** |

The windows are near-stationary (**v0 median 1.40 m/s**), so the gate demands
`kappa >= 0.15/(v0*T) = 0.0536 1/m` — **a 19 m radius**. ⇒ **3/9 windows demand MORE than the shipped
`GOAL_KAPPA_TURN = 0.08`; 6/9 demand more than the corrected 0.02.** **A yaw gate calibrated for
junction turns is unreachable at walking pace.**

⇒ ⛔⛔ **`M64`'s named defect — "refav1's planner emits ZERO left turns" — and every turn-recall number
in `M57`, `M58`, `M64` and `M69`, is measured through a gate the GROUND TRUTH FAILS TWO-THIRDS OF THE
TIME.** ⚠️ The stream that produced `M64` is independently flagged as **likely sharing the defect**.

### 2. ⭐⭐ Under a reachable criterion the answer INVERTS — and it is a YES

| arm | curvature MAE on turn windows | reading |
|---|---|---|
| **`wk15`** | **0.04578** | ⭐ **best of every banked arm, −23 % vs the straight floor** |
| perfectly-straight floor | 0.05936 | the bar |
| `ccos_argmax` (recall 0.3636) | ⛔ **0.07935** | **WORSE THAN DRIVING IN A STRAIGHT LINE** |

⇒ ⭐⭐⭐ **THERE IS AN ARM THAT BOTH DRIVES ACCURATELY AND TURNS: `wk15`** — at ADE **0.9877** on turn
windows — **and it is the arm this entire campaign called "accurate because it does not turn."**
⛔ **`ccos_argmax` does not track the road; it OVER-TURNS, and the recall gate rewarded it for that.**

### 3. ⭐ CLASS `ARCH-D` — and it completes the night's family

⛔ **A METRIC WITH A THRESHOLD CARRIES THE REGIME IT WAS CALIBRATED FOR, AND THE GROUND TRUTH MUST BE
SCORED THROUGH THE SAME GATE BEFORE ANY ARM IS.** **One line would have caught it** — and nobody ran it,
through four separate packages.

⇒ This is the **fourth** member of tonight's scope family, and the sharpest:
* `M52` — a result carries its **MODEL**;
* `M61` — a floor carries its **RIG**;
* `M69` — a lever carries its **OPERATING POINT**;
* **`M74` — a threshold carries its REGIME.**
⭐ **The general form: score the ground truth through every gate before scoring an arm.** A gate the
human fails is not measuring skill; it is measuring the gate.

### 4. ⛔ "The penalty defeats the hierarchy" is WITHDRAWN — the defect is the GOAL VOCABULARY

The originating stream withdrew its own hypothesis: **the penalty does fight the goal — but the goal
commands ~5x the human's heading change, so fighting it moved the plan TOWARD the human.**

⇒ ⭐ **This converges with `M69`**, which measured the same thing from the corpus side (`0.08` = R 12.5 m
on a corpus curving at R 100–1000 m, where obeying is worse than driving straight). ⇒ ⛔ **The defect is
the GOAL VOCABULARY, not the cost** — which **qualifies `M54`/`M58`/`M64`'s "the cost is the defect."**
The cost does discard the commanded candidate; **the commanded candidate was wrong, and discarding it
was right.**

⚠️ **What survives from `M54`:** the *search* is still exonerated (it optimises better with a perfect
goal and drives worse), and the factorial's `W_KAPPA` → `turn_left` 0.0000 in all four cells is still a
true statement **about the recall metric** — which we now know is the broken one.

### 4b. ⭐⭐⭐ THE TWO STREAMS DO NOT CONFLICT — THEY COMPOSE, and the synthesis is the real result

The turn-asymmetry stream landed the **causal** half at seed 0, paired on the **same 75 windows**, one
variable (`W_KAPPA` 0 → 15.11245, same plan seed):

| | recall | delta | separated |
|---|---|---|---|
| **`turn_left`** | 0.3667 → **0.0000** | **−0.3667 [−0.6333, −0.1000]** | **YES** |
| `turn_right` | 0.5667 → 0.4333 | **−0.1333 [−0.2286, −0.0357]** | **YES** |

⭐ And the mechanism, in one cell — on windows where the head decoded the **correct** token:
`W_KAPPA = 0` tracks the goal at **full ±0.08 on 29 of 29** and turns **SYMMETRICALLY** (left 7/9 =
0.7778 vs right 16/20 = 0.8000); with the penalty, the 11 windows losing full curvature are **all 9
left plus 2 right**. ⛔ **At `W_KAPPA = 0` the L/R gap is NOT separated** ⇒ *without the penalty the
planner is not measurably asymmetric.*

⇒ ⭐⭐ **RECONCILED: the `ccos` arm clears the unreachable gate PRECISELY BECAUSE it emits the full
0.08 — a 12.5 m radius — while the human turns at median dyaw 0.0431.** ⇒ **BOTH are true:**
1. ⭐ **the penalty genuinely removes turns, asymmetrically, and that is now well-powered** (30/30, 6
   clusters, separated in both directions, with the speed confound re-refuted at n=30/30 — left is
   **0/17 SLOW and 0/13 FAST**, zero in *both* bands ⇒ **direction, not speed**);
2. ⛔ **what it removes is an OVER-TURN that tracks the road WORSE THAN A STRAIGHT LINE**
   (`ccos_argmax` curvature MAE **0.07935** vs the straight floor's 0.05936).

⇒ ⛔ **So the CAUSAL claim STANDS and is well-powered. What is withdrawn is the VALUE reading: that
suppressing those turns is a DEFECT.** `W_KAPPA` is suppressing a **wrong command**, and the recall
gate scored it as a loss because the gate rewards emitting 0.08.

⚠️ ⭐ **This is why "the cost discards the correct candidate" must be restated**: the cost discards the
**commanded** candidate, and the commanded candidate is **not the correct one.**

### 5. ⭐ And the seed floor is a property of the ARM, not only the rig — amends M61

MEASURED: a **22x spread** — **0.0047 / 0.0607 / 0.1035** — **across configurations of ONE model on ONE
panel.** ⇒ `M61` established floors are **rig**-dependent; this adds that they are **arm**-dependent too.
⛔ **A floor is therefore identified by FOUR things: the metric, the statistic, the rig, and the ARM.**
Quoting a remembered scalar is inadmissible — read `raw/seed_floor*.txt` for the exact configuration.

### 6. What must now be re-read

1. ⛔ **Every turn-recall claim in `M57`, `M58`, `M64`, `M69`** — re-score with the ground truth passed
   through the same gate, or replace recall with **curvature MAE against the straight-line floor**,
   which is reachable and which the human passes by construction.
2. ⛔ **`M64`'s "OUTCOME A, the asymmetry is real"** — the *separation* may survive (it was measured
   consistently across arms), but **"zero left turns" as a capability claim does not**, because the
   human scores 3/9 on the same gate.
3. ⭐ **`wk15` may now be quoted as a lateral fix — on curvature MAE, with the straight floor beside
   it** — which is the opposite of `M64` §P5's ruling, and for a better reason.

## M75. ⭐⭐⭐ THE REFAV1 LATERAL THREAD CLOSES — the asymmetry is real, the penalty causes it, and "zero left turns" means REDUCED MAGNITUDE

### 1. The causal result, complete and well-powered

All four arms of the 2×2 (`W_KAPPA` ∈ {0, 15.11245} × seed ∈ {0, 1}) landed. Paired, same 75 windows,
one variable:

| seed | `turn_left` | `turn_right` |
|---|---|---|
| s0 | 0.3667 → **0.0000** · **−0.3667 [−0.6333, −0.1000] SEPARATED** | −0.1333 separated |
| s1 | 0.3667 → **0.0000** · **−0.3667 [−0.6333, −0.1000] SEPARATED** | −0.0667 **not** separated |

⭐ **The left effect is IDENTICAL and separated at both seeds; the right effect is marginal.**

⭐⭐ **And the control is what converts this from an observation into a mechanism: at `W_KAPPA = 0`
there is NO measurable asymmetry** — pooled +0.2000 not separated, within-episode +0.1583 not
separated, and **retention 1.0000 / 1.0000 with the difference EXACTLY +0.0000**: the plan tracks the
goal at **full ±0.08 on every turn-goal window in both directions.**

⇒ ⛔ **The asymmetry is NOT a property of refav1's planner. It is a property of refav1's planner UNDER
A CURVATURE PENALTY.**

### 2. ⭐⭐⭐ THE RECONCILING FACT — and it closes `M74` and this stream into one account

**153/153 retained plans across all four arms curve the way their goal asked. THE SIGN IS NEVER WRONG;
THE FAILURE IS MAGNITUDE.**

⇒ That single control dissolves the apparent contradiction between this stream and `M74`:

1. `W_KAPPA` **reduces turn MAGNITUDE** — never direction (153/153);
2. the `|dyaw| > 0.15` gate reads reduced magnitude as **"zero turns"**;
3. ⛔ **the HUMAN's own magnitude is also below that gate** — she passes **3 of 9** (median dyaw 0.0431);
4. ⭐ and **curvature MAE says the reduced magnitude is CLOSER to the human**: `wk15` **0.04578** vs the
   straight floor's 0.05936 (−23 %), while the unpenalised `ccos_argmax` reads **0.07935 — worse than
   driving straight.**

⇒ ⭐⭐ **THE COMPLETE ACCOUNT: the asymmetry is real and the penalty causes it. Whether that is a
DEFECT depends on whether the reduced magnitude lands closer to or further from the human — and the
reachable metric says CLOSER.** ⛔ **"Zero left turns" is therefore a statement about the RECALL
METRIC, not about driving**, and it must never be quoted without the curvature MAE beside it.

⚠️ **What is NOT resolved by this**: the asymmetry itself. Even if reduced magnitude is *better*, the
penalty reduces it **2.75× harder on the left**, on a cost that is **bit-exactly sign-symmetric**.
**That remains unexplained and is the real open question.**

### 3. Where the defect is NOT — four candidates closed

* ⛔ **not the search** — the goal's `+0.08` candidate is **handed** to iCEM on every left-goal window
  and **loses on cost**;
* ⛔ **not the goal head** — where it decodes left correctly, the plan still produces **0/5**;
* ⛔ **not a direction-aware seed pool** — refuted; that candidate is already there;
* ⛔ **not another weight** — the ladder is closed, and the cost is **bit-exactly sign-symmetric**.

⇒ ⭐ **The named next instrument is cheap and is NOT another arm: a goal-latent COSINE LANDSCAPE SWEEP
over κ for matched left/right windows — a forward pass, not a planner run.** If the latent cost
surface is itself asymmetric in κ for matched geometry, that is the answer; if it is symmetric, the
asymmetry lives in the rollout.

### 4. The methodological work that made the answer trustworthy

* ⭐ **A power target derived BEFORE looking**: `2/n <= 0.0750` ⇒ n ≥ 27 per direction, ≥ 5 clusters.
  ⛔ The banked panel's 11/8 resolved only to **0.0909 / 0.1250 — coarser than the floor it was being
  compared against.** **It could never have decided anything, and nobody had checked.**
* ⛔ **The banked panel was structurally UNATTRIBUTABLE, not merely thin** — its two turn-goal strata
  occupied **disjoint episodes**, making direction and episode the same variable. New retraction class
  with a durable fix.
* ⭐ **Two `ccos` arms read EXACTLY identical** — the signature of a duplicated arm (`M40`). Verified
  genuine three ways: their **trajectories differ by 7.75 m on 14/75 windows** and simply never cross a
  turn-label boundary. **The suspicious identity was real and benign, and it was checked rather than
  assumed.**
* ⭐ **A refutation re-run because its evidence was inadmissible**: the speed confound had been refuted
  on the panel later proved degenerate. Re-run at n=30/30 it **holds** — left is **0/17 slow and 0/13
  fast**, zero in *both* bands ⇒ **direction, not speed.**

### 5. ⚠️ The remaining ceiling is the corpus

**8 episodes, 4 carrying both directions, one flipping sign.** Raising n from 11/8 to 30/30 **did not
raise the cluster count**, which is what the decision estimator consumes. ⇒ **Sharpening further needs
more EPISODES, not more windows or more compute** — the narrowed corpus request from `M64` §4, now
confirmed on a completed 2×2 rather than a partial read.


## M76. ⛔ SESSION API LIMIT HIT ~02:10 BERLIN — read this FIRST after the 04:20 reset

### 1. What happened

Four agents terminated with **HTTP 429, session limit, resets 04:20 Europe/Berlin**: the refav1
make-it-drive stream, the refav1 gap-closing stream, the refav1 longitudinal stream, and an E-DDA-4
probe. ⭐ **No GPU work was affected** — every arm is a host-side process on Thor / the A40 / the
dev box and does not touch the API, and the host-side finishers (`finalize_thor.sh`,
`finalize_lon.sh`, `queueK.sh`, `queueLON4.sh`) are shell scripts that keep running.

⛔ **What died is the AGENT-SIDE READING of results that land between ~02:10 and 04:20.** The arms
will finish and write their records; nothing will interpret them until the loop resumes.

### 2. ⭐ READ THESE FIRST after the reset — in this order

1. **`seambase` / `seamon`** (dev box, `W_JERK = 0.02` LIVE, ETA was ~50 min from 02:00) — the
   **first arm the cost-pricing instrument ever APPROVED** (`M65`). ⛔ **A null here is now a REAL
   answer about the cost side**, not arithmetic, because the weight is live. Read via
   `finalize_lon.sh`; committed outcomes are in `raw/queueLON4.sh`'s header.
2. **`best_seed1`** (dev box) — the inference-seed replicate of `best`'s `kamm_over 0.0000`.
   ⛔ **`combined`'s identical zero DIED at seed 1** (`M56`), so this is the test that decides whether
   `best`'s safety number is a property or a draw. Both outcomes pre-committed in `raw/queueK.sh`.
3. **Thor's `goal_reach_s` panel** (`T_grsCTL` / `T_grs1` / `T_grs8` / `T_grs1_s1`) — ⛔ **READ
   `T_grsCTL` FIRST: if it does not reproduce `T_lonshift`, every other arm in that panel is VOID.**
   `bash /home/nvidia/refav1_lon/finalize_thor.sh` lands it.
4. **`A4a_gk_kt02` / `A4b_kt02`** (Thor) — the goal-conditioned lateral cost at
   `--goal-kappa-turn 0.02`. Its refutation branch is committed: **if it fails, the defect is upstream
   of the cost and the next lever is `--lat-logit-bias`, not another cost term.**
5. **refcv4b** — step ~33,350/40,284 at 00:38 UTC, **ETA ~08:20 UTC**, watcher armed and proven to
   fire on completion, failure AND channel death. Its landing sequence is preflighted to a single
   changed argument.

### 3. ⚠️ Read these BEFORE quoting anything that lands

* ⛔ **`M74`/`M75`: the `|dyaw| > 0.15` turn gate is UNREACHABLE and the HUMAN fails it 3/9.** Any
  turn-RECALL number is a statement about the metric. Quote **curvature MAE with the straight-line
  floor beside it** instead.
* ⛔ **`M70`: the shared committer is running WITHOUT its name-preservation guards.** Verify every
  commit by a **content marker inside the committed blob** with a non-zero control. Three commits
  tonight exited 0 and landed nothing.
* ⛔ **A floor is identified by FOUR things** — metric, statistic, **rig** (`M61`) and **arm**
  (22× spread on one model, one panel). Never quote a remembered scalar.
* ⭐ **`M73`: the ego-channel question is a POLICY call with a recommended default (admit all five)**,
  not an open evidence question — no channel reads the future.

### 4. Do NOT do these

⛔ **Do not spawn agents before 04:20** — they will 429 and burn budget.
⛔ **Do not repair `mktree_commit.py` or reflow `GOALS_AND_CLAIMS.md`** while streams are live
(`M70`, `M72`) — both are queued for a genuinely quiet window.
⛔ **Do not launch refcv5 training** until the A40 frees AND the corpus-flag preconditions are met
(`--agents oracle` is the first rung and needs no detector; `--sampler ddim` needs an `--anchor-file`
with declared `control_units`). ⚠️ **Do NOT set `--sel-refined` on the first arm** — measured
**0.0259 m separated WORSE** (`M72`): the ranking head was never trained.


## M77. ⭐⭐⭐ `best`'s ZERO-VIOLATION RESULT REPLICATES — the programme's first replicated, NON-VACUOUS safety result

### 1. The measurement, with every control passing

Re-run of `feas_audit.py` over the banked dumps, `FEAS_VMIN=2`, n=27 windows. ⭐ **Two controls, both
pass:** the ground-truth path `g` reads `kamm_over` **0.0000** on every arm (it must — a real vehicle's
recorded motion is feasible by construction), and the **known-value control reproduces `M56` exactly**.

| arm | `kamm_over` | max \|kappa\| | `peak_g` max |
|---|---|---|---|
| **`best`** (seed 0) | **0.0000** | 0.0800 | **0.332** |
| ⭐ **`best_seed1`** | ⭐ **0.0000** | 0.0800 | **0.332** |
| `combined` | 0.0000 | 0.1505 | 0.618 |
| ⛔ `combined_seed1` | ⛔ **0.0741** | 0.1672 | **0.702 — over the mu = 0.7 circle** |
| **`g` — the human** | **0.0000** | 0.1701 | 0.373 |
| `ha0_ext` floor | 0.1852 | 0.7672 | 1.436 |
| `ol` (T0 contract) | 0.1111 | 0.4400 | 1.016 |

### 2. ⭐⭐ What this establishes, and it CORRECTS M56

⭐ **`best`'s zero HOLDS at a second inference seed** — 0.0000 at both, `peak_g` max **0.332 at both**
(mean 0.082 / 0.081), max|kappa| **0.0800 at both**. ⛔ **`M56` concluded *"no arm in this programme has
a replicated zero friction-circle violation rate."* THAT IS NOW FALSE: `best` has one.**

⭐ **And it is NOT VACUOUS by `M58`'s own gate.** That gate failed `bestlad` on **max|kappa| 0.0049 <
0.02**; `best` reads **0.0800 — 4x the threshold** — and runs at **89 % of the human's peak lateral
load** (0.332 vs 0.373). ⇒ **It is not a stopped arm buying safety by declining to act**, which is the
degenerate solution this programme has caught three separate times tonight.

⭐⭐ **And unlike the turn gate (`M74`), THIS metric is one the GROUND TRUTH PASSES** — `g` reads
0.0000 by construction. ⇒ **`kamm_over_rate` is a sound criterion**, and `best` matches the human on it
at both seeds. **The `ARCH-D` test was applied before quoting the number, and it passes.**

### 3. Why `best` holds where `combined` died — the mechanism

`combined` (`W_KAPPA = 0` + cap + ladder) reaches max|kappa| **0.1505**, nearly **2x the commanded
0.08**, and at seed 1 overshoots to 0.1672 with `peak_g` **0.702 — outside the circle**. `best`
(`W_KAPPA = 15.11245` + cap) reads max|kappa| **0.0800 at both seeds — exactly the goal command.**

⇒ ⭐ **The curvature penalty is what makes the safety result STABLE**: it holds the plan at the
commanded magnitude instead of letting iCEM overshoot into the friction circle. ⇒ this is the same
mechanism `M75` measured from the other side (*the sign is never wrong; the failure is magnitude*) —
**and here the magnitude control is exactly what buys the replicated zero.**

⚠️ **Not claimed:** that `best` is therefore the arm to ship. It still loses the longitudinal family,
and `M74` shows its curvature magnitude is *below* the human's on turn windows. **Safe and stable is
established; complete is not.**

### 4. ⚠️ My own invocation was wrong twice, and the control caught both

The first run read **ABSENT on all six arms including the control**. Cause: `audit(dumpdir, label)` but
`__main__` passes `argv[1], argv[2]` — ⛔ **I passed label-then-dir, the reverse** — and `VMIN` reads
the env var **`FEAS_VMIN`** (default 0) where the banked run used **2.0**.
⇒ ⭐ **Had I omitted the `combined` control I would have read six ABSENTs and concluded the dumps were
bad.** *A probe whose known-value control fails is inadmissible — and it is the cheapest possible
protection against reading one's own argument order as a finding.*


## M78. ⭐⭐ THE JERK SEAM IS FIRST-ORDER WHEN ITS COEFFICIENT IS LIVE — the pricing instrument's approval was right

### 1. The pair, one variable, `W_JERK = 0.02` LIVE

| arm | `W_JERK` | seam | ADE | LON spd | LON accel | LON along | LAT curv | LAT head |
|---|---|---|---|---|---|---|---|---|
| `seambase` | 0.02 | off | 0.9085 | 0.8183 | 0.8871 | 0.6769 | 0.03500 | 15.125 |
| `seamon` | 0.02 | **a0** | 0.8148 | 0.5823 | 0.6848 | 0.5386 | 0.03792 | 19.830 |
| `lonshift` | 0.0 | off | 0.7868 | 0.5657 | 0.6526 | 0.5232 | 0.04116 | 16.856 |
| **`loncomb3`** | 0.02 | a0 | **0.7739** | **0.4665** | **0.5698** | **0.4159** | 0.04117 | 16.501 |

**SEAM EFFECT (`seamon` − `seambase`), the clean one-variable contrast:**
**LON speed −0.2360 · accel −0.2023 · along −0.1383 · ADE −0.0937**, against
**LAT heading +4.705 deg · curvature +0.0029.**

### 2. ⭐⭐ The pricing instrument was RIGHT, and that closes its validation

`M65` approved this pair on a 0-GPU cost pricing: the seam delta was **0.494** of everything it trades
against on `wk15` and **4.05** on `lonshift` — *"first-order, the pair earns its GPU."* ⇒ **It
delivered a first-order longitudinal effect.** ⭐ The same instrument **declined D4** (~4.6 % of the
gap) and **refuted D5**; this is its first approval and it was correct. ⇒ **`M65`'s claim that the
instrument is DISCRIMINATING rather than merely conservative is now established on both sides.**

### 3. ⭐ It closes the inert-arm lesson POSITIVELY

The seam read **`+0.0000` on all eleven metrics at `W_JERK = 0.0`**, confirmed on two rigs. ⛔ That was
never a null about the seam — it was a null about a term switched off. **With the coefficient live the
same lever is first-order.** ⇒ *A lever multiplied by a zero coefficient is not a null about the
lever* is now demonstrated in **both** directions, which is what makes it a rule rather than an excuse.

### 4. ⚠️ What is NOT claimed

* ⛔ **These are POINT ESTIMATES. No paired interval, no seed replicate.** `M61`/`M56` bind: a delta
  without its interval is not a separated result, and the floor is **arm- and rig-dependent**. **The
  paired bootstrap and a seed replicate are owed before any of this is quoted as a lever effect.**
* ⚠️ **The seam trades LON for LAT** — heading +4.705 deg is a real cost and the four families are
  **never pooled**. ⚠️ Note it does NOT reproduce in `loncomb3` vs `lonshift` (16.501 vs 16.856), but
  that contrast moves **two** variables (`W_JERK` and the seam) and is therefore **not attributable**.
* ⭐ `loncomb3`'s committed branch — *"if `loncomb3` is worse than `lonshift`, report it; do not retune
  the weight to rescue it"* — **did not fire**: it is better on every longitudinal metric.


## M79. ⭐⭐⭐ THE `W_KAPPA` SWEEP, READ ON THE SOUND METRIC — a clean interior optimum, and refav1 DOES track the road

### 1. The sweep, re-read on curvature MAE instead of turn recall

`A2` was designed to find where turn **RECALL** dies. `M74` showed recall is measured through a gate
the **human fails 3/9**, so the sweep is re-read on **curvature MAE**, which the human passes by
construction and whose floor is a **perfectly straight plan**.

| arm | `W_KAPPA` | ADE | **curv MAE** | head MAE | turn_L (broken metric) |
|---|---|---|---|---|---|
| `ccos_argmax` | 0 | 1.3272 | ⛔ **0.05537** | 23.458 | 0.3636 |
| `wk1` | 1 | 0.9388 | 0.03941 | 20.373 | 0.3636 |
| `wk3` | 3 | 0.9301 | 0.03874 | 20.195 | 0.3636 |
| `wk7` | 7 | 0.8935 | 0.03352 | 19.084 | 0.2727 |
| ⭐ **`wk15`** | 15.11245 | **0.8934** | ⭐ **0.03098** | **15.270** | 0.0000 |
| `wk151` | 151.1245 | 0.9084 | 0.03802 | 15.379 | 0.0000 |
| **FLOOR `ha0`** (perfectly straight) | — | — | **0.040083** | 20.137 | — |

### 2. ⭐⭐ What it establishes

* ⛔ **The UNPENALISED arm is WORSE THAN DRIVING STRAIGHT** — 0.05537 vs the floor's 0.040083.
  **Independent confirmation of `M74` from a six-point sweep** rather than a two-arm contrast.
* ⭐ **`wk1` already crosses the floor** (0.03941): the penalty buys road-tracking from its very
  first unit.
* ⭐⭐ **A CLEAN INTERIOR OPTIMUM AT `W_KAPPA` ~ 15** — monotone improvement 0 → 1 → 3 → 7 → 15 on
  **both** curvature and heading, then **degradation at 151** (0.03098 → 0.03802). ⇒ **not a
  saturating knob and not a cliff: a real optimum**, which is what a well-posed cost term should show.
* ⭐⭐⭐ **`wk15` beats the straight floor by 23 %** (0.03098 vs 0.040083) — **the best lateral
  tracking in the programme.**

### 3. ⚠️ `wk7` vs `wk15` — and the answer is NOT what the recall column suggests

`wk7` is the only arm that keeps a **non-zero turn_left recall (0.2727)** while matching `wk15`'s ADE
(0.8935 vs 0.8934). ⛔ **But recall is the broken metric.** On every SOUND metric `wk15` wins:
curvature **0.03098 vs 0.03352**, heading **15.270 vs 19.084**, ADE tied.
⇒ ⭐ **`wk15` is the better arm, and `wk7`'s only advantage is on the gate the human fails.**
**Reading the sweep on recall would have selected the worse arm** — which is precisely the failure
`ARCH-D` describes, caught here before it cost anything.

### 4. ⭐ The answer to the PI's central question

**"Is there an arm that both drives accurately and tracks the road?" — YES: `wk15`.**
ADE **0.8934** (best in the sweep) with curvature MAE **23 % better than a perfectly straight plan**,
and — with `M77` — a **replicated, non-vacuous zero friction-circle violation rate** at 89 % of the
human's peak lateral load.
⛔ **What it still is NOT:** better than `ha0_ext` on the longitudinal family, and `M66` warns that
`ha0_ext` is itself infeasible on 18.5 % of windows, so that comparison needs its feasibility rate
stated. **Lateral and safety are established; longitudinal is the open blocker.**


## M80. ⛔⛔ v7f's L1 "NO COLLAPSE" PASS IS ARM-SUBSTITUTED — the PI's doubt was correct, and the mechanism is new

### 1. The measurement

**MEASURED, identical across THREE independent panels** (`leadsplit.json`, `gatec_panel.json`,
`gateb_panel.json` in `…/2026-08-19-simwam-analysis/raw/`):

| arm | step | participation_val | heldout24 | vs the quoted 8.56 bar |
|---|---|---|---|---|
| **`rdw8p30k`** | 30000 | **25.583** | 26.965 | **above** |
| `rdw8s30k` | 30000 | 18.075 | 18.845 | above |
| `scale1` | 30000 | 8.541 | 7.537 | val above, held24 below |
| ⛔ **`splitp30k`** | 30000 | **6.383** | 7.629 | ⛔ **BELOW** |
| `champ30k` | 30000 | 6.499 | 7.029 | below |

### 2. ⛔⛔ The substitution, on one two-row table

`V7_LAUNCH_GATE.md:237-238` scores L1 and L2 on the **same table** with **DIFFERENT ARMS**:

* **L1** *"no collapse, >= 8.56"* is ✅ on **3.80/3.62 → 25.58/26.96** — that is `rdw8` → **`rdw8p30k`**;
* **L2** *"the latent CARRIES the environment"* is ✅ on **`splitp30k`** (+0.1220 vs DINOv3 +0.0998);
* **L3** names **`splitp30k`** as the deliberate-regression arm.

⇒ ⛔ **One arm clears the collapse bar. A DIFFERENT arm carries the decodability story. And that second
arm would NOT clear the bar the first is credited with clearing.**

⚠️ **The number is not hidden** — 6.38 / 7.63 sits un-bolded beside the bolded 25.58/26.96 in
`MODEL_REGISTRY.md:4332` and `GOALS_AND_CLAIMS.md:1388`. ⛔ **It simply never appears on the gate
line.** The row reads as one coherent pass while resting on two objects.

### 3. ⭐ The honest status of L1, and the PI was right

Combined with the separately established fact that **the 8.56 bar is itself unreproducible** — its own
artifacts read **20.23** n-matched and **5.76** on the same 12 clips, and
`h_rank16_floor_valclips.json` says verbatim *"do not fail any arm on it"* — the honest status is:

⇒ ⛔ **THE BAR IS UNSETTLED **AND** THE PASS IS NOT ATTRIBUTABLE TO THE ARM THE REST OF THE PANEL RESTS
ON. That is not a clean ✅.**

⭐⭐ **The PI flagged non-collapse specifically** — *"non-collapse (was marked as solved)"* — and
**his doubt is now vindicated with a mechanism**, not merely a caveat. ⭐ Note what it is NOT: the
numbers are all real and all banked. **Nothing was fabricated; the arms were swapped between rows.**

### 4. ⭐⭐ NEW CLASS — A GATE ROW CARRIES ITS ARM

⛔ **Substituting arms across rows of one gate manufactures a coherent-looking pass out of true
numbers.** Each row is individually honest; the panel as a whole asserts something none of them does.

⇒ This is the same family as tonight's scope errors, with the scope being the **ROW's SUBJECT**:
* a result carries its **MODEL** (`M52`) · a floor carries its **RIG and ARM** (`M61`, `M79`) ·
  a lever carries its **OPERATING POINT** (`M69`) · a threshold carries its **REGIME** (`M74`) ·
  ⭐ **a GATE ROW carries its ARM (`M80`).**

⇒ ⛔ **DURABLE FIX: every gate row prints the ARM NAME that supplies it, and a panel whose rows do not
share an arm must say so on its face.** A cross-row ✅ is only admissible when one arm clears every row
— otherwise the gate is scoring a chimera. **One column would have caught this.**

### 5. Consequence for v7f

⛔ **L1 must be re-scored on the arm that carries L2/L3** (`splitp30k`, 6.383) **or the panel must
declare that its rows use different arms.** Until then *"collapse is solved"* is not quotable.
⭐ This does **not** touch the earlier finding that the three T1-read arms sit at **0.90–0.98 of
`rdw8p30k`** and are therefore **not collapsed** — that reading is arm-explicit and stands.
