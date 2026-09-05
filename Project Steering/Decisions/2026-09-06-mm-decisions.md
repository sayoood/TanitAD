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
