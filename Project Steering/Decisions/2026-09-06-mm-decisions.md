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
