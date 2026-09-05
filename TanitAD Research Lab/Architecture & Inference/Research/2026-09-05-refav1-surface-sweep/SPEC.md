# SPEC — H-REFAV1-SURFACE-1, the cost-surface weight sweep and the goal-source discriminator

**PRE-REGISTERED 2026-09-05, BEFORE any sweep arithmetic and before any GPU arm.** The
SELECT/SCORE split (`raw/split.json`) was generated and banked in the same commit as this file,
so no weight can be chosen on the windows it is reported on.

```yaml
hypothesis: H-REFAV1-SURFACE-1     # exists in GOALS_AND_CLAIMS.md (verified 2026-09-05)
one_variable: the cost weight triple (W_JERK, W_KAPPA, W_VEND) — one coordinate per arm
held_constant: [checkpoint, cost_metric=ccos, corpus, window grid, stride 40, K=10 @0.2s,
                action-units kappa, --no-navshuf, plan {n_samples 300, n_iters 30,
                n_elites 30, seed 0}, lead block, estimator, n_boot 2000]
success: "on the SCORE half, some swept setting's four families beat `ha0_ext`
          paired-separated with the echo gate holding"
failure: "no setting in the swept range beats `ha0_ext` on the four families"
controls: [constant_only, trivial_floor, deliberate_regression, known_value]
splits: {select: "70 of the 141 episode clusters, rng(0) — weights are chosen here",
         score:  "the other 71 — scored, NEVER tuned on"}
```

## 0. What is being tested, and what is NOT

**Tested:** the PLANNER'S COST SURFACE over a frozen world model. **Not tested:** the world
model, which is separately banked as sound (TURN_L +0.502 [+0.383, +0.579], TURN_R −0.387
[−0.498, −0.267], separated). No verdict here may be restated as a statement about the WM.

⚠️ **Why the skill's G-RANK and G-DECODE gates do not apply, stated rather than skipped.** Both
are representation gates. This change touches **no weights of the model and no representation** —
it re-scores candidate action sequences under a different cost. The applicable gate is **G-DRIVE**:
T1, four metric families, paired episode-cluster bootstrap. Running a participation number here
would be a number without a matched reference, which the skill forbids.

## 1. The two stages, and why the screen is not the result

| stage | cost | what it decides | what it may NOT conclude |
|---|---|---|---|
| **A — the exhaustive weight screen** | **ZERO GPU** | which weight settings are worth a real arm | anything about the four families |
| **B — real arms** | GPU | the committed outcomes | — |

**Stage A is exact arithmetic on banked fields, not a simulation.** `raw/box_panel_282.json`
(Thor) holds, per window, for each of **26 candidates** (`cv`, `decel_1.5`, ±κ ∈ {0.2, 0.1, 0.05,
0.02, 0.01}, ±a ∈ {4, 2, 1.5, 0.5}, `turnL/R_slow`, `rampL/R`, **`seed0`**, **`proposal`**): the
goal term under each metric, and the raw penalties `jerk_raw`, `kap_raw`. The planner's own closure
`total = goal + W_JERK·jerk_raw + W_KAPPA·kap_raw` reproduces to **1.6e-08**, so
`argmin_n [goal_ccos(n) + W_JERK·jerk_raw(n) + W_KAPPA·kap_raw(n)]` is the planner's own
comparison at any weight triple, computed exactly, for free.

⛔ **The honest limit, stated in advance:** the box is 26 designed candidates; the real planner
searches **300 samples × 30 iterations**. Stage A therefore answers *"at these weights, which
member of the injected/seeded set wins"* — and the banked seed-channel result is why that is
informative (**every** emitted plan on this model is an injected baseline or the seed; **0**
searched plans over 282 windows). It is a SCREEN. **No committed outcome is decided by Stage A.**

## 2. Stage A — the grid

`W_VEND` is never charged on this eval path (`target_speed=None`), so it is held at its compensated
value and not swept; this is stated rather than silently dropped. One variable at a time, log
spaced, around the compensated triple `(12.859430, 32.148575, 64.297150)`:

| arm | variable | values (× the compensated value) |
|---|---|---|
| `S-JERK-k` | `W_JERK` | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 |
| `S-KAPPA-k` | `W_KAPPA` | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 |
| `S-BOTH-k` | both, jointly | the 6 × 6 product (free; reported as a map, not as 36 arms) |
| `S-ZERO` | both = 0 | the **pure goal** surface: what the goal term alone would choose |
| `S-REG` ⛔ | `W_JERK` × 1e4 | **DELIBERATE REGRESSION — must select `cv` on ~100 % of windows** |

**Screen readouts, per setting** (SELECT half only for choosing):
1. fraction of windows whose argmin is a **non-trivial** candidate (not `cv`, not `decel_1.5`);
2. fraction with **κ ≠ 0**, and on the **38 TURN-goal windows** the fraction whose κ sign agrees
   with the decoded goal;
3. **agreement with GT at the action level**: sign(κ_selected) vs sign(κ̄_GT), and |a_selected − ā_GT|,
   where the GT kinematics come from `four_families` on the banked GT trajectory. ⚠️ Labelled an
   ACTION-LEVEL proxy, never reported as the lateral or longitudinal family.

**Selection rule, committed now:** the setting taken to Stage B is the one maximising **(2)** — κ-sign
agreement with GT on the SELECT half's turning windows — subject to **(1) ≥ 0.10** so a setting that
merely never turns cannot win by default. Ties break toward the setting closest to the compensated
triple. ⛔ Computed on SELECT only.

## 3. Stage B — the real arms

Every arm: `taniteval/tools/refav1_arm.py`, same checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt`
step 21,109, `--cost-metric ccos`, `--window-stride 40 --no-navshuf --action-units kappa`, full
141-episode grid so it is paired-comparable with every banked arm; **the primary verdict is read on
the SCORE half**, the full grid reported as secondary.

| arm | one variable | status |
|---|---|---|
| **B1 `sweep-best`** | the weight triple from Stage A | the headline |
| **B2 `oracle-goal`** | `--with-oracle-goal-arm` at the SAME weights as B1 | ⛔ **T0 ORACLE — the true future field as the planning goal. An UPPER BOUND, never a deployable arm.** |
| **B3 `chord_shipped`** ⛔ | the uncentred goal form at shipped weights | **DELIBERATE REGRESSION**, already running on Thor (launched by the predecessor's chain) |

⭐ **Why B2 is in the spec although the PI did not ask for it.** The banked goal provenance is
`{'tactical_imagined': 1.0}` — **every** goal is the tactical head's imagination. If the goal itself
is wrong, no weight fixes it. B2 replaces only the goal source and holds the weights, so it is the
cheapest discriminator between *"re-balance the surface"* and *"re-form the goal"*. It costs no
extra run: `--with-oracle-goal-arm` rolls `cl_oraclegoal` beside `cl` in the same pass.

## 4. Controls — every one must read a known value

| control | must read | where |
|---|---|---|
| **known-value** | a dump against ITSELF: exactly 0.0000 [0, 0] on every metric | `tools/paired_delta_refav1.py`, PASSED in phase 1 |
| **constant-only** | the panel's zero-model control (all candidates = the cv field) reads ptp **0.0 exactly** | `box_panel_282.json` `controls.zero_model_ptp_*` |
| **trivial floor** | `ha`, `ha0`, `ha0_ext` — a planner that does not beat the trivial kinematic controls has added nothing. This is the raw-input floor of the skill's rule, in planner form | banked, bit-identical across dumps |
| **deliberate regression** | `S-REG` must select `cv` on ~100 % of windows; `chord` at shipped weights must NOT improve | Stage A / B3 |
| **printed n and d** | every table prints n_windows AND n_episode_clusters | enforced by the tools |

## 5. Committed outcomes — written before the data, all three

1. ⭐ **A weight setting whose four families beat `ha0_ext` paired-separated on the SCORE half,
   with the echo gate holding** ⇒ **the surface IS the lever**; refav1's planner is repairable by
   re-weighting; report the setting and hand it to the Master Mind as a default candidate.
2. ⚠️ **No setting in the swept range beats `ha0_ext`, WHILE the T0 ORACLE-GOAL arm does** ⇒ **the
   GOAL, not the weights, is the binding constraint.** The next work is the goal source (the
   tactical head's imagined goal), not the cost surface. `H-REFAV1-SURFACE-1` is refuted and a
   successor hypothesis on the goal source is registered.
3. ⛔ **Neither** ⇒ **the iCEM planner over this world model is not repairable by re-weighting or
   re-goaling, and the line closes.** This is stated plainly, not softened: it is a real result for
   the programme's thesis — it says the hierarchy's failure is in the PLANNER, while the world
   model underneath it is sound.

⚠️ **A fourth outcome is possible and is pre-declared so it cannot be reported as (1):** a setting
that beats `ha0_ext` on SOME families and loses on others. That is **NOT** outcome (1). It is
reported per family with no composite, and the verdict is "no setting beats the floor on all four".

## 6. What would make this SPEC's own result inadmissible

* a weight chosen on the SCORE half (guarded by `raw/split.json`, banked before the screen);
* an arm differing from B1 in more than the one variable (guarded by diffing the launch commands
  and the `cost` block each dump stamps);
* a family reported as pooled, or ADE alone (binding rule);
* a number quoted without its tier — `cl` is **T1**, `cl_oraclegoal` is **T0**, `ol` is **T0**;
* the STRATEGIC family silently omitted rather than reported UNAVAILABLE with its reason and n.
