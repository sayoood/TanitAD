# SPEC — C-ENV-1: the constraint channel (two-sided speed envelope + clearance)

**Date** 2026-09-06 (Europe/Berlin) · **Stream** Architecture & Inference ·
**Branch** `agent/arch-inf-20260803` · **GPU-days spent: 0. No model was trained.
No pod was touched. The A40's refcv5 run was not approached.**

**Answers PI instructions (1), (8) and (10) of the refcv4b video review.**

⛔ **WHAT THIS SPEC IS.** §1–§4 pre-register a MODEL arm that has **not run**.
§5 records the INSTRUMENT characterisation that was executed today on banked
data, whose purpose is to establish that the gate can fail before any arm is
allowed to pass it. Nothing in §5 is a claim about any TanitAD model.

---

## 0. The pre-registered block

```yaml
hypothesis: H-CONSTRAINT-2SIDED-1     # GOALS_AND_CLAIMS.md, appended 2026-09-06
one_variable: constraint_loss_enabled  # the ONLY difference between arms
held_constant: [corpus, seed, steps, batch, window, lr, anchors, labels,
                encoder, decoder_width, anchor_control_units]
success: >
  On the OVER side, paired episode-cluster bootstrap (taniteval/ci.py, unit =
  episode, B=2000): frac_over_ceiling DECREASES vs the no-constraint arm, CI
  excludes zero, AND |delta| exceeds the arm's own INFERENCE-seed replicate
  floor. Reported jointly with ade_m: a constraint arm that buys compliance by
  destroying the trajectory has FAILED.
failure: >
  Not separated (NULL), or frac_over_ceiling rises (REGRESSION), or ade_m
  degrades beyond its replicate floor while compliance improves (TRADED).
controls: [constant_only, raw_input_floor, deliberate_regression, replicate_seed]
splits: {fit: "B1 v7.2 TRAIN 4,572", val: "carved from TRAIN only",
         test: "B1 v7.2 EVAL, scored, never tuned on"}
```

⛔ **The UNDER side is deliberately NOT in `success`.** §5.4 measures that it is
not scoreable against any ceiling available today. Committing a criterion to a
degenerate statistic is how a gate gets passed by an artifact.

---

## 1. The question, and why the obvious design is inadmissible

The PI asked for a MAX SPEED the model adapts to, approaching it "when the
situation allows". The obvious design — mint a max-speed label, feed it, score
adherence — is **inadmissible**, and not by argument:

| candidate ceiling | layer | status |
|---|---|---|
| `g_tac.goals.SPEED_BAND.v_hi_ms` | v7.2 augmented | ⛔ `[min,max]` of the EGO'S OWN future speed over 2–6 s. Zero horizon guard. |
| `a_tac.lon_args.v_target_ms` | v7.2 augmented | ⛔ `round(m.v_min,2)`, the MINIMUM over the plan window. Ego-future. |
| `vtarget_guarded` | lake | ⛔ ADMISSIBLE as a label, **INADMISSIBLE supplied** (ΔR² +0.0996). |
| `strata.road_class` prior | v7.2 augmented | ⛔ DEFINED by a speed threshold ⇒ circular. |
| posted sign limit | — | ⛔ Does not exist. F-14 blocker; sign-text gate 0/31. |
| map / lane graph | raw corpus | ⛔ Does not exist (five probes; the card says so). |

⭐ **THE DESIGN CONSEQUENCE, and it is the whole idea: CONSTRAINTS ARE SCORED ON
THE OUTPUT, NEVER SUPPLIED AT THE INPUT.** Both ceilings are computed from the
model's own proposed trajectory and from perception it must already produce, so
**no leak is possible by construction** — an input that does not exist cannot
echo. This sidesteps the entire admissibility problem rather than negotiating
with it.

* **kinematic ceiling** `v ≤ sqrt(a_lat_max / |κ|)` — κ from the PROPOSED path.
* **clearance ceiling** `v ≤ (gap − d0)/τ` — `gap` from the in-corridor lead.

---

## 2. Arms — one variable, both outcomes committed

| arm | difference | committed reading |
|---|---|---|
| `base` | none (the incumbent) | the reference |
| `cons` | `+ constraint loss` | **PASS** if §0 `success`; **FAIL** if `failure` |
| `cons_replicate` | `cons` flags, **inference seed changed** | the noise floor `cons` must beat |
| ⛔ `cons_regress` | constraint loss with the sign **flipped** | **MUST FAIL.** If the gate cannot fail an arm trained to violate, a PASS is meaningless |

⛔ `oracle_sel` / `anchor_acc` / `sel_agrees_oracle` are INVALID on refcv4b-family
arms and are not reported. The `|dyaw| > 0.15` gate is never used.

---

## 3. Controls that must read known values

| control | must read | why |
|---|---|---|
| `constant_only` | frac_over **exactly 0.0** | a constant-speed arm below every ceiling cannot violate; anything else is an instrument bug |
| `raw_input_floor` | GT trajectory's own rate | a model worse than the human it imitates has added nothing |
| `paired(a, a)` | **exactly 0.0000** | a paired bootstrap of an arm against itself |
| `deliberate_regression` | frac_over **rises** | mutation, not inspection |
| `manoeuvre_rate` | reported **beside every number** | ⛔ MEASURED 2026-09-06: a friction-circle zero was bought by a `turn_left` recall of exactly 0.0000. A safety zero from an arm that declines the manoeuvre is not a safety result. |

---

## 4. Four families, never pooled · T-tier

LONGITUDINAL (envelope over + under, lead gap, TTC) · LATERAL (heading,
curvature, yaw-rate, cross-track) · TACTICAL (manoeuvre decision + anchor) ·
STRATEGIC (**UNAVAILABLE, n = 0** — no map, no route label; stated per family
with its reason, never silently dropped). Estimator: paired episode-cluster
bootstrap. ⛔ A one-seed separated CI is **necessary, not sufficient**
(`H-ESTIM-SEED-1`, 6/42 = 14.3 % false-positive rate); a replicate arm is
mandatory, and on a sampling planner it must vary the **inference** seed
(refav1 floor ≈ 0.30 m ADE, curvature 0.00200).

---

## 5. INSTRUMENT CHARACTERISATION — executed today, 0 GPU

⛔ Trajectory arms over banked GT. **No TanitAD model was run**, so none of this
scores a model. It exists to prove the gate has dynamic range and can fail.

### 5.1 The gate fires in both directions

B1 EVAL split, 139/141 clips joined, **1,951 windows**, tactical band 2.0–6.0 s,
`raw/constraint_panel_b1eval.json`. `gt_x2speed` and `gt_creep` hold `gt`'s PATH
fixed, so the ceilings are bit-identical and only the speed channel moves.

| arm | `frac_over_ceiling` | n over | `frac_under_when_allowed` |
|---|---|---|---|
| `gt` | **0.1229** | 6,564 | 0.7960 |
| ⛔ `gt_x2speed` | **0.4755** | 25,406 | 0.4502 |
| ⛔ `gt_creep` | **0.0041** | 220 | **1.0000** |

⇒ the OVER side separates GT from a doubled-speed arm **3.9×**, and the UNDER
side catches the creeping arm at **1.0000** where the OVER side reads ~0. **A
one-sided metric would have scored `gt_creep` as near-perfect.**
**Vacuity gate: manoeuvre_rate 0.2019 (394/1,951).**

### 5.2 ⛔ A defect the GT control caught, and the rule it produced

The first run priced the ceiling off the **minimum distance to ANY agent within
60 m**. GT then "violated" its own envelope on **71.2 %** of steps and a 1 m/s
creeping arm "over-drove" on **27.3 %** — impossible for a calibrated ceiling.
Cause: an adjacent-lane or parked car 2 m to the side is not a longitudinal
constraint. ⇒ **two different quantities, and conflating them was the defect:**
`trajectory_clearance` (min distance to any agent) is a PROXIMITY statistic and
must never become a speed ceiling; `lead_gap` (in-corridor, ahead,
`gap = cx − l/2`) is what bounds speed. Pinned by
`test_an_agent_BESIDE_the_ego_is_not_a_lead`.

Second defect, same run: an unclamped double difference gave a kinematic ceiling
minimum of **0.279 m/s = κ 43.7 1/m = a 2.3 cm turning radius** — the vtarget
jitter trap in curvature costume. Clamped at `KAPPA_MAX_1_M = 0.2`.

### 5.3 Clearance, with its `n` and its censoring

`n = 68,977` steps with a computable clearance; **p05 1.99 m · p50 8.06 m ·
p95 37.26 m**. Censoring is reported per family, never dropped: **49.67 %** of
steps have no kinematic ceiling (straight path), **65.09 %** no clearance ceiling
(empty corridor), **33.20 %** neither. **Datum: rig-origin to box EDGE**, ego
footprint NOT subtracted — ⛔ this is `lead_source`'s convention and **not**
`alpasim-gsplat`'s bumper-to-bumper `EGO_LEN = 4.7 m`; the two differ by 4.7 m.

⚠️ **This is AGENT clearance, not infrastructure clearance.** All 11
`obstacle.offline` classes are dynamic agents (vendor `is_dynamic: true` on every
one); "world-static" is a per-TRACK property (1,756/2,778 = 63.2 %). Parked cars
are a *proxy* for the roadside edge and must be named as one.

### 5.4 ⛔ THE UNDER SIDE IS NOT SCOREABLE TODAY — measured, not asserted

GT is competent human driving. A criterion that flags GT as under-driving on
most windows is measuring the CEILING's uninformativeness.

| `allows` ≥ | margin | n | GT under-rate | LEAD | NO_LEAD |
|---|---|---|---|---|---|
| 8 | 3 | 43,298 | 0.7960 | 0.6710 | 0.9011 |
| 12 | 3 | 35,797 | 0.8641 | 0.7577 | 0.9354 |
| 16 | 3 | 29,089 | 0.9087 | 0.7687 | 0.9858 |
| **20** | **3** | 22,932 | 0.9276 | 0.7605 | **0.9987** |
| 20 | 10 | 22,932 | 0.8451 | 0.6361 | 0.9340 |

⇒ **the GT under-rate never falls below 0.51 anywhere on the grid and RISES with
the threshold** — the signature of a degenerate criterion. The LEAD/NO_LEAD split
localises it exactly: where a real lead gap binds it is 0.29–0.77; where it does
not (**23,534 of 43,298 steps = 54.4 %**) the only ceiling is kinematic, ~50 m/s
on a straight road, and the criterion collapses to *"GT is not driving at
50 m/s"* — true on **99.87 %** of steps.

⭐ **This is the measured answer to the PI's instruction (10): the "reach max
speed when the situation allows" half REQUIRES a road-speed ceiling that exists
in no layer.** It is a dataset extension, not a modelling choice.

---

## 6. What would unblock the under side — two honest routes, costed

| route | what it needs | error it would carry |
|---|---|---|
| **A. Import a limit** | map-matching our traces to OSM | ⛔ **IMPOSSIBLE on PhysicalAI**: `egomotion` carries no lat/lon (clip-local metres). Would need a corpus that does. |
| **B. Per-clip free-flow ceiling** | the clip's own high-percentile unobstructed speed as a per-clip constant | ⚠️ ego-derived ⇒ a **LABEL only**, and it cannot say what the road *allowed* where the ego never went fast. Under-estimates every congested clip. |
| **C. NuRec `map.xodr`** | the strategic-map path already identified | speed limits only where NuRec scenes exist — a small subset, not B1 |

⛔ **I did not invent a proxy and score against it.** Route B is the cheapest and
is *still* not admissible as a supplied input; it would serve only to supervise a
PREDICTED ceiling head. **This is a PI decision and is escalated, not assumed.**

---

## 7. Deliverable manifest

| artifact | path | state |
|---|---|---|
| constraints module | `stack/tanitad/eval/constraints.py` | repo, staged |
| its tests (35) | `stack/tests/test_constraints.py` | repo, staged |
| this pre-registration | `…/Research/2026-09-06-constraints/SPEC.md` | repo, staged |
| result | `…/2026-09-06-constraints/RESULT.md` | repo, staged |
| SPEED_BAND census | `code/speed_band_census.py` + `raw/speed_band_census_train.json` | repo, staged |
| strata ceiling probe | `code/strata_ceiling_probe.py` + `raw/strata_ceiling_probe.json` | repo, staged |
| constraint panel | `code/run_constraint_panel.py` + `raw/constraint_panel_b1eval.json` | repo, staged |
| ⭐ vision→target-speed | `raw/vt_band_from_vision.json` (the 2026-08-04 script's first run) | repo, staged |
| poses-view builder | `code/build_poses_view.py` | repo, staged |
