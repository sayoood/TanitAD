# AMENDMENT 1 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): Master Mind rulings on the metric-code ambiguities

**Date:** 2026-11-07 ~22:20Z.
**Ruled before** any model rollout exists. No checkpoint has been scored by the behaviour code.
**Scope:** no threshold, window, rule or adoption bar changes. These rulings fix only the definitions that the frozen text left open. Item numbers follow `code/AMBIGUITIES.md`.

## Rulings

| id | ruling |
|---|---|
| A-1 | **ADOPTED as implemented.** Free road requires no lead within G **and** no lead with TTC ≤ 8 s. The literal parse contradicts the rationale. |
| A-2 | **ADOPTED.** The sim must emit `agent_cls` (nuPlan tracked-object type). Until it does, outputs carry `agent_cls_defaulted = true`. |
| A-3 | **ADOPTED.** Scalar speeds as written. A path-projected lead speed is welcome but not required. |
| A-4 | **ADOPTED.** The sim is asked to log true `ay` and yaw rate if available. The `v * yaw_rate` proxy stays until then. |
| A-5 | **ADOPTED: `C1_MODE = "clear_persistent"`** (the default). The literal mode would score a legitimate stop at a light that turns red as a stall, contradicting c2. |
| A-6 | **ADOPTED.** |
| A-7 | **ADOPTED.** |
| A-8 | **ADOPTED.** A creep at ≤ 1.0 m/s counts as stopped. The creep-stall is the defect this rule exists for. |
| A-9 | **ADOPTED.** |
| A-10 | **REPLACED: f1 must reproduce NAVSIM's at-fault classification from SOURCE, not from memory.** The source exists on the dev box: `F:/Archive/devbox-C/navsim/devkit/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py:363-420` (`_calculate_no_at_fault_collision`) and `pdm_scorer_utils.py:12` (`get_collision_type`). The reconstruction differs from source in at least two places: ego "stopped" is `speed <= 5e-02` in source, not 0.005; and ACTIVE_FRONT is a front-bumper `LineString` intersecting the track polygon, not a ±45° bearing.<br>The implementation must follow the source's order: STOPPED_EGO, STOPPED_TRACK (`is_track_stopped`), ACTIVE_REAR (`is_agent_behind`), ACTIVE_FRONT (front bumper), else ACTIVE_LATERAL.<br>At fault = STOPPED_TRACK or ACTIVE_FRONT, or LATERAL while the ego is in multiple lanes or non-drivable area. On the lifted lane-less road the `|cross-track| > 0.5 m` stand-in for "multiple lanes" stays, flagged `lane_proxy = true`.<br>A track whose first collision was not at fault is exempt afterwards, as in source. Collisions with non-agent objects are at fault: NAVSIM scores them 0.5; for f1 either is a FAIL.<br>Vendor the functions unmodified and sha-pinned, as was done for `ego_is_comfortable`. **Until this is done f1 is UNVERIFIED and no adoption may rest on it.** |
| A-11 | **ADOPTED** (the test-arm design). |
| A-12 | **ADOPTED.** |
| A-13 | **ADOPTED.** |
| A-14 | **WORK ITEM.** The TACTICAL executed-decision confusion table needs an event-detection definition. Tier-1 reads report TACTICAL as N/A with this reason. The definition must be added by a further amendment before the first tier-2 ADOPT read. A missing family is a work item, not a pass. |
| A-15 | **ADOPTED.** The primary statistic is the paired pass indicator. Continuous-margin statistics may be reported as DESCRIPTIVE secondary rows and never decide. The point-effect-vs-floor reading is adopted; the CI-bound variant stays reported. |
| A-16 | **ADOPTED** (causal). |
| A-17 | **ADOPTED.** |
| A-18 | **ADOPTED** (scenario-wide). Gate H handles a human who does the same. |

## Tier stamp

The scorecard's interim stamp "T2-class, closed-loop sim" is replaced by **`T2 (re-perception closed-loop sim; MM ruling 2026-11-07)`**. The sim re-renders the camera views, which is T2 under `EVAL_DOCTRINE.md`. The sim session acknowledged this.

## Gate H

The first real run of the suite scores the human log against itself (every human-relative rule must read PASS) before any checkpoint is scored. A Gate H failure is an instrument defect, not a model result.
