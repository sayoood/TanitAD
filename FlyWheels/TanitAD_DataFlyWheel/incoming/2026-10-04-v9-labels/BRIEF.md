# WP-A — the v9 label release: per-frame tactical goals/actions with constraints, per-frame nav, route checkpoint

*Commissioned by the Master Mind, 2026-10-04, for the PI's refcv8 requirements R8-1, R8-2, R8-3
(`Project Steering/PLAN_REFCV8.md` §0). Owner: TanitAD_DataFlyWheel. Schema: TANITAD_PROGRAMME.md §3 (SPEC / code /
tests / raw / RESULT / COMMS) in this folder.*

## The PI's requirement (verbatim, abridged)
"Each camera frame should have corresponding tactical goals and actions in the future time frame (2 to 8 seconds from
the current reference frame) with corresponding constraints like distance and time (not only one per clip). Each frame
should have a corresponding nav command as already defined, not only one per clip. We should also introduce a route
checkpoint extracted from the ego trajectory defining the next reference route point in the vehicle coordinate system;
this is not a label, this is just the route way point goal as training and inference way point in addition to the nav
command simulating the nav system of the car."

## What is wrong today (MEASURED 2026-10-04 — read these packages first)
`TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/`: D1 `COVERAGE.md`, D2 `RESULT.md` +
`SEMANTICS.md`, D3 `RESULT.md`, D4 `USAGE_AUDIT.md` + `REFCV8_LABEL_DESIGN.md` (the design you start from), D5
`EFFECTS_INVENTORY.md`; and the route package `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following/`
(RESULT.md, RESULT_A5.md, RESULT_A6.md, SPEC_ADDENDUM_A7.md).
* ONE v8 record per clip, anchored at raw t0 = 8.0 s; tactical labels exist on **23.22 %** of 746,946 train windows,
  on **35.1 %** of turn windows (D1).
* NUDGE labels curves (10–12 % agree with geometry); longitudinal computed over [0, 6] not [2, 6]; effective
  ACCELERATE bar +1.0 not +1.5 (D2). 3 of 8 lateral and 1 of 8 longitudinal classes have ZERO windows (D1).
* The nav token is per clip: on **74.3 %** of L/R windows no turn starts in the next 6 s (D1). `build_v8_nav30s.py:83`
  reads `suppressed` while the emitter writes `applied` (66 suppressed turns stay); a 30 s cap vs `nav_command`'s 35 s;
  curve-first records; `nav_30s.entries` also carries NAV_FOLLOW_ROAD pseudo-entries with null `dyaw_deg` (D1, D2, A6).
* `LANE_CHANGE_L` was supervised as a negative on 100 % of tactical windows because `v7_labels._MEASURED_GEOMETRY_TOKENS`
  is module state that the eval-blob load overwrote before the DataLoader workers started (D1). ⛔ The v9 reader must
  hold NO mutable module-level state.
* The max-speed input was a future-speed oracle (R² 0.988 in band, D2); D4 measured past-only options (N2 leak 3.5 %).
* Split: 3–5 eval clips share a source recording with train clips; 2 train pairs are the same video; the ego appears as
  an agent box in 18 train clips (D3).

## Deliverables, in PRIORITY ORDER — bank each; report to the Master Mind after (1) and after (3)
1. **SPEC.md — the v9 definitions, BEFORE any build** (the Master Mind reviews it against the PI's words):
   * **Per-frame tactical ACTIONS over [NOW+2 s, NOW+8 s]** for every 10 Hz frame NOW of every clip (train 4,369 +
     eval139): lateral (LANE_KEEP / TURN_L / TURN_R; LANE_CHANGE_L/R only where SAM3 lane lines make it measurable —
     D4 showed ego motion alone cannot separate lane change from nudge), longitudinal (e.g. STOP / DECELERATE /
     ACCELERATE / KEEP / FOLLOW / CREEP / HOLD) — each with its **constraints**: start and end time (s from NOW),
     arc-length distance (m), Δyaw and radius for turns, target speed v_target with t_reach / d_reach, stop distance and
     time, lead distance and time gap for FOLLOW. Literal thresholds, stated with their physical justification.
   * **Per-frame tactical GOALS** (the goal-token family) over the same band with the same constraint fields: TURN_L/R,
     STOP_POINT, SPEED target, FOLLOW_LANE, traffic-light reaction (VLM-only: D4's propagation rule — RED over the stop
     episode — or anchor-bound), yield / evade / overtake / merge / exit (VLM, re-timed to the VLM's own window, D4),
     lane change (SAM3). Every goal carries provenance (geometry / agents / sam3 / vlm) and a validity mask; NEGATIVES
     defined per token (geometry negatives; the PI's 2026-09-16 caption-absence negatives rebuilt against the v9 blob).
   * **Per-frame NAV**: announced junction turns from a WHOLE-RECORDING turn table (fix the three builder defects
     above), token + distance + time to the turn start; FOLLOW when no announced turn is inside the announcement horizon
     or after the turn ends. State the horizon (distance and time) and why.
   * **ROUTE CHECKPOINT (an INPUT at training and inference, not a label)**: the next reference route point in the NOW
     vehicle frame (x forward, y left), from the ego's driven path, plus validity and arc-length distance. Define at
     least two variants — RC-A a fixed arc-length lookahead L ∈ {30, 50, 80} m along a SMOOTHED path (no speed
     dependence), RC-B the next decision point (the end of the next announced manoeuvre), clamped — and the deployment
     analogue (a navigation system's route polyline; for NavSim, the scene's route centreline — say what the
     EvalFlyWheel bridge must compute).
   * Admissibility statement per field (PI 2026-08-03: labels may use ego/future/agents/maps; inference is vision +
     the supplied nav, route checkpoint and speed limit; a goal input must not carry the situation classifier's
     output). The nav and the route checkpoint are optimistic on PhysicalAI by construction — say so.
2. **The ECHO / LEAK study for the route checkpoint and nav (CPU, before any training)**, on eval139 (the D1 eval
   truth table `D1_label_census/tables/label_truth_eval139.npz` gives every window): (a) a trivial planner that drives
   to the checkpoint at constant v0 — its ADE, heading-within-15° and turn direction-correct vs GT and vs refcv7's
   banked picks (`D:/refcv7_route_bin/2026-10-04/eval_s0g.npz`); (b) speed leak: OOF R² of the 6-s future speed
   profile from v0 vs from v0 + checkpoint (must be ≈ equal); (c) lateral leak: the checkpoint's lateral offset vs a
   heavily smoothed route at the same arc length; (d) per variant. This decides L and the variant; it is the
   literature's known risk (CARLA "target point" conditioning and its shortcut, e.g. Jaeger et al., "Hidden Biases of
   End-to-End Driving Models", ICCV 2023 — find, read and BANK the primary with `tools/kb_add.py` before citing it).
3. **The build**: eval139 first, then train. Sources: the 100 Hz egomotion over the WHOLE recording (so every frame
   has its 8-s future and nav can look 30+ s ahead — find where D2 read it, `D2_label_validity/raw/d2_speed.py`), the
   v8 records (VLM semantics, `alpamayo.*` windows), the agent join (lead / FOLLOW / distance keeping), SAM3 map GT
   (lane lines). Frame time = the trainer's own clock (`refc_v3_train.py::_now_s`, raw offset 2, clip-clock sidecar;
   D1 reproduced it bit-exactly) so a row joins a training window without guessing. Release files with md5, a manifest
   (sources + md5s + builder commit), a schema doc.
4. **Validation report (RESULT.md)**: coverage per channel (target ≥ 95 % of windows for actions; every turn window);
   agreement of every geometry-derived field with an INDEPENDENTLY written derivation (not an import of the builder)
   — targets ≥ 0.95 on TURN side and STOP; constraint error (distance, time) vs the egomotion; class support per split;
   analytic controls (a circle reads TURN with the right radius; a constant-deceleration track reads STOP at the
   right distance); a MUTATION per field that must go red; VLM-propagation coverage; the echo/leak table from (2).
5. **Reader + integration contract**: `stack/tanitad/data/v9_labels.py` (pure functions, no mutable module state;
   per-window lookup by stable clip id + NOW) with tests `stack/tests/test_v9_labels*.py`; and `INTEGRATION.md` — the
   exact tensors (names, shapes, dtypes, masks, units) the trainer and the model will receive, so WP-B
   (Architecture & Inference) wires them. ⛔ Do NOT edit `stack/scripts/refc_v3_train.py` or model code — WP-B owns them.
6. **Corpus manifest for refcv8**: the train clips to DROP (the recording partners of eval clips, one of each duplicate
   video pair — D3 `raw/c1_*`), and the ego-agent-box frames to MASK (D3 `raw/c3b_agents_detail.json`). Keep eval139
   unchanged so refcv8 stays comparable to refcv7.
