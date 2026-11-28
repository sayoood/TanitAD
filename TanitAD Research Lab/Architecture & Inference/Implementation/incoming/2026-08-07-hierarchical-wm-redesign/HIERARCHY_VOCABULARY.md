# HIERARCHY VOCABULARY — strategic & tactical goals/actions as a rich tokenizable
# vocabulary, + the frozen-latent interpretation-head suite (PI directive 2026-08-11)

**PI resequencing order (supersedes VLM_STRATEGIC_LABELING.md §7): the VLM/algorithmic
pipeline moves PRIOR to v6 training — it now supplies the vocabulary's label stream and
starts IMMEDIATELY on pod4 (the VLM pod: all three PH0 arms prefetched); pod5 carries the
remaining programme tasks.** The complete, correctly-wired 4B hierarchy is a v6 REQUIREMENT:
every layer emits ACTIONS conditioning its own predictor and GOALS conditioning the layer
below, drawn from the vocabulary defined here.

## 0b. Engine C — SAM-family segmentation in the labeling pipeline (PI idea 2026-08-11)

Add Meta's SAM line (SAM 2/3 video segmentation) as **Engine C** beside Engine A
(geometry) and Engine B (VLM): offline, per-clip semantic/instance masks — drivable
surface, lane furniture, agent instances tracked through time — giving (a) richer
grounding for the VLM's claims (a sign/light claim must overlap a mask), (b) mask-level
agent tracks that cross-check `obstacle.offline` and extend labels to classes it lacks
(static furniture), (c) lane/road-surface geometry PhysicalAI's labels never had — the
closest admissible thing to the missing map. Usage stays labeling-side only (frozen-probe /
goal-head / eval-strata discipline unchanged). Feasibility smoke on pod4 rides the PH0
window (SAM checkpoints are small next to the VLM arms); PH0's prereg gains an OPTIONAL
Engine-C column — measured mask-vs-join agreement on the 50 pilot clips decides whether
Engine C enters PH1.

**Image-coordinate requirement (PI 2026-08-11):** every semantic claim any engine emits
(VLM sign/light/agent fields, SAM instances) MUST carry its image-space grounding —
bounding box `[x0,y0,x1,y1]` and, where the engine provides it, the pixel contour/mask
reference + frame index. Ungrounded claims are `disputed` by default in the fusion gate.

## 1. Sources the vocabulary is derived from

1. **Alpamayo-2-Super inference data** (our banked 4,800-clip augmentation, 5 tasks incl.
   meta-action parsing) — the empirical distribution of manoeuvre phrases and meta-actions
   real driving exhibits on our corpus.
2. **The VLM pipeline** (PH0→PH1): scenario/domain/signage/strategic fields per clip,
   two-engine adversarially fused (Engine A hindsight geometry disposes).
3. **E4.1/E7.1 machinery**: the factored LAT/LON axes and curvature-relative corridor
   events — the geometric backbone every token must ground to.

## 2. Design rules (binding, inherited)

- **Tokenizable = finite discrete head + typed continuous slots.** Every token is
  `(TYPE, [args])`; args are physical units (m, s, m/s). No free text at inference.
- **Factored LAT × LON everywhere** (the 5-way-mixed-softmax defect is retired by design).
- **Goal/situation disjointness** (2026-08-03): no goal token may be derivable from the
  situation classifier's output; provenance tags (`path|signage|vlm-fused`) travel with
  every supervised instance.
- **Labels supervise GOAL/INTERPRETATION HEADS only — never any WM trunk loss** (JEPA
  thesis; the aux-label retraction stands).
- **Every token must be HINDSIGHT-DERIVABLE** from Engine A geometry alone (VLM enriches;
  geometry guarantees) — so the vocabulary works even where the VLM abstains.
- **Every goal token carries OPTIONAL temporal and spatial constraint slots (PI
  2026-08-11):** `within_m` / `by_time_s` / `at_arc_m` / `hold_for_s` — uniformly typed,
  so "change lane within 500 m", "stop at arc 82 m", "hold corridor for 12 s" are all
  expressible without new token types. Unset = unconstrained.
- **Near-field is TIME-scaled, not metre-scaled (PI correction):** a fixed 40 m band
  cannot cover a 6 s horizon (180 m at 30 m/s). All near-field weighting (V6 measure O2)
  and constraint defaults use TIME-TO-REACH (arc_length / v_ego, capped at the 6 s
  horizon) — speed-adaptive by construction.

## 3. STRATEGIC vocabulary (horizon 8–30 s+, conditions the tactical layer)

**Goals** `g_str`:
| token | args | derivation |
|---|---|---|
| `KEEP_CORRIDOR` | target_arc_m | hindsight path curvature-relative follow |
| `LANE_TARGET` ⛔ **DERIVATION REFUTED — NOT EMITTED** | lane_offset_idx, deadline_m | ~~lateral displacement events (E4.1 LAT)~~ **REMOVED 2026-08-16 (`06b8782`), and note this is the SPEC that was wrong, not only its implementation:** "lateral displacement events" cannot distinguish a lane change from road curvature — a pure road curve clears the gate at any highway speed, the PI adjudicated the resulting labels ~78 % wrong (14/18 with an opinion), and a 34-feature search over the adjudicated set found **no separating ego-geometry feature** (best `yaw_lin_r2`, FWER p=0.29, no perfect separation, n=18 — the study could have found one: min attainable two-sided p = 0.00065). ⇒ **`LANE_TARGET` is now derived from what the ROUTE DEMANDS, not from what the ego DID** (`s2_derive.py` §LC, `lane_change_requirement()`), which needs a LANE CONTEXT (`n_lanes_same_direction`, `ego_lane_idx`, `route_lane_idx`, `lane_continues`) that PhysicalAI-AV does not ship — so `required` is `None` (UNKNOWN) for **801/801** clips and the token is emitted by no path today. ⚠️ It **stays in `STRATEGIC_GOAL_TOKENS`**: the tuple sizes an embedding table (`v6.py:3281`) and the live v6F run resumes tensor-level. A lane change *observed* inside a route-following segment is a TACTICAL event — §4's `a_tac: LANE_CHANGE_L/R` — not a strategic goal. |
| `EXIT_RIGHT` / `EXIT_LEFT` | distance_m | corridor split geometry |
| `TURN_LEFT` / `TURN_RIGHT` / `STRAIGHT_THROUGH` | intersection arc | E7.1 turn events |
| `ROUTE_TO` | text_token_id (city/POI vocab from OCR), evidence_id | ONLY with signage OCR (G1-gated); abstains otherwise |
| `STOP_AT` | distance_m | signage/geometry (stop events in hindsight speed profile) |
| **`FOLLOW_MAIN_ROAD`** | — | **THE DEFAULT strategic goal whenever no navigation route is set up (PI 2026-08-11)** — hindsight-derivable as the corridor-continuation prior; replaces `NONE_ABSTAIN` as the no-route baseline (abstain remains only for genuinely ambiguous geometry) |
| `NONE_ABSTAIN` | — | honest ceiling (ambiguous geometry only, given FOLLOW_MAIN_ROAD above) |

**Actions** `a_str` (emitted by S, condition S's own predictor; horizon-typed):
`PREPARE_LANE_CHANGE(dir, within_m)` · `HOLD_CORRIDOR(arc_m)` · `REDUCE_TO(v_target, within_m)`
· `PREPARE_EXIT(dir, within_m)` · `PREPARE_STOP(within_m)` · `RESUME_CRUISE(v_target)`

## 4. TACTICAL vocabulary (2–8 s, conditions the operative layer)

**Goals** `g_tac` (what S hands down / what T selects):
| token | args | grounding |
|---|---|---|
| `ANCHOR_GOAL` | anchor_id ∈ fan vocab, t_reach_s | the geometric goal-point lever (the +4.7 PDMS class) |
| `CORRIDOR_OFFSET` | lat_offset_m, arc_m | curvature-relative corridor frame |
| `GAP_TARGET` | agent_slot_id, time_gap_s | from the perception head's agent slots (§6) |
| `SPEED_BAND` | v_lo, v_hi | LON axis — **SET BY THE TACTICAL LAYER (PI decision 2026-08-11): target speed is a tactical responsibility, computed from traffic-sign inputs (VLM/OCR speed-limit fields) and prior speed information (corridor speed statistics), bounded by the strategic layer's `REDUCE_TO` only as an upper envelope** |
| **`YIELD_AT`** | position_arc_m, gap_slot | **PI addition**: yield point + the gap being yielded to (merge/roundabout/unprotected turn) |
| **`STOP_POINT`** | position_arc_m, reason ∈ {sign, light, queue, hazard} | **PI addition**: tactical stop with grounded position |
| **`WAIT_FOR_ONCOMING`** | narrow_arc_m, oncoming_slot | **PI addition**: hold before a narrows/parked-lane pinch until oncoming traffic clears (narrow-road negotiation) |
| **`EVADE_IN_CORRIDOR`** | lat_offset_m, obstacle_slot, past_arc_m | **PI addition**: in-corridor lateral evasion around open doors / cyclists / pedestrians / parked vehicles — bounded by corridor, NOT a lane change |
| **`TRAFFIC_LIGHT_REACT`** | light_slot_id, state ∈ {red, yellow, green}, stopline_arc_m | **PI addition**: light state from the VLM/signage fields (eval + goal-head supervision; never trunk) — proceed/prepare-stop/stop resolved by the LON action given this goal |

**Actions** `a_tac` (factored, each with continuous envelope args):
- LAT: `LANE_KEEP` · `LANE_CHANGE_L/R(within_m)` · `ABORT_LC` · `NUDGE_L/R(lat_m)`
- LON: `FOLLOW(time_gap_s)` · `CRUISE(v)` · `YIELD/MERGE(gap_slot)` · `BRAKE_TO(v, within_m)` · `CREEP` · `HOLD`
Meta-action phrases from Alpamayo inference map onto this token set (mapping table =
PH1 deliverable; unmappable phrases get logged, not silently dropped — vocabulary
completeness is MEASURED as coverage of the 4,800-clip distribution).

**Operative layer** keeps continuous (a, κ) unicycle controls (W4/W4r-proven) — no
tokenization at 10 Hz.

## 4b. TRAJECTORY HORIZON SPEC (PI clarification 2026-08-11 — BINDING for v6)

**Every planned trajectory spans up to 6 s — covering BOTH the operative and the tactical
horizon in one kinematically consistent rollout.** Concretely:
- The fan/plan is a **60-step control sequence (a, κ) @10 Hz integrated through ONE
  unicycle rollout 0→6 s** — never two stitched trajectories. Segment semantics: 0–2 s is
  the operative band (fine control authority), 2–6 s is the tactical band (the same
  controls, shaped by `g_tac` conditioning) — the shared integrator makes the 2 s seam
  discontinuity-free BY CONSTRUCTION (the X2 seam metrics verify, not repair).
- The tactical layer's goals therefore ground at 2–6 s (the earlier 2–8 s note is
  superseded); the strategic layer conditions beyond 6 s via goals only.
- Emission heads scale k=20 → k=60; the diffusion proposal generator diffuses the full
  6 s control sequence; W7-style roll selection rolls to 6 s (roll-k scales with it).
- **Eval consequence:** four families + oracle/selected reported at BOTH 0–2 s and 0–6 s;
  T1/P5 compounding measured to 6 s. **E-H1/W5 (6 s on the w120 trunk, gate ADE(6s) ≤
  3×ADE(2s)) is promoted from queued to REQUIRED precursor** — it baselines v5.8f at 6 s
  before v6 trains against it.

## 5. Wiring (the "correctly wired" requirement, explicit)

S-predictor: `z_str_{t+K} = P_S(z_str_t, a_str_t)` · T-predictor:
`z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)` · O-predictor:
`z_op_{t+j} = P_O(z_op_t, (a,κ)_t | g_tac)`. Goals flow DOWN only; latents flow UP through
stop-grad/EMA (gradient-isolation matrix, V6_TRAINING_MEASURES X3). Each layer's selector
is roll-cost-based over ITS OWN predictor (W7 pattern per level); per-level sel_gap via
E8.1. Token embeddings are shared between the goal-emitting head (above) and the
goal-consuming conditioner (below) — one vocabulary, two views.

## 6. FROZEN-LATENT interpretation-head suite (all trunk-frozen readouts; labels admissible
## here by design — they measure and expose, never train the trunk)

| head | output | supervision | status |
|---|---|---|---|
| **PERCEPTION-AGENTS** | K agent slots: bbox (cx, cy, yaw, l, w) + state (v, heading-rate, occluded) + characteristics (class, size-prior) | obstacle.offline join (26k records) | NEW — design here; DETR-style slot decoder ~2-4 M params on spatial tokens |
| P8-OCC | BEV occupancy (+ per-cell class later) | join rasters | attempt-2 training NOW (both pods) |
| LEAD-STATE | lead slot: gap, closing speed, TTC | join (vehicle-filtered) | after LF0 locates the information |
| KIN-READOUT | v, yaw-rate, curvature | ego (free) | P1-proven (R² 0.99/0.86/0.84) |
| UNCERTAINTY | fan spread / roll-cost variance | self-supervised | P7-proven (ρ 0.716) |
| P9-SALIENCY | input-space attribution per head | none (gradients) | rides P8 harness |
| SCENARIO/DOMAIN (eval-only) | strata tags | VLM fields | PH1 deliverable; NEVER an input (disjointness) |

The PERCEPTION-AGENTS head doubles as: (a) the P4/P8 permanence instrument at object
level; (b) the `GAP_TARGET`/`agent_slot` grounding for tactical tokens; (c) the
interpretability deliverable (draw the WM's believed agents on the camera/BEV).

## 7. Execution plan (STARTED tonight)

1. **Pod4 = the VLM pod** (PI): after W7-FULL completes tonight, PH0 runs — video-template
   smoke ×3 arms FIRST (chained tonight), then the 50-clip pilot per `PREREG_PH0_VLM.md`
   (sampler + runner tomorrow AM). PH1 spend decision from measured s/clip.
2. **Pod5 = everything else**: p8c2 → H-COTRAIN → T1 rows/four-families for the v5.8f
   release row (unchanged), then W5/E5 restorations.
3. Vocabulary v0 = THIS DOC; the Alpamayo meta-action mapping table + coverage measurement
   = first PH1-adjacent deliverable; tokenizer freeze before any v6 head is built.
4. PERCEPTION-AGENTS head prereg + trainer next (pod5, after release row; ~1 GPU-day).

- [ ] smoke ×3 · [ ] ph0_clips.json · [ ] 50-clip pilot ×3 · [ ] mapping table +
  coverage · [ ] tokenizer freeze · [ ] perception-head prereg
