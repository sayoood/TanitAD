# HIERARCHY VOCABULARY — strategic & tactical goals/actions as a rich tokenizable
# vocabulary, + the frozen-latent interpretation-head suite (PI directive 2026-08-11)

**PI resequencing order (supersedes VLM_STRATEGIC_LABELING.md §7): the VLM/algorithmic
pipeline moves PRIOR to v6 training — it now supplies the vocabulary's label stream and
starts IMMEDIATELY on pod4 (the VLM pod: all three PH0 arms prefetched); pod5 carries the
remaining programme tasks.** The complete, correctly-wired 4B hierarchy is a v6 REQUIREMENT:
every layer emits ACTIONS conditioning its own predictor and GOALS conditioning the layer
below, drawn from the vocabulary defined here.

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

## 3. STRATEGIC vocabulary (horizon 8–30 s+, conditions the tactical layer)

**Goals** `g_str`:
| token | args | derivation |
|---|---|---|
| `KEEP_CORRIDOR` | target_arc_m | hindsight path curvature-relative follow |
| `LANE_TARGET` | lane_offset_idx, deadline_m | lateral displacement events (E4.1 LAT) |
| `EXIT_RIGHT` / `EXIT_LEFT` | distance_m | corridor split geometry |
| `TURN_LEFT` / `TURN_RIGHT` / `STRAIGHT_THROUGH` | intersection arc | E7.1 turn events |
| `ROUTE_TO` | text_token_id (city/POI vocab from OCR), evidence_id | ONLY with signage OCR (G1-gated); abstains otherwise |
| `STOP_AT` | distance_m | signage/geometry (stop events in hindsight speed profile) |
| `NONE_ABSTAIN` | — | honest ceiling |

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
| `SPEED_BAND` | v_lo, v_hi | LON axis |

**Actions** `a_tac` (factored, each with continuous envelope args):
- LAT: `LANE_KEEP` · `LANE_CHANGE_L/R(within_m)` · `ABORT_LC` · `NUDGE_L/R(lat_m)`
- LON: `FOLLOW(time_gap_s)` · `CRUISE(v)` · `YIELD/MERGE(gap_slot)` · `BRAKE_TO(v, within_m)` · `CREEP` · `HOLD`
Meta-action phrases from Alpamayo inference map onto this token set (mapping table =
PH1 deliverable; unmappable phrases get logged, not silently dropped — vocabulary
completeness is MEASURED as coverage of the 4,800-clip distribution).

**Operative layer** keeps continuous (a, κ) unicycle controls (W4/W4r-proven) — no
tokenization at 10 Hz.

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
