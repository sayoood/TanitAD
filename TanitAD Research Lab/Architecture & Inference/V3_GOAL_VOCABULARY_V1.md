# TanitAD v3 Goal Vocabulary — v1 (FROZEN 2026-07-19)

*The shared, tokenizable goal vocabulary for v3 flagship + refa-v3. One vocabulary everywhere: dataset
labels, CoC reasoning traces, conditioning embeddings, planner option space, and evaluation. Frozen after
the 2026-07-19 co-design (Sayed's capability model + Michon/Donges/J3016/SOTIF/Rulebooks anchoring).
Changes require a version bump (v1.1…) with a migration note — labels are minted against this.*

## Design rules (binding)
R1 every token **discrete/banded** (VLM-labelable; no continuous leakage) · R2 goals are **compositional
tuples** of slot tokens · R3 every label carries **provenance** `{kinematic|map|vlm|human|sim}` ·
R4 the tactical vocabulary **is** the planner's option space · R5 one shared vocabulary, namespaced
`SLOT:token` (VLM/CoC-friendly — frozen decision Q4) · R6 anti-shortcut: goal-dropout ≥0.5 during
training; conditioning always via the level-above's *token*, never raw future signals.

## Frozen decisions on the five open questions
| Q | Decision | Rationale |
|---|---|---|
| Q1 VTARGET granularity | **Non-uniform**: 1 m/s bands below 10 m/s, 2.5 m/s bands 10–40 m/s | stop-and-go lives at low speed; uniform 2.5 wastes resolution where longitudinal control is hardest |
| Q2 LONMODE size | **9 core modes**; decel intensity moved to DYN param | keeps modes orthogonal; `decel_soft/hard` are intensities, not modes |
| Q3 Agent-referenced tokens | **v3.0 minimal**: k ∈ {lead, merger} only; full tracked-agent slots v3.1 | lead-state labels are already required for longitudinal; full tracking is a data cost we stage |
| Q4 Tokenization | **One shared vocab, slot-prefixed** (`LON:follow_lead`) | reads naturally in CoC traces + VLM prompts; single codebook simplifies embedding + LM-compat |
| Q5 Sequential goals | **Single compound goal in v3.0**; `THEN`-chains v3.1 | TACPOINT anchors carry most sequencing implicitly; chains add planner branching we don't need yet |

## STRATEGIC goal = ⟨MISSION, ROUTE, LANEOBJ, SPEEDPOLICY, STYLE, RISK, ODD⟩ — 34 tokens
*Sayed 2026-07-19: **changing the SET SPEED is a TACTICAL decision** — VTARGET/VSOURCE moved to tactical.
The strategic layer owns the slow speed **intent/envelope** (SPEEDPOLICY: cap + regime), not the concrete
per-moment target; STYLE biases it. The tactical layer picks the actual set-speed responding to sign/lead/curve.*
| Slot | n | Tokens |
|---|---|---|
| MISSION | 5 | `route_follow` `free_navigate` `explore` `plan_pullover` `mrm_now` |
| ROUTE | 9 | `follow` `straight` `turn_left` `turn_right` `exit_left` `exit_right` `merge` `u_turn` `roundabout` |
| LANEOBJ | 4 | `keep` `prefer_left_faster` `prefer_right_exit` `any` |
| SPEEDPOLICY | 4 | `nominal` `cap_low` `cap_med` `cap_high` — strategic speed **envelope/cap** (from ODD/mission/comfort), the ceiling the tactical set-speed respects; NOT the concrete target |
| STYLE | 5 | `max_availability` `comfort` `eco` `dynamic` `degraded_caution` — **= planner cost-weight presets** (bias the tactical set-speed too) |
| RISK | 4 | `nominal` `elevated_weather` `elevated_visibility` `elevated_anomaly` — scales risk budget + planner compute |
| ODD | 3 | `in_odd` `odd_exit_ahead` `capability_degrading` |

## TACTICAL goal = ⟨VTARGET, VSOURCE, LONMODE, LATMANEUVER, HEADWAY, DYN, RULECTX, SIGNAL, INTERACT, TACPOINT, LIGHTSTATE⟩ — 80 tokens, 11 slots
| Slot | n | Tokens |
|---|---|---|
| VTARGET | 23 | `v_stop`, `v(0-1]`…`v(9-10]` (10 × 1 m/s), `v(10-12.5]`…`v(37.5-40]` (12 × 2.5 m/s) — **the concrete set-speed, a tactical choice** from sign/lead/curve, capped by strategic SPEEDPOLICY, biased by STYLE |
| VSOURCE | 5 | `sign_limit` `lead_constrained` `curve_constrained` `road_class_default` `traffic_flow` — WHY this tactical set-speed (teaches sign→speed + lead/curve reasoning) |
| LONMODE | 9 | `free_cruise` `follow_lead` `close_gap` `open_gap` `stop_at_point` `hold_stop` `launch` `creep` `coast` |
| LATMANEUVER | 9 | `lane_keep` `lc_left` `lc_right` `abort_lc` `merge_in` `yield_merge` `nudge_left` `nudge_right` `pull_over` |
| HEADWAY | 5 | `hw_0.8s` `hw_1.2s` `hw_1.45s` `hw_1.75s` `hw_2.5s+` |
| DYN | 4 | `gentle` `normal` `firm` `max` (intensity of the active mode; absorbs decel_soft/hard) |
| RULECTX | 5 | `conform` + `justified_deviation{obstacle_avoidance, rescue_corridor, stopped_vehicle_pass, instructed}` |
| SIGNAL | 6 | `none` `indicator_left` `indicator_right` `hazard` `headlight_flash` `horn` (parallel acts) |
| INTERACT | 5 | `none` `yield_to_k` `assert_gap_k` `cooperate_merge_k` `respond_emergency` (k∈{lead,merger} in v3.0) |
| TACPOINT | 5 | `none` `stop_line` `merge_point` `creep_point` `clear_point` |
| LIGHTSTATE | 4 | `proceed` `prepare_stop` `stop_at_line` `creep_check` |

**Total: 114 tokens, 18 slots** (strategic 34 / 7 + tactical 80 / 11). Operative level stays
continuous (trajectory + [steer, accel, v0]); a coarse steer×accel bin grid exists only as the CEM
sampling space, not vocabulary.

> **Header-arithmetic correction, 2026-07-20** — the section headers previously read "76 tactical /
> 110 total / 17 slots"; the per-slot `n` rows have always summed to **80 / 114 / 18**. Only the
> summary arithmetic was wrong: **no token, slot, or semantic changed**, so this is NOT a version
> bump and no label migration is required. The **enumerated per-slot rows are authoritative** — the
> implementation (`stack/tanitad/lake/vocab.py`) pins each slot's count so any future drift between
> the table and the code fails loudly instead of silently mis-sizing an embedding.

## Label minting (per slot, provenance-stamped)
- `VTARGET`: 85th-pct future free-flow speed over 10–20 s in unconstrained segments, capped by sign/map
  limit → band. `VSOURCE` from VLM sign-reads / map / flow stats. **kinematic+vlm**
- `LONMODE/HEADWAY`: classified from kinematics + **lead state** (presence, gap, closing speed) — minted by
  the perception-enrichment pass (detector+monodepth or Qwen3-VL/Cosmos-Reason2 pipeline). **kinematic+vlm**
- `LATMANEUVER`: sustained lateral displacement vs lane width + yaw signature (extends labels-v2
  curvature-relative logic). **kinematic**
- `ROUTE`: labels-v2 v2 (curvature-relative, AMBIGUOUS masked). **kinematic**
- `STYLE/RISK/ODD/MISSION`: scene-level VLM tags (weather/visibility) + engineered ODD monitors; largely
  **inference-time inputs**, trained only where labels are honest (weather→RISK). **vlm+engineered**
- `RULECTX/SIGNAL/INTERACT/TACPOINT/LIGHTSTATE`: VLM+detector passes (lights, indicators visible on ego?
  no — SIGNAL is imitation-limited in v3.0: labeled only where CAN carries it, e.g. L2D blinker channel;
  else planner-emitted only). **mixed — honest gaps stamped `unknown`**

## Anti-shortcut enforcement (from H25/H26 lessons)
Goal-dropout 0.5 on every conditioning path · gated/ReZero conditioning (init 0.1) with **norm parity
monitored** (intent_proj vs action_emb — the H26 swamping bug class) · VTARGET conditioning is the
*strategic layer's predicted token* at inference (teacher-forced + dropout in training) · causality eval
mandatory: swap a goal token → conditioned output must change (CI-separated) before a seam counts as live.
