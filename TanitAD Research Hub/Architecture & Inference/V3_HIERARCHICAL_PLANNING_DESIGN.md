# TanitAD v3 — Hierarchical World-Model Planning · Design Draft 1

*2026-07-19 · co-design working doc (Sayed + Claude). Draft 1 for iteration — the goal vocabulary especially.
Inputs: REF-A deep analysis (frozen ceiling, ~94% longitudinal), H26 hierarchy panel (strategic echo, weak
tactical head, mis-scaled intent seam), DINO-WM literature comparison, longitudinal planning/control survey.*

---

## 0. Design thesis

**Planning = evaluating alternative tactical plans against a strategic goal, using the world model to predict
each plan's consequences.** The hierarchy stops being three supervised heads (which degenerate: route echoes
the command, tactical regresses to the mean, operative hedges speed) and becomes a **goal → options →
consequences → cost → choice** pipeline. Supervised heads survive only as *proposal priors*, never as
decision-makers. This one change addresses simultaneously: the longitudinal mean-regression, the degenerate
strategic seam, and the "alignment asserted but not demonstrated" H26 gap.

Two v3 arms share everything below except the encoder:
- **flagship-v3** — own trained encoder (incremental from v2: keep JEPA+SIGReg, metric grounding, speed
  channel, decorr, anchored decoder).
- **refa-v3** — frozen-encoder matrix: (a) frozen generic DINOv2 + feature-prediction + CEM (faithful
  DINO-WM control); (b) **own-SSL-then-freeze** (continued DINO/JEPA pretraining on our driving video, then
  frozen — the scientifically clean adaptation); (c) LoRA/partial-FT with LP-FT warmup. No naive full FT.

---

## 1. Abstraction levels — tasks, horizons, timesteps

| Level | Horizon / Δt | TASK (what it alone decides) | Predictor | Grounding |
|---|---|---|---|---|
| **Strategic** | 10–30 s / 1 s | Sets the GOAL: route intent, **target-speed band + source**, lane objective. Reads navigation + scene (signs, road class, traffic flow). | coarse: predicts progress/lane/traffic-density evolution under a strategic goal | route-level metrics |
| **Tactical** | 2–8 s / 0.5 s | Proposes and shapes PLAN OPTIONS: longitudinal mode × lateral maneuver × parameters (headway, lateral offset). Does NOT pick the winner. | mid: predicts scene evolution *conditioned on a tactical goal token* — "what if lane_change_left @ headway 1.45 s?" | maneuver-level metrics |
| **Operative** | 0–2 s / 0.1 s | Predicts CONSEQUENCES of concrete action sequences (incl. lead-gap evolution); executes the chosen plan as trajectory + controls. | fine: action-conditioned latent transition (KEEP — 0.427 m grounded, proven) | step-displacement grounding (KEEP) |
| **Planner** (new, cross-level) | — | EVALUATES tactical options via operative/tactical rollouts against the strategic goal cost; outputs one safe/smooth/rule-conform trajectory. | — | closed-loop metrics |

Top-down conditioning: each predictor is conditioned on the level above's **goal token** — with the H26
lesson enforced: gated/ReZero conditioning (learned scale, init small), goal-dropout during training, and a
**causality eval** (swap the goal token → the conditioned output must change appropriately; measured, not
assumed).

---

## 2. The Goal Vocabulary — **FROZEN as v1, see `V3_GOAL_VOCABULARY_V1.md`** (110 tokens / 17 slots; the five open questions decided there). The draft below is retained for provenance.

Design rules: (R1) every token is **discrete/banded** (VLM-labelable, tokenizable, no continuous leakage);
(R2) compositional — a goal is a short token tuple, not a monolithic class; (R3) each token carries a
**provenance stamp** (kinematic-auto / VLM / human / map); (R4) the tactical vocabulary doubles as the
**planner's proposal set** — the vocabulary IS the option space; (R5) shared across dataset labels, CoC
reasoning traces, conditioning embeddings, and eval — one vocabulary everywhere.

### 2.1 Strategic goal = ⟨ROUTE, VTARGET, VSOURCE, LANEOBJ⟩
| Slot | Tokens (draft) | Notes |
|---|---|---|
| ROUTE | `follow` `turn_left` `turn_right` `straight` `exit_right` `exit_left` `merge` `u_turn` `roundabout_N` | extends current 3-class route; curvature-relative per labels-v2 |
| VTARGET | 16 bands × 2.5 m/s: `v[0-2.5)` … `v[37.5-40)` | **the target-speed representation** — banded, never a raw float |
| VSOURCE | `sign_limit` `road_class_default` `traffic_flow` `nav_profile` `unknown` | *why* this target — teaches sign→speed extraction |
| LANEOBJ | `keep` `prefer_left_faster` `prefer_right_exit` `any` | couples lane choice to speed/route desire |

### 2.2 Tactical goal = ⟨LONMODE, LATMANEUVER, HEADWAY, PARAM⟩
| Slot | Tokens (draft) | Notes |
|---|---|---|
| LONMODE | `free_cruise` `follow_lead` `close_gap` `open_gap` `decel_soft` `decel_hard` `stop_at_point` `hold_stop` `launch` `creep` `coast` | the longitudinal mode set — IDM free-flow↔following made explicit + stop-and-go states |
| LATMANEUVER | `lane_keep` `lc_left` `lc_right` `abort_lc` `merge_in` `yield_merge` `nudge_left` `nudge_right` `pull_over` | lane-change is a *tactical goal* serving LANEOBJ/ROUTE |
| HEADWAY | bands: `0.8s` `1.2s` `1.45s` `1.75s` `2.5s+` | openpilot/ACC-anchored time-headway |
| PARAM | small banded set per maneuver (e.g., lateral offset band, gap-target band) | keeps tuples short |

### 2.3 Operative
Continuous (trajectory + steer/accel + v0 channel) — NOT tokenized for execution. A coarse discretized action
set (steer × accel bins) exists ONLY as the CEM sampling space.

### 2.3b Vocabulary v0.2 — Sayed's capability model folded in (2026-07-19 brainstorm)

**Scientific anchoring.** Sayed's three-level capability model is a modern instantiation of **Michon's (1985)
driver-task hierarchy** — strategical (route/trip planning, minutes+), tactical/maneuvering (seconds),
operational/control (sub-second) — and of **Donges' Navigation / Bahnführung / Stabilisierung**. What his
model ADDS beyond Michon/Donges, and what makes it novel in a WM context: (1) **driving style as a strategic
goal**, (2) **ODD/mission management incl. MRM** (SAE J3016 minimal-risk-maneuver / UNECE R157), (3) **risk
management as graded behavioral degradation** (ISO 21448 SOTIF triggering-condition handling), (4) **rule
deviation as a deliberate tactical act** (Censi et al. *Rulebooks*: a PRIORITY LATTICE of rules, where
collision-avoidance outranks lane-markings — "break the rule to avoid the accident" is lattice-consistent,
not lawless), (5) **communication as first-class tactical action** (interaction-aware planning: Sadigh et
al. — actions chosen partly to shape other agents' beliefs; Schwarting et al. social value orientation).

**New STRATEGIC slots (v0.2):**
| Slot | Tokens | Grounding |
|---|---|---|
| MISSION | `route_follow` `free_navigate(dir)` `explore` `plan_pullover` `mrm_now` | route-less navigation is a strategic capability; MRM = overriding goal (J3016/R157) |
| STYLE | `max_availability` `comfort` `eco` `dynamic` `degraded_caution` | **implemented as a planner cost-weight PRESET** — style token ⇒ weight vector (w_v, w_c, headway floor, maneuver set), so "change driving style" = re-parameterize the cost, zero new heads |
| RISK | `nominal` `elevated_weather` `elevated_visibility` `elevated_anomaly` | graded degradation: scales cost weights (↓v_target cap, ↑headway floor, restrict LATMANEUVER set) AND ↑planner budget |
| ODD | `in_odd` `odd_exit_ahead(T)` `capability_degrading` | predictive ODD monitoring feeds MISSION transitions |

**New TACTICAL slots (v0.2):**
| Slot | Tokens | Notes |
|---|---|---|
| RULECTX | `conform` `justified_deviation(reason)` | tokenized rule-break with reason + provenance (cross solid line to avoid obstacle; temporary human-like limit-leaving). Rulebooks lattice arbitrates in the planner cost |
| SIGNAL | `indicator_L/R` `hazard` `headlight_flash` `horn` `none` | communication ACTS, output in parallel with motion; behavior-based intent communication (early lateral nudge = announced cut-in) is a PARAM of LATMANEUVER |
| INTERACT | `yield_to_k` `assert_gap_k` `cooperate_merge_k` `respond_emergency_vehicle` | agent-referenced (needs tracked-agent slots — v3.0 keeps k∈{lead, merger}; full tracking v3.1) |
| TACPOINT | `stop_line` `merge_point` `creep_point` `clear_point` | "set the tactical points" — discrete spatial anchors the plan must satisfy (traffic-light stop-line, occlusion creep-point) |
| LIGHTSTATE | reaction tokens for traffic lights/signs: `proceed` `stop_at(stop_line)` `prepare_stop` `creep_check` | |

**Two syntheses worth highlighting:**
- **Risk budget:** the strategic layer owns a scalar risk budget; RISK/STYLE tokens set it; the tactical
  planner SPENDS it (tighter gaps, higher dynamics cost more). Degradation = shrinking the budget. This makes
  "focus on the most important aspects" quantitative: attention/compute allocation follows the budget.
- **"Stop to think" = anomaly-triggered deliberation:** `elevated_anomaly` doesn't just slow down — it
  RAISES THE PLANNER BUDGET (more CEM samples, longer horizon, finer options) while lowering speed; if
  uncertainty stays high → `creep`/`hold_stop` → escalate to `plan_pullover`/`mrm_now`. Deliberation-on-
  demand is a native strength of planner-based (vs head-based) architectures — a v3 EDGE argument.

### 2.4 Open vocabulary questions (for iteration NOW)
V1 band width 2.5 m/s — right? (finer at low speed, e.g. 1 m/s below 10 m/s?) · V2 LONMODE: is 11 too many
(merge decel_soft/hard into close_gap params?) or missing states (`overtake_commit`?) · V3 do we need an
INTERACTION slot (`yield_to_agent_k`, `gap_accept_k`) referencing tracked agents, or defer to v3.1? ·
V4 tokenization: per-slot codebooks vs one shared vocab with slot prefixes (VLM-friendlier)? · V5 sequence
goals (e.g., `lc_left THEN exit_right`) — chain tokens or single compound?

---

## 3. The longitudinal solution (three reinforcing pieces)

**Diagnosis recap:** models regress to a mean speed because nothing tells them what speed *should* be held
(v0 says where you are, not where you ought to be), and single-trajectory L2 collapses multimodal
speed-profiles to their conditional mean (survey: established).

**(1) Dataset — target speed as a LABEL (Sayed's ask, done safely).** Mint per-window:
- `v_target_band`: robust future free-flow speed = 85th-percentile of driven speed over the next 10–20 s
  **in unconstrained segments**, capped by sign/map limit where known → banded. Provenance-stamped.
- `v_source`: sign-read (VLM pass on frames) / road-class default / traffic-flow / unknown.
- `lonmode` label: classified from kinematics + lead state (free_cruise / follow_lead / stop / launch …).
- `lead` state: presence, gap (m), closing speed, headway (s) — **requires a perception-enrichment pass**
  (lightweight detector + monodepth or the Qwen3-VL/Cosmos-Reason2 pipeline) since our caches store only
  frames+poses+actions. This is a TanitDataSet enrichment column, same pipeline as scene tags/CoC.
- **Anti-shortcut safeguards:** v_target enters training as (a) the *prediction target* of the strategic
  level (learn to infer it from scene+nav), and (b) the *conditioning input* to tactical/operative **as the
  strategic layer's token** — banded, goal-dropout ~0.5, never the raw future-realized speed. At inference
  the strategic layer *produces* it, so nothing is fed that the car wouldn't know.

**(2) Architecture.** Strategic outputs ⟨VTARGET, VSOURCE⟩ from vision+nav (this IS "extract set speed from
signs" as a learned task); tactical chooses LONMODE conditioned on it; operative WM predicts consequences
including lead-gap evolution (needs the lead-state label to ground the gap channel).

**(3) Planner cost (where the modes toggle emergently).**
`J(plan) = w_v·(v̂ − v_target)² + Barrier(gap, TTC, headway) + w_c·(jerk² + |accel|²) + w_p·progress
+ Rules(speed-limit cap, lane legality, signals) + w_g·strategic-goal-consistency`
IDM's insight (survey-established): free-cruise ↔ follow-lead needs NO mode switch at execution — the
target-speed term and the gap barrier compete and the active constraint wins. The LONMODE tokens are for
*proposal seeding, labeling, and interpretability* — the cost arbitrates. Lane-change-for-speed emerges:
a `lc_left` plan scores better on (v̂−v_target)² when the ego lane is blocked → chosen "because no better
alternative," exactly as specified.

---

## 4. The planner/decoder (the new component)

1. **Strategic pass** → goal ⟨ROUTE, VTARGET, VSOURCE, LANEOBJ⟩ (from nav + scene).
2. **Proposal generation** → K tactical options compatible with the goal: vocabulary-enumerated
   (LONMODE × LATMANEUVER × HEADWAY, pruned by context) + the **anchored multi-mode decoder reused as a
   learned proposal prior** (its K modes = warm-start trajectories — nothing thrown away).
3. **Consequence rollout** → per option: operative WM fine rollout (0–2 s) + tactical predictor coarse
   extension (2–8 s). Feature-space rollout for refa-v3 arms (DINO-WM recipe); latent rollout for flagship-v3.
4. **Scoring** → J(plan) above. Optimizer: **CEM v3.0** (simple, DINO-WM-faithful, debuggable) →
   **Diffusion-ES v3.1** (SOTA for non-differentiable driving rewards; DiffusionDrive experience via REF-C
   transfers).
5. **Output** → best plan → smooth, rule-conform trajectory + actions (jerk-limited, speed-limit-capped)
   → closed-loop feedable (imagination-in-the-loop now, AlpaSim when hosted).

## 5. Evaluation redesign (H26 made falsifiable)
- **Counterfactual plan-ranking**: does the chosen plan beat the non-chosen options on realized outcome?
  (the direct test of "evaluating alternatives" — new, primary)
- **Goal-causality**: swap strategic tokens → measure the induced change in tactical proposals/plan choice
  (kills the echo pathology; must be CI-separated).
- **v_target tracking**: |v̂ − v_target| profile + mode accuracy vs labels (longitudinal-specific).
- **Closed-loop drift** (existing harness) per arm; open-loop ADE stays but demoted (survey: it hides
  longitudinal collapse).

## 6. Keep / replace (incrementality contract)
KEEP: operative predictor + step grounding (0.427 m) · v0 speed channel · JEPA+SIGReg · anchored multi-mode
decoder (→ proposal prior) · decorr lever · labels-v2 curvature logic · TanitEval harness + viz standard.
REPLACE: unimodal heads as deciders → planner · route-echo strategic head → strategic goal module ·
ungated intent seam → gated goal conditioning. DROP: REF-A anchored-decoder retrain (ceiling is the encoder).

## 7. Build order (proposal)
P0 data-enrichment spec (lead state, v_target, sign-reads, mode labels — VLM+detector pipeline on val first)
→ P1 vocabulary freeze after iteration → P2 planner skeleton on TOP OF frozen v1/v2 (CEM + cost over the
existing operative WM — measurable win without any retraining!) → P3 goal-conditioned tactical predictor
(train) → P4 refa-v3 matrix + flagship-v3 (full recipe) → P5 Diffusion-ES upgrade + closed-loop eval.
P2 is the key de-risk: the planner is testable on today's checkpoints before we train anything new.

---

## 8. Architecture spec v3.0 (tight — both arms)

**Shared spine (flagship-v3 and refa-v3 differ ONLY in the encoder + transition objective):**

| # | Module | Spec | Trained on |
|---|---|---|---|
| M1 | **Encoder** | flagship-v3: trained ViT (v2 lineage: JEPA+SIGReg+decorr, staged levers per the 10k-gate findings). refa-v3 matrix: (a) frozen DINOv2 (control) (b) own-SSL-then-freeze (c) LoRA+LP-FT | SSL (+aux) |
| M2 | **Strategic module** | predicts strategic tuple ⟨MISSION…ODD⟩ from (pooled latent, nav, ego); MISSION/ODD/RISK transitions wrapped in an **engineered safety envelope** (MRM triggering is scaffolding, not learned) | CE on minted labels; VTARGET is its key head |
| M3 | **Tactical proposal generator** | vocabulary enumeration conditioned on the strategic tuple (context-pruned) **+ the anchored multi-mode decoder as learned proposal prior** (K≈10 warm-start trajectories — validated by REF-B v2 0.646) → M ≈ 8–24 candidate plans (tactical tuple + seed trajectory) | anchored decoder: WTA+cls (existing) |
| M4a | **Operative predictor** | action-conditioned transition, 0–2 s @ 0.1 s. flagship-v3: latent + step-displacement grounding (KEEP, 0.427 m). refa-v3: **feature-prediction objective** (latent-MSE, DINO-WM) + separately-calibrated geometric readout | self-supervised transition |
| M4b | **Tactical predictor** (new) | goal-token-conditioned coarse transition, 2–8 s @ 0.5 s on pooled latents; predicts scene/gap/lane evolution under a tactical tuple | feature-prediction + gap/lane aux |
| M5 | **Cost** | J(plan) = lexicographic Rulebooks lattice: **L0 safety** (collision/TTC/gap barrier) ≻ **L1 rules** (speed-limit cap, lane legality, LIGHTSTATE; RULECTX-justified deviations re-rank inside L1) ≻ **L2 mission** (w_v·(v̂−VTARGET)² + progress + goal-consistency) ≻ **L3 comfort** (jerk/accel/DYN). STYLE/RISK tokens select the weight preset + risk budget; hysteresis: plan-switch cost δ | weights engineered, presets tuned on val |
| M6 | **Optimizer** | v3.0: **CEM** — per-proposal warm start, N=64 samples, 3 iterations, elite-8, horizon 2 s fine + 8 s coarse; `elevated_anomaly` ⇒ budget ×4 (deliberation-on-demand). v3.1: Diffusion-ES upgrade | — |
| M7 | **Output head** | best plan → jerk-limited, rule-conform trajectory + [steer,accel] + SIGNAL acts; closed-loop feedable | — |

**Conditioning flow (top-down, all gated):** strategic tuple —ReZero(0.1)+dropout 0.5→ M3/M4b;
tactical tuple —same→ M4a rollout context. Norm-parity monitored (H26). Bottom-up: M4 rollouts return
predicted consequences to M5; no head decides anything.

**Losses:** L_M1 (SSL per arm) + L_M2 (slot-CE, VTARGET-weighted) + L_M3 (WTA+cls) + L_M4a
(transition + grounding) + L_M4b (coarse transition + aux) — **no trajectory-imitation loss on the
planner path** (the planner is not trained; it is evaluated). Imitation lives only in the proposal prior.

**Eval gates (per §5):** G1 counterfactual plan-ranking beats proposal-prior-argmax by CI-separated
margin · G2 goal-causality per seam · G3 v_target tracking |v̂−VTARGET| · G4 closed-loop drift < head
baseline (1.69 m) · G5 open-loop non-regression vs v1 (0.452 m) on the flagship arm.

**P2 de-risk instantiation (build FIRST, this week):** M5+M6 (cost+CEM) over the **frozen v1 flagship**
operative WM, proposals = v1's anchored decoder modes + vocabulary enumeration, VTARGET minted offline
for val. Measure G1/G4 against the 3.38 m tactical head and 1.69 m closed-loop drift. A win here
validates the entire v3 thesis at zero training cost.
