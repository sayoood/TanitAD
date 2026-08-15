# Campaign science addendum — the seven topics the consolidation report pointed at but did not extract

**Written 2026-08-15 ~22:00 CEST**, at the PI's request: *"Im missing a lot of other results like
the tactical/strategic hierarchy vocabulary, the Alpamayo 2 super quantization and inference,
Augmentation of data based on Alpamayo and later based on vlm/sam3, the combination of diffusion
planner with MPC, the learnings from alpamayo action space and trajectory representation etc, etc.
I want you to extract all these information, look also in the updated scientific paper of the
programme."*

**Companion to** `Project Steering/Reports/2026-08-15-2100-program-report.md`, which covered the
training line (v5f/v1arch finals, Stage-A, v5.8f/W-wedge, P-battery echo, G1, PH1 fusion counts,
Orbis-2 headline). **This document is the missing extraction** and is written to be read WITHOUT
opening the sources: every number is verbatim, carries its evidence class, and names the file that
owns it.

**Reading rules in force.** `Project Steering/MODEL_REGISTRY.md` + raw JSON/parquet are the only
quotable sources for model facts; design docs own their own design decisions. Capability numbers
carry their **T-tier** (`Project Steering/EVAL_DOCTRINE.md`: T0 = teacher-forced WM diagnostic,
never driving performance · T1 = action-closed loop, the primary offline tier · T2 = perception-
closed loop, still missing). Intervals name their estimator. Where two documents disagree, the
disagreement is **recorded, not smoothed**.

---

## 1. THE HIERARCHY VOCABULARY — the actual token sets, their sizes, and how they bind to v6

**Owner:** `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-07-hierarchical-wm-redesign/HIERARCHY_VOCABULARY.md`
(the handover cites it without a path; it is in the `2026-08-07-hierarchical-wm-redesign` folder,
alongside `HIERARCHICAL_WM_REDESIGN.md`). Evidence class: **DESIGN — a PI-directed specification**,
not a measurement. The doc's own header calls it "vocabulary v0 = THIS DOC".

### 1.1 The design rules that generate the vocabulary (binding, verbatim from §2)

- **"Tokenizable = finite discrete head + typed continuous slots."** Every token is `(TYPE, [args])`;
  args are physical units (m, s, m/s). **"No free text at inference."**
- **"Factored LAT × LON everywhere** (the 5-way-mixed-softmax defect is retired by design)."
- **Goal/situation disjointness** (PI 2026-08-03): no goal token may be derivable from the situation
  classifier's output; provenance tags (`path|signage|vlm-fused`) travel with every supervised instance.
- **"Labels supervise GOAL/INTERPRETATION HEADS only — never any WM trunk loss"** (JEPA thesis; the
  aux-label retraction stands).
- **"Every token must be HINDSIGHT-DERIVABLE from Engine A geometry alone"** (VLM enriches; geometry
  guarantees) — so the vocabulary works even where the VLM abstains.
- Every goal token carries **optional constraint slots** `within_m` / `by_time_s` / `at_arc_m` /
  `hold_for_s`; unset = unconstrained.
- ⭐ **Near-field is TIME-scaled, not metre-scaled** (PI correction): *"a fixed 40 m band cannot cover
  a 6 s horizon (180 m at 30 m/s)."* All near-field weighting uses **time-to-reach** = `arc_length /
  v_ego`, capped at the 6 s horizon.

### 1.2 STRATEGIC vocabulary `g_str` — **11 goal tokens** (§3)

| token | args | derivation |
|---|---|---|
| `KEEP_CORRIDOR` | `target_arc_m` | hindsight path curvature-relative follow |
| `LANE_TARGET` | `lane_offset_idx, deadline_m` | lateral displacement events (E4.1 LAT) |
| `EXIT_RIGHT` / `EXIT_LEFT` | `distance_m` | corridor split geometry |
| `TURN_LEFT` / `TURN_RIGHT` / `STRAIGHT_THROUGH` | intersection arc | E7.1 turn events |
| `ROUTE_TO` | `text_token_id` (city/POI vocab from OCR), `evidence_id` | **ONLY with signage OCR (G1-gated); abstains otherwise** |
| `STOP_AT` | `distance_m` | signage/geometry (stop events in hindsight speed profile) |
| **`FOLLOW_MAIN_ROAD`** | — | **THE DEFAULT strategic goal whenever no navigation route is set up** (PI 2026-08-11) — the corridor-continuation prior; replaces `NONE_ABSTAIN` as the no-route baseline |
| `NONE_ABSTAIN` | — | honest ceiling (ambiguous geometry only) |

**Strategic actions `a_str` — 6 verbs:** `PREPARE_LANE_CHANGE(dir, within_m)` · `HOLD_CORRIDOR(arc_m)`
· `REDUCE_TO(v_target, within_m)` · `PREPARE_EXIT(dir, within_m)` · `PREPARE_STOP(within_m)` ·
`RESUME_CRUISE(v_target)`.

**The count is independently confirmed in code**: `PH0_COVERAGE_AUDIT.md` §2 reports
`GOAL_KINDS` and `ACTION_VERBS` in `ph0_v2.py:34-38` are a **"term-for-term match"** to §3's tables —
**11 `g_str` tokens, 6 `a_str` verbs** — and rates the strategic layer **"✅ COMPLETE"**.

### 1.3 TACTICAL vocabulary `g_tac` — **9 goal tokens** (§4)

| token | args | grounding |
|---|---|---|
| `ANCHOR_GOAL` | `anchor_id ∈ fan vocab, t_reach_s` | the geometric goal-point lever (**"the +4.7 PDMS class"**) |
| `CORRIDOR_OFFSET` | `lat_offset_m, arc_m` | curvature-relative corridor frame |
| `GAP_TARGET` | `agent_slot_id, time_gap_s` | from the perception head's agent slots |
| `SPEED_BAND` | `v_lo, v_hi` | LON axis — **"SET BY THE TACTICAL LAYER (PI decision 2026-08-11): target speed is a tactical responsibility, computed from traffic-sign inputs (VLM/OCR speed-limit fields) and prior speed information (corridor speed statistics), bounded by the strategic layer's `REDUCE_TO` only as an upper envelope"** |
| `YIELD_AT` | `position_arc_m, gap_slot` | PI addition: merge/roundabout/unprotected turn |
| `STOP_POINT` | `position_arc_m, reason ∈ {sign, light, queue, hazard}` | PI addition |
| `WAIT_FOR_ONCOMING` | `narrow_arc_m, oncoming_slot` | PI addition: narrow-road negotiation |
| `EVADE_IN_CORRIDOR` | `lat_offset_m, obstacle_slot, past_arc_m` | PI addition: in-corridor evasion around doors/cyclists/parked vehicles — **"bounded by corridor, NOT a lane change"** |
| `TRAFFIC_LIGHT_REACT` | `light_slot_id, state ∈ {red, yellow, green}, stopline_arc_m` | PI addition; light state from VLM/signage fields — **"eval + goal-head supervision; never trunk"** |

**Tactical actions `a_tac` — factored on two axes, 11 verbs total:**
- **LAT (6):** `LANE_KEEP` · `LANE_CHANGE_L/R(within_m)` · `ABORT_LC` · `NUDGE_L/R(lat_m)`
- **LON (6):** `FOLLOW(time_gap_s)` · `CRUISE(v)` · `YIELD/MERGE(gap_slot)` · `BRAKE_TO(v, within_m)` · `CREEP` · `HOLD`

**Operative layer keeps continuous `(a, κ)` unicycle controls (W4/W4r-proven) — "no tokenization at 10 Hz."**

### 1.4 ⛔ The measured gap between the vocabulary and what is actually extracted

**Source:** `…/2026-08-07-hierarchical-wm-redesign/PH0_COVERAGE_AUDIT.md`, evidence class **MEASURED
(read from source, file:line)**, answering the PI's 2026-08-12 question *"Did you include all the
information we want to extract like scenario etc"*. **Its verdict, verbatim: "Answer: NO. Strategic
is complete; tactical is largely empty; and the SCENARIO CLASS the question names is genuinely absent."**

- **`HIERARCHY_VOCABULARY.md` §4 defines nine `g_tac` tokens. PH0 emits none of them.** B4 emits only
  strategic goals; **"there is no tactical call at all."**
- Per token: `ANCHOR_GOAL`, `CORRIDOR_OFFSET`, `SPEED_BAND` are ⚠️ *derivable but not assembled*;
  `GAP_TARGET`, `YIELD_AT`, `STOP_POINT`, `WAIT_FOR_ONCOMING`, `EVADE_IN_CORRIDOR` are ⛔ **NO**;
  `TRAFFIC_LIGHT_REACT` is 🟡 PARTIAL (state only, no slot id, no stopline distance).
- ⭐ **"The single root cause: PH0 extracts NO AGENTS."** Five of the nine tactical goals need an
  agent/obstacle slot. This is also why the binding four-family **LONGITUDINAL** family (headway /
  time-gap / TTC) cannot be computed — **"88.7 % of the oracle gap is longitudinal."**
- Two unwired supplies exist: **SAM3 text-prompted detection** (added 2026-08-12) and
  **`obstacle.offline` — "3D agent cuboids on 97.44 % of the corpus, 87,481 cuboids over 10 dynamic
  classes"**, of which *"our ingest reads 4 of 36 features"*. Called **"already-paid-for ground truth
  we are not reading — the cheapest fix in the table."**
- `a_tac` LAT's **`ABORT_LC` has no detector**; `a_tac` LON `FOLLOW(time_gap_s)` needs a lead agent —
  **"LF0 RC1 (the zero-parameter geometric lead read) was REFUTED, so distance-keeping still has no
  instrument."**
- **SCENARIO/SITUATION is ⛔ ABSENT** and is exactly what the PI asked about. Classes are
  `lane_change` / `intersection` / `roundabout`, defined in `stack/tanitad/data/situations.py`
  (thresholds frozen by the 2026-07-26 prereg; roundabout deliberately unpowered at n=26). B1's
  `road_type`/`domain` are **static scene descriptors, not the dynamic situation** — *"A clip driving
  straight through a junction and a clip turning left at one get the same B1 `domain=intersection`
  and are different situations."* Label side is already solved algorithmically (a deterministic
  function of the ego pose track, `emit_situation_labels.py:54-62`); the **vision-only read** is the
  missing piece, and if added it must be **information-disjoint from B4**.

**Priority order to close it (cheapest first, from §6):** 1. wire `obstacle.offline` → agent slots
(no GPU, no PI decision) · 2. Engine-A situation labels (free, exact) · 3. B5 tactical + B6 vision-only
situation calls · 4. SAM3 `--mode text` tracks · 5. range from intrinsics · 6. `ABORT_LC` detector.

### 1.5 The binding 6-second horizon (§4b) — the clause that reshaped v6

**"Every planned trajectory spans up to 6 s — covering BOTH the operative and the tactical horizon
in one kinematically consistent rollout."** Concretely (verbatim):
- **"a 60-step control sequence (a, κ) @10 Hz integrated through ONE unicycle rollout 0→6 s — never
  two stitched trajectories."** 0–2 s is the operative band, 2–6 s the tactical band; *"the shared
  integrator makes the 2 s seam discontinuity-free BY CONSTRUCTION (the X2 seam metrics verify, not
  repair)."*
- The tactical layer therefore grounds at **2–6 s** — **"the earlier 2–8 s note is superseded"**
  (a self-recorded internal correction: §4 of the same document still carries the 2–8 s heading).
- Emission heads scale **k=20 → k=60**; the diffusion proposal generator diffuses the full 6 s control
  sequence; W7-style roll selection rolls to 6 s.
- **Eval consequence:** four families + oracle/selected reported at **BOTH 0–2 s and 0–6 s**; T1/P5
  compounding measured to 6 s. **E-H1/W5 is "promoted from queued to REQUIRED precursor"** — it
  baselines v5.8f at 6 s before v6 trains against it. *(Status: still open — see §7 and §8.)*

### 1.6 The wiring (§5) — "goals flow DOWN only; latents flow UP"

```
z_str_{t+K} = P_S(z_str_t, a_str_t)
z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)
z_op_{t+j}  = P_O(z_op_t, (a,κ)_t | g_tac)
```
Latents flow up through **stop-grad/EMA** (gradient-isolation matrix, `V6_TRAINING_MEASURES` X3).
Each level's selector is **roll-cost-based over ITS OWN predictor (the W7 pattern per level)**;
per-level `sel_gap` via E8.1. **"Token embeddings are shared between the goal-emitting head (above)
and the goal-consuming conditioner (below) — one vocabulary, two views."**

### 1.7 How it binds to the v6 staged trainer (S-T / S-S)

**Source:** `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINER_DESIGN.md`, evidence class **MECHANISM
+ MEASURED at instantiation**. Its own tier stamp: *"No number produced by this trainer is quotable
as driving performance."*

- **`V6Stack` implements §5 verbatim.** Three things are enforced **mechanically**, not by comment:
  1. **X3 gradient isolation** is *"a real autograd probe"* (`torch.autograd.grad(..., allow_unused=True)`)
     over three edges: `planner → encoder`, `tactical → below`, `strategic → below`. It temporarily
     makes every parameter differentiable **for the probe only**, because *"a frozen parameter records
     no autograd edge, so a check run mid-S-T would find the encoder 'isolated' simply because it is
     frozen — a vacuous pass, which is how an isolation guarantee rots."*
  2. **One vocabulary, two views** — the goal-token table is *"the same `nn.Module` object"* in the
     emitting head and the consuming conditioner, **"pinned by `is` identity, not equality."**
  3. **§4b seam-free by construction** — `V6Config` **refuses** a band gap or overlap at construction.
- **S-T** (`--stage S-T`, 10,000 steps, `--w-t1 1.0`, `λ_plan` defaults 1.0) trains `layer_tac`
  (adapter, `P_T`, `goal_head_tac`, **factored LAT/LON action heads**, `vocab_tac`, `vocab_a_lat/lon`)
  and the planner; **everything below frozen** — *"this is Drive-JEPA's shape (the planner is a
  post-trained consumer)"*. Plan loss reports `plan_ade_0_2s` and `plan_ade_2_6s` **separately**,
  *"because a pooled 0–6 s number cannot see the seam."* ⚠️ The tactical target sits one **tactical**
  tick ahead (`stride_tac = 5`): *"Predicting one operative tick ahead and calling it a tactical
  prediction is an identity map wearing a hierarchy's name."*
- **S-S** (8,000 steps, `--w-s1 1.0`) trains `layer_str` only; `stride_str = 20`. ⚠️ **"S2 (`g_str`
  supervision) is not wired here and must not be faked. It arrives from the PH0→PH1→PH2 VLM/geometric
  pipeline. Until it lands, S-S trains the strategic latent prediction (S1) only, and the STRATEGIC
  metric family is reported as `n/a` with its reason and its n — never silently dropped."**
- **Stage gates** (pre-registered): S-T requires the **TACTICAL family** + `sel_gap ≤ 0.5× the fan
  oracle at T1 tier` and TACTICAL confusion improving on E4.1-derived strata; S-S requires the
  STRATEGIC family **"computable at all (measured vs `n/a` today)"** and S1 ADE(8–30 s) beating
  CV/corridor baselines at T1. **Three verdicts, and `pass: null` ≠ `pass: true`: "INCONCLUSIVE IS
  NOT A PASS."** A `pass: false` is **"REFUSED, and there is no override."**
- ⚠️ **`--init-from` is REQUIRED** for S-T/S-S/S-J: *"A gate saying 'S-W passed' is worthless if this
  stage then trains on a randomly-initialised trunk — that is not the staged protocol, it is four
  unrelated models with a gate between them."* Load is `strict=True`; the config records the **md5 of
  the loaded trunk**.

### 1.8 The relation to D-TAC1 (the early-August factored-tactical line)

The vocabulary's *"Factored LAT × LON everywhere"* rule is the design-side answer to a defect that
D-TAC1 measured in code and in evals. From `Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md`
(evidence class **MEASURED**, file:line cited per row):

- **"REF-C emits one 5-way softmax: `N_MANEUVERS = 5` over `(lane_keep, turn_left, turn_right,
  accelerate, brake_stop)` — 3 lateral + 2 longitudinal classes in one mutually-exclusive simplex"**
  (`stack/tanitad/refs/refc.py:100`).
- The label is minted by a **PRIORITY collapse** of two orthogonal axes, `turn > brake > accel >
  lane_keep` (`stack/scripts/refb_labels.py:100-109`, `:339-347`).
- On canonical val at **n = 859 windows / 39 episodes** the head predicts **`accelerate` 0 / 93** and
  **`brake_stop` 7 / 78**; `never_predicted: ["accelerate"]`.
- It is **INVARIANT to `nav_cmd`** (all four nav arms give a bit-identical manoeuvre histogram) and
  **"the manoeuvre head never sees the ego speed."**
- Class balance: **steady 510 / brake 78 / accel 93 = 74.9 % / 11.5 % / 13.7 %**.
- ⛔ **The prereg's own §6.3 claim was retracted by the measurement**: *"READOUT-limited ⇒ F2+F3
  sufficient"* is marked **"TOO OPTIMISTIC"** — F2 alone (τ=0) gives brake recall **0.072** / accel
  **0.045**; F3 lifts `brake_stop` to **0.503** at τ=0.5 with no retrain, but **`accelerate` never
  exceeds 0.153 at ANY τ**. **Revised lever ordering: measured F3 > F2 > F1** (was F1 > F2 > F3).
  Results live in `…/incoming/2026-08-03-dtac1-tactical-head/DTAC1_RESULTS.md` + `DTAC1B_RESULTS.md`.

⇒ **The vocabulary does not merely re-state the defect: it removes it as representable.** The 5-way
mixed simplex is replaced by `a_tac` LAT (6) × LON (6) with severity carried in continuous envelope
args — and §1.9 below shows a 34 B production system already factorises this way.

---

## 2. ALPAMAYO-2 SUPER — QUANTIZATION AND INFERENCE

**Owners:** `TanitAD Research Hub/Benchmarks & Eval/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md`
(the campaign's single richest result document, 764 lines, and **not referenced by path anywhere in
the handover** — the handover points only at "chronicle rows in `PROJECT_STATE.md`", where **no A2
rows exist**; see §2.6) + `…/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md`
+ the HF dataset `Sayood/tanitad-alpamayo2-augmentation`.

### 2.1 What the model is (PUBLISHED, primary sources retrieved 2026-08-05)

| | |
|---|---|
| **Total parameters** | **34.3 B** = 32 B VLM backbone + **2.3 B** action expert |
| **Backbone** | **Qwen3-VL** (`vlm_class: Qwen3VLForConditionalGeneration`), branded **Cosmos 3 Super Reasoner** |
| **On-disk** | **71.65 GB**, 15 safetensors shards, 32 files, `dtype: bfloat16` |
| **Licence** | weights **OpenMDW-1.1** (permissive, **commercial redistribution allowed**); code Apache-2.0 |
| **Gating** | HF API reports **`gated: false`** ⚠️ while the GitHub README still says *"gated"* — **the README is stale on this point** (a recorded source disagreement) |
| **Training data** | **≈115,000 h** video · **>1 billion** images · **≈3,700,000** Chain-of-Causation traces |
| **Published results** | LingoQA (Lingo-Judge) **79.2** · AlpaSim closed-loop, 910 NuRec scenarios **1.50 ± 0.13** · open-loop **minADE₆ @ 6.4 s = 0.911 m** on 1,434 challenging PhysicalAI-AV samples |
| **NVIDIA's own memory profile** | **72,115 MiB** peak device memory on **1 × H100 80 GB**; *"Other GPU architectures have not yet been validated."* |

⚠️ **`minADE₆` is best-of-6** — *"a best-of-6 oracle-selection metric, not a single-shot error"*,
explicitly the same distinction as our `oracle_ade` vs shipped ADE.

### 2.2 ⭐ THE QUANTIZATION RESULT — 34.3 B runs on a 46 GB A40 at **25.84 GiB peak**

**MEASURED 2026-08-05, pod4 (A40 46,068 MiB)**, `ALPAMAYO2_SUPER_ANALYSIS.md` §10. Verbatim log:

```
[quant] NF4 backbone · BF16 skip-list ['visual','lm_head','expert','action_in_proj','action_out_proj'] · attn=sdpa
[quant] Linear4bit modules: 448
[quant] weights resident: 23.58 GiB
Chain-of-Causation: "Nudge left to avoid the cones on the right side."
minADE: 1.4222202 meters
[quant] PEAK device memory: 25.84 GiB (A40 capacity 45.0 GiB)
A2_RUN_RC=0
```

- **72,115 MiB (H100, NVIDIA) → 25.84 GiB (A40, ours) = a 2.79× reduction**, and *"the model produces
  correct, semantically grounded output."*
- **What was quantised:** `vlm.model.language_model.*` only — **448 `Linear4bit` modules**, NF4
  double-quant with bf16 compute. **Kept BF16:** `vlm.model.visual.*`, `vlm.lm_head`, and **all of
  `expert.*`** including `action_in_proj`/`action_out_proj`.
- ⛔ **The stated reason, verbatim:** *"Quantising the action expert would have been the wrong saving.
  It is the module that emits the trajectory; 4-biting 2.3 B to save ~3.5 GB, when the 31 B backbone
  is where the memory actually sits, corrupts the measured output to buy almost nothing."*
- **Semantic validation, not just a non-crash:** the clip (`030c760c-ae38-49aa-9ad8-f5650a545d26`,
  `t0_us 5,100,000`) is a roadworks scene; *"cones are visible in the cross-right, front-wide and
  rear-right views, and the predicted path nudges left. The model reasoned about the correct object
  and acted on it."*
- **Exactly one line of NVIDIA's code is patched** — `inference_smoke.py:133`.
- ⛔ **Admissibility label, binding: `QUANTISED-4BIT-UNVALIDATED`.** *"The 1.4222 m minADE here may
  NOT be compared to their published 0.911 m"* — different estimator (minADE₆ vs 1 sample), different
  denominator (1,434 curated vs 1 clip), different precision. **"Three reasons, any one of which is
  disqualifying."**
- **Practical inference notes (MEASURED):** Python **3.12** via `uv`; torch 2.8.0+cu128, transformers
  4.57.1, flash-attn 2.8.3 built from source, bitsandbytes 0.50.0. ⚠️ *"Do not override `HF_HOME`"* —
  doing so produced a `GatedRepoError 401` that *"looks like a permissions problem and is actually a
  path problem."* ⚠️ **The venv on MooseFS costs ~6 minutes of import time per run** — put it on local
  disk. Load is **~18 s/shard × 15 shards ≈ 4.5 min** from MooseFS.

### 2.3 ⭐ THE QUANTIZED-VS-OURS COMPARISON — "the over-speed is OURS, not the data's"

**MEASURED 2026-08-06, 39 paired clips**, both truncated to a **2.0 s / 20-waypoint** horizon on the
same PhysicalAI OOD-val clips at the same nominal `t0 = 5.1 s` (§11 of the same doc).

| | Alpamayo 2 Super (NF4) | TanitAD flagship-v1arch |
|---|---|---|
| ADE @2 s | **0.2703 m** | 0.3303 m |
| **speed bias** | **+0.0569 m/s** | **+0.4245 m/s** |
| **frac faster than human** | **59.0 %** | **84.6 %** |
| along bias @2 s | +0.1132 m | +0.8176 m |
| frac ahead at 2 s | 59.0 % | 79.5 % |
| (Alpamayo native 6.4 s ADE) | 2.4026 m | — |

⇒ **"A 34 B model trained on ~115,000 h of the same-family data does NOT run ahead of the human. So
the bias does not live in the ground truth or the label convention — it is ours, and it is ours to
fix."** And the counter-reading: **"The ADE gap is 0.2703 vs 0.3303 — a factor of 1.22 — against
~115× the parameters and ~190× the input pixels."**

**How the GT was obtained and why it is trustworthy:** the trajectory capture returned empty GT, so
GT was reconstructed from the staged `egomotion` parquet and then **proven** — recomputing ADE between
Alpamayo's own captured prediction and the reconstructed GT reproduces the `min_ade_m` NVIDIA's code
printed **to within 0.0220 m (mean 0.0047 m) over 39 samples**; *"the check refuses to write a GT that
does not reproduce their metric."*

**Input profile, recorded per sample not asserted:** every one of the 40 runs carries
`image_frames_shape: [6, 4, 3, 1080, 1920]` — **6 cameras × 4 frames at native 1920×1080** — with
`camera_indices: [0, 1, 2, 3, 5, 6]`, NVIDIA's own `DRIVING_SIX_CAMERA_FOUR_FRAME` profile
(`src/alpamayo2_super/input_profiles.py:40-41`). The withheld camera (id 4, `camera_rear_tele_30fov`)
is **a VQA camera**: *"excluding it for trajectory IS the validated configuration. Giving it would
have been the deviation."*

⛔ **Caveats carried with every number:** n = 39, unweighted mean over clips (**not** an episode-cluster
bootstrap) · **NF4 quantisation is not an NVIDIA-validated configuration** · ⛔ **CONTAMINATION
UNRESOLVED** — *"these clips are PhysicalAI-AV, which Alpamayo lists as a TRAINING dataset. Any
advantage may be contamination rather than capability"* · not like-for-like on 5 axes · window origins
sit on an 0.8 s stride so |dt| ≤ 0.4 s.

### 2.4 The four families on the same 39 clips — **"only one of them favours the 34 B model"**

§12, computed by `taniteval.four_families` (not re-implemented). Estimator stated: **unweighted mean
over 39 paired clips, one window per clip** — ⛔ *"with one window per episode the episode-cluster
bootstrap degenerates to the i.i.d. case, so no CI is quoted here rather than a wrong one."*

**LONGITUDINAL — Alpamayo wins decisively:**

| | Alpamayo 2 Super | flagship-v1arch | ratio |
|---|---|---|---|
| speed MAE (m/s) | **0.3833** | 0.7050 | 1.84× |
| speed bias (m/s) | **+0.0569** | +0.4245 | 7.5× |
| **accel MAE (m/s²)** | **0.5077** | **1.7644** | **3.48×** |
| along bias (m) | +0.0315 | +0.1543 | 4.9× |
| along final bias @2 s (m) | +0.1132 | +0.8176 | 7.2× |
| ego-progress ratio (GT = 1.0) | 0.9836 | 1.0566 | — |
| under-progress rate | 0.3714 | 0.1714 | — |
| distance keeping (headway/time-gap/TTC) | **UNAVAILABLE** | **UNAVAILABLE** | — |

⭐ *"The new finding is `accel MAE`, not the speed bias. Our arm's acceleration profile is 3.48× worse
— a metric ADE cannot see at all."* ⛔ **`distance_keeping` UNAVAILABLE for both arms "is a WORK ITEM,
not a pass"** — the banked OOD-val lead block is keyed on our 0.8 s grid while Alpamayo ran at clip
`t0 = 5.1 s`, so **half of the binding LONGITUDINAL family is missing from this comparison**.

**LATERAL — ours is competitive and better on curvature:**

| | Alpamayo 2 Super | flagship-v1arch |
|---|---|---|
| heading MAE (deg) | **0.6794** | 0.7800 |
| yaw-rate MAE (deg/s) | 1.8285 | **1.7791** |
| **curvature MAE (1/m)** | 0.009162 | **0.007551** |
| cross-track MAE (m) | **0.0398** | 0.0493 |
| cross-track bias (m) | +0.0036 | −0.0241 |

⭐ *"A sub-0.3 B front-crop-only model matches a 34.3 B six-camera model on heading and yaw-rate, and
beats it on curvature MAE. The deficit that produced the ADE gap is NOT lateral"* — independent
confirmation of the programme's own "88.7 % of the oracle gap is longitudinal".

**STRATEGIC — UNAVAILABLE for BOTH arms, "and *that* is the finding."** Neither can be scored because
PhysicalAI-AV ships no map, lane graph, junction annotation or route signal (five independent probes;
the card says verbatim *"we do not include open maps data"*). Scoring against the ego's own future is
what produced flagship v1's `route_head_eq_logged = 1.0000`, *"an echo of its own nav input read as
skill"*. ⇒ **"The programme's central thesis — that the hierarchy works — remains unmeasured at its
top level, for us and for a 34 B reference system alike."**

**TACTICAL — ⛔ §12's block was RETRACTED by §14 (two instrument defects, both ours).**
1. Net yaw was summed over steps where the ego was not moving (one stopped window contributed **π**).
   Excluding steps below `MIN_DS_MPS = 0.5` **moved Alpamayo's executed-manoeuvre κ from 0.3333 to
   0.4882**. *"Third appearance of the same trap … a quantity that is undefined in a regime,
   aggregated over that regime, read as a measurement."*
2. The **0.15 rad direction gate is mis-scaled for 2 s windows**: on the human's own paths **median
   |net yaw| over 2 s is 0.023 rad, p90 0.185, and only 17.9 % of windows exceed the 0.15 gate**;
   `hierarchy.DIR_YAW_RAD` is *"~6.5× the typical turn."*

| gate (rad) | A2 declared κ | flagship declared κ | A2 executed κ | flagship executed κ |
|---|---|---|---|---|
| **0.15** (as published in §12) | 0.1961 | **0.4402** | 0.4882 | **0.6176** |
| 0.10 | 0.3004 | 0.3743 | **0.7292** | **0.7263** |
| 0.06 | 0.2639 | 0.2835 | 0.7222 | 0.7132 |
| 0.04 | **0.4059** | 0.2390 | 0.6277 | 0.6848 |
| 0.03 | **0.4553** | 0.1986 | 0.6926 | 0.5752 |
| 0.01 | **0.4660** | 0.1159 | 0.8077 | 0.3953 |

⛔ **RETRACTED: "our executed-manoeuvre κ 0.4968 beats Alpamayo's 0.3333."** At the best-matched gate
(0.10) the two arms are **indistinguishable — 0.7263 vs 0.7292**. ⭐ **The substantive replacement
finding: the two arms' declarations move in OPPOSITE directions as the gate tightens** — Alpamayo's
rises **0.196 → 0.466**, ours falls **0.440 → 0.116**. *"Its declaration carries fine lateral
information — the nudges — that our gate discards; ours carries only coarse information … That is
exactly what a vocabulary with severity (`Steer Left` vs `Sharp Steer Left`) buys, and our 5-way
softmax has no severity axis at all."* Gate-free cross-check: **0.7143 for both arms**, but over
**n = 21** declared turns for Alpamayo against **n = 7** for ours — *"It declares 3× as many lateral
actions, and is right about them just as often."*

⛔ **Blast radius stated in the doc:** `DIR_YAW_RAD = 0.15` is `taniteval/hierarchy.py:164` and feeds
`consistency.maneuver_vs_trajectory`, `commanded_route_vs_maneuver`, `commanded_route_vs_trajectory`
and every `*_turn_subset` — **"i.e. every published manoeuvre-coherence κ in the programme. They
should be re-read at 0.10 and the sensitivity published."** Logged in `RETRACTION_LOG.md`.

### 2.5 ⭐ THE AUGMENTATION RECORD SET — MEASURED BY ME FROM THE PARQUET, 2026-08-15

The handover says *"The quantisation arms are recorded per-row in `records.parquet` itself — quote
from there, not from prose."* I pulled it. Evidence class: **MEASURED (ours)** — direct read of
`Sayood/tanitad-alpamayo2-augmentation/records.parquet`, **25,970,018 bytes**, this session;
extraction script `…/scratchpad/pull_a2_parquet.py`, facts JSON `…/scratchpad/a2_parquet_facts.json`.

**Schema — 12 columns:**
`clip_id` (large_string) · `t0_us` (int64) · `task` (large_string) · `model_id` (large_string) ·
`quantisation` (large_string) · `seed` (int64) · `wall_s` (double) · `error` (null) · `vqa_qid`
(large_string) · `vqa_category` (large_string) · `question` (large_string) · `raw_json` (large_string).

**Counts (MEASURED):** **23,644 rows**, **4,729 unique `clip_id`**, `model_id` = `nvidia/Alpamayo2-Super`
on **all 23,644 rows**, `seed` = **42** on all rows, **`error` non-null on 0 rows**.

| task | rows |
|---|---|
| `trajectory` | 4,729 |
| `meta_action` | 4,729 |
| `auto_labeling` | 4,729 |
| `vqa` | 4,729 |
| `grounding_via_vqa` | **4,728** |

⇒ The handover's *"23,644 rows = 4,729 clips × 5 tasks"* is **arithmetically 23,645**; the true row
count is 23,644 because **`grounding_via_vqa` is one row short (4,728)**. A one-row discrepancy, but
recorded rather than smoothed.

⭐ **THE QUANTISATION COLUMN — a single arm, and it is labelled as unvalidated:**

| `quantisation` value | rows |
|---|---|
| **`NF4-backbone-4bit-UNVALIDATED`** | **23,644 (100 %)** |

⇒ **There is exactly ONE quantisation arm in the entire augmentation set**, and its label carries the
`UNVALIDATED` stamp into every row — the §2.2 discipline is enforced *in the data*, not just in prose.
**There is no bf16 arm and therefore no quantized-vs-full comparison inside this dataset.** The only
quantized-vs-full evidence in the programme is the *indirect* one in §2.2 (our NF4 A40 number vs
NVIDIA's published H100 number, explicitly declared non-comparable).

**Inference cost, MEASURED from `wall_s` (this is the throughput record the DESIGN doc said the pilot
would produce):**

| task | mean s/clip | median s/clip | n |
|---|---|---|---|
| `auto_labeling` | **17.40** | 18.1 | 4,729 |
| `meta_action` | **11.74** | 10.0 | 4,729 |
| `trajectory` | **11.47** | 9.5 | 4,729 |
| `grounding_via_vqa` | **10.45** | 10.1 | 4,728 |
| `vqa` | **8.59** | 8.2 | 4,729 |
| **total** | — | — | **78.4 wall-hours** |

⇒ **The full 5-task battery costs ~59.6 s/clip measured**, against the DESIGN doc's
**"ESTIMATE ~200 s/clip"** — the real pipeline is **~3.4× cheaper than the estimate**, and the
earlier single-task MEASURED anchor (*"meta-action = 40 s/clip (1,561 s / 39)"*) was **3.4× slower
than the batch rate the full run achieved**. This is a materially better cost basis for any future A2
pass and it has not appeared in any report.

**Trajectory output shape (MEASURED from `raw_json`):** `pred_xyz_shape = [1, 1, 1, 64, 3]` on every
row that carries it — **64 waypoints**, `num_trajectory_samples: 1`. ⚠️ **Only 255 of 4,729 trajectory
rows carry `min_ade_m`** (the rest have `num_trajectory_samples: None`, i.e. the GT-dependent metric
block is absent). On those **255**: **min_ade_m mean 2.3469 m / median 1.5233 m**; **min_fde_m mean
6.8726 m / median 4.6711 m** — at Alpamayo's **native 6.4 s horizon**, and at **1 sample**, so these
are ⛔ **not comparable to the published minADE₆ 0.911 m** for the same three reasons in §2.2.

### 2.6 ⚠️ Where the handover's pointers do not land

- The handover (§1.A) says the A2 phase's record is in **"chronicle rows in `PROJECT_STATE.md`"**.
  **MEASURED: there are none.** `grep` for `a2venv`, `4,729`, `4729`, `23,644`, `alpamayo_vs_flagship`
  over `PROJECT_STATE.md` returns **zero hits**; the only `alpamayo` matches in that file are three
  **Opponent-Analyzer** rows from 2026-07-13/17/24 about the 32 B competitor, not about our run.
- It also says the strategy is in **`DataEng/DATA_STRATEGY.md`**. **That file is the v1.0 of
  2026-07-06** (`git log`: last touched by `47a89c4`, the D-014 MetaDrive retirement) and **contains
  no Alpamayo content at all**. See §3.1.
- The A2 work's actual home is `TanitAD Research Hub/Benchmarks & Eval/Research/2026-08-05-alpamayo2-super/`
  (analysis, 764 lines) and `…/2026-08-06-alpamayo-augmentation/DESIGN.md` (the dataset design).
- `alpamayo_vs_flagship` and the `retime` arms are named in the handover as pod-side products in
  `a2_batch_out/`. **The pods are dead.** What survived into the repo is the §2.3/§2.4 comparison
  (inside `ALPAMAYO2_SUPER_ANALYSIS.md`) and the re-timing A/B under
  `…/incoming/2026-08-06-v1-defect-triage/results/RETIME_AB_RESULT.md` (extracted in §5.3 below) —
  which is a *different* "retime" (our own trajectory re-timing, not an A2 arm). **If an A2-side
  `retime` arm existed beyond those, it did not reach the repo**; searched by content (`retime`,
  `alpamayo_vs_flagship`) across `*.md`/`*.py`/`*.json` and by path shape across `stack/` and the Hub.

---

## 3. DATA AUGMENTATION — Alpamayo first, then VLM/SAM3 (PH0), and what it buys against the 139× gap

### 3.1 ⚠️ First, a pointer correction

The handover cites **`DataEng/DATA_STRATEGY.md`** as the owner of the augmentation strategy. **It is
not.** That file is **v1.0 dated 2026-07-06** ("implements D-009/D-010/D-012"); `git log` shows its
last commit is `47a89c4` (MetaDrive retirement), it predates Alpamayo-2 Super's release (2026-08-04)
by a month, and it contains **no Alpamayo, no VLM, no SAM3 content**. Its still-current contribution
is the corpus-role table (PhysicalAI-AV as *"urban diversity backbone … 1,727 h, 25 countries, 2,500+
cities"*; comma2k19 as the *"BOOTSTRAP + PUBLIC ANCHOR"* with real CAN; the license firewall: *"public
claims/demos/publications use comma2k19 + MetaDrive (+ own data) only"*). **The augmentation strategy
actually lives in four documents**: `…/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md`,
`…/2026-08-07-hierarchical-wm-redesign/{PREREG_PH0_VLM.md, PH0_PIPELINE_VALIDATION.md, PH0_COVERAGE_AUDIT.md}`,
with the requirement framing in `…/V6_DATA_REQUIREMENT.md`.

### 3.2 WHY augment at all — the measured data deficit

**Source:** `…/2026-08-07-hierarchical-wm-redesign/V6_DATA_REQUIREMENT.md`, evidence class **MEASURED
from the running job** (answering the PI's 2026-08-13 question about deriving and reducing the data
requirement). **Headline verbatim: "we are data-limited by roughly two orders of magnitude, and
compute-limited by none."**

| quantity | value |
|---|---|
| corpus | **2,400 episodes ≈ 13.3 hours** of driving |
| valid windows at k=60 | **319,002** (415,002 at k=20) |
| samples seen at 30 k × batch 8 | 240,000 → **0.75 epochs** |
| unique latent-steps | 319,002 × 66 = **21.1 M** |
| model | **336.5 M** parameters |
| Chinchilla-style target (20 tokens/param) | **6.73 B** |
| **unique data ÷ that target** | **0.31 %** |
| V-JEPA 2 pretraining | **>1,000,000 hours** → ratio **1 : 75,000** |
| ⭐ **Orbis 2 pretraining** (same task, same model class) | **5,890 hours**, 1 epoch → ratio **1 : 443** |
| ⭐ **hours per M-param: Orbis 2 vs v6** | **5.52 vs 0.040** → **139× under, size-normalised** |

⚠️ The doc reads its own Chinchilla row down: *"20 tokens/parameter is an LLM-on-text law … the
constant is not transferable. What IS transferable is the order of magnitude."* The **Orbis 2 rows are
the load-bearing ones** because they are **task-matched** (a hierarchical *driving* world model,
1,067 M params, fine-tuned on **the same PhysicalAI-AV corpus we use**). ⚠️ Evidence stamp on those:
`ORBIS2_ANALYSIS.md` marks them **PUBLISHED-via-snippet, "one confidence step below PUBLISHED-exact"**
— arxiv is egress-blocked from that environment, so they came from search-index snippets of
arXiv:2607.15898 plus the GitHub README. **Before any of them decides a GPU-day, someone with unblocked
egress must confirm against the PDF.**

⭐ **The uncomfortable implication, verbatim:** *"Config E/F at 336 M was chosen to sit in the PI's
250–350 M band. On 13.3 hours of unique video that is heavily over-parameterised by frontier standards
— the model has ~16 parameters for every unique latent-step it will ever see. … the binding constraint
moved from parameters to data."*

**The lever ranking that follows (§2–§4 of the same doc):** only **P0** (a larger corpus under a
**new, declared** parity key — *"never a silent widening"*; evaluation set unchanged) or **P1** (a
frozen pretrained encoder, importing someone else's hours) can close a 443× gap. The rest are
single-digit factors: **train longer** (*"repeated data keeps nearly the value of fresh data for the
first ~4 epochs … We are at 0.75. So the same corpus supports ~120 k steps"* ⇒ *"the current 30 k step
budget is not a data limit, it is an arbitrary stopping point"*), **O4 saliency sampling** (already on,
`weight_max_over_min` **15.5× at k=60**, 30.8× at k=20; `alpha=1.0`/`floor=0.25` **never swept**), and
**latent near-duplicate pruning** (untried). ⚠️ *"Do not read LIMO/s1 as 'we can learn driving physics
from 13 hours' — they apply to our S-T/S-S stages … and no help at all for S-W."*
⚠️ **What the doc refuses to claim:** *"none of these is measured to fix our defects."*

### 3.3 Layer 1 — the ALPAMAYO augmentation: what it contributes and why

**Source:** `…/2026-08-06-alpamayo-augmentation/DESIGN.md`, evidence class **DESIGN + MEASURED facts**.

- **The PI's brief, verbatim intent:** select **100 hours** of well-distributed PhysicalAI-AV data,
  run it through Alpamayo 2 Super inference, log its **complete outputs — no exceptions**, link every
  output to the index of the corresponding sensor + ego data (**unique link, NO camera images in the
  dataset**), publish as an HF public gated dataset; VQA gets **≥500 questions** assigned randomly.
- **Corpus size MEASURED: 306,152 clips ≈ 1,701 h** (`data_collection.parquet` row count, 2026-08-06)
  ⇒ 100 h = **18,000 clips** = a **5.9 % stratified sample**.
- ⚠️ **"No road-type column found yet"** (highway/urban/intersection) — the fallback is deriving road
  class from egomotion statistics, admissible under the labels-may-use-ego rule.
- **What the augmentation buys, per the redesign doc** (`HIERARCHICAL_WM_REDESIGN.md` §3.7): *"The
  Alpamayo augmentation dataset supplies **auxiliary distillation targets**: its meta-actions (3-axis,
  severity) and CoC traces are exactly tactical-level supervision — a second, independent teacher for
  stage 0. (Contamination caveat travels.)"* And per `HIERARCHY_VOCABULARY.md` §1, it is **source #1**
  of the vocabulary itself: *"the empirical distribution of manoeuvre phrases and meta-actions real
  driving exhibits on our corpus."*
- **The ranked leverage list** (`ALPAMAYO2_SUPER_ANALYSIS.md` §8) puts **"Use Alpamayo 2 as an
  AUTO-LABELLER, not a competitor"** at Tier-1 #2: *"Our strategic brain has no map, no lane graph, no
  junction annotation — settled at five probes — which is precisely why our route head degenerated to
  a constant predictor (`{left 0, straight 1737, right 0}`). **Alpamayo can manufacture the strategic
  supervision PhysicalAI-AV does not ship.** OpenMDW-1.1 permits derivative use. This is the single
  biggest unlock available to us."*
- ⛔ **What NOT to do, verbatim:** *"Do not reposition TanitAD as 'beating Alpamayo'"* (34 B vs 0.3 B,
  6 cameras vs 1, 115,000 h vs ~9 h) · *"Do not fine-tune Alpamayo as our deliverable"* · *"Do not
  quote its numbers as a target on our corpus until the contamination question is settled."*
- **Delivered (MEASURED, §2.5 above):** 4,729 clips × 5 tasks, 78.4 wall-hours, zero errors, on HF.
  ⚠️ **`alpamayo_rows = 0` on every PH0 clip — "engine D contributed nothing. Undiagnosed."**
  (`PH0_COVERAGE_AUDIT.md` §5) — i.e. the Alpamayo layer is banked but **was not joined into the PH0
  records at pilot time**; the PH1 fuser later reports **56 of 600 with the Alpamayo layer**.

### 3.4 Layer 2 — PH0 (VLM + SAM3): pre-registration, then the measured verdict

**Pre-registration:** `…/2026-08-07-hierarchical-wm-redesign/PREREG_PH0_VLM.md` — three arms on an
identical 50-clip pilot, **clip_ids committed WITH the sampler line that produced them**, seed 0:
**A** `Qwen/Qwen3.5-9B` bf16 · **B** `Qwen/Qwen3.5-27B-FP8` · **C** `google/gemma-4-31B-it-qat-w4a16-ct`
(*"survey benchmark leader (MMMU-Pro 76.9) with NO published OCRBench — exactly the unknown PH0's
sign-OCR gate measures"*). Gates bound in advance: **G1 sign-OCR precision ≥ 0.9** · **G2 schema
compliance ≥ 0.9 on pass 1** · **G3 consistency reported with CI, no pass/fail**. Both outcomes
committed: a G1 FAIL on all three arms ⇒ *"PH1 drops signage-text fields entirely and the strategic
goal degrades to hindsight-geometry corridor intent — the honest ceiling stands, no re-prompting
rescue runs without a new prereg."* Discipline: *"report exact counts (x/n) with Wilson intervals,
never bare percentages."*

**The measured verdict:** `…/PH0_PIPELINE_VALIDATION.md`, answering the PI's 2026-08-13 *"Did you
validate the correctness and the semantic quality of the results? Can we start the production…"*.
**Verdict verbatim: "three of four channels VALIDATED, one FAILED — production may start with the
failed channel demoted."** All numbers **MEASURED on the 50-clip pilot** (`/workspace/ph0_pilot50/`),
not the 8-clip smoke.

| channel | engine | status | evidence |
|---|---|---|---|
| **Schema / structure** | B (Qwen3.5-9B, grammar-constrained) | ✅ **PASS** | **50/50 clips all-calls-valid; 0 parse failures, 0 violations, 0 retries.** Gate G2 ≥ 0.90 → measured **1.00** |
| **Agent / object detection** | C (SAM3) | ✅ **PASS** | **1,383 detections / 50 clips**: car 703 · sign 292 · light 167 · pedestrian 130 · truck 57 · bus 22 · cyclist 12. Correct abstention verified (0 cars on an empty road, 16 trees on a treed one) |
| **Geometry / ego** | A (integrated path) | ✅ **PASS** | deterministic; route token + speed profile + situations + pre-decision ego state on every clip |
| **Sign box grounding** | B (call B3) | ⛔ **FAIL** | **2 / 49 agree with SAM3** on the *exact same frame* |

**Semantic quality, MEASURED:** *"49 signs read across 50 clips, 31 with non-empty OCR text; kind
distribution `other 24 · speed 16 · light 4 · nav 3 · yield 2`. Goals `follow_main_road 45 · route_to
5`, and every `route_to` is backed by a nav sign index (the abstention rule held — no invented
cities). Throughput **48.9 s/clip median, 52.5 s p90**."*

**The B3 failure, decomposed (§3):** *"The earlier 0/8 was my confound (I compared frames up to ~3.5 s
apart)."* Frame-exact: **26 of 49** — SAM3 saw no sign on that exact frame; **23 of 49** — both engines
saw a sign on the same frame, **and only 2 overlap** ⇒ **agreement 2/23 ≈ 9 %**. ⇒ **"B3 is demoted to
diagnostic-only. SAM3 supplies all boxes and pixels; the VLM supplies symbols and OCR."** ⚠️ What is
explicitly *not* claimed: *"this does not show the VLM's sign classification is wrong — B2 and B3 are
separate calls, and B2 passed schema validation on every clip."*

**Production readiness table (§4):** schema stability ✅ 50/50 zero retries · resume-safety ✅ · per-clip
failure isolation ✅ · corpus bridged ✅ **2400/2400 clips, 5.46 GB** · **cost MEASURED 48.9 s/clip →
~64 h for 4,729 clips (VLM), + SAM3** · ⛔ blocker at the time: the 5.46 GB had not been transferred
pod5 → pod4. **"The one thing production must NOT do is treat `grounding` as ground truth."**

⚠️ **A recorded correction on the VLM arms** (paper commit `4531868`, PI-raised): an earlier claim that
*"all three arms unusable (text-only / OOM / broken 4-bit)"* was **wrong on two of three**. Re-read from
`ph0_smoke.json`: Qwen3.5-9B and gemma both hit `BlockingIOError [Errno 11] - [swscaler] Failed
initializing scaling graph` — **a video-decode resource failure, not a model capability limit**
(*"thread/FD exhaustion, i.e. ENVIRONMENTAL — same family as the torch-113-threads trap"*); only the
27B-FP8 OOM is real (**43.23 GiB on a 44.43 GiB card**). ⇒ *"D7 is therefore probably NOT a decision:
fix the decode first, the 9B we already have may be fine."* Root-cause class named in the commit:
*"I carried a stale characterisation from an earlier smoke run into a PI decision request instead of
re-probing."*

**The full extracted structure** (`PH0_PIPELINE_VALIDATION.md` §2, schema `ph0-v2.2`) — Engine B emits
**B1 SCENE** (`illumination` day|dusk|night|dark · `weather` clear|rain|snow|fog|unclear · `road_type`
highway|urban|rural|junction|unclear · `domain` +roundabout|intersection · `lanes_visible` · `lane_ego`
· `conf`), **B2 SIGNS** (≤6 × {`kind` ∈ light|speed|nav|stop|yield|other · `text` **verbatim OCR, ""
if illegible** · `state` · `applies_to_ego`}), **B3 GROUNDING** (diagnostic only), **B4 SYMBOLS**
(the **11 g_str tokens** + `goal_evidence_sign` + ≤3 × {`verb` (6) · `direction`}); Engine A emits
`route{token, token_valid, dist_m, arc_m, maneuver_dyaw_rad, graded_route}` · `lane_change_events[]` ·
`speed_events[]` · `speed_profile{…}` · `peak_kappa_per_m` · **`situations{lane_change, intersection,
roundabout, *_windows_s}` — "frozen detectors, deliberately NOT routed into the goal prompt"** (the
goal/situation-disjointness rule enforced at the prompt boundary); Engine C emits per-clip masklets
with **exact RLE masks** and a frame-aligned `vlm_cross_check`. **Every call's PROMPT + RAW OUTPUT is
banked** (`_calls`), with `_all_valid`, `_n_parse_fail`, `_n_violation_fail`, `_n_retried`.

**The image-coordinate rule** (`HIERARCHY_VOCABULARY.md` §0b, PI 2026-08-11): *"every semantic claim
any engine emits MUST carry its image-space grounding — bounding box `[x0,y0,x1,y1]` and, where the
engine provides it, the pixel contour/mask reference + frame index. **Ungrounded claims are `disputed`
by default in the fusion gate.**"*

**Why SAM3 is in the pipeline at all** (same §0b, PI idea): Engine C gives *"(a) richer grounding for
the VLM's claims (a sign/light claim must overlap a mask), (b) mask-level agent tracks that cross-check
`obstacle.offline` and extend labels to classes it lacks (static furniture), (c) lane/road-surface
geometry PhysicalAI's labels never had — **the closest admissible thing to the missing map**."*
Usage stays **labeling-side only**.

### 3.5 The coverage arithmetic — 201 done, 4,472 blocked

**MEASURED (handover §1.G, far-side-verified at stop; the HF listing confirms the artifacts):**
of the **4,729** augmented clips, **257 have w120 caches**, **56 were already done**, and the
**201 runnable were all processed and pushed** (`aug120_pipeline.py`, per-batch pull/process/push/
delete, bounded disk). ⇒ **257 − 56 = 201 processed; 4,729 − 257 = 4,472 clips have no w120 cache.**

⛔ **The blocker is a build, not a model:** *"the source stores the 120° camera as chunked zips +
per-feature parquet chunks, so the build path is chunk-index → `v2_compressed.py build --only-clips`
(scoped in the stop runbook, not started)."* It is ranked **#4 in the open-work order** and is
*"the single biggest data job; also unlocks the G1 native-res re-run."*

**What the 4,472 buys against the 139× gap — the honest arithmetic.** The augmentation set is
**4,729 clips ≈ 26 h** (`V6_DATA_REQUIREMENT.md` §2: *"roughly doubles the corpus"* against the
13.3 h in training). Against Orbis-2's **5,890 h**, doubling 13.3 h → ~26 h moves the size-normalised
deficit from **139×** to **~70×** — still two orders short, which is exactly why the same document
ranks the **new-parity-key corpus enlargement (P0)** and the **frozen pretrained encoder (P1)** above
it: *"only lever 2 or lever 3 can close a 443× gap. Levers 1, 4 and 5 are worth single-digit factors."*
⭐ And per `ORBIS2_ANALYSIS.md` §5.1, **P1 is now "the best-supported unrun experiment in the
programme"** — Orbis 2 is *"the third system"* to route around the data requirement with a frozen
pretrained encoder, *"and the first to do so for driving, hierarchically, with a compression
projection, and to report that the compression itself stabilised training and improved long-horizon
rollout"* — the regime our 6 s / K=60 contract lives in. ⚠️ **The cost we still own:** *"DINOv2B is
3-channel, narrow-FOV, non-driving-pretrained. Ours is 9-channel wide-FOV cylindrical. The adapter
that bridges that is real work and is precisely what the arm must measure."*

⇒ **Stated plainly: the PH0/PH1 augmentation is NOT a fix for the S-W data deficit.** Per
`V6_DATA_REQUIREMENT.md` §3.4, label-side curation *"applies to our S-T/S-S stages (goal supervision
on a frozen trunk), where a few thousand well-chosen labelled clips genuinely may suffice — which is
a real and encouraging result for the PH0 pipeline, and **no help at all for S-W**."* The augmentation
stream and the corpus-size stream are answering **different** questions, and conflating them would
misprice both.

---

## 4. THE DIFFUSION PLANNER × MPC COMBINATION — the design, and what W7 measured of it

**Owners:** `…/incoming/2026-08-06-mpc-planner-design/MPC_WM_DESIGN.md` (the MPC planner, *"Design
only — no numbers claimed"*) · `…/2026-08-06-mpc-planner-design/DIFFUSION_MPC_SYNTHESIS.md` (the
combination, answering the PI's 2026-08-07 question) · `…/2026-08-07-hierarchical-wm-redesign/DIFFUSION_PLANNER_COMPARISON.md`
(why our fan looked jumpy) · `…/2026-07-23-frozen-wm-learned-planner/` (the four measured precursor
experiments) · `MODEL_REGISTRY.md` §1.13c/§1.14 (the W7 results).

### 4.1 The design — why the fit is exact, verbatim

*"MPPI/CEM's weak point is its **proposal distribution** (Gaussian around a warm start — poor coverage
of discrete choices like branch/gap selection). v5f's anchor+denoise fan is a **learned, scene- and
goal-conditioned proposal** that MEASURED reaches oracle 0.21 m — better than any hand-crafted sampling
prior we could write. … v5f's weak point is **choosing** (sel_gap ≈ 2×, static over 17k steps) and its
imagination conditioning has **no per-candidate axis** (one shared consequence summary for the whole
fan). MPC is precisely a per-candidate consequence check with an explicit, auditable cost."*

### 4.2 The four integration levels (`DIFFUSION_MPC_SYNTHESIS.md`) — **DESIGN**

- **L1 — MPC as re-ranker (cheapest, first).** Everything frozen. Selector prunes the fan **256 → top-8**;
  each survivor is converted to a unicycle action sequence, the predictor rolls it **per candidate**
  (batched), each is scored on **its own imagined future** with the `MPC_WM_DESIGN` cost terms, and the
  argmin is emitted. **Pre-registered gate X0: "MPC re-rank closes ≥ 50 % of sel_gap on the same fan
  (plan 0.42–0.48 → ≤ 0.33 at oracle 0.21), evaluated at T1."**
- **L2 — cost-guided denoising.** `x ← denoise(x) − η·∇C(x)`, classifier-guidance style, injected into
  the (only 2) truncated denoise steps. *"guidance shapes, MPC still chooses."* ⚠️ **Diversity guard:**
  *"oracle_ade is the canary: if oracle worsens while plan improves, guidance is eating the
  multimodality that makes the fan valuable."*
- **L3 — receding-horizon warm start.** Next cycle's denoise starts from the previous selected candidate
  time-shifted instead of the static anchor vocabulary — *"the diffusion equivalent of MPC's warm start"*,
  aimed at replan-jump/temporal stability.
- **L4 — amortisation (fast/slow closed).** Distil L1's MPC choices back into the learned selector
  (DAgger-style). *"Deployed shape: selector always (fast); MPC verification triggered by low
  score-margin, high ensemble disagreement, or high-stakes context flags (slow). The PI's fast/slow-
  thinking pattern at the operative level, with the slow path defined by explicit costs rather than a
  second network."*

**The experiment ladder** (all post-30k, frozen-trunk): X0 L1 re-rank ~2 GPU-h, gate ≥50 % sel_gap
closed · X1 L2 η sweep ~2 GPU-h, gate *plan ↓ with oracle NOT ↓* · X2 L4 distilled selector ~3 GPU-h,
gate *distilled ≥ 80 % of MPC's gain at zero runtime cost* · X3 full stack ~4 GPU-h.

**Prerequisites, all named in advance:** **P1 action-controllability of the trunk** (*"if the predictor
under-weights actions, per-candidate rolls are fiction"*) · **P2 feasible action conversion**
(*"REF-C-XL fan: 72 % were not [near-feasible]"*) · **P3 cost-term feasibility** (lead-gap/curvature
decodability from ẑ).

**Where it sits in the hierarchy, verbatim:** *"This synthesis lives entirely at the **operative**
level … g_tac (E5) conditions the denoise (upgrading cond_route's 3-way class to a geometric goal),
the tactical layer's own fan-and-select repeats the same pattern one level up … **Nothing here competes
with the hierarchy — it is the hierarchy's bottom layer done right.**"*

### 4.3 The MPC planner's own design (`MPC_WM_DESIGN.md`) — **DESIGN**, with the motivation MEASURED

**Motivation, registry §1.12 (MEASURED 2026-08-06):** with recorded future actions removed,
*"hold-action reproduces **0 %** of GT yaw-reversals (open-loop: **97.9 %**), closed-loop ADE degrades
**0.34→0.47** (v1.6) / **0.28→0.46** (v1.7), net-yaw **×6.7**. **The open-loop lateral signal was an
action echo.** Any deployable configuration must therefore SUPPLY future actions rather than consume
them — which is exactly what MPC does by construction."*

- **Sampler: MPPI/CEM over the UNICYCLE action space** (steer, accel at 0.1 s, K = 20), *"parameterise
  candidates as controls, never free XY — every candidate dynamically feasible, |κ| ≤ 0.33, a ∈ vehicle
  envelope (**same bounds Alpamayo publishes**)"*. Warm-start each cycle from the previous solution
  shifted one step.
- **Rollout engine: the frozen predictor, batched** — *"N = 64 with 2 CEM iterations ≈ 128 rolls per
  replan; A40 feasibility to be MEASURED, Thor deployment budget is the binding constraint and
  batch-8 SM saturation applies."*
- **The cost — "where the programme's instruments become the objective":** imitation prior (distance to
  the readout head's own decode) · progress/goal (**"goal = predicted goal point (goal-input rule:
  never the situation classifier's output)"**) · comfort (*"the SAME losses that trained v1.6/v1.7
  become runtime costs"*, train-corpus p99 barriers) · safety/distance-keeping (headway/TTC vs a
  predicted lead track — *"probe first: ridge from ẑ to lead gap … R² decides feasibility"*) · latent
  plausibility (distance of ẑ to the predictor's unconditional roll).
- ⛔ **Attribution discipline:** *"every term has a weight flag and a pre-registered zero-weight
  ablation; a planner that improved with ten live terms is unattributable (the `--v2` failure)."*
- **vs Alpamayo:** *"its diffusion expert samples 6 trajectories with NO explicit cost; minADE₆ quietly
  assumes an oracle selector. **MPC makes the selector explicit and auditable.**"*
- **The headline discriminator named in advance:** *"The S-curve rate under MPC … candidates that
  counter-steer exist in the sampler by construction; whether the cost selects them tests whether the
  LATENTS carry the road geometry."*

### 4.4 The precursor evidence — the frozen-WM planner quartet (MEASURED, 2026-07-23/24, 12-ep val)

All four in `…/incoming/2026-07-23-frozen-wm-learned-planner/`, all on the frozen v1 WM, all with
**pre-registrations committed in advance** and episode-cluster CIs.

| planner over the frozen WM | ADE@2s | what it is |
|---|---|---|
| **CEM search, warm-init** (refines GT actions) | **0.1322** [0.087, 0.184] | *"the strongest plan the WM admits per window"* — 4.53× better than W |
| **CEM search, cold-init** (μ=0, GT cost) | **0.2471** [0.149, 0.375] | 2.42× better than W |
| oracle-action ceiling (frozen WM, raw GT actions) | 0.4045 [0.310, 0.514] | v1's own operative number |
| **W — feed-forward analytic-gradient planner (3.77 M)** | **0.5989** [0.374, 0.854] | the baseline |
| CV floor | 0.8463 | trivial |
| amortised prior (naive action-BC distill) | **1.3987** | ⛔ the DEPLOYABLE fast planner — **fails** |
| **V-search (learned value, NO GT) — deployable** | **1.0162** [0.809, 1.273] | ⛔ **+0.417 [+0.237, +0.605] SEP WORSE than W** |

**Three verdicts, each against its own pre-registration:**
1. **Frozen-WM + learned planner is `VIABLE`, not `BOTTLENECKED`** — W beats CV (0.846) and hold-v0
   (0.788), is *"6.1× better than decoding the trajectory statically off the same frozen latent
   (3.65 m)"*, and lands within 1.48× of the WM's own oracle-action ceiling — *"provided the planner
   reads the WM through its dynamics (outputs actions the frozen predictor rolls) rather than decoding
   the trajectory off the static latent."*
2. **Feed-forward capacity is NOT the lever** — *"Scaling the planner 11× … is a FLAT line — W 0.599 →
   mlpbig(30.8 M) 0.601 → mlpwide(42.6 M) 0.599 … bigger query-decoder planners OVERFIT and do worse
   (0.82–0.86). No feed-forward variant approaches the ≤0.45 bar."* ⇒ *"matching the 0.132 search
   ceiling fundamentally needs test-time search."*
3. ⛔ **The learned value model does NOT close the gap** — V-search 1.0162 vs W 0.599, paired-separated
   **worse**; diagnostics: held-out cost-prediction L1 **0.115 m**, within-window rank-corr(V, true
   cost) **0.613**, and *"V-search − cold-GT-search +0.7682 [+0.587, +0.988] ⇒ the entire gap to the
   search ceiling is V's imperfection."* The mechanism was **predicted in advance**: *"A value model
   can only learn `E[cost | state]`; it cannot know this window's actual future … CEM will instead
   adversarially exploit V's errors."*

⇒ **These four say the fan-and-search topology is right and the *cost* is the hard part** — which is
exactly what W7 then tested at scale.

### 4.5 ⭐ WHAT W7 MEASURED — the design confirmed in mechanism, refuted as a headline route

All from `Project Steering/MODEL_REGISTRY.md` §1.13c/§1.14, **[T0, 881 grid]** unless stamped otherwise.

**(a) The prerequisite P1 failed first, and its repair is the campaign's cleanest win.**
W3 measured *"the WM's action-response gain at ~0.27 (¼ physical)"* ⇒ **"W4b/W4c's scoring failures,
W7's ceiling, and §1.12's action echo are ONE DEFECT: the trunk under-weights actions in its
rollout."** **Stage-A post-training** (predictor-only, 3,000 steps, encoder/head/emission frozen with
md5-proof) — **MEASURED 2026-08-11 ~07:20Z [T1-diagnostic]: ALL GATES PASS**: *"lateral gain 0.27 →
**0.971/0.966** (gate [0.5, 2.0]); longitudinal sign 0.745/0.787 → **1.0/1.0** (gate ≥0.95); lateral
sign stays 1.0; longitudinal gain 0.972; **P6 subspace stays exactly 3-dim**; no-harm passed."*

**(b) W7 gate FAILED at every K — and the failure converged the night onto one root cause.**
**MEASURED 2026-08-11 ~02:45Z:** K=8 sel **0.5772** (shortlist oracle 0.4401 — pruner-starved) *"but
the roll-cost is the programme's **FIRST CALIBRATED selection signal (Spearman ρ 0.399 vs 0.05–0.26
for every learned scorer)**"*; K=32/64 shortlist oracle improves hugely (**0.182 / 0.142**) *"yet
selection stalls (**0.5173 / 0.5319**) and cost-calibration **COLLAPSES (0.106 / 0.047)** — with many
similar good candidates, the WM's rolled consequences barely differ, so the cost drowns."*

**(c) W7 on the repaired trunk — the calibration nearly doubled.** **MEASURED 2026-08-11 ~07:35Z:**
gate FAIL (selected 2.3468 vs thr 0.4505) *"but the failure is **INSTRUMENT COMPOSITION**, not a repair
verdict"* — the W4 emission head + frozen selector were trained on **frozen-trunk** features and no
longer compose (in-run frozen selector reads **3.448** vs banked 0.7933; fan oracle degrades **0.289**
vs banked 0.1077; winner-in-shortlist only **19.6 %**). **Evidence FOR the repair in the same run:
roll-cost calibration ρ 0.7164 [0.5847, 0.7696] (episode-cluster bootstrap, n=881; P7 gate PASS —
~1.8× the frozen trunk's 0.399, the strongest calibration measured in the programme)**, and W7's pick
≤ the in-run frozen pick on **71.1 %** of windows. Families reported per the binding rule: LON speed
MAE 0.989 m/s, accel MAE 1.067; LAT heading MAE 0.173 rad, curvature MAE 6.05 pm, yaw-rate MAE 0.250;
TACTICAL winner-hit 0.9 %, sel-rank 29.9 %; STRATEGIC n/a (no route label — settled).

**(d) ⭐ W7-FULL — selector-free, all 256 candidates: the question closes with a MECHANISM.**
**MEASURED 2026-08-12 ~00:15Z:** every stale component removed at once (repaired trunk, W4r head refit
on it, **no shortlist — topk = 256 so `winner_in_shortlist_frac` = 1.0**), selection = argmin of
roll-cost + kinematic cost. Result: **fan oracle 0.1273** (excellent) but **selected 3.3348**,
sel_gap **3.207**; the frozen selector on the same fan reads **4.4159**, so *"W7 beats it on 67.5 % of
windows and still fails absolutely."*

**The diagnostic pair that explains it, verbatim:** *"the cost's within-window rank correlation over
the 256 candidates is **ρ_mean 0.445 / ρ_median 0.497** (it ranks broadly correctly) while across-window
calibration is **ρ 0.3185 [0.2064, 0.4086]** — yet the **argmin is 26× the oracle**. That combination
is the **winner's curse**: with a noisily-correlated cost over 256 candidates, the minimiser selects
for cost UNDER-estimation, not for quality, and **enlarging the candidate set makes argmin worse** —
the same direction as the earlier frozen-trunk K-sweep (0.577 → 0.517 → 0.532 at K = 8/32/64)."*
The paper (§7.16) adds the sharpest form: **"the argmin's error-rank is 132 of 256, the median"**, and
*"the quantity that governs selection is lower-tail dependence, which is zero here."*

**Two consequences, both load-bearing, verbatim from the registry:** *"(1) **v5.8f ships the
FROZEN-trunk assembly** (rescorer-top8+kincost, selected **0.4815 [0.393, 0.577]**) — the stage-A
trunk's physics is better but every frozen consumer trained on the old features mis-ranks on it
(frozen selector 0.7933 → 4.4159), and **you cannot repair a trunk and keep its planner**; (2) that
sentence IS the staged-training argument for v6 — consumers must be (re)trained ON the trunk they
consume (S-W → S-T → S-S), and **argmin-over-a-large-fan must be replaced by a noise-robust rule
(top-m aggregation / sharpened cost), pre-registered before it is used**."*

**(e) W7-PROG — the pre-registered fix for the degenerate minimiser: PARTIAL, and it flipped a sign.**
**MEASURED 2026-08-12 ~05:40Z [EXPLORATORY, 881 grid]**, `PREREG_W7_PROG.md` written before the run
with all three outcomes bound; only `--w-prog` changes against the W7-FULL control. Primary endpoint
declared to be **the argmin's mean ERROR-RANK over the 256-candidate fan, not ADE**, *"because the
mechanism under test is a ranking claim"*:

| arm | `--w-prog` | **error-rank of argmin** /256 | gate `w7_selected_ade` |
|---|---|---|---|
| control (= W7-FULL) | 0.0 | **132.3** | 3.3348 |
| `w7-prog-01` | 0.1 | **130.31** | 3.4360 |
| `w7-prog-05` | 0.5 | **126.69** | 3.7398 |

*"The rank falls **monotonically** with the weight — so the degenerate-minimiser account is **not**
wrong … But the whole effect is **5.6 rank positions of 256 (2.2 % of the fan)** … and the gate ADE
gets **monotonically worse**."* ⇒ **PARTIAL** by the pre-registered rule ⇒ **binding consequence fixed
in advance: "the cost needs a goal-conditioned component, not a larger anti-degeneracy weight, and
W7-style self-consistency selection is retired as a headline route."** ⚠️ And: *"with the progress term
the across-window calibration **flips sign** — Spearman(cost, realised error) goes **+0.3185 (control)
→ −0.4244 (w-prog 0.1)**. A cost that is negatively calibrated across windows is worse than an
uninformative one."*

⇒ **Scorecard against the L1 design.** X0's gate (*≥ 50 % of sel_gap closed at T1*) is **not met** —
L1 was measured at T0 and **failed absolutely**. What the design got **right**: the roll-cost is the
programme's only calibrated selection signal (ρ 0.399 frozen, **0.716 repaired**), the per-candidate
axis was the missing ingredient, and the prerequisite P1 was correctly identified as a blocker and
correctly repaired. What the design got **wrong**: it assumed a calibrated cost implies a good argmin.
The paper's own statement (§7.16): the published loops we copied — **V-JEPA 2-AC, DINO-WM** — *"minimise
distance to a **goal**, which inaction cannot minimise, and we had dropped that term."* **L2/L3/L4 were
never run.**

### 4.6 The companion finding — why our fan looked "jumpy", and the fix that halved the oracle

**Source:** `…/2026-08-07-hierarchical-wm-redesign/DIFFUSION_PLANNER_COMPARISON.md` (2026-08-10),
triggered by the PI's observation that v5f's fan looks far less rich than REF-C's. **Both heads run
the same algorithm** — *"v5f's `V15Decoder` literally subclasses REF-C's `AnchoredDiffusionDecoder`
(`stack/tanitad/models/flagship_v15.py:210-217`) and overrides only the KV source and the scoring
fix"* — so every difference is configuration, not algorithm.

| quantity | REF-C-XL (native 4 wp @ 0.5 s) | v5f subsampled to the SAME 4 wp | v5f native dense (20 wp @ 0.1 s) |
|---|---|---|---|
| fan accel MAE (all candidates) | **6.39 m/s²** | **17.00 (2.7×)** | **252 m/s²** |
| selected-candidate accel MAE | **0.366** | **0.627 (1.7×)** | **8.10** |
| candidate-infeasible fraction | 71.7 % | 89.0 % | **97.6 %** of all 256×20×881 steps; **100 % of candidates** |
| selected ADE @2s | 0.4714 m (registry §1.5/§4) | — | **0.4011 m** (§1.8) |
| oracle-in-fan | 0.1640 m | — | 0.1975 m (§1.8) |

⚠️ **Evidence class split, as the doc states it:** the matched-resolution row is **INHERITED** (supplied
by the orchestrating brief citing `fan_refc-xl-30k.pt` + pod5 `x0_fan_dump.npz` censuses, *"not re-run
by this agent"*); the §1.13/§1.8 rows are MEASURED and registry-banked.

**H1 — the confirmed root cause, and it is arithmetic:** the offset head emits one independent (x, y)
per step, so *"for a per-waypoint position error ε, discrete acceleration is a second difference
divided by dt² … the SAME ε costs **1.0 m/s² at dt = 0.5 s and 24.5 m/s² at dt = 0.1 s — a 25×
amplification for identical waypoint-level fit quality**."* **H4 CLOSED by census** (2026-08-10
~19:05Z): the raw dense anchor vocabulary alone scores **accel MAE 1.97 m/s², step-infeasible 10.6 %**
against the refined fan's 252 / 97.6 % ⇒ *"the vocabulary is data-plausible and largely innocent — the
roughness is manufactured by the per-step offset + truncated-denoise path."*

**W4's constructive confirmation — `UnicycleEmission`** (same frozen trunk + head, **109 k new
params, 4 k steps**): **violations 0.0 · selected accel MAE 0.774 · oracle 0.1077 (nearly halved)**
against the original fan's 0.1991 / 9.297. *"A parameterisation change alone removed the roughness AND
improved coverage — **the jitter was hiding coverage, not providing it**."* Trunk provably untouched
(md5 identical, `w4_gate.json.trunk_frozen_proof`).

⭐ **The "richness" question answered, MEASURED 2026-08-10 ~19:30Z:** selector entropy **REF-C 0.97 vs
v5f 2.22**; modes > 1 % **4.6 / 4 vs 12.0 / 12**; windows with ≥3 modes **87.9 % vs 99.2 %**.
⇒ **"The perceived 'richness' of REF-C is NOT more hypotheses — it is FEWER, cleaner, more decisive
ones."** And the visualisation consequence: *"the comparison the PI made was **structurally unequal**
— REF-C's drawn fan is band-limited to 4 segments while v5f's 20-point polylines faithfully display
10 Hz jitter REF-C could not have shown; **the fix is NOT to smooth the drawing**"* — render natively
and add a matched-resolution 0.5 s pane whenever the two are shown together.

**What REF-C still does better and is worth porting (§5):** *"Supervised-at-the-ranked-object scoring
on stable features"* — the E-S1-0 dose-response is *"the sharpest statement in the program"*: the
supervised t=0 conf selects at **0.4728** while **the SAME weights'** unsupervised refined readout
selects at **1.3100** — *"a 2.8× penalty purely for scoring off-distribution."* Plus the reachability
prefilter: REF-C's S2/S2b band *"deletes **72.08 %** of its fan for a paired ΔADE of exactly **0.0000**
and a **3.5×** per-candidate compute saving"* — *"the cheap admission ticket for W7's per-candidate WM
rolls."*

---

## 5. ALPAMAYO'S ACTION SPACE AND TRAJECTORY REPRESENTATION — what it taught us, and what v6 adopted or rejected

This is the topic with the **longest measured chain** in the campaign: a published config → a
comparison → an inverse-dynamics measurement → four head-design decisions → two trained arms → the v6
emission contract. Every link is named.

### 5.1 What Alpamayo's representation actually is (PUBLISHED, read off `config.json`)

From `ALPAMAYO2_SUPER_ANALYSIS.md` §1, verbatim from the released config:

```
action_space_cfg: UnicycleAccelCurvatureActionSpace
    n_waypoints 64 · dt 0.1
    accel_bounds     [-9.8, +9.8]    accel_mean 0.0290   accel_std 0.6810
    curvature_bounds [-0.33, +0.33]  curv_mean  0.000269 curv_std  0.02615
    ridge/lambda terms on a, kappa, theta, v   (least-squares action fit)
diffusion_cfg: FlowMatching
    int_method euler · train_timestep_sampler beta
    inference_guidance_weight 3.0 · use_classifier_free_guidance false
action_in_proj: PerWaypointActionInProjV2 (Fourier feats: 20, max_freq 100, 2 enc layers, h=512)
expert llm_config: 64 layers · 16 heads / 8 KV heads (GQA) · expert_non_causal_attention: true
cotrain_expert_vlm: false
```

**Five readings, verbatim:**
1. **"The action space is `(acceleration, curvature)` on a unicycle model"**, integrated at dt = 0.1 to
   64 waypoints. *"The model does not emit free XY. **Every output is dynamically feasible by
   construction** — no bounds check, no post-hoc smoother, no 'physically impossible path' failure
   mode."* The bounds `a ∈ [−9.8, 9.8]`, `κ ∈ [−0.33, 0.33]` (**≈3 m minimum turn radius**) *"are the
   vehicle envelope, enforced in the parameterisation."*
2. **Flow matching, not DDPM** — Euler, **10 steps** at inference: *"10 function evaluations of a 2.3 B
   expert, not of the 32 B backbone."*
3. **The expert is 64 layers at hidden 1536** — π0-style interleaved, attending into the backbone's
   per-layer KV cache; `expert_non_causal_attention: true` means *"the expert sees the whole trajectory
   jointly (it is denoising a full 64-step plan, not autoregressing it)."*
4. **`cotrain_expert_vlm: false`** — *"the expert is trained against a **frozen** backbone. The
   reasoning and the acting are decoupled at training time."*
5. **A SECOND, discrete trajectory path exists**: a full trajectory *tokenizer* — `future_vocab_size
   3000`, `history_vocab_size 1000`, `traj_vocab_size 4000`, `tokens_per_future_traj 128`,
   `tokens_per_history_traj 45` + `future_start/end/pad`, `history_start/end/pad` special tokens.
   *"The VLM can read and write trajectories as text tokens, while the diffusion expert produces the
   precise continuous plan. **The language model reasons *about* trajectories in its own vocabulary.**"*

**Output contract:** *"64 waypoints, 0.1 → 6.4 s at 0.1 s, ego-frame XYZ **+ 3×3 rotation per waypoint**
(full SE(3), not just position)."* ⭐ **Independently confirmed in our own banked data** (§2.5, MEASURED
by me from `records.parquet`): `pred_xyz_shape = [1, 1, 1, 64, 3]` on every trajectory row.

**Meta-action: three independent axes with severity.** MEASURED over **4,729 clips** by me from the same
parquet (the campaign only ever published the n=39 slice) — this is the **empirical vocabulary
distribution** `HIERARCHY_VOCABULARY.md` §1 names as source #1, and it has not appeared in any report:

| axis | observed vocabulary over 4,729 clips (parsed **4,729 / 4,729**) |
|---|---|
| **Longitudinal** (7) | Gentle Deceleration **1,594** · Maintain Speed **1,225** · Gentle Acceleration **1,151** · Stop **304** · Strong Deceleration **267** · Strong Acceleration **182** · Reverse **6** |
| **Lateral** (7) | Go Straight **2,504** · Steer Right **1,020** · Steer Left **649** · Sharp Steer Right **135** · Sharp Steer Left **115** · Reverse Right **1** · Reverse Left **1** |
| **Lane** (7) | Lane Keep **4,035** · Turn Right **101** · Turn Left **85** · Right Lane Change **82** · Slightly Shift Left **69** · Slightly Shift Right **31** · Left Lane Change **22** |

⇒ **21 distinct meta-action tokens across three axes, with an explicit severity ladder on each**
(Gentle/Strong; Steer/Sharp Steer; Slightly Shift/Lane Change/Turn). Against our 5-way mixed softmax,
the representable space is **7 × 7 × 7 = 343 combinations vs 5 mutually-exclusive labels**.
⚠️ **The `Stop` short-circuit, MEASURED at n=39 and worth re-checking at 4,729:** *"In all 5 rows where
the longitudinal action is `Stop`, generation ends before the Lateral and Lane axes are emitted. The
axes are therefore *not* fully independent in Alpamayo's own scheme — a factorisation we borrow must
decide deliberately whether to reproduce that."*
⚠️ **Sampled, not modal:** *"`generate_text` runs at temperature 0.6; one draw per clip, seed 42.
Stability across draws is UNMEASURED and is a work item — a κ computed on single samples has a variance
floor we have not quantified."* (`seed = 42` on all 23,644 rows is confirmed MEASURED in §2.5.)

### 5.2 ⭐ THE LEARNING — "our arm commands 5.18× the acceleration a human does"

`ALPAMAYO2_SUPER_ANALYSIS.md` §13, **MEASURED 2026-08-06, 39 clips, 2 s, dt = 0.1 s**: recover the
controls each arm's path *implies* (inverse unicycle) and compare in the units a control head emits.

| | Alpamayo 2 Super | TanitAD flagship | human (GT) |
|---|---|---|---|
| implied accel RMS (m/s²) | 1.027 | **4.1656** | **0.8048** |
| **× the human's accel magnitude** | **1.27×** | **5.18×** | 1.00× |
| implied accel MAE vs human (m/s²) | 0.5090 | 1.7350 | — |
| implied accel **bias** (m/s²) | +0.1076 | **+0.7160** | — |
| implied curvature RMS (1/m) | 0.030829 | 0.032496 | 0.020914–0.024488 |
| implied curvature MAE (1/m) | 0.008497 | **0.008275** | — |
| entry transient MAE (m/s²) | 1.4039 | 1.5367 | **0.4249 (floor)** |

⭐ *"**Our arm commands 5.18× the acceleration magnitude a human does.** Alpamayo commands 1.27×. That
is the single most specific statement the programme has about the longitudinal defect, and **no ADE at
any horizon can express it** — a path can match position while thrashing the throttle."*
⭐ *"**And the launch is NOT where the two arms differ.** Both sit at ~1.4–1.5 m/s² of entry transient
against the instrument's own floor of 0.4249 … **The defect is in the sustained acceleration profile**,
which is exactly what an integrated `(accel, curvature)` head constrains and a free-waypoint head does
not."* ⚠️ On curvature the two arms are the same and **both over-command** relative to the human —
*"the lateral channel is not our problem."*

⛔ **Two traps this work caught "both of which would have shipped a number":**
1. **An off-by-one in the inverse map** — *"`rollout_unicycle` advances position on the speed at the
   start of each step … The naive version drifted **1.2233 m** over 2 s and returned every control one
   step late. Caught only because the round-trip test integrates → recovers → re-integrates."*
2. **"`tanh` is not a safe saturating squash."** MEASURED: *"`1 - tanh(51)**2` is **exactly 0.0** in
   float32, so a control far outside its limit has an underflowed gradient — the same silent dead-head
   a hard `clamp` produces, moved out to where nobody tests. Replaced with softsign `x / (1 + |x|/limit)`,
   whose 1/x² decay leaves ~3.7e-4 at the same overshoot."*
3. **Curvature at a standstill is undetermined, not large** — ungated, implied-curvature MAE came back
   as **1.6 × 10⁶** and **7.6 × 10³** 1/m; gated at `MIN_DS_MPS = 0.5`.

⚠️ **We already had half the machinery and it was dead code:** *"`rollout_bicycle` — a differentiable
kinematic-bicycle integrator with a Kamm-circle penalty — has been in the repo since H14 Track 1, is
exported from `models/__init__`, is NaN-tested, and is imported by **no model and no trainer** (verified
by grep over `stack/`)."*

### 5.3 The retrain-free precursor — re-timing, and its honest price

`…/incoming/2026-08-06-v1-defect-triage/results/RETIME_AB_RESULT.md`, **MEASURED 2026-08-06**,
`flagship-v1arch-v2bal-30k` @ step 29999, **6,834 windows / 40 OOD-val episodes**, artifact
`results/retime_ab_v1arch_tangent.json.xz`. Limits are *"a **rule**, not a tuning"* — the human's own
p99 on this corpus computed in the same pass: **accel 2.6890 m/s², jerk 6.3686 m/s³**.

| | BEFORE | AFTER | human | change |
|---|---|---|---|---|
| ADE @2 s (m) | 0.3585 | **0.3205** | 0 | **−10.6 %** |
| speed bias (m/s) | +0.3794 | **+0.0328** | 0 | **−91 %** |
| speed MAE (m/s) | 0.7172 | **0.4803** | 0 | −33 % |
| along-track bias @2 s (m) | +0.7527 | **+0.0647** | 0 | **−91 %** |
| accel RMS (m/s²) | 3.8166 | **1.1919** | 0.9075 | 4.21× → **1.31×** human |
| **jerk RMS (m/s³)** | **52.1281** | **4.9535** | 1.7051 | 30.6× → **2.90×** human |
| entry transient (m/s²) | 1.9825 | **0.0064** | 0.5487 | **−99.7 %** |
| **curvature MAE (1/m)** | 0.006103 | **0.006922** | 0 | ⚠️ **+13.4 % — WORSE** |
| **replan accel jump mean (m/s²)** | 1.1021 | **0.4336** | — | **−61 %** |

⭐ *"**The frame-to-frame control jump fell 61 %, and I predicted it would not.** … **A prediction
committed in advance and falsified by the measurement — recorded as such rather than quietly dropped.**"*
⛔ And the counter-discipline: the doc's *own* mechanism for the curvature regression was then **tested
at full scale and refuted** — the arc-extrapolation fix changed curvature MAE by **−0.03 %** (0.006922
→ 0.006920) and left every temporal metric **byte-identical**. On sampling-independent quantities:
**net yaw error over 2 s 0.1201 → 0.1944 (+62 %, genuinely WORSE)** while **cross-track at 2 s 0.1961 →
0.1565 (−20 %, genuinely BETTER)**. ⛔ *"**No third hypothesis is offered.** … The honest position:
7 metrics improve substantially, cross-track improves, heading degrades, and why the heading degrades
is an OPEN QUESTION."* ⚠️ *"**No CI** … The decision-grade form is the episode-cluster bootstrap over
the 40 episodes — a work item."* And **it must not be deployed on the arithmetic alone**: *"heading
error is what a lane-keeping failure looks like, and 4° is not nothing."*

⚠️ **A superseded figure recorded rather than smoothed:** *"The ADE gain is smaller at scale than on
39 clips (**−10.6 % vs −19.0 %**). The 39-clip figure was optimistic; quote this one."*

### 5.4 The head design — four optimisations, each from a measurement

`…/incoming/2026-08-06-v1-defect-triage/OPTION2_UNICYCLE_READOUT.md`. First, **what v1arch's head
actually is** (⛔ a correction: *"`flagship-v1arch-v2bal-30k` has **no anchored-diffusion head**"* —
the earlier root-cause analysis pointed at the v1.5/v4/v5f lineage and *"is retracted"*): 20 waypoints
from `rollout_decode(...)` decoded by `StepDisplacementReadout` into a **free (dx, dy, dyaw)**.
Params: encoder **87.02 M** · predictor **91.36 M** · **`step_readout` (the head) 6.32 M** · world
model total **263.44 M** ⇒ *"6.32 M trainable against a 178 M frozen trunk. That is why Option 2 is
hours, not days."*

**Why a free (dx, dy, dyaw) decode produces exactly the measured defects:** *"`dx_j` is free ⇒ the
implied speed can jump between steps ⇒ jerk RMS **52.13** vs a human **1.71**. `dx_1` is free of the
true `v0` ⇒ launch transient **1.98 m/s²** vs a **0.55** floor. `dy_j` is free ⇒ the decoder may
translate the ego **sideways**. A road vehicle cannot. `dyaw_j` is independent of speed ⇒ the decoder
can turn while stopped."* ⭐ *"The unicycle removes all four **as representable states** … `dy == 0`
**is** the non-holonomic constraint."*

⭐ **The target was proven reachable before any GPU was spent** (MEASURED, 39 clips): re-integrating the
controls the **human's own path** implies gives **mean position residual 0.0477 m** (p90 0.1239, max
0.2614) and **net-yaw error of the reconstruction 0.00114 rad** against v1arch's 0.1201 and re-timed
0.1944. *"0.00114 rad is 0.065°, against v1arch's 6.9°. The ceiling is ~100× better than where we are.
⇒ **Therefore any heading error remaining after Option 2 is a LEARNING or LATENT-INFORMATION limit,
not a representation limit.** That is the single most useful thing to know before the run, and it cost
30 seconds."*

**The four head-parameterisation optimisations (§7), each from the human's own recovered controls:**
1. **Per-channel output scaling — a 38.5× imbalance.** accel std **0.80438** vs curvature std
   **0.02091**; one `Linear(hidden, 2)` gives both the same initial gradient scale ⇒ *"the curvature
   channel is ~38× under-resolved."*
2. **Predict YAW RATE, not curvature.** `kappa = yaw_rate / v` explodes as v → 0: curvature kurtosis
   **38.9** vs yaw-rate **10.4**; |·| p99 at v<3 m/s **0.1748 vs 0.3452**, at v>8 m/s **0.0184 vs
   0.1521** ⇒ **low/high-speed tail ratio 9.5× vs 2.3×**. ⛔ **And the trap in doing so:** predicting
   yaw rate unbounded *"would quietly restore turn-in-place"* ⇒ clamped to `±|v|·kappa_max`, so at
   v = 0 the bound is 0.
3. **Speed as an input** (the 9.5× tail ratio *is* a conditional dependence on v). ⚠️ Not privileged:
   *"`v0` is already a model input under `speed_input=True`."*
4. ⭐ **Predict the DELTA, not the level — the biggest one.** accel: abs std 0.80438, **delta std
   0.17494 (ratio 0.22)**, **lag-1 autocorr +0.977** ⇒ *"a **4.6× easier target** … nowhere near white,
   which is the condition under which delta-prediction would hurt."* ⭐ *"**A delta head's natural
   output scale IS THE JERK.** Smoothness becomes the DEFAULT rather than something a barrier term has
   to fight the head for."*
⛔ *"Every choice is a separate flag … **An arm that changed four things at once and improved would be
UNATTRIBUTABLE** — the `--v2` conflation failure."*

⚠️ **And the constants were then re-measured on train — the val estimate was materially wrong.**
MEASURED 2026-08-06 over **1,500 train episodes / 33,004 windows / 660,080 samples — 846×** the 780
val samples: **yaw-rate std 0.06930 → 0.13091 (nearly 2×)**, accel std 0.80438 → 0.88361, accel p99
2.172 → 2.7847. *"**The val estimate was wrong on the channel that matters** … training with the val
constant would have left the lateral channel ~2× under-scaled … **A 780-sample estimate of a tailed
quantity is not an estimate.**"* ⭐ Every design choice survived and two got stronger: **yaw-rate
delta/abs 0.50 → 0.195 with autocorr +0.874 → +0.981** (*"a 5× easier target on train, not the 2× val
suggested"*), tail ratio **2.3× → 1.46×**. ⚠️ One constant is still val-derived and flagged: `jerk_limit`
6.369.

### 5.5 ⛔ The WM-reliance guard — the risk §7's own optimisations created

*"Sayed, 2026-08-06: 'we need to be very careful not to train a driving dynamic predictor not using
the wm.' ⭐ **The risk is real and §7's own optimisations created it.**"* Two of the four choices open
WM-bypassing paths: **speed as an input** (*"on a mostly-straight, mostly-constant-speed corpus `v`
alone reconstructs most of the trajectory"*) and **delta prediction with carried `a_prev`/`yr_prev`**
(*"lag-1 autocorrelation +0.983, so `a_j ≈ a_{j−1}` is an excellent predictor **using no latents at
all**"*). *"A head taking both would score well on ADE and carry **none** of the world model's content
… **I did not flag this when proposing them.**"*

⛔ **"'Real beats none' is NOT the test"** — the strict test is **real vs MEAN**. Five arms: `real` ·
`mean` (per-window content removed, distribution kept) · `shuffled` (pairing removed, marginal kept
exactly) · `frozen` (temporal content only) · `cv` (the analytic floor). The metric:

```
wm_reliance = 1 − (ADE(cv) − ADE(mean)) / (ADE(cv) − ADE(real))
```

*"Of everything the decoder adds over constant velocity, what fraction REQUIRES the world model's
per-window content?"* **Gate pre-registered at 0.5**, *"deliberately not 0.9: the ego-state pathway is
legitimate."* **UNAVAILABLE** when the head does not beat CV — *"reporting 0.0 would read as 'bypassed'
when the truth is 'not computable'."* Counter-measures in the head: **`shortcut_dropout`** (0.1) blanks
the `(v, a_prev, yr_prev)` **input** columns while the integrator keeps the real values — ⛔ *"Dropping
the LATENTS instead would be exactly backwards: that teaches robustness to missing latents, i.e. it
**rewards** the shortcut"* — and **`detach_feedback`**.

### 5.6 ⭐ THE TRAINED RESULT — the head that cannot cheat beats the head that could

`…/incoming/2026-08-06-v1-defect-triage/results/UNICYCLE_RUN5_RESULT.md`, **MEASURED 2026-08-06**,
3,000 steps / 58 min, frozen `flagship-v1arch-v2bal-30k` trunk (**md5 `c11575…` identical before/after**),
**2.11 M trainable params**, val = the fixed 128-window OOD-val q90 batch.

| val | baseline (displacement readout) | **run 5 (latents-only)** | human | pre-reg clause |
|---|---|---|---|---|
| ADE (m) | 0.3612 | **0.3571** | 0 | (not required) **−1.1 %** |
| speed bias (m/s) | +0.4094 | **−0.0962** | 0 | \|·\| < 0.15 ✅ |
| accel RMS (m/s²) | 3.5298 | **0.7273** | 0.91 | ≤ 2× human ✅ |
| jerk RMS (m/s³) | 44.0103 | **1.2891** | 1.71 | ≤ 3× human ✅ |
| **net-yaw err (rad)** | 0.0328 | **0.0130 (−60 %)** | 0 | ≤ baseline ✅ |
| **wm_reliance** | (8.72) | **0.6233** | — | ≥ 0.5 ✅ **GATE PASS** |

**Run 4 (the shortcut head), for contrast: ADE 0.4078, net-yaw 0.0298, reliance 0.089 FAIL.**
⭐ *"**The head that cannot cheat beats both the head that could and the original.** Removing the
shortcut inputs (v, control feedback) moved reliance 0.089 → 0.623 **and** ADE 0.408 → 0.357. The
crutch was not even buying accuracy — it was pure gradient-descent path-of-least-resistance."*

⭐ **This is the direct answer to the re-timing trade-off in §5.3:** re-timing bought 6 of 7 metrics but
cost **+62 % net yaw**; the trained unicycle head bought them **and improved net yaw by 60 %** — i.e.
*"Re-timing can only re-time a curve it did not choose. Option 2 chooses the curve."*

⚠️ **A confound that travels with every v1arch number, stated in the same doc:** `rollout_decode` is
conditioned on the **TRUE future actions** ⇒ *"The deployed 20-waypoint trajectory is therefore a
grounded, action-conditioned world-model rollout, **not an autonomous plan** … no number from this arm
may be presented as closed-loop planning performance."* (This is the **T0/T1** distinction that
`EVAL_DOCTRINE.md` later made binding.)

### 5.7 What v6 ADOPTED, and what it REJECTED

**ADOPTED — the unicycle emission, as the v6 planner's output contract.** `V6_TRAINER_DESIGN.md` §1.1
imports `scripts/train_v58f_unicycle_head.py::UnicycleEmission` for the 60-step emission, citing it as
*"W4-gated: `a = a_max·tanh`, `κ = κ_max·tanh` — **feasible by construction**, census violations 0.0"*.
⚠️ **Note the residual disagreement between two of our own documents:** the v6 emission uses **`tanh`**,
while `ALPAMAYO2_SUPER_ANALYSIS.md` §13 measured that *"`tanh` is not a safe saturating squash"* and
replaced it with **softsign** in `kinematic.py`. Both are in the repo; the v6 path takes the tanh form.
**Recorded, not smoothed** — it is a real, small, checkable divergence and a cheap thing to re-verify
before S-T trains the emission head.

**ADOPTED — the 6 s / 60-step plan contract**, and this is downstream of Alpamayo directly.
`HIERARCHICAL_WM_REDESIGN.md` §4 lists the PUBLISHED anchors: *"Alpamayo 2 Super emits **6.4 s @ 10 Hz
(64 wp)**. nuPlan planning evaluates **8 s**; Waymo motion/planning benchmarks use **8 s**; UniAD plans
**6 s**; nuScenes prediction convention is 6 s. **2 s is short of every planning-grade reference.**"*
⇒ recommendation *"operative emitted plan → **6 s (60 wp)**"* ⇒ `HIERARCHY_VOCABULARY.md` §4b makes it
**BINDING** ⇒ `V6_TRAINER_DESIGN.md` runs `o5_k=60` with `--max-horizon 60`. **So `o5_k=60 / 6 s` is
Alpamayo-anchored in its origin and PI-bound in its authority** — the handover's guess that it is
"likely downstream" is **confirmed, via the horizon-anchor table, not via the action space**.
⚠️ **And it has a measured price:** *"the window COUNT drops … `120 − 6 − 20 = 94` windows at v4's
horizon and `120 − 6 − 60 = 54` at 6 s — **≈43 % fewer**. That is a real distribution change vs v5f and
belongs in the run row."* ⚠️ **The trap it nearly walked into:** *"MEASURED dev-side (2026-08-11):
`_plan(_eval_cfg())` … returns **`max_horizon = 20`**, i.e. each window carries **2 s** of future. A v6
trainer that inherited it would make §4b's 6 s horizon **structurally untrainable**, and it would fail
*looking like a corpus limitation*."*

**ADOPTED — the factored, severity-carrying manoeuvre vocabulary** (§1.3 above), on the explicit
evidence that *"this is a working system that does not have [the mixing defect], with the measured
consequence"* of §2.4's opposite-direction κ finding.

**ADOPTED — the goal point as the strategic/tactical lever.** `HIERARCHY_VOCABULARY.md` §4's
`ANCHOR_GOAL` is annotated *"the geometric goal-point lever (**the +4.7 PDMS class**)"*, matching the
binding PI rule of 2026-08-03 (*categorical command +0.2 PDMS; goal point **+4.7***).

**REJECTED / NOT TAKEN — with reasons:**
- ⛔ **Do not fine-tune or chase Alpamayo** (§8's explicit "what NOT to do"), and *"the interesting axis
  is capability per parameter and per camera."*
- **Flow matching + Euler-10** is **Tier-2 "strong" and was never run** — the campaign's sampler work
  went to W4/W7 instead.
- **The trajectory tokenizer** (3000-bin future / 1000-bin history) is Tier-2 #6, *"the missing bridge
  if we ever want a text-reasoning layer above the hierarchy"* — **not adopted**; v6 tokenises **goals**,
  not trajectories, and keeps *"continuous (a, κ) unicycle controls — no tokenization at 10 Hz."*
- **`PerWaypointActionInProjV2` Fourier features**, **non-causal joint denoising**, **CoC traces as a
  training signal**, **LingoQA as external calibration**, **the camera-count ablation** — all Tier-2/3,
  **none run**. ⚠️ *"Our programme has **no external calibration at all**"* remains true.
- ⛔ **`cotrain_expert_vlm: false` is the one Alpamayo choice v6 independently re-derived**: our own
  W7-FULL consumer-invalidation result (*"you cannot repair a trunk and keep its planner"*) produced
  the same **frozen-trunk, post-trained-consumer** shape — and `V6_TRAINER_DESIGN.md` §3.2 names the
  precedent as *"Drive-JEPA's shape (the planner is a **post-trained consumer**)"*.

---

## 6. THE UPDATED SCIENTIFIC PAPER — what the campaign changed, and its current claims list

**Owner:** `Paper/TANITAD_PAPER.md` — **living paper, v1.0 (2026-08-12)**, **3,275 lines**.

### 6.1 The campaign's diff, from `git log -- Paper/`

**`git diff be2da04..eb877a1 --stat -- Paper/`: 12 files changed, 2,712 insertions, 5 deletions**,
including four new figure assets (`v6_architecture.svg/.png`, `winners_curse.svg/.png`, plus the
v5.8f results figure). Six commits touch `Paper/` in the campaign window, newest first:

| commit | what it landed |
|---|---|
| `4531868` | **Two PI-raised corrections** — the VLM-arm mischaracterisation (§3.4 above) and **"THE BEV CONTAINS NO ROAD STRUCTURE"** (§6.4 below), plus the BEV figure that makes the second visible |
| `85cee64` | ⭐ **T1 PSEUDO-CLOSED-LOOP LANDS** — the single most consequential result of the campaign (§6.3) |
| `31a0841` | Figure 1 (v6 architecture) referenced in §3; Figure 2 (v5.8f repair arc) in §7.13 |
| `cdc565b` | The v5.8f results figure, **"generated from registry literals so it cannot drift"**; palette run through the dataviz validator |
| `f279dd2` | v6 figure footer trim |
| `ddcfbe4` | PH0 pilot package (2,221 lines, **37 CPU tests green**) + the paper-quality **v6 architecture SVG/PNG** (4 layers, per-layer predictors, goals down / latents up through **tested** gradient-isolation barriers, the 6 s single-rollout trajectory band, frozen interpretation heads, the staged S-W/S-T/S-S/S-J strip) |
| `0aeee08` | **Paper → v0.9** (+582 lines): new theory §3.9, §5 extended with the episode-cluster bootstrap maths + the binding T0/T1/T2 doctrine, results §7.12b–§7.15, roadmap 7–11, 4 new references |

⚠️ **A versioning discipline the paper applies to itself, verbatim:** *"This round was requested as
'v0.7' but that number is taken — it is therefore v0.9, and the numbering is again not silently
reused."* And v1.0's justification: *"It is v1.0 rather than v0.10 because it is the first version in
which a hypothesis this paper itself carried is **retracted by our own measurement** in the paper's own
voice, §7.17."*

### 6.2 The current claims list — v1.0's four verdicts, "two of them negative and both reported as results"

Verbatim from the abstract's sixth-round update:

1. ⛔ **"The selector-free planner fails, and the mechanism is the winner's curse."** *"selection by
   argmin of a world-model roll-consistency cost plus a kinematic cost scores **3.3348 m against a
   0.4505 m gate over a fan whose oracle is 0.1273 m** … The cost's within-window rank correlation is
   0.445/0.497 and its across-window calibration ρ 0.3185 [0.2064, 0.4086] — yet **the argmin's
   error-rank is 132 of 256, the median**, and the mean error inside its top-m set is flat at the fan's
   own mean for every m. **Rank correlation is a bulk statistic; argmin is an extreme one, and the
   quantity that governs selection is lower-tail dependence, which is zero here (§3.12).** A
   self-consistency cost has a **degenerate minimiser** — a near-stationary candidate — so deepening
   the fan makes a minimiser monotonically worse while the oracle improves; **the published loops we
   copied (V-JEPA 2-AC, DINO-WM) minimise distance to a *goal*, which inaction cannot minimise, and we
   had dropped that term.**"*
2. ⛔⭐ **"The standing hypothesis that planner co-training erodes physical representation is REJECTED
   within the measured range and retracted in this paper's own voice"**: *"across the λ_plan ramp every
   probed physical variable became *more* decodable (**curvature 0.213 → 0.551 encoded, yaw-rate
   0.583 → 0.869**), the latent's participation ratio **expanded 53 % (4.53 → 6.94 of 2048)** and the
   P1 battery went FAIL → PASS — which also **validates SIGReg** under a full planner gradient
   (**retention 1.53× against a ≥ 0.8× gate**), with three scope bounds stated rather than buried."*
3. **"The predicted latent retains the environment"**: *"a frozen-latent BEV occupancy readout gives
   **retention 0.932 at k = 10** (gate ≥ 0.80, **τ\* chosen on the encoded arm so the gate can only
   harden**), and occluded-agent recall is **not worse** than visible — the latent carries agents the
   camera cannot see (§7.18; **absolute IoU ≈ 0.02, so the ratio is the quotable claim**)."*
4. **"Imagination is load-bearing"**: *"zeroing it collapses selected ADE **0.4011 → 7.6493 m** and
   shuffling it — which preserves marginals and destroys only correspondence — gives **1.2492 m**, so
   the planner reads imagination as **content, not as a bias term**."*

Plus the **consumer-invalidation** result that defines the next generation: *"repairing a trunk moves
the frozen selector **0.7933 → 4.4159 m**, which is why the shipped assembly is the frozen-trunk one at
**0.4815 m [0.3928, 0.5771]**"* ⇒ *"these define the programme's next generation as a **staged** ladder
in which every consumer is trained on the trunk it consumes (§10)."*

**The v0.9 round's two binding methodology corrections**, also in the abstract: the **eval-tier
doctrine** forced by the measured action echo (*"97.9 % S-curve reproduction under the true action
transcript and ~5 % closed-loop / **0.0 % hold-action**"*), and *"the completed retirement of the
historical `overlapping_holdout_se` estimator, which biased **point estimates** (−6.67 % to +11.69 %,
sign-flips on paired deltas) and not only intervals."* Plus the physics-proof battery locating *"**one
root defect under four separate failures** — the trunk responds to counterfactual actions with the
right sign (99.5 % lateral) inside a **3-dimensional action subspace** but at **~¼ the physical gain**
(0.27)"* and its stage-A repair, after which *"the world-model-roll selection cost became the
programme's best-calibrated selection signal (**Spearman ρ 0.716 [0.585, 0.770] vs ≤ 0.26 for every
learned scorer**)"*. And the sharpest absence: *"**no readable lead-vehicle distance at any probe
capacity** (`p1_lead_transforms.json`), answered by a **label-free lever program rather than label
injection**, per the PI's standing rule that labels into the trunk would break the self-supervised
thesis."*

**Model size as the paper states it (§ abstract):** *"263.4 M parameters measured — `total_model`
263,442,838, `trainable` 277,404,073, `MODEL_REGISTRY §1.2`; the design budget every arm is matched to
is 261 M, D-008."* ⚠️ **This is the pre-v6 number.** v6 config E is **336.5 M** with the `param_budget`
raised to 350 M — i.e. **the paper's headline parameter count is one generation behind the live
architecture**, which is expected for a paper whose §10 treats v6 as future work, but it is the first
thing to update when v6 produces a result.

### 6.3 ⭐ THE PAPER'S NEWEST RESULT — the T1 pseudo-closed-loop block (registry §1.14, T1 = PRIMARY)

**MEASURED, 6,844 windows / 40 val episodes, stride 1, episode-cluster bootstrap,
`overlapping_holdout_se` used nowhere.** This is the campaign's last science commit (`85cee64`) and it
is **not in the consolidation report**.

| arm | surface | tier | ADE dense (m) | FDE last (m) | LON speed MAE (m/s) | LON along MAE (m) | LAT cross MAE (m) | LAT heading MAE (°) |
|---|---|---|---|---|---|---|---|---|
| `v5f-30k` | `cl` | **T1** | **23.9837** [21.442, 26.347] | 53.4756 | 26.9356 | 23.8965 | 0.9993 | 3.6204 |
| `v5f-30k` | `ol` | T0 | 0.9397 [0.8162, 1.0679] | 2.8003 | 1.4431 | 0.8762 | 0.1947 | 3.3483 |
| `v5f-30k` | `ha` | **T1** | 0.9597 [0.8361, 1.0879] | 2.8631 | 1.4531 | 0.8901 | 0.2072 | 4.5954 |
| `stage-a-repaired` | `cl` | **T1** | **9.3697** [6.6822, 12.2576] | 19.5256 | 9.7291 | 9.2655 | 0.7446 | 5.3945 |
| `stage-a-repaired` | `ol` | T0 | 0.3659 [0.2926, 0.4521] | 1.0231 | 0.5113 | 0.2990 | 0.1534 | 1.8351 |
| `stage-a-repaired` | `ha` | **T1** | 0.4246 [0.3500, 0.5132] | 1.2242 | 0.5671 | 0.3487 | 0.1689 | 3.9859 |

**FINDING 1 — the stage-A repair wins on every surface, separated.** Paired `stage-a-repaired − v5f-30k`,
same windows: **`cl` ADE −14.6139 [−16.9319, −12.2010]** (`p_delta_gt0` 0.0) with **`cl` LON speed MAE
−17.2064 [−19.7815, −14.4927]**; `ol` ADE −0.5739 [−0.7002, −0.4570]; `ha` ADE −0.5351 [−0.6644,
−0.4181]. *"The repair … improves **exactly the axis it targeted** — a clean confirmation, not a
coincidence."*

**FINDING 2 — ⛔ THE CLOSED LOOP DIVERGES, AND THE HOLD-ACTION CONTROL BEATS IT BY 22×.** Repaired arm
`ha` **0.4246** vs `cl` **9.3697**; v5f `ha` 0.9597 vs `cl` 23.9837. Within-arm paired `cl − ol`:
**+9.0039 [6.3659, 11.8487]** and **+23.0439 [20.5613, 25.3884]**, both separated. ⇒ **"No closed-loop
driving competence may be claimed for either arm."** *"The T0 number (0.3659) and the T1 number (9.3697)
differ by **25×** on the same checkpoint and the same windows — this row is the strongest evidence yet
for the tier doctrine."*

⭐ **THE DIVERGENCE IS ~99 % LONGITUDINAL — visible ONLY because the four families are reported.**
*"Of the repaired arm's `cl` ADE 9.3697, **LON along-track MAE is 9.2655** while **LAT cross-track MAE
is 0.7446**; for v5f, 23.8965 of 23.9837 against 0.9993 lateral. **The car holds its lane and its SPEED
integrates away** (LON speed MAE 9.73 and 26.94 m/s — both physically implausible, i.e. a true blow-up
rather than a graceful degradation). This **sharpens the standing '88.7 % of the oracle gap is
longitudinal' (T0) to ~99 % at T1**, and it converges with the P1 verdict that the latent lacks a
readable lead-distance variable. ⚠️ A scalar ADE would have shown a 25× gap and NOT shown that it is one
axis — this is the four-family rule earning its cost in a single row."*

**Consequences for v6, verbatim and load-bearing:** *"(1) The staged ladder is now empirically, not just
architecturally, motivated: **S-W must produce a world model stable under its OWN actions before any
planner is attached** — that is precisely the quantity `cl − ol` measures, **and it is the natural S-W
gate**. (2) The longitudinal channel is the design target, not the lateral one. (3) **`ha` is the floor
any closed-loop claim must clear first**; clearing `ol` is not evidence of anything driving-related."*

⚠️ Scope stamped: *"`ha` is a strong baseline partly **because** the corpus is short-horizon and
near-constant-speed — that is what makes it the right 'do nothing clever' floor, not a reason to
discount it."* TACTICAL/STRATEGIC were UNAVAILABLE in this run; **TACTICAL has since been closed at
source for all future T1 runs** (`t1_eval.py` now passes `tactical_from_traj=True, tier=t` — *"at T1
the driven path IS the manoeuvre decision"*), STRATEGIC stays `n/a` with reason + n.
⚠️ Instrument hazard recorded: both arms rolled all 40 episodes and *"then died in `analyze()` on
`from taniteval import selgap`"*; the dumps survived so the numbers come from `--analyze-only` with
**zero GPU recompute** — *"an analysis-time import that fails after 11 minutes of rollout is a standing
hazard."*

### 6.4 Claims in the paper NOT yet surfaced in any report

1. ⛔ **"THE BEV CONTAINS NO ROAD STRUCTURE"** (commit `4531868`, PI-raised). From source
   (`bev_raster.py:79-82`) the raster is `obstacle.offline`, *"whose full measured enum is 10 classes,
   **ALL DYNAMIC AGENTS**. No lane boundary, no road edge, no drivable area."* Two of the programme's
   own phrasings are corrected: *"the 'ego corridor' is a **HAND-DEFINED ±1.5 m band** (cols 29-34),
   my assumption and not a perception; and 'knows where to go' is **bearing accuracy toward the human's
   future position, not road understanding**."* ⇒ *"This sharpens the longitudinal story — the model
   has **no road-structure representation at all**."*
2. **§3.12 gives the winner's curse a formal treatment** (*"lower-tail dependence"*) with a dedicated
   figure (`winners_curse.svg/.png`) — the mechanism is now a **methods contribution**, not just a
   failed arm.
3. **§3.10 "the label-free commitment, stated as an admissibility algebra"** and **§3.11 "the 4-brain
   hierarchy, formally: per-layer predictors, goal conditioning, and the gradient-isolation matrix"** —
   the paper carries the v6 hierarchy's mathematics **before** v6 has produced a result.
4. **§3.13 "SIGReg, effective rank, and what 'retention' measures"** — pins the spectral instrument so
   *"the training monitor and the offline instrument cannot drift apart."*
5. **§5.5 promotes the four metric families to a named methodological section** ("the evaluation
   contract"), and **§5.4** does the same for the tier doctrine — i.e. both binding rules are now
   **publishable method**, not internal policy.
6. **§7.14's P1–P9 + I4 battery is framed as "a methods contribution"** including *"the lead-distance
   missing-variable verdict and its label-free response."*
7. **Figure discipline worth copying:** the v5.8f figure is *"generated from registry literals so it
   cannot drift"* — a figure that cannot disagree with the registry.
8. ⚠️ **Three numbers are flagged INHERITED in-text rather than silently promoted** (commit `0aeee08`):
   *"Alpamayo counts pending a registry row; P8 positives fraction pending tonight's banked gate; one
   truncated digit described qualitatively."* ⇒ **The Alpamayo augmentation counts still have no
   registry row** — §2.5 of this addendum supplies the measured values (23,644 / 4,729 / one
   quantisation arm) that would close that flag.

---

## 7. THE "ETC." SWEEP — result-bearing campaign docs not covered above or in the consolidation report

Scanned: `git diff --name-only be2da04..eb877a1 -- "*.md"` (89 files), plus the `2026-08-0[5-9]` and
`2026-08-1*` incoming folders and top-level `Project Steering/*.md`. Already covered elsewhere and so
excluded: v5f/v1arch finals, Stage-A, v5.8f/W-wedge, P-battery echo, G1, PH1 fusion counts, Orbis-2
headline (consolidation report); the seven topics above. One paragraph each for the remainder.

**`E_ENC_RESULT.md` — the encoder-width arm, decided at step 500 (MEASURED, pod5, same seed 0, same
corpus, back-to-back).** PI decision D-A (2026-08-13) paused S-W to run this before committing ~60 A40-h.
Arm (a) **384×8, 87.89 M total** vs arm (c) **768×12, 159.93 M**: total loss **2.9720 vs 3.5924**, O1
factual **0.3672 vs 0.8824**, **O1 factual ADE 0.6141 vs 1.4405 (2.35× worse on the wide encoder)**,
O3 masked-cell 0.0270 vs 0.0333, O6 SIGReg 15.75 vs 16.17, **s/step 7.19 vs 10.76 (1.50× faster)**.
⇒ **"(a) wins on 7 of 8 objectives and is 1.50× faster per step."** ⚠️ **The handover's own caveat
stands and matters: "the ViT-5-form width question is UNMEASURED — the running encoder is a different
architecture than either arm"** (config E's ViT-5 has registers + RoPE). So this result rules out
*plain* width, not the shipped encoder's width. Re-running E-ENC in ViT-5 form is on the deferred queue.

**`V6_TRAINER_DESIGN.md` §2 — the parameter budget and the matched-params rule.** MEASURED at
instantiation: shared-encoder **87.89 M** vs per-layer-encoders **120.74 M**. ⚠️ **"E-ENC decides at
MATCHED TOTAL PARAMS, not at matched per-layer widths. Matching by eye is how an arm wins on capacity
and gets read as winning on architecture."** The matched pair is `--per-layer-encoders` (120.74 M) vs
shared with `--pred-dim 960` (**118.11 M**), **a 2.2 % residual gap — "quote the gap, because 'matched'
with a 30 % gap is not matched."** Prior stated: *"every frontier system (V-JEPA2, DINO-WM, Drive-JEPA)
uses ONE encoder with downstream consumers. Separate encoders must **earn** their params; a tie goes to
(a)."*

**`V6_TRAINER_DESIGN.md` §5/§7 — cost, and a suite hazard filed rather than hidden.** S-W cost is
**ESTIMATED 175–290 A40-hours** for 30 k steps (whole ladder ≈ **220–370 A40-h**), with the honesty
stamp *"the programme's own history says estimates here run ~11 % low (v4's '~53 h ESTIMATED' against
MEASURED 59.04 h)"* ⇒ **"THE FIRST ACTION ON THE POD IS TO RE-COST FROM THE RUN'S OWN LOG."**
*(The campaign's realised number, from the consolidation report: **17.37 s/step**, i.e. ~145 A40-h for
30 k — inside the low end of the band.)* §7.0 also files **23 pre-existing suite failures** with a
proper control (*"the same 23 failures occur with and without this work"*): 1× missing `onnx`, 2× a
Windows-basename assertion in `test_resim.py`, **20× suite-order-dependent in `test_rig_clean_fix.py`**
(all 20 pass in isolation). *"They are not v6 work — filed here so they are visible rather than
rediscovered by whoever next needs a green suite."*

**`Project Steering/LEAK_v1arch_val_2026-08-05.md` — v1arch trained WITHOUT parity, and the canonical
val is INSIDE its training pool (MEASURED, manifests read directly).** From the arm's own config:
`"v2_parity": {"parity": false, "corpus_key": null, "checked": false, "clips_present": 9000}` and
`"require_parity": false`. It trained on a **9,000-clip** pool, not the canonical 2,376-episode
`physicalai-train-e438721ae894` (skip-hash `f09e44db`). ⭐ **Found "while preparing the 30k gate — i.e.
BEFORE a number was published, not after."** This is why the v1arch four-family block is reported on the
**OOD-val 290** corpus rather than the canonical val, and it is the third member of the leak family
after REF-A I-JEPA (~80 % val leak) and the nav-echo.

**`Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md` + `REACHABILITY_IS_COMPUTE_NOT_SELECTION_2026-08-05.md`
+ `DRIVE_DOC_DELTA_IS_CRLF_2026-08-05.md`.** The first defines the clean OOD-val protocol the leak
forced; the second is a titled negative result (**the reachability filter is a compute lever, not a
selection lever** — consistent with REF-C's S2 band deleting 72.08 % of the fan for **ΔADE exactly
0.0000**, §4.6); the third is an ops finding (a Drive-visible doc delta that was purely CRLF).

**`…/incoming/2026-08-06-v1-defect-triage/` — the rest of the triage family.** `V1_DEFECT_TRIAGE.md`
(the parent), `PREREG_V161_SPEEDLOSS.md` (a pre-registered speed-loss arm), `UNICYCLE_RETRAIN_PLAN.md`,
and three further result docs: `GATE_RERUN_RESULT.md`, `TEMPORAL_STABILITY_RESULT.md`,
`UNICYCLE_RUN4_RESULT.md` (**run 4 = the shortcut head: ADE 0.4078, net-yaw 0.0298, wm_reliance 0.089
FAIL** — the control that makes run 5's 0.6233 PASS interpretable, §5.6).

**`WM_PHYSICS_PROOF.md` + `JEPA_PHYSICS_SURVEY.md` — the label-free doctrine's evidence base.** The
survey was written *"after the PI's course correction"* (*"if we add a lot of labels, the whole idea
looses its charme"*) and **RETRACTS the auxiliary lead-readout-loss lever** while keeping the P1 H-absent
measurement (*"frozen probes that never train the trunk are LeCun-orthodox instrumentation; it is the
RESPONSE that must be self-supervised"*). Its four root-cause hypotheses for the missing lead-distance
variable — **RC1 pooling bottleneck** (*"the physical detail lives in spatial tokens; **pooling is where
geometry goes to die**"* — DINO-WM), **RC2 the objective never needs it** (*"rare-event variables are
exactly what a fixed-rank latent sacrifices first"*), **RC3 no spatial masking pressure**, **RC4
horizon** — are each label-free-testable, and RC3 is directly why v6 ships the **O3 contiguous-block
masking** measure. It also notes *"V-JEPA 2's planning loop is structurally OUR W7 roll-cost re-rank"*
and that V-JEPA 2.1 found *"dense spatio-temporal features do NOT emerge reliably from standard SSL;
fixes are **loss-shaping, not labels**."*

**`V6_TRAINING_MEASURES.md` / `V6_SIZING.md` / `V6_SIZE_VS_FRONTIER.md` / `V6_GO_PACKAGE.md` /
`V6_ARCHITECTURE_REVIEW.md` / `PI_DECISIONS_2026-08-12.md`.** The measure catalogue (O1–O6, T1–T5,
S1–S3, C1–C2, X1–X5) whose primitives §1.7 quotes; the sizing ladder (configs A–F) whose **config D vs C**
question Orbis-2 then answered externally (*"D is the only in-band configuration that grows the
**hierarchy** (56.7 M, 19.6 %) rather than growing the operative predictor further"*, against v6-as-running
at **68.6 % operative / 12 % hierarchy** — *"No system in this class allocates the way we do"*); and the
PI decision record that selected **config E**.

**The prereg family: `PREREG_E6_EFFICIENCY.md`, `PREREG_H_COTRAIN.md`, `PREREG_SCALING_LADDER.md`,
`PREREG_W4B_SELECTOR.md`, `PREREG_W4C_SPATIAL_SCORING.md`, `PREREG_W7_PROG.md`.** Six pre-registrations
with outcomes bound in advance; **W4B and W4C both FAILED their gates and were retired by their own
rules** (W4b feat: *"selected ADE 0.5600 vs ≤ 0.45"*, with the diagnosis *"train monitor 0.21–0.33 vs
held-out 0.56 — it memorises train-window selection"*), **H-COTRAIN's hypothesis was rejected** (§6.2
claim 2), and **W7-PROG returned PARTIAL** (§4.5e). ⭐ **The scaling ladder** (`PREREG_SCALING_LADDER.md`,
the §3.6 efficiency experiment) is *"the cheapest discriminating experiment for the programme's central
thesis"* — {150, 300, 600} episodes × hierarchy-vs-monolithic at matched added params, ~6 GPU-hours,
**both outcomes committed** (*"H ≤ M everywhere ⇒ claim refuted at this scale, and the next lever is the
goal-space (not more layers)"*). **It has still not been run** — and it is the only pre-registered test
of the hierarchy's data-efficiency claim.

**`V58F_FUSION.md` / `V58F_INTERP_PLAN.md` / `V5F_ARCHITECTURE_REVIEW.md` / `V5F_DATA_WIRING_AUDIT.md` /
`V18_BACKLOG.md`.** The v5.8f assembly and interpretation plans, the two v5f audits that found the
defects the repair arc then fixed, and the V18 backlog from which E3.4 (stage-A) was drawn.

**`VLM_MODEL_SURVEY.md` / `VLM_STRATEGIC_LABELING.md` / `PH0_TARGET_STRUCTURE_v2.md` /
`PH1_FUSION_STRATEGY.md`.** The VLM arm selection (survey → the three PH0 arms), the strategic-labelling
design that `HIERARCHY_VOCABULARY.md` **supersedes at §7** (*"PI resequencing order … the VLM/algorithmic
pipeline moves PRIOR to v6 training"*), the v2 target structure, and the fusion strategy whose
*"jurisdiction not averaging"* principle, **2-of-3 vocabulary voting emitting the REAL v6 tokens
(imported, cannot drift)**, and *"conflicts recorded, never merged"* rules are the reason the PH1 counts
(175 corroborations / 41 conflicts / 56 with the Alpamayo layer) are interpretable at all.

**`MORNING_REPORT_2026-08-11.md` / `OVERNIGHT_PLAN_2026-08-11.md` / `OVERNIGHT_RESULTS_2026-08-12.md` /
`…/2026-08-11-ops-bundle/README.md` / `stack/ops/POD_ACCESS_2026-08-04.md` /
`stack/scripts/T1_ADAPTER_NOTES.md` / `Project Steering/POD_HANDOVER_2026-08-13.md` /
`GATE_PREP_2026-08-04.md` / `FLEET_STATE_2026-08-04-1830Z.md`.** Operational records — the overnight
planning/results pair is the audit trail for the 08-11/08-12 verdict night; the ops bundle and pod
handover carry the HF silent-push-failure fix and the pod-access recipes; `T1_ADAPTER_NOTES.md` is the
T1 harness adapter whose staging is flagged in `V6_TRAINER_DESIGN.md` §7 as a **foreign staged entry**.

**`Project Steering/eval_corpus/{README.md, V2EP_FORMAT_SPEC.md}`.** The eval-corpus format spec — the
contract any future corpus build (including the 4,472-clip job) must satisfy.

**`TanitAD Research Hub/Evaluation/Videos/*` (6 READMEs + `INDEX.md`).** The video deliverables incl.
`v1arch-oodval-openloop-2026-08-05/READ_THIS_TOO-the-ranked-reels-are-a-speed-split.md` — a titled
warning that the ranked reels are a **speed split**, not a quality ranking. ⚠️ Per the standing note,
`*.mp4` is gitignored: the READMEs are in the repo, **the videos need `git add -f`** or they live only
on the producing disk.

**`Project Steering/BACKLOG.md`, `RETRACTION_LOG.md`, `CLAUDE.md`, `MODEL_REGISTRY.md`, `EVAL_DOCTRINE.md`.**
All updated by the campaign; `EVAL_DOCTRINE.md` is new-in-force (the T0/T1/T2 tiers), and `CLAUDE.md`
gained this campaign's traps (the cgroup page-cache trap, the supervisor-manifest trap, the flock race).

### 7.1 What I did NOT extract, and why

- **`PROJECT_STATE.md`'s A2 chronicle rows** — searched by content (`a2venv`, `alpamayo_vs_flagship`,
  `4,729`, `4729`, `23,644`, `quantis`) and by date-row scan. **They do not exist** (§2.6). Not a gap in
  the extraction: a gap in the campaign's chronicle. The A2 record lives in the two Hub folders instead.
- **Pod-side `a2_batch_out/` artifacts** (`alpamayo_gt`, the retime arms) — **the pods are gone**
  (verified in the consolidation report: connection refused/timeout on all four). Only what reached HF
  or the repo is quotable, which is what §2.5 measures.
- **The full `raw_json` payloads** of the 4,729 auto-labeling / VQA / grounding rows — I extracted their
  schema, counts, wall-times and the meta-action distribution, but did **not** attempt a semantic audit
  of 23,644 free-text generations. That is a real work item (a VQA-category-stratified QC pass) and is
  named as one, not silently skipped.
- **`ORBIS2_ANALYSIS.md` beyond §4/§5** and the deeper `V6_SIZING.md` config ladder — the consolidation
  report already carries the Orbis-2 headline; I extracted only the two rows the PI's topics touch
  (the 139× and the inverse allocation).
- **Numbers behind an egress wall** — every Orbis-2 figure carries the **PUBLISHED-via-snippet** stamp
  and I did not attempt to promote any of them.

---

## 8. WHAT THIS CHANGES FOR v6 — connecting each topic to the live S-W resume

The live line is: **v6 config E (336.5 M), S-W stopped cleanly at step 6,300 / last ckpt 6,250 of
30 k**, resuming on the Thor. Each topic lands on it as follows.

**From §1 (the vocabulary) — S-T is blocked on data, not on code.** The trainer imports the token
tables and enforces one-vocabulary-two-views by `is` identity, so the *mechanism* is ready. But
`PH0_COVERAGE_AUDIT.md` measures that **PH0 emits none of the nine `g_tac` tokens**, and **five of
them need agent slots that are not extracted**. ⇒ **The highest-value 0-GPU item in the programme is
wiring `obstacle.offline`** (87,481 cuboids / 10 dynamic classes on 97.44 % of the corpus; we read
4 of 36 features). It unblocks five tactical goals **and** the binding LONGITUDINAL family's
distance-keeping — the family that §6.3 just showed owns **~99 % of the T1 divergence**. Second
0-GPU item: **Engine-A situation labels** (free, exact, and information-disjoint from B4).
⚠️ **S-S must not be faked**: S2 `g_str` supervision comes from PH0→PH1→PH2 or the STRATEGIC family
stays `n/a` **with its reason and its n**.

**From §2 (A2 quantization + inference) — two things transfer to the live line.** (a) The **measured
throughput** (59.6 s/clip for the full 5-task battery, 78.4 wall-hours for 4,729 clips) is a real cost
basis: a second A2 pass over the 4,472-clip build is **~78 GPU-hours**, not the ~200 s/clip design
guess. (b) The **`NF4-backbone-4bit-UNVALIDATED` label on all 23,644 rows** means every downstream use
of this augmentation as a *teacher* inherits an unvalidated-quantisation caveat — it must travel into
any S-T distillation row. ⚠️ And the **contamination question is still unresolved**: A2 lists
PhysicalAI-AV as training data, so A2-derived labels on our corpus may be partly self-referential.
For *label* purposes that is much weaker than for *comparison* purposes, but it is not zero and must
be stated per arm.

**From §3 (augmentation) — do not expect it to move S-W.** The augmentation roughly doubles 13.3 h to
~26 h; the size-normalised deficit goes **139× → ~70×**. ⇒ **The S-W-relevant levers are the ones the
data doc ranks above it**: run past 30 k (we are at **0.75 epochs**, and repeated data holds value to
~4 ⇒ **~120 k steps** are available on the same corpus), enlarge the corpus **under a new declared
parity key** with the **same 40 val episodes**, and run the **frozen-pretrained-encoder arm (P1)** that
Orbis-2 upgraded to *"the best-supported unrun experiment in the programme."* The augmentation's real
target is **S-T/S-S**, where a few thousand well-chosen labelled clips plausibly suffice.

**From §4 (diffusion × MPC) — the v6 selector must not repeat W7's argmin.** The registry's own
consequence is binding: *"argmin-over-a-large-fan must be replaced by a noise-robust rule (top-m
aggregation / sharpened cost), **pre-registered before it is used**"*, and W7-PROG bound the follow-up:
*"the cost needs a **goal-conditioned** component."* v6 already has the goal — `g_tac`'s `ANCHOR_GOAL`
is exactly the *"distance to a goal, which inaction cannot minimise"* term the paper says we dropped.
⇒ **The S-T selector should be goal-conditioned roll-cost with a noise-robust aggregator, and its
`sel_gap ≤ 0.5× fan oracle at T1` gate should be met on that design, not on an argmin.** The L2/L3/L4
levels of the synthesis remain unrun and are the natural post-S-T ladder.

**From §5 (action space) — the emission contract is settled, and one small divergence is open.** The
unicycle `(a, κ)` emission at 60 steps / 6 s is imported into v6 and is feasible-by-construction; run 5
proved a latents-only head **passes the wm_reliance gate at 0.6233 while beating the shortcut head on
ADE**. ⚠️ **Open item: v6's `UnicycleEmission` uses `tanh` while our own measurement replaced `tanh`
with softsign for exactly this saturation reason.** Cheap to check before S-T trains the head.
⚠️ **Also carried:** the 6 s horizon costs **~43 % of the windows** (94 → 54 per 120-frame episode) —
a real distribution change vs v5f that belongs in the v6 run row, and one the PI was asked to accept or
rebuild the cache for.

**From §6 (the paper) — the S-W gate is now defined by a measurement, not by architecture.** The T1
block says: **`cl − ol` is the quantity that matters, `ha` is the floor to clear, and the divergence is
~99 % longitudinal.** ⇒ **S-W's own gate should be `cl − ol` stability under the model's own actions**,
exactly as the registry states, and the 10 k P-battery should be read with that framing.
⚠️ **Two paper-maintenance items fall out of this addendum:** the headline parameter count is still
**263.4 M** (one generation behind config E's 336.5 M), and the **Alpamayo counts are still flagged
INHERITED "pending a registry row"** — §2.5 supplies the measured values to close that flag.

**From §7 (the sweep) — three unrun items with outsized leverage.** (1) **`PREREG_SCALING_LADDER.md`**
is the only pre-registered test of the hierarchy's data-efficiency claim, costs ~6 GPU-hours on a frozen
trunk, and has both outcomes committed — it is the cheapest test of the programme's thesis and it is
still unrun. (2) **E-ENC in ViT-5 form** — the shipped width question is genuinely UNMEASURED. (3) The
**`DIR_YAW_RAD = 0.15 → 0.10` re-read**, which the A2 correction showed touches *"every published
manoeuvre-coherence κ in the programme"* — a 0-GPU re-analysis over banked windows that could change
the TACTICAL family's history.

---

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md` | repo working tree + index | **staged** (verified with `git ls-files --cached`) |
| `a2_parquet_facts.json` (schema, counts, quantisation value-counts, wall_s stats) | scratchpad `…/8fc25020-…/scratchpad/a2_parquet_facts.json` | ⚠️ **scratchpad only** — the *numbers* are banked in §2.5 of this report; the JSON itself is reproducible in ~2 minutes from the script below |
| `pull_a2_parquet.py` (the extraction script; reads the HF token in place from `Keys.txt`, never prints it) | scratchpad `…/scratchpad/pull_a2_parquet.py` | ⚠️ **scratchpad only** — say the word and it belongs in `stack/scripts/` as `a2_records_stats.py` |
| `records.parquet` (25,970,018 bytes) | HF `Sayood/tanitad-alpamayo2-augmentation` (far side) + scratchpad cache | far side is authoritative; nothing new was pushed |

**Nothing was committed and nothing was pushed** — the `AGENT_OPERATING_STANDARD` contract.

## Escalations

1. ⭐ **The A2 augmentation set has no registry row, and the paper carries its counts as INHERITED.**
   §2.5 measures them (23,644 rows / 4,729 clips / 5 tasks / **one** quantisation arm,
   `NF4-backbone-4bit-UNVALIDATED` / 78.4 wall-hours / zero errors). **This wants a `MODEL_REGISTRY.md`
   row so the paper can promote the flag** — that is a registry edit and belongs to whoever owns it,
   not to this report.
2. ⚠️ **`DataEng/DATA_STRATEGY.md` is a month stale** (v1.0, 2026-07-06) and is still cited by the
   handover as the augmentation strategy's owner. Either it gets a v2.0 that points at the four real
   documents, or the handover's pointer should be corrected. **A stale pointer in a handover is how the
   next session loses a stream.**
3. ⚠️ **The `tanh`-vs-softsign divergence** between v6's `UnicycleEmission` and our own measured
   saturation finding (§5.7) — a cheap check that should happen **before** S-T trains the emission head.
4. ⚠️ **The `DIR_YAW_RAD` 0.15 → 0.10 sensitivity re-read** is still owed; it touches every published
   manoeuvre-coherence κ, needs no GPU, and the A2 correction already logged it in `RETRACTION_LOG.md`.

