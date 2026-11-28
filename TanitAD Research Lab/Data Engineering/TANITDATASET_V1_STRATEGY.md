<title>TanitDataSet — Two-Dataset + VLM Augmentation Strategy</title>

# TanitDataSet — Two Datasets + VLM Augmentation Strategy

*Rev 4 · 2026-07-20 · **the VLM pilot actually ran**: §8.5-8.8 add verified gating, the deployed model + real serving recipe, measured throughput/adherence, and the two design changes the data forced (mint `VTARGET` + `LONMODE` kinematically). Rev 3 §8.1-8.4 kept for provenance. Rev 3 content (curation + filtering, record schema, v3 label mapping, phased plan) and Rev 2 content (two-dataset doctrine, source tables, VLM engine, prompts) preserved.*
Sources: [sensor survey](Research/2026-07-19-tanitdataset-v1-sensor-survey.md) · [semantic survey](Research/2026-07-19-tanitdataset-v1-semantic-survey.md) · [VLM survey](Research/2026-07-19-vlm-augmentation-survey.md) · vocabulary [V3_GOAL_VOCABULARY_V1](../Architecture%20%26%20Inference/V3_GOAL_VOCABULARY_V1.md) · code `stack/tanitad/lake/schema.py`, `stack/scripts/refb_labels.py`, `stack/tanitad/data/_contract.py`

---

## TL;DR

Build **two** camera-first datasets under **one schema and one pipeline**; generate the semantic/reasoning/goal labels ourselves with a **single VLM deployment**; make quality and balance a **mechanical, provenance-stamped filter**, not a vibe.

- 🟢 **TanitDataSet-C (Commercial)** — permissive sources only → shippable, HF-publishable, commercially deployable.
- 🔬 **TanitDataSet-R (Research)** — C **plus** all non-commercial sources → maximum diversity, **internal only, never redistributed**.
- **Superset relation** R = C ∪ NC: one build, one schema; a per-record `tier` stamp (already computed by `LakeRecord.commercial_ok`) decides what may ship.
- **VLM engine:** **Qwen3-VL-8B** (Apache-2.0) for bulk captions/tags + **Cosmos-Reason2-8B** (NVIDIA Open Model License, a Qwen3-VL post-train → same serving path) for Chain-of-Causation, risk, physics. Both commercial-clean. Frontier models (Gemini/GPT/Claude) are **research-set only**.
- **Every VLM label maps to the frozen [v3 goal vocabulary](../Architecture%20%26%20Inference/V3_GOAL_VOCABULARY_V1.md)** (VTARGET / VSOURCE / LONMODE / HEADWAY / lead-state / sign-reads / scene tags), **banded and provenance-stamped** `{kinematic|map|vlm|human|sim}` — §7.4.
- **Curation is the moat** (§7): stratified scene-balance so we don't drown in straight-highway free-flow, **up-sampling of the known weakness strata** (high-speed longitudinal, stop-and-go, cut-ins, merges, curves), a mined **safety-event split**, and a **frozen per-tier eval slice**.
- ✅ **VLM pilot EXECUTED 2026-07-20 (§8.5-8.8):** **48 clips labeled** into the frozen v3 tactical vocabulary on the eval-pod A40 — **100 % JSON parse, 99.1 % token adherence** (all word-token slots ~100 %), **~257 clips/GPU-hr** uncontended, **zero packages installed** on the shared pod. Deployed model = **`nvidia/Cosmos-Reason1-7B`** (ungated, 16.58 GB, stock `transformers`). **Cosmos3-Nano is ungated but unservable on the current fleet** — 16 B / 34.89 GB, docker-only official path and RunPod pods can't run docker → dedicated-hardware follow-up. Two design changes forced by the data: **mint `VTARGET` and `LONMODE` kinematically, not from the VLM.** Full write-up: [pilot note](Research/2026-07-20-cosmos3-vlm-pilot.md).
- **Infra asks (§9):** a dedicated ≥32 GB labeling GPU (or a scheduled eval-pod window); ~~one-click Cosmos-Reason2 license acceptance~~ **obsolete — the chosen model is ungated and needs no token**; ~15-60 TB for the C-core normalized cache + the L2D-subset scope decision.

---

## 1. The two datasets

| | 🟢 **TanitDataSet-C (Commercial)** | 🔬 **TanitDataSet-R (Research)** |
|---|---|---|
| **Sources** | permissive only (Apache/MIT/CC-BY/OpenMDW) + self-gen + CARLA | C **+** all NC (nuScenes/Waymo/BDD/A2D2/PhysicalAI-AV + their semantic layers) |
| **VLM labels** | Qwen3-VL + Cosmos-Reason2 (commercial-clean) | + frontier (Gemini/GPT/Claude) gold slices |
| **Use** | train + **ship** the world model; publishable to HF | train **research** models; internal eval/distillation |
| **Redistribution** | ✅ yes (after anonymization) | ❌ **never** — internal only |
| **Enforcement** | `tier:ship` records (`commercial_ok=True`) | `tier:nc` / firewalled records |

**The firewall is already mechanical in code.** `stack/tanitad/lake/schema.py` sets `license_class ∈ {owned-safe, nc-research, gated-confidential}` from a per-source CONSTANT (`SOURCE_REGISTRY`), never inferred; `commercial_ok = owned-safe AND not share_alike`; and `assemble_lake_record` **raises `PermissionError`** if a `gated-confidential` source (PhysicalAI-AV) is ever fed to the lake. Rev 3 formalizes the record-level `tier` on top of these (§7.1):

| `tier` | condition (from the existing schema fields) | shard placement |
|---|---|---|
| `ship` | `commercial_ok == True` (owned-safe, not share-alike) | commercial shards, HF-publishable |
| `ship-sa` | `owned-safe AND share_alike` (ZOD) | **segregated** copyleft shard, never co-mingled |
| `nc` | `license_class == nc-research` | internal-only shards |
| `firewalled` | `license_class == gated-confidential` (PhysicalAI-AV) | **never enters the lake** — recipe-only |

---

## 2. TanitDataSet-C — the shippable core (🟢 permissive)

| Source | License | Cameras | Ego / actions | Sensors | Scale | Role |
|---|---|---|---|---|---|---|
| **L2D** (yaak-ai) | Apache-2.0 | **6 surround** | **real CAN** + RTK | GPS/IMU | ~5,000 h | **anchor** — surround + real off-expert (expert/student split) |
| **comma2k19** | MIT | front | real CAN steer+speed | — | ~33 h *(loaded)* | highway action anchor |
| **PandaSet** | CC-BY-4.0 | 6 cam | pose | LiDAR | 8 h | clean SF-urban + LiDAR |
| **Cosmos-Drive-Dreams** | CC-BY-4.0 | front | poses→actions | — | synthetic *(loaded)* | weather/night/VRU synthetic |
| **WorldModel-Synthetic** | OpenMDW-1.1 | video | pose-less | — | 264 k clips | long-tail + free captions/tags |
| **Udacity** | MIT | front | steer/throttle | — | small | real-action augment |
| **CARLA** (self-gen) | ours | any | privileged expert | full | ∞ | off-expert + closed-loop + CoT GT |
| 🟡 **ZOD** | CC-BY-**SA** | multi | RTK | 3×LiDAR+radar | EU 14-country | urban diversity — **segregated `ship-sa` shard** |

## 3. TanitDataSet-R adds (🔴 NC — research only)

nuScenes (+ its whole semantic economy: DriveLM, NuScenes-QA, Talk2Car, OmniDrive, DriveLMM-o1), Waymo (+ WOMD-Reasoning), BDD100K (+ BDD-X human justifications), Argoverse 2 (NC-SA), KITTI-360, ONCE, A2D2 (CC-BY-**ND**), **PhysicalAI-AV/Alpamayo** (our own — commercial-OK for internal AV dev but no-derivatives → firewalled, `recipe-only`), DRAMA + Rank2Tell (human risk/importance gold). All ingest under the same schema, stamped `tier:nc`.

---

## 4. VLM augmentation engine — one deployment, two licenses

The elegant result from the survey: **the workhorse and the specialist are the same architecture.** Cosmos-Reason2-8B is Qwen3-VL-8B-Instruct post-trained → identical tokenizer/serving path, one container, hot-swap the weights.

| Role | Model | License / output rights | Use |
|---|---|---|---|
| **Bulk workhorse** | **Qwen3-VL-8B** (32B AWQ for the quality tier) | **Apache-2.0** — sell labels, train-and-ship, no field-of-use limits | dense captions, scene tags, VQA, OCR of signs |
| **Physical-AI specialist** | **Cosmos-Reason2-8B** (Qwen3-VL-8B post-train) | NVIDIA Open Model License — commercial OK, NVIDIA disclaims output ownership | Chain-of-Causation, risk, critical-agent, physics-plausibility |
| **Frontier teacher** (🔬 R-set only) | Gemini 2.5/3 Pro (video), GPT-5/o-series, Claude Opus | ToS forbids outputs→competing shipped models → **never touches C** | mint a human-verified **gold slice**; LLM-judge over open-weight labels |

**The output-license gate (protects the commercial set) — unchanged from rev 2:** frontier APIs never label shippable data; Llama-3.2-Vision is OUT (EU-excluded + naming tax); verify per-checkpoint backbones (Qwen-Research NC hides inside small InternVL/Ovis/Molmo variants); driving specialists (DriveVLM/Senna/OmniDrive/EMMA) are schema references not labelers; keep a human-verified gold slice as the only reasoning eval GT.

---

## 5. Professional system prompts (the labeling contract)

Schema-locked and anti-hallucination by design; run on Qwen3-VL (bulk) and Cosmos-Reason2 (reasoning). **Rev 3 change: every enumerated field is now constrained to a frozen v3-vocabulary token** (§7.4), and the model must emit `unknown` rather than a nearest-guess — honest gaps are stamped, not fabricated (vocabulary rule R3).

**A · Scene tagging** (Qwen3-VL, every clip):
```
You are an expert autonomous-driving scene annotator. You see a short sequence of
forward-camera frames. Output ONLY a JSON object matching the schema — no prose.
Annotate ONLY what is visible; when uncertain use "unknown" — never guess or infer
beyond the frames.
{ "weather":[clear|overcast|rain|snow|fog], "time_of_day":[dawn|day|dusk|night],
  "road_type":[highway|urban|suburban|rural|intersection|parking],
  "surface":[dry|wet|snow|unknown], "traffic_density":[empty|light|moderate|heavy],
  "ego_maneuver":[cruise|accelerate|brake|lane_change_L|lane_change_R|turn_L|turn_R|stop],
  "vru_present": true|false,
  "notable_events": ["≤5-word rare/safety-relevant items", ...] }  # [] if none
Base every field on visual evidence only; never fabricate.
```

**B · Chain-of-Causation reasoning trace** (Cosmos-Reason2, Alpamayo schema):
```
You are an expert driving-behavior reasoner. Given a forward-camera clip, the ego
speed, and the ego's actual next-2s action, produce a causally-linked trace that
EXPLAINS the decision from what is observable. Do NOT restate the action as its own
justification. Do NOT invent agents that are not visible. If the scene is
unremarkable, say so plainly. Output JSON:
{ "observation": "road, agents, signals actually visible",
  "critical_agents": [ {"agent":"...", "why_critical":"..."} ],   # ≤2 unless clearly more
  "justification": "the causal reason the action fits the critical agents",
  "decision": "high-level (maintain speed | yield | slow for lead | ...)",
  "action": "concrete control (hold lane | decelerate to X m/s | ...)",
  "physics_flag": "note any physically implausible element, else null" }
Every link must be grounded in the clip or the given ego state.
```

**C · Risk / critical-agent grounding** (Cosmos-Reason2 + Molmo for 2D points):
```
Identify agents that could causally affect the ego in the next 3 s. For each: type,
a 2D image point (or bbox), predicted interaction (cut-in | crossing | braking-lead |
merging | none), and risk [low|med|high] with a one-clause reason grounded in motion
cues. Return [] if none. Never list agents that cannot affect the ego path.
```

**D · Label verification / LLM-judge** (frontier, R-set gold only):
```
You are a strict reviewer. Given the clip and a candidate label (tags or CoC trace),
score faithfulness [0-1] and list any element NOT supported by the visual evidence.
Reject hallucinated agents, unsupported weather/time, and circular justifications.
Output {score, violations[], verdict:[accept|revise|reject]}.
```

**E · Sign & speed-limit read** (Qwen3-VL OCR, feeds VSOURCE/VTARGET — new in rev 3):
```
Read every speed-limit or regulatory sign clearly legible in these frames. For each:
{ "type":[speed_limit|stop|yield|no_entry|school_zone|other],
  "value_kph": <int or null>, "image_point":[x,y], "confidence":[0-1] }
Return [] if none legible. Report ONLY signs you can actually read; never infer a
limit from road type.
```

Governance: every generated field is stamped `{model, model_version, prompt_version, provenance:vlm, source_license}`. A **human-verified gold slice** (frontier-judged, then human-checked) calibrates auto-label quality and is the only reasoning GT used for eval.

---

## 6. Data-level augmentation (unchanged from rev 1/2, still core)

1. **Cross-source harmonization** — canonical rig (intrinsics/FOV), **per-clip `rig_id`/`crop_center`** (the two-rig lesson), unified action contract (→ 3-ch + v0 speed). *(The cosmos camera-projection bug is exactly this: heterogeneous rigs need real per-source calibration, not an assumed shared crop.)*
2. **Neural weather/lighting via Cosmos** — re-render owned clips into fog/night/rain/snow (we own Cosmos-DD); aligned to the OOD gap (comma 0.85 / cosmos 0.58 vs in-dist 0.43).
3. **Weakness-targeted long-tail** — oversample high-speed / sharp-turn / VRU / off-expert; L2D expert-vs-student + CARLA supply recovery data. *(Now driven mechanically by the curation weights, §7.3.)*
4. **CARLA counterfactual / off-expert** — perturbation + recovery + CoT GT — antidote to the open-loop→closed-loop collapse (1.69 m drift).
5. **VLM semantic self-generation** — §5, at scale on owned pixels.
6. **Temporal resampling** to `{0.5,1,1.5,2 s}@10 Hz`.
7. **Anonymization (mandatory)** — face/plate blur on real EU footage (L2D/ZOD/PandaSet) — GDPR — a hard gate before any `tier:ship` promotion (§7.2 stage 6).

---

## 7. Smart curation + filtering strategy  *(Deliverable 1 — the moat)*

The doctrine: **filtering removes what is unusable; curation decides what we see how often.** Both operate on one canonical record, and both write their verdicts back onto it as banded, provenance-stamped fields — so "why is this clip in/weighted" is always one column away, and a training run is a *query*, not a bespoke pull.

### 7.1 The record schema — extends the live `LakeRecord`

The core tensors (`frames [T,9,256,256] uint8`, `actions [T,2]`, `poses [T,4]`) and their contract stay **byte-for-byte** what every adapter emits today (`_contract.assemble_episode`). Everything below is **catalog metadata (Parquet) + a per-episode JSON sidecar** for the language/goal labels — nothing rewrites the shard blob. Fields already present in `schema.py` are marked ✓; rev-3 additions are marked ＋.

| Group | Field | Type / values | Provenance | Notes |
|---|---|---|---|---|
| **identity** ✓ | `episode_id`, `split_unit_id`, `sha256`, `source`, `attribution_id` | — | — | `split_unit_id` = the I3 route/clip unit; **never split its windows across train/val/test** |
| **license/tier** ✓＋ | `license_class`, `license_name`, `share_alike`, `commercial_ok` ✓ · `tier` ＋ | enum | source CONSTANT | `tier` derived (§1); never inferred, cannot drift |
| **geometry** ✓ | `camera_model`, `intrinsics_native`, `f_eff_px`, `image_size`, `hz` | — | source/calib | drives the rig/cy sanity gate (§7.2.4) |
| **motion** ✓ | `action_source` (can\|pose_derived\|idm\|vo\|none), `has_can` | — | kinematic | `has_can` gates honest SIGNAL labels |
| **rig** ＋ | `rig_id`, `crop_center_cy` | int / px | calib | per-clip; the two-rig fix (cy≈543 vs ≈755) |
| **quality** ＋ | `blur_band`, `exposure_band`, `truncation_frac`, `egomotion_sane`, `corrupt` | banded/bool | kinematic/cv | filter outputs (§7.2); banded so "hard-but-valid" is up-sampled not dropped |
| **dedup** ＋ | `phash`, `geo_cell`, `dedup_cluster_id`, `is_exemplar` | — | cv/gps | §7.2.5 |
| **scene tags** ＋ | `scene_tags{weather,time_of_day,road_type,surface,traffic_density,vru_present,notable_events[]}` | tokens | **vlm** | prompt A; also curation strata |
| **lead state** ＋ | `lead_state{present,gap_m,closing_speed_ms,ttc_s}` | banded | **vlm** (detector+monodepth / Cosmos-Reason2) | feeds LONMODE/HEADWAY/VSOURCE |
| **sign reads** ＋ | `sign_reads[{type,value_kph,image_point,conf}]` | list | **vlm** (OCR) | prompt E; feeds VSOURCE=sign_limit + VTARGET cap |
| **goal labels** ＋ | `goal{STRATEGIC⟨MISSION,ROUTE,LANEOBJ,SPEEDPOLICY,STYLE,RISK,ODD⟩, TACTICAL⟨VTARGET,VSOURCE,LONMODE,LATMANEUVER,HEADWAY,DYN,RULECTX,SIGNAL,INTERACT,TACPOINT,LIGHTSTATE⟩}` | v3 tokens | **per-slot** `{kinematic\|map\|vlm\|human\|sim}` + `unknown` | the frozen [v3 vocabulary](../Architecture%20%26%20Inference/V3_GOAL_VOCABULARY_V1.md); mapping §7.4 |
| **reasoning** ✓＋ | `language{caption, coc_trace{observation,critical_agents[],justification,decision,action,physics_flag}, qa[]}` ✓ · `label_stamp{model,model_version,prompt_version,provenance,source_license}` ＋ | struct | **vlm/human** | rides the existing `language`/`language_source` fields + `modality_flags.has_language/has_qa/has_maneuver` |
| **curation** ＋ | `curation{strata[], weight, safety_event, is_eval_holdout}` | — | derived | §7.3; `weight` = the training sampler probability multiplier |

**Provenance is per-slot, not per-record** (vocabulary R3). `VTARGET` is `kinematic+vlm` (85th-pct future free-flow speed, *capped* by a VLM sign-read); `SIGNAL` is `human/can` where CAN carries the blinker (L2D) and `unknown` everywhere else — the label never pretends to know what the sensor can't see. `SOURCE_REGISTRY` must be **extended** for rev-3 ingest: L2D (Apache), PandaSet (CC-BY), Udacity (MIT), WorldModel-Synth (OpenMDW), ZOD (CC-BY-SA, `share_alike=True`), plus the R/NC sources as `nc-research`.

### 7.2 Filtering pipeline — ordered stages, cheap→expensive, verdicts written back

1. **License stamp (structural, first).** `SOURCE_REGISTRY` sets the license axis + `tier` at ingest. C is not a separate build — it is the **query** `tier == 'ship'`. The `PermissionError` guard makes a firewalled source physically unable to enter the lake.
2. **Corrupt-clip skip.** Extend the existing parity skipset (skip-hash `f09e44db`, 24 corrupt PhysicalAI clips) into a per-source `skipset` keyed by clip hash: decode failure, NaN/Inf poses, zero-length, frame-count mismatch, all-black/all-frozen frames. Skipsets are committed artifacts so a rebuild is deterministic (the strict-parity discipline, key `e438721ae894`).
3. **Quality gates (banded, not binary).**
   - **Blur** — variance-of-Laplacian per frame → `blur_band`; drop only if >X % of frames are below the hard floor (motion blur at speed is *signal*, keep it).
   - **Exposure** — over/under-exposed pixel fraction → `exposure_band`; night-blown/sun-washed clips are downweighted, not dropped, unless outside any night/glare stratum.
   - **Truncation/occlusion** — ego-hood / wiper / rain-on-lens fraction → `truncation_frac`; downweight.
   - All banded so curation can deliberately **up-sample "hard but valid"** (rain-on-lens, dusk glare) rather than blanket-drop the exact OOD we're weak on.
4. **Ego-motion sanity (the rig/cy lesson — the discipline that burned us).**
   - **Multi-rig detection:** cluster each source's per-clip principal-point `cy`; PhysicalAI front-wide has **two rigs** (cy≈543 rig-A / cy≈755 rig-B) and a geometric-center crop is ~215 px wrong for rig-B → cross-rig vertical inconsistency in training frames. Set `rig_id` + `crop_center_cy` **per clip**; crop around the per-clip cy, or filter to one rig. Never assume a shared crop.
   - **Calibration gate:** if `intrinsics_native` is missing AND `camera_model ≠ pinhole` → route to the D-016 R1 pad-crop+undistort repair, else skip. (The cosmos f-theta projection bug is this gate failing open.)
   - **Kinematic plausibility:** reject clips with `|accel| > 8 m/s²`, implausible jerk, negative speed, or pose teleports. Verify against the metric *definition* and multiple samples before flagging (the "step_s is accumulated over log-every, not per-step" false-alarm lesson) → `egomotion_sane`.
5. **Dedup (two independent passes).**
   - **Perceptual-hash near-dup (within-source, cheap first):** pHash/dHash per keyframe → LSH buckets → collapse near-identical clips (re-uploads, overlapping comma segments); keep one `is_exemplar`, tag the rest with `dedup_cluster_id`.
   - **GPS/time overlap (cross-source):** for GPS-bearing sources (L2D RTK, comma GNSS, ZOD RTK, PandaSet GPS) build a `geo_cell`×timestamp index; collapse true content-dups (same drive, same source re-hosted). **Same-road / different-time traversals are KEPT and tagged** — multi-traversal (Open-MARS-style) is a *wanted* world-model consistency signal, not a dup. This pass catches cross-source re-hosts (comma2k19 GitHub vs HF) that pHash within-source misses.
6. **Anonymization gate (hard, mandatory).** Real EU footage (L2D/ZOD/PandaSet) triggers GDPR regardless of license. `tier` cannot become `ship`/`ship-sa` until `anonymized == True` (face/plate blur; ZOD ships de-identified — its notice must travel). This is a **blocking gate**, not a downweight.

### 7.3 Curation — the smart part

**Goal:** the corpus is violently imbalanced — comma2k19 is ~74 % straight highway; `free_cruise + lane_keep + high-VTARGET` free-flow dominates every real source. Un-curated, the world model over-fits the easy majority and stays weak exactly where the memory says v1 is weak (high-speed longitudinal, closed-loop divergence). Curation fixes the *sampling distribution*, using the VLM tags + kinematic labels as the strata.

1. **Stratified scene-balance.** Define strata over the crossproduct of two axes:
   - *scene* = `road_type × weather × time_of_day × traffic_density` (from VLM scene_tags),
   - *behavior* = `LONMODE × VTARGET-band × LATMANEUVER` (from kinematics + lead_state).
   Compute the empirical histogram; set `curation.weight = clamp(target_density / empirical_density)`. This is exactly the **inverse-frequency, clamped** up-weighting `refb_train` already uses for the route-heading aux CE — generalized to the full stratum grid.
2. **Up-sample the known weakness strata** (boosted target densities):
   - **high-speed longitudinal** — `VTARGET ≥ v(30-40]`, `VSOURCE ∈ {sign_limit, road_class_default}` (the flagship longitudinal lever: 83 % of 2 s error is along-track at high speed).
   - **stop-and-go** — `LONMODE ∈ {stop_at_point, hold_stop, launch, creep}`, the low-speed bands where VTARGET's 1 m/s resolution lives.
   - **cut-ins** — `INTERACT = yield_to_lead` + `lead_state` closing + a lateral incursion (Cosmos-Reason2 critical-agent).
   - **merges** — `ROUTE = merge`, `LATMANEUVER ∈ {merge_in, yield_merge}`.
   - **curves** — `peak_kappa` in the turn band with `LATMANEUVER = lane_keep` (the v2 curvature-relative labels separate a road-following sweep from a junction turn; AMBIGUOUS gray-zone stays masked, `valid=False`).
3. **Safety-event mining → a curated split.** Set `curation.safety_event`:
   - `hard_brake` — strong `DV_BRAKE` + `DYN ∈ {firm,max}` + jerk spike (**kinematic**, cheap, high-precision).
   - `near_miss` — `lead_state.ttc_s < 1.5 s` + evasive lateral (`nudge_*` / `abort_lc`) (**kinematic+vlm**).
   - `anomaly` — `coc_trace.physics_flag ≠ null` OR `notable_events` flags a rare item (VRU incursion, wrong-way, debris) (**vlm**).
   This is the license-clean, owned-pixel analog of DRAMA/Rank2Tell — over-represented for the safety eval and the closed-loop stress tests.
4. **Frozen per-tier eval slice.** Freeze `is_eval_holdout` **per tier** (C-eval, R-eval), split at `split_unit_id` granularity (never leak a route's windows), stratified to cover every stratum incl. the weakness + safety strata, and **hash-pinned** so eval never drifts into train and stays comparable across runs (mirror the strict-parity key discipline). The **gold reasoning slice** (frontier-judged → human-verified, prompt D) is a subset of R-eval and is the *only* reasoning GT used for scoring.

### 7.4 The v3-vocabulary mapping — VLM output → frozen tokens, provenance-stamped

The augmentation is worthless if it doesn't feed the planner's option space. Each VLM/kinematic signal maps to a **banded** [v3 slot token](../Architecture%20%26%20Inference/V3_GOAL_VOCABULARY_V1.md) (never a continuous leak — rule R1), stamped with its provenance:

| Signal (source) | → v3 slot(s) | Provenance | Rule |
|---|---|---|---|
| future free-flow speed, 85th-pct 10-20 s, capped by sign/map | `VTARGET` band (23, non-uniform: 1 m/s <10, 2.5 m/s 10-40) | **kinematic + vlm** (cap) | the concrete tactical set-speed |
| `sign_reads.value_kph` (OCR, prompt E) | `VSOURCE = sign_limit` + caps `VTARGET` | **vlm** | teaches sign→speed |
| `lead_state{present,gap,closing}` | `LONMODE ∈ {free_cruise\|follow_lead\|close_gap\|open_gap}` + `HEADWAY` band (5) + `VSOURCE = lead_constrained` | **vlm** (detector+monodepth / Cosmos-Reason2) | longitudinal reasoning |
| `peak_kappa` band + sign-of-curvature | `VSOURCE = curve_constrained`, curve stratum | **kinematic** | curve up-sample |
| curvature-relative maneuver (`refb_labels` v2) | `LATMANEUVER`, `ROUTE`; AMBIGUOUS→`unknown` | **kinematic** | R3 honest gaps |
| `scene_tags.weather/visibility` | `RISK ∈ {nominal\|elevated_weather\|elevated_visibility\|elevated_anomaly}`, `ODD` monitors | **vlm + engineered** | mostly inference-time input |
| `critical_agents` (CoC prompt B/C) | `INTERACT ∈ {yield_to_k\|assert_gap_k\|cooperate_merge_k}` (k∈{lead,merger}) | **vlm** | cut-in/merge mining |
| traffic lights (prompt C / detector) | `LIGHTSTATE`, `TACPOINT = stop_line` | **vlm** | signal reasoning |
| blinker (CAN, L2D only) | `SIGNAL`; else `unknown` | **human/can** | imitation-limited, honestly stamped |
| `coc_trace.decision/action` | cross-check vs kinematic `LONMODE/LATMANEUVER`; disagreement → flag hard/ambiguous | **vlm** (verify) | anti-hallucination |

`STRATEGIC` slots (MISSION/ROUTE/LANEOBJ/SPEEDPOLICY/STYLE/ODD) are largely **inference-time inputs / engineered monitors**, trained only where the label is honest (weather→RISK). The mapping is faithful to the vocabulary's own "Label minting" section — no slot is invented, and every gap is `unknown`, not guessed.

---

## 8. VLM deployment recipe & pilot status  *(Deliverable 2)*

> **✅ REV 4 — PILOT EXECUTED 2026-07-20.** 48 clips labeled end-to-end into the frozen v3 tactical
> vocabulary on the eval-pod A40. **100 % JSON parse, 99.1 % token adherence** on all word-token
> slots. Full write-up: [2026-07-20-cosmos3-vlm-pilot](Research/2026-07-20-cosmos3-vlm-pilot.md).
> §8.1–8.4 below are the superseded rev-3 recipe-only plan, kept for provenance; **§8.5–8.8 are the
> verified state.**

### 8.1 Live fleet check (2026-07-19, read-only)

| Pod | GPU | Util / mem | State | Verdict |
|---|---|---|---|---|
| pod2 (flagship) | A40 48 GB | **100 % / 15.3 GB** | training | do not touch |
| pod3 (REF-A) | A40 48 GB | **100 % / 28.8 GB** | training | do not touch |
| tanitad-eval | A40 48 GB | 0 % / 0 GB (GPU idle); **RAM 429 GB free; load-avg 21-27** | mid evals+P2 cycle; **no comma2k19/PandaSet staged** | GPU free but box contended + no data |
| pod1 (tanitad-pod) | RTX A6000 48 GB | 0 % / 2 MB (GPU idle) | **comma2k19 `_epcache` staged**, but a **live babysitter** waits on `axis6-relaxed` to fire a GPU diagnostic + grounded-rollout | data present but pending GPU job |
| local dev box | RTX 4060 **8 GB** | — | — | **impossible** (Cosmos-Reason2 needs ≥32 GB) |

**Decision: recipe-only, no contention.** Two GPUs read as idle, but neither is a *clean* target: the eval-pod A40 is on the box the brief flagged as "running evals+P2" (load-avg 21-27, still cycling) with no driving data staged, and pod1's A6000 sits behind a babysitter poised to launch GPU evals the moment `axis6-relaxed` completes, with comma2k19 staged. Standing the persistent 16-18 GB vLLM server on either would contend with the pod's actual job — the exact "do NOT contend" the brief forbids — and Cosmos-Reason2 carries a gated-access + vLLM-version precondition that could stall on a live eval box. The recipe below is **one command away** the moment a GPU frees (eval-pod when P2 clears, or a dedicated labeling pod).

### 8.2 Model facts (verified against the HF card, 2026-07-19)

- **`nvidia/Cosmos-Reason2-8B`** — **gated**: someone must accept the NVIDIA Open Model License + share contact info **once** on the HF account before any pull. Base = **Qwen3-VL-8B-Instruct** (8.77 B). **BF16-only, 32 GB-min GPU** (tested H100/A100 → A40/A6000 qualify; the 4060 cannot). 256 K context; inputs = text + video(mp4)/image(jpg); card recommends `fps=4`, `max_tokens=4096`.
- **`Qwen/Qwen3-VL-8B-Instruct`** — Apache-2.0, **ungated**, same architecture → same serving image.

### 8.3 The deployment recipe (one command away)

```bash
# 0) one-time: accept the NVIDIA Open Model License for nvidia/Cosmos-Reason2-8B on HF
#    (the Sayood token has read access once accepted). Qwen3-VL needs no gate.
export HF_TOKEN=<from Keys.txt>

# 1) serve (vLLM >= 0.11.0 for Qwen3-VL arch; one image serves both by weight swap)
#    A40/A6000, BF16. Cap frames via --limit-mm-per-prompt; verify it is honored on
#    the pinned vLLM build (known Qwen3-VL caveat) else fall back to the transformers
#    path (Qwen3VLForConditionalGeneration, fps=4, max_new_tokens=4096, per the card).
vllm serve nvidia/Cosmos-Reason2-8B \
  --dtype bfloat16 --max-model-len 32768 \
  --limit-mm-per-prompt image=16 \
  --served-model-name cosmos-reason2 --port 8000
#   bulk workhorse (hot-swap): vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 ...

# 2) augment: sample 16 keyframes/clip @ ~0.5-1 MP from the staged epcache, POST the
#    OpenAI-compatible /v1/chat/completions with prompts A+E (Qwen3-VL) then B+C
#    (Cosmos-Reason2), parse JSON, map_to_vocab() (§7.4), write the per-episode sidecar.
```

Harness sketch (CPU-testable end-to-end against the served endpoint):
```python
for ep in staged_epcache_subset(n=50):                 # pod1: /workspace/data/comma2k19/_epcache
    frames = sample_keyframes(ep, k=16, max_mp=1.0)    # downscale+subsample = the throughput lever
    tags   = vlm(PROMPT_A, frames, model="qwen3-vl")   # scene_tags
    signs  = vlm(PROMPT_E, frames, model="qwen3-vl")   # sign_reads -> VSOURCE/VTARGET cap
    coc    = vlm(PROMPT_B, frames, ego=ep.speed,       # Chain-of-Causation
                 next2s=ep.action, model="cosmos-reason2")
    risk   = vlm(PROMPT_C, frames, model="cosmos-reason2")   # critical agents -> INTERACT
    goal   = map_to_vocab(tags, signs, coc, risk,      # -> frozen v3 tokens, provenance=vlm
                          kinematic=refb_labels_v2(ep.poses))
    write_sidecar(ep.id, dict(scene_tags=tags, sign_reads=signs,
                              coc_trace=coc, lead_state=risk, goal=goal,
                              label_stamp=stamp("cosmos-reason2","2026-07", "vA/B/C/E")))
```

### 8.4 Throughput estimate (A40, 8B BF16, vLLM continuous batching)

At 16 keyframes/clip downscaled to ~0.5-1 MP (~8 k vision tokens prefill), terse JSON out, batch 16-32:
- **Scene-tags + sign-reads (Qwen3-VL, ~200-400 tok out):** ≈ **1.5-3 k clips / GPU-hr**.
- **Chain-of-Causation (Cosmos-Reason2, ~0.8-1.5 k reasoning tok out):** ≈ **0.6-1.2 k clips / GPU-hr**.
- **Key optimization:** do NOT CoC every clip — tags on *all*, CoC only on the curated + safety strata (§7.3), which is ~10-20 % of the corpus. This cuts the expensive pass ~5-10×.
- **50-clip pilot:** minutes (well under one GPU-hour). Deliverable = 50 sidecars with scene_tags + sign_reads + lead_state + CoC + the mapped v3 `goal` tuple, provenance-stamped — a live proof of the §7.1 schema and §7.4 mapping.

---

### 8.5 ✅ Verified gating (byte-pull checked 2026-07-20)

| Model | Gated? | License | Size | Serving |
|---|---|---|---|---|
| **`nvidia/Cosmos3-Nano`** | **NO** — ungated | OpenMDW-1.1 (commercial + NC OK) | **16 B / 34.89 GB** BF16 | `cosmos3_omni` — **docker / vllm-omni / sglang-diffusion only** |
| `nvidia/Cosmos3-Super` | **NO** — ungated | OpenMDW-1.1 | 64 B / 30 shards | same |
| **`nvidia/Cosmos-Reason1-7B`** | **NO** — ungated | NVIDIA Open Model License | 7 B / **16.58 GB** BF16 | **stock `transformers`** (`qwen2_5_vl`) |
| `nvidia/Cosmos-Reason2-8B` | **yes** (auto-accept) | NVIDIA Open Model License | 8.77 B | vLLM ≥0.11 |
| `Qwen/Qwen3-VL-8B-Instruct` | NO | Apache-2.0 | 8 B | vLLM / transformers |

**Rev-3's "accept the Cosmos-Reason2 license" infra ask is obsolete** for the pilot path: the chosen
model needs **no token and no license click-through**. (Reason2-8B remains gated; Cosmos3 is not.)

### 8.6 Chosen model: **`nvidia/Cosmos-Reason1-7B`** — and why not Cosmos3

Sayed's 2026-07-20 call was Cosmos3. On evidence it **cannot be served on the current fleet**, so the
pilot ran the sanctioned fallback and Cosmos3 moves to a dedicated-hardware follow-up:

> **⚠️ RE-CHECKED 2026-07-20 (bulk pass, timeboxed 30 min) — the docker blocker below is
> OBSOLETE, but the verdict is unchanged.** `transformers` **5.14.1** (already on tanitad-eval)
> registers the architecture natively: `cosmos3_omni → Cosmos3OmniForConditionalGeneration`, and on
> the pod **`AutoConfig.from_pretrained('nvidia/Cosmos3-Nano')` → `Cosmos3OmniConfig`** and
> **`AutoProcessor` → `Qwen3VLProcessor`** both load with **zero installs** — no docker, no
> `vllm-omni`, no `sglang`. The real blocker is the **weight layout**: the root
> `model.safetensors.index.json` maps the *language* tensors (`layers`, `embed_tokens`, `lm_head`,
> `norm` — 792 layer entries) **into the same `transformer/diffusion_pytorch_model-*` shards as the
> DiT**. Understanding and generation share ONE 16 B backbone; there is no smaller
> understanding-only subset to load. Minimum resident = **31.50 GB** (`total_size` in that index),
> which on a 46 GB A40 shared with live TanitEval jobs leaves ~14 GB. **Verdict stands: Cosmos3-Nano
> needs a dedicated card — for a memory reason, not a serving-stack reason.** On that card it should
> be a one-liner (`AutoModelForImageTextToText.from_pretrained('nvidia/Cosmos3-Nano',
> dtype=bfloat16)`); the only untested step left is whether `transformers` resolves this
> diffusers-layout checkpoint through that index.

- ~~**Official Cosmos3 serving path is a docker container** (`docker pull vllm/vllm-omni:cosmos3`) —
  **RunPod pods cannot run docker; they are containers.** Hard blocker.~~ **Superseded — see the
  re-check above: stock `transformers` serves the arch with zero installs.**
- The pip alternative `vllm-omni` 0.24.0 carries **81 dependencies** incl. an exact
  `diffusers==0.38.0` pin → must be venv-isolated; installing it on a pod that also runs
  TanitEval/training risks re-pinning torch under a live job.
- **34.89 GB of weights on a 46 GB A40 leaves ~11 GB** for vision/KV/VAE — the model card itself
  prescribes `--enable-layerwise-offload` / `--tensor-parallel-size` for this case. Single-A40 is the
  marginal path, not the happy path.
- Cosmos3-Nano's checkpoint is dominated by `transformer/diffusion_pytorch_model-*` (≈33.9 GB) — it
  is a **generation** omnimodel. Its real value to TanitAD is **neural weather/lighting re-rendering
  (§6.2) + action-conditioned world simulation**, not bulk labeling.
- ⚠️ **Trap:** PyPI **`cosmos` (0.6 MB, 0 deps) is NOT NVIDIA's framework.** The real one is
  `pip install -e` from `github.com/NVIDIA/cosmos-framework`.

### 8.7 The verified serving recipe (zero environment mutation)

```bash
# tanitad-eval already had torch 2.8.0+cu128 + transformers 5.14.1 -> NOTHING was installed.
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Reason1-7B')"
#   16 GB in ~10 s, no HF token required (ungated).

# load (NB: device_map= needs `accelerate`, absent -> load on CPU then move):
#   AutoModelForImageTextToText.from_pretrained(id, dtype=torch.bfloat16).to("cuda")
#   -> 16.64 GB resident on the A40, 10 s load.

# label: 8 keyframes/clip @ 896x512, two schema-locked passes per clip
#   PASS A perception  -> scene_tags + sign_reads + lead_state
#   PASS B tactical+CoC-> v3 TACTICAL slots + chain-of-causation
```
Scripts on `tanitad-eval:/root`: `prep_clips.py` → `label_clips.py --tag reason1` (resumable,
checkpoints per clip) → `analyze_pilot.py`. Sidecars in `/root/vlm_pilot/out/reason1/`.
**vLLM is NOT required** — and avoiding it is what kept the shared eval pod safe.

### 8.8 Measured results — and the two design changes they force

**Throughput:** 48 clips / 96 VLM calls in **23.5 min**. ~30 s/clip while sharing the card with a
TanitEval probe (**123 clips/GPU-hr**); **~14 s/clip once uncontended → ~257 clips/GPU-hr**, ~10 953
tok in / 336 out per clip. This is *unbatched transformers*; vLLM continuous batching should add
several × on top, so §8.4's 0.6–1.2 k clips/GPU-hr remains the right planning number for a batched
production pass.

**Adherence:** JSON parse **48/48 (100 %)** both passes. VSOURCE / LONMODE / HEADWAY / DYN / TACPOINT
/ LIGHTSTATE / INTERACT / RISK = **100 % in-vocab**; LATMANEUVER 92 %; **overall 94 %, or 99.1 %
excluding VTARGET**.

> **Design change 1 — never ask a VLM for `VTARGET`.** It fabricates band edges (`v(7.3-8.2)`,
> `v(11.7-13.5]`) rather than copying from the 23-token grid → **48 % violation**, while every
> *word*-token slot is ~100 %. 92 % of the fabrications snap back to the grid, but the snapped value
> agrees with the kinematic 85th-pct band only **45 %** of the time. §7.4 already specifies VTARGET
> as `kinematic + vlm(cap)` — **mint it from the pose track** (exact, free) and use the VLM only for
> the sign cap and for `VSOURCE` (the *why*). Enumerated numeric ranges are the wrong ask for a 7 B VLM.

> **Design change 2 — mint `LONMODE` kinematically too.** The 48 clips are **24 drives × 2 weather
> renders with identical ego motion**, giving a free label-stability probe: LATMANEUVER/LIGHTSTATE
> **100 %** stable, VTARGET/DYN 92 %, HEADWAY 79 %, TACPOINT 75 %, VSOURCE 67 %, **LONMODE 62 %**
> (RISK 33 %, correctly tracking weather). The VLM's **longitudinal** reasoning is not
> appearance-invariant — the same axis the flagship is weakest on. **Adopt render-pair stability as a
> standing QA metric** (we own Cosmos-DD; re-rendering costs nothing and needs no human GT) and gate
> any slot below ~80 % stability out of training labels until kinematically minted.

**Quality vs ground truth** (render condition carried in the filename): weather **75 %** (15/20),
time_of_day **75 %** (21/28).

**Mandatory consistency gates** — cheap, and each caught a real defect in 48 clips:

| Gate | Pilot violation rate |
|---|---|
| reject `value_kph` when `type != speed_limit` (a *directional* sign was read as a 65 kph limit) | — |
| `VSOURCE=sign_limit` ⇒ a sign read must exist | **4/48 (8 %)** |
| `HEADWAY=unknown` ⇒ when no lead present | **6/48 (12.5 %)** |
| VLM `LONMODE` vs kinematic mint — disagreement ⇒ flag hard/ambiguous | implement per §7.4 |
| `LONMODE=follow_lead` ⇒ lead present | 0/48 ✅ (already consistent) |

Also: `lead_state.present` fired on **42/48 (88 %)** — likely over-detection, tighten the prompt.
And **`coc_trace` returned the structured object in only 3/48 (6 %)** — the rest collapsed to (good)
prose, which blocks `critical_agents` → `INTERACT` mining and the safety-event `physics_flag`. Fix
with a separate CoC call, a flattened schema, or constrained decoding **before** the CoC pass scales.

**Data gotchas found (cost hours if rediscovered):** weather suffixes can contain `_`
(`Golden_hour`) so parse filenames structurally, not by `split("_")`; and `vehicle_pose/` + `pose/` +
`pinhole_intrinsic/` ship alongside `generation/` — real ego motion is available and should be used
rather than asking the VLM to guess speed.

> **❌ CORRECTION 2026-07-20 (bulk pass) — every speed in §8.8 above is a factor 2 TOO LOW.**
> The pilot derived the clip span from the filename timestamps (`t1-t0 = 2e7`, read as µs → 20 s)
> and got 300 poses / 20 s = **15 Hz**. The dataset card settles it the other way, twice over:
> (a) it states the corpus is **"5,843 10-second clips"**, and (b) its sensor table documents
> `lidar_raw` at **10 FPS** with keys `000000`, `000003`, `000006`, … — an index **stride of 3**,
> so the index rate is **30 Hz**. 300 indices ÷ 30 Hz = 10 s, self-consistent with (a); `pose/`
> (2 100 = 300 × 7 cameras) and `all_object_info/` (300) carry the same 300-index grid.
> **The true rate is 30 Hz over a 10 s clip.** Consequences: the pilot's "0–17.4 m/s (0–62 km/h),
> plausible urban/suburban" is really **0–34.8 m/s (0–125 km/h)** — a highway-heavy corpus — and
> the kinematic VTARGET the pilot declared authoritative was **one to several bands low on every
> clip**. Re-derived at 30 Hz the staged corpus is v_mean **0–37.3 m/s, median 17.9 m/s (64 km/h)**.
> *Lesson: derive the sample rate from a documented sensor, never from a filename timestamp whose
> unit is a guess.* Also unresolved and flagged in every bulk record: the render is **121 frames**
> against a 300-index clip, so which 121 indices it covers is not documented — pixel-motion
> correlation was inconclusive (generated-video texture churn), so kinematics are minted from the
> full 10 s track.

---

## 9. Phased build plan + infra asks  *(Deliverable 3)*

**Storage model.** The cost driver is the **normalized `[T,9,256,256]` uint8 cache**, not raw footage. At the D-015 3-frame RGB stack, a normalized episode is ≈ **0.15-0.4 GB uint8** (memory anchor: 584 float32 eps ≈ 400 GB → ~0.17 GB/ep uint8). Labels/sidecars are negligible (~KB/clip). Raw is *transient staging*, deleted after normalization.

| Phase | Sources | VLM pass | Storage (normalized cache) | New infra |
|---|---|---|---|---|
| **0 — schema + pilot** *(now)* | — | 50-clip proof (recipe §8, first free uncontended GPU) | negligible | none (recipe ready) |
| **1 — C core (ship)** | comma2k19 ✓ + PandaSet + Cosmos-DD ✓ + Udacity + WorldModel-Synth (tags only, pose-less) + **stratified L2D subset** | Qwen3-VL tags on all; Cosmos-Reason2 CoC on curated+safety strata; anonymize EU-real; freeze C-eval; HF push | **~15-60 TB** (≈50-150 k normalized eps) + transient L2D raw stage | **labeling GPU + storage** |
| **2 — R superset (internal)** | + nuScenes economy, Waymo/WOMD, BDD/BDD-X, A2D2, Argoverse2, Honda DRAMA/Rank2Tell; **PhysicalAI-AV recipe-only** | + frontier gold slice (Gemini/GPT/Claude) → human-verified R-eval reasoning GT | **~10-30 TB** internal | frontier API budget |
| **continuous** | CARLA off-expert CoT (license-clean reasoning GT); ZOD `ship-sa` shard (segregated) | privileged-expert CoT | grows with CARLA gen | — |

**VLM-hours for the v1 C-core** (≈1 M candidate clips, ~15 % to CoC): ~350-700 GPU-hr tags + ~150-250 GPU-hr CoC ≈ **~500-900 A40-GPU-hours** (banded) — ~1 week on ~4-6 cards, or a rolling background job on one dedicated card.

### Infra asks (explicit, to Sayed)

1. **A dedicated ≥32 GB labeling GPU** (A40/A6000-class), **or** a scheduled eval-pod window when evals+P2 are idle. A GPU is technically free *now* (eval-pod A40) but the box is mid-cycle — a dedicated card avoids contending with the running evals. Budget ~500-900 A40-GPU-hours for the v1 C-core pass.
2. ~~**Accept the Cosmos-Reason2 gated license**~~ — **OBSOLETE (2026-07-20).** The deployed labeler `nvidia/Cosmos-Reason1-7B` is **ungated** (as are Cosmos3-Nano/Super and Qwen3-VL) and pulls with no token at all. Nothing to click. *New ask in its place:* **a pod with docker (or a dedicated GPU + isolated venv) if we want Cosmos3-Nano** — see §8.6.
3. **Storage** — ~15-60 TB for the C-core normalized cache + a transient raw-staging area for the L2D-subset download (watch the MooseFS ~466 GB per-pod quota that `df` hides). ~10-30 TB more for the R internal cache.
4. **Confirm the anonymization pipeline** as a hard `ship`-gate before any EU-real ingest.
5. **The five open decisions below** still need sign-off before TBs move.

### Open decisions for sign-off (carried from rev 2, still binding)

1. **L2D scope** — ~90 TB raw / phased rollout → which stratified subset (cities × conditions × expert/student, target hours)? *This sets Phase-1 storage and VLM-hours.*
2. **ZOD ShareAlike** — include the `ship-sa` copyleft shard (segregated) or skip to keep C fully Apache/MIT/CC-BY?
3. **VLM hardware** — dedicated labeling pod vs. scheduled eval-pod windows (throughput vs. the running evals). *See ask #1.*
4. **Anonymization** — confirm as the hard `ship`-gate.
5. **v1.0 target size / storage budget** — sets the subset sampling densities (§7.3).
6. **nuScenes/nuPlan paid commercial path** — pursue (moves the nuScenes economy from R into C legally) or stay firewalled?

---
*Watch-items: ~~Cosmos 3~~ **RESOLVED 2026-07-20 (§8.6): confirmed heavier unified platform, NOT a drop-in labeler — 16 B/34.89 GB generation omnimodel, docker-only serving; keep it for neural re-rendering (§6.2), not labeling**; ROVR Open (ego-motion undocumented); Open MARS (license TBD — its multi-traversal signal is wanted, so verify). Verify at build: per-checkpoint backbone licenses for any InternVL/Ovis/Molmo; the vLLM `--limit-mm-per-prompt` honoring on the pinned Qwen3-VL build; Cosmos-Reason2 A40 throughput at our 16-frame clip length.*
