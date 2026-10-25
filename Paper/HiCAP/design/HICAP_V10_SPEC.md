# HiCAP v10 extensions — design specification (rev b, 2026-10-25)

Single source for the four extensions the PI requested on 2026-10-25 and for the poster / motion-graphics
explainer. The paper (`Paper/HiCAP/hicap.tex`) is the authority for everything that existed before; this
file adds and does not repeat. Evidence classes as in `CLAUDE.md`: MEASURED (ours + artifact),
PUBLISHED (cited), INHERITED, ESTIMATED, HYPOTHESIS.

PI request (verbatim gist): *agent supervision from the corpus's agent bounding boxes; an audio encoder
with a bridge to the same embedding space as in the Qwen omni / Gemma models; check whether the AV dataset
has audio; the text reasoning extracted from Cosmos as an optional output; traffic-rule retrieval as
knowledge; the country as a strategic label learned by the strategic layer, with each country's traffic
rules as embeddings; augment the data for v10 or reference the original data; then a detailed
architecture poster and a motion-graphics video.*

---------------------------------------------------------------------------------------------------------

## 0. Facts established today (all MEASURED, 2026-10-25, scratch dir `/tmp/claude-0/probe/`)

| fact | value | how |
|---|---|---|
| Country per clip exists upstream | `metadata/data_collection.parquet` of `nvidia/PhysicalAI-Autonomous-Vehicles`: columns `country, month, hour_of_day, platform_class, radar_config`, 306,152 clips, 25 countries | HF download + pandas |
| Country coverage of our corpus | **train 4,369/4,369, eval139 139/139** mapped (sha12 → UUID via `index/clip_index_v7.2_{train,eval}.json`) | join |
| Country balance (train) | 25 countries, **68 (Slovenia) – 278 (Denmark)** clips each; United States 265 | join |
| Country balance (eval139) | 23 countries; US 13, Sweden 11, Netherlands 11 … 5 countries with n=1 | join |
| Audio in PhysicalAI-AV | **none**: the 36-feature list (`features.csv`) is egomotion ×2, obstacle.offline, 6 calibration, 7 cameras, 1 LiDAR, 19 radars | HF download |
| Audio in shipped video | **none**: `ffprobe` of 2/2 corpus `camera/<clip>.mp4` → one stream, `hevc` 1920×1080@30, no audio stream | ffprobe |
| Agent tracks in our corpus | `agents/obstacle_offline/*.parquet` (+`index.parquet`): **4,585 / 4,719** clips, 29,415,112 cuboid rows, 6 clips with zero cuboids, 134 clips without a source file | `AGENTS_MANIFEST.json`, `index.parquet` |
| Reasoning text in our corpus | `alpamayo/records.parquet`: `nvidia/Alpamayo2-Super` outputs for **4,729 clips at one anchor t0 = 5.1 s**: tasks `trajectory, meta_action (CoT + meta action), auto_labeling (chain_of_causation), vqa, grounding_via_vqa` | pandas |
| NVIDIA OOD reasoning | `reasoning/ood_reasoning.parquet`: 1,740 clips with per-event chain-of-causation strings; **50** overlap our corpus | pandas |
| Cameras in our corpus | front-wide only (`camera/<clip>.mp4`, 4,719 files, 61.6 GB); the other 6 cameras exist only upstream (chunked zips) | `MANIFEST.json:camera` |
| The 6 other cameras upstream | **all 7 cameras present for 4,508/4,508 v9 clips**, in 1,363 upstream chunks; whole-chunk pull 11.5 TB, member-only range-read **≈ 384 GB ESTIMATED** | `stack/scripts/build_v10_camera_ref.py` → `DataEng/v10/camera_ref/MANIFEST.json` |

Consequences: the country label can be built **now** and losslessly by reference; audio needs external
data; reasoning has a ready supervision source at one anchor per clip; the imagination supervision
producer has all the inputs it needs in our own corpus.

---------------------------------------------------------------------------------------------------------

## 1. Which data is missing for which training (the PI's question)

| trained part | needs | have | missing | how it is filled | blocks |
|---|---|---|---|---|---|
| Fusion, level queries, critics, factored vocabulary (core HiCAP) | frozen-VLM tap caches, motion frames, ego, v9 tactical labels | v9 labels ✓, egomotion ✓, front-wide mp4 ✓ | **VLM tap caches** (GPU build, `build_hicap_cache.py`, never run) | pod job E27 | all real training |
| Strategic level (route decision) | strategic labels | v8.1: one label per clip at t0 = 8 s ✓ | denser strategic supervision | v10 keeps v8.1 by reference | weak but runnable |
| **Strategic level — country** (new) | country per clip | ✓ upstream, joined 100 % | nothing | `stack/scripts/build_v10_country.py` (today) | — |
| Imagination Tier 0 (outcome-aware targets) | per-candidate teacher cost J(a) | agent cuboids ✓ (all 4,508 v9 clips), ego ✓, train vocab ✓ (`DataEng/v10/vocab`) | producer ✓, eval139 + 100-clip train built; **full train build** running | `supervision.py` | Tier 0 |
| Imagination Tier 1 (exogenous E-head) | lead gap / closing / lead-acc at 0.5–3 s, mode, onset | agent cuboids ✓; v9 lead gap cross-check: 90.1 % lead agreement, 92.1 % gap within 2 m after +3.55 m offset (MEASURED, eval139) | full train build | `supervision.py` | Tier 1 |
| Imagination Tier 2 (counterfactual cost heads) | cost of top-K candidates vs logged agent futures | agent cuboids ✓ | full train build (same job) | `supervision.py` | Tier 2 |
| Elastic sensing gate | per-camera-option decision errors | front-wide only | **6 other cameras' caches** + the per-option error runs | pull upstream chunks by reference (v10 `camera_ref`), then per-option eval | gate |
| Knowledge — node towers | frozen-text-tower embeddings of node definitions | definitions ✓ | **embedding job** (GPU, minutes) | pod job | real (not hashed) node embeddings |
| **Knowledge — traffic rules** (new) | per-country rule texts + their embeddings | country ✓ | **RuleBook content + embedding job** | `rules.py` + `rulebook_v1.json` (stream R) | rule memory |
| **Audio bridge** (new) | audio–text pairs; in-domain driving audio | **no audio in any programme corpus** (MEASURED) | **all of it** | external audio–caption sets (§3.4) + synthetic events rendered from agent tracks | audio arms |
| **Reasoning text output** (new, optional) | (scene, decision, explanation) triples | Alpamayo2-Super CoC 4,729 clips × 1 anchor ✓; NVIDIA OOD CoC 50 clips ✓ | denser anchors (optional) | zero-shot first; optional text-channel LoRA on the CoC set | none for zero-shot |
| Student (optional) | trained teacher | — | teacher | after core training | student |
| Contamination control (H-HC2) | exclusion list | ✓ (4/139 eval clips) | — | — | — |

---------------------------------------------------------------------------------------------------------

## 2. The unifying idea: one shared embedding space

Every non-visual source of information enters HiCAP in **the frozen backbone's text-embedding space**
(width d_v; 2,048 for the Cosmos3-Edge reasoner, 4,096 for the 8 B tier):

| source | how it reaches the shared space | trained? |
|---|---|---|
| camera frames | the frozen VLM itself (prefill taps at 3 depths) | frozen |
| node definitions (strategic / tactical tokens) | frozen text path of the same VLM, last-token state | frozen, cached |
| **traffic rules** (per country) | frozen text path, one embedding per rule text | frozen, cached |
| **country** | its node embedding = text path of "driving in ⟨country⟩" | frozen, cached |
| **audio** | frozen audio encoder → **trainable bridge** (perceiver + MLP) → tokens in d_v, aligned to the text path's embedding of an event caption | bridge trained |
| **reasoning text (output)** | the frozen VLM decodes from its own prefix | frozen (optional LoRA on the text channel only) |

This is the omni-model pattern (Qwen2.5-Omni / Qwen3-Omni: audio encoder → projector into the LLM's
embedding space; Gemma 3n: audio encoder → soft tokens in the LM embedding space — PUBLISHED, model
reports; re-verify the exact encoder names before quoting them in the paper) applied to a frozen
driving backbone, so that one HiCAP-side mechanism — **zero-init gated cross-attention from the level
queries into a set of d_v tokens** — consumes audio, rules and any future knowledge source, and every
such source is bit-exactly removable (gate = 0 at initialisation and when the source is absent).

---------------------------------------------------------------------------------------------------------

## 3. The four extensions

### 3.1 Agent supervision for imagination (stream S → `stack/tanitad/hicap/supervision.py`)

Input per clip: ego poses (egomotion, clip-local metres, rig frame +x forward +y left), obstacle.offline
cuboids (track id, class among 10 dynamic classes, centre, size, yaw, timestamp). Output per v9 row
(trainer clock, 10 Hz):

* **Lead identification**: the nearest agent whose footprint overlaps the ego's *logged future corridor*
  (ego path ± half-width + margin) within 0–80 m along-path; `has_lead`.
* **Tier 1 targets** (24 outputs, matching `imagination.EHead`): gap, closing speed, lead longitudinal
  acceleration at 0.5 / 1 / 2 / 3 s (gap = lead arc position − v0·h, may be negative; the E-head gap output is linear); gap log-σ trained
  by a heteroscedastic Gaussian NLL of the forecast (track jitter is ~1 cm, MEASURED, and is not a target); **mode** ids index
  `imagination.MODE_ACC` = {brake −3, hold 0, accelerate +1.5, brake hard −6 m/s²}, which is what Tier 1 integrates; cut-outs are IGNORE
  for the mode head and are emitted separately (`exo_mode4` = {cruise, brake, accelerate, cut-out}; cut-in flagged); **onset**
  q10/q50/q90 of the mode change. Rows whose agent labels cover < 3 s carry J_w = 0 and do not train Tier 0 or the cost heads.
* **Tier 0 / Tier 2 teacher cost** J(a) for each composed vocabulary candidate a (or for the top-K at
  Tier 2): `ttc` (min over agents of time-to-collision of the candidate's swept footprint vs the agents'
  *logged* futures), `clearance` (min distance), `progress deficit` (vs the logged path) — the 3 channels
  of `HiCAPConfig.n_cost`.
* **Cross-check** against v9's own `row__lead_gap_m / row__lead_tg_s` (agreement statistics MEASURED and
  reported, not assumed).
* Counterfactual caveat (carried into the paper): logged agents do not react to a counterfactual ego, so J
  is an *open-loop, non-reactive* cost — the standard nuPlan/NAVSIM caveat; it is a target shape, not a
  simulator.

### 3.2 Audio encoder with a bridge to the shared space (stream A → `stack/tanitad/hicap/audio.py`)

```
mic (mono, or array) ─► log-mel, 2 s window, 0.5 s hop
   └► FROZEN audio encoder  (default: Qwen3-Omni AuT, 647.9 M, Apache-2.0, ~6.8 ms/window on Thor ESTIMATED;
                            contender: BEATs-base ~90 M; decided by E-N0 — note R-G)
         └► TRAINABLE BRIDGE: 2-layer perceiver, 8 latent queries → MLP → 8 tokens ∈ R^{d_v}
               ├─(a, default) HiCAP fusion: zero-init gated cross-attention (existing GatedAudio, d_audio = d_v)
               └─(b, ablation) native: 8 soft tokens appended to the frozen VLM prefill (omni pattern)
```

Training, three stages, backbone and HiCAP frozen during A1:
* **A1 bridge alignment** (contrastive, CLAP-style, symmetric InfoNCE + cosine to the caption): pooled
  bridge output vs the frozen text path's embedding of the caption, on public audio–caption / tagged
  sets (AudioCaps, Clotho, AudioSet strong labels for siren / horn / train horn / reversing beeper /
  tyre squeal / crash, UrbanSound8K, ESC-50, emergency-siren sets). Licences checked per set.
* **A2 event heads** on the bridge tokens: class, bearing (array only), distance bin, approaching — from
  event metadata only (never driving labels; the label-echo argument of `sec_elastic.tex`).
* **A3 gate**: HiCAP's gate g trained on synthetic events rendered from **real agent tracks** of our corpus
  (emitter = an obstacle.offline track; delay, 1/r gain, Doppler, absorption, ITD/ILD, ego-noise floor),
  supervised through *knowledge*, not trajectories: the event caption embedding retrieves the rule
  "yield to emergency vehicles" (§3.4), so no logged trajectory has to "react" to a synthetic siren.

All seven audio controls of the paper (null, shuffle, bearing flip, dose-response, onset shift,
siren-without-vehicle, audio-only label probe) remain mandatory. Audio-off is every family's primary
number; audio-on is a paired arm.

### 3.3 Reasoning text as an optional output (stream T → `stack/tanitad/hicap/reasoning.py`)

* **Off the control path, one-way.** The command is produced by HiCAP's scorer; the text explains it.
  Nothing from the text channel enters the control computation in the default system (audit mode).
* **How**: after HiCAP selects (strategic node, tactical cell, path, profile), the decision is rendered as
  text (`render_candidate`) together with the retrieved rule texts (§3.4); the frozen Cosmos reasoner
  decodes ≤ 48 tokens **reusing the KV cache of the latest refresh prefill** (no second vision pass), at
  ≤ 0.5 Hz in leftover refresh duty (D ≤ 0.5 constraint unchanged). Prompt template:
  *"Scene (cached). Decision: ⟨lat⟩, ⟨lon⟩, target ⟨v⟩ m/s. Applicable rules: ⟨r1; r2⟩. In one sentence,
  explain why."*
* **Consistency monitor**: the explanation is embedded by the same frozen text path and mapped to the
  nearest tactical node; disagreement with the executed node is logged (and, as a registered ablation
  only, may raise the System-2 verifier's trigger).
* **Optional training**: a LoRA on the reasoner's **text channel only** (disabled during tap extraction so
  the cached taps stay byte-identical), trained on Alpamayo2-Super chain-of-causation (4,729 clips × 1
  anchor) + NVIDIA OOD CoC (50 overlapping clips), conditioned on the *executed* decision. Metric:
  text→node agreement rate, decision-consistency, and a human audit sample; no BLEU-only claim.
* Contamination note: Alpamayo2-Super texts are a teacher's output (INHERITED quality), not ground truth.

### 3.4 Traffic-rule knowledge and the country label (stream R → `stack/tanitad/hicap/rules.py`)

* **Country as a strategic label.** `y_country ∈ {25 countries}` per clip (v10, §4), broadcast to every
  row. The **strategic level query (s1)** gets a contrastive country readout against **country node
  embeddings** (text path of "driving in ⟨country⟩"), loss λ_c = 0.1. Cues are visual (signs, markings,
  plates, road furniture). Optional GNSS geofence input at inference as a *supplied* arm (a production car
  knows its country); the *predicted* arm is primary.
* **RuleBook** (`rulebook_v1.json`, versioned like `NodeLibrary`): records
  `{rule_id, countries, text, args (e.g. speed km/h), road_type, effect ∈ {prune, prior, memory}, source}`.
  Initial content, decision-relevant only: driving side; default speed limits urban / rural / expressway /
  motorway; right-turn-on-red (US permitted unless signed; EU generally prohibited, DE only with the green
  arrow sign); priority to the right at unsigned junctions; roundabout entry yields to circulating
  traffic; yield to emergency vehicles; pedestrian-crossing yielding; tram priority; school-bus stop (US);
  daytime lights. Every fact carries its source (statute or official page), PUBLISHED, and is marked
  UNVERIFIED where a second source was not found.
* **Embeddings**: each rule text → frozen text path → r_i ∈ R^{d_v} (cached, once per RuleBook version).
* **Retrieval** (per tick, cheap): country posterior p(c) from s1 → rule mask weight
  w_i = Σ_c p(c)·1[i ∈ R_c]; relevance s_i = ⟨W_q q_s2, W_r r_i⟩; top-k = 8 rules by w_i · softmax(s_i).
* **Injection, three routes** (same as the three knowledge routes of the paper):
  1. **memory**: retrieved rule tokens → zero-init gated cross-attention into the s1/s2 queries
     (`GatedMemory`, the GatedAudio mechanism generalised; rules-null is bit-exact);
  2. **prune**: rules with numeric args become masks / priors in `prune.py` (e.g. speed-profile cap at
     limit × (1 + tolerance) when the road type is known; right-on-red candidates removed where
     prohibited); a rule may only remove candidates the R1 feasibility mask has not already removed, and
     the guard set always survives;
  3. **verify / explain**: retrieved rule texts are part of the System-2 verifier's prompt and of the
     reasoning-output prompt (§3.3).
* **Admissibility** (the goal-input rule, CLAUDE.md 2026-08-03): country and rules are **not** the
  situation classifier's output (HiCAP has no situation classifier head); the goal / route path does not
  read the country posterior; the rule memory feeds the level queries, not the goal token. Leak check:
  the country label comes from upstream metadata, which is never an inference input unless the supplied
  (GNSS) arm is selected — both arms are reported.
* **What the corpus can and cannot test** (HYPOTHESIS until measured): all 25 countries drive on the
  right, so driving side is untestable; right-on-red differs only for 265 US clips and the corpus has
  no traffic-light labels; speed-limit compliance is testable through the longitudinal family against
  `speed_max_input` (v8.1). Pre-registered: H-HC18 country accuracy (vision-only) vs majority and chance
  1/25, per-country with n; H-HC19 rules on/off paired deltas per family; H-HC20 **country-swap
  counterfactual**: swapping the country posterior changes the decision only on rule-relevant cells
  (a specificity test, both outcomes committed).

---------------------------------------------------------------------------------------------------------

## 4. v10 data release: augment by reference, never by copy

`v10 = v9 labels + v8.1 strategic (unchanged, by reference) + new components keyed by sha12 / clip_id`:

| component | file (proposed, HF `Sayood/tanitad-refcv8-v10-labels`) | source | status |
|---|---|---|---|
| country | `country/v10_country.parquet` (`split, sha12, clip_id, country, country_id`) | upstream `data_collection.parquet` | **built today**, repo copy `DataEng/v10/` |
| imagination supervision | `imag/v10_imag_{train,eval139}.npz` | corpus `agents/obstacle_offline` + egomotion | producer written (stream S); full build CPU/pod |
| rules | `rules/rulebook_v1.json` + `rules/rule_emb_<backbone>.npy` | curated + frozen text path | content stream R; embeddings pod |
| reasoning | `reasoning/v10_coc.parquet` (`sha12, t0_us, source, text`) | corpus `alpamayo/records.parquet` + upstream OOD CoC | builder in stream T |
| extra cameras | `camera_ref/v10_camera_ref.parquet` (per clip: chunk, per-camera presence and chunk path) | upstream `clip_index.parquet` + `feature_presence.parquet` | **built today** (`DataEng/v10/camera_ref/`); pull members on the pod |
| audio | — (external sets listed with licences) | AudioCaps / Clotho / AudioSet / … | not in v10 |

Publishing to HF is the PI's decision; everything above is staged in the repo first.

---------------------------------------------------------------------------------------------------------

## 5. Frozen vs trained (the poster's colour key)

| part | state | params |
|---|---|---|
| Cosmos3-Edge reasoner (vision tower + LM) — taps, text path, verifier scoring, reasoning decode | **FROZEN** (optional text-channel LoRA for the explanation only) | 2.43 B (PUBLISHED card) |
| Audio encoder | **FROZEN** | candidate-dependent |
| Motion adapter, projectors, fusion (6 L, d 512), level queries, readouts, node/numeric towers, re-ranker | trained | 56.4 M (MEASURED, `HiCAP.n_params`) |
| Imagination E-head, Tier-2 heads, camera gate, config embedding | trained | +1.88 M (MEASURED, `HiCAPFull`) |
| Rule memory (retrieval 0.33 M + gated cross-attention 2.10 M), country readout 3.67 M | trained | **6.10 M** (MEASURED, `HiCAPFull.knowledge_params`, d 512, d_text 2,048) — complete model **64.37 M** without the audio bridge |
| Audio bridge + event heads | trained | **10.13 M** at d_v 2,048 mono (MEASURED, `rg_audio_bridge_params_result.json`); HiCAPFull with bridge and d_audio = d_v: **74.62 M** |
| Node / rule / country text embeddings | cached once from the frozen text path | 0 trainable |

## 6. Training schedule (poster / video) — as implemented (`losses.STAGES`, `full.FullLoss`; rev b aligns the spec to the code)

Stage 0 (offline, once): VLM tap caches; text embeddings of nodes, rules, countries; vocabulary build; imagination supervision;
country labels. Stage A1 (independent): audio bridge alignment.
Stage 1: level critics on every level (soft targets), goal loss, **country readout** (λ 0.1); Tier-0 target shaping and the camera gate
whenever their supervision is in the batch. Stage 2: + hard negatives, consistency KL, 40 % replay. Stage 3: + cost heads, refinement,
L^cfg, audio, imagination (E-head, Tier-2 heads); the re-ranker is trained only here. The rule-memory and audio gates start at 0 and are
trained by the main losses from the moment their source is attached. Optional: student distillation; text-channel LoRA for explanations.

## 7. Inference data flow (one tick, 10 Hz command clock; 2 Hz refresh clock)

refresh (async, 2 Hz): cameras → frozen VLM prefill → taps cache (+KV cache for explanations);
audio (async, 2 Hz, optional): mic → encoder → bridge → 8 tokens.
tick (10 Hz): motion frames + ego (v, a_lon, yaw rate) + nav/route + cached taps (+ audio tokens) → fusion
→ level queries → s1: strategic node + **country posterior → rule retrieval** → s2: tactical cell (lat ×
lon) → s3: path × profile via factored vocabulary → prune (R1 physics, route, **rules**, conformal) →
re-rank (+ Tier 1 analytic imagination, Tier 2 veto/benefit gate) → System-2 verifier only if uncertain →
command. Optional: explanation text (≤ 0.5 Hz, off the control path).
