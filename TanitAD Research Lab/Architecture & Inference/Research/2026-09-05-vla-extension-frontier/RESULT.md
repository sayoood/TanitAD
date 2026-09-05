# VLA extension frontier research — a language backbone for the tactical and strategic layers of REF-C v5

**status: IN PROGRESS — done: §0 (our architecture, the Alpamayo CoT we hold, Thor) / next: §1 survey (primaries banking in progress), §2 design options, §3 grounding, §4 hypotheses, §5 traps, plan (B)**

- **Agent:** TanitAD Research Lab (Architecture & Inference), time-boxed ~4 h, zero GPU, no sub-agents.
- **Branch:** `agent/arch-inf-20260803`. **Date:** 2026-09-05.
- **Ask (PI, verbatim summary):** extend REF-C (+ planned gap-closing extensions) and the 4b architecture with a
  vision-language part that shares REF-C's embedding space, processes nav commands, question queries, images and ego
  data, emits a system-initiated chain of thought (text) explaining the behaviour — distilled from the Alpamayo CoT we
  extracted — grounded in the scene and consistent with the trajectory hypotheses (no hallucinated internal thoughts),
  able to process AND emit strategic and tactical goals, complementing the current strategic/tactical layers, at a
  300–500 ms tact, ≤ 1 B parameters (smaller better), on the Jetson Thor.
- **Companion plan:** `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` (deliverable B).
- **Evidence classes used:** PUBLISHED-PRIMARY (table/section cited, primary banked in the Library) ·
  PUBLISHED-SECONDARY (abstract/aggregator/web page only — inadmissible for the registry) · MEASURED (ours + artifact
  path) · INHERITED (another doc, not re-verified here) · ESTIMATED (derivation shown) · HYPOTHESIS.

## 0. What OUR architecture is (read, not re-derived)

### 0.1 REF-C v4b interfaces the module must read and write

All rows INHERITED from source-read documents (each MEASURED there against `refc.py` / `refc_v3.py` and the live run's
`config.json`): `…/2026-09-04-refcv4b-seam-state/SEAM_STATE.md`, `Project Steering/DESIGN_REFCV4_NAV_WIRING.md` §1–§4,
`Project Steering/MODEL_REGISTRY.md` §4.5–4.6. Line numbers are theirs and were re-pinned by content on 2026-09-04.

| node | what it is | shape / width | where | status today |
|---|---|---|---|---|
| `pooled` | vision-only ResNet-34-style trunk → **8 × 20 = 160 perspective-view tokens** at t0 | `feat_dim` per token (the decoder's `feat_proj: Linear(feat_dim, d)` reads it, `refc.py:1206`) | `refc.py:1900`, `:1574` | live; the ONLY scene carrier for core tactical/strategic aux heads |
| `pooled_seq` | the same tokens over the 8-frame window | [B, 8, …] | `refc.py:2190` | feeds `ctx` and `z_tac` |
| `ctx` | strategic GRU context (`StrategicCtxConfig(hidden=512, d_ctx=256)`, widened 64 → 256 on 2026-09-02) | **256** | `refc_v3.py:436, :648` | live; `ctx = ctx + nav_s` inside `hook()` (`:861`, zero-init `nav_to_str`) |
| `g_str` | **cascade strategic goal** = `str_goal_head: Linear(256, 3)` → (cos, sin) route bearing + `tanh` along-track preference | 3 | `refc_v3.py:655, :883` | live but **NAV_BLIND** (nav-compliance Δ_shuffle = 0.0000, `…/2026-09-05-nav-compliance-metric/RESULT.md` §0); its only gradient is the zero-init FiLM into `z_tac` |
| E19 (planned) | widens `str_goal_head` to `Linear(256, 3 + 4 + K + 4 + K)`, K = 6: `g_str_geo(3)`, `p_man(4)` NEXT routing manoeuvre ∈ {left, right, straight-through→folded into none, none}, `t_bin(6)` time-to-event over [0, 30] s (0-2, 2-5, 5-9, 9-14, 14-21, 21-30), `p_man2(4)`, `t_bin2(6)` | 23 | `DESIGN_REFCV4_NAV_WIRING.md` §3.4 | planned; label from `manoeuvre_sequence`, `STRATEGIC_S = (8.0, 30.0)` (`s2_geom_emit_v7.py:52`) |
| `z_tac` | tactical latent `phi_tac(pooled_seq)` → FiLM'd by `g_str` (zero-init) | `d_tac = 512` | `refc_v3.py:334, :652, :900-902` | live; input to the scorer and to `target_latent` |
| `lat/lon_logits_tac` | the **v7.2 8-wide** factored tactical heads (`lat_head_tac`, `lon_head_tac`) | `_nlat`, `_nlon` | `refc_v3.py:758-759` | live; **reach the decoder only through E7/E9 latent conditioning** (SEAM_STATE) |
| `g_tac` | tactical goal (`tac_goal_head`) — E14 echo-quotiented form `ha0_ext + zero-init residual` | — | `refc_v3.py:902-907` | live; reaches selection through `goal_gate·scorer`, gate **opened, 0.1744** |
| H19 prior | anchor prior reweighted every step by the model's own **kin3** factored heads via `derive_man5_logprobs` on `tac_in = pooled` (vision-pure) | 5-way surface | `refc.py:2290-2305` | live; the EXTERNAL port `maneuver_logits` (an outside tactical brain speaking the 5-way surface, `refc.py:2306-2313`) is **never filled** |
| decoder | 117-anchor v0-conditioned kinematic vocabulary (v4b; `anchors.pt` units = lateral accel, m/s², `--anchor-control-units alat`), cross-attention from anchor queries to the 160 PV tokens + FiLM on the measurement `m`, confidence + offset heads, 1 + `steps` deterministic passes (no diffusion: `…/2026-09-05-refc-vs-diffusiondrive-audit/RESULT.md` §0) | fan of 117 | `refc.py:1470, :1596-1603` | live |
| selector | `sel_score → sel_score_v3 → argmax` | — | `refc.py:1639-1690`, `refc_v3.py:1058-1073` | live; sees nav only diluted through `m`→`conf0` and through `man5`; S7/S8 (nav / `g_str` → selector) planned |
| `m` | the measurement encoder — `v0` (+ derived ego channels, per-sample **ego withholding 0.5** + X15 presence bit) ⊕ nav one-hot; the ONLY carrier of nav today | — | `refc.py:2273-2275`, `:1503` | live |
| nav | **3-way v7.2 token** `NAV_COMMAND_TOKENS = (NAV_FOLLOW_ROAD, NAV_TURN_L, NAV_TURN_R)`, provenance `ego-future`, a **per-CLIP constant**; `nav_known` bit (E21) | 3 (+1) | `refc_v3_train.py:266-326` | `--nav-from-v7` ON in refcv4b; E15/E17/E18 (nav → core tactical / selector, `g_str` → selector) planned |

**Sizes (MEASURED, registry §4.5):** refcv3 base = **107,032,901** params (core 104,879,522 · phi_tac 1,757,440 ·
tac_latent_proj 262,656 · gstr_cond 66,816 · nav_inject 50,176 · tac_heads 14,364 · scorer 1,156 · str_goal_head 771).
The whole cascade above the trunk is **~2.15 M** parameters — the language module will be 100–400× larger than the
layers it complements, which is the first design constraint (§2): the VLM must not be allowed to *replace* the
cascade's decisions, only to read and propose into them.

**The label vocabulary the module must speak (INHERITED, `DESIGN_REFCV4_NAV_WIRING.md` §3):** three time bands from
`s2_geom_emit_v7.py:50-59` — `OPERATIVE_S (0, 2)`, `TACTICAL_S (2, 6)`, `GAP_S (6, 8)` belongs to no layer,
`STRATEGIC_S (8, 30)`; `LOOKAHEAD_S = 30`. Band census (train, `…/2026-08-30-v72-copy-adjudication/raw/band_census.json`):
1,882 manoeuvres start in [8, 30), 463 in [2, 6), 687 in [0, 2), 256 beyond 30 s. The tactical vocabulary is the
v7.2 8-way factored lat/lon set (`stack/tanitad/data/vocab_v7.py`); the strategic vocabulary is E19's
(`p_man`, `t_bin`, `p_man2`, `t_bin2`) plus `g_str_geo`.

**What is measured about the seam today (MEASURED by siblings, T1 early-training diagnostic on refcv4b ~step 700–2,000):**
nav-compliance of the selected anchor **0.390 [0.303, 0.472]** → **0.291** under nav-SHUFFLE (paired Δ **+0.099
[+0.041, …]**, 12 % of the always-commanded ceiling 0.837); `g_str` **NAV_BLIND**; hold-action `ha` still ties or beats
`os` on ADE@6 s (3.560 vs 3.517 m, n.s.). ⇒ The strategic layer the VLM is asked to *complement* does not yet follow
its own command; the module must therefore be designed so that it can be evaluated **independently** of whether
`g_str` works (§3.4), or it inherits an unfalsifiable seam.

### 0.2 The Alpamayo CoT we hold — n, format, alignment (MEASURED today)

**MEASURED 2026-09-05** on the local verified copy `C:\Users\Admin\tanitad-data\alpamayo\records.parquet`
(25,970,018 B — byte-identical size to the HF `Sayood/tanitad-alpamayo2-augmentation` listing in the registry §11.1;
sha256 `ecae276db9969de1…` per D-DATA-ALPA-FULL), read with pyarrow in `venvs/tanitad`:

| fact | value |
|---|---|
| rows / clips / tasks | **23,644 rows · 4,729 unique `clip_id` · 5 tasks** (`trajectory` 4,729 · `meta_action` 4,729 · `auto_labeling` 4,729 · `vqa` 4,729 · `grounding_via_vqa` 4,728) — the registry's "cot" task is the `cot` FIELD carried by every row's `raw_json`, not a task |
| anchor | `t0_us` = **5,100,000** on every row (Alpamayo's default), vs our s2 label anchor **8.0 s** — every join is 2.9 s off unless re-anchored |
| producer | `nvidia/Alpamayo2-Super` (34.3 B), seed 42, temperature 0.6, **ONE draw per clip** (D-DATA-COT-HALLUC) |
| **`cot` (the "Chain of Causation")** | **one sentence**: median **57 chars / 10 words**, p10 43 / p90 86 chars (7 / 14 words), max 216 chars / 41 words, 0 empty. Example: *"Slow down due to the lead vehicle ahead."* |
| `auto_labeling` JSON | `chain_of_causation` 4,729/4,729 (median 64 chars) · `critical_components_analysis` **3,140** (median 179 chars; `type: none` on 853 clips is an explicit clean negative, D-DATA-ALPA-FULL) · `ego_vehicle_motion_analysis` **2,943** (median 314 chars; typed, time-stamped segments, 97.1 % parse) · `trajectory_analysis` **133** (median 703 chars) |
| `meta_action` triplet | LON 7 classes: Gentle Decel 1,594 · Maintain 1,225 · Gentle Accel 1,151 · Stop 304 · Strong Decel 267 · Strong Accel 182 · Reverse 6. LAT 7: Go Straight 2,504 · Steer Right 1,020 · Steer Left 649 · Sharp R 135 · Sharp L 115 · Reverse L/R 1+1. LANE 7: Lane Keep **4,035** · Turn Right 101 · Turn Left 85 · Right LC 82 · Slightly Shift L 69 · Slightly Shift R 31 · Left LC 22 |
| `trajectory` | Alpamayo's own predicted trajectory (`pred_xyz`, `pred_rot`, `logprob`) on **4,474** rows (native horizon 6.4 s per registry §11.1; the nested shape did not parse in this pass — **UNVERIFIED**); 255 rows carry the rendered-figure metadata with `ade_m`/`fde_m` against GT |
| `grounding_via_vqa` | 2-D boxes (`[{bbox_2d, label}]`) on 4,728 rows — Car 2,292 · Pedestrian 1,079 · traffic light 889 clips (D-DATA-ALPA-FULL) — **the fusion gate**: the only channel that lets a CoT token be checked against image space |
| `vqa` | ~5 sampled questions per clip; spot-check only, never a dense label (D-DATA-ALPA-FULL) |
| quality, MEASURED by siblings | visually checkable CoT claims **3 correct / 2 wrong (n = 5)** incl. a hallucinated cyclist through a 68–82° junction turn (D-DATA-COT-ACC, D-DATA-COT-HALLUC); LATERAL and LANE axes **at chance** vs shuffle on the 39-clip s2 subset (D-DATA-ALPA-LAT) but `meta_action` lateral lift **×1.68** on n = 1,584 (D-DATA-ALPA-FULL); longitudinal `meta_action` monotone with Cohen's d 1.59; CoT-vs-geometry lateral disagreement **41.7 %** of 4,416 non-null on the release blob (D-LAT-AGREE, unadjudicated); five tactical referent tokens recoverable by term search (GAP_TARGET 22.0 %, STOP_POINT 18.2 %, EVADE_IN_CORRIDOR 16.7 %, TRAFFIC_LIGHT_REACT 13.3 %, YIELD_AT 8.4 %; 83.0 % of CoTs carry ≥ 1 referent — D-DATA-COT-TOKENS, all `disputed=true`) |
| alignment to our clips | joins by `clip_id` + `t0_us` to PhysicalAI-AV; the B1 training corpus **IS** the Alpamayo-labelled set (4,719 clips, D-CORPUS-B1); v7.2 labels 4,572 train / 147 eval (D-V72-SPLIT); the deployed val40 carries v7 labels on only 6/40 clips (D-VAL40-NOLABELS) |

**What this means for "distil the CoT we extracted":**

1. ⛔ **It is not a chain of thought.** It is **4,729 one-sentence causal summaries (~47 k words in total)** plus
   structured JSON on 62–66 % of clips. There are no intermediate reasoning steps to distil; Alpamayo-R1's multi-step
   *Chain-of-Causation traces* (the paper's training target, lib `2511.00088`) were never generated for our corpus —
   Alpamayo2-Super was run with a one-sentence `cot` prompt. A text decoder can be fine-tuned on this to emit the same
   *kind* of sentence; it cannot learn multi-step reasoning from it.
2. ⭐ The **structured fields are the better distillation target**: `chain_of_causation` (cause → action),
   `critical_components_analysis` (the forcing object, with an explicit `none`), `ego_vehicle_motion_analysis`
   (time-stamped segments), the `meta_action` triplet, and the 2-D boxes. Together they are exactly a
   *template-structured CoT* (critical object → its state → the causal rule → the manoeuvre), which is the form both
   FastDriveCoT (`2602.02864`, parallel-decodable) and XCoT-VLA (`2608.10976`, 2–6 executable tokens) exploit for
   latency. Our data already has that shape.
3. ⚠️ **Nothing in it may be believed without a grounding pass** — the corpus's own measurements (3/5 checkable claims,
   hallucinated cyclist, lateral axis at chance) match the 2026 external audits of Alpamayo-R1 itself (42.5 % reasoning
   fidelity, 48.3 % reasoning-action consistency, `2605.17268`; 33.3 % of CoTs unreliable, `2608.29583`). A teacher that
   is wrong half the time on what it *sees* is a teacher only for the *form* of the explanation; the content must come
   from our own labels and detections (§3).
4. The anchor (5.1 s vs 8.0 s) and the CoT's per-clip (not per-window) granularity mean the distillation set is
   **≤ 4,729 (image window, ego, nav) → sentence pairs at one instant per clip**; every window-level explanation label
   beyond that must be synthesised from the v7.2 manoeuvre sequences + `obstacle.offline` tracks (the label may use ego,
   PI 2026-08-03), which is a DataFlyWheel work package (plan §WP-2).

### 0.3 Thor as the latency target

- Fleet/target facts (MEASURED, CLAUDE.md 2026-08-03/08-16): Jetson Thor, unified memory; **only in-process
  `torch.cuda.max_memory_allocated()` is admissible**; the 20 SMs **saturate at batch 8** — throughput flat at
  **12.3–14.1 windows/s across a 6× batch range** for the v6F *training* step (this is a training-throughput number at
  batch 8, not an inference latency, and it is for a different trunk; the REF-C trunk's Thor inference latency is
  **UNMEASURED** and is the first item of the plan's latency budget).
- Toolchain (PUBLISHED-SECONDARY, Jetson AI Lab "TensorRT Edge-LLM" page, fetched 2026-09-05): Thor (SM110) supports
  **FP16, INT4-AWQ, NVFP4, FP8**; supported families include Qwen3 dense 0.6B–14B, **Qwen3.5/3.6 text 0.8B–27B**,
  Qwen3-VL 2B–8B, **InternVL3/3.5 1B–14B**, Nemotron-Nano, and **Alpamayo R1**; the exporter splits the LM and the
  visual encoder into two engines. ⚠️ The page lists the Qwen3.5 small models under *text*, so Qwen3.5-0.8B's native
  vision path is not confirmed in the SDK — irrelevant for option (c), which feeds our trunk's tokens and never runs
  a second vision encoder, and a measurement item for option (a).
- Qwen3.5-0.8B (PUBLISHED-SECONDARY, HF model card, fetched 2026-09-05; family primary = Qwen3.5-Omni report
  `2604.15804`): 0.8 B, hidden 1024, **24 layers in 6 × (3 × Gated-DeltaNet → FFN, 1 × Gated-Attention → FFN)**,
  FFN 3584, GQA 8 Q / 2 KV heads × 256, native context 262,144, early-fusion multimodal (image + video), Apache-2.0,
  released 2026-02. The 3:1 linear-attention hybrid matters for us: **decode cost per token is nearly
  context-independent** and the KV cache is a quarter of a dense model's — the property a 2–3 Hz tact with a long
  standing prefix (scene tokens + goal state + prior thoughts) wants. ⚠️ Not "Qwen 3.6 0.8B": the 3.6 release
  (2026-04) is 27B dense / 35B-A3B only; the sub-1B member of the family is 3.5-0.8B.
- **Decode budget on Thor (ESTIMATED; must be MEASURED in WP-0):** small-model decode is weight-bandwidth-bound. With
  a 0.8 B decoder the per-token floor is `bytes(weights) / bandwidth`: bf16 1.6 GB, FP8 0.8 GB, NVFP4 0.4 GB. At the
  Thor spec bandwidth (PUBLISHED-SECONDARY, NVIDIA spec page — see §1.3 for the fetched figure) the floors are of order
  **5–6 ms/token bf16, ~3 ms FP8, ~1.5 ms NVFP4**, *before* the vision/prefill share and *before* the REF-C trunk
  itself. A 300–500 ms tact therefore holds roughly **40–70 generated tokens at bf16, 100–150 at FP8** if the language
  module owned the whole tact — and it does not: the trunk, the cascade and the decoder run in the same budget. ⇒ the
  emitted CoT per tact must be **short and structured (≤ ~40 tokens)**, or **latent/executable** with the text rendered
  asynchronously at a lower rate (§2). Every number in this bullet is a *floor from a spec*; the plan's first work
  package replaces it with a measurement on `tanitad-thor-wifi`.

## 1. Survey — primaries banked
_(in progress — batch 1 banked: `2410.23262` EMMA · `2402.12289` DriveVLM · `2503.23463` OpenDriveVLA · `2405.01533` OmniDrive · `2410.22313` Senna · `2312.14150` DriveLM · `2312.07488` LMDrive · `2503.19755` ORION; batch 2 in flight)_

### 1.1 Driving VLM/VLA line
### 1.2 Small-VLM line (≤ 1 B)
### 1.3 CoT-for-control and latency line
### 1.4 Grounding / faithfulness line

## 2. Design options for OUR module (priced)
### 2.a Pretrained small VLM + projector from `pooled`/`z_tac`/`ctx`
### 2.b From-scratch small decoder on our corpus + distilled CoT
### 2.c Hybrid — frozen small LM + LoRA, REF-C trunk as the ONLY vision encoder
### 2.d Comparison table

## 3. The grounding / consistency mechanism — the USP
### 3.1 Consistency losses text ↔ `g_tac` / selected anchor
### 3.2 Verification against the fan
### 3.3 Contradiction detection and grounding tokens (`obstacle.offline`)
### 3.4 EXPLANATION FAITHFULNESS — the fifth metric family, with controls

## 4. Own hypotheses — `H-VLA-*` (validation on the tiny rig / dev box, both outcomes committed)

## 5. What we must NOT do

## 6. Deliverable manifest
_(pending)_
