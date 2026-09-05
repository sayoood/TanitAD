# VLA extension frontier research — a language backbone for the tactical and strategic layers of REF-C v5

**status: IN PROGRESS — done: §0 (our architecture, the Alpamayo CoT we hold, Thor), §1 survey (58 primaries banked) / next: §2 design options, §3 grounding, §4 hypotheses, §5 traps, §6 manifest, plan (B)**

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

**Banking status (MEASURED, `TanitAD Research Lab/Library/library.json`):** 40 primaries carry the tag
`vla-frontier-2026-09-05` from batch 1–2, plus 18 added in batch 3; every paper named below is a **banked PDF with a
sha256**, so numbers read from it are `PUBLISHED-PRIMARY`. Where a number comes from the paper's own **abstract** it is
marked *(abstract)*; where it comes from a table or a numbered section, the table is named. Rows marked
`PUBLISHED-SECONDARY` come from a model card or a vendor page and are **inadmissible for `MODEL_REGISTRY.md` or the
paper** — they are here only to size an option.

⭐ **The four numbers that decide our design, before any table.**

1. **Alpamayo-R1's famous "99 ms" is a WORKSTATION number and 71 % of it is the CoT.** `2511.00088` Table 14 measures
   on an **NVIDIA RTX 6000 Pro Blackwell** — not the vehicle computer: vision encoder **3.43 ms**, prefill **16.54 ms**,
   **reasoning decoding 70 ms for 40 tokens**, trajectory decoding 8.75 ms (5 flow-matching steps); total **99 ms**.
   The trajectory-only baseline is **29 ms**, and decoding the trajectory autoregressively instead (127 tokens) costs
   **312 ms**. ⇒ On a workstation Blackwell, **1.75 ms/token** for a 10 B model, and the language is the budget.
   *(⛔ Quoting "99 ms" as an on-vehicle latency is the datacentre-GPU trap the PI's brief forbids; the paper's own
   on-vehicle section claims real-time deployment but publishes **no embedded breakdown**.)*
2. **The only published EMBEDDED driving-VLM latency table says a ~1.8 B model decodes at 79.6 tok/s.** `2402.12289`
   (DriveVLM) Table 6, quantised, **on an OrinX chip**, prompt ~1,078 tokens, 59 output tokens: **Qwen1.8B** prefill
   **0.23 s** (4,709 tok/s), decode **79.6 tok/s** ⇒ **12.6 ms/token**, decode latency 0.74 s; MobileLLaMA-1.4B
   **117.4 tok/s** (8.5 ms/token); Gemma-2B 40.9 tok/s; Qwen4B 44.5 tok/s. Their conclusion — *"wide and shallow"
   (Qwen-series) beats "narrow and deep" on Orin* — is an architecture-selection fact, not a size fact.
3. **DriveVLM-Dual ships at 410 ms on OrinX — inside the PI's 300–500 ms tact — precisely because the VLM does NOT
   drive.** `2402.12289` §6: two OrinX processors, a high-frequency end-to-end driver on OrinX-1 and the VLM on
   OrinX-2, **operating asynchronously**; "average inference speed of **410 ms** on OrinX". This is the published
   existence proof for the architecture the PI is asking for, and it is asynchronous by construction.
4. **A short, structured, small-vocabulary CoT is 4–30× cheaper than a free-form one, and the levers are additive.**
   `2602.02864` (FastDriveCoT): a template CoT is **300–500 tokens**, and parallel decoding over its dependency graph
   gives **3.1–4.1×**. `2608.10976` (XCoT-VLA): **2–6 executable tokens** replace **40–80** free-form ones.
   `2402.12289` Table 9, **on the deployed OrinX stack**: q4f16 base 109 tok/s → EAGLE + fine-tune **295 tok/s (2.7×)**
   → **plus shrinking the vocabulary to the 1,024 most frequent words, 518.3 tok/s (4.33×)**, decode latency
   0.328 s → **0.071 s**.

### 1.1 Driving VLM/VLA line

**(a) The systems that emit GOALS rather than trajectories — the line REF-C actually needs.**

| system (lib key) | what it does | parameters | latency (and on WHAT hardware) | how goals/actions are emitted | faithfulness measured? |
|---|---|---|---|---|---|
| **DriveVLM / DriveVLM-Dual** `2402.12289` | CoT of three modules — scene *description* → scene *analysis* → **hierarchical planning**; Dual adds a classical 3D-perception + planner fast path | backbone < 4 B by design ("limited memory and bandwidth of the vehicle's hardware"); deployed backbone Qwen-series | **410 ms on OrinX** (§6, production vehicle, 2 × OrinX, asynchronous); Table 6 per-LLM breakdown | **meta-actions → decision description → waypoints**, i.e. a *hierarchy of goals* in text, with the fast path doing high-frequency trajectory refinement | no; hallucination penalised in the SUP-AD score only |
| **Senna** `2410.22313` | Senna-VLM (decision) + Senna-E2E (VAD) — high-level planning decoupled from trajectory | LVLM + E2E planner (VLM not sub-1 B) | not published for embedded HW | **meta-actions in natural language**, executed by a separate numeric planner — the paper's core claim is that LVLMs are *not* suited to precise numerical prediction *(abstract)* | no |
| **RAD** `2503.13861` | retrieval-augmented **meta-action** decisions with a VLM | — | — | discrete manoeuvre, retrieved-example conditioned | no |
| **XCoT-VLA** `2608.10976` | replaces rationales with **executable CoT tokens**; the XCoT sequence stays in context and conditions fixed trajectory queries through shared self-attention; Reason-FFN / Control-FFN routing; flow-matching trajectories | not published | "within the real-time planning budget"; **2–6 tokens vs 40–80** | a **fixed interpretable vocabulary**: `KEEP_SPEED`, `DECELERATE`, `LANE_KEEPING`, `NAV_LEFT_LANE_CHANGE`, `RIGHT_TURN_PREPARE`, `RIGHT_EXIT_PREPARE`, `RED_LIGHT_HOLD`, `LEFT_OVERTAKE`, `VISIBILITY_CAUTION`, `HAZARD_YIELD`; **deterministic ordering** (primary lateral/nav manoeuvre first, then interaction/longitudinal, then rule/safety modifiers) so one decision has one canonical sequence | indirectly — the tokens are *executable*, so a wrong token is a wrong action |
| **Alpamayo-R1** `2511.00088` | Cosmos-Reason VLM + **flow-matching trajectory decoder**; Chain-of-Causation SFT then RL on reasoning quality **and reasoning-action consistency** | **10 B** released; ablations 0.5 B → 7 B, monotone | **99 ms on an RTX 6000 Pro Blackwell** (Table 14) | text CoC + 128 discrete trajectory tokens at training (**64 waypoints × 2 quantised control values: acceleration and curvature**), flow matching at inference | **yes, as an RL reward**: +45 % reasoning quality, +37 % reasoning-action consistency from RL post-training *(abstract)* |
| **ORION** `2503.19755` | QT-Former long-horizon memory + LLM + generative planner, **aligned in one latent space**; Bench2Drive | — | — | reasoning space aligned to action space via a shared latent *(abstract)* | ⚠️ audited: **ORION's CoT is epiphenomenal** under intervention (`2606.12706`) |
| **BLUE** `2606.08684` | a **0.11 M-parameter gate on frozen VLA hidden states** decides *per frame* whether to generate language at all — because "language matters on only a small fraction of routes" *(abstract)* | +0.11 M | — | gating, not emission | — |

**(b) The rest of the driving line, banked, compressed.** `2410.23262` **EMMA** (Waymo): every non-sensor input and
output — navigation instructions, ego status, waypoints, 3-D boxes — is **natural-language text** in one Gemini-scale
model; nuScenes planning SOTA; **no on-vehicle latency is claimed**, and the model is not deployable at our budget.
`2503.23463` **OpenDriveVLA**: Qwen at **0.5 B / 3 B / 7 B** over 3-D-aware BEV/agent/map tokens with a *hierarchical
vision-language alignment* that projects 2-D and 3-D structured visual tokens into one semantic space — the closest
published instance of "a projector from an existing perception trunk into an LM's embedding space", and its 0.5 B
variant is the closest published size to ours. `2405.01533` **OmniDrive**: sparse 3-D Q-Former queries into a 7 B LLM,
plus **counterfactual** QA ("what if I went straight?") — a supervision idea we can reuse against our own fan.
`2312.14150` **DriveLM**: Graph-VQA over perception → prediction → planning, the standard structured-QA supervision.
`2312.07488` **LMDrive**: closed-loop CARLA under language navigation instructions. `2503.07608` **AlphaDrive**: GRPO
with four planning-specific rewards + a two-stage reasoning curriculum. `2506.08052` **ReCogDrive**: AR VLM + diffusion
planner + simulator-assisted RL, motivated explicitly by *"format-violating outputs, infeasible actions, and slow
inference"* from language-space trajectories *(abstract)*. `2505.19381` **DiffVLA**: VLM output as guidance for a
sparse-dense diffusion planner (NAVSIM v2). `2502.14917` **Sce2DriveX**: MLLM CoT from local video + global BEV.
`2412.09951` **WiseAD**: a MobileVLM-based *small* driving VLM trained jointly on driving knowledge and planning.
`2511.22532` **CoT4AD**, `2607.08375` **WCog-VLA** (Game-theoretic CoT + a generative world model),
`2605.23270` **ChainFlow-VLA** (AR causal factorisation ∪ global diffusion refinement), `2601.05611` **FLARE**
(future-aware latent distilled from a 4.6 B VLM, **no language annotations at all**), `2603.11219` **Senna-2**
(aligning the VLM decision and the E2E policy for *consistent* decision-making — the nearest published neighbour of
our USP), `2607.19194` (dual-process planning with an explicit **verification** step), `2605.10744` **C-CoT**
(counterfactual CoT), `2609.04070` (latent-aligned planning from discrete LM tokens to continuous plans).

**What (a)+(b) settle for us.** Three architectural families exist, and only one of them fits a 300–500 ms tact beside
an existing planner: **(i) the VLM IS the planner** (EMMA, OpenDriveVLA, CoT4AD) — incompatible with REF-C's cascade
and with our parameter budget; **(ii) the VLM guides a numeric planner** (DriveVLM-Dual, Senna, DiffVLA, ReCogDrive,
SimpleVSF `2510.17191`) — the VLM emits a *decision* and the planner emits the *trajectory*; **(iii) the VLM's
reasoning is compressed into tokens that are themselves the interface to control** (XCoT-VLA, LCDrive, LaRA-VLA).
Family (ii) is what the PI describes ("complements our current strategic and tactical layer"), and family (iii) is how
it is made to fit in the tact. **The design in §2 is (ii) in the interface and (iii) in the representation.**

### 1.2 Small-VLM line (≤ ~1 B)

| model (lib key) | composition | parameters | the fact that matters for us | class |
|---|---|---|---|---|
| **SmolVLM-256M** `2504.05299` | SigLIP-B/16 **93 M** + SmolLM2-135M | **256 M** | *"uses less than 1 GB GPU memory during inference"*; 8 k context for the small variants (16 k for 2.2 B) | PRIMARY (abstract, §3) |
| **SmolVLM-500M** `2504.05299` | SigLIP-B/16 93 M + SmolLM2-**360M** | **500 M** | the mid variant explicitly aimed at "moderate-resource edge devices" | PRIMARY (§3) |
| **SmolVLM-2.2B** `2504.05299` | SigLIP-SO400M **428 M** + SmolLM2-1.7B | 2.2 B | above our ceiling; listed for the scaling trend | PRIMARY |
| **InternVL3-1B** `2504.10479` | InternViT-300M + Qwen2.5-0.5B, *native* multimodal pretraining (not a post-hoc adaptation), V2PE positions | ~1 B | the only ≤ 1 B model whose vision and language were pretrained **jointly** | PRIMARY (abstract) |
| **Qwen3.5-0.8B** (family primary `2604.15804`; sizes from the HF card) | hidden 1024, **24 layers = 6 × (3 × Gated-DeltaNet → FFN, 1 × Gated-Attention → FFN)**, FFN 3584, GQA 8 Q / 2 KV × 256, native ctx 262,144, Apache-2.0 | **0.8 B** | the 3:1 **linear-attention hybrid**: decode cost per token is nearly context-independent and the KV cache is ~¼ of a dense model's — exactly what a long standing prefix at 2–3 Hz wants | card = **SECONDARY**; the hybrid-block claim also in `2601.22156`, `2608.14586` |
| **FastVLM** `2412.13303` | FastViTHD hybrid conv/transformer encoder → far fewer vision tokens, TTFT-optimised | encoder-side | the *prefill* lever: fewer vision tokens is the only way to cut time-to-first-token without touching the LM | PRIMARY (abstract) |
| **PaliGemma** `2407.07726` | SigLIP-So400m + Gemma-2B | 3 B | reference point for "small" in 2024; above budget | PRIMARY |
| Moondream, nanoLLaVA | — | ~0.5–1.9 B | **no primary exists** (model cards only) ⇒ **inadmissible as evidence**; not carried into §2 | SECONDARY |

⭐ **SmolVLM's Finding 1 is the single most useful result in this line for us** (`2504.05299` §2.1, Figure 3-left):
*"Compact multimodal models benefit from a balanced encoder–LM parameter allocation, making smaller vision encoders
preferable for efficiency."* Measured: pairing the 428 M SigLIP-SO400M with a 135 M LM **degrades** performance; at
360 M LM the big encoder buys +11.6 % for **+66 % parameters**; only at 1.7 B is a big encoder a 10 % increment.
⇒ **A 0.36–0.8 B LM is the right partner for a ~20 M-parameter REF-C trunk, and adding a second SigLIP-class encoder
next to it would be the exact mistake this paper measures.** Two further transferable findings: extended context
matters more than size for small VLMs (§2.2), and small VLMs prefer **more aggressive** token compression
(pixel-shuffle r = 4, i.e. 16× fewer visual tokens) than large ones use (§2.2, Figure 3-right).

⚠️ **The vision-token budget is where the published systems agree with our trunk by accident.** Alpamayo-R1's default
tokenizer yields **160 tokens per image** at 448 × 280 (`2511.00088` §3.2.1); its multi-camera triplane tokenizer
gets a 7-camera rig to **288 tokens per timestep** (41.1 per image). **REF-C's trunk emits exactly 160 PV tokens**
(`refc.py:1900`). Our vision side is therefore already at the published operating point *for one camera*, at
**zero extra parameters** — which is the whole argument for option (c) in §2.

### 1.3 CoT-for-control and the latency line

**(a) Make the CoT shorter.**

| work | mechanism | measured effect | class |
|---|---|---|---|
| **XCoT-VLA** `2608.10976` | executable semantic-action tokens from a fixed taxonomy, built offline from logged trajectories + scene semantics | **2–6 tokens** replace 40–80; longitudinal ADE **1.645 → 1.323**; lane-change lateral FDE **1.616 → 0.648** *(abstract)* | PRIMARY |
| **FastDriveCoT** `2602.02864` | template CoT → **dependency graph**; independent fields decoded **in parallel in one forward pass** with a custom attention mask; KV cache shared | template CoT is **300–500 tokens**; **3.1–4.1×** speedup, downstream gains preserved *(abstract, §4)* | PRIMARY |
| **DualCoT-VLA** `2603.22280` | parallel visual + linguistic CoT via learnable query tokens, single-step forward reasoning | removes step-by-step decoding and its compounding errors *(abstract)* | PRIMARY |
| **BLUE** `2606.08684` | 0.11 M gate on frozen hidden states: *generate language only on frames that benefit* | language helps on a small fraction of routes, and hurts on others *(abstract)* | PRIMARY |

**(b) Move the CoT out of token space entirely.** `2412.06769` **Coconut** (feed the last hidden state back as the
next input embedding — reasoning without decoding); `2512.10226` **LCDrive** (latent CoT = interleaved **action-proposal
tokens in the model's own action vocabulary** + **world-model tokens** expressing the outcome of the action being
considered — this is the closest published thing to reasoning *over our own anchor fan*); `2602.01166` **LaRA-VLA**
(a curriculum from explicit text/visual CoT **to** latent reasoning, *"up to 90 % latency reduction"* — the standard
recipe for keeping the text at training time and dropping it at inference); `2603.01928` **LaST-VLA** (latent
spatio-temporal CoT with geometric + world-model distillation, because unconstrained latent CoT is
"physics-agnostic"); `2512.22939` **ColaVLA** (cognitive latent reasoning for **hierarchical parallel** planning);
`2503.22020` **CoT-VLA** (visual CoT: predict future frames as goals); `2407.08693` **ECoT** (embodied CoT with
grounded sub-steps).
⚠️ **A latent CoT is not an explanation.** The PI's ask is explicitly *"chain of thought as text as explanation of the
behavior"*. Latent reasoning buys latency and loses the deliverable. §2 resolves this by **separating the two rates**:
an executable/latent path at the control rate and a text rendering at a lower one.

**(c) Make each token cheaper.** `2211.17192` speculative decoding; `2401.15077` **EAGLE** (feature-level
autoregression); `2403.06764` **FastV** (prune visual tokens after layer 2); `2606.31160` reasoning-aware speculative
decoding *for driving VLAs specifically*; `2608.12932` **FlashDrive**; `2606.14010` **RT-VLA** (real-time via
knowledge distillation); `2608.14586` (block-layer **parallel** inference on hybrid linear/softmax architectures —
directly relevant to a Qwen3.5-style 3:1 hybrid); `2608.01035` **WAM-Diff2** (AR→diffusion distillation).
⭐ And the deployment-proven combination is already in `2402.12289` **Table 9, on OrinX**: q4f16 → EAGLE + post-quant
fine-tune → **plus a 1,024-word vocabulary** = **4.33×** and a **0.071 s** decode for ~40 tokens. Our CoT vocabulary is
a closed driving taxonomy; the 1,024-word shrink is not a hack for us, it is the natural vocabulary.

**(d) Know where the time actually goes on edge silicon.** `2607.09520` *"Seeing is Free, Speaking is Not"* — the
energy/latency bottleneck of edge VLM inference is **decode**, not vision; `2606.27906` *"Phase Matters"* — prefill and
decode are different hardware regimes on a mobile SoC; `2607.12659` **Jetson-PI** — onboard real-time control by
**foresight-aligned asynchronous inference** on Jetson, i.e. the slow model runs off-cycle and its output is aligned
forward in time. `2607.08029` (component-wise quantisation on Jetson Orin NX/AGX): **INT4 saves VRAM but can SLOW
generation** through dequantisation overhead, and sensitivity is governed by the **structural paradigm (MoE vs dense),
not scale**; SigLIP encoders incur disproportionate INT8 latency on Jetson Ampere. `2606.06527` (NVFP4 ablation):
block size B = 16 is the practical accuracy/storage trade-off (4.51 bits/input at N = 4096), and FP8/FP16 weights buy
only modest gains over FP4 weights.
⛔ **Consequence for our plan: "quantise it to INT4 and it gets faster" is refuted on Jetson by `2607.08029`.** The
quantisation format is a **measurement item on Thor**, not an assumption (plan WP-0).

### 1.4 Grounding / faithfulness line — and why this is the USP

| work | what it measures | headline numbers | class |
|---|---|---|---|
| **VLADriveBench** `2606.12706` | **observational** metrics (mentioning, hallucination, contradiction, action alignment) **+ a CoT INTERVENTION protocol** | the two views **diverge sharply**: ORION scores highest observationally yet its CoT is **epiphenomenal**; Alpamayo v1.5 scores lower yet its CoT is **strongly causal**, with visual salience gating CoT influence *(abstract)* | PRIMARY |
| **Is VLA Reasoning Faithful?** `2605.17268` | 300 Alpamayo-R1-10B inferences over 100 **PhysicalAI-AV** scenarios — *our corpus* | reasoning fidelity **42.5 %**; **94 missed pedestrians** in a third of pedestrian-relevant scenes; **97.7 % trajectory fragility** under mild visual perturbation; mean reasoning-action consistency **48.3 %**, with **37.9 %** of stop-claimed cases continuing instead *(abstract)* | PRIMARY |
| **Drive the Thoughts** `2608.29583` | DriveAlignBench: 150 manually-labelled CoT–trajectory pairs from **Alpamayo 1.5** | **33.3 % of CoTs unreliable**; among reliable ones **74 % consistent** with the trajectory; best runtime monitor **F1 0.75** *(abstract)* | PRIMARY |
| **Deferred Exposure** `2608.01755` | what happens when the CoT **teacher is shown the logged GT future** | **trajectory anchoring bias**: the teacher *rationalises the revealed outcome* instead of inferring a decision from evidence ⇒ less causally faithful CoTs and **substantially more severe hallucinations**, worst in causally hard scenes *(abstract)* | PRIMARY |
| **DriveBench** `2501.04003` | VLM reliability across 17 settings incl. **corrupted and text-only** inputs (19,200 frames) | VLMs *"often generate plausible responses derived from general knowledge or textual cues rather than true visual grounding"* — concealed by dataset imbalance *(abstract)* | PRIMARY |
| **Lanham et al.** `2307.13702` | intervention tests on the CoT itself — truncate it, paraphrase it, **insert a mistake**, and see whether the answer moves | conditioning on the CoT varies hugely by task; **larger models produce LESS faithful reasoning** | PRIMARY |
| **Turpin et al.** `2305.04388` | biasing features the model never mentions | CoT explanations **systematically misrepresent** the true cause of the prediction | PRIMARY |
| **RL-finetuned VLM CoT consistency** `2602.12506` | robustness of CoT under misleading captions / wrong CoT traces | RL fine-tuning raises accuracy but open models stay vulnerable — an **accuracy–faithfulness trade-off**, so `2511.00088`'s "+37 % consistency from RL" must not be read as *solved* | PRIMARY |
| **Neuro-Symbolic Drive** `2606.23938` | rule-grounded faithful reasoning for driving VLAs | symbolic grounding of the rationale as the mechanism, not a metric | PRIMARY |

⛔⛔ **THE FINDING THAT CHANGES OUR DISTILLATION PLAN.** `2511.00088` §5 states that the CoC **auto-labelling pipeline
provides the teacher with auxiliary signals "including the ego vehicle's trajectory, dynamic states, and meta
actions"**, with video sampled at 2 Hz. `2608.01755` measures that exposing a teacher to the logged future **induces
trajectory anchoring bias** — rationalisation instead of inference. Our banked corpus (§0.2) was produced by
`nvidia/Alpamayo2-Super`, ONE draw per clip, and it carries a `meta_action` triplet and a **predicted** trajectory —
so we do **not** know from our side whether the generating prompt exposed the GT future.
⇒ **This is a work item with a cheap discriminating test, not an assumption** (plan §WP-2, `H-VLA-6`): if the CoT is
*anchored*, its causal claims should track the **executed/labelled** manoeuvre even where the scene evidence
contradicts it, and should track the model's own **predicted** trajectory where the two differ; if it is *inferential*,
the residual should be symmetric. The 41.7 % CoT-vs-geometry lateral disagreement already MEASURED on our blob
(D-LAT-AGREE) is the population to run it on.
⚠️ And the two audits above are **on our own corpus and our own teacher** (`2605.17268` is 100 PhysicalAI-AV scenarios;
`2608.29583` is Alpamayo 1.5). We are not extrapolating from a different domain: **the teacher we are being asked to
distil has a published 42.5 % reasoning fidelity and 48.3 % reasoning-action consistency.** That is the quantitative
justification for §3 — the module must be trained against *our* labels and *our* detections, with the teacher supplying
form, and it must be **measured with an intervention protocol**, because `2606.12706` proves observational agreement
and causal influence come apart.

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
