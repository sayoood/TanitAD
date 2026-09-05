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

### 2.0 The Thor latency model every option is priced against

⛔ **No datacentre number appears in this section.** Two facts anchor it, both fetched from NVIDIA's own module pages
on 2026-09-05 (`PUBLISHED-SECONDARY` — a vendor spec page, not a paper):

| | Jetson AGX Thor (T5000) | Jetson AGX Orin (64 GB) | ratio |
|---|---|---|---|
| AI compute | **2070 TFLOPS FP4 (sparse)** | 275 TOPS INT8 class | ~7.5× (NVIDIA's own claim) |
| memory | 128 GB 256-bit LPDDR5X | 64 GB 256-bit LPDDR5X | — |
| **memory bandwidth** | **273 GB/s** | **204.8 GB/s** | **1.33×** |

⭐⭐ **THE CENTRAL LATENCY FACT: autoregressive decode is WEIGHT-BANDWIDTH-BOUND, so Thor's 7.5× compute advantage
does NOT transfer to it — the transferable ratio is 1.33×.** Every design decision below follows from this one line.
Prefill and the vision trunk are compute-bound and *do* get the 7.5×; the CoT does not. This is the same family as the
`step_s`/`df` scope traps: a true number (7.5×) quoted where it does not apply.

**Calibrating the efficiency factor from the ONE published embedded measurement.** `2402.12289` Table 6 reports
Qwen1.8B on OrinX at **79.6 tok/s (12.56 ms/token)** with a *"Model Size"* column of **3.7 GB** — which is the FP16
weight size (1.8 B × 2 B). At FP16 the bandwidth floor would be 3.7 / 204.8 = **18.1 ms/token**, i.e. *slower than
what was measured*, which is impossible. ⇒ **The table's own numbers prove the deployed weights are quantised**
(consistent with its caption *"after quantization"* and with Table 9's `q4f16`). At 4-bit (~0.95 GB) the floor is
**4.64 ms/token**, so the achieved efficiency is **η = 4.64 / 12.56 = 0.37** of the bandwidth roof. The same
arithmetic on MobileLLaMA-1.4B (2.5 GB FP16 → ~0.63 GB at 4-bit, floor 3.08 ms, measured 8.52 ms) gives **η = 0.36**.
Two independent rows, the same η — the derivation is at least self-consistent.

**The resulting per-token cost model (ESTIMATED — derivation shown, `η = 0.37`, effective 101 GB/s on Thor):**

| decoder | bf16 (2 B/param) | FP8 (1 B/param) | NVFP4 (0.5 B/param) |
|---|---|---|---|
| **0.8 B** (Qwen3.5-0.8B class) | 15.8 ms/tok | **7.9 ms/tok** | **4.0 ms/tok** |
| **0.5 B** (SmolVLM-500M's LM class + heads) | 9.9 ms/tok | 4.9 ms/tok | 2.5 ms/tok |
| **0.36 B** (SmolLM2-360M) | 7.1 ms/tok | **3.6 ms/tok** | 1.8 ms/tok |

⚠️ Every cell is an **ESTIMATE from a spec sheet plus one published embedded calibration**, and it ignores the hybrid
linear-attention advantage (`2604.15804`: a 3:1 Gated-DeltaNet stack reads a much smaller KV cache, which raises η at
long context) and the INT4 dequantisation penalty MEASURED on Jetson by `2607.08029` (which *lowers* it). **These two
push in opposite directions and neither is quantified for Thor. WP-0 replaces this whole table with a measurement**
using `torch.cuda.max_memory_allocated()` — the only admissible memory probe on Thor — and a real token-rate timing.

**The tact budget (ESTIMATED), for a 400 ms tact:**

| stage | cost | basis |
|---|---|---|
| REF-C trunk + cascade + 117-anchor decoder | **UNMEASURED** — allowance 60 ms | ⛔ the first item of WP-0; there is no Thor inference number for REF-C anywhere in the programme |
| projector `pooled` (160 tokens) → LM embedding | < 1 ms | one `Linear`, 160 × 512 × 1024 ≈ 84 MFLOP |
| prefill ~230 tokens (160 vision + ~40 state/goal + ~30 query) | ~15 ms | Orin measured 4,709 tok/s prefill for Qwen1.8B (`2402.12289` Table 6) = 49 ms for 230 tokens; prefill is compute-bound, so Thor's 7.5× applies — 15 ms is deliberately conservative |
| **decode — what is left** | **~325 ms** | ⇒ **41 tokens at 0.8 B FP8** · **81 at 0.8 B NVFP4** · **90 at 0.36 B FP8** |

⭐ **41 tokens is exactly Alpamayo-R1's reasoning budget** (`2511.00088` Table 14: 70 ms for 40 tokens). Two
independent routes to the same number is a good sign — and it is **an order of magnitude below FastDriveCoT's
300–500-token template CoT** (`2602.02864`). ⇒ **A free-form prose CoT cannot be emitted every tact on Thor. This is a
hard constraint, and it is what makes §2.c's two-rate design necessary rather than merely tidy.**

### 2.a Pretrained small VLM + projector from `pooled` / `z_tac` / `ctx`

**Shape.** Take SmolVLM-500M (`2504.05299`) or InternVL3-1B (`2504.10479`) whole — its own SigLIP/InternViT encoder
included — feed it the camera image on its own path, and *additionally* project `pooled`, `z_tac` and `ctx` into its
text-embedding space as extra prefix tokens (the OpenDriveVLA recipe, `2503.23463`, which projects 2-D and 3-D
structured tokens into one semantic space).

| | |
|---|---|
| **params** | 500 M (SmolVLM-500M) or ~1 B (InternVL3-1B) + projector ~1.6 M. **At the ceiling, not below it.** |
| **Thor tact** | 0.5 B FP8 ≈ 4.9 ms/tok, but the **second vision encoder is an extra prefill and an extra weight stream**: SigLIP-B/16 at 512² emits **1,024 tokens before pixel-shuffle** (`2504.05299` §2.2); at r = 4 that is 64. Prefill roughly doubles. |
| **training data** | it already speaks; only alignment + our CoT are needed. **Lowest data risk of the three.** |
| **goal interface** | text in, text out; goals must be parsed out of strings, or extra heads bolted on. |
| ⛔ **why it is the runner-up, not the choice** | It **violates the PI's "sharing the same embedding space as refc"** in the only sense that has teeth: two vision encoders means two scene representations, and nothing forces them to agree — the explanation can then be grounded in the VLM's *own* view of a scene the planner never saw. It is also the exact configuration `2504.05299` **Finding 1** measures as wasteful (a 428 M encoder next to a small LM buys +11.6 % for +66 % params), and `2607.08029` MEASURED that SigLIP encoders take a disproportionate INT8 latency hit on Jetson. |

### 2.b From-scratch small decoder on our corpus + distilled CoT

**Shape.** A 30–80 M transformer decoder trained from zero on our own `(pooled, ego, nav) → sentence` pairs.

| | |
|---|---|
| **params** | 30–80 M — comfortably the smallest. |
| **Thor tact** | ~0.7–1.6 ms/token at FP8: **essentially free**; hundreds of tokens per tact. |
| **training data** | ⛔ **this is where it dies.** §0.2 MEASURED our corpus: **4,729 one-sentence CoTs, ~47 k words in total**, one instant per clip. That is roughly **five orders of magnitude** below what `2502.02737` (SmolLM2) needs for a 360 M model. A from-scratch decoder trained on 47 k words can reproduce the *template* and nothing else — no question answering, no unseen-object naming, no world knowledge, which are three of the four things the PI asked for. |
| **verdict** | ⛔ **REFUSED as the language module** — but ⭐ **KEPT as the mandatory FLOOR ARM.** CLAUDE.md's probe rule ("a learned representation that does not beat raw input has added nothing") applies verbatim: a pretrained LM that does not beat a from-scratch template decoder on explanation faithfulness **has bought us nothing but parameters**. This arm is cheap and it is the control that makes the whole comparison admissible (`H-VLA-2`). |

### 2.c ⭐ CHOSEN — hybrid: frozen small LM + LoRA, **REF-C's trunk as the ONLY vision encoder**

**Shape (`TanitLang`).** No second ViT anywhere. The REF-C trunk's **160 perspective-view tokens** are the vision
input, projected once into the LM's embedding space; a frozen pretrained LM supplies language; LoRA adapts it; and a
small set of heads writes back into REF-C's own goal nodes.

```
  REF-C trunk (frozen, already running)  ──► pooled  [160 × feat_dim]
                                                │
                          proj_in: Linear(feat_dim → d_lm)          ─┐
  ctx [256] · z_tac [512] · g_str [3] · v0 · nav one-hot [3+1]      ─┤  prefix tokens
                          state_proj: MLP(→ d_lm), 4 tokens          │
  question / system prompt (tokenised text)                         ─┘
                                                │
                                   ┌────────────▼────────────┐
                                   │  frozen LM + LoRA r=16  │   0.36 B or 0.8 B
                                   └────────────┬────────────┘
                       ┌────────────────────────┼────────────────────────┐
             XCoT head (2–6 exec. tokens)   goal heads (write-back)   text head (LM head)
                  every tact                   every tact              ≤ 1 Hz / on demand
```

**Parameter budget (ESTIMATED, arithmetic shown):**

| part | count |
|---|---|
| frozen LM — SmolLM2-360M (`2502.02737`) *or* Qwen3.5-0.8B | **362 M** or **800 M** |
| `proj_in`: `Linear(feat_dim=512 → d_lm)` + 1 hidden layer | 512·1024 + 1024·1024 ≈ **1.6 M** |
| `state_proj`: MLP over (`ctx` 256, `z_tac` 512, `g_str` 3, `v0`, nav 4) → 4 prefix tokens | ≈ **0.8 M** |
| LoRA r = 16 on q/k/v/o + FFN, 24 layers, hidden 1024, FFN 3584 | ≈ **3.5 M** |
| write-back heads: E19 strategic (23) + v7.2 lat/lon (8+8) + geometric goal point (3) + XCoT vocabulary (~64) | 1024 × 106 ≈ **0.11 M** |
| **total resident** | **≈ 368 M** (SmolLM2-360M) or **≈ 806 M** (Qwen3.5-0.8B) — **both ≤ 1 B, the first well below it** |
| **total trainable** | **≈ 6 M** — 0.7 % of the 0.8 B option; a dev-box-scale fine-tune, not a cluster job |

**Why the trunk-as-only-encoder is the right call and not merely the cheap one.**
1. It is the **only** reading of *"sharing the same embedding space as refc"* with a testable consequence: the language
   module and the planner are looking at **the same tensor**, so a disagreement between explanation and plan is a
   disagreement about *reasoning*, never about *perception*. With two encoders that distinction is unrecoverable.
2. `2504.05299` **Finding 1** says a small LM prefers a small encoder — and our trunk is ~20 M, an excellent partner
   for a 0.36 B LM. Adding SigLIP-SO400M (428 M) would more than double the module for a measured +11.6 %.
3. Our trunk already emits **160 tokens**, which is exactly Alpamayo-R1's per-image tokenizer output
   (`2511.00088` §3.2.1). **Zero additional vision parameters, zero additional vision latency** — the trunk runs for
   REF-C anyway.
4. It is the configuration the faithfulness literature demands: `2501.04003` (DriveBench) measures VLMs answering from
   *textual cues rather than visual grounding*; the ablation that detects it is **corrupt the vision input and see
   whether the text changes**. With one shared encoder that intervention is a single tensor swap (§3.4).

**The two-rate emission — how the 41-token budget is actually spent.** ⛔ A prose CoT every tact is arithmetically
impossible (§2.0). So the module runs **two clocks**:

| clock | what is emitted | tokens | cost at 0.36 B FP8 |
|---|---|---|---|
| **every tact (300–500 ms)** | **XCoT-style executable tokens** (`2608.10976`) — 2–6 from a closed driving taxonomy — plus the goal write-back heads, which are **not decoded at all** (one forward pass, argmax/regression, ~0 tokens) | **2–6** | **7–22 ms** |
| **≤ 1 Hz, or on demand ("why did you brake?")** | the prose explanation, from the *same* KV cache, rendered with FastDriveCoT-style field parallelism (`2602.02864`, 3.1–4.1×) over a template with a small vocabulary (`2402.12289` Table 9: 4.33× from a 1,024-word vocabulary) | 40–80 | 60–120 ms, **off the control path** |

⭐ This is `2402.12289`'s deployed dual-clock architecture (OrinX-1 fast / OrinX-2 VLM at 410 ms, asynchronous) and
`2607.12659`'s foresight-aligned asynchronous inference, applied inside one module instead of across two chips.
⭐ And it is why the executable-token path is the **primary** interface: `2606.08684` (BLUE) MEASURED that language
helps on only a small fraction of frames — so a per-frame prose budget is spent mostly on frames that do not need it.

**The goal interface (both directions).** *"Process AND emit strategic and tactical goals"* means the projector is
bidirectional, and both directions bind to nodes that already exist in `refc_v3.py`:

| direction | node | mechanism |
|---|---|---|
| **reads** | `nav` (3-way v7.2 token + `nav_known`), `g_str` (3), `ctx` (256), `z_tac` (512), `pooled` (160 tokens), `v0` | prefix tokens via `proj_in` / `state_proj` |
| **emits — strategic** | E19's surface: `g_str_geo` (3), `p_man` (4), `t_bin` (6) over `STRATEGIC_S = (8, 30) s`, `p_man2` (4), `t_bin2` (6) | a 23-wide head on the LM's last hidden state, **zero-init**, summed into `str_goal_head`'s output exactly as the existing FiLM edges are gated |
| **emits — tactical** | the **v7.2 8-way factored** `lat_logits_tac` / `lon_logits_tac`, and a **geometric goal point** | zero-init residual heads; the geometric goal point is the lever the literature measures as the big one (§0: categorical command +0.2 PDMS vs goal point +4.7) |
| **emits — explanation** | XCoT tokens every tact; prose at the slow clock | LM head over a restricted vocabulary |

⛔ **Every emitted edge is zero-init and individually gated**, matching the seven existing cascade edges
(`DESIGN_REFCV4_NAV_WIRING.md` §2.2, +7,682 params, all zero-init). A language module that cannot be switched off
edge-by-edge cannot be ablated, and an un-ablatable module cannot be attributed.
⛔ **The goal heads may NOT read the situation classifier's output** (PI 2026-08-03) — see §5.

### 2.d Comparison

| | **(a)** pretrained VLM + projector | **(b)** from scratch | **(c)** ⭐ frozen LM + LoRA, trunk-only vision |
|---|---|---|---|
| resident params | 0.5–1.0 B | 0.03–0.08 B | **0.37 B** / 0.81 B |
| trainable params | 0.5–1.0 B (or LoRA) | all | **~6 M** |
| vision encoders | **two** | one (ours) | **one (ours)** |
| shares REF-C's space | partly (extra tokens) | yes | **yes, by construction** |
| tokens per 400 ms tact | ~30 (0.5 B FP8, doubled prefill) | hundreds | **90 (0.36 B FP8)** — spends 2–6 |
| language competence | **high** | ⛔ none (47 k words) | **high** (frozen pretrained LM) |
| answers open questions | yes | no | yes |
| faithfulness intervention | hard (two views of the scene) | easy | **easy (one tensor)** |
| risk | parameter ceiling, encoder disagreement | data poverty | LoRA under-capacity for a new modality; projector cold-start |
| **role** | runner-up; the fallback if (c)'s projector fails to align | **mandatory floor arm** | **CHOSEN** |

⚠️ **(c)'s real risk, stated plainly:** a frozen LM has never seen a 160-token REF-C scene representation, and a
~6 M-parameter adapter may be too small a bridge. The mitigation is a **staged unfreeze** (projector-only → +LoRA →
+top-k blocks) with the parameter count re-checked at every stage against the 1 B ceiling, and `H-VLA-1` is exactly the
tiny-rig arm that answers it before any GPU-day is spent.

## 3. The grounding / consistency mechanism — the USP

### 3.0 The consistency paradox, and the two modes that resolve it

⛔ **State the trap before the mechanism.** *"Train the text to agree with the plan"* is the obvious answer and it is
**self-defeating**: a CoT optimised to match the plan is trivially consistent, carries no independent information, and
— fatally — **can never flag a wrong plan**, which is the one thing an explanation channel is for. This is the same
shape as the programme's own echo defects: refcv3's route head scored **1.0000** by reproducing its own nav input
(369/369 and 81/81), and the S-curve "lateral skill" was **97.9 % open-loop, 0.0 % hold-action**. A consistency loss
without an anti-echo control manufactures exactly that number again, in a new costume.

⭐ **The resolution: our causal claim is deliberately weaker than the literature's, and therefore testable.**
VLADriveBench (`2606.12706`) asks whether the CoT *causes* the action, and finds ORION's does not. We do not want ours
to cause the action by default — the 2.15 M-parameter cascade is the authority, and a 368 M language module silently
overriding it would dissolve the hierarchy that is the programme's thesis. So the module ships in **two modes**, and
the mode is a **gate**, not a rewrite:

| mode | write-back gates | what the module is | what must be measured | first deployable |
|---|---|---|---|---|
| **M — monitor** | **closed** (all zero-init heads gated to 0) | a **runtime consistency monitor** in the sense of `2608.29583` — it explains and it flags, it does not steer | F1–F6 below; **F7 (causal influence) must read EXACTLY 0.0000**, because there is no path | ⭐ **yes — zero planner risk** |
| **I — influence** | opened **one edge at a time** | the CoT is causally in the loop through `g_str` / `g_tac` / the geometric goal point | F7 becomes non-zero *and must earn its edge* with a T1 four-family delta and the echo gate | only after M passes |

**The claim mode M actually supports** — and it is the one to write in the paper — is: *the explanation and the plan
are both produced from the same scene tensor, and a divergence between them is a detectable, measurable fault.* Not
*"the CoT is the reason"*. `2307.13702` (Lanham) and `2305.04388` (Turpin) are the reason to be this careful: stated
reasoning routinely misrepresents the true cause, and larger models are **less** faithful, not more.

### 3.1 Consistency between the emitted text goal and `g_tac` / the selected anchor

**The interface that makes consistency mechanical rather than rhetorical** is the executable-token vocabulary
(`2608.10976`): every emitted token has *executable semantics*, so it can be compared to kinematics rather than to
another string. Our vocabulary is not invented — it is the **v7.2 factored 8-way lat/lon set**
(`stack/tanitad/data/vocab_v7.py`) plus a small closed set of rule/interaction modifiers, in XCoT's deterministic
order (primary lateral/nav manoeuvre → longitudinal adjustment → rule/safety modifier).

| loss | form | against what | why it is not an echo |
|---|---|---|---|
| **L_tac** | CE(XCoT primary lat/lon token, v7.2 label) | the **label**, not the model's own head | the target is the corpus label; matching the model's own `lat_logits_tac` would be self-distillation and would read 1.0 by construction |
| **L_agree** | KL(XCoT lat/lon distribution ‖ `lat_logits_tac` / `lon_logits_tac`), **weight ramped from 0 and capped**, and reported as a *measured* rate, not driven to 1 | the cascade's tactical heads | the cap is the mechanism: consistency is a **calibration target, not a constraint**. A model at 100 % agreement is refused — it has become the echo |
| **L_goal** | Smooth-L1(emitted geometric goal point, label goal point) + CE(E19 `p_man`, `t_bin`) | the v7.2 `manoeuvre_sequence` labels over `STRATEGIC_S = (8, 30) s` | the 30 s label is a **training target and never an input** (`refused_edges`, invariant I1) |
| **L_ref** | CE(referent pointer, `obstacle.offline` critical agent \| NULL) | §3.3 | 853 clips carry an explicit `type: none` — a real negative class, so NULL is learnable and not a degenerate default |
| **L_cot** | token CE on the prose head, **slow clock only** | the Alpamayo sentence (form) + our template (content) | §0.2: the teacher supplies the *kind* of sentence; the content comes from our labels |

⚠️ **`L_agree` is the one that needs a deliberate-regression arm** (per `TanitAD_ValidateAIDesign`): an arm trained
with `L_agree` at 10× weight must **FAIL** the F5 scene-sensitivity gate. If it passes, the gate is not measuring
grounding and the whole family is invalid.

### 3.2 Verification against the fan — "name a manoeuvre the plan actually contains"

REF-C emits **117 anchors** with confidences and picks one (`sel_score → sel_score_v3 → argmax`). Every anchor is a
constant-`(a_lon, a_lat)` control rolled from the measured `v0`, so **every anchor has a computable v7.2 class** — the
machinery already exists (`derive_man5_logprobs` on the 5-way surface, `refc.py:2290-2305`; the factored 8-way set in
`vocab_v7.py`). Two verifications follow directly, and both are free at eval time because the fan is already dumped:

- **FAN-SUPPORT@k** — the fraction of windows where the explanation's primary manoeuvre lies in the class set spanned
  by the **top-k anchors by confidence**. `k = 117` must read **1.0000** exactly (every class is present) — that is a
  control that must read a known value, and if it does not, the class mapping is broken, not the model.
  `k = 1` is the strict form and coincides with PLAN-NAME below.
- **PLAN-NAME accuracy** — the explanation's primary manoeuvre equals the **selected** anchor's class.

⭐ Why this is stronger than a text-similarity score: it is a **statement about the plan the car will execute**, in the
plan's own vocabulary, computed from the same tensor. `2608.29583`'s monitor reaches **F1 0.75** doing an approximate
version of this on Alpamayo 1.5 with a *learned* judge; ours is exact and rule-based because our fan is enumerable.

### 3.3 Grounding tokens — the referent must POINT, not describe

⛔ **The single defect the corpus already exhibits is a hallucinated referent** — the cyclist invented through a
68–82° junction turn (D-DATA-COT-HALLUC), and `2605.17268` measures **94 missed pedestrians** in a third of
pedestrian-relevant PhysicalAI-AV scenes. A string cannot be checked; a pointer can.

**Mechanism.** The refcv5 plan's `E-AGT-1` already produces **agent tokens from `obstacle.offline`** (3-D tracks on
**97.44 %** of the corpus, 10 dynamic classes over 87,481 cuboids). The language module emits, for each referent slot,
a **softmax over {agent tokens} ∪ {NULL}** — so *"the lead vehicle"* is emitted as `REF(k)` and rendered to text only
afterwards. Consequences:

1. **A referent that does not exist cannot be emitted.** The hallucinated cyclist is structurally impossible, not
   merely penalised.
2. **The negative class is real.** `critical_components_analysis` carries `type: none` on **853** clips (§0.2), so
   NULL has genuine supervision and the head cannot collapse to "always point at something".
3. **The 2-D boxes are the cross-modal check.** `grounding_via_vqa` gives boxes on **4,728** rows (Car 2,292 ·
   Pedestrian 1,079 · traffic light 889) — the only channel that ties a CoT token to image space, and therefore the
   only way to audit the *teacher's* referents before distilling them.
4. ⛔ **`obstacle.offline` is a TRAIN-TIME label only.** Labels may use ego and other privileged channels (PI
   2026-08-03); at inference the referent pointer must resolve against **agent tokens produced by our trunk**, never
   against the offline track file. This is invariant I3 and it is the same test as the vision-only rule.

**Rule-based contradiction detection** (the `2606.23938` neuro-symbolic idea, made cheap by executable tokens). Each
token carries a checkable predicate over the **selected anchor's rolled control**, so a violation is a fact, not a
judgement:

| token | predicate on the plan | fires when |
|---|---|---|
| `DECELERATE` / `STOP_POINT` | selected anchor `a_lon < −τ_dec` ; terminal speed ≈ 0 within the horizon for `STOP_POINT` | the text says brake and the plan accelerates — `2605.17268` measures this exact failure at **37.9 % of stop-claimed cases** |
| `KEEP_SPEED` | \|Δv\| over the horizon below τ_keep | — |
| `NAV_LEFT/RIGHT_LANE_CHANGE` | cumulative lateral offset > ½ lane width **and** nav token consistent | — |
| `HAZARD_YIELD` / `GAP_TARGET` | referent pointer ≠ NULL | an interaction claim with nobody to interact with |
| `RED_LIGHT_HOLD` | referent class ∈ {traffic light} | — |

⇒ **CONTRADICTION RATE**, reported **per rule with its own n** (a pooled rate hides which rule fires, and CLAUDE.md's
"never pooled" rule applies to this family too).

### 3.4 ⭐ EXPLANATION FAITHFULNESS — the FIFTH metric family, with controls that must read known values

**This is the deliverable that makes the USP a claim rather than a slogan.** It sits **beside** the binding four
(LONGITUDINAL · LATERAL · TACTICAL · STRATEGIC), never inside them, never pooled with them, and it carries the same
apparatus: the **paired episode-cluster bootstrap** over the 141 eval clips (`taniteval/ci.py`), never
`overlapping_holdout_se`; a **T-tier stamp** on every row; and `n` **and** `d` printed in the table (the probe rule).

| id | metric | definition | reported with |
|---|---|---|---|
| **F1** | **plan consistency** | rate(emitted primary manoeuvre = selected anchor's v7.2 class) | + the class prior; + `Δ_plan-shuffle` |
| **F2** | **fan support@k** | primary manoeuvre ∈ classes of the top-k anchors | k ∈ {1, 3, 10, 117}; **k = 117 must read 1.0000** |
| **F3** | **referent grounding** | precision / recall / NULL-accuracy of `REF(k)` against the critical-agent label | per class; explicit-none subset separately |
| **F4** | **contradiction rate** | per-rule violation rate over §3.3 | per rule, with n |
| **F5** | **scene sensitivity** | 1 − agreement(explanation \| scene tokens, explanation \| **shuffled** scene tokens) | ⭐ the anti-echo gate |
| **F6** | **nav-echo quotient** | accuracy of the emitted strategic manoeuvre **conditioned on nav** — i.e. what the explanation adds beyond the nav token — plus `Δ_nav-shuffle` | ⛔ a bare accuracy here is inadmissible |
| **F7** | **causal influence** | change in the selected anchor when an XCoT token is forced to a different value (`2606.12706`'s intervention) | **must read exactly 0.0000 in mode M** |

**The control rig — five controls, each of which must read a KNOWN value.** *(This exists because on 2026-08-22 four
distinct probe failures in one afternoon each produced a confident publishable-looking number, and every one was
caught only by a control reading the same value as the thing being measured.)*

| control | construction | the value it MUST read | what its failure proves |
|---|---|---|---|
| **C-A constant** | always emit the modal token sequence | F1 = the class prior, **F5 = 0.0000 exactly** | if C-A scores like the model, F1 is measuring the prior |
| **C-B plan-echo** | explanation computed from the **plan alone**, scene tokens withheld | F1 ≈ 1.0, **F5 = 0.0000** | ⭐ this is the control that gives F5 its meaning: it is the only thing separating *grounded* from *echo* |
| **C-C scene-only** | plan/goal prefix tokens withheld | F1 < full model | if equal, the plan input is doing nothing and mode I can never work |
| **C-D nav-only** | only the nav token and `v0` as input | F6 ≈ 0 beyond nav | ⛔ this is refcv3's 141/141 route bijection, pre-armed |
| **C-E floor** | the §2.b from-scratch template decoder, same targets, no pretrained LM | the pretrained LM **must beat it** | a learned representation that does not beat the cheap floor has added nothing |

**The two interventions the PI's brief names, made precise:**

- *"shuffle the scene ⇒ the explanation must change"* → **F5 with the C-B control.** Shuffle `pooled` across windows
  **within an episode** (so the ego state and nav are unchanged and only the scene moves); the explanation must change
  on a majority of windows. A module reading only ego/nav reads F5 ≈ 0 — which is precisely `2501.04003`'s finding that
  VLMs answer from textual cues rather than visual grounding, turned into a gate.
- *"shuffle the plan ⇒ consistency must drop"* → **F1 with `Δ_plan-shuffle`.** Replace the selected anchor with one
  drawn from another window; F1 must fall to the class prior. If it does not, F1 was never measuring agreement.

⚠️ **Two openly-stated weaknesses of this family, so nobody inherits them as folklore.**
1. **F1 and F5 trade off by construction.** A perfectly plan-consistent module is at risk on F5; a perfectly
   scene-sensitive one is at risk on F1. ⇒ they are reported **as a pair, never as a mean**, and the target is a
   *region* (F1 well above prior **and** F5 well above 0), not a maximum on either.
2. **F3 is only as good as its labels**, and the labels come from a teacher measured at **42.5 %** reasoning fidelity
   (`2605.17268`). ⇒ F3's *label* set is `obstacle.offline` (geometry, ours), **not** the Alpamayo referents; the
   Alpamayo `grounding_via_vqa` boxes are used to *audit the teacher*, never as ground truth.

**Integration.** The four families are enforced by the criteria registry
(`products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.6.0, INHERITED from `REFCV5_DESIGN_PLAN.md` §2 I5). A fifth family
is a **registry version bump plus checker rules**, not a paragraph in a report — CLAUDE.md's own C126 lesson is that
"no prose correction can reach a glob". ⇒ plan **WP-6**: add `explanation_faithfulness` to the registry with F1–F7 and
the five controls as *required* keys, so an eval that omits them is a **violation**, not an omission.

## 4. Own hypotheses — `H-VLA-*` (validation on the tiny rig / dev box, both outcomes committed)

Registered in `Project Steering/GOALS_AND_CLAIMS.md` in the same turn as this section. Every row names its rig, its
PASS **and** its FAIL, and **what changes on FAIL** — a hypothesis whose refutation changes nothing is not a
hypothesis. All eight are 0-GPU or dev-box-scale except `H-VLA-3` and `H-VLA-8`, which need Thor and no training.

| id | hypothesis | rig | PASS | FAIL — and what changes |
|---|---|---|---|---|
| **H-VLA-1** | A **~6 M-parameter bridge** (`proj_in` + LoRA r=16) is enough for a **frozen** SmolLM2-360M to read REF-C's 160 PV tokens: predicting the v7.2 lat/lon class *through the LM* beats a **linear probe on `pooled` directly**. | dev box (RTX 4060, 8.6 GB); ~500 clips of banked `pooled`; projector-only then +LoRA; λ and any PCA basis fit on the FIT split only; `n` and `d` printed | macro-F1(LM path) > macro-F1(linear probe on `pooled`) with a **paired** episode-cluster CI excluding 0 | the bridge is too small ⇒ **staged unfreeze** (top-k blocks), and if that also fails, **§2.a becomes the design** and the ≤ 1 B ceiling is re-checked. ⚠️ A *negative* here is only a statement about **this** bridge — a linear-probe-class negative is not a learnability negative (CLAUDE.md). |
| **H-VLA-2** | A **pretrained** LM beats the **from-scratch template decoder** (§2.b) jointly on F1/F3/F5. | both arms on the same 141 eval clips, mode M | pretrained wins on **all three**, paired CI excluding 0 on each | the language competence is not being used ⇒ **ship the template generator instead** (30–80 M, ~free on Thor) and retire the ≥ 300 M module. This is a genuinely acceptable outcome and would *save* the programme parameters. |
| **H-VLA-3** | On Thor, **0.36 B FP8** emits **≥ 6 tokens + the non-decoded goal heads in ≤ 150 ms**, and REF-C + module together stay **≤ 500 ms** at batch 1. | `tanitad-thor-wifi`, inference only, no training on the box; **`torch.cuda.max_memory_allocated()` only** for memory; wall-clock at **batch 1** | both budgets met | drop to **NVFP4** (2× cheaper per the §2.0 model) and/or 0.2 B; if still short, the text head moves **entirely** off-cycle and only the executable tokens run per tact. ⚠️ `2607.08029` MEASURED that INT4 can *slow* generation on Jetson, so the fallback is itself a measurement, not an assumption. |
| **H-VLA-4** | ⭐ **THE USP.** In mode M the module's **F5 (scene sensitivity)** is significantly above the **C-B plan-echo control**. | 141 eval clips; paired episode-cluster bootstrap (`taniteval/ci.py`) | F5(model) − F5(C-B) > 0 with the paired CI excluding 0, **while F1 stays above the class prior** | the explanation is an **echo of the plan** ⇒ the USP is **not** established, and it is published as a negative with the number. No re-scoping: the next move is the diagnostic (which input is it reading?), not a softer claim. |
| **H-VLA-5** | Opening the **`g_str` write-back edge** (mode I, one edge) raises nav-compliance `Δ_shuffle` and E19 `t_bin` accuracy at **T1** without regressing the four families. | refcv5 arm, one lever, pre-registered | `Δ_shuffle` and `t_bin` both up, four families not worse, echo gate passed | the language path cannot fix a **NAV_BLIND** strategic head (`Δ_shuffle` = 0.0000 today) ⇒ the seam is a *wiring* problem, not an *information* problem, and E15/E17/E18 take priority over the VLA edge. |
| **H-VLA-6** | Our Alpamayo draw is **trajectory-anchored** (`2608.01755`'s bias), i.e. the CoT rationalises a revealed outcome. | 0 GPU. The **4,416 non-null clips** with a MEASURED 41.7 % CoT-vs-geometry lateral disagreement (D-LAT-AGREE): partition by agreement and test whether the CoT's causal claim tracks the **executed/labelled** manoeuvre or the model's **own predicted** trajectory where the two differ | anchoring detected | not anchored ⇒ the CoT may also supply **weak content** supervision on the agreeing subset. **Either way the corpus supplies the FORM**; anchoring only decides whether it may also supply content. |
| **H-VLA-7** | **2–6 executable tokens carry as much decision-relevant information as the 40–80-token prose form.** | 0 GPU on banked text: predict the v7.2 label from (a) the XCoT sequence, (b) a sentence embedding of the full CoT, (c) a **constant** control | (a) ≈ (b) ≫ (c) | prose carries more ⇒ the compact path loses information and the slow clock must carry *decision-relevant* content, not only exposition — which changes the tact budget in §2.0. |
| **H-VLA-8** | The **1,024-word vocabulary shrink** reproduces `2402.12289` Table 9's **4.33×** decode speedup on Thor for our closed taxonomy. | Thor, inference only, two vocabularies, same weights | ≥ 2× | the lever is Orin-specific ⇒ the token budget in §2.0 stands without it, and the slow clock gets proportionally less prose. |

⚠️ **Every one of these is scored with `n` and `d` printed, hyper-parameters fit on the FIT split only, and a control
that must read a known value** — the four probe failures of 2026-08-22 (target-energy normalisation, λ on the point
estimate, λ on the test set, n ≪ d) each produced a confident publishable number, and only a control caught them.

## 5. What we must NOT do

Each row is a trap **this programme has already paid for**, restated in the shape it will take inside a language
module. A design review of any VLA arm checks this list before it checks the results.

| # | the trap | the shape it takes here | the guard |
|---|---|---|---|
| **N1** | ⛔ **The nav echo.** refcv3's route metric was an **exact bijection of its own nav input** (369/369, 141/141) and scored **1.0000**. | An explanation or strategic head that reads the nav token and is scored on a nav-derived label. Nav is a **per-CLIP constant**, so a per-clip metric cannot distinguish skill from copying. | **F6 conditions on nav** and reports `Δ_nav-shuffle`; **C-D (nav-only)** is pre-armed and must read ~0 beyond nav. A bare accuracy on a nav-derived target is **inadmissible**. |
| **N2** | ⛔ **A goal input carrying the situation classifier's output** (PI 2026-08-03). | The LM sees everything; if a situation posterior, argmax, embedding, or *any feature derived from them* reaches the goal heads, attribution dies and the classifier stops being independently evaluable. | The goal path and the situation path stay **information-disjoint at inference**. For every input, ask *"could this have been computed from the situation classifier's output?"* — if yes, it is inadmissible until shown otherwise. State, per arm, what the goal is computed from. |
| **N3** | ⛔ **Reopening the ego echo through the language path.** | REF-C closed this with `ego_dropout 0.5` + the X15 presence bit (`H-ECHO-8`). A state-prefix token stream that hands the LM ego channels **unconditionally** reopens it, and the module would score well by reading ego rather than the scene. | The **same dropout and presence bit apply to the module's state prefix**. Admissible: the measured `v0` at t0 (PI 2026-09-02). ⛔ Inadmissible: future ego, and any label the target was derived from. |
| **N4** | ⛔ **Quoting a datacentre latency.** Alpamayo-R1's **99 ms is on an RTX 6000 Pro Blackwell** (`2511.00088` Table 14). | It reads exactly like an on-vehicle number and it is the number everyone cites. | Only **OrinX-measured** (`2402.12289` Tables 6/9) and **Thor-measured** numbers may size the tact. Every latency row names its silicon. |
| **N5** | ⚠️ **The sibling of N4: quoting Thor's 7.5× for decode.** | Decode is **weight-bandwidth-bound**; the transferable ratio is **273 / 204.8 = 1.33×**, not 7.5×. | §2.0. Prefill and the trunk get the compute ratio; the CoT does not. |
| **N6** | ⚠️ **"Thor saturates at batch 8."** | That is a **v6F *training* throughput** on a **different trunk** (12.3–14.1 windows/s across a 6× batch range). Inference here is **batch 1, latency-bound** — a completely different regime, and quoting the training fact would size the module wrongly. | `H-VLA-3` measures batch-1 latency explicitly. And on Thor **only `torch.cuda.max_memory_allocated()` is admissible** — `mem_get_info`, `free`, `tegrastats` and `VmRSS` all lie, in both directions. |
| **N7** | ⛔ **A second vision encoder.** | The obvious "just fine-tune SmolVLM" path. | It contradicts `2504.05299` Finding 1 (MEASURED), takes a disproportionate INT8 hit on Jetson (`2607.08029`, MEASURED), and — decisively — **destroys the faithfulness intervention**, because explanation and plan would then look at different tensors and a divergence could no longer be attributed to reasoning. |
| **N8** | ⛔ **Distilling the teacher's CONTENT.** | Our teacher is measured at **42.5 % reasoning fidelity / 48.3 % reasoning-action consistency** on **our own corpus** (`2605.17268`), **33.3 % of CoTs unreliable** (`2608.29583`), and our own spot check read 3/5 with a hallucinated cyclist. | Distil the **form**; take the content from our labels (`manoeuvre_sequence`, v7.2) and our detections (`obstacle.offline`). `H-VLA-6` decides whether even weak content supervision is admissible. |
| **N9** | ⛔ **Shipping a latent CoT as "the explanation".** | Latent reasoning (`2412.06769`, `2602.01166`, `2512.10226`) is the cheapest latency answer and it is **not a deliverable** — the PI asked for *"chain of thought as text as explanation"*. | The text head is mandatory; latency is bought with the **two-rate** design (§2.c), not by deleting the text. |
| **N10** | ⛔ **A supplied route or goal taken from the ego's own future.** | On PhysicalAI our only route supplier is the ego's future path — optimistic by construction, and at inference it is a privileged channel. | Prefer a **predicted** goal point. The 30 s strategic label is a **training target, never an input**. |
| **N11** | ⛔ **Pooling F1–F7 into one "faithfulness score".** | A composite hides exactly the F1/F5 trade-off the family exists to expose. | Per-metric, with its own estimator, CI, and `n` — the same rule the four families carry. |
| **N12** | ⛔ **Letting the module replace the cascade.** | The language module is **100–400× larger** than the 2.15 M-parameter cascade it complements. Ungated, it will dominate any joint objective. | Every write-back edge is **zero-init and individually gated**; mode M ships first with all gates closed; mode I opens **one edge at a time**, each earning a T1 four-family delta. |
| **N13** | ⚠️ **Quoting a paper from a library NOTE rather than the banked PDF.** | The notes in `library.json` are a previous agent's reading and are **INHERITED**, not PRIMARY. | Every number in §1 names its table or is marked *(abstract)*; `kb_add.py --verify` re-hashes the PDFs, and a file that exists with changed bytes is worse than a missing one. |
| **N14** | ⛔ **Reporting ADE, or the fifth family, instead of the four.** | A new metric family is the most tempting thing to lead with. | The fifth family is **ADDED**, never substituted. An eval without LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC is incomplete regardless of how good F1–F7 look, and every number carries its **T-tier**. |

## 6. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| this file (§0–§6) | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-extension-frontier/RESULT.md` | **in the repo**, branch `agent/arch-inf-20260803`, committed in increments |
| the refcv5 VLA extension plan (deliverable B) | `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` | **in the repo** (same branch) |
| 58 banked primaries tagged `vla-frontier-2026-09-05` | `TanitAD Research Lab/Library/papers/*.pdf`, indexed in `TanitAD Research Lab/Library/library.json`, rendered to `LIBRARY.md` | **in the repo**; `kb_add.py --verify` output in the plan's manifest |
| `H-VLA-1 … H-VLA-8` | `Project Steering/GOALS_AND_CLAIMS.md` | **registered** |
| extracted paper text used for the load-bearing tables (Alpamayo-R1 Table 14, DriveVLM Tables 6/8/9, SmolVLM §2–3, XCoT-VLA §3, FastDriveCoT §1/§4) | scratchpad only — **derivable from the banked PDFs with `pdftotext`**; nothing load-bearing lives only there | not banked (intentionally: the PDFs are the artifact) |

**Nothing in this deliverable exists only on one disk or only in one agent's context.**

## 7. Reconciliation with the contract (`REFCV5_DESIGN_PLAN.md` §9), and three corrections to §2

§9 landed in HEAD **after** §2 of this file was written. It is *the contract* (§9.0: where the two disagree about a
**port**, §9 wins). The design survives unchanged — §9.1's contract term is the same conclusion §2.c reached
independently — but three statements above need correcting, and one addition is load-bearing.

**C1 — `pooled` is a VECTOR; the 160 tokens are `fmap`.** §9.2 separates them: `pooled [B, feat_dim]` is the pooled
feature at t0, `pooled_seq [B, 8, feat_dim]` is the hook's first argument, and the **160 perspective-view tokens** are
`fmap [B, feat_dim, 8, 20]`, consumed by the decoder's `feat_proj`. ⇒ **Everywhere §2.c says the module reads
"`pooled`'s 160 tokens", read `fmap`** (with `pooled_seq` as the temporal context). The design is unaffected; the
port name was wrong.

**C2 — `feat_dim` is 704 at `size base`, not 512, and it must never be hardcoded.** §9.2/§9.8: `feat_dim =
base_width · 8 = 704` at base, **256** at the `tiny` rig rung, and *"the VLA must read them from the built model,
never hardcode 704 / 512 / 256"*. ⇒ `proj_in` is `Linear(feat_dim → d_lm)` with `feat_dim` read at construction;
§2.c's ≈ 1.6 M projector estimate becomes ≈ **1.4 M** at 704 → 960 → 960 and the ≈ 368 M / ≈ 6 M totals are unchanged
at this precision.

**C3 — the geometric goal port is `g_tac [B, 12]`, not a single point.** §9.2: `GOAL_DIMS = 4`, layout
`(x, y, heading, speed)` per τ ∈ **{2.0, 4.0, 6.0} s**, ego frame x forward / y left / z up. ⇒ the write head is
12-wide, and `goal_consistency` is measured **per τ**.

**A1 — ⭐ the mount point already exists, and one write port is LIVE.** §9.1: `RefCModel.forward(...,
hierarchy_hook=...)` is called as `hook(pooled_seq, ctx) -> dict` and fills ports **only where the caller passed
`None`**, so *"an explicitly supplied port always wins"*. `maneuver_logits [B, 5]` reweights the anchor prior (H19)
and is **never filled today** — *"an outside tactical brain speaking the 5-way surface"*. ⇒ **the first mode-I edge
costs zero new wiring**, and it is the cheapest possible causal test of the whole thesis (plan WP-8a). And §9.1's
docstring supplies the latency argument in advance: the hook exists so an external hierarchy can read the window
*without encoding the frames a second time*, because **the encoder is ~90 % of a REF-C tick**.

**A2 — the information-state declaration, sharpened.** §9.7 splits arms into **PROPOSER** (does not see the pick;
consistency is a real measurement) and **DESCRIBER** (sees `sel_idx`/`g_tac`; consistency ≈ 1.0 *by construction* and
is **not a result**). Our §3.0 modes M/I are about **write-back**; §9.7's regimes are about **read-access**. They are
**orthogonal and both required**, so the plan declares three read tiers — **P0** scene-only · **P1** scene + cascade
state *upstream of the pick* (the headline) · **D** describer — and every table naming `tac_consistency` states its
tier. ⚠️ A P1 number and a D number are not comparable.

**A3 — the constant control's known value is now a NUMBER.** §9.7: the corpus is **86.59 % `lane_keep`, 75.76 %
`steady`, 67.18 % both** ⇒ a module that always says `(LANE_KEEP, CRUISE)` reads ≈ **0.67**, and **any F1 below 0.67
is worse than a constant**. That converts C-A from "must read the prior" into a pre-registered threshold.

**A4 — two refusals §9 adds that §5 did not have.** (i) `ctx` **already contains nav** (`ctx = ctx + nav_s`), so a
module reading `ctx` may not claim a vision-only reading — it carries the nav controls instead. (ii) A **question**
is admissible as an input to the module only; **its answer may not route into a write port in the same tick**, or a
question becomes a control channel and the goal path stops being information-disjoint from the situation path.

**Library verification (MEASURED, 2026-09-05):** `python tools/kb_add.py --verify` →
**`verified 437 entries, 0 orphan(s), 0 problem(s)`** — every banked PDF re-hashed against `library.json`.
