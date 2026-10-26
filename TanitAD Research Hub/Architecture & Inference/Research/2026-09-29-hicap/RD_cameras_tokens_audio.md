# HiCAP stream R-D — elastic camera/token usage, and audio in the shared embedding space (2026-09-29)

**Status:** COMPLETE — staged in the index, **not committed**. Scope: the PI's two new demands on REF-F/HiCAP — (1) *"adapt the amount of encoded state and cameras by
varying the usage of the cameras"* (a compute-elastic state), (2) *"add audio as an additional tokenised input, in the same embedding space as language and vision"*.
Maximal compute efficiency is the overriding constraint. No pod, no torch, no GPU touched; one numpy toy was **built and run** (§5).

**Evidence classes** (as tagged inline): `PUBLISHED` (title/id/venue seen this session; **-P** = read in primary code/README I cloned, **-S** = abstract or search-result snippet only — re-verify
before spending a GPU-day) · `RECALLED` (id from memory, not re-seen this session) · `MEASURED (ours)` (script named) · `ESTIMATED` (arithmetic shown) · `HYPOTHESIS` · `INHERITED` (another
repo doc, not re-verified) · `UNVERIFIED`. TanitAD model facts are quoted from nothing here except where tagged `INHERITED`; **no registry number is used**.

## 0. Headline — escalations first, then findings

**Escalations (each blocks ONE item, not the programme):**
1. ⛔ **PI ruling needed on the gate's cues.** A camera gate is a *situation classifier in disguise* (situation → camera set). The binding of 2026-08-03 makes **inference vision-only** for the scenario
   classifier; HiCAP's brief includes ego in the state. `MEASURED (toy)`: an **ego-cue gate reaches 95.1 %** of decisions at 1.64 camera-units, a **vision-only gate only 81.2 %**, and the vision-only gate *with its own
   previous choice fed back* **collapses (80.4 %, wrong camera held 41–62 ticks ≈ 4–6 s)**; it recovers (96.6 %) only when the always-on front-wide embedding carries ≈3× the situation information (§5.4). Until ruled,
   §3 carries **G-V (vision-only, primary/deployable under the binding)** and **G-E (+ego, ablation)**; **G-T (+previous tactical token) is inadmissible until it passes the §3.2 information-disjointness tests.**
2. **Extra cameras need new caches** — every TanitAD cache is front-wide only. ≤197 source chunks × 1.6–2.5 GB per camera = **321–497 GB transient download per camera, 2.55 TB for all six** (§3.7, `ESTIMATED`
   from `MEASURED` chunk sizes). Provision/spend is the PI's; the first camera (front-tele, 321 GB) is gated on experiment E-C3 (§6), which needs a *side-study* of ~30 dense chunks (49 GB), not the parity corpus.
3. **Audio cannot be validated as driving skill on any corpus we have or can download** (§4.3). What can be built and tested is plumbing + a synthetic/sim path; the PI decides whether that is worth its cost.

**Findings:**
1. `PUBLISHED` systems budget multi-camera tokens three ways: **decouple tokens from camera count** (Alpamayo-R1 triplane ≈288 tokens/timestep, 3.9× fewer, up to 72 % fewer / 50 % faster inference at equal open-loop
   accuracy — -S), **prune** (ST-Prune 90 % reduction near-lossless; MVPruner 87.3 % FLOPs cut, 4.97× prefill — -S), or **select views** (DriveMoE: front pair + one router-chosen view; 260 ms vs 700 ms for six fixed views — -S).
   The closest precedent to HiCAP's gate is DriveMoE; its router code is read in §1.5 (**no FLOPs term in its loss; 71.8 % of its router labels are one default class**).
2. **No clean published front-only-vs-N-camera planning ablation, stratified by manoeuvre, was found** (nuScenes/UniAD/VAD: absent at every location probed; NAVSIM: front-only LTF competitive with surround-view
   UniAD/PARA-Drive). Open-loop metrics under-credit side/rear cameras (and our own `INHERITED` finding: ≈64 % of an anticipation gap survives with the scene destroyed). **The value of extra cameras on our data is unproven — E-C3 is the first experiment.**
3. `MEASURED (toy)` the gate can serve **95.1 % of decisions at 1.64 camera-units vs 7** (4.3× cheaper), but **the lead over fixed-all (+1.4…+1.8 pt) is a head-routing-difficulty artefact**: the honest references are the
   **oracle gate 96.7 %** (gap −1.6 pt) and the **all-cameras-plus-true-situation ceiling 97.6 %** (gap −2.5 pt). **Below ≈1.6 units accuracy falls off a cliff** (91.5 % at 1.49, 82.5 % at 1.25).
4. `MEASURED (toy)` **negative results that shape the design:** confidence-triggered escalation to all cameras buys *nothing* (−0.03…−1.5 pt) for +2…+145 % cost (τ ≤ 0.9); hedging with the runner-up camera ≤ +0.06 pt; **symmetric dwell hysteresis costs 0.6 pt and
   EMA is neutral (both halve switches)**; **asymmetric (escalate fast, de-escalate slow) is the only variant that gains (+1.4 pt vs none for +6 % cost)**. Gate errors are *confident* (follow-far→cruise 28 %).
5. **Matryoshka *dimensions* save ≈0 FLOPs at HiCAP's scale** (scoring 10³–10⁵ cached candidates × 256-d is ≪1 ms next to a 17–120 ms encoder). The nesting that carries compute is over **cameras and tokens**, and it works
   only if the heads are **budget-conditioned** — a frozen backbone cannot be Matryoshka-trained, so the nesting lives in a trainable pooler + heads.
6. **Audio compute is negligible; audio *data and validation* are the problem.** A 2-s window through a BEATs-base-class tower ≈ 18 GFLOP (≈0.7 ms on Thor) vs ≈17–31 ms for the first camera. **PhysicalAI-AV (four locations), nuScenes and comma2k19
   (two each), Waymo Open and AV2 (repo grep only) carry no audio; AlpaSim strips it; CARLA has no native sound.** The one untested location is the PhysicalAI mp4 container itself (E-A0, ≈10 minutes).

## 1. Multi-camera driving: how published systems budget tokens (priority 1)

### 1.1 What the systems do

| system | cameras · frames | token budget mechanism | evidence |
|---|---|---|---|
| **Alpamayo-R1** (arXiv 2511.00088) released code | **4 cams (CL, FW, CR, FT) × 4 frames** default (`alpamayo/src/alpamayo_r1/load_physical_aiavdataset.py:52-55,73-78`) | pixel window 163,840–196,608 px/image (`helper.py:23-24`) ⇒ ≈160–250 tokens/image (paper snippet: 448×280 → 160 tokens); 16 images ⇒ ≈2,600 tokens/tick | -P code |
| Alpamayo-R1 paper (same id) | up to 7 | **multi-camera triplane tokenizer ≈288 tokens/timestep, 3.9× fewer than per-camera; video tokenizer "Flex" 8–45 tokens/frame** | -S snippet; triplane tokenizer **not released** (`NVlabs/alpamayo1.5` issue #4, fetched: "only the Qwen3-VL patch-based encoder is in releases") |
| Triplane tokenization (Ivanovic et al., arXiv 2506.12251, NVIDIA/Stanford) | any | fixed-size 3-D triplane ⇒ tokens agnostic to camera count *and* resolution; **up to 72 % fewer tokens, up to 50 % faster policy, same open-loop accuracy, better closed-loop off-road** | -S abstract |
| **Alpamayo 2 Super** code (`alpamayo2`@5e7975f) | canonical **7-camera ring**; **fixed per-task profiles**: trajectory/meta-action/auto-label/grounding IDs **[0,1,2,3,5,6] = CL, FW, CR, RL, RR, FT (rear-tele dropped)**; VQA **[0..5] (front-tele dropped)**; 4 frames; camera-ID embeddings (`constants.py:28-37`) | 6 cams × 4 frames peaks ≈67 GiB on the VLM GPU (`README.md:222`) — NVIDIA selects cameras **per task, statically** | -P code |
| **DriveMoE** (arXiv 2505.16278; CVPR 2026 listing) | 6 cams; **front + previous-front always, + ONE router-chosen view** | Scene-Specialized Vision MoE; **latency 260 ms vs 240 ms (two-view) vs 700 ms (six fixed views)**; Bench2Drive DS 74.22 vs 55.85, SR 48.64 % vs 30.00 % (**confounded with its Action MoE**; removing either hurts) | -S snippet; **router code -P (§1.5)** |
| OmniDrive (2405.01533) | 6 | Q-Former3D compresses multi-view into a fixed query set | -S; **per-camera token count not obtained → UNVERIFIED** |
| EMMA (2410.23262), OpenDriveVLA (2503.23463), ORION (ICCV 2025), DriveVLM, Senna | 6 / front | per-camera token counts **not obtained this session** (search budget) | **UNVERIFIED — do not quote** |
| SV-WAM (2609.03602, 2026-09) | six-view, ≈5 B backbone | keeps all six views; argues single-front designs "restrict spatial coverage in lane changes, merges, turns" | -S |
| Prune2Drive (2508.13305), **ST-Prune** (2604.19145), **MVPruner** (2606.27660) | multi-view driving VLMs | training-free pruning; ST-Prune 90 % reduction near-lossless (4 benchmarks: perception/prediction/planning); **MVPruner allocates per-*view* budgets by information diversity: 87.3 % FLOPs cut, 4.97× prefill, 98.5 % accuracy (DriveMM on DriveLM)** | -S; VQA-type benchmarks, not closed-loop |

### 1.2 Which cameras matter for which manoeuvre

| manoeuvre | camera(s) | support | class |
|---|---|---|---|
| cruise, lead at low speed | front-wide | every system's default | `PUBLISHED` |
| lead/distance keeping at speed, far objects | front-tele (30°) | in AR1 default and AR2 trajectory profile; **no measured benefit found** | `HYPOTHESIS` (E-C3) |
| junction turn | cross camera on the exit side | DriveMoE's label rule (paper text via snippet): *in-junction ∧ turn command → front-side camera facing the exit* | `PUBLISHED-S` |
| lane change / merge | rear-side, same direction (front-side if the target lane is oncoming) | DriveMoE label rule; its action tokens weight front-left + rear most in a left change | `PUBLISHED-S` |
| pull-over / kerb / reverse | rear | DriveMoE default when no critical view exists | `PUBLISHED-S` |

⚠️ **DriveMoE's router labels are 71.8 % one class** (`src/model/DriveMoE/loss.py:39`: counts FL 35,105 · FR 8,865 · **BACK 161,717** · BL 13,671 · BR 5,990 = 225,348; `"NULL"` also maps to `BACK`, `camera_scenario_map.py`).
A router that always answers "default" scores 71.8 % ⇒ **gate quality must be reported as per-class recall / balanced accuracy, never accuracy.**

### 1.3 Measured value of extra cameras — what exists and what does not
* DriveMoE's +18.4 DS / +18.6 pt SR over its Drive-π0 base is the **joint** effect of the Vision MoE and Action MoE; the camera-only ablation numbers were **not obtained** ⇒ not attributable to view selection.
* NAVSIM (arXiv 2406.15349): a snippet reports the wide-angle **front-camera-only LTF as competitive with surround-view UniAD/PARA-Drive** (source paper of the sentence not identified; I did not obtain the PDMS table). `HYPOTHESIS`:
  non-reactive open-loop replay rarely rewards seeing a rear-approaching vehicle, so front-only "ties" say little about lane-change or junction skill.
* Absence at one location is not absence: **nuScenes/UniAD/VAD front-only-vs-six-camera planning ablations were searched in 4 phrasings and not found** — I do not claim they do not exist.
* `INHERITED` (`PROJECT_STATE.md:351`, SC-13 run 5): **≈64 % of the open-loop anticipation gap survives with the scene destroyed** — open-loop planning metrics are dominated by ego kinematics. ⇒ camera value must be measured
  **stratified by manoeuvre and closed-loop/four-family**, which is exactly what §3.8 pre-registers.

### 1.4 Camera dropout / robustness
Sensor-dropout training (random suppression of a camera/modality per sample) is the standard recipe (progressive sensor dropout; `PUBLISHED-S`). Planning-level tolerance benchmarks now exist: **DriveDegrade** (2608.29005: 16 corruption
families × 5 severities incl. camera loss, nuScenes + NAVSIM + CARLA anchor; "clear breakpoints for blur, JPEG, noise and camera loss") and **Bench2Drive-Robust** (2605.18059); **M²-Occ** (2603.09737) handles incomplete camera sets for occupancy.
⇒ HiCAP's camera-dropout curriculum is conventional; DriveDegrade's camera-loss family is a candidate external sanity check for the elastic heads (protocol details **UNVERIFIED**).

### 1.5 The one published precedent for a view router — read from code (`Thinklab-SJTU/DriveMoE`@e39df2f; -P: primary code, read by me, not run)
| aspect | what the code does | consequence for HiCAP |
|---|---|---|
| router input | learned query cross-attends over **[waypoint token; front-camera tokens]** → LN → MLP → logits over **5 views** (`router.py:31-58`; `num_camera_views_selected: 5` in `config/*/DriveMoE/*.yaml`) | the 2-D `waypoints` tensor comes from the batch (`eval.py:76`; presumably the navigation target — **UNVERIFIED**): a *supplied route*, optimistic by construction on PhysicalAI → not copyable |
| always-on set | front + previous-front images (`drivemoe.py:119,145`) | = HiCAP's "FW always on" |
| training | stage 1: **supervised** with camera-id labels, focal loss, inverse-frequency α (`loss.py:39-45`); stage 2: **`F.gumbel_softmax(tau=1, hard=True)`** (`drivemoe.py:124`) with **all five candidate views encoded** (`:120`) | training costs all cameras; labels come from simulator annotations (privileged — legal for *labels*) |
| inference | encode the front pair → router `argmax` → **encode only the selected view** (`drivemoe.py:135-158`) | a real saving; exactly one dynamic view, no "none", no hedge |
| cost control | `CombinedLoss` = camera-router + action-router + action terms; **no FLOPs/budget term** | budget is the fixed top-1, not a learned trade-off; **HiCAP's λ-penalised gate is new** |
| dynamics | none: no hysteresis, no cold-start handling | §5 measures what they cost |

## 2. Elastic compute methods that work with a FROZEN backbone (priority 2)

The sibling stream R-A tabulates NVIDIA's *measured* Thor rows and a token-compression table for frozen VLMs (`RA_vlm_backbones.md` §5.1, §5.4, `INHERITED`); this section adds what R-A does not cover and states applicability to
**a frozen VLM whose output is a pooled embedding** (HiCAP's state encoder).

| method (id, status) | acts on | reported trade-off | fine-tuning? | frozen-VLM + pooled output? |
|---|---|---|---|---|
| static masks / fixed ROI (sky, hood; `ENCODER_MULTICAM_OPTIMIZATION.md`; MaskVD 2407.12067 as cited there, `INHERITED`) | pixels | 20–35 % tokens free | none | **yes, exact, cuts tower *and* LLM** |
| **FastV** (2403.06764 R), **SparseVLM** (2410.04417 R), **VisionZip** (2412.04467 R), on LLaVA-1.5-7B at 64/576 tokens | in-LLM attention / tower output | **75.3 % / 88.5 % / 94.1 %** of full score (VisPruner 2412.01818, ICCV 2025, -S) | none | FastV needs attention maps ⇒ incompatible with fused attention (R-A: avoid on Thor); SparseVLM needs text guidance (HiCAP's prompt is fixed ⇒ nothing to guide); **VisionZip acts at the tower output — frozen-safe, cuts LLM cost only** |
| **DART** (2502.11494, -S) duplication-aware | LLM early layer (index **UNVERIFIED**) | prunes **88.9 %** tokens, **1.99× total / 2.99× prefill**, FlashAttention-compatible; "importance" ≈ or worse than random | none | yes; cuts LLM part only |
| ToMe (2210.09461 R), PruMerge (2403.15388 R), FitPrune (2409.10197 R) | tower / LLM | merge/prune | none | static-ratio variants export to TRT (`ENCODER_MULTICAM…`); dynamic shapes do not |
| **ST-Prune / MVPruner** (§1.1) | driving multi-view | 90 % near-lossless / 87.3 % FLOPs | none | yes; **per-view budgets = the allocator HiCAP's gate would learn**; VQA-scored only |
| **Matryoshka**: MRL (2205.13147), M3 (2405.17430), **Matryoshka-Adaptor** (2407.20243) | embedding dims / visual tokens | M3: ~9 tokens ≈ 576-token accuracy on COCO-style benchmarks; Adaptor: 2–12× smaller embeddings without loss on BEIR, **post-hoc on frozen/black-box embeddings** (-S) | MRL/M3 train the backbone; **the Adaptor does not** | only the **Adaptor-style trainable head** applies to a frozen VLM |
| Mixture-of-Depths (2404.02258 R), VideoLLM-MoD (2408.16730), AdaVSkip (2609.15131), MoLe-VLA (2503.20384) | layers / tokens | router-selected token or layer skipping | **train routers** | not without LoRA + router; frozen-friendly variant = probe heads at layer k (`HYPOTHESIS`) |
| adaptive resolution / foveation (`UNIFIED_FOV_FOVEATED_PATCHING.md`) | patch layout | fixed foveated layout is TRT-static | needs positional adaptation | no (frozen position embeddings) |
| **StreamingVLM** (2510.09608, -S) | KV cache | sink + short vision window + long text window; **SFT on overlapped chunks aligns train with streaming** | **yes** | **not exact for a frozen VLM**; exact reuse = cache *tower outputs* per camera-frame and the static prompt prefix |
| **Per-camera update rate** (2 Hz rear vs 10 Hz front) | scheduling | — | none | **exact and free** (`HYPOTHESIS`; this is what a "slow lane" is, §3.2) |
| **Learned budget routers**: DynamicViT (2106.02034) Gumbel/ST + budget loss; DriveMoE Gumbel-ST without budget term | gate | — | trains the gate only | yes — but with all cameras cached offline **full-information cost-sensitive training needs no RL** (§5) |

2026 successors seen **by title only** (not read; do not cite for numbers): "From Token Importance to Conditional Removability" (2609.26484), ResPrune (2603.21105), "Balancing Saliency and Coverage" (2603.14892), "Prune Redundancy, Preserve Essence" (2603.09480), head-aware pruning (2608.25332), E2S-Pruner (2608.23253),
"Look Less, Think Faster: Joint Token-Compute Adaptation for Multimodal LLMs" (2607.20357 — the closest title to a *joint* token-and-compute budget). **RL for view selection: no source found this session (`UNVERIFIED`).** The toy suggests RL is unnecessary while the counterfactual camera sets are computable offline;
it becomes necessary only for effects that need closed-loop rollouts (a camera choice that changes *future* states).

**Applicability verdict.** For a frozen VLM with pooled output: (a) **input-side** elasticity (camera on/off, resolution, update rate, static masks) is exact and cuts tower *and* LLM; (b) **tower-output** pruning/merging (VisionZip/DART-style, or a
**trainable attention pooler to k tokens/camera**, CoVer-style, `INHERITED` from `2026-09-29-reff-prior-art-and-theory.md` §4.3) cuts the LLM part (≈55–75 % of a 4 B tick per R-A) but **not the tower**; (c) anything that
needs router training *inside* the VLM is out. ⚠️ **Training-free pruning changes the frozen model's output embedding**, so heads trained on unpruned embeddings see a shift: measure cos(e_full, e_config) before adopting (E-R1, §6)
and train heads on embeddings produced under the same configs (§3.6).

## 3. A concrete elastic-state design for HiCAP (priority 3)

### 3.1 Camera tiers (fast lane = front-wide @10 Hz always; extras chosen per tick)
| tier | cameras | triggered by (targets — learned, not hard-coded) | tokens (extras) | Thor ms, 2B-class / 4B-class (§3.5) |
|---|---|---|---|---|
| T0 | FW | default | — | 16.6 / 30.8 |
| T1a | + front-tele | speed high and a lead/far object in FW; distance keeping | 64–160 | 20.6–26.1 / 37.0–45.6 |
| T1b | + one cross camera (side = turn/route side) | junction approach, turn intent, stop line | 64 | 20.6 / 37.0 |
| T1c | + one rear-side, or + rear-tele | lane-change intent / merge; pull-over, kerb, reverse | 64 | 20.6 / 37.0 |
| T2 | FW + 2 extras (e.g. FT + cross, both cross at an unprotected junction) | two-factor situations | 64 each | 24.5 / 43.3 |
| T3 | all 7, peripheral budget | **probe / degraded-gate state only, not steady state** | 6 × 64 | 40.4 / 68.2 |

Fixed alternatives for scale: **AR1's 4 cams × 4 frames ≈ 2,624 tokens ≈ 19.5 TFLOP/tick** (`ESTIMATED`; ≫100 ms) — **frames per camera is a bigger lever than camera count**: HiCAP should use **1 frame/camera/tick** (+ ego history tokens).

### 3.2 The gate
* **Inputs, three arms (the disjointness statement CLAUDE.md requires):**
  **G-V (primary):** the FW pooled embedding (already computed for the decision, gate cost ≈10⁶ FLOPs: a 2048→256→8 MLP) + the gate's own previous camera-set (keep-alive state). **Vision-only** ⇒ admissible under the 2026-08-03 binding. Goal path and situation path are
  information-disjoint *except through the shared FW trunk* — stated, and tested by the config-invariance test below.
  **G-E:** + ego kinematics (speed, yaw-rate, indicator). Needs PI ruling (escalation 1).
  **G-T:** + previous-tick tactical token. ⛔ **Not information-disjoint**: the tactical token decides which pixels the goal head sees next tick — a back door of the *selection* kind (attribution dies; errors can lock in). Admissible **only if** it beats G-V/G-E by the pre-registered
  margin **and** passes (i) **config-invariance** — the goal head scored on *fixed-camera* states must not degrade relative to gated states; (ii) **gate-shuffle** — replace the gate's camera-set indicator by a shuffled one.
* **Labels may use everything** (binding): per-camera relevance = nearest dynamic agent in that camera's FOV within a TTC/distance threshold from **`obstacle.offline`** (3-D tracked cuboids, 97.44 % coverage, `PHYSICALAI_FEATURE_PROBE.md:53`) + egomotion-derived manoeuvre intent + `blurred_boxes.parquet` (free 2-D evidence shipped with every camera, `:78-80`, `:243`).
* **Training — full-information cost-sensitive** (validated in §5): with all cameras cached offline, the per-option error E_j(x) of the *frozen* VLM + heads is computable for every training frame, so the gate minimises Σ_j p_j(x)[E_j(x) + λ·cost_j].
  Add a **calibrated situation head** (cross-entropy on the label-derived situation) as an auxiliary/ensemble: in the toy it is **as good as the cost-sensitive gate (95.8 % vs 95.1 %)** when the situation→camera map is one-to-one; cost-sensitivity earns its keep only when the map is not.
* **Asymmetric loss:** the toy's regret matrix (§5.3) says a wrong camera costs ≈0 in cruise, ≈22 pt for tele, **≈41–46 pt (chance) for cross/rear situations**. Weight E_j by situation criticality — a missed junction is not a missed cruise.
* **Hysteresis = keep-alive:** switch *up* (or sideways) in 1 tick, *down* after 3 ticks with margin 0.10. **Not symmetric dwell** (−0.6 pt); EMA is neutral (§5.5).
* **Cold start & prefetch:** a newly enabled camera has no temporal cache: `MEASURED (toy)` −0.7 pt even for a perfect gate; **prefetching 2 ticks ahead recovers +0.45 pt for +0.03 units.** Rule: switch a camera on in *shadow mode* (encode, don't use) for 2 ticks before its first use; drive prefetch from intent (route side/indicator).
* **Slow lane for the unknown unknown:** confidence-triggered escalation did not work (gate errors are confident, §5.6), so add a **scheduled probe**: rear/cross cameras encoded at 2 Hz, 64 tokens, **tower + pooler only (no LLM pass)** feeding a small peripheral-risk head. `ESTIMATED` 6 cams × 2 Hz × 0.21 TFLOP ≈ **2.6 TFLOP/s ≈ 19 %** of the FW fast lane (10 Hz × 1.34 TFLOP = 13.4 TFLOP/s). `HYPOTHESIS` — untested.

### 3.3 Token budgets per camera (a config knob in v1, not learned)
FW 160 tokens (448×280-class window; the v6 cylindrical 256×640 grid = 640 patches → 160 merged tokens, the R-A convention). Extras 64 tokens (256 patches; 0.42× the marginal cost of a 160-token extra, `ESTIMATED`). Front-tele is 30°, so even 64 tokens (8×8) is ≈2× the horizontal angular token density of a 160-token FW (16×10 over 120°) —
its geometry needs its own crop (not the cylindrical FW canvas).

### 3.4 Nested embeddings — what nests, what does not
* **Dimensions (256/128/64) — adopt as a free option, expect nothing.** Scoring |C| cached candidates costs |C|×d MACs: 10⁵ × 256 = 2.6·10⁷ ≪ 1 ms next to a 17–120 ms encoder. Value: **cached-action storage/bandwidth** if the vocabulary is ≥10⁵–10⁶ (10⁶ × 256 × 2 B = 512 MB vs 128 MB at 64-d) and a coarse-to-fine shortlist.
* **State information (cameras, tokens) — the nesting that matters.** Loss for a config c with the frozen VLM: **L = Σ_c w_c Σ_{d∈{64,128,256}} InfoNCE(f_d(e_c), a_d)  +  β·(1 − cos(e_c, sg(e_full)))**: heads share weights across configs, a **config embedding** is appended to the pooled state, and the **full-camera embedding is the distillation target** for reduced-camera ones.
  Trainable: pooler, adapter (Matryoshka-Adaptor-style down/up + skip), heads. **Frozen: the VLM.**

### 3.5 Cost model (`ESTIMATED`; `rd_cost_model.py`; ±25 % band, excludes heads/ego encoder/pre-processing/hand-off)
Latency is **not derived from peak FLOPs**: the "TRT" columns scale NVIDIA's published Thor rows as tabulated by R-A (Qwen3-VL-2B: 16.6 ms 1 cam / 35.5 ms 3 cams; Qwen3.5-4B-class: 30.8 / 60.5; `INHERITED`, NVFP4 LLM + FP16 ViT, batch 1); the marginal camera is linear (R-A: batching cameras helps the ViT only ≈18 %).
"ms@25 / @100" bracket my own FLOP model with 25 TFLOP/s (≈ eager BF16, R-A: 24.6) and 100 TFLOP/s effective. **No Thor latency exists for any HiCAP configuration — E-L (§6) measures it.**

| config (1 frame/camera) | LLM tokens | tower + LLM TFLOP (my model: 0.56 TF per 640-patch image; 2·1.7 B·N + attention) | ms@25 | ms@100 | **2B TRT** | **4B TRT** |
|---|---:|---|---:|---:|---:|---:|
| C1 FW@160 | 224 | 0.56 + 0.77 = 1.34 | 53 | 18 | **16.6** | **30.8** |
| C2 FW + 1×@160 | 384 | 1.13 + 1.34 = 2.47 | 99 | 25 | 26.1 | 45.6 |
| C2p FW + 1×@64 | 288 | 0.78 + 1.00 = 1.77 | 71 | 20 | 20.6 | 37.0 |
| C3 FW + 2×@160 | 544 | 1.69 + 1.92 = 3.61 | 144 | 36 | 35.5 | 60.5 |
| C4 four cams @160 (AR1 cams, 1 frame) | 704 | 2.25 + 2.51 = 4.76 | 190 | 48 | 45.0 | 75.3 |
| C4p FW + 3×@64 | 416 | 1.20 + 1.45 = 2.66 | 106 | 27 | 28.5 | 49.5 |
| C7 seven cams @160 | 1184 | 3.94 + 4.35 = 8.29 | 332 | 83 | 73.3 | **119.9 (> 100 ms)** |
| **C7p FW@160 + 6×@64** | 608 | 1.84 + 2.15 = 3.99 | 160 | 40 | 40.4 | 68.2 |
| AR1 default: 4 cams × 4 frames @160 | 2624 | 9.01 + 10.50 = 19.5 | 780 | 195 | — | — |

Reading: with peripheral cameras at 64 tokens **even all seven fit a 100 ms tick on a 2B-class TRT stack (40 ms) and marginally on 4B (68 ms)**; unbudgeted C7 does not fit 4B. The LLM stage is weight-streaming-bound below ≈92–366 tokens (crossover at 25–100 TFLOP/s;
LLM weights 3.4 GB fp16 ÷ 273 GB/s = 12.5 ms floor), so **the first extra camera's LLM tokens are nearly free; its tower pass is not** ⇒ *pre-tower* savings (resolution, static mask, update rate, on/off) matter more than post-tower pruning. Thor bandwidth 273 GB/s: `INHERITED`, `FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md:295`.

### 3.6 Training recipe
1. **Config cache (0 new architecture):** run the frozen VLM under the 8 options {FW, FW+one extra ×6, ALL} on every 5th frame of the parity corpus (94.5 k frames): tower ×7 + 8 LLM passes = **17.1 TFLOP/frame ⇒ 1.6 EFLOP ⇒ ≈11 A40-hours at 40 TFLOP/s** (`ESTIMATED`; ≈2× for a 4B-class). Cache pooled embeddings (fp16, ≈4 KB × 8 × 94.5 k ≈ 3 GB).
2. **Heads with a camera-dropout curriculum** (the mask mix that worked in the toy): 40 % the situation's oracle camera · 20 % oracle + one random extra · 15 % all · 25 % FW + a random 0–2 cameras. **Budget-conditioned:** config embedding + the §3.4 loss.
3. **Gate** on the cached per-option errors (§3.2). 4. **Joint** fine-tune of pooler/heads only if E-R1 shows drift. **Do not** train the gate against embeddings whose config distribution differs from deployment.

### 3.7 Data reality — extra cameras need new caches (`ESTIMATED` from `MEASURED` chunk sizes, `PHYSICALAI_FEATURE_PROBE.md:70-76`)
| camera | chunk MB (measured) | per-clip MB (chunk/97) | transient GB, 197 chunks | 30 dense chunks |
|---|---:|---:|---:|---:|
| front-tele 30° | 1,627 | 16.8 | **321** | 48.8 |
| cross-left 120° | 2,261 | 23.3 | 445 | 67.8 |
| cross-right 120° | 2,521 | 26.0 | 497 | 75.6 |
| rear-left 70° | 2,130 | 22.0 | 420 | 63.9 |
| rear-right 70° | 2,392 | 24.7 | 471 | 71.8 |
| rear-tele 30° | 2,001 | 20.6 | 394 | 60.0 |
| **all six** | 12,932 | — | **2,548 (2.55 TB)** | **388** |

* **Time:** 2.55 TB at the dev-box's measured 2.1 MB/s (`2026-08-04-instrument-durability/INTAKE.md:52`) = 14 days (⇒ pod-side only); at the pod rate implied by `Research/2026-07-09-physicalai-r1-selection-…md:49-56` (60 GB in ~1 h ≈ 17 MB/s) = 42 h; at HF↔pod 118 MB/s = 6 h. **Build:** front-wide ran at 26 clips/min (5 workers, `V2_PHASE2_BUILD.md`) ⇒ 2,376 clips ≈ 91 min/camera ⇒ **≈9 h for six**. **Cache:** JPEG q90 ≈ 2.9 MB/episode at 256² ⇒ 6.9 GB/camera (≈17 GB at 256×640).
* ⚠️ **Chunk geometry:** the parity set came from **197 chunks** (`V2_CORPUS_DESIGN.md` §6; exact chunk count for the 2,376 subset **UNVERIFIED**), R0/R1's 1,926 clips sit in **30 chunks** (64 clips/chunk) — a 30-chunk side-study costs 8× less per clip but **is not the parity corpus** ⇒ decision-grade for "does camera X help at all", **not** for cross-arm comparison. Chunk sizes disagree across docs (v2 design implies ≈1.21 GB/chunk for front-wide vs the probe's 2,051 MB); I use the probe's per-feature measurements.
* **Parity is sacred:** extra cameras are keyed by `clip_id` onto the *same* 2,376 episodes / skip-hash `f09e44db`; nothing here re-selects episodes.

### 3.8 Pre-registered evaluation (paired; four families; per family, never pooled)
**Arms:** A0 fixed FW · A1 fixed AR1-4 (CL, FW, CR, FT) · A2 fixed all-7 @ peripheral budget · **A3 gated G-V** · A4 gated G-E · A5 gated G-T · A6 oracle gate from the label-derived relevance (upper bound) · **A7 random gate with A3's cost distribution** (is the gate *informative*, or only cheap?).
**Metrics (binding — ADE alone is INCOMPLETE):** LONGITUDINAL (target-speed accuracy, headway/TTC), LATERAL (heading, curvature, yaw-rate, cross-track), TACTICAL (manoeuvre decision + goal selection, confusion), STRATEGIC (route/goal quality) — each with the **paired episode-cluster bootstrap** (`taniteval/ci.py`) over the 40 val episodes on the same windows;
**stratified** junction / lane-change / pull-over / high-speed follow / other with n per stratum (n < 30 windows ⇒ "not evaluable", said so). **Cost:** measured Thor p50/p95, mean camera-units, switch rate. **Gate:** per-class recall, mean wrong-run length, and the config-invariance + gate-shuffle tests.
**Decision rule (both outcomes committed):** *adopt* G-V iff, in every family, the paired CI of (A3 − A2) excludes a harm margin δ_f (set **before launch** from ≥3-seed sd of A2), mean cost ≤ 0.35 × A2 and p95 ≤ 100 ms; a stratum with n ≥ 30 and harm beyond δ_f ⇒ that situation gets a **hard tier rule** (no gate). *Reject elasticity* (ship FW-only, an automatic compute win) iff E-C3 finds **no** stratum where the oracle camera beats FW-only beyond δ_f.

## 4. Audio in the shared embedding space (priority 4)

### 4.1 How the field puts audio into the token space
| system (id, status) | audio encoder → token rate | joined to LLM/vision by | training / alignment | note |
|---|---|---|---|---|
| Qwen2-Audio (2407.10759 R) | Whisper-class encoder | LLaVA-style projector → LLM | pre-train + SFT | speech + sound |
| **Qwen2.5-Omni** (2503.20215 R; rate -S) | native audio encoder **25 Hz** | **TMRoPE**: absolute-time positions align audio with video | joint multimodal | Thinker–Talker |
| **Qwen3-Omni** (2509.17765, -S) | **AuT, 12.5 Hz (≈80 ms/token)**, trained on very large audio (snippet: 20 M h) | TMRoPE | joint | 30 B-A3B; R-A: min memory 68.7 GB |
| Gemma 3n (Google dev blog, -S) | **USM-based, 160 ms/token (6.25 tok/s)** | native | native | R-A: Gemma-4 E2B/E4B 305 M USM-Conformer, 40 ms chunks, Thor rows exist (`INHERITED`) |
| Phi-4-multimodal (2503.01743, -S) | 3 conv + 24 conformer, ×8 subsampling ⇒ **80 ms/token** | **Mixture-of-LoRAs** (title -S); language backbone frozen while modality LoRAs train — RECALLED | modality LoRAs | closest to "add a modality, keep the LLM" |
| **video-SALMONN** (2406.15704) / **2** (2506.15220) | Whisper-class + visual | **MRC Q-Former**; **diversity loss + unpaired audio-visual mixed training** to stop modality dominance; v2 keeps backbone + both encoders **frozen** | -S | AV-LLMs favour vision (2604.14129 title) |
| ImageBind (2305.05665 R), LanguageBind (2310.01852 R), CLAP (2211.06687 R) | audio tower | **contrastive binding** to image (ImageBind), language (LanguageBind), text (CLAP) | InfoNCE | no LLM in the loop |
| BEATs (2212.09058 R), AudioMAE (2207.06405 R), Whisper (2212.04356 R) | general-audio SSL / ASR | frozen features + projector | — | Whisper is speech-centric |
| **Modality-Gated Deep Adapters** (2609.26182, **title only**) | — | *"adding a modality to a frozen embedding model with exact preservation"* | **content UNVERIFIED** | exactly HiCAP's shape |

**Recommended pattern for a frozen state encoder — a gated *residual* audio path, not audio tokens inside the LLM.** frozen audio tower (BEATs-base-class, 90 M, or an ASR-class encoder if the chosen VLM family ships none: R-A lists Qwen3-ASR-0.6B's 180 M AuT, Parakeet-0.6 B) → 2-layer perceiver to **k = 8 tokens per 2-s window** →
**e′ = e + σ(g)·h(a, e)**, `g` initialised so σ(g)≈0. **Audio-null ≡ the original embedding bit-for-bit**, which preserves parity and gives the exact-preservation property of the title above (**mechanism UNVERIFIED — I saw only the title**). Ablation: soft audio tokens inside the LLM (+25 tokens ≈ +0.085 TFLOP/tick ≈ +6 % of C1).
**Alignment to "the same space as language and vision":** train `h` so e′(image, audio) ≈ the frozen VLM's own embedding of (image, *a caption of the sound event*) — `L_align = 1 − cos(e′, e_teacher(image, "siren approaching from behind, ~40 m"))` — the teacher is the frozen text path, so audio is tied to language without audio–text LM training; the caption comes from the **event generator's metadata** (class, bearing, distance bin, approaching), *not* from the driving label, which is what keeps the echo out (§4.3).
Optional pre-alignment: InfoNCE(audio embedding, VLM text embedding of the caption) on public caption sets (AudioCaps/Clotho/WavCaps — sizes not verified).

**What a frozen audio tower loses:** (i) **direction of arrival** — mono log-mel discards inter-channel phase/level; needs a multi-mic front-end (GCC-PHAT / intensity-vector features, SELD-style) *before* the tower; (ii) **fine pitch trend / Doppler** at 128-bin resolution; (iii) **speech-centric features** (Whisper) suppress non-speech; (iv) **temporal context** — the tower sees a window, not a stream; (v) **ego-noise invariance** is not in the pre-training data.

### 4.2 What audio is actually useful for driving
| sound | driving relevance | published detectability | latency need | failure modes |
|---|---|---|---|---|
| **emergency siren** | yield / pull over / hold at a junction; the occluded-EV case vision cannot see | 2109.14797 (-S): **99.16 % recall** for presence, direction error **median 9.64° / mean 19.18°**, distance error **median 9.30 m / mean 10.58 m within 10–50 m**, two roof-rear microphones, **<50 ms** model latency; degrades **beyond ≈70 m**. AVNet (2609.16535): audio-visual transformer + KD, **degrades gracefully when either modality is absent**. 2507.01563: efficient CNNs on embedded HW (no numbers seen). AudioSet-EV + a siren/road-noise dataset (Sci. Data 2022) exist (-S) | ≥1 s integration (siren cycle) | **wind noise** at speed (one patent-text snippet: only sirens >95 dB survive — **UNVERIFIED**), cabin insulation, urban multipath, own-vehicle occlusion |
| **horn** | someone signals — *whom* is ambiguous | AudioSet/UrbanSound8K/ESC-50 have horn classes (RECALLED) | 0.2–1 s event | needs bearing; frequent false alarms |
| **tyre squeal / skid / impact** | imminent-collision confirmation | not found this session | lead time ≲0.5 s ⇒ **reactive, marginal** | — |
| **reversing beeper, crossing bell** | low-speed near-field / rail crossing | not found | seconds | — |
| **speech** | passenger/police commands | EchoVLA (2601.12142, -S) **synthesised** speech on nuScenes; assumes clean audio, **not tested under traffic/wind** | 1–3 s | noise, accent, intent ambiguity |
| **road/tyre noise ("hearing the road")** | surface/friction prior | **no source found** ⇒ `UNVERIFIED` | slow | ego-noise confound |

**Latency arithmetic** (`ESTIMATED`; speeds assumed): a siren audible at 70–100 m and an EV overtaking at +10 m/s gives 7–10 s; an oncoming/crossing EV at 40–60 m/s closing gives **1.2–2.5 s** ⇒ integrate ≈1 s, decide within 0.3–0.7 s ⇒ **audio state refreshed at ≥2 Hz**; HiCAP's 10 Hz tick can consume a 0.5-s-hop audio state. Audio is a **prior for behaviour and risk**, not a sensor of last resort.

### 4.3 Data reality — what carries audio, and what a test can and cannot show
| corpus / tool | audio? | evidence |
|---|---|---|
| **PhysicalAI-AV** | **ABSENT** in the feature inventory | `pai_features.csv` = **36 features** (3 label, 6 calibration, 7 camera, 1 lidar, 19 radar); `pai_tree_l1.json` dirs = calibration/camera/labels/lidar/metadata/radar/**reasoning**; `pai_tree_l3_sample.json` = the same 36; card sensor sentence `pai_card.md:333` ("multi-camera (7) … LiDAR (1) … radar (up to 10). Ego motion, calibration, and machine labels"); camera table `PHYSICALAI_FEATURE_PROBE.md:66-76`, `pai_card.md:382-402` ("visual RGB data (i.e., videos)"); `grep -i 'audio|sound|micro|acoust|speech|siren|horn'` over probe md/card/csv/json = **0 real hits** (the only matches are the substrings "dynamic"/"economic" via `mic\b`, `MEASURED (ours)`); sibling R-A (`RA_vlm_backbones.md`) reports the same. **Not yet probed: the mp4 container's own stream list ⇒ E-A0 (`ffprobe` one file).** |
| nuScenes | ABSENT | `nuscenes-devkit`@b40adc4 `docs/schema_nuscenes.md:198`: `modality: {camera, lidar, radar}`; EchoVLA **synthesised** audio for it |
| comma2k19 | ABSENT | `README.md:2` sensor list (camera, GPS, thermometers, IMU, raw GNSS, CAN); files are raw `video.hevc` (an elementary stream carries no audio track) |
| Waymo Open / Argoverse 2 | no audio reference | `waymo-open-dataset`@99a4cb3, `av2-api`@b7321d1: 0 hits for audio/microphone/sound/acoustic over md/py/proto/yaml/json (README does not enumerate modalities ⇒ `PUBLISHED`-negative by absence only) |
| **AlpaSim** | **strips audio** | `alpasim`@affc2ea `src/eval/src/eval/aggregation/main.py:296`: ffmpeg `"-an",  # Remove audio`; that is the only hit in the repo |
| **CARLA** | **no native sound** | GitHub issues #442, #1787 (-S); community add-ons: FMOD `carla-audio-engine`, DReyeVR engine sounds |
| audio-visual driving data that *does* exist | yes | **A3CarScene** (Data in Brief 2023, -S: 8 microphones in/around the cabin + 2 dashcams; size/labels/ego log **UNVERIFIED**); the siren/road-noise dataset; AudioSet-EV |

**Synthetic augmentation recipe.** Sources: AudioSet-EV / ESC-50 (2,000 × 5 s) / UrbanSound8K (8,732 × ≤4 s) siren + horn classes (sizes RECALLED). Render from **world state**, not from labels: take an actual agent track from `obstacle.offline` (automobile/bus/heavy_truck), *designate* it the emitter, and apply delay r/c, gain ∝ 1/r (−6 dB per doubling, r ≥ 2 m), **Doppler f′ = f·c/(c − v_r)** (c = 343 m/s; ±6 % at 20 m/s) as a time-varying fractional delay, air-absorption low-pass, ITD Δt = d·sinθ/c and ILD across a virtual mic array, and an **ego-noise floor as a function of speed** mixed at a sampled SNR.
**Modality dropout:** p(audio dropped) = 0.3, plus "ego-noise only" and "shuffled audio" samples (video-SALMONN's unpaired-mixed-training logic).

⛔ **The label-consistency trap.** A real PhysicalAI ego trajectory never yields to a *synthetic* siren, so an audio-augmented clip has **no valid driving label** unless the expert was already yielding/stopping — and then audio is present *only where the label is "yield"*, which is a **label echo**. Therefore: (a) train the audio path on **event metadata** (class, bearing, distance, approaching) only; (b) test *behavioural* effect **in closed-loop sim**; (c) counterfactual trajectory synthesis is a later, model-based step.

**Controls that prove audio is used (pre-registered, both outcomes committed):**
| control | construction | if audio is used | if it is an echo/shortcut |
|---|---|---|---|
| **audio-null** | silence + speed-matched ego-noise floor | siren-window metrics move toward the no-audio arm | no change ⇒ audio ignored |
| **audio-shuffle** | audio from another window, matched on speed bin, noise floor and event rate | true − shuffle = audio-specific benefit; shuffle ≈ null | shuffle ≈ true ⇒ the effect rides on timing/marginal statistics |
| **bearing flip** | mirror L/R channels | predicted pull-over side / gap choice flips | no flip ⇒ not using direction |
| **level dose-response** | attenuate 0/−10/−20/−30 dB | monotone decay | non-monotone ⇒ shortcut |
| **onset shift** | move siren onset ±1–2 s from the label-consistent time | effect follows the audio | effect stays at the label time ⇒ leak |
| **audio-only label probe** | train an audio-only classifier for the *driving* label on the injected clips (the generator saw no driving label) | ≈ chance | above chance ⇒ audio carries the label (the vision-only-rule leak test in a new costume) |
| **no-vehicle / no-siren** | siren with no visible EV; visible EV without siren | still yields on siren-only (occluded case) | — |

**What can be tested on the current corpus:** (T1) **parity** — audio-null reproduces the original embedding bit-exactly (deterministic); (T2) **event heads** — classification/localisation on public audio-visual sets and synthetic events; (T3) **usage controls on synthetic-event-injected PhysicalAI clips** — a *capacity/plumbing* test that the decision path can read audio when it is the only informative channel, **not driving skill**; (T4) **no-regression** — noisy or absent audio leaves the four-family metrics unchanged (paired). **What cannot:** any claim that audio *improves driving* — that needs real audio-visual logs with the ego's response, or closed-loop sim with an acoustic layer. **Sim path:** AlpaSim/CARLA give per-tick actor poses; compute the acoustic layer **offline** from them and feed it as a recorded stream (circular — our own physics — so validate detection realism on real recordings first).

### 4.4 Audio compute on Thor (`ESTIMATED`; `rd_cost_model.py`)
| tower | params | per 2-s window | ms @50 TFLOP/s · weight-stream floor (273 GB/s) | note |
|---|---:|---:|---|---|
| BEATs-base-class | 90 M (RECALLED) | 2·90 M·99 patches + attention = **18.2 GFLOP** (49.6 patches/s) | 0.36 · 0.66 | at 2 Hz refresh: 36 GFLOP/s |
| Whisper-small encoder, **fixed 30 s window** | 88 M | **347 GFLOP** | 6.9 | ⚠️ do not run un-truncated; truncated to 2 s ≈ 18 GFLOP |
| AuT-class | ≈0.65 B (**UNVERIFIED**) | 2·0.65 B·25 tokens = 32.5 GFLOP | — · **4.8 (memory-bound)** | weights 1.3 GB |
| LLM-side audio tokens (ablation) | — | 25 tokens = 85 GFLOP; 8 pooled = 27 GFLOP | ≈+6 % / +2 % of C1's 1.34 TFLOP | |
⇒ audio ≈ **0.4–5 ms per 0.5-s refresh (<2 % of a 10 Hz FW stream)**; extra weights 0.2–1.3 GB. **Compute is not the constraint; data and validation are.**

## 5. Toy — `rd_camera_gating_toy.py` (numpy, ≈90–97 s single-thread, bit-deterministic: a rerun reproduced the JSON exactly; `MEASURED (toy)` — mechanisms under *my chosen assumptions*, not driving evidence)

**World.** 7 cameras (FW FT CL CR RL RR RT), 7 sticky situations (stay 0.97; mean dwell ≈33 ticks) each with **exactly one** camera carrying the decision bit z (proceed/hold; amplitude 2.0 ⇒ ≈97.7 % from the right camera); FW leaks z weakly (0.15; 0.7 in follow-far) and carries a weak situation code
(radius 1.5). Decision y = [z>0]; **miss** = true hold predicted proceed. All constants are `ASSUMPTIONS` in the script and JSON. **Gate** MLP (15→32→8) trained by **full-information cost-sensitive learning** on per-option errors of a camera-dropout-curriculum head (§3.6). Policies are evaluated tick-by-tick on 300 held-out episodes × 200 ticks with a closed-loop previous-option cue and a **cold-start penalty** (a newly enabled camera's signal scaled 0.3/0.6 for 2 ticks).
Reproduce: `python rd_camera_gating_toy.py` (`--quick` = 1 seed, no sensitivity, ≈30 s). Intervals: **episode-cluster bootstrap (B = 2000) over the 300 test episodes, per seed, paired for two arms**; variation across training seeds is shown as sd (n = 3). ⚠️ This toy scores **one binary decision**, not the four metric families; it is a mechanism study, never "the result".

### 5.1 Headline — cost vs decision quality (mean of 3 seeds; sd in brackets)
| policy | cost (camera-units) | decision acc | miss rate | note |
|---|---:|---:|---:|---|
| fixed FW only | 1 | 0.732 (0.002) | 0.257 | |
| fixed AR1-default 4 cams (CL, FW, CR, FT) | 4 | 0.830 (0.007) | 0.158 | fails every rear situation by construction |
| fixed AR2 trajectory profile (all but rear-tele) | 6 | 0.899 (0.005) | 0.093 | fails pull-over by construction |
| fixed all 7, head H=32 | 7 | 0.936 (0.001) | 0.062 | |
| fixed all 7, head H=128, 2× epochs (**honesty control**) | 7 | 0.942 (0.001) | 0.055 | routing difficulty, not information |
| **ceiling**: all 7 + true situation given to the head (privileged, reference only) | 7 | **0.976** (0.001) | 0.023 | |
| oracle gate, warm | 1.65 | 0.974 | 0.025 | |
| oracle gate, cold start | 1.65 | **0.967** (0.001) | 0.031 | |
| oracle gate, cold start + 2-tick prefetch | 1.68 | 0.972 | 0.027 | prefetch recovers +0.45 pt |
| **learned gate λ*=0.08** (asym. hysteresis, cold start) | **1.635 (0.003)** | **0.951 (0.003)** | 0.047 | paired Δacc vs fixed-all(H32) **+1.51 [1.17, 1.81] / +1.39 [1.10, 1.68] / +1.80 [1.50, 2.08] pt** (per seed); Δmiss −1.85 / −1.66 / −1.09 pt |

⚠️ **Do not read "+1.5 pt over fixed-all" as "fewer cameras are better."** The fixed-all head must *learn to route* from noisy cues (ceiling 97.6 % vs 93.6 %: a 4-pt routing-learning gap); the honest comparison is **gated vs oracle gate −1.6 pt** (gate errors) and **vs ceiling −2.5 pt**. λ* was picked from the *test* frontier by a knee rule (optimistic by ≲0.2 pt of plateau spread); the per-λ rows below are the unselected numbers.

### 5.2 The accuracy–cost frontier (gate, asymmetric hysteresis, cold start; mean of 3 seeds)
| λ | cost | acc | miss | switches/100 ticks | gate = oracle option | ticks on ALL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.75 | 0.940 | 0.054 | 6.7 | 0.47 | 0.36 |
| 0.02 | 2.02 | 0.951 | 0.047 | 4.5 | 0.78 | 0.04 |
| 0.04 | 1.74 | 0.953 | 0.044 | 4.2 | 0.86 | 0 |
| **0.08** | **1.64** | **0.951** | 0.047 | 3.4 | 0.92 | 0 |
| 0.16 | 1.49 | 0.915 | 0.083 | 2.8 | 0.79 | 0 |
| 0.32 | 1.25 | 0.825 | 0.174 | 1.5 | 0.58 | 0 |
A flat plateau (1.6–2.0 units, 95.1–95.3 %) then a **cliff**: −3.6 pt at 1.49, −12.6 pt at 1.25. λ = 0 buys nothing (the gate chooses ALL 36 % of the time and is *worse*: the ALL head routes worse than a right single camera).

### 5.3 Cost of a wrong camera — head accuracy by (true situation × camera set actually served; mean of 3 seeds)
| true situation \ served | FW | +FT | +CL | +CR | +RL | +RR | +RT | ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cruise (FW) | **0.977** | 0.930 | 0.968 | 0.970 | 0.963 | 0.956 | 0.966 | 0.933 |
| follow-far (FT) | 0.758 | **0.974** | 0.748 | 0.749 | 0.743 | 0.742 | 0.747 | 0.935 |
| junction-L (CL) | 0.537 | 0.537 | **0.972** | 0.528 | 0.517 | 0.531 | 0.527 | 0.930 |
| junction-R (CR) | 0.521 | 0.507 | 0.526 | **0.975** | 0.522 | 0.511 | 0.512 | 0.936 |
| lane-change-L (RL) | 0.547 | 0.532 | 0.520 | 0.541 | **0.971** | 0.519 | 0.538 | 0.907 |
| lane-change-R (RR) | 0.556 | 0.536 | 0.533 | 0.544 | 0.538 | **0.969** | 0.534 | 0.881 |
| pull-over (RT) | 0.536 | 0.537 | 0.532 | 0.526 | 0.532 | 0.522 | **0.970** | 0.910 |
A wrong camera costs **≈0 in cruise, ≈22 pt for the tele case, ≈41–46 pt (chance) whenever the needed cross/rear camera is off**. Injecting gate error into an oracle (warm cameras, wrong *single* camera): ε = 5 % → −1.3 pt; 10 % → −2.6; 20 % → −5.3; 30 % → −7.9; 50 % → −13.1 pt (miss 0.025 → 0.150). Gate confusion (λ*): **cruise served correctly 91.5 %, follow-far only 69.9 % (28.3 % sent to FW-only)**, all cross/rear situations 96.6–99.2 % — the errors sit on the *tele-vs-wide ambiguity*, not on the rare junctions.

### 5.4 Which cues may the gate use? (`gate_*` at λ*, cold start on; mean of 3 seeds)
| gate inputs | acc | cost | gate = oracle option | mean wrong-run length (ticks) |
|---|---:|---:|---:|---:|
| ego cues + FW features, **no** previous token (≈ G-E) | 0.929 | 1.72 | 0.87 | 2.0 |
| + previous option, **closed loop** (≈ G-E + own history) | 0.951 | 1.64 | 0.92 | **5.1** |
| + previous option, open loop (noisy-oracle previous situation) | 0.954 | 1.69 | 0.94 | 1.8 |
| supervised situation classifier (CE on label-derived situation, cost-blind) + previous | 0.958 | 1.69 | 0.93 | — |
| **vision-only** (FW features only) — **G-V** | **0.812** | 1.78 | 0.53 | 3.2 |
| **vision-only + own previous option, closed loop** | **0.804** | 1.72 | 0.41 | **49.2** (62 / 45 / 41) |
* The previous-token cue is worth **+2.3 pt** but **closed-loop feedback lengthens error runs 2.5×** (5.1 vs 2.0 ticks); with weak visual evidence it **locks in**: a wrong camera set held ≈5 s.
* **How much situation information must FW carry for a vision-only gate?** (seed 0, λ = 0.04; FW situation-code radius, base 1.5):
| FW code radius | fixed-all acc | ego-cue gate + prev | vision-only + prev | vision-only | wrong-run (vision-only + prev) |
|---:|---:|---:|---:|---:|---:|
| 1.5 (base, 3 seeds) | 0.936 | 0.951 | 0.804 | 0.812 | 49 ticks |
| 3.0 | 0.952 | 0.966 | 0.952 | 0.901 | 11 ticks |
| 4.5 | 0.956 | 0.968 | **0.966** | 0.950 | 2.9 ticks |
G-V matches G-E only when the always-on FW embedding separates the situations ≈3× better than in the base world. **That is an empirical quantity of the frozen VLM on real frames — E-G1 (§6) measures it before anything else is built.** `INHERITED` warning sign: the banked scenario-classifier ranking in `CLAUDE.md` has the image-only arm weakest (`head_ego 0.0697 > head_img_ego 0.0525 > head_img 0.0376`; metric not restated there).

### 5.5 The value of hysteresis (gate at λ*; mean of 3 seeds; cold start on / off)
| policy | acc (cold on) | acc (cold off) | cost | switches/100 ticks |
|---|---:|---:|---:|---:|
| none (raw argmax) | 0.937 | 0.945 | 1.54 | 4.10 |
| EMA 0.6 on probabilities | 0.938 | 0.943 | 1.55 | 2.19 |
| symmetric dwell (2 ticks, margin 0.10) | 0.932 | 0.937 | 1.54 | 2.24 |
| **asymmetric: up/sideways 1 tick, down 3 ticks, margin 0.10** | **0.951** | 0.959 | 1.64 | 3.44 |
| asymmetric, down 8 ticks | 0.951 | 0.958 | 1.74 | 3.51 |
Versus none (cold on): **EMA halves flicker at no accuracy change (+0.02 pt)**; **symmetric dwell halves it at −0.6 pt** (it delays needed escalations *and* equal-cost swaps); the asymmetric keep-alive **gains +1.4 pt for +0.10 units (+6 %)** with −16 % switches — its gain is mostly *not letting a needed camera drop for one noisy tick*. Cold start costs 0.55–0.8 pt under every variant. **No paired CI was computed between hysteresis variants — treat as directional.** The oracle itself switches 2.4/100 ticks.

### 5.6 Escalation and hedging under uncertainty — **negative**
* The cost-trained gate is **over-confident**: max-probability > 0.99 on 90 % of ticks (88.7/90.2/92.1 %), so a confidence trigger barely fires; the calibrated situation head is better-behaved but **still gains nothing**.
* Escalate to ALL when max-prob < τ (situation head): τ = 0.5 / 0.7 / 0.8 / 0.9 → cost 1.81 / 2.40 / 2.94 / 4.14, acc 0.957 / 0.956 / 0.953 / 0.943 vs **0.958 at 1.69 unescalated** — cost +7…+145 % for **−0.03…−1.5 pt**. Cost-gate: cost 1.64→3.86 (τ 0.99), acc 0.951→0.943.
* Hedge (also serve the runner-up camera, cost +1): situation head τ = 0.6 / 0.8 / 0.9 / 0.99 → acc 0.958 / 0.958 / 0.956 / 0.947 for cost 1.70 / 1.71 / 1.74 / 1.87 — **≤ +0.01 pt** (cost-trained gate: ≤ +0.06 pt at +0.004…+0.05 units).
Reason (from §5.3/5.5): the residual errors are **confident** (follow-far ↔ cruise) or **transition lag**, neither of which a confidence threshold sees; and ALL routes worse than a right single camera. ⇒ use **scheduled probes and keep-alive**, not confidence escalation.

### 5.7 Limits — what this toy cannot say
One binary decision, memoryless head, **exactly one informative camera per situation** (real junctions need both cross cameras; real cameras overlap), uniform per-camera cost, an ad-hoc cold-start model, sticky Markov situations, a **chosen** FW situation-code SNR (the §5.4 sensitivity shows the conclusion about vision-only gates depends on it), no fisheye/tele geometry, λ* selected on test, one head architecture. Absolute accuracies mean nothing; **signs and orderings** (cliff below ≈1.6 units; asymmetric > symmetric hysteresis; escalation useless; closed-loop lock-in; vision-only sensitivity) are the deliverable, and each is a **`HYPOTHESIS` for real data** until E-G1/E-C3 run.

## 6. Recommendation, ranked risks, cheapest discriminating experiments (priority 6)

**Camera/token policy (v1):** FW always @160 tokens, 1 frame/tick · **tiers T0–T2** (§3.1), extras at **64 tokens** · **G-V gate** (FW embedding + own keep-alive state; G-E as ablation; G-T only if it passes §3.2) trained by full-information cost-sensitive learning **plus** a calibrated situation head, situation-criticality-weighted ·
**asymmetric keep-alive hysteresis (1 up / 3 down, margin 0.10), 2-tick shadow-mode prefetch on intent** · **2 Hz tower-only slow lane** for the rest · budget-conditioned heads + camera-dropout curriculum · Matryoshka **dims as a free option**, nesting over configs via the §3.4 loss · cached per-config embeddings (§3.6).
Target cost: **mean ≤ 1.7 camera-units; TRT ≈ 20–30 ms (2B-class) / 37–50 ms (4B-class) for T1–T2**, all-seven probe ≤ 40 / 68 ms.
**Audio path (v1):** frozen BEATs-base-class tower (≈90 M, 0.18 GB fp16) → 2-layer perceiver (k = 8 tokens / 2 s, ≈15 M trainable) → **zero-initialised gated residual on the pooled state**, 0.5-s hop, modality dropout 0.3; **alignment to a caption teacher** (§4.1); event heads on metadata; **no driving-skill claim** until a closed-loop acoustic sim exists. Cost ≈ 0.4–5 ms per refresh.

**Ranked risks and the cheapest experiment for each (proposals; none run; both outcomes committed before data):**
| # | risk | cheapest discriminating experiment | outcome A | outcome B |
|---|---|---|---|---|
| 1 | **Extra cameras do not help our decisions/metrics** (elasticity solves a non-problem) | **E-C3** oracle-camera value: side-study of ~30 dense chunks; add **front-tele first (49 GB)**, then cross pair (143 GB); paired four-family on 300 windows per stratum | oracle camera beats FW-only beyond δ_f in ≥1 stratum ⇒ tier only those strata; buy that camera's parity cache | no stratum ⇒ **retire elasticity, ship FW-only** (a compute win by default), spend nothing more on caches |
| 2 | **A vision-only gate is too weak** (and closed-loop feedback locks in) | **E-G1**: frozen-VLM FW embedding → tiny MLP → 7-way situation (labels from ego + `obstacle.offline`), per-class recall + calibration on val-40; then run the closed-loop lock-in test (wrong-run length) | per-class recall ≥ 0.9 on every tele/cross/rear class **and** closed-loop wrong-run ≲3 ticks (toy r = 4.5: option-match 0.93–0.97, wrong-run 2.9) ⇒ G-V viable | weak ⇒ PI must rule on ego cues (G-E) or the design falls back to **hard, speed/route-driven tier rules** |
| 3 | **Embedding drift** from pruning/config change breaks heads | **E-R1**: cos(e_full, e_config) and head loss under config shift on cached embeddings (0 new GPU beyond §3.6 cache) | drift ≲ head noise ⇒ frozen heads OK | large ⇒ budget-conditioned training mandatory; drop training-free pruning |
| 4 | **Thor latency unknown** for any HiCAP config | **E-L**: TRT-Edge-LLM builds of the chosen VLM at C1/C2p/C4p/C7p, p50/p99 ≥ 50 iterations, `torch.cuda.max_memory_allocated()` only | T1–T2 p95 ≤ 100 ms with margin ⇒ tiers as designed | not ⇒ smaller VLM or fewer tokens/extra; re-derive λ |
| 5 | **Audio track already in the PhysicalAI mp4s** (we would be wrong to say ABSENT) | **E-A0**: `ffprobe` one chunk mp4 (~10 min) | none ⇒ absence confirmed (three-location + container) | present ⇒ this whole §4.3 changes; audio becomes trainable on the parity corpus |
| 6 | **Audio echo/shortcut** (a synthetic label proxy) | the §4.3 control table on injected events; **audio-only label probe** first | probe ≈ chance and controls behave ⇒ plumbing validated (not skill) | above chance ⇒ audio generator leaks the label; regenerate from state |
| 7 | Cold-start / flicker dominate | shadow-mode prefetch ablation: 0 / 1 / 2 / 3 ticks | ≥2 ticks recover ≥ the toy's +0.45 pt ⇒ keep | no gain ⇒ drop prefetch |
| 8 | Front-tele geometry breaks parity/tooling | calibrate f-theta for the 30° camera; crop, not the FW cylindrical canvas | clean crop at ≥2× the FW angular token density | distortion ⇒ front-tele deferred |

## 7. Implications for HiCAP (≤ 12)
1. **Measure value before building elasticity (E-C3).** No published evidence isolates extra-camera value by manoeuvre; our own open-loop finding says ego kinematics dominate. A null result makes FW-only the answer *and* the cheapest.
2. **Frames per camera outweigh camera count**: AR1's 4×4 is ≈19.5 TFLOP/tick; HiCAP's 1 frame × 1–3 cameras is 1.3–3.6.
3. **Pre-tower savings beat post-tower pruning**: the first extra camera's LLM tokens are nearly free (weight-streaming-bound), its tower pass is not; batching cameras saves only ≈18 % of ViT time (R-A).
4. **Peripheral cameras at 64 tokens make even all seven fit** a 100 ms tick on 2B-class TRT (≈40 ms) — unbudgeted seven do not fit 4B (≈120 ms).
5. **The gate must be cheap, asymmetric and cost-aware**: full-information cost-sensitive training works because every camera is cached offline; weight errors by situation (a wrong camera is ≈0 pt in cruise, ≈41–46 pt at a junction); report **per-class recall**, never accuracy (DriveMoE's default class is 71.8 %).
6. **Use asymmetric keep-alive hysteresis and shadow-mode prefetch**; symmetric dwell (−0.6 pt) and confidence-triggered escalation (no gain, +2…+145 % cost) measured *negative*, EMA neutral; use a slow-lane probe for the unknown unknowns.
7. **Cue admissibility decides the design**: G-V (vision-only) is the only gate admissible under the 2026-08-03 binding, and closed-loop own-choice feedback can lock in for seconds; the previous tactical token (G-T) is a selection-type back door to the goal head and needs the config-invariance and gate-shuffle tests. **Escalation 1 needs a PI ruling.**
8. **Matryoshka dims are free but worthless for compute here**; nest over *cameras/tokens* with config-conditioned heads and a full-camera distillation target — the VLM stays frozen, so the nesting lives in pooler + heads (Matryoshka-Adaptor-style).
9. **Extra cameras are a data project, not a config flag**: 321 GB (front-tele) … 2.55 TB (six) transient, ≈9 h build for six on a pod; a 30-chunk side-study (49–388 GB) is the cheap first step but is not the parity corpus.
10. **Audio is cheap to compute and impossible to validate today**: no audio in PhysicalAI/nuScenes/Waymo/AV2/comma2k19, AlpaSim strips it, CARLA has none; run E-A0, build the gated-residual path with a bit-exact audio-null, and claim plumbing only.
11. **Audio must be aligned through the frozen text path and generated from world state, not from labels** — otherwise a result is an echo; the seven-control table (§4.3) is the acceptance test, and the audio-only label probe is the first gate.
12. **Every elasticity claim is reported in the four metric families, paired, stratified, with cost** — the toy's single binary decision is a mechanism study, not a result.

## 8. Access log, citation status, gaps (fail loud)
**Budget used:** WebSearch **30 / 30**; WebFetch 4 attempts (`arxiv.org` and `emergentmind.com` **egress-blocked**; `github.com/NVlabs/alpamayo1.5/issues/4` and `github.com/Thinklab-SJTU/DriveMoE` read). Shallow clones (scratchpad, not staged): `alpamayo`@11a0e01, `alpamayo1.5`@36aeb4c, `alpamayo2`@5e7975f, `DriveMoE`@e39df2f, `alpasim`@affc2ea, `comma2k19`@4c7f1a6, `nuscenes-devkit`@b40adc4, `waymo-open-dataset`@99a4cb3, `av2-api`@b7321d1, `navsim` (README/docs only).
**Not read:** every arXiv PDF (blocked) — abstract-level numbers come from search snippets; DriveMoE's ablation tables; the NAVSIM PDMS table; EMMA/OpenDriveVLA/ORION/Senna/DriveVLM per-camera tokens; A3CarScene's contents; the content of 2609.26182; sibling streams: `RA_vlm_backbones.md` read in part (§1.3–1.5, §2 table, §5, §6 rows), P1/P2/P3 not read.

| id | claim used | status |
|---|---|---|
| 2511.00088 (AR1) | 448×280→160 tok; triplane ≈288 tok/timestep, 3.9×; Flex 8–45 tok/frame | -S snippet; code facts -P |
| 2506.12251 (triplane) | ≤72 % fewer tokens, ≤50 % faster, same open-loop accuracy | -S abstract |
| 2505.16278 (DriveMoE) | DS 74.22 vs 55.85, SR 48.64 vs 30.00, 260/240/700 ms; label rules | -S (numbers, rules); **router code -P** |
| 2609.03602 (SV-WAM), 2604.19145 (ST-Prune), 2606.27660 (MVPruner), 2508.13305 (Prune2Drive) | as §1.1 | -S |
| 2412.01818 (VisPruner) | FastV 75.3 / SparseVLM 88.5 / VisionZip 94.1 % @64 tokens | -S snippet |
| 2502.11494 (DART) | 88.9 % pruned, 1.99× / 2.99× | -S |
| 2205.13147 (MRL), 2405.17430 (M3), 2407.20243 (Matryoshka-Adaptor) | as §2 | -S |
| 2510.09608 (StreamingVLM), 2408.16730 (VideoLLM-MoD), 2609.15131 (AdaVSkip), 2503.20384 (MoLe-VLA), 2106.02034 (DynamicViT) | titles/mechanisms | -S |
| 2608.29005 (DriveDegrade), 2605.18059 (Bench2Drive-Robust), 2603.09737 (M²-Occ), 2406.15349 (NAVSIM) | as §1.3–1.4 | -S; NAVSIM sentence's source paper unidentified |
| 2509.17765 (Qwen3-Omni), 2503.01743 (Phi-4-Mini/multimodal), 2406.15704 & 2506.15220 (video-SALMONN 1/2), Gemma 3n blog | token rates, mechanisms | -S |
| 2109.14797, 2609.16535 (AVNet), 2507.01563, 2301.12808, 2602.07668, 2601.12142 (EchoVLA), 2603.05058 | as §4.2 | -S (numbers for 2109.14797 from snippets) |
| 2609.26182 (Modality-Gated Deep Adapters) | **title only** | mechanism **UNVERIFIED** |
| **RECALLED ids (not re-seen):** FastV 2403.06764, SparseVLM 2410.04417, VisionZip 2412.04467, ToMe 2210.09461, PruMerge 2403.15388, FitPrune 2409.10197, MoD 2404.02258, Qwen2-Audio 2407.10759, Qwen2.5-Omni 2503.20215, ImageBind 2305.05665, LanguageBind 2310.01852, CLAP 2211.06687, BEATs 2212.09058, AudioMAE 2207.06405, Whisper 2212.04356 | | **re-verify before citing** |
| OmniDrive 2405.01533, EMMA 2410.23262, OpenDriveVLA 2503.23463 | ids seen in results; token counts absent | UNVERIFIED numbers |

## 9. Deliverable manifest
| artifact | where it lives | staged? | single-location? |
|---|---|---|---|
| this report | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RD_cameras_tokens_audio.md` | yes | no (repo) |
| toy (numpy, ≈95 s, deterministic) | `repo:…/2026-09-29-hicap/rd_camera_gating_toy.py` | yes | no |
| toy result (3 seeds + sensitivity; all §5 numbers) | `repo:…/2026-09-29-hicap/rd_camera_gating_toy_result.json` | yes | no |
| cost-model arithmetic (§3.5, §3.7, §4.4) | `repo:…/2026-09-29-hicap/rd_cost_model.py` | yes | no |
| shallow clones of Alpamayo/DriveMoE/AlpaSim/dataset devkits | scratchpad only (public GitHub; re-clonable at the SHAs in §8) | no (not deliverables) | **yes — reproducible from SHAs** |
Nothing was committed or pushed; no `Keys.txt`, token, or pod was touched. **Integration flags:** escalations 1–3 in §0 need PI decisions; nothing here needs a merge.
