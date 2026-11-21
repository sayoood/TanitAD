<title>Qwen-Drive-1.0-4B teardown — it wins every open-loop board and loses the closed-loop one</title>

# Qwen-Drive-1.0-4B: a full teardown, and the table that inverts its own headline

`TanitAD Research Lab · Opponent Analysis · 2026-09-10 · LAB-RUN-011 · ⭐ PI-REQUESTED`
`Primary: arXiv 2609.00111v1 (retrieved 2026-09-10, 40 pp., 32,477,575 B, valid %%EOF) — FULL TEXT.`
`Qwen Team + Huazhong University of Science and Technology · 2026-09-02 · Apache 2.0 · weights on HF and ModelScope, code on GitHub.`
`Serves: guideline S-1 (T1 burden of proof) · the three-planner hierarchy directive · GS-4 · FS9-5 · FS9-3 · backlog rows 11, 12, 18, 22, 23, 32 · register debt D-12`
`Search log: raw/search_log.md`

---

## 0. THE HEADLINE FINDING, BEFORE ANYTHING ELSE

**Qwen-Drive-1.0 posts the best NAVSIM PDMS (90.7) and the best test-split WOD-E2E RFS (7.91) among its comparisons — and finishes at or near the BOTTOM of its own closed-loop table.**

**Table 7, verbatim — 916 AlpaSim scenarios, PAI-AV-NuRec v26.02:**

| Method | Params | Close encounter, all ↓ | at-fault ↓ | Off-road ↓ | Progress ↑ | AlpaSim all ↑ | AlpaSim at-fault ↑ |
|---|---|---|---|---|---|---|---|
| Alpamayo-R1 | 9.8 B | **19.0** | 6.0 | 17.0 | **67.0** | **0.36** | **0.58** |
| Alpamayo-1.5 | 9.8 B | 37.0 | 11.0 | 16.0 | 59.0 | 0.23 | 0.45 |
| DriveWAM | 15.0 B | 56.0 | **5.0** | **8.0** | 35.0 | 0.10 | 0.53 |
| SimWAM | 6.0 B | 35.0 | 22.0 | 19.0 | 62.0 | 0.22 | 0.30 |
| **Qwen-Drive-1.0-SFT w/ reasoning** | **5.0 B** | 38.0 | 12.0 | 24.0 | 54.0 | **0.16** | **0.27** |
| **Qwen-Drive-1.0-RL** | **5.0 B** | 41.0 | 11.0 | 12.0 | 48.0 | **0.16** | 0.37 |

⭐⭐⭐ **One model, one paper, four evaluation tiers, and the ranking REVERSES.** Best-in-class open-loop and pseudo-closed-loop; **2.25× worse than Alpamayo-R1 on the closed-loop AlpaSim score (0.16 vs 0.36)** and worst on at-fault (0.27). Published by the model's own authors, in the same table set, with every method *"reproduced under the same evaluation setting."*

⇒ **This is the strongest external evidence the programme has ever had for guideline S-1** (*"make T1 the burden of proof; any T0-only claim is unevaluated"*). It is no longer an argument from our own measurements plus Waymo's rhetoric — it is a published rank inversion inside a single opponent's result set.

⚠️ **And scale does not rescue the open-loop ordering either:** DriveWAM is **15.0 B**, the largest model in the table, and posts the worst close-encounter rate (56 %), the worst progress (35 %) and the worst AlpaSim-all (0.10).

---

## 1. ARCHITECTURE

**Base.** Qwen3.5-4B, a *natively multimodal* VLM. ⭐ **The VLM architecture is not modified at all** — two external modules read from the shared pathway. Total 5.0 B parameters *"excluding the LLM token embeddings."*

### 1.1 Input format — cheap, architecture-free, and directly stealable

- **View tags:** eight canonical directions, `<FRONT VIEW>`, `<FRONT RIGHT VIEW>`, `<RIGHT VIEW>`, `<BACK RIGHT VIEW>`, `<BACK VIEW>`, `<BACK LEFT VIEW>`, `<LEFT VIEW>`, `<FRONT LEFT VIEW>`. **Frame tags:** `frame: k`.
- ⭐ **Both use ordinary vocabulary tokens** — verbatim: *"require no additional special tokens or architectural modifications."*
- ⭐⭐ **Serialization order is task-dependent and load-bearing.** QA uses **frame-major** (`frame: 0 <FRONT> <FRONT RIGHT> … frame: 1 …`); planning uses **view-major** (`<FRONT VIEW> frame: 0 frame: 1 … <FRONT RIGHT VIEW> …`), because view-major *"places consecutive observations from each view adjacent in the token sequence and exposes temporal variation within that view, which is important for control in dynamic environments."*
- ⭐ **Asymmetric resolution:** historical images at **320p**, current images at **720p** — *"retains more spatial detail in the current observation while representing the motion history with fewer visual tokens."*

### 1.2 BEV perception head — an inspectable 3D probe

Single-frame surround-view, `Nv ∈ {6, 8}` cameras. **Two complementary feature streams:**

| stream | source | role |
|---|---|---|
| `F^v_i` | vision encoder, **before** the VLM | low-level appearance |
| `F^m_i` | image tokens **after** the full VLM | broader scene context, the semantic source |

- **Geometry path:** a depth-based lift-splat view transform maps `F^v` into a 3D volume. A lightweight depth net (residual blocks + atrous spatial pyramid) predicts a per-pixel categorical distribution over `N_d` depth bins ⛔ **without depth supervision**. `V(p) = Σ_i D_i(u_i,v_i,d_i) · F^v_i(u_i,v_i)`. The volume **retains the height dimension** for occupancy.
- **Semantic path:** a simple feature pyramid expands `F^m`; a query-based BEV transformer aggregates it onto the BEV plane, alternating self-attention over the BEV grid with deformable cross-attention. ⭐⭐ **Its queries are initialised from the height-collapsed `V̄`, which supplies an explicit geometric prior.**
- **Three branches from the fused `B`:** DETR-style deformable decoder (3D detection) · `B` expanded along height and fused with `V`, then a shallow 3D UNet (semantic occupancy) · UNet-style head (BEV map segmentation). `L_perc = L_det + L_occ + L_map`.
- ⭐⭐⭐ **The head shapes the encoder**, verbatim: *"During joint training, the perception losses propagate through `F^m_i`, providing an additional gradient path to the vision encoder alongside the direct path through `F^v_i`."*

### 1.3 Planning Expert — flow matching on cached VLM state

- **Conditions:** sensor images `s`, historical ego trajectory `τ_hist`, **current ego state `e`**, navigation instruction `n`, and an *optional* textual planning reason `r`.
- ⭐⭐ **It conditions on cached VLM keys and values** — the VLM is not re-run per denoising step.
- **Structure (Fig. 3b):** noisy trajectory tokens → 4× GQA layers + AdaLN, ×8; and 1× GQA + 3× GDN layers.
- **Output:** clean trajectory, **5 s at 10 Hz = 50 waypoints**, expressed in the current ego frame, with headings.
- **Sampler:** K = 10 deterministic Euler steps, endpoint parameterisation.

---

## 2. TRAINING — four stages, and the freeze pattern inside them

| stage | trained | frozen | objective | output |
|---|---|---|---|---|
| **1. Perception head pretraining** | BEV head only | vision encoder + VLM | `L_perc` | initialised head |
| **2. Perception + VQA joint** | **BEV head + vision encoder + VLM** | — | `L_perc` on perception samples, `L_ntp` on VL samples | the shared representation |
| **3. Planning Expert pretraining** | Planning Expert only | vision encoder + VLM | `L_plan` only, **no text objective** | **Qwen-Drive-1.0-SFT** |
| **4. Reinforcement learning** | Planning Expert only | vision encoder + VLM | task-level rewards | **Qwen-Drive-1.0-RL** |

### 2.1 ⭐⭐⭐ Stage 2 states our own measured problem, and unfreezes to fix it

Verbatim: *"head-only training yields limited perception performance, **indicating that the pretrained representations do not directly expose sufficient 3D structure for driving perception.** The head-only setting thus probes how readily the pretrained features support explicit 3D prediction."*

**That is our v6/v7 readout-geometry ceiling, stated by someone else, and their remedy is to unfreeze the trunk.** Engineering details worth copying:
- **The BEV head runs at 20× the VLM's learning rate.**
- Mixed minibatches; **dummy inputs feed inactive branches** to keep the computation graph consistent across distributed workers, and the dummy outputs are excluded from the loss.

### 2.2 Stage 4 RL — how to get a policy gradient out of a deterministic flow

The sampler is deterministic once the noise is drawn, so there is no transition likelihood to differentiate. Their construction:
1. **Share the initial trajectory noise within each rollout group** (GRPO-style).
2. **Inject stochasticity only over the final three of ten Euler steps**, `W = {7, 8, 9}`, `σ = 0.03`. Rationale, verbatim: *"Under the endpoint parameterization, perturbations near t = 1 affect the emitted trajectory more directly, while earlier perturbations are increasingly attenuated."*
3. **A restoring score** from the Gaussian conditional, `s_θ = −(τ − t·τ̂₁)/(1−t)²`, folded into the transition mean as `μ = τ + v_θ·Δt + (σ²/2)·s_θ`; `1−t_k` lower-bounded at `ε = 0.1`, `τ̂₁` clipped to `[−1,1]`.
4. ⭐⭐⭐ **Low-frequency exploration.** Verbatim: *"Independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity."* They project noise onto the **first M = 6 orthonormal cosine modes** over the N = 50 waypoints: `τ⁽ᵏ⁺¹⁾ = μ⁽ᵏ⁾ + σ_k Φ Z_k`, `Φ ∈ ℝ^{50×6}`, `ΦᵀΦ = I`.

**Why RL at all**, verbatim: *"A recorded trajectory represents only one of several acceptable futures, so other safe behaviors may be penalized, particularly when the four training sources exhibit different ego-motion distributions."*

---

## 3. DATA — the most transferable part of the paper

### 3.1 Perception data and cross-dataset unification

- **nuScenes** (6 views, OccNet occupancy labels): official split, **28 K train / 6 K val** keyframes. **OpenScene** (8 views): 16 logs held out *"to balance city and time of day"* → **607 K train / 9 K val**.
- **Label unification** at *"the coarsest mutually compatible granularity"*: **7** detection classes, **10** occupancy classes, **6** map classes. Source-only classes are supervised only on their own source's samples.
- **Offline label completion** from auxiliary annotations: rasterise the nuPlan vector map to add `driveable` to OpenScene occupancy; generate `generic_object` pseudo-labels from nuScenes 3D boxes — ⭐ *"we change only voxels that already carry semantic labels"*, and no ray masking. **Class frequencies are recomputed after remapping** for the class-balanced focal loss.
- ⭐⭐⭐ **Spatial unification is done on the PREDICTED FEATURES, never on the labels.** The two sources use different physical extents (nuScenes ±40 m, z ∈ [−1.0, 5.4], 0.4 m voxels vs OpenScene ±50 m, z ∈ [−4.0, 4.0], 0.5 m voxels) and different LiDAR-to-ego transforms (nuScenes has a ~1.84 m vertical offset; OpenScene is identity). Rather than resample labels — *"resampling categorical labels would distort the supervision"* — **one differentiable trilinear sampling maps the predicted volume onto each dataset's native grid**, so one head serves both *"without dataset-specific branches."*
- ⚠️ **Honest about residue:** *"Label unification does not remove noise from the source annotations… Offline completion adds missing semantic labels but retains these artifacts"* (floating voxels from LiDAR aggregation).

### 3.2 Vision-language data — 24 public sets, LLM-rewritten, then consistency-filtered

- **24 public driving VL datasets** aggregated (CODA-LM, DRAMA, DriveAction, DriveGPT4, DriveLM, DrivingVQA, Impromptu VLA, LingoQA, MapLM, MM-AU, NAVSIM-ReCogDrive, NuInstruct, NuPlanQA, nuScenes-MQA, nuScenes-QA, **the OOD reasoning-label subset of PhysicalAI-AV**, OmniDrive, ROADWork, Senna, STSBench, SURDS, SUTD-TrafficQA, Talk2Car, WaymoQA). **Training splits only, to avoid leakage.**
- **Two-step pipeline, and the two steps do different jobs:** Qwen3.5-Plus **rewrites** every prompt/response into one conversational schema; then a **separate consistency filter** (Qwen3.5-Flash) checks each rewritten response against its source annotation. Verbatim: *"Rewriting standardizes the format but does not validate the source annotation."*
- ⭐ **A measured retention rate: 5.53 M → 3.09 M, 55.9 %.** Format mix after filtering: 61.6 % multi-view, 20.1 % single-view, 10.1 % single-view temporal, 4.4 % multi-view temporal, 3.7 % video.
- Boxes normalised to `[0, 1000)` image coordinates; view/frame tags inserted; textual view references rewritten to match.

### 3.3 Self-constructed data — three components

1. **Planning reasoning (Chain-of-Causation), explicitly credited to Alpamayo-R1.** Built from NAVSIM, Waymo and PAI-AV futures. ⭐⭐⭐ **A rule-based classifier derives LONGITUDINAL AND LATERAL manoeuvre components separately, which jointly form a "motion prior"**; Qwen3.7-Plus then writes the trace conditioned on multi-view images, history, the motion prior, and the navigation instruction.
   **The audit is the good part:** ⭐⭐ *"Rather than requesting scalar quality scores, judge models answer classification questions, and their decisions are aggregated programmatically."* It checks the predicted manoeuvre against the ground-truth trajectory, classifies the causal role of each cited factor, and ⭐⭐ ***"rejects traces that reveal future information"*** — a leak guard at generation time. Qwen3.5-Flash assigns a **rarity score**, and **rare scenes get higher sampling priority.** Two response formats: trace alone, or trace + JSON-serialised future trajectory.
2. **Camera ordering** — shuffle the surround views, strip all view tags, and require the model to find the front view from visual cues and recover the clockwise order. ⭐ A free self-supervised cross-view spatial task.
3. **In-house perception QA** — 30 K examples from Chinese road scenes: traffic-light grounding and 3D detection with the camera pose given in text and 3D boxes returned in the global frame.

### 3.4 Mixtures

- **Stage 2:** ~20 % sampled from each public source, **stratified by task and question type**, plus self-constructed and general VL → **1.54 M** examples (64.3 % driving VL / 26.0 % general VL / 9.7 % perception). **Group-specific repetition factors** then shift the effective mixture to **12.7 % perception / 31.0 % general VL / 56.3 % driving VL**, with VL repeated 2–3 epochs.
- **Stage 3 planning:** **2.83 M** samples — NAVSIM + OpenScene 890 K from 2.5 K clips; WOD-E2E 557 K from 2 K clips; **PAI-AV 1.38 M from 156 K clips**. ⭐ **Only 685 K (24.2 %) carry a reasoning trace; 75.8 % train with `r = ∅`.** NAVSIM and OpenScene are kept as separate sources *"because their ego-motion distributions differ"* despite sharing a nuPlan origin.
- **Trajectory preprocessing gotchas worth stealing:** WOD-E2E is 4 Hz → prepend the current position, fit a time-parameterised natural cubic spline, evaluate on a 10 Hz grid, take derivatives for velocity and acceleration; ⚠️ **apply a factor-of-four scale correction to the raw `accel_x`/`accel_y` metadata**; heading rate `θ̇ = (v_x a_y − v_y a_x)/‖v‖²`, **zeroed below 0.3 m/s**; reject samples whose historical or future acceleration exceeds **9.8 m/s²**; two complementary heading checks (derivative-based and adjacent-step) at a **1.2 rad/s** threshold.

---

## 4. RESULTS — and the three places the tables argue with the abstract

### 4.1 Perception and general capability

**nuScenes 43.95 mAP / 60.99 map mIoU · OpenScene 43.45 mAP / 71.27 map mIoU.** On ten general VL benchmarks the model stays **within one point of the base Qwen3.5-4B average** — the "foundation" property is preserved. Their commercial argument, verbatim: *"one instance can cover both domains. This removes the need for a separate cockpit model and the associated compute and maintenance cost."*

### 4.2 Planning, by tier

| tier | benchmark | result |
|---|---|---|
| open-loop | PAI-AV, avg ADE 3 s/5 s, 644-split | SFT w/o reasoning **0.38 / 1.07** · w/ reasoning **0.37 / 1.07** · RL **0.42 / 1.11** |
| open-loop | PAI-AV, **minADE** 3 s/5 s, 644-split | SFT **0.34 / 0.96** vs **Alpamayo 0.16 / 0.48** |
| open-loop | PAI-AV, leakage-free 700-frame subset | SFT w/ reasoning **0.42 / 1.23**, minADE **0.39 / 1.11** |
| open-loop | WOD-E2E test RFS | **7.91** (best in comparison) |
| pseudo-closed | **NAVSIM v1.1 navtest** PDMS | SFT w/ reasoning **88.2** (vs DiffusionDrive 88.1) → RL **90.7** |
| **closed-loop** | **AlpaSim, 916 scenarios** | **AlpaSim score 0.16 all / 0.27 at-fault — near the bottom** |

⛔ **Three contradictions inside their own numbers:**

1. **Reasoning is worth about a centimetre.** PAI-AV avg ADE 3 s: 0.38 without reasoning, 0.37 with. At 5 s: 1.07 both. On the leakage-free subset, 0.43 → 0.42 and 1.24 → 1.23. **The Chain-of-Causation condition — the most expensive component of the data pipeline — moves open-loop displacement by 0.01 m.**
2. ⭐⭐ **Their six candidates are not diverse, and they say how to see it.** Verbatim: *"SimWAM substantially improves average ADE to 0.43 m, while its minADE remains close at 0.40 m, **indicating more limited diversity**"*; *"DriveWAM achieves a much lower minADE than average ADE, suggesting **broader candidate coverage** but weaker typical trajectory accuracy."* By their own instrument, Qwen-Drive (0.42 avg / 0.39 min, a gap of **0.03**) is in the low-diversity class, while Alpamayo (0.36 / 0.17, a gap of **0.19**) is not.
3. ⛔ **RL trades the metric everyone reports for the metric that matters.** RL costs 3–5 cm of open-loop ADE and buys *"improved preference alignment on WOD-E2E, higher pseudo-closed-loop PDMS on NAVSIM, and **a halving of the closed-loop off-road rate** in AlpaSim"* (24.0 % → 12.0 %; at-fault AlpaSim score 0.27 → 0.37).

### 4.3 Ablations

- **Stage-2 mixture (Table 8).** VL training lifts Driving QA average by **+6.55** points with a *"substantially larger gain in CoC reasoning"*; adding 3D perception supervision keeps all three VL aggregates **within one point**. ⭐⭐ **But the authors refuse the causal claim**, verbatim: *"It also yields the highest RFS after Stage 3, although the margin over the other variants is small… but **does not establish explicit 3D supervision as the source of the improvement**."*
- **RL reward design (Fig. 11).** NAVSIM-only: PDMS 90.4 → **90.8** when a shared ADE term is added to the source-specific PDMS reward. Multi-source: 90.6 → 90.7. ⭐⭐⭐ **On WOD-E2E validation the shared ADE term lowers RFS 8.68 → 8.45 while cutting 5 s ADE from 2.24 m to 1.27 m** — a 43 % displacement improvement bought for 0.23 RFS. Verbatim: *"The shared displacement reward anchors preference optimization to the recorded motion and limits excessive [deviation]."*
- **Data scaling on PAI-AV** (avg ADE 5 s / avg FDE 5 s, 644-split):

| training scale | ADE 5 s | FDE 5 s |
|---|---|---|
| 0.17 M | 1.34 | 4.18 |
| 0.35 M | 1.25 | 3.92 |
| 0.69 M | 1.17 | 3.62 |
| 1.04 M | 1.09 | 3.41 |
| 1.38 M | **1.05** | **3.24** |

⭐ **An 8× data increase buys −21.6 % ADE and −22.5 % FDE, and the curve has not flattened.**

### 4.4 The data gap, quantified by the authors

Verbatim: *"Alpamayo-1.5 uses **80,000 hours** of driving trajectories and 3M CoC reasoning traces, whereas PAI-AV contains 156K clips, corresponding to approximately **900 raw hours** before sparse frame sampling."* ⇒ **an ~89× data gap between the closed-loop leader and the public-data challenger**, named by the challenger.

---

## 5. CONCESSIONS — the sentences that cost the authors something

1. ⭐⭐⭐ **Multi-timescale causal structure is unsolved**, verbatim: *"Driving mixes causes that act at different time scales. A red light 20 m ahead calls for early, gradual deceleration, whereas a child emerging 5 m ahead demands an immediate response. When such causes coexist, **the model remains unstable in identifying the governing cause and its temporal scope.** Even when the suggested trend is appropriate, the decision executed within the next 1 to 2 s may not reflect the stated immediate cause."* Their named fix: *"multi-timescale causal modeling."*
2. ⭐⭐ **Reasoning-action consistency fails, and the reasoning gain may be an artifact**, verbatim: *"the generated trajectory does not always adhere to the textual rationale. Although reasoning improves downstream planning performance, **part of this gain may stem from the additional model-internal information that the self-generated trace contributes to the conditioning context.**"*
3. **Cross-task transfer is limited by their own input design**: *"The three tasks currently use different input formats, temporal contexts, and image resolutions, which may limit the transfer of learned representations."*

---

## 6. WHAT THIS CHANGES FOR TANITAD

### 6.1 ⭐⭐⭐ The evidence tier (guideline S-1) — adopt immediately, zero cost

Qwen-Drive is a **published, single-source, same-setting rank inversion between open-loop and closed-loop.** Put it in the paper beside our own three lines (action echo 97.9 % vs 0.0 % hold-action; Alpamayo's open-loop metric flat across 3.3× params; Waymo L5). It converts S-1 from a position into a citation.

⭐ **And it retires a weaker argument of ours.** We have leaned on *"nobody evaluates closed-loop."* They did, on 916 scenarios, and published the table that hurts them. **The correct claim is narrower and stronger: the field reports open-loop headlines while its own closed-loop numbers disagree.**

### 6.2 ⭐⭐⭐ The three-planner hierarchy — an opponent's stated open problem

Their limitation 1 **is** the strategic/tactical/operative directive (PI, 2026-07-21), described as an unsolved instability in a 4 B foundation model, with *"multi-timescale causal modeling"* named as the fix. Combined with today's other two findings — Drive-HWM's unmatched slow/fast ablation (+0.8 PDMS) and `2601.00844`'s independent hierarchy proposal with a vanishing-gradient mechanism — **three independent 2026 sources now converge on temporal hierarchy in one week, and none of them has tested it at matched parameters.** That is precisely the gap backlog row 18 / A10-1 is specified to fill.

### 6.3 ⭐⭐ Architecture: two mechanisms we can take without adopting their stack

1. **The dual-stream BEV head answers GS-4 with a design.** GS-4 asks whether an explicit geometry loss on the v7 readout moves azimuth-bin decodability. Qwen-Drive's head is a worked instance: a **geometry** path (depth-binned lift, height retained) initialises the **queries** of a **semantic** path, and the two are fused before the occupancy branch. ⭐ Critically, its depth net is trained **without depth supervision** — which matters because our corpus has none. ⛔ **But their own ablation refuses to credit the perception supervision for the planning gain**, so GS-4's pre-committed read must stay as written; this supplies the *architecture*, not the *evidence*.
2. ⛔ **The trunk-freeze question gets a third data point today, and it points the same way.** Stage 1 (frozen) *"yields limited perception performance"*; Stage 2 unfreezes and runs the head at **20× LR**. Today's other two findings — `2601.00844`'s value-shaping loss and the LDAD/AITS pattern — are the same shape. **Four independent lines now say a frozen trunk caps what a head can recover.** This belongs in the v7f freeze decision as a stated cost.

### 6.4 ⭐⭐⭐ Data curation and augmentation — the richest yield

| what to take | why it fits TanitAD |
|---|---|
| **Rewrite ≠ validate.** An LLM normalises format; a *separate* consistency filter checks the rewrite against the source annotation. Measured retention **55.9 %** | We have no measured retention rate for any curation step. Kairos deferred its second level entirely (today's DataEng package); **Qwen-Drive shipped one and published the number** |
| ⭐⭐ **Judges answer CLASSIFICATION questions, aggregated programmatically — never scalar quality scores** | Directly applicable to every LLM-judge we run, and it removes the calibration problem that makes scalar judge scores uninterpretable |
| ⭐⭐ **Reject any generated trace that reveals future information** | A leak guard at *generation* time. Same family as our nav-echo bijection and the `situations.py` provenance leak — an opponent independently arriving at our discipline |
| **Rarity score → higher sampling priority for rare scenes** | The control-relevance selection Kairos only proposed. Pairs with new row **DE10-1** (corpus-rate census of the six control-relevant families): **Kairos supplies the taxonomy, Qwen-Drive supplies a working selection signal** |
| ⭐⭐⭐ **Separate LONGITUDINAL and LATERAL manoeuvre components in the motion prior** | **Independent corroboration of our longitudinal-blindness root cause** — one 5-way manoeuvre softmax mixing LAT and LON explains 0/881 accelerate. Qwen-Drive never mixes them |
| ⭐⭐⭐ **Spatial unification on PREDICTED FEATURES, not on labels** (one differentiable trilinear resample onto each source's native grid) | **The mechanism for combining PhysicalAI-AV with ZOD / PandaSet / YouTube (P3) without the parity violation that relabelling would cause.** Our corpora differ in extent, resolution and ego-frame convention exactly as theirs do |
| **Label unification at the coarsest mutually compatible granularity**, source-only classes supervised source-only, class frequencies recomputed after remapping | The taxonomy problem the owned-data tier (backlog row 26, ZOD) will hit the moment a second corpus lands |
| **Trajectory hygiene**: cubic-spline resampling to a common rate, the ≤ 9.8 m/s² acceleration filter, the 1.2 rad/s dual heading check, heading rate zeroed below 0.3 m/s | Cheap, mechanical admissibility filters for the Y-pilot-50 YouTube pipeline (row 23) and for any resampled corpus |
| ⚠️ **The factor-of-four correction to WOD-E2E's raw `accel_x`/`accel_y`** | A concrete instance of our own recurring class — **a field whose units are not what its name implies.** If we ever ingest WOD-E2E, this is a landmine already mapped |

### 6.5 ⭐⭐ Training method

- **Stage-4's RL construction transfers wholesale to REF-C.** Getting a policy gradient out of a deterministic flow by perturbing only the **last three of ten** Euler steps, with a restoring score and shared group noise, is a complete recipe. ⭐⭐⭐ **And the low-frequency projection — noise onto the first 6 cosine modes over 50 waypoints, because *"independent waypoint noise primarily introduces high-frequency jitter rather than meaningful maneuver diversity"* — is a direct, tested answer to REF-C's fan-diversity problem.**
- ⭐⭐⭐ **The shared-ADE anchor is the four-families doctrine, measured.** Adding a displacement term to a preference reward cost **0.23 RFS** and bought **43 % of 5 s ADE** (2.24 → 1.27 m). ⇒ **FS9-5's per-family reward decomposition is not a hygiene preference; without an anchoring family the RL drifts 2.24 m from the recorded motion and the aggregate reward never says so.**
- **Group-specific repetition factors** (perception repeated harder than VL) is a cheap mixture knob we do not currently use.
- ⚠️ **Ego state is an input to their Planning Expert.** This is another instance of backlog row 11's problem: **every competitive planner consumes ego status, and our vision-pure rule makes our rows either non-comparable or rule-breaking unless the two-arm design is approved.** The PI decision is now overdue against a third instance.

### 6.6 Distillation

⚠️ **The honest reading is that this model is a poor distillation teacher for us, and saying so is the finding.**

- Apache 2.0, 5.0 B, weights public — **licensing and access are not the obstacle** (and our open-source usage is sanctioned).
- ⛔ **But it is last-place closed-loop.** Distilling a teacher that scores 0.16 AlpaSim-all against Alpamayo-R1's 0.36 would transfer the ranking we are trying to beat. **Its trajectory head is exactly where it is weakest.**
- ⛔ **And its candidate diversity is measurably narrow** by its own avg-vs-minADE instrument (gap 0.03 vs Alpamayo's 0.19), which is the property REF-C's fan needs most.
- ⭐ **Where it IS a good teacher: the semantic channel our trunk lacks (B1).** Its 3D-perception and driving-VQA capabilities are strong, general capability is preserved, and the BEV head is *"an explicit, inspectable interface to 3D scene structure"*. **A perception/semantic distillation target, not a planning one.**
- ⭐ **Better still, it is a free LABELLER.** Traffic lights, 3D boxes, occupancy, map elements and CoC traces on our own parity corpus — which is the DataFlyWheel's augmentation lane, not the model lane.

---

## 7. WHAT TO DO — ≤3 recommendations

1. ⭐⭐⭐ **Adopt the rank inversion as the programme's headline evidence for S-1, and narrow our claim.** Replace *"nobody evaluates closed-loop"* with *"the field reports open-loop headlines that its own closed-loop tables contradict,"* citing Table 7 against Tables 5–6. **Zero cost, and it is the strongest external support our binding open-vs-closed-loop ruling has ever had.**
2. ⭐⭐⭐ **Take the low-frequency exploration basis into REF-C now** (first 6 cosine modes over the waypoint horizon, σ ≈ 0.03, perturbing only the last ~30 % of integration steps), **and pair it with the avg-vs-minADE gap as a diversity instrument.** Both are free, both are published, and both attack the fan-diversity defect FS5-1 and FS9-5 are circling.
3. ⭐⭐ **Use Qwen-Drive as a labeller and a perception teacher, never as a planning teacher.** It is Apache 2.0, strong where our trunk is weak (semantics, 3D probes) and weakest exactly where we would be distilling (closed-loop trajectory quality, candidate diversity).

## 8. PRE-REGISTERED EXPERIMENT

**The diversity-gap instrument on our own banked fan.** Compute avg-ADE and minADE over REF-C's N candidates on the banked 201 windows, and report the **gap**, against three controls: a constant-velocity floor, a shuffled-candidate control that must read the no-information value, and a single-candidate degenerate baseline whose gap must read **0**.
⛔ **Committed in advance: if our gap is ≤ 0.05 m (Qwen-Drive's class), our fan is a mode-collapsed sampler and the low-frequency exploration basis is the first fix, ahead of any scorer work; if the gap is ≥ 0.15 m (Alpamayo's class), diversity is adequate and FS5-1's scorer-blindness is the binding defect instead.** Report `n` per cell. **0 GPU** — the windows are banked.

## 9. ESCALATIONS

- **To the PI (decision, now third instance):** backlog row 11's ego-status two-arm design. Qwen-Drive's Planning Expert consumes current ego state `e`. Every competitive planner does. **Without the two-arm approval our published rows are either non-comparable or rule-breaking.**
- **To the Master Mind (v7f freeze, blocking):** a **fourth** independent line today says a frozen trunk caps what a head can recover — Stage 1 vs Stage 2 here, plus LDAD, AITS and the IQL value loss. Record the freeze decision and its measured cost.
- **To the Master Mind (register):** Qwen-Drive is an opponent-doctrine-bearing publication and its claims should be adjudicated into `OPPONENT_CLAIMS_REGISTER.md` on the next Band-D pass. **Debt D-12 (AlpaSim fidelity) is partly addressable now: an AlpaSim protocol with 916 scenarios and six reported metrics exists in the literature and we run AlpaSim.**
- ⚠️ **Split-stamp note, consistent with today's Benchmarks package:** their 90.7 is **NAVSIM v1.1 navtest PDMS**, stamped by the authors themselves. It belongs to the navtest population and may not be compared to navhard numbers.
