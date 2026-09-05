# refcv5 — Design Plan: closing the DiffusionDrive gaps without giving up the hierarchy

**status: COMPLETE — §0 summary · §1 gap table · §2 invariants · §3 BEV/LiDAR · §4 diffusion mechanism · §5 selection · §6 RL · §7 implementation ladder · §8 claims and refutations · §9 the VLA contract · §10 registered decisions · manifest. ⛔ Nothing here is a result: it is a design with pre-registrations, and it launches nothing.**
**author:** Architecture & Inference FlyWheel · **date:** 2026-09-05 · **branch:** `agent/arch-inf-20260803`
**GPU spent by this document:** 0 (design synthesis; every number is cited to its artifact)
**Owner for integration:** Master Mind. **Sibling streams:** `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` (VLA, §9 interface; not yet present in HEAD at the time of writing), DataFlyWheel (§3 work package), `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/` (§6 vehicle), `…/Architecture & Inference/Research/2026-09-05-withheld-bank-panel/` (§2, pending panel).

## PI's ask (verbatim, 2026-09-05)

> "What were the results of the deep analysis and comparison with both DiffusionDrive papers? Do we have a plan to close the gaps without giving up our USPs like the hierarchy, how can we implement the missing pieces, I need a plan for it. This will be refcv5 combining our learning with the missing pieces. I think the AV dataset includes lidar, we need a plan to engineer and create the BEV + other missing pieces."

## How to read this document

Every number carries its evidence class — `MEASURED (artifact)` · `PUBLISHED (library key / paper / pinned code)` · `INHERITED (doc, not re-verified)` · `ESTIMATED` · `HYPOTHESIS` — and, where it is an eval number, its tier (T0 / T1). Decisions are registered as `D-REFCV5-PLAN-*` in `Project Steering/GOALS_AND_CLAIMS.md` (§10). A section marked `[PENDING]` is not yet decided. Line numbers are as read on 2026-09-05 from the worktree; the files move, so every reference also names the function.

---

## §0 Executive summary — the answer to the PI's four questions

**1. What the deep analysis found.** Two audits, both banked with primaries (`D-REFC-DDAUDIT-1..6`, `D-DDV2-CODE-1..5`, `D-DDV2-RES-1`, `D-DDV2-ORDER-1`; `H-DDA-1..7` pre-registered). REF-C implements DiffusionDrive's *skeleton* (anchor vocabulary → anchor queries cross-attending scene features → per-anchor confidence + offset → argmax) but **not its mechanism**: there is no noise schedule, no anchored Gaussian, no x0/DDIM update, no sampling at inference, and the shipped ranking scores a fan two passes staler than the one it emits (`sel_idx_base` unchanged on 201/201 windows across `steps ∈ {0,…,8}`, MEASURED on refcv3 @ 40,284). Our cross-attention exists but attends ungrounded perspective-view (PV) tokens through a FiLM; DD's is grounded three ways — sampled **at the candidate's own waypoints** in a metric BEV, on **30 detected-agent queries**, and on the ego planning query — and we have none of the three because we have no BEV, no detector and no LiDAR. V2 keeps v1's generator and adds a head-only RL stage and a sub-metric selector; the code shows the RL "exploration" is **two scalars per anchor (along-track scale, lateral scale)**, exactly the axis on which 92.2 % of our own deficit sits (`D-REFCV3-AXIS1`), and that no diffusion sampler is needed to port it.

**2. Do we have a plan that keeps the USPs?** Yes — §1 decides every audited component (IMPLEMENT / ADAPT / KEEP OURS / SKIP), and §2 states the USPs as invariants that every adopted mechanism is checked against: the three-level hierarchy with the 30 s strategic label as supervision and nav as a conditioning input (PI ruling R5), the nav-COMPLIANCE metric and the post-training hierarchy tests (R1, R2), vision-only inference with ego-dropout + X15 anti-echo, the v0-conditioned kinematic vocabulary (with the realised-speed κ clamp, `D-REFCV4B-FLYLOW1`), and the four-family / T-tier / paired-bootstrap doctrine. DD's diffusion is adopted **in control space** (noise on `(a_lon, a_lat)`, re-rolled through the unicycle) so every sample stays flyable and the vocabulary stays the prior (§4).

**3. How the missing pieces get implemented.** As a ladder of one-lever, pre-registered tiny-rig arms (§7), ordered by what is free first: E-DDA-4 (a zero-cost readout from the refcv4b dump), E-DDA-2/2b (score the emitted fan; the four-family selector), E-DDA-1 (waypoint-indexed sampling in PV — no BEV needed), E-AGT-1 (agent tokens from `obstacle.offline`), E-DDA-3 (the sampler, in control space), E-DDA-3b → E-DDA-6 (the head-only scale-policy RL), E-DDA-5 (loss form), the withheld-bank panel (`H-EGO-LIT-4`, running), and the vocabulary v5 build (κ clamp + 13×11 grid, MODEL-FREE gates). Everything except the LiDAR path can start now at 0 GPU; the full-scale refcv5 launch follows refcv4b's final eval on the same A40.

**4. LiDAR and the BEV.** ⭐ **The PI is right: PhysicalAI-AV carries LiDAR.** Three independent probes agree (§3): `lidar/lidar_top_360fov` — a 128-row spinning LiDAR (`lidar_intrinsics.offline`: `row-offset-spinning`, 128 rows × 3600 columns), 10 Hz, ~200 spins per 20 s clip, Draco-compressed, on **298,326 of 306,152 clips (97.44 %)**, 3,146 chunks of **31.7 GB** each = **99.6 TB, 68 % of the dataset by bytes**. Its extrinsics are in `sensor_extrinsics.offline` (8 sensors = 7 cameras + LiDAR), in the same rig frame `obstacle.offline` uses. The plan (§3): a DataFlyWheel work package that builds a DD-faithful **256 × 256 @ 0.25 m/px, ±32 m, 2-height-bin BEV histogram** per episode frame as a **sidecar** to the `*.v2ep.pt` cache (priced against the consumer's loader: ≈ 3–8 MB/episode PNG, ESTIMATED, against 34.0 MB/episode of camera), plus the agent raster the repo already has (`tanitad/data/bev_raster.py`, from `obstacle.offline`). ⚠️ Whether LiDAR is an **inference input** (TransFuser-style sensor fusion, in the Mission Plan and ROADMAP L4) or a **training-time teacher** for a camera-only BEV lift is a doctrine call the PI owns (§2 I3, `D-REFCV5-PLAN-7`); this plan builds the data either way and pre-registers both arms so the sensor's contribution is attributable.

**The bar refcv5 must clear (§8):** at T1, on the four families with the echo gate, beat **both** `ha` and `ha0_ext` per horizon — the bar refcv3 never cleared (`os` 0.4419 m vs `ha` 0.2996 m at 2 s, MEASURED, `taniteval/results/refcv3-40284-openloop.ARM.json`) — and pass the post-training hierarchy tests (`PREREG_REFCV4B_HIERARCHY_EVAL.md`) with the new 30 s `t_bin` primary.

---

## §1 The gap table, closed out

Sources: `…/2026-09-05-refc-vs-diffusiondrive-audit/RESULT.md` §1.1–1.6 (our side MEASURED at the worktree + refcv3 @ 40,284; DD side PUBLISHED, `hustvl/DiffusionDrive@9b52ed0`, `hustvl/DiffusionDriveV2@1cd12a1`, arXiv 2411.15139v3 / 2512.07745v1, library keys `2411.15139` / `2512.07745`). Decision legend: **IMPLEMENT** = adopt DD's mechanism · **ADAPT** = their idea under our constraints · **KEEP OURS** = we are ahead or the divergence is deliberate · **SKIP** = not applicable, with the reason. Attribution rule: **one lever per pre-registered tiny-rig arm** (§7); the full-scale refcv5 bundles only levers that passed their arm.

### 1.1 Truncated diffusion (audit #1–12)

| # | component | ours (MEASURED, audit) | DD mechanism (PUBLISHED) | DECISION | arm |
|---|---|---|---|---|---|
| 1 | Anchored Gaussian prior | none — `x = bank + offset`, train-only N(0, 0.1 m) jitter that moves the fan by 0.062 m vs 8.41 m of offset | τ_t = √ᾱ_t·a_k + √(1−ᾱ_t)·ε in normalised coords; σ ≈ 0.90/0.73 m at t = 8 | **IMPLEMENT — in CONTROL space** (§4): noise on `(a_lon, a_lat)`, re-rolled through `rollout_unicycle`, so every sample is flyable and the vocabulary stays the prior | E-DDA-3 |
| 2 | Truncation timestep | none (constant `noise_std 0.1`) | train t ~ U[0, 50) of T = 1000; infer from t = 8 | **IMPLEMENT** (same numbers, normalised control units) | E-DDA-3 |
| 3 | Denoising iterations, train | 1 classifier pass + 2 refinement passes | 0 (one noised sample, one decoder call) | **ADAPT**: keep the classifier pass — it is our anchor-CE surface and the H19 / factored / LAN / goal priors live on it — and make the refinement passes DDIM steps on a noised sample | E-DDA-3 |
| 4 | Denoising iterations, inference | 2 passes + classifier | 2 DDIM steps | **KEEP the count (2)**; mechanism from #1/#5/#9 | E-DDA-3 |
| 5 | Noise schedule | none | `diffusers.DDIMScheduler(1000, scaled_linear, prediction_type="sample")` | **IMPLEMENT** (behind `DecoderConfig.sampler` so the classifier path stays byte-identical) | E-DDA-3 |
| 6 | Prediction target | residual on the previous estimate, no √ᾱ scaling | x0 ("sample"): residual on the noisy input | **IMPLEMENT** the x0 form | E-DDA-3 |
| 7 | Timestep embedding | `nn.Embedding(steps+1, d)`, 3 rows, additive once, saturates at row 2 for extra passes; live (−1.48 m if removed) | sinusoidal → MLP → per-layer AdaLN scale/shift | **IMPLEMENT** (comes with the sampler; the audit's rank-6 item) | E-DDA-3 |
| 8 | Loss per step | CE on t = 0 confidence; L1 on the final fan at the assigned anchor; gradient through the whole pass chain; refined confidences unsupervised | per cascade layer `(reg, cls)`, one random t per sample, layer-to-layer coordinates detached | **ADAPT**: per-pass loss with detach between passes; keep the t = 0 CE (classifier surface) **and** supervise the emitted-fan confidence (`sel_ce_reach` / `score_emitted`) | E-DDA-2, E-DDA-3 |
| 9 | Sampler | none (deterministic residual loop) | DDIM, η = 0, final step → x0 exactly | **IMPLEMENT** | E-DDA-3 |
| 10 | Classifier-free guidance | none | none | **SKIP** — N/A on both sides | — |
| 11 | Coordinate normalisation | raw metres | `norm_odo` to [−1, 1] (x ∈ [−1.2, 55.7], y ∈ [−20, 26] m) | **ADAPT**: normalise in control units (`a_lon/4.0`, `a_lat/3.0` — the 13 × 9 grid's own ranges) — a metre-space normaliser would put the noise back on the geometry | E-DDA-3 |
| 12 | Does `_decode` denoise? | no — non-contractive residual refiner; endpoint spread 104 → 42 m by 8 passes | scheduled denoiser, 2 steps | replaced by #1–#9; **the pass count is never again a free knob** (mode collapse is controlled by truncation, not iteration) | E-DDA-3 |

### 1.2 Multimodal generation (audit #13–18)

| # | component | ours | DD | DECISION | arm |
|---|---|---|---|---|---|
| 13 | Number of modes | 117 (13 `a_lon` × 9 `a_lat`), `N_infer ≡ N_anchor` | 20 k-means, `N_infer` may differ (Tab. 6) | **KEEP 117** as the base; **ADAPT** `N_infer ≠ N_anchor` via G samples per anchor once #15 lands; the 13 × 11 grid (+26 anchors, −0.0418 m [−0.0566, −0.0286] MODEL-FREE, `…/2026-09-04-refcv4b-turn-coverage/RESULT.md` §6) is the vocabulary-v5 candidate, gated MODEL-FREE per `GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md` | V5-VOCAB |
| 14 | Anchor construction | v0-conditioned constant-`(a_lon, a_lat)` controls rolled per window (`refc.py::roll_bank`); κ = `a_lat / max(v0, 4)²` clamped ±0.12 | fixed k-means table in metres, `requires_grad=False` | **KEEP OURS** — a per-window kinematic prior DD lacks (audit #14) — **subject to the running `H-EGO-LIT-4` panel** (§2 I3): if the speed-blind arm A4 matches on the echo instrument and the families, the roll is retired for the field's design; **plus** the `D-REFCV4B-FLYLOW1` fix (clamp κ against the realised speed) and a Kamm filter at build time (+0.0043 m oracle cost, removes the 16 % dead pool) | V5-VOCAB, H-EGO-LIT-4 |
| 15 | Diversity via noise | none; two calls bit-identical | each anchor independently noised at inference; diversity D 74 % vs 11 % | **IMPLEMENT** via #1 — **only after E-DDA-4** says the bank is a prior (‖offset‖ < 1 m); readouts = oracle-in-fan at equal N + DD's diversity score D | E-DDA-4 → E-DDA-3 |
| 16 | Mode-collapse prevention | nothing structural; iterating collapses the fan | truncation + per-mode cls | **IMPLEMENT** (truncation) and retire the unbounded pass count | E-DDA-3 |
| 17 | Which modes are supervised | L1 on the assigned anchor, sum-of-squares assignment | L1 on the nearest anchor, mean-L2 assignment | **KEEP**, with the assignment metric switched to mean-L2 as arm (c) of E-DDA-5 (a different anchor on some windows — attributable, cheap) | E-DDA-5 |
| 18 | Where multimodality is lost (our four causes) | (a) no sampling; (b) selection decided before refinement; (c) reach clamp deletes 18–37 % (2 s numbers); (d) non-assigned offsets unsupervised, mean 8.41 m | — | (a) → #15; (b) → #28; (c) **SKIP** — MEASURED inert at the 6 s band (0.00 % killed, 117.0 survivors/window, turn-coverage §2); (d) → E-DDA-4 decides, and #1's per-anchor noising supervises every anchor's x0 | — |

### 1.3 Cross-attention (audit #19–26) — the PI's direct question

| # | component | ours | DD | DECISION | arm |
|---|---|---|---|---|---|
| 19 | Spatial attention to scene features | content MHA from 117 anchor queries to the 8 × 20 = 160 PV tokens (`CrossAttnLayer`, `refc.py:1114-1131`); the query carries the trajectory only through `traj_proj(x_est)` — the attention never looks *where the candidate goes* | `GridSampleCrossBEVAttention`: bilinear-samples the BEV value map at the 8 waypoints of each noisy trajectory, softmax over points, re-sampled every layer | **IMPLEMENT in two stages**: (i) **E-DDA-1** — waypoint-indexed sampling on the **PV** map (project each candidate's waypoints through the banked `camera_intrinsics` / `sensor_extrinsics` into the 256 × 640 **cylindrical** frame — `col = f_ref·φ + W/2`, `f_ref` 305.577, `calib.CanonicalFrame(projection="cylindrical")`, `bev_raster.readout_column_index` — and bilinear-sample the 8 × 20 map; zero-init gate; control = random sampling locations at equal parameter count); (ii) **E-BEV-1** — the same module on the metric BEV map once §3 delivers it (no projection: the fan and the BEV share the rig frame) | E-DDA-1, E-BEV-1 |
| 20 | Agent cross-attention | none (no detector, no agent tokens) | MHA to 30 detection queries from the TransFuser decoder | **IMPLEMENT — ADAPTED to our labels**: a DETR-style agent-token head (K = 30 queries, Hungarian matching, box + class) trained on `obstacle.offline` (3D tracks on 97.44 % of the corpus, pod-side join `stack/scripts/build_obstacle_join.py`); the decoder layer gains a second cross-attention `cross_agent(q, agents, agents)`, zero-init gate. At inference the tokens come from the trunk — no label is read (§2 I3) | E-AGT-1 |
| 21 | Ego / status cross-attention | FiLM on the MLP input from `cond_proj(m)` (`m` = `[v0/10, nav one-hot]`) + zero-init `ctx_to_cond`, `lan_to_cond`, `tgt_film`; E11' ego block reaches the goal path under `ego_dropout 0.5` + X15 | `cross_ego_attention(traj, ego_query, ego_query)`; the per-layer `status_encoding` argument is dead code; **no ego dropout anywhere** (`H-EGO-LIT-2`) | **KEEP OURS** — the FiLM + dropout + X15 design is the one lever MEASURED to make an arm read the scene (`H-ECHO-8`); DD's ego query is an unguarded channel. **SKIP** the ego-query attention | — |
| 22 | Self-attention over candidates | none (S2b prefilter exactness rests on it, `SelectionConfig` `refc.py:467-472`) | none in the generator; **present in the V2 selector** | **KEEP** (generator); **IMPLEMENT** in the selector only (#31), where candidates must see each other | E-DDA-2b |
| 23 | Depth / width | 4 layers, d = 384, 8 heads, ff ×4 | 2 layers, d = 256, ff 1024; V2's validated decoder is **1** layer (`D-DDV2-CODE-4`) | **KEEP OURS** — depth is not a lever either way | — |
| 24 | Query initialisation | `nn.Linear(S·2, d)` on raw metres | sinusoidal Fourier features per waypoint | **IMPLEMENT** (cheap; bundled with E-DDA-3, and it is what the selector's candidate embedding needs anyway) | E-DDA-3 |
| 25 | Timestep conditioning inside the layer | additive bias on the input query | per-layer AdaLN | **IMPLEMENT** (= #7) | E-DDA-3 |
| 26 | Map cross-attention | none | none in v1's released code (map enters as the BEV-semantic aux); no map in PhysicalAI-AV (pinned, `test_physicalai_feature_readset.py`) | **SKIP** on PhysicalAI; the only non-oracle map we hold is AlpaSim/NuRec `map.xodr` (W6 of `DESIGN_REFCV4_NAV_WIRING.md`) | — |

### 1.4 Confidence / selection (audit #27–31)

| # | component | ours | DD v1 / V2 | DECISION | arm |
|---|---|---|---|---|---|
| 27 | Score head | `Linear(d, 1)` per anchor, softmax CE over N on the t = 0 pass, weights traj 1.0 / cls 1.0 | v1: per-mode sigmoid, focal BCE (γ 2, α 0.25), cls 10 / reg 8; V2: the head is untrained and unread | **ADAPT-if-separates**: focal BCE as arm (a) of E-DDA-5; calibration (ECE of the winner) is the readout | E-DDA-5 |
| 28 | Object scored vs emitted | ranking reads the t = 0 confidence; emitted fan is two passes later (201/201 unchanged) | scores the fan it emits (`argmax(poses_cls[-1])` on `poses_reg[-1]`) | **IMPLEMENT** — `sel_refined + sel_score_emitted (+ sel_ce_reach)`, 0 params, flags exist (`refc.py:1703-1710`); first free read = `H-SEL-1` in refcv4b's post-training ablation | E-DDA-2 |
| 29 | Priors on the score | H19 `maneuver_to_anchor`, factored `lat/lon_to_anchor`, LAN / goal geometric compatibilities, consequence gate | none in v1 | **KEEP OURS (beyond DD)** — and add the two nav / strategic selector gates S7 / S8 (E17 / E18 of the nav wiring) | E17/E18 |
| 30 | Physical filter | S2 reach band (`sel_accel_max 2.0`, 6 s) — MEASURED inert at 6 s | v1 none; V2 penalises collision / off-drivable (−1) and scores NC/DAC/TTC/EP/C | **KEEP** the band as a guard; **ADD** the build-time Kamm filter (#14) and the selector's collision / TTC heads (#31) | V5-VOCAB, E-DDA-2b |
| 31 | V2 mode selector | none (`refc_rescorer.py` v1.2 was REFUSED — winner's curse; E9 goal scorer exists) | coarse (1 layer, five PDM sub-score heads, BCE + margin-rank 0.05 × 2) → top-32 → fine (3 layers), 800 candidates, candidate self-attention | **IMPLEMENT — ADAPTED**: our four-family heads {collision/TTC/headway vs the `obstacle.offline` replay, comfort, progress, tactical-consistency (v7.2)} — no PDM, no map (§5) | E-DDA-2b |

### 1.5 Encoder (audit #32–36)

| # | component | ours | DD (TransFuser) | DECISION | arm |
|---|---|---|---|---|---|
| 32 | Inputs | camera only: 3-frame RGB stack (9 ch) at 256 × 640 cylindrical, W = 8 frames for the strategic GRU, `v0` (+ nav) | 3 cams stitched 1024 × 256 + LiDAR BEV histogram 256 × 256 @ 0.25 m/px (±32 m) + 8-d status | **IMPLEMENT the BEV data (§3)**; the *input* question is a PI doctrine call (`D-REFCV5-PLAN-7`): default = LiDAR/agent BEV as a **training-time teacher** for a camera BEV lift; arm = LiDAR BEV as an **inference input** (E-LIDAR-1) | WP-DE-BEV-1, E-BEV-1, E-LIDAR-1 |
| 33 | Fusion | single ResNet-34-style trunk → 8 × 20 PV map; no lift to BEV | two ResNet-34s fused by GPT blocks at 4 scales, top-down decoder → metric BEV | **ADAPT — late fusion first**: a small BEV encoder (ResNet-18-class, 256 × 256 × 2 → 8 × 8 tokens) whose tokens join the decoder's KV set (`kv = feat_proj(PV) ⊕ bev_proj(BEV)` with a modality embedding); TransFuser's 4-scale mid-fusion only if E-LIDAR-1 separates and the Thor tact budget (§9) allows a second trunk | E-LIDAR-1 |
| 34 | Auxiliary heads | LAW latent (w 0.5), route head (w 0.1), maneuver / factored heads (w 0.1), goal cascade | 3-D agent detection (30 boxes, Hungarian) + BEV semantic segmentation (7 classes) — these **produce** the agent queries | **IMPLEMENT**: the agent-token head (#20) and a BEV-occupancy aux target from the LiDAR histogram + the P8 agent raster (`tanitad/data/bev_raster.py`, exists) | E-AGT-1, E-BEV-1 |
| 35 | What is structurally lost | no metric frame to index, no agents to attend — the LONGITUDINAL (headway/TTC) and TACTICAL grounding | — | closed by #19 / #20 / #32–#34 | — |
| 36 | V2 backbone | — | unchanged from v1 | n/a | — |

### 1.6 V2 additions (audit #37–43)

| # | component | ours (`stack/tanitad/rl/`) | V2 (pinned code) | DECISION | arm |
|---|---|---|---|---|---|
| 37 | Denoising policy with log-prob | absent (deterministic decoder) | `DDIMScheduler_with_logprob`, RL rollout 10 steps | **IMPLEMENT with E-DDA-3** — but the RL stage does **not** wait for it (`D-DDV2-ORDER-1`): the released mechanism never uses the chain's randomness | E-DDA-3 |
| 38 | Scale-adaptive multiplicative noise | `refcv3_adapter.py::sample_offsets` is per-coordinate | two scalars per trajectory (along, lateral), σ floor 0.04; additive term × 0 | **IMPLEMENT** the 2-scalar mode — in **control space** (scale `a_lon`, `a_lat`), one-line change | E-DDA-3b |
| 39 | Intra-anchor GRPO | `advantage.py::grpo_advantage` (centre-only default; std divisor a flag), G ≥ 2 refused otherwise | G = 4, `(r − mean_G)/(std_G + 1e-4)`, discount 0.8 | **KEEP OURS** + **E-DDA-6** decides the grouping key (anchor vs (a_lon, a_lat) cell vs tactical class) | E-DDA-6 |
| 40 | Inter-anchor truncated advantage | `truncated_inter_anchor_advantage` (clamp ≥ 0, veto pinned −1) | `clamp(min=0) · (r > r_GT − 1e-6)`; −1 on collision **or** off-drivable | **IMPLEMENT the ≥ GT bar** (one line, behind a config field; the GT trajectory is scored under the same rule reward); the drivable-area half **SKIP** (no map) | E-DDA-3b |
| 41 | Reward | rule-based components (progress 0.30, collision 1.00, headway 0.30, feasibility 0.50, comfort 0.20), `FORBIDDEN_REWARD_INPUTS`, `HACKABLE_WEIGHTS` regression arm; MEASURED to prefer hold-v0 to the human on 78.3 % of lead windows (`D-RL-REWARD-FLOOR-1`) | NAVSIM PDM simulator score, EP re-normalised to the GT's progress | **ADAPT** to PDMS's *structure*: `NC(replay) × (5·TTC/headway + 2·comfort + 5·progress)/12`, ≥ GT bar, collision → −1, **no DAC**, **no GT-future term** (§6 echo table); the reward-floor defect is fixed **before** any arm | E-DDA-3b |
| 42 | Mode selector | none | coarse-to-fine sub-metric selector | = #31 | E-DDA-2b |
| 43 | What would be needed | — | (i) stochastic decoder, (ii) T1-admissible reward, (iii) selector-disjointness audit | (i) E-DDA-3; (ii) the `obstacle.offline` replay join (exists pod-side: `build_obstacle_join.py`, `taniteval/lead_source.py`); (iii) `rl/audit.py` exists | — |

**Net:** 19 IMPLEMENT, 9 ADAPT, 9 KEEP OURS, 5 SKIP (each with its reason); 10 pre-registered arms carry them (§7).

---

## §2 The USPs that must survive — stated as invariants

Every adopted mechanism in §1 is checked against these six. A violation is a refusal, not a trade-off.

| id | invariant | pinned by | why the adopted DD mechanisms do not violate it |
|---|---|---|---|
| **I1 — the three-level hierarchy, supervised at the top** | strategic (`g_str` → E19: `Linear(d_ctx, 3+4+K+4+K)`, the **30 s label** `STRATEGIC_S = (8, 30)` from the v7.2 `manoeuvre_sequence`, event-anchored, causal, `none` as a class, censoring modelled) conditions tactical (E4 FiLM + **E16 urgency**), which reaches the operative decoder (E7 `target_latent`) and selection (E9 `goal_gate·scorer`); **nav is a conditioning input at every level** (E20 v7.2 source — ON in refcv4b; E21 `nav_known`; E15 nav → core tactical; E17 / E18 nav and `g_str` → selector; E13 already live); H19 is the **internal** seam (core kin3 heads → anchor prior; `SEAM_STATE.md`). PI ruling R5 (2026-09-05) | `DESIGN_REFCV4_NAV_WIRING.md` §2.2 (7 edges, +7,682 params, all zero-init and individually gated), §3.4 (E19), §3.6 (E16); `Decisions/2026-09-05-pi-rulings.md` R5 | The sampler (§4) noises the operative fan only; the priors from the hierarchy stay on the classifier surface and are re-applied to the refined readout (`refc.py:1677-1683`). The selector (§5) reads the fan and the scene plus the S7/S8 gates — it does not bypass `g_str`/`z_tac`. Agent tokens and BEV enter the operative decoder's KV, never the goal nodes. The RL stage trains the head only, trunk and goal heads frozen (V2's own recipe). The 30 s label is a **training target**, never an input (`refused_edges`). |
| **I2 — compliance, not reproduction, and the post-training hierarchy tests** | the nav-COMPLIANCE metric (`taniteval/taniteval/nav_compliance.py`, criteria v2.6.0: a rate without its shuffle + zero controls is two violations); `PREREG_REFCV4B_HIERARCHY_EVAL.md` (H-NAVC-1..3, H-SEAM-1, H-H19-1, H-CONS-1, H-SEL-1; deliberate-regression arm must FAIL the echo gate); the new **`t_bin` accuracy** primary for E19 (a per-clip-constant nav cannot echo *when*) | PI rulings R1, R2; `D-NAVCOMP-1..3` | Every refcv5 number that touches nav or `g_str` carries the shuffle / zero (and flip) controls; every added seam (E15–E21, E-AGT-1, E-BEV-1, S7/S8) ships with its eval-time switch so the ablation table of the prereg extends by one row per edge. |
| **I3 — vision-only at inference, with anti-echo** | no ego state beyond the measured `v0` at t0 (PI 2026-09-02: velocity at cycle time is admissible), no future, no label, no situation-classifier output in any goal input (PI 2026-08-03); `ego_dropout 0.5` + the X15 presence bit are the measured lever (`H-ECHO-8`: the only `READS_BOTH` arm); the withheld-row bank follows the **`H-EGO-LIT-4` panel outcome** (A1 predicted-speed bank vs A4 speed-blind vocabulary; running on the 4060 — cited as pending); goal/situation disjointness | CLAUDE.md binding rules; `H-ECHO-8`, `H-EGO-LIT-1..4`, `D-EGOLIT-WITHHELD1`, `H-EGODROP-PRED`; `provenance_roles()` (`refc_v3.py:786-833`) measured interventionally | Grid-sample attention reads *scene features* at candidate waypoints (no label, no future). Agent tokens are produced by the trunk at inference; `obstacle.offline` is read only at train time (labels may use anything). The selector's targets are rule-based sub-scores against **replayed agents**, never the ego's GT future. The RL reward table (§6) refuses every GT-future term. ⚠️ **LiDAR at inference is a *sensor*, not a privileged channel — but the constitution's literal wording is "vision-only"**; the Mission Plan (`Mission Plan.md:181`) and ROADMAP L4 foresee LiDAR/radar fusion. ⇒ default keeps inference camera-only (LiDAR as teacher); the LiDAR-input arm needs the PI's ruling (`D-REFCV5-PLAN-7`). |
| **I4 — the v0-conditioned control vocabulary, flyable, self-describing** | 117 constant-`(a_lon, a_lat)` controls rolled per window from the measured `v0` (`refc.py::roll_bank`); vocabulary v5 adds the **realised-speed κ clamp** (`D-REFCV4B-FLYLOW1`: the cap bounds curvature, not load — 5.51 g at v0 = 4 m/s) and a build-time Kamm filter; artifacts carry their units (`tanitad/refs/anchor_meta.py`, `SCHEMA tanitad.anchor_artifact/1`) and MODEL-FREE gates are never compared to achievements (`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`) | `D-REFCV4-VOCAB1`, `D-REFCV4B-FLYLOW1`, `D-REFCV4B-TURNCOV1`, `anchor_meta.py` | **This is why the diffusion noise lives in control space** (§4): a metre-space Gaussian around a rolled anchor produces samples no car can drive; a control-space Gaussian re-rolled through the unicycle cannot. V2's 2-scalar exploration is likewise a scale on `(a_lon, a_lat)`. The vocabulary stays the prior; the sampler explores around it. |
| **I5 — four families, tiers, estimator** | LONGITUDINAL (speed, along-track, headway / time-gap / TTC to the lead) · LATERAL (heading, curvature, yaw-rate, cross-track) · TACTICAL (decision quality, confusion, goal setting) · STRATEGIC (compliance, `t_bin`) — never pooled; T0 = diagnostic, **T1 = the capability tier** (self-action open loop; never "driving"); paired episode-cluster bootstrap over the 141 EVAL episodes (`taniteval/ci.py`), never `overlapping_holdout_se`; hold-action floors `ha`, `ha0`, `ha0_ext` + the echo gate; criteria checker 0 violations | `EVAL_DOCTRINE.md`, CLAUDE.md, `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.6.0 | The selector (§5) is *trained* on family-shaped sub-scores and *read* on the families (`H-DDA-7` refuses the ADE-only reading in advance). The RL stage's capability claim is T1 with the four families, never the reward. |
| **I6 — corpus and parity** | refcv5 trains on refcv4b's corpus (B1-v7.2: 4,572 train / 141 eval clips, labels md5 `0ff90213…` / `aa12c948…`) so refcv5-vs-refcv4b is a valid **arm** delta; the sacred parity corpus (`physicalai-train-e438721ae894`, 2,376 episodes) is untouched; the BEV sidecar (§3) adds bytes to no `*.v2ep.pt` — the frames stay byte-identical | CLAUDE.md invariants; `MODEL_REGISTRY.md` §4.6; `stack/tanitad/data/parity_manifest.json` | A sidecar cannot change episode membership or frame bytes; the parity assertion at trainer start is unchanged. |

---

## §3 The BEV / LiDAR plan — the data question settled first

### 3.1 Does PhysicalAI-AV carry LiDAR? Yes — three independent probes

| probe | mechanism | what it says | evidence class |
|---|---|---|---|
| **P1 — HF blob listing** (2026-09-05, this stream, `api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles?blobs=true`, token read in place) | file tree + sizes | `lidar/lidar_top_360fov/` — **3,146 chunk zips** (the same count as every camera feature), **99,601.7 GB total, 31,659.8 MB per chunk**; `calibration/lidar_intrinsics.offline/` — 3,146 parquets, 16.2 GB, 5.2 MB/chunk; 18 radar features (306–993 chunks each); dataset total **133,214 GB** — matches the card's published 133 TB. Naming: `lidar_top_360fov.chunk_NNNN.zip` | **MEASURED** (this stream; the listing is reproducible with the command above) |
| **P2 — the DataFlyWheel's 36-feature probe** (2026-07-26, `…/Data Engineering/Implementation/incoming/2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md` §1, groups B and D; `pai_features.csv` pins the 36) | one measured chunk × chunk count; `metadata/feature_presence.parquet` (306,152 × 36) | row 17 `lidar_top_360fov`: **Draco-compressed point clouds, ~200 spins/clip @ 10 Hz, coverage 97.44 %, 32,340 MB/chunk, 101.7 TB — 68 % of the dataset by bytes**; row 6 `lidar_intrinsics.offline`: `model_type = 'row-offset-spinning'`, **128 rows × 3600 columns**, 10 Hz, per-row elevations; row 8 `sensor_extrinsics.offline`: 8 sensors = 7 cameras + LiDAR; row 3 `obstacle.offline`: 3D tracked cuboids, ~39 tracks/clip, 97.44 % | **MEASURED** by that stream (INHERITED here, consistent with P1 to 2 %) |
| **P3 — the dataset card** (`README.md` at `main`, fetched 2026-09-05) | publisher's statement | "LiDAR coverage for 298,326 clips" of 306,152 (= 97.44 %); each chunk holds ~100 `<clip_uuid>.lidar_top360_fov.parquet` files with "approximately 200 lidar spins (i.e. 10 Hz capture rate for a 20 sec clip)"; decodable with `DracoPy`; changelog 25.10 → 26.03: LiDAR present "for 97 % of clips; per-clip lidar presence is now recorded in `metadata/feature_presence.parquet`" | **PUBLISHED** (dataset card) |

**Why our ingest never saw it:** the pinned read-set (`stack/tests/test_physicalai_feature_readset.py`) is 2 / 5 / 6 features by layer; `physicalai.py` declares `_FRONT_WIDE_CAM`, `_CALIB_INTR`, `_CALIB_EXTR`, `_CALIB_VEH` + `egomotion` and nothing else (the drift detector fails on any new path — so **adding LiDAR is a declared change to that test, not a silent edit**). `grep -i lidar stack/tanitad/` returns nothing, with a same-breath control (`egomotion`: 293 files in the Lab) — a real absence at two probes.

**Per-clip cost (derived from P1/P2):** 306,152 clips / 3,146 chunks = 97.3 clips per chunk ⇒ **≈ 325 MB of Draco-compressed LiDAR per 20 s clip** (ESTIMATED from the chunk mean; per-clip variance unknown until the pilot), ≈ 1.6 MB per spin; a full 128 × 3600 spin is 460,800 points max ⇒ ≈ 7.4 MB uncompressed at 4 × float32. For the B1 corpus (4,572 + 141 = 4,713 clips): **≈ 1.53 TB of LiDAR must transit** (ESTIMATED) — whole-chunk pulls cost `n_chunks × 31.7 GB` and the B1 selection's chunk spread is not banked (the r0 tool selects 30 egomotion chunks stratified by country, `physicalai_r0.py:17`; B1 is a later, larger selection). ⛔ **A pod cannot hold it**: the MooseFS quota is ~466 GB and `df` cannot see it (CLAUDE.md) — the build must **stream** (chunk in → rasters out → chunk deleted), or use HTTP-range member reads from the zip central directory (`huggingface_hub.HfFileSystem` + `zipfile`; HYPOTHESIS until the pilot measures it).

### 3.2 What we already read, and the projection facts

| fact | value | evidence |
|---|---|---|
| calibration in the episode build | `camera_intrinsics` (f-theta `poly`, `cx`, `cy`, `width`, `height`), `sensor_extrinsics` (per-clip × 8 sensors: `qx..qw, x, y, z`), `vehicle_dimensions` (`wheelbase` etc.) — all 100 % coverage | `physicalai.py:232-235` (consumed at `:354`, `:407`, `:456`), pinned by `test_physicalai_feature_readset.py` |
| LiDAR → rig transform | in `sensor_extrinsics(.offline)` (8 sensors = 7 cameras + LiDAR) — the rig frame is the frame `obstacle.offline` cuboids are expressed in (`reference_frame == "rig"` on every row; **x forward, y left, z up**, MEASURED by the parked-car experiment, 7.4× over the nearest alternative — `bev_raster.py` docstring, join doc §2) and the same convention as `refb_labels.ego_frame` (`refb_labels.py:86-90`) | `bev_raster.py:1-96`, `build_obstacle_join.py` header |
| ⇒ a LiDAR BEV in the rig frame is **directly indexable by the anchor fan** (both are ego-frame metres, +x forward, +y left); no camera projection is involved in building or reading it | — | derived |
| our frames | **256 × 640, CYLINDRICAL, `f_ref` 305.577** — the column is linear in azimuth: `HFOV = 2·(W/2)/f_ref = 2·320/305.577 = 2.0944 rad = 120.0°`, matching the rig's own `camera_front_wide_120fov`; the pinhole formula `2·atan((W/2)/f)` gives 92.6° and is **wrong** on this projection (CLAUDE.md FOV trap) | `calib.py:75-90` (`CanonicalFrame.projection`), `physicalai.py:107-131` (`projection_mode="cylindrical"`, `cylindrical_rectify`) |
| projecting a rig-frame point into our image (needed only for E-DDA-1 PV sampling and for overlays) | `φ = atan2(y_cam, z_cam)` → `col = f_ref·φ + W/2`; vertical is tan-based (`row = f_ref·(y_up/ρ) + H/2`, ρ = horizontal range), per `CanonicalFrame` (`calib.py:88-90`, `:151-157`); `bev_raster.readout_column_index(projection="cylindrical")` already maps azimuth → column | `calib.py`, `bev_raster.py:388-441` |
| episode time grid | realised **~0.1007 s**, not 0.1 s (`register_poses_to_time`); LiDAR spins are 10 Hz ⇒ nearest-spin matching with a stated tolerance; agents are a per-track temporal lookup (`agents_at_time`, tol `DEFAULT_TOL_S`) | `bev_raster.py` docstring, `taniteval/lead_source.py` |
| an agent BEV raster **already exists** | `tanitad/data/bev_raster.py` (P8): ego-frame occupancy over `obstacle.offline` cuboids, grid 60 m forward × ±16 m at 0.5 m → [120, 64], with `fov_mask`, `readout_column_index`, `fov_census`; fed by the pod-side join `build_obstacle_join.py` (jsonl per labelled episode frame, with the P4 visible/occluded flag) | source read 2026-09-05 |

### 3.3 The engineering plan (DataFlyWheel work package **WP-DE-BEV-1**, owner: DataFlyWheel; consumer: Arch+Inference)

**Target artifact — priced against the CONSUMER's loader, per CLAUDE.md.** The trainer's consumer is `tanitad/data/v2_dataset.py::V2CompressedCache._payload` → `_decode_stacked` (reads `jpeg_buf`, `jpeg_len`, `n_stack`, `codec` from `*.v2ep.pt`, written by `scripts/v2_compressed.py::build_compressed`; frames decoded only for the `[a:b]` window). The BEV goes in a **sidecar `<clip_id>.v2bev.pt`** beside each `*.v2ep.pt` — `{"bev_buf", "bev_len", "bev_codec": "png", "bev_grid": {...}, "align": {...}, "provenance": {...}}` — decoded for the same `[a:b]` frame range by a `_V2BevProxy` mirroring `_V2FramesProxy`. **Rationale:** zero rewrite of the 161 GB camera cache, frames byte-identical (I6), old readers unaffected, and a missing sidecar is a loud `NO_BEV` state, never zeros (the `jpeg_buf`-is-PNG / zero-memmap trap, CLAUDE.md).

| item | spec | cost (class) |
|---|---|---|
| BEV grid | DD/NAVSIM-faithful: **256 × 256 @ 0.25 m/px**, x ∈ [−32, 32] m, y ∈ [−32, 32] m (ego-centred; LiDAR is 360°), **2 height bins** split at the NAVSIM ground threshold (0.2 m above the rig ground plane), uint8 point counts clipped at 255; stored PNG (lossless; `v2_dataset.py:325` refuses to sub-frame a lossy cache, and PNG keeps that property) | PUBLISHED grid (`transfuser_config.py`, audit §1.5 #32) |
| channels | ch0 = below-threshold counts, ch1 = above-threshold counts (TransFuser's 2-bin histogram); optional ch2 = the P8 agent raster re-gridded to the same frame (agents from `obstacle.offline`, train-time only — a **label**, kept in a separate key `agent_buf` so it can never be fed at inference by accident) | — |
| alignment | per episode frame `t_f` (the realised grid) → nearest LiDAR spin `|t_spin − t_f| ≤ 50 ms`, else `NO_SPIN` (a state, never an empty raster); spin timestamps from the parquet; motion compensation of the 100 ms sweep with `egomotion` (~111 Hz) if the parquet carries per-point timestamps — the pilot decides | — |
| per-frame bytes | raw 256 × 256 × 2 = 131,072 B; PNG **ESTIMATED 15–40 KB** (sparse far field); per 199-frame episode raw 26.1 MB / PNG **≈ 3–8 MB** vs 34.0 MB of camera (MEASURED, CLAUDE.md `*.v2ep.pt`); B1 corpus (4,713 eps): raw 123 GB / PNG **≈ 14–38 GB** — fits beside the 161 GB cache under the ~466 GB pod quota | ESTIMATED; the pilot measures it |
| decode at train time | PNG decode of a 256 × 256 × 2 image ≈ 1 ms; 8 frames per window ⇒ negligible next to the 9-channel 256 × 640 camera decode | ESTIMATED |
| LiDAR transit | ≈ 325 MB/clip × 4,713 = **≈ 1.53 TB** (ESTIMATED), streamed; at the HF ~118 MB/s measured on pods (CLAUDE.md) ≈ **3.6 h of transfer**; Draco decode ≈ 200 spins × (10–30 ms) ≈ 2–6 s/clip (ESTIMATED; `DracoPy`) + histogram (numpy `bincount`, ms) ⇒ **≈ 3–8 h single-core, parallelisable** — the pilot on ONE chunk turns every ESTIMATE here into a MEASURED figure before the corpus build is scheduled | ESTIMATED |
| where it runs | a pod with HF quota and the B1 cache (never the training pod while it trains); dev-box pilot on 10 clips first | — |

**Steps and gates (verify by CONTENT, never by exit code):**
1. **Pilot (dev box, 10 clips, ≤ 1 day):** pull `lidar_intrinsics.offline` + `sensor_extrinsics.offline` for the chunk; range-read or pull ONE LiDAR chunk; decode 10 clips with `DracoPy`; **print** points/spin, field names (x, y, z, intensity, per-point timestamp?), spin timestamps and rate, decode time; confirm the 128 × 3600 layout against `lidar_intrinsics.offline`. Bank `raw/lidar_pilot.json`. *Gate:* the rig-frame point cloud of a known parked-car frame shows the car where `obstacle.offline` puts it (the same cross-check that settled the axis convention) — a control that must read a known value.
2. **Rasteriser (`stack/tanitad/data/lidar_bev.py`, pure numpy, tests first):** LiDAR → rig frame via the extrinsics → 2-bin histogram on the grid above; asserts on content (non-zero, mean occupancy printed, an all-zero raster is a refusal); a deliberate-regression test feeds a wrong extrinsic and must see the parked car move.
3. **Cross-check with the agent raster:** point density inside `obstacle.offline` footprints ≫ outside (a ratio that must exceed a pre-registered bar on the pilot clips) — this is what makes the LiDAR raster trustworthy before any model sees it.
4. **Sidecar builder (`scripts/v2_bev_sidecar.py`):** streams chunk → `<clip>.v2bev.pt`, `MANIFEST-last` protocol, sha256 per file, provenance stamp (builder sha256, grid, split threshold, tolerance, `NO_SPIN` count); **re-times the real path on one chunk** before the corpus run (CLAUDE.md: never port an old timing onto a new format).
5. **Loader (`v2_dataset.py`):** `_V2BevProxy`; `LazyV2Episode` gains `.bev` (and `.agent_bev` train-only); `V3Dataset.__getitem__` (`refc_v3_train.py`) emits `item["bev"]` [W, 2, 256, 256] u8; parity assertion unchanged; **a test that a window without a sidecar raises `NO_BEV`**, never returns zeros.
6. **Declare the read-set change:** `test_physicalai_feature_readset.py` gains `lidar_top_360fov`, `lidar_intrinsics.offline`, `sensor_extrinsics.offline` as a **new layer ("the BEV sidecar build")** — the layer is named, the counts are pinned, the failure message names the docs (CLAUDE.md's four-times-rotted count).
7. **Corpus build** on the pod, streamed, B1 train + eval; publish the sidecars PRIVATE on the PI's HF account (an augmented dataset, per the constitution).

### 3.4 The model side, in three arms (each one lever; §7 for the ladder)

| arm | what enters the model | supervision | inference input | invariant check |
|---|---|---|---|---|
| **E-AGT-1** — agent tokens | the trunk's PV tokens (and later BEV tokens) → DETR-style head, K = 30 queries → `cross_agent` in every decoder layer (zero-init gate) | `obstacle.offline` boxes (Hungarian; train-time label) | camera only | I3 ✔ (labels at train time; the trunk emits the tokens at inference) |
| **E-BEV-1** — camera BEV lift, LiDAR-taught | a lift from the PV map to a metric BEV (LSS-style depth-distribution splat or BEVFormer-style deformable queries; ⚠️ the 8 × 20 map is 6° per azimuth bin — ≈ 3 m lateral at 30 m — so the lift reads the trunk's stride-8 stage, 32 × 80, a trunk change to be priced on the tiny rig); grid-sample attention (#19-ii) on the lifted BEV | **the LiDAR histogram + agent raster as BEV-occupancy targets** (a teacher, train-time only) | camera only | I3 ✔ (LiDAR is a label here) |
| **E-LIDAR-1** — LiDAR BEV as input | the LiDAR histogram through a small BEV encoder → tokens into the decoder KV (late fusion, #33); grid-sample attention on the LiDAR BEV | as E-BEV-1 | **camera + LiDAR** | ⛔ gated on the PI's doctrine ruling (`D-REFCV5-PLAN-7`); paired against E-BEV-1 on the same windows so **what the sensor buys beyond what the camera can be taught** is the measured quantity |

**If LiDAR turns out too costly or is refused at inference:** the camera-only path is complete without it — E-DDA-1 (waypoint-indexed PV sampling, no BEV at all), E-AGT-1 (agent tokens from `obstacle.offline`, the attention DD has and we lack), and E-BEV-1 with the **agent raster alone** as the BEV target (the P8 rasteriser exists today, zero new data). The order of the ladder (§7) puts these first precisely so that the LiDAR build is never on the critical path.

**New primaries the library stream should bank for §3 (not banked by this stream — a sibling owns `kb_add.py` this morning):** LSS — *Lift, Splat, Shoot* (Philion & Fidler 2020, arXiv 2008.05711); SimpleBEV (Harley et al. 2022, arXiv 2206.07959); PointPillars (Lang et al. 2019, arXiv 1812.05784). Already banked and cited here: TransFuser `2205.15997`, BEVFormer `2203.17270`, DiffusionDrive `2411.15139`, DiffusionDriveV2 `2512.07745`, GTRS `2506.06664`, DriveSuprim `2506.06659`, DIVER `2507.04049`, DPPO `2409.00588`.

---

## §4 The diffusion mechanism (`H-DDA-3`) — anchored Gaussian, DDIM, x0, AdaLN — in CONTROL space

### 4.1 What is adopted from DD, item by item (PUBLISHED, `hustvl/DiffusionDrive@9b52ed0`; audit §1.1)

| item | DD | refcv5 |
|---|---|---|
| schedule | `diffusers.DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample")` (+ V2's `steps_offset=1`) | same object, behind `DecoderConfig.sampler = "ddim"` (default `"none"` = today's decoder, byte-identical, pinned by a parity test like `tests/test_refc_v4.py::test_v3_parity`) |
| forward noising | τ_t = √ᾱ_t·a_k + √(1−ᾱ_t)·ε in normalised metres | same formula on the **normalised control sequence** (§4.2) |
| train timestep | t ~ U[0, 50) | same |
| inference | fresh ε at t = 8, two DDIM steps `[10, 0]`, η = 0 | same; G ≥ 1 samples per anchor; eval at a **fixed seed** (bit-reproducible at the seed; a seed sweep is the variance readout — the old "two calls bit-identical" property is replaced, not lost) |
| prediction target | x0 ("sample"): `poses_reg = delta + noisy_input` | x0 in control space: `u0_hat = u_t + Δ(u_t, t, scene)` |
| timestep embedding | `SinusoidalPosEmb(d) → Linear → Mish → Linear`, injected **per layer** as AdaLN scale/shift after the FFN | same; `CrossAttnLayer` gains a `time_mod: FiLM(d_t, d)` after its MLP; the 3-row `nn.Embedding` is retired under the flag |
| query init | sinusoidal Fourier features of each waypoint (64/pt) → MLP | same, on the **rolled waypoints** of `u_t` (the query still sees geometry) |
| loss | per cascade layer `(reg, cls)`, one random t per sample, coordinates detached between layers | per refinement pass: L1 on the **rolled** waypoints (today's `loss_traj`) + L1 on the controls vs `unicycle_controls_from_path_varstep(GT)` (the x0 loss); **detach** `u` between passes; keep the t = 0 CE on the classifier surface; add the emitted-fan confidence CE (§5.1) |
| mode-collapse control | truncation (start near the anchor, 2 steps) + per-mode cls | same — **the pass count stops being a knob** (`D-REFC-DDAUDIT-2`: passes beyond 2 collapse the spread 104 → 42 m) |

### 4.2 The design question ours raises — noise in CONTROL space, not metre space. Decision: CONTROL space.

**The state.** A candidate is a control sequence `u ∈ ℝ^{8×2}` — one `(a_lon, a_lat)` pair per slot of the 8-slot 0.5…6 s grid — rolled through the programme's own integrator (`kinematic.rollout_unicycle_varstep`, `unicycle_decode_varstep`, `kinematic.py:741,887`) from the window's `v0`. The anchor `a_k` is the constant pair `(a_lon_k, a_lat_k)` repeated over the 8 slots — exactly today's `anchor_controls` (`refc.py::roll_bank`, `:1378-1439`) with the κ derivation `a_lat / max(v, 4)²` clamped ±0.12 unchanged (vocabulary v5 adds the realised-speed clamp, §2 I4). Normalisation for the scheduler: `a_lon / 4.0`, `a_lat / 3.0` — the grid's own ranges (`a_lon ∈ [−4.0, +2.9167]`, `a_lat ∈ [−3.0, +3.0]`, `MODEL_REGISTRY.md` §4.6) — so [−1, 1] means the same thing the vocabulary means. The GT target in control space is `unicycle_controls_from_path_varstep(GT waypoints, dts)` (`kinematic.py:778`), which exists and is tested.

**Why control space (the argument):**

1. **Flyability is preserved by construction.** ESTIMATED from the `scaled_linear` schedule (arithmetic, this stream, reproducing the audit's 0.899 / 0.727 m at t = 8 as a control): √(1−ᾱ₈) = **0.0316**. In DD's metre normalisation that is σ_x 0.90 m / σ_y 0.73 m **per waypoint, independently** — over a 0.5 s slot a 0.73 m lateral excursion is an implied lateral acceleration of ≈ **5.8 m/s²** (0.6 g) and a 0.91 m along-track one a **1.8 m/s** speed jump; the sampled fan would be full of trajectories no car can drive, which is precisely what the selected-trajectory smoothness finding (`D-REFCV4B-EGODROP2` §3: jerk 2.42 vs the human's 0.86 m/s³) says we must not add. In control units the same schedule gives σ(t = 8) = **0.126 m/s² along / 0.095 m/s² lateral**, i.e. ≈ **2.3 m along-track and ≈ 1.7 m lateral at the 6 s endpoint** (the lateral figure is speed-independent because κ = a_lat/v² cancels against path length) — comparable to DD's ~0.9 m endpoint noise at their 4 s horizon, but every sample is a smooth, integrable path. At t = 49 (the training maximum) σ = 0.377 / 0.283 m/s².
2. **The vocabulary stays the prior.** DD's premise — anchors near the modes, small noise, two steps — holds only if the bank is a real prior; `E-DDA-4` (a free readout from the refcv4b dump: classifier-pass ‖offset‖ mean, oracle-in-bank vs oracle-in-fan) decides that before the sampler is trained. If offsets are still metres-scale at 40 k, the fork is *not* a bigger σ; it is a denser / re-centred grid (§1.2 #13), because a sampler cannot fix a vocabulary (`D-REFCV4-VOCAB1`).
3. **One parametrisation for the sampler and the RL policy.** V2's released exploration is two scalars per anchor scaling the mean trajectory along-track and laterally (`D-DDV2-CODE-2`). In control space that is a scale on `(a_lon, a_lat)` — the *same* two numbers the sampler noises. E-DDA-3b (§6) and E-DDA-3 share the state, the integrator and the flyability guard, and the RL stage needs no chain density (`D-DDV2-ORDER-1`).
4. **The resolution objection is answered by the sequence, not by metres.** A constant `(a_lon, a_lat)` held for 6 s cannot trace a real 6 s trajectory (turn-coverage §5: the ceiling degrades 0.87 → 1.97 m with turn magnitude, a model-class limit, not a coverage hole). The **8-slot control sequence** has the same 16 degrees of freedom as the waypoint offset it replaces, so nothing expressible before becomes inexpressible — and `control_smoothness_losses` (`kinematic.py:820`) regularises it in the programme's own units.

**What is given up, stated:** DD's x0 is directly the waypoints, so its loss is one L1; ours rolls through the integrator, which is ~60 sequential steps in float32 per candidate (`roll_bank` already pays this per window: 117 × 60 steps) — the sampler adds G × 2 refinement rolls per window. On the A40 the current forward is 3.844 s/step at batch 20 (`MODEL_REGISTRY.md` §4.6); the roll is a small fraction of that (ESTIMATED — the tiny-rig arm measures it, and the `--anchor-prefilter` S2b path can still decode only reach-survivors because candidates stay independent).

**Deliberate-regression arm for E-DDA-3 (must FAIL):** the DD-literal metre-space sampler (`norm_odo` on waypoints). Pre-registered expectation: it fails the flyability gate (Kamm μ = 0.7 violations on the sampled fan ≫ the control-space arm's) and raises selected-trajectory jerk; if it does *not*, the flyability instrument cannot see what it is cited for and the arm is VOID. **Control that must read a known value:** the sampler at σ ≡ 0, t = 0, reproduces the deterministic decoder bit-for-bit.

### 4.3 File-level plan (0 GPU to implement; validated on the tiny rig per §7)

| file | change |
|---|---|
| `stack/tanitad/refs/refc.py` `DecoderConfig` (`:380-387`) | `sampler: str = "none"` (`"none" | "ddim"`), `sampler_train_t_max: int = 50`, `sampler_infer_t: int = 8`, `sampler_steps: int = 2`, `sampler_groups: int = 1`, `control_norm: tuple = (4.0, 3.0)`; `noise_std` retired under `"ddim"` |
| `AnchoredDiffusionDecoder.__init__` (`:1134-1260`) | under `"ddim"`: `time_mlp` (sinusoidal → Linear → Mish → Linear), per-layer `time_mod` FiLM in `CrossAttnLayer`, `wp_embed` (Fourier features of rolled waypoints → d); `offset_head` becomes `control_head: Linear(d, 8·2)` predicting Δu; a `DDIMScheduler` instance (diffusers; a 30-line local re-implementation if the dependency is refused on pods) |
| `forward` (`:1541-1770`) | classifier pass unchanged (t = 0 token, priors, reach band); under `"ddim"`: `u_t = add_noise(u_anchor, ε, t)` (train: t ~ U[0, 50); eval: t = 8, G groups), each refinement pass rolls `u_t` → waypoints → `wp_embed` → layers with `time_mod(t)` → `Δu` → `u0_hat`; DDIM step to the next `u_t`; detach between passes; emit the rolled fan `[B, G·N, S, 2]` + `u0_hat` |
| `stack/tanitad/models/kinematic.py` | no change — `rollout_unicycle_varstep`, `unicycle_controls_from_path_varstep`, `control_smoothness_losses` exist; add a batched `roll_controls(v0, u, dts)` wrapper with a test against `roll_bank`'s constant-control case (must be bit-identical for a constant `u`) |
| `stack/scripts/refc_v3_train.py` `compute_losses_v3` (`:537-575`) | `steps` logic unchanged; add `loss_u0 = L1(u0_hat[a_star], u_gt)` with `u_gt` from `unicycle_controls_from_path_varstep`, per-pass `loss_traj` on the rolled fan, `loss_smooth` (weight 0 by default), all stamped in `config.json` |
| `taniteval/tools/refcv3_arm.py`, `taniteval/plan_fan.py` | dump `u0_hat`, the G-fan, the seed; implement DD's diversity score D (Eq. 3 of `2411.15139`) and the sampled-fan Kamm violation rate as readouts |
| tests | parity (sampler off ≡ refcv4b), σ ≡ 0 identity, constant-`u` roll ≡ `roll_bank`, deliberate-regression (metre-space arm fails flyability) |

---

## §5 Selection — score the fan actually emitted, then the four-family selector

### 5.1 Step 1 — E-DDA-2, zero parameters: the ranked object becomes the emitted object

MEASURED defect (`D-REFC-DDAUDIT-3`): the ranking reads the t = 0 confidence and the fan leaves two passes later; `sel_idx_base` is unchanged on 201/201 windows for every `steps`. The fix exists in source: `SelectionConfig.refined` + `score_emitted` (+ `score_emitted_t`) make one extra confidence-only pass on the emitted fan (`refc.py:1703-1710`), and `sel_ce_reach` supervises that confidence over the reach-surviving set (E-SEL-0 showed the *unsupervised* refined readout ranks worse — so the arm is the supervised one, `H-DDA-2`). The priors (H19, factored, LAN, goal) are re-applied to the refined readout (`refc.py:1677-1683`), so turning this on deletes nothing from the hierarchy's seam. **First free read:** `H-SEL-1` in refcv4b's post-training ablation (eval-time switch; pick changes on ≥ 10 % of windows without ADE loss). **Ladder:** paired vs the all-off control on the tiny rig; readouts = the selection gap (`oracle_sel − os`; 0.0751 m [0.0618, 0.0884] on refcv3, `D-REFCV3-40284a`) and the four families. Under the §4 sampler the "emitted fan" is the G·N rolled fan and this pass scores it — the two levers compose without a second mechanism.

### 5.2 Step 2 — E-DDA-2b, V2's coarse-to-fine sub-metric selector with OUR heads (no PDM, no map)

**What V2 does** (`D-DDV2-CODE-4`, `diffusiondrivev2_model_sel.py` @ `1cd12a1`): candidate embedding (sinusoid of (x, y) + heading) → coarse scorer (1 layer: grid-sample BEV attention → agent MHA → **self-attention among candidates** → ego MHA → FFN; five heads NC / EP / DAC / TTC / C; BCE against PDM sub-scores + `MarginRankingLoss(0.05)` on EP pairs × 2) → top-32 → fine scorer (3 layers) → argmax of `σ(NC)·σ(DAC)·(5σ(TTC)+5σ(EP)+2σ(C))/12`. Trained with the generator frozen, 20 epochs; the v1 classifier is abandoned. Authors' control: the selector alone bought **+1.0 PDMS in the hackish direction** (NC 98.2 → 97.2, TTC 94.7 → 92.2, EP 82.2 → 86.8; Tab. 9) — the pattern `H-DDA-7` pre-registers for us.

**Our adaptation — heads are the four families, targets are the environment, never the answer:**

| head | target (train-time, rule-based, against the `obstacle.offline` REPLAY join) | family | instrument that exists |
|---|---|---|---|
| `NC` | no contact with any replayed agent over the 6 s candidate (time-aligned moving lead, `rl/rewards.py::_collision` — the static-lead defect is fixed, `D-RL-READY-1` #3) | LONGITUDINAL (distance keeping) | `rewards.py`, `taniteval/lead_metrics.py`, `lead_source.py` |
| `TTC/headway` | min TTC ≥ threshold and time-gap ≥ threshold along the candidate vs the lead's own track (`ttc_violation`, `_headway`; thresholds from `PREREG_D_SAFE_CAL`, calibrated so the human is not flagged — `H-RL-THRESH-1` class clear at 0.047 on fit8) | LONGITUDINAL | same |
| `comfort/feasibility` | jerk, lateral acceleration, curvature, yaw-rate within bounds on the rolled path (`_comfort`, `_kinematic_feasibility`; `flyability.py` Kamm) | LATERAL + comfort | `stack/tanitad/instruments/flyability.py` |
| `progress` | along-track displacement relative to `v0` (hackable — see §6; a *head*, never the only head) | LONGITUDINAL (progress) | `_progress` |
| `tactical-consistency` | the candidate's `(a_lon, a_lat)` cell class agrees with the v7.2 factored label (`vocab_v7.py`) — ⚠️ a label derived from the ego future, i.e. a **partial echo** (V2 analysis §3.3); admissible as a selector target only if reported separately and **never used as both the RL grouping key and a reward** (`H-DDA-6` rule) | TACTICAL | `refc_tactical.py`, `v7_labels.py` |
| `compliance` | the candidate's terminal heading agrees with the nav command on informative windows — the compliance metric's own predicate (`nav_compliance.py`, τ_plan from GT + label only); the selector is *told* the command here (this is S7 in learned form), so the read is the shuffle / zero delta | STRATEGIC | `taniteval/taniteval/nav_compliance.py` |

Score composition: `σ(NC) · (5·σ(TTC) + 5·σ(progress) + 2·σ(comfort) + w_c·σ(compliance) + w_t·σ(tactical)) / (12 + w_c + w_t)` — PDMS's structure with the map half removed and the two hierarchy heads added; `w_c`, `w_t` are config, reported, and ablated as switches (I2). ⛔ `DAC` (drivable area) is not constructible on PhysicalAI (no map, pinned); on the AlpaSim/NuRec arm `map.xodr` supplies it (W6).

**Fan and augmentation.** Candidates = the 117-fan (× G under the sampler) + **2 multiplicative augmentations in control space** (`u·(1+s)`, `s ~ U(0.1, 0.2)`, V2's train recipe) + ≈ 1 % foreign bank (the 6 s k-means vocabulary `D-REFCV4-VOCAB1` that the arm does not use — V2's GTRS-vocabulary trick) ≈ 350–700 candidates. Scorer width d = 256, 8 heads (V2: 512 / 16 over 800 — half our budget, Thor-minded); coarse 1 layer → top-32 → fine 3 layers; candidate self-attention lives **only here** (the generator keeps candidates independent, §1.3 #22). Scene attention = the E-DDA-1 / E-BEV-1 grid-sample module, reused; agent MHA = E-AGT-1 tokens if present, else omitted (ablatable). Losses: BCE per head (V2's NC 0.5 → 0 mapping; comfort masked where undefined) + `MarginRankingLoss(0.05)` on progress pairs × 2. **Generator frozen** (stage II), trained on the dev box from the refcv4b final checkpoint's fan — it can start the day refcv4b lands, at 0 pod GPU.

**Why this is not the re-scorer SEL-1 refused (winner's curse):** v1.2's `refc_rescorer.py` was trained toward the fan's own GT-distance winner; these targets are environment predicates that the GT does not enter (except the tactical head, quarantined above), and E9's goal detach stays hard.

**Readouts (pre-registered in `H-DDA-7`):** the selection gap closes with CI excluding 0 **and** fan-collision-vs-replay does not rise ⇒ adopt; closes but collision rises ⇒ the Tab. 9 hackish direction, refuse; no change ⇒ selection is not binding at this fan quality, bank. Four families always; ADE-only reading refused in advance. Deliberate regression: the **progress-only head** must buy progress with collisions, or the collision readout cannot see collisions and the panel is VOID. Nav shuffle / zero controls on the compliance head (I2).

**Files:** new `stack/tanitad/refs/refc_selector.py` (`SubMetricSelector`, coarse/fine, heads, composition); `stack/scripts/refc_selector_train.py` (frozen generator, dumps in, rule targets from the replay join — reuses `rl/rewards.py`); `taniteval/tools/refcv3_arm.py` gains `--selector <ckpt>` and dumps `sel_idx_selector` beside `sel_idx` / `sel_idx_base`; the criteria registry gains the selector's per-head calibration (ECE) as a reported, not gated, criterion.

---

## §6 The RL stage — a head-only scale policy whose job is to remove collision-prone and infeasible candidates from the fan

### 6.1 Purpose, re-scoped by the PI (2026-09-05)

Not to chase ADE. V2's published shape is the target: the raw fan's **floor** rose (PDMS@10 75.3 → 84.4, +9.1) while the top barely moved (@1 93.5 → 94.9) and diversity fell 28 % (`D-DDV2-RES-1`, Tab. 3) — the fan got *safer*, not *closer*. Our corresponding readout does not exist yet and is a work item: **`fan_floor@k`** — the k-th-best candidate's four-family score (k = 1, 5, 10) and the **fan-collision rate vs replay** — implemented in `taniteval/plan_fan.py` beside DD's diversity D (§4.3). A capability claim after RL is T1 on the four families; the reward is never quoted as a result (`EVAL_DOCTRINE.md`, I5).

### 6.2 Mechanism (V2's released code, `D-DDV2-CODE-2/3`, ported one-to-one where our constraints allow)

| ingredient | V2 (`_model_rl.py` @ `1cd12a1`) | refcv5 | state in `stack/tanitad/rl/` |
|---|---|---|---|
| trainable surface | `_trajectory_head` only; trunk + agent/BEV heads frozen **and in eval mode** | `core.decoder` minus the selector tensors (9.21 M / 107.03 M = 8.60 % on refcv3, MEASURED `D-RL-READY-1`); goal heads, `phi_tac`, `str_goal_head`, the §5 selector and the trunk frozen | `posttrain.py::select_trainable(freeze_trunk=True)` + the 15-prefix tripwire — exists |
| policy | 2 scalars per anchor scale the DDIM mean along / laterally, `σ ≥ 0.04` (the additive term × 0) | 2 scalars per anchor scale `(a_lon, a_lat)` of the emitted control sequence, `u' = u ⊙ (1 + 0.04·ε_{lon,lat})`, re-rolled → always flyable (§4.2) | `refcv3_adapter.py::sample_offsets` is per-coordinate → **add the 2-scalar control-scale mode** (one-line change, integration request #1 of the V2 analysis) |
| likelihood | isotropic Gaussian, σ 0.1, summed over 16 coords — not the sampler's density | same class (the surrogate `logp`, `D-DDV2-CODE-3`); stated limit; the chain density comes with E-DDA-3 later | exists |
| groups | G = 4 per anchor, `(r − mean_G)/(std_G + 1e-4)`, discount 0.8 over 10 steps | G = 4 per anchor, centre-only by default (Dr. GRPO), std divisor a flag; one step (no chain) until E-DDA-3 | `advantage.py::grpo_advantage` — exists |
| inter-anchor truncation | `clamp(min=0) · (r > r_GT − 1e-6)`; −1 on collision **or** off-drivable | `clamp(min=0) · (r ≥ r_GT)` with the GT scored under the same rule reward; −1 on collision (**no** drivable-area term: no map) | `truncated_inter_anchor_advantage` exists (veto → −1); **add the ≥ GT bar** (one line behind a config field) |
| IL term | L1 of all 80 chains to the GT, λ = 0.1 (1.0 on rows with no positive sample) | same, on all G·N samples | `config.py` refuses `gt_similarity` + `w_imitation` double counting — keep that refusal: the IL term is the trainer's, not a reward component |
| trust region | none (no KL, no ref policy) | the existing L2 anchor to the frozen fan (`anchor.py`, `w_anchor` 1.0, monotone in the pilot sweep) — **ours, not V2's**; stated | exists |
| grouping key | anchor | anchor (default) vs `(a_lon, a_lat)` cell vs v7.2 tactical class — E-DDA-6 decides; **the class is the key XOR a reward term, never both** | E-DDA-6 |
| recipe | 10 epochs, lr 2e-4, wd 1e-4, cosine, batch 512, cold start = the IL checkpoint | 2,000 steps on the 120-clip fit set, lr per the pilot ladder, cold start = the refcv4b (later refcv5) final checkpoint | `rl_refcv3_min.py` + `launch_refcv3_rl_min.sh`, STAGE 0 passes, cost MEASURED 0.862 s/step / 1.78 GB on the 4060 |

### 6.3 The reward — PDMS's structure, none of its map terms, and the echo traps named

`R = NC(replay) · (5·TTC/headway + 5·progress + 2·comfort) / 12`, the ≥ GT bar, collision → −1; scored at **train time** on the model's own emitted candidates against **replayed** `obstacle.offline` agents (NAVSIM's non-reactive assumption; privileged inputs are admissible at train time). Selector disjointness (`FORBIDDEN_REWARD_INPUTS`) stands: no term reads any model-produced ranking.

| candidate term | reads | verdict |
|---|---|---|
| ADE / FDE to the GT future | the ego's GT future, whole shape | ⛔ **ECHO — it is the IL loss.** RL on it re-learns imitation with variance |
| "stay within X m of the recorded path" | the GT future, banded | ⛔ ECHO with a dead zone — the recorded path is the answer |
| target-speed / speed-profile match to GT | the GT future (speed) | ⛔ ECHO — the longitudinal IL target |
| route / strategic-goal alignment | `nav_command` = ego-future oracle (4,719/4,719) | ⛔ ECHO of the strategic IL target |
| v7.2 tactical class agreement | a label derived from the ego future | ⚠️ partial echo — admissible only as the *grouping key* (E-DDA-6), never as a reward term while it is the key |
| progress normalised by the GT's progress (V2's EP) | one GT scalar | ⚠️ acceptable as a **normaliser**, a trap as a term ("match the human's distance" is the longitudinal echo in ratio form) |
| ≥ GT mask | the GT's **score**, not its shape | ✅ a bar, not a target |
| collision / clearance / TTC / headway vs replay | other agents' futures | ✅ the environment; caveat: agents were recorded reacting to the human (non-reactive replay, NAVSIM's own caveat) |
| comfort / feasibility | candidate geometry only | ✅ — but see the floor defect below |
| progress (along-track vs `v0`) | candidate geometry | ✅ hackable (`HACKABLE_WEIGHTS` regression arm); needs the collision term to fire |
| drivable area / lane keeping | map | ✗ unavailable on PhysicalAI (AlpaSim `map.xodr` only) |

⛔ **The reward-floor defect is fixed BEFORE any arm, and it is a design item, not a launch.** MEASURED (`D-RL-REWARD-FLOOR-1`, fit8, 318 lead windows): the DEFAULT composition ranks the hold-v0 straight path **above the human's own future on 78.3 %** of lead windows — feasibility (0.50) and comfort (0.20) are maximal for a straight constant-speed path and progress reads 1.0 for it. The ≥ GT bar does **not** repair this: hold-v0-like samples *beat* the human under such a reward and collect positive advantage. Fix: move feasibility and comfort from the *reward* into **vetoes** (thresholds calibrated on the demonstration, `PREREG_D_SAFE_CAL`), keep the environment terms (NC, TTC/headway, progress) as the reward, and gate the design on the existing `humanflag` instrument (`rl_refcv3_min.py --mode humanflag`): **G-REWARD — hold-v0 ≥ human on ≤ 30 % of lead windows on the 120-clip fit set**, pre-registered; a reward that fails it may not train anything.

### 6.4 The vehicle, and what it decides

`H-RL-MIN-1` (`…/2026-09-05-refc-rl-readiness/`: driver `stack/scripts/rl_refcv3_min.py`, chain `launch_refcv3_rl_min.sh`, STAGE 0 PASS, four arms `base · rl · reg_echo · ctrl0`, verdict order VOID → FAIL-COLLAPSE → FAIL-GUARD → FAIL-FAN → PASS → SPLIT → REJECT-SELECTOR → NULL) is re-scoped as **E-DDA-3b**: base = the **refcv4b final** checkpoint (not refcv3) once it lands; the one-variable delta vs the surrogate per-coordinate version = the two code-only V2 ingredients (≥ GT bar + 2-scalar control-scale exploration); `reg_echo` (the GT future inside the advantage) **must FAIL the G-FAN gate or the run is VOID**; `ctrl0` (lr 0) must read Δ = 0 exactly. Outcomes committed in `H-DDA-5`: the along-track oracle gap shrinks with CI excluding 0 **and** fan-collision does not rise ⇒ the V2 lever transfers on the predicted axis, schedule E-DDA-3's chain; along shrinks but lateral worsens ⇒ scale-only exploration too coarse, stop; neither ⇒ V2's map half was load-bearing, park RL until a lane signal exists. Then **E-DDA-6** (grouping key) on the same rig. Cost ≈ 2.5–3 h of the 4060 serial per panel (MEASURED/ESTIMATED, `D-RL-READY-1`), 0 pod GPU. ⛔ **Launch is the Master Mind's / PI's call** (`LAUNCH_APPROVED=1`); this plan launches nothing.

## §7 Implementation ladder

> ⚠️ **SUPERSEDED FOR ORDERING BY §7A** — the PI's 2026-09-05 release split (pure vision first, then LiDAR) postdates this section, so §7 below is single-track; **§7A is the ordering of record** and carries the canonical names, while every per-WP detail, SPEC block, param delta and cost figure in §7 stands unchanged and is what §7A points back to.

**The ordering principle.** The ladder is sorted by *what is free first*, not by what is most
interesting. Every rung is **one lever**, pre-registered per `TanitAD_ValidateAIDesign` with a
deliberate-regression arm and controls that must read known values, and validated on the **`tiny` rig
rung** — `V3_RIG_SIZES = {"tiny": (32, (2, 2, 4, 2))}`, **16,989,725 params** MEASURED by building
(`refc_v3.py:492-518`), deliberately outside `V3_SIZES` so `test_refc_v3_scale_matrix.py` keeps
guarding the registered ladder. ⛔ **Nothing measured at the rig rung is a model claim** — it
validates the *wiring and the gate*, never the architecture's quality, and no registry row may cite
it (`refc_v3.py:511-514`).

### 7.0 The two clocks — what starts NOW at zero GPU, and what waits for refcv4b

refcv4b is 🟢 **TRAINING** on `tanitad-refcv3` (= `tanitad-a40`), 40,284 steps at **3.844 s/step**
marginal median, ETA **~2026-09-06 08:00 UTC** (ESTIMATED, `MODEL_REGISTRY.md` §4.6 *Pace*). ⛔ Never
eval on that pod while it trains; ⛔ never modify its run dir except to add a sidecar. That single
fact partitions the ladder:

| clock | rungs | why |
|---|---|---|
| ⭐ **NOW, ZERO GPU** (code, data, instruments, prereg) | **WP-0** instrument gaps · **WP-1** V5-VOCAB build · **WP-2** E-DDA-2 wiring · **WP-3** E-DDA-1 module · **WP-4** E-DDA-3 sampler · **WP-5** E-DDA-5 loss forms · **WP-6** E-AGT-1 head + join · **WP-7** E-DDA-2b selector module · **WP-8** nav wiring E15–E21 / S7 / S8 · **WP-11** LiDAR pilot scripts + rasteriser + tests | all are source changes, CPU-only artifact builds, or pre-registrations. None reads a refcv4b weight. |
| **NOW, DEV-BOX GPU** (RTX 4060, 8,187 MiB; ⚠️ currently carrying the `H-EGO-LIT-4` panel — schedule serially) | the tiny-rig panels for WP-2…WP-8 as each module lands; **WP-10** `H-EGO-LIT-4` (running) | the rig rung fits the 4060 (~29 min/arm; ~17 min/arm on Thor). |
| ⏳ **WAITS FOR A refcv4b CHECKPOINT** | **WP-9a** E-DDA-4 final read · **WP-9b** the post-training hierarchy ablation panel (`PREREG_REFCV4B_HIERARCHY_EVAL.md`, incl. the free `H-SEL-1` read of E-DDA-2) · **WP-12** E-DDA-2b selector *training* (frozen generator) · **WP-13** E-DDA-3b / E-DDA-6 RL (cold start = the refcv4b final) | each needs trained weights. ⭐ **E-DDA-4 does NOT have to wait for the FINAL**: `ckpt_step9500.pt` (1,285,301,425 B) is already on the dev box at `C:\Users\Admin\navcomp\ckpt\` and gives the readout *today* as an early-training diagnostic, exactly as `D-REFCV4B-EGODROP2` did. |
| ⏳ **WAITS FOR THE PI** | **WP-11b** LiDAR corpus build + **E-LIDAR-1** (inference-input doctrine, `D-REFCV5-PLAN-7`) · **WP-14** the refcv5 A40 launch (`LAUNCH_APPROVED=1`) | §2 I3 and the never-launch rule. |

⇒ **Ten of fourteen work packages start today at zero pod GPU.** The LiDAR build is deliberately
*last* on the model critical path (§3.4) so it can never gate the launch.

### 7.1 The work packages

Param deltas are **ESTIMATED by arithmetic** from `DecoderConfig(d=384, n_heads=8, layers=4,
ff_mult=4)` and `feat_dim = base_width·8 = 704` at `size base` (`refc.py:273, :291, :380-387`); each
one is **counted by building** in its own WP and the launch preflight pins the delta the way
`REGISTERED_DELTA_KEYS_V4` pins v4's (`refc_v3.py:527-536`) — an asserted delta, not a described one.

| WP | delivers | primary files | arm | one variable | Δ params (EST) | box · cost | starts | blocked by |
|---|---|---|---|---|---|---|---|---|
| **WP-0** | the three instrument gaps that block reads, not runs | `taniteval/tools/refcv3_arm.py`, `taniteval/plan_fan.py`, `stack/tanitad/eval/echo_gate.py` | — | — | 0 | CPU · ~1 d | **NOW** | — |
| **WP-1** | vocabulary **v5**: realised-speed κ clamp + build-time Kamm filter + the 13 × 11 grid | `stack/scripts/build_refc_anchors.py`, `stack/tanitad/refs/anchor_meta.py`, `stack/tanitad/instruments/flyability.py` | **V5-VOCAB** | the anchor family | 0 (a data artifact) | CPU · ~0.5 d | **NOW** | — |
| **WP-2** | score the fan actually emitted (0 new params) | `refc.py` `SelectionConfig` (`:467-472`, `:1703-1710`), `refc_v3_train.py` | **E-DDA-2** | `sel.refined+score_emitted+sel_ce_reach` on/off | **0** | tiny rig · 4 arms ≈ 2 h | **NOW** | — |
| **WP-3** | waypoint-indexed grid-sample cross-attention on the **PV** map | new `stack/tanitad/refs/refc_gridsample.py`, `refc.py::CrossAttnLayer` (`:1114-1131`), `stack/tanitad/data/calib.py` (read-only), `bev_raster.readout_column_index` | **E-DDA-1** | where the attention reads | ≈ **0.27 M** (one `Linear(704, 384)` sample projection + 4 zero-init gate scalars; ~0 if `feat_proj` is shared) | tiny rig · 4 arms ≈ 2 h | **NOW** | — |
| **WP-4** | the **control-space** DDIM sampler (anchored Gaussian, x0, AdaLN, Fourier query init) | `refc.py` `DecoderConfig` / `AnchoredDiffusionDecoder` (`:1134-1260`, `:1541-1770`), `stack/tanitad/models/kinematic.py` (+`roll_controls`), `refc_v3_train.py::compute_losses_v3` (`:537-575`) | **E-DDA-3** | `sampler: "none" → "ddim"` | ≈ **1.67 M** (`time_mlp` 0.30 M + 4 × `time_mod` FiLM 1.18 M + `wp_embed` 0.20 M; `control_head` replaces `offset_head` at parity) | tiny rig · 5 arms ≈ 2.5 h | **NOW** (code) | E-DDA-4 for the *decision to train it* (§4.2 #2) |
| **WP-5** | loss form: focal BCE score head; mean-L2 anchor assignment | `refc_v3_train.py`, `refc.py` score head | **E-DDA-5** | (a) CE → focal BCE; (c) assignment metric — **two separate arms, never bundled** | 0 | tiny rig · 4 arms ≈ 2 h | **NOW** | — |
| **WP-6** | agent tokens: DETR head on `obstacle.offline` + `cross_agent` in every decoder layer | new `stack/tanitad/refs/refc_agents.py`, `refc.py` decoder, `stack/scripts/build_obstacle_join.py` (exists, pod-side), `stack/tanitad/data/bev_raster.py` | **E-AGT-1** | the second cross-attention | ≈ **6 M** (30 queries 11.5 k + 2 detection layers ≈ 3.5 M + 4 × `cross_agent` ≈ 2.4 M) | tiny rig · 4 arms ≈ 2.5 h | **NOW** (code); the **join must be run for B1** | the B1 obstacle join (pod-side, CPU) |
| **WP-7** | the four-family coarse-to-fine sub-metric selector | new `stack/tanitad/refs/refc_selector.py`, new `stack/scripts/refc_selector_train.py`, `stack/tanitad/rl/rewards.py` (reused), `taniteval/tools/refcv3_arm.py` (`--selector`) | **E-DDA-2b** | the selector head, generator frozen | ≈ **3.5 M**, **outside the generator** (stage II) | 4060 · ≈ 3–5 h | module **NOW**; training after refcv4b | a refcv4b checkpoint's fan |
| **WP-8** | nav / strategic wiring: E15, E16, E17, E18, E19, E20, E21 + selector gates S7 / S8 | `refc_v3.py` (`hook()`, `str_goal_head` `:655`, `:883`), `refc.py` selector (`:1639-1690`), `refc_v3_train.py` | **E15–E21, S7/S8** | **one edge per arm** (7 edges, all zero-init and individually gated) | **+7,682** total (INHERITED, `DESIGN_REFCV4_NAV_WIRING.md` §2.2) | tiny rig · 1 panel per edge ≈ 6 h total | **NOW** | — |
| **WP-9a** | **E-DDA-4** — is the bank a prior? `‖offset‖` on the classifier pass, oracle-in-bank vs oracle-in-fan at equal N | `taniteval/tools/refcv3_arm.py`, `taniteval/plan_fan.py` | **E-DDA-4** | — (a readout, not an arm) | 0 | 4060 · ≈ 0.5 h | **NOW on `ckpt_step9500.pt`** (diagnostic); repeat on the final | — / the final ckpt for the decision |
| **WP-9b** | the refcv4b post-training hierarchy ablation panel (12 arms) | `PREREG_REFCV4B_HIERARCHY_EVAL.md` §3, `taniteval/tools/refcv3_arm.py` + WP-0 flags | H-NAVC-1..3, H-SEAM-1, H-H19-1, H-CONS-1, **H-SEL-1** | one ablation per arm | 0 | 4060 · ≈ 6–10 h | after refcv4b | ⛔ **WP-0** — the switches are not yet CLI flags |
| **WP-10** | the withheld-bank panel (5 arms) | `refc.py::roll_bank` (`:1318-1368`) `ref_speed` switch; `--withheld-bank {fixed,pred,random,none}` | **H-EGO-LIT-4** | the bank's reference speed | 0 | 4060 · **RUNNING** | — | — |
| **WP-11** | **WP-DE-BEV-1** (DataFlyWheel): LiDAR pilot → rasteriser → sidecar builder → loader → read-set declaration → corpus build | new `stack/tanitad/data/lidar_bev.py`, new `stack/scripts/v2_bev_sidecar.py`, `stack/tanitad/data/v2_dataset.py`, `stack/tests/test_physicalai_feature_readset.py` | — (a data WP) | — | 0 (data) | dev box pilot (10 clips) · then a pod, **≈ 3–8 h** streamed (ESTIMATED, §3.3) | pilot **NOW** | corpus build: HF quota + a non-training pod |
| **WP-12** | camera BEV lift taught by LiDAR; LiDAR BEV as input | `refc.py` trunk + KV set, `refc_gridsample.py` (reused) | **E-BEV-1**, **E-LIDAR-1** | the BEV path | E-BEV-1 ≈ 2–4 M lift; E-LIDAR-1 + ≈ 2–4 M encoder | tiny rig · 4 arms ≈ 3 h each | after WP-11 | ⛔ **E-LIDAR-1 needs `D-REFCV5-PLAN-7`** |
| **WP-13** | head-only scale-policy RL: the two V2 code ingredients, then the grouping key | `stack/tanitad/rl/refcv3_adapter.py::sample_offsets`, `advantage.py`, `rewards.py`, `stack/scripts/rl_refcv3_min.py`, `launch_refcv3_rl_min.sh` | **E-DDA-3b** → **E-DDA-6** | 3b: ≥ GT bar + 2-scalar control-scale exploration (a *pair*, see below) · 6: the grouping key | 0 (head-only; trainable **8.60 %**) | 4060 · ≈ 2.5–3 h per panel | after **G-REWARD** | ⛔ **G-REWARD** (§6.3) + a refcv4b final |
| **WP-14** | refcv5 assembly, preflight, launch | `refc_v3_train.py`, `refc_v3.py` registered delta, `sup_refcv5*.sh` | — | — | sum of the passed levers | **A40, 40,284 steps ≈ 43 h at 3.844 s/step** | after refcv4b's verdict | ⛔ PI / Master Mind `LAUNCH_APPROVED=1` |

⚠️ **WP-13's "one variable" is honestly a pair.** `H-RL-MIN-1`'s registered delta is *both* V2
code-only ingredients (≥ GT bar **and** 2-scalar control-scale exploration) against the surrogate
per-coordinate version. They are bundled on purpose — separately neither is V2's mechanism — so the
outcome is attributable to *"V2's exploration+truncation pair"* and **not** to either half. Stated
here rather than discovered later; if the pair separates, a follow-up single-lever panel splits it.

### 7.2 The pre-registrations — one SPEC per rung

Each rung gets `TanitAD Research Lab/Architecture & Inference/Research/<date>-<slug>/SPEC.md` in the
skill's schema **before** any compute. The blocks below are the committed content; only `hypothesis`
ids already in `GOALS_AND_CLAIMS.md` are used.

```yaml
# WP-2  E-DDA-2 — the ranked object becomes the emitted object
hypothesis:   H-DDA-2
one_variable: sel.refined && sel.score_emitted && sel_ce_reach     # 0 new params
held_constant: [anchors, corpus, steps, batch, seed, lr, window, all other flags]
success: "pick changes on >= 10 % of windows AND the selection gap (oracle_sel - os)
          shrinks with a paired episode-cluster CI excluding 0, with no separated
          loss on any of the four families"
failure:  "pick unchanged on > 95 % of windows (the audit's 201/201 defect survives
           the fix -> the flag is not the mechanism), OR the gap shrinks while
           LONGITUDINAL or LATERAL degrades separated"
controls: [all_off_paired_control,                 # today's decoder, same seed
           unsupervised_refined_arm,               # E-SEL-0's known-worse reading
           deliberate_regression: score_emitted_with_sel_ce_reach_OFF]
splits:   {fit: B1 train, val: carved from FIT, test: the 141 EVAL clips, scored once}
```

```yaml
# WP-3  E-DDA-1 — waypoint-indexed grid-sample attention on the PV map
hypothesis:   H-DDA-1
one_variable: attention_sampling_locations         # content-MHA -> candidate waypoints
held_constant: [d, layers, heads, ff_mult, param_count_matched, anchors, corpus, seed]
success: "TACTICAL and LONGITUDINAL improve with paired CIs excluding 0 AND the
          candidate-conditional readout separates: the attention map moves when the
          candidate moves (a permutation test over candidates within a window)"
failure:  "no separated family gain, OR the attention map is candidate-invariant
           (the module is a re-parameterised content MHA)"
controls: [random_sampling_locations_at_equal_params,     # the floor
           frozen_zero_init_gate,                          # must read delta == 0 exactly
           projection_control: pinhole_formula_arm_must_MISPROJECT]   # the FOV trap, on purpose
splits:   {fit: B1 train, val: carved from FIT, test: the 141 EVAL clips}
notes: "cylindrical, f_ref 305.577, col = f_ref*phi + W/2. The pinhole arm exists to
        prove the projection instrument can see a wrong projection (CLAUDE.md FOV trap)."
```

```yaml
# WP-4  E-DDA-3 — the control-space DDIM sampler
hypothesis:   H-DDA-3
one_variable: DecoderConfig.sampler                # "none" -> "ddim"
held_constant: [anchors, corpus, steps, batch, seed, lr, layers, d, heads]
success: "diversity D (Eq. 3 of 2411.15139) rises AND oracle-in-fan at EQUAL N improves
          with a paired CI excluding 0 AND the sampled fan's Kamm(mu=0.7) violation rate
          does not rise above the deterministic fan's"
failure:  "oracle-in-fan at equal N does not improve (the noise is not exploring the
           right space), OR selected-trajectory jerk rises separated
           (D-REFCV4B-EGODROP2 measured 2.42 m/s^3 vs the human's 0.86 -- we may not
           make it worse)"
controls: [sigma_zero_identity,          # MUST reproduce the deterministic decoder BIT-FOR-BIT
           constant_u_roll_equals_roll_bank,    # a known value, to the bit
           deliberate_regression: DD_literal_metre_space_norm_odo_sampler]
splits:   {fit: B1 train, val: carved from FIT, test: the 141 EVAL clips}
notes: "The deliberate-regression arm MUST fail the flyability gate. If it does not,
        the flyability instrument cannot see what it is cited for and the panel is VOID."
```

```yaml
# WP-6  E-AGT-1 — agent tokens
hypothesis:   H-DDA-1                    # same audit family (#20); a sibling arm
one_variable: cross_agent_layer          # present / absent (zero-init gate)
held_constant: [d, layers, heads, anchors, corpus, seed, lr]
success: "detection AP > prior AND > pixel floor (paired), AND the LONGITUDINAL family
          (headway / time-gap / TTC) improves with a CI excluding 0"
failure:  "AP at or below the prior floor (the head learned nothing), OR AP good and no
           family moves (the tokens are produced but not read -- report the gate value)"
controls: [zero_init_gate_reads_delta_zero,
           shuffled_agent_tokens,          # across windows: the gain must vanish
           constant_only_detection_control, raw_pixel_detection_floor,
           printed_n_and_d]
splits:   {fit: B1 train, val: carved from FIT, test: the 141 EVAL clips}
notes: "obstacle.offline is a TRAIN-TIME label (I3). At inference the tokens come from
        the trunk. An arm that reads the join at inference is refused, not fixed."
```

```yaml
# WP-7  E-DDA-2b — the four-family sub-metric selector
hypothesis:   H-DDA-7
one_variable: selector                   # argmax(conf) -> SubMetricSelector
held_constant: [generator weights FROZEN, fan, corpus, seed, windows]
success: "the selection gap closes with a CI excluding 0 AND fan-collision-vs-replay
          does NOT rise"
failure:  "the gap closes and collision rises  -> V2 Tab. 9's hackish direction, REFUSE;
           nothing changes                     -> selection is not binding at this fan
                                                  quality, bank the negative"
controls: [progress_only_head,     # the deliberate regression: MUST buy progress WITH
                                   # collisions, else the collision readout is blind -> VOID
           nav_shuffle_and_nav_zero_on_the_compliance_head,
           per_head_calibration_ECE_reported]
splits:   {fit: the 120-clip fit set, val: carved from FIT, test: the 141 EVAL clips}
notes: "ADE-only reading refused in advance. Four families always."
```

```yaml
# WP-13  E-DDA-3b — the head-only scale policy (V2's two code ingredients)
hypothesis:   H-DDA-5                    # vehicle: H-RL-MIN-1
one_variable: "the V2 pair: (>= GT bar) + (2-scalar control-scale exploration)"
held_constant: [base ckpt = refcv4b FINAL, reward, fit clips, lr schedule, seed, steps]
success: "the ALONG-TRACK oracle gap shrinks with a CI excluding 0 AND fan-collision
          does not rise  -> the V2 lever transfers on the predicted axis (92.2 % of our
          deficit, D-REFCV3-AXIS1); schedule E-DDA-3's chain"
failure:  "along shrinks but lateral worsens -> scale-only exploration too coarse, stop;
           neither moves                     -> V2's map half was load-bearing, park RL
                                                until a lane signal exists"
controls: [ctrl0_lr_zero_reads_delta_exactly_zero,
           reg_echo_GT_future_in_the_advantage,   # MUST FAIL G-FAN or the run is VOID
           G-REWARD_precondition]
splits:   {fit: the 120-clip fit set, val: carved from FIT, test: the 141 EVAL clips}
gate_before_any_arm: "G-REWARD -- hold-v0 >= human on <= 30 % of lead windows on the
                      120-clip fit set (today: 78.3 %, D-RL-REWARD-FLOOR-1). A reward
                      that fails G-REWARD may not train anything."
```

The remaining rungs (WP-1 V5-VOCAB, WP-5 E-DDA-5 a/c, WP-8's seven edges, WP-12 E-BEV-1 / E-LIDAR-1)
take the same shape; WP-1's is **MODEL-FREE** and its gate is stamped so per
`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md` — ⛔ a raw-anchor ceiling is never compared to `ha`, `os` or
`oracle_sel` (the scope error that nearly killed refcv4, `D-REFCV4-GATE1`).

### 7.3 WP-0 in full — the three gaps that block *reads*, not runs

These are cheap, they are on the critical path of two other packages, and each is a MEASURED absence.

1. ⛔ **The eval-time ablation switches are not CLI flags.** `PREREG_REFCV4B_HIERARCHY_EVAL.md`
   already carries this as **ESCALATED**: the twelve arms of §3 (nav-ZERO / SHUFFLE / FLIP, g_str-ZERO
   / SHUFFLE, E7-OFF, E9-OFF, H19-OFF, EGO-ZERO, SEL-REFINED, the frame-blind deliberate regression)
   need flags on `taniteval/tools/refcv3_arm.py`. **0 GPU, and it must precede the post-training run**
   — WP-9b cannot start without it, and WP-9b is where E-DDA-2 gets its first free read.
2. ⛔ **`ha0_ext` is not an arm of the REF-C harness.** MEASURED today: `refcv3_arm.py:961` is
   `arms = ["os", "ha", "ha0"]` and `ARM_TIERS` (`:237-238`) has no `ha0_ext`; the banked
   `refcv3-40284-openloop.ARM.json` carries `arm_keys = [os, ha, ha0, os_navshuf, os_navzero,
   oracle_sel]` — **six arms, no `ha0_ext`**. The implementation exists
   (`stack/tanitad/eval/echo_gate.py::ha0_ext`) and the *refav1* harness runs it
   (`taniteval/tools/refav1_arm.py:639`, `t1_eval.py:162`). ⇒ **§8's bar — "beat both `ha` and
   `ha0_ext`" — is not readable on the 4,823-window surface today.** Porting the arm is ~30 lines and
   one shared call; ⛔ it must be the **same** `echo_gate.ha0_ext` (the docstring at `:180-183` says
   why: a control re-implemented beside the thing it controls drifts, and then the gate measures the
   drift). Until it lands, `ha0_ext` appears in §8 only on the **compliance** readout, where
   `nav_compliance.py:1125` already computes it.
3. **The fan readouts do not exist.** `fan_floor@k` (k = 1, 5, 10 — the k-th-best candidate's
   four-family score), the fan-collision-vs-replay rate, DD's diversity score D, and the sampled-fan
   Kamm violation rate. Without them **WP-4, WP-7 and WP-13 have no primary readout at all**: §6.1's
   whole point is that V2's gain was a *floor* effect the top-1 number cannot see. Home:
   `taniteval/plan_fan.py`, tests first, with a synthetic fan whose floor is known by construction.

### 7.4 Cost, and the critical path

**GPU budget (ESTIMATED; the rig figures are anchored on the skill's MEASURED ~29 min/arm on the
4060, ~17 min/arm on Thor):**

| box | packages | hours | note |
|---|---|---|---|
| dev-box RTX 4060 (8,187 MiB) | WP-2, 3, 4, 5, 6, 8 rig panels (≈ 21 arms) | **≈ 12–16 h** serial | ⚠️ shares the box with WP-10 (running) and WP-9a/9b; schedule serially and set `OMP_NUM_THREADS=6` before any multi-arm panel (CLAUDE.md: 7 concurrent arms sat at 0–6 % `sm` for 50 min) |
| dev-box RTX 4060 | WP-9a (0.5 h) + WP-9b (6–10 h) + WP-7 training (3–5 h) + WP-13 (2.5–3 h × 2 panels) | **≈ 15–22 h** | all after a refcv4b checkpoint |
| Jetson Thor | overflow rig panels; the §9 latency budget (WP-0 of the VLA plan) | **≈ 6 h** | ⚠️ `--batch 8` (20 SMs saturate there); only `torch.cuda.max_memory_allocated()` is admissible |
| A40 pod (`tanitad-refcv3`) | **WP-14 refcv5**, 40,284 steps @ 3.844 s/step | **≈ 43 h** | ⛔ free only after refcv4b's ETA ~2026-09-06 08:00 UTC |
| a non-training pod | WP-11 corpus build, streamed | **≈ 3–8 h** + ≈ 3.6 h transfer | ⛔ never the training pod; ⛔ `df` cannot see the ~466 GB MooseFS quota — use a `dd` write test |

**The critical path to a refcv5 launch** (⇒ = "gates"):

```
refcv4b finishes (~2026-09-06 08:00 UTC)
  => WP-9a E-DDA-4 final read      -> decides WHETHER the sampler is the fork at all (§4.2 #2)
  => WP-9b hierarchy panel         -> decides which hierarchy edges survive into refcv5
       ^-- requires WP-0.1 (flags), which requires NOTHING and must be done first
  => WP-7 selector training + WP-13 RL   (both cold-start from the final checkpoint)

in PARALLEL, off the critical path, starting today:
  WP-0 -> WP-1 (V5-VOCAB) -> [MODEL-FREE gate]  ------------------\
  WP-2, WP-3, WP-4, WP-5, WP-6, WP-8 rig panels ------------------ >-- WP-14 bundles ONLY
  WP-11 pilot -> rasteriser -> (PI ruling) -> corpus -> WP-12 ----/     the levers that PASSED
```

⭐ **The bundling rule, stated before the fact.** refcv4b already forfeited attribution by carrying
five levers at once (`MODEL_REGISTRY.md` §4.6, caveat 1). refcv5 must not repeat it *silently*: the
launch bundles only levers whose rig arm passed, the registered delta is **pinned as a frozenset**
the preflight refuses to deviate from, and the bundle is declared in the registry row as a bundle —
so a refcv5-vs-refcv4b delta is an **arm** delta, never a lever attribution. ⚠️ Any lever whose rig
arm has not run by the launch date is **left out**, not launched hopefully.

**What is NOT on the path, and why that is deliberate:** the LiDAR/BEV work (WP-11/WP-12). §3.4's
camera-only fallback — E-DDA-1 (no BEV at all), E-AGT-1 (the attention DD has and we lack), and
E-BEV-1 taught by the **agent raster alone**, which `bev_raster.py` produces today from data we
already hold — is a complete path to closing the cross-attention gap without a single byte of LiDAR.
The LiDAR build raises the ceiling; it never blocks the launch.


## §7A The reconciled release ladder (v5a / v5b)

> ⚠️ **STATUS: PART 1 of 6 LANDED — 2026-09-05, Arch+Inference FlyWheel, integration task delegated by
> the Master Mind. Parts 2–6 are being filled in and committed one at a time; a heading with
> `*(pending)*` under it is NOT a finding, it is work in flight.**

This section reconciles two completed documents that describe overlapping work under two
incompatible naming schemes:

* **(A) `Project Steering/REFCV5_DESIGN_PLAN.md` §7** — this file, immediately above: the
  **14-work-package ladder WP-0…WP-14** with file-level plans, param deltas, per-rung SPEC blocks,
  GPU budgets and a critical path. It was written **before** the PI's release split and is therefore
  **single-track**.
* **(B) `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vision-only-maximum/RESULT.md`**
  — the **ten-rung `WP-V5A-*` camera-only ladder** (≈ 13.4 rig-GPU-h, four rungs starting at zero pod
  GPU), plus the v5a/v5b boundary stated as a decision rule, and 11 `D-V5A-*` + 8 `H-V5A-*` register
  rows.

⛔ **Nothing here is new research.** The adjudication rule the Master Mind set, applied throughout:
**(B)'s evidence wins on the camera-only questions** — it measured the camera inventory, the bytes and
the azimuth coverage; **(A)'s structure wins on everything else** — its SPEC blocks, gates, claims and
refutations are the programme's format. Where the two conflict on a **fact** rather than a name, both
readings are recorded with their evidence and the conflict is **escalated** (§7A.7), not adjudicated
here.

**Why this section exists (the PI's ruling, `Project Steering/Decisions/2026-09-05-mm-decisions.md`
§M7, verbatim):** *"i prefer to do the environment extensions in two versions/steps, let start by
pure vision and then add lidar. So check, what we can do maximally with vision, bev, und was else?"*

**What §7 keeps.** §7 is **not rewritten**. Its per-WP detail — primary files, one-variable
statements, param deltas, SPEC blocks, box-and-cost figures, the bundling rule — stands unchanged and
is what §7A points back to. §7 is superseded **for ordering and for names only**, and it is preserved
deliberately: it is the record of what was planned before the split.

### 7A.1 Crosswalk — every name in both documents, and the single canonical name

**How to use it.** Take any name you have seen today, find its row, read the **CANONICAL** column.
A name marked ⛔ **RETIRED** means two documents used it for two different arms; it is never quotable
bare again.

#### 7A.1.a ⛔ The one name that means two different things — `E-AGT-1`

This is the collision the Master Mind flagged, and it is real:

| the name `E-AGT-1` in… | means | evidence |
|---|---|---|
| **(A) §7.1 WP-6 / §3.4 / §1.3 #20** | the **whole agent-token mechanism in one arm**: a DETR-style head on `obstacle.offline` **plus** `cross_agent` in every decoder layer, ≈ 6 M params, one variable = "the second cross-attention" | plan §7.1 row WP-6; SPEC block `# WP-6 E-AGT-1 — agent tokens`, `one_variable: cross_agent_layer` |
| **(B) §3.0 WP-V5A-2 / §3.3** | **only the accuracy-budget sweep**: range-noise σ ∈ {0, 0.5, 1, 2, 4} m + a separate miss-rate sweep applied to **oracle** boxes, 6 arms ≈ 2.9 h, one variable = σ | study §3.0 table; §3.3 `one_variable: the range-noise sigma applied to the oracle boxes` |

⇒ **`E-AGT-1` is RETIRED.** The mechanism is (B)'s three-rung decomposition — which is the
camera-only refinement, and (B) wins there — under three names that cannot be confused:

| CANONICAL | what it is | was called |
|---|---|---|
| **`E-AGT-ORACLE`** | the **ceiling**: `cross_agent` wired and fed **ground-truth** boxes at inference. Deliberately inadmissible as a capability claim, run **once**. If it does not separate on LONGITUDINAL **and** TACTICAL, the whole mechanism is refused for **zero further GPU-days** | (B) `E-AGT-0` / `WP-V5A-1` · no counterpart in (A) |
| **`E-AGT-BUDGET`** | the **derived accuracy bar**: degrade the oracle boxes with calibrated range noise σ and a miss rate, and read where the separation dies. **The σ that comes out is the detector specification** — and it is also the number that prices v5b (§7A.4) | (B) `E-AGT-1` / `WP-V5A-2` · no counterpart in (A) |
| **`E-AGT-HEAD`** | the **deliverable arm**: tokens from a **learned** monocular 3D head (DETR-style, K = 30, Hungarian on the 10 dynamic classes) + `cross_agent` in every decoder layer, zero-init gated | (A) `E-AGT-1` / `WP-6` · (B) `E-AGT-2` / `WP-V5A-3` |

⚠️ **The hypothesis ids also disagree and are canonicalised here.** (A)'s WP-6 SPEC block carries
`hypothesis: H-DDA-1` with the note *"same audit family (#20); a sibling arm"* — but `H-DDA-1` is
already the hypothesis of the **waypoint-sampling** arm `E-DDA-1`, so under (A) one id covers two
different mechanisms. (B) registered `H-V5A-AGT-1/2/3` for exactly the three rungs above. ⇒
**canonical: `H-V5A-AGT-1` → `E-AGT-ORACLE`, `H-V5A-AGT-2` → `E-AGT-BUDGET`, `H-V5A-AGT-3` →
`E-AGT-HEAD`; `H-DDA-1` is reserved for `E-DDA-1` alone.** Both id sets are live in
`GOALS_AND_CLAIMS.md`; this row is the mapping, not a deletion.

#### 7A.1.b The second overlap the collision hid — the camera→BEV lift

Not a name collision, a **scope** collision, and it is the one that actually straddles the release
boundary:

| | (A) `E-BEV-1` (WP-12) | (B) `E-BEVA-1` (WP-V5A-8) |
|---|---|---|
| what enters | a lift from the PV map to a metric BEV; grid-sample attention on the lifted BEV | an **LSS-style polar-native** lift from the stride-8 stage into an ego-frame grid, indexed directly by the fan |
| **supervision** | **the LiDAR histogram + agent raster** as BEV-occupancy targets | **the agent raster ALONE** (zero new data) |
| scheduled | §7.1: **after WP-11** (the LiDAR corpus build) | §3.0: **after WP-V5A-4 separates**, zero new bytes |
| mandatory | — | ⛔ an explicit **`fov_mask` channel with every loss masked by it** (an out-of-frustum cell is UNOBSERVED, not free) |

⭐ (A)'s own §3.4 already names the camera-only variant — *"E-BEV-1 with the **agent raster alone** as
the BEV target (the P8 rasteriser exists today, zero new data)"* — but (A)'s §7.1 nevertheless
schedules `E-BEV-1` **after WP-11**, i.e. behind the LiDAR build. **The split resolves that tension:**

| CANONICAL | release | what it is |
|---|---|---|
| **`E-BEVA-1`** | **v5a** | the polar camera→BEV lift taught by the **agent raster alone**, `fov_mask` mandatory — **no LiDAR, no new bytes, not blocked by WP-11** |
| **`E-BEV-1`** | **v5b** | the same lift taught by the **LiDAR histogram** — the strictly-later, strictly-better teacher |
| **`E-LIDAR-1`** | **v5b** | LiDAR BEV as an **inference input** — additionally gated on `D-REFCV5-PLAN-7` (PI doctrine) |

#### 7A.1.c The full crosswalk

Every WP and experiment name that appears in either document. **"—" = no counterpart.**

| name as written | in | counterpart in the other document | **CANONICAL** | release |
|---|---|---|---|---|
| `WP-0` (three instrument gaps) | A §7.1, §7.3 | — | **`WP-0.1` / `WP-0.2` / `WP-0.3`** (split; §7A.5) | v5a |
| `WP-1` · `V5-VOCAB` | A §7.1 | — | **`WP-1` · `V5-VOCAB`** | v5a |
| `WP-2` · `E-DDA-2` | A §7.1, §7.2 | — | **`WP-2` · `E-DDA-2`** | v5a |
| `WP-3` · `E-DDA-1` | A §7.1, §7.2 | **`WP-V5A-4` · `E-DDA-1`** — same name, same arm | **`WP-3` · `E-DDA-1`** | v5a |
| `WP-4` · `E-DDA-3` (sampler) | A §7.1, §7.2 | — | **`WP-4` · `E-DDA-3`** | v5a |
| `WP-5` · `E-DDA-5` (loss form, arms a/c) | A §7.1 | — | **`WP-5` · `E-DDA-5`** | v5a |
| `WP-6` · `E-AGT-1` | A §7.1, §7.2 | **`WP-V5A-3` · `E-AGT-2`** | ⛔ RETIRED ⇒ **`WP-6` · `E-AGT-HEAD`** | v5a |
| `WP-7` · `E-DDA-2b` (selector) | A §7.1, §7.2 | — | **`WP-7` · `E-DDA-2b`** | v5a |
| `WP-8` · `E15`–`E21`, `S7`/`S8` | A §7.1 | — | **`WP-8` · nav/strategic wiring** | v5a |
| `WP-9a` · `E-DDA-4` | A §7.1 | — | **`WP-9a` · `E-DDA-4`** | v5a |
| `WP-9b` · the hierarchy panel | A §7.1 | — | **`WP-9b` · the refcv4b hierarchy panel** | v5a |
| `WP-10` · `H-EGO-LIT-4` | A §7.1 | — | **`WP-10` · `H-EGO-LIT-4`** (RUNNING) | v5a |
| `WP-11` · `WP-DE-BEV-1` (LiDAR data) | A §3.3, §7.1 | — (⚠️ **not** `WP-V5A-9`, which is a *camera* pull) | **`WP-11a…d`** (pilot / rasteriser / sidecar+loader / corpus) | **v5b** |
| `WP-12` · `E-BEV-1`, `E-LIDAR-1` | A §7.1 | **`WP-V5A-8` · `E-BEVA-1`** covers only the camera-taught half | **`WP-12a` · `E-BEV-1`** · **`WP-12b` · `E-LIDAR-1`** | **v5b** |
| `WP-13` · `E-DDA-3b` → `E-DDA-6` (RL) | A §7.1, §7.2 | — | **`WP-13` · `E-DDA-3b`, `E-DDA-6`** | v5a |
| `WP-14` · the refcv5 launch | A §7.1 | **`WP-V5A-F`** — the refcv5a full-scale run | **`WP-14a`** (v5a launch) · **`WP-14b`** (v5b launch) | both |
| `WP-V5A-0` · `E-THOR-MV` | B §3.0, §3.1 | — | **`WP-V5A-0` · `E-THOR-MV`** | v5a |
| `WP-V5A-1` · `E-AGT-0` | B §3.0, §3.2 | — | **`WP-V5A-1` · `E-AGT-ORACLE`** | v5a |
| `WP-V5A-2` · `E-AGT-1` | B §3.0, §3.3 | — | ⛔ RETIRED ⇒ **`WP-V5A-2` · `E-AGT-BUDGET`** | v5a |
| `WP-V5A-3` · `E-AGT-2` | B §3.0, §3.4 | **`WP-6` · `E-AGT-1`** | **`WP-6` · `E-AGT-HEAD`** | v5a |
| `WP-V5A-4` · `E-DDA-1` | B §3.0, §3.5 | **`WP-3` · `E-DDA-1`** | **`WP-3` · `E-DDA-1`** | v5a |
| `WP-V5A-5` · `E-DDA-1b` | B §3.0, §3.5 | — (the idea is inside A §3.4's `E-BEV-1` row, **not** a separate arm there) | **`WP-V5A-5` · `E-DDA-1b`** | v5a |
| `WP-V5A-6` · `E-DEPTH-0` | B §3.0, §3.6 | — | **`WP-V5A-6` · `E-DEPTH-0`** | v5a |
| `WP-V5A-7` · `E-CAM-1` | B §3.0, §3.7 | — | **`WP-V5A-7` · `E-CAM-1`** | v5a |
| `WP-V5A-8` · `E-BEVA-1` | B §3.0, §3.8 | the camera-taught half of **`E-BEV-1`** | **`WP-V5A-8` · `E-BEVA-1`** | v5a |
| `WP-V5A-9` · surround corpus pull | B §3.0, §3.7 | — (⚠️ **not** `WP-11`: different sensor, **shared fetcher**) | **`WP-V5A-9` · the surround corpus pull** | v5a |
| `WP-V5A-F` · the refcv5a run | B §3.0 | **`WP-14`** | **`WP-14a`** | v5a |

⚠️ **Two `WP-11`-shaped items are NOT the same work and must never be merged:** `WP-11` pulls
**LiDAR** (≈ 1.53 TB transit for B1) and `WP-V5A-9` pulls **cameras** (149 GB cross pair / 429 GB all
six). What they share is the **HTTP-range zip-member fetcher**, which is a HYPOTHESIS in (A) §3.1 and
on v5b's critical path — and (B) §3.7 validates it on ≈ 3 GB of camera first. ⇒ **one shared
component, two corpora** (escalation §7A.7 #4).

⚠️ **An internal inconsistency inside (A), noted not fixed:** §7.0's clock table places
*"**WP-12** E-DDA-2b selector *training*"* in the refcv4b-gated column, but §7.1 assigns `E-DDA-2b` to
**WP-7** and `E-BEV-1`/`E-LIDAR-1` to **WP-12**. §7.1's own WP-7 row (*"module **NOW**; training after
refcv4b"*) says what was meant. §7A reads it as **WP-7**. §7 is left as written.

### 7A.2 v5a — camera-only, ordered

*(pending)*

### 7A.3 v5b — LiDAR, ordered, off the model critical path

*(pending)*

### 7A.4 The release boundary as a decision rule, not a date

*(pending)*

### 7A.5 The two blockers, carried into the ladder as rungs

*(pending)*

### 7A.6 The arithmetic that constrains the whole ladder

*(pending)*

### 7A.7 Conflicts recorded, not adjudicated — escalated to the Master Mind

*(pending)*

## §8 What refcv5 claims, and what would refute it

### 8.0 The shape of every claim in this section

A refcv5 claim is admissible only in this form: **a named quantity · at a named tier · on the named
surface · against a named control · with a paired episode-cluster CI · in its metric family**. The
doctrine is not restated here (`EVAL_DOCTRINE.md`, CLAUDE.md, §2 I5); what follows is the *content*
of the claims, with their refutations committed **now**, before any refcv5 weight exists.

**The surface.** n = **4,823 windows / 141 episodes** of the v7.2 EVAL split (labels md5
`aa12c948f062181c3297265b51526ec5`), grid `2s`, dt 0.5 s, 4 horizon steps; estimator = episode-cluster
bootstrap **B = 2,000, seed 0, paired for every delta** (`MODEL_REGISTRY.md` §4.5). ⛔ The corpus is
**NON-PARITY** (`v2_parity.parity false`), so refcv5 is comparable to **refcv4b and refcv3** (same
`/root/data`) and to the shared floors — **never** to `refc-base` / `refc-xl`.

**The bar, quoted from the artifact, not from prose** (MEASURED, `taniteval/results/refcv3-40284-openloop.ARM.json`,
md5 of the JSON `5cfe3258c18871218bd85d691904eb20`; `D-REFCV3-40284`):

| arm | ADE 0–2 s (m) | FDE (m) | what it is |
|---|---|---|---|
| ⭐ **`ha`** | **0.2996** [0.2755, 0.3278] | 0.6588 [0.6044, 0.7192] | hold-action: the `(a, steer)` closing at t0, held — **the bar** |
| `oracle_sel` (T0) | 0.3668 [0.3437, 0.3914] | 0.7770 | GT-nearest anchor's refinement — a ceiling, not an arm |
| `os` (refcv3 @ 40,284) | 0.4419 [0.4098, 0.4743] | 0.9288 | the model, own choice, oracle nav |
| `os_navzero` | 0.4659 [0.4310, 0.5010] | 0.9655 | **the deployment condition** |
| `ha0` | 0.6723 [0.6007, 0.7469] | 1.4029 | constant velocity — beating it is **necessary, not sufficient** |

⛔ **`ha0_ext` is not yet readable on this surface** — WP-0.2 (§7.3): `refcv3_arm.py:961` is
`arms = ["os", "ha", "ha0"]`, and the banked ARM JSON's `arm_keys` has six entries, none of them
`ha0_ext`. **Until WP-0.2 lands, every `ha0_ext` clause below is a claim we cannot yet read**, and
saying so is part of the pre-registration. On the *compliance* readout `ha0_ext` already exists
(`nav_compliance.py:1125`) and binds today.

⭐⭐ **The one framing fact that decides what refcv5 may claim at all.** On this surface
`oracle_sel − os` = **−0.0751** [−0.0884, −0.0618] and `os − os_navzero` = **−0.0239**
[−0.0428, −0.0089]. **Perfect selection plus the oracle route is 0.099 m — the gap to `ha` is
0.1423 m.** ⇒ **The deficit is in the FAN, not in selection and not in routing.** Any refcv5 claim
built only on §5's selector or on nav wiring is, by arithmetic, incapable of clearing the bar; the
fan levers (§4 sampler, §1.2 #13/#14 vocabulary, §1.3 #19/#20 grounding) are the ones that can. This
is also why §7 puts the *free* selection work first and still calls the sampler and the grounding the
load-bearing rungs.

### 8.1 The claims, with their refutations

| id | CLAIM (both outcomes committed now) | REFUTED IF | ties to |
|---|---|---|---|
| **C1 — the headline** | refcv5's `os` **beats `ha`** on ADE 0–2 s with a paired CI **excluding 0 in the right direction**, on the 4,823-window surface — the bar refcv3 never cleared (+0.1423 [+0.1187, +0.1658] the wrong way at 40,284, and +0.1803 at 30,000: 10,284 further steps closed **21 %** and did not close it) | `os − ha` ≥ 0 with a separated CI, **or** unseparated (a tie with a trivial control is not a capability claim). ⚠️ A win on ADE alone does **not** establish C1: it must hold **jointly with C2** | `D-REFCV3-40284`, `H-ECHO-1` |
| **C2 — the four families** | at **T1**, refcv5 beats `ha` on **LONGITUDINAL** (target-speed accuracy, headway / time-gap / TTC to the lead) and **LATERAL** (heading, curvature, yaw-rate, cross-track), **each family separately, never pooled**, and is not separated-worse on any family | any family separated-worse than `ha`, **or** the LONGITUDINAL family unimproved — 92.2 % of the deficit is along-track (`D-REFCV3-AXIS1`) and 88.7 % of the oracle gap is longitudinal, so a refcv5 that improves only cross-track has not addressed its own defect | §2 I5, `D-REFCV3-AXIS1` |
| **C3 — the echo gate** | the deployed arm reads **`READS_BOTH`** on `stack/tanitad/eval/echo_gate.py` — scene degradation **and** ego degradation both separated, the `H-ECHO-8` signature | `ECHOING` (ego separated, scene not) ⇒ ⛔ **the arm is refused whatever its ADE**. MEASURED precedent: `ego_dropout 0.0` reads `ECHOING` *and* is the best 6 s displacement arm (13.27 m vs the `READS_BOTH` arm's 17.86) — **an ADE-scored gate would have selected the echoing arm** | `H-ECHO-8`, `H-EGO-LIT-4` |
| **C4 — the fan, not the pick** | refcv5's **oracle-in-fan at equal N** improves separated vs refcv4b's, and `fan_floor@k` (k = 1, 5, 10) rises without the fan-collision-vs-replay rate rising | oracle-in-fan flat ⇒ the sampler/vocabulary levers bought coverage nowhere and C1 (if it happened) came from selection — which §8.0's arithmetic says cannot reach the bar; report it as such | `H-DDA-3`, `H-DDA-4`, `D-REFCV4-VOCAB1` |
| **C5 — the mechanism is a sampler** | with `sampler="ddim"` the decoder is a **diffusion model by its own definition**: a noise schedule exists, two calls at different seeds differ, diversity D rises, and `σ ≡ 0, t = 0` reproduces the deterministic decoder **bit-for-bit** | the σ ≡ 0 identity fails ⇒ the implementation is wrong, not the idea; two seeds identical ⇒ the sampler is inert and `D-REFC-DDAUDIT-1`'s verdict ("an unrolled 3-pass anchor-refinement regressor") still stands for refcv5 and the registry must keep saying so | `D-REFC-DDAUDIT-1`, `H-DDA-3` |
| **C6 — flyability is not traded away** | the sampled fan's Kamm(μ = 0.7) violation rate does not exceed the deterministic fan's, and **selected-trajectory jerk does not rise** above refcv4b's | jerk rises separated. Context: refcv4b @ 9,500 already reads **2.42 m/s³** kept / **3.65** withheld against the human's **0.86** and `ha`'s **0.03** (`D-REFCV4B-EGODROP2`). A sampler that buys ADE with jerk is buying the wrong thing | §2 I4, `D-REFCV4B-FLYLOW1`, `D-REFCV4B-EGODROP2` |
| **C7 — the hierarchy is live, not a training-time story** | the post-training ablation panel of `PREREG_REFCV4B_HIERARCHY_EVAL.md` §3–§4, run on refcv5: **H-NAVC-1 AND H-NAVC-2 AND H-SEAM-1**, with the deliberate-regression (frame-blind) arm **FAILED by the echo gate** | any of the named failure modes: **PATH_FOLLOWS_WITHOUT_GOAL** (H-NAVC-1 ✓, H-NAVC-2 ✗ — nav reaches the operative layer through `m` and *not* the strategic decision, which is what refcv4b reads at step 9,500 and what refcv3 measured); **SEAM_FAILURE_RIGHT_GOAL_WRONG_PATH**; or E7-off / E9-off / H19-off **all** unseparated ⇒ the seams are inert at eval and **the registry may not call the hierarchy "live"** | `D-NAVCOMP-1..3`, `H-NAVC-1..3`, `H-SEAM-1`, `H-H19-1`, `H-CONS-1`, `H-SEL-1`, `D-REFCV4-NAV-WIRING` |
| **C8 — compliance clears its nav-blind floor** | plan-compliance with the TRUE command drops ≥ **0.10 absolute** under **both** nav-SHUFFLE and nav-ZERO (separated, paired), the changed subset follows the **FED** command under nav-FLIP, and the rate clears **`ha0_ext`** on the same windows (H-NAVC-3) | the shuffle/zero deltas unseparated ⇒ the `nav_true` rate is scene coincidence and may not be cited; compliance below `ha0_ext` ⇒ refcv5 follows nav **less often than a constant-curvature extrapolation of its own state complies by coincidence** — nav-following exists and is weak (refcv4b @ 9,500: **0.390 vs 0.418**, i.e. H-NAVC-3 **FALSE** at 23 % of training) | `D-NAVCOMP-3`, `H-NAVC-3` |
| **C9 — each adopted mechanism earned its place** | every lever in the refcv5 bundle passed its own **one-lever rig arm** (§7.2), and the bundle equals a **pinned frozenset** the preflight refuses to deviate from | a lever is in the bundle without a passed arm ⇒ the run is an unattributable bundle like refcv4b's five-lever arm, and its registry row must say so in the *Role* field rather than in a caveat | §7.4, `REGISTERED_DELTA_KEYS_V4` precedent |

**Margins, fixed here and not renegotiable afterwards:** 0.10 absolute on compliance deltas; 2 %
relative on ADE@6 s; family deltas separated. **A separated CI on a smaller margin is a real but
unusable difference and is reported as such** — never rounded up into support
(`PREREG_REFCV4B_HIERARCHY_EVAL.md` §4).

### 8.2 The conjunction — and the four ways refcv5 can "win" and still be refused

⭐ **C1 alone is not the claim.** The claim is **C1 ∧ C2 ∧ C3 ∧ C6** for capability, and
**C7 ∧ C8** for the hierarchy thesis. The programme has already measured all four ways an arm can
post a good headline and still be wrong, so each is named as a refusal, not a caveat:

1. **The ADE win with an echoing arm.** MEASURED (`H-ECHO-8`): the `ego_dropout 0.0` arm is the best
   6 s displacement arm in its panel *and* reads `ECHOING`. ⇒ **C3 refuses it.**
2. **The ADE win bought with jerk.** MEASURED (`D-REFCV4B-EGODROP2`): jerk 2.42 / 3.65 vs the human's
   0.86. ⇒ **C6 refuses it.**
3. **The ADE win from selection.** By §8.0's arithmetic, selection + oracle nav cannot span the
   0.1423 m gap; an ADE win with a flat oracle-in-fan is a *selection* result mislabelled. ⇒ **C4
   catches it, and the registry row must say "selection", not "driving".**
4. **The pooled win that hides a family.** ⇒ **C2 refuses it.** And `D-REFCV3-40284b` is the
   precedent: the pooled +0.1423 is the model's *best* case — the loss is **2.87× larger** on
   manoeuvre windows, worst on `brake_stop` (2.31×, +0.4930 on 13.08 % of windows). **refcv5's
   headline must be reported stratified**, or it repeats a reading that was already corrected once.

### 8.3 What would refute the *programme's* thesis, not just this arm

These are the sharper statements, and they are worth more than C1.

| # | refutation of the thesis | the reading that establishes it | consequence |
|---|---|---|---|
| **R1** | **The hierarchy conditions nothing.** E7-off, E9-off, g_str-ZERO, g_str-SHUFFLE and H19-off are **all** unseparated on every family and on compliance, on a fully-trained refcv5 with the seams open | `PREREG_REFCV4B_HIERARCHY_EVAL.md` §3, run on refcv5 | the three-level hierarchy is a training-time regulariser at best. `PROGRAM_OVERVIEW.md` and the paper must say so; the USP claim (§2 I1) is withdrawn, not softened |
| **R2** | **The anchored-vocabulary class is the ceiling.** refcv5 clears `ha0` and loses to `ha` *again*, with oracle-in-fan improved and `oracle_sel` still above 0.2996 | C1 fails **while** C4 succeeds | the fan can be made to cover the answer and the model still cannot pick or refine to it ⇒ the binding constraint is **perception**, not vocabulary (the residual risk `D-REFCV4-GATE1` already named: `os` 0.4419 and `oracle_sel` 0.3668 differ by only 0.0751 off a 1.0838 m raw anchor) |
| **R3** | **Diffusion adds nothing here.** C5 holds (the sampler is real) and C4 fails (oracle-in-fan at equal N flat, diversity up) | `H-DDA-3`'s committed failure branch | truncated diffusion is a *diversity* mechanism whose diversity we cannot cash; keep the deterministic refiner and **rename it honestly** in the registry. This is a publishable negative about DD's transfer to a control-space vocabulary |
| **R4** | **The grounding gap was not the gap.** E-DDA-1 and E-AGT-1 both unseparated on the LONGITUDINAL family at the rig, and E-BEV-1 too after WP-11 | `H-DDA-1`'s failure branch, twice | the PV tokens already carry the geometry; the cross-attention deficit the audit found is real in *source* and inert in *behaviour*. ⇒ do not spend on BEV lifting, and say so |
| **R5** | **Vision-only at this scale cannot beat a hold-action control on this corpus.** C1 fails across refcv3, refcv4b and refcv5 while C3 holds (`READS_BOTH`) and every fan/selection lever is exhausted | three arms, one bar | this is the honest form of the programme's hardest possible negative, and it is a *result*: a sub-300 M vision-only planner on 4,572 clips of PhysicalAI-AV does not clear a trivial dynamics extrapolation at 2 s. It would redirect the programme to data scale, to closed loop (AlpaSim), or to the VLA extension (§9) — and it is why `ha`/`ha0`/`ha0_ext` are recomputed on every arm rather than being quoted from this document |

### 8.4 What may NOT be cited as support for any refcv5 claim

Carried from `PREREG_REFCV4B_HIERARCHY_EVAL.md` §5, `EVAL_DOCTRINE.md`, and this programme's own
retractions:

- ⛔ **T0 numbers as capability.** `oracle_sel`, oracle-in-fan and oracle-in-vocabulary are **ceilings**;
  `anchor_acc`, `goal_str` and any `metrics.jsonl` value are **T0 loss-surface diagnostics** (C1).
- ⛔ **MODEL-FREE gates compared to achievements.** A raw-anchor ceiling is never set against `ha`,
  `os` or `oracle_sel` (`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`; the error that nearly killed refcv4).
- ⛔ **A high `nav_true` compliance rate on its own** — without its shuffle **and** zero controls it is
  two named criteria violations (registry v2.6.0), and it can be scene coincidence.
- ⛔ **The RL reward as a result.** The reward is a training signal; the claim is T1 on the four
  families (§6.1).
- ⛔ **A rig-rung number as a model claim** (`refc_v3.py:511-514`).
- ⛔ **`overlapping_holdout_se`** in any form — it biases the point estimate as well as the interval.
- ⛔ **A learning-curve exponent** without its fit window, R² and n; and never as a restart argument.
- ⚠️ **Any refcv4b number from the unpatched `refcv3_arm.py`** (`D-NAVCOMP-2`: `run_dump` passed no
  `ego_state`, starving an `ego_state_inject` model's goal path). refcv5 inherits E11' and therefore
  inherits this trap; the manifest field `ego_state_fed` must read true on every refcv5 dump.

### 8.5 The claim register

Each row above is registered in `GOALS_AND_CLAIMS.md` (§10) as `D-REFCV5-PLAN-*` with status
**OPEN — pre-registered, both outcomes committed**, and is closed by the artifact named in its row —
not by a summary, a report, or this document.


## §9 Interface for the VLA extension (sibling stream)

### 9.0 What this section is, and what it is not

This is **the contract the language module codes against** — the ports, their shapes, the vocabularies,
the budget and the one measurement that makes the extension a USP rather than a caption generator.
⛔ **It does not design the VLA.** The design lives with the sibling stream:
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-extension-frontier/RESULT.md`
(§0 banked at commit `7d4b0b7`; §1–§6 in progress) and `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md`
(not yet in HEAD at the time of writing). Where the two disagree about a *port*, this section wins and
the disagreement is escalated to the Master Mind; where they disagree about the *module*, that stream
wins.

⚠️ **The first constraint, stated before any port.** The whole cascade above the trunk is **≈ 2.15 M
parameters** (MEASURED, registry §4.5: `phi_tac` 1,757,440 · `tac_latent_proj` 262,656 · `gstr_cond`
66,816 · `nav_inject` 50,176 · `tac_heads` 14,364 · `scorer` 1,156 · `str_goal_head` 771). A ≤ 1 B
language module is **100–400× larger than the layers it complements**. It may **read and propose
into** the cascade's decisions; it may **not replace** them, because a module that large will win any
gradient competition it is allowed to enter, and the hierarchy's attributability (§2 I1) dies with it.

### 9.1 The mount point — `hierarchy_hook`, the port that already exists

⭐ **REF-C already has a documented external-brain port, and it is unfilled.**
`RefCModel.forward(..., hierarchy_hook=...)` (`stack/tanitad/refs/refc.py`, the `hierarchy_hook` block
at `:2268-2300`, docstring `:2230-2255`) is called as

```
hook(pooled_seq: Tensor[B, W, F], ctx: Tensor[B, d_ctx]) -> dict
```

and its returned `"maneuver_logits"`, `"target_latent"` and `"bank_speed_pred"` entries **fill the
model's ports ONLY where the caller passed `None`** — *"an explicitly supplied port always wins, so
the hook can never silently override an experiment's input"* (`refc.py:2241-2243`). REF-C v3's own
hierarchy is supplied through exactly this hook (`refc_v3.py`), so the VLA is not a new mechanism: it
is a **second supplier on an existing, gated, experiment-safe seam**.

⭐⭐ **And the docstring names the reason it exists, which is the VLA's latency argument in advance:**
*a hierarchy built outside the class can read the window it is conditioning on* **without encoding the
frames a second time** *— the encoder is ~90 % of the tick, and a cached one would be a stale-latent
hazard* (`refc.py:2236-2240`). ⇒ **The language module must consume REF-C's trunk tokens. It must not
run a second vision encoder.** That is a contract term, not a preference — it is where the 300–500 ms
budget of §9.6 is won or lost.

⛔ **One port is live and one is not, and the difference matters.** `maneuver_logits [B, 5]` reaches
`refc.py:1645-1647` and reweights the anchor prior (H19) — *"an outside tactical brain speaking the
5-way surface"*, and it is **never filled today** (`D-REFCV4-NAV-WIRING`, VLA RESULT §0.1). It is the
**only** port through which an external module already changes behaviour with zero new wiring. Every
other write below needs a new, zero-init, individually-gated edge (§2 I2).

### 9.2 READ ports — what the module may consume

Widths are for `size base` (`V3_SIZES["base"] = (88, (3, 6, 16, 6))`, `feat_dim = base_width · 8 =
704`, `refc.py:273, :291`); at the `tiny` rig rung `feat_dim = 256`. ⚠️ **Line numbers drift — the
function/attribute name is the stable reference.**

| port | shape | where | what it is | notes for the module |
|---|---|---|---|---|
| `pooled` | `[B, 704]` (`feat_dim`) | `RefCModel.forward`, `refc.py::forward` hierarchy branch | the trunk's pooled features at t0 | the *pooled* vector, not the map |
| `pooled_seq` | `[B, W=8, 704]` | same, `pooled_all.reshape(b, w, -1)` | the same over the 8-frame window | ⭐ **the hook's first argument** — the module's primary visual input |
| `fmap` (PV token map) | `[B, 704, 8, 20]` → 160 tokens | `refc.py::forward`; consumed by `feat_proj: Linear(feat_dim, d)` (`:1206`) | the **8 × 20 = 160 perspective-view tokens**, 256 × 640 **cylindrical**, `f_ref` 305.577, HFOV **120.0°** | ⛔ the pinhole formula gives 92.6° and is **wrong** here (§3.2); use `calib.CanonicalFrame(projection="cylindrical")` and `bev_raster.readout_column_index` for any token↔azimuth mapping |
| `ctx` | `[B, 256]` (`StrategicCtxConfig(hidden=512, d_ctx=256)`) | `refc_v3.py::StrategicCtx`, supplied as the hook's **second argument** | strategic GRU context over the window; `ctx = ctx + nav_s` inside `hook()` (zero-init `nav_to_str`) | already carries nav — see §9.5 before treating it as vision-only |
| `g_str` | `[B, 3]` → `[B, 23]` under E19 | `refc_v3.py::str_goal_head` | `(cos, sin)` route bearing + `tanh` along-track preference; E19 widens to `g_str_geo(3) + p_man(4) + t_bin(6) + p_man2(4) + t_bin2(6)` | ⚠️ **NAV_BLIND today** (nav-compliance Δ_shuffle = 0.0000). Do not build on it without checking `H-NAVC-2`'s outcome |
| `z_tac` | `[B, 512]` (`d_tac`) | `refc_v3.py::phi_tac`, FiLM'd by `g_str` (zero-init `gstr_film`) | the tactical latent; input to the scorer and to `target_latent` | the natural conditioning vector for a tactical language head |
| `lat_logits_tac` / `lon_logits_tac` | `[B, 8]` each | `refc_v3.py::lat_head_tac`, `lon_head_tac` | the **v7.2 8-way factored tactical heads** | reach the decoder only through E7/E9 (`SEAM_STATE.md`) |
| `g_tac` | `[B, k·4]`, k = 3 taus `(20, 40, 60)` → **12**; layout `(x, y, heading, speed)@τ` (`GOAL_DIMS = 4`, `models/tactical.py:78`) | `refc_v3.py::tac_goal_head` | tactical **geometric goal points**; E14 echo-quotiented form = `ha0_ext + zero-init residual` | ⭐ **this is the geometric goal port** the module must read and may propose into |
| the fan | `anchor_traj [B, N, n_steps, 2]`, `anchor_logits [B, N]`, `offset`, `sel_idx [B]` | `RefCModel.forward` return dict (`refc.py:2247-2250`) | the 117 emitted candidates, their scores, and the pick | ⭐ **the object the consistency constraint (§9.7) is measured against** |
| `m` (measurement) | `[B, d_m]` | `refc.py` measurement encoder | `v0` (+ E11' ego channels, per-sample withholding 0.5 + the **X15 presence bit**) ⊕ nav one-hot | ⛔ **read-only, and see §9.5** |

### 9.3 WRITE ports — what the module may emit, and where it lands

| port | shape | lands in | gate | status |
|---|---|---|---|---|
| `maneuver_logits` | `[B, 5]` (log-probs over the kin3-derived 5-way surface) | `refc.py:1645-1647` — reweights the **anchor prior** (H19) | already gated: fills only when the caller passed `None`; `maneuver_to_anchor` must be present | ⭐ **LIVE, unfilled, zero new wiring** |
| `target_latent` | `[B, d_tac]` | the decoder's `tgt_film` (E7) | as above | LIVE, filled today by `refc_v3`; a VLA arm must **replace or blend**, and say which |
| `g_tac_proposal` | `[B, k·4]` = `[B, 12]` | a **new** zero-init residual edge into `tac_goal_head`'s output | **new edge — must be zero-init and individually switchable** (§2 I2) | to be added by the VLA stream |
| `p_man` / `t_bin` proposal | `[B, 4]` / `[B, 6]` (E19's layout) | a **new** zero-init residual edge into `str_goal_head`'s E19 slots | as above | needs E19 first (WP-8) |
| `selector_prior` | `[B, N]` additive log-prior on the fan | `refc.py:1677-1683`, beside the H19 / factored / LAN / goal priors | as above | a natural port for a "which of these 117 do you endorse" reading |
| `text` (the explanation) | tokens | **no model node** | — | ⛔ **the explanation must not close a loop into the planner.** It is scored (§9.7), never consumed |

**Every write port ships with its eval-time switch**, so the post-training ablation table of
`PREREG_REFCV4B_HIERARCHY_EVAL.md` extends by one row per VLA edge (§2 I2). An edge without a switch
is not admissible.

### 9.4 The vocabularies the module must speak — verbatim, from source

⛔ **These are pinned constants (`stack/tanitad/models/vocab_v7.py`), not descriptions.** A module
that emits a token outside them is emitting an unlabelled string, and no consistency metric can score
it.

**Time bands** (`stack/scripts/s2_geom_emit_v7.py:50-59`): `OPERATIVE_S = (0.0, 2.0)` ·
`TACTICAL_S = (2.0, 6.0)` · `GAP_S = (6.0, 8.0)` — **belongs to no layer, deliberately** ·
`STRATEGIC_S = (8.0, 30.0)` · `LOOKAHEAD_S = 30.0`.

**Strategic (the `STRATEGIC_S = (8, 30)` event vocabulary), 8 goals + 7 actions + 2 arg slots:**

```
STRATEGIC_GOAL_TOKENS_V7  = (FOLLOW_ROUTE, TURN_LEFT_FOLLOW_ROUTE, TURN_RIGHT_FOLLOW_ROUTE,
                             STOP_AT_FOLLOW_ROUTE, EXIT_LEFT_FOLLOW_ROUTE, EXIT_RIGHT_FOLLOW_ROUTE,
                             LANE_CHANGE_L_FOLLOW_ROUTE, LANE_CHANGE_R_FOLLOW_ROUTE)
STRATEGIC_ACTION_TOKENS_V7 = (HOLD_MAIN_ROAD, PREPARE_TURN_L_FOLLOW_ROUTE,
                             PREPARE_TURN_R_FOLLOW_ROUTE, PREPARE_STOP_FOLLOW_ROUTE,
                             PREPARE_EXIT_FOLLOW_ROUTE, PREPARE_LANE_CHANGE_FOLLOW_ROUTE,
                             RESUME_CRUISE_FOLLOW_ROUTE)
STRATEGIC_ARG_SLOTS       = ("within_m", "by_time_s")     # a token WITHOUT these is ambiguous
```

⚠️ The arg slots are **not optional**: the PI's own counter-example (`01bee851`, three manoeuvres
+76° / −43° / +69°) is distinguishable only by distance and time. E19's `t_bin` bins over `[0, 30]` s
are `(0-2, 2-5, 5-9, 9-14, 14-21, 21-30)`. Band census (train): **1,882** manoeuvres start in
`[8, 30)`, 463 in `[2, 6)`, 687 in `[0, 2)`, 256 beyond 30 s.

**Tactical — the 8-way v7.2 factored classes** (this is *the* pair the module's tactical claim is
scored on):

```
TACTICAL_LAT_ACTIONS_V7 = (LANE_KEEP, LANE_CHANGE_L, LANE_CHANGE_R, ABORT_LC,
                           NUDGE_L, NUDGE_R, TURN_L, TURN_R)
TACTICAL_LON_ACTIONS_V7 = (FOLLOW, CRUISE, YIELD_MERGE, BRAKE_TO, CREEP, HOLD,
                           ADAPT_SPEED_FOR_CURVE, ACCELERATE)
LAT_ACTION_ARG_SLOTS = ("within_m",)   LON_ACTION_ARG_SLOTS = ("v_target_ms", "within_m")
```

plus the **24 `TACTICAL_GOAL_TOKENS_V7`** (a *set*, expressed against the anchor reference:
`FOLLOW_LANE`, `TURN_L/R`, `YIELD*`, `STOP_POINT`, `SPEED_BAND` — **always present** —,
`CORRIDOR_OFFSET`, `EVADE_IN_CORRIDOR`, `OVERTAKE_VEHICLE`, `MERGE`, `GAP_TARGET`,
`REACT_ON_ONCOMING`, `TAKE_EXIT_L/R`, `TRAFFIC_LIGHT_REACT{,_RED,_YELLOW,_GREEN}`,
`LANE_CHANGE_L/R`) with `ANCHOR_ARG_SLOTS = ("goal_x_m", "goal_y_m", "t_reach_s", …)`, and the
admissibility maps `GOAL_ADMISSIBLE_LAT` / `GOAL_ADMISSIBLE_LON` — **an emitted (goal, action) pair
that violates them is a contract violation the module must not produce.** ⚠️ `OVERTAKE_VEHICLE` vs
`EVADE_IN_CORRIDOR` is about the **object** (moving vs static/VRU), not the manoeuvre; they share the
verb, and conflating them files 419 parked-car evasions with 13 real overtakes.

**Nav (input):** `NAV_COMMAND_TOKENS = (NAV_FOLLOW_ROAD, NAV_TURN_L, NAV_TURN_R)`,
`NAV_ARG_SLOTS = ("distance_m", "time_s")`, `NAV_PROVENANCE = ("nav-system", "ego-future")`, plus the
`nav_known` bit (E21). ⚠️ On PhysicalAI the provenance is **`ego-future`** — an oracle, and a
**per-clip constant** (train `follow 2,897 / left 811 / right 864`). ⇒ Every VLA number conditioned on
nav carries the **shuffle and zero controls**, exactly like every other nav number here.

**Geometric goal points:** `GOAL_DIMS = 4`, layout `(x, y, heading, speed)` per τ, τ ∈ `(20, 40, 60)`
decisteps = **2.0 / 4.0 / 6.0 s**, ego frame **x forward, y left, z up** (the rig frame
`obstacle.offline` uses; MEASURED by the parked-car experiment, 7.4× over the nearest alternative).

**Question input.** A question/instruction port is admissible **as an input to the language module
only**. ⛔ It may not reach any planner goal node, and the module's answer to a question may not be
routed into a write port of §9.3 in the same tick — otherwise a question becomes a control channel and
the goal path stops being information-disjoint from the situation path (PI, 2026-08-03). The CoT/VQA
material we hold is described in the sibling's §0.2 (23,644 rows / 4,729 clips; `t0_us` = 5,100,000
against our 8.0 s anchor ⇒ **every join is 2.9 s off unless re-anchored**).

### 9.5 What the module may NOT read at inference — the invariants, restated as port rules

1. ⛔ **No ego state beyond the measured `v0` at t0**, and no future of any kind (§2 I3; PI 2026-09-02
   allows measured velocity at cycle time and nothing more). The five E11' channels reach `m` under
   `ego_dropout 0.5` + the X15 presence bit; a VLA that reads `m` **must honour the same withholding
   draw** — `ego_keep` is a caller-supplied port precisely so the two paths are withheld *together*
   (`refc.py` E11' comment: "a second, unsynchronised dropout" is the defect it prevents).
2. ⛔ **No situation-classifier output in any goal input**, in any form — posterior, argmax, embedding,
   or anything derived from them (PI 2026-08-03). The admissibility check is literal: *"could this have
   been computed from the situation classifier's output?"* If yes, inadmissible until shown otherwise.
3. ⛔ **No label at inference.** `obstacle.offline`, the v7.2 labels and the Alpamayo CoT are
   **train-time** material. A grounding token that resolves against the join at inference is a leak,
   not a feature.
4. ⚠️ **`ctx` already contains nav** (`ctx = ctx + nav_s`). A module reading `ctx` and claiming a
   vision-only reading is wrong about its own inputs; either read `pooled_seq` or state that nav is in
   the path and carry the controls.
5. ⚠️ **LiDAR at inference is unresolved doctrine** (`D-REFCV5-PLAN-7`, §2 I3). The VLA plan must not
   assume it.

### 9.6 The budget — 300–500 ms per tact on Thor

The tact is the **whole** tick: trunk + cascade + decoder + selector + the language module. The module
gets what is left, not the whole budget.

| item | figure | class |
|---|---|---|
| REF-C trunk inference latency on Thor | ⛔ **UNMEASURED** — and it is *the* number that sets the module's share. First item of the VLA plan's WP-0 | — |
| the encoder's share of a REF-C tick | **~90 %** (`refc.py:2238`) — which is why §9.1 forbids a second vision encoder | INHERITED (source comment) |
| Thor batching | 20 SMs **saturate at batch 8**; throughput flat 12.3–14.1 windows/s across a 6× batch range | MEASURED (CLAUDE.md, a *training* step on a different trunk — quoted for the saturation point only) |
| Thor memory probes | ⛔ only in-process `torch.cuda.max_memory_allocated()` is admissible; `mem_get_info`, `free`, `tegrastats` and `VmRSS` all lie, in both directions | MEASURED (CLAUDE.md) |
| decode floor for a 0.8 B model | `bytes(weights)/bandwidth` ⇒ **~5–6 ms/token bf16 · ~3 ms FP8 · ~1.5 ms NVFP4** | ESTIMATED (spec floor; VLA RESULT §0.3) |
| ⇒ tokens affordable in 300–500 ms **if the module owned the whole tact** | ≈ **40–70 bf16**, **100–150 FP8** — and it does not own it | ESTIMATED |

⇒ **Two contract terms follow.** (a) The emitted per-tact output is **short and structured
(≤ ~40 tokens)**, or it is **latent/executable** with prose rendered asynchronously at a lower rate.
(b) The module's write ports (§9.3) must be producible **without** generating prose — a `p_man` vector
or a `[B, 5]` log-prob costs one forward, not forty decode steps. ⛔ **A design in which the planner
waits on text is refused at the interface, not at the eval.**

### 9.7 The consistency constraint — the USP, and the one way it can be faked

**The requirement.** The module's explanation must be **consistent with the trajectory hypotheses the
planner actually emits** — concretely, with (i) `g_tac`'s geometric goal points `(x, y, heading,
speed)@τ` and (ii) the **v7.2 factored class of the SELECTED anchor**, obtained by mapping the winner's
`(a_lon, a_lat)` cell through `refc_tactical.factor_from_kinematics` — the same stratifier
`D-REFCV3-40284b` used, so the mapping is already exercised and tested.

**The readout** — a **fifth metric family** beside the four (§2 I5), reported per-family and never
pooled into them:

| readout | definition | must-read control |
|---|---|---|
| `tac_consistency` | agreement rate between the module's stated (LAT, LON) pair and the selected anchor's cell class | **constant-only**: a module that always says `(LANE_KEEP, CRUISE)` must read the class **prior** — our corpus is 86.59 % `lane_keep`, 75.76 % `steady`, 67.18 % both, so the prior is ≈ 0.67 and any consistency below it is worse than a constant |
| `goal_consistency` | distance between a named goal point and `g_tac`'s `(x, y)@τ`, per τ | **shuffle**: the module's output paired with a *different* window's plan must read the marginal |
| `contradiction_rate` | emitted (goal, action) pairs violating `GOAL_ADMISSIBLE_LAT/LON` | must read **0** on a compliant module — a known value |
| `fan_endorsement` | does the module's endorsed candidate survive the reach band and the Kamm filter? | a **random-candidate** control must read the fan's own survival rate |

⛔⛔ **THE FAKE, NAMED IN ADVANCE — AND IT IS THE ECHO FAMILY AGAIN.** If the module can **see**
`sel_idx` / `g_tac` when it explains, then **consistency is satisfiable by description** and the
number measures nothing: a post-hoc describer scores ≈ 1.0 by construction, exactly as flagship v1's
route head scored 1.0000 by echoing its own nav input (369/369 and 81/81), and exactly as
`D-NAVCOMP-1` refused the old route metric for being a bijection of the fed token. ⇒ **The contract
requires the module's information state at explanation time to be declared, and the two regimes to be
reported separately:**

| regime | sees the plan? | what consistency means | what must be reported instead |
|---|---|---|---|
| **PROPOSER** | ⛔ no — it emits before/independently of the pick | a **real** measurement: two systems agreeing from the same scene | consistency is the headline; the shuffle control must drop it |
| **DESCRIBER** | ✅ yes | ≈ 1.0 **by construction — not a result** | the headline becomes **faithfulness to the SCENE** (does the named object exist? `grounding_via_vqa`'s 2-D boxes are the only channel that lets a CoT token be checked against image space), never agreement with the plan |

⚠️ And the sharper form of the same rule: **a consistency loss trained into the planner makes the
planner explainable-by-construction and unfalsifiable at the same time.** If text conditions the
trajectory *and* is scored against it, the pair is a loop — the C6 confound in language costume. If
such a loss is used, the eval must include an arm with the loss **off at inference**, and the
deliberate-regression arm is a module fed **shuffled scenes** whose consistency must collapse. If it
does not collapse, the instrument cannot see what it is cited for and the panel is **VOID**.

⚠️ **Do not build the consistency claim on `g_str`.** It is **NAV_BLIND** today (Δ_shuffle = 0.0000),
and `H-NAVC-2` is open. A strategic-consistency readout is admissible but must be reported as
*conditional on `H-NAVC-2`*, or the module inherits an unfalsifiable seam (the sibling's §0.1 makes
the same point).

### 9.8 Contract versioning

The contract is **the source**, not this table: `refc.py::RefCModel.forward` (ports and return dict),
`refc_v3.py` (`StrategicCtx`, `phi_tac`, `str_goal_head`, `tac_goal_head`, `lat_head_tac`,
`lon_head_tac`), `stack/tanitad/models/vocab_v7.py` (vocabularies), `stack/scripts/s2_geom_emit_v7.py`
(bands), `stack/tanitad/models/tactical.py` (`GOAL_DIMS`). **Widths change with `size`**; the VLA must
read them from the built model, never hardcode 704 / 512 / 256.

**What breaks the contract, and therefore needs a note to the Master Mind rather than a local fix:**
E19 widening `str_goal_head` 3 → 23 (WP-8); the vocabulary v5 rebuild changing the anchor grid
13 × 9 → 13 × 11 (WP-1, changes the selected-anchor class mapping); the §4 sampler changing the fan
from `[B, N, S, 2]` to `[B, G·N, S, 2]` (WP-4, changes `sel_idx`'s meaning); and any BEV token added
to the decoder KV set (WP-12).


## §10 Registered decisions

The rows below are appended to `Project Steering/GOALS_AND_CLAIMS.md` in the same turn as this
document, under the heading **`D-REFCV5-PLAN` — the refcv5 design plan (2026-09-05, Arch+Inference)**.
⛔ **This table is a mirror for readability. The register is the source**; where they disagree, the
register wins and this table is fixed.

| id | decision | status | where it is argued |
|---|---|---|---|
| **D-REFCV5-PLAN-1** | The DiffusionDrive gap table is **closed out**: 43 audited components → **19 IMPLEMENT · 9 ADAPT · 9 KEEP OURS · 5 SKIP**, each with its reason, carried by **10 pre-registered one-lever arms**. No component is left "to be decided" | **DECIDED** | §1 |
| **D-REFCV5-PLAN-2** | The six USP invariants **I1–I6** are **refusal conditions, not trade-offs**: an adopted mechanism that violates one is refused, not negotiated. Every DD mechanism in §1 is checked against them individually | **DECIDED** | §2 |
| **D-REFCV5-PLAN-3** | ⭐ **PhysicalAI-AV carries LiDAR** — 128-row spinning, 10 Hz, ~200 spins/clip, **97.44 % coverage**, 99.6 TB = 68 % of the dataset by bytes — and the artifact we build from it is a **DD-faithful 256 × 256 @ 0.25 m/px, ±32 m, 2-height-bin BEV histogram** shipped as a **PNG sidecar `<clip>.v2bev.pt`** beside the existing `*.v2ep.pt`, so the 161 GB camera cache is not rewritten and frames stay byte-identical (I6) | **SUPPORTED (MEASURED, three independent probes)** — build = **WP-DE-BEV-1**, owner DataFlyWheel | §3 |
| **D-REFCV5-PLAN-4** | ⭐ **The diffusion noise lives in CONTROL space, not metre space**: `ε` on `(a_lon, a_lat)` normalised by the grid's own ranges `(4.0, 3.0)`, re-rolled through `rollout_unicycle_varstep`, so **every sample is flyable by construction** and the v0-conditioned vocabulary stays the prior. DD's metre-space normaliser is kept as the **deliberate-regression arm that must FAIL flyability** | **DECIDED**, with both outcomes committed | §4.2 |
| **D-REFCV5-PLAN-5** | **The ranked object becomes the emitted object.** `sel_refined + sel_score_emitted + sel_ce_reach` (0 new params, flags already in source) replaces ranking a fan two passes staler than the one emitted (`sel_idx_base` unchanged on **201/201** windows). The four-family sub-metric selector (§5.2) is the second step, with **no PDM and no map**, and `DAC` is refused as unconstructible on PhysicalAI | **DECIDED** | §5 |
| **D-REFCV5-PLAN-6** | ⭐ **The RL stage's job is the fan's FLOOR, not ADE**: V2's published shape is PDMS@10 +9.1 with @1 +1.4 and diversity −28 %. The vehicle is a **head-only scale policy** (trainable 8.60 %); **`G-REWARD` is a precondition, not a milestone** — hold-v0 must beat the human on **≤ 30 %** of lead windows (today **78.3 %**), and a reward that fails it may not train anything | **DECIDED** | §6 |
| ⛔ **D-REFCV5-PLAN-7** | **PI DECISION REQUIRED — is LiDAR an INFERENCE INPUT or a TRAINING-TIME TEACHER?** The constitution's literal wording is *vision-only*; the Mission Plan (`:181`) and ROADMAP L4 foresee LiDAR/radar fusion. **Default = teacher** (camera-only inference, E-BEV-1); the input arm (**E-LIDAR-1**) is built and pre-registered but **not run** without the ruling. Either way the data is built once and the two arms are paired on the same windows, so *what the sensor buys beyond what the camera can be taught* is the measured quantity | **OPEN — PI** | §2 I3, §3.4 |
| **D-REFCV5-PLAN-8** | The ladder is ordered by **what is free first**: **10 of 14 work packages start today at zero pod GPU**, four need a refcv4b checkpoint, and the **LiDAR build is deliberately OFF the model critical path** — the camera-only closure (E-DDA-1 + E-AGT-1 + E-BEV-1 on the agent raster alone) is complete without a single byte of LiDAR | **DECIDED** | §7.0, §7.4 |
| ⛔ **D-REFCV5-PLAN-9** | **THREE INSTRUMENT GAPS BLOCK READS, NOT RUNS — each MEASURED, all ESCALATED.** (1) the 12 eval-time ablation switches of `PREREG_REFCV4B_HIERARCHY_EVAL.md` are **still not CLI flags** of `refcv3_arm.py`, so the post-training hierarchy panel cannot start; (2) **`ha0_ext` is NOT an arm of the REF-C harness** — `refcv3_arm.py:961` is `arms = ["os","ha","ha0"]`, `ARM_TIERS` has no `ha0_ext`, and the banked ARM JSON's six `arm_keys` do not include it, while the implementation exists (`stack/tanitad/eval/echo_gate.py::ha0_ext`) and the **refav1** harness runs it ⇒ **the refcv5 acceptance bar "beat both `ha` and `ha0_ext`" is unreadable on the 4,823-window surface today**; (3) `fan_floor@k`, fan-collision-vs-replay, diversity D and the sampled-fan Kamm rate **do not exist**, so the sampler, selector and RL rungs have **no primary readout at all** | **ESCALATED — 0 GPU, must precede WP-9b, WP-4, WP-7, WP-13** | §7.3 |
| ⭐ **D-REFCV5-PLAN-10** | **THE DEFICIT IS IN THE FAN, NOT IN SELECTION AND NOT IN ROUTING** — arithmetic on banked numbers: `oracle_sel − os` = **−0.0751** [−0.0884, −0.0618] and `os − os_navzero` = **−0.0239** [−0.0428, −0.0089], so **perfect selection plus the oracle route is 0.099 m against a 0.1423 m gap to `ha`**. ⇒ any refcv5 claim built only on the selector or on nav wiring is, by arithmetic, incapable of clearing the bar | **SUPPORTED (MEASURED, `refcv3-40284-openloop.ARM.json`)** | §8.0 |
| **D-REFCV5-PLAN-11** | The acceptance bar is a **conjunction**: **C1 ∧ C2 ∧ C3 ∧ C6** for capability and **C7 ∧ C8** for the hierarchy thesis, with all nine refutations committed in advance. **Four ways to post a good headline and still be REFUSED** are named, each already MEASURED here: the echoing arm (`H-ECHO-8`), the jerk purchase (`D-REFCV4B-EGODROP2`), the selection result mislabelled as driving (D-REFCV5-PLAN-10), and the pooled win that hides a family (`D-REFCV3-40284b`, 2.87× on manoeuvre windows) | **DECIDED — pre-registered, both outcomes committed** | §8.1, §8.2 |
| **D-REFCV5-PLAN-12** | ⭐ **The VLA mounts on `hierarchy_hook`, which already exists and is unfilled.** `hook(pooled_seq, ctx) -> dict` fills ports **only where the caller passed `None`**; `maneuver_logits [B, 5]` reweights the H19 anchor prior with **zero new wiring**. Contract terms: **no second vision encoder** (the encoder is ~90 % of a tick); ≤ ~40 tokens per tact or a latent/executable output; every new write edge zero-init, individually gated, and carrying its eval-time switch; and the module's **information state at explanation time must be DECLARED** — **PROPOSER** (consistency is a measurement) vs **DESCRIBER** (consistency ≈ 1.0 by construction and is **not a result**) | **DECIDED — the contract; the module's design stays with the VLA stream** | §9 |
| **D-REFCV5-PLAN-13** | **refcv5 bundles ONLY levers whose one-lever rig arm PASSED**, the bundle is pinned as a **frozenset the preflight refuses to deviate from**, and the registry row declares it a bundle. A lever whose arm has not run by the launch date is **left out, not launched hopefully**. *(refcv4b already forfeited attribution with five levers in one arm; refcv5 may repeat that only deliberately and in writing.)* | **DECIDED** | §7.4 |


## Manifest

**GPU spent by this document: 0.** Every number carries its evidence class; nothing here was measured
by running a model.

### Deliverables

| artifact | where it lives | state |
|---|---|---|
| `Project Steering/REFCV5_DESIGN_PLAN.md` — this document, §0–§10 + Manifest | **repo**, branch `agent/arch-inf-20260803` | committed across `cc3c57f` (§1–§3), `ee459a6` (§4–§6), then §7 · §8 · §9 · §10+Manifest, each verified in HEAD by a length-guarded blob comparison |
| `Project Steering/GOALS_AND_CLAIMS.md` — the `D-REFCV5-PLAN-1..13` block | **repo**, same branch | appended (INSERT, never rewrite) in the same turn; the six sibling rows of today (`D-REFCV4B-EGODROP2`, `H-EGO-LIT-4`, `D-REFC-DDAUDIT-1`, `D-DDV2-CODE-1`, `D-NAVCOMP-1`, `D-RL-READY-1`) verified present before and after |
| the §7 ladder as a work-package list (WP-0 … WP-14) | inside this document | ⛔ **not yet mirrored into `Project Steering/BACKLOG.md`** — a Master Mind action, see Escalations |

**Nothing produced by this stream lives in only one place.** No pod was touched; no worktree was used;
no file was left off-repo.

### Escalations — integration the Master Mind must schedule, not read about

| # | what | why it cannot wait |
|---|---|---|
| **E1** | ⛔ **WP-0.1 — the 12 eval-time ablation switches are still not CLI flags of `taniteval/tools/refcv3_arm.py`.** Already ESCALATED inside `PREREG_REFCV4B_HIERARCHY_EVAL.md`; repeated here because it now blocks a second thing | refcv4b's ETA is **~2026-09-06 08:00 UTC**. The post-training hierarchy panel — and with it E-DDA-2's first free read (`H-SEL-1`) — **cannot start** without it. 0 GPU, ~1 day |
| **E2** | ⛔ **WP-0.2 — `ha0_ext` is not an arm of the REF-C harness.** MEASURED today: `refcv3_arm.py:961` = `arms = ["os","ha","ha0"]`; `ARM_TIERS` (`:237-238`) has no `ha0_ext`; the banked `refcv3-40284-openloop.ARM.json` lists six `arm_keys`, none of them `ha0_ext`. The implementation exists (`stack/tanitad/eval/echo_gate.py::ha0_ext`) and `taniteval/tools/refav1_arm.py:639` + `t1_eval.py:162` already run it | **The refcv5 acceptance bar in every steering document says "beat both `ha` and `ha0_ext`" — and on the 4,823-window surface that bar is currently UNREADABLE.** ⛔ It must be ported as the **same shared call**, per `echo_gate.ha0_ext`'s own docstring (a control re-implemented beside the thing it controls drifts, and then the gate measures the drift) |
| **E3** | ⛔ **WP-0.3 — the fan readouts do not exist**: `fan_floor@k` (k = 1, 5, 10), fan-collision-vs-replay, DD's diversity D, the sampled-fan Kamm rate. Home: `taniteval/plan_fan.py` | Without them **WP-4 (sampler), WP-7 (selector) and WP-13 (RL) have no primary readout at all** — §6.1's whole point is that V2's gain was a *floor* effect a top-1 number cannot see |
| **E4** | ⛔ **`D-REFCV5-PLAN-7` is a PI decision**: LiDAR as inference input vs training-time teacher | It gates **E-LIDAR-1** only. It does **not** gate the LiDAR data build, E-BEV-1, or anything else — the plan is written so the ruling can arrive late |
| **E5** | **WP-DE-BEV-1 belongs to the DataFlyWheel**, with Arch+Inference as consumer. §3.3's seven steps and their content gates are the brief | The pilot (10 clips, dev box) turns every ESTIMATE in §3.3 into a MEASURED figure **before** any corpus build is scheduled. ⛔ Never on the training pod; ⛔ `df` cannot see the ~466 GB MooseFS quota |
| **E6** | **The §7 ladder should become `BACKLOG.md` pull-items**, so a gated turn can pull a rung | Ten of fourteen WPs are zero-GPU and independent — exactly the shape the ≥ 5-parallel-streams rule wants |

### Primaries this document needs but did NOT bank

⛔ **This stream did not run `tools/kb_add.py`** — the VLA/library stream owns the writer today, and
two agents writing `library.json` is how a generated index gets corrupted. Listed here so the banking
is a work item and not an omission:

| needed for | primary | arXiv |
|---|---|---|
| §3.4 E-BEV-1 (camera → BEV lift) | *Lift, Splat, Shoot* — Philion & Fidler 2020 | `2008.05711` |
| §3.4 E-BEV-1 (the simpler baseline lift) | *SimpleBEV* — Harley et al. 2022 | `2206.07959` |
| §3.3 (LiDAR encoder alternative to the histogram) | *PointPillars* — Lang et al. 2019 | `1812.05784` |

**Already banked and cited by this document** (library keys): TransFuser `2205.15997` · BEVFormer
`2203.17270` · DiffusionDrive `2411.15139` · DiffusionDriveV2 `2512.07745` · GTRS `2506.06664` ·
DriveSuprim `2506.06659` · DIVER `2507.04049` · DPPO `2409.00588`.

### Evidence this document rests on (all banked, none re-measured here)

| package | path |
|---|---|
| the REF-C vs DiffusionDrive audit (`D-REFC-DDAUDIT-1..6`, `H-DDA-1..4`) | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refc-vs-diffusiondrive-audit/` |
| the DiffusionDriveV2 code+paper analysis (`D-DDV2-*`, `H-DDA-5..7`) | `…/Architecture & Inference/Research/2026-09-05-diffusiondrive-v2-analysis/` |
| the ego-dropout burden on the trained model (`D-REFCV4B-EGODROP2`, `H-EGODROP-PRED`) | `…/Architecture & Inference/Research/2026-09-05-refcv4b-egodrop-burden/` |
| the ego-input literature panel (`H-EGO-LIT-1..4`) | `…/Architecture & Inference/Research/2026-09-05-ego-input-literature/` |
| RL readiness and the reward-floor defect (`D-RL-READY-1`, `D-RL-REWARD-FLOOR-1`) | `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/` |
| the nav-compliance metric (`D-NAVCOMP-1..3`) | `…/Benchmarks & Eval` stream + `taniteval/taniteval/nav_compliance.py`, `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.6.0 |
| the vocabulary and its gates (`D-REFCV4-VOCAB1`, `D-REFCV4-GATE1`, `D-REFCV4B-FLYLOW1`, `D-REFCV4B-TURNCOV1`) | `…/2026-09-04-refcv4b-vocabulary/`, `…/2026-09-04-refcv4-gate-validation/`, `…/2026-09-04-refcv4b-turn-coverage/` |
| the headline surface (`D-REFCV3-40284`, `D-REFCV3-40284b`, `D-REFCV3-AXIS1`) | `taniteval/results/refcv3-40284-openloop.ARM.json`, `…-stratified.json`, `MODEL_REGISTRY.md` §4.5 |
| the nav wiring design (E13–E21, S7/S8) | `Project Steering/DESIGN_REFCV4_NAV_WIRING.md`, `…/2026-09-04-refcv4b-seam-state/SEAM_STATE.md` |
| the PhysicalAI feature probe (LiDAR, `obstacle.offline`) | `…/Data Engineering/Implementation/incoming/2026-07-26-physicalai-feature-probe/` |
| the hierarchy eval pre-registration | `Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md` |

### Sibling streams this document is bound to

| stream | artifact | relation |
|---|---|---|
| **VLA extension** | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-extension-frontier/RESULT.md`; `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` | ⭐ §9 is **the contract**; that plan is the module. Where they disagree about a **port**, §9 wins and the disagreement is escalated; where they disagree about the **module**, that plan wins |
| **DataFlyWheel** | WP-DE-BEV-1 (§3.3) | owner of the LiDAR/BEV build; Arch+Inference is the consumer |
| **Deployment & Optimization** | `…/2026-09-05-refc-rl-readiness/` | owner of the RL vehicle (`rl_refcv3_min.py`, `launch_refcv3_rl_min.sh`); §6 re-scopes it as E-DDA-3b |
| **Benchmarks & Eval** | `nav_compliance.py`, `CRITERIA_REGISTRY.json` v2.6.0 | owns the compliance instrument every C7/C8 claim is read on |
| **Master Mind** | `MODEL_REGISTRY.md`, the launch decision | ⛔ **this plan launches nothing** |

### Honest scope

- **Nothing here is a result.** It is a design with pre-registrations; the first refcv5 number does not
  exist and will not until the ladder runs.
- **T1 is self-action open loop.** No claim in §8 is a *driving* claim; closed loop needs AlpaSim or a
  vehicle (the standing PI ruling).
- **Line numbers were read from the worktree on 2026-09-05 and drift.** Every reference also names its
  function or attribute; the function name is the stable one, and §9.8 says so explicitly for the
  contract.
- **Parameter deltas in §7.1 are ESTIMATED by arithmetic**, not counted by building. Each WP counts its
  own, and the preflight pins the total.
- **The `ha0_ext` clauses of §8 are claims we cannot yet read** (E2). Stating that is part of the
  pre-registration, not a footnote to it.

