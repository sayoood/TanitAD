# REF-C vs DiffusionDrive v1 / DiffusionDriveV2 — implementation audit

**Stream:** Architecture & Inference FlyWheel · **Date:** 2026-09-05 · **Branch:** `agent/arch-inf-20260803`
**PI question (verbatim):** *"review if we implemented all important parts like the truncated diffusion decoder, the generation of multimodal trajectories, transfuser, etc.. did we implemented the correct diffusion mechanisms. Review both v1 and v2 diffusiondrive papers, did we implement the cross attention mechanism?"*

**Deliverables (this directory):** `RESULT.md` (this file) · `refc_diffusion_mechanism_check.py` (the numerical check) · `raw/mechanism_check.json` + `raw/mechanism_check.log` (full run, 201 windows) · `raw/smoke.json` (2-window validation run) · `raw/sources.md` (exact paper versions, repo commits and files read).

**Sources (evidence class PUBLISHED, primaries banked):** DiffusionDrive v1 — arXiv **2411.15139v3** (library key `2411.15139`, sha256 `6ad4f8a3…`), repo `hustvl/DiffusionDrive` at commit **`9b52ed0ec06b073d82d6f392ab084c7b301c8681`** (2025-12-08); DiffusionDriveV2 — arXiv **2512.07745v1** (library key `2512.07745`, sha256 `076ce47e…`), repo `hustvl/DiffusionDriveV2` at commit **`1cd12a1e155c34dcc471261835444c8d5587580b`** (branch `master`, 2025-12-29). Files read are listed in `raw/sources.md`. Our side is **MEASURED**: source at the worktree (`stack/tanitad/refs/refc.py` md5 `54777902…`, `refc_v3.py` `815fdd9a…`, `refc_tactical.py` `df0d63ef…`, `models/kinematic.py` `2b0fa0da…`, `scripts/build_refc_anchors.py` `d00d4cb2…` — md5-identical between the G: worktree and the off-Drive clone) plus the numerical check on the shipped **refcv3 step-40,284** weights (`ckpt_step40284_frozen.pt` md5 `b1ed7075ff730d0993d2eaa3c86f6b56`, strict load `missing_keys: []`).

---

## 0. The answer in one paragraph

REF-C implements the **skeleton** of DiffusionDrive — a fixed anchor vocabulary, anchor-queries that cross-attend scene features, a per-anchor confidence + offset head, a small number of refinement passes with a timestep embedding, and argmax selection — but **not its diffusion mechanism**. There is **no noise schedule, no anchored Gaussian prior, no x0/DDIM update rule, no sampling at inference, and no per-step supervision**: `_decode` is called 1 + `steps` times with a constant-metres residual update `x ← x + off` (`refc.py:1596-1603`), a **0.1 m** jitter that exists only in training (measured: injecting it at eval moves the fan by **0.062 m** against **8.41 m** of deterministic offset — it is numerically decorative), and a timestep table of 3 rows that **saturates at index 2** for every extra pass. Iterating it beyond the two trained passes does not converge: the per-pass displacement decays by only ×0.72–0.85 per pass (5.68 → 4.28 → 3.30 → 2.82 → 2.04 m) while the fan's endpoint spread **collapses from 104 m to 42 m by 8 passes** — the mode collapse DiffusionDrive's truncation exists to prevent, controlled here by the pass count. **The cross-attention mechanism is implemented, but it is not the same one:** ours is a content-based `nn.MultiheadAttention` from 128 anchor queries to the 160 perspective-view conv-map tokens plus a FiLM on the measurement condition (`refc.py:1096-1113`); theirs is a **trajectory-coordinate-indexed BEV grid-sample attention** (sampling the BEV map *at the candidate's own waypoints*), followed by cross-attention to **30 detected-agent queries** and to the **ego planning query**, with the timestep entering as **per-layer AdaLN modulation** (`transfuser_model_v2.py::CustomTransformerDecoderLayer`, `modules/blocks.py::GridSampleCrossBEVAttention`). We have none of the three grounded attentions, because we have no BEV, no detector and no LiDAR — TransFuser cannot apply to a vision-only trunk as-is. The confidence/selection path is the most consequential documented divergence: DiffusionDrive scores **the fan it emits** (the last cascade layer's cls head), whereas our ranking reads the **t = 0 classifier confidence** and is therefore two passes stale — measured here as **`sel_idx_base` unchanged on 201/201 windows across `steps ∈ {0,1,2,3,4,8}`**. Everything V2 adds (intra-anchor GRPO, scale-adaptive multiplicative noise, inter-anchor truncated advantage, the coarse-to-fine metric selector) is **MISSING by construction**, and its prerequisite — a stochastic denoising policy with a log-probability — is absent from our decoder.

---

## 1. Component-by-component table

Status legend: **IMPLEMENTED** (same mechanism) · **PARTIAL** (present, materially different) · **DIVERGED** (present, different mechanism) · **MISSING** (absent) · **N/A** (absent on both sides).

### 1.1 Truncated diffusion

| # | component | DiffusionDrive v1 (PUBLISHED: paper §3–4; `navsim/agents/diffusiondrive/transfuser_model_v2.py` @ `9b52ed0`) | REF-C (MEASURED: worktree `refc.py` / trainers) | status |
|---|---|---|---|---|
| 1 | Anchored Gaussian prior | Eq. 4: τ_t^k = √ᾱ_t·a_k + √(1−ᾱ_t)·ε in **normalised** coordinates (`norm_odo`: x∈[−1.2, 55.7] m → [−1,1], y∈[−20, 26] m → [−1,1]); at inference fresh ε at **t = 8** (`forward_test`, `trunc_timesteps = 8`) → σ ≈ **0.90 m (x) / 0.73 m (y)** around each anchor (computed from diffusers `scaled_linear`, β 1e-4→0.02; `raw/sources.md`) | No Gaussian around the anchor. Train: `x = bank + offset` (classifier pass) then `x_in = x + N(0, 0.1 m)` **on the refined estimate**, constant σ at every pass (`refc.py:1598-1601`); eval: `torch.zeros_like` (`:1600`). Measured train σ = **0.0994 m** (cfg 0.1) | **DIVERGED** |
| 2 | Truncation timestep | Train: t ~ U[0, 50) of T=1000 (`forward_train`, `torch.randint(0, 50)`), i.e. the paper's "50/1000"; inference starts at t=8 with DDIM timesteps **[10, 0]** (`step_ratio = 20/2`) — note the paper/code asymmetry | No schedule; the analogue is the constant `noise_std = 0.1` (`DecoderConfig`, `refc.py:369`) | **MISSING** (no schedule to truncate) |
| 3 | Denoising iterations, train | **0 iterations**: one noised sample per example, one decoder call (2 cascade layers), no loop | **2 passes + 1 classifier pass** (`refc_v3_train.py:516`: `steps = diffusion_steps if mode == "diffusion"`; loop `refc.py:1597`) | **DIVERGED** (we iterate at train, they do not) |
| 4 | Denoising iterations, inference | **2** DDIM steps × 2 cascade layers = 4 refinement layers on 20 queries | **2** passes + classifier pass = 3 decoder calls × 4 cross-attn layers = 12 layer-evaluations on 128 queries; harness provenance `decoder_steps: 2` (`refcv3_arm.load_model`) | **PARTIAL** (count matches, mechanism does not) |
| 5 | Noise schedule | diffusers `DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample")` | none | **MISSING** |
| 6 | Prediction target | **x0** ("sample"): per-layer `poses_reg[..., :2] = delta + noisy_traj_points` (residual on the noisy input) | residual on the current estimate: `x = x_in + off` (`refc.py:1603`) — same *form*, but with no √ᾱ scaling the "noisy input" is just the previous estimate | **PARTIAL** |
| 7 | Timestep embedding | `SinusoidalPosEmb(d) → Linear → Mish → Linear` (`time_mlp`), injected **per layer** as AdaLN scale/shift after the FFN (`ModulationLayer`: `x·(1+scale)+shift`) | `nn.Embedding(diffusion_steps+1, d)` (`refc.py:1183`) added **once** as a bias to the input query (`:1376`), clamped `t_idx = min(i+1, diffusion_steps)` (`:1598`) → passes ≥ 2 reuse row 2 (measured trace for steps=8: `[0,1,2,2,2,2,2,2,2]`). Rows are distinct and live: forcing `t_idx = 0` at steps=2 costs **+1.48 m** ADE (SE 0.06, n=201) | **PARTIAL** |
| 8 | Loss per step | Per **cascade layer** (both layers' `(reg, cls)` summed; `forward_train`), one random t per sample; layer-to-layer coordinates **detached** (`traj_points = poses_reg[...,:2].clone().detach()`) | CE on the **t = 0** confidence only; L1 on the **final** fan at the assigned anchor (`refc_v3_train.py:551-553`) — gradient flows through the whole pass chain (no detach); the t ≥ 1 confidences (`refined_logits`) are **unsupervised** unless the D-SEL flags are on (off in refcv3/v4: `_v3_core_base`, `refc_v3.py:406-450`) | **DIVERGED** |
| 9 | Sampler | DDIM, η = 0 (deterministic given the initial noise), `set_timesteps(1000)`, final step → predicted x0 exactly | none (deterministic residual loop) | **MISSING** |
| 10 | Classifier-free guidance | not used (no guidance code in either repo) | not used | **N/A** |
| 11 | Coordinate normalisation | `norm_odo` / `denorm_odo` to [−1, 1], clamp after noising | raw metres throughout | **DIVERGED** (minor; matters only once a schedule exists) |
| 12 | Does `_decode` actually denoise? | — | It **refines**: `x_est` is updated between passes (measured per-pass displacement 5.68 / 4.28 / 3.30 / 2.82 / 2.04 m for passes 1/2/3/4/5–8); `_decode(t=0)` vs `_decode(t=2)` on identical `x` differ by **1.81 m** in offset and **0.50** in confidence logits. It does **not** denoise in the diffusion sense: nothing was noised at inference, and the loop is not a contraction to a fixed point (ratio ≈ 0.72–0.85 per pass; spread 104 → 42 m at 8 passes) | — |

### 1.2 Multimodal trajectory generation

| # | component | DiffusionDrive v1 | REF-C | status |
|---|---|---|---|---|
| 13 | Number of modes | 20 (NAVSIM) / 18 (nuScenes); `N_infer` may differ from `N_anchor` (Table 6: 10/20/40) | 128 (refcv3, refcv4), 117 (refcv4b, the 13 × 9 (a_lon, a_lat) grid); `N_infer ≡ N_anchor` | **PARTIAL** |
| 14 | Anchor construction | k-means over GT trajectories (`kmeans_navsim_traj_20.npy`, 20 × 8 × 2), fixed in metres | refcv3: FPS over a **synthetic** unicycle pool (`refc.default_anchors`; D-REFCV3-SMOOTH1); refcv4: k-means slot-normalised over 635,331 GT windows (D-REFCV4-VOCAB1); refcv4b: **v0-conditioned** constant-(accel, a_lat) controls rolled per window through `rollout_unicycle` (`refc.py:1318-1368`, `kinematic.py:220-266`; curvature `a_lat / max(v0, 4)²` clamped ±0.12) | **DIVERGED** (v4b goes beyond DD: a per-window kinematic vocabulary DD does not have) |
| 15 | How noise creates diversity | Each anchor is independently noised (σ ≈ 0.9/0.7 m) at inference → **sampled** fan; diversity score D (Eq. 3) 74 % vs 11 % vanilla | **No sampling at inference** (`refc.py:1600`); the fan is the vocabulary plus deterministic offsets; two calls are bit-identical (Part B: `eval_deterministic_steps2 = True`) | **MISSING** |
| 16 | Mode-collapse prevention | Truncation (start near anchors, only 2 steps) + per-mode cls | Nothing structural; **measured**: iterating collapses the fan (endpoint spread 103.7 → 81.7 (k=2) → 42.0 m (k=8)); at the trained k=2 the fan is still wide because there are only two passes | **DIVERGED** |
| 17 | Which modes are supervised | Only the anchor nearest the GT gets the L1 (`LossComputer`: `best_reg` gathered by `argmin` of **mean L2** to the clean anchors) | Same shape: L1 on the assigned anchor only (`refc_v3_train.py:552`), assignment by **sum of squared L2** over valid slots (`:544-548`) — a different metric, so a different anchor on some windows | **IMPLEMENTED** (assignment metric differs) |
| 18 | Where multimodality is lost in our path | — | (a) no sampling — one deterministic fan; (b) the emitted mode is decided by the t=0 classifier before any refinement (measured `sel_idx_base` change vs steps=0: **0/201**); (c) the reach clamp deletes 18–37 % of candidates (D-REFCV4-CLAMP1, D-REFCV3-SMOOTH6); (d) non-assigned anchors' offsets are unsupervised, so the mean classifier-pass offset is **8.41 m** — the fan is not "anchored" in the DD sense (their inference noise is < 1 m) | — |

### 1.3 Cross-attention (the PI's direct question)

| # | component | DiffusionDrive v1 (`CustomTransformerDecoderLayer.forward`) | REF-C (`CrossAttnLayer`, `refc.py:1096-1113`; decoder `forward`, `:1470-1511`) | status |
|---|---|---|---|---|
| 19 | Spatial attention to scene features | **Trajectory-coordinate-indexed**: `GridSampleCrossBEVAttention` bilinearly samples the BEV value map **at the 8 waypoints of each noisy trajectory** and mixes them with a softmax over points (`blocks.py:44-107`); each cascade layer re-samples at the updated points | Content-based `nn.MultiheadAttention(q, kv, kv)` from anchor queries to the **8 × 20 = 160 perspective-view tokens** of the last frame (`kv = feat_proj(fmap.flatten(2))`, `:1502`); the query carries the trajectory only through `traj_proj(x_est)` (`:1375`) — the attention never looks *where the candidate goes* | **DIVERGED** |
| 20 | Agent cross-attention | `cross_agent_attention(traj, agents_query, agents_query)` — 30 detection queries from the TransFuser decoder | none (no detector, no agent tokens) | **MISSING** |
| 21 | Ego / status cross-attention | `cross_ego_attention(traj, ego_query, ego_query)` — the TransFuser planning query that already attended BEV + `status_encoding` (command 4 + velocity 2 + acceleration 2) | The condition path is a **FiLM** on the MLP input: `cond = cond_proj(measurement)` (+ zero-init `ctx_to_cond`, `lan_to_cond`, target-latent FiLM) (`:1503-1509`, `:1112`); measurement = `[v0/10, nav one-hot(4)]` (`:2223-2245`) — no acceleration channel in refcv3; v4 adds the measured ego block to the hierarchy's goal path (E11') | **PARTIAL** (conditioning exists, mechanism differs) |
| 22 | Self-attention over candidates | none (commented out in their code) | none — and `SelectionConfig` relies on it: candidates are independent, which is what makes the S2b pre-filter exact (`refc.py:449-455`) | **IMPLEMENTED** (same absence) |
| 23 | Depth / width | 2 cascade layers, d=256, 8 heads, FFN 1024 | 4 layers (base/v3), d=384, 8 heads, ff_mult 4 (`DecoderConfig`, `:362-369`) | **PARTIAL** |
| 24 | Query initialisation | sinusoidal Fourier features of each waypoint (`gen_sineembed_for_position`, 64/pt → 512 → MLP) | `nn.Linear(S·2, d)` on raw metres (`:1181`) | **DIVERGED** (minor) |
| 25 | Timestep conditioning inside the layer | AdaLN modulation per layer | additive bias on the input query only (see #7) | **PARTIAL** |
| 26 | Map cross-attention | none in v1 (the paper's "agent/map queries" — the released NAVSIM code attends agents only; map enters as the BEV semantic aux) | none; no map exists in PhysicalAI-AV (CLAUDE.md: no map/lane graph in the 36 features) | **N/A** |

**Verdict on the PI's question:** yes, a cross-attention decoder is implemented (`CrossAttnLayer`), and it does what the docstring says — anchor queries attend a conv map under a FiLM condition. It is **not** DiffusionDrive's mechanism: theirs is grounded three ways (at the candidate's own waypoints in a metric BEV, on detected agents, on the ego planning query); ours attends ungrounded PV tokens and gets its ego/route information through a FiLM. Of the three grounded attentions we have zero.

### 1.4 Confidence / mode selection

| # | component | DiffusionDrive v1 / V2 | REF-C | status |
|---|---|---|---|---|
| 27 | Score head | v1: per-mode sigmoid, `bias_init_with_prob(0.01)`, trained with **focal BCE** (γ=2, α=0.25) on the one-hot nearest anchor (`multimodal_loss.py::LossComputer`), weights cls 10 / reg 8 | `conf_head = Linear(d, 1)` per anchor (`:1187`), **softmax CE** over N on the t=0 pass (`refc_v3_train.py:551`), weights traj 1.0 / cls 1.0 (refc-base `config.json` `loss_weights`) | **DIVERGED** |
| 28 | Object that is scored vs emitted | The **last** cascade layer scores the fan it emits; `argmax(poses_cls[-1])` gathers from `poses_reg[-1]` (`forward_test`) | Ranking reads the **t=0** confidence (`anchor_logits`); the emitted fan is two passes later. Measured: `sel_idx_base` identical to the steps=0 selection on **201/201** windows for every `steps`; the v3 goal-scorer graft (E9, `refc_v3.py:1020-1078`) changes the winner on 10.9 % of windows because it reads fan geometry. Documented in-source as D1/S1/S1b (`refc.py:1613-1636`; `refc_select.py:11-31`, the 45.4 % ranking failure on XL) | **DIVERGED** |
| 29 | Priors on the score | none in v1 | H19 `maneuver_to_anchor(log_softmax(man))` + factored `lat_to_anchor`/`lon_to_anchor` added in log-space (`:1571-1586`); LAN / goal geometric compatibilities (`_lan_anchor_prior`, `_goal_along_prior`); consequence gate (S3) | **beyond DD** (extra, gated) |
| 30 | Physical filter | none in v1; V2 penalises collision / off-drivable with advantage −1 and its selector scores NC/DAC/TTC/EP/C | S2 `reach_keep` bounded-acceleration band masks the argmax (`:1671-1686`; a = 2.0 at 6 s in refcv4, D-REFCV4-CLAMP1) | **PARTIAL** (different criterion) |
| 31 | V2 mode selector | Separate stage: coarse scorer (1 layer: BEV grid-sample + agent + ego attention) with five sub-metric heads (NC, EP, DAC, TTC, C) trained by BCE against PDM sub-scores + `MarginRankingLoss(margin 0.05)` ×2, top-32 → fine scorer (3 layers), argmax (`diffusiondrivev2_model_sel.py` @ `1cd12a1`) | none; the nearest thing is `refc_rescorer.py` (v1.2) and the E9 goal scorer | **MISSING** |

### 1.5 Encoder

| # | component | DiffusionDrive v1 (TransFuser, `transfuser_backbone.py`, `transfuser_config.py`) | REF-C | status |
|---|---|---|---|---|
| 32 | Inputs | Camera: 3 cams cropped + stitched to **1024 × 256**; LiDAR: BEV histogram **256 × 256 at 0.25 m/px** (±32 m); status vector 8-d | Camera only: 3-frame RGB stack (9 ch) at **256 × 640** (v3), window W=8 frames for the strategic GRU; `v0` (+ nav) | **MISSING** (LiDAR/BEV, by the vision-only programme rule) |
| 33 | Fusion | Two ResNet-34 (timm) trunks fused by GPT blocks at **4 scales** (avg-pooled to 8×32 image / 8×8 LiDAR tokens, `n_layer 2, n_head 4`), top-down decoder → `bev_feature_upscale` (metric BEV) | Single custom ResNet-34-style trunk (`ResNetEncoder`, `:1001-1058`; base_width 64/88, blocks (3,6,16,6)) → **8 × 20 PV** map; no lift to BEV | **MISSING** |
| 34 | Auxiliary heads | 3-D agent detection (30 boxes, Hungarian, w 10/1) + BEV semantic segmentation (7 classes, w 14) — these produce the agent queries the decoder attends | LAW latent (0.5 s-ahead pooled feature MSE, w 0.5), route head (w 0.1), maneuver / factored lat-lon heads (w 0.1), v3: goal cascade | **DIVERGED** |
| 35 | What is structurally lost | — | No metric frame for the decoder to index; no agents to attend; no BEV semantic supervision of the trunk. For the four families this is the LONGITUDINAL (headway / TTC to a lead agent) and TACTICAL grounding — exactly where `obstacle.offline` is read only pod-side (CLAUDE.md read-set table) | — |
| 36 | What V2 changed | Nothing: `diffusiondrivev2/transfuser_backbone.py` (19,898 B) vs v1 (19,896 B), same ResNet-34, config adds `num_groups: int = 4` only | — | — |

### 1.6 V2 additions

| # | component | DiffusionDriveV2 (paper §4; `diffusiondrivev2_model_rl.py`, `_rl_config.py` @ `1cd12a1`) | REF-C / `stack/tanitad/rl/` | status |
|---|---|---|---|---|
| 37 | Denoising policy with log-prob | `DDIMScheduler_with_logprob.step`: Gaussian per step, `std_dev_t_mul = clip(std_dev_t, min=0.1)`, `log_prob` summed over points; RL training runs **`step_num = 10`** steps, inference 2; `eta` 1 (train) / 0 (inference) | Absent: the decoder has no sampling distribution at inference and no log-probability (Part B: eval deterministic) | **MISSING (prerequisite)** |
| 38 | Scale-adaptive multiplicative noise | `prev_sample = prev_mean · (1 + ε_mul) + std_add · ε_add`, ε_mul two scalars per trajectory (longitudinal, lateral) | `rl/` documents the estimator; no decoder support | **MISSING** |
| 39 | Intra-anchor GRPO | G = `num_groups` = 4 samples per anchor; advantage = (r − mean_G) / (std_G + 1e-4); discount 0.8^(step_num−i−1); loss `L_RL + λ·L_IL`, λ = 0.1 (1.0 when a row has no positive sample) | `rl/advantage.py` implements the intra-anchor estimator (centre-only "Dr. GRPO" default) — critic-free, matches the paper's Eq. 7–8 in form | **PARTIAL** (estimator present, no policy to apply it to) |
| 40 | Inter-anchor truncated advantage | collision / off-drivable → −1; `advantages.clamp(min=0) · (reward > reward_gt − 1e-6)` — only samples at least as good as the GT's own PDM score get positive advantage | `rl/advantage.py` truncated advantage; `rl/rewards.py` rule-based components with `FORBIDDEN_REWARD_INPUTS` (selector disjointness) and a deliberate `HACKABLE_WEIGHTS` regression arm | **PARTIAL** |
| 41 | Reward | NAVSIM PDM sub-scores (NC, DAC, EP, TTC, C, driving direction) from the metric cache + PDM simulator in a process pool (`get_pdm_score_para`) | rule-based proxies (collision / feasibility / headway) — no PDM simulator on PhysicalAI | **DIVERGED** |
| 42 | Mode selector | see #31 | none | **MISSING** |
| 43 | What would be needed | — | (i) a stochastic decoder: anchored Gaussian start + a scheduler with a log-prob (#1, #5, #37); (ii) a reward instrument that is T1-admissible on PhysicalAI (`obstacle.offline` join for collision/TTC); (iii) the selector-disjointness audit already in `rl/audit.py`. RL-readiness is the sibling's deliverable (`stack/tanitad/rl/`, D-RL-REFCV3) — cited, not duplicated | — |

---

## 2. Numerical check of OUR mechanism (MEASURED, dev-box RTX 4060, 67 s)

**Setup.** `refc_diffusion_mechanism_check.py` — shipped **refcv3 @ 40,284** (`run_refcv3_viz/ckpt/ckpt_step40284_frozen.pt`, config rebuilt by the harness's own `refcv3_arm.load_model`, strict), the three local B1-v7.2 EVAL clips (`6ed4ef7a…`, `ca11a2a2…`, `d85682b8…`), stride 2 → **201 windows** (59 skipped: horizon beyond episode), `nav_cmd = None` (the harness's `os_navzero` conditioning), `v0` measured, each window encoded **once** (memoised encoder) so all decoder configurations are paired on bit-identical `kv`/`cond`. Code executed: the pod-identical tree (`run_refcv3_viz/repo`, `refc.py` md5 `b3365609…`); `_decode`, the denoise loop and `CrossAttnLayer` were **diff-verified identical** to the worktree before running.
**Estimator disclosure:** intervals below are **window-level paired SEs over overlapping windows from 3 clips** — a mechanism probe. They are **not** the decision-grade episode-cluster bootstrap and must not be quoted as benchmark numbers. `OMP_NUM_THREADS=6`.

| steps (`k`) | ADE 0–6 s (m) | ADE 0–2 s (m) | oracle-in-fan (m) | endpoint spread (m) | per-pass displacement (m) | `sel_idx` ≠ k=0 |
|---|---|---|---|---|---|---|
| 0 (classifier) | 6.415 | 3.184 | 3.942 | 103.7 | — | — |
| 1 | 5.153 | 2.108 | 3.061 | 92.3 | 5.683 | 14.4 % |
| **2 (shipped)** | **3.927** | **0.651** | **1.721** | **81.7** | **4.284** | 10.9 % |
| 3 | 3.992 | 0.979 | 1.784 | 72.5 | 3.301 | 11.9 % |
| 4 | 3.873 | 0.920 | 1.713 | 64.5 | 2.822 | 11.4 % |
| 8 | 3.727 | 0.901 | 1.697 | 42.0 | 2.036 | 12.4 % |

Paired deltas (mean, window-level SE, n = 201): ADE(k0) − ADE(k2) = **+2.488** (0.105) · ADE(k1) − ADE(k2) = **+1.226** (0.061) · ADE(k4) − ADE(k2) = −0.054 (0.049) · ADE(k8) − ADE(k2) = −0.200 (0.065) · oracle(k8) − oracle(k2) = −0.024 (0.070). Fraction of windows where k=8 beats k=2: **46.8 %** (k=4: 45.8 %); ADE 0–2 s **worsens** 0.651 → 0.901 m at k=8.

Answers to the seven questions the brief asks:

| Q | question | answer (MEASURED) |
|---|---|---|
| 1 | Is `x_est` updated between passes; how many passes run at inference? | Yes — per-pass displacement 5.68 / 4.28 / 3.30 / 2.82 / 2.04 m (passes 1 / 2 / 3 / 4 / 5–8). Inference runs **1 classifier pass + `diffusion_steps` = 2** refinement passes (harness provenance `decoder_steps: 2`, `decoder_mode: diffusion`); the same 2 at train (`refc_v3_train.py:516`). |
| 2 | Does the timestep embedding vary across passes? | Rows 0/1/2 are distinct (norms 20.18 / 19.96 / 19.44; pairwise L2 27.4 / 26.7 / 29.0) and **live**: forcing `t_idx = 0` on both refinement passes moves the fan by **1.86 m** and costs **+1.48 m** ADE (SE 0.06). But the table has only 3 rows: the trace for `steps = 8` is `[0, 1, 2, 2, 2, 2, 2, 2, 2]` — every pass beyond the second reuses row 2. |
| 3 | Does `_decode` at `t = 0` differ from `t = N − 1`? | On identical `x` and `kv`/`cond`: offsets differ by **1.81 m** (mean point-wise), confidences by **0.50** logits; the classifier-pass offset itself averages **8.41 m** — the head moves anchors by ~8 m, i.e. the vocabulary is not a tight prior. |
| 4 | Same noise schedule at train and inference? | **No schedule on either side.** Train: N(0, 0.1 m) added to the refined estimate at every pass (measured σ = 0.0994); inference: exactly zero. Injecting the train noise at eval moves the fan by **0.062 m** (0.7 % of the offset scale), ADE −0.005 (SE 0.006, n.s.), winner unchanged on 99.7 % of windows — the "diffusion noise" is numerically negligible. |
| 5 | Do more iterations move the fan toward the target? | From 0 → 2 passes, strongly (ADE −2.49 m, oracle −2.22 m). Beyond the trained 2, marginally at 0–6 s (−0.20 m at k=8, 47 % of windows) while the 0–2 s ADE **worsens** (+0.25 m): the extra passes reuse embedding row 2 and keep translating the fan. |
| 6 | Is the loop a contraction (DDIM-like convergence)? | **No.** Displacement decays ×0.75, ×0.77, ×0.85, ×0.72 per pass and is still 2.0 m/pass at pass 8; the endpoint spread collapses 104 → 42 m. It is an unrolled residual regressor, not a scheduled denoiser. |
| 7 | Is the ranking affected by refinement? | **No** — `sel_idx_base` (t=0 confidence + priors, reach-masked) is unchanged on **201/201** windows for every `steps`; only the v3 goal-scorer graft, which reads fan geometry, moves the winner (10.9–14.4 %). Selection is decided before refinement. |

Structural checks on a random-init `refc_smoke_config` model (Part B, CPU): eval deterministic at steps 0 and 2 ✔; `steps = 0` fan ≡ `bank + offset` ✔; `anchor_logits` identical across steps ✔ (`refined_logits` differ) ✔; train-mode noise σ = 0.103, mean 0.001 ✔; train mode non-deterministic ✔; `t_idx` trace for steps=4 `[0, 1, 2, 2, 2]` ✔.

---

## 3. Verdict per component

| component | verdict |
|---|---|
| Truncated diffusion decoder | **Not implemented as a diffusion model.** What exists is a 3-pass deterministic residual refiner with a 3-row timestep table and a train-only 0.1 m jitter. The docstring's "truncated diffusion" (`refc.py:43-45`) overstates the mechanism; the registry should call it what the measurement shows: an unrolled anchor-refinement regressor. |
| Multimodal generation | **Partial.** Multimodality is the vocabulary (fixed or, in v4b, v0-conditioned) plus deterministic refinement; no sampling, no diversity control; iterating collapses the fan. The v4b kinematic vocabulary is a genuine advance DD lacks — but it is a *prior*, not a generator. |
| Cross-attention | **Implemented, different.** Content attention to PV tokens + FiLM condition, vs DD's waypoint-indexed BEV sampling + agent + ego attention with per-layer AdaLN. |
| Confidence / selection | **Diverged, and measured to matter.** Scores the t=0 object, emits the 2-pass object; softmax CE vs focal BCE; extra priors and a physical filter DD does not have. |
| TransFuser | **Missing by construction** (vision-only trunk; no BEV/LiDAR/detector). V2 changed nothing here. |
| V2 (GRPO, multiplicative noise, selector) | **Missing**; prerequisite (a stochastic policy with log-prob) absent; estimators exist in `stack/tanitad/rl/`. |

---

## 4. Ranked divergences by expected impact on the four families, each with the cheapest discriminating experiment

Pre-registration rule: both outcomes are committed before the run; every arm reports the four families (LONGITUDINAL · LATERAL · TACTICAL · STRATEGIC) with the paired episode-cluster bootstrap on the same windows; every ladder run passes `TanitAD_ValidateAIDesign` (deliberate-regression arm + controls that must read known values).

| rank | divergence | families hit | cheapest discriminating experiment (pre-registered) | outcome A ⇒ | outcome B ⇒ |
|---|---|---|---|---|---|
| 1 | **Ungrounded attention** (#19–21): the decoder never looks at the scene *where a candidate goes*, sees no agents | LONGITUDINAL (headway/TTC), TACTICAL | **E-DDA-1 "waypoint-indexed PV sampling", v7-tiny ladder, 0-GPU-day.** Project each candidate's waypoints through the banked `camera_intrinsics`/`sensor_extrinsics` (already in the episode build) onto the 8 × 20 map, bilinear-sample, softmax-mix over slots (DD's `GridSampleCrossBEVAttention` in PV), add through a **zero-init gate** to the query (bit-identical at step 0). Control: same parameter count, random sampling locations. Readouts: oracle-in-fan, selection gap, lead-block TTC/headway on the `obstacle.offline` join | grounded arm beats the random-location control on oracle-in-fan AND the longitudinal family ⇒ grounding is the lever; schedule a base-size run | no separation ⇒ PV tokens already carry the geometry; move rank 2 up and do not spend on BEV lifting yet |
| 2 | **Stale ranking** (#28): the t=0 confidence ranks a fan it never saw; refined confidences unsupervised | all four (selection is where 0.0751 m of oracle gap sits, D-REFCV3-40284a; 45.4 % ranking failure on XL) | **E-DDA-2 "score the emitted fan", 0 parameters.** Flags already exist: `sel_refined + sel_score_emitted + sel_ce_reach` (`refc.py:525-560`). v7-tiny ladder, paired against the all-off control; E-SEL-0 already showed the *unsupervised* refined readout ranks worse, so the arm is the *supervised* one | selection gap closes on the ladder ⇒ turn the flags on in refcv4c | no change ⇒ selection is not the binding defect at this fan quality; bank and stop re-litigating S1 |
| 3 | **No anchored Gaussian / no sampling** (#1, #5, #15): deterministic fan, decorative 0.1 m jitter, diversity collapses with passes | TACTICAL (mode coverage), and the whole V2 path (GRPO needs a log-prob) | **E-DDA-3 "DD-faithful sampler", ~1 dev-box hour.** Add `diffusers.DDIMScheduler(1000, scaled_linear, prediction_type="sample")`, normalise to the corpus ranges, train t ~ U[0, 50), infer from t = 8 with 2 DDIM steps, x0 prediction — behind a config flag so the classifier path stays byte-identical. Readouts: oracle-in-fan at equal N, the DD diversity score D (Eq. 3 of 2411.15139) on 8 samples, and the four families | oracle-in-fan improves at equal N or D rises without ADE loss ⇒ sampling adds coverage the vocabulary lacks; this is also the prerequisite for GRPO | no gain ⇒ on this corpus the vocabulary ceiling (D-REFCV4-VOCAB1) binds, not sampling; keep the deterministic refiner and rename it honestly |
| 4 | **The anchors are not a prior** (#14, #18d): mean classifier-pass offset 8.41 m on refcv3 | LONGITUDINAL (D-REFCV3-AXIS1: the deficit is along-track) | **E-DDA-4, zero cost — a readout from the running refcv4b dump.** Report `‖offset‖` mean and the oracle-in-*bank* (pre-offset) vs oracle-in-fan at 40 k | offset mean < 1 m and oracle-in-bank ≈ oracle-in-fan ⇒ v4b's v0-conditioned vocabulary is a real prior; DD's premise now holds | offset still metres-scale ⇒ the head is doing the vocabulary's job; revisit N and the (a_lon, a_lat) grid density before anything else |
| 5 | **Loss form** (#8, #17, #27): softmax CE over 128 vs focal BCE over 20; sum-of-squares vs mean-L2 assignment; no heading channel | LATERAL (heading), TACTICAL (calibration) | **E-DDA-5, v7-tiny ladder, +0 / +2·d params.** Arms: (a) focal BCE (γ 2, α 0.25) in place of CE; (b) heading channel on `offset_head` with the `tanh·π` decode; control: CE. Readouts: heading error, curvature error, anchor accuracy, calibration (ECE of the winner) | either arm separates on its family ⇒ adopt | no separation ⇒ record as N/A on this corpus |
| 6 | **Timestep conditioning** (#7, #25): additive input bias, table saturates at row 2 | none measurable now (the embedding is live: −1.48 m if removed) | none — bank the measurement; if E-DDA-3 lands, the AdaLN form comes with it | — | — |
| 7 | **TransFuser / BEV** (#32–35): structural, vision-only rule | LONGITUDINAL, TACTICAL | not a cheap experiment; the vision-only analogue (LSS/BEVFormer-style lift) is a programme decision. Prerequisite readout is E-DDA-1: if PV grounding already separates, BEV lifting is the next lever; if not, it is not | — | — |
| 8 | **V2 stack** (#37–43): GRPO, multiplicative noise, selector | all (V2's PDMS@10 floor +9.1 vs v1) | gated on E-DDA-3 (a policy with a log-prob) and on a T1-admissible reward; the estimators and the reward audit exist in `stack/tanitad/rl/` (sibling's RL-readiness assessment) | — | — |

---

## 5. What this audit does NOT claim

* No benchmark number here is registry-grade: n = 3 clips, window-level SEs, `nav_cmd = None`. The mechanism conclusions (no schedule, no sampling, stale ranking, non-contractive loop, saturating timestep table) are **source-level facts** confirmed by the run, and do not depend on the estimator.
* DiffusionDrive's own code/paper asymmetries are reported as found (train t ∈ [0, 50), inference from t = 8 with steps [10, 0]); I did not run their code.
* The V2 RL code (`diffusiondrivev2_model_rl.py`, 52 KB) was read as extracted excerpts, not in full; the config and `blocks.py` were read verbatim (`raw/sources.md`).
* `refcv4b-b1-v72-40k` is live and was not touched; the check ran on the frozen refcv3 copy on the dev box.

---

## 6. Deliverable manifest

| artifact | where it lives |
|---|---|
| this report | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refc-vs-diffusiondrive-audit/RESULT.md` (repo, staged) |
| numerical check script | `…/2026-09-05-refc-vs-diffusiondrive-audit/refc_diffusion_mechanism_check.py` (repo, staged); working copy `C:\Users\Admin\refc_dd_audit\` (dev box) |
| full run (201 windows) | `…/raw/mechanism_check.json`, `…/raw/mechanism_check.log` (repo, staged) |
| validation run (2 windows) | `…/raw/smoke.json` (repo, staged) |
| sources note | `…/raw/sources.md` (repo, staged) |
| claims register rows | `Project Steering/GOALS_AND_CLAIMS.md` — section `D-REFC-DDAUDIT` (repo, staged) |
| library citations | `TanitAD Research Lab/Library/library.json` — `cited_by` added to keys `2411.15139`, `2512.07745` via `tools/kb_add.py` (no re-download; both were already banked) |

**Integration requests (escalated, not written into a doc):** (1) `MODEL_REGISTRY.md` §4.5/4.6 should describe the refcv3/refcv4 decoder as an *unrolled anchor-refinement regressor (3 passes, no schedule, no sampling)* rather than "truncated diffusion" — the registry is the quotable source and the current wording invites the comparison this audit refutes; (2) E-DDA-4 is a free readout from the refcv4b dump when it lands — please add it to that eval's work list; (3) E-DDA-1/2/3 are ladder-sized and need a slot on the dev box, not a pod.
