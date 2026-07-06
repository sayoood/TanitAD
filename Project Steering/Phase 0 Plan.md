# TanitAD — Phase 0 Plan (v1.0, 2026-07-05)

**Duration.** 6 weeks: 2026-07-06 → 2026-08-15.
**Constitution scope.** "Phase 0 includes a running architecture with the most important hypotheses:
4B architecture, world-model approach, real existing dataset, ideally open loop and closed loop, single
front camera with good performance including reasoning, and the MoE-tactical steering of additional
sensors." This plan operationalizes exactly that.

---

## 1. Reframed Phase 0 goals (measurable)

| # | Goal (reframed) | Measured by |
|---|---|---|
| G0.1 | A **running 4B latent world-model stack** (operative + tactical imagine-and-select + strategic graph + fallback monitor) trained self-supervised on driving data | training runs healthy (no collapse, `sig`≈1.1, effective rank > 8), end-to-end demo videos |
| G0.2 | **Imagination you can trust**: action-conditioned latent prediction decodable to metric trajectories via frozen calibrated probes | Gates D1–D3 |
| G0.3 | **The hierarchy edge on driving topology** at matched parameters | Gates D5–D6 |
| G0.4 | **Open-loop competence on real data** (single front camera) | ADE/FDE@1s/2s on comma2k19 held-out routes vs persistence + supervised baseline |
| G0.5 | **First closed-loop driving** in MetaDrive (procedural maps) | route completion / success rate + solve videos |
| G0.6 | **Self-monitoring baseline** (imagination-error OOD) + custom metric suite live | D8 baseline AUROC; LAL/TMS/OKRI/CNCE/LOPS implemented |
| G0.7 | **Modality-steering mechanism demonstrated** (Phase-0 exit demo, not full ABMS) | tactical selects {front / +side / +rear} on multi-view clips; quality-vs-FLOPs Pareto plot |
| G0.8 | **Honest efficiency ledger** from day 1 | params, FLOPs/decision, batch-1 latency (RTX 4060; Orin projection) per experiment |

Non-goals of Phase 0 (explicitly deferred): NAVSIM leaderboard entry, RMFM alignment, LLM bridge,
latent-RAG retrieval in the control path, radar/lidar fusion, on-target Orin deployment.

## 2. Decision recommendations (the ones you asked for)

### 2.1 Model architecture — **decided recommendation** *(updated 2026-07-06 per D-008: ≥250 M)*

**Main track (arm A): TanitAD-4B-M, from-scratch hierarchical latent world model, 261 M params
(measured at instantiation; enforced by `tests/test_imagination.py::test_base250_parameter_budget`).**

| Brain / component | Config (`base250_config`) | Params (measured) |
|---|---|---|
| Operative — perception (ViT encoder) | d 768 × 14 blocks, patch 16, batch-free norms | 99.5 M |
| Operative — dynamics (causal predictor) | d 768 × 12 blocks, FiLM actions, horizons 1/2/4 | 107.7 M |
| Tactical — maneuver-horizon predictor | d 512 × 6 blocks, horizons 8/16 (MoE upgrade WP4: 8 experts top-2, +~40 M total at equal *active* params) | 26.5 M |
| H15 — imagination field | advection flow + 3 refine blocks d 768 + σ head | 22.1 M |
| Grounding — inverse dynamics | MLP (2·state → 1024 → action) | 5.2 M |
| Readout (spatial grid 4×4, d_r 128) | linear projection | 0.1 M |
| Strategic — VQ(256) + latent transition graph | **non-parametric by design** | ~0.1 M |
| Fallback — monitors | out-of-gradient | 0 |
| **Total** | | **261.1 M** |

Inference-rate layering keeps the efficiency moat despite the size: operative path (207 M) at 10–20 Hz,
tactical (26 M) at 1–2 Hz, H15 heads only where sectors are gated, strategic at 0.1 Hz (graph lookup,
µs). The Phase-1 LLM bridge (frozen 1 B, ~20 M trainable adapters) sits outside this budget.

Detailed component rationale (unchanged from kickoff):

| Component | Choice | Why / provenance |
|---|---|---|
| Encoder | ViT patch 16, 2-frame tubelets, **batch-free norms** (LayerNorm only) | ALPS-4B validated; patch 8 measured worse; batch-free norms guarantee I2 |
| SSL objective | LeJEPA: SIGReg (Epps–Pulley, 512 slices, λ=0.1) on embeddings AND predictions, all levels; no EMA/stop-grad crutches | A1 + LeJEPA theory; single hyperparameter |
| Operative predictor | causal transformer, window 6–8, depth 8, action-conditioned (FiLM on continuous (steer, accel)), **residual/delta prediction + change-weighted latent loss**, multi-horizon heads k∈{1,2,4} | A4 bake-off winner (0.97 vs 0.71 MSE vs 0.44 flow); MTP = free speedup+signal (H5) |
| Grounding | inverse-dynamics regression head (z_t, z_{t+1}) → (steer, accel) | A5; ego-motion is free proprioception; seed of the H7 IDM |
| Readout | spatial grid readout (never global pooling) + **frozen probes calibrated on imagined latents** (probe_real + probe_imag pair) | A7 + A3 (0.97 vs 0.66) — our answer to trajectory decode |
| Trajectory head | probe → waypoints at 0.5/1/1.5/2 s **through a differentiable kinematic bicycle layer**; WTA loss when multi-modal (K=6) | H14 Track 1; 100 % physically realizable outputs; PKQT/Deep-Think-13 design |
| Tactical | discrete maneuver vocabulary (9–15 bins: lane-keep, follow, creep, turn-L/R, …) as MoE experts; imagine-and-select: imagine each maneuver's post-latent, decode with probe_imag, score = progress + safety (TTC-proxy) + comfort | §6 of AD_TRANSFER_RESEARCH; World4Drive pattern, hierarchical + label-free |
| Strategic | VQ codes (64–128) over pooled latents + latent transition graph, shortest-path routing; re-route on imagination-plan divergence | A6 (+58 % topology edge); port from ALPS-4B |
| Fallback (brain 4) | imagination-error monitor (A9) + collapse watchdog (rank/variance/NaN) → deterministic MRC hook (brake profile in sim) | free; regulation-aligned (ISMR trigger substrate) |
| Command input | discrete navigation command embedding (left/straight/right/goal-node) via FiLM | H12 minimal form |

**Comparison track (arm B): same predictor/tactical/strategic stack on a frozen DINOv3-S/DINOv2-S
encoder (+LoRA r=16 later).** Answers H4 at every gate. Decision between arms only on measured gates.

**Explicitly rejected for Phase 0:** pixel/diffusion world models (compute, planning latency), VLA-style
language-core (H12 stance), CEM/continuous planners (discrete tactical vocabulary is milliseconds),
BatchNorm anywhere in the inference path (I2).

### 2.2 Training data — **exact specification** *(updated 2026-07-06 per D-008; ordering updated per D-009)*

> **D-009 (Sayed): real camera data first.** B1 (comma2k19, HF mirror `commaai/comma2k19`) is the
> bootstrap corpus from day 1 under config `base250cam` (6-ch 2-frame RGB @ 256 px).
> A1 (toy) is a CI fixture only — zero training time. A2 (MetaDrive) remains required for the
> closed-loop gates D5/D6 and scripted-occluder LOPS, scheduled after the supervised source install.
>
> **D-012 update (Sayed): the FIRST RICH corpus is PhysicalAI-AV**, used now with tagged exposure
> and the license resolved later — comma2k19 is highway-only and stays the license-clean anchor for
> public numbers. Full corpus roles, staged ingestion (R0 urban starter 500 clips → R1 2 000 clips →
> R2 multi-view), composition targets (~60/25/15) and the license-management plan:
> **`DataEng/DATA_STRATEGY.md`** (supersedes the volume figures in the table below where they differ).

| # | Dataset | Exact spec | Volume | Role / gates |
|---|---|---|---|---|
| A1 | TanitAD driving toy (in repo, zero-dep) | 1 000 episodes × 300 steps, ego-centric BEV 128×128, bicycle-model actions | 300 k frames | pipeline + H15 rehearsal; runs TODAY on pod & local |
| A2 | **MetaDrive** (adapter merged; live rollout after supervised source install — PyPI no-go on py3.13) | train: 1 000 procedural maps (3-block), eval: 100 unseen (3–7 block) + blocked-route variants; BEV 128² + front cam 224² @ 10 Hz; continuous (steer, accel) | ~500 k frames ≈ 14 h | D1–D6, D9 (scripted occluders), closed-loop G0.5 |
| B1 | **comma2k19** — full 33 h | 10 Hz resample; front cam center-crop → 224²; actions: yaw-rate from CAN steering + accel from speed derivative; pose targets from GNSS; **split by chunk: 1–8 train, 9 val, 10 test (I3)** | 33 h | G0.4 real-data open-loop; the data-efficiency anchor |
| B2 | **PhysicalAI-AV** (loader in `DataEng/AVDataSetLoader`; needs rotated HF token; license review before public claims) | 2 000 front-wide 20 s clips (diversity aug) + 500 multi-view clips (front+L+R+rear) reserved for the G0.7 modality demo; egomotion → actions | ~11 h + demo set | Stage-B diversity; H2 demo |
| P | nuScenes-mini | 10 scenes, never trained on | ~20 min | D8 OOD probes only |

**Total trained volume Phase 0: ~14 h sim + ~44 h real** — the "tens of hours" data-efficiency claim
stays intact at 261 M params. Own GoPro/smartphone data and YouTube corpora: Phase 1 (H7 IDM pipeline).

Own smartphone/GoPro data, YouTube/dashcam corpora: Phase 1 (H7 pipeline). License review of
PhysicalAI-AV terms before any public claim (open task, DataEng agent).

### 2.3 Eval metrics — **decided recommendation**

1. **Standard (recognizable):** ADE/FDE @1/2 s open-loop (route-level splits only); MetaDrive closed-loop
   success / route completion / infractions; later NAVSIM PDMS-style sub-scores.
2. **Baselines every table must contain:** persistence, supervised-at-matched-data, (arm A vs arm B).
3. **Instrument rows first (D-004):** I1 oracle-decode, I2 batch-consistency, I3 route-splits, I4
   `imag relative` < 1.
4. **Custom TanitAD metrics (Deep Think 14):** LAL (anticipation latency), TMS (maneuver stability),
   OKRI (occluded-kinematic risk), CNCE (compute-normalized causal efficacy), LOPS (object permanence).
5. **Safety/compliance:** rule-violation rate (stop line, speed, collision) on decoded trajectories;
   friction-circle violation rate; D8 OOD AUROC.

### 2.4 Compute — **decided recommendation** *(updated per D-008: 261 M model)*

RTX 4060 (8 GB): CI, smoke tests, and 261 M pipeline-debug runs at batch ≤ 8 / 128 px only — NOT for
real training at this scale. **A40 48 GB (RunPod) is the Phase 0 training workhorse**: Stage-A run
(A1+A2 data, 60 k steps, batch 64) ≈ 12–24 h ≈ $10–20 at $0.40–0.85/h — inside the $50/wk guardrail
(runbook: `stack/RUNPOD_RUNBOOK.md`). A100 80 GB only if Stage-B 224² multi-view memory demands it.
Every run logged in the experiment record format (protocol §6) + RESOURCE_LEDGER row before launch.

## 3. Work packages

- **WP1 stack scaffold** — `stack/` package: configs, encoder, SIGReg, operative predictor, inverse
  dynamics, readout/probes, kinematic head, instruments, toy data, training loop, tests. *(started at kickoff)*
- **WP2 driving-toy pipeline** — MetaDrive wrapper with Two-Rooms observation contract; synthetic
  kinematic BEV toy (zero-dep) for CI. Gate rehearsal D1–D3 at toy scale on the 4060.
- **WP3 bake-offs** — loss (residual+change-weighted vs MSE vs ego-compensated), readout (grid vs pool),
  probe (imag vs real calibration). One lever per run.
- **WP4 hierarchy** — tactical vocabulary + imagine-and-select; strategic VQ graph; blocked-route tasks;
  D4–D6 at matched params vs flat.
- **WP5 real data** — comma2k19 ingestion (route-level splits), Stage-B training, G0.4 numbers;
  PhysicalAI-AV front-cam subset.
- **WP6 eval suite** — metric implementations + gate runner + first `Benchmarks & Eval/LEADERBOARD.md`.
- **WP7 monitoring** — imagination-error monitor + D8 harness (in-dist vs unseen-town/weather).
- **WP8 modality-steering demo** — G0.7 Pareto experiment on multi-view clips.
- **WP9 phase report** — gate table, honest verdicts, Phase 1 go/no-go.

## 4. Gates (falsifiable, with instrument rows)

| Gate | Claim | Threshold | Ablation |
|---|---|---|---|
| D1 | encoder state decodable | frozen-probe ADE@1s < 0.5 m (BEV) / < 1.0 m (camera); I2, I3 pass | vs global-pool |
| D2 | imagination usable for selection | calibrated direction acc > 0.7; imag-rel < 0.8; I1 ≈ 1.0 first | vs persistence; vs flow head |
| D3 | trajectory decode from imagination | imagined-ADE@2s ≤ 1.5× oracle-decode ADE@2s | probe_real vs probe_imag |
| D4 | tactical beats greedy | +15 % success on interactive scenarios | tactical off |
| D5 | strategic routing beats greedy on topology | blocked-route success: 4B ≫ operative (≥ +30 % abs) | graph off / random waypoints |
| D6 | hierarchy generalizes simple→complex | success-degradation slope 4B < flat at matched params (20 train maps → 100 unseen, 3→7 blocks) | flat matched-params |
| D7 | memory helps rare scenarios *(stretch)* | repeat-exposure improvement, no interference regression | RAG off |
| D8 | monitor detects OOD | AUROC > 0.85 (unseen town/weather) | vs Mahalanobis-encoder baseline |
| D9 | **H15: imagination in unobserved areas** (added per D-008) | hidden-sector cosine ≥ shuffled baseline + 0.2; calibration gap > 0 (σ higher where blind); object-level LOPS uplift vs no-advection on scripted occluders | advection off; refine blocks off |

**Program rule:** no architecture change may be motivated by a gate that hasn't passed its instrument rows.

## 5. Baselines to beat / compare (Phase 0 realism)

- Persistence + constant-velocity (floor). — Supervised BC at matched data/params (the honest E2E
  reference). — Arm B frozen-encoder (H4). — Published context numbers (LAW-class ADE, MetaDrive PPO
  reference) quoted, not re-run, in Phase 0.

## 6. Week-by-week

| Week (2026) | Focus | Exit |
|---|---|---|
| W28 (Jul 6–12) | WP1 finish + WP2 toy pipeline + CI green (I1–I4) | smoke train on 4060; toy D1–D3 rehearsal runs |
| W29 | WP3 bake-offs on MetaDrive BEV (A40 if needed) | D1–D3 measured; A3/A4 orderings reproduce on driving |
| W30 | WP4 tactical+strategic; blocked-route tasks | D4/D5 first numbers; solve videos |
| W31 | WP4 D6 ladder + WP7 monitor | D6 slope + D8 baseline |
| W32 | WP5 comma2k19 Stage-B + front-camera MetaDrive | G0.4 first real-data numbers |
| W33 (–Aug 15) | WP8 modality demo + WP6 leaderboard + WP9 report | Phase 0 Report + go/no-go |

## 7. Definition of Done (Phase 0)

1. D1–D3 pass on MetaDrive; D5 or D6 shows the hierarchy edge; D8 baseline measured.
2. Real-data (comma2k19) open-loop table published with all baselines and instrument rows.
3. Closed-loop MetaDrive demo videos; efficiency ledger for every experiment.
4. G0.7 modality-steering Pareto demo done.
5. Phase 0 Report written; PROJECT_STATE and ledgers current; all pushed to GitHub.
