# TanitAD — MODEL REGISTRY

> **Purpose (Sayed, 2026-07-20):** *"be able to reconstruct any of our major model versions from a code
> point of view and from a documentation point of view."*
>
> **The acceptance test for every row below:** a reader with this repo + the pods can rebuild that exact
> model — architecture, params, training command, data, code state — and knows what it scored and why it
> exists. Where that chain is broken, the row says so in a **RECONSTRUCTION RISK** block rather than
> pretending.
>
> **Compiled:** 2026-07-20 · **Method:** every architecture fact read from `stack/tanitad/` source; every
> training command read from the run's own `config.json` / run-manifest / live `ps` on the pod; every ADE
> read from the raw eval JSON on `tanitad-eval`, not from prose. Pods were read-only.
>
> Companion docs: `Project Steering/PROGRAM_OVERVIEW.md` (whole-program briefing) · `DECISIONS.md` (ADR
> log) · `TanitAD Research Hub/HYPOTHESIS_LEDGER.md` (H-numbers) · `Paper/TANITAD_PAPER.md`.

**Verification legend**

| Mark | Meaning |
|---|---|
| ✅ | Verified this session against source / run config / raw result JSON |
| ⚠️ | Verified but carries a caveat stated inline |
| 🟥 **UNVERIFIED** | Could not be confirmed from a primary artifact — do not rely on it for a rebuild |

---

## 0. Shared substrate — true for every row unless a row overrides it

### 0.1 Training corpus and parity

| Item | Value | Source |
|---|---|---|
| Corpus | **NVIDIA PhysicalAI-AV**, front-wide camera, **2,376 episodes** (train split only) | run logs (`[refa+] … 2376 eps / 406099 windows`), cache-guard `(2376 files)` ✅ |
| Strict-parity build key | **`physicalai-train-e438721ae894`** | `/workspace/data/physicalai_phase0/PARITY_OK`; `refb_pipeline.sh: EXPECT_KEY`; `stack/tanitad/lake/filtering.py:60 STRICT_PARITY_BUILD_KEY` ✅ |
| Corrupt-clip skip-hash | **`f09e44db`** — 24 corrupt front-wide clips excluded | `stack/tanitad/lake/filtering.py:59 PARITY_SKIP_KEY`; `TANITDATASET_V1_STRATEGY.md` ✅ |
| Cache paths | pod1/pod2 `/workspace/data/physicalai_phase0/_epcache` · pod3 `/workspace/pai_epcache` | run configs ✅ |
| Episode contract | `ep_*.pt` per episode: `frames_u8 [T,9,256,256] uint8` (3 RGB frames at 100 ms spacing, channel-stacked — D-015), `actions [T,2]` = (steer, accel), `poses [T,4]` = (x, y, yaw, v) | `taniteval/registry.py` CORPORA note; `stack/tanitad/data/` ✅ |
| Rebuild-from-origin | `stack/scripts/build_pai_cache.py`, `stack/scripts/physicalai_r0.py fetch-camera`, `stack/scripts/rebuild_pai_rolling.py --expect-key e438721ae894 --skip-idx …`; full chained supervisor at `/workspace/refb_pipeline.sh` (pod1) | ✅ |
| Parity gate | `refb_pipeline.sh` refuses to launch training unless the build reproduces `EXPECT_KEY` with zero per-clip-intrinsics fallbacks and no disk-guard firing | ✅ |

**Known data caveat (memory-of-record):** PhysicalAI front-wide contains **two camera rigs** (cy ≈ 543 rig A,
cy ≈ 755 rig B). The phase-0 cache crops around the per-clip principal point. Any rebuild that uses a
geometric-center crop will be ~215 px off for rig B. ⚠️

### 0.2 REF-A feature cache (frozen-encoder arms only)

| Item | Value |
|---|---|
| Path (pod3) | `/root/phase0_dinofeats/` → `physicalai-train-e438721ae894-dinov2-b14`, `physicalai-val-0c5f7dac3b11-dinov2-b14` ✅ |
| `META.json` | `{"encoder":"dinov2-b14","size":224,"grid":"16x16","dim":768,"note":"latest-frame features; 3-frame windows reconstructed at train time from consecutive rows"}` ✅ |
| Builder | `stack/scripts/dino_precompute.py` — tries `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated) first, **falls back to `dinov2_vitb14` via torch.hub** and records which ran. Phase-0 ran the **fallback**. ✅ |
| I-JEPA variant | `/workspace/tmp/ijepa_feats` (pod3), `d_dino=1280`, frozen I-JEPA ViT-H/14 ✅ |
| 320-ep variant | `/workspace/tmp/dino_feats_320` (pod3), `d_dino=768` ✅ |

### 0.3 Evaluation substrate — **TanitEval**

| Item | Value |
|---|---|
| Location | `/root/taniteval/` on **`tanitad-eval`** (A40). **NOT in this repo.** ✅ |
| Val set | `physicalai-val-0c5f7dac3b11` — **40 episodes → 881 windows**, episode-disjoint from train ✅ |
| Protocol | window 8, stride 8, K = 20 steps @ 10 Hz, waypoints `[5,10,15,20]` = 0.5/1/1.5/2 s, metric-BEV ego frame, `nav=follow`, operative step **intent-free** ✅ |
| Statistic | **8-split episode-disjoint jackknife**, `val_frac 0.2` → `heldout mean ± CI95`. `full_set` = plain mean over all 881. **Both are published; they differ. Always name which.** ⚠️ |
| Trivial floor | **CV ADE@2s = 0.8248 heldout / 0.8377 full-set**; CTRV oracle 0.523; best-of-3 kinematic floor 0.5005; learned ego-status (no-vision) ceiling 0.5735 ✅ |
| Invocation | `python3 -m taniteval.runner run --model <key> --episodes 40` → `results/<key>.json`; also `ab`, `imagination`, `hierarchy`, `report`; `python3 -m taniteval.closedloop --arm <key>`; `python3 -m taniteval.planner_p2 --arm <key>` ✅ |
| Model registry | `/root/taniteval/taniteval/registry.py` — the mapping from arm key → checkpoint path → arch flags. **This file is the eval-side twin of this document.** ✅ |

> 🟥 **RECONSTRUCTION RISK — TanitEval is uncommitted.** Every headline ADE in this registry was produced
> by code that exists only on `tanitad-eval:/root/taniteval`. If that pod is lost, the numbers become
> unreproducible even though the checkpoints survive. The in-repo evaluators
> (`stack/scripts/evaluate_checkpoint.py`, `eval_grounded_rollout_4b.py`, `eval_metric_rollout.py`,
> `compare_arms.py`) implement the **older camera-frame D1/D2/D3 gate**, which is *not* the same metric —
> `LEADERBOARD.md`'s newest row is still camera-frame ADE@1s @27k and is stale.

### 0.4 In-repo reference implementations (the code side of the acceptance test)

| Component | File |
|---|---|
| All flagship/REF-A architecture presets | `stack/tanitad/config.py` |
| 4-brain assembly (encoder → operative → tactical → strategic + grounding) | `stack/tanitad/models/fourbrain.py` |
| Flagship loss + grounding + `v0` speed channel | `stack/tanitad/train/flagship_losses.py` (`v0 = pose_last[:,3]/10.0`, **SPEED_SCALE = 10.0**) ✅ |
| REF-A frozen-DINO adapter + predictor | `stack/tanitad/refs/refa.py`, `stack/experiments/reset-speed4b/refa_plus.py` |
| REF-B end-to-end BC stack | `stack/tanitad/refs/refb.py` |
| REF-C anchored-diffusion stack | `stack/tanitad/refs/refc.py` |
| Label derivation (maneuver / nav / route / path targets, v1 + v2) | `stack/scripts/refb_labels.py` |
| Trainers | `stack/scripts/train_flagship4b.py`, `refa_train.py`, `refa_train4b.py`, `refb_train.py`, `refc_train.py`, `stack/experiments/reset-speed4b/refa_train_plus.py` |

---

## 1. FLAGSHIP — 4-brain latent world model, trained ViT encoder

All five versions share `--config flagship4b` → `flagship4b_config()` (`stack/tanitad/config.py:307`):

```
encoder            ViT  in_ch 9, 256 px, patch 16 → 16×16 grid, d768 × depth 12, 12 heads, grad-ckpt
operative pred     d768 × depth 10, 12 heads, window 8, horizons (1,2,4), residual, change-weighted
tactical_pred      d512 × depth 6,  8 heads, window 8, horizons (8,16)
tactical_policy    d512 × depth 6,  8 heads, 5 maneuvers, wp (5,10,15,20), d_intent 256, cadence 5
strategic_policy   d384 × depth 4,  6 heads, 4 nav cmds, d_cmd 128, d_ctx 256, n_route 3, cadence 20
readout            spatial grid 4×4, d_readout 128  →  state_dim 2048   (A7: never global-pool)
h15 imagination    enabled, mask_prob 0.5, weight 0.5, depth 3, observed_weight 0.1
loss               SIGReg n_slices 512, β 1.0, w 0.1, free_dims 64 · pred 1.0 · inv_dyn 0.5
optimizer          AdamW lr 3e-4, wd 0.05, betas (0.9, 0.95), warmup 2000, cosine
```

Conditioning flow: `strategic ctx --FiLM--> tactical --intent FiLM--> operative predictor`.
Grounding heads live **outside** the model (separate ckpt keys) so a vanilla `WorldModel` still loads a
4b checkpoint.

---

### 1.1 flagship-v1 **no-speed** — `flagship4b-phase0-30k` ⚠️ *(commonly mistaken for "the deployed v1")*

| Field | Value |
|---|---|
| **Status** | **SUPERSEDED** — killed by the 2026-07-14 speed reset at step **22,950**; retained as the causal ablation control |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship4b-phase0-30k/` (`ckpt.pt`, `config.json`, `train_log.jsonl`, `gate_step{1k,5k,10k}.json`) |
| **Distinguishing flags** | `speed_input=false`, `action_dim=2`, `jerk_weight=0.0`, `aux_accel=false`, `rollout_k=4` |
| **Params (from run config)** | encoder 87,121,280 · operative 96,607,490 · tactical_pred 26,534,912 · tactical_policy 22,736,141 · strategic_policy 8,385,027 · h15 22,055,683 · grounding_heads 13,432,338 → **total_model 263,440,533 / trainable 276,872,871** ✅ |
| **Data** | `physicalai-train-e438721ae894`, skip-hash `f09e44db`, cache `/workspace/data/physicalai_phase0/_epcache` |
| **Exact command** | run manifest `/workspace/ops/runs.d/flagship-phase0.env.disabled`:<br>`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/train_flagship4b.py --data cached --cache-dirs /workspace/data/physicalai_phase0/_epcache --config flagship4b --sigreg-free-dims 64 --rollout-k 4 --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint --lr 3e-4 --warmup 2000 --ckpt-every 1000 --log-every 50 --workers 4 --guard-limit-gb 45 --out /workspace/experiments/flagship4b-phase0-30k` ✅ |
| **Code state** | `stack/scripts/train_flagship4b.py` with **no** optional flags; pod tree at `main@0f93b98`. Reconstructible from the repo. |
| **Results** (TanitEval key `flagship-nospeed`, ckpt ≈22k) | ADE@2s **2.9176 ± 0.3558** heldout / **3.0175** full-set · FDE@2s 4.9395 · miss@2m 0.7395 · **does not beat CV** ✅ |
| **HF** | `Sayood/tanitad-flagship-4b-phase0` (gated, public+manual) |
| **Why it matters** | It is the *causal control* for the speed fix: identical architecture and data, only `speed_input` differs → 2.918 vs 0.452. Do not delete. |

---

### 1.2 flagship-v1 **speed+jerk** — `flagship4b-speedjerk-30k` — ⭐ **THE DEPLOYED MODEL**

| Field | Value |
|---|---|
| **Status** | ✅ **DEPLOYED / operative arm.** `summary.json`: `done: true, final_step: 29999, wallclock_s: 191206.2` (~53 h A40) |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship4b-speedjerk-30k/` · eval copy `tanitad-eval:/root/models/flagship-30k/ckpt.pt` |
| **TanitEval keys** | `flagship-30k` (step 29999 FINAL) · `flagship-speed` (19k relay ckpt, same run) |
| **Distinguishing flags** | `--speed-input --jerk-weight 0.02 --aux-accel`, `rollout_k=4`, `action_dim=3` |
| **The speed channel (the single most important reconstruction detail)** | `v0 = poses[t,3] / 10.0` appended as the **3rd action channel** to both `actions` and `future_actions` in `stack/tanitad/train/flagship_losses.py:228`. **SPEED_SCALE = 10.0** — this constant is a hard contract with `eval_grounded_rollout_4b_speed.py`; get it wrong and the checkpoint decodes garbage. ✅ |
| **Params** | encoder 87,121,280 · operative 96,609,283 · tactical_pred 26,535,424 · tactical_policy 22,736,141 · strategic_policy 8,385,027 · h15 22,055,683 · grounding 13,432,338 · aux_accel 528,897 → **total_model 263,442,838 / trainable 277,404,073** ✅ |
| **Data** | identical to §1.1 (strict parity) |
| **Exact command** | run manifest `/workspace/ops/runs.d/flagship-speed.env`:<br>`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/train_flagship4b.py --data cached --cache-dirs /workspace/data/physicalai_phase0/_epcache --config flagship4b --sigreg-free-dims 64 --rollout-k 4 --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint --lr 3e-4 --warmup 2000 --ckpt-every 1000 --log-every 50 --workers 4 --guard-limit-gb 45 --speed-input --jerk-weight 0.02 --aux-accel --out /workspace/experiments/flagship4b-speedjerk-30k` ✅ |
| **Code state** | ⚠️ The v1 run trained with a **pod-side trainer that was never committed at the time** (noted in the 2026-07-18 review). The `--speed-input` flag was subsequently committed to `stack/scripts/train_flagship4b.py` (`config.py:211 speed_input`, `train_flagship4b.py:525`) so a rebuild from HEAD is now possible. `--jerk-weight` / `--aux-accel` are **not** in the committed `train_flagship4b.py` arg list — see risk block. |
| **HF** | `Sayood/tanitad-flagship-4b-speedjerk` (gated-manual). Also pushed 07-16 mid-run as `Sayood/flagship-4b-phase0` / `Sayood/tanitad-flagship-4b-phase0` |

**Results — step 29999 (`flagship-30k`), 881 windows** ✅ *(read from `results/flagship-30k.json`)*

| Metric | heldout mean ± CI95 | full-set |
|---|---|---|
| ADE@0.5s | 0.0762 ± 0.0046 | — |
| ADE@1s | 0.1584 ± 0.0149 | — |
| ADE@1.5s | 0.2883 ± 0.0227 | — |
| **ADE@2s (`ade_0_2s`)** | **0.4522 ± 0.0312** | **0.4271** |
| FDE@2s | 0.9437 ± 0.0630 | — |
| miss@2m | 0.0602 ± 0.0121 | — |

Clears every trivial bar on the same 881 windows: best-of-3 kinematic floor 0.5005 · CTRV oracle 0.523 ·
no-vision ego-status ceiling 0.5735 · CV 0.8248.

**Strata — skill-vs-floor** (model ÷ per-stratum floor; <1 beats floor): straights **1.032** · gentle
**0.679** · sharp **0.599** · **high-speed top decile 1.785 ← the one open weakness.**

**Failure signature (`pathspeed.py`, 2 s):** long-RMSE 1.04 m / lat-RMSE 0.36 m → **89 % of squared error is
longitudinal**; speed bias **+0.19 m/s**; along-track overshoot +0.38 m; speed-decoupled cross-track only
0.10 m. At high speed: speed over-prediction **+0.66 m/s**, long-RMSE 1.38 m vs CTRV 0.077 m.
**Interpretation: the residual is a longitudinal/speed problem, not a geometry problem.**

**Genuine-prediction (causal) panel:** on high-CTRV-divergence windows it beats the CTRV *oracle* by
+0.796 m on 94 % of them; mean-replacing the scene inverts this to −0.529 m → **vision effect +1.325 m,
CI [+1.04, +1.64]** (CI-separated). Upcoming-curvature decode R² 0.254 vs 0.031 ego-only.

**OOD:** physicalai (in-dist) 0.427 vs floor 0.523, win 49.7 % ✅ | comma2k19 0.849 vs floor 0.372, win
17.5 % ✗ | cosmos 0.583 vs 0.358, win 29.4 % ✗. **Generalization is the open gap.**

**Closed-loop (imagination-in-the-loop, no renderer):** closed_bike ADE@2s **1.685 ± 0.098**, FDE 3.530,
divergence >5 m 22.2 %. Open-loop 0.452 → closed-loop 1.685: **open-loop does not predict closed-loop.**

**Efficiency — TWO DIFFERENT TICKS. Neither may be quoted without its definition.**

| tick | what it actually measures | hardware | ckpt / corpus | p50 | Hz |
|---|---|---|---|---|---|
| **decision tick** = `encode(1 frame) + select_K9` | imagine-and-select only — **does NOT include the 20-step rollout** | RTX 4060 (declared Orin proxy, single-stream) | step **6,500** / **comma2k19** | **11.16 ms** fp16+CUDA-graph (17.75 fp32, **1.59×**) | 89.6 |
| **planning tick** = `encode(8-frame window) + 20 SEQUENTIAL predictor steps → per-step metric Δpose → SE(2) accumulate` | the intent-free operative path that **produces the trajectory ADE@2s scores** | A40, exclusive (contamination-checked) | step **29,999** / **physicalai val** | **103.42** fp32 · **93.76** tf32 · **104.49** amp16 | 9.7 / 10.7 / 9.6 |

⚠️ **DEFECT CORRECTED 2026-07-20.** This line previously read *"deploy tick 11.16 ms / 89.6 Hz"* with no
definition, hardware, checkpoint or corpus, and propagated in that bare form to `PROGRAM_OVERVIEW.md`,
`Progress Reports/2026-W33.md` and the 360-review. It is a **1-frame encode plus a K=9 select, on a
different GPU, a different checkpoint and a different corpus** — it is **not** the latency of the
trajectory the leaderboard scores. The two figures differ in **five** dimensions at once and are not
comparable; the 9× apparent discrepancy is definitional, not a regression.

**The planning tick MISSES the 10 Hz budget at p99 in all three precisions** (146.60 / 102.71 /
113.13 ms). Rollout = **83.7 → 96.7 %** of it (20 sequential steps @ 4.35–5.08 ms/step); encoder only
15–26 %. Achieved **3.7–4.3 TFLOPs** ⇒ **launch/serialisation-bound, not arithmetic-bound**. `amp16` is
*slower* than `tf32` here — precision cannot help a launch-bound dependent chain. Peak 1217 MB.
Encoder-caching (encode only the new frame) → 84.74 ms. Batched: best **34.8 windows/s** @ batch 32.

**✅ OPTIMISED PLANNING TICK — MEASURED 2026-07-21** (A40, batch 1, exclusive under `gpu_lock.sh`,
`contamination_check.valid` sampled before/after *and between* every variant; raw:
`taniteval/results/eff_levers_flagship-30k.json`). Eager reference this session **100.29 / 113.98 ms**:

| lever | tick p50 | ×p50 | max abs dev | 10 Hz @p99 |
|---|---:|---:|---:|:--:|
| eager fp32 | 100.29 | 1.00 | — | ❌ |
| **L1b** CUDA-graph the 20-step rollout | 57.18 | **1.75** | **0.0 m (exact)** | ✅ |
| L1d `torch.compile(reduce-overhead)` | 52.89 | 1.90 | 3.8e-6 m | ✅ |
| L2 encoder cache alone | 95.11 | 1.05 | 1.9e-6 m | ❌ |
| L3 fp16 weights alone | 98.47 | 1.02 | 0.024 m | ❌ |
| L7 drop 2 unused horizon heads alone | 100.47 | 1.00 | 0.0 m | ❌ |
| **L4 = L1+L2+L3+L7 composed** | **18.75** | **5.35** | 0.024 m | ✅ **53.3 Hz** |

**The flagship MEETS the 10 Hz budget at p99 with 5.3× headroom (18.76 ms).** Rollout *stage*
95.03 → 28.73 ms (**3.31×**, 4.75 → 1.44 ms/step); the tick multiple is diluted by the eager encoder.
Orthogonality confirmed: fp16 gives the encoder **3.81×** and the rollout **1.01×**; the graph gives
the rollout 3.31×. Free refinement: keeping SE(2) accumulation in fp32 halves fp16's deviation
(0.0241 → 0.0127 m).

⚠️ **Levers are SEQUENCED, not additive — capture FIRST.** L2/L3/L7 are worth ~1.0× *before* L1 and
24 / 32 / 0.6 ms *after* it. The 2026-07-18 "levers compose additively" result **does not generalise**
from a 1-step select to a 20-step rollout.

⚠️ **`torch.compile(reduce-overhead)` beats manual capture on Linux (52.89 vs 57.18) — the opposite of
the Windows result** (`TritonMissing`; the `cudagraphs` backend ran 20× slower). It is **not
bit-identical**, so **manual capture stays the deploy default**.

⭐ **CEM is NOT latency-blocked.** An 8-candidate imagine-and-select fan costs **20.82 ms p50 /
23.72 p99** (K=32: 28.41 ms); marginal candidate ≈ **0.3 ms** — provided you **encode once and
broadcast** (re-encoding per candidate costs +5.6 ms at K=8, +26.9 ms at K=32). This **refutes** the
`n_candidates × horizon × per_step` arithmetic that projected 723 ms.

Predictor **2.57×** is the *`predict_1pass` stage* on an RTX 4060 — a stage figure, never a tick.

Sources: `taniteval/results/eff_flagship-30k.json` (2026-07-20, `taniteval.efficiency`) ·
`agent/prod-opt-20260718` combined-tick note (2026-07-18) — ⚠️ **that harness is NOT in HEAD**
(`combined_tick_harness.py`, reconstruction gap; see §6).

**✅ DEPLOYMENT EXPORT — MEASURED 2026-07-22** (A40 SM 8.6 **proxy**; raw in
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-22-orin-thor-deployment/artifacts/`;
full staged plan `DEPLOYMENT_PLAN.md` in that folder). The **exact deployed arch** (`action_dim=3`,
263.44 M = `total_model 263,442,838`) exports to **static-shape ONNX** (encoder `[1,9,256,256]→[1,2048]`,
predictor `states[1,8,2048], actions[1,8,3]`), torch-vs-ORT parity ≤ **1.9e-6** (`export_report.json`);
builds to a **TensorRT-FP16** engine — encoder **1.205 ms** / predictor **0.666 ms** p50, and **TRT fuses
our MHA** (no standalone softmax → the NVIDIA #4537 ViT-fusion risk does not bite on SM 8.6)
(`trt_fp16_report.json`); and an independent A40 reproduces the **CUDA-graph rollout** lever — eager
**96.40 → graph 27.87 ms** p50, K=20, **3.46×** (`bench_latency_report.json`), matching the eval pod's
rollout-stage 95.03→28.73 ms to 3 %. **Per-chip precision map (PUBLISHED, vendor specs):** Orin (Ampere
SM 8.7) = FP16 baseline, INT8 only per-layer-gated, **no FP8/FP4**; Thor (Blackwell) = FP16/FP8 + **NVFP4**
(Thor-only 4× weight-traffic win). ⚠️ **A40 is a PROXY — TRT engines are not portable across GPU
architectures;** real Orin/Thor throughput, the on-device engine build, and any NVFP4 number are
**hardware-blocked** (silicon not on hand), not fabricated. Consistent with the §1.2 composed tick
(18.75 ms) via a different (TRT) route.

**Results — 19k relay (`flagship-speed`), same run** ✅

ADE@2s **0.6277 ± 0.0551** heldout / **0.6152** full-set · FDE 1.3173 · miss 0.1799 · **first CV-beater.**

> ⚠️ **Number-hygiene note that resolves a repo-wide conflict.** The 19k relay is quoted in docs as
> 0.628, 0.615 and 0.640. From the raw JSON: **0.6277 = heldout**, **0.6152 = full-set** (this is the
> "0.615" in the H26 hierarchy panel — same eval, different statistic), and **0.640 is derived
> arithmetic** (0.4522 + the paired 0.188 m win-delta), never a measured mean. Cite 0.628 (heldout).

> 🟥 **RECONSTRUCTION RISK — v1 speedjerk.** The committed `stack/scripts/train_flagship4b.py` arg parser
> has **no `--jerk-weight` and no `--aux-accel`** (verified: `grep add_argument` returns neither), yet the
> run's `config.json` records `jerk_weight: 0.02, aux_accel: true` and `summary.json` books an
> `aux_accel: 528897`-param head. **A clean-checkout rebuild of the deployed v1 is therefore not
> byte-exact today.** The pod2 working tree at `/workspace/TanitAD/stack` still carries the modified
> trainer (`git status` shows `M stack/scripts/train_flagship4b.py`). **Action: commit the pod2 trainer
> diff, or add the two flags, before pod2 is recycled.**

---

### 1.3 flagship-v2 — `flagship4b-v2-30k` — **ABANDONED at step 7,800**

| Field | Value |
|---|---|
| **Status** | ❌ **ABANDONED.** Launched 2026-07-18 19:48:09Z on pod2; last logged step **7,800**; killed after the 6k diagnostic. Superseded by v3enc. |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship4b-v2-30k/` (`ckpt.pt`, `ckpt_step5000.pt`, `config.json`, `train_log.jsonl`, `supervisor.log`) |
| **Distinguishing flag** | `--v2` — one flag that turns on the whole lever pack |
| **Params** | operative 96,609,284 · tactical_pred 26,535,424 · tactical_policy 30,098,063 · strategic 8,385,411 · encoder 87,121,280 · h15 22,055,683 · grounding 13,432,338 → **total_model 272,906,913 / trainable 286,339,251** ✅ (+9.5 M over v1: the anchored tactical decoder) |
| **Exact command** | run manifest `/workspace/ops/runs.d/flagship-v2.env`; identical to §1.2 plus `--v2` (which implies `--speed-input`, `--labels-v2`, and defaults `rollout-k` to 12) |
| **Code state** | `stack/scripts/train_flagship4b.py:513 --v2`; lever definitions `stack/tanitad/config.py:164-237`; labels `stack/scripts/refb_labels.py` v2 path; decorr `stack/tanitad/train/decorr.py`. Commits **`f583bb4`** (six levers), **`b8d3fc8`** (labels-v2), **`a01ad24`** (v2 levers + v3enc schedule). ✅ |
| **HF** | none |

**The lever pack as actually recorded in the run's `config.json`** (this is the definitive list — the
`--v2` help string in the trainer names only six; ten flags are set):

| # | Flag | v2 value | What it does |
|---|---|---|---|
| 1 | `v2_ego_to_planners` | `true` | feed `[v0, yr0]` to the strategic + tactical brains |
| — | `v2_ego_dropout` | `0.25` | ego-vector dropout — shortcut guard for (1) |
| 2 | `v2_fa_dropout` | **`0.30`** | future-action dropout inside the rollout loss |
| 3 | (via `--v2`) `rollout_k` | **`12`** | K-step recursive rollout, up from 4 |
| 4 | `v2_goal_decode` | `true` | goal-conditioned trajectory head |
| 5 | `v2_nav_dropout` | `0.5` | nav-command dropout → route must come from vision |
| 6 | `v2_traj_jerk` | `0.02` | jerk penalty on predicted waypoint paths |
| 7 | `v2_gated_intent` | `true` | ReZero gate on the intent→operative term (H26: ungated intent was net-harmful) |
| 8 | `v2_anchor_tactical` | `true` | **time-anchored multi-anchor (DiffusionDrive-style) tactical decoder** replacing the unimodal `wp_heads` — the +9.5 M |
| 9 | `v2_route_from_vision` | `true` | always-on nav-zeroed route aux (weight 0.3) — fixes the command-echo strategic head |
| 10 | `v2_encoder_ego_decorr` | `true` | linear decorrelation penalty (weight 0.05) between pooled `z_t` and fed ego `[v0, yr0]` |
| — | `v2_invdyn_gradscale` | **`0.25`** | gradient scale on encoder latents feeding the inverse-dynamics real-pair term |
| — | `v2_labels` | `true` | curvature-relative strategic/tactical labels (data-side only, no param change) |

**Results — step 6,000 (`flagship-v2-6k`)** ✅ *(from `results/flagship-v2-6k.json`)*

ADE@0.5s **1.2389** · ADE@1s 2.3276 · ADE@1.5s 4.0048 · **ADE@2s 6.179 ± 1.2845** (7.4× CV) · FDE@2s
12.7015 · miss@2m 0.8407. Offline reproduction (full-set) 5.94.

**Diagnosis (why it was killed, not continued):**
- Encoder speed-probe R² **0.30** (v1: 0.861); operative step-1 decoded-speed R² 0.723 (v1: 0.9987);
  step-1 speed error **+2.39 m/s** (v1: +0.06). Rollout speed diverges 15.1 → 23.9 m/s vs flat GT ≈12.7.
- Error is **79 % longitudinal** at 2 s, signed **+9.74 m overshoot**.
- **Learning-rate-of-improvement, the decisive read:** same-step `g_op_fwd_ade_m` power-law exponent
  **v2 = −0.50 vs v1 = −0.84**; the v2/v1 ratio *widened* 1.51 → 4.33 over 0–7 k. v1 reached v2's 7.5 k
  value at **step ~250** (~30× faster). Projection to 30 k: v2 0.273 vs v1's actual 0.030 → **~9× worse
  for the same ~4 days of A40**. ✅
- Per-lever telemetry was otherwise healthy (anchored decoder converging, no NaN, no gnorm spike) — the
  problem was **all ten levers at once**, not any one of them.

---

### 1.4 flagship-v3enc — `flagship4b-v3enc-30k` — ⏹️ **STOPPED at step 10,800** · 🟥 **10 k GATE: `RESTART`** (2026-07-21)

| Field | Value |
|---|---|
| **Status** | ⏹️ **STOPPED 2026-07-21 ~14:35 local (12:35 UTC) at step 10,800**, on Sayed's decision after the 10 k gate returned `RESTART`; superseded by **flagship v4** (`V4_FLAGSHIP_DESIGN.md`). Killed by explicit parent PID 1388768 — clean exit, no orphaned workers, GPU released. `ckpt.pt` was still the step-10,000 write (md5-identical to `ckpt_step10000.pt`), so only 800 uncheckpointed steps were lost. Ran on **`tanitad-pod`** (RTX A6000) from **2026-07-20 05:27 UTC, step 0** (fresh, not resumed). ⚠️ **DO NOT RECYCLE `tanitad-pod`** — `ckpt_step10000.pt` was never on D-032's archive list and is the only 10 k state that will ever exist; it is the evidence behind the RESTART verdict and two settling experiments need it. |
| **Prior attempt** | `tanitad-pod2` launched 2026-07-19 21:42 UTC → **died at step 1,950** on 2026-07-20 03:56 UTC in the checkpoint write (`PytorchStreamWriter failed writing file data/967`); pod2 overlay was 98 % full with a stale 3.36 GB `ckpt.tmp`. Dead dir preserved at `tanitad-pod2:/workspace/experiments/flagship4b-v3enc-30k/`. ✅ |
| **Distinguishing flags** | `--v2 --staged-levers` |
| **Params** | identical to v2: **total_model 272,906,913 / trainable 286,339,251** ✅ |
| **Exact command** *(read live from `ps` on `tanitad-pod`)* | `cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 scripts/train_flagship4b.py --data cached --cache-dirs /workspace/data/physicalai_phase0/_epcache --config flagship4b --sigreg-free-dims 64 --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint --lr 3e-4 --warmup 2000 --ckpt-every 1000 --log-every 50 --workers 4 --guard-limit-gb 45 --speed-input --v2 --staged-levers --out /workspace/experiments/flagship4b-v3enc-30k` ✅ |
| **Code state** | `stack/scripts/train_flagship4b.py:519 --staged-levers`, schedule at `:293-306`, per-step application via `staged_lever_schedule()`. Commit **`a01ad24`**. ✅ |
| **HF** | none |

**What "staged" means** — v3enc is v2 with the four **encoder-grounding** levers softened and time-staged,
while every **decode-side** lever (anchored tactical, gated intent, goal decode, labels-v2, jerk,
ego→planners, route-from-vision) stays on from step 0:

| Lever | v2 | v3enc |
|---|---|---|
| `decorr_weight` | 0.05 constant | **0.0 until step 10,000, then 0.02** |
| `rollout_k` | 12 constant | **4 (<5k) → 8 (<10k) → 12** |
| `v2_invdyn_gradscale` | 0.25 | **0.5** (softer decouple) |
| `v2_fa_dropout` | 0.30 | **0.15** (gentler withhold) |

**Results — step 10,000 (`flagship-v3enc-10k`), 881 windows / 40 episodes** ✅ *(read from
`taniteval/results/flagship-v3enc-10k.json`; eval 2026-07-21 12:56 CEST on `tanitad-eval` under
`gpu_lock.sh acquire v3enc-gate`, `TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2`)*

The gate ckpt is `tanitad-pod:/workspace/experiments/flagship4b-v3enc-30k/ckpt_step10000.pt`
(`atomic_archive`, ckpt `step` field verified 10000) → `tanitad-eval:/root/models/flagship-v3enc-10k/ckpt.pt`,
md5 `3654a99935d456a56874359e93934b70` identical both ends. **10 k was NOT on D-032's archive list
(5k/15k/20k/30k) — this is the only 10 k state that will ever exist.**

| Metric | episode-cluster bootstrap (PRIMARY) | `overlapping_holdout_se` (DEPRECATED) |
|---|---|---|
| ADE@0.5s | 0.5629 [0.4719, 0.6631] | 0.6146 ± 0.0727 |
| ADE@1s | 0.9382 [0.7918, 1.0972] | 1.0160 ± 0.1065 |
| ADE@1.5s | 1.4178 [1.1981, 1.6517] | 1.5329 ± 0.1552 |
| **ADE@2s (`ade_0_2s`)** | **1.9654 [1.6556, 2.2859]** | 2.1072 ± 0.2020 |
| FDE@2s | 3.6084 [2.9973, 4.2290] | 3.8298 ± 0.3600 |
| miss@2m | 0.6901 [0.6039, 0.7682] | 0.7187 ± 0.0492 |

**Paired vs v1 `flagship-30k`, same 881 windows** (`paired_episode_cluster_bootstrap`, B=2000):
ADE@2s Δ **+1.5383 [+1.2697, +1.8159]** — CI-separated **against** v3enc (v1 0.4271). ⚠️ 10 k vs 30 k is
**not** an equal-budget comparison. Against the trivial floor v3enc is also CI-separated **worse than CV**
(Δ +1.1277 [+0.8741, +1.4134]; CV 0.8377, hold-v0 0.7876).

**Driving panel (TanitEval v2 tier-0)** — along-track 3.2166 [2.699, 3.757] · cross-track 1.0627
[0.734, 1.429] · speed MAE **1.8075** [1.550, 2.079] vs hold-v0 **0.4818** · straight-road heading MAE
**3.642°** vs CV 1.399°. `where_the_win_lives = neither axis separated`; `tracks_speed_better_than_cv = False`.
**Notable:** straight-road heading 3.642° is **better than half of v1's 7.980°** on the identical windows.

**Pre-registered gate — `Project Steering/Gates/flagship-v3enc.card.json`, adjudicated by
`stack/scripts/run_gate.py check` → `Gates/flagship-v3enc-gate-10k-2026-07-21.json`:**

| criterion | threshold | measured | |
|---|---|---|---|
| primary `ade_0_2s` | ≤ 2.5 m | **1.9654** (upper CI 2.2859) | ✅ PASS |
| `encoder_speed_probe_r2` | ≥ 0.55 | **0.393** (v2 0.300 · v1 0.861) | 🟥 **FAIL** |
| `highspeed_long_overshoot_m` | ≤ 8.0 | **+2.195** (v2 +23.7 · v1 +0.831) | ✅ PASS |

### 🟥 **VERDICT: `RESTART`** — one pre-registered secondary failed. Restart budget was 1/2 for lever family `encoder-grounding`; a second restart exhausts it, and a third failure **refutes the family**.

**Secondaries are `diag_v2mech.py` verbatim** — the pod copy at `tanitad-eval:/root/diag_v2mech.py` is
byte-identical to the repo's `taniteval/diag_v2mech.py` (already rescued in `4124be0`), and it was driven
unmodified by `taniteval/diag_v3enc_gate.py` (which only appends a registry entry in memory).
Definitions per `2026-07-19-flagshipv2-6k-diagnostic.md:196-199`:
`probe_speed_r2` = ridge z_t→v0, per-episode held-out (8 eps), λ∈(1e-2,1e-1,1,10); `op_long2s_high` =
mean **signed** along-track error at the 2 s waypoint in the high-speed tercile.

**🔑 The finding that reframes the failure — decorr was NEVER ON.** `train_flagship4b.py:92` sets
`decorr_w = 0.0 if step < 10000`. The whole 0–9,999 window carries **zero** decorrelation penalty, so the
gate measured the arm *before* the staged lever under test was applied. D-031 attributed v2's collapsed
probe (0.300) to decorr "strangling speed capacity" — but with decorr removed entirely for 10 k steps the
probe only reached **0.393**, not v1's 0.861. **Decorr is not the main cause.** The levers still active at
the gate step and absent from v1 are `invdyn_gradscale` **0.5 vs v1's 1.0** (default `config.py:224`) and
`v2_ego_dropout` **0.25 vs 0.0** — those are the surviving suspects. (In-sample `ego_r2` runs 0.79–0.85
while the episode-held-out probe reads 0.393: the encoder's speed content exists but does not generalise
across episodes.)

**Matched-step ratio on `g_op_fwd_ade_m`** (`run_gate.py`, no exponent): mean **2.629** CI [2.454, 2.807]
over steps 50–10,200; first 0.757 → last 3.314, **WIDENING**.
🔴 **RETRACTED 2026-07-21 — "v1 reached v3enc's current 0.4101 at step 450 (~23× step-efficiency)".**
Both the step and the multiple are withdrawn. That was a **single-row noise artifact**: v1's raw rows
around step 450 swing 0.758 / 0.616 / 0.404 / 0.687 / 0.384 / 0.816 — the 0.404 is one draw from a wide
distribution, not a level v1 had reached. `g_op_fwd_ade_m` is a **per-batch (B=16)** train metric that
swings ~2× between adjacent logged rows, which the then-current 3-point rolling median in
`run_gate.py:reference_reached_at` could not smooth.

**Replacement figure and its estimator** — v1 reaches 0.4101:

| estimator | result | step-efficiency vs v3enc's 10 k |
|---|---|---|
| **fixed-width 2 k bucket mean** (crossing bucket, primary) | **2 k–4 k**, bucket mean **0.3389** (raw log) · 0.3465 (gate artifact) | **2.5–5×** |
| **k=3 consecutive crossings** (point estimate) | **step ≈ 2 500** | **≈ 4×** |

**Quote it as ≈4× (point), 2.5–5× (interval) — never 23×.** The interval's low end is 2.5×, not the
3.5× of this row's first draft: 3.5× came from interpolating between bucket centres, whereas the bucket
the crossing actually falls in only licenses 2.5–5×. The k-consecutive point estimate is computed on the
gate artifact's matched-step reference series (`per_step[].ref`, n=204); recomputation against the raw
pod-side log is **pending** and may move it within the interval — the interval itself is unaffected, both
sources put the crossing in 2 k–4 k. *(Same failure mode as the retired exponent gate: a scalar read off a
noisy curve at one point. Prefer bucket means or the matched-step ratio, never a single row.)*
**Fixed in code 2026-07-21:** `reference_reached_at` now requires k consecutive crossings, returns the
bucket interval alongside the point, and ships a mandatory `estimator` field so the statistic cannot be
quoted without its rule; regression-tested in `stack/tests/test_run_gate_reached_at.py`. The two gate
JSONs still on disk (`Gates/flagship-v3enc-gate-2026-07-20.json`,
`Gates/flagship-v3enc-gate-10k-2026-07-21.json`) retain `"reached_at_step": 450` under
`"smoothing": "3-point rolling median"` — **that field is void in both**; they were not re-run (no GPU).
Budget 10.34 vs 10.89 s/step → 348 vs 331 steps/GPU-h; **29.3 GPU-h** spent. Per-1k bucket vs the
v2 table in the diagnostic §4:

| bucket | v1 | v3enc | v3enc/v1 | v2 | v2/v1 |
|---|---|---|---|---|---|
| 0–1k | 0.954 | 1.304 | 1.37 | 1.443 | 1.51 |
| 1–2k | 0.484 | 0.842 | 1.74 | 1.222 | 2.52 |
| 3–4k | 0.267 | 0.532 | 1.99 | 0.862 | 3.23 |
| 5–6k | 0.201 | 0.451 | 2.25 | 0.627 | 3.12 |
| 6–7k | 0.140 | 0.482 | 3.45 | 0.604 | 4.31 |
| 9–10k | 0.100 | 0.548 | **5.50** | — | — |
| 10–11k | 0.110 | 0.431 | 3.90 | — | — |

**The staging worked, and it was not enough.** v3enc's ratio is below v2's at *every* matched bucket
(≈half the excess early) and the level is far better (ADE 1.97 vs v2's 5.94, overshoot +2.2 vs +23.7) — but
the ratio is still **widening**, so D-A7's own falsifier ("no improvement in same-step forward-consistency
vs v1 at 10 k") is also not cleared. *(Exponent logged as a diagnostic only: R²=0.179 over 1500–10000,
n=171 — **UNSUPPORTED**, below the 0.80 floor. It decides nothing.)*

⚠️ **CONFOUND that travels with this row: v3enc trained on the PRE-v2.1 (broken) route labels** — 26 %
coverage, 63 % of genuine turns unlearnable, masked windows still emitting `ROUTE_STRAIGHT`, and no token
for any longitudinal mode (`stop_at_point`/`hold_stop`/`creep`). Measured in this run's log: `nav_valid_frac`
≈ 0.21, `route_acc` 0.68–0.84 ≈ the majority-class rate. **This does not explain the failing criterion** —
a 3-way *lateral* topology label cannot remove *scalar speed* from a linear encoder probe — but it does
invalidate any route/strategic reading from this arm, and it is a live confound on the wp/goal heads via
the gated-intent path.

**Remaining gates (unchanged, not yet run):**
- **Acceptance gate should be the OOD panel, not in-distribution:** beat the comma2k19 floor (0.372 m) on
  **≥ 35 %** of windows, up from v1's 17.5 %.

---

### 1.4b flagship-v1.6 — `flagship-v16-ab-ft` — ✅ **COMPLETE at 5,999** · ⭐ best ADE in the program

LP-FT completion of the v1.5 ladder: the `ab` head warm-started, then **4 encoder blocks + the
predictor UNFROZEN** (head-LR 1e-4 / trunk-LR 1e-5, 500-step ramp). pod2, 20:01→01:02 UTC, 18,038 s.

**CANONICAL eval — `eval_flagship_v16.py`, 881 windows, run 2026-07-21 02:20 on pod2 under
`gpu_lock.sh`** ✅ *(`/workspace/experiments/flagship-v16-ab-ft/eval_v16.json`)*

| | **v1.6** | v1.5 `ab` | flagship v1 | REF-C-XL |
|---|---:|---:|---:|---:|
| **ADE@2s heldout** | **0.4886 ± 0.0800** | 0.5437 | **0.4522** | 0.4577 |
| **ADE@2s full-set** | **0.43746** | — | **0.4271** | 0.4714 |
| **WM canary** | **1.1022** | 0.4521 | *0.452 (base)* | n/a |

**❌ v1.6 does NOT beat v1.** `beats_cv` ✅ · **G1 (beat REF-C 0.458) ❌ · G2 (beat v1 0.4522) ❌ ·
G3 (miss ≤0.10) ❌.** On heldout it is the **worst of the three** (0.4886 vs 0.4522 / 0.4577); on the
full set it sits between them but still **behind v1** (0.43746 vs 0.4271).

> 🔴 **RETRACTED — "v1.6 ADE 0.44201, the best in the program."** That figure is the **trainer's own
> in-loop val**, a *different protocol*, and it is **~10 % optimistic** versus the canonical harness
> (0.44201 vs 0.4886 heldout). I entered it into this registry as a headline, which is precisely what
> §0 forbids: **a training-log number is not an eval number.** Trainer val is for watching a curve;
> only `eval_*.py` output may be quoted. *(Third revision of this arm in one night: "decisive failure"
> → "best in program" → this. Each error came from quoting a faster-moving source than the harness.)*

**The finding stands, with the direction unchanged and the magnitude corrected:** unfreezing bought
fan quality (oracle 0.3073 → 0.2815, in-loop) and **cost the world model 144 %** (canary 0.452 →
1.1022) — and on the canonical harness the ADE trade came out **net negative**. Unfreezing 4 ViT
blocks is not the route to a REF-C-grade fan, and it damages the substrate v3.5 is built on.

**✅ PAIRED EPISODE-CLUSTER BOOTSTRAP — run 2026-07-21, and it settles the arm:**

```
Δ(v1.6 − v1) = +0.0104 m   CI95 [−0.0888, +0.1147]   separated = FALSE
full-set 0.4375 vs 0.4271 · per-window corr 0.453 · 40 episodes · 881 windows
```

**v1.6 and v1 are statistically INDISTINGUISHABLE on ADE.** Neither of the earlier claims survives:
not "best in the program", not "clearly worse". **Unfreezing changed nothing measurable on ADE while
costing 144 % of the world model** — a cleaner and more damning result than either framing, because
we paid a large measured price and got back nothing that survives a valid test.

⚠️ **Two eid FAMILIES exist and their `heldout` means are NOT comparable.** `bench.py` clusters on
**file indices 0–39**; `eval_flagship_v15/v16.py` deliberately cluster on the **real `episode_id`**
(`real_episode_ids()`, e.g. 808464434) *because the eval pod's estimator does* — see the v15 docstring:
*"using file indices instead would produce a DIFFERENT episode partition and therefore a different
heldout mean."* Both partition the same 40 episodes (verified: a consistent 1-to-1 relabel), so
**episode-cluster bootstrap and full-set are unaffected** — but `split_by_episode` hashes the id
*values*, so the 8-split `heldout` numbers come from **different random partitions across the two
families and must never be compared directly**. v1.6's 0.4886 heldout vs v1's 0.4522 is such a
cross-family comparison; the **paired bootstrap above is the valid one**. Alignment for pairing was
proven from the data, not assumed: `gt` and `cv` are identical **elementwise, max diff 0.0**.
⚠️ **Use `eval_flagship_v16.py` ONLY**: it re-encodes val frames through the unfrozen trunk; cached
`states_val.pt` would silently score the OLD trunk. The pod copy was stale (248 lines, no
`windows_*.pt` persistence) and was synced from HEAD before this run.

**The result, stated honestly — unfreezing TRADES the world model for planning quality:**
- planning **improved**: ADE −18.7 % vs `ab`, and the fan improved **−8.4 %** (0.3073 → 0.2815);
- the world model was **destroyed**: the intent-free canary went **0.452 → 1.1022, +144 % (2.44×)**.

**G-A verdict: PARTIAL — 2 of 5.** ✅ G1 (<0.458) ✅ G2 (<0.4522) · ❌ oracle ≤0.22 (0.2815)
· ❌ miss ≤0.10 (0.1067, narrow) · ❌ canary flat (+0.650). **Branch 1 does not fire.**

**What it settles:** unfreezing 4 ViT blocks buys only **8.4 %** of the fan gap — nowhere near REF-C's
0.1640. Frozen-vs-trained encoder is confirmed as the *direction*, but this is **not** the route to a
REF-C-grade fan, and the WM cost makes it actively hostile to a design built on that WM (v3.5).

⚠️ **Process note (mine).** At step 2500 a transient spike (oracle 2.08, gnorm 161) plus a monotone
canary trend led me to report a "decisive failure". **It recovered completely** — 5,999 is the best
ADE in the program. The confirming-eval discipline saved the run; the premature *communication* did
not. Second such call this session. **A single post-spike eval is not a verdict.**

### 1.5 flagship-v4 line — three planners over the v1 world model, **trained JOINTLY, nothing frozen**

The v4 family is a single architecture (`stack/tanitad/models/flagship_v4.py`, `FlagshipV4Head` +
`V4Config`; trainer `stack/scripts/train_flagship_v4.py`) trained across a lineage of restarts —
**v4 → v4.1 → v4.2 → v4.2b (+ a from-scratch fallback)** — that differ **only** in one lever each
(trunk learning-rate / λ_plan-canary-controller behaviour / warm-start-vs-random-init). Every headline
below is read from the **raw held-out eval JSON**, not from `LOOP_STATE.md` prose; the first
decision-grade v4 numbers were produced 2026-07-23 by `stack/scripts/eval_flagship_v4.py` after that
harness was validated on the known v1 checkpoint (MODE A: **0.42148** vs registry full-set **0.4271**,
`taniteval/results/v1-validation.json`, `HARNESS_VALIDATED: true`).

**Shared architecture** — v1's trunk (trained-from-scratch ViT encoder + action-conditioned operative
predictor) + a **strategic planner** (its own predictor in a compressed 128-d subspace) + **tactical and
operative anchored-diffusion planners** (two `FlagshipV15Head` / DiffusionDrive-style instances, 256
anchors), λ_plan curriculum coupling the planner gradient into the trunk. It warm-starts the trunk from
the deployed v1 (`flagship4b-speedjerk-30k`, step 29999) and then **trains everything jointly** —
except the from-scratch fallback, which random-inits the trunk. Head config (from the run's own
`config.json`): `cond_imagination=false`, `cond_vtarget=true`, `cond_route=true`, `factorised=true`,
`n_anchors=256`, dense 1..20-step horizons, decoder d384×4L.

**Params — MEASURED by instantiation** (`V4_FLAGSHIP_DESIGN.md` §3.1; `scratchpad/v4_param_budget.py`
under venv `C:/Users/Admin/venvs/tanitad`; the same script reproduces `WorldModel(flagship4b_config())`
= **263,440,533** byte-identical to §1.1 as the faithfulness check; G0-preflight re-verifies vs §3.1 at
launch, `V4_FLAGSHIP_DESIGN.md` §17 line 1737):

| Module | Params | vs v1 |
|---|---:|---|
| shared trunk — encoder+readout **87,121,280** · operative predictor `action_dim 3` **96,609,283** · H15 imagination **22,055,683** | (v1 verbatim) | unchanged |
| ① strategic planner — `E_strat` 2048→128 + strategic predictor in the 128-d subspace + option-prior + goal-scalar + KV proj | **5,152,911** | NEW |
| imagination-horizon-scaling direct-head baselines (falsifier control arm) | **3,149,824** | NEW |
| ② tactical planner — `FlagshipV15Head` d384×4L, 256 anchors, 5 s coarse (diffusion #1) | **9,767,320** | NEW |
| ③ operative planner — `FlagshipV15Head` d384×4L, 256 anchors, **dense 20-step** (diffusion #2) | **9,778,604** | NEW |
| factorised LAT(8)/LON(7)/DIST(8) heads + 3 zero-init anchor grafts | **≤ 811,543** | NEW |
| removed from v1 — `tactical_policy` −22,736,141 · `tactical_pred` −26,534,912 · `strategic_policy` −8,385,027 · aux-accel −528,897 | | REMOVED |
| grounding heads (op/tac/str, outside the model) | 13,432,338 | v1 verbatim |
| **v4 TRAINABLE TOTAL** | **≈ 247,878,786** | ✅ **~30 M *smaller* than v1's 277,404,073; 62 % of the 400 M cap** |

> ⚠️ The **247,878,786** total is MEASURED by local instantiation, not printed in the run's
> `config.json`. Do **not** quote "~247.9 M" from prose bare: its authority is §3.1's per-module
> instantiation + the G0-preflight faithfulness check, and it is the single authoritative figure (an
> earlier "≈239 M" was the pre-strategic-planner count; O-02 CLOSED). The `flagship-v4.card.json` gate
> carries no param field.

**Not frozen — MEASURED run-side, two independent probes** (both required, per the operating standard's
"two probes" rule): the v4.2 run's `config.json` records `not_frozen_proof` = `{not_frozen: true,
encoder_params_requires_grad "149/149", predictor_params_requires_grad "159/159", trunk_tensors_frozen
0, trunk_group_lr 1e-4}` (`taniteval/results/trainlogs/flagship-v4.2-step4000_config.json`); and the
v4.1 trainer banner reads `[v4] warm-started trunk+grounding from …/flagship4b-speedjerk-30k/ckpt.pt
step=29999 (TRAINABLE)` (`…/incoming/2026-07-23-v41-10k-gate/v4.1_train.log:1`). ✅

> 🔬 **The root-cause through-line — HYPOTHESIS (with MEASURED support), tied to the pending v4.2b test.**
> Every warm-start v4 arm stresses the world model: coupling a *new* anchored-diffusion planner's
> gradient into v1's **already-prediction-converged** WM yanks it off-manifold (v4 hot-trunk canary
> **0.452 → ~1.3**; v4.1 avoided it only by *starving* the planner; v4.2 protects the planner but the WM
> canary degrades to **0.7222**). v1 avoided this by co-evolving WM + planner **jointly from scratch**
> (canary held **0.42**). **HYPOTHESIS: the degradation is a warm-start artifact, not intrinsic.**
> MEASURED support: the from-scratch smoke-loop co-evolves the WM from random init with **no collapse**
> (canary 1.52 → 1.165, §1.5.5). The pending **v4.2b Phase-B canary** is the discriminating test: if a
> floored-λ planner still breaches the canary, the coupling is *not* floor-tunable ⇒ from-scratch.

---

#### 1.5.1 flagship-v4 (original, hot trunk) — `flagship-v4-30k` — ❌ **KILLED ~step 3,500**

| Field | Value |
|---|---|
| **Status** | ❌ **KILLED / superseded by v4.1** — Sayed-authorized restart 2026-07-22 ~18:10 local, after Phase-B showed the WM degrading. No held-out gate eval was ever run (killed before the 10 k gate). |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship-v4-30k/`. Launched 2026-07-22 16:00 local, PID **75844**, step-0 canary baseline **0.42148**. |
| **Distinguishing lever** | **`--lr-trunk 3e-4` (hot)**; batch 16, accum 1 (eff batch 16); λ_plan `sched`. |
| **Result** | ⚠️ **No eval JSON exists** (never gated). MEASURED **in-loop** (trainer WM-integrity canary, NOT a held-out headline): canary ran **0.452 → ~1.3 by ~step 3,500** (`V4_FROMSCRATCH_LAUNCH.md` §0), oscillating with peaks creeping up; WM loss rose **2.3 → 4.24** — and it kept degrading with the planner gradient fully clamped (`lam_mult=0`), so the **hot trunk LP-FT itself** (lr_trunk 3e-4), not the planner, was the culprit. *(In-loop canary ≠ eval output — quotable only as the kill trigger, per CLAUDE.md C1.)* |
| **Exact command** | `PYTHONPATH=/workspace/TanitAD/stack python scripts/train_flagship_v4.py --train-cache …/physicalai-train-e438721ae894 --val-cache …/physicalai-val-0c5f7dac3b11 --trunk …/flagship4b-speedjerk-30k/ckpt.pt --anchors-dense …/flagship_v4_anchors_dense.pt --out …/flagship-v4-30k --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 --strategic full --long-horizon-k 50 --steps 30000 --gate-step 10000 --batch 16 --lr-head 1e-4 --lr-trunk 3e-4 --eval-every 500 --save-every 1000 --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda` (LOOP_STATE launch record). |
| **HF** | none |

---

#### 1.5.2 flagship-v4.1 (lr_trunk 3e-5) — `flagship-v4.1-30k` — 🟥 **10 k GATE: primary FAILS** (2026-07-23)

| Field | Value |
|---|---|
| **Status** | 🟥 **10 k gate primary FAILS**; superseded by v4.2/v4.2b. WM stayed healthy; the **planner** was gradient-starved. Sayed decision pending (kill, or bank the healthy WM to 30 k) — not killed unilaterally. |
| **Location** | trained `tanitad-pod2:/workspace/experiments/flagship-v4.1-30k/`, PID **79542**. Gate ckpt `ckpt_step10000.pt` (**3,243,109,310 B**, md5 `8ae1ca6890bc73c7c32816ab6a4228fb`) → read-only eval copy `tanitad-eval:/root/models/flagship-v4.1-10k/ckpt_step10000.pt`. ⚠️ **single pod disk** — HF-back it once a transfer path is chosen. |
| **Distinguishing lever** | **`--lr-trunk 3e-5`** (10× cut from v4's 3e-4) + the canary controller ran **naive halve-to-zero** (a controller *bug* vs the design's cap-and-hold/O-14): `lam_mult` decayed 2.4e-4 → **1.5e-5** by step 10 k, i.e. the planner gradient was ≈ OFF since ~step 2,000. Batch 16, accum 1 (eff batch **16**). |
| **Params** | shared v4 line: **≈ 247,878,786** (see §1.5 preamble). |
| **Data** | `physicalai-train-e438721ae894`, skip-hash `f09e44db` (strict parity); val `physicalai-val-0c5f7dac3b11`. |
| **Exact command** | as §1.5.1 with `--out …/flagship-v4.1-30k --lr-trunk 3e-5` (no `--accum` → eff batch 16). Config: `taniteval/results/trainlogs/flagship-v4.1-10k_config.json`. |
| **Code state** | `stack/scripts/train_flagship_v4.py`, `stack/tanitad/models/flagship_v4.py` (STAGED). |
| **HF** | none |

**Results — step 10,000 (`flagship-v4.1-10k`), 881 windows / 40 episodes** ✅ *(read from
`taniteval/results/flagship-v4.1-10k.json`, produced by `eval_flagship_v4.py` MODE B, gate stream
`a938e1c0`; harness validated first — MODE A on v1 = 0.42148 vs registry 0.4271)*

| Metric | episode-cluster bootstrap (PRIMARY) | full-set | gate bar | |
|---|---|---|---|---|
| ADE@0.5s | 0.2376 [0.2146, 0.2601] | 0.2376 | — | |
| ADE@1s | 0.4075 [0.3643, 0.4521] | 0.4075 | — | |
| ADE@1.5s | 0.6304 [0.5591, 0.7073] | 0.6304 | — | |
| **ADE@2s (`ade_0_2s`)** | **0.8522 [0.7468, 0.9800]** | **0.8522** | ≤ 0.60 | 🟥 **FAIL** (~1.9–2.0× v1's 0.4271; CI entirely above the bar) |
| FDE@2s | 1.5176 [1.2563, 1.8213] | 1.5176 | — | |
| **miss@2m (`miss_at_2m`)** | **0.2486 [0.1714, 0.3379]** | 0.2486 | ≤ 0.10 | 🟥 **FAIL** |
| **`oracle_in_fan`** (4wp best-in-256-anchor) | **0.4838** | — | ≤ 0.30 | 🟥 **FAIL** (worse than v1.5-`ab`'s *frozen*-trunk 0.3073 → the KILL condition; dense-20 oracle 0.3603 also fails) |
| **`wm_canary_ade_2s`** (plan-free WM) | **0.4599** | — | ≤ 0.55 | ✅ **PASS** (v1 base 0.452, Δ+0.008 ≈ unchanged → **WM HEALTHY**) |
| `seam_norm_ratio_max` | 0.1796 | — | ≤ 1.0 | ✅ PASS |
| `encoder_touching_levers` | 2 | — | ≤ 2 | ✅ PASS (door closed) |

**Where the failure lives (paired episode-cluster bootstrap vs floors, from `driving_flagship-v4.1-10k.json`):**
aggregate ADE **ties** CV (Δ −0.0145 [−0.1508, +0.1448], not separated) — the point estimate 0.8522 is
even slightly worse than CV 0.8377 / hold-v0 0.7876. **Speed/longitudinal is decisively worse than every
trivial floor:** speed-MAE vs CV Δ **−0.3662** [−0.4908, −0.2446] (separated, favours floor); steady-cruise
speed (n=639) vs hold-v0 Δ **−0.5593** [−0.6482, −0.4689] (separated, favours floor). Straight-road heading
8.25° vs CV 1.399° (separated worse). **The one genuine win: speed-decoupled path GEOMETRY beats CV**
(Δ +0.1145 [+0.0171, +0.24], separated, favours model). ⇒ **the fault is the PLANNER (speed/selection),
not the WM** (consistent with the canary PASS).

**Pre-registered gate — `Project Steering/Gates/flagship-v4.card.json`, adjudicated by
`stack/scripts/run_gate.py check` → `Project Steering/Gates/flagship-v4-gate-10k-2026-07-23.json`:**

| verdict field | value |
|---|---|
| formal machine verdict | **`INCOMPLETE`** — 3 of 8 KILL secondaries have **no emitter anywhere in the codebase** (`speed_benefit_recovered_frac`, `deploy_tick_p99_ms`, `nonav_route_beats_majority`; the last needs the un-landed strategic ROUTE head) |
| primary | `ade_0_2s` 0.8522 ≤ 0.60 → **pass: false** |
| measurable KILL secondaries | 3 PASS (`wm_canary`, `seam_norm_ratio_max`, `encoder_touching_levers`) / **2 FAIL** (`oracle_in_fan`, `miss_at_2m`) |

### 🟥 **VERDICT: formal `INCOMPLETE`, substantively FAIL.** The primary fails outright (CI [0.7468, 0.98] sits **entirely above** the 0.60 bar — not a marginal miss) and 2 of the 5 measurable KILL secondaries fail. Restart budget **0/2** for lever family `joint-planner-wm` (nothing forces `REFUTE_LEVER_FAMILY`; nothing supports `CONTINUE`). Reads as `RESTART`-shaped once the 3 missing instruments are accepted as open — **Sayed's call**.

> ⚠️ **Discrepancy noted (raw JSON wins):** `LOOP_STATE.md` shorthands this gate as "FAIL". The raw
> `run_gate.py` verdict is **`INCOMPLETE`** (`pass:false` on the primary + 3 unmeasured secondaries).
> Both are true at different resolutions: formally INCOMPLETE, substantively a decisive primary FAIL.
> The gate-completeness gap (3 secondaries with no v4 producer) is itself a reconstruction risk — see
> `…/incoming/2026-07-23-v41-10k-gate/STATUS_BLOCKED.md`.

> 🔴 **Do NOT quote the in-loop `train.log` numbers as the gate primary** (`val.ade@2s 0.7054 /
> oracle 0.3598 / miss 0.2486` at step 10 k). Those are a **dense-20-step mean** (0.1–2.0 s) — a
> *different* convention that dilutes the 2 s endpoint and reads **lower** than the 4-waypoint
> `ade_0_2s`. C1 class: a trainer-log number is not an eval number. The held-out 4wp `ade_0_2s`
> (0.8522) is the quotable figure.

---

#### 1.5.3 flagship-v4.2 (cap-and-hold, floor 0.25) — `flagship-v4.2-30k` — ❌ **superseded @ ~step 5 k**

| Field | Value |
|---|---|
| **Status** | ❌ **superseded by v4.2b** (killed, both ckpts preserved). Interim @ step 4,000 measured to decide continue-vs-restart before its own 10 k gate. |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship-v4.2-30k/` → eval copy `tanitad-eval:/root/models/flagship-v4.2-step4000/ckpt.pt` (relay md5 `c42ae39cfbd6afd4aae58e5713d05d67`). |
| **Distinguishing lever** | canary controller **cap-and-hold, floor 0.25** (`--lam-mult-floor 0.25`; `config.json` `canary_controller.kind = "cap-and-hold-floor (v4.2 fix for v4.1 halve-to-zero)"`) + **`--lr-trunk 1e-4`** (raised back from v4.1's 3e-5) + **eff batch 64** (batch 16 × **accum 4**, matching v1). |
| **Exact command** | as §1.5.1 with `--out …/flagship-v4.2-30k --batch 16 --accum 4 --lr-trunk 1e-4 --lam-mult-floor 0.25`. Config: `taniteval/results/trainlogs/flagship-v4.2-step4000_config.json`. |

**Results — INTERIM step 4,000 (`flagship-v4.2-step4000`), 881 windows** ✅ *(read from
`taniteval/results/flagship-v4.2-step4000.json`, `eval_flagship_v4.py` MODE B)*

| Metric | value | note |
|---|---|---|
| **ADE@2s (`ade_0_2s`)** | **0.9869 [0.8795, 1.1088]** (4wp cluster bootstrap; full-set 0.9869) | worse than v4.1@10k's 0.8522 at **<½ the steps** |
| **`wm_canary_ade_2s`** | **0.7222** | 🟥 breaches the KILL bar (≤ 0.55) — the floor protects the planner **at the WM's expense**; independent harness canary matched the in-loop log to the digit (0.72224) |
| miss@2m | 0.2940 [0.2216, 0.3716] | |
| oracle (4wp) | 0.5009 | |

**Read:** v4.2@4 k is worse than v4.1@10 k on **every** measured axis (0.9869 vs 0.8522 primary; 0.7222 vs
0.4599 canary) — not a "needs more time" pattern. In-loop canary trend **0.86@2k / 0.72@4k / 0.77@5k**.
Per the pre-registered rule this **confirms "floor too high"** → v4.2b (floor 0.15) warranted.

---

#### 1.5.4 flagship-v4.2b (floor 0.15) — `flagship-v4.2b-30k` — 🟡 **LIVE / PENDING** (do not quote a number)

| Field | Value |
|---|---|
| **Status** | 🟡 **LIVE, in-flight** on `tanitad-pod2` (streams table PID **99197**; v4.2 killed, both ckpts preserved; fresh warm-start from v1). As of `LOOP_STATE` LAST_UPDATED 2026-07-23: ~step 900, Phase A, in-loop canary 0.495 (= v4.2's Phase A — indistinguishable until λ_plan ramps at step 2000). |
| **Distinguishing lever** | canary controller floor lowered to **0.15** (`--lam-mult-floor 0.15`); otherwise byte-identical to v4.2 (eff batch 64, lr_trunk 1e-4, λ_plan sched). |
| **Result** | 🟡 **PENDING — no held-out eval; DO NOT fabricate a number.** THE TELL is the **Phase-B canary @ steps 2500–3000**. Pre-registered rule (v4.2 hit 0.86/0.72/0.77): **≤0.55 & <v4.2 & gnorm_pred↑ → PASS**, continue to 10 k; **≥0.65 (~v4.2) → FAIL**, fire from-scratch; **0.55–0.65 → floor 0.10 or pivot** per planner trend. |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship-v4.2b-30k/` (in progress). |

---

#### 1.5.5 flagship-v4 from-scratch fallback — `flagship-v4-fromscratch` — ✅ **READY, not launched**

| Field | Value |
|---|---|
| **Status** | ✅ **CODE STAGED + VALIDATED, NOT LAUNCHED.** Fires **only** if v4.2b's Phase-B canary runs away, and only on Sayed's go. Zero GPU-day committed. Spec: `…/incoming/2026-07-23-v4-fromscratch/V4_FROMSCRATCH_LAUNCH.md`; artifact `a05a5c9e`. |
| **Distinguishing lever** | **`--from-scratch`** = skip `_warmstart_trunk`, random-init the trunk (the not-frozen gate then passes trivially). Byte-identical to v4.2b in every other flag (eff batch 64, lr_trunk 1e-4, floor 0.25, λ_plan sched) → **one-lever attributability**: if v4.2b's canary runs away and this one does not, the warm-start coupling is confirmed as the cause. |
| **Validation** | MEASURED: full `pytest -q` **786 passed / 2 skipped** + 5 new from-scratch tests (14 passed in `test_train_flagship_v4.py`). `--smoke-loop` from random init: canary baseline **1.5189** → drops monotonically **1.385 → 1.312 → 1.232 → 1.179 → 1.165** as the WM co-evolves — **improves, does not collapse** (the existence-proof for the through-line above). |
| **Cost** | **~53 h / 30 k** (ESTIMATED, MEASURED basis: v4.1 ~1.57 s/step at accum 1 → ~6.3 s/step at accum 4 → ~52.5 h + eval overhead). Gate read at step 10 000 (~18 h). |
| **Code state** | `--from-scratch` / `--trunk none` sentinel in `stack/scripts/train_flagship_v4.py` (STAGED); tests in `stack/tests/test_train_flagship_v4.py` (STAGED). |

> 🟥 **GATE-COMPLETENESS / RECONSTRUCTION RISK (the whole v4 line).** The v4 held-out eval **driver was
> the standing blocker** — it did not exist until `eval_flagship_v4.py` was built + O-03-validated on
> 2026-07-23. **3 of 8 pre-registered KILL secondaries still have no emitter** anywhere in the codebase
> (`speed_benefit_recovered_frac`, `deploy_tick_p99_ms`, `nonav_route_beats_majority`), so **no v4 gate
> can render a *complete* formal verdict today** even with a held-out primary. The `run_gate.py`
> comparative matched-step-ratio path is also dead for v4: `g_op_fwd_ade_m` is computed in
> `v4_loss_step` but never reaches `train_log.jsonl` (whitelist gap in `_training_loop`). Checkpoints
> live on single pod disks (v4.1 3.24 GB, v4.2 both on pod2) and are **not HF-backed**.

---

### 1.6 Flagship variants that exist but are not "versions"

| Run | What it is | Where |
|---|---|---|
| `p0-sB01-realmix` | the pre-reset 2-corpus (comma+PAI) realmix run; source of the **27 k D1/D2/D3 gate ladder** and the 3000-sample spectral fit; **stale since 2026-07-12 @ step 28,600** — the `gate-eval` skill still targets it ⚠️ | pod1 `/workspace/experiments/p0-sB01-realmix/` |
| `axis6-clean` / `axis6-relaxed` | `flagship4b_reduced` (d384, encoder 8, ~53 M model / 66 M trainable) A/B pre-check of `--sigreg-free-dims 64` vs `0` on comma2k19 | pod1 `/workspace/experiments/` |

---

## 2. REF-A — the frozen-encoder arm (H4)

**Shared:** frozen **DINOv2-B/14** features (224 px, 16×16 grid, dim 768) precomputed once; only the
adapter + predictor(s) train. `refa4b_config()` returns the **identical** `StackConfig` as
`flagship4b_config()` — REF-A consumes only the non-encoder fields, so **the flagship and REF-A differ in
exactly two things: (1) the encoder, (2) the SIGReg target** (`pred_only` vs `full_relaxed`).

Trainers: `stack/scripts/refa_train.py` (base, `--adapter pool|grid`) → `stack/scripts/refa_train4b.py`
(4-brain from features) → `stack/experiments/reset-speed4b/refa_train_plus.py` (**the one that trained
every post-reset arm**; `--adapter pool|grid|temporal`, `--speed-input`, `--yaw-input`, `--dyn-input`,
`--four-brain`, `--aux-egomotion`, `--aux-accel`, `--ego-dropout`, `--d-dino`, `--anchor-tactical`).

| Run | Steps | `action_dim` | adapter | key flags | Result (ADE@2s heldout) | Status |
|---|---|---|---|---|---|---|
| `refa-phase0-30k` | 30 k | 2 | grid | rollout_k 4 | **3.726 / "3.73"** | superseded — the pre-fix baseline |
| `refa-plus-speed-30k` | 30 k | 3 | temporal | `--speed-input --aux-egomotion --aux-accel`, rollout_k 12 | in-training fwd-ADE **3.73 → 0.83**, speed-R² 0.61 → 0.965 | the **isolated proof** of the speed fix — preserve |
| `refa-4brain-speed-30k` | 30 k | 3 | temporal | `+ --four-brain` | **2.1322 ± 0.1821** | the canonical "REF-A DINOv2 4B" |
| `refa-4brain-speedyaw-30k` | 30 k | 4 | temporal | `+ --yaw-input` | 🟥 no TanitEval record | superseded by dyn-in |
| `refa-ijepa-4brain-speed-15k` | 15 k | 3 | temporal | `--d-dino 1280`, I-JEPA ViT-H/14, **320 eps** | fwd-ADE 3.194 (vs DINOv2 3.796) @15k; best 2.816 @7k | ⚠️ **val-leaked, see below** |
| `refa-dino320-4brain-speed-15k` | 15 k | 3 | temporal | `--d-dino 768`, 320-ep DINO feats | the matched control for the I-JEPA arm | diagnostic |
| `refa-dynin-4brain-30k` | 30 k | 4 | temporal | `+ --dyn-input --ego-dropout 0.25` | **2.9196 ± 0.3937** | the H4 **final answer** |

---

### 2.1 REF-A **dinov2-4b** — `refa-4brain-speed-30k` — the canonical frozen-encoder reference

| Field | Value |
|---|---|
| **Status** | ✅ **ACCEPTED AS REFERENCE** (ceiling proven; see §8 D-A5) |
| **Location** | `tanitad-pod3:/workspace/experiments/refa-4brain-speed-30k/` · eval `tanitad-eval:/root/models/tanitad-refa-dinov2-4b/ckpt.pt` |
| **Architecture** | frozen DINOv2-B/14 + **temporal adapter** → 4-brain (`flagship4b` brains verbatim), predictor `d768 × depth 10, 12 heads, window 8, horizons (1,2,4), action_dim 3` ✅ |
| **Training args (from `config.json`)** | `--data-root /root/phase0_dinofeats --steps 30000 --rollout-k 12 --batch 64 --lr 3e-4 --warmup 500 --invdyn-weight 2.0 --fwd-weight 1.0 --pose-scale 10.0 --fwd-step-weight 0.5 --adapter temporal --aux-egomotion --aux-speed-weight 1.0 --aux-yaw-weight 1.0 --aux-accel --aux-accel-weight 1.0 --scale-weight 0.5 --jerk-weight 0.02 --speed-input --four-brain --log-every 50 --save-every 500 --seed 0` ✅ |
| **Trainer** | `stack/experiments/reset-speed4b/refa_train_plus.py` (committed, archive copy) |
| **Results** | ADE@2s **2.1322 ± 0.1821** heldout / **2.1675** full-set · FDE 3.2619 · miss 0.6245 · **does not beat CV** ✅<br>Pod-side gate (`refa4b_gate_30k.json`, different harness): 2.1355 ± 0.1963 / 2.1688 — cross-checks to within 0.003 m. |
| **Mid-run** | 14 k = 2.05 → 30 k = 2.14 → **plateaued** |
| **HF** | `Sayood/tanitad-refa-dinov2-4b` |

---

### 2.2 REF-A **ijepa-4b** — `refa-ijepa-4brain-speed-15k`

| Field | Value |
|---|---|
| **Status** | ⚠️ **DIAGNOSTIC ONLY — the canonical-val number is unusable** |
| **Architecture** | frozen **I-JEPA ViT-H/14**, `d_dino = 1280`, otherwise identical to §2.1 |
| **Args** | as §2.1 but `--data-root /workspace/tmp/ijepa_feats --steps 15000 --d-dino 1280 --yaw-input false` ✅ |
| **Results** | fwd-ADE **3.194 vs DINOv2's 3.796 at 15 k**; its own best was **2.816 @ 7 k** (7 k beats 15 k = overfit) |
| 🟥 **Why the number is unusable** | `taniteval/registry.py` records: *"320-ep variant … Canonical val 80 % LEAKED into its train set → guard excludes; clean number lives on the f1b378 val (pod3 gates)."* Both arms also overfit hard at 320 eps, so the I-JEPA-beats-DINOv2 read is an **overfit-regime ranking with data binding, not a feature-quality verdict.** ✅ |
| **HF** | `Sayood/tanitad-refa-ijepa-4b` |

---

### 2.3 REF-A **dyn-in** — `refa-dynin-4brain-30k` — the H4 final answer

| Field | Value |
|---|---|
| **Status** | ✅ **COMPLETE (step 29999).** The last frozen-encoder attempt; its result closed H4. |
| **Location** | `tanitad-pod3:/workspace/experiments/refa-dynin-4brain-30k/` — with **milestone ckpts at 5 k / 15 k / 20 k / 30 k** (D-032) |
| **Distinguishing flags** | `--dyn-input` (ego `[v0, yr0]` → `action_dim 4`) `--ego-dropout 0.25` `--four-brain --speed-input` |
| **Architecture** | frozen DINOv2-B/14 + temporal adapter; predictor `d768 × depth 10, 12 heads, window 8, horizons (1,2,4), **action_dim 4**` ✅ |
| **Args (`config.json`)** | `--data-root /root/phase0_dinofeats --out /workspace/experiments/refa-dynin-4brain-30k --steps 30000 --rollout-k 12 --batch 64 --lr 3e-4 --warmup 500 --invdyn-weight 2.0 --fwd-weight 1.0 --pose-scale 10.0 --fwd-step-weight 0.5 --adapter temporal --aux-egomotion --aux-speed-weight 1.0 --aux-yaw-weight 1.0 --aux-accel --aux-accel-weight 1.0 --scale-weight 0.5 --jerk-weight 0.02 --speed-input --dyn-input --ego-dropout 0.25 --d-dino 768 --four-brain --seed 0` ✅ |
| **Code state** | commit **`35956b2`** ("guarded yaw-rate conditioning for the frozen-DINO REF-A arm — SUPPLY dynamics + anti-shortcut ego-dropout, keep DINOv2 frozen"); milestone archiving **`6808c2d`** |
| **Final training metrics** | step 29999: `fwd_ade 0.6489` (train), `aux_speed_r2 0.9825`, `aux_yaw_r2 0.7575`, `aux_accel_r2 0.7569`, `man_acc 0.8438` ✅ |
| **HF** | none (eval copy scp'd to `tanitad-eval:/root/models/refa-dynin-30k/`) |

**Results — 881 windows** ✅ *(`results/refa-dynin-30k.json`)*

| Metric | heldout | vs flagship-30k |
|---|---|---|
| ADE@0.5s | 1.2680 ± 0.1657 | 0.0762 |
| ADE@1s | 1.8201 ± 0.2440 | 0.1584 |
| ADE@1.5s | 2.3650 ± 0.3209 | 0.2883 |
| **ADE@2s** | **2.9196 ± 0.3937** (full-set 3.047) | 0.4522 |
| FDE@2s | 4.5832 | 0.9437 |
| miss@2m | 0.7246 | 0.0602 |

Paired A/B (881 windows): flagship wins **95.9 %**, Δ **+2.62 m, CI95 [2.447, 2.798]**.

**The overfitting question — answered NO** (this is why D-032 milestone archiving matters): the curve is
**monotonically improving**, 5 k **3.755** → 15 k **3.694** → 20 k **3.016** → 30 k **2.920** (best is
last). Held-out error is not rising, so REF-A is **not overfitting — it is at a capability ceiling.**

**Failure signature:** long-RMSE 6.21 m / lat-RMSE 1.54 m → **94.2 % longitudinal**; speed bias +0.77 m/s;
overshoot +1.53 m; train fwd-ADE 0.65 → held-out 2.92 (**4.5× generalization gap**). Earlier ablation on
the pre-fix arm: `vision_use` **3.4 %**, imagination 1.5 % → "a dynamics integrator", earning ~96 % of its
accuracy from integrating `v0`.

---

### 2.4 The "4-brain variant" — clarification

There is **no separate model called "the 4-brain REF-A."** `--four-brain` is a *flag* on
`refa_train_plus.py` that swaps the bare predictor for the full flagship brain stack
(`FeatureWindowDataset4B` + the ported `flagship_loss`). It is **on** for `refa-4brain-speed-30k`,
`refa-4brain-speedyaw-30k`, `refa-dino320-…`, `refa-ijepa-…` and `refa-dynin-…`; **off** for
`refa-phase0-30k` and `refa-plus-speed-30k`. Given by hand during the 2026-07-14 reset and
CPU-smoke-validated before launch.

---

## 3. REF-B — hierarchical vision→action, **NO world model** (H1/D4 control)

**Shared** (`refb_config()`, `stack/tanitad/refs/refb.py`) — budget-matched to the flagship within ±2 %:

```
encoder      the SAME ViTEncoder class, trained from scratch: 9-ch 256 px patch16, d768 × depth 25
             (the ~130 M freed by having no predictor/imagination buys 25 blocks instead of 14)
readout      spatial grid 4, d_readout 128 ;  window 8
operative    d768 × depth 6, 12 heads, action_dim 2, action_seq 5 (0.5 s DIRECT heads, no recursion)
tactical     d512 × depth 6, 8 heads, 5 maneuvers, wp (5,10,15,20), d_intent 256   ← rev2 depth 4→6
strategic    d384 × depth 4, 6 heads, 4 nav cmds, d_cmd 128, d_ctx 256, n_route 3  ← rev2: real transformer
fallback     ConfidenceHead (hidden 512, fully DETACHED) + FeatureOOD (frozen buffers, 0 trainable)
optimizer    read PROGRAMMATICALLY from base250cam_config().train — lr 3e-4, wd 0.05, betas (0.9,0.95),
             warmup 2000, cosine.  Loss weights: action 1.0, seq 1.0, wp 1.0, man 0.5, route 0.5,
             inv 0.5, conf 1.0, route_ce_clamp 10.0
```

REF-B **structurally cannot** do imagination-error (D8), latent rollout (LOPS/SC-02), imagine-and-select
(D4), or closure reasoning (SC-01). That is the point of the arm. It is also **architecturally excluded**
from the closed-loop harness (no operative latent predictor + metric step-readout).

| Version | Run dir | Distinguishing flags | Params (total) | ADE@2s | Status |
|---|---|---|---|---|---|
| **v1 initial (4-layer)** | `refb-phase0-30k` | none (`speed_input=false`) | 262,509,213 | **0.8682 ± 0.0817** @6 k | superseded |
| **rev2 (2026-07-11)** | *(architecture revision, not a separate run)* | strategic → real d384×4 transformer + per-window nav derivation + route aux CE; tactical 4→6 | 260.7 M (rev1 `e616b23`), −0.124 % at rev2 `38cf9ca` | 🟥 never separately evaluated | folded into v1 |
| **speed-input reset (07-14)** | `refb-speed-30k` | `--speed-input --jerk-weight 0.02` → `speed_input`, `aux_accel` | 262,771,870 | **0.8255 ± 0.0992** @10 k | superseded |
| **refbpatch (07-17)** | `refb-refbpatch-30k` | `--refbpatch` → `+ aux_yaw`, `ego_dropout 0.5`, `path_dists (2,5,10,20)` | 263,038,375 | 🟥 crashed ≈step 500 | dead |
| **v2 / arch-v2 (07-18)** | `refb-refbpatch-v2-30k` | `--arch-v2 --refbpatch` | **271,619,880** | **0.5921 ± 0.0685** @29999 | ✅ **FINAL** |

---

### 3.1 REF-B v1 — `refb-phase0-30k`

Launched by the chained supervisor `/workspace/refb_pipeline.sh` (pod1), which gates the launch on
reproducing `physicalai-train-e438721ae894`. Params: encoder 179,263,616 · operative 51,731,724 ·
tactical 21,685,517 · strategic 8,385,027 · fallback 1,443,329 → **262,509,213**. ✅
Result (`refb`, 6 k): ADE@2s **0.8682 ± 0.0817** heldout / 0.8629 full-set, FDE 1.7341, miss 0.3343 —
**does not beat CV**. Mechanism: rotation-gain in curves 0.03, **yaw-rate probe R² = 0.11 (yaw-blind)**.
HF: `Sayood/tanitad-refb-speed` (the lineage repo).

### 3.2 REF-B rev2 (2026-07-11)

An **architecture revision**, not a distinct trained artifact. Fixed a real defect: the strategic head had
been training on a constant `follow` command. rev2 gives it a genuine `d384 × 4` causal transformer,
per-window nav commands derived from 15–25 s of future heading (`refb_labels.nav_command`), and its own
auxiliary route-heading CE. Tactical depth 4→6, funded by encoder 27→25. Commits `e616b23` (rev1, 260.7 M,
−0.82 %, 204 tests) → `38cf9ca` (rev2, −0.124 %). 🟥 No standalone eval exists.

### 3.3 REF-B speed-input reset — `refb-speed-30k`

`--speed-input` sets `cfg.speed_input` **and** `cfg.aux_accel` together (gated: off = byte-identical
state_dict, old ckpts resume). Command:
`python scripts/refb_train.py --data-root /workspace/data/physicalai_phase0/_epcache --out /workspace/experiments/refb-speed-30k --steps 30000 --speed-input --jerk-weight 0.02 --grad-checkpoint --ood-warmup 2000 --save-every 500 --seed 0` ✅
Result (`refb-10k`): **0.8255 ± 0.0992** heldout / 0.8372 full-set, FDE 1.6714, miss 0.2641 — turns
+0.255 m better than the 6 k, straights slightly worse.

> ⚠️ **Lineage trap, documented:** `refb-speed-30k/ckpt_prepatch_step8500.pt` is **byte-identical (md5)**
> to `refb-speed-30k/ckpt.pt`. There is one checkpoint, and it is at **step 10,000**, not 8,500. The file
> name is wrong.

### 3.4 REF-B refbpatch — `refb-refbpatch-30k` (crashed)

`--refbpatch` bundles: `aux_yaw=True`, `ego_dropout=0.5`, `path_dists=(2,5,10,20)` and implies
`--speed-input` (`stack/scripts/refb_train.py:398-406`). Crashed at ≈step 500:
`RuntimeError: DataLoader worker exited unexpectedly — bus error — insufficient shared memory`, root cause
a MooseFS mmap slice crossing the worker boundary. Fixed by cloning mmap window tensors in `__getitem__`
(commit **`986b688`**) — the same fix that unblocked `--workers>0` and the REF-C launch.

---

### 3.5 REF-B **v2 (arch-v2)** — `refb-refbpatch-v2-30k` — ⭐ FINAL, 0.592

| Field | Value |
|---|---|
| **Status** | ✅ **COMPLETE at step 29999** (`metrics.json: {"final": {"step": 29999, …}, "steps": 30000}`, ckpt written 2026-07-19 20:19 UTC) |
| **Location** | `tanitad-pod:/workspace/experiments/refb-refbpatch-v2-30k/` · milestones `tanitad-pod:/root/refb_milestones/ckpt_step{5000,15000,20000}.pt` · eval copy `tanitad-eval:/root/models/refb-v2-30k/ckpt.pt` |
| **Distinguishing flags** | `--arch-v2 --refbpatch` (arch-v2 **implies** refbpatch) |
| **The v2 architecture delta** | `yaw_input=True` (**B2**: ego proprioception widened `v0` → `[v0, yr0]`) and `anchored_tactical=True` (**B1**: a DiffusionDrive/VADv2-faithful **time-anchored** tactical decoder replacing the unimodal `wp_heads`), with `anchor_space="time"`, `anchor_n=128`, `anchor_pool=4096`, `anchor_d=384`, `anchor_layers=4`, `anchor_heads=8`. Anchors are **FPS over real GT trajectory targets** built from the dataset at launch, not the synthetic default. ✅ |
| **Params** | encoder 179,263,616 · operative 52,256,526 · **tactical 30,270,742** (was 21.7 M) · strategic 8,385,667 · fallback 1,443,329 → **271,619,880** ✅ |
| **Exact command** | `/root/launch_v2.sh` on `tanitad-pod`:<br>`cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts nohup setsid python3 scripts/refb_train.py --data-root /workspace/data/physicalai_phase0/_epcache --out /workspace/experiments/refb-refbpatch-v2-30k --arch-v2 --refbpatch --jerk-weight 0.02 --steps 30000 --grad-checkpoint --save-every 500 --workers 4 --prefetch 2 --amp --milestone-dir /root/refb_milestones --seed 0` ✅ |

**Results** ✅ *(`results/refb-v2-30k.json` and `refb-v2-20k.json`)*

| Metric | @20 k | @30 k (29999) |
|---|---|---|
| ADE@0.5s | — | 0.1033 ± 0.0120 |
| ADE@1s | — | 0.2173 ± 0.0260 |
| ADE@1.5s | — | 0.3793 ± 0.0450 |
| **ADE@2s** | **0.6462 ± 0.0548** | **0.5921 ± 0.0685** (full-set 0.5904) |
| FDE@2s | 1.3050 | 1.2305 |
| miss@2m | 0.2132 | 0.2025 |

@20 k it became the **first REF-B to beat the CV floor in every speed stratum** — validating the
time-anchored proposal decoder. Final training metrics at 29999: `man_acc 0.75`, `route_acc 1.00`,
`anchor_acc 0.469`, `n_modes 33`, `aux_yaw_r2 −0.106`.

> ✅ **Conflict resolved this session.** Weekly report W33 recorded this run as *"DEAD at 22,600/30,000"*.
> That was a **stale-log misread**: `refb-refbpatch-v2-30k.log` stopped being written at 12:08 UTC while
> training continued; `ckpt.pt` and `metrics.json` are both timestamped 20:19 UTC with `final.step 29999`,
> and 7,400 steps × ~3.9 s/step ≈ 8 h exactly accounts for 12:08 → 20:19. **The run completed. 0.592 is
> real and is confirmed in the raw eval JSON.**

> 🟥 **RECONSTRUCTION RISK — REF-B v2 is UNCOMMITTED.** `--arch-v2`, `anchored_tactical`, `yaw_input`,
> `anchor_space` **do not exist in this repo.** Verified: `stack/tanitad/refs/refb.py` has no
> `anchored_tactical`; `stack/scripts/refb_train.py` has no `--arch-v2`. The only copies are
> **pod-side, on `tanitad-pod`**:
> - `/root/refb_train_v4.py` (38,802 B) and `/root/refb_v4.py` (35,975 B) — the authored versions
> - `/workspace/TanitAD/stack/scripts/refb_train.py` and `.../tanitad/refs/refb.py` — the working-tree
>   copies actually executed (`git status` on that pod shows both as `M`, on `main@0f93b98`)
> - originals backed up as `/root/refb_train_orig_backup.py`, `/root/refb_orig_backup.py`
>
> The eval side is equally pinned: `taniteval/registry.py` notes the v2 checkpoints need
> `TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2b`.
> **The best-scoring reference arm we have cannot be rebuilt from this repo today. Highest-priority
> commit.**

---

## 4. REF-C — Anchored-Diffusion-C (DiffusionDrive-style)

`stack/tanitad/refs/refc.py` (committed: `6025769` redesign, `7e9c402` sizing). Replaces the old TCP-C GRU
trajectory/control branches with a fixed **anchor vocabulary** whose queries cross-attend the conv feature
map, emitting per-anchor confidence + offset, optionally refined by truncated denoising. Anchors are built
by **furthest-point sampling** (not k-means — comma2k19 is ~74 % straight and k-means collapses onto the
straight mode). Kept from TCP-C: torchvision-free ResNet encoder, measurement encoder with per-sample
ego-dropout, the **LAW** latent-world-model aux, the strategic-ctx hierarchy graft, the REF-C.1
target-speed head. Grafts: `hierarchy`, `graft_maneuver` (H19 maneuver→anchor prior, live from step 0),
`graft_imagination` (H15 belief field — **XL only**).

**Three presets exist in code; only two were ever instantiated, and only one was really trained.**

| Preset | `--config` | Encoder | Decoder | Anchors | Imagination | Measured params | Trained? |
|---|---|---|---|---|---|---|---|
| `refc_small_config()` | `small` | base_width 64, blocks (3,6,16,6) | d256, 4 heads, 3 layers | 64 / pool 2048 | off | **54,690,001** ✅ | ✅ **yes** — `refc-diffusion-small-v21-30k`, 30 k complete + evaluated (§4.2) |
| `refc_config()` | `base` | base_width 88, blocks default | d384, 4 layers | 128 / pool 4096 | off | **104,191,577** ✅ *(measured 2026-07-20; the docstring's "~110 M" was 5.6 % high)* | ✅ **yes** — `refc-diffusion-base-v21-30k`, 30 k complete + evaluated (§4.3) |
| `refc_xl_config()` | `xl` | base_width 124, blocks (3,8,20,6) | d512, 8 heads, 6 layers | 256 / pool 4096 | **on** | **251,932,584** ✅ | ✅ **yes** |

---

### 4.1 REF-C-XL — `refc-diffusion-xl-30k` — ✅ **COMPLETE at step 29,999**

| Field | Value |
|---|---|
| **Status** | ✅ **FINISHED** on `tanitad-pod3` 2026-07-20 09:19 UTC at step **29,999 / 30,000**. GPU released. |
| **Location** | `tanitad-pod3:/workspace/experiments/refc-diffusion-xl-30k/` (source) · **final eval copy `tanitad-eval:/root/models/refc-xl-30k/ckpt.pt`, md5 `966d4eff1ea5ddf86efba01b8344e198`** (pulled after the trainer exited, so the file was quiescent) · superseded mid-training snapshots: `refc-xl-snap` (~16 k / 28 k) |
| **Architecture** *(from run `config.json`)* | encoder 9-ch 256 px, `base_width 124`, blocks (3,8,20,6) → 8×8×F map · measurement `{hidden 128, d_out 128}` · trajectory horizons (5,10,15,20) · anchors `{n 256, pool 4096, seed 0}` · decoder `{d 512, 8 heads, 6 layers, ff_mult 4, aux_hidden 512, diffusion_steps 2, noise_std 0.1}` · law `{hidden 2048}` · strategic `{hidden 768, d_ctx 96}` · imagination `{d 512, depth 6, 8 heads, ff_mult 4, head_hidden 1024}` · `ego_dropout 0.5` · `hierarchy true`, `graft_maneuver true`, `graft_imagination true`, `graft_target_latent false`, `grounded_selector false`, `refc1 false`, `path_dists (2,5,10,20)`, `speed_bins 4`, `speed_max 30.0` ✅ |
| **Params** | encoder 199,496,532 · measurement 17,280 · strategic 4,133,472 · decoder 22,702,345 · imagination 20,986,339 · aux 513,960 · law 4,082,656 → **251,932,584** ✅ |
| **Optimizer** | **Adam** (DiffusionDrive/TCP convention, *not* AdamW), lr **1e-4**, warmup 2000, cosine. Loss weights: traj 1.0, cls 1.0, law 0.5, route 0.1, man 0.1, speed_cls 0.2 ✅ |
| **Data** | `/workspace/pai_epcache` (pod3 copy of the same 2,376-ep parity set); anchors `/workspace/experiments/refc_anchors_full.pt` built by `stack/scripts/build_refc_anchors.py` |
| **Exact command** *(live `ps`)* | `cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 scripts/refc_train.py --data-root /workspace/pai_epcache --out /workspace/experiments/refc-diffusion-xl-30k --steps 30000 --mode diffusion --config xl --anchors /workspace/experiments/refc_anchors_full.pt --batch 20 --workers 6` ✅ |
| **Code state** | `stack/tanitad/refs/refc.py` + `stack/scripts/refc_train.py` are **committed in this repo** (`6025769`, `7e9c402`, 15 refc tests). ⚠️ On pod3 the file shows as **untracked** (`?? stack/tanitad/refs/refc.py`) — the pod predates the commit; verify the pod copy matches HEAD before claiming byte-parity. |
| **HF** | none |

**Results — FINAL step 29,999 (`refc-xl-30k`), 881 windows** ✅ *(read from the raw eval run 2026-07-20)*

| Metric | **FINAL 29,999** | 28 k (provisional) | ≈16 k snapshot |
|---|---|---|---|
| **ADE@2s** | **0.458 ± 0.057** | 0.470 ± 0.057 | 0.5645 ± 0.0447 |
| FDE@2s | **0.972** | — | 1.1076 |
| miss@2m | **0.146** | 0.154 | 0.1495 |
| TMS | 0.203 | — | — |

> ❌ **RETRACTED 2026-07-20 — "REF-C-XL finishes 0.006 m behind the deployed flagship v1 (0.4522)".**
> Wrong magnitude **and** wrong conclusion. The 0.006 m was a difference of *split-means* (0.4522 vs
> 0.4577); on the full 881 windows the gap is **0.0443 m — 8× larger**. The leaderboard was ranking on
> a statistic that **compresses between-arm differences**. Under the **paired episode-cluster
> bootstrap** (`taniteval/ci.py`, 2000 resamples over the 40 val episodes) the two arms are
> **NOT separated**: Δ(REF-C − flagship) **+0.0443 m, CI95 [−0.0544, +0.1465]**, P(Δ>0) = 0.809.
>
> **Correct statement: flagship v1 and REF-C-XL are statistically indistinguishable on ADE@2s.** A
> budget-matched direct-head diffusion arm ties the world-model stack. Leaderboard rows 1–2 are a tie,
> not an ordering — and per-window correlation is only 0.207 here, so pairing buys little power
> (1.02×); the tie is real, not an artefact of a weak test.
> Source: `Project Steering/CI_RECOMPUTE_2026-07-20.json`. **The `± ci95` values in the table above are
> the deprecated `overlapping_holdout_se`** (measured **1.28–2.06× too narrow** across 10 arms; see §7).
> Decision-grade: full-set **0.4714**, bootstrap CI95 **±0.0830** → **[0.3896, 0.5556]**.

> 🔬 **The selection flaw — REF-C ranks with the UN-refined anchor's score.**
> Read from `refc.py::AnchoredDiffusionDecoder.forward`: all 256 anchors ARE denoised (no
> top-K gate), **but selection uses the t=0 classifier score over the ORIGINAL anchors — the
> denoise passes return `_, off` and their own confidences are DISCARDED.** Geometry is
> refined; ranking is not.
>
> **CORPUS figures (n=881, canonical val — use THESE):**
> selected **0.4714** full-set · **oracle-in-fan 0.1640** · gap **0.3075 m** ·
> `frac_sel_2x_worse` **0.454**. ⚠️ An earlier revision of this section quoted
> *0.295 / 65 %* — those were **single-clip** (ep11, stride-1) values mis-stated as corpus
> figures. The ep11 illustration itself stands: ADE(selected) 2.572 m vs oracle 0.305 m (8.4×),
> and the raw vocabulary already held a 0.290 m plan, so it is neither a coverage nor a
> refinement failure.
>
> **⛔ THE ORACLE GAP IS ~92 % IRREDUCIBLE — stop quoting it as available headroom.**
> REF-C v1.2 settled this across **47 trained arms**: a learned re-scorer recovers at most
> **8.4 % of the gap on its own training data**. Not capacity (smaller heads are worse), not
> overfitting (dev tracks train). The 0.1640 oracle is a **minimum over 256 candidates scored
> against ONE realised future** — most of the distance below the incumbent is that minimum's
> *statistics over aleatoric outcomes*, not recoverable signal. An earlier revision of this
> section (mine) claimed "oracle within top-8 = 87 % of the gap, so fund the learned ranker".
> That framing was wrong: top-8 bounds where the *lottery* is least severe, not what is
> *learnable*. **Selection is no longer the productive lever on REF-C.**
>
> ❌ **REFUTED — do not add a target-speed term to the selection score.** REF-C v1.0 measured
> it: cost re-ranking recovers **0.0 %** (best blend point is λ=0, the unmodified baseline;
> pure cost −171 %). A **GT-perfect speed-matcher scores 1.1236, WORSE than baseline**, and a
> GT-perfect along-track-only ranker caps at 34 %. VTARGET sits +1.42 m/s above v0 and is a
> 10–20 s free-flow *aspiration* — used as a 2 s reference it is worse than holding v0
> (MAE 1.65 vs 0.475) and makes braking windows **+0.51 m worse**. Right quantity, wrong
> timescale.
>
> ⚠️ **Do NOT naively "score the refined trajectory" either.** Selecting on the discarded
> refined-pass confidence scores **1.36593 — 2.9× WORSE than baseline** — because
> `refc_train` never supervises the conf head at denoise timesteps, so that signal is
> *unsupervised noise*. This retroactively explains why flagship v1.5's version of the same
> fix degraded as its fan sharpened.
>
> **The two selection experiments, both settled:**
>
> | | approach | full-set ADE@2s | verdict |
> |---|---|---|---|
> | **REF-C v1.0** | hand-written cost re-rank, 0 new params | 0.4714 (λ=0 best) | **0.0 % recovered** — best blend point is the untouched baseline; pure cost −171 % |
> | **REF-C v1.2** | learned re-scorer, soft distance-weighted target, frozen decoder | **0.46251** vs 0.47144 | **+2.9 % of the gap; NOT significant** (paired Δ +0.00893, CI [−0.0062, +0.0250]) |
>
> v1.2's one clean win is over v1.0: **a learned ranker does what a hand-written cost provably
> cannot** — qualitatively real, quantitatively small. Also established there: **hard-argmin is
> the worst target in all five feature configurations** (pointwise ≈ warm-listwise >
> cold-listwise > hard), the frozen decoder embedding is nearly worthless (+3.01 % on geometry
> + frozen logit alone vs +3.61 % with it), and top-K is target-dependent (`regress` collapses
> on the full fan, `soft` tolerates it; K=8–32 a flat plateau).
>
> **Where the lever actually is now:** REF-C **proposes** ~2× better than flagship v1.5
> (oracle 0.164 vs 0.338) while v1.5 **mis-ranks** about half as often (0.235 vs 0.454). The
> two arms fail in opposite directions, so the open question is proposal quality and the
> architecture that produces it — not the ranker.
> Evidence: `taniteval/taniteval/plan_fan.py`, `taniteval/taniteval/refc_rerank.py`,
> `stack/tanitad/models/refc_rescorer.py`, `Benchmarks & Eval/PLANNER_VIZ_CONCEPT.md`,
> `Research/2026-07-20-refc-cost-rerank-tier0.md`,
> `Research/2026-07-20-refc-v12-learned-rescorer.md`.

**Strata — FINAL step 29,999** *(read live from `results/refc-xl-30k.json`; an earlier revision of this
row printed the ~16 k model column against the FINAL header — the CV baselines were right, the model
numbers were stale, which understated REF-C everywhere)*:

| stratum | model ADE@2s | CV | n |
|---|---|---|---|
| speed **high** | **0.3243** | 0.6468 | 294 |
| speed med | 0.4989 | 0.9345 | 293 |
| speed low | 0.5912 | 0.9322 | 294 |
| curv straight | 0.3865 | 0.4393 | 634 |
| curv gentle | 0.6751 | 1.3566 | 125 |
| curv sharp | 0.7040 | 2.3764 | 122 |

**Beats CV in every stratum, including straight** (0.3865 vs 0.4393 — the 16 k row had it LOSING there
at 0.523 vs 0.439; that ✗ was an artefact of the stale numbers). Overall full-set 0.47144, miss@2m
0.14188, FDE 1.00614, TMS 0.21351.

> 🔬 **The high-speed win is bigger than previously briefed.** Against flagship v1's 0.5513 in the same
> stratum, REF-C FINAL scores **0.3243** — not the 0.330-vs-0.551 quoted from the 28 k provisional. This
> is the stratum flagship is weakest in, and a direct-head diffusion arm beats the world-model stack
> there by ~41 %.
Evaluated through `taniteval.refc_eval` — REF-C has its **own** trajectory decoder, no grounded operative
rollout (`step_readout = None`).

---

### 4.2 REF-C-small (54.7 M) — `refc-diffusion-small-v21-30k` — ✅ **COMPLETE at step 29,999 · EVALUATED 2026-07-22** — closes the D-030 ladder

*(History: the prior `small` instantiation was a **150-step classifier smoke** at
`tanitad-pod3:/workspace/experiments/refc-smoke320/`, `param_breakdown.total = 54,690,001`, never trained
on the full set. As of 2026-07-22 the real 30 k run below **supersedes it** — small is now trained on the
2,376-ep parity set and evaluated on the canonical val.)*

**The ambiguity, stated precisely — three distinct claims are in circulation:**

1. **Task-brief / some docs:** *"REF-C small 54.7 M (DiffusionDrive-scale preset) vs XL ~252 M (the one
   actually trained, 0.565@16k)."* — correct on both sizes and on which was trained.
2. **A conflicting framing in the eval-path note** described the live run as *"diffusion-XL … ~54.7 M."*
   That is **wrong**: the live run passes `--config xl` and its own `config.json` books **251,932,584**
   params. `taniteval/registry.py` carries an explicit corrective: *"this is the XL scale arm, ~252 M —
   NOT the 54.7 M `small`/DiffusionDrive-scale preset."* ✅
3. **Internal drift in `small`'s own size:** commit `36d979f` introduced it as **~28 M**; commit `7e9c402`
   re-anchored it to **54.7 M** ("research-anchored, DiffusionDrive scale, per Sayed"); the docstring now
   says "~55 M, tests pin the 45–65 M band". The 54,690,001 measurement confirms the *current* code.

**RESOLVED 2026-07-22 — the bottom rung landed.** `refc-diffusion-small-v21-30k` (tanitad-pod2 A40,
30,000 steps, ~7 h 10 m, PID 57658). **Same command as base §4.3, only `--config small` + the 64-anchor
vocab differ; `--labels v21` held constant, so small-vs-base isolates SCALE with NO label confound.**
Parity proven live: **2,376 eps / 406,099 windows**, v21 label coverage **[0.121 / 0.5645 / 0.115 /
UNKNOWN 0.1995]** bit-identical to base, `param_breakdown.total = 54,690,001`. Anchors
`refc_anchors_small64.pt` = **base128[:64] == full256[:64]** (bit-exact nested FPS prefix, seed 0), so the
scale-A/B matched-vocabulary control nests vs both base and XL (`nested vocabulary: True` in both runs).
Eval **identical to base/XL**: `taniteval.refc_eval` on the canonical 40-ep / 881-window val, nav=follow,
2 truncated-denoise steps.

**Results — FINAL step 29,999 (`refc-small-30k`), 881 windows** ✅ *(raw:
`taniteval/results/refc-small-30k.json`, `scaleab_refc-small-30k_vs_refc-{base,xl}-30k.json`; repo copies
in `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-22-refc-small-30k/`)*

| Metric (full-set, episode-cluster bootstrap B=2000) | **small** (54.7 M) | base (104.2 M) | XL (251.9 M) |
|---|---|---|---|
| **ADE@2s (selected)** | **0.5261 [0.4295, 0.6262]** | 0.4728 [0.3835, 0.5699] | 0.4714 [0.3896, 0.5556] |
| FDE@2s | 1.1115 [0.9041, 1.3224] | 1.0031 | 1.0061 |
| miss@2m | 0.1714 [0.1168, 0.2281] | 0.1419 | 0.1419 |
| TMS-openloop | 0.159 | 0.1957 | 0.2135 |
| plan tick p50 fp32 | **11.50 ms** | 21.78 ms | 44.06 ms |

**Paired (small − X), same 881 windows:** vs base **+0.0533 [+0.0167, +0.0925] SEPARATED** · vs XL
**+0.0546 [+0.0189, +0.0940] SEPARATED**. Beats CV (0.8377) with margin, and in every stratum
(high 0.358 / med 0.549 / low 0.672 · straight 0.408 / gentle 0.813 / sharp 0.848).

**Oracle-in-fan (proposal quality) — the read the decision needs:**

| | small (64 anc) | base (128) | base@64 | XL (256) | XL@64 |
|---|---|---|---|---|---|
| oracle-in-fan | **0.2213** | 0.1914 | 0.2833 | 0.1640 | 0.4368 |
| sel_gap | 0.3048 | 0.2813 | 0.1895 | 0.3075 | 0.0346 |
| frac_sel_2x_worse | 0.3825 | 0.4109 | 0.2940 | 0.4540 | 0.1771 |

Paired oracle: small − base **full** +0.0299 SEPARATED (small worse — fewer anchors); small − base **@64
matched** **−0.0620 [−0.0801, −0.0435] SEPARATED — small BETTER**; small − XL@64 **−0.2155 SEPARATED —
small far better**. Oracle-over-first-K: small ≤ base ≤ XL at **every** shared K (4→64), i.e. the small
encoder's fan is the **tightest per-anchor**.

### ✅ **VERDICT — the ladder has a small knee, and it is ANCHOR-COUNT, not encoder scale.**
Small does **not** hold the base≈XL tie on the shipping metric: selected ADE@2s is **SEPARATED-worse than
both base and XL by ~0.053 m** (base≈XL was +0.0013, not separated — so small is the ladder's first
separation). **But the mechanism is decisive.** On the matched 64-anchor vocabulary small's fan is
**better** than base's and far better than XL's, and small's full-fan oracle (0.221) trails base's (0.191)
**only because small carries half the anchors** (64 vs 128) — anchors are ~0.05 MB buffers, not params.
The **2.4× encoder/param cut (48 M vs 90 M encoder) did NOT degrade proposal quality**; the smaller
encoder proposes at least as tightly per-anchor. This **extends §4.3 a full rung down: the fan lever is
anchor-vocabulary WIDTH, not encoder scale.**

**For v4's decoder budget:** REF-C's encoder is over-provisioned even at base — v4 can shrink the encoder
toward small's **~48 M with no measured loss of fan quality**, provided the anchor vocabulary stays wide
(**≥128**, nearly free). The selected-ADE knee is recoverable by anchor count, not encoder params.
small-vs-base is the **clean** scale test (shared v2.1 labels); small-vs-XL still carries the base/XL label
confound. Efficiency bonus: small's **11.5 ms** plan tick is 1.9× faster than base, 3.8× faster than XL,
at 12 % of the 100 ms budget. Evidence class: **MEASURED** (this run + eval, artifacts staged in-repo).

---

### 4.3 REF-C-base (medium, 104.2 M) — `refc-diffusion-base-v21-30k` — ✅ **COMPLETE at step 29,999 · EVALUATED 2026-07-21**

| Field | Value |
|---|---|
| **Status** | ✅ **FINISHED** on `tanitad-pod3` 2026-07-21 04:44 UTC at step **29,999 / 30,000** (`metrics.json` `final.step` 29999, `steps` 30000 — report it as step-29999). GPU released. Evaluated on `tanitad-eval` 2026-07-21 05:18–05:19 UTC under the `refc-base-eval` GPU lock. |
| **Location** | `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/` (source, + 5 k/15 k/20 k milestones) · **eval copy `tanitad-eval:/root/models/refc-base-30k/ckpt.pt`, md5 `8f10d6f934f4199e11ddc7352e074939`** (pod3→eval direct agent-forwarded scp, 1.25 GB in 70 s ≈ 17.9 MB/s, md5 identical both ends; the pod3 trainer had already exited so the source was quiescent) · TanitEval key **`refc-base-30k`** |
| **Params** *(measured at instantiation)* | encoder 90,458,632 · decoder 8,634,505 · strategic 1,903,680 · law 2,902,720 · aux 274,760 · measurement 17,280 · imagination **0** (graft off, XL-only) → **104,191,577** ✅ |
| **Parity with XL** | same corpus `physicalai-train-e438721ae894` (2,376 eps / 406,099 windows), 30 k steps, **Adam** lr 1e-4 / warmup 2000 / cosine, same loss weights, `--mode diffusion`, `--batch 20 --workers 6` |
| **Deliberate differences** | `--config base` (2.42× smaller) · **128** FPS anchors — verified a **strict prefix of XL's 256** (`refc_anchors_base128.pt`, same script/source/pool-cap/seed) · H15 imagination OFF (preset design) · **route labels v2.1** |
| **⚠️ Confound** | XL trained with **v1** route labels (`route_target(nav_cmd)` — circular *and* straight-by-default; `labels_v2` was never set in `refc_train.py`). This run uses **v2.1** (`route_from_future_v21`, `use_net_dyaw=False`, ROUTE_UNKNOWN=3 **masked** out of the 0.1-weight CE, never clamped). **medium-vs-XL therefore conflates scale and labels.** Calibration: the flagship v1.5 end-to-end label effect was +0.025 m, not CI-separated. |
| **Label coverage** *(4,000-window sample, in `config.json`)* | left 0.121 · straight 0.5645 · right 0.115 · **UNKNOWN 0.1995 (masked out)** → 80.05 % judgeable, vs v1's straight-by-default target |
| **Code** | `stack/scripts/refc_train.py` gained `--labels {v1,v21}` (**default `v1` = XL-reproducible**), `RouteV21Dataset`, a fail-loud masked route CE, and 5 k/15 k/20 k/30 k **milestone archiving** (the gate series XL lacks). 15/15 `tests/test_refc.py` pass. Pod3 drift repaired before launch (`refb_labels.py` still had `use_net_dyaw=True`; `ckpt_io.py` was absent) — backups in `/workspace/ops/backup-20260720-refcmed/`. |
| **Eval** | canonical `taniteval.refc_eval` path, **identical to XL**: n=881 windows / 40 val eps / `/root/valdata/physicalai-val-0c5f7dac3b11`, window 8 / stride 8, nav=follow, 2 truncated-denoise steps. Parity proven three ways: same 881 `eid`s, bit-identical GT, and **bit-identical CV baseline in every stratum** (0.6468 / 0.9345 / 0.9322 high/med/low, 0.4393 / 1.3566 / 2.3764 straight/gentle/sharp — the same numbers §4.1 prints for XL). Registry entry `refc-base-30k` added to `taniteval/taniteval/registry.py` with `config_preset="base"`. |
| **Note** | `TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-20-refc-medium-scaling.md` (pre-registered the reading rule) |

**Results — FINAL step 29,999 (`refc-base-30k`), 881 windows** ✅ *(raw: `taniteval/results/refc-base-30k.json`)*

| Metric | **REF-C-base** (104.2 M) | REF-C-XL (251.9 M) | paired Δ (base − XL), episode-cluster bootstrap |
|---|---|---|---|
| ADE@2s *(full-set, decision-grade)* | **0.4728** · CI95 [0.3835, 0.5699] | 0.4714 · [0.3896, 0.5556] | **+0.0013 [−0.0281, +0.0316] — NOT separated** |
| FDE@2s *(full-set)* | **1.0031** · [0.8148, 1.2087] | 1.0061 · [0.8301, 1.1875] | **−0.0030 [−0.0619, +0.0584] — NOT separated** |
| miss@2m *(full-set)* | **0.1419** · [0.0874, 0.2000] | 0.1419 · [0.0943, 0.1918] | **+0.0000 [−0.0261, +0.0272] — NOT separated** |
| TMS-openloop *(full-set)* | 0.1957 | 0.2135 | — |
| *legacy `heldout ±` (deprecated, for continuity with §4.1's published row)* | *0.4523 ± 0.0497* | *0.458 ± 0.057* | — |

> **Verdict: REF-C-base and REF-C-XL are statistically indistinguishable on everything that ships.**
> All three paired intervals straddle zero and the point deltas are ≤0.003 m — a 2.42× parameter cut
> and a **2.20× encoder cut** (90,458,632 vs 199,496,532) cost **nothing measurable** on this corpus.
> Per-window ADE correlation 0.789, so the pairing is doing real work (the test is not weak).
> Estimator: `taniteval/ci.py` episode-cluster bootstrap, B=2000 over the 40 val episodes; paired form
> for the deltas. Reproduce from `taniteval/results/{windows,fan}_refc-{base,xl}-30k.pt` with
> `taniteval/refc_scale_ab.py analyze`.

**Strata — FINAL step 29,999** *(CV column is bit-identical to §4.1's, which is the parity proof)*:

| stratum | base ADE@2s | XL ADE@2s | CV | n |
|---|---|---|---|---|
| speed **high** | 0.3510 | **0.3243** | 0.6468 | 294 |
| speed med | **0.4483** | 0.4989 | 0.9345 | 293 |
| speed low | 0.6189 | **0.5912** | 0.9322 | 294 |
| curv straight | 0.3866 | 0.3865 | 0.4393 | 634 |
| curv gentle | 0.6778 | 0.6751 | 1.3566 | 125 |
| curv sharp | 0.7105 | 0.7040 | 2.3764 | 122 |

**Beats CV in every stratum**, including straight. The two arms trade strata (base wins med by 0.051,
XL wins high by 0.027 and low by 0.028) — no stratum-level ordering survives as a scale story.

**Fan quality — the read the decision actually needs** *(raw: `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`)*

| | base (128 anchors) | XL (256 anchors) | XL restricted to its first **128** |
|---|---|---|---|
| **oracle-in-fan** | **0.1914** [0.1654, 0.2184] | **0.1640** [0.1414, 0.1902] | 0.2624 [0.2262, 0.3011] |
| sel_gap (selected − oracle) | 0.2813 | 0.3075 | 0.2091 |
| `frac_sel_2x_worse` | 0.4109 | 0.4540 | 0.3190 |

*Paired:* base − XL(256) **+0.0275 [+0.0142, +0.0405] SEPARATED** (XL better) · base − XL(128)
**−0.0710 [−0.0965, −0.0502] SEPARATED** (base better).

> 🔬 **The fan lever is anchor-vocabulary WIDTH, not encoder scale.** base's 128 anchors are a
> **bit-exact prefix** of XL's 256 (verified at load: `max|A − B[:128]| = 0`), so the fans can be
> compared over the identical vocabulary. Oracle-in-fan over the first K anchors:
>
> | K | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
> |---|---|---|---|---|---|---|---|
> | **base** | 3.193 | **1.686** | **0.813** | **0.527** | **0.283** | **0.191** | — |
> | **XL** | 3.535 | 2.274 | 1.226 | 0.806 | 0.437 | 0.262 | **0.164** |
>
> base is at least as good at **every matched K**, and XL's entire oracle advantage arrives with
> anchors 129–256. Anchors are a **buffer, not parameters** (0.048 M of buffers total) and the decoder
> is only **~1.7 ms of base's 21.8 ms tick** (encoder 90.7 %), so widening the vocabulary is nearly
> free while widening the encoder demonstrably bought nothing here.
> ⚠️ **Two caveats, both real.** (1) A prefix restriction structurally penalises XL: its
> winner-takes-all training spread modes across 256 slots, so the interstitial anchors nearest ~half
> its targets are exactly what the restriction removes. Read the curve's shape, not one K. (2) The
> **label confound below is of the same magnitude as every oracle delta here** and points the same
> way as base — so "base's encoder proposes better" is NOT established; "XL's bigger encoder does not
> buy fan quality" is what the evidence supports.

> ⚠️ **CONFOUND — SCALE, ANCHOR COUNT AND LABELS MOVE TOGETHER.** base trained on route labels
> **v2.1**, XL on **v1** (row above). The matched-K control removes the anchor-count confound; nothing
> removes the label one. Calibration from flagship v1.5, the only place the label change was measured
> end-to-end: ADE **+0.025 m (not CI-separated)** but **oracle −0.058 m** — i.e. v2.1 labels
> *improved the proposal set* by more than either oracle delta measured here, and base is the arm that
> had them. **Do not present a clean scaling conclusion.** What IS separable: on ADE/FDE/miss the arms
> tie, so the label effect would have to be ≥0.03 m *and* exactly cancel a scale effect to hide one —
> possible but unevidenced. What is NOT separable: the sign and size of the encoder-scale effect on
> oracle-in-fan. The clean resolution remains one control run (XL-with-v2.1 or base-with-v1).

**Efficiency — batch 1, one A40, identical precision flags** *(raw: `results/eff_refc-base-30k.json`)*

| | base | XL | ratio |
|---|---|---|---|
| plan tick p50 fp32 / tf32 / amp16 | **21.78 / 15.81 / 15.88 ms** | 44.06 / 27.78 / 21.00 ms | **1.32–2.02× faster** |
| p99 fp32 | **22.33 ms** | 44.44 ms | meets 10 Hz in all 3 precisions (both arms do) |
| GFLOPs / peak MB | **292.5 / 556.7** | 702.2 / 1178.4 | 0.42× / 0.47× |
| encoder share of the tick | 90.7 % | 88.7 % | — |

**What this settles for v3.5** (`V35_DESIGN.md` §3.6 fires the "base ≈ XL" branch): (i) the decoder
geometry can be trimmed to base's **d384 / 4-layer / 128-anchor** (8.6 M vs XL's 22.7 M); (ii) ⭐
REF-C-**base**'s 90.5 M encoder is **validated** as the second-KV candidate that makes §2.3
alternative ② fit under the 400 M cap (≈348 M) — the XL form (≈457 M) stays over cap and now has no
measured accuracy argument for itself either; (iii) the ~140 M headroom should **not** be spent
widening this encoder — on the only near-matched test we have (base's 90.5 M vs flagship v1's
87.1 M encoder, within 3.8 %), 2.2× the encoder bought **0.001 m**.

---

### 4.4 REF-C CLOSED-LOOP — AlpaSim NuRec suite (n = 12) · ⚠️ **RECONSTRUCTION-OOD CONFOUNDED** — MEASURED 2026-07-22

The program's first **external-simulator** closed-loop numbers (the §8.1-#3 imagination-in-the-loop harness
was self-referential). AlpaSim on **NuRec** photoreal reconstructions, **480×854**, 20 s rollouts. Raw
(`…/incoming/2026-07-22-alpasim-closedloop-evalpod/`): `REFC_suite_results.json`
(+ `REFC_suite_{base,xl}_results.json`), open-loop control `REFC_openloop_diagnostic.json`, flagship
`Flagship_v1_results-summary.json`. **"pass" = no at-fault collision AND no off-road** (`score_criteria`).

> ⚠️⚠️ **These numbers are ENV-CONFOUNDED, not a clean model result (`RETRACTION_LOG.md` C6, 07-22).**
> The open-loop control settles it: **REF-C's open-loop ADE *on the AlpaSim reconstructions* is 1.52 m
> (de@2s 2.58), 3.21× its taniteval real-footage 0.4728** (4 scenes / 288 predictions, per-scene
> 1.40–1.77 m; `REFC_openloop_diagnostic.json`). REF-C is fed NuRec input **~3× off its training
> distribution** → the at-fault / pass rates measure **model × reconstruction-fidelity, NOT the model.**
> The base-vs-XL *ordering* survives (same OOD both); **"REF-C collides closed-loop" is NOT a clean model
> indictment.**

| metric | **REF-C-base** (104.2 M) | **REF-C-XL** (251.9 M) |
|---|---|---|
| at-fault collision | **33.3 % (4/12)** | **33.3 % (4/12)** |
| off-road | 16.7 % (2/12) | 25.0 % (3/12) |
| pass rate | **6/12** | 5/12 |
| mean score | **0.345** | 0.246 |
| dist-to-GT trajectory (m) | **1.642** | 1.973 |
| progress-rel | 0.877 | 0.885 |

**⚠️ n = 12 subset** of the 916-scene public suite (one scene = 8.3 pp; the raw JSON's own caveat is "wide
binomial CIs at n = 12"). NuRec reconstructions, not real-world; 480×854 (single-scene runs were
1080×1920). Both arms' collisions are entirely at-fault (`collision_any == collision_at_fault`).

**What DOES survive the confound.** (1) **base ≥ XL ordering** — both arms eat the same reconstruction-OOD,
so the ordering is readable even though the levels are not a clean model result: base scores **0.345 > XL
0.246**, passes **6/12 vs 5/12**, closer to GT (**1.64 vs 1.97 m**); scale bought no closed-loop advantage,
consistent with the open-loop tie at 2.4× fewer params (§4.2/§4.3). (2) **Flagship v1 CAN drive closed-loop
and passes the scene REF-C crashes (n = 1, directional):** on `01d503d4` (41-actor highway; all three REF-C
variants collide at-fault), flagship v1 (WM + `tactical_policy` head) drives **collision-free — PASS,
at-fault 0.0, score 0.699, dist-to-GT 4.25 m** (rollout `71f9740c`, `Flagship_v1_results-summary.json`;
⚠️ cite that rollout ONLY — the file's aggregate is contaminated by a stray REF-C-small rollout `17e55c6a`,
collision 1.0, on the same clip). This **corrects an earlier "v1 can't drive closed-loop" claim** — v1
drives from observations via its tactical policy (same reconstruction-OOD caveat applies).

> ⚠️ **`RETRACTION_LOG.md` 07-22:** **C5** — the n=1 *"REF-C collides at-fault"* over-read the worst-case
> scene `01d503d4`. **C6** — the n=12 *"REF-C fails ~half closed-loop"* is **reconstruction-OOD confounded**
> (open-loop-on-reconstructions control 3.21×): run the open-loop-vs-known control **before** attributing a
> closed-loop failure to the model.

---

## 5. P2 — CEM planner over the frozen v1 world model

**Not a trained model. A reconstructible evaluation artifact — and the evidence base for the v3 pivot.**

| Field | Value |
|---|---|
| **Status** | ✅ **MEASURED, both gates PASS.** Nothing trained, nothing committed. Built 2026-07-19 on Sayed's greenlight (V3_HIERARCHICAL_PLANNING_DESIGN §8). |
| **Location** | `tanitad-eval:/root/taniteval/taniteval/planner_p2.py` → `results/planner_p2_flagship-30k.json` |
| **Frozen substrate** | `flagship-30k` (step 29999, `action_dim 3`) operative predictor + `grounding.step['op']` metric step-readout, loaded strict via `loaders.load`. Rollout is the **exact gate path** `metric_dynamics.rollout_decode` (encode window → predictor K steps under the action sequence → per-step metric Δpose → SE(2) accumulate). **Nothing is fit.** |
| **Decision variable** | future action sequence `[steer, accel] × 20` steps (2 s @ 0.1 s). The `v0` channel is the observed current speed, held constant (leakage-safe, matches every trainer). Open-loop holds the observed last action fixed; closed-loop lets the planner emit `a0` directly. |
| **Proposal set** | v1 has no multi-mode decoder, so: a **5 steer × 3 accel + coast = 16-seed constant-action grid**, plus the v1 tactical head's own 0.5 s control as one learned proposal-prior seed. CEM initialises from the best seed per window. |
| **CEM** | **N = 64 samples, 3 iterations, elite-8**, per-window Gaussian over the 20×2 action tensor, clamped to `|steer| ≤ 0.03`, `|accel| ≤ 2.5`. Fully batched over windows × samples. Closed-loop uses a lighter **N = 48 × 2 iterations**. |
| **Cost** | `J = w_v·(v̂ − v_target)² + w_c·(accel² + jerk²) + w_s·steer_rate² − w_p·progress`, weights **(w_v, w_c, w_s, w_p) = (1.0, 0.1, 50, 0.02)** — **engineered from physical scales, not fit to GT ADE** (fitting would make G1 circular). **Gap/TTC barrier deliberately SKIPPED (v0)** — the data has no lead-agent boxes or HD map. |
| **VTARGET** | per window, the **85th percentile of future speed over the next 10–20 s**, dropping steps braking harder than 1.5 m/s² (free-flow only); falls back to current speed when the free-flow sample < 3 s. Valid on **94.2 %** of windows. Provenance **kinematic** — no VLM sign-read on the eval pod (an honest gap). |
| **Repro** | `python3 -m taniteval.planner_p2 --arm flagship-30k --episodes 40` (G1)<br>`python3 -m taniteval.planner_p2 --arm flagship-30k --closed-loop --cl-episodes 20 --replan-every 1` (G4) |

**Results**

| Gate | Metric | Planner | Baseline | Verdict |
|---|---|---|---|---|
| **G1** open-loop, 880 windows / 40 eps | ADE@2s | **0.893 ± 0.114** | tactical head **3.150** | **PASS** — Δ **+2.257 ± 0.329 m, CI-separated**; 72 % error reduction |
| **G4** closed-loop, 221 windows / 20 eps | ADE@2s | **1.038 ± 0.202** | v1 head **1.685 ± 0.098** | **PASS** — 38 % less drift, CI-separated |
| | FDE@2s | 2.194 ± 0.455 | 3.530 | 38 % closer |
| | divergence >5 m | **8.7 % ± 4.6** | 22.2 % | **2.5× fewer** |

Reference points on the same pass: CV 0.825 · operative rollout with **true** actions (the WM ceiling)
0.452 open-loop / 0.424 closed-loop.

**Strata:** straight (634 windows, 72 %) planner **0.564** vs true-action 0.393 vs CV 0.439 vs head 3.297.
Curved (top-10 % curvature, 89) planner **2.114** vs true-action 0.484 vs CV 2.426 vs head 3.344.

**The honest signature:** long-RMSE 1.41 / lat-RMSE 1.97 → **only 34 % of the 2 s squared error is
longitudinal; 66 % is lateral.** Speed-decoupled cross-track 0.445 m; speed bias +0.47 m/s. The planner
tracks its own minted `v_target` to **1.03 m/s** — *better than the GT log tracks it (1.54 m/s)*.
This is mechanism, not surprise: **the P2 cost has no lateral/route/goal term** (that is P3). The lateral
residual *is the measurement of what P3/P4 must add.*

**Robustness:** a 3×3 sweep of `w_c ∈ {0.05,0.1,0.2} × w_p ∈ {0.01,0.02,0.04}` (a 4× band) moves planner
ADE only **0.647 → 0.669 m (3.4 %)** and beats the head in **all 9** configs. G1 is not a tuning artifact.

> 🟥 **RECONSTRUCTION RISK — P2 is uncommitted.** `planner_p2.py` exists only on `tanitad-eval`. It is the
> single strongest piece of evidence for the v3 direction and it is one pod-loss away from gone.

---

## 6. Cross-arm leaderboard — identical harness, identical 881 windows

> ⚠️ **ESTIMATOR DEFECT — CORRECTED 2026-07-20. The `± CI95` column below is NOT decision-grade.**
> This block was historically labelled *"8-split episode-disjoint jackknife"*. It is **neither a
> jackknife nor a valid SE**: `bench.py` draws 8 **independent random 20 % holdouts** from the same 40
> episodes and takes `1.96·std/√8` over overlapping estimates — Monte-Carlo CV, measuring
> **split-selection noise**, not model uncertainty. Measured **1.28–2.06× too narrow** across 10 arms
> (median 1.51×). Coverage simulation: naive **62.3 %** vs cluster-bootstrap **93.8 %** (target 93–97 %).
> The **mean** is also a split-mean and **compresses between-arm gaps** (rows 1–2: 0.006 m here vs
> **0.0443 m** on the full set).
>
> **Decision-grade intervals: `taniteval/ci.py` episode-cluster bootstrap** (2000 resamples over the 40
> val episodes); for two arms on the same windows use the **paired** form, never a quadrature
> combination. All 10 corrected intervals: `Project Steering/CI_RECOMPUTE_2026-07-20.json`.
>
> **UPDATE 2026-07-21 — rank 1 is now a THREE-WAY tie.** `refc-base-30k` (104.2 M) is paired-tied with
> `refc-xl-30k` on all three headline metrics: ADE Δ **+0.0013 [−0.0281, +0.0316]**, FDE Δ **−0.0030
> [−0.0619, +0.0584]**, miss Δ **+0.0000 [−0.0261, +0.0272]** — none separated (§4.3). The 1= slot is
> therefore held by a **263 M world model, a 252 M diffusion arm and a 104 M diffusion arm** that no
> paired test can order. base is also the cheapest tick in the table (21.8 ms fp32 p50).
>
> **Ranks 1 and 2 are a TIE, not an ordering** — paired Δ **+0.0443 m, CI95 [−0.0544, +0.1465]**,
> P(Δ>0) = 0.809, **not separated**. Ranks that DO survive the paired test: flagship > REF-B v2
> (+0.1642, [0.043, 0.285]) · flagship > REF-A (+2.6200, [2.0945, 3.2570]) · REF-C > REF-B v2
> (+0.1199, [0.0649, 0.1771]) · v1-30k > v1-19k (+0.1881, [0.1512, 0.2265]). Every `Beats CV` ✅ was
> re-verified and **holds**.

*Rows below are the legacy heldout split-mean ± `overlapping_holdout_se` (deprecated, retained so
published figures stay traceable), physicalai val, read from the raw eval JSONs on `tanitad-eval`.*

| Rank | Arm | TanitEval key | Step | Params | ADE@2s | FDE@2s | miss@2m | Beats CV |
|---:|---|---|---:|---:|---:|---:|---:|:--:|
| **1=** | **Flagship v1 (speed+jerk) FINAL** | `flagship-30k` | 29 999 | 263.4 M | **0.4522 ± 0.0312** *(full-set 0.4271, boot [0.3675, 0.4871])* | 0.9437 | 0.0602 | ✅ |
| **1=** | **REF-C-XL** (anchored diffusion) **FINAL** | `refc-xl-30k` | 29 999 | 251.9 M | **0.458 ± 0.057** *(full-set 0.4714, boot [0.3896, 0.5556])* | 0.972 | 0.146 | ✅ |
| **1=** | **REF-C-base** (anchored diffusion) **FINAL** | `refc-base-30k` | 29 999 | **104.2 M** | **0.4523 ± 0.0497** *(full-set 0.4728, boot [0.3835, 0.5699])* | 0.954 | 0.135 | ✅ |
| — | **REF-C-small** (anchored diffusion) FINAL — SEPARATED 3rd rung (§4.2) | `refc-small-30k` | 29 999 | **54.7 M** | **0.5007 ± 0.0671** *(full-set 0.5261, boot [0.4295, 0.6262])* | 1.045 | 0.171 | ✅ |
| — | *best-of-3 kinematic floor* | — | — | — | *0.5005* | — | — | — |
| — | *CTRV oracle* | — | — | — | *0.523* | — | — | — |
| — | *no-vision ego-status ceiling* | — | — | — | *0.5735* | — | — | — |
| 3 | **REF-B v2** (arch-v2) FINAL | `refb-v2-30k` | 29 999 | 271.6 M | **0.5921 ± 0.0685** | 1.2305 | 0.2025 | ✅ |
| 4 | Flagship v1, 19 k relay | `flagship-speed` | 19 000 | 263.4 M | 0.6277 ± 0.0551 | 1.3173 | 0.1799 | ✅ |
| 5 | REF-B v2 @20 k milestone | `refb-v2-20k` | 20 000 | 271.6 M | 0.6462 ± 0.0548 | 1.3050 | 0.2132 | ✅ |
| — | **Constant velocity (the floor)** | — | — | 0 | **0.8248** | 1.7081 | — | — |
| 6 | REF-B speed | `refb-10k` | 10 000 | 262.8 M | 0.8255 ± 0.0992 | 1.6714 | 0.2641 | ✗ |
| 7 | REF-B v1 | `refb` | 6 000 | 262.5 M | 0.8682 ± 0.0817 | 1.7341 | 0.3343 | ✗ |
| 8 | **P2 CEM planner** over frozen v1 | `planner_p2` | (n/a) | 0 trained | 0.893 ± 0.114 | — | — | ✗ |
| 9 | REF-A DINOv2 4B | `refa-dinov2` | 29 999 | — | 2.1322 ± 0.1821 | 3.2619 | 0.6245 | ✗ |
| 10 | Flagship **no-speed** (ablation) | `flagship-nospeed` | ~22 000 | 263.4 M | 2.9176 ± 0.3558 | 4.9395 | 0.7395 | ✗ |
| 11 | REF-A dyn-in 4B | `refa-dynin-30k` | 29 999 | — | 2.9196 ± 0.3937 | 4.5832 | 0.7246 | ✗ |
| 12 | Flagship **v2** (killed) | `flagship-v2-6k` | 6 000 | 272.9 M | 6.179 ± 1.2845 | 12.7015 | 0.8407 | ✗ |
| — | Flagship v3enc | — | running | 272.9 M | 🟥 not evaluated | — | — | — |
| — | Flagship v1 tactical **head** (not rollout) | `plan_flagship-30k` | 29 999 | — | 3.38 (3.150 in the P2 pass) | — | — | ✗ |

**Two readings that matter:**
1. The **trained-encoder** arms occupy every slot above CV. The **frozen-encoder** arms occupy slots 9 and
   11. That is H4, in one table.
2. The flagship's supervised **tactical head** (3.38 m) is *worse than CV*, while the same model's
   operative rollout is 0.452 m. **The head is a lossy readout of a good world model** — which is exactly
   what P2 exploits and what v3 is built on.
3. **Ranks 1–2 tie on accuracy, so latency is the tiebreaker — and it is not close.** Measured
   2026-07-20 on one A40, batch 1, identical precision flags (`taniteval/results/eff_*.json`):
   flagship planning tick **103.42 / 93.76 / 104.49 ms** (fp32/tf32/amp16) vs REF-C **44.28 / 27.84 /
   26.12 ms** — **2.3–4.0× faster**, and REF-C **meets the 10 Hz budget at p99 in all three precisions
   while the flagship misses in all three**. REF-C does **1.75× the FLOPs** (702 vs 402 GFLOPs) and is
   still faster because it achieves **15.9–25.2 TFLOPs vs 3.7–4.3**: the flagship's 20-step sequential
   rollout is **launch/serialisation-bound**, not arithmetic-bound. ⚠️ Direction reverses for *batched*
   throughput (flagship 34.8 vs REF-C 29.9 windows/s @ batch 32) — REF-C wins latency, the flagship wins
   bulk eval. See §1.2 for the two-tick definition and the retracted "11.16 ms" framing. ⚠️ These three
   latency figures conflict with the committed `eff_*.json` — see **R14**.
4. **The 1= tie is not a tie on driving — and latency is no longer the only separator.** MEASURED
   2026-07-21, TanitEval v2 tier-0 over these same 881 windows (`taniteval/results/driving_*.json`;
   `Benchmarks & Eval/LEADERBOARD.md` §2). Splitting the 2 s residual on the GT path tangent:
   **REF-C-XL beats CV along-track by +0.2170 [+0.0584, +0.3783] and REF-C-base by +0.2300
   [+0.0773, +0.3816], both CI-separated — while flagship v1's along-track win is +0.2543
   [−0.0278, +0.5304] and is NOT separated.** All three separate on cross-track. **Among its own rank
   tier the flagship is the only arm with no CI-separated longitudinal competency**; its entire
   separated advantage over CV is lateral (+0.7720 [+0.4166, +1.1914]). Two further reads a single ADE
   column hid: **(a)** every one of the 14 arms with a window dump is CI-separated *against* the
   hold-v0 floor on the 639 longitudinally steady windows (flagship v1 **−0.2122 [−0.2778, −0.1443]**,
   i.e. 2.0× worse than doing nothing) while winning brake/accel — cruise quality and transient
   response point in opposite directions program-wide; **(b)** `flagship-v16-ab-ft` (§1.4b), an ADE tie
   with v1, is the **only arm in the program whose speed MAE beats CV with a separated interval**
   (+0.0785 [+0.0066, +0.1516]) and it paid for that laterally (cross 0.423 vs v1's 0.237,
   path-geometry 0.204 vs 0.111, κ-sign 0.865 vs 0.954). §1.4b's "unfreezing changed nothing
   measurable" is exact **on ADE** and wrong in both directions on the split: unfreezing **traded
   lateral geometry for longitudinal tracking**.
5. **Closed-loop was measured on an EXTERNAL simulator — but the REF-C numbers are RECONSTRUCTION-OOD
   confounded (§4.4, RETRACTION_LOG C6).** AlpaSim NuRec (n=12, 2026-07-22): REF-C-base 33 % at-fault /
   6-of-12 pass, XL 5-of-12 — **but REF-C's open-loop ADE *on the reconstructions* is 1.52, 3.21× its
   real-footage 0.4728**, so those rates measure model × reconstruction-fidelity, not the model. What
   survives: **base ≥ XL ordering** (score 0.345 vs 0.246, same OOD both), and — the ONLY admissible
   reading of the v1 scene — **v1 does drive closed-loop at all via its tactical policy** (`01d503d4`
   collision-free, PASS score 0.699, rollout `71f9740c`).
   ⚠️ **CORRECTED 2026-07-25 — do NOT quote that scene as "v1 beats REF-C".** The n=1 framing
   ("the one scene all three REF-C variants crash") was a **lucky scene** and is **retracted**
   (`RETRACTION_LOG` C7). It reverses under power, twice, on independent instruments:
   **n=12 paired AlpaSim** — REF-C base vs flagship v1 pass **8/12 vs 2/12**, score 0.496 vs 0.066,
   paired Δ **−0.430 [−0.646, −0.215]**, sign-test 8-0 (p = 0.008), collisions TIED; and
   **n=40 real-footage low-OOD** (1.02–1.20× OOD, ≪ NuRec's 3.2×) — ADE@2s **0.564 [.452,.676] vs
   1.488 [1.329,1.647]**, departure-rate 0.0134 vs 0.0318 (§4.4, LEADERBOARD §5.5).
   **Standing closed-loop ordering: REF-C base > flagship v1, triple-confirmed.** v1's deficit is
   **longitudinal, not lane-keeping**; its tactical head is a high-deviation planner
   (plan_dev 1.12 vs 0.34) → offroad, not collision.

---

## 7. Reconstruction-gap register — what would block a rebuild today

| # | Gap | Severity | Where the only copy lives | Fix |
|---|---|---|---|---|
| R1 | 🟠 **RE-OPENED 2026-07-20 (I closed this prematurely — on file *presence*, not file *identity*).** The code is rescued, but the registry named the **wrong file** as the as-trained artifact. The trained arm (0.5921, **271,619,880 / tactical 30,270,742**) is reproduced by **`refb_v3.py`**, NOT `refb_v4.py`. The two differ by exactly **Δ255 params** in the tactical brain = `LayerNorm(128)`'s 256 params minus v4's 1-param gate; their **only** difference is the H19 prior mechanism. *Independently reproduced: v3 271,090,974 / tac 30,266,638 vs v4 271,090,719 / tac 30,266,383 — Δ255 exactly, on a config missing two aux flags so neither hits the absolute figure; the exact match to 271,619,880 is the wiring-comparison agent's measurement.* **So the as-trained REF-B v2 used the LayerNorm-pinned prior** — a scaler now on the never-worked list (it pins ‖prior‖ at √N instead of bounding it), which makes v4 an untrained *improvement*, not the artifact | 🟠 high | in-repo: `stack/experiments/refb-v2/{refb_v3.py,refb_v4.py,refb_train_v4.py,launch_v2.sh}` | point reproduction at **`refb_v3.py`**; record v4 explicitly as the later untrained successor so nobody rebuilds the wrong arm |
| R2 | ✅ **CLOSED 2026-07-20.** TanitEval vendored — **68 files tracked** incl. `registry.py`, `bench.py`, `closedloop.py`, `refc_eval.py`, `hierarchy.py`, `plan_fan.py`, `ci.py`, `efficiency.py` | — | in-repo: `taniteval/` | done |
| R3 | ✅ **CLOSED 2026-07-20.** P2 planner vendored | — | in-repo: `taniteval/taniteval/planner_p2.py` | done |
| R4 | **Flagship v1 `--jerk-weight` / `--aux-accel` missing from the committed trainer** — the deployed model is not byte-rebuildable from HEAD | 🟠 high | `tanitad-pod2:/workspace/TanitAD/stack/scripts/train_flagship4b.py` (shows `M`) | commit the pod2 diff |
| R5 | ✅ **CLOSED 2026-07-21.** `Benchmarks & Eval/LEADERBOARD.md` rewritten from §6 with units labelled on every column (**metric-BEV `ade_0_2s`, m**); the camera-frame ADE@1s @27 k gate ladder is retained but demoted to a clearly-labelled historical section (§8) so it cannot be read as current. Added: the TanitEval v2 **tier-0 driving-capability** tables (along/cross split, speed MAE vs hold-v0, cruise vs transient, heading by curvature, κ-sign) for **all 14 arms with a window dump**, plus the 04b latency column. Regenerable offline: `python -m taniteval.runner driving-all` → `python -m taniteval.driving --leaderboard` | — | in-repo: `Benchmarks & Eval/LEADERBOARD.md`, `taniteval/taniteval/driving.py`, `taniteval/results/driving_*.json` | done |
| R14 | **Two primary sources disagree on the flagship's planning tick.** §6 reading 3 quotes **103.42 / 93.76 / 104.49 ms** (fp32/tf32/amp16, "measured 2026-07-20, `taniteval/results/eff_*.json`"); the committed `eff_flagship-30k.json` says **97.32 / 97.70 / 123.83**, its own kept replicate says **97.13** fp32, and `eff_repeatability.json` (5 clean reps, exclusive GPU) says **99.03–100.05** p50 — three values, none matching §6. REF-C-XL's fp32/tf32 agree across both (44.28≈44.06, 27.84≈27.78) but amp16 does not (**26.12 vs 21.00**). The two sets were evidently taken in different sessions and only one survived into the repo. **No conclusion changes** (REF-C is 2.2–4.6× faster; the flagship misses 10 Hz at p99 in all three precisions in every version) — but a prose figure and its cited artifact must not disagree | 🟡 medium | in-repo artifacts + this table | one reconciliation pass: re-measure on an idle A40 or restate §6 from the committed JSONs. `LEADERBOARD.md` §5 currently quotes the artifact and flags the conflict |
| R6 | **`gate-eval` skill targets a dead run** (`p0-sB01-realmix`, frozen since 2026-07-12 @ step 28,600) | 🟡 medium | `.claude/skills/gate-eval/SKILL.md` | retarget to the live arm |
| R7 | **REF-C three-size scaling study never ran** — 🟡 **middle rung now training** (`base` measured 104,191,577 and launched 2026-07-20, §4.3); `small` still only smoked, and the `base` run carries a **scale/label confound** (v2.1 labels vs XL's v1) | 🟡 medium | n/a | let `base` finish + evaluate; then either add a label-controlled arm or state the confound wherever the ladder is quoted |
| R8 | **REF-A I-JEPA canonical-val result is leak-contaminated** (80 % of val in its train set) | 🟡 medium | flagged in `taniteval/registry.py` | re-evaluate on the `f1b378` val before any comparative claim |
| R9 | **REF-B rev2, `refa-4brain-speedyaw-30k`** have no eval record | 🟢 low | n/a | either evaluate or mark explicitly superseded |
| R10 | **`refb-speed-30k/ckpt_prepatch_step8500.pt` is misnamed** — it is step 10,000 and byte-identical to `ckpt.pt` | 🟢 low | pod1 | rename |
| R11 | ~~`combined_tick_harness.py` is not in HEAD~~ ✅ **NOT A GAP — retracted within the hour it was written 2026-07-20.** The harness and its raw JSONs **are** tracked: `Production & Optimization/Implementation/combined_tick/{combined_tick_harness.py, combined_tick_20260718.json, vram_fp16/fp32_20260718.json}`. I asserted the gap from `REPO_TRIAGE_2026-07-20.md`, which was written **before** this session's merges landed, instead of checking `git ls-files` — the exact prose-over-primary-source failure this document exists to prevent. Kept visible as a worked example, not deleted | — | in-repo | ⚠️ `REPO_TRIAGE_2026-07-20.md` is now stale on this point and should be date-stamped or retired |
| R12 | ✅ **CLOSED 2026-07-21 — measured.** The planning tick now has an optimised variant and **the 10 Hz miss is resolved**: composed **18.75 ms p50 / 18.76 p99 = 53.3 Hz**, 5.35× (§1.2). ⚠️ **Four predictions written into this row on 07-20 were REFUTED by the measurement** — recorded rather than deleted, because the pattern is the lesson: (1) *"capturing all 20 steps in ONE graph should beat 2.57×"* — **no**, whole-rollout capture equals per-step capture to **7.7 µs/step** (57.18 vs 57.33 ms); inter-step CPU round-trips were **never** the cost, and a constrained runtime that can only capture one step loses **0.3 %**; (2) tick-level gain is **1.75×**, not 2.57× — 2.57× was a *stage* figure and stage speedups do not equal tick speedups; (3) `torch.compile(reduce-overhead)` **wins on Linux** (52.89) though it failed on Windows; (4) levers are **sequenced, not additive** — capture first, everything else is worth ~1.0× before it | — | `taniteval/results/eff_levers_flagship-30k.json`, `taniteval/taniteval/efficiency.py`, `Production & Optimization/Research/2026-07-20-flagship-v1-inference-levers-measured.md` | done |
| R13 | **Three code sites were escalated as CUDA-graph "prerequisites" and are NOT** — capture succeeded with zero build errors and **exact** equivalence despite ~38 allocating `torch.cat`s (`metric_dynamics.py:241-242`) and a per-call mask rebuild (`predictor.py:112`), because allocations *inside* a capture come from the graph's own private memory pool. They remain real waste (L7's 2 discarded horizon heads = ~252 MB/tick of needless DRAM reads) and matter on a bandwidth-bound Jetson, but they **never blocked anything** | 🟢 low | in-repo | treat as an efficiency cleanup, not a blocker; the "static-address" rule applies to tensors crossing the capture boundary, not to internal allocation |
| R15 | **`dynenc-branchB` trained ckpt (step 40000, md5 `a0d7e7c1…`) has no HF backup** — the gated `Sayood/tanitad-dynenc-branchB` push was **classifier-blocked** (pod3 has no HF auth; credential-move refused). Arch/trainer/eval are in-repo (`stack/tanitad/models/dynamics_encoder.py`, `train_dynamics_encoder.py`, `run_branchb_transfer.py`), so the recipe is rebuildable, but the exact trained weights (the evidence behind the §10 FAIL) live on **pod3 + MooseFS only** | 🟡 medium | `tanitad-pod3:/workspace/experiments/dynenc-branchB/ckpt.pt` (durable MooseFS, dd-verified) | Sayed/user authorizes HF token handling, or push from an HF-auth box (`push_ckpt.py` precedent). Failed-arm evidence, so lower urgency than a deployed model — but still one pod-loss from gone (§10) |

---

## 8. Decision log — the *why* behind every row above

Chronological. IDs `D-0xx` reference `DECISIONS.md`; `D-Axx` are lineage decisions recorded here because
they were made in the operator loop and never got an ADR.

| # | Date | Decision | Rationale | Evidence | Superseded / affected |
|---|---|---|---|---|---|
| **D-003** | 07-05 | Main track = from-scratch 4-brain latent world model; **frozen-encoder is a comparison arm, not a hedge to adopt** | Every component individually validated in ALPS-4B or externally (LAW, IDOL, V-JEPA-2-AC); the from-scratch arm is what makes the data-efficiency claim disruptive | ALPS-4B assets A1–A10 | Created the flagship / REF-A split that everything below reads against |
| **D-008** | 07-05 | Model scale **≥ 250 M**; H15 imagination promoted into Phase 0 | A scale where hierarchy is expressible and Orin/Thor is still reachable | — | Fixed the ~261 M budget every arm is matched to |
| **D-009** | 07-06 | **Real camera data first**; toy demoted to CI fixture | Toy proofs don't transfer; comma2k19 gives real actions at zero annotation cost | — | `base250cam` becomes primary; later PhysicalAI-AV |
| **D-015** | 07-06 | Encoder input = **3 RGB frames at 100 ms spacing, channel-stacked (9 ch, 256 px)** | Acceleration and curvature become observable *inside one encoder input* | — | The `[T,9,256,256]` contract every arm shares |
| **D-027** | 07-10 | **K-step rollout loss** (`rollout_k=4`) for all post-30k training | Multistep-as-augmentation (2512.24497) | K-step bake-off probe | v1 runs at k=4; v2/v3enc raise it to 12 |
| **D-A1** | 07-11 | **REF-B rev2**: strategic becomes a real `d384×4` transformer with per-window nav commands + its own route CE | **A defect, not an upgrade** — the strategic head had been training on a constant `follow` | `refb_labels.nav_command` derivation over 15–25 s of future heading | `e616b23` → `38cf9ca`; the strategic block all later arms inherit |
| **D-A2** | 07-13/14 | **The 3-arm parity design**: flagship / REF-A / REF-B, one pod each, on the *identical* 2,376-ep set | Only strict same-data parity makes the encoder axis (H4) and the hierarchy axis (H1/D4) causally readable. Each arm isolates exactly one thing | parity key `e438721ae894` + skip-hash `f09e44db`; `refb_pipeline.sh` **refuses to launch** unless the build reproduces the key | Every comparison in §6 depends on this |
| **⭐ D-A3** | **07-14** | **`v0` (current speed) as the 3rd action channel** — and **restart all three arms from scratch** to get it | Actions are *derivatives* `[steer, accel]`; absolute displacement needs `v0`, which a frozen encoder cannot recover from pixels. The models were being asked to integrate without an initial condition | Validated **in isolation before committing the retrain**: REF-A operative fwd-ADE **3.73 → 0.83 m**, speed-decodability **R² 0.61 → 0.965**. Later confirmed causally: flagship no-speed **2.918** vs speed **0.452** on identical data/arch, paired A/B **+2.21 m [2.04, 2.39]**, win-rate 83.8 % | Voided all pre-07-14 REF-A numbers (the 14.2/17.0/20.2/7.6 m spread); created `flagship4b-speedjerk-30k`, `refa-plus-speed-30k`, `refb-speed-30k`; archived to `stack/experiments/reset-speed4b/`. **`SPEED_SCALE = 10.0` is a hard contract** |
| **D-A4** | 07-14 | REF-A given the **full 4 brains by hand** (`--four-brain`) | Without it, REF-A vs flagship confounds *encoder* with *hierarchy*. With it, the two differ in exactly two things (encoder, SIGReg target) | CPU-smoke-validated before launch; `refa4b_config()` returns the identical `StackConfig` | Makes `refa-*-4brain-*` the only fair REF-A arms |
| **⭐ D-A5** | 07-17→19 | **REF-A accepted as a frozen-encoder REFERENCE — H4 closes negative** | The ceiling is **capability, not overfitting**: the milestone curve is monotonically *improving* (5 k 3.755 → 15 k 3.694 → 20 k 3.016 → 30 k **2.920**, best is last). Held-out error never rises. Every remedy was tried — speed input, yaw input, dyn-input `[v0,yr0]`, ego-dropout, temporal adapter, 4 brains, I-JEPA features — and it still plateaus above CV | dyn-in 2.9196 ± 0.394 vs flagship 0.4522; paired win-rate 95.9 %, Δ +2.62 m CI [2.447, 2.798]; train fwd-ADE 0.65 → held-out 2.92 (4.5× gap); pre-fix ablation `vision_use` 3.4 % → "a dynamics integrator" | A clean publishable negative. Motivates the trained encoder. Ends the REF-A retrain line; anchored-decoder retrain dropped |
| **D-030** | 07-18 | **REF-C redesigned** to a DiffusionDrive anchored truncated-diffusion decoder + a 3-size scaling study (55/104/252 M) | The 2022-era tiny-TCP GRU was not a fair modern reference. Anchored multimodal decoding is the published standard and directly tests H19 (maneuver→anchor graft). FPS not k-means because ~74 % of the data is straight | REF-C-XL @16 k = **0.5645** — 2nd of the trained-encoder arms, beats CV in all three speed terciles | `6025769`, `36d979f`, `7e9c402`. ⚠️ **The scaling study itself was never run — only XL exists (§4.2, OPEN)** |
| **D-032** | 07-18 | **Milestone checkpoints at 5 k/15 k/20 k/30 k** instead of overwriting `ckpt.pt` | No earlier checkpoint survived for re-gating, overfitting curves, or lineage forensics | The REF-A overfitting-vs-ceiling verdict (D-A5) is **only possible because of this** | `b298cef`, `6808c2d`. Costs disk — and pod2's 98 %-full overlay is what killed v3enc's first attempt |
| **D-A6** | 07-17 | **flagship-v2: ten levers at once** (six named in the directive, ten set) | The 30 k flagship had three named weaknesses: high-speed longitudinal overshoot, a command-echo strategic head (`route_skill_vs_chance = 0.0`), and an encoder redundantly re-encoding ego dynamics (`vision_use` ~12 %). Each lever targets one | Every lever individually motivated by an H25/H26 measurement | `f583bb4`, `b8d3fc8`; run `flagship4b-v2-30k` |
| **⭐ D-031** | **07-19** | **Kill flagship-v2 at 7.8 k; do not grind it to 30 k** | The 6 k number (6.18 m) alone would *not* justify killing — it was correctly diagnosed as **mechanism-A**: the levers removed the kinematic speed shortcut **by design** (encoder speed-probe R² 0.30 vs v1's 0.86). **The decisive read was the rate of learning, not the level:** the same-step v2/v1 forward-consistency ratio *widened* 1.51 → 4.33; the power-law exponent was **−0.50 vs v1's −0.84**; v1 reached v2's 7.5 k value at **step ~250**. Projection to 30 k: 9× worse for the same 4 days of A40 | `flagshipv2-6k-diagnostic.md`; per-lever telemetry otherwise healthy (no NaN, no gnorm spike, anchored decoder converging) → **the failure was simultaneity, not any single lever** | `flagship4b-v3enc-30k` |
| **D-A7** | 07-19 | **v3enc restart with STAGED levers** | Keep every *decode-side* lever from step 0 (they were healthy); soften and time-stage only the four *encoder-grounding* levers: decorr off until 10 k then 0.02, rollout-k 4→8→12, invdyn_gradscale 0.25→**0.5**, fa_dropout 0.3→**0.15** | The diagnostic isolated the encoder-grounding group as the optimization burden | `a01ad24`; `--staged-levers`. **Pre-registered falsifier:** no improvement in same-step forward-consistency vs v1 at 10 k → restart again |
| **D-A8** | 07-19 | **Acceptance gate for v3enc should be the OOD panel, not in-distribution** | v1 already passes in-distribution (0.427 vs floor 0.523) but fails OOD (comma 0.849 vs floor 0.372, 17.5 % win-rate). Optimizing the passed gate teaches nothing | the OOD panel | Proposed target: **≥ 35 %** win-rate vs the comma floor |
| **⭐ D-033** | **07-19** | **v3 pivot: hierarchical world-model PLANNING. Supervised heads demote to proposal priors** | Three measured pathologies all trace to head-supervision: longitudinal mean-regression (REF-A 94 % longitudinal, flagship high-speed the only above-floor stratum), a degenerate strategic seam (`route_skill_vs_chance` 0.0 — pure command echo), and an actively **harmful** intent→operative seam (cos vs-none **−0.238**). Making target-speed and mode-switching a **planning cost** instead of a head fixes all three at once | **P2 passed both decisive gates at zero training cost:** G1 open-loop **0.893 ± 0.114 vs head 3.150** (+2.257 ± 0.329, CI-separated, 72 % error reduction); G4 closed-loop **1.038 ± 0.202 vs 1.685 ± 0.098** (38 % less drift), divergence **8.7 % vs 22.2 %** (2.5× fewer). Weight-sweep robust across a 4× band (0.647→0.669, wins all 9) | `V3_HIERARCHICAL_PLANNING_DESIGN.md` + `V3_GOAL_VOCABULARY_V1.md` (frozen). v1 remains the operative arm. Sayed's framing: v3 = the **original DINO-WM recipe** (frozen encoder + feature-prediction of action-consequences, **no supervised head**) + CEM/diffusion/MPC planner |
| **D-A9** | 07-19 | **VTARGET moved strategic → tactical** | P2 measured that the planner tracks its minted `v_target` to **1.03 m/s** — *better than the GT log tracks it (1.54 m/s)*. The longitudinal target is a **control-rate** quantity that must be re-derived faster than the strategic cadence (20 ticks); leaving it at strategic starves the cost function between updates | P2 §2.3 + §5.2; strategic route decode remains at the strategic level | `V3_GOAL_VOCABULARY_V1.md`. ⚠️ The exact vocabulary-level wording should be read from that doc before implementation — this row records *that* it moved and *why* |
| **D-A10** | 07-19 | **P2's residual is 66 % lateral — that is the P3/P4 scope, not a failure** | P2's cost is longitudinal + comfort + progress **only**; it carries no lateral/route/goal term by design. So it nails longitudinal and defaults laterally to the smoothest option. Curved-window error 2.114 m *is* the measured cost-of-no-lateral-goal | long-RMSE 1.41 / lat-RMSE 1.97; speed-decoupled cross-track 0.445 m; true actions reach 0.484 on the same curved windows | P3 = strategic lateral goal in the cost; P4 = goal-conditioned tactical predictor to lift the WM from imitation-era to planning-grade |
| **D-A11** | 07-20 | **Cosmos-Reason1-7B chosen as the dataset VLM labeler; Cosmos3 is not a labeler** | Byte-pull gating check (2026-07-20): Cosmos3-Nano/Super (OpenMDW 1.1 omnimodel, commercial-OK), Cosmos-Reason1-7B and Reason2-32B are **ungated**; only Reason2-2B/8B are gated. The pilot verdict then separated *serving* from *labeling*: Cosmos3 needs `vllm-omni`/`sglang` rather than vanilla vLLM and did not behave as a labeler | commit **`547c8ec`** "dataset: VLM pilot verdict — Cosmos-Reason1-7B for labeling, Cosmos3 is not a labeler"; pilot artifacts in `TanitAD Research Hub/Data Engineering/` | ⚠️ Sayed had earlier asked for **Cosmos3 for the dataset**; the pilot changed the answer. Every VLM label maps onto the frozen `V3_GOAL_VOCABULARY_V1` |

### 8.1 Open questions this log deliberately does not close

1. **REF-C scale (§4.2/§4.3).** `base` (104,191,577 — measured) is **training since 2026-07-20**;
   `small` is still a 150-step smoke. And `base` runs **v2.1** route labels while XL ran **v1**, so the
   first medium-vs-XL number conflates **scale with labels** — a clean rung needs a label-controlled
   arm. The "where does bigger help vs overfit" claim D-030 commissioned stays **unsupported** until
   §4.3 is evaluated. 🟡
2. **v3enc has no result.** Every statement about staged levers working is a hypothesis with a
   pre-registered falsifier, not a finding. 🟥
3. **Closed-loop is self-referential — PARTLY UNBLOCKED 2026-07-22, but the first external run is
   reconstruction-OOD confounded.** The imagination-in-the-loop harness uses the world model as both
   estimator and simulator. The external cure **ran** (AlpaSim NuRec, n=12, REF-C base+XL, §4.4) — but the
   open-loop-on-reconstructions control shows REF-C is fed **3.21×-OOD** input (open-loop ADE 1.52 vs
   real-footage 0.4728), so the failure rates confound model with reconstruction-fidelity (RETRACTION_LOG
   C6). What survives: **base ≥ XL ordering**, and that **v1 drives closed-loop at all** (`01d503d4`
   collision-free, n=1, §4.4). ⚠️ **UPDATED 2026-07-25 — the OOD confound is now RESOLVED and the n=1
   v1-vs-REF-C reading is RETRACTED (C7).** The real-footage low-OOD instrument eliminated the
   reconstruction confound by construction (on-policy OOD **1.02–1.19×** vs NuRec's 3.2–3.75×; Δ=0
   open-loop ADE 0.4045 reproduces the real level) and at **n=40** gives **REF-C base > flagship v1**
   (ADE@2s 0.564 vs 1.488) — the same ordering as the n=12 paired AlpaSim run (8/12 vs 2/12), so it is
   **not** a reconstruction artifact. Still open: the low-OOD instrument is **map/agent-free** → it emits
   drift/lane-departure but **never off-road/collision** (those need reactive agents = a sim = OOD); that
   gap is ~fundamental until a lower-OOD *reactive* renderer exists. CARLA pixels remain host-blocked.
   D5/D6 remain 🟡.
4. **D2/D3 gate evidence is stale** — measured at step 27 k on the *pre-reset* `p0-sB01-realmix` run, in
   camera-frame units, and never re-gated after the speed reset.
5. **VTARGET's exact placement in the frozen goal vocabulary** should be read from
   `V3_GOAL_VOCABULARY_V1.md` before implementation (D-A9). ⚠️
6. **Supervised-IDM does not transfer cross-domain (2026-07-22 finding — not a model row).** A 2.9 M
   inverse-dynamics head reaches PhysicalAI held-out speed R² **0.930** but **fails cross-domain** —
   comma2k19 speed R² 0.657 / yaw R² 0.000, and even same-corpus **rig-B speed R² −2.465** (ADE ratio
   2.40 / 4.01; both FAIL the cross-domain >0.9 gate). The YouTube-scale IDM data pipeline is **gated on
   the re-gate** and does not proceed on these numbers. Raw:
   `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-22-idm-proof/results.json`.
   **UPDATE 2026-07-24 — the fix was built and it FAILED.** The own dynamics-encoder designed to give this
   head a rig-robust latent (from-scratch, GAIA-2 all-block camera-conditioned, `dynenc-branchB` @ 40k) was
   measured on the decisive held-out-rig transfer gate and **REFUTED**: best cross-rig speed R² **−0.667**
   vs the plain frozen flagship-v1 encoder's **+0.657** (paired dR2 CI excludes 0, Branch B worse, on 3/4
   arms). See **§10**. The YouTube-scale IDM thesis resting on this encoder is now **not supported** by
   measurement, not merely gated. 🟥

---

## 9. Maintenance contract

- **One row per major version.** A run that only changes hyperparameters within a version is a milestone,
  not a row.
- **Every row must carry:** architecture + measured param count, the *exact* command, corpus + parity key,
  the flags/commit that define it, results with the statistic named, HF repo, status.
- **Numbers come from raw result JSONs, never from prose.** Where two numbers exist, name both and name
  the statistic (`heldout` vs `full_set`).
- **A fact that cannot be verified is marked 🟥 UNVERIFIED, not guessed.** A wrong reconstruction recipe
  is worse than a flagged gap.
- **Refresh:** at every version boundary (new run launched, run completes, new eval lands), and whenever
  `PROGRAM_OVERVIEW.md` is refreshed.

---

## 10. OWN DYNAMICS-ENCODER LINE — rig-robust IDM substrate (SIDE model, NOT the WM parity arm)

*(New top-level section, appended 2026-07-24 so §1–§9 keep their numbers. This is a distinct experimental
line, **not** a REF-C sub-arm.)*

A **side model**: a from-scratch dynamics-estimation encoder built to give a small inverse-dynamics (IDM)
head a **rig-robust** latent — the substrate the YouTube-scale IDM-pretraining thesis
(`IDM_VIDEO_PRETRAIN_DESIGN.md`, `Research/2026-07-22-encoder-strategy-and-vjepa2ac.md`) depends on. It
**never** reads the WM parity key `e438721ae894` / skip-hash `f09e44db` and never re-selects parity
episodes (splits are **by rig / by corpus**, orthogonal to the WM selection). It is therefore **not** on the
§6 trajectory leaderboard — it is scored by a **cross-rig IDM probe** (speed/yaw R² on a held-out camera
rig), a different harness. Design docs:
`…/incoming/2026-07-22-own-dynamics-encoder/{DESIGN,LAUNCH_PLAN,PRE_REGISTRATION}.md`.

**The refutation chain (every step MEASURED, artifact-cited).** The target is a latent whose cross-rig
speed R² clears the **>0.9** gate:

| step | recipe | cross-rig rig-B speed R² | artifact |
|---|---|---|---|
| frozen v1-encoder IDM | supervised IDM head on frozen flagship-v1 | **−2.465** | `…/incoming/2026-07-22-idm-proof/results.json` (§8.1 #6) |
| light-FT | unfreeze last 4 ViT blocks | **−1.65** single-dom / **−1.61** multirig (data-diversity **refuted**) | `results_regate.json` / `results_multirig.json` |
| **Branch A** | warm-start + suffix camera-conditioning (ON) | **−2.25** rig / **−2.06** multirig (refuted) | `…/own-dynamics-encoder/RESULTS_camcond.md` |
| **Branch B** | **from-scratch, all-block conditioning, multi-rig** | **§10.1 below** | `…/incoming/2026-07-24-branchb-transfer-eval/` |

> ⚠️ **Regime note (C5/C6):** the four rows above are **not** a single comparable series — the first three
> are light-FT / suffix regimes, Branch B is frozen-encoder + a converged fresh head. The Branch-B verdict
> is anchored on the **paired same-regime contrast vs flagship-v1** and the **regime-free own-head number**
> (§10.1), never on a cross-regime point comparison. Do not read "−0.667 beats −1.61 so Branch B improved."

---

### 10.1 dynenc-branchB (step 40,000) — 🟥 **FAIL** — decisive held-out-rig transfer

| Field | Value |
|---|---|
| **Status** | 🟥 **FAIL — REFUTED.** From-scratch GAIA-2 camera-conditioning does **not** engineer rig-robustness at 40 k steps / 2 466 clips, and is a **weaker** dynamics substrate than the plain flagship-v1 encoder. **The own-encoder / YouTube-IDM thesis resting on it is not supported.** Transfer eval landed 2026-07-24 ~01:46 UTC on pod3 (A40); training completed 40 k on pod3 (`BRANCHB_LAUNCH.md`, launched 2026-07-23, Sayed-approved). |
| **Location** | `tanitad-pod3:/workspace/experiments/dynenc-branchB/ckpt.pt` — **step 40000, md5 `a0d7e7c19e8105cde04e743f6ed6ee26`** (weights identical to the step-40000 save; final save re-inits the optimizer only) — + `history.json` + `milestone_step2000.pt`; durable MooseFS (dd-verified, 500 MB @ 389 MB/s). Eval-cached latents `pod3:/workspace/tmp/branchb_eval/`. |
| **Distinguishing design** | **From scratch** (no warm-start); GAIA-2 **all-block** camera conditioning — separate intrinsics/extrinsics/distortion embeds + known/unknown mask, per-clip `cy`, **zero-init per-block inject** (grown to weight-norms ~10–18 across the 12 blocks by step 10 k, verified rig-discriminating); objective = masked-latent SSL + action-cond forward-pred + SIGReg (λ 0.1) + supervised IDM head + odometry metric grounding. `geom_augment` (±12 px vertical shift + matched cam params) every window. |
| **Params (MEASURED at instantiation)** | ViT **87.0 M** + GAIA-2 all-block cond **7.4 M** + readout 0.1 M + IDM head 2.9 M → **deployable 97.4 M**; + predictor 3.49 + masked-pred 2.63 + invdyn 2.37 → **total(train) 105.9 M** — sub-300 M. (`BRANCHB_LAUNCH.md`; `stack/tests/test_dynamics_encoder.py`, `smoke_report.json`.) |
| **Data (SIDE — NOT parity)** | **2 466 clips**: PhysicalAI **rig-A 637** + **rig-B 1 739** (per-clip cy, f-theta fisheye) + **comma2k19 90** (rectilinear). Multi-rig, 3 geometries. Corpus-sampled fixed (resume-safe) standardizer. **Never the parity key `e438721ae894`.** |
| **Exact command (trainer)** | `PYTHONPATH=/workspace/TanitAD/stack python3 scripts/train_dynamics_encoder.py … --out experiments/dynenc-branchB` — `DynEncConfig(grad_checkpoint=True)`, AdamW lr 3e-4 / wd 0.05 / warmup 1000 / grad-clip 1.0, batch 16, grad-checkpoint, **40 000 steps**; memory-safe SHARD loader (48 clips resident, rotate every 200). Supervisor `pod3:/workspace/tmp/dynenc_supervise.sh` (auto-resume; staged copy in the design folder). |
| **Code state** | ✅ **In-repo (staged, `git ls-files` confirms):** `stack/tanitad/models/dynamics_encoder.py` (`CameraConditionedEncoder` / `DynamicsEncoderModel` / `DynEncConfig`), `stack/scripts/train_dynamics_encoder.py`, transfer runner `stack/scripts/run_branchb_transfer.py`, `stack/tests/test_dynamics_encoder.py`. Arch + trainer + eval are rebuildable from HEAD. |
| **HF** | ❌ **none — push BLOCKED.** The backup to a gated `Sayood/tanitad-dynenc-branchB` was **gated by the safety classifier** (pod3 has no HF auth; the credential-move was refused and not worked around). Ckpt preserved on pod3 + durable MooseFS (not the sole copy). **Action for Sayed/user:** authorize the token handling or push from an HF-auth box (precedent: `push_ckpt.py`). |

**Gate (pre-registered, frozen — `PRE_REGISTRATION.md` / `BRANCHB_LAUNCH.md`):** on the held-out **rig-B**
contrast, **cross speed R² > 0.9 AND yaw R² > 0.9 AND ADE@2s < 1.5× in-domain**.

**Results — MEASURED, converged head (epochs = 50, decision-grade)** ✅ *(read from
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-24-branchb-transfer-eval/results_branchb_transfer_e50_CONVERGED.json`;
ckpt md5s above; flagship-v1 frozen paired control md5 `b5f07d9e3dd2ca643949bc86832e6585`, step 29999;
episode-cluster bootstrap over rig-B eval clips, 2000×; clips train rigA 100 / rigB 120 / comma 80, val
rigA 26 / rigB 54, **episode-disjoint**. LOOP_STATE operational run-tag: `ad4e13c4`.)*

Cross = held-out **rig-B** speed R² (the pre-registered headline). Paired dR2 = Branch B − flagship-v1 on
**identical windows + identical converged head-fit** (the C6-clean, regime-robust contrast).

| arm | rig-B set | **Branch B** cross | **flagship-v1** cross | **paired dR2** [95% CI] frac+ | PASS |
|---|---|---:|---:|---|:--:|
| rig_train | train-cache (⚠ LEAKED) | −2.662 | −2.948 | +0.286 [−0.270, +0.988] .83 | ❌ |
| multirig_train | train-cache (⚠ LEAKED) | −1.703 | **+0.382** | −2.085 [−2.902, −1.519] **.00** | ❌ |
| **rig_val** (clean, disjoint) | val-cache | −1.923 | −1.169 | −0.755 [−1.336, −0.108] .01 | ❌ |
| **multirig_val** (clean, disjoint) | val-cache | **−0.667** | **+0.657** | −1.325 [−2.295, −0.801] **.00** | ❌ |

**Branch B's OWN 40k-trained head, in-sample on rig-B** (regime-free — no fresh-head fit): speed R²
**0.156** (train-cache) / **0.242** (val-cache); yaw R² ≈ 0. The deployed model reads rig-B speed at
R² ≈ 0.2 *even where its head trained on rig-B*.

**Verdict — three findings, ranked by robustness:**
1. **Cross-rig transfer FAILS the gate by a wide margin, at every regime.** Best Branch B cross-rig speed R²
   = **−0.667** (gate +0.9); no arm passes; cross **yaw R² is negative on every cross set**.
2. **Weaker dynamics substrate than the plain flagship-v1 encoder — even in-domain.** With a converged
   fresh head (the same fit that gives flagship-v1 in-domain rig-A speed R² **+0.862 / +0.910** — see the
   harness check), Branch B's own in-domain rig-A reads **+0.039 / −0.603**; corroborated by its own 40 k
   head (in-sample rig-B **0.24**). Two independent heads read the latent weakly ⇒ **it is the
   representation, not the head.**
3. **Paired, same-regime: Branch B ≤ flagship-v1 cross-rig.** dR2 CI excludes 0 (Branch B worse) on **3 of
   4** arms (multirig_train −2.085, rig_val −0.755, multirig_val −1.325). The only Branch-B-favoring arm
   (rig_train, +0.286) has a **CI spanning 0** and is the **leaked** arm (Branch B trained SSL + supervised
   IDM on those exact rig-B clips; the edge vanishes on disjoint clips). Flagship-v1 frozen **does** transfer
   to rig-B with a converged multi-domain head (+0.382 / +0.657); Branch B does not.

**Controls + caveats (why this is honest, not an artifact):**
- **Harness validated (MEASURED):** flagship-v1 frozen in-domain (train rig-A held-out, converged head)
  speed R² **+0.862 / +0.910**, reproducing the known frozen-flagship quality (registry frozen in-dist
  ~0.93) — the probe works; Branch B's low numbers are its own.
- **Leakage controlled:** the decision uses the `*_val` sets (`physicalai-val-f1b378f295ae`, episode-disjoint
  from Branch B's `…-train-e438721ae894` training); the `*_train` sets are **best-case** (Branch B trained
  on those rig-B clips) and flagged ⚠.
- **"Held-out rig" = seen GEOMETRY, disjoint EPISODES:** rig-B's cy ≈ 753 geometry *was* in Branch B's
  multi-rig SSL (the GAIA-2 "conditioning ⊗ multi-rig, both required" regime, by design). It fails the
  **easier** seen-geometry test, so the stricter never-seen-rig (YouTube) question is moot.
- **C5 — head-fit convergence is a large lever:** e10 → e50 moved cross-rig R² by 1–3.5 pts. Both JSONs are
  staged (`…_e50_CONVERGED.json` = decision-grade; `…_e10_UNDERFIT.json` = the head-fit-sensitivity lesson).
  The verdict is anchored on the **paired same-regime contrast** + the **regime-free own-head number**, not
  on external point estimates in a different (light-FT) regime.
- **⚠ Residual confound on finding #2:** Branch B trained with `geom_augment`; eval feeds **clean** frames —
  this train/eval mismatch may depress its clean-frame readout. But the deployment target *is* clean
  heterogeneous video, and the cross-rig paired contrast (findings #1/#3) holds regardless. Cheapest
  follow-up: re-encode with matched augmentation to isolate — it does not touch #1/#3.

**⭐ Positive finding (flag, do not over-claim):** flagship-v1's **trained** encoder, frozen + a converged
multi-domain head, is the **stronger cross-rig substrate** — it transfers to rig-B at **+0.657**
(multirig_val) where Branch B collapses (−0.667). **But NOT uniformly rig-robust:** on the single-domain
rig_val arm flagship-v1 is **−1.169**. So the cross-rig problem is **partly narrowed, not solved.** (C5
head-fit note: this +0.38/+0.66 is materially better than the prior −1.61 light-FT baseline — a
head-fit-sensitivity flag for the encoder line, **not** a relitigation of prior MEASURED artifacts.)

> **Decision (pre-registered outcome = "≈ ablation → conditioning insufficient", and stronger, a
> regression).** Both the cheap warm-start ablation (Branch A, −2.1) and the expensive from-scratch Branch B
> are refuted → **explicit camera-conditioning at this scale does not close the cross-rig problem**; the
> deficit is upstream of rig-invariance — representation *quality*. Any further encoder-line spend
> (Plücker / PRoPE geometry-as-input, YouTube pretrain) must be **re-pre-registered** against this evidence.
>
> **HYPOTHESIS pivot — NOT proven; a Sayed-gated NEW training arm, NOT auto-launch.** Since the paired data
> says the frozen flagship-v1 encoder is the stronger cross-rig substrate, a **flagship-warm-started,
> longer-trained, augmentation-matched** encoder variant is the more promising lever than more from-scratch
> conditioning. This is a **new GPU-day arm requiring Sayed's explicit go** (LOOP_STATE: pod3 is held for
> exactly this decision); it does **not** auto-launch. Because flagship-v1 is better-but-not-uniformly-
> rig-robust (−1.169 on rig_val), this pivot is a hypothesis to test, not a fix in hand.

**RETRACTION_LOG:** no retraction (fresh pre-registered result, reported plainly per Operating-Standard rule
5); it re-demonstrates **C5** — cross-rig R² is head-fit-sensitive, so the decision-grade read uses a
converged head + the paired same-regime contrast, never an under-converged point estimate.
