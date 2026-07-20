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

**Efficiency:** deploy tick 11.16 ms / 89.6 Hz (fp16, 1.59× vs fp32); predictor CUDA-graph 2.57×.

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

### 1.4 flagship-v3enc — `flagship4b-v3enc-30k` — 🟢 **RUNNING NOW**

| Field | Value |
|---|---|
| **Status** | 🟢 **RUNNING** on **`tanitad-pod`** (RTX A6000), relaunched **2026-07-20 05:27 UTC from step 0** (fresh, not resumed). At 05:45 UTC the log showed step 100. |
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

**Results:** 🟥 **NONE — no checkpoint has been evaluated.** The only telemetry is the in-training log.

**Pre-registered gates (from the diagnostic + orchestrator recommendation):**
- @10 k: primary ADE@2s ≤ **2.5 m**; encoder speed-probe R² ≥ **0.55**; high-speed long overshoot @2s ≤ **8 m**.
- **Acceptance gate should be the OOD panel, not in-distribution:** beat the comma2k19 floor (0.372 m) on
  **≥ 35 %** of windows, up from v1's 17.5 %.
- **Falsifier:** if v3enc@10 k does not improve the same-step forward-consistency ratio vs v1, restart again.

---

### 1.5 Flagship variants that exist but are not "versions"

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
| `refc_small_config()` | `small` | base_width 64, blocks (3,6,16,6) | d256, 4 heads, 3 layers | 64 / pool 2048 | off | **54,690,001** ✅ | ⚠️ **only a 150-step CPU/GPU smoke** |
| `refc_config()` | `base` | base_width 88, blocks default | d384, 4 layers | 128 / pool 4096 | off | **104,191,577** ✅ *(measured 2026-07-20; the docstring's "~110 M" was 5.6 % high)* | 🟡 **training** — `refc-diffusion-base-v21-30k` (§4.3) |
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

**REF-C-XL finishes 0.006 m behind the deployed flagship v1 (0.4522)** — a budget-matched
direct-head diffusion arm essentially level with the world-model stack.

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

### 4.2 REF-C-small — 🟥 **the scale ambiguity — OPEN**

The **only** instantiation of `refc_small_config()` in the entire fleet is `tanitad-pod3:/workspace/
experiments/refc-smoke320/` — a **150-step, `--mode classifier`, `--config small`** smoke on
`/workspace/refc_smoke_data`, `param_breakdown.total = 54,690,001`. It has **never been trained on the
2,376-ep set and has never been evaluated.** ✅

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

**What is genuinely OPEN:** the three-size scaling study (**small 55 M / base 104 M / XL 252 M on identical
data, read at the 5 k/15 k/20 k/30 k milestone gates**) that D-030 commissioned. The **middle rung is now
training** (§4.3, launched 2026-07-20) and `base` is no longer a docstring estimate — it measures
**104,191,577**. `small` is still a 150-step smoke, and the `base` run carries a label confound, so any
statement about "where bigger helps vs overfits on our data" remains **unsupported until §4.3 finishes and
is evaluated**. 🟡

---

### 4.3 REF-C-base (medium, 104.2 M) — `refc-diffusion-base-v21-30k` — 🟡 **TRAINING**

| Field | Value |
|---|---|
| **Status** | 🟡 launched `tanitad-pod3` 2026-07-20 ~16:40 UTC, 30,000 steps, ~1.23 s/step → **ETA ≈ 10.5 h**. GPU A40, peak 14.4 GiB of 44.4. |
| **Location** | `tanitad-pod3:/workspace/experiments/refc-diffusion-base-v21-30k/` · log `tanitad-pod3:/tmp/refc-base-v21-30k.log` · ⚠️ **single copy, pod-only** |
| **Params** *(measured at instantiation)* | encoder 90,458,632 · decoder 8,634,505 · strategic 1,903,680 · law 2,902,720 · aux 274,760 · measurement 17,280 · imagination **0** (graft off, XL-only) → **104,191,577** ✅ |
| **Parity with XL** | same corpus `physicalai-train-e438721ae894` (2,376 eps / 406,099 windows), 30 k steps, **Adam** lr 1e-4 / warmup 2000 / cosine, same loss weights, `--mode diffusion`, `--batch 20 --workers 6` |
| **Deliberate differences** | `--config base` (2.42× smaller) · **128** FPS anchors — verified a **strict prefix of XL's 256** (`refc_anchors_base128.pt`, same script/source/pool-cap/seed) · H15 imagination OFF (preset design) · **route labels v2.1** |
| **⚠️ Confound** | XL trained with **v1** route labels (`route_target(nav_cmd)` — circular *and* straight-by-default; `labels_v2` was never set in `refc_train.py`). This run uses **v2.1** (`route_from_future_v21`, `use_net_dyaw=False`, ROUTE_UNKNOWN=3 **masked** out of the 0.1-weight CE, never clamped). **medium-vs-XL therefore conflates scale and labels.** Calibration: the flagship v1.5 end-to-end label effect was +0.025 m, not CI-separated. |
| **Label coverage** *(4,000-window sample, in `config.json`)* | left 0.121 · straight 0.5645 · right 0.115 · **UNKNOWN 0.1995 (masked out)** → 80.05 % judgeable, vs v1's straight-by-default target |
| **Code** | `stack/scripts/refc_train.py` gained `--labels {v1,v21}` (**default `v1` = XL-reproducible**), `RouteV21Dataset`, a fail-loud masked route CE, and 5 k/15 k/20 k/30 k **milestone archiving** (the gate series XL lacks). 15/15 `tests/test_refc.py` pass. Pod3 drift repaired before launch (`refb_labels.py` still had `use_net_dyaw=True`; `ckpt_io.py` was absent) — backups in `/workspace/ops/backup-20260720-refcmed/`. |
| **Eval plan** | canonical `taniteval` n=881 / 40 val eps / 8-split jackknife vs XL's **0.458 ± 0.057**, plus `plan_fan` oracle-in-fan + `frac_sel_2x_worse` (note: a 128-wide fan is half XL's, so a worse oracle is partly pure coverage). Registry entry needs only `config_preset="base"` — the loader already resolves it. |
| **Note** | `TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-20-refc-medium-scaling.md` |

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

*All heldout mean ± CI95, 8-split episode-disjoint jackknife, physicalai val. Read from the raw eval JSONs
on `tanitad-eval` this session.* ✅

| Rank | Arm | TanitEval key | Step | Params | ADE@2s | FDE@2s | miss@2m | Beats CV |
|---:|---|---|---:|---:|---:|---:|---:|:--:|
| 1 | **Flagship v1 (speed+jerk) FINAL** | `flagship-30k` | 29 999 | 263.4 M | **0.4522 ± 0.0312** | 0.9437 | 0.0602 | ✅ |
| 2 | **REF-C-XL** (anchored diffusion) **FINAL** | `refc-xl-30k` | 29 999 | 251.9 M | **0.458 ± 0.057** | 0.972 | 0.146 | ✅ |
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

---

## 7. Reconstruction-gap register — what would block a rebuild today

| # | Gap | Severity | Where the only copy lives | Fix |
|---|---|---|---|---|
| R1 | **REF-B v2 (`--arch-v2`, anchored tactical, `yaw_input`) is uncommitted** — our 3rd-best arm | 🔴 critical | `tanitad-pod:/root/refb_train_v4.py`, `/root/refb_v4.py`, + the modified pod working tree | commit the pod diff to `stack/tanitad/refs/refb.py` + `stack/scripts/refb_train.py` |
| R2 | **TanitEval is uncommitted** — every headline ADE in this document | 🔴 critical | `tanitad-eval:/root/taniteval/` (incl. `registry.py`, `bench.py`, `closedloop.py`, `refc_eval.py`, `generalization.py`, `pathspeed.py`, `hierarchy.py`) | vendor it into the repo, or at minimum `registry.py` + the metric definitions |
| R3 | **P2 planner is uncommitted** — the entire v3 evidence base | 🔴 critical | `tanitad-eval:/root/taniteval/taniteval/planner_p2.py` | commit alongside R2 |
| R4 | **Flagship v1 `--jerk-weight` / `--aux-accel` missing from the committed trainer** — the deployed model is not byte-rebuildable from HEAD | 🟠 high | `tanitad-pod2:/workspace/TanitAD/stack/scripts/train_flagship4b.py` (shows `M`) | commit the pod2 diff |
| R5 | **`LEADERBOARD.md` is stale and in the wrong units** — newest row is camera-frame ADE@1s @27 k (2026-07-12); every current number is metric-BEV `ade_0_2s` | 🟠 high | in-repo | rewrite from §6 above, and label units |
| R6 | **`gate-eval` skill targets a dead run** (`p0-sB01-realmix`, frozen since 2026-07-12 @ step 28,600) | 🟡 medium | `.claude/skills/gate-eval/SKILL.md` | retarget to the live arm |
| R7 | **REF-C three-size scaling study never ran** — 🟡 **middle rung now training** (`base` measured 104,191,577 and launched 2026-07-20, §4.3); `small` still only smoked, and the `base` run carries a **scale/label confound** (v2.1 labels vs XL's v1) | 🟡 medium | n/a | let `base` finish + evaluate; then either add a label-controlled arm or state the confound wherever the ladder is quoted |
| R8 | **REF-A I-JEPA canonical-val result is leak-contaminated** (80 % of val in its train set) | 🟡 medium | flagged in `taniteval/registry.py` | re-evaluate on the `f1b378` val before any comparative claim |
| R9 | **REF-B rev2, `refa-4brain-speedyaw-30k`** have no eval record | 🟢 low | n/a | either evaluate or mark explicitly superseded |
| R10 | **`refb-speed-30k/ckpt_prepatch_step8500.pt` is misnamed** — it is step 10,000 and byte-identical to `ckpt.pt` | 🟢 low | pod1 | rename |

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
3. **Closed-loop is self-referential.** The imagination-in-the-loop harness uses the world model as both
   state estimator and simulator. An external photoreal sim (AlpaSim/NuRec) is the only cure and is
   currently unrunnable — AlpaSim needs nested docker in an unprivileged container; CARLA pixels are
   host-blocked. D5/D6 remain 🔴 **BLOCKED — infra**.
4. **D2/D3 gate evidence is stale** — measured at step 27 k on the *pre-reset* `p0-sB01-realmix` run, in
   camera-frame units, and never re-gated after the speed reset.
5. **VTARGET's exact placement in the frozen goal vocabulary** should be read from
   `V3_GOAL_VOCABULARY_V1.md` before implementation (D-A9). ⚠️

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
