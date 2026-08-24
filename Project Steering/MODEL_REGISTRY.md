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
> log) · `TanitAD Research Lab/HYPOTHESIS_LEDGER.md` (H-numbers) · `Paper/TANITAD_PAPER.md`.

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

#### 0.1.1 ⚠️ NAMED REGIME BOUNDARY — the steer-label wheelbase (option B, PI-approved 2026-07-26)

**Two label regimes exist. They are NEVER comparable, and no paired test may cross this line.**

| regime | steer label | cache-key contribution | who is in it |
|---|---|---|---|
| **`const2p9`** (LEGACY) | `atan(2.9 · κ)` for every clip | **none** — `e438721ae894` keeps its exact current meaning | **every arm, cache and published number in this registry** |
| **`per_clip_v1`** (CORRECTED) | `atan(L_clip · κ)`, `L_clip` joined per clip from the dataset's own `calibration/vehicle_dimensions` | adds `wheelbase_mode`, so a corrected cache **cannot collide** with a legacy one | new arms only, from 2026-07-26 |

**Why the boundary exists.** `WHEELBASE = 2.9` is an approximation, not a platform fact — **no Hyperion
platform in this corpus has a 2.9 m wheelbase.** MEASURED over the 197 chunks the parity corpus draws
from: **2.730 (47.0 %, the mode) · 2.850 (1.8 %) · 3.135 (13.9 %) · 3.165 (25.5 %) · 3.216 (11.8 %)**,
clip-mean **2.9568 m** ⇒ **98.2 % of clips carry a >5 % error**, and the populations are geographically
coherent (`I(wheelbase; country) = 0.769` of `H = 1.880` bits; the 2.85 m slice is **100 % United States**).

**What it does NOT change — the parity statement.** Option B is **fix-forward only**. It changes **no**
existing cache, **no** existing arm and **no** published number. **It does not re-select episodes**: the
cache key hashes the *ordered clip-id list* plus build params, and the clip list is untouched — the fix
alters label VALUES for future builds only. `e438721ae894` (2,376 episodes, skip-hash `f09e44db`) is
bit-identical, enforced in code by `physicalai.label_params()` returning `{}` for the legacy regime and
asserted by `stack/tests/test_wheelbase_regime.py`.

**Measured size of the defect** (input-side bound on flagship-v1, paired episode-cluster bootstrap,
B = 2000, 881 windows / 40 val episodes): ADE **+0.00560 [+0.00070, +0.01130]** m (**+1.31 %**,
CI-separated) — but **cross-track@2s 0.2742 → 0.2977 m, +8.6 % relative**, which is the largest honest
statement of its size. A re-baseline (option A) was rejected at **~53 A40-days** to move numbers by
**9.3 % of their own CI half-width**. Full measurement:
`…/incoming/2026-07-26-wheelbase-impact/WHEELBASE_IMPACT.md`; execution:
`…/incoming/2026-07-26-trafficsim-wheelbase/TRAFFICSIM_WHEELBASE.md`.

⚠️ **Related, and NOT the same constant:** `taniteval/closedloop.py` mints and integrates steer at
**2.7 m**. The closed-loop *path* is wheelbase-invariant (it cancels), but the action *fed to the model*
is 2.7-derived while the model trained on 2.9 — a **+7.41 % train/serve skew**, open-loop cost
**+0.0026 [−0.0006, +0.0062]** (not separated). Documented in code, **not** silently changed, because
aligning it would move every future closed-loop number. **Open PI decision.**

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
| Statistic | 🟥 **`heldout` = `overlapping_holdout_se` — DEPRECATED, and BOTH its mean and its interval are defective.** *(This row read "**8-split episode-disjoint jackknife**" until 2026-07-26 — a **retracted label**, corrected in §6 1,300 lines below but not here, so every reader who stopped at §0.3 inherited it.)* `val_frac 0.2`, 8 **overlapping** random holdouts: the interval is **1.107–3.100× too narrow** and the "mean" is a **mean-of-split-means** that shifts the point estimate **−6.67 % to +11.69 %, bidirectionally** (27 dumps = 25 distinct arms — C126). **Decision-grade = `full_set` mean + `taniteval/ci.py` episode-cluster bootstrap; paired for two arms.** Both are published; they differ. **Always name which — and never decide on `heldout`.** |
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
| **Location** | 🔴 **`tanitad-pod2` IS TERMINATED.** The run dir `/workspace/experiments/flagship4b-phase0-30k/` went with it, and it is **absent from pod4's rescue dump** (MEASURED 2026-08-03, read-only `ls /workspace/rescue/experiments/`). **Reachable copies — of `ckpt.pt` ONLY:** (a) dev box `_pod_backup/pod2-2026-08-03/ckpts/` file `flagship4b-phase0-30k_ckpt.pt`, **3 302 176 350 B**, md5 `74be81035699c362e2fd0e5197880506` (per `_pod_backup/pod2-2026-08-03/ckpts/BACKUP_LOG.txt`) — ⚠️ **git-ignored by `.gitignore:48`, so it is NOT in the repo and one disk failure ends it**; (b) HF `Sayood/tanitad-flagship-4b-phase0`. ⛔ **`config.json`, `train_log.jsonl` and the per-gate JSONs have no reachable copy.** |
| **⛔ UNRESOLVED citation** | This row used to cite `gate_step{1k,5k,10k}.json`, which is wrong twice over: a **shell brace expansion names no file**, and the stem is wrong as well. MEASURED 2026-08-03 from the emitters — `stack/scripts/watch_gates.py:213` and `stack/scripts/evaluate_checkpoint.py:201` both write `f"gates_step{step}.json"` — so the real name is **gates_step<full-integer>.json** (plural `gates`, no `k` abbreviation), and **no `gate_step*` writer exists anywhere in the tree**. ⛔ **I did not rewrite it to a guess**: the host is terminated and the dir is not in the rescue dump, so neither the filenames nor the gate steps can be verified, and a confidently wrong citation is worse than an obviously broken one. Tracked as the single `unresolved` entry in `tools/registry_paths_allow.json`; resolving it means finding the run dir, then lowering that file's `max_unresolved`. |
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
| **ADE@2s (`ade_0_2s`)** ⚠️ **= `wm_fidelity_ade_2s` — a WORLD-MODEL FIDELITY number, NOT a planning number** | **0.4522 ± 0.0312** | **0.4271** |
| FDE@2s | 0.9437 ± 0.0630 | — |
| miss@2m | 0.0602 ± 0.0121 | — |

⛔ **WHAT THIS ROW MEASURES — verified in code 2026-07-27, and it is NOT what it has repeatedly been
quoted as.** `taniteval/rollout.py:170` sets **`actions_source="expert_future"`** and `:174` names the
metric **`wm_fidelity_ade_2s`**. ⇒ **the model is HANDED THE EXPERT'S TRUE FUTURE ACTIONS and asked
only to roll them out.** `0.4271` therefore measures **how well v1's world model integrates known
actions**, not how well anything *chooses* them.

⚠️ **CONSEQUENCE: `0.4271` IS NOT A PLANNING BAR AND MUST NOT BE USED AS ONE.** A selector that must
*choose* actions is not expected to beat a world model that was *given* them — that bar is closer to
an oracle than to a baseline, which is part of why nothing in the program has cleared it. *(Flagged
independently by two streams on 2026-07-26 and 2026-07-27; it had already propagated into `V5_PLAN.md`'s
STRONG bar and into several agent briefs, mine included. The legitimate same-surface bar is the
in-sample re-scoring ceiling **0.4907**.)*

⚠️ **AND `0.4907` CARRIES ITS OWN DEPLOYMENT TAG — added 2026-07-27, because it is being quoted
against 600-episode numbers.** **`0.4907` is an 881-window / 40-episode number** (the canonical val
deployment, the same surface as `0.4271`), on which the `a0` as-trained reference reads **0.4714**.
**MEASURED** — `…/incoming/2026-07-26-bar-a-selector/raw/bar_a_produced.json`:
`in_sample_ceiling.ce.ade_0_2s_in_sample = 0.4907`, `_cache.n_eval_windows = 881`,
`_cache.n_episodes = 40`, `_goal_mode "produced"`, `_ckpt_step 29999` — i.e. an **IN-SAMPLE** ceiling
(*"fit and scored on the same windows … NOT deployable, NOT a generalization number"*, the artifact's
own `_read`) over the **frozen** produced-surface fan of `flagship-v4-fromscratch` @ 29 999 (§1.5.5).
`0.4714` = REF-C-XL's full-set `ade_0_2s` on that same deployment (§1.5's v1.6 table, row *ADE@2s
full-set*). ⛔ **The 600-episode panels are a DIFFERENT deployment** — 13 198 windows / 600 episode
clusters, where the same `a0` reads **0.5015** (`…/incoming/2026-07-28-egoal-4-joint/raw/e4_summary.json`,
`deployment.a0_as_trained`). **A 0.4907-vs-600-episode comparison is cross-deployment and is not a
result**, by the same rule that forbids substituting v1's 600-ep 0.4108 for its 0.4271 (§1.2a).

Clears every trivial bar on the same 881 windows: best-of-3 kinematic floor 0.5005 · CTRV oracle 0.523 ·
no-vision ego-status ceiling 0.5735 · CV 0.8248 heldout / **0.8377 full-set**.
⚠️ **Read this comparison LIKE-FOR-LIKE (noted 2026-07-26).** The three bars are **full-set means by
construction** (`bench.py:511`, `:558`) while `0.4522` is a split-mean — so the row as originally written
compared two different estimators. On the full set the verdict **survives unchanged: 0.4271 vs
0.5005 / 0.523 / 0.5735 / 0.8377.**

#### 1.2a ⭐ v1's TWO val DEPLOYMENTS — added 2026-07-26. **Two rows, never one number.**

`flagship-30k` now has a reference on the **600-episode** clean val as well as the canonical 40. The
600 build is a strict **order-preserving superset** of the 40 (`published40[i] == val600[i]` for every
`i ∈ [0,39]`, MEASURED — `…/2026-07-26-pod2-eval-host/artifacts/prefix_disjointness_result.json`), so
parity is not violated. **The two numbers are still not interchangeable**, and both are recorded here so
nobody has to choose which one "the" v1 number is.

| deployment | `ade_0_2s` **full-set mean** | **episode-cluster bootstrap** CI95 (B = 2000, unit = val episode) | n windows | **n episode clusters** | CV floor on the SAME deployment | paired Δ vs CV | artifact |
|---|---|---|---:|---:|---|---|---|
| **40 eps — CANONICAL** (`--episodes 40`) | **0.4271** | **[0.3675, 0.4871]** hw 0.0299 | 881 | **40** | **0.8377** | +0.4106 [+0.2050, +0.6240] ✅ sep | `taniteval/results/driving_flagship-30k.json` · re-run `…/2026-07-26-pod2-eval-host/artifacts/RESULT_v1_40ep_preflight.json` |
| ⭐ **600 eps — NEW, NOT a correction** (`--episodes 600`) | **0.4108** | **[0.3956, 0.4273]** hw 0.0159 | **13 198** | **600** | ⚠️ **0.6917** | +0.2809 [+0.2457, +0.3142] ✅ sep | `…/2026-07-26-pod2-eval-host/artifacts/RESULT_v1_600ep.json` |

> ⛔ **THE 600-EPISODE NUMBER IS NOT A CORRECTION TO 0.4271 AND MUST NEVER BE SUBSTITUTED FOR IT.**
> It is a **different deployment**, and the reason is measurable rather than stylistic: **the 600 is an
> EASIER corpus.** The trivial CV floor moves **0.8377 → 0.6917**, i.e. the 560 added episodes are on
> average *more predictable* than the published 40. v1's margin over the floor therefore **falls**
> (paired Δ **+0.4106 → +0.2809**) even though its absolute ADE improves. **A 40-vs-600 delta is
> confounded by corpus composition and is not a model result.**
>
> **Rules, binding:**
> 1. **Never mix.** Every arm in a comparison must be on the *same* deployment. The §5 leaderboard is the
>    **40-episode / 881-window** table; `0.4108` may not appear in it, and no arm's 40-episode number may
>    be compared to another arm's 600-episode number.
> 2. **Always quote n as BOTH counts** — windows *and* episode clusters. The episode cluster is the
>    resampling unit; stride changes windows by up to **×5.9** and clusters by **exactly 0**
>    (MEASURED, `…/2026-07-26-pod2-eval-host/artifacts/horizon_analysis.json`).
> 3. **Estimator named on every interval:** `episode_cluster_bootstrap` (`taniteval/ci.py`) B = 2000,
>    paired form for any two arms on shared windows. The `legacy_overlapping_holdout_se` block in both
>    JSONs is **not quoted here and must not be**.
>
> ⭐ **What the 600 buys, MEASURED not projected:** the CI half-width shrinks **×2.8–3.9 (mean ≈ 3.4)**
> across eight open-loop metrics, against the **×3.87** that `√15` predicts. One driving-panel verdict
> flips on power alone — `along_track_vs_cv` goes δ 0.2543 **[−0.0278, +0.5304] "tie"** at 40 to
> δ 0.2525 **[+0.1926, +0.3104] "model wins, separated"** at 600, with the point estimate moving **0.7 %**.
> ⚠️ **Consequence for this registry: any verdict here that rests on a 40-episode "not separated" is
> UNPOWERED, not refuted.** (`…/2026-07-26-pod2-eval-host/artifacts/v1_40_vs_600.json`.)
>
> Both rows are `wm_fidelity_ade_2s` under `pc2` — the scored pass does not traverse the hierarchy
> (`actions_source=expert_future`). Same protocol on both, which is why they are comparable *to each
> other* in kind, and not in value.

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

**Closed-loop (imagination-in-the-loop, no renderer):** closed_bike ADE@2s **1.7318** [1.5707, 1.9070]
(episode-cluster bootstrap, 881 win / 40 ep), FDE **3.6190** [3.2453, 4.0215], divergence >5 m **23.50 %**
[16.80 %, 30.27 %]. Open-loop **0.4271** → closed-loop **1.7318** (**4.05×**): **open-loop does not
predict closed-loop.**

> ✅ **THE 2026-08-03 "estimator NOT STATED" FLAG IS ANSWERED AND CLOSED — 2026-08-17.** It was
> `overlapping_holdout_se`, the banned estimator. This line previously read *"1.685 ± 0.098 … FDE 3.530,
> divergence >5 m 22.2 %"*; **all three were legacy `heldout` split-means**, and all three are corrected
> above. MEASURED from `…/incoming/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json`
> (`cluster_bootstrap.model` vs `legacy_overlapping_holdout_se.heldout.closed_bike`):
>
> | | superseded `heldout ± overlapping_holdout_se` (BANNED) | **decision-grade full_set [episode-cluster bootstrap]** | point shift | CI too narrow by |
> |---|---|---|---|---|
> | ADE@2s | *1.6852 ± 0.0977* | **1.7318** [1.5707, 1.9070] | **−2.69 %** | **1.722×** |
> | FDE@2s | *3.5296 ± 0.2548* | **3.6190** [3.2453, 4.0215] | **−2.47 %** | **1.523×** |
> | divergence >5 m | *0.2216 ± 0.0431* | **0.2350** [0.1680, 0.3027] | **−5.70 %** | **1.564×** |
>
> ⭐ **All three moved the same direction: the closed-loop failure was UNDERSTATED.** ⚠️ **And 1.6852 is
> exactly the number that became the `G4_pass` threshold** (§5) — a legacy split-mean promoted to a gate
> bar, making the old gate 2.69 % *harder* than the honest one. The open-loop→closed-loop degradation
> ratio also sharpens from 3.73× to **4.05×**.

**Efficiency — TWO DIFFERENT TICKS. Neither may be quoted without its definition.**

| tick | what it actually measures | hardware | ckpt / corpus | p50 | Hz |
|---|---|---|---|---|---|
| **decision tick** = `encode(1 frame) + select_K9` | imagine-and-select only — **does NOT include the 20-step rollout** | RTX 4060 (declared Orin proxy, single-stream) | step **6,500** / **comma2k19** | **11.16 ms** fp16+CUDA-graph (17.75 fp32, **1.59×**) | 89.6 |
| **planning tick** = `encode(8-frame window) + 20 SEQUENTIAL predictor steps → per-step metric Δpose → SE(2) accumulate` | the intent-free operative path that **produces the trajectory ADE@2s scores** | A40, exclusive (contamination-checked) | step **29,999** / **physicalai val** | **103.42** fp32 · **93.76** tf32 · **104.49** amp16 | 9.7 / 10.7 / 9.6 |

**A THIRD instance of the *decision* tick exists and was missing from this registry — added 2026-07-26.**
`Paper/TANITAD_PAPER.md §7.10` publishes **14.331 ms p50** (encode 9.273 + K = 9 select 5.058) as the
beyond-ADE suite's headline latency — rounded to **14.33 ms** in `PROGRAM_OVERVIEW.md:54` and
`LOOP_STATE.md:64` — and that value appeared **nowhere in this document** until now: the exact defect
corrected below for "11.16 ms", repeated on a second tick. Traced and MEASURED:
`TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/2026-07-24-traffic-light-scenario-metric/real_tms_cnce.json`
(`latency.decision_tick_p50_ms`), generator `real_telemetry_tms_cnce.py:109`. **Conditions:** RTX 4060 ·
**fp32 eager** (no autocast, no CUDA graph) · **comma2k19 val, n = 30 episodes** · log-replay ·
architecture **`base250cam`, `params_billions` 0.2628 = 262.8 M, instantiated fresh (random init)** —
latency is weight-independent, so this is an *architecture* read and **not** a read of the deployed
263.44 M flagship. It is the same tick definition as row 1: **14.331 fp32 (base250cam) vs 17.75 fp32
(step 6,500)** differ by architecture config, not by regression. Do not quote it as "the deployed
architecture".

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
`TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-22-orin-thor-deployment/artifacts/`;
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
> arithmetic** (0.4522 + the paired 0.188 m win-delta), never a measured mean.
> ⛔ **THE CITE INSTRUCTION IS REVERSED — CORRECTED 2026-08-17.** This note ended *"Cite 0.628
> (heldout)"*, i.e. it **instructed readers to quote the banned split-mean** over the decision-grade
> full-set value sitting next to it. That is backwards under §0.3's own rule (*"Decision-grade =
> `full_set` mean + episode-cluster bootstrap … never decide on `heldout`"*). ⇒ **Cite 0.6152
> [0.5422, 0.6951] (full-set, §6).** The note's actual contribution — that 0.640 is derived arithmetic
> and never a measured mean — stands and is the reason to keep it. *(This is the sharpest instance of the
> class in the document: not a stale number, but a **standing instruction to prefer the banned one**.)*

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

ADE@0.5s **1.2389** · ADE@1s 2.3276 · ADE@1.5s 4.0048 · **ADE@2s 6.179 ± 1.2845** (7.4× CV; ⚠️ **estimator label added 2026-08-03** — `±` in this document is `overlapping_holdout_se`, **DEPRECATED and BIASED in the point estimate** (§1.4's header states the convention). The decision-grade interval is the episode-cluster bootstrap; §6 gives it as **5.9396 [4.3273, 7.6249]**) · FDE@2s
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

### 1.4b flagship-v1.6 — `flagship-v16-ab-ft` — ✅ **COMPLETE at 5,999** · ⚖️ **TIED with the deployed v1 (NOT "best in program")**

> 🔴 **HEADLINE CORRECTED 2026-07-25.** This header read *"⭐ best ADE in the program"* for four days
> **after its own body had retracted that claim** (C1, 07-21) — the retraction edited the prose and left
> the headline standing. Decision-grade re-derivation (**paired episode-cluster bootstrap**, B=2000,
> 40 episodes / 881 windows, corr 0.453, reproduced digit-for-digit):
> **Δ(v1.6 − v1) = +0.0104 m, CI95 [−0.0888, +0.1147] — NOT separated**, and v1.6 is *behind* on the
> point estimate (**0.43746 vs v1's 0.42711**; lower is better). The tie is **not** a power artifact —
> the paired half-width (±0.1018) is *tighter* than the invalid quadrature (±0.1199) and still spans 0.
> ⚠️ The table's `0.4886 ± 0.0800` below is the **deprecated `overlapping_holdout_se`** (verified:
> 1.96 × 0.1155 / √8 = 0.0800 exactly) — **not quotable**; widths run 1.30×/1.92× narrow vs the paired
> bootstrap, inside the documented 1.28–2.06× band. **Both G-A gates are TIES → treat as UNRESOLVED,
> not pass/fail** (v1.6 vs REF-C-XL: Δ −0.0340 [−0.1060, +0.0511], also not separated), which settles
> the contradiction between the `❌ ❌` and `✅ ✅` gate lines further down this section.
> Raw: `…/incoming/2026-07-25-v16-paired-interval/v16_vs_v1_paired_bootstrap.json`.
> ⚠️ **STRANDING: v1.6's ckpt exists on exactly ONE disk (pod2, currently training v4 = off-limits);
> `Sayood/flagship-v16-ab-ft` holds NO weights and §1.4b has no `Location` row.**

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

⚠️ **`heldout` means are NOT comparable across arms — v1.6's 0.4886 vs v1's 0.4522 is an invalid
comparison, and the paired bootstrap above is the valid one.** *(Conclusion unchanged. **MECHANISM
CORRECTED 2026-07-26** — the reason previously given here was wrong, and wrongly reassuring; see below.)*

Two eid FAMILIES do exist: `bench.py` clusters on **file indices 0–39**, while
`stack/scripts/eval_flagship_v15.py` and `stack/scripts/eval_flagship_v16.py` deliberately cluster on
the **real `episode_id`** (`real_episode_ids()` — `eval_flagship_v15.py:125`, `eval_flagship_v16.py:171`;
e.g. 808464434) *because the eval pod's estimator does* — see the v15 docstring
(`stack/scripts/eval_flagship_v15.py:128-130`): *"using file indices instead would produce a DIFFERENT
episode partition and therefore a different heldout mean."*

> 🔧 **CORRECTION (2026-07-26). This block used to say `split_by_episode` "hashes the id *values*", so
> the two families drew "different random partitions". IT DOES NOT HASH.** MEASURED from the emitter:
> `stack/tanitad/eval/gates.py:139-152` takes `sorted(set(int(e)))` and hands it to
> `stack/tanitad/instruments/checks.py:49-58` (`i3_episode_split`), which calls
> `torch.randperm(len(episode_ids))` — it permutes **positions in the sorted list**, never the values.
> An **order-preserving** relabelling therefore yields the **identical** partition. MEASURED
> 2026-07-26 on the two dumps themselves: both `windows_flagship-30k.pt` (0–39) and
> `windows_flagship-v16-ab-ft.pt` (real ids) are order-preserving w.r.t. file order, and
> `split_by_episode` returns **byte-identical val index lists for all 8 seeds** (176/881 windows each).
> The v15 docstring's warning is a **real hazard for an order-CHANGING relabel** — it just is not what
> happened here.
>
> **Why the correction matters rather than being pedantry:** the old wording implied that *same-family*
> split-means ARE comparable. **They are not.** The defect is the **estimator itself** — the split-mean
> is biased −6.67 % to +11.69 % per arm, bidirectionally (§6), which is present *within* a family and is
> larger than most gaps being compared. v1.6 is the extreme case in the whole program: its split-mean
> 0.4886 overstates its full-set 0.4375 by **+11.69 %**, while v1's 0.4522 overstates 0.4271 by
> +5.88 % — so the legacy Δ reads `+0.0364` where the true full-set Δ is `+0.0104` (**×3.5**). No part of
> that distortion needs two eid families to occur. *(Root-cause class: a plausible mechanism was inferred
> from a correct observation and never read off the code — the same class as the `df`/quota and
> `step_s` traps in `CLAUDE.md`.)*

Both families partition the same 40 episodes, so **episode-cluster bootstrap and full-set are
unaffected** either way. Alignment for pairing was proven from the data, not assumed: `gt` and `cv` are
identical **elementwise, max diff 0.0** (re-verified 2026-07-26).
⚠️ **Use `eval_flagship_v16.py` ONLY**: it re-encodes val frames through the unfrozen trunk; cached
`states_val.pt` would silently score the OLD trunk. The pod copy was stale (248 lines, no
`windows_*.pt` persistence) and was synced from HEAD before this run.

**The result, stated honestly — unfreezing TRADES the world model for planning quality:**
- planning **improved**: ADE −18.7 % vs `ab`, and the fan improved **−8.4 %** (0.3073 → 0.2815);
- the world model was **destroyed**: the intent-free canary went **0.452 → 1.1022, +144 % (2.44×)**.

**G-A verdict: PARTIAL — 2 of 5.** ✅ G1 (<0.458) ✅ G2 (<0.4522) · ❌ oracle ≤0.22 (0.2815)
· ❌ miss ≤0.10 (0.1067, narrow) · ❌ canary flat (+0.650). **Branch 1 does not fire.**

**What it settles:** unfreezing 4 ViT blocks is **not** the route to a REF-C-grade fan, and the WM
cost makes it actively hostile to a design built on that WM (v3.5). Frozen-vs-trained encoder is
confirmed as the *direction* only.

⛔ **CORRECTED 2026-08-03 — the earlier sentence here conflated two different quantities.** It read
*"buys only 8.4 % of the fan gap — nowhere near REF-C's 0.1640"*. Both halves were wrong:

| quantity | value |
|---|---|
| **relative** improvement in the fan number (0.3073 → 0.2815) | **8.40 %** ← *this is what 8.4 % actually is* |
| fraction of the gap **to REF-C-XL** (0.1640) closed | **18.0 %** |
| fraction of the gap **to REF-C-base** (0.1914) closed | **22.3 %** |
| fraction of the distance to **this gate's own `oracle ≤ 0.22` bar** closed | **29.55 %** |

**8.4 % is a relative change in the fan, not a fraction of a gap.** And **0.1640 is REF-C-XL's fan,
not base's** — base is **0.1914**, so comparing a base-scale lever against XL's number overstates
the shortfall. The verdict is unchanged (18–22 % of a gap is not closing it, and the canary went
+144 %), but the magnitude must be quoted correctly.

⚠️ **DO NOT MERGE THIS 8.4 % WITH THE OTHER ONE.** §4's *"a learned re-scorer recovers at most 8.4 %
of the gap across 47 trained arms"* is an unrelated quantity that coincidentally shares the digits —
that one **is** a fraction-of-gap, about a re-scorer, on REF-C. This one is a relative change, about
unfreezing, on the flagship. Two numbers, one string; quote the section, never the bare figure.
⚠️ The re-scorer 8.4 % resolves to a **prose note, not a results JSON** — it is `INHERITED` until
someone re-derives it from the 47 arms' raw output, and it is load-bearing for the D-SEL adverse
prior (`PREREG_D-SEL_REFC_SELECTION_SURFACE.md`).

⚠️ **Process note (mine).** At step 2500 a transient spike (oracle 2.08, gnorm 161) plus a monotone
canary trend led me to report a "decisive failure". **It recovered completely** — 5,999 lands ~~the best
ADE in the program~~ **STATISTICALLY TIED with the deployed v1** (corrected 2026-07-25; paired bootstrap
Δ +0.0104 [−0.0888, +0.1147], v1.6 behind on the point estimate — see the header note). *This sentence
wrapped across a newline and so evaded the line-based grep of the 07-21 retraction sweep — presence-
detection hitting the same trap as absence-detection; retraction sweeps must be MULTILINE.*
The confirming-eval discipline saved the run; the premature *communication* did
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
> ⚠️ **STATUS MOVED 2026-07-27 — this block was written while the from-scratch arm was still a
> *fallback*; it has since RUN TO 30 k** (§1.5.5, corrected the same day). Its real-run canary is
> **1.1409** from a random-init baseline of **15.6742** — i.e. **no collapse**, consistent with the
> hypothesis, but still far from v1's **0.452**. **This line records the new evidence and does NOT
> re-adjudicate the hypothesis**, which remains owned by the v4.2b Phase-B read.

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

#### 1.5.5 flagship-v4 from-scratch — `flagship-v4-fromscratch` — ✅ **COMPLETE (30 k, `rc=0`)**

> 🔴 **ROW CORRECTED 2026-07-27 — it recorded an INTENT and was never advanced to an OUTCOME.**
> **Superseded text, kept visible on purpose** (a registry whose history is erased cannot be audited).
> Until 2026-07-27 the heading read *"✅ **READY, not launched**"* and the Status row read verbatim:
> *"✅ **CODE STAGED + VALIDATED, NOT LAUNCHED.** Fires **only** if v4.2b's Phase-B canary runs away,
> and only on Sayed's go. **Zero GPU-day committed.**"*, with **Cost** *"~53 h / 30 k (ESTIMATED)"*.
> **All of that was false from 2026-07-23T21:54:44Z onward** — the arm launched that night, ran
> **59.04 h** to `final_step 29999`, exited `rc=0`, and by 2026-07-27 was **the substrate of every
> selection experiment of the week** (Bar A, T3, E-V5-1, the fan work, E-GOAL-1→4). A reader auditing
> those results was told their substrate did not exist. **Every field below was re-read from the run's
> own artifacts on pod2 on 2026-07-27** (`metrics.json`, `supervisor.log`, `config.json`, `train.out`,
> `ls -la`) — **not** from any summary, changelog or report. Root-cause class and the standing check
> that would have caught it: `Project Steering/RETRACTION_LOG.md`, entry **2026-07-27 / class C41**.

| Field | Value |
|---|---|
| **Status** | ✅ **LAUNCHED 2026-07-23T21:54:44Z, RAN TO COMPLETION 2026-07-26T09:01:37Z.** **MEASURED** (mine, direct pod2 reads 2026-07-27): `metrics.json` → **`final_step 29999`**; `supervisor.log` → *"launch attempt (**restarts=0**)"*, *"trainer **pid=108011**"*, **`trainer exited rc=0`**, *"clean finish"*, *"supervisor exiting (run complete)"*. ⚠️ *"clean finish (**summary.json** done)"* is a **fixed log string** — `summary.json` does **not** exist in the run dir; the branch of `is_done()` that actually fired is the trainer's own terminal line in `train.out`: `{"done": true, "final_step": 29999, …}` (both re-read; state it this way so nobody hunts a `summary.json` artifact). **Parity intact** — the run's `config.json` carries `train_corpus_key physicalai-train-e438721ae894`, `skip_hash f09e44db`. Spec: `…/incoming/2026-07-23-v4-fromscratch/V4_FROMSCRATCH_LAUNCH.md`; launch record `…/incoming/2026-07-23-v4-fromscratch-launch/LAUNCH_CONFIRMED.md`. |
| **Distinguishing lever** | **`--from-scratch`** = skip `_warmstart_trunk`, random-init the trunk (the not-frozen gate then passes trivially). Byte-identical to v4.2b in every other flag (eff batch 64, lr_trunk 1e-4, floor 0.25, λ_plan sched) → **one-lever attributability**: if v4.2b's canary runs away and this one does not, the warm-start coupling is confirmed as the cause. |
| **Validation** | MEASURED: full `pytest -q` **786 passed / 2 skipped** + 5 new from-scratch tests (14 passed in `test_train_flagship_v4.py`). `--smoke-loop` from random init: canary baseline **1.5189** → drops monotonically **1.385 → 1.312 → 1.232 → 1.179 → 1.165** as the WM co-evolves — **improves, does not collapse** (the existence-proof for the through-line above). |
| **Cost** | **MEASURED 59.04 h** — `metrics.json` `wallclock_s` **212 544.6** (= 212 544.6 / 3600), on **pod2 / NVIDIA A40 46 068 MiB**, host `08f6ce7d8e55` (the supervisor's *"supervisor UP on 08f6ce7d8e55"* matches pod2's live `hostname`, both MEASURED 2026-07-27). ⚠️ The superseded row carried **~53 h ESTIMATED**: the estimate was **~11 % low**, and *"zero GPU-day committed"* understated the real spend by **≈2.5 GPU-days**. |
| **Result** | ⚠️ **Two protocols — never mix them.** **(a) TRAINER in-loop val** (`metrics.json`, n = **881**): `ade@2s` **0.5063**, `oracle_ade@2s` 0.1892, `sel_gap@2s` 0.3172, `miss@2m` 0.2145; `canary_ade@2s` **1.1409** against a random-init `canary_baseline` **15.6742** (the co-evolution signature — no collapse). ⛔ **Trainer val is NOT quotable against v1** (§0, and the v1.6 retraction at §1.5's sibling row). **(b) DECISION-GRADE** `eval_flagship_v4.py`, **episode-cluster bootstrap**, **881 windows / 40 val episodes**: @ step **15 000** `ade_0_2s` **0.5839 [0.4962, 0.6821]**, **paired** vs deployed v1 **+0.1568 [+0.0630, +0.2504]** (`p_delta_gt0` 1.0, ✅ separated; **positive = BEHIND v1**) — `…/incoming/2026-07-25-flagship-v4-midtrain-eval/flagship-v4-fromscratch-15k.json` + `…_lateral_and_paired.json`. @ step **29 999**: deployable **produced** surface **0.8563 [0.7282, 1.0035]**, `beats_cv` **❌** (CV floor 0.8377 on the same windows); **oracle-goal** surface 0.6423 [0.5348, 0.7586], `beats_cv` ✅ but `goal_provenance.deployable = false` ⇒ **not a leaderboard number** — `…/incoming/2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-produced.json` and `…/incoming/2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-oracle.json` (both `ckpt_step 29999`). ✅ **Both raw files re-read 2026-08-03 and both match this row exactly**: `cluster_bootstrap.model.ade_0_2s` 0.8563 [0.7282, 1.0035] and 0.6423 [0.5348, 0.7586], `goal_provenance.deployable` `true` and `false`. |
| **Code state** | `--from-scratch` / `--trunk none` sentinel in `stack/scripts/train_flagship_v4.py`; the run's own `config.json` proves the lever was **in force**: `"from_scratch": true`, `trunk.init "from-scratch (random)"`, `trunk.ckpt null`, `trunk.step -1`. **As launched** (`config.json` → `args`): `steps 30000, batch 16, accum 4 (eff 64), lr_head 1e-4, lr_trunk 1e-4, warmup 2000, lam_mult_floor 0.25, phase_a_steps 2000, phase_b_steps 8000, gate_step 10000, labels v3, strategic full, dense_plan true, rollout_k 4, eval_episodes 40, seed 0`. Tests in `stack/tests/test_train_flagship_v4.py`. |
| **Location** | `tanitad-pod2:/workspace/experiments/flagship-v4-fromscratch/` — `ckpt.pt` **3 243 109 310 B** (step 29999) + `ckpt_step5000.pt`, `ckpt_step10000.pt`, `ckpt_step15000.pt`, `ckpt_step20000.pt` (written out 2026-08-03; the ckpt_step<full-integer>.pt form is MEASURED from the HF sibling `ckpt_step20000.pt` listed later in this row); `train_log.jsonl` **661 rows**, last row step **29999** (`plan_ade 0.3659`, `oracle_ade 0.1647`). ⭐ **Partly HF-backed** (MEASURED 2026-07-27, HF API): `Sayood/flagship-v4-fromscratch`, public + **`gated: manual`**, `lastModified` **2026-07-26T09:12:06Z**, files `ckpt.pt`, `ckpt_step20000.pt`, `config.json`, `metrics.json`, `train_log.jsonl`, `README.md`. 🔴 **`ckpt_step5000.pt`, `ckpt_step10000.pt` and `ckpt_step15000.pt` have NO REACHABLE COPY** (MEASURED 2026-08-03, read-only `ls`: `tanitad-pod2` is terminated and this run dir is **absent from pod4's rescue dump** `/workspace/rescue/experiments/`; the HF file list in this row is `ckpt.pt` + `ckpt_step20000.pt` only) — the 15 k milestone that carries this arm's decision-grade read is **gone**, unless a copy exists somewhere not yet probed (three locations were probed: pod2, pod4's rescue dump, HF). |

> 🟥 **GATE-COMPLETENESS / RECONSTRUCTION RISK (the whole v4 line).** The v4 held-out eval **driver was
> the standing blocker** — it did not exist until `eval_flagship_v4.py` was built + O-03-validated on
> 2026-07-23. **3 of 8 pre-registered KILL secondaries still have no emitter** anywhere in the codebase
> (`speed_benefit_recovered_frac`, `deploy_tick_p99_ms`, `nonav_route_beats_majority`), so **no v4 gate
> can render a *complete* formal verdict today** even with a held-out primary. The `run_gate.py`
> comparative matched-step-ratio path is also dead for v4: `g_op_fwd_ade_m` is computed in
> `v4_loss_step` but never reaches `train_log.jsonl` (whitelist gap in `_training_loop`). Checkpoints
> live on single pod disks (v4.1 3.24 GB, v4.2 both on pod2) and are **not HF-backed**.
> *(⭐ **One exception, MEASURED 2026-07-27:** `flagship-v4-fromscratch` **is** HF-backed —
> `Sayood/flagship-v4-fromscratch`, gated:manual — but only `ckpt.pt` + `ckpt_step20000.pt`; its
> 5 k / 10 k / **15 k** milestones are still single-disk. See §1.5.5 **Location**.)*

---

### 1.6 Flagship variants that exist but are not "versions"

| Run | What it is | Where |
|---|---|---|
| `p0-sB01-realmix` | the pre-reset 2-corpus (comma+PAI) realmix run; source of the **27 k D1/D2/D3 gate ladder** and the 3000-sample spectral fit; **stale since 2026-07-12 @ step 28,600** — the `gate-eval` skill still targets it ⚠️ | pod1 `/workspace/experiments/p0-sB01-realmix/` |
| `axis6-clean` / `axis6-relaxed` | `flagship4b_reduced` (d384, encoder 8, ~53 M model / 66 M trainable) A/B pre-check of `--sigreg-free-dims 64` vs `0` on comma2k19 | pod1 `/workspace/experiments/` |

---

### 1.7 flagship-v2**corpus** — `flagship-v2corpus-30k` — 🟢 **RUNNING** (launched 2026-07-25T02:41Z)

**The corpus experiment: §1.3's recipe, unchanged, on 3.8× the data.** Every lever
matches `flagship4b-v2-30k`; the **corpus is the only differing variable**.

> ⚠️ **STATUS ANNOTATION APPENDED 2026-08-16 by the stale-blocker sweep — nothing below is changed,
> corrected or restated; this note adds only a re-probe flag.** This row's status reads
> **🟢 RUNNING** with **ETA 2026-07-29T01:10Z**, which is **18 days in the past**, and no completion
> row, final-step row or final-eval row for `flagship-v2corpus-30k` was found anywhere in this
> registry. Meanwhile `Project Steering/BACKLOG.md` C1 records pod1 — this run's host — as having
> `/dev/nvidia*` **empty**. ⇒ **Treat the 🟢 RUNNING status as UNVERIFIED, not as a live run.**
> ⛔ **This annotation does NOT assert that the run finished, and does NOT assert that it died** —
> neither was probed (this sweep is CPU-only and does not touch pods). It asserts only that the
> status is stale on its face and must be re-probed before anything waits on it or quotes it.
> Downstream: `BACKLOG.md` C2 is gated on this run and has been marked UNVERIFIABLE for the same
> reason. ⚠️ Do not confuse this arm with `flagship-v1arch-v2bal-30k`, which shares the **v2bal
> corpus** but is the **v1-architecture / every-v2-lever-false** arm.

| Field | Value |
|---|---|
| **Status** | 🟢 RUNNING on `tanitad-pod` (pod1, RTX A6000). Trainer PID **699286** (parent `bash -c` wrapper 699284). ETA **≈2026-07-29T01:10Z** (30,000 × MEASURED 11.3 s/step ≈ 94 h). |
| **Location** | `tanitad-pod:/workspace/experiments/flagship-v2corpus-30k/` · log `tanitad-pod:/tmp/flagship-v2corpus-30k.log` |
| **Corpus** ⚠️ | **`physicalai-v2bal-4b7eeeac222d`** — 9,000 clips / **49.742 trainable h** / 22.3 GiB JPEG cache. **NOT the parity corpus.** Key **`4b7eeeac222d`** recomputed from the built clip-ids at consolidation. **Record it here because the run's own `config.json` does NOT** — under `--v2-cache` the trainer writes `"cache_dirs": null, "data": "realmix"`. |
| **Windows** | **1,538,710** (parity arms: ~198 k) |
| **Params** | operative 96,609,284 · tactical_pred 26,535,424 · tactical_policy 30,098,063 · strategic 8,385,411 · encoder 87,121,280 · h15 22,055,683 · grounding 13,432,338 → **total_model 272,906,913 / trainable 286,339,251** ✅ — **identical to §1.3**, verified from the run's own `[init]` line |
| **Levers** | `v2_labels true` · `speed_input true` · `rollout_k 12` · `anchor_tactical true` · `gated_intent true` — each read from **both** config.jsons and confirmed equal to §1.3 |
| **Exact command** | run manifest `/workspace/ops/runs.d/flagship-v2corpus-30k.env` (`TRAIN_CMD` copied verbatim from `/proc/699286/cmdline`):<br>`PYTHONPATH=/workspace/TanitAD/stack PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u scripts/train_flagship4b.py --v2-cache /workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d --config flagship4b --v2 --sigreg-free-dims 64 --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint --lr 3e-4 --warmup 2000 --workers 8 --v2-lru 64 --guard-limit-gb 45 --ckpt-every 1000 --log-every 50 --out /workspace/experiments/flagship-v2corpus-30k` ✅ |
| **Code state** | repo `bdb6ba1` synced to pod1 and verified by full trainer import. Requires `tanitad/data/v2_dataset.py` (`--v2-cache` lazy loader) and the `finetune_traj.py` OOM-guard fix (the guard matched **0** files on any v2 cache before it). Both staged, uncommitted at launch. ⚠️ |
| **Supervision** | `supervise_run.sh` attached **death-only** (`TRAIN_MATCH` prevents double-launch); heartbeat `/workspace/ops/heartbeats/flagship-v2corpus-30k.json` |
| **HF** | none |

**Pre-registered comparison (committed BEFORE the run):** vs **§1.3
`flagship4b-v2-30k`**, **corpus only**. Primary = **ADE@2s** at **matched step
5,000** (the control's cleanly archived `ckpt_step5000.pt`), via **episode-cluster
bootstrap** over the 40 val episodes, paired where windows match — never
`overlapping_holdout_se`. Secondary = turn-stratified ADE. 30k-vs-30k does not
exist: the control's log ends at step **7,700** (registry text elsewhere says 7,800)
with resume-duplicated rows in its tail.

> ⚠️ **Turn share depends on the labeler, the scenes do not.** This corpus reads
> **28.04 %** turns under the v1 labeler (its selection target) and **18.83 %**
> under the v2 labeler now in force — the same 9,000 clips either way. Quote
> **18.83 %** for this run's labels, **28.04 %** for the corpus design target.

> ⚠️ A first launch at 02:21Z used `--v2 --no-labels-v2` and was killed at ~20 min
> (no checkpoint had been written); it is preserved at
> `…/flagship-v2corpus-30k_ABORTED-labelsFALSE-20260725T0221Z/`. `--no-labels-v2`
> would have made **labels a second differing variable** vs §1.3.

Full provenance: `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-07-25-v2-launch-readiness/V2_LAUNCH_READINESS.md`.

---

### 1.8 flagship-v5f — `flagship-v5f-w120-30k` — ✅ **COMPLETE at 30,000** (2026-08-09T19:23Z)

**FINAL EVAL [TIER T0 — teacher-forced WM; see EVAL_DOCTRINE.md] — MEASURED 2026-08-09,
`eval_flagship_v4.py` on the full 600-episode w120 val corpus (881 windows, frame parity
`176x624f305.5775cyl` MATCHES the ckpt config; imagination probes fed via the run's frozen
`probe_vocab.pt`):**

| metric (dense 1..20 @ 10 Hz) | value |
|---|---|
| **ade@2s (selected)** | **0.4011 m** |
| **oracle_ade@2s (best-in-fan)** | **0.1975 m** |
| **sel_gap** | **0.2036 m — the selector still leaves ~half** |
| miss@2m | 0.1487 |
| 4-wp convention: ade / oracle | 0.5191 / 0.2453 |
| seam_norm_ratio_max | 0.099 (clamp 1.0 — grafts healthy) |
| wm_canary_ade@2s | 1.245 *(inert-controller regime, benign by construction)* |

⚠️ **w120 geometry ⇒ NOT comparable to any 256×256-pinhole number** (cross-frame). ⛔ All
values T0; T1 closed-loop is queued (E1.4). **ckpt frozen `ckpt_30k_final.pt`, HF:
`Sayood/tanitad-flagship-v5f-w120` (public, gated=auto: ckpt + config + train log + eval
JSON).** Post-30k queue: T1 eval · fan-feasibility dump · X0 diffusion-MPC re-rank ·
selector ranking retrofit · w120 E-H1. Eval-harness fixes this required (both committed):
frame args via the existing shared helper, and `_imagination_inputs` + `probe_vocab.pt`
threading for cond-imagination heads.

#### (history) run record — was 🟢 RUNNING (step 3,650 @ 2026-08-03T18:57Z)

**The 120° wide-FOV cylindrical arm with conditional imagination.** Added to the registry
2026-08-03 — it had been the programme's headline live run for days **with no registry row**, so
every quoted v5f fact was un-anchored. All fields below are MEASURED off the pod in one probe.

| Field | Value |
|---|---|
| **Status** | 🟢 RUNNING on `tanitad-new` (pod5, `69.30.85.106:22039`), trainer PID **19412**. Migrated from the faulty pod2. Resumed `[resume] step 1001`. |
| **Location** | `tanitad-new:/workspace/experiments/flagship-v5f-w120-30k/` · log `…/train_log.jsonl` (+ `/workspace/v5f_run.log`) |
| **Corpus** ✅ | **PARITY VERIFIED by the trainer itself**: `physicalai-train-e438721ae894-w120-256x640cyl`, **2400 clips**, clip sha256 `e61a04553df5…` matching the committed manifest; sibling of `physicalai-train-e438721ae894`, skip-hash **`f09e44db`**. Val: `physicalai-val-0c5f7dac3b11-w120-256x640cyl`, 600 clips, sha256 `0b176d2e5cb4…`. |
| **Geometry** | 256×640, `f_ref` 305.5775, **cylindrical**, HFOV 120°, subframe 176×624, `frame_tag` `256x640f305.5775cyl` |
| **Levers** | `--from-scratch` (trunk *init* only) · `--cond-imagination` · `--batch 4 --accum 16` (eff_batch **64**) · `--anchors-dense` · `--save-every 250` · ⚠️ `--no-heldout-gate` with `--heldout-off-reason` *"PI directive: no heldout gate. Migrated from the faulty pod2…"* |
| **Trainability** | `[v4][not-frozen]` self-check PASSES: **631/631** trunk tensors require grad and sit in the AdamW `trunk` group at lr 1e-4; encoder **149/149**, predictor **159/159** trainable; `gnorm_trunk > 0` confirmed in the log. |
| **Throughput** | **MEASURED 19.58 s/step** over the last 250 steps ⇒ **143.3 h ≈ 6.0 days** remaining to 30 k. |
| **Container RAM** | ⚠️ `memory.max` = **49,999,998,976 B (50 GB)** — the *same* cap as the pod2 that was OOM-killed 6×. The migration runbook asked for more and did not get it. |
| **HF** | `Sayood/tanitad-flagship-v5f-w120` (ckpt + config + probe_vocab) |

#### ⛔ The 13:00 "v5f IS GOING THE WRONG WAY" alarm is WITHDRAWN

MEASURED, **500-step block medians** (n ≥ 4 per block — never one logged line):

| step block | `g_op_fwd_ade_m` | `plan_ade` | `oracle_ade` | `sel_gap` | `rank_acc` | `frac_sel_2x_worse` |
|---|---|---|---|---|---|---|
| 1000–1500 | 0.3522 | 1.4935 | 0.8798 | 0.4510 | 0.2500 | 0.2500 |
| 1500–2000 | 0.2933 | 1.3949 | 0.8382 | 0.4878 | 0.0000 | 0.3750 |
| **2000–2500** | **0.4191** ← *the alarm* | 1.6597 | 0.9450 | 0.5681 | 0.1250 | 0.2500 |
| 2500–3000 | 0.1784 | 1.0714 | 0.5902 | 0.3980 | 0.1250 | 0.2500 |
| 3000–3500 | 0.2389 | 0.9762 | 0.5663 | 0.3432 | 0.0000 | 0.2500 |
| 3500–4000 *(n=4)* | 0.1893 | 1.0251 | **0.5254** | 0.4715 | 0.3750 | 0.5000 |

The 13:00 report read **0.64–1.02 around step 2300** and headlined *"v5f IS GOING THE WRONG WAY"*.
Block medians show that was the **2000–2500 bump**, and the run has since printed 0.1784 / 0.2389 /
0.1893 — **better than anything before the bump**. ⇒ **No restart. The HYPOTHESIS stated alongside
the alarm (LR warm-up under a changed `--batch 4 --accum 16` regime) is the reading the data now
supports**, and the recommendation *"hold to 5 k, do not restart on 3 noisy points"* was correct.
⭐ **Root-cause class: a 3-point read inside an LR warm-up is not a trend — block-median it or do
not raise it.** Same family as the three earlier `g_op_fwd_ade_m` misreads, in both directions.

#### ⭐ What the same table shows instead — the fan is good and the SELECTOR is the defect

- `oracle_ade` **improves monotonically after the bump**: 0.9450 → 0.5902 → 0.5663 → **0.5254**.
  The best candidate in the fan is getting substantially better.
- `sel_gap` = `plan_ade − oracle_ade` **does not close**: 0.4510 / 0.4878 / 0.5681 / 0.3980 /
  0.3432 / 0.4715 — no trend across 2,650 steps.
- `rank_acc` sits at **0.000–0.375** and `frac_sel_2x_worse_than_oracle` at **0.25–0.50**: the
  selector picks a candidate **≥2× worse than the oracle on a quarter to a half of logged steps**.
- ⇒ At step 3,650 `plan_ade` **1.0251** vs `oracle_ade` **0.5254** — **the arm would be ~2× better
  if it merely chose correctly among candidates it already generates.**

⇒ **This is the TACTICAL family failing live in the flagship's own training log**, independent of
REF-C's manoeuvre head (§ the D-TAC1 stream) and of the sitclf work — three instruments now point at
selection rather than generation. ⛔ It is **not** visible in `g_op_fwd_ade_m`, which is why an
ADE-only read of this run says "healthy".

⚠️ **These are TRAINER-log numbers, not eval output.** Per the operating standard they are a curve
watch and are **not quotable as a result** — trainer val has run ~10 % optimistic vs `eval_*.py`
before. The 5 k gate must be adjudicated on `stack/scripts/run_gate.py`, on the four families, with
the paired episode-cluster bootstrap.

---

### 1.9 flagship-**v1arch** — `flagship-v1arch-v2bal-30k` — ✅ **COMPLETE at step 29999** (2026-08-05)

**The v1 ARCHITECTURE on more, better-distributed data.** ⛔ **Every `v2_*` lever in its own
`config.json` is `false`** — MEASURED off the run's `cfg` block on pod4, 2026-08-05:
`v2_anchor_tactical`, `v2_ego_to_planners`, `v2_gated_intent`, `v2_goal_decode`, `v2_labels`,
`v2_route_from_vision`, `v2_encoder_ego_decorr` all `false`; `v2_ego_dropout`, `v2_fa_dropout`,
`v2_nav_dropout`, `v2_traj_jerk` all `0.0`; `v21_route_labels` `false`. So **architecture is held
constant and only the data varies** — which is what makes the PI's question (*the effect of more
and better-distributed data*) attributable. ⚠️ **It is NOT a v2-architecture arm and must never be
quoted as one.**

| Field | Value |
|---|---|
| **Status** | ✅ COMPLETE, clean finish at step **29999**. Supervisor **adopted** it (`trainer ALREADY RUNNING outside this supervisor (pid 9076)`), did not launch it, and exited without relaunching. |
| **Location** | `tanitad-pod4:/workspace/experiments/flagship-v1arch-v2bal-30k/` (`ckpt.pt`, `ckpt_step5000.pt`, `ckpt_step15000.pt`, `ckpt_step20000.pt`, `config.json`, `train_log.jsonl`) |
| **Checkpoint shape** | keys `['grounding','model','opt','step']` — **no `head` key**. ⛔ `eval_flagship_v4.py` gates its full metric path on `is_v4 = … and ("head" in ck)`, so on this checkpoint it can ONLY run `MODE_A_canary_only_validation`: it exits 0, prints an ADE, and **emits no per-window `pred`/`gt` at all**. Use `taniteval/tools/eval_four_families.py`. |
| **Architecture** | `predictor` d_model 768 / depth 10 / heads 12 / window 8 / **action_dim 3** (`speed_input` true) · `tactical_policy` **n_maneuvers 5**, wp horizons [5,10,15,20], cadence 5 · `strategic_policy` n_commands 4, **n_route 3**, d_ctx 256, cadence 20 · `h15` enabled (mask_prob 0.5) · state_dim 2048 |
| **Corpus** ⛔ | `"v2_parity": {"parity": false, "checked": false, "clips_present": 9000}`, `"require_parity": false`. **This arm is OFF the parity corpus by design** (that was the experiment). ⛔ **21 of the 40 canonical val episodes are INSIDE its 9000-clip training pool** — see `LEAK_v1arch_val_2026-08-05.md`. Canonical-val numbers for this arm are train-contaminated and inadmissible. |
| **HF** ⭐ | `Sayood/tanitad-flagship-v1arch-v2bal` — **PUBLIC + GATED (`auto`)**, published 2026-08-05. `ckpt.pt` (3,302,203,998 B, the exact evaluated artifact), `config.json`, `summary.json`, `train_log.jsonl`, model card. Verified ANONYMOUSLY: metadata 200, `private False`, `gated auto`, and an unauthenticated `ckpt.pt` fetch returns **401 — the gate is enforced**, while the card (7,439 B) is publicly readable so the caveats reach a reader before the gate does. ⚠️ The milestone checkpoints are NOT published: `ckpt.pt` is the artifact every number below was measured on, and shipping the others invites *"which one produced this?"*. |
| **The admissible eval corpus** | `physicalai-oodval-6f4b94e4c7ce-q90` — PhysicalAI-AV's **own official eval split** (HF dataset `nvidia/PhysicalAI-Autonomous-Vehicles`, `ood_reasoning.parquet` under its `reasoning` prefix; split sizes `{train 1450, val 290}`), **290 clips, ZERO overlap** with the training pool, JPEG-q90 round-tripped to the training format. 6,382 windows / 290 episode clusters. |

#### Result — the first COMPLETE four-family block in the programme (`_complete: true`)

⛔ **ADE is one row of four.** Estimator: **episode-cluster bootstrap** over the 290 clips,
n_boot 2000. ⛔ **Not comparable to any canonical-val number** until other arms are scored here.

| family | headline | reading |
|---|---|---|
| — (ADE) | `ade_mean_4wp` **0.5752** [0.5370, 0.6142] · `fde_2s` **1.4018** [1.3040, 1.5010] | |
| **LONGITUDINAL** | `speed_bias` **+0.484 m/s** · `along_final_bias` **+0.943 m** · `ego_progress` **1.0795×** · time-gap at 15+ m/s **1.43 s** | ⛔ systematic **over-speed**, and MEASURED it is **71.95 %** of windows ahead at 2 s and **75.51 %** faster than the human — a **prior**, not a tail |
| **LATERAL** | `cross_mae` **0.0552 m** [0.0500, 0.0611] · `heading_mae` 0.806° · `curvature_bias` −0.000126 | tight; not where effort belongs |
| **TACTICAL** | κ **0.6033** (SUBSTANTIAL), agreement 0.8881 [0.8740, 0.9021] · **`seams_beneficial_of_3` = 0** | the manoeuvre label is honest; the **seam is FALSIFIED at this checkpoint** |
| **STRATEGIC** | `route_acc_follow` **0.8031** == `majority_straight_rate` **0.8031** · `follow_pred_distribution` **{left 0, straight 1737, right 0}** · `route_acc_nav` 1.0000 | ⛔ **no vision-only route skill at all** — a constant predictor. `route_acc_nav` is an **echo** of its own input. Confirmed **off-leak**. |

Distance-keeping (n = 2,846 lead windows): headway **25.53 m**, time-gap **5.76 s**, min-TTC
**14.73 s** (632 censored at the 30 s cap). Window states LEAD 3,002 / NO_LEAD 2,752 / NO_LABEL 628.

**JPEG-format control:** raw uint8 vs the q90 round-trip on the same 6,382 windows — max |pixel
delta| **185**, every metric moves **< 0.03** (largest: heading 0.0236°); tactical and strategic
identical. **q90 is the headline** because it is format-faithful.

⚠️ **Two harnesses disagree 0.8 %** on this corpus: `eval_flagship_v4.py`'s MODE-A canary gives
**0.5705**, `eval_four_families.py` gives **0.5752**. Recorded, not smoothed over; unresolved.

**Artifacts:** `TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/2026-08-05-v1arch-oodval-four-families/`
(RESULT.md + raw JSON) · protocol `EVAL_PROTOCOL_OODVAL_2026-08-05.md` · videos
`TanitAD Research Lab/Evaluation/Videos/v1arch-oodval-openloop-2026-08-05/` · pod4
`/workspace/evalout/` (windows dumps, lead block).

---

### 1.10 flagship-**v1.6** — `flagship-v16-unicycle` — ✅ **COMPLETE** (2026-08-06) — the unicycle trajectory readout on the FROZEN v1arch trunk

⛔ **NAME DISAMBIGUATION, load-bearing:** "v1.6" ALSO names §1.4b's `flagship-v16-ab-ft`
(2026-07, the arm behind the retracted "best-in-program" claim). These are **different
models**. Quote the registry key, never the bare "v1.6".

| | |
|---|---|
| **architecture** | `flagship-v1arch-v2bal-30k` trunk (encoder 87.02 M + predictor 91.36 M + policies), **entirely frozen** (md5 `c1157528…` proved identical before/after training) + `UnicycleStepReadout` **2.11 M** trainable — latent transition `(z_prev, z_hat)` → per-step `(accel, yaw_rate)`, integrated non-holonomically (`dy ≡ 0`, `yaw_rate` bounded by `\|v\|·κ_max`, v carried from the TRUE v0). ⛔ `speed_input=False, predict_delta=False` — the head reads ONLY the WM latents; the v0/feedback shortcut surface was REMOVED after run 4 failed the reliance gate at 0.0891 (`…/2026-08-06-v1-defect-triage/results/UNICYCLE_RUN4_RESULT.md`). |
| **training** | 3,000 steps, batch 32, AdamW lr 3e-4 cosine, 58 min on one A40. Loss = pos-L1 + 0.3·heading + 0.5·net-yaw + 0.05·accel-barrier + 0.05·jerk-barrier (barriers above the TRAIN-corpus human p99; dense 0.1 s grid). Train corpus: 600 eps of `epcache-physicalai-v2bal-4b7eeeac222d` (local copy). Head warm-started from `grounding.step.op`, zero-init output (= constant-velocity start). |
| **parity** | trunk untouched ⇒ inherits v1arch's parity (skip-hash `f09e44db` lineage). Eval corpus: `physicalai-oodval-6f4b94e4c7ce-q90`, 40 episodes, stride-1 rollout grid, **6,834 windows** — the SAME grid as every banked v1arch temporal/kinematic number. |
| **ckpt** | `pod4:/workspace/experiments/unicycle-readout-v2-latentsonly/unicycle_readout.pt` · **banked in-repo 2026-08-06**: `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-08-06-v1-defect-triage/results/unicycle_readout_v16.pt` (8.4 MB, md5 `81f7f3a19ad0da97fb55ed9270f2f884` verified matching the pod copy — single-disk risk closed) |

**RESULTS [TIER T0 — teacher-forced; see EVAL_DOCTRINE.md] — MEASURED 2026-08-06, paired episode-cluster bootstrap over 40 episodes, 2,000
draws** (`…/2026-08-06-v1-defect-triage/results/v16_full_eval.json.xz`). Δ = v1.6 − v1arch,
same windows, same frozen-trunk latent rolls (decoder-only contrast by construction):

| metric | v1arch | **v1.6** | Δ [CI95] | separated |
|---|---|---|---|---|
| ADE 2 s (m) | 0.3584 | **0.3398** | −0.0186 [−0.0706, +0.0293] | ✗ (parity) |
| speed bias (m/s) | +0.3793 | **−0.0265** | −0.4058 [−0.5162, −0.3022] | ✅ |
| along-final bias (m) | +0.7524 | **−0.0511** | −0.8036 [−1.0197, −0.5851] | ✅ |
| accel RMS (m/s²) | 2.9465 | **0.7172** | −2.2293 [−3.0886, −1.4641] | ✅ (human ≈ 0.91) |
| accel MAE (m/s²) | 1.8240 | **0.5499** | −1.2741 [−1.6534, −0.9324] | ✅ |
| jerk RMS (m/s³) | 36.1682 | **1.1334** | −35.0348 [−46.6907, −24.8134] | ✅ (human ≈ 1.71) |
| **net-yaw err (rad)** | 0.0307 | **0.0108** | −0.0199 [−0.0248, −0.0153] | ✅ (−65 %) |
| heading MAE (rad/step) | 0.0027 | **0.0015** | −0.0012 [−0.0015, −0.0009] | ✅ |
| cross-track MAE (m) | 0.0502 | **0.0363** | −0.0139 [−0.0193, −0.0091] | ✅ |
| replan shift (m, 0.1 s) | 0.0947 | **0.0604** | — | (point est.) |
| **replan accel jump (m/s²)** | 1.1310 | **0.1016** | — | (point est., **11×** lower) |

**WM-reliance** (canary-grade, 128-window fixed batch): final **0.6233 — gate PASS ≥ 0.5**;
per-arm decomposition shows real latents ~halve the heading error vs batch-mean latents at
every canary. Baseline control: the displacement readout scores 8.72 (cannot function without
latents), and its **frozen-latents arm collapses to the CV floor** — what both decoders consume
is the **predictor's rolled prediction**.

**DISTANCE-KEEPING — MEASURED 2026-08-06** (`…/2026-08-06-v1-defect-triage/results/v16_distance_keeping.json`,
built from a fresh 40-episode lead block — `lead_block_40.report.json`: 880 stride-8 windows, LEAD 419 /
NO_LEAD 329 / NO_LABEL 132; 5 clips lack `obstacle.offline`, 1 stationary clip fails registration — joined
row-for-row to the stride-8 subset of the SAME scored dump as the table above, no re-inference; paired
episode-cluster bootstrap, 30 lead-bearing episodes, 2,000 draws; sign below is **v1arch − v1.6** / **GT − v1.6**):

| metric | v1arch | **v1.6** | GT | v1arch−v1.6 [CI95] | GT−v1.6 [CI95] |
|---|---|---|---|---|---|
| min headway (m) | 25.23 | **25.52** | 25.39 | −0.238 [−0.385, −0.091] ✅ | −0.026 [−0.225, +0.153] ✗ |
| min time-gap (s) | 7.71 | **7.98** | 7.98 | −0.138 [−0.252, −0.049] ✅ | −0.010 [−0.064, +0.042] ✗ |
| min TTC (s) | 14.97 | **17.82** | 17.10 | **−2.818 [−3.892, −1.784] ✅** | −0.597 [−1.393, +0.099] ✗ |

Reading: v1arch is CI-separated **more aggressive** on all three (2.8 s lower min-TTC — the longitudinal
speed/accel defect showing up against real traffic), while **v1.6 is statistically indistinguishable from
GT on every distance-keeping metric** at this n. ⚠️ 40-episode subset grid, not the 290-corpus grid the
v1arch OOD-val LEAD numbers use — compare within this table only.

**⛔ Standing caveats:** (1) all trajectory numbers are **action-conditioned WM rollouts under
TRUE future actions — NOT closed-loop planning**; applies equally to both arms and to every
v1arch number banked before. (2) STRATEGIC UNAVAILABLE (no map — unchanged). ~~distance_keeping
UNAVAILABLE~~ — closed above; the 2026-08-06 claim "lead block not on pod4" was itself a **stale
absence-claim** (the 290-episode block and v1arch's complete LEAD JSON already existed at
`pod4:/workspace/evalout/` — `v1arch_oodval_q90_4fam_LEAD.json`, `families_unavailable=[]`).
(3) TACTICAL **declared-head** metrics are identical to v1arch's (policies untouched — the
declared dwell 0.55 s toggling defect is NOT fixed by v1.6 and still needs its own lever).
The **EXECUTED-manoeuvre** side, MEASURED 2026-08-06
(`…/results/v16_tactical_executed.json`, `classify_maneuver` over each window's 2 s path,
6,834 stride-1 windows; ⚠️ yaw at horizon from last-segment heading — the dump stores xy only):
agreement with GT-executed **v1arch 0.5016 → v1.6 0.7694**; executed toggle rate v1arch 0.0620
→ v1.6 **0.0309** vs GT 0.0318 — paired Δ(v1.6−v1arch) −0.0311 [−0.0462, −0.0177] separated,
Δ(v1.6−GT) −0.0009 [−0.0066, +0.0055] NOT separated; executed dwell 2.61 s → 4.38 s (GT 3.52 s
— v1.6 slightly over-steady: it under-executes `accelerate`, 577 vs GT 975, the conservative
tail of its speed profile). v1arch's executed distribution over-calls `accelerate` 2,668 vs
GT 975 — the longitudinal defect visible in decision space. (4) reliance CI not computed
(canary batch); families CI'd as above.

### 1.11 flagship-v1.7 — `flagship-v17-speedloss` — pre-registered run 6, outcome **B**

**MEASURED 2026-08-06** (`…/2026-08-06-v1-defect-triage/PREREG_V161_SPEEDLOSS.md` — gates and
both outcomes were committed BEFORE launch). v1.6's exact recipe + **one change**:
`--w-speed 0.5` speed-profile L1. Trunk frozen; head 2.11 M latents-only; 3,000 steps, 55 min.
**ckpt:** `pod4:/workspace/experiments/unicycle-readout-v3-speedloss/unicycle_readout.pt`, banked
in-repo as `results/unicycle_readout_run6.pt` (md5 `e389f638cc2e7ac4d67bf57479936b7f`).

**Gates verdict [TIER T0 — teacher-forced; see EVAL_DOCTRINE.md] — the PRIMARY gates FAILED;
the pre-registered hypothesis is REFUTED:**

| gate | target | measured (eval-grade, same 6,834-window grid) | verdict |
|---|---|---|---|
| P1 decel response ratio | ≥ 0.40 | **0.1547** (v1.6: 0.1623 — unchanged) | ❌ |
| P2 accel lag | ≤ +0.15 s | **+0.173 s** (from +0.28) | ❌ |
| N1 ADE vs v1.6 | not CI-worse | **−0.0549 [−0.0713, −0.0412] CI-BETTER** (0.2849 vs 0.3398) | ✅ |
| N2 jerk RMS | ≤ 3.42 | 1.567 (human 1.71) | ✅ |
| N3 net-yaw | not CI-worse | Δ +0.0001, not separated | ✅ |
| N4 reliance | ≥ 0.5 | **1.1849** (>1: cannot function without WM) | ✅ |
| N5 replan accel jump | ≤ 0.30 | 0.125 *(GT-frame approx transform)* | ✅ |

⇒ **Outcome B binds:** the position-loss hypothesis for the decel ramp is refuted for the
speed-L1 lever — near-term response barely moved while everything global improved. The next
pre-registered lever is the **event-weighted near-term accel-matching term**, NOT weight tuning.
v1.7 **is** CI-separated better than v1.6 on ADE (−16 %) and speed MAE (−0.072 [−0.094, −0.051]),
with every non-regression gate green — registered as the best open-loop head in the lineage,
**not** as the lag fix. Speed_l1 evidence: `results/run6_train_log.jsonl.xz`; analysis:
`results/closed_loop_analysis.json`.

### 1.12 CLOSED-LOOP (decoder-conditioned predictor) — MEASURED 2026-08-06 [TIER T1 — the PRIMARY eval per EVAL_DOCTRINE.md]

**The predictor rolled on the DECODER'S OWN actions** (steer = atan(2.9·κ), accel direct — the
`signals_at` contract), no recorded future anywhere; perception context unchanged (imagination
closed loop, not re-perception). `tools/closed_loop_dump.py` → `results/closed_loop_analysis.json`;
grid identical to §1.10 (o16 arm reproduces the banked v1.6 dump **byte-exactly, max |Δ| = 0.0**).

| metric | v1.6 open | v1.6 closed | v1.7 open | v1.7 closed | CV floor |
|---|---|---|---|---|---|
| ADE (m) | 0.3398 | **0.4714** (+0.132 [0.112, 0.152]) | 0.2849 | **0.4616** (+0.177 [0.148, 0.208]) | 0.5352 |
| net-yaw err (rad) | 0.0108 | **0.0725** (×6.7) | 0.0109 | 0.0761 | — |
| speed MAE (m/s) | 0.441 | 0.606 | 0.369 | 0.598 | — |
| **S-curve reproduction** | **0.9785** | **0.0538** | 0.9785 | 0.0430 | 0 |

**⛔ The finding that re-frames the open-loop numbers:** the **hold-action arm reproduces 0.0 %
of S-reversals** (and closed-loop ~5 %) vs 97.9 % open-loop — the counter-steer in the open-loop
eval came from the TRUE-action conditioning, **not from vision**. Open-loop LATERAL skill is
largely an action echo; closed-loop the stack drives near-straight with speed control, retaining
**~33 %** of its open-loop ADE advantage over CV (v1.6: (0.535−0.471)/(0.535−0.340)). The §1.10
vision-attribution (35 % of the CV gap, swap-latents) still holds at the ADE level — vision
carries speed/scene content — but the lateral reversal channel specifically is action-driven.
This is the measured content of the standing "NOT closed-loop planning" caveat, and the
programme's next lever set (policy-actions closed loop, lateral-capable heads, MPC-style search
over the predictor) starts from these numbers.

### 1.13 v5.8f wedge ladder — W1/W2/W2b (X0-lite) — MEASURED 2026-08-09 [TIER T0 diagnostic on the v5f 30k fan]

One GPU pass over the exact §1.8 grid (881 windows, oracle goal, `sel_idx` = the head's own
pick), full 256-candidate `anchor_traj` fan in **float32** (an f16 first pass gave the same
census to 3 decimals — the jitter is model output, not storage precision). Artifacts:
`…/incoming/2026-08-07-hierarchical-wm-redesign/{x0_lite_f32.json, tools_x0_lite.py}`.
Point estimates on the fixed grid; no interval computed (wedge diagnostic, not an eval row).

| quantity | value |
|---|---|
| **W1 (PRE-REGISTERED gate: kinematic re-rank of top-8 closes ≥ 30 % of sel_gap)** | **REFUTED: −16.7 %** (re-rank WORSENS: sel ADE 0.4011 → 0.4351; gap 0.2036 → 0.2376) |
| W2 census: fan steps violating \|a\|≤4 ∨ \|yr\|≤0.33v+0.05 | **97.6 %** of all 256×20×881 steps |
| W2 census: candidates infeasible (>5 % bad steps) — selected / oracle / all | **100 % / 100 % / 100 %** |
| W2: mean \|accel\| over ALL candidates | 252.1 m/s² |
| W2: selected-candidate accel MAE | **8.10 m/s²** — cross-validates §1.8 four-families' 8.11 on an independent code path |
| W2b (exploratory, NOT pre-registered): 3-tap [.25 .5 .25] smoother | sel ADE 0.4011→**0.3975**, oracle 0.1975→**0.1879**, sel accel MAE 8.10→**3.09**, sel mean \|yaw-rate\| 81.9→49.0 °/s, still 32 % steps infeasible; smoothed-cost re-rank still fails (−17.1 %) |

**Reading.** The fan's step-level jitter is truncated-denoise residue AROUND the true path —
smoothing improves BOTH ADE numbers, which noise-around-signal predicts and signal-content does
not. Consequences: (1) any waypoint-space kinematic cost ranks jitter, not manoeuvre quality —
W1's failure is structural, so **W7 (WM-roll re-rank) must run on a kinematically clean fan**;
(2) a free smoother is a partial "W4-lite" (accel 3.09, still 2× the W4 gate of 1.5) — worth
stacking, not a substitute; (3) **W4 (unicycle-anchor emission head) stays the load-bearing
v5.8f wedge**, launched 2026-08-09 ~22:45Z on pod5 (`train_v58f_unicycle_head.py`, gates
pre-registered in `w4_gate.json`: selected accel MAE < 1.5 ∧ oracle ADE ≤ 1.10×0.1975).

**W4 RESULT — MEASURED 2026-08-10 [TIER T0 diagnostic, same 881-window grid]: both
pre-registered gates PASS.** `UnicycleEmission` (109,096 trainable params, 2-layer MLP off the
offset-head query, a=4·tanh / κ=0.2·tanh, unicycle-integrated; trunk+head frozen, md5-identical
before/after; 4,000 steps, 5.7 h — two prior attempts died in the 2026-08-10 ~05:00Z MooseFS
I/O incident, third ran clean). Artifact: `…/2026-08-07-hierarchical-wm-redesign/w4_gate.json`;
weights: HF `Sayood/tanitad-flagship-v5f-w120` `/w4/`.

| quantity (new unicycle fan vs original, SAME grid) | new | original |
|---|---|---|
| oracle ADE | **0.1077 m** | 0.1991 m |
| selected-candidate accel MAE | **0.774 m/s²** (winner 0.261; census-violation frac **0.0**) | 9.297 m/s² |
| selected ADE (FROZEN selector's pick) | 0.7933 m | 0.4056 m |

**Reading.** The re-parameterised fan is feasible BY CONSTRUCTION *and* its oracle nearly
halves — the waypoint jitter was hiding coverage, not providing it. The one regression is the
**frozen selector**: its scores were learned against the old fan's geometry, so its argmax on
the new fan is near-uninformed (0.79 ≈ CV-floor territory). That is a selector-calibration
defect, NOT a fan defect — and it is exactly the seam W7 (MPC re-rank) and an L4-style selector
re-distill were built for. Next rung: re-rank/retrain the SELECTOR on the frozen new fan
(cheap, selector-only) — pre-register before running. Estimator note: corpus-grid point
estimates; the episode-cluster bootstrap runs before any leaderboard/publication claim.

**W4b RESULT (feat variant) — MEASURED 2026-08-10 ~18:20Z, held-out 881 grid, per
PREREG_W4B_SELECTOR.md: G1 FAILED, G2 engaged, pruner NOT viable.** Selected ADE **0.5600**
vs gate ≤ 0.45 (frozen-selector reference 0.7933 — the rescorer recovers a large fraction but
not enough); top-8 oracle **0.3185** vs pruner threshold ≤ 0.15 (full oracle 0.1077 — the
rescorer's ranking does not concentrate the good candidates). ⚠️ The TRAIN monitor sat at
0.21–0.33 while held-out is 0.56 — the rescorer memorises train-window selection rather than
generalising it; the offset-query feature alone does not carry a generalising selection
signal for the unicycle fan. Per the prereg's bound G2 consequence: **W7 (WM-roll re-rank on
the clean fan) is now the primary selection mechanism** for v5.8f; the kin variant (adds
(a,κ) inputs) is running and reported when it lands. Artifact:
`…/2026-08-07-hierarchical-wm-redesign/w4b_gate_feat.json`.

**W4b kin variant — MEASURED 2026-08-10 ~21:30Z: G1 FAILED IDENTICALLY** (held-out selected
ADE **0.5637** vs feat's 0.5600; top-8 oracle 0.3155, pruner not viable). Adding the
candidates' own (a,κ) kinematics to the scorer moved the held-out number by <0.004 —
**the failure is not the input surface; pooled-feature per-candidate scoring on this trunk
memorises train-window selection.** Per PREREG_W4C_SPATIAL_SCORING.md this ACTIVATES W4c
(spatial cross-attention scoring, the REF-C conf mechanism, no grafts/gating) as the last
fast-selector attempt; its G-null retires fast scoring to a W7-distillation target.
Artifact: `w4b_gate_kin.json`.

**W4c RESULT — MEASURED 2026-08-11 ~00:50Z: G-NULL ENGAGED — the spatial port ALSO fails**
(held-out selected ADE **0.6609** vs gate ≤ 0.45; entropy 5.37 — still smeared; final
train-vs-heldout gap 0.139 — the memorisation signature persists even with spatial
attention + dropout). **Three independent scoring surfaces (pooled query / +kinematics /
spatial cross-attention) have now failed the same held-out gate.** Per the bound G-null:
⛔ **fast per-candidate scoring on this trunk is RETIRED** — no fourth attempt without new
evidence; selection moves ENTIRELY to **W7 (WM-roll re-rank)**, and a fast selector may
return only as a DISTILLATION of W7 (L4). Scientifically this is the programme's own thesis
arriving by elimination: the selection information is not in light readouts of the trunk's
features — it is in the CONSEQUENCES, i.e. rolling the world model. Artifact:
`w4c_gate.json`.

### 1.13b E4.4 — tactical stage-0, first trained instance — MEASURED 2026-08-10 ~21:50Z [TIER T0 diagnostic, 881-grid val, n@4s = 761]

**Pre-registered gate (goal FDE@4s < CV-extrapolated): FAILED — but by SELECTION, not
generation.** Goal FDE of the SELECTED tactical goal: 5.89 / **12.86** / 21.50 m at 2/4/6 s
vs CV baseline 1.65 / 6.09 / 12.46. The FAN's oracle: 2.38 / **5.28** / **9.92** —
**the 8-candidate goal fan BEATS CV at 4 s and 6 s; the selector throws the advantage away**
(sel_gap_tac 8.95 in the mixed-unit ordering quantity). Trainer: `train_tactical_stage0.py`
(6.39 M trainable on the frozen trunk, 4,000 steps + eval-only gate recovery after the
reporting-bug fix `0f6367e`). Artifact: `e44_gate.json`.

**Cross-level finding (programme-defining, now measured at BOTH hierarchy levels):**
operative fan oracle 0.108 vs selected 0.56–0.79; tactical fan oracle 5.28 vs selected
12.86. **Fans generate adequate hypotheses; learned pooled-feature selectors fail to find
them.** Convergent suspects: (a) the scoring features (W4c tests the spatial-attention
alternative), (b) the ranking objective itself (margin at GT-nearest under a MIXED-UNIT
error for the tactical level — flagged by the artifact's own units_note), (c) selection as
argmax at all (W7's roll-and-cost is the structural alternative). E5 goal-conditioning
should train with ORACLE (hindsight) goals and treat predicted-goal selection as the
separately-gated component it has now proven to be.

### 1.13c STAGE-A POST-TRAINING — MEASURED 2026-08-11 ~07:20Z [TIER T1-diagnostic, 881 grid]: **ALL GATES PASS — the action interface is REPAIRED**

Predictor-only post-training (`train_stage_a.py`, 3,000 steps, L_ctrl response-form vs the
unicycle analytic + L_factual + L_scene; encoder/head/emission frozen, md5-proof). Before →
after on the full held-out W3 pack: **lateral gain 0.27 → 0.971/0.966** (gate [0.5, 2.0]);
**longitudinal sign 0.745/0.787 → 1.0/1.0** (gate ≥0.95); lateral sign stays 1.0;
longitudinal gain 0.972 (reported); **P6 subspace stays exactly 3-dim**; no-harm passed.
The single root defect behind the action echo, the three scoring failures and W7's ceiling
is closed at head-scale cost. Repaired ckpt: `/workspace/experiments/stage-a-predictor/ckpt_stage_a.pt`
(pod copy gone — hosts terminated; durable copy verified 2026-08-18 on HF
`Sayood/tanitad-flagship-v5f-w120` at `release/v58f/ckpt/` `ckpt_stage_a.pt`, per the
`stack/scripts/release_v58f.py` manifest + HF tree listing);
artifact `stage_a_gate.json`. W7-on-repaired (K=32) ran ~07:35Z: gate FAIL by INSTRUMENT
COMPOSITION (the frozen-trunk-trained W4 head/selector don't compose with the repaired
trunk — §1.14), while roll-cost calibration nearly doubled (ρ 0.716) — the repair's
signal survives; W4r head refit on the stage-A trunk is the queued next pairing.

### 1.14 v5.8f — FIRST ASSEMBLED T0 NUMBERS — MEASURED 2026-08-10 ~22:35Z [TIER T0, 881 grid; families + cluster-CI rescore pending on the banked windows]

Assembly = frozen v5f-30k trunk + W4 UnicycleEmission fan + selector per gate
(`tanitad/models/v58f.py`, eval `eval_v58f.py`; artifacts HF `/v58f/`, windows banked for
rescore). Two arms, same 881 windows:

| arm | selected ADE | oracle ADE | sel accel MAE | vs v5f baseline (0.4011 / 0.1975 / 8.10) |
|---|---|---|---|---|
| **v58f rescorer-top8-kincost** (gate-decided) | **0.4815** | **0.1077** | **0.515** | ADE +0.08 WORSE · oracle 1.8× BETTER · accel **16× BETTER** |
| v58f frozen-argmax (control) | 0.7933 | 0.1077 | 0.774 | reproduces the W4 selector-mismatch reference |

**Honest reading.** v5.8f currently trades +0.08 m selected ADE against a 16× kinematic
improvement (0.515 vs 8.10 m/s², violations ~0) and a fan whose oracle nearly halves. The
whole deficit is SELECTION (sel_gap 0.374 vs v5f's 0.204) — and unlike v5f's, this gap sits
over a feasible fan with 0.37 m of recoverable headroom. Selection ladder state: W4b
feat/kin FAILED (memorisation), W4c (spatial attention) TRAINING, W7 (WM-roll re-rank)
primary. Notable: top8+kinematic-cost (0.4815) beats the trained rescorer argmax (0.560) —
on a clean fan the W1-refuted kinematic cost becomes USEFUL as a tie-breaker, exactly as the
fusion doc predicted. NOT yet a release row: four families + episode-cluster CIs on the
banked windows, then T1 (E1.4), complete it.

**W7 RESULT + K-SWEEP — MEASURED 2026-08-11 ~02:45Z [T0, 881 grid]: gate FAILED at every K,
and the failure CONVERGES the night onto one root cause.** K=8: sel 0.5772 (shortlist oracle
0.4401 — pruner-starved) but the roll-cost is the programme's FIRST CALIBRATED selection
signal (Spearman ρ 0.399 vs 0.05–0.26 for every learned scorer). K=32/64: shortlist oracle
improves hugely (0.182 / 0.142) yet selection stalls (0.5173 / 0.5319) and cost-calibration
COLLAPSES (0.106 / 0.047) — with many similar good candidates, the WM's rolled consequences
barely differ, so the cost drowns. **Why: W3 measured the WM's action-response gain at ~0.27
(¼ physical). W4b/W4c's scoring failures, W7's ceiling, and §1.12's action echo are ONE
DEFECT: the trunk under-weights actions in its rollout.** ⇒ **Stage-A post-training
(V18 E3.4: L_ctrl gain repair, targets measured by W3 — lateral gain into [0.5, 2],
longitudinal sign ≥95 %, preserve the 3-dim action subspace) is THE critical path** for
selection AND closed-loop capability; W7 re-runs after it. Artifacts:
`…/incoming/2026-08-07-hierarchical-wm-redesign/w7_gate_k8.json` + `w7_gate_k32.json` + `w7_gate_k64.json`.

**W7-ON-REPAIRED (stage-A trunk, K=32) — MEASURED 2026-08-11 ~07:35Z [T0, 881 grid]: gate
FAIL (selected 2.3468 vs thr 0.4505; frac closed −2.27) — but the failure is INSTRUMENT
COMPOSITION, not a repair verdict.** Mechanism inside the same artifact: the in-run FROZEN
selector on the recomputed fan reads **3.448** (banked 0.7933 on the frozen trunk) and the
fan oracle degrades **0.289** (banked 0.1077) — stage-A necessarily moved the trunk's
feature/state distribution, and the W4 emission head + frozen selector, both trained on
FROZEN-trunk features (`w4_meta.w4_base_ckpt = ckpt_30k_final.pt`), no longer compose with
it; winner-in-shortlist only 19.6 %. Evidence FOR the repair in the same run: across-window
roll-cost calibration **ρ 0.7164 [0.5847, 0.7696]** (episode-cluster bootstrap, n=881; P7
gate PASS — ~1.8× the frozen trunk's 0.399, the strongest calibration measured in the
programme), and W7's pick ≤ the in-run frozen pick on **71.1 %** of windows. Families
stated per the 2026-08-02 rule: LON speed MAE 0.989 m/s, accel MAE 1.067; LAT heading MAE
0.173 rad, curvature MAE 6.05 pm, yaw-rate MAE 0.250; TACTICAL winner-hit 0.9 %, sel-rank
29.9 %; STRATEGIC n/a (no route label — settled). ⇒ **W4r: refit the 4000-step unicycle
emission head on the stage-A trunk (cheap), then W7 re-runs — that pairing is the selection
verdict.** Artifact `w7_gate_repaired_k32.json`; per-window arrays `w7_eval_windows.pt`
(pod5, `w7-repaired-k32/`).

**P1/P2 (decodability battery) — MEASURED 2026-08-11 ~02:30Z [T0-diagnostic]:** driving-state
decodability from the PREDICTED latent EXCEEDS the encoded one (speed R² 0.99 vs 0.73,
curvature 0.84 vs 0.51, yaw-rate 0.86 vs 0.80 at k=5) — the rollout carries the
action-implied state strongly (caveat stamped: partly the action conditioning itself, the
P5/T1 lens applies). Lead-gap probe NOT JUDGEABLE (negative R² both sides — the
class-agnostic join needs the vehicle-class filter; instrument fix queued, not a model
verdict). Artifact: `p12_gate.json`.

**W7-FULL (selector-free, all 256 candidates) — MEASURED 2026-08-12 ~00:15Z [T0, 881
grid, pod4]: GATE FAIL 3.3348 vs 0.4505 — and it closes the selection question with a
MECHANISM, not another confound.** This run removed every stale component at once: the
repaired stage-A trunk, the W4r head refit on it, NO shortlist (topk = 256, so
`winner_in_shortlist_frac` = 1.0 — the true best candidate is always available), selection
= argmin of roll-cost + kinematic cost. Result: fan **oracle 0.1273** (excellent) but
**selected 3.3348**, sel_gap 3.207; the frozen selector on the same fan reads 4.4159, so
W7 beats it on 67.5 % of windows and still fails absolutely.
**The diagnostic pair that explains it:** the cost's *within-window* rank correlation over
the 256 candidates is **ρ_mean 0.445 / ρ_median 0.497** (it ranks broadly correctly) while
*across-window* calibration is ρ 0.3185 [0.2064, 0.4086] — yet the **argmin** is 26× the
oracle. That combination is the **winner's curse**: with a noisily-correlated cost over 256
candidates, the minimiser selects for cost UNDER-estimation, not for quality, and enlarging
the candidate set makes argmin worse — the same direction as the earlier frozen-trunk
K-sweep (0.577 → 0.517 → 0.532 at K = 8/32/64). ⇒ **Two consequences, both load-bearing:**
(1) **v5.8f ships the FROZEN-trunk assembly** (rescorer-top8+kincost, selected 0.4815
[0.393, 0.577], §1.14 above) — the stage-A trunk's physics is better but every frozen
consumer trained on the old features mis-ranks on it (frozen selector 0.7933 → 4.4159),
and **you cannot repair a trunk and keep its planner**; (2) that sentence IS the staged-
training argument for v6 — consumers must be (re)trained ON the trunk they consume
(S-W → S-T → S-S), and argmin-over-a-large-fan must be replaced by a
noise-robust rule (top-m aggregation / sharpened cost), pre-registered before it is used.
Artifact: `w7_full_gate.json`; per-window arrays `/workspace/experiments/w7-full-roll/w7_eval_windows.pt`
[⛔ UNBANKED — lived on pod4 (terminated) per this block's header, emitter `stack/scripts/w7_roll_rerank.py:715`;
never committed, and NOT in the v5.8f HF release (tree listed 2026-08-18)].

**P8 BEV-OCCUPANCY READOUT (attempt 2) — MEASURED 2026-08-12 ~00:05Z [T0-diagnostic,
881 grid, pod4]: GATE PASS — the PREDICTED latent retains the environment.** Attempt 1's
all-empty collapse was an instrument failure (unweighted BCE on overwhelmingly empty
rasters, IoU 2.7e-4 — see `p8_gate_attempt1.json`); attempt 2 (measured `pos_weight`
79.7 from 1.239 % positive cells, + soft-Dice, + a 9-point threshold sweep with τ* chosen
on the ENCODED arm — the conservative side) lifts the readout **74×** and makes the gate
computable: at k=10, **IoU(decode(ẑ)) 0.01869 vs IoU(decode(z_enc)) 0.02005 → retention
ratio 0.932** (gate ≥0.80) at **τ* = 0.7** (interior maximum of the sweep;
0.0149→0.0191→0.0180 across 0.05→0.7→0.8). ⇒ **rolling the predictor forward loses only
~7 % of the scene the encoder itself exposes** — the environment survives prediction, which
is the property P8 exists to test. **Object permanence (P4's gate, now VISUALISED):
occluded-agent recall is NOT worse than visible — enc 0.2178 occluded vs 0.1881 visible;
pred 0.1743 vs 0.1717 at k=10** (n 194/548), i.e. the latent carries agents the camera
cannot see. ⚠️ Honest limitation stamped: the ABSOLUTE IoU is low (~0.02) — a 1 M-param
readout on frozen latents against sparse rasters — so the admissible claim is the
RETENTION RATIO (one instrument, two inputs), not the absolute occupancy quality; and the
occluded≥visible parity must be re-checked against a diffuseness control before it is
quoted as permanence on its own. Artifacts: `p8_gate_attempt2.json`; reel
`p8-occupancy-c/reel/` (camera | decode(ẑ) | belief∩truth at the same τ*).

⛔⭐ **P4 PREDICATE STAMP — MEASURED 2026-08-16, and it changes how this row may be
read AND how it may be "fixed".** The join's `occ` flag (`build_obstacle_join.
visibility_occ`) and `bev_raster.fov_mask` are **THE SAME PREDICATE** — 0 of 7 680 cells
disagree at every half-angle tested (30/60/90/117/120/150/179°), and the two defaults are
**bit-identical** IEEE doubles (`0x1.0c152382d7365p+0`); they differ only in granularity
(agent centre vs cell centre). Three consequences:
(1) ⛔ **The `occluded` arm IS the out-of-field set, so the `_infov` twin that the
2026-08-16 `bev_raster` consumer audit added everywhere else MUST NOT be added here — it
would EMPTY the finding, not correct it.** MEASURED over 10 000 occluded agents per
extent: a sub-cell agent keeps 0.0–1.5 % of its cells, an automobile 10.8 % (60.4 %
emptied outright), a heavy truck 26.7 % — the survival fraction **rises with vehicle
length**, i.e. the twin re-selects the population by extent. Guarded by
`stack/tests/test_p4_fov_predicate.py`, which fails if a twin is ever added.
(2) ⚠️ **The two arms are DISJOINT REGIONS, not just two visibilities** — the occluded arm
is scored entirely inside a 590-cell wedge (7.68 % of the grid, all at x < 9.24 m) and the
visible arm over the other 92 %. That is the concrete form the required diffuseness
control must take: `--p4-region-control` (`train_p8_occupancy.py`) adds
`visible_near`/`visible_far` at the same range boundary and reports
`occluded_over_visible_near` — **≈1.0 ⇒ the gap is REGIONAL, not permanence; >1.0 ⇒ it
survives.** Pre-registered with both outcomes; NOT YET RUN (needs `p8_head.pt` + the join
file on a pod; ~5 min GPU).
(3) ⚠️ **The `pred` half of the sentence above is k=10-only.** Re-read of the banked
artifact across all four k: the ENC ordering occluded > visible holds at k = 5/10/15/20,
but the PRED ordering holds at **k = 10 alone** and reverses at 5/15/20 (−0.0035 /
−0.0034 / −0.0057). k=10 is the pre-registered gate k, so the choice is principled — but
quote the enc arm, or quote the pred arm *with* this. Also: the join flagged at the
sensor's 120° while the encoder saw the 117° sub-frame, which puts unseen agents in the
*visible* bucket and can only SHRINK this gap ⇒ the banked number is **conservative**.
Artifacts: `…/incoming/2026-08-16-p4-fov-predicate/P4_FOV_PREDICATE.md` +
`raw/p4_predicate_identity.json`; both `p8_gate_attempt1.json` + `p8_gate_attempt2.json` annotated in place.

**I4a IMAGINATION ABLATION — MEASURED 2026-08-11 ~19:40Z [T0, 881 grid]: the imagination
channel is LOAD-BEARING, not decorative.** Three arms, same checkpoint, same grid, only the
imagination input changed (`eval_flagship_v4 --imagination-ablate`): **intact ADE 0.4011**
(byte-matches the banked v5f baseline — an in-run instrument-parity proof), oracle 0.1975,
miss@2m 0.149; **zeroed 7.6493** (19× collapse; oracle 1.457, miss@2m 0.805); **shuffled
1.2492** (3.1×; oracle 0.426). The ordering zero ≫ shuffle ≫ intact is the discriminating
result: shuffling preserves the marginal statistics and destroys only the
window↔consequence correspondence, so the planner is reading imagination as CONTENT, not
as a bias term. ⚠️ Caveat stamped: the head was TRAINED with imagination present, so this
measures the dependence of THIS architecture, not the value of retraining without it;
I4b (occluded-split stratification) is the next refinement. Artifacts: three JSONs in
pod5 `/workspace/experiments/i4a/` (local stems flagship-v5f-w120-30k-i4a-none/zero/shuffle;
pod terminated, never committed) — durable copies verified 2026-08-18 on HF
`Sayood/tanitad-flagship-v5f-w120` at `release/v58f/gates/` as `i4a_none.json` /
`i4a_zero.json` / `i4a_shuffle.json`, per the `stack/scripts/release_v58f.py` manifest +
HF tree listing; banked in-repo (md5-verified against that manifest) at
`…/incoming/2026-08-18-v58f-artifact-banking/gates/i4a_none.json` + `i4a_zero.json` +
`i4a_shuffle.json`.

**W4r + W7-w4r — MEASURED 2026-08-11 ~19:10Z [T0, 881 grid]: the repair arc closes on ONE
remaining stale part.** W4r (unicycle head refit ON the stage-A trunk, 4000 steps, trunk
md5-frozen): **GATE PASS — fan oracle 0.1273** (cap 0.2173 vs the 0.1975 reference ✓),
winner accel MAE 0.276, selected-accel 0.697, violations 0.0. The fan on the repaired
trunk is HEALTHY. Same fan through the FROZEN selector: selected **4.416** — and W7-w4r
(K=32, frozen-selector shortlist) FAILS at **3.614** (thr 0.4505) because the shortlist
itself is poisoned: fan oracle 0.127 vs shortlist ceiling set by a selector still trained
on frozen-trunk features. Chain of eliminations now complete: trunk repaired (§1.13c),
head refit PASS, roll-cost calibrated (ρ 0.716) — **the frozen selector is the last stale
component, and it sits in W7's PRUNER, not its cost.** ⇒ **W7-FULL queued (topk 256 = no
shortlist, selector-free): roll-cost + kinematic cost over the whole healthy fan — the
first selection read of the fully-repaired pipeline with NO stale part anywhere** (pod4,
behind p8c; W4r head relayed via HF /battery/). Artifacts: `w4r_gate.json`,
`/workspace/experiments/w7-repaired-w4r-k32/w7_gate.json` (pod5, terminated — durable copy
verified 2026-08-18 on HF `Sayood/tanitad-flagship-v5f-w120` at `release/v58f/gates/` as
`w7_w4r_k32_gate.json`, per the `stack/scripts/release_v58f.py` manifest + HF tree listing;
banked in-repo, md5-verified against that manifest, at
`…/incoming/2026-08-18-v58f-artifact-banking/gates/w7_w4r_k32_gate.json`).

**P1 LEAD-GAP RESOLUTION — MEASURED 2026-08-11 ~17:20Z (two runs, pod4): the instrument
was fixed AND the failure survived it — MODEL VERDICT "missing state variable".**
Run 1 (class filter applied, provenance in-artifact, n=266 vehicle-lead windows): still
R²(enc) ≤ 0; P1's other 3 targets PASS their retention gates (speed pred 0.993/enc 0.744).
Run 2 (transform + capability-ceiling probes, episode-disjoint OOF, CPU off the banked
arrays): log1p/inverse/TTC-proxy linear ALL fail and the 2-layer MLP ceiling reads
**−0.334** — the latent lacks a readable lead-distance variable in any parameterization
tested (small-n caveat stamped in the battery doc). Load-bearing consequence: 88.7 % of
the oracle gap is longitudinal, and this is the missing longitudinal state variable.
⚠️ **The aux-label-loss lever originally proposed here was RETRACTED the same day by the
PI (labels into the trunk break the JEPA self-supervised thesis); the response is the
LABEL-FREE lever program in `JEPA_PHYSICS_SURVEY.md`** — LF0 (locate: probe PRE-POOL
spatial tokens + P8 decoded-BEV read-off; probe-only, admissible) then
interaction-weighted sampling / masked-latent objectives / dense near-field loss shaping,
each gated on the SAME frozen P1 lead battery. Headway/TTC stay GT-join instruments.
Artifacts: `p12_gate_clsfilter.json`, `p1_lead_transforms.json` (+ HF /battery/ arrays).

**⭐ T1 PSEUDO-CLOSED-LOOP — MEASURED 2026-08-11 ~23:27Z, analysed 2026-08-12 ~00:10Z
[TIER T1 = PRIMARY; 6 844 windows / 40 val episodes, stride 1; episode-cluster bootstrap;
pod5]. THE HEADLINE MEASUREMENT OF THE v5.8f CAMPAIGN, and it is two findings, not one.**

Instrument `taniteval/tools/t1_eval.py` (`--v2-val-cache --grounding-readout`), three
surfaces per arm: **`cl`** = action-closed loop (the model conditioned on its OWN actions,
**T1**), **`ol`** = teacher-forced (**T0**, a WM diagnostic and never driving performance),
**`ha`** = hold-action control (**T1**). Point estimates are `full_set` pooled means;
intervals are the episode-cluster bootstrap and cross-arm deltas the **paired** version on
the same windows — `overlapping_holdout_se` is used nowhere.

| arm | surface | tier | ADE dense (m) | FDE last (m) | LON speed MAE (m/s) | LON along MAE (m) | LAT cross MAE (m) | LAT heading MAE (°) |
|---|---|---|---|---|---|---|---|---|
| `v5f-30k` | `cl` | **T1** | **23.9837** [21.442, 26.347] | 53.4756 | 26.9356 | 23.8965 | 0.9993 | 3.6204 |
| `v5f-30k` | `ol` | T0 | 0.9397 [0.8162, 1.0679] | 2.8003 | 1.4431 | 0.8762 | 0.1947 | 3.3483 |
| `v5f-30k` | `ha` | **T1** | 0.9597 [0.8361, 1.0879] | 2.8631 | 1.4531 | 0.8901 | 0.2072 | 4.5954 |
| `stage-a-repaired` | `cl` | **T1** | **9.3697** [6.6822, 12.2576] | 19.5256 | 9.7291 | 9.2655 | 0.7446 | 5.3945 |
| `stage-a-repaired` | `ol` | T0 | 0.3659 [0.2926, 0.4521] | 1.0231 | 0.5113 | 0.2990 | 0.1534 | 1.8351 |
| `stage-a-repaired` | `ha` | **T1** | 0.4246 [0.3500, 0.5132] | 1.2242 | 0.5671 | 0.3487 | 0.1689 | 3.9859 |

**FINDING 1 — the stage-A repair wins on every surface, decisively and separated.**
Paired `stage-a-repaired − v5f-30k`, same windows: `cl` ADE **−14.6139 [−16.9319, −12.2010]**
(`p_delta_gt0` 0.0, ✅ separated) with `cl` LON speed MAE **−17.2064 [−19.7815, −14.4927]`;
`ol` ADE −0.5739 [−0.7002, −0.4570]; `ha` ADE −0.5351 [−0.6644, −0.4181]. The repair was
designed to restore action-response gain (0.27 → 0.971/0.966, longitudinal sign 1.0, §1.13c)
and it improves **exactly the axis it targeted** — a clean confirmation, not a coincidence.

**FINDING 2 — ⛔ THE CLOSED LOOP DIVERGES, AND THE HOLD-ACTION CONTROL BEATS IT BY 22×.**
For the repaired arm `ha` **0.4246** vs `cl` **9.3697**; for v5f `ha` 0.9597 vs `cl` 23.9837.
A control that simply holds the last action is an order of magnitude better than the model
driving itself. Within-arm paired `cl − ol`: **+9.0039 [6.3659, 11.8487]** (repaired) and
**+23.0439 [20.5613, 25.3884]** (v5f), both separated at `p_delta_gt0` 1.0. ⇒ **No
closed-loop driving competence may be claimed for either arm.** The T0 number (0.3659) and
the T1 number (9.3697) differ by **25×** on the same checkpoint and the same windows — this
row is the strongest evidence yet for the tier doctrine, and it is exactly the failure
`EVAL_DOCTRINE.md` was written to expose.

**⭐ THE DIVERGENCE IS ~99 % LONGITUDINAL — visible ONLY because the four families are
reported.** Of the repaired arm's `cl` ADE 9.3697, **LON along-track MAE is 9.2655** while
**LAT cross-track MAE is 0.7446**; for v5f, 23.8965 of 23.9837 against 0.9993 lateral. The
car holds its lane and its SPEED integrates away (LON speed MAE 9.73 and 26.94 m/s — both
physically implausible, i.e. a true blow-up rather than a graceful degradation). This
sharpens the standing "88.7 % of the oracle gap is longitudinal" (§1.14, T0) to **~99 % at
T1**, and it converges with the P1 verdict that the latent lacks a readable lead-distance
variable. ⚠️ A scalar ADE would have shown a 25× gap and NOT shown that it is one axis —
this is the four-family rule earning its cost in a single row.

**Consequences for v6, all load-bearing.** (1) The staged ladder is now empirically, not
just architecturally, motivated: **S-W must produce a world model stable under its OWN
actions before any planner is attached** — that is precisely the quantity `cl − ol` measures,
and it is the natural S-W gate. (2) The longitudinal channel is the design target, not the
lateral one. (3) `ha` is the floor any closed-loop claim must clear first; clearing `ol` is
not evidence of anything driving-related.

⚠️ Scope stamped honestly: `ha` is a strong baseline partly *because* the corpus is
short-horizon and near-constant-speed — that is what makes it the right "do nothing clever"
floor, not a reason to discount it. TACTICAL/STRATEGIC were `UNAVAILABLE` in this run
(`_families_unavailable`); TACTICAL has since been closed at source for all future T1 runs
(`t1_eval.py` now passes `tactical_from_traj=True, tier=t` — at T1 the driven path IS the
manoeuvre decision), and STRATEGIC stays `n/a` with reason + n because PhysicalAI-AV carries
no map, lane graph or route signal. Distance-keeping remains UNAVAILABLE pending a lead
block on this dense grid (`tools/build_lead_block.py`) — a WORK ITEM, not a pass.

> ⚠️ **PATH ANNOTATION APPENDED 2026-08-16 by the stale-blocker sweep — the numbers and the verdict
> above are untouched; only the citation is corrected.** **`tools/build_lead_block.py` does not
> exist.** The repo-root `tools/` holds 16 files and none of them is `build_lead_block.py` (probed
> three ways: `git ls-tree HEAD -- tools/`, `git ls-files -- tools/`, filesystem). The instrument
> **does exist**, at **`taniteval/tools/build_lead_block.py`**, with its pure-join sibling
> `taniteval/taniteval/lead_source.py` and metric `taniteval/taniteval/lead_metrics.py`.
> ⇒ **This is exactly the defect §4.2 of this registry names — "it sends the next reader to a path
> that does not exist" — and here it is worse than a dead link: it makes a BUILT instrument look
> UNBUILT, which invites re-commissioning work that is already done.**
> ⛔ **The WORK ITEM itself STANDS and is NOT closed:** the lead block has not been built *for this
> dense 6 844-window T1 grid*, so distance-keeping is genuinely UNAVAILABLE **on these rows**. What
> is wrong is only the path — and the implication that the tool is missing.
> ⚠️ **Scope note, because these two facts are easy to read as a contradiction:** distance-keeping
> **is** closed on the **OOD-val 290/40-episode grid** (§ above, `families_unavailable=[]`,
> `v1arch_oodval_q90_4fam_LEAD.json`) and **not** closed on **this T1 dense grid**. Both are true;
> they are different grids and different arms. Always carry the grid with the availability claim.

⚠️ Instrument note for anyone reproducing: both arms rolled all 40 episodes and then died in
`analyze()` on `from taniteval import selgap` (pod5's package predates the module). The
dumps survived, so the numbers above come from `--analyze-only` over them with **zero GPU
recompute** — but an analysis-time import that fails after 11 minutes of rollout is a
standing hazard, and the same class as the `UnicycleStepReadout` failure earlier the same
night. Artifacts: `t1_v58f_summary.json` (banked in
`…/incoming/2026-08-07-hierarchical-wm-redesign/`), per-arm `t1_v5f_30k.json` /
`t1_stage_a_repaired.json` and the 80 episode dumps (pod5:`/workspace/experiments/t1-v58f/`).

**⭐ T1 FOUR-FAMILY RESCORE — MEASURED 2026-08-12 ~00:55Z [6 arms, 15 paired contrasts,
`FF_EXIT=0`, same 6 844-window grid; `taniteval/tools/ff_rescore.py`]. THE MECHANISM BEHIND
THE DIVERGENCE, AND IT IS DIRECTIONAL, NOT NOISE.** The binding rule is now satisfied on
these rows (`rule_satisfied: true`, `_families_unavailable: ['strategic']` only).

**LONGITUDINAL — the closed loop ACCELERATES AWAY.** `stage-a-repaired · cl`: speed MAE
**9.7291** m/s with **speed BIAS +9.3892** — the bias is ~96 % of the error, so this is a
systematic over-speed, not scatter. Along-track MAE 9.2655 with **bias +9.0407** and final
bias **+18.5801 m**; accel MAE **19.0948 m/s²** (>1.9 g — physically impossible, i.e. a true
blow-up); **ego progress ratio 1.7279** (median 1.0994), so the arm drives **1.73× the
distance the human did**. Target-speed accuracy: **0.3398 / 0.5069 / 0.6564** within
0.5 / 1.0 / 2.0 m/s. ⇒ **the v6 longitudinal work item is a runaway-acceleration failure with
a known sign** — far more actionable than "ADE 9.37".

**LATERAL — healthy, and NOT the problem.** heading MAE **3.8776°**, yaw-rate MAE
**4.9188 °/s**, curvature MAE **0.0186 1/m** (bias −0.0024), cross-track MAE **0.7446 m**
(final 2.1565). n_steps 128 988 heading / 122 151 curvature, 7 892 steps excluded below
`min_ds` 0.05 m. All four LATERAL members the rule names are present — heading, curvature,
yaw-rate and cross-track — and none of them is where the failure lives.

**TACTICAL — the factored view shows longitudinal decision-making is AT CHANCE.**

| decision axis | accuracy | Cohen's κ |
|---|---|---|
| lateral (`stage-a-repaired · cl`) | 0.7515 | **0.3795** |
| **longitudinal** (`stage-a-repaired · cl`) | 0.3327 | ⛔ **0.0405** |
| collapsed 5-way | 0.3036 | 0.1404 |
| lateral — **hold-action control** | 0.8675 | 0.6427 |
| longitudinal — **hold-action control** | 0.5586 | 0.2072 |

κ 0.0405 is chance agreement. The collapsed 5-way (κ 0.1404) sits *between* the two axes and
reports neither — **the direct measurement of the lat/lon-mixing softmax defect CLAUDE.md
names as our largest known architectural problem**, and it is only visible because the family
is reported factored. Per-class lateral: `lane_keep` recall 0.8092 / precision 0.8747;
`turn_left` 0.4994 / 0.6627; `turn_right` recall 0.6003 / **precision 0.2879** (1 195
predicted against 573 true — right turns over-predicted 2.1×). ⚠️ The hold-action control
beats the model on BOTH axes, which is the tactical restatement of Finding 2 above.

**TACTICAL goal-setting — the DIRECTION is right and the DISTANCE is wrong.** Goal bearing
MAE **4.8098°** (bias −1.8323°, n 6 603, 241 windows excluded below 0.5 m) against goal range
ratio **1.7584** and long-bias **+18.5801 m** vs lat-bias **−1.2061 m**. ⇒ **the model knows
WHERE to go and not HOW FAR** — one clean sentence that ADE, and even ADE-plus-FDE, cannot
express. (`goal_point_error_m` 19.5256 is FDE under another name and is labelled as such in
the artifact, not sold as a new metric.)

**STRATEGIC — `n/a` with reason and n = 6 844**, per clause 5 of the binding rule:
PhysicalAI-AV carries no map, no lane graph, no junction/roundabout label, no traffic-light
feature and no route/goal signal (the dataset card says verbatim *"we do not include open
maps data"*). No rescore can close this; the programme's instrument is the VLM pipeline
PH0→PH1→PH2. Distance-keeping (the other half of LONGITUDINAL) also remains UNAVAILABLE
pending a lead block on this dense grid (`tools/build_lead_block.py`) — a WORK ITEM, not a
pass, and the half where 88.7 % of the T0 oracle gap was measured to live.

> ⚠️ **PATH ANNOTATION APPENDED 2026-08-16 (same correction as §1.13's, repeated here because this
> line is quoted independently).** The instrument is **`taniteval/tools/build_lead_block.py`**, not
> `tools/build_lead_block.py` — the latter does not exist. The **WORK ITEM stands** for this dense
> grid; only the citation and the "tool is missing" implication are wrong. Distance-keeping **is**
> closed on the OOD-val grid (`families_unavailable=[]`) — carry the grid with the claim.

**⭐ PH0 v2 VLM EXTRACTION — GATE PASS — MEASURED 2026-08-12 [8-clip smoke, pod4,
`Qwen/Qwen3.5-9B` via `AutoModelForImageTextToText`, grammar-constrained].** The PI's
decided stack (engine B Qwen3.5-9B · engine C SAM3 · engine A algorithmic integrated ego
path) against the pre-registered v2 gate in `PH0_TARGET_STRUCTURE_v2.md` §5:

| criterion | v0.1 MEASURED | v2 threshold | v2 MEASURED | |
|---|---|---|---|---|
| clips with ALL calls valid | 1 / 8 | ≥ 6 / 8 | **8 / 8** | ✅ |
| hard `no parseable JSON` failures | 3 / 8 | 0 / 8 | **0 / 8** | ✅ |
| B1 scene valid | — | ≥ 7 / 8 | **8 / 8** | ✅ |
| B4 goal ∈ vocabulary | — | ≥ 7 / 8 | **8 / 8** | ✅ |

`retried = 0` — every call was valid on its first attempt; the retry path never fired.
Extracted symbols are coherent: `urban/day` 3 signs → `follow_main_road`+`hold_corridor`;
`urban/night` → `turn_left`+[`prepare_lane_change`,`reduce_to`]; `highway/snow` 0 signs →
abstained to `follow_main_road`; one `route_to` carrying its required `goal_evidence_sign`.

**What made it solvable** (all four are mechanisms, not tuning): the single 5-section /
4-level / ~30-field object split into **four flat calls**; grammar-constrained decoding, so
an unparseable output is impossible by construction; `max_consecutive_whitespaces=1`, since
the default 12 let the model burn its budget on tabs and truncate mid-object; and
`force_json_field_order=True`, which makes `n_signs` genuinely precede `signs[]` rather than
by luck. ⭐ **The organising principle is the load-bearing part: the VLM chooses SYMBOLS, the
algorithm supplies NUMBERS, SAM3 supplies PIXELS.** Every metric slot (`within_m`,
`by_time_s`, `at_arc_m`, `hold_for_s`, `v_target_ms`) was REMOVED from the VLM's job —
Engine A measures them — and a test fails if any reappears.

⚠️ **Two of the defects on the way were OURS, and both had misled a report.** (1) The arm was
called "unusable / text-only" for a day; it was a `[swscaler]` EAGAIN (ffmpeg sizing its pool
to the HOST's 96 CPUs) plus loading a VLM through the **text-only** `AutoModelForCausalLM`,
which loads fine and then rejects the vision kwargs at `generate()`. (2) `bbox
[952,100,975,160]` was read as an out-of-frame hallucination against a 448 px maximum —
Qwen-VL emits **normalized 0–1000** coordinates, its own trained convention, so the model was
self-consistent and the check was wrong twice over (frames are 179×448, so y never reaches
448 either). Also retracted: duplicate actions are a padding artifact, not a content error,
and rejecting the record for them discarded a good `goal_kind`.

⚠️ **Scope stamp: n = 8 is a SMOKE, not a measurement.** The prereg is explicit that PH1's
50 clips produce quotable rates; this gate exists to decide whether PH1 is worth launching,
and it says yes. Still open: the processor reports `fps=24` for a 2 fps sample (a temporal
mismatch that B4's hindsight premise depends on), `alpamayo_rows = 0` on every clip (engine D
contributed nothing), and SAM3's real API is installed but not yet wired.
Artifacts: `/workspace/ph0_mini/v2/ph0_v2.json` (every prompt + raw model output banked
per call) [⛔ UNBANKED — lived on pod4 (terminated) per this block's header, out-dir per
`stack/scripts/ph0_v2_chain.sh:20`; never committed, and NOT in the v5.8f HF release
(tree listed 2026-08-18)],
`ph0_mini/v2/viz/` (overlay MP4 + stills); instruments `stack/scripts/ph0_v2.py`,
`ph0_v2_chain.sh`, `ph0_v2_overlay.py`, 44 CPU tests.

**⛔ LF0 — MEASURED 2026-08-12 ~08:30Z [T0 diagnostic, 900 windows scanned / 129 labelled,
pod4]: THE DECODED BEV DOES NOT READ THE LEAD GAP. RC1 IS NOT SUPPORTED — there is no
zero-training fix.** `scripts/lf0_bev_lead.py`, a **zero-parameter geometric read**: walk the
decoded occupancy raster forward along the ego corridor, return the range of the first cell
≥ τ. τ = 0.7 **inherited** from the P8 gate, never re-tuned. Nothing is fitted, so a spatial
arm cannot win by having more capacity than P1's pooled probe — which is why this and not a
spatial-token probe is the admissible first test.

**Reader sanity gate: PASSED**, and it is a precondition, not a formality. GT reads at other
corridor widths rank-agree with the headline: `gt@1.0` ρ **1.0** (R² 0.9998, MAE 0.0144 m),
`gt@2.0` ρ **0.9596** (R² 0.8904, MAE 0.2829 m). The corridor geometry is right and the reader
recovers the labelled scene essentially exactly.

| arm (corridor 1.5 m) | R² | ρ | MAE (m) | n paired | **censored on labelled** |
|---|---|---|---|---|---|
| `gt@1.5` (= truth, by construction) | 1.0 | 1.0 | 0.0 | 129 | 0 % |
| **`enc@1.5`** (decoded ENCODED latent) | **−21.00** | 0.3826 | **26.85** | 24 | ⛔ **81.4 %** |
| **`pred@1.5`** (decoded PREDICTED latent) | **−16.12** | −0.7091 | **42.65** | 10 | ⛔ **92.3 %** |

**⭐ THE FINDING IS THE CENSORING RATE, not the R².** In **81.4 %** (encoded) and **92.3 %**
(predicted) of the windows where the ground truth has a lead vehicle in the ego corridor, the
decoded BEV has **no occupied cell there at all** — it shows an empty lane. That statistic rests
on all **129** labelled windows and is decision-grade. When the decode does fire, the read is
wrong by **26.85 m / 42.65 m** on a grid only 60 m deep. ⚠️ The R² and ρ values sit on n = 24 and
n = 10 (the latter exactly at the pre-declared floor) and are **NOT decision-grade** — in
particular `pred`'s ρ = −0.71 on n = 10 is noise, not an inverted signal, and must not be quoted
as one.

**This is the concrete consequence of P8's own stamped limitation, not a contradiction of it.**
§1.14 already records that P8's absolute IoU is ~0.02 and that "the admissible claim is the
RETENTION RATIO, not the absolute occupancy quality". LF0 is what that caveat means in practice:
the decode preserves enough *relative* structure to score retention 0.932, and is far too diffuse
to support the statement *"there is a vehicle at 18 m in my lane"*. Both results stand together.

**⭐ REFINEMENT (2026-08-12, from the rendered panels — `Paper/figures/lf0_bev_panels.svg`):
"the decode shows an empty lane" UNDERSTATES it. The decode is NOT blank.** In the three
inspected windows it puts **40 / 43 / 45** (encoded) and **68 / 43 / 35** (predicted) cells
above τ — *comparable to the ground truth's 33 / 31 / 31* — but essentially none inside the
ego band. The failure is **confident MISLOCATION, not absence of output**, which is exactly
what IoU ≈ 0.02 beside a retention ratio of 0.932 means: the relative structure survives
prediction, the absolute placement does not.

**⚠️ A "small lateral offset" hypothesis was raised from eyeballing those panels (the decoded
mass sits just outside the band) and TESTED — it is NOT SUPPORTED.** Corridor-width sweep on
the same run:

| width | `enc` censored | `enc` n | `enc` R² | `enc` MAE | `pred` censored | `pred` n |
|---|---|---|---|---|---|---|
| ±1.0 m | 82.17 % | 23 | −21.31 | 27.93 | 92.25 % | 10 |
| ±1.5 m | 81.40 % | 24 | −21.00 | 26.85 | 92.25 % | 10 |
| ±2.0 m | **68.22 %** | 41 | **−28.66** | **30.71** | 92.25 % | 10 |

Widening to ±2.0 m recovers 14 points of encoded censoring but makes **R² and MAE WORSE**, so
the extra detections are **other traffic at wrong distances**, not a laterally-displaced lead —
a genuine offset would have recovered the lead at the *right* range and improved the fit. The
predicted arm is **completely flat across all three widths** (identical censoring, n, R² and
MAE), i.e. widening the band adds nothing whatsoever. ⇒ the failure is not a calibration or
band-geometry artefact.

**⇒ Consequences, and they are load-bearing.** (1) **The survey's RC1 is REFUTED**: exposing an
existing read-off does not close the lead gap, so there is **no zero-training fix** and the
longitudinal lever must be training-side. (2) This is the **second independent test**, with a
different instrument class — a zero-parameter geometric read versus P1's fitted probe (linear
R²(enc) ≤ 0, every transform failed, 2-layer MLP ceiling **−0.334**) — reaching the same verdict:
**the lead gap is not readable from this latent in any form yet probed**. (3) It converges with
the T1 result above: the closed loop is ~99 % longitudinal and over-accelerates (progress ratio
1.7279, speed bias +9.3892 m/s), and the model cannot see the vehicle in front of it. Artifacts:
`/workspace/experiments/lf0-bev-lead/lf0_gate.json` (pod4, terminated) [⛔ UNBANKED — out-dir per
`stack/scripts/lf0_chain.sh:17`; never committed, and NOT in the v5.8f HF release (tree listed
2026-08-18)]; instrument `stack/scripts/lf0_bev_lead.py` + `lf0_chain.sh`,
21 CPU tests.

**W7-PROG — MEASURED 2026-08-12 ~05:40Z [EXPLORATORY, 881 grid, pod4]: PRE-REGISTERED
OUTCOME = PARTIAL. The anti-degeneracy term is REAL, MONOTONE, and FAR TOO SMALL.**
Pre-registration `PREREG_W7_PROG.md` (written before the run, all three outcomes bound). Only
`--w-prog` changes against the W7-FULL control, so the contrast is attributable. PRIMARY endpoint
was declared to be the **argmin's mean ERROR-RANK over the 256-candidate fan**, not ADE, because
the mechanism under test is a ranking claim.

| arm | `--w-prog` | **error-rank of argmin** /256 | gate `w7_selected_ade` |
|---|---|---|---|
| control (= W7-FULL) | 0.0 | **132.3** | 3.3348 |
| `w7-prog-01` | 0.1 | **130.31** | 3.4360 |
| `w7-prog-05` | 0.5 | **126.69** | 3.7398 |

The rank falls **monotonically** with the weight (132.3 → 130.31 → 126.69) — so the degenerate-
minimiser account is **not** wrong: switching on the progress term does move the argmin toward
better candidates, exactly as predicted. But the whole effect is **5.6 rank positions of 256
(2.2 % of the fan)**, the argmin still sits essentially at the median, and the gate ADE gets
**monotonically worse** (3.3348 → 3.4360 → 3.7398). ⇒ **PARTIAL**, by the pre-registered rule
(`w7-prog-05` clears the <128 clause; neither arm approaches the <100 CONFIRM threshold and
neither improves ADE). Per the pre-registration's own PARTIAL branch, the consequence is binding
and was fixed in advance: **the cost needs a goal-conditioned component, not a larger
anti-degeneracy weight, and W7-style self-consistency selection is retired as a headline route**
(the V-JEPA-2-AC / DINO-WM goal-cost contrast in the paper is the next lever).

⚠️ Also measured and worth its own line: with the progress term the across-window calibration
**flips sign** — Spearman(cost, realised error) goes **+0.3185 (control) → −0.4244 (w-prog 0.1)**.
A cost that is *negatively* calibrated across windows is worse than an uninformative one, which is
the mechanism behind the ADE regression and independent evidence that this cost family is not
merely under-tuned. ⚠️ EXPLORATORY stamp holds: this re-uses the W7 scoring windows, so no arm
here is quotable as a v5.8f number. Artifacts: `w7_gate.json` + `rules.json` in each arm run dir,
w7-prog-01/ and w7-prog-05/, on pod4 [⛔ UNBANKED — pod terminated, never committed, and NOT in
the v5.8f HF release (tree listed 2026-08-18)].

Instrument change that made this possible: `t1_eval.py` now calls
`all_families(win, tactical_from_traj=True, tier=t)`. At T1 nothing steers the rollout but the
arm's own actions, so the driven path IS its manoeuvre decision; at T0 the same block is
stamped as substantially an ACTION ECHO so a teacher-forced tactical number can never be read
as skill. Artifacts: the six per-tier JSONs ff_{stageA,v5f30k}_{cl,ol,ha}.json + `ff_comparison.json`
(pod5:`/workspace/experiments/t1-v58f/four_families/`, terminated — all seven durable, verified
2026-08-18 on HF `Sayood/tanitad-flagship-v5f-w120` at `release/v58f/gates/four_families/`, same
basenames, per the `stack/scripts/release_v58f.py` manifest + HF tree listing; all seven banked
in-repo, md5-verified against that manifest, at
`…/incoming/2026-08-18-v58f-artifact-banking/gates/four_families/` — six under the same
basenames, the full comparison as `ff_comparison.full.json`, basename-disambiguated on purpose
from the smaller in-repo demo
`…/incoming/2026-08-07-hierarchical-wm-redesign/ff_rescore_val40_demo/ff_comparison.json`).

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
| 🟥 **Why the number is unusable** | `taniteval/registry.py` records: *"320-ep variant … Canonical val 80 % LEAKED into its train set → guard excludes; clean number lives on the f1b378 val (pod3 gates)."* Both arms also overfit hard at 320 eps, so the I-JEPA-beats-DINOv2 read is an **overfit-regime ranking with data binding, not a feature-quality verdict.** ✅ ⛔ **THE QUOTED REMEDY IS INVERTED — flagged 2026-07-28.** `f1b378` is **not** the clean val: **62 of its 80 episodes (77.5 %) are bit-identical to parity-train episodes** (MEASURED by content — sha256 of raw `poses` and `frames_u8`). The clean val is `physicalai-val-0c5f7dac3b11` (0/40, 0/600 by content). The wording lives in `taniteval/registry.py:85-87` and must be fixed there; see R8 and `…/incoming/2026-07-28-leaky-cache-audit/`. |
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

| Metric | heldout 🟥 **DEPRECATED — `overlapping_holdout_se`** | vs flagship-30k 🟥 **also heldout** |
|---|---|---|
| ADE@0.5s | 1.2680 ± 0.1657 | 0.0762 |
| ADE@1s | 1.8201 ± 0.2440 | 0.1584 |
| ADE@1.5s | 2.3650 ± 0.3209 | 0.2883 |
| **ADE@2s** | **2.9196 ± 0.3937** (**full-set 3.0471** [2.4984, 3.6878] — decision-grade, §6) | 0.4522 (**full-set 0.4271**) |
| FDE@2s | 4.5832 | 0.9437 |
| miss@2m | 0.7246 | 0.0602 |

⚠️ **BOTH COLUMNS ARE `heldout` SPLIT-MEANS, so the right-hand column is a CROSS-ARM COMPARISON OF TWO
SPLIT-MEANS — which §4.1b of this document explicitly rules invalid** (*"`heldout` means are NOT
comparable across arms"*). Only the ADE@2s row has both forms; the other five rows have **no
decision-grade value published anywhere**. Flagged 2026-08-17, not fixed — closing them needs a
re-emission from `driving_refa-dynin-30k.json` / `driving_flagship-30k.json`, which is a §6 job.

Paired A/B (881 windows): flagship wins **95.9 %**, Δ **+2.62 m**.
⛔ **INTERVAL CORRECTED 2026-08-17 — this line published `CI95 [2.447, 2.798]`, which is the BANNED
`overlapping_holdout_se` paired form.** The decision-grade paired episode-cluster bootstrap for the *same
delta* is already published in §6 of this same document: **[+2.0945, +3.2570]**. ⚠️ **The banned interval
was 3.31× too narrow** (width 0.351 vs 1.1625) — **above the top of the programme-wide 1.107–3.100×
band**, and it is a **paired delta**, the exact statistic the 2026-07-25 blast radius measured errors of
up to **×−4.15 including a sign flip** on. *The same document carried both intervals for the same
quantity, 800 lines apart, with the narrow one unlabelled.* ✅ **The verdict is untouched** — the delta is
separated from zero by a wide margin under either estimator. *Superseded, kept visible: [2.447, 2.798].*

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

> ⚠️ **Lineage trap, documented:** `tanitad-pod4:/workspace/rescue/experiments/refb-speed-30k/ckpt_prepatch_step8500.pt`
> is **byte-identical (md5)** to
> `tanitad-pod4:/workspace/rescue/experiments/refb-speed-30k/ckpt.pt`. There is one checkpoint, and it is
> at **step 10,000**, not 8,500. The file name is wrong.
>
> 📍 **Host added 2026-08-03 (this citation was a host-less fragment and resolved to nothing).** The run
> was produced on `tanitad-pod` at `/workspace/experiments/refb-speed-30k/`; that pod refuses connections
> as of 2026-08-03 and the **only reachable copy is pod4's rescue dump**. MEASURED (mine, read-only `ls`
> over ssh, 2026-08-03): both files present, **3 153 889 214 B each** — equal size, consistent with the
> byte-identical claim above, though I did **not** re-run md5 (that would put sustained disk I/O on a pod
> that is training `flagship-v1arch-v2bal-30k`). ⛔ **No HF copy**: no repo under `Sayood/` holds a file of
> that size (corpus-durability sweep, 2026-08-03) — so this arm is **single-disk on a rented pod**.

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
| **Location** | `tanitad-pod:/workspace/experiments/refb-refbpatch-v2-30k/` · milestones `tanitad-pod:/root/refb_milestones/ckpt_step5000.pt`, `tanitad-pod:/root/refb_milestones/ckpt_step15000.pt`, `tanitad-pod:/root/refb_milestones/ckpt_step20000.pt` · eval copy `tanitad-eval:/root/models/refb-v2-30k/ckpt.pt`. ⚠️ **Reachability MEASURED 2026-08-03:** both `tanitad-pod` and `tanitad-eval` return `Connection refused` at their recorded addresses (a RunPod volume resize presents the same way, so this is *unreachable*, not *proven destroyed*), and `/root/refb_milestones/` is **not** in pod4's rescue dump. The **run dir itself IS rescued** — `tanitad-pod4:/workspace/rescue/experiments/refb-refbpatch-v2-30k/` — and its `ckpt.pt` size-matches the `ckpt.pt` in HF `Sayood/tanitad-refb-speed`. ⛔ **The three milestones have no reachable copy.** |
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
| **HF** | ✅ **`Sayood/tanitad-refc-xl`** — public + **gated `manual`** (access by owner approval), pushed 2026-07-25. Files: `ckpt.pt` **3,024,021,445 B** (md5 `966d4eff1ea5ddf86efba01b8344e198`, re-verified against this row immediately before upload), `config.json`, `metrics.json`, model card. An anonymous `HEAD` of `https://huggingface.co/Sayood/tanitad-refc-xl/resolve/main/ckpt.pt` returns **401 `GatedRepo`** *(re-verified by me 2026-08-03, unauthenticated request, `X-Error-Code: GatedRepo`)* — weights are NOT world-downloadable (verified, not assumed). Now 3 copies (HF + eval pod + pod3). Note: `…/incoming/2026-07-25-refc-hf-push/NOTE.md` |

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
> **8.4 % of the ORACLE gap, on its own training data** *(qualifier added 2026-08-03: an unrelated **8.4 %** at §1.6 is a **relative** change in the flagship's fan — the two were merging)*. Not capacity (smaller heads are worse), not
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

**Results — FINAL step 29,999 (`refc-small-30k`), 881 windows** ✅

⛔ **CITATION CORRECTED 2026-08-03.** This block previously cited
`taniteval/results/refc-small-30k.json` as *the raw* and the hub directory as *"repo copies"*.
**`taniteval/results/refc-small-30k.json` DOES NOT EXIST** (probed at `taniteval/results/`,
`taniteval/taniteval/results/`, and repo-wide by name). The hub directory is not a copy — **it is
the only source**, and it holds more than the dead citation claimed:

```
TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/2026-07-22-refc-small-30k/
  refc-small-30k.json                          ← the raw eval output
  scaleab_refc-small-30k_vs_refc-base-30k.json
  scaleab_refc-small-30k_vs_refc-xl-30k.json   ← the brace form `{base,xl}` was never a real path
  windows_refc-small-30k.pt   fan_refc-small-30k.pt
  refc_anchors_small64.pt     provenance.json  eval_registry_after.py
```

⇒ **RULE: a shell brace expansion is not a citation.** `…_vs_refc-{base,xl}-30k.json` names no file
and cannot be checked by anything that reads the registry literally — which is every reader and
every script. Write both paths.
⇒ **RULE: never label the only copy of an artifact a "copy".** It invites deletion of the thing the
number rests on, and it sends the next reader to a path that does not exist.

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
| **Note** | `TanitAD Research Lab/Benchmarks & Eval/Research/2026-07-20-refc-medium-scaling.md` (pre-registered the reading rule) |
| **HF** | ✅ **`Sayood/tanitad-refc-base`** — public + **gated `manual`**, pushed 2026-07-25. Files: `ckpt.pt` **1,250,838,325 B** (md5 `8f10d6f934f4199e11ddc7352e074939`, re-verified immediately before upload), `config.json`, `metrics.json`, model card. An anonymous `HEAD` of `https://huggingface.co/Sayood/tanitad-refc-base/resolve/main/ckpt.pt` returns **401 `GatedRepo`** *(re-verified by me 2026-08-03, unauthenticated request, `X-Error-Code: GatedRepo`)*. Now 3 copies (HF + eval pod + pod3). Note: `…/incoming/2026-07-25-refc-hf-push/NOTE.md` |

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
> for the deltas. Reproduce from `taniteval/results/windows_refc-base-30k.pt`, `taniteval/results/windows_refc-xl-30k.pt`, `taniteval/results/fan_refc-base-30k.pt` and `taniteval/results/fan_refc-xl-30k.pt` (four files, all verified present 2026-08-03) with
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
(+ `REFC_suite_base_results.json` and `REFC_suite_xl_results.json`, both verified present 2026-08-03), open-loop control `REFC_openloop_diagnostic.json`, flagship
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

> ⛔ **ESTIMATOR — CORRECTED 2026-08-17. Every number this section published on 2026-07-19 was
> `overlapping_holdout_se`, the BANNED estimator, and it was DECIDING both gates.** The artifact says so
> in its own protocol field: `planner_p2_flagship-30k.json` → `protocol.ci = "8-split episode jackknife"`,
> which is the mislabel `taniteval/ci.py:5-27` documents as *neither a jackknife nor a valid SE*.
> ✅ **NEITHER `G1_pass` NOR `G4_pass` FLIPS** — re-decided CPU-only on banked per-window data
> (`…/incoming/2026-08-16-jack-in-gates/JACK_IN_GATES.md`, raw `raw/g1_g4_both_estimators.json`).
> **Both values are printed below. The superseded number is never deleted.** MEASURED.
>
> **What the correction moves even though no verdict does** — carry these with the numbers, they are not
> footnotes: point estimates shift **−6.9 % to +6.8 %, bidirectional *within this single artifact***
> (head −6.9 %, operative +5.9 %); intervals were **1.17×–2.17× too narrow**; the **divergence rate — the
> safety-shaped number — was overstated by +20.3 %** (8.7 % → **7.2 %**); and the **G4 threshold itself
> was a legacy `heldout` mean, 2.69 % LOW** (1.6852 vs **1.7318**) — *i.e. the old gate was HARDER than
> the honest one, so the correction only strengthens the PASS.* The banned statistic also gave **7 of the
> 40 val episodes weight exactly 0** (ids 1, 9, 22, 23, 27, 28, 34) — a **wrong-population** defect
> (class **C73**), not a precision one.

| Gate | Metric | Planner — **decision-grade** [episode-cluster bootstrap CI95] | Baseline — **decision-grade** | Verdict | *superseded `± overlapping_holdout_se` (BANNED)* |
|---|---|---|---|---|---|
| **G1** open-loop, **881** windows / 40 eps | ADE@2s | ⚠️ **PENDING — arm not banked per-window** (see below) | tactical head **3.3839** [2.8336, 3.9722] | **PASS** — ⚠️ **PARTIAL re-decision**; sign and separation confirmed, magnitude pending | *planner 0.893 ± 0.114 · head 3.150 ± 0.347 · Δ +2.257 ± 0.329, 72 % reduction* |
| **G4** closed-loop, 221 windows / 20 eps | ADE@2s | **0.9799** [0.7456, 1.2312] | v1 head **1.7318** (881 win/40 ep) | **PASS**, and `hi` **1.2312 < 1.7318** ⇒ **CI-separated** | *1.038 ± 0.202 vs 1.685 ± 0.098, "38 % less drift"* |
| **G4 — PAIRED** ⭐ *(NEW 2026-08-16, first ever computed)* | ADE@2s δ | **−0.7375** [−0.9362, −0.5295], **p(δ>0) = 0.0000** | head **1.7174** on the *same* 221 windows | **PASS — the strongest form of this claim the programme has** ⇒ **42.9 % less closed-loop drift** | *−0.6873 ± 0.2191 (banned paired)* |
| | FDE@2s | **2.0583** [1.5463, 2.6134] | **3.6190** [3.2453, 4.0215] ⚠️ unpaired, 881/40 | 43.1 % closer ⚠️ *scope-mismatched* | *2.194 ± 0.455 vs 3.530* |
| | divergence >5 m | **7.24 %** [2.25 %, 14.09 %] | **23.50 %** [16.80 %, 30.27 %] ⚠️ unpaired, 881/40 | **3.2× fewer** ⚠️ *scope-mismatched* | *8.7 % ± 4.6 vs 22.2 %* |

⭐ **The PAIRED G4 row is strictly stronger than the two-interval claim it replaces.** The published G4
compared a 221-window/20-episode planner against an 881-window/40-episode baseline and rested its
"CI-separated" claim on **two independent intervals, both banned**. The planner's windows are the
**stride-16 subset** of the baseline's stride-8 windows on the same 20 episodes (verified by GT waypoint
equality to `atol=1e-5`, window-for-window, for all 221), which made a paired test possible for the first
time. It agrees with the verdict it replaces — at `p(δ>0) = 0.0000`.

⚠️ **All THREE baseline closed-loop numbers were the banned estimator, not just the threshold** — MEASURED
2026-08-17 from `closedloop_flagship-30k.CORRECTED.json`: ADE 1.6852 → **1.7318** (−2.69 %), FDE 3.5296 →
**3.6190** (−2.47 %), divergence 0.2216 → **0.2350** (−5.70 %); CI widening **1.722× / 1.523× / 1.564×**.
**All three moved the same way — the v1 head baseline is WORSE than published**, so every P2-vs-head
margin in this section widens under correction. The registry previously carried only the threshold
correction; the FDE and divergence baselines are corrected here for the first time.

⚠️ **`n` was also wrong: G1 is 881 windows, not 880.** `planner_p2_flagship-30k.json` →
`open_loop.n_windows = 881`. Corrected here.

⚠️ **UNSEEDED CEM — this is a property of every P2 number above, not a note about the code.**
`planner_p2.py` draws `torch.randn` with **no seed**, so no P2 figure is bit-reproducible and each one
carries a **sampling component that has been measured but never bounded** (drift **0.019 %** between the
2026-07-19 published run and the 2026-07-26 re-drive, `planner_p2_G4.CORRECTED.json`). *Measured is not
bounded*: 0.019 % is one observation of the drift, not a bound on it. Any re-drive of these rows must
seed first, or state that it did not.

⚠️ **G1's re-decision is PARTIAL, and the cell is marked partial rather than clean.** `collect_openloop`'s
`plan_wp` — the open-loop CEM arm — **was never dumped per-window** (probed at three locations). What can
be said rigorously: the corrected head is **3.3839**, so for G1 to flip the planner's corrected mean would
have to reach **≥ 3.3839**, i.e. the banned estimator would have to have been wrong on that one arm by
**−73.6 %**, against a MEASURED envelope on this exact window set and split structure of **−6.9 % to
+5.9 %** (and a programme-wide 27-arm envelope of −6.67 % to +11.69 %). **A flip needs an error ~11×
larger than anything ever measured for this estimator.** G1 does not flip. Closing it properly costs
**~400 s of GPU** (`wall_s = 400.4` in the artifact) re-running `collect_openloop` **with the `plan_wp`
dump this time**; `…/2026-08-16-jack-in-gates/code/recompute_g1_g4.py` needs no changes.

> ⛔ **A THIRD VERDICT ON THIS PATH WAS NEVER RE-DECIDED, AND UNLIKE G1/G4 ITS FLIP IS REACHABLE.**
> MEASURED 2026-08-17 (this pass, not inherited): the published artifact carries **five** boolean
> verdicts, not two — `G1_pass`, `G1…separated`, `G4_pass`, **`planner_beats_cv = False`**, and
> `weight_sensitivity.beats_head_all = True`. The jack-in-gates re-decision covered the first three.
> **`planner_beats_cv` is banned on BOTH sides** (planner 0.8929 vs CV 0.8248) and its corrected CV floor
> is **0.8377** — *higher*, which moves the comparison toward the planner. For the verdict to flip to
> "beats CV" the planner's corrected mean must fall below **0.8377**, i.e. the banned estimator must have
> overstated it by **+6.59 %** — against a measured local upper edge of **+5.877 %** and a programme-wide
> upper edge of **+11.69 %**. ⇒ **This one is genuinely UNDECIDED, not "no flip"**, and the same ~400 s
> GPU re-drive settles it. Do not quote `planner_beats_cv` in either direction until then.
> *(`beats_head_all` is **not** a banned-estimator verdict — it compares raw point values, 0.6476 vs
> 3.1342 — but see the scope stamp on the Robustness line below.)*

> ## ⭐ **UPDATE 2026-08-18 (C101): the QUESTION `planner_beats_cv` stood for is now ANSWERED — against the planner — at the PRIMARY tier, from banked data and with NO GPU.**
>
> ⚠️ **First, a scope correction to the block above:** `planner_beats_cv` is computed inside
> **`analyze_openloop`** (`planner_p2.py:621`, fn at `:555`) from `collect_openloop`'s
> `plan_wp`/`cv_wp` — **OPEN LOOP, 881 windows / 40 episodes / stride 8.** The banked
> `p2win_flagship-30k.pt` is the **CLOSED-loop** collection (221 win / 20 ep / stride 16). **Different
> tier, different windows, different episode count** ⇒ the banked path cannot reach that verdict, and
> **no open-loop CEM planner arm is banked anywhere** (confirmed at three probes including an
> exhaustive walk of every `.pt` in the repo — independently reproducing `JACK_IN_GATES.md` §3.1).
>
> ⛔ **But the published G4 compared planner vs HEAD, never vs CV — so "does the planner beat CV?" had
> never actually been asked.** Computed from banked data, **paired**, **[TIER T1 — PRIMARY]**:
>
> | | value |
> |---|---|
> | **planner − CV** | **+0.2585 m [+0.0869, +0.4309]**, CI-separated, **p(δ>0) = 0.9975** |
> | ⇒ | **the CEM planner is 35.8 % WORSE than constant velocity, closed-loop** |
> | **operative under TRUE actions − CV** | **−0.3151 m** ⇒ the WM rolls out *better* than CV when handed true actions |
>
> ⇒ ⭐⭐ **THE LOSS IS IN THE ACTION SEARCH, NOT IN THE WORLD MODEL.** The CEM cannot find actions that
> exploit a predictor that demonstrably works.
>
> ⛔ **AND IT LOSES ON THE FAMILY IT IS DESIGNED FOR.** Per family, never pooled:
> **LONGITUDINAL 1.9062 vs CV 1.6705 m**, speed error **0.9431 vs 0.7607 m/s**, bias **+0.2737 vs
> −0.0995 m/s**. The *lateral* loss its own scope note predicts; **the longitudinal loss it does not.**
> **TACTICAL** and **STRATEGIC** are genuine **N/A with reasons**: the CEM emits no manoeuvre class,
> and the cost carries **no route/goal term**. Distance-keeping/TTC uncomputable — no lead-agent track.
>
> ⚠️ **The OPEN-LOOP verdict remains UNDECIDED and still needs the re-drive** — the flip requirement
> above (−6.589 % against a −6.909 %…+5.877 % local envelope) is reproduced to 4 dp, and **no bound
> closes it**: the banned estimator gives **7 of 40 episodes weight exactly 0**, leaving those windows
> unconstrained by the published mean. ⏳ Its only missing input is the **4.70 GB val cache** — a **PI
> decision**, not an agent's.
> ⭐ **The CEM is now SEEDED** (`cem_seed`, default 0, `fa4b3d1`, 9 pinning tests), so the re-drive is
> reproducible; the previously "unbounded sampling component" is measured at **0.0193 % drift** on the
> closed-loop reproduction gate, with `cv` and `open_grnd` **bit-exact**.
> ⚠️ **Verdict inventory correction:** the artifact holds **14 boolean instances across 6 distinct
> names**, not five — the block above collapsed the 9 `beats_head` grid entries.
> Source: `…/Benchmarks & Eval/Implementation/incoming/2026-08-18-planner-beats-cv-redrive/`.

Reference points on the same pass: CV **0.8377** [0.6234, 1.0716] *(legacy 0.825)* · operative rollout with
**true** actions (the WM ceiling) **0.4271** [0.3675, 0.4871] open-loop *(legacy 0.452)* / **0.4063**
[0.3293, 0.4907] closed-loop *(legacy 0.424)*.

**Strata:** straight (634 windows, 72 %) planner **0.564** vs true-action 0.393 vs CV 0.439 vs head 3.297.
Curved (top-10 % curvature, 89) planner **2.114** vs true-action 0.484 vs CV 2.426 vs head 3.344.
✅ **These are NOT banned-estimator numbers** — MEASURED 2026-08-17 from the artifact: the stratum entries
carry no `ci95` and no estimator field, i.e. they are plain `.mean()` over the stratum, full-set by
construction on their own subset (same situation as the `bench.py` trivial bars in §6). They are
comparable to each other but **not** to the split-mean headline figures they sit beside — which is why
the stratum head (3.297 / 3.344) does not equal the headline head (3.150).

**The honest signature:** long-RMSE 1.41 / lat-RMSE 1.97 → **only 34 % of the 2 s squared error is
longitudinal; 66 % is lateral.** Speed-decoupled cross-track 0.445 m; speed bias +0.47 m/s. The planner
tracks its own minted `v_target` to **1.03 m/s** — *better than the GT log tracks it (1.54 m/s)*.
This is mechanism, not surprise: **the P2 cost has no lateral/route/goal term** (that is P3). The lateral
residual *is the measurement of what P3/P4 must add.*

**Robustness:** a 3×3 sweep of `w_c ∈ {0.05,0.1,0.2} × w_p ∈ {0.01,0.02,0.04}` (a 4× band) moves planner
ADE only **0.647 → 0.669 m (3.4 %)** and beats the head in **all 9** configs. G1 is not a tuning artifact.
⚠️ **SCOPE STAMP, added 2026-08-17 — this sweep is on an 8-EPISODE SUBSET, not the 40-episode val set**
(`weight_sensitivity.note`: *"8-ep subset; planner ADE@2s vs w_c,w_p; weights NOT selected on GT ADE"*),
which is why its head reads **3.1342** rather than the headline 3.1501. ✅ Its `beats_head_all` verdict is
**not** a banned-estimator verdict — it compares raw point values across a 4.8× gap — so it stands as
written. But the sweep was published without its `n` for three weeks; quote it as *"8 episodes"* or not
at all.

> 🟥 **RECONSTRUCTION RISK — P2 is uncommitted.** `planner_p2.py` exists only on `tanitad-eval`. It is the
> single strongest piece of evidence for the v3 direction and it is one pod-loss away from gone.

---

## 6. Cross-arm leaderboard — identical harness, identical 881 windows

> ⚠️ **ESTIMATOR DEFECT — CORRECTED 2026-07-20. The `± CI95` column below is NOT decision-grade.**
> This block was historically labelled *"8-split episode-disjoint jackknife"*. It is **neither a
> jackknife nor a valid SE**: `bench.py` draws 8 **independent random 20 % holdouts** from the same 40
> episodes and takes `1.96·std/√8` over overlapping estimates — Monte-Carlo CV, measuring
> **split-selection noise**, not model uncertainty. Measured **1.107–3.100× too narrow, median 1.499×**
> across **27 dumps = 25 distinct arms** *(two double-banked pairs — C126,
> `taniteval/results/dump_exclusions.json`)* — MEASURED 2026-07-25,
> `TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/2026-07-25-jack-blast-radius/jack_recompute.json`.
> *(The older **1.28–2.06×, median 1.51×** band was never wrong, only **under-sampled at 10 arms**: all
> 10 reproduce bit-for-bit against `Project Steering/CI_RECOMPUTE_2026-07-20.json`.)* Coverage
> simulation: naive **62.3 %** vs cluster-bootstrap **93.8 %** (target 93–97 %).
> **The `mean` column is ALSO defective, and that is the newer and larger finding.** It is a
> mean-of-split-means, so besides **compressing between-arm gaps** (rows 1–2: 0.006 m here vs
> **0.0443 m** on the full set) it **shifts the single-arm point estimate by −6.67 % to +11.69 %,
> bidirectionally — 11 inflated / 14 deflated over the 25 distinct arms (11/16 over dumps — C126), none flat.** No legacy point estimate may be
> assumed conservative, and no ranking may be read off two split-means.
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

**RE-EMITTED 2026-07-26 under the corrected estimator** (from the 2026-07-25 blast-radius recompute).
The primary ADE column is now the **full-set
mean over all 881 windows** with its **episode-cluster bootstrap** CI95 (`taniteval/ci.py`, B = 2000,
resampling unit = the val episode); every ordering claim comes from the **paired** bootstrap on the same
windows, never from differencing two column values. The final column retains the superseded figure,
explicitly labelled **`legacy_split_mean ± overlapping_holdout_se` (DEPRECATED)** — kept rather than
deleted so every previously published number stays traceable. Sources: the per-row inline
`src:` drift pointers below (machine-checked by `tools/registry_lint.py`), cross-validated against
`…/incoming/2026-07-25-jack-blast-radius/jack_recompute.json` (27 dumps = 25 distinct arms recomputed from the raw
`windows_*.pt` dumps, dev-box CPU, no GPU). MEASURED.

> ⚠️ **THE ROW ORDER CHANGED — the rows were NOT shuffled, the ranking moved.** Recomputing all 27
> in-repo arms moves **10 of 27 positions** (dump-level ranking; 27 dumps = 25 distinct arms — C126) in the cross-arm ranking. Three of those order changes land
> inside *this* table, and each is a finding rather than a re-render:
> 1. **REF-C-XL and REF-C-base SWAP.** Legacy: base 0.4523 ahead of XL 0.4577. Full-set: **XL 0.4714
>    ahead of base 0.4728.** The paired delta **flips sign** — legacy `+0.0054` → true **−0.0013,
>    CI95 [−0.0316, +0.0281]**. Neither is separated, so **the 1= three-way tie stands**; what moved is
>    the sign of a non-significant gap, which is exactly why it must not be read as an ordering.
> 2. **`refb-10k` crosses the CV floor.** Legacy 0.8255 vs CV 0.8248 → ✗. Full-set 0.8372 vs CV 0.8377
>    → ✅. The paired bootstrap does **not** separate them, so the honest verdict is **TIE** — neither
>    the ✗ this table used to print nor the ✅ `LEADERBOARD.md` prints. The arm sits *between* the two
>    circulating CV floors (0.8248 split-mean / 0.8377 full-set), which is the whole reason the two
>    documents have been contradicting each other on this one row.
> 3. **Flagship v3enc is no longer "🟥 not evaluated"** — `driving_flagship-v3enc-10k.json` is in-repo
>    and gives 1.9654 [1.6556, 2.2859]. That row contradicted §1.4's own RESTART verdict; it now ranks,
>    which pushes REF-A/no-speed/v2 down one slot each. Ranks were renumbered accordingly (old 3–12 →
>    new 5–15); the `Beats CV` column is unchanged for every arm except `refb-10k`.
>
> ⚠️ **The three trivial bars are deliberately NOT re-emitted — they were never split-means.** MEASURED
> from the emitter: `bench.py:485-511` (`kinematic_floor` → `best_of_3_ade_0_2s`) and `bench.py:558`
> (`ctrv_ade`) both take a plain `.mean()` over all 881 windows. **Consequence, and it is not small:
> every legacy "clears the floor" verdict in this program compared a *split-mean* model number against a
> *full-set* floor** — an estimator mismatch, not a like-for-like test. Re-checked like-for-like on the
> full set: v1 (0.4271), REF-C-XL (0.4714) and REF-C-base (0.4728) still clear all three bars, but
> **REF-C-small no longer clears the CTRV oracle** (0.5261 vs 0.523; its legacy 0.5007 did). The CV row
> is the one floor that has both forms, and both are printed.

> ⛔ **DEPLOYMENT, stamped 2026-07-26: this table is the 40-EPISODE / 881-WINDOW val deployment, every
> row.** A second deployment now exists (600 episodes / 13,198 windows, §1.2a) and it is a **different and
> EASIER corpus** — CV floor 0.8377 → 0.6917. **Do not substitute a 600-episode number into this table**
> (v1 reads 0.4108 there, which is not an improvement on 0.4271, it is a different measurement), and do
> not compare any row here against any number measured on 600. n is **881 windows / 40 episode clusters**
> for every ranked row.

| Rank | Arm | TanitEval key | Step | Params | **ADE@2s — full-set mean [episode-cluster bootstrap CI95]** | FDE@2s | miss@2m | Beats CV | `legacy_split_mean ± overlapping_holdout_se` (DEPRECATED) |
|---:|---|---|---:|---:|---|---:|---:|:--:|---|
| **1=** | **Flagship v1 (speed+jerk) FINAL** | `flagship-30k` | 29 999 | 263.4 M | **0.4271** [0.3675, 0.4871] <!-- src: taniteval/results/driving_flagship-30k.json#headline.ade_0_2s.mean --> | 0.9075 | 0.0454 | ✅ sep | *0.4522 ± 0.0312* |
| **1=** | **REF-C-XL** (anchored diffusion) **FINAL** | `refc-xl-30k` | 29 999 | 251.9 M | **0.4714** [0.3896, 0.5556] <!-- src: taniteval/results/driving_refc-xl-30k.json#headline.ade_0_2s.mean --> | 1.0061 | 0.1419 | ✅ sep | *0.4577 ± 0.0572* |
| **1=** | **REF-C-base** (anchored diffusion) **FINAL** | `refc-base-30k` | 29 999 | **104.2 M** | **0.4728** [0.3835, 0.5699] <!-- src: taniteval/results/driving_refc-base-30k.json#headline.ade_0_2s.mean --> | 1.0031 | 0.1419 | ✅ sep | *0.4523 ± 0.0497* |
| 4 | **REF-C-small** (anchored diffusion) FINAL — SEPARATED 3rd rung of the REF-C ladder (§4.2) | `refc-small-30k` | 29 999 | **54.7 M** | **0.5261** [0.4295, 0.6262] | 1.1115 | 0.1714 | ✅ sep | *0.5007 ± 0.0671* |
| — | *best-of-3 kinematic floor* — full-set by construction (`bench.py:511`) | — | — | — | *0.5005* | — | — | — | *n/a — never went through the estimator* |
| — | *CTRV oracle* — full-set by construction (`bench.py:558`) | — | — | — | *0.523* | — | — | — | *n/a* |
| — | *no-vision ego-status ceiling* — full-set by construction | — | — | — | *0.5735* | — | — | — | *n/a* |
| 5 | **REF-B v2** (arch-v2) FINAL | `refb-v2-30k` | 29 999 | 271.6 M | **0.5913** [0.4766, 0.7131] <!-- src: taniteval/results/driving_refb-v2-30k.json#headline.ade_0_2s.mean --> | 1.2434 | 0.2066 | ✅ sep | *0.5921 ± 0.0685* |
| 6 | Flagship v1, 19 k relay | `flagship-speed` | 19 000 | 263.4 M | 0.6152 [0.5422, 0.6951] <!-- src: taniteval/results/driving_flagship-speed.json#headline.ade_0_2s.mean --> | 1.3168 | 0.1669 | ✅ sep | *0.6277 ± 0.0551* |
| 7 | REF-B v2 @20 k milestone | `refb-v2-20k` | 20 000 | 271.6 M | 0.6435 [0.5410, 0.7516] <!-- src: taniteval/results/driving_refb-v2-20k.json#headline.ade_0_2s.mean --> | 1.3218 | 0.2157 | ✅ sep | *0.6462 ± 0.0548* |
| 8 | REF-B speed | `refb-10k` | 10 000 | 262.8 M | 0.8372 [0.6753, 1.0218] <!-- src: taniteval/results/driving_refb-10k.json#headline.ade_0_2s.mean --> | 1.6964 | 0.2679 | ⚠️ **TIE** (flip) | *0.8255 ± 0.0992* |
| — | **Constant velocity (the floor)** | — | — | 0 | **0.8377** [0.6234, 1.0716] <!-- src: taniteval/results/driving_flagship-30k.json#floor_values.cv.ade_0_2s.value --> | 1.7406 | 0.3042 | — | *0.8248* |
| 9 | REF-B v1 | `refb` | 6 000 | 262.5 M | 0.8629 [0.6928, 1.0385] <!-- src: taniteval/results/driving_refb.json#headline.ade_0_2s.mean --> | 1.7351 | 0.3178 | ✗ | *0.8682 ± 0.0817* |
| 10 | **P2 CEM planner** over frozen v1 | `planner_p2` | (n/a) | 0 trained | ⛔ **THIS CELL WAS WRONG — CORRECTED 2026-08-16.** It read *"NOT RECOMPUTABLE — no raw JSON and no `windows_*.pt` in the repo"*. **Both are in-repo, at depth 6–8**, and the recompute is **CPU-only**. That stale absence claim is why this gate sat un-re-decided for **21 days** (classes **C69** — an absence asserted from a depth-bounded search — and **C70** — a blocker never revisited). ✅ **NOW MIGRATED AND RE-DECIDED:** `planner_p2.py` uses `episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`; the legacy value is kept beside it under `legacy_overlapping_holdout_se` with its `ci_width_ratio` and `point_estimate_shift_pct`. **NEITHER G1 NOR G4 FLIPS** (details in `…/incoming/2026-08-16-jack-in-gates/JACK_IN_GATES.md`). ⚠️ **THE ADE CELL STAYS BLANK ON PURPOSE — the open-loop CEM arm was never dumped per-window**, so this arm has *no* decision-grade point estimate; §5 states the flip requirement (−73.6 % vs a measured −6.9 %/+5.9 % envelope) and the ~400 s GPU that closes it. ⛔ **The `✗ Beats CV` in this row is ITSELF an un-re-decided banned verdict** (`planner_beats_cv`, banned on both sides) and its flip IS reachable — see §5. Treat it as **UNDECIDED**, not ✗. Every P2 number also carries an **unbounded unseeded-CEM sampling component** | — | — | ⚠️ **undecided** | *0.893 ± 0.114* |
| 11 | Flagship **v3enc** (RESTART, §1.4) | `flagship-v3enc-10k` | 10 000 | 272.9 M | 1.9654 [1.6556, 2.2859] <!-- src: taniteval/results/driving_flagship-v3enc-10k.json#headline.ade_0_2s.mean --> | 3.6084 | 0.6901 | ✗ | *2.1072 ± 0.2020* |
| 12 | REF-A DINOv2 4B | `refa-dinov2` | 29 999 | — | 2.1675 [1.9081, 2.4212] <!-- src: taniteval/results/driving_refa-dinov2.json#headline.ade_0_2s.mean --> | 3.2803 | 0.6129 | ✗ | *2.1322 ± 0.1821* |
| 13 | Flagship **no-speed** (ablation control) | `flagship-nospeed` | ~22 000 | 263.4 M | 3.0175 [2.5450, 3.5444] <!-- src: taniteval/results/driving_flagship-nospeed.json#headline.ade_0_2s.mean --> | 5.0282 | 0.7423 | ✗ | *2.9176 ± 0.3558* |
| 14 | REF-A dyn-in 4B | `refa-dynin-30k` | 29 999 | — | 3.0471 [2.4984, 3.6878] <!-- src: taniteval/results/driving_refa-dynin-30k.json#headline.ade_0_2s.mean --> | 4.7642 | 0.7412 | ✗ | *2.9196 ± 0.3937* |
| 15 | Flagship **v2** (killed) | `flagship-v2-6k` | 6 000 | 272.9 M | 5.9396 [4.3273, 7.6249] <!-- src: taniteval/results/driving_flagship-v2-6k.json#headline.ade_0_2s.mean --> | 12.4011 | 0.8524 | ✗ | *6.179 ± 1.2845* |
| — | Flagship v1 tactical **head** (not rollout) | `plan_flagship-30k` | 29 999 | — | **3.3839** [2.8336, 3.9722] <!-- src: TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json#G1.arms.tactical_head.corrected_mean --> ⚠️ **the "🟥 no windows dump — legacy only" that stood here is REFUTED (2026-08-16):** `clwin_flagship-30k.pt`'s `plan_direct` **is** this arm — it reproduces the legacy 3.1501 ± 0.3472 **bit-exactly at 4 dp**, and its `full_set` mean is 3.3839. Same stale-absence class (**C69**/**C70**) as the P2 row above | — | — | ✗ | *3.38 (3.150 ± 0.347 in the P2 pass)* |

*Arms recomputed but not ranked here (same 881 windows; full table in `jack_recompute.json`):*
**v1.6** `flagship-v16-ab-ft` 0.4375 [0.3423, 0.5501] (legacy 0.4886 — the largest single-arm bias in the
program, **+11.69 %**; paired-TIED with v1, §1.4b) · **v4.1-10k** 0.8522 [0.7468, 0.9800] (legacy 0.8707)
· **v4.2-step4000** 0.9869 [0.8795, 1.1088] (legacy 1.0490) · the REF-C v1.2 family and the REF-A
milestone ladder. **Every one of these was previously published as a split-mean.**

**Two readings that matter:**
1. The **trained-encoder** arms occupy every slot above CV. The **frozen-encoder** arms occupy slots **12
   and 14** (slots 9 and 11 before the 2026-07-25 re-emission renumbered the table). That is H4, in one
   table — and it **survives the estimator correction**: REF-A's gap to v1 is a paired **+2.6200 m
   [+2.0945, +3.2570]**, separated by ~40× the largest measured bias.
2. The flagship's supervised **tactical head** (**3.3839** [2.8336, 3.9722]) is *worse than CV*
   (**0.8377**), while the same model's operative rollout is **0.4271 m** [0.3675, 0.4871]. **The head is
   a lossy readout of a good world model** — which is exactly what P2 exploits and what v3 is built on.
   ✅ **UPGRADED 2026-08-16/17 — this reading is now decision-grade on BOTH sides.** The caveat that stood
   here (*"both sides are still legacy statistics … `plan_flagship-30k` has no windows dump, and P2 is
   un-recomputable"*) was **wrong on both counts**: the head's windows are in `clwin_flagship-30k.pt`
   (`plan_direct`) and the rollout's are in `taniteval/results/windows_flagship-30k.pt`. The head/rollout
   ratio is **7.92×** on full-set means (legacy 6.96×) — the correction *widens* it. ⚠️ The one thing here
   that remains un-recomputable is the **P2 planner arm**, which this reading does not depend on.
3. **Ranks 1–2 tie on accuracy, so latency is the tiebreaker — and it is not close.** Measured
   2026-07-20 on one A40, batch 1, identical precision flags. ⛔ **UNRESOLVED SOURCE (2026-08-03).** This reading cited a wildcard, which names no file — and the figures below are **not in any committed artifact**, so it cannot simply be repointed. MEASURED 2026-08-03, `plan_step.p50_ms` in the committed files: `taniteval/results/eff_flagship-30k.json` → **97.32 / 97.70 / 123.83** and `taniteval/results/eff_refc-xl-30k.json` → **44.06 / 27.78 / 21.00** (fp32/tf32/amp16). ⇒ **neither** triple below matches its own artifact, which extends **R14** (that row had REF-C's fp32/tf32 as agreeing; amp16 is 26.12 vs **21.00**). ⛔ **Do not re-cite these six figures** — re-measure on an idle A40, or restate from the committed JSONs. The ranking they support is unaffected:
   flagship planning tick **103.42 / 93.76 / 104.49 ms** (fp32/tf32/amp16) vs REF-C **44.28 / 27.84 /
   26.12 ms** — **2.3–4.0× faster**, and REF-C **meets the 10 Hz budget at p99 in all three precisions
   while the flagship misses in all three**. REF-C does **1.75× the FLOPs** (702 vs 402 GFLOPs) and is
   still faster because it achieves **15.9–25.2 TFLOPs vs 3.7–4.3**: the flagship's 20-step sequential
   rollout is **launch/serialisation-bound**, not arithmetic-bound. ⚠️ Direction reverses for *batched*
   throughput (flagship 34.8 vs REF-C 29.9 windows/s @ batch 32) — REF-C wins latency, the flagship wins
   bulk eval. See §1.2 for the two-tick definition and the retracted "11.16 ms" framing. ⚠️ These three
   latency figures conflict with the committed `taniteval/results/eff_flagship-30k.json` and `taniteval/results/eff_refc-xl-30k.json` (and so does the REF-C row — see the UNRESOLVED-SOURCE note at reading 3) — see **R14**.
4. **The 1= tie is not a tie on driving — and latency is no longer the only separator.** MEASURED
   2026-07-21, TanitEval v2 tier-0 over these same 881 windows (`taniteval/results/driving_refc-xl-30k.json`, `taniteval/results/driving_refc-base-30k.json` and `taniteval/results/driving_flagship-30k.json`, key `vs_floor_paired.cv.long_abs_2s_m` — all three re-read 2026-08-03, exact, `estimator paired_episode_cluster_bootstrap`;
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
| R8 | **REF-A I-JEPA canonical-val result is leak-contaminated** (80 % of val in its train set) | 🟡 medium | flagged in `taniteval/registry.py` | 🟥 **REMEDIATION CORRECTED 2026-07-28 — this cell used to read "re-evaluate on the `f1b378` val", which prescribed the 77.5 %-LEAKED split as the cure for a leak.** Re-evaluate on the **content-verified clean** `physicalai-val-0c5f7dac3b11` (0/40 and 0/600 by sha256 of raw `poses` and `frames_u8`, 2026-07-28). ✅ **ALL FOUR SITES CLOSED 2026-07-28** (`…/incoming/2026-07-28-leaky-cache-audit/`): the **source** `taniteval/registry.py:85-97` now names `0c5f7dac3b11` and records why (taniteval 773 passed); `Paper/TANITAD_PAPER.md` §7.2 corrected — ⚠️ **that paper contradicted itself, since its own §5 already recorded the 78 % overlap AND that the harness "refuses it in code"**; `DOC_CORRECTION_SWEEP.md` corrected with its superseded wording preserved. **§2.2 is deliberately NOT reworded** — it *quotes* the defective source, so the quote stands and carries an inline ⛔. ⇒ **One `note=` string propagated into four documents because each inherited the phrase instead of checking the artifact.** |
| R9 | **REF-B rev2, `refa-4brain-speedyaw-30k`** have no eval record | 🟢 low | n/a | either evaluate or mark explicitly superseded |
| R10 | **`tanitad-pod4:/workspace/rescue/experiments/refb-speed-30k/ckpt_prepatch_step8500.pt` is misnamed** — it is step 10,000 and byte-identical to `ckpt.pt` | 🟢 low | ⚠️ **host corrected 2026-08-03: NOT "pod1".** The originating pod (`tanitad-pod`) refuses connections; the only reachable copy is `tanitad-pod4:/workspace/rescue/experiments/refb-speed-30k/` (MEASURED, read-only `ls`, both files 3 153 889 214 B) | rename — ⛔ **but not while pod4 is training**; and this arm has **no HF copy**, so the rename must not be a move that risks the only copy |
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
| **⭐ D-A5** | 07-17→19 | **REF-A accepted as a frozen-encoder REFERENCE — H4 closes negative** | The ceiling is **capability, not overfitting**: the milestone curve is monotonically *improving* (5 k 3.755 → 15 k 3.694 → 20 k 3.016 → 30 k **2.920**, best is last). Held-out error never rises. Every remedy was tried — speed input, yaw input, dyn-input `[v0,yr0]`, ego-dropout, temporal adapter, 4 brains, I-JEPA features — and it still plateaus above CV | dyn-in **3.0471** [2.4984, 3.6878] vs flagship **0.4271** [0.3675, 0.4871] (full-set, decision-grade); paired win-rate 95.9 %, Δ **+2.62 m [+2.0945, +3.2570]** (paired episode-cluster bootstrap, §6); train fwd-ADE 0.65 → held-out 2.92 (4.5× gap); pre-fix ablation `vision_use` 3.4 % → "a dynamics integrator". ⛔ **ESTIMATOR CORRECTED 2026-08-17** — this row published *2.9196 ± 0.394 vs 0.4522* and *CI [2.447, 2.798]*, all `overlapping_holdout_se`; the paired interval was **3.31× too narrow**. ✅ **The H4 verdict is untouched and in fact strengthens** (the gap widens 2.467 → 2.620 m on full-set means). *Superseded, kept visible: 2.9196 ± 0.394 · 0.4522 · [2.447, 2.798].* | A clean publishable negative. Motivates the trained encoder. Ends the REF-A retrain line; anchored-decoder retrain dropped |
| **D-030** | 07-18 | **REF-C redesigned** to a DiffusionDrive anchored truncated-diffusion decoder + a 3-size scaling study (55/104/252 M) | The 2022-era tiny-TCP GRU was not a fair modern reference. Anchored multimodal decoding is the published standard and directly tests H19 (maneuver→anchor graft). FPS not k-means because ~74 % of the data is straight | REF-C-XL @16 k = **0.5645** — 2nd of the trained-encoder arms, beats CV in all three speed terciles | `6025769`, `36d979f`, `7e9c402`. ⚠️ **The scaling study itself was never run — only XL exists (§4.2, OPEN)** |
| **D-032** | 07-18 | **Milestone checkpoints at 5 k/15 k/20 k/30 k** instead of overwriting `ckpt.pt` | No earlier checkpoint survived for re-gating, overfitting curves, or lineage forensics | The REF-A overfitting-vs-ceiling verdict (D-A5) is **only possible because of this** | `b298cef`, `6808c2d`. Costs disk — and pod2's 98 %-full overlay is what killed v3enc's first attempt |
| **D-A6** | 07-17 | **flagship-v2: ten levers at once** (six named in the directive, ten set) | The 30 k flagship had three named weaknesses: high-speed longitudinal overshoot, a command-echo strategic head (`route_skill_vs_chance = 0.0`), and an encoder redundantly re-encoding ego dynamics (`vision_use` ~12 %). Each lever targets one | Every lever individually motivated by an H25/H26 measurement | `f583bb4`, `b8d3fc8`; run `flagship4b-v2-30k` |
| **⭐ D-031** | **07-19** | **Kill flagship-v2 at 7.8 k; do not grind it to 30 k** | The 6 k number (6.18 m) alone would *not* justify killing — it was correctly diagnosed as **mechanism-A**: the levers removed the kinematic speed shortcut **by design** (encoder speed-probe R² 0.30 vs v1's 0.86). **The decisive read was the rate of learning, not the level:** the same-step v2/v1 forward-consistency ratio *widened* 1.51 → 4.33; the power-law exponent was **−0.50 vs v1's −0.84**; v1 reached v2's 7.5 k value at **step ~250**. Projection to 30 k: 9× worse for the same 4 days of A40 | `flagshipv2-6k-diagnostic.md`; per-lever telemetry otherwise healthy (no NaN, no gnorm spike, anchored decoder converging) → **the failure was simultaneity, not any single lever** | `flagship4b-v3enc-30k` |
| **D-A7** | 07-19 | **v3enc restart with STAGED levers** | Keep every *decode-side* lever from step 0 (they were healthy); soften and time-stage only the four *encoder-grounding* levers: decorr off until 10 k then 0.02, rollout-k 4→8→12, invdyn_gradscale 0.25→**0.5**, fa_dropout 0.3→**0.15** | The diagnostic isolated the encoder-grounding group as the optimization burden | `a01ad24`; `--staged-levers`. **Pre-registered falsifier:** no improvement in same-step forward-consistency vs v1 at 10 k → restart again |
| **D-A8** | 07-19 | **Acceptance gate for v3enc should be the OOD panel, not in-distribution** | v1 already passes in-distribution (0.427 vs floor 0.523) but fails OOD (comma 0.849 vs floor 0.372, 17.5 % win-rate). Optimizing the passed gate teaches nothing | the OOD panel | Proposed target: **≥ 35 %** win-rate vs the comma floor |
| **⭐ D-033** | **07-19** | **v3 pivot: hierarchical world-model PLANNING. Supervised heads demote to proposal priors** | Three measured pathologies all trace to head-supervision: longitudinal mean-regression (REF-A 94 % longitudinal, flagship high-speed the only above-floor stratum), a degenerate strategic seam (`route_skill_vs_chance` 0.0 — pure command echo), and an actively **harmful** intent→operative seam (cos vs-none **−0.238**). Making target-speed and mode-switching a **planning cost** instead of a head fixes all three at once | **P2 passed both decisive gates at zero training cost — and BOTH VERDICTS SURVIVE the estimator correction (2026-08-16/17).** ⛔ **The numbers this row published were `overlapping_holdout_se`, the BANNED estimator, and it was DECIDING both gates.** ✅ **Neither flips.** Decision-grade (episode-cluster bootstrap; §5 for the full table): **G4 closed-loop 0.9799 [0.7456, 1.2312] vs threshold 1.7318 → PASS, CI-separated**, and ⭐ **the first-ever PAIRED form −0.7375 [−0.9362, −0.5295], p(δ>0) = 0.0000, n = 221 win / 20 ep — strictly stronger than the two-interval claim it replaces**, ⇒ **42.9 % less drift**; **divergence 7.24 % [2.25 %, 14.09 %] vs 23.50 %** ⇒ 3.2× fewer. **G1: head 3.3839 [2.8336, 3.9722]; the planner arm was never dumped per-window ⇒ the re-decision is PARTIAL** — sign and separation hold, magnitude pending ~400 s GPU; a flip would need a **−73.6 %** error against a measured **−6.9 % to +5.9 %** envelope. ⚠️ **Carry with these numbers:** point estimates moved **−6.9 % to +6.8 %, bidirectional within one artifact**; intervals were **1.17×–2.17× too narrow**; the **divergence rate — the safety-shaped number — by +20.3 %**; the **G4 threshold itself was a legacy heldout mean 2.69 % LOW (1.6852 vs 1.7318), so the old gate was HARDER than the honest one**; and the banned statistic gave **7 of 40 val episodes weight exactly 0** (**C73**). ⚠️ **`planner_beats_cv` is a THIRD verdict on this path, still UNDECIDED, and its flip IS reachable** (§5). ⚠️ **Unseeded CEM** ⇒ every P2 figure carries an unbounded sampling component. *Superseded, kept visible: G1 0.893 ± 0.114 vs head 3.150, Δ +2.257 ± 0.329, "72 % reduction"; G4 1.038 ± 0.202 vs 1.685 ± 0.098, "38 % less drift"; divergence 8.7 % vs 22.2 %, "2.5× fewer".* Weight-sweep robust across a 4× band (0.647→0.669, wins all 9) — ⚠️ **on an 8-EPISODE subset**, not banned-estimator-decided | `V3_HIERARCHICAL_PLANNING_DESIGN.md` + `V3_GOAL_VOCABULARY_V1.md` (frozen). v1 remains the operative arm. Sayed's framing: v3 = the **original DINO-WM recipe** (frozen encoder + feature-prediction of action-consequences, **no supervised head**) + CEM/diffusion/MPC planner |
| **D-A9** | 07-19 | **VTARGET moved strategic → tactical** | P2 measured that the planner tracks its minted `v_target` to **1.03 m/s** — *better than the GT log tracks it (1.54 m/s)*. The longitudinal target is a **control-rate** quantity that must be re-derived faster than the strategic cadence (20 ticks); leaving it at strategic starves the cost function between updates | P2 §2.3 + §5.2; strategic route decode remains at the strategic level | `V3_GOAL_VOCABULARY_V1.md`. ⚠️ The exact vocabulary-level wording should be read from that doc before implementation — this row records *that* it moved and *why* |
| **D-A10** | 07-19 | **P2's residual is 66 % lateral — that is the P3/P4 scope, not a failure** | P2's cost is longitudinal + comfort + progress **only**; it carries no lateral/route/goal term by design. So it nails longitudinal and defaults laterally to the smoothest option. Curved-window error 2.114 m *is* the measured cost-of-no-lateral-goal | long-RMSE 1.41 / lat-RMSE 1.97; speed-decoupled cross-track 0.445 m; true actions reach 0.484 on the same curved windows | P3 = strategic lateral goal in the cost; P4 = goal-conditioned tactical predictor to lift the WM from imitation-era to planning-grade |
| **D-A11** | 07-20 | **Cosmos-Reason1-7B chosen as the dataset VLM labeler; Cosmos3 is not a labeler** | Byte-pull gating check (2026-07-20): Cosmos3-Nano/Super (OpenMDW 1.1 omnimodel, commercial-OK), Cosmos-Reason1-7B and Reason2-32B are **ungated**; only Reason2-2B/8B are gated. The pilot verdict then separated *serving* from *labeling*: Cosmos3 needs `vllm-omni`/`sglang` rather than vanilla vLLM and did not behave as a labeler | commit **`547c8ec`** "dataset: VLM pilot verdict — Cosmos-Reason1-7B for labeling, Cosmos3 is not a labeler"; pilot artifacts in `TanitAD Research Lab/Data Engineering/` | ⚠️ Sayed had earlier asked for **Cosmos3 for the dataset**; the pilot changed the answer. Every VLM label maps onto the frozen `V3_GOAL_VOCABULARY_V1` |

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
   comma2k19 speed R² 0.657 / yaw R² 0.000, and even same-corpus **rig-B speed R² −2.4654** (ADE ratio
   2.40 / 4.01; both FAIL the cross-domain >0.9 gate).

   ⛔ **CORRECTED 2026-08-03 — do NOT quote "0.930 → −2.465" as the cross-rig drop.** Those two
   numbers come from **two different experiments**: 0.930 is the *held-out* read of one run, while
   −2.4654 is the *cross-rig* read of another, **whose own within-rig baseline is +0.7863**. The
   honest cross-rig pair is therefore **+0.7863 → −2.4654**; pairing it with 0.930 borrows a
   baseline from an experiment that was not the one generalised across rigs, and overstates the drop.
   ⇒ **RULE: a "X → Y" degradation pair must come from ONE experiment.** If the baseline and the
   stressed read are from different runs, the difference contains the between-run delta as well as
   the effect, and nothing separates them afterwards.
   *(MEASURED in the D-APPEAR audit, 2026-08-03; see also R-2026-08-03-appear.)*
   ⚠️ **And the cross-rig collapse itself does not reproduce in the current cache**: A→A +0.7052,
   B→A +0.7127, paired **+0.0075 [−0.0318, +0.0502] NOT separated**. The −2.4654 is **not
   attributed** — cache geometry (the rigs' horizons agree on 8 of 256 rows here) and MLP
   extrapolation both remain live explanations. Do not cite it as an established mechanism. The YouTube-scale IDM data pipeline is **gated on
   the re-gate** and does not proceed on these numbers. Raw:
   `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-22-idm-proof/results.json`.
   **UPDATE 2026-07-24 — the fix was built and it FAILED.** The own dynamics-encoder designed to give this
   head a rig-robust latent (from-scratch, GAIA-2 all-block camera-conditioned, `dynenc-branchB` @ 40k) was
   measured on the decisive held-out-rig transfer gate and **REFUTED**: best cross-rig speed R² **−0.667**
   vs the plain frozen flagship-v1 encoder's **+0.657** (paired dR2 CI excludes 0, Branch B worse, on 3/4
   arms). See **§10**. The YouTube-scale IDM thesis resting on this encoder is now **not supported** by
   measurement, not merely gated. 🟥
   > 🔴 **LABEL-PROTOCOL CORRECTION 2026-07-27 (C29) — the `yaw R² 0.000` above is STALE-PENDING, and
   > the `speed R²` numbers beside it are NOT.** Every comma2k19 yaw number in this program before
   > 2026-07-27 was scored with **`heading_repair` OFF**: comma's heading is `arctan2` of the ENU
   > velocity and is **undefined at standstill**. MEASURED (`…/2026-07-27-idm-v3/results/labels_v3.json`):
   > **26.27 % of comma frames below 0.5 m/s are physically impossible, 0.000 % above it; PhysicalAI is
   > zero in every bin.** ⇒ **`0.000` is a fact about the LABEL, not about transfer.** The superseded
   > value is kept above deliberately. **It is NOT replaced here**, because no repaired measurement
   > exists on *this* substrate (PhysicalAI-trained head, 12,420 comma windows, **no `v_min` gate**);
   > the repaired A0 number lives on a different split and is not substitutable.
   > **What IS measured, on the v3 val split (`heading_repair` ON, `v_min` 0.5, 2,992 comma windows,
   > nothing retrained):** the deployed head reads comma2k19 `yaw_rate` **R² +0.3308** (was **+0.0114**
   > with the repair off), and a head *retrained* on repaired labels reads **+0.679**.
   > ⭐ **Honesty condition, which must travel with this correction:** on comma2k19 alone the repair
   > moves **R² +0.0114 → +0.3308** and **MAE −42.5 %**, but **medAE moves only −1.1 % and nMedAE gets
   > 8.0 % WORSE**, with Spearman ρ flat (+0.001). **The repair fixes the tail and the summary
   > statistic, not typical accuracy** — a correction quoting only the R² jump overstates it.
   > Full inventory + what is still stale-pending: `TanitAD Research Lab/Benchmarks & Eval/
   > Implementation/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`.
   > ⚠️ **PhysicalAI/rig-B numbers in §8.1 #6 and §10 are UNAFFECTED and must not be re-issued.**
   >
   > 🔴 **AMENDED 2026-07-27 (`anchor-settlement`, class C43) — `+0.3308` IS WITHDRAWN.**
   > *(The block above keeps every value and its date. The `yaw R² 0.000` cell is unchanged and
   > remains STALE-PENDING; this amendment changes only what "what IS measured elsewhere" says.)*
   > Settled **BY CONTENT** — sha256 of the raw `poses` float32 bytes **and** of the raw
   > `frames_u8 [300,9,256,256]` sensor bytes, on both hosts, 6 hash families agreeing, filenames and
   > `episode_id` used only as a cross-check: **2 of the 22 comma val episodes the `+0.3308` was
   > scored on are bit-identical to 2 of the deployed head's own 40 comma TRAINING clips**
   > (`76b:ep_00018 ≡ 61c:ep_00008`, `76b:ep_00039 ≡ 61c:ep_00020`). Those 2 carry the entire figure:
   > without them the same head reads comma yaw **R² −0.746 (CI [−1.574, −0.177])**. Its published
   > interval **[−1.2982, +0.7047]** already spanned zero, and the OFF→ON contrast measures
   > **+0.3194, CI [−1.262, +0.6425], NOT separated**. ⛔ **Do not quote `+0.3308` anywhere.**
   > ✅ **`+0.679` is NOT withdrawn** — it is `R0` (= the shipped `V3F`'s rotation head), trained on a
   > content-disjoint split. But on the 20 content-clean episodes it reads **+0.3038
   > (CI [+0.054, +0.479], separated)**, and all 18 persisted v3 arms lose 0.36–0.58 R² the same way.
   > ⇒ **The registry's quotable statement is now:** *comma2k19 yaw is TESTABLE — a retrained head
   > reads **+0.3038 [+0.054, +0.479]** on content-verified-disjoint comma val clips — and the
   > DEPLOYED head does not recover it on any content-disjoint comma substrate* (strict-admissible
   > **−0.288** on its own held-out clips, ρ 0.211, nMedAE 2.36). **Testable ≠ working.**
   > ⚠️ PhysicalAI remains UNAFFECTED and was **re-measured, not inherited**: `n_pai_changed = 0`,
   > yaw R² **+0.903482 bit-identical** under legacy, repaired and strict-admissible protocols.
   > Record: `TanitAD Research Lab/Benchmarks & Eval/Implementation/incoming/
   > 2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md` (raw: `anchor_overlap.json`,
   > `anchor_resettlement.json`, `arms_resettlement.json`).

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
`…/incoming/2026-07-22-own-dynamics-encoder/DESIGN.md`, `…/incoming/2026-07-22-own-dynamics-encoder/LAUNCH_PLAN.md` and `…/incoming/2026-07-22-own-dynamics-encoder/PRE_REGISTRATION.md` (all three verified present 2026-08-03).

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
`TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-24-branchb-transfer-eval/results_branchb_transfer_e50_CONVERGED.json`;
ckpt md5s above; flagship-v1 frozen paired control md5 `b5f07d9e3dd2ca643949bc86832e6585`, step 29999;
episode-cluster bootstrap over rig-B eval clips, 2000×; clips train rigA 100 / rigB 120 / comma 80, val
rigA 26 / rigB 54, ~~**episode-disjoint**~~ 🟥 **FALSE — CORRECTED 2026-07-28.** The val clips are
`physicalai-val-f1b378f295ae`: **rig-A 21 of 26 (80.8 %) and rig-B 41 of 54 (75.9 %) are BIT-IDENTICAL to
episodes in the parity train corpus the flagship-v1 control was trained on** (MEASURED by content, see the
Leakage bullet below). LOOP_STATE operational run-tag: `ad4e13c4`.)*

Cross = held-out **rig-B** speed R² (the pre-registered headline). Paired dR2 = Branch B − flagship-v1 on
**identical windows + identical converged head-fit** (the C6-clean, regime-robust contrast).

| arm | rig-B set | **Branch B** cross | **flagship-v1** cross | **paired dR2** [95% CI] frac+ | PASS |
|---|---|---:|---:|---|:--:|
| rig_train | train-cache (⚠ LEAKED) | −2.662 | −2.948 | +0.286 [−0.270, +0.988] .83 | ❌ |
| multirig_train | train-cache (⚠ LEAKED) | −1.703 | **+0.382** | −2.085 [−2.902, −1.519] **.00** | ❌ |
| **rig_val** ~~(clean, disjoint)~~ 🟥 **75.9 % LEAKED** | val-cache `f1b378` | −1.923 | −1.169 | −0.755 [−1.336, −0.108] .01 | ❌ |
| **multirig_val** ~~(clean, disjoint)~~ 🟥 **75.9 % LEAKED** | val-cache `f1b378` | **−0.667** | **+0.657** | −1.325 [−2.295, −0.801] **.00** | ❌ |

🟥 **The "(clean, disjoint)" labels on the two `*_val` rows were FALSE and are struck, 2026-07-28.** The
val cache is `physicalai-val-f1b378f295ae`; **3,600 of the 4,742 rig-B eval windows (75.9 %) are
bit-identical to parity-train episodes**, and the paired `dR2` column contrasts a **parity-trained**
control (flagship-v1, which saw them) against an arm whose overlap is **unmeasured**. The two paired
`dR2` values are **CONFOUNDED, not clean**; Branch B's absolute failure of the +0.9 gate is not.

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
- **Leakage — CORRECTED 2026-07-25, RE-MEASURED BY CONTENT 2026-07-28 (was WRONG here):** this bullet
  previously called `physicalai-val-f1b378f295ae` *"episode-disjoint from …-train-e438721ae894"*. It is
  **NOT**.
  **CURRENT FIGURE (MEASURED BY CONTENT, 2026-07-28): 62 of its 80 episodes = 77.5 % are IN the parity
  train `e438721ae894`** — sha256 of the raw `poses` float32 bytes **and** of the raw `frames_u8` sensor
  bytes, six identifying hash families agreeing at 62, `poses_bitwise_equal` true for every pair
  (`…/incoming/2026-07-28-parity-leak-check/raw/KNOWNPOSITIVE_f1b378_x_train.json`; first content
  measurement at the poses level `…/incoming/2026-07-26-s3-decision-grade/disjointness_result.json`).
  Frame mass: **12,328 of 15,906 frames = 77.5 %**.
  ⚠️ **SUPERSEDED VALUE, KEPT: "62 of its 79 populated episodes (78.5 %)"** — carried here from
  **2026-07-25** to 2026-07-28, sourced from
  `incoming/2026-07-25-closedloop-horizon-and-shift/E1a_E2a_RESULTS.md` §1.1, an **`episode_id`
  intersection**. The denominator was wrong: the cache holds **80** `ep_*.pt` files with **80 distinct**
  poses/frames hashes but only **79 distinct `episode_id`s** (two files share an id). **The figure is
  hereby upgraded from an `episode_id` claim to a CONTENT claim** — and it is the one place where
  `episode_id` happened to agree with the bytes.
  ⛔ **`episode_id` is not evidence in either direction** (RETRACTION_LOG **C43**): on
  `physicalai-val-0c5f7dac3b11` @600 it manufactures **20 false positives** (20 val episodes share an id
  with a train episode and share **no content**), and it is not even a key inside train (**2,342 distinct
  ids for 2,376 episodes**). Any "episode-disjoint" claim below that rests on `episode_id` is flagged, not
  trusted.
  f1b378 is **hard-refused in code** since 07-23 (`data.list_val_episodes(..., allow_leaky=False)` raises;
  canonical CLEAN val is `physicalai-val-0c5f7dac3b11`, **content-verified 0/40 and 0/600**, 2026-07-28).
  **Effect on the Branch-B contrast — the previous mitigation is WITHDRAWN.** This bullet used to argue
  *"the leak inflates both arms' val R² equally … so the WORSE-Branch-B ordering is conservative, not
  manufactured."* That is **not established and the asymmetry runs the other way**: flagship-v1 is the
  parity-trained control and **provably** saw the leaked episodes (MEASURED); Branch B trained on a
  different 2,466-clip set whose content overlap with f1b378 is **UNMEASURED**. A memorisation advantage
  for flagship-v1 is therefore a live candidate explanation for the very gap the paired `dR2` rows
  report. **Finding 1 survives** (Branch B's own cross-rig speed R² −0.667 fails the +0.9 gate on its own
  absolute value) and so does **Finding 2** (in-domain / own-head, not measured on f1b378); **Finding 3,
  the paired `dR2` on the two `*_val` rows, is CONFOUNDED and must not be quoted as a clean contrast**
  until Branch B's own overlap with f1b378 is measured by content.
  Per-row leaked mass, MEASURED 2026-07-28
  (`…/incoming/2026-07-28-leaky-cache-audit/leak_mass_by_result_set.json`):
  **rig-A val 26 clips → 21 leaked (80.8 %), 1,848 / 2,286 windows; rig-B val 54 clips → 41 leaked
  (75.9 %), 3,600 / 4,742 windows.** The `*_train` sets remain separately flagged ⚠ best-case.
  *(R8/§ line ~1556 — I-JEPA's leak vs its OWN 320-ep subset — is a DIFFERENT overlap, still unmeasured;
  do not assume f1b378 is clean there. Related and MEASURED by content 2026-07-26: the 320-episode
  `physicalai-train-51f40f5ebc21` overlaps the parity train on **256 / 320 = 80 %**.)*
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

---

## 11. PRODUCED DATASETS — artifacts this programme *created* (not models)

> Why this section exists: the Alpamayo-2 augmentation set was the programme's largest
> single inference spend (78.4 wall-hours) and **had no registry row for 11 days**, so
> `Paper/TANITAD_PAPER.md` carried its counts as **INHERITED** with the flag *"pending a
> registry row"*. Same failure mode as v5f being the headline live run with no row (§1.8).
> A produced dataset is quotable only from here or from its own parquet.
>
> **Rows: §11.1** the A2 augmentation set · **§11.2** the PH1-fused hierarchy layer derived from it.
>
> ⛔ **The counting rule for this section, learned twice the hard way: COUNT RECORDS, NEVER FILES.**
> A listing probe sees a **missing** file but never a **short** one — that is how an 8-clip gap was
> really 115 (§11.2), and how a 1-row hole was really 356 (§11.1). Every count here is a row tally,
> a per-clip task tally, or a per-record key census, and each states its **n**. Where a stage's
> coverage is not conserved (`n_out != n_in`), the row says so with the number.

### 11.1 Alpamayo-2-Super augmentation of PhysicalAI-AV — `Sayood/tanitad-alpamayo2-augmentation` — ✅ **COMPLETE** (produced 2026-08-06 → 2026-08-09)

**Every count below is MEASURED BY ME, 2026-08-15, by direct aggregation over
`records.parquet`** — not from the dataset card, which disagrees (see the ⛔ block). The
local copy is proven to BE the published artifact: sha256
`ecae276db9969de115abb3caa1e87d97eae0535544be8f5edcc33ec45d925ed2` matches the far-side
LFS oid, and both sides are **25,970,018 B**. Extraction/verification script
`…/2026-08-06-alpamayo-augmentation/tools/verify_a2_parquet.py`; the earlier read is
`Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md` §2.5, which this
row reproduces independently and then extends.

✅ **RE-VERIFIED 2026-08-16 by a second, independently written probe**
(`…/2026-08-06-alpamayo-augmentation/tools/reverify_a2_counts.py` → `…/tools/a2_reverify_2026-08-16.json`), which recomputed every
count from the parquet rather than re-running the first script. **All headline counts below
reproduced exactly** (23,644 / 4,729 / per-task / 78.36 h / 59.65 s per clip / one quantisation
arm / 0 errors), the far-side sha256 + byte-size match was re-confirmed, and the manifest
accounting was re-derived from a fresh pull of `selection_manifest.json`. **One paragraph did
NOT survive** — the `trajectory`-task claim, corrected in its own block below. Counts are
**RECORD counts** throughout: rows and per-clip task-row tallies, never a file listing.

| Field | Value |
|---|---|
| **What it is** | 5-task `nvidia/Alpamayo2-Super` (34.3 B) inference over a stratified selection of PhysicalAI-AV. ⛔ **No raw sensor data** — every row joins back by `clip_id` + `t0_us`. |
| **Source corpus** | `nvidia/PhysicalAI-Autonomous-Vehicles` — **306,152 clips ≈ 1,701 h** (MEASURED, `data_collection.parquet` row count 2026-08-06, `…/2026-08-06-alpamayo-augmentation/DESIGN.md`). |
| **Coverage of source** | **4,729 / 306,152 clips = 1.54 %** of the corpus by clip. ⚠️ The PI brief targeted **100 h ≈ 18,000 clips (5.9 %)**; delivered is **26.3 % of that target** — a scope fact, not a failure, and it is the number a "scale the augmentation" decision starts from. |
| **Delivery vs selection** | **4,729 delivered / 4,800 selected = 98.52 % of the manifest**; **23,644 / 24,000 rows = 98.52 %**. |
| **Pipeline entry point** | `TanitAD Research Lab/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/tools/a2_alltasks.py` (the 5-task battery), driven per batch by `…/2026-08-05-alpamayo2-super/tools/a2_batch.py`; the single-clip quantised runner is `…/2026-08-05-alpamayo2-super/tools/a2_quant_run.py` (+ its `run_4bit_a40/` copy). Design + stage plan: `…/2026-08-06-alpamayo-augmentation/DESIGN.md`. ⚠️ **The orchestration ran pod-side and the pods are dead** — §2.6 of the addendum records that `a2_batch_out/` did not reach the repo, so the *driver invocation* is 🟥 UNVERIFIED even though the entry points are in-repo. |
| **Artifacts live** | HF `Sayood/tanitad-alpamayo2-augmentation` (5 files, far-side listing MEASURED 2026-08-16) · in-repo tooling + road-class labels under `…/2026-08-06-alpamayo-augmentation/` (`aug_road_class.json`, `a2_records_stats.json`, `tools/`, `pilot_rows_10clips.jsonl`, `aug_candidates_phase1.json.xz`) · local verified copy of the parquet in the session scratchpad (not committed — 25.97 MB). |
| **Status** | ✅ COMPLETE, on HF, **`error` non-null on 0 of 23,644 rows** |
| **HF** | `Sayood/tanitad-alpamayo2-augmentation` · 5 files: `records.parquet` (25,970,018 B), `README.md`, `selection_manifest.json` (340,800 B), `vqa_bank_500.json`, `.gitattributes` |
| **Rows / clips** | **23,644 rows · 4,729 unique `clip_id` · 5 tasks · 12 columns** |
| **Schema (12 cols)** | `clip_id` · `t0_us` · `task` · `model_id` · `quantisation` · `seed` · `wall_s` · `error` · `vqa_qid` · `vqa_category` · `question` · `raw_json` (all `large_string` except `t0_us`/`seed` int64, `wall_s` double, `error` **null-typed**) |
| **Determinism** | `seed` = **42** on **23,644/23,644**; `model_id` = `nvidia/Alpamayo2-Super` on **23,644/23,644**; `t0_us` = **5,100,000** on every row (single distinct value — the loader default) |
| **Parity** | ⛔ **NOT the training-parity corpus.** Selection is its own manifest, not `e438721ae894`/`f09e44db`. Nothing here re-selects training episodes, so §0.1 parity is untouched — but no A2 row may be joined to a parity arm without an explicit clip-level join. |
| **License** | `nvidia-physicalai-derivative` (card), inheriting `nvidia/PhysicalAI-Autonomous-Vehicles`. 🟥 **OPEN PI DECISION — two in-repo documents disagree and this row does NOT pick a side.** See the ⚖️ block below. |

**Per-task rows and measured cost** (`wall_s`, the throughput record the DESIGN doc said the
pilot would produce):

| task | rows | mean s/clip | median s/clip |
|---|---|---|---|
| `trajectory` | 4,729 | 11.47 | 9.5 |
| `meta_action` | 4,729 | 11.74 | 10.0 |
| `auto_labeling` | 4,729 | **17.40** | 18.1 |
| `vqa` | 4,729 | 8.59 | 8.2 |
| `grounding_via_vqa` | **4,728** | 10.45 | 10.1 |
| **total** | **23,644** | **59.65 s/clip** (full 5-task battery) | **78.4 wall-hours** |

⇒ Against the DESIGN doc's **ESTIMATE ~200 s/clip**, the realised pipeline is **~3.4×
cheaper**; the earlier single-task MEASURED anchor ("meta-action 40 s/clip") was **3.4×
slower than the batch rate**. This is the cost basis for any future A2 pass.

⭐ **THE QUANTISATION COLUMN — one arm, and it is stamped unvalidated in the data:**

| `quantisation` value | rows |
|---|---|
| `NF4-backbone-4bit-UNVALIDATED` | **23,644 (100.0 %)** |

⛔ **There is NO bf16 arm, therefore NO quantized-vs-full comparison inside this dataset.**
Anyone reaching for one is reaching for a column that does not exist. The only
quantized-vs-full evidence in the programme is the *indirect* one below, explicitly declared
non-comparable.

**The quantisation measurement itself (MEASURED, addendum §2.2; INHERITED into this row and
so marked):** NF4 applied to `vlm.model.language_model.*` only — **448 `Linear4bit`
modules**, expert and vision tower left BF16 — ran the 34.3 B model at **25.84 GiB peak on a
46 GB A40** against NVIDIA's published **72,115 MiB on 1 × H100 80 GB** = **2.79×**.
⚠️ **Not a like-for-like reduction claim**: different GPU, different profile methodology,
and NVIDIA states other architectures are unvalidated. The `UNVALIDATED` stamp in the data
is the honest carrier of that.

### ⛔ THE DATASET CARD DISAGREES WITH THE PARQUET, AND IT UNDERSTATES THE HOLE BY 356×

The card (`README.md` on HF) claims *"4,800 clips … 23,999 inference rows"* and, under
**Known holes**, *"One task row of 24,000 missing (23,999)."* **MEASURED against the
selection manifest and the parquet:**

| quantity | card | MEASURED | |
|---|---|---|---|
| clips | 4,800 | **4,729** | −71 |
| rows | 23,999 | **23,644** | **−355** |
| rows missing vs 4,800 × 5 = 24,000 | "one" | **356** (1.48 %) | ⛔ **356× understated** |

**The exact accounting** (it closes to the row):

| effect | clips | rows |
|---|---|---|
| selected in `selection_manifest.json` (4,800 unique `clip_id`, all `t0_us` = 5,100,000) | 4,800 | 24,000 |
| ⛔ **selected but produced ZERO rows** | **−81** | **−405** |
| ⛔ **present in the parquet but NEVER in the manifest** | **+10** | **+50** |
| ⚠️ one clip missing its `grounding_via_vqa` row (4,728 clips have 5 tasks, 1 has 4) | — | **−1** |
| **delivered** | **4,729** | **23,644** ✅ |

⚠️ **The +10 is the more serious of the two.** The delivered set is **not a subset of the
selected set**, so nothing may be inferred about the delivered 4,729 from the manifest alone.

⭐ **But the stratification SURVIVES INTACT, and this is MEASURED, not assumed.** Joining the
parquet's own `clip_id` to
`TanitAD Research Lab/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/aug_road_class.json`
(3,592 labelled clips; `_rule` = *"ego-derived (labels-may-use-ego); highway frac(v>20)>0.6;
intersection_rich stop+turn or big turn at low speed; urban vmed>2; else unstructured"*):

| stratum | manifest (4,800) | **delivered (4,729)** |
|---|---|---|
| urban | 1,884 | **1,884** |
| intersection_rich | 1,241 | **1,241** |
| highway | 384 | **384** |
| unstructured | 83 | **83** |
| *(unlabelled remainder)* | 1,208 | **1,137** |

⇒ **Every one of the 81 zero-row clips and all 10 off-manifest clips fall in the UNLABELLED
remainder** — the four stratified classes are delivered **100 % complete**. The card's
stratum table is therefore correct for the delivered set as well as the manifest, and the
1.48 % hole is confined entirely to the unstratified remainder. *(`no_ego` = 0, so the
ego-derived rule labelled every clip it could.)*
⚠️ The road-class labels are **ego-derived** — admissible as labels under the binding
label/inference rule, ⛔ **never as an inference-time input.**

✅ **The stratum table above was RE-DERIVED INDEPENDENTLY 2026-08-16** (fresh manifest pull +
fresh parquet join, `…/2026-08-06-alpamayo-augmentation/tools/reverify_a2_counts.py`): manifest 4,800 → urban 1,884 ·
intersection_rich 1,241 · highway 384 · unstructured 83 · unlabelled 1,208; delivered 4,729 →
identical on all four labelled strata, unlabelled **1,137**; and the **81 zero-row clips are
81/81 unlabelled** and the **10 off-manifest clips 10/10 unlabelled**. `aug_road_class.json`
carries `counts` + `classes` (**3,592 labelled clip_ids**) and is git-tracked at
`TanitAD Research Lab/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/aug_road_class.json`.
*(An earlier draft of this row asserted that file "IS NOT THERE" — that absence-claim was wrong
and is retracted; it is present and tracked. Operating-Standard rule 2: absence at one probe is
not absence.)*

### ⚖️ OPEN PI DECISION — MAY THIS DATASET BE USED IN A PUBLIC CLAIM? TWO DOCS DISAGREE

**Recorded, not resolved. This row takes no side; the PI decides.** The two documents are not
obviously talking about the same object, which is exactly why it needs a decision rather than a
reading:

| document | claim | scope it names |
|---|---|---|
| `TanitAD Research Lab/Data Engineering/TANITDATASET_V1_STRATEGY.md:61` | **"PhysicalAI-AV/Alpamayo … commercial-OK for internal AV dev but no-derivatives → firewalled, `recipe-only`"**; `:42` `firewalled` = *"never enters the lake — recipe-only"*; `:35` `assemble_lake_record` **raises `PermissionError`** if PhysicalAI-AV is fed to the lake | the **source dataset** |
| `…/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md:24, :162, :229` | **"OpenMDW-1.1 … fine-tuning, derivatives, commercial redistribution"**, and **"OpenMDW-1.1 permits derivative use"** cited as the licence basis for auto-labelling | the **model weights** |

⚠️ **This dataset is precisely the object the two scopes collide on:** it is a **derivative of
the restricted DATASET, produced by the permissive MODEL**, and it is **already published on
HF**. Under the strategy doc's reading it is `firewalled`/`recipe-only`; under the analysis
doc's reading the derivative is permitted.
⛔ **Until the PI rules, no public claim, demo, paper figure or external release may rest on
this dataset**, and no agent may cite either document as having settled it. Internal
programme use (auto-labelling, distillation targets, the PH0/PH1 fusion in §11.2) is unaffected
by the dispute as framed — both readings permit internal AV development.

⚠️ **Two provenance fields the card promises are ABSENT from the data.** The card lists
*"`model_id`, `seed`, `_quantisation` (4-bit bnb load), `_contamination` note, `peak_gib`"*.
MEASURED, two independent probe shapes (JSON-key parse of every `raw_json`, and a raw
substring scan of all 23,644 rows): `model_id` ✅ and `seed` ✅ are top-level **columns**;
the column is `quantisation`, not `_quantisation`; **`_contamination` = 0/23,644** and
**`peak_gib` = 0/23,644**. ⇒ The per-row contamination note and the memory figure exist only
in prose, not in the artifact.

### ⛔ CORRECTED 2026-08-16 — THE `trajectory` TASK IS NOT 94 % EMPTY; IT HAS TWO SCHEMA VARIANTS

**The first read of this paragraph was wrong and is retracted here.** It said *"only 255 of
4,729 `trajectory` rows carry `min_ade_m` — the rest have `num_trajectory_samples: None`,
i.e. the GT-dependent metric block never ran."* **MEASURED on re-verification** (independent
re-derivation from the same parquet, `…/2026-08-06-alpamayo-augmentation/tools/reverify_a2_counts.py` →
`…/tools/a2_reverify_2026-08-16.json`): `raw_json` holds **two disjoint schema variants**, and a
key-presence probe over a heterogeneous column mistook the minority variant's vocabulary for
the whole task's completeness.

| variant | rows | keys present | GT-referenced error | waypoints |
|---|---|---|---|---|
| **A — waypoints** | **4,474** (94.61 %) | `cot` · `pred_xyz` · `pred_rot` · `logprob` · `ade_vs_gt_m` (all 5 on 4,474/4,474) | **`ade_vs_gt_m` on 4,474/4,474, 0 null** | **`pred_xyz` = 64 points on 4,474/4,474** |
| **B — metric block** | **255** (5.39 %) | 15+, incl. `figure_style` · `clip_id` · `t0_us` · `camera_indices` · `input_profile` · `coc_label` · `cots` · `pred_xyz_shape` · `num_trajectory_samples` · `min_ade_m` · `min_fde_m` | `min_ade_m` / `min_fde_m` on 255/255 | `pred_xyz_shape` = `[1,1,1,64,3]`; ⛔ **no `pred_xyz` on any of the 255** |

⛔ **No row carries both.** `pred_xyz` = 4,474 rows, `pred_xyz_shape` = 255 rows, disjoint;
`4,474 + 255 = 4,729` = one trajectory row per clip. ⛔ **The key `num_trajectory_samples` is
ABSENT ENTIRELY** on the 4,474 variant-A rows — 255 substring hits over the whole column — so
*"the rest have `num_trajectory_samples: None`"* is false, and so is the inference drawn from
it.

**The better-powered read is variant A, and it corroborates variant B at 17.5× the n:**

| statistic | variant A `ade_vs_gt_m` | variant B `min_ade_m` |
|---|---|---|
| **n** | **4,474** | 255 |
| mean (m) | **2.2584** | 2.3469 |
| median (m) | **1.5245** | 1.5233 |
| p10 / p90 (m) | 0.4061 / 5.0252 | — |
| nulls | **0** | 0 |
| (`min_fde_m`, variant B only) | — | mean 6.8726 / median 4.6711 |

⇒ **What the `trajectory` task can be used for is BROADER than previously recorded:** predicted
waypoints exist for **4,474 of 4,729 clips** (not 255), and a GT-referenced along-path error
exists for **4,729 of 4,729** (4,474 `ade_vs_gt_m` + 255 `min_ade_m`).

⚠️ **Whether `ade_vs_gt_m` and `min_ade_m` are the SAME estimator is 🟥 UNVERIFIED** — they come
from different code paths and are differently named. Their agreement (2.2584 vs 2.3469 mean;
1.5245 vs 1.5233 median) is *consistent with* identity but does not prove it. ⛔ **Do not pool
the two columns into one distribution** without settling that first.
⛔ **Neither is comparable to the published minADE₆ 0.911 m** — both are **1 sample** at
Alpamayo's native **6.4 s** horizon on a different corpus slice (the §2.2 three-reasons rule).

**Root-cause class:** the same family as **C18** (a defect scoped by the probe that found it) —
here a key-presence probe over a **heterogeneous** JSON column, which can see a key MISSING but
cannot see that a *different key answering the same question* is present. Logged in
`RETRACTION_LOG.md`.

**Reconstruction risk.** 🟥 The production pods are dead; the run is reproducible only from
`…/2026-08-06-alpamayo-augmentation/DESIGN.md` plus the manifest. The 81 zero-row clips have
**no PER-CLIP recorded reason** — `error` is null-typed for the whole column, so a clip that
failed before writing a row is indistinguishable from one never attempted. ⇒ A rebuild must
re-derive completeness by joining the manifest, exactly as this row does.
⚠️ **Corrected 2026-08-16 — a RUN-LEVEL cause *is* recorded, and an earlier draft of this row
said there was none.** The card's *Known holes* section names it verbatim: the missing rows
occurred **"after two MooseFS I/O incidents during"** the run (card text re-read directly from
HF, 1,746 B, this session). ⇒ The gap is **not** unexplained: it is attributed to storage I/O,
which is consistent with the losses falling in an unstratified, non-systematic pattern (§ the
stratum table above). What is missing is only the **per-clip** attribution.
*(Operating-Standard rule 2 again: "no recorded reason" was an absence-claim made without
reading the one document that records it.)*

**Card text verified verbatim 2026-08-16** (direct fetch, not inherited): *"**4,800 clips
(~26.7 h of driving), 23,999 inference rows, 5 tasks per clip**"*; *"One task row of 24,000
missing (23,999)"*; `license_name: nvidia-physicalai-derivative`; and the card's pointer to
*"per-clip road classes: `aug_road_class.json` in the TanitAD repo"* — **which resolves
correctly** (git-tracked, 3,592 labelled clips). ⇒ **Delivered scale: 4,729 clips ≈ 26.3 h**
by the card's own per-clip rate.

**Sources.** `records.parquet` (sha256 verified above) · `selection_manifest.json` ·
`Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md` §2.2/§2.5 ·
`TanitAD Research Lab/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md` ·
comparison analysis `…/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md`.

⇒ **`Paper/TANITAD_PAPER.md` may now promote its A2 counts from INHERITED to MEASURED,
citing this row — but it must quote 4,729 / 23,644, NOT the card's 4,800 / 23,999.**
*(Done 2026-08-16 at `TANITAD_PAPER.md` §7.15 and §8-item-9.)*

#### 11.1a — What the A2 labels ACTUALLY CONTAIN, and the two limits on using them (added 2026-08-17)

**Why this subsection exists:** §11.1 establishes *how much* A2 there is. It does not say *what the
labels are* or *what they cannot do* — and both usage limits below are the kind that get discovered by
a trainer eating the corpus rather than by a reader.

⛔ **PARITY — state this before anything else. These clips are NOT in the canonical train corpus.**
A2 covers 4,729 clips of `nvidia/PhysicalAI-Autonomous-Vehicles`; the parity set is
`physicalai-train-e438721ae894` (2,376 episodes, skip-hash `f09e44db`). **A2 is a SEPARATE labelled
corpus, never an extension of the parity set.** Any trainer that eats it across that boundary breaks
cross-arm comparability and must be refused — the standing rule in `CLAUDE.md` under *Parity is sacred*.

| Field | Value |
|---|---|
| **Identity** | `Sayood/tanitad-alpamayo2-augmentation` → `records.parquet`. **4,729 clips × 5 tasks** (`trajectory`, `meta_action`, `auto_labeling`, `vqa`, `grounding_via_vqa`). ⚠️ **It is LABELS, not video.** |
| **Label structure** | Three axes — **Longitudinal / Lateral / Lane** — **7 values each**. |
| **Parse rate** | **Zero unparsable rows.** The 304 apparent "unparsed" are exactly the **304 `Stop` rows**, where a stopped vehicle legitimately emits one axis. |
| ⭐ **LAT×LON simultaneity** | **40.62 % of clips declare non-trivial action on LON and LAT at the same time** — direct label-side support for v6's LAT×LON factoring, and an argument against any single mixed softmax over the two (the defect named in `longitudinal-blindness-root-cause`). |
| **Reasoning fields** | `meta_action.cot` at **100 % coverage, 1,103 distinct strings**. `auto_labeling.chain_of_causation` is only **22.18 % string-identical** to it ⇒ semi-independent, usable as a repeat measurement. |
| ⛔ **`answer` is NOT corroboration** | `answer` is a **BYTE-DUPLICATE of `cot` on 4,729/4,729**. It may **never** be cited as agreement with `cot` — it is the same string. *(Named here because it is exactly the field someone reaches for to "confirm" a finding.)* |

**⛔ TWO USAGE LIMITS — these belong with the labels, not in a footnote.**

1. **`Stop` is a STATE, not an action — a naive `Stop → BRAKE_TO` mapping is wrong on most of them.**
   MEASURED: on `Stop` rows the ego is at **v(t₀) = 0.51 m/s** and **rising to 2.95 m/s by the 2 s
   horizon** (Δv **+2.44**), and the `cot` strings say *"Resume speed from stop"*. The label describes
   the situation the clip **starts in**, not the manoeuvre it executes.
2. **The LON axis always abstains ⇒ Alpamayo cannot corroborate a longitudinal tactical label.**
   This is why **`a_tac_lon` is independently corroborated on 0/201** clips of the aug120 fused corpus
   (§11.2). Any fusion design that assumes A2 votes on longitudinal is assuming a signal that is not
   in the corpus.

**AGREEMENT — and why the old 2-of-3 vote is retired.**

| pair | κ | reading |
|---|---|---|
| Alpamayo ↔ VLM | **0.1717** | low |
| VLM ↔ ego | **0.7608** | ⛔ **NOT an independent agreement** |

⛔ **The VLM and ego legs are not independent:** `_ego_prompt_mode == 'past'` on **201/201**, i.e. the VLM
was shown the very fields the ego voter reads. **The old 2-of-3 majority vote was one source counted
twice**, and it is **retired**. ⇒ **Alpamayo is the only trustworthy leg of the three.**

⚠️ **BUT — aggregate corroboration and per-clip discrimination are DIFFERENT PROPERTIES, and A2 has only
the first.** MEASURED on the PI's 80 removed lane-change records: A2 covers only **23/80**; its single
`Left Lane Change` call lands on a clip the PI adjudicated **WRONG**; and **3 of his 4 CORRECT clips read
`Lane Keep`**. ⇒ **A per-clip rule built on A2 would have promoted a wrong clip and flattened three right
ones.** Use A2 to corroborate a distribution; do not use it to adjudicate a clip.

**COVERAGE AT OUR GEOMETRY — the operationally important number.**

| slice | clips | note |
|---|---|---|
| have w120 video | **257** | |
| processed (the aug120 corpus, §11.2) | **201** | ⊂ the 257 |
| **no w120 cache at all** | **4,472** | **94.6 % of the corpus** |
| | **= 4,729** | ✅ 257 + 4,472 = 4,729 — identity checked by me, 2026-08-17 |

⇒ **Closing it is an EXTRACTION job, not a labelling one — the labels are already valid for all 4,729.**
Cost: **1,418 of 3,146 camera chunks ≈ 1.84 TB**. **Deferred by PI decision (2026-08-17)** to Thor
post-30k, **densest-chunks-first** (top 50 chunks ⇒ **1,317 clips / 65 GB**, i.e. 28 % of the remaining
clips for 3.5 % of the bytes).

**Evidence class.** All figures in 11.1a are **MEASURED** by this session's streams and **INHERITED into
this row and so marked** — owning packages: `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md`,
`…/incoming/2026-08-17-aug120-refuse/AUG120_REFUSE.md`, `…/incoming/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md`
(all three paths verified to resolve, 2026-08-17 — note they live under **Data Engineering**, not
*Benchmarks & Eval*). ⚠️ **The 1.84 TB is MEASURED counts × an ESTIMATED mean chunk size (1.30 GB)** —
1,418 × 1.30 GB = 1.84 TB, arithmetic checked by me; the *estimate* is the chunk size, not the count.
⚠️ **I did not re-derive the κ values, the Δv, or the 80-record adjudication from the parquet** — they
are quoted from the owning packages and are marked INHERITED for that reason.

⚠️ **STALE-TEXT FLAG for §11.2, raised not fixed (2026-08-17):** `goal_evidence: grounded` is
**retired**, the **geometric lane-change gate is removed**, and the **aug120 corpus was re-fused**. Any
§11.2 text describing the old label emission or the 2-of-3 vote is therefore stale. Left for the owning
stream rather than rewritten blind — but it is flagged here so it is corrected rather than re-derived.

---

### 11.2 PH1-fused hierarchical label layer over the aug120 slice — `Sayood/tanitad-ph0-aug120/fused_aug120/` — ✅ **COMPLETE, with a NAMED 57.2 % perception hole** (produced 2026-08-15)

**Why it is a registry row:** it is the first **derived** product of §11.1 — the layer that turns
A2 rows into v6 hierarchy tokens — and its headline coverage number was **understated 14×** in the
document that first reported it. A produced dataset with a hole that large is quotable only from
here or from its own summary JSON.

**Evidence classes, stated per group rather than blanket-stamped:**

- ⭐ **MEASURED BY ME 2026-08-16, re-derived from the artifacts** (`…/2026-08-06-alpamayo-augmentation/tools/reverify_a2_counts.py`
  P3 over the in-repo raw JSONs, **plus a direct re-read of all 201 fused records**): the record
  count, the SAM3 coverage, corroborations/conflicts, the per-batch conservation columns, **every
  vocabulary tally**, the voter-majority counts, the winning-vote tally, `sign_text_status`,
  `inference_admissible`, the `goal_evidence` verdict split, and the A2 join. **All reproduced the
  owning document EXACTLY** — 8/8 aggregate columns and 100 % of the token counts.
  ⭐ The strongest of these is a **cross-artifact join**: the 201 aug120 `clip_id`s are **201/201
  present in §11.1's `records.parquet`** and pull **exactly 1,005 rows = 201 × 5, 201 per task** —
  so §11.1 and §11.2 are joined by measurement, not by assertion.
- **MEASURED by the owning run, INHERITED into this row and so marked:** the far-side verification
  (204/204 files, 6/6 md5 byte round-trip), the test counts, **G1 CLOSED at 0/31**, and the SAM3
  sign-class reliability note (⅔ of best crops had no sign). *(Locally I confirm 204 files = **201
  records + 3 meta** — `_summary.json`, `_batch_accounting.json`, `_label_sources.json` — which is
  the very "count records, not files" distinction this row exists to make.)*

| Field | Value |
|---|---|
| **What it is** | Adversarially-fused per-clip hierarchy labels (`schema ph1-fused-v1`): ego geometry + VLM (PH0 Engine B) + SAM3 perception + **the §11.1 Alpamayo layer**, voted 2-of-3 into v6 `g_str` / `g_tac_lat` / `g_tac_lon` tokens. |
| **Source population** | The aug120 slice = `records.parquet` clips (§11.1) ∩ w120 corpus − the 600 already-fused val clips. Reconstructed from primary sources as **201 clips**, matching the pipeline's own `todo=201` log line. |
| **Records** | **201 fused records** (201/201 of the population; `n_v2` 201, `no_v2` 0, `no_ego` 0). |
| **Alpamayo coverage** | **201/201 (100 %)** — the slice is *defined* by carrying an A2 record; **1,005 A2 rows = 201 × 5 tasks**, all five task keys present on all 201. |
| ⛔ **SAM3 (perception) coverage** | **86 / 201 = 42.8 %.** **115 clips (57.2 %) carry NO SAM3 record**, stamped `perception.absent = "AUG120_SAM3_STAGE_GAP"` per record. |
| **Corroborations / conflicts** | **88 / 10** — ⚠️ a **floor, not a rate**: with SAM3 absent on 115 clips, 2 of the 6 checks cannot fire at all. |
| **Artifacts live** | HF `Sayood/tanitad-ph0-aug120/fused_aug120/` (204 files, far-side verified 204/204 + a 6/6 md5 byte round-trip) · in-repo raw + code at `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-15-aug120-fusion/` (`raw/fused_aug120_summary.json`, `raw/fused_aug120_batch_accounting.json`, `raw/fused_aug120_label_sources.json`, `raw/aug120_coverage.json`). |
| **Pipeline entry point** | `stack/scripts/ph1_fuse.py` (fuser) · `stack/scripts/aug120_pipeline.py` (label production) · `stack/scripts/ph0_sam3.py` (perception stage) · run driver `…/2026-08-15-aug120-fusion/code/aug120_fuse_run.py`. Strategy: `…/2026-08-07-hierarchical-wm-redesign/PH1_FUSION_STRATEGY.md`. |
| **Parity** | ⛔ **NOT the training-parity corpus** (inherits §11.1's position). Disjoint by construction from the val-600 fused set. |
| **License** | Inherits §11.1's 🟥 **OPEN PI DECISION** verbatim — it is a derivative of a derivative. |

**Vocabulary emitted (MEASURED, tokens imported from `tanitad.models.v6`, not re-declared):**
`g_str` FOLLOW_MAIN_ROAD 151 · ROUTE_TO 31 · TURN_LEFT 17 · TURN_RIGHT 2 · `g_tac_lat` LANE_KEEP
186 · LANE_CHANGE_L 7 · LANE_CHANGE_R 7 · null 1 · `g_tac_lon` CRUISE 112 · BRAKE_TO 52 · HOLD 36
· YIELD_MERGE 1. True 2-of-3 majority backs **178/201 lateral** and **61/201 longitudinal**;
winning-vote sources ego 316 · alpamayo 245 · vlm 211.

⛔ **THE 14× UNDERSTATEMENT, AND ITS ROOT CAUSE (both MEASURED).** `STOP_2026-08-15_RESUME_RUNBOOK.md`
§6.11 recorded the SAM3 gap as **`batch_00184`, 8 clips**. The true gap is **115 of 201 (57.2 %)**,
spread across *every* batch — **14.4× larger**.

| | runbook §6.11 | **MEASURED** | |
|---|---|---|---|
| clips missing SAM3 | 8 (one batch) | **115** (every batch) | ⛔ **14.4×** |
| SAM3 files present | 25 of 26 prefixes | 25 — *all present* | the listing looked fine |
| **SAM3 RECORDS inside them** | not counted | **93** (25 files × exactly 4), **86 distinct clips** | the hole was *inside* the files |

- **Mechanism:** `stack/scripts/aug120_pipeline.py` passed `--n` to the bridge and to the VLM and
  **omitted it for SAM3**, whose default is **4** (`ph0_sam3.py:387`, consumed at `:411` `[:a.n]`).
  Every batch got SAM3 on its first 4 clips **and printed `SAM3_RC=0`**. Fixed in place
  (`--n str(len(batch))`); ⚠️ **the fix does not retro-fill** — the 115 need a GPU re-run.
- **Root-cause class C18** (`RETRACTION_LOG.md`): *a defect found by a structural listing probe is
  bounded by that probe's granularity* — **a listing sees a MISSING file, never a SHORT one.**
  ⇒ **The standing check is a conservation count `n_out == n_in` per stage per batch.**

**Known gaps, each with its n — none silently dropped:**
1. 🟥 **115/201 clips have no perception layer.** Needs ~30 min GPU + a re-fuse; the fuser resumes,
   so only new records are written. **This is the only thing between this row and a complete layer.**
2. ⚠️ **`scene_vs_situations` fired 0/201** — the frozen situation detectors never ran on these
   batches and the v2 records carry no `situations` key. *Absence of an instrument, recorded as
   absence*, not a zero.
3. ⚠️ **`fused_w120val/` (the 600-clip baseline, NOT this row) still carries 4 records fused with a
   silently empty perception layer.** Correcting them would re-baseline the published 175/41/56 →
   flagged as a PI call, not silently redone.
4. ⚠️ `sign_text_status == pending_g1_gate` on **201/201** — **G1 is CLOSED at 0/31**, so sign text
   stays extraction-only and never reaches a goal token.
5. 🟥 **15 `goal_evidence: grounded` verdicts** rest on SAM3 sign tracks whose class reliability is
   flagged (⅔ of best crops had no sign) — a threshold study is scoped but not run.

**Owning document.** `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md`
(+ `MANIFEST.md`, `NEXT_4472_BUILD_INPUTS.md` for the follow-on build). Tests:
`stack/tests/test_ph1_fuse.py` 14 passed; full suite 2,812 passed / 0 failed / 17 skipped / 2 xfailed.

---

## 12. FROZEN-TRUNK READOUT LINE — what the v6 latent does and does not carry [TIER T0-DIAGNOSTIC throughout]

⛔ **Everything in this section is a WORLD-MODEL DIAGNOSTIC and may NEVER be quoted as driving
performance.** T1 capability claims live in §1.12, §1.13c and §5.

**Substrate:** frozen `v6F-SW-30k` snapshots (`/home/nvidia/ckpt_snaps`, fp16 weights-only).
**Estimator:** paired episode-cluster bootstrap throughout. **Instrument:** `pc6_linear_readout`
ridge — ⛔ **pass `intercept_col=-1`**; the default is deliberately the incumbent (biased) behaviour
so banked `pc6_ridge_*` JSONs under `…/incoming/2026-08-17-probe-positive-control/raw/`
reproduce bit-exactly (C92).

### 12.1 ⛔ The 40:1 pooling bottleneck is REFUTED — the ENCODER is the constraint (C104, 2026-08-18)

| | |
|---|---|
| **Design** | pre-registered; 4 pooling ratios (40:1 deployed / 10:1 / 4:1 / 1:1) differing **only in the kernel**, each forced to exactly 2,048 features by a **fixed random projection**, 5 seeds, 1,302 train / 1,507 eval windows in 70 episode clusters |
| **Result** | on the four rungs the hypothesis was built to explain, **removing the pool entirely moves r² by \|Δ\| ≤ 0.0002, CI containing zero on all five seeds** (`lead_closing` Δ = +0.00001 [−0.00597, +0.00504]) ⇒ **`R1 DROPPED` by its own criterion** |
| **Discriminator** | through the **same** deployed `AvgPool2d((4,10))` on the **same** windows, `facebook/dinov2-base` reads `lead_gap` **0.44997** vs ours **0.00496**, `ego_v0` **0.71733** vs **0.05240**, `lead_closing` **0.01713** vs **0.00000** |
| **Not capacity** | DINOv2-B/14 **86 M** params against our encoder's **87.3 M** |
| ⇒ | **the information is in our images, it SURVIVES the pool, and neither pooling nor corpus narrowness is the constraint. The encoder gap is 91×.** |

⚠️ **CONTROL CORRECTION (C109): `PC-2OBJ` — the positive control this result originally cited — is
INERT AT 40:1 BY CONSTRUCTION** (two *opposing* plants inside one cell cancel; at p40 it reproduced
the un-planted arm to 5e-05). **The verdict is unchanged** — `PC-LOCAL`/`PC-DIST` do fire (our own
trained tokens through the deployed pool: 0.0596 → 1.0000, K1 9/9) — **but PC-2OBJ must not be cited
as the control.**

### 12.2 ⚠️ Random init vs trained encoder — `ego_v0` only, and NOT as a ratio (C109 supersedes C106)

⛔ **Write this row from C109, never from C106.** C106 published *"random init beats the trained
encoder 3.6× on both rungs"*; C109 attacked it five ways.

| claim | status |
|---|---|
| `ego_v0`: random init reads better | ✅ **SURVIVES, with a real estimator** — paired episode-cluster bootstrap on Δr²c, **+0.150 [+0.055, +0.226], p(Δ>0) = 1.000**, positive in **27/27 cells** (3 init × 3 projection × 3 ridge seeds) |
| `lead_gap` half | ⛔ **DIES** — 0 of 27 cells CI-separated, p 0.71–0.76, **sign flips in 9/27** |
| the **3.6× ratio** | ⛔ **WITHDRAWN.** It compares a near-constant predictor (`pred_sd/gt_sd` 0.014) to a live one (0.89), and re-drawing the **ridge inner split** moves it to 2.8×/2.0× |
| C106's bracket `[0.1736, 0.2011]` | ⚠️ **a projection-seed SPREAD, not a confidence interval** |
| ⭐ **the reframing** | **our trained arm is NOT CI-separated from its own matched-random null** (`lead_gap` 0/9, `ego_v0` 3/9) while the random arm **is** (9/9) ⇒ **signal vs no-signal on one rung, not a ratio** |

**Mechanism, verified from the weights:** random init has residual fraction **0.0002**, cos **1.0000**
against its own linear path — it **is** the raw-pixel linear map; the trained arm **has moved**
(LayerScale **70× init**, residual **0.38**). **Trajectory:** `ego_v0` **0.1346 → 0.0801** and
`lead_gap` **0.0123 → 0.0059** from step 2000 → 9000, **then flat**. ⚠️ No snapshot exists before
≈step 9100 (whole-filesystem probe) — the step-0 sweep is **UNAVAILABLE**, not unretrieved.
⭐ **New and monitorable from step 0:** the trained token field is **rank-collapsed** — **97.6 %** of
token-channel variance in one direction, effective rank **1.223 vs 67.1–68.2**. A **co-symptom**
(PCA-whitening lifts both arms ~3× and closes nothing), but observable where the headline is not.

### 12.3 ⛔ At three seeds the linear-readout ladder's substantive count is ZERO (C107)

Of **87** banked separated-FAILs, re-read with the C92 repair **and** the C97 degeneracy guard, on
**both** repair routes: **58 dead on all three seeds · 11 flip to PASS (the positive controls) · 9
survive at \|K1B\|/gt_sd ≤ 0.013 — all `ego_yawrate`, one on a random-latent null · 9 seed-unstable ·
⭐ 0 SUBSTANTIVE.** Reproduction gate **3465/3465** fields identical.

⚠️ **A repair can DESTROY apparent stability:** the defective instrument picked the same α on all 3
seeds for **132 of 165** rows, the repaired one for **42** — a **3.1× drop caused by the repair**,
because the biased fit had frozen the α sweep. *(Counter-column, published: max K1 seed spread is
larger on the incumbent, 4.239 vs 2.812.)*
⛔ **Trivial-proxy result: the ego-speed SCALAR matches or beats the 2,048-dim latent on 120 of 154
paired rows** — and wins **all four** of the latent's only 3-seed-stable guarded PASSes.
⇒ ⛔ **"The v6 latent reads scene density" is WITHDRAWN — it is ~80 % `v0`.**

### 12.4 ⛔ PARITY: 201 of 4,729 Alpamayo clips are ALREADY IN THE TRAIN CORPUS (C112)

**MEASURED** against `physicalai-train-e438721ae894` — and **the live trainer reads exactly that
cache** (`--v2-cache …e438721ae894-w120-256x640cyl`, from `/proc/25477/cmdline`, not from a doc).
mtimes prove genuine selection membership, not later contamination.

⛔ **THE 4.3 % FIGURE IS WRONG IN THE FLATTERING DIRECTION — CORRECTED 2026-08-18 (C113). QUOTE
78.21 %.** 4.3 % is 201 of 4,729 *catalogue records*, but **a split can only contain clips that
EXIST**: only **257** of the 4,729 have w120 video built, and **201 of those 257 are parity-train**
⇒ **the Alpamayo eval split buildable today is 78.21 % contaminated — REF-A I-JEPA SCALE (~80 %)**,
not "the same class at smaller scale". ⇒ **ROOT-CAUSE CLASS: a contamination rate quoted over the
CATALOGUE rather than over the BUILDABLE SET. The denominator that flatters is the one that is easy
to count** — the `df`-reports-the-cluster family.

⭐ **And the 201 do not COINCIDE WITH the aug120 perception corpus — they ARE it, exactly.**
`fused_aug120_v2_index.jsonl` and `v3` both hash to `80632f17…`, **byte-identical to the exclusion
list, 201/201 = 100 %** inside parity train. Mechanism is one line — `aug120_pipeline.py:53`,
`todo = (records ∩ w120_corpus) − done`, where the w120 corpus **is** the parity geometry sibling.
**The cohort was SELECTED FROM the train corpus.** ⇒ *A matching count between two sets is a prompt
to test set EQUALITY, not a coincidence to note.*

⭐⭐ **THE OTHER DIRECTION IS WORSE AND NOBODY WAS LOOKING AT IT: 6 of the 40 canonical val episodes
(15.0 %) are inside the Alpamayo record set** — verified two ways, 40/40 `clip_sha8` agreements
against an independently banked artifact. Not *"an eval split contains train clips"* but ⛔ **"a
train corpus is about to swallow the deployed val"** — the episode set behind **every published
open-loop number** (881 stride-8 windows). **Blast radius TODAY is ZERO** (nothing trains on those
labels) and the **trigger is already scheduled: the 4,472-clip build.** ⛔ **No existing guard
fires** — `parity.py` §9 checks a cache against *its own* corpus digest, and an augmentation corpus
is a different corpus by construction. ⇒ **Whoever runs that build MUST call
`parity.filter_train_clips()` first.**

✅ **BLAST RADIUS ON PUBLISHED NUMBERS: ZERO** — all **73** JSONs in `taniteval/results/` opened;
`registry.py:288` lists three eval corpora, **none Alpamayo**; every registry hit sits in §11
*PRODUCED DATASETS*. The aug120 numbers that exist are **label-quality only**, and the ADE numbers
near the word "Alpamayo" are the **Alpamayo-2-Super model** on the 290-clip OOD-val corpus — a
different corpus.

⚠️ **Root cause: non-overlap was ASSUMED FROM PROVENANCE ("different source ⇒ disjoint") rather than
COMPUTED FROM IDS — but it was UNANSWERABLE, not lazy.** The manifest carries only
`clip_id_sha256_sorted`, a whole-list digest: a set **identity** that cannot test one element, and
the ids are pod-only. ⇒ **The fix is the missing ORACLE, not a reminder:** `parity.py` §10/§10b plus
committed per-clip `sha256` sets — **membership exact, enumeration impossible**, §9's confidentiality
rule preserved. The mint **refuses to write** unless its source reproduces the committed corpus
digest, and the chain re-walks inside the repo with **no pod access**. **Derived, never hand-listed**
(C99/C105) — the next 4,472 clips need no update.

**Owning documents:** `…/incoming/2026-08-18-pooling-ladder-ER10/`, `…/2026-08-18-c106-adversarial/`,
`…/2026-08-18-ladder-3seed/`, `…/2026-08-17-thor-concurrency-pilot/`; classes **C92, C97, C100,
C103, C104, C106, C107, C109, C112** in `RETRACTION_LOG.md`.

## 13. v7-tiny WORLD-MODEL LINE — the collapse-elimination arms [TIER T0-DIAGNOSTIC throughout]

⛔ **NOTHING IN THIS SECTION IS A DRIVING NUMBER.** Every row is T0 (world-model diagnostic).
Per `EVAL_DOCTRINE.md` a T0 number may never be presented as driving performance — these arms have
no scorer, no planner and no closed loop.

⛔⛔ **`lead_gap_m` IS A SUPERSEDED TARGET (C150) — every `lead_gap_m` cell in §13 was computed against it.** MEASURED over all 25,790 labelled frames: **4.8 %** of frames have no lead and took an **80.0 m default**, carrying **59.9 % of the total variance**, and 80.0 is **not** a sentinel outside the data (real leads reach **180.84 m**). Split into `lead_present` + `lead_range_m` (lead-present frames only), **the repair flips verdicts**: `scale1` and `champ30k` "beat pixels" on the defective target (t 4.23 / **10.28**) and do **not** on the sound one, where **every arm of ours is worse than a constant** and only frozen DINOv3 carries range (+0.0336). ⭐ Cause identified (E-DEC-25): the readout's **128→64 projection**, not its pooling — at the identical grid the *unprojected* tokens read **+0.0719** against the readout's **−0.1611**. Cells below are kept as the historical record and **may not be quoted as range decodability**; `n_agents` is unaffected. Raw: `…/raw/leadsplit.json`, `…/raw/rowladder.json`.

### 13.0 ⭐⭐⭐ `rdw8p30k` — the settled readout geometry at PARITY SCALE — COMPLETE at 30,000 (2026-08-24)

**The arm that re-scopes the whole collapse campaign.** Identical architecture and identical
two-term objective to the 2k tiny screen `rdw8` — **no teacher, no external target, no PSG** —
run on the sacred corpus for 30,000 steps. Sourced from raw JSON
(`TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/raw/p30k_panel.json`).

| field | value |
|---|---|
| recipe | two-term (**O5 rollout + O6 SIGReg only**; O1/O2/O3 **off**), `--o5-k 1`, `--readout-grid 4 --readout-grid-w 8 --readout-dim 64`, `--sigreg-subspaces 32`, `--sigreg-slices 512`, `--o5-form l1`, `--w-o5 1.0 --w-o6 0.1` |
| encoder | `enc-dim 128`, `enc-depth 3`, `enc-heads 4` (0.97 M) · total **19,300,297** params, trainable 10,169,731 |
| train corpus | **`physicalai-train-e438721ae894-w120-256x640cyl`**, `require_parity` **true** |
| steps / batch | 30,000 / 8 · elapsed **29,758 s** on Jetson Thor · `summary.json` `done: true` |
| ckpt md5 | `6e382ebe721ba4b7df97e8305f695767` (verified BOTH sides of the pull) |
| gate | `INCONCLUSIVE` (`stage_gate.json`) |
| eval tier | **T0-DIAGNOSTIC** |
| estimator | paired LOEO over clips; probes fit lambda and PCA basis on the FIT split only; every panel carries a constant control at exactly 0.0000 and a raw-pixel floor |

| metric | `rdw8` (2k, 130 clips) | **`rdw8p30k`** | vs `rdw8` | vs RAW-PIXEL floor (paired) |
|---|---|---|---|---|
| participation val / held24 | 3.80 / 3.62 | **25.58 / 26.96** | — | — |
| predictor cos h=1 (z) | 0.0541 (3.99) | **0.6224 (30.55)** | — | — |
| speed | +0.2830 | +0.1482 | t −3.71 | +0.0326 (t 0.64) |
| `d_ego` | +0.3601 | +0.0963 | t −6.29 | +0.1744 (t 2.77) |
| `lead_gap_m` | −0.3290 | **+0.0063** | t 6.67, 22/24 | −0.0001 (t −0.01) |
| `n_agents` | −1.0407 | **−0.0180** | t 10.06, **24/24** | **+0.1976 (t 10.21, 24/24)** |

⭐ **Participation ~7×**, **predictor cos 11.5× the tiny arm and 3.3× the best previously measured
anywhere in the programme**, and **the first time any trained arm of ours beats the raw-pixel floor
on environment content** — with no external target of any kind.

⚠️ **Bounds, which are not small.** `n_agents` is still **marginally below the constant control**
(−0.0180 vs 0.0000) and **well below frozen DINOv3** (+0.2754); on this clip set the pixel floor
(−0.2156) is itself below the constant control, so clearing it is the weaker of the two bars.
`lead_gap_m` **+0.0063** is above the constant control and level with pixels, just under DINOv3
(+0.0294) — ⚠️ **on the SUPERSEDED target (C150); on the repaired one this arm reads `lead_range_m`
−0.0167, below a constant.** **Ego degrades** — which under E-DEC-17 is the least informative axis, because a frozen
RANDOM encoder reads the best speed of any arm.
⛔ **Confounded three ways** — corpus, steps and batch all moved. Direction unambiguous, cause not
isolated.

⭐ **Validity, checked rather than assumed (MEASURED pod-side):** **130 of 130** lead-corpus clips
are INSIDE the parity train corpus and **0** are in the val cache. ⇒ the ENV rows are **in-sample
for BOTH arms** — a valid contrast, **not** a generalisation claim — and the EGO rows are **held out
for both**. *(Clip ids are gated-confidential and live only on the pod; only the count was returned.)*

⛔⛔ **Consequence for the campaign:** E-DEC-7/14/17/18 were all measured on 2k tiny arms. Per
H-SCALE-2 that is valid for ARCHITECTURE and invalid for CAPABILITY LEVELS. The degeneracy
derivation stands as mathematics; the conclusions *"a frozen part is the only thing that works"* and
*"a teacher-free content source is the whole problem"* were drawn **inside the small-data regime**
and must be re-tested at parity before deciding the v7 design.

### 13.0d ⭐⭐⭐ `splitp30k` — frozen distilled encoder at PARITY, 30,000 steps — the first arm to beat the predictor floor AND carry high scene content

Frozen distilled encoder + trainable readout and predictor, O5+O6 only, `--o5-k 4`, PARITY corpus,
30,000 steps, batch 8. Raw: `…/2026-08-19-simwam-analysis/raw/gatec_panel.json` and
`gatec_meanpred.json`.

| field | value |
|---|---|
| train corpus | `physicalai-train-e438721ae894-w120-256x640cyl`, `require_parity` **true** |
| init / freeze | `init_from = distill_init.pt`, `--freeze-encoder` — **confirmed from the LIVE process args**, not the config's stage block (C146) |
| steps / elapsed | 30,000 / **27,041 s** on Jetson Thor · `summary.json` `done: true` |
| ckpt md5 | `4348cad27dbf1895654c40681d92ea97` (verified BOTH sides of the pull) |
| eval tier | **T0-DIAGNOSTIC** |

| metric (same 24-clip ENV set / 12-clip val) | `rdw8p30k` | **`splitp30k`** | `splitfrz10k` (130 clips) | frozen DINOv3 |
|---|---|---|---|---|
| **predictor `nrmse` vs constant floor** | **0.7903** ✅ | **0.8416** ✅ | 4.1757 ❌ | — |
| predictor cos h=1 (z) | 0.6224 (33.61) | 0.5481 (21.75) | 0.0007 (0.40) | — |
| participation val / held24 | **25.58 / 26.96** | 6.38 / 7.63 | 2.68 / 2.47 | — |
| `n_agents` | −0.0180 | **+0.3881** (t 28.21, 24/24) | +0.4156 | +0.2754 |
| `n_agents` vs raw-pixel floor (paired) | +0.1976 (t 10.21) | **+0.6037 (t 30.86, 24/24)** | +0.6313 (t 36.67) | +0.4911 |
| `lead_gap_m` | **+0.0063** | −0.0940 (t −6.22) | −0.0123 | +0.0294 |
| speed / `d_ego` | +0.1482 / +0.0963 | +0.0971 / −0.0399 | +0.2594 / +0.2939 | +0.4081 / +0.3238 |

⭐⭐⭐ **`splitp30k` is the FIRST arm in the programme to BEAT the constant-predictor floor AND carry high scene content.** It joins `rdw8p30k`, `scale1` and `champ30k` as the fourth arm ever to beat the predictor floor, and its `n_agents` **+0.3881 is above frozen DINOv3 (+0.2754)** and **30.86σ above raw pixels**. ⇒ **E-DEC-20's "no arm wins both axes" is SUPERSEDED.**

⭐ **AND THE FROZEN-FIELD PREDICTOR COLLAPSE WAS ALSO A SMALL-DATA ARTEFACT.** The same frozen encoder at 130 clips (`splitfrz10k`) gives `nrmse` **4.1757 — worse than a constant**; at parity it gives **0.8416 — beating it**. Freezing does **not** destroy the predictor; the 130-clip corpus did. That is the fourth conclusion of this campaign re-scoped by scale (after E-DEC-7/14/17/18).

⚠️ **THE COSTS ARE REAL AND ARE NOT ROUNDED AWAY.** (1) Its predictor is **worse than `rdw8p30k`'s** (0.8416 vs 0.7903) — freezing still costs prediction quality. (2) **`lead_gap_m` goes the WRONG WAY**: +0.0063 → **−0.0940**, below the raw-pixel floor (t −3.91) — ⚠️ on the SUPERSEDED target; on the repaired one it reads `lead_range_m` **−0.1611**, the worst of any arm, so the direction holds and the magnitude is larger (C150 / E-DEC-24) — **the two environment targets dissociate again**, so `n_agents` alone must never be quoted as "environment". (3) **Participation drops 4×** (25.58 → 6.38) and the pre-registered kill-gate REJECTS the arm on rank; that is weighed against C131/C135/E-DEC-7 (rank does not track capability) but it is not dismissed. (4) Ego degrades, `d_ego` **−0.0399 below the constant control** — least informative under E-DEC-17 (ego is free) but recorded. (5) ⛔ **It is TEACHER-DEPENDENT AT INIT**, so it does **not** satisfy the PI's "clear preference without pretrained labels"; the teacher-free question is unchanged and open.

⭐ **VALIDITY:** completed 30,000 steps, `summary.json` `done: true`, elapsed 27,041 s; ckpt md5 `4348cad27dbf1895654c40681d92ea97` verified on BOTH sides of the pull; `--freeze-encoder` + `init_from=distill_init.pt` + `require_parity` confirmed from the LIVE process args, not the config's stage block (C146). ⚠️ ENV rows are IN-SAMPLE for all parity arms (130/130 lead clips are inside parity train, 0 in val — MEASURED pod-side); EGO rows are held out.

### 13.0b ⭐⭐ `splitfrz10k` — frozen distilled encoder, 10,000 steps — the programme's best SCENE-CONTENT carrier, with a predictor worse than a constant

Frozen distilled encoder + trainable readout and predictor, O5+O6 only, `--o5-k 4`,
130-clip corpus, 10,000 steps. Raw:
`…/2026-08-19-simwam-analysis/raw/depth10k_panel.json` and `meanpred_all.json`.

⭐ **The freeze is verified by CONTENT at BOTH checkpoints**: all 41 `encoder.*`
tensors byte-identical to the distill init, **max|Δ| = 0** (2k and 10k alike),
while every trainable arm moved 0.08–0.33. *(The arm's own `config.json` reports
`'encoder': {'trainable': 972032, 'frozen': 0}` — that is the STAGE PLAN, not the
`--freeze-encoder` override. See **C146**.)*

| metric (24-clip in-sample ENV set) | `splitfrz` 2k | **`splitfrz10k`** | `rdw8p30k` | frozen DINOv3 |
|---|---|---|---|---|
| `n_agents` | +0.4035 | **+0.4156** (t 10.65, 24/24) | −0.0180 | +0.2754 |
| `n_agents` vs raw-pixel floor (paired) | +0.6191 (t 36.27) | **+0.6313 (t 36.67, 24/24)** | +0.1976 (t 10.21) | — |
| `lead_gap_m` | −0.0172 | −0.0123 | +0.0063 | +0.0294 |
| speed / `d_ego` | +0.2465 / +0.2878 | +0.2594 / +0.2939 | +0.1482 / +0.0963 | +0.4081 / +0.3238 |
| participation val / held24 | 2.28 / 2.17 | 2.68 / 2.47 | **25.58 / 26.96** | — |
| **predictor nrmse vs constant floor** | 0.9902 / 0.9982 | **4.8321** / 0.9982 | **0.7845** / 0.9978 | — |

⭐ **Highest `n_agents` in the programme — above `rdw8p30k` AND above a frozen
DINOv3 that never saw our data, on the same clips, 36σ over raw pixels — and it
HOLDS and improves over 5× more training,** with ego intact.
⛔ **Its predictor is WORSE THAN A CONSTANT** (nrmse 4.83 against a 0.9982 floor),
with **82 %** of its output a fixed offset and its h=1 head **grown 2.676×** — so
it is **miscalibrated, not dead**. ⚠️ The earlier reading "the predictor dies" is
superseded by this sharper one; the cos-only figure (0.1872 → 0.0007) was quoted
before the constant floor existed (**C149**).

⇒ **Content and prediction DISSOCIATE across arms, and no arm in the registry wins
both.** `splitp30k` (frozen distilled at PARITY scale) is the arm that tests
whether both are obtainable at once.

### 13.0c ⭐⭐⭐ THE PREDICTOR CENSUS — exactly three arms in the programme beat a constant predictor

> ⛔⛔ **READ THIS BEFORE QUOTING ANY `nrmse` BELOW — E-DEC-30 (MEASURED
> 2026-08-24).** Every `nrmse` in this section is **UNCHANGED TO FOUR DECIMALS
> when the predictor's actions are replaced by a sequence from a random other
> moment**: `rdw8p30k` **0.7845 → 0.7845** (prediction moves 0.77 %),
> `scale1` 0.8200 → 0.8199, `splitp30k` 0.8683 → 0.8680.
> `…/2026-08-24-action-conditioning-and-heldout/raw/nrmse_shuf.json`
>
> ⇒ **BEATING THE CONSTANT FLOOR IS NOT EVIDENCE OF AN ACTION-CONDITIONED WORLD
> MODEL.** These arms beat the floor by TEMPORAL EXTRAPOLATION — predicting that
> the scene continues — which is a real and non-trivial capability and is
> correctly recorded here. It is **not** the capability of answering *"what
> happens to the scene if I brake?"*, and this census must never be cited for
> that. The full channel factorial (`raw/actchan.json`, 444 windows, 3 arms,
> positive control passing on all three) puts the action pathway at **2–9 %** of
> the latent pathway, with a hard-left→hard-right sign flip moving `rdw8p30k`
> by **1.1 %**.
>
> ⚠️ **The C149 floor did its job and is not in question** — it is why "≈ a
> constant" is 16 arms rather than a ranking of noise. The gap C149 did not
> close is that a *constant-predictor* floor cannot distinguish extrapolation
> from action-conditioning; only a **shuffled-action** control can, and that
> control did not exist until E-DEC-30. ⇒ **Any future row in this census
> carries `nrmse` AND `nrmse_SHUFFLED` side by side.** A row with only the
> former is incomplete in the same way an ADE-only eval is incomplete.
>
> The fix under test is **O11-CF** (`train_v6_staged.py`), pre-registered with
> four outcomes in `PREREG_O11_COUNTERFACTUAL_ACTION.md`.

`meanpred_all.json` — 30 finished arms re-scored against the floor **C149** added
(`nrmse = ||d̂ − t|| / ||t||` versus a dataset-mean-delta predictor;
`nrmse_zero` = 1.0 by construction), h=1, 10 held-out val clips. `rdw8s30k`
excluded: its checkpoint is a mid-run periodic save.

| verdict | n | arms (nrmse) |
|---|---|---|
| **BEATS a constant** | **3** | `rdw8p30k` **0.7845** · `scale1` **0.8200** · `champ30k` **0.9348** |
| ≈ a constant | 16 | every healthy 2k arm — `o5k4` 0.9988 · `o5k8` 0.9984 · `o5k16` 0.9984 · `splitfrz` 0.9902 · `postrain10k` 0.9967 |
| **WORSE than a constant** | 11 | all four O1 (`o1sgw3` **22.72**, `o1sg` 17.47) · all five PSG (`psgenc01` **32.38**) · `frzrand` 3.98 · `splitfrz10k` 4.83 · `rdw8o3` 1.07 |

⭐ **The separation is exact and was checked programmatically, not by eye:** every
arm that beats the floor is **30,000 steps on the PARITY corpus**; every
30k-parity arm beats it; **no sub-30k arm ever does** (`arms <30k that beat` = **[]**,
`30k arms that do not` = **[]**). `champ30k` and `scale1` do it at **batch 4**.

⚠️⚠️ **This census CANNOT separate steps from data** — all three winners are 30k
*and* parity, always together. `rdw8s30k` (30k on 130 clips) is the arm that
separates them and its verdict is pre-registered.

⭐ **A single-number screen falls out:** `mean_fraction_of_prediction` orders the
three classes almost perfectly — BEATS **0.0719 / 0.1148 / 0.1790**, ≈constant
~0.34–0.82, WORSE 0.73–0.87. **A predictor whose output is mostly a fixed offset
is the failure mode**, and this detects it without any baseline arm.

⛔ **Consequence for every row in §13:** a predictor `cos` with a permutation `z`
establishes *there is signal*, never *the predictor works*. Rows written before
2026-08-24 quote the former. **The `--o5-k` depth ranking is entirely inside the
floor** (0.9988 / 0.9984 / 0.9984 against 0.9981 / 0.9975 / 0.9985) and its
ordering does not even match the cos ordering it was drawn from.

### 13.1 `champ30k` — two-term (LeWM-style) + k=1 + 32 SIGReg subspaces — COMPLETE at 30,000

The first arm in the programme whose predictor **beats the HOLD baseline at all**, and the arm
E-PROOF-1 was built to adjudicate. Sourced from raw JSON
(`TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/raw/e_proof1_champ30k.json`).

| field | value |
|---|---|
| recipe | two-term objective (**O5 rollout + O6 SIGReg only**; O1/O2/O3 **off**), `--o5-k 1`, `--sigreg-subspaces 32`, `--sigreg-slices 512`, `--o5-form l1`, `--w-o5 1.0 --w-o6 0.1` |
| steps | 30,000 |
| eval corpus | `physicalai-val-0c5f7dac3b11`, **12 clips**, parity `true` |
| eval tier | **T0-DIAGNOSTIC** |
| estimator | episode-cluster bootstrap; probes fit lambda and PCA basis on the FIT split only |

**A - RANK** — participation **6.489** (n=1680, d=2048, top-8 share 0.8805).
⛔ Recorded as FAIL against `O6_PARTICIPATION_FLOOR = 8.56`, **and that FAIL is NOT admissible as of
2026-08-23**: §13.3 shows no live instrument reproduces 8.56, and the reference is d=1024 against
this arm's d=2048. **Status of the rank gate: UNDECIDABLE, not failed.**

**B1 - PREDICTOR vs HOLD** — the positive result, and it is narrow:

| horizon | explained movement | CI95 | beats HOLD |
|---|---|---|---|
| h=1 (0.1 s) | **+0.1303** | [+0.1079, +0.1512] | **yes — the first in the programme** |
| h=2 (0.2 s) | **-0.0** | [-0.0, +0.0] | no |
| h=4 (0.4 s) | **-0.0** | [-0.0, -0.0] | no |

⚠️ **The h>=2 rows are EXACTLY zero with a ZERO-WIDTH CI, which is a signature, not a small effect.**
`EM = 1 - ||z_hat - z_plus||^2 / ||z - z_plus||^2` is identically 0 iff `z_hat = z`, i.e. **the
predictor returns its input unchanged**. Read together with B2 below, the arm behaves as an
**identity map beyond one tick** — and that is now **MEASURED, not inferred: H-PROOF-2 SUPPORTED**
(`…/2026-08-19-simwam-analysis/raw/h_proof2_identity.json`, 8 held-out val clips).
`mean||z_hat(h) - z_last|| / mean||z_true(h) - z_last||` reads **0.0002 at h=2** (0.008 against a
true motion of 46.19) and **0.0002 at h=4** (0.008 against 49.22), while the action-sensitive
control arm `fixed` reads **0.207 / 0.154** on the same probe -- three orders of magnitude larger,
which is what shows the probe measures the arm rather than the harness.
⇒ read with B2, **the two-term recipe learned the TRIVIAL SOLUTION: copy the input, ignore the
actions.** Beating HOLD at h=1 is its only non-trivial behaviour.
⚠️ `fixed` is not thereby a healthy predictor (0.15-0.21 is still near-identity; its h=1 ratio of
30x overshoots because its own latent barely moves, 0.093/tick against champ30k's 42.6). The
contrast is decisive about IDENTITY, not a certificate for `fixed`.

**B2 - COUNTERFACTUAL DIVERGENCE (anti-echo)** — ⛔ **THE "ACTION-BLIND" VERDICT IS RETRACTED
(C137, 2026-08-23). The predictor IS action-conditioned; the metric was confounded.**

The original statistic divided by **true per-tick latent movement**, which is a property of the
ARM, not of the predictor, and spans **0.088 to 41.3 across our arms — a factor of 468**. Two
predictors with identical action sensitivity therefore score up to 468x apart, and champ30k's
"0.0000" meant only *its delta is small in absolute terms*.

Re-measured scale-free — normalising by the predictor's OWN delta,
`rel = ||z_hat(a x100) - z_hat(a)|| / ||z_hat(a) - z_t||`
(`…/2026-08-19-simwam-analysis/raw/action_response_scalefree.json`, 6 val clips):

| arm | relative action response | old (confounded) divergence |
|---|---|---|
| **champ30k** | **0.7194** | 0.0000 |
| `lewm` | 0.5426 | 0.000 |
| `lewm_o1` | 1.1640 | 1031.8 |
| `lewm_o1_detach` | 0.7806 | 254.5 |
| `fixed` | 0.8358 | 925.3 |

**Every arm is substantially action-conditioned. No arm ever ignored actions.**

⭐ Corroborated independently at the WEIGHT level
(`…/raw/wiring_audit.json`): `FiLM.to_scale_shift` is **exactly zero at init**, so its norm is the
entire learned action pathway — champ30k's is **10.92**, the LARGEST of all five arms (2.15x
`fixed`'s 5.08), and its residual heads grew **55.4x** past the down-scaled init. A starved action
path was the first hypothesis and the weights refute it.

⭐ **THE REAL DEFECT, and it is sharper than the retracted one: OUTPUT MAGNITUDE and HORIZON DECAY.**
champ30k's delta is **0.264x the true movement at h=1** (~4x too small) and **0.0002x at h>=2**
(the identity map, §13.1 B1 / H-PROOF-2, which never used this metric and is unaffected). No arm is
magnitude-calibrated: `lewm` **0.020x**, `lewm_o1_detach` **3.57x**, `lewm_o1` **17.6x**,
`fixed` **25.8x**. ⇒ the lever is the delta's SCALE and its decay with horizon, NOT the action
wiring. This also reframes `RESIDUAL_HEAD_INIT_SCALE = 1e-3`: it removed a 1000x-too-large delta and
left us ~4x too small at h=1.

⛔ **ROOT-CAUSE CLASS (C137): a ratio whose DENOMINATOR is not held fixed across the things being
compared** — the same scope family as §13.3's floor and as `df` / `free` / cgroup / `step_s`.

**C - DECODABILITY** — does not clear the raw-pixel floor, and the point estimates are not even
significantly positive (n=700, d=128 PCA, lambda on the FIT split):

| target | latent | pixel floor | frozen DINOv3 | constant control |
|---|---|---|---|---|
| speed | +0.2557 **[-0.4179, +0.3855]** | +0.1976 [-0.1099, +0.3440] | **+0.5045** [+0.1982, +0.5509] | 0.0000 [0, 0] |
| `d_ego` | +0.2256 **[-0.3662, +0.3597]** | +0.1820 [-0.0875, +0.2984] | **+0.5588** [+0.2137, +0.6285] | 0.0000 |
| yaw-rate | -0.0102 [-0.0753, +0.0002] | +0.0049 [-0.1148, +0.0160] | -0.0005 | 0.0000 |

⚠️ **The latent's CI straddles zero on every target**, so "the latent decodes speed at +0.26" is
NOT a supported statement; the honest reading is *indistinguishable from the pixel floor and from
zero*. The **constant control reads exactly 0.0000 with zero width** on all three, which is what
makes the rest of the panel trustworthy. Frozen DINOv3 on the **same clips, same probe** separates
cleanly on two of three, so the target IS decodable and this representation does not carry it.

**VERDICT — `PASS_ALL: false`, and the gate-by-gate reading CHANGED on 2026-08-23 (C137):**

| gate | status |
|---|---|
| **A** rank | **UNDECIDABLE** — the 8.56 floor is not reproducible and is measured at half this arm's `d` (§13.3). Neither pass nor fail. |
| **B1** beats HOLD | **PASSES at h=1 only** (+0.1303 [+0.1079, +0.1512]); identity map at h>=2. |
| **B2** action-conditioned | **PASSES** — relative action response **0.7194**. The earlier FAIL was the retracted metric. |
| **C** decodable above pixels | **SPLIT — and the split is the finding (E-DEC-1, 2026-08-23).** `z_op` (AFTER the 4×4 readout) does NOT beat the pixel floor (`z_op − pix` t = 0.71 on speed). But the **ENCODER TOKENS DO, decisively**: `enc − pix` **+0.2535 (t = 3.68), 12/12 episodes** on speed and **+0.3603 (t = 4.21), 11/12** on `d_ego`, and the encoder is statistically indistinguishable from frozen DINOv3 on `d_ego`. ⇒ **the trunk learned; the readout discards it.** The original FAIL judged `z_op` and concluded about the encoder. |

⇒ the arm proves the collapse can be *prevented* (H-RANK-11) and that the predictor *is* action-
conditioned, **without proving the representation learned anything a raw pixel does not already
carry**. The binding failure is **C (decodability)**, supported by the magnitude/horizon defect in
B1-B2: a correctly-directed delta at ~1/4 the right size one tick out, and none at all thereafter.

### 13.2 The O1 coupling — rank versus action-response strength (H-RANK-18, MEASURED 2026-08-23; divergence column superseded by C137)

Matched 2k-step arms, same 130-clip corpus, same instrument, `.../raw/h_rank18_readout.json`:

| arm | terms | participation (held24 / val) | divergence x100 actions | v0 x3 |
|---|---|---|---|---|
| `lewm` | O5+O6 | **4.43** / 3.42 | **0.000** | 0.000 |
| `lewm_o1` | O5+O6+**O1** | **2.94** / 2.59 | **516.6** | 388.5 |
| `fixed` | all six | 2.94 / 2.61 | 474.9 | 1159.8 |

**O1 is the single term that BOTH buys action-conditioning AND causes the collapse.** Adding it to
the collapse-free recipe restored divergence from exactly zero *and* returned participation to the
six-term value to three significant figures. **H-RANK-18 REFUTED** — the two failure modes are
coupled, not separable by adding O1 at full weight.
⚠️ These are **NON-PARITY 130-clip arms at 2k steps**: the admissible read is the *shape* across
matched arms, never an absolute number, and none is comparable to §13.1's parity numbers.

### 13.3 ⛔ THE PARTICIPATION FLOOR IS NOT REPRODUCIBLE, AND A SCALAR FLOOR IS NOT VALID (2026-08-23)

Frozen DINOv3 ViT-L/16, patch tokens mean-pooled per frame, through **`spectrum_report` itself**
(the gate's own function), **n=1440 in every row**:

| sample | participation |
|---|---|
| 12 physicalai-val clips — *the corpus `8.56` is sourced to* | **5.756** |
| 130-clip lead corpus — *the corpus `40.77` is sourced to* | **20.228 +- 0.327** |
| 130-clip lead corpus, full n=5617 | **20.516** |

**Neither published number survives.** The **3.51x** between rows one and two is **episode diversity
alone** — same encoder, same d=1024, same n=1440, same instrument, only the clips differ
(**H-RANK-23 SUPPORTED**). Sample size is **not** the confound (**H-RANK-21 REFUTED**): a synthetic
control with closed-form truth reads **0.974x / 1.002x** of truth at n=1440 for the concentrated
spectra we actually have, and the real bank is flat in n (19.29@360 -> 20.52@5617).

A participation value is comparable only at matched **CORPUS**, matched **EPISODE COUNT** and
matched **AMBIENT DIMENSION**. ⛔ **No arm may currently be failed on the O6 participation clause**,
and ⛔ **the reverse claim is equally inadmissible** — champ30k is d=2048 and every banked reference
is d=1024, so "6.489 beats 5.756" is not supported either. Recorded in code beside the constant
(`stack/tanitad/models/v6.py`, `O6_PARTICIPATION_REFERENCES`) and pinned by
`stack/tests/test_participation_floor_provenance.py`.
**ROOT-CAUSE CLASS: a number true for one scope, quoted where that scope does not apply** — the
`df` / `free` / cgroup / `step_s` family, this time inside a gate that decides.

**Raw artifacts** — all under `TanitAD Research Lab/Architecture & Inference/Research/`:

* `…/2026-08-19-simwam-analysis/raw/e_proof1_champ30k.json` — the §13.1 battery
* `…/2026-08-19-simwam-analysis/raw/h_rank18_readout.json` — the §13.2 O1 coupling
* `…/2026-08-19-simwam-analysis/raw/h_rank16_floor_reconcile.json` — DINOv3 on the 130-clip corpus, swept in n
* `…/2026-08-19-simwam-analysis/raw/h_rank16_floor_valclips.json` — DINOv3 on the 12 val clips (the apples-to-apples floor)
* `…/2026-08-19-simwam-analysis/raw/h_rank21_partic_nbias.json` — the synthetic finite-n control with closed-form truth
