# NAVSIM port of v8F-mini -- scope for the PI's compute decision

Scoping only. 2026-10-25. No GPU, no download, nothing staged in the repo. Repo tip used: `f547c11b`.
Evidence classes: **PUBLISHED** (primary, key + table + page) · **PUBLISHED-SECONDARY** (read in another paper's table / a docs page through a summariser) ·
**MEASURED** (ours, artifact path) · **INHERITED** (another agent's document, not re-verified) · **ESTIMATED** (my arithmetic, assumptions named) · **HYPOTHESIS**.
Library keys are `lib:<arXiv id>`; text extracted from the banked PDFs (sha256 checked against the tip's `library.json`) is in `papertext/<id>.txt` beside this file.

## 0. What changes the answer: most of the NAVSIM machinery already exists (MEASURED / INHERITED, all at tip `f547c11b`)

The programme has already scored models on NAVSIM, on both generations. A v8F-mini port is mostly a *model adapter* plus a *navtrain feature/label build*, not a new harness.

| asset | state | where |
|---|---|---|
| NAVSIM **v1.1 navtest PDMS harness** (devkit `3e8291b`) | built, controls pass, full-split runs done: CV 20.65, STOP 61.82, HUMAN 94.55 (12,146 tokens, 136 logs) | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/RESULT.md` (MEASURED) |
| navtest **frame bank** (3-camera L0/F0/R0 stitch to 256x640 cylindrical, same geometry as the PhysicalAI "wide120" frames) | 12,146 scenes from 17,872 unique frames, sha-verified, **8.18 GiB**; also on the dev box at `E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank` (8.3 GB by `du`) | same RESULT.md section 1.4; builders `code/build_navtest_frames.py`, `bank_finalize.py` |
| navtest **metric cache** | built (137 entries under `E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest`); W3 estimated 5-6.5 h to build, 15-25 min per scored arm | same RESULT.md (cache cost ESTIMATED from a 20-token smoke) |
| NAVSIM **v2 warmup / navhard two-stage** harness (devkit `0a380a9`), metric caches | built (`E:/Archive/devbox-C/navsim/exp/metric_cache_{warmup,navhard}_two_stage`); floors measured: navhard STOP 29.85 [27.37, 32.53] vs CV 11.48 | `Benchmarks & Eval/LEADERBOARD.md` section 0E.1 (MEASURED) |
| a **bridge from a TanitAD planner to NAVSIM** (declared-input seam, ego-withholding controls K5/K6, frame/command/curvature round-trips K1-K9) | built for refcv4b, zero-shot: warmup stage-2 A1 0.4670 vs CV 0.3971 but **STOP 0.5212**; navtest **58.97 < STOP 61.82** | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/{RESULT.md,code/tanitad_navsim_bridge.py}`; navtest in `.../2026-09-19-navsim-v1-navtest/RESULT.md` headline 2 |
| navtrain pixel pull + training pipeline | REFe (frozen DINOv3 ViT-L/16 + LoRA, 4 cameras, 322 M total / 19 M trainable) was trained on navtrain on an A40 pod and scored on navtest: **PDMS 83.24 -> 88.82** (P2 seed 0), 89.15 (seed 1) -- with 7-D ego status and a route-oracle goal; the A40 pod was unreachable at the last probe (2026-10-11, volume fate unknown), checkpoints and banks are on HF / the dev box | `LEADERBOARD.md` section 0E.3; `Research/2026-09-20-refe-plan/{POD_HANDOFF.md,TRAINING_TIME.md,STOP_CONSEQUENCES_2026-10-11.md}` |
| dev-box disk | **E: has 102 GB free of 3.7 TB** (98 % used); C: has about 1.5 GB | `df` this session (MEASURED) |

Consequence: the raw navtrain pixels must be handled on a pod (449 GB will not fit on E:); only features (about 6-12 GB) come back.

## 1. What NAVSIM provides

| item | value | class / source |
|---|---|---|
| scenes | navtrain **103,288** (103k), navtest **12,146** (12k); built from OpenScene (nuPlan at 2 Hz), filtered so a constant-velocity agent scores < 0.8 and the human scores >= 0.8 | PUBLISHED `lib:2406.15349` p.5; counts 103,288 / 12,146 also in `lib:2608.07468` p.6 |
| sizes (logs / sensors) | navtrain 14 GB / **445 GB (300 GB without the history archives)**; navtest 983 MB / 223 GB; navhard_two_stage 892 MB / 31 GB; warmup_two_stage 27 MB / 1.2 GB; private_test_hard_two_stage 14 MB / 11 GB | PUBLISHED-SECONDARY: GitHub `docs/splits.md` read by WebFetch 2026-10-25; the paper says "450 GB" (`lib:2406.15349` p.5); the programme measured navtrain_current 304.5 GB + navtrain_history 144.5 GB = 449.0 GB (INHERITED, `POD_HANDOFF.md`) |
| camera-only vs lidar | the navtrain archives bundle cameras and merged lidar in one gzip stream; MEASURED on one shard (`navtrain_current_25`): 4 cameras 27.5 %, the other 4 cameras 28.9 %, lidar `.pcd` 43.8 % of bytes. **Lidar can be skipped on disk, not in transit.** navtest cameras ship as separate camera-only archives (32 shards, 127.9 GB, MEASURED, all sha256 == HF ETag) | INHERITED `POD_HANDOFF.md`; MEASURED W3 RESULT section 1.4 |
| cameras | 8 cameras 1920x1080 (+ merged 5-lidar cloud); LTF's only input is the stitched front 3-camera view | PUBLISHED `lib:2406.15349` p.5, p.7 |
| history | current + 3 past frames at **2 Hz** (paper: 1.5 s span; docs page says "2 seconds") | PUBLISHED `lib:2406.15349` p.5; docs `agents.md` (SECONDARY) |
| ego status | velocity, acceleration (+ pose) in the local frame, and a discrete **driving command**: left / straight / right (+ "unknown" used to filter training data); index 0 = left (+y), 2 = right (-y) | PUBLISHED `lib:2406.15349` p.4; index order MEASURED (K9: 99.6 % / 100 % of 283 turning windows each), `navsim-refcv4b-bridge/RESULT.md` |
| command is a route-level oracle | derived from the expert's driven path at roadblock granularity; "no route-following skill may be claimed from a command-conditioned row" | MEASURED / PUBLISHED-CODE, `LEADERBOARD.md` section 0E.1 (W2 D-NAVSIM-ROUTE-LEAK) |
| speed limit | **not an input** (agent input = sensors + ego pose / velocity / acceleration + command) | PUBLISHED-SECONDARY, docs `agents.md`. A lane speed limit may exist in the nuPlan map API (map-derived, not vision-only): UNVERIFIED here |
| trajectory expected | local-frame poses (x, y, heading); the scorer simulates **4 s at 10 Hz** (LQR + kinematic bicycle); devkit default sampling is 8 poses at 0.5 s | PUBLISHED `lib:2406.15349` p.4 (4 s, 10 Hz); 8 poses at 2 Hz PUBLISHED-SECONDARY in `lib:2608.07468` p.6 |
| metric v1 | PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) / 12, x100; DDC logged with weight 0 | PUBLISHED `lib:2406.15349` p.4; MEASURED C1 identity (max abs diff 0.0) |
| metric v2 | EPDMS = NC x DAC x DDC x TLC x (5 TTC + 5 EP + 2 HC + 2 LK + 2 EC) / 16, with a "human also fails" filter; **two-stage** (stage 1 real scene, stage 2 synthetic follow-ups weighted by a Gaussian kernel on the stage-1 endpoint, product of the two); official v2 column = **navhard two-stage** | PUBLISHED-SECONDARY (docs `metrics.md`); `lib:2506.04218` p.7-8 |
| rules | navtest / navhard / warmup may not be used for training a challenge submission; leaderboard entries must release code and weights | PUBLISHED-SECONDARY (docs `splits.md`); `lib:2406.15349` |
| v1 vs v2 devkit | v1 PDMS must come from the v1.1 branch (`3e8291b`); `main`/`0a380a9` computes EPDMS; never mixed | MEASURED / INHERITED `LEADERBOARD.md` section 0E.3 |
| two scores that are NOT the official v2 column | "EPDMS on navtest" (single stage) is what many method papers print; the pseudo-simulation paper discourages it | PUBLISHED `lib:2506.04218`; web search 2026-10-25 (SECONDARY) |

**Floors that every row must sit beside (MEASURED, our runs, 12,146 tokens, log-cluster bootstrap over 136 logs):** STOP (all-zero plan) **61.82 [60.70, 63.08]**, CV 20.65 [19.20, 22.22], human 94.55. Mechanism: the split keeps scenes the human passes, so a stopped car holds NC/DAC/TTC near 1 (5/12 of PDMS free) and EP = 1 by rule on 8.28 % of tokens. Published blind ego-status MLP: 65.6 (PUBLISHED `lib:2406.15349` p.7).
Navhard two-stage: STOP 29.85 vs CV 11.48 (MEASURED, `LEADERBOARD.md` section 0E.1).

## 2. Published baselines (navtest, NAVSIM v1 PDMS x100 unless stated)

"Camera" is **the authors' claim**; the same method is labelled differently across papers (flagged). Params / compute only where published.

| method | PDMS | inputs as claimed | params / training compute | source (primary unless noted) | class |
|---|---|---|---|---|---|
| Constant velocity | 20.6 | ego status | - | `lib:2406.15349` Tab. 1 p.7 | PUBLISHED |
| Ego-status MLP | 65.6 | blind (ego status incl. command) | - | `lib:2406.15349` Tab. 1 p.7 | PUBLISHED |
| UniAD | 83.4 | camera (6-8 surround), video | 125 M (Fig. 1 of `lib:2507.17596`) | `lib:2406.15349` Tab. 1 | PUBLISHED |
| **LTF (Latent TransFuser)** | 83.8 | **camera only (front 3-cam stitch)** | - | `lib:2406.15349` Tab. 1; text p.7 "camera-only LTF" | PUBLISHED || TransFuser | 84.0 | camera + lidar | 56 M; 1 GPU x 1 day (navtrain) | `lib:2406.15349` p.7; params `lib:2507.17596` Fig. 1 | PUBLISHED |
| PARA-Drive | 84.0 | surround cameras | 80 GPUs x 3 days | `lib:2406.15349` p.7 | PUBLISHED |
| LAW / World4Drive | 83.8 / 85.1 | `C & L` in `lib:2601.22032` Tab. 2; "camera-only" in `lib:2601.05083` Tab. 1 -- **labels conflict** | 21 M encoder, ~20 h video each | `lib:2601.22032` Tab. 1 p.4, Tab. 2 p.8 | PUBLISHED |
| Epona | 86.2 (Tab. 1 of the same paper prints 86.1) | camera | 1.1 B encoder, 128 h | `lib:2601.22032` Tab. 1, Tab. 2 | PUBLISHED |
| PRIX | 87.8 | **camera-only, raw pixels** | 37 M (variant "Shared SA 512") | `lib:2507.17596` ablation table; Fig. 1 | PUBLISHED |
| DiffusionDrive | 88.1 | camera + lidar (`lib:2601.22032`), ResNet-34 | 60 M, 45 FPS on a 4090 | `lib:2411.15139` Tab. 2 p.7 | PUBLISHED |
| WoTE / DIVER | 88.3 / 88.3 | camera + lidar | - | `lib:2504.01941` Tab. 1; `lib:2507.04049` | PUBLISHED |
| **Drive-JEPA, perception-free** | **89.0** | **front camera only**, 2 frames 512x256, ego status + command | ViT-L 307 M encoder, 330 h video pre-training (8 x H800, ~3 days); planners 2 x A30, 20 epochs | `lib:2601.22032` Tab. 1 p.4, Tab. 2 p.8, Sec. 4.2 | PUBLISHED |
| AutoVLA / DriveVLA-W0 | 89.1 / 90.2 | camera (1x for W0) | VLM-scale | `lib:2506.13757`; `lib:2510.12796` | PUBLISHED |
| ReCogDrive | 90.8 | camera (1x), VLM | InternVL3-2B/8B fine-tuned (Fig. 1 labels) | `lib:2506.08052` main navtest table (row "ReCogDrive (ours)") | PUBLISHED |
| DriveFuture | 90.7 | front + rear camera streams | 8 x RTX 5090 | `lib:2605.09701` abstract p.1, Tab. 1 | PUBLISHED |
| Hydra-MDP++ | 91.0 | camera (ResNet-34 / V2-99 / ViT) | - | `lib:2503.12820` | PUBLISHED |
| DiffusionDriveV2 | 91.2 | ResNet-34 (C + L per `lib:2604.03581`) | 8 x L20, 10 epochs RL | `lib:2512.07745` abstract | PUBLISHED |
| **SimWAM** | **91.5** | **single front camera**, 384x672 | Wan2.2-5B video expert + 1024-d action DiT; 100 epochs | `lib:2608.07468` abstract, Sec. 5.1 | PUBLISHED |
| iPad | 91.7 | camera, ResNet-34 (own table); `lib:2601.22032` Tab. 2 prints 91.1 for ResNet-34 and 91.7 for ViT-L -- conflict, primary used | - | `lib:2505.15111` main navtest table (row "Ours, Camera, ResNet-34") | PUBLISHED |
| SparseDriveV2 | 92.0 | ResNet-34 | - | `lib:2603.29163` abstract | PUBLISHED |
| Adaptive-WAM | 90.8 (1 trajectory) / **92.6** (64 proposals) | **front camera, no lidar** | Wan2.2-5B backbone, 170 ms on an A100 | `lib:2608.06008` abstract | PUBLISHED |
| Centaur | 92.6 | listed camera-only in `lib:2601.05083` Tab. 1; test-time training | - | `lib:2503.11650` abstract | PUBLISHED |
| DriveSuprim | 93.5 | camera (ViT-L, per `lib:2601.22032` Tab. 2) | - | `lib:2506.06659` abstract | PUBLISHED |
| DrivoR (navtrain / trainval / +134k synthetic) | 93.1 / 93.7 / **94.6** | **4 cameras, camera-only**; DINOv2 ViT-S + LoRA | ~40 M; 4 x A100, 10 epochs (25 for the competition split) | `lib:2601.05083` Tab. 1 p.5, Sec. 4 | PUBLISHED |
| RAP-DINO | 93.8 | camera; trained on 10x navtrain | ViT-H | `lib:2510.04333` Tab. 2; `lib:2601.05083` Tab. 1 | PUBLISHED |
| Drive-JEPA, full framework | 93.7 | front camera only (same inputs), + multimodal trajectory distillation | as above | `lib:2601.22032` Tab. 2 p.8 | PUBLISHED |
| DrivoR + TOAD (test-time search) | **94.7** | 4 cameras + control-space CEM search | - | `lib:2606.07170` Tab. 1 p.5 | PUBLISHED |
| DriveZero | **95.3** (abstract) / 94.8 (no human trajectory supervision) | camera-only, multi-view, RL-teacher distillation, 100K navtrain + 237K synthetic | 338.5 M total / 18.6 M LoRA; 16 x H20, 25 epochs | `lib:2609.06055` abstract, Table A12 | PUBLISHED |
| PDM-Closed (privileged) / human | 89.1 / 94.8 | ground-truth perception / logged | - | `lib:2601.05083` Tab. 1; `lib:2406.15349` | PUBLISHED |

Single-front-camera rows quoted in SimWAM's own comparison table and NOT re-read in their method primaries (PUBLISHED-SECONDARY): SGDrive 91.1, ExploreVLA 90.4, UniWorldVLA 89.4, PWM 88.1, Vega 87.9, ImagiDrive 87.4 (`lib:2608.07468` Tab. 1 p.6).

**NAVSIM v2 (EPDMS) -- two protocols, never merged.**
* *navtest single-stage "EPDMS" (method papers; NOT the official column):* Transfuser 76.7, Hydra-MDP++ ViT-L 85.6, iPad ViT-L 85.8, DriveSuprim ViT-L 87.1, Drive-JEPA ViT-L 87.8 (`lib:2601.22032` Tab. 3 p.8); Latent-WAM 89.3, WA-JEPA 91.7 (multi-view, 4 frames, 64 + 32 A800), DriveFuture 89.9 (`lib:2609.34085` Tab. 3 p.9; `lib:2605.09701` abstract); **SUV 91.0, single front camera, ~6 B parameters, 8 x H200, 60 epochs** (`lib:2608.03084` abstract); Adaptive-WAM 89.9 (`lib:2608.06008`); human 90.3 (`lib:2604.03581`, SECONDARY via the programme table).
* ***Closest analog to v8F-mini:*** `lib:2609.34085` Tab. 3: **frozen DINOv3 ViT-L, single view, 4 frames, perception-free: 80.2 EPDMS with a random projector, 85.4 with the AD-E2E-JEPA projector**, trained on navtrain. (PUBLISHED, navtest v2 single-stage, so not comparable to the v1 numbers above.)
* *navhard two-stage (official):* PDM-Closed 56.6 (privileged), DrivoR + TOAD 56.3, DriveFuture 55.5, DrivoR 54.6, SimScale 53.2, ZTRS 48.1, LTF 25.1, ego-history MLP 14.1, CV 11.4 (`lib:2506.04218` Tab. 2 p.8; `lib:2606.07170` Tab. 2; `lib:2605.09701`); SUV 36.9 single front camera (`lib:2608.03084` abstract). Our STOP floor on this column: 29.85.
* Our own arms for orientation (MEASURED, LEADERBOARD section 0E.3, all with 7-D ego status + a route-oracle goal, 4 cameras, one training seed): refcv4b zero-shot **58.97** (< STOP); REFe model_final 83.24, deployed 84.46, SFT-4 88.25, P2 seed 0 **88.82** [87.86, 89.77], P2 seed 1 89.15.

**Camera-only number to beat -- three bars, from low to high.** (a) *Floors:* STOP 61.82 (MEASURED), ego-MLP 65.6. (b) *Same-class bar for a perception-free single-front-camera planner:* **Drive-JEPA perception-free 89.0** (PUBLISHED, 307 M encoder + 330 h pre-training); same-class small models: LTF 83.8, PRIX 87.8 (37 M), AD-E2E-JEPA analog 85.4 on the v2 navtest protocol. (c) *Camera-only ceiling:* **DrivoR + 134k synthetic 94.6 / + TOAD 94.7** (4 cameras, ~40 M), DriveZero 95.3; the human is 94.8, so v1 is saturated at the top.

## 3. The port

**Inputs (v8F-mini -> NAVSIM).**

| v8F-mini input | NAVSIM source | mapping / issue |
|---|---|---|
| front-wide camera, 120 deg cylindrical 256x640 | L0 + F0 + R0 stitched to 256x640 cylindrical | **Already solved and bit-exact**: the E2/W3 builders produce the `PHYSICALAI_WIDE120_256x640` geometry (KB1: 20/20 scenes bit-exact; 17 distinct rigs handled). Fallback: CAM_F0 alone (about one third of the pixel pull, no transfer from the PhysicalAI adapter) |
| 4 history frames at 0.2 s (0.6 s span) | 4 frames at **0.5 s** spacing | cannot be reproduced. Options: **ST** (current frame repeated; what REFe and the refcv4b bridge did, needs only `navtrain_current`), **NT** (nearest-time), or **retrain with a native 0.5 s grid** (needs `navtrain_history`, 144.5 GB). Zero-shot A1NT vs ST differed by only +0.016 on warmup (MEASURED) |
| frozen DINOv3-L, pooled to 40 tokens per frame (4x10 of the 16x40 patch grid) | re-extract | see cost below; I found no NAVSIM features in this pooled-4x10 frozen form (REFe's model is LoRA-trained and 4-camera; its `data/refe_trunk_cache` was not inspected) |
| measured ego motion: speed, yaw rate, history | `ego_velocity`, `ego_acceleration`, poses of the 3 past frames | yaw rate is **derived** (from pose history at 2 Hz; the refcv4b bridge used kappa = a_y / max(v0, 4)^2, sign agreement 0.969, Pearson 0.958, MEASURED KC) |
| nav command (follow / left / right) + distance to the manoeuvre | 4-way driving command, **no distance** | 3-way token maps directly (index order MEASURED); set the existing args-valid flag to invalid for the distance. Route-oracle caveat applies to every row |
| speed limit | **absent** | mask the channel (or feed a map-derived limit as a declared non-vision input; needs a PI call) |
| strategic checkpoints (30/50/80 m), tactical goal/action logits, constraint cells (v9 labels, partly VLM-CoT-derived) | not in OpenScene | route checkpoints can be re-derived from the logged future path (labels may use ego); the CoT-derived tactical tokens cannot be built without a VLM pass. The existing valid-masks let the T/S losses be dropped, but then **the port exercises the operative layer only** |
| output: 6 s at 10 Hz (60 poses) | 4 s at 10 Hz (or 8 poses at 0.5 s) | take the first 4 s; 6 s labels need `num_future_frames` >= 12 at 2 Hz and a spline to 10 Hz (round-trip error 2.5 s mean 0.0085 m, MEASURED K2). The 2 s beyond the horizon is unscored |

**What must be re-extracted, and what it costs (ESTIMATED unless marked).**
* Frames: navtrain_current has the 103,288 current frames; with the native history the unique-frame count is about **152 k** (17,872 unique frames per 12,146 navtest scenes, MEASURED, scaled to navtrain: range 130-170 k).
* Stitch (CPU): 2.04 h for 17,872 frames = **0.41 s/frame** (MEASURED, dev box) -> 103 k frames = 11.7 CPU-h, 152 k = 17.3 CPU-h; about 1-1.5 h / 1.5-2.5 h on a 16-vCPU pod.
* DINOv3-L encode: the banked cache builder ran **4.72 s per clip on the RTX 4060** (about 100 cache steps per clip at 5 Hz, my reading of the 4 MB/clip fp8 size -> about 21 frames/s; `refav1-hj-mini/HANDOFF.md`, "banked"). A40 about 2.5x (spec ratio, `TRAINING_TIME.md`) -> about **50 frames/s**; the FLOP bound is about 0.43 TFLOP per 256x640 frame, so roughly 140 frames/s at 40 % of the A40's 149.7 TFLOPS. => 103 k frames: **0.4-1.0 A40-h**; 152 k frames: **0.6-1.5 A40-h**; navtest (17,872 frames): **3-14 min**.
* Feature storage (fp8, pooled 4x10 x 1024 = 41 KB per frame): navtrain 152 k = **6.2 GB** (103 k = 4.2 GB; fp16 doubles it); navtest 0.73 GB.
* Raw pixels on the pod: navtrain_current transit 304.5 GB; 3 cameras kept = about 20.6 % of the bytes (from the measured 4-camera share) -> **about 63 GB** on disk (4-camera REFe figure is 84 GB); history adds 144.5 GB transit / about 30 GB kept. Streaming shard by shard (download -> extract 3 cameras -> stitch -> encode -> delete) needs only about 20 GB peak.
* Download: **7.76 MB/s on the dev box** (MEASURED, 304.5 GB = 10.9 h); on a pod ESTIMATED 100 MB/s = 0.85 h (current), 1.3 h (449 GB). The dev box is the wrong place (also 102 GB free).
* Training on an A40 (the model is 5.2 M (M1) / 7.2 M (M2) trainable, about 65 GFLOP per step at batch 64): the recipe budget 4S = 19,416 steps = 1.24 M samples = **about 12 navtrain epochs**; MEASURED ~4 h on the 8 GB box (the brief), the repo's per-step guess is 0.05-0.15 s (input-bound). ESTIMATED **2-4 A40-h per arm, up to 8** if input-bound; 25 epochs (2.58 M samples) is about double.
* Scoring: each navtest arm 15-25 min on one CPU worker (W3 estimate from a 20-token smoke; full STOP/CV/HUMAN runs completed); navhard about 1-2 h per arm (the first N1 run aborted at 4,477 s on a RAM guard; the completed run's duration is not recorded here).

## 4. Paths

**M0 -- zero-shot, no navtrain (step 0, nearly free).** Pack the existing navtest frame bank into 17,872 stitched frames, encode with DINOv3-L pooled 4x10, run the **existing M1/M2 checkpoints** through the refcv4b bridge adapter (ST frames, v0/a0/yaw-rate from the NAVSIM ego status, command token, speed limit and distance masked), score with the v1.1 harness plus the STOP / CV / ego-blind / vision-blind controls.
* Cost: encode 3-14 min GPU; scoring 15-25 min per arm; **under 1 GPU-h**, 0 new raw storage, features 0.73 GB; engineering 1-2 days (adapter to the 40-token cache format + gate controls).
* Value: a legitimate number and a test of the input mapping. **Expectation (HYPOTHESIS, anchor: refcv4b zero-shot 58.97 below STOP 61.82):** below STOP. Report it as a waypoint, not a deliverable.

**M1 -- minimal path to a trained navtest number (the first publishable PDMS).**
1. One A40 pod, >= 150 GB volume, >= 16 vCPU. Pull `navtrain_current_{1..32}` (304.5 GB transit), keep CAM_L0/F0/R0, stitch, encode the current frames; **ST history** (no `navtrain_history`). Pull the 14 GB trainval logs for labels.
2. Build windows/labels from the OpenScene logs: logged future path -> 10 Hz targets, route checkpoints from the future path, nav token from the NAVSIM command; T/S losses masked where no labels exist.
3. Train M1 (and M1-nav-shuffle / vision-blind controls), 2 seeds; score on navtest with the existing bank, cache and harness; add the paired log-cluster bootstrap, STOP/CV/HUMAN rows and the four metric families (human futures exist on navtest, so all four are computable; W3 measured the families at 6.5 s).
* **Compute:** download 1-3 h, stitch about 1-1.5 h (overlapped), encode 0.4-1.0 h, train 2 seeds x 2-4 h, navtest encode + score about 1 h => **about 5-10 A40-h, 8-14 h pod wall-clock**; plus about 3-5 CPU-h of navtest scoring for roughly 10 arms (15-25 min each) on the dev box.
* **Storage:** pod volume 150 GB (63 GB kept pixels, or about 20 GB streaming); returned artifacts about 4.2 GB features + checkpoints; dev box +1 GB (navtest features) -- fits E: and C: only if the cache stays off C:.
* **Engineering:** 3-5 days (navtrain windows/labels builder and the stitch/encode pipeline; the bank builder, bridge, controls and scorer are reused).

**F -- the full path (an outsider-grade result).** M1 plus: `navtrain_history` (449 GB transit total) with a native 0.5 s history re-design, >= 3 seeds of the headline arm and replicate arms for every lever claim (`H-ESTIM-SEED-1`), the control set (nav-only, nav-shuffle, vision-blind, STOP, ECHO), a navhard two-stage run (frames for 450 + 5,462 scenes from the 31 GB the leaderboard page says is already local (INHERITED); builder proven bit-exact on warmup), optionally a PDM-score-aware head (the published jump from about 84-89 to 91-95 comes from scoring / vocabulary / RL methods, not from perception).
* **Compute (ESTIMATED):** download 449 GB 1.3 h, stitch + encode 152 k frames 2-4 h, 15-25 training runs x 2-4 h (up to 8) = **about 50-100 A40-h (range 30-200)**, navtest scoring 15-25 arms x 20 min, navhard 5-10 arms x 1-2 h on CPU; **about 2-3 weeks elapsed**.
* **Storage:** pod 500 GB (or about 100 GB streamed); features 6.2 GB (fp8) / 12.4 GB (fp16); no other change on E:.
* **Engineering:** 1.5-3 person-weeks (history grid redesign, label derivation, replicate protocol, navhard runner for the new model).

## 5. Alternatives, briefly

* **nuPlan closed loop (Val14 / Test14-hard / Test14-random, CLS-NR / CLS-R):** the simulator consumes structured state (vector map + tracked objects), not images; DriveZero's 93.57 mean is for a privileged structured-state teacher, not camera (`lib:2609.06055` abstract). A vision-only entry cannot be scored without a camera renderer. Not suitable. (`lib:2106.11810`, `lib:2403.04133`.)
* **Bench2Drive (CARLA v2, 220 routes, driving score / success rate):** the closest to genuine closed loop. Official training data 2 M frames from 13,638 clips; base subset 1,000 clips (950 / 50 split) trained on 1-8 GPUs; evaluation of 220 routes cost 4 x A6000 x 6 h for AD-MLP/TCP and 8 x H800 x 2 days for VAD/UniAD (`lib:2406.03877` Sec. D, Tab. 5). Published driving scores: AD-MLP 18.05, UniAD 45.81, VAD 42.35, TCP 59.90, DriveTransformer 63.46, Drive-JEPA 64.52 (`lib:2601.22032` Tab. 4); SparseDriveV2 89.15 DS / 70.00 SR (`lib:2603.29163` abstract). Cost: a CARLA simulator, a synthetic-domain retrain (the PhysicalAI features/labels do not carry), and a tactical-label pipeline; a different project. Worth it only if the claim must be closed loop.

## 6. Risks (ranked)

1. **The STOP floor and the EP gap.** NAVSIM rewards not moving (STOP 61.82 on navtest, 29.85 on navhard; our zero-shot refcv4b 58.97). A small imitation head on frozen features has no PDM-score supervision; the large published gains (84-89 -> 91-95) come from scorers / vocabularies / RL, and the dominant losses are EP and DAC. Zero-shot cannot clear it; NAVSIM-side training is mandatory, which breaks PhysicalAI-only parity (allowed under the research-use ruling, but must be declared). *Anchors for a realistic first trained number: REFe model_final 83.24 (4 cameras, ego, route oracle), frozen-DINOv3-L single-view AD-E2E-JEPA 85.4 EPDMS. A first v8F-mini score in the low-to-mid 80s is plausible and above the 89.0 same-class bar is not (HYPOTHESIS, not measured).*
2. **The port does not test the thesis.** NAVSIM gives 2 Hz keyframes (no 0.2 s history, no 10 Hz frames), a 4-way command without distance, no speed limit, and none of the v9 tactical/strategic labels. With those masked the run measures the operative layer plus a route-oracle command -- the 4-brain hierarchy would be untested. Also "7.2 M parameters" will be read as **about 311 M** (7.2 M trainable + about 303 M frozen DINOv3-L; Drive-JEPA lists its 307 M ViT-L as the model size), so the efficiency claim must be stated with the encoder counted.
3. **Protocol and acceptance.** v1 navtest is saturated at the top (94.6-95.3 against a 94.8 human) and is non-reactive open-loop scoring; the official v2 column is navhard two-stage with synthetic 3DGS renders that frozen-feature pipelines have not seen; "EPDMS on navtest" is not the official v2 protocol. Every row also needs: STOP beside CV, a replicate seed (a separated log-cluster CI is necessary, not sufficient -- `H-ESTIM-SEED-1`), the route-oracle disclosure, and tier / loop stamps (T1-family, loop OPEN, non-reactive logged agents).
4. Camera-only gap vs lidar: not the issue here -- camera-only entries already lead (DrivoR 94.6, DriveZero 95.3); the gap that matters is single-camera vs 4-camera (single-front-camera leaders: Drive-JEPA 93.7, Adaptive-WAM 92.6, SimWAM 91.5, all with 300 M - 6 B parameters).
5. Rig and domain shift: 17 distinct camera rigs in navtest, 3-camera stitch artifacts, and a frozen trunk with no adaptation; REFe needed LoRA to reach 83.
6. Labels / leakage: the NAVSIM command is a route-level oracle (no route-following claim); ego-status input is accepted in the field but must be declared (our vision-only rule says inference is vision-only -- a NAVSIM row reads ego status and command, as every leader does; the PI ruled this admissible for REFe on 2026-10-04).
7. Engineering reality: the pod that held REFe's navtrain pixels was unreachable at the last probe (volume fate unknown), the navtrain re-pull is the long pole, and E: (102 GB free) cannot hold it.

## 7. Open decisions for the PI

* Authorise M0 (under 1 GPU-h) now; authorise M1 (about 5-10 A40-h, one pod session) if M0 shows the input mapping is sound.
* ST history (cheap) vs a retrained native 0.5 s history (needs `navtrain_history`).
* Speed limit: masked, or a map-derived declared input.
* Whether the claim is "Pareto: PDMS per parameter / GPU-hour" (credible) or "above Drive-JEPA perception-free 89.0" (hypothesised unlikely for a frozen-feature head without PDM-score supervision).

## 8. Banking manifest (research-banking rule)

* `kb_add.py` resolves its Library as `<tools parent>/TanitAD Research Lab/Library`, so running it from the fresh export (`tools_export/`, from tip `f547c11b`) writes ONLY inside this directory, never into E: or D:. The export was made from the mirror at `f547c11b` (the brief's second instruction) rather than `8d6449a7`.
* **Newly banked here (3 papers, 53 MB, `kb_add --verify`: 3 entries, 0 problems), tag `navsim`, `--cited-by` = the path in the brief:** `lib:2608.03084` SUV (3.9 MB), `lib:2608.06008` Adaptive-WAM (11.2 MB), `lib:2609.03602` SV-WAM (37.8 MB). They live in `tools_export/TanitAD Research Lab/Library/papers/` with a scratch `library.json` / `LIBRARY.md`. **They are NOT in the repo's Library:** the caller must move the three PDFs and merge the three rows into the real `library.json` (e.g. `kb_add.py --local` or a row merge) and run `--reindex`.
* **Already banked at the tip, cited here, `--cited-by` NOT updated (it needs the real Library):** 2406.15349, 2506.04218, 2601.22032, 2609.34085, 2606.07170, 2605.09701, 2609.06055, 2512.07745, 2604.03581, 2603.29163, 2601.05083, 2411.15139, 2608.07468, 2507.17596, 2504.01941, 2507.04049, 2506.08052, 2505.15111, 2506.06659, 2503.12820, 2503.11650, 2510.12796, 2510.04333, 2506.13757, 2406.03877, 2106.11810, 2403.04133. Each PDF's sha256 was checked against the tip's `library.json` when its text was extracted.
* Not banked (no arXiv PDF): the NAVSIM GitHub docs pages (`install.md`, `splits.md`, `metrics.md`, `agents.md`), read via WebFetch on 2026-10-25 and recorded in `raw/WEB_DOCS_NOTES.md`; they could be banked as `--doc-type source` rows with the URLs. Numbers quoted from them are PUBLISHED-SECONDARY.
* Never touched: `Keys.txt`, any git index, any E:/D: path (E: was listed read-only).

## 9. Files in this package (all under the OUT directory)

`NAVSIM_SCOPE.md` (this file) · `raw/WEB_DOCS_NOTES.md` · `papertext/*.txt` (extracted text of the cited banked PDFs + the three new ones) · `tipdocs/` (copies of the tip documents read: LEADERBOARD, E2 bridge, W3 navtest, REFe, v8F-mini handoffs, `published_results.json`) · `library_tip.json` (the tip's Library index, used to check which papers were already banked) · `scripts/pdf_txt.py` · `tools_export/` (fresh tools export + the scratch Library with the 3 new PDFs).
