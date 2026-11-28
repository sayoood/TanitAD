<title>REFe — DriveZero reproduced on nuPlan/NAVSIM with ONE change (DINOv3-only), scored on their benchmarks against their own table</title>

# REFe — plan v2 (corrected 2026-09-20; scope and state refreshed 2026-09-21)

⛔ **SCOPE CHANGED 2026-09-20 BY THE PI'S SECOND RULING AND THIS FILE LAGGED IT.** REFe is now the
paper's **FOUR-camera** rig (`CAM_F0, CAM_L0, CAM_R0, CAM_B0`), so it differs from DriveZero in
**ONE** variable — DINOv3 instead of the agglomerative DriveVFM. Wherever the text below still says
*"one camera"* or *"at one camera"*, read it as the superseded single-camera design; the committed
reads in §6 are restated for four cameras in `REFE_MODEL.md` and `TRAINING_TIME.md`.

`Research Lab · 2026-09-20 (LAB-RUN-017) · Architecture & Inference · PI directive 2026-09-20 · companion to ../2026-09-20-drivezero-deep-analysis/`
⛔ **v1 of this file mis-read the scope as "REFe on our PhysicalAI corpus" and built a corpus-adapter stage around it. The PI's intent is an experiment ON THEIR DATA (nuPlan / NAVSIM) to measure the quality of the resulting architecture. v2 replaces v1; the withdrawn parts are listed in §5 so nothing vanishes silently.**

---

## 0 · The experiment, stated precisely

**REFe** = DriveZero's end-to-end planner, trained **exactly their way** on **their data**, with two deliberate changes:

| | DriveZero | **REFe** |
|---|---|---|
| cameras | 4 (`F0 B0 L0 R0`) | **1 — `CAM_F0` only** |
| visual backbone | DriveVFM (DINOv3 + SigLIP2 + SAM + DA2 distilled) | **DINOv3 only** (frozen); later **TanitVFM** = their VFM recipe with DINOv3 as the sole teacher |
| everything else | registers, M = 64 WTA proposals, detached 6-component scorer, rank-32 Q/V LoRA, AdamW 2e-4 cosine 25 ep, teacher rollouts + **goal augmentation**, no human trajectories | **identical** |
| teacher | DriveRL, their checkpoint | **their released `checkpoint_2400.pt`** — validated first |
| training data | navtrain (100 K) [+ SimScale 237 K for `-Scale`] | **navtrain (100 K)**, no SimScale |
| evaluation | navtest PDMS · navhard EPDMS · HUGSIM | **navtest PDMS · navhard EPDMS** (HUGSIM optional, later) |

⭐ **The published anchor REFe is measured against exists already.** DriveZero Table 7 ran their full recipe with a **plain DINOv3 ViT-S backbone, 4 cameras, navtrain only: PDMS 93.88** (NC 98.93 · DAC 99.01 · EP 91.31 · TTC 95.77 · Comfort 99.97). REFe differs from that row by **one variable — the camera count** — so `93.88 − REFe` is **the price of the single front camera**, cleanly. The same table's supervision block (human 93.92 · teacher 93.61 · teacher+goal-aug 94.41, DriveVFM ViT-S) is the second anchor.

## 1 · Two design notes that stand from v1 (they are about their data, not ours)

**1.1 TanitVFM's goal is efficiency, not quality — say so up front.** Their Table 7: DINOv3-only backbone **93.88** vs the four-teacher DriveVFM **94.41**; a distillation from a single teacher cannot exceed that teacher. Their Table A13: **DriveVFM ViT-S 94.41 ≈ DINOv3 ViT-L 94.55** — a ~22 M backbone at ~304 M parity. ⇒ **TanitVFM = DINOv3 ViT-L → ViT-S distillation (PHI-S, QK-Clip, two-stage 256² → 512²) on navtrain `CAM_F0` frames**, pre-registered **in two tiers** (Stage 4): *minimum* = no worse than REFe-DINOv3-S at the same size; *stretch* = REFe-DINOv3-L parity at ~1/14 the parameters. Their only single-teacher-adjacent row — the two-teacher DINOv3 + SigLIP2 student at **93.69, below frozen DINOv3-S (93.88)** — says the minimum is not automatic and the stretch is a hypothesis. A single-teacher distillation is a **new** row either way — they never ran one.

**1.2 Order: planner on frozen DINOv3 first, TanitVFM second.** The frozen-DINOv3 REFe is the control TanitVFM must beat, it is days not weeks, and it delivers the headline comparison (§0) on its own.

**1.3 One number to carry, not a risk to fear:** the teacher sees every agent; REFe sees the front camera. Generate each target twice — teacher on full inputs, teacher on **FOV-masked** inputs — and log the per-frame divergence. It is the number that explains whatever `93.88 − REFe` turns out to be, and it costs one extra rollout.

## 2 · Stages

⭐ **What trains in which stage (PI question 2026-09-20 — this is DriveZero's own two-part structure).** DriveZero pretrains its backbone **separately** (DriveVFM, their §2.2) and then trains the planner with that backbone **frozen plus rank-32 Q/V LoRA** (their §2.3; Table A12: *"Visual backbone DriveVFM ViT-L (frozen) · Backbone adaptation Q/V LoRA, rank 32 · Trainable parameters 18.58 M · Full 338.46 M"*). So "end-to-end planner training" in their recipe **already includes light encoder adaptation via LoRA**; what it does *not* include is encoder **pretraining**. REFe keeps exactly that split:

| stage | backbone weights come from | what TRAINS | what stays FROZEN | what it delivers |
|---|---|---|---|---|
| **3 — planner training** | **Meta's released DINOv3** (ViT-S and ViT-L), as-is | rank-32 Q/V **LoRA** on the backbone + 16 registers + trajectory decoder + scoring decoder (their ≈ 18 M trainable) | the DINOv3 trunk itself | **REFe-DINOv3** and the headline number vs their DINOv3-row **93.88** — comparable because their ablation rows used the same frozen-plus-LoRA planner recipe |
| **4a — encoder pretraining** | a **new ViT-S student** (DINOv3-S init or scratch) | the whole student, by **distillation** from frozen DINOv3 ViT-L: summary + patch tokens, PHI-S, QK-Clip, 256² → 512² | the DINOv3-L teacher | **TanitVFM-S** — their DriveVFM recipe with one teacher |
| **4b — planner re-training** | **TanitVFM-S** | same as Stage 3 (LoRA + heads), re-run | the TanitVFM-S trunk | **REFe-TanitVFM**: must match REFe-DINOv3-L within the replicate floor at ~1/14 the backbone parameters |

⇒ **Yes, the first end-to-end test (Stage 3) runs without any encoder pretraining of ours** — it is DriveZero's planner recipe on the off-the-shelf DINOv3, LoRA included. Stage 4 is where *we* make an encoder, and Stage 3 is the bar it has to reach.

### Stage 0 — Teacher running and validated ← **IN PROGRESS, dev box (4060)**

| step | status |
|---|---|
| clean clone, short NTFS path (`C:/Users/Admin/dz/DriveZero`, 554 files) | ✅ (the scratchpad path hit Windows **MAX_PATH**) |
| Python 3.11 env; **torch 2.7.1+cu128, CUDA on the 4060**; ray 2.51.1, hydra 1.3.2; `driverl` + vendored `nuplan-devkit 1.2.2` editable | ✅ (TLS proxy ⇒ `uv --native-tls`) |
| `fcntl` (Unix-only, one advisory `flock` in `gpkg_mapsdb.py`) | ✅ documented no-op shim, single-process-safe |
| **`PREFLIGHT_OK`** — `artifact=ok · simulation_import=ok · policy_load=ok agent=VanillaNetAgent update=2400 interval_s=0.2` | ✅ **MEASURED 2026-09-20** |
| `nuplan-maps-v1.0` (1.4 GB, four cities) | ✅ **already local** at `D:/Projects/TanitAD/data/nuplan-maps/` |
| **nuPlan v1.1 `mini` split** | ⭐ **MEASURED 2026-09-20: the S3 object is PUBLIC** — `HEAD …/public/nuplan-v1.1/nuplan-v1.1_mini.zip` → 200, `Content-Length 8,550,100,030` (8.55 GB zip; the "13 GB" was the unzipped DBs). No login needed. The PI's Chrome download was at 89.8 % @ 13 MB/s, so `code/finalize_mini.py` **adopts it** (move to D:, byte-count + CRC of every member, extract, re-point the junction, run 8 scenarios, render) with curl-to-D: as the fallback for every named failure mode. **Data on D: only** (`D:/Projects/TanitAD/data/nuplan`) |
| teacher closed-loop run on mini, all six reward components logged; TTS N=8 ≥ N=1 (their margin rule guarantees it — a control that must read a known value) | ⏳ on the zip |
| **mini-split wiring** — their runner hard-codes six tasks, but exposes `DRIVERL_EVAL_DB_LINK_ROOT` + `DRIVERL_EVAL_SCENARIO_FILTER_OVERRIDE`; `code/run_mini_teacher.sh` re-points `val14_{nr,r}` at mini through an NTFS **junction** (D: is exFAT, no symlinks) and the devkit's split-agnostic `one_of_each_scenario_type` filter | ✅ **DRY-RUN PASSED 2026-09-20** (`PREFLIGHT_OK → START_EVAL db=…/dblinks/driverl_val14 filter=one_of_each_scenario_type → DRY_RUN → SUITE_DONE status=0`). ⛔ Two Windows traps on the way, both recorded: `cmd.exe /c` from Git Bash does **nothing** with exit 0 (MSYS rewrites `/c`→`C:/`; `MSYS_NO_PATHCONV=1` fixes it); first light runs `worker_mode=sequential` (no Ray) |
| **first light — FIVE environment faults, each one layer deeper, all MEASURED and fixed 2026-09-20 11:22–12:05** | (1) my own `MSYS_NO_PATHCONV=1` export for the whole run made the native Python look for `C:\c\Users\…` (exit 2) → scoped to the `mklink` line only; (2) **an NTFS junction (C:) into the exFAT volume lists EMPTY** to Python/`cmd dir`/`ls` → devkit "No log files found!", 0 scenarios → **link-free layout**: the DBs live as a plain dir at `D:/…/data/nuplan/dblinks/driverl_val14` (their runner only needs the path to *exist*); (3) `NUPLAN_MAPS_ROOT` must be the dir that *directly* holds `nuplan-maps-v1.0.json` → "Building simulations from **8** scenarios" and the teacher is driving | ✅ each fix is in `run_mini_teacher.sh` / `finalize_mini.py`; the junction fact is in memory |
| ⛔ **fault 4/5 — Windows MAX_PATH at the SIMULATION-LOG WRITE, and it presents as a healthy run** | Run #4 simulated **5 of 8** scenarios, wrote **4** logs, then died `FileNotFoundError [Errno 2]` on a **263-char** path (limit **259**). The messenger lies twice: the parent dir existed and was created fine, and only ONE scenario type was over budget — so the suite looked normal until THEIR `exit_on_failure=true` aborted it. ⭐ **The budget belongs to the CONSUMER that composes the deepest path** (`simulation_log_callback.py:147` appends `<sim>_<task>/simulation_log/<planner>/<scenario_type>/<log_name>/<token>/<token>.msgpack.xz`), and it must be priced at its **worst case**: type 42 here but **54** possible (`starting_straight_traffic_light_intersection_traversal`), +8 for the six-task suite ⇒ tail **231** ⇒ prefix budget **28**. The old prefix was **52**. | ✅ output root is now `C:/dzo` + a ≤7-char run id (`m-nr-n`, `m-nr-64`) = **14**, 14 chars spare. Verified: the exact scenario that killed run #4 now writes at **224** chars. |
| ⭐ **BEV renderer VALIDATED ON REAL DATA** (it had only ever been compile-checked) | `following_lane_with_lead` / token `485e78d3d4035b52`: **149 frames, 990×990 @ 5 fps**, content-asserted (per-frame std 30.7–32.2, never flat — an all-black mp4 is a valid file and would have passed a presence check). Fixed a real defect found by LOOKING at it: the title printed the token twice, clipped. | ✅ `media/teacher_mini_485e78d3d4035b52.mp4` + `_sheet.png`; sent to the PI 2026-09-20 |
| ⭐ **per-step policy outputs are IN the log** — `SimulationHistorySample.trajectory` is their `DriveRLActionTrajectory`, which pickles `jerk_long · lat_command · raw_action (Beta) · acceleration_control · steering_control · goal_points (ego frame) · debug_info{model_input{ego, other_agents, road_graph as the policy saw it}, test_time_scaling{candidate_actions/values/scores/total_rewards/dones/valid, selected_candidate, margin, gamma}}` | ✅ **MEASURED from `trajectory.py` + `planner.py`** ⇒ no sidecar; the renderer and the TTS sweep read the log directly |
| **BEV mp4 renderer** `code/render_teacher_bev.py`: map lanes + the policy's **own** road-graph view (cyan) + agents + ego + DriveRL plan + expert log + mission goal + the policy's near/far goal anchors, with a text overlay of speed, jerk, lateral command, raw Beta action and the TTS selection | ✅ written against the verified schema; **untested until the first real log** (the release ships no renderer; mini has no camera blobs ⇒ BEV-only for the teacher) |
| **TTS N-sweep analysis** `code/tts_sweep_analysis.py` (DZ-10): paired per-scenario deltas vs N=1 on identical tokens (refuses otherwise), switch rate, margin-when-switched, and a **harness check** (a switch below the configured margin is an error, by their own rule) | ✅ written; untested until logs exist |
| DZ-10: the **TTS N-sweep** on mini (N ∈ {1, 8, 16, 32, 64}) — a paired within-scenario ablation on their own policy and critic | ⏳ on the zip; **answers I-1's quality half** |
| ⭐ **DZ-11 re-priced — the full six-task headline reproduction is a dev-box job**: `test.zip` **95.9 GB** + `val.zip` **97.0 GB** = **193 GB on D:, ~4 h at 13 MB/s**, single-GPU `ray_local` run; the 1.8 TB figure was the *whole dataset incl. train*, which the six benchmark filters never touch | ⏳ **needs the PI's OK for a 193 GB pull** (nuPlan non-commercial terms apply); then we are the **first external check** of `checkpoint_2400.pt` against their Table 2 |

### Stage 1 — Goal augmentation on nuPlan (dev box)

The hook is `route_points` → `route_goal_positions(horizon 12 s, min 5 m/s)` → near/far anchors. On nuPlan the two augmentation forms are both **native**: (a) **lane-centre snapping** — their own PPO training semantic (Table A3: *"the four nearest lane-center projections within 20 m"*, weight `[max(1 − d/20, 0)]²`); (b) **route alternatives** — the nuPlan map API's roadblock/lane-connector graph, which the released feature builder already consumes (`route_roadblock_ids`, `correct_route_roadblock_ids`). Deliverable: `augment_routes(scene) → [route_k]`, K ≈ 3–5 per frame; teacher rollout per `(frame, route_k)`; the 20-step `(x, y, yaw)` target; the six component scores from the released calculators.

⭐ **Read from their code 2026-09-20, in `STAGE1_GOAL_AUGMENTATION.md` — three facts that change this stage:**
1. **The teacher's goal input is TWO points**, not one (`driverl_teacher.yaml:12` `goal_count_probs: [0.5, 0.5]`;
   `env/config.py:27` defines `num_goal_positions = len(goal_count_probs)`), and PPO drew the effective count
   50/50 between 1 and 2. A single-goal reproduction is **not** the released recipe, and `apply_goal_pair_mode`
   gives us the collapse ablation for free.
2. **The snap sampler is released and correct** (`goal_position_utils.py`), so Stage 1 does not write one. But
   ⛔ **their own docstring says the draw weight is `1/distance` and the implementation computes `(1 − d/max)²`** —
   the code's inline comment says the shape was chosen to avoid exactly that blow-up, and the variable is still
   named `inv_dist`. Table A3 and the code agree; the **docstring** is the stale artifact. Implementing from it
   would put ≈0.94 of the mass on the nearest lane instead of ≈0.42.
3. **Snapping is NOT enabled in the released runtime config** (`grep goal_snap release/configs` → no hits ⇒
   `keep_prob=None` ⇒ branch skipped). ⚠️ **State the layer:** that is the *inference* config and is **not**
   evidence about how PPO was trained; the training config is not in the release.

⇒ What is genuinely ours in Stage 1 shrinks to the **route-alternative generator** over the nuPlan roadblock
graph, plus the rollout loop and the controls. The free knobs (`horizon_s`, `min_speed_mps`, `goal_pair_mode`)
are env vars we already set in Stage 0.

**Controls:** `route_0` (the logged route) reproduces the un-augmented rollout **exactly**; the calculators score the **logged human** trajectory at a banked value before scoring anything else.

### Stage 2 — Target generation on navtrain (pod)

DriveZero: *"each frame in the log pairs the multi-view images with the structured state, so a trajectory produced by the teacher from the state can supervise the student on the images."* Their released feature builder is the **online** nuPlan variant; its own `temporary_limitations` string says their training pipeline used an **offline WebDataset with expert future ego poses**. ⇒ we write the offline builder: **OpenScene/navtrain annotations (ego, agents, route, traffic lights) + nuPlan maps → `ScenarioData`**, then Stage 1's rollouts per frame × K goals. Output: `(scenario_token, frame, route_k, navigation command, τ_T[20×3], six scores, FOV-divergence)`.

⭐ **The builder's inputs all exist in NAVSIM's own `Scene` (schema read from `navsim/common/dataclasses.py`, 2026-09-20):** `Frame.ego_status{ego_pose, ego_velocity, ego_acceleration, driving_command}` → the 9-D kinematics; `Frame.annotations{boxes, names, velocity_3d, track_tokens}` → the 16-D agent rows with 5-frame history (`num_history_frames`); `Frame.roadblock_ids` → the route polyline via the map API (`Scene.map_api`, `SceneMetadata.map_name` — our local `nuplan-maps-v1.0`); `Frame.traffic_lights: List[(lane_connector_id, bool)]` → the traffic-light state folded into the 10-D lane rows; `Cameras.cam_f0{image, intrinsics, sensor2lidar_*}` → REFe's single input. Nothing has to be invented; it is a re-indexing of their pickles into the teacher's `ScenarioData`. ⭐ **NAVSIM ships a `mini` split WITH sensor blobs (`./download_mini`)** ⇒ Stages 2→3→5 can be exercised **end-to-end at small scale on the dev box** before the pod exists — the cheapest possible rehearsal of the whole REFe pipeline.
**Budget (ESTIMATED):** 100 K frames × K goals × 20-step rollouts of a 5.7 M net is GPU-trivial; the annotation → `ScenarioData` build is CPU-bound — hours to a day on the pod.

### ⭐ STAGE 2c STATUS — THE SIX-COMPONENT TARGETS EXIST AND DISCRIMINATE (2026-09-20)

The scorer half of Stage 2 is **built and gated**, on nuPlan rather than navtrain, from banked
teacher rollouts. Full account and the four wrong versions that preceded it: `REFE_MODEL.md` §8.1.

| piece | state |
|---|---|
| the six released calculators, fed a `ScenarioData` rebuilt offline | ✅ works, nothing re-derived |
| step-wise rollout scoring, the way the RL engine scores | ✅ `score_proposals.score_proposal_rollout` |
| a gate whose controls are **analytic**, not assumed | ✅ `refe/scorer_gate.py` |
| reproducibility across processes | ✅ domain randomization pinned from their released yaml |
| target bank over a structured candidate set | ✅ `refe/build_scorer_targets.py`, building |
| targets consumed by training | ✅ `train.py`, nearest-proposal assignment |
| **online scoring of the student's OWN proposals (their method)** | ⛔ not affordable here — stated, not hidden |

**The gate, and why it is the only number worth quoting here.** On the teacher's own realised path
`OffRoad.info` reads **0.0000**; on a path aimed through the nearest curb it reads **1.0000** with
reward **−1.6994**; **10 of 28** signals separate under step-wise scoring against **3** at the
endpoint. ⛔ Every weaker check was green while the pipeline was useless, including one that ran
end to end and returned plausible PDM numbers for all three candidates.

⚠️ **Cost moves the plan.** MEASURED single-core CPU: **~12 s per frame** for 9 candidates at
stride 2, of which the `ScenarioData` build is **2.5–4.9 s**. The Stage-2 budget line below calls
the annotation → `ScenarioData` build "CPU-bound, hours to a day on the pod" — that is the right
order and this is the measured constant to plan with. A batched multi-proposal path gives only
**2.5× at N=8**, so parallel processes are the lever, not batching.

## ⭐ STAGE 3 STATUS — THE MODEL EXISTS (2026-09-20). Full account: `REFE_MODEL.md`

| piece | state |
|---|---|
| architecture (`refe/model.py`) | ✅ **ViT-L, FOUR cameras**, **322,070,854 total / 18,991,426 trainable (5.90 %)** vs the paper's 338.46 M / 18.58 M (5.49 %) — **+2.21 % on trainable**. ⚠️ The **8.06 %** this row used to quote predates the `reg_mlp_ratio = 1` ruling; `RegisterCompress` costs **6,303,744**, not 12,598,272 |
| real DINOv3 weights | ✅ all three sizes load **strictly** — every checkpoint tensor consumed |
| training loop (`refe/train.py`) | ✅ overfit control **1.9316 → 0.3730 (80.7 %)** at four cameras with **gradient accumulation**; `TRAIN_CONTROL_OK` |
| target bank (`refe/build_targets.py`) | ✅ **2,182 tuples** = 1,091 × 2 **genuinely distinct** ranks, 4 channel paths per row, 73.2 % step coverage |
| planner wrapper (`refe/planner.py`) | ✅ four-camera end to end, runs in THEIR harness |
| camera frames for a real rehearsal | ✅ L0/R0/B0 fetched into containers; **982 of 1,091 tuples (90.0 %) resolve all four and decode**, the remainder is one log being fetched now |
| scorer targets reaching training | ✅ MEASURED **10.4 %** sample coverage with a live score loss (rank-1 bank rebuilding to raise it) |
| training cost | ✅ MEASURED — full paper scale **2,380–3,080 A40-hours = 99–128 days on ONE A40** (`TRAINING_TIME.md`) |
| **trained weights** | ⛔ **not yet — this is what the pod is for** |

⭐ **Fidelity cross-check that was not fitted:** rebuilt in THEIR configuration (ViT-L, 4 cameras)
the reconstruction gives **5.52 %** trainable against their published **5.49 %**. The heads were
sized from the paper's prose, so landing there independently is evidence the architecture is right.

⚠️ **Anchor changed with the backbone.** With ViT-L the comparable published row is **DINOv3
ViT-L = 94.55** (Table A13), not the ViT-S 93.88 this plan originally named. The reason ViT-S was
abandoned is stronger than the +0.67: their Table 7 shows **DINOv2 ViT-S = DINOv3 ViT-S = 93.88**,
so at that scale the backbone provably does not matter and a DINOv3-only experiment there would
have been **uninformative by construction**.

⚠️ **Consequence:** REFe is **319.0 M** and exceeds the programme's **sub-300M** thesis by 19.0 M.
⭐ **PI RULING 2026-09-20: "no problem with the 316.9M"** — accepted, including the rise to 319.0 M
from the 4-layer proposal decoder the paper specifies. REFe is a DriveZero reproduction, not the
TanitAD flagship, and the sub-300M thesis is the flagship's.
Recorded, not buried — REFe is a reproduction on their data, not the deployed product.

⛔ **Two defects caught by controls before any rented GPU:** DINOv3's own `cls_token` and 4
`reg_token`s were missing from the trunk (a silently-damaged "pretrained" backbone), and the target
horizon was **halved** because the simulation history logs at 10 Hz while their head is 20 steps
@ 5 Hz. Both produced entirely plausible artifacts and were found only by a control.
### Stage 3 — REFe planner training on the released DINOv3 (frozen trunk + LoRA + heads) (pod, A40)

Their planner verbatim, one camera at **960 × 512**: 3D position embeddings, **16 registers**, **M = 64** proposals, 256-d, 4-layer proposal decoder, **WTA L1** to `τ_T`, **detached** 6-component scoring branch (BCE on the released calculators' scores), **rank-32 Q/V LoRA** on frozen DINOv3, AdamW 2e-4, cosine to zero, 25 epochs, batch 256, FP32.
**Arms:** `DINOv3-S` and `DINOv3-L` backbones × supervision ∈ {**human-only · teacher-only · teacher + goal-aug**} — the second axis reproduces their Table 7 supervision block at one camera. **Committed reads:** (i) REFe-S (teacher+aug) vs the anchor **93.88** = the single-camera price; (ii) teacher+aug > human > teacher-only reproduces their mechanism at one camera, or it does not and we learn that; (iii) DINOv3-L vs -S = the backbone-scale slope at one camera (their A13 slope was +0.42 S→L for DriveVFM).
**Budget (ESTIMATED, measure on the pod first):** their 608 GPU-h covered 337 K scenes × 4 cams × ViT-L; at 100 K × 1 cam × ViT-S this is of order **0.5–1 A40-day per arm**.

### Stage 4 — TanitVFM: (a) our own encoder by distillation, then (b) Stage 3 re-run on it (pod, A40, weeks)

DINOv3 ViT-L (frozen teacher) → ViT-S student on navtrain `CAM_F0` frames: summary + patch tokens, cosine + MSE, **PHI-S** on patch features, **QK-Clip**, two-stage 256² → 512², **scaled-down iteration count pre-registered** (their 800 K × 2,048 is out of reach on one A40 — an epoch count over the navtrain frames is the honest unit). Then Stage 3's best arm with TanitVFM-S swapped in. **Committed, in two tiers, because their own numbers support only the first as an expectation (PI question 2026-09-20):**
- **Minimum — "distillation did not hurt":** REFe-TanitVFM-S within the replicate floor of **REFe-DINOv3-S** (same backbone size). Their nearest row says this is *not* automatic: the **two-teacher (DINOv3 + SigLIP2) student scores 93.69, i.e. 0.19 BELOW frozen DINOv3 ViT-S (93.88)** — a DINOv3-fed distillation only overtook frozen DINOv3 once SAM was added (94.10).
- **Stretch — "parity at 1/14 the size":** within the replicate floor of **REFe-DINOv3-L**. Their evidence for this is the *four-teacher* DriveVFM-S (94.41) against DINOv3-L (94.55); no single-teacher row exists, so this tier is a **HYPOTHESIS**, not an expectation.
- **And** ≥ 3× faster on Thor. Adopt on the stretch tier; on the minimum tier only, TanitVFM is a *size* win to be priced against its accuracy cost, not an automatic adoption.

### DZ-11 — the headline reproduction: pipeline, costs and the two traps already guarded (2026-09-20)

⛔ **Three steps, each with a guard. None may be skipped.**

| step | command | guard |
|---|---|---|
| 1 extract | `code/verify_extract_split.py` (already wired into the running pull) | byte count vs the S3 `Content-Length`, CRC via `zipfile` |
| 2 **arrange** | `code/arrange_splits.py --apply` | dry run by default; **refuses a non-empty target**; asserts the DB count survives the move |
| 3 run | `code/run_headline_suite.sh` | refuses a mini-sized split, an incomplete extraction, and an inherited `DB_LINK_ROOT` — all three proven by mutation (`raw/dz11_guard_mutation_test.txt`) |

⚠️ **Trap 1 — the layout.** MEASURED by parsing the local file headers of the **partially downloaded**
archive (no need to wait for 100 %): the zips contain **`data/cache/<split>/`**, while the runner reads
**`nuplan-v1.1/splits/{trainval,test}`** (`run_driverl_nuplan_eval.sh:192-194`). Extraction alone
yields *"No log files found!"* and 0 scenarios — the Stage-0 symptom from a different cause. D: is
exFAT so neither link type works; `arrange_splits.py` does a **same-volume rename** (instant).

⛔ **Trap 2 — the silent wrong score.** Their runner overrides the DB dir when
`DRIVERL_EVAL_DB_LINK_ROOT` is set and `$ROOT/$FILTER` exists (`:197-198`). Stage 0 left
`dblinks/driverl_val14` holding the **64 MINI DBs**. Inheriting that variable makes `val14_nr/_r`
report a "val14" score computed on mini logs **without erroring**. The runner now refuses it (exit 4).

**Cost, from the shipped filters' own token counts (MEASURED, not estimated):**

| task pair | tokens | scenarios | sequential @ 114 s/scen |
|---|---|---|---|
| test14hard_nr/_r | 272 | 544 | 17.2 h |
| test14random_nr/_r | 261 | 522 | 16.5 h |
| val14_nr/_r | 1118 | 2236 | 70.8 h |
| **total** | | **3302** | **104.6 h** |

⇒ **run the four test14 tasks first** (the rows their headline table reports, one third of the cost);
`ray_local` + **`OMP_NUM_THREADS=6`** are mandatory — torch spawns ~113 threads per process and
concurrent arms otherwise sit at 0–6 % GPU looking hung.

**Disk, MEASURED:** mini expands **1.68×** (7.96 GB zip → 13.37 GB). Projected peak with both zips
kept and both splits extracted = **481 GB**; D: has **945 GB** free. Not a constraint.
### Stage 5 — Evaluation and the comparison table

**navtest PDMS** (NAVSIM v1) and **navhard EPDMS two-stage** (NAVSIM v2) through the NAVSIM devkit, plus the number nobody in this family publishes — **REFe's measured latency on Thor**. Report against DriveZero's Table 4/5/7 rows with their stamps carried.
⭐ Prior art in-house: `Data Engineering/Implementation/incoming/2026-08-28-navsim-corpus-adaptation/code/build_navsim_eval.py` (unmerged, `incoming/`) already touches NAVSIM eval plumbing — read it before writing Stage 5's scorer glue. ⛔ **No NAVSIM/OpenScene data exists on the dev box** (probed 2026-09-20: no HF-cache, no D: dirs); the backlog-row-3 pull is still PI-gated, so any holding would be pod-side and unverified from here.

## 3 · Data and where it lives

| data | size | where | why |
|---|---|---|---|
| nuPlan **maps** v1.0 | 1.4 GB | ✅ dev box | teacher + builder |
| nuPlan v1.1 **mini** (DB only) | **8.55 GB zip (MEASURED, S3 HEAD)** → ~13 GB of DBs (RELAYED) | **D:** `D:/Projects/TanitAD/data/nuplan` (**link-free**: DBs live as a plain dir `dblinks/driverl_val14`; run OUTPUT goes to `C:/dzo` for MAX_PATH) | Stage 0/1 validation, videos, TTS sweep |
| **NAVSIM `mini`** (`./download_mini`: `navsim_logs/mini` + `sensor_blobs/mini`, 8 cameras) | **not stated in the docs — read it off the OpenScene page** | dev box (if it fits C:'s 135 GB free) | ⭐ end-to-end rehearsal of Stages 2→3→5 before the pod |
| **navtrain** (`./download_navtrain`, "a small portion of trainval": `navsim_logs/trainval` + `sensor_blobs/trainval`) + `CAM_F0` blobs | annotations small; sensor blobs are the bulk — the programme's prior figure for the full NAVSIM pull is **~450 GB** (backlog row 3; **to confirm on the OpenScene page**) | **pod only** | Stage 2/3/4 |
| **navtest** annotations + `CAM_F0` blobs, **navhard** (v2, incl. stage-2 renders) | to confirm | **pod only** | Stage 5 |
| nuPlan **`test`** DBs — `nuplan-v1.1_test.zip` | ⭐ **95.9 GB (MEASURED, S3 HEAD 200, 2026-09-20)** | **D:** (1.1 TB free), ~2 h at the measured 13 MB/s | **4 of the 6 headline tasks** (`test14hard_nr/r`, `test14random_nr/r`) — the DZ-11 gap is closed; **no pod needed** |
| nuPlan **`val`** DBs — `nuplan-v1.1_val.zip` | **97.0 GB (MEASURED)** | D:, ~2 h | the remaining 2 tasks (`val14_nr/r`; Val14's 1,118 tokens are drawn from the nuPlan val split — INHERITED from the Val14 definition, to be confirmed by the filter hitting 1,118) |
| *(for scale)* one train part — `nuplan-v1.1_train_boston.zip` | 38.2 GB (MEASURED) | — | **not needed**: the six-task reproduction is **~193 GB, not the 1.8 TB whole-dataset figure** |

⛔ Nothing multi-GB is pulled over the dev box's measured 1.2 MB/s uplink; the pod pulls its own data.
⭐ **Downlink, MEASURED 2026-09-20 for the nuPlan pull (PI "start"):** single curl stream to D: **5.2 MB/s uncontended** (1.0 MB/s while the mini extraction was writing to the same drive); a 4-stream range probe reached only **6.6 MB/s aggregate** ⇒ the path is **link-capped, not per-connection-capped**, so no parallel downloader. Chrome's earlier 13 MB/s was not reproducible by curl. ⇒ `test.zip` ≈ **5 h**, both archives ≈ **10 h**, sequential, resumable (`code/pull_nuplan_splits.sh`, running detached with a 10-min progress monitor). D: is a **SanDisk Extreme Pro USB SSD (4 TB, exFAT)**; C: is NVMe.

## 4 · What only the PI can do

1. ~~Log in and download mini~~ **superseded 2026-09-20**: the S3 objects under `/public/nuplan-v1.1/` answer without a session, so the Lab can pull nuPlan splits itself (curl, resumable, straight to D:). The PI's parallel Chrome download of mini is being adopted. ⭐ Still useful: **the listed size of `nuplan-v1.1_test`** from the download page, or the Lab HEAD-probes it next pass. ⚠️ nuPlan's terms (non-commercial) still apply to our use regardless of how the bytes were fetched.
2. **The A40 pod**, with the NAVSIM pull done pod-side (navtrain + navtest + navhard); the pod is ready to be useful as soon as Stage 1 exists.
3. **TanitVFM's scoped-down recipe** (epochs over navtrain frames instead of their 1.64 B samples) — a pre-registration sign-off.

## 5 · Withdrawn from v1 (recorded, not deleted)

- **Route "B" / the PhysicalAI → nuPlan-structured adapter (v1 Stage 2)** — not the experiment. Backlog row **RE-2 withdrawn**.
- **"The teacher is unvalidatable on our corpus"** — moot; on nuPlan it is validated by the six-task suite.
- The v1 framing of TanitVFM's corpus as PhysicalAI — it is navtrain `CAM_F0`.
