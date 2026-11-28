# Why refcv8 drives poorly — and the refcv9 plan (PI question, 2026-10-21)

*Master Mind, 2026-10-21 19:25 Berlin (clock read 19:22). Answers the PI's question of 2026-10-21, verbatim scope: validated
improvements for refcv9; the 3D box head and the BEV head; the tactical layer; route following; consistency between
trajectory and tactical plan; the diffusion planner and its conformance to DiffusionDrive v1 / v2; the main reasons for the
poor results; the plan for excellent driving results.*

**Evidence base.** Three read-only evidence passes, run on 2026-10-21 at branch tip `32a0c6a`:

| pass | covers | model |
|---|---|---|
| A | box and map heads | sonnet |
| B | tactical layer, route following, consistency | sonnet |
| C | diffusion planner vs papers | opus |

Plus the registry (`Project Steering/MODEL_REGISTRY.md` §14–§16).

Pass B's recomputations were RE-RUN by the Master Mind at 19:21. Every quoted value reproduced. Scripts are in `code/an*.py`
and outputs in `raw/tactical/*.txt`. Those scripts import `taniteval/taniteval/tactical_conditioning.py`; the copy they ran
against is blob `b5f0b15c`, identical to the repo file.

**Stamps unless stated otherwise:**
- refcv8 = `refcv8-r101-s0` EMA-48,852, per-frame nav feed (the PI's option (a)).
- PhysicalAI eval139: T1 self-action open loop on logged frames.
- One training seed.
- Paired episode-cluster bootstrap. It answers *another draw of EPISODES* only — not another training run, and not another
  inference run.
- NavSim: NAVSIM v1.1 PDMS, non-reactive pseudo-simulation, paired log-cluster bootstrap.

## 1. Validated levers for refcv9 (MEASURED)

| lever | measured effect | source |
|---|---|---|
| Protect the decoder | Decoder-only blend with refcv7 at λ 0.5: navtrain **76.36 / 75.68** vs EMA-48,852 63.39 / 63.02, **+12.97 [+10.55, +15.68]** / +12.66.<br>Full θ0.5 on navtest: 72.78 / 72.90 vs 57.84 (raw plan R8_LEGAL).<br>Blending only the tactical stack is WORSE than EMA (−3.58* / −4.57*). | registry §16.8, §16.11, §16.12 |
| Safety-scoring selector | REFe scorer-only SFT-4: **+3.79 [+3.05, +4.54]** navtest PDMS (on REFe, not refc). | §14.2, D-REFE-SFT-4 |
| Re-timing / progress | Frozen train-gain re-timing: E1 **−0.032** separated.<br>The ego-scalar ridge floor reaches −0.033, so an in-model progress head must beat it. | PLAN_REFCV9 L1 |
| K-draw averaging | E1 **−0.0225 / −0.0210** at 4× inference. | H-R8-KDRAW-1 |
| Box NMS at inference | refcv7: AP@2m 0.249 → **0.349**, boxes per object 2 → 1.07.<br>⚠️ EVAL-DIAG with thresholds fit on eval. The train-side refit is running (`C:/Users/Admin/refcv8_pf/perc8/`). | WP-D `raw/box_probe0.json` |

**Eliminated (not refcv9 levers):**
- ×4 map weight (H-R8-MAPW-1).
- Tactical class decisions: v8F A1 (aa8366d) puts 0.07–0.12 of the headroom in decisions.
- RL ports on refcv7 (H-DDV2RL-1/2).

## 2. Box head and map (BEV) head

**Box head**

refcv7 @50,400, eval-grade (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/step50400_a6text/perc_score.json`; 4,532 windows / 139 episodes; T0):
- Pooled AP@2m 0.2491.
- Class-mean mAP@0.5/1/2/4 m: 0.006 / 0.028 / 0.070 / 0.102.
- AUROC 0.907.
- **Only 119 confident boxes at the declared 0.5 gate across 4,532 windows** (recall 0.008).
- About 2 boxes per object.
- The top-scored slot is the nearest object on only 34.5 % of windows.

Root causes, all MEASURED:
- Unstable slot assignment: the one-frame overfit ladder fails on every rung.
- The 300 learned anchors never moved (0.031 m).
- Presence never becomes confident.
- The error is mostly longitudinal: |Δx| 0.96 m vs |Δy| 0.38 m.
- About one epoch of training.

refcv8 has **no eval-grade read**: `perc_pass_r8.py` ran only as a 4-window smoke, and the eval-grade pass is running now on Thor. Trainer-val (121 windows): AP@2m 0.317 vs 0.229, with the same structure (conf_ratio 0.009).

**Map head** (refcv7, eval-grade, 4,686 windows)

| class | IoU by distance band | reference |
|---|---|---|
| drivable | 0–20 m .606 · 20–40 m .604 · 40–60 m .538 · 60–80 m .504 · 80–100 m .478 | positional prior .500 / .442 / .395 / .378 / .370 |
| lane | .195 near, ≈ 0 beyond 20 m | |
| crosswalk / arrow | ≈ 0 beyond 20 m | |
| edge | .001 | |

- BAR-M7-1..4 all FAIL.
- refcv8's in-run map (16 windows) is unchanged.
- Causes: about 1.3 stride-8 feature rows cover 40–100 m, and the shared 0.25 m BEV encoder takes 97 % of its update from the box3d loss (P-GRAD).
- The SAM3 GT noise ceiling for thin classes is NOT MEASURED.

⚠️ **The planner never reads the box3d slots.**
- They are consumed only by the box loss and by eval: `refc_v3_train.py:6418-6420` and `refc.py:5211` at fb47a58.
- The planner reads the separate 300-query agent head (`refc.py:5037-5043`, `:5174`) and the pooled BEV (`map_hires_pool`, `refc.py:4826`).
- The causal effect of perception on planning (zero or shuffle the agent tokens and the BEV) has **never been measured**. That is the next queued test.

Trunk update share (P-GRAD, refcv7):

| loss | trunk update share |
|---|---|
| agent | 50.1 % |
| box3d | 35.7 % |
| trajectory | **0.016 %** |
| map | 0.014 % |

**Registry corrections owed** (logged in the register; not edited here):
1. §15.2 labels box 0.248 / 0.131 (NMS 0.349) and lane 0.164 / edge 0.042 as "trainer-val". They are **EVAL-DIAG** (1,061 / 1,112 windows, oracle thresholds). The true trainer-val values are AP 0.2287 and lane 0.062 / edge 0.0.
2. The B1 guard reference (0.21211) is the minimum of refcv7's **RAW** in-run AP over 40–50.4k. §16 calls it "EMA", but refcv7 had no EMA. At step 1,000 the guard compared an EMA that was 82 % warm-start weights against that reference, while the raw head read 0.160. The guard was blind to that regression.

## 3. Tactical layer (seed 0; the heads are deterministic) — `raw/tactical/an2.txt`, `an6.txt`

**Lateral**
- Accuracy **0.9415** vs constant 0.9039 (+0.0376 [+0.0124, +0.0611]).
- A **lookup of the fed nav token alone scores 0.9189**, so the head adds +0.0226 [+0.0071, +0.0408].
- Token FOLLOW: 3,352 / 3,352 predicted LANE_KEEP.
- Turn recall with nav zeroed: TURN_L 0.585 → 0.255; TURN_R 0.751 → 0.132.
- Vision-only (nav zeroed) 0.8873, **below the constant** (−0.0166 [−0.0343, −0.0015]).
- ⇒ The lateral layer is an **echo of the route command**.

**Longitudinal**
- Accuracy **0.6103** vs constant 0.3352 (+0.2751 [+0.2095, +0.3369]); balanced 0.466.
- BRAKE_TO missed 55.7 %; CREEP 2 of 35 correct; HOLD 10 of 31.
- False-decel on label CRUISE 0.103.

**Strategic:** UNAVAILABLE (`--no-strategic`, `--r8-no-rc`), n = 0.

**Headroom location (T0 analogy, v8F-mini A1):** continuous constraints carry 0.88–0.90 of it, class decisions 0.07–0.12. The read on refcv8's own continuous targets is running (`C:/Users/Admin/refcv8_pf/tacconstr/`).

## 4. Route following (proxy; S6 grid; 461 turn windows) — `raw/tactical/an4_s{0,1}.txt`

| arm | turn direction correct | heading within 15° |
|---|---|---|
| refcv8 | **0.870 [0.794, 0.925]** | 0.564 (bar 0.70) |
| refcv7 (same windows) | 0.807 [0.714, 0.882] | 0.490 |
| refcv8, nav zeroed | 0.716 | |

- The direction bar of 0.95 is **NOT MET**.
- By nav token: left 0.729 (n 85), right 0.886 (n 166).
- Controls: GT-vs-GT 1.0 and hold-straight 0.0 (both as required).

NavSim (registry §16, `PRIOR_GAP_DIAG`), R8_RT − kinematic prior:
- straight-command tokens **−19.64 [−25.4, −14.9]** (n 1,444; losses in NC and TTC);
- right turns **+32.5 [+22.9, +40.7]** (n 162).

## 5. Consistency between plan and tactical head — `raw/tactical/an5_*.txt`, `an7.txt`

**Head vs plan agreement:**

| arm | lateral (permuted-head null) | longitudinal 3-way (null) |
|---|---|---|
| refcv8 | **0.877** (0.791) | **0.613** (0.391) |
| refcv7 | 0.903 (0.838) | 0.618 |

**Braking obedience:** when the head says decel, refcv8's plan decelerates **646 / 883 = 73 %**; refcv7's does 40 %.

**False-decel on GT cruise** (S6, 1,949 windows):

| arm | false-decel |
|---|---|
| refcv8 | **0.150 [0.108, 0.194]** |
| refcv7 | 0.0965 |
| paired difference | **+0.053 [+0.024, +0.087]** (s1 +0.048 [+0.017, +0.082]) |

- Missed-decel: refcv8 0.428 vs refcv7 0.548.
- Nav zeroed: false-decel 0.257.

**Turn obedience is weak:** plan-derived turn recall is 0.46 (L) / 0.47 (R), vs the head's own 0.59 / 0.75.

**Mechanism found in source (H-R9-ALLOC-ORACLE-1):**
- Training always forces the GT tactical hypothesis into ≥ 4 of the 32 allocated candidates (`refc.py:2512-2513`).
- Inference allocates by the head's posterior, with no gating.

The formal bar CONS-PICK-TAG ≥ 0.95 is not yet scored; it needs per-candidate tags (A6 dense capture).

**Control notes (an3):**
- The S6 / decisions join is verified: `plan_full` slots [1,3,4,5,6,7] equal the S6 `os` with max abs diff **0.0**.
- The second printed "control" (g vs `gt_future_ext`, 485 m) compares two different frames. It is non-informative and is not used by any quoted number.

## 6. The diffusion planner vs DiffusionDrive v1 (`2411.15139`) and v2 (`2512.07745`) — both PDFs banked

| element | DiffusionDrive v1 / v2 | refcv8 | verdict |
|---|---|---|---|
| Truncated schedule | t < 50 / 1000; inference from t = 8, 2 DDIM steps | same (`--f1-random-t --f2-dd-step`, `refc.py:546-549`) | MATCH |
| Training losses and timestep conditioning | per-layer loss / detach, AdaLN timestep, focal BCE | same (`--f3 --f4-adaln --f5-focal --f6-w-u0-zero`) | MATCH |
| Noise space | metres | (a_lon, a_lat) control space (D-REFCV5-PLAN-4) | DEVIATION, deliberate |
| Anchors | 20 k-means paths | 117 speed-conditioned + 32 allocated = 149 | DEVIATION (F9) |
| Output | x0 waypoints | residual on `ha0_ext_pose` | DEVIATION |
| Cross-attention | camera + LiDAR BEV | camera-lifted SAM3 BEV, agent head, ego FiLM; one camera | PARTIAL |
| Selection | v1: classifier argmax. v2: sub-score selector (NC / EP / DAC / TTC / C) | priors + grafts + clamps + 4 log-imitation critics; drivable critic OFF | v2 selector ABSENT |
| RL post-training | v2: GRPO with PDM reward | none (the WP-RL port on refcv7 is paused at 159 / 2,000) | ABSENT |
| Sensors and data | 3 cameras + LiDAR; navtrain | 1 camera, 416×1024, R101; PhysicalAI | DEVIATION (PI rulings) |

**Quality on PhysicalAI** (2 s; `…/2026-10-20-refcv8-perframe-final/raw/step48852ema_pf/battery_pf/analysis_s{0,1}.json`):

| plan | 2 s ADE |
|---|---|
| refcv8 own pick | **0.2956 / 0.2985 m** (seed 0 / seed 1) |
| refcv7 pick (same windows) | 0.2784 |
| kinematic prior `ha0_ext_pose` | **0.2886** |
| anchor nearest the human (oracle) | 0.2003 |

- pick − oracle: −0.0953 [−0.1117, −0.0802].
- The pick matches that oracle on 44.2 % of windows.
- One anchor is picked on 42.4 % of windows.
- ⇒ **At 2 s, refcv8's pick does not beat the kinematic prior.**

The fan ceiling (best of 149) and the random-pick control are running on Thor (`C:/Users/Admin/refcv8_pf/fanceil/`).

**NavSim navtest PDMS:**

| arm | PDMS |
|---|---|
| refcv8 raw plan | 57.84 |
| θ0.5 blend | 72.78 |
| refcv7 (identical diffusion flags) | 70.20 |
| REFe | 88.82 / 89.15 |
| DiffusionDrive v1 (PUBLISHED) | 88.1 |
| DiffusionDriveV2 (PUBLISHED) | 91.2 |

⇒ The mechanism is not the gap. The refcv7 → refcv8 drop is decoder drift (§16.11).

## 7. Main reasons, ranked by measured effect

1. **Zero-shot transfer from one front camera on PhysicalAI.** REFe (navtrain, 4 cameras, DINOv3 ViT-L) reaches 88.8–89.2, +16 over the best refc blend. REFe changes data, cameras and backbone together, so they are not yet separable. The decision experiment is in §8.
2. **Decoder drift in the late training phase:** about −12 PDMS, recovered by decoder-only blending. On PhysicalAI, raw S6 ADE also worsened from 40k to 48,852 (+0.168).
3. **The trunk is shaped by perception, not planning:** trajectory loss 0.016 % of the trunk update; box3d + agent 86 %.
4. **A selector with no safety terms.** REFe's gap to the human is 4.15 of 6.30 points in NC/DAC-zeroed tokens (D-REFE-GAP-1b).
5. **Tactical problems:** the lateral echo of the route token, and braking over-coupled to the head (the train/inference allocation mismatch).
6. **Weak perception,** and the box head is not wired to the planner. Its causal value is unmeasured.
7. **No strategic layer.**

## 8. Plan

**Running now (no new authorisation):**
- fan ceiling (Thor);
- refcv8 eval-grade perception plus train-side NMS / gate refit (Thor, after the fan job);
- refcv8 continuous tactical targets (CPU);
- SHIP-R8 on the WiSE candidates (dev-box queue).

**Next (queued):** the perception → planner causal test (zero or shuffle the agent tokens and the BEV).

**Decision experiment (needs PI authorisation):** fine-tune θ_DEC (front camera only) on NAVSIM navtrain, ~1–2 GPU-days (ESTIMATED).
- **≥ ~85** ⇒ data was the gap; keep the refc architecture.
- **≤ ~75 while REFe is at 89** ⇒ sensors / backbone; refcv9 adopts REFe's DINOv3 backbone and multi-camera input.

Both outcomes must be registered before the run.

**refcv9 recipe, in order of measured effect:**
1. Decoder protection (anchor to refcv7's decoder, or freeze late).
2. A DDv2-style safety selector (NC / DAC / TTC / EP / C). Labels: SAM3 drivable area plus agent boxes on PhysicalAI; the PDM scorer on NavSim.
3. A planner share of the trunk update well above 0.016 % (HYPOTHESIS).
4. Tactical:
   - anneal the GT allocation floor and train on predicted classes;
   - a nav-blind lateral loss;
   - constraint (speed-target) accuracy over classes.
5. Progress head (must beat the scalar floor) and K-draw distillation.
6. Perception fixes (anchored / HQS queries, quality-aware presence, CDN; map-owned encoder plus a stride-4 near lift; drivable area first) — only if the causal test shows perception matters for the plan.
7. Multi-camera input and the strategic layer with a predicted goal (PI).

**Parallel NavSim track:** REFe 89.15 → beyond DDv2's 91.2:
- seed-ensemble proposals (no training);
- add TTC to the selector (TTC is P2-s1's adverse sub-score);
- then GRPO RL.

**PI decisions:**
- compute for the navtrain fine-tune of refc;
- whether to push REFe on the A40;
- multi-camera ingest;
- D-R8-WISE-NAVTEST.
