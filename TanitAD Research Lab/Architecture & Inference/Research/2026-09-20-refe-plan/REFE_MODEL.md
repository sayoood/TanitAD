# REFe — the model exists, loads real DINOv3, and learns. Built 2026-09-20.

**Evidence class: MEASURED (ours)** unless marked PUBLISHED. Artifacts in `refe/` and `raw/`.
**Scope (PI, second ruling 2026-09-20):** reproduce DriveZero's camera planner on THEIR data with
**ONE** deliberate difference — **DINOv3 alone** instead of their agglomerative DriveVFM.
⚠️ This line read *"two deliberate differences — one front camera instead of four"* until
2026-09-21; the PI's four-camera ruling superseded that, and the rig is now the paper's.

## 1 · What was built

**State as of 2026-09-21 01:00.** Four cameras (`CAM_F0, CAM_L0, CAM_R0, CAM_B0`), ego **7 real
kinematic scalars**, and the frustum **inverts the lens distortion**.

| piece | file | state |
|---|---|---|
| architecture | `refe/model.py` | ✅ 4 cameras · ego 7-D · distortion inverted · **322,070,854 total / 18,991,426 trainable (5.90 %)** vs paper 338.46 M / 18.58 M (5.49 %) |
| architecture validation + regression arms | `refe/validate_model.py` | ✅ `VALIDATE_OK` — ⛔ it previously **died before section 4** on a 1-camera tensor, so no runtime arm had run since the extension |
| real DINOv3 weight loader | `refe/load_dinov3.py` | ✅ `LOAD_OK` on all three sizes |
| Stage-2 target builder | `refe/build_targets.py` | ✅ **2,182 tuples** = 1,091 × 2 **genuinely distinct** ranks, 4 channel paths per row, 73.2 % step coverage. ⛔ It **crashed on its own success path** until 2026-09-21 |
| Stage-2b per-proposal PDM targets | `refe/score_proposals.py` | ✅ self-test green, ✅ **now reaching training** — MEASURED 10.4 % sample coverage with a live score loss |
| nuPlan planner wrapper | `refe/planner.py` | ✅ 4-camera end to end; ⛔ was single-camera and would have raised the rig guard on step 1 |
| bank filter that actually scales | `code/driverl_mini_bank.yaml` | ✅ 14 val14 types, per-type cap |
| training loop + overfit control | `refe/train.py` | ✅ `TRAIN_CONTROL_OK` (L1 1.9316 → 0.3730, 80.7 %) at 4 cameras, with **gradient accumulation** |
| consumer conformance | `refe/diag_consumer_conformance.py` | ✅ `CONSUMERS_CONFORM` + `SELF_TEST_OK` (8/8 arms go red alone) |
| rank distinctness | `refe/diag_rank_distinctness.py` | ✅ `RANKS_ARE_DISTINCT` + `SELF_TEST_OK` |
| four-camera image path | `build_targets.index_images` + `train.FrameStore` | 🟡 **655 of 1,091 tuples (60.0 %) resolve all four cameras and decode**; the rest await the L0/R0/B0 fetch still running |
| training cost | `TRAINING_TIME.md` | ✅ measured — full paper scale **2,380–3,080 A40-hours = 99–128 days on ONE A40** |

## 2 · Backbone — ViT-L, by PI decision

⛔ **I defaulted to ViT-S and the PI was right to challenge it.** The honest accounting:

| backbone | REFe total | trainable | published DINOv3 PDMS |
|---|---|---|---|
| ViT-S | 30.2 M | 7.92 M | **93.88** (their Table 7) |
| ViT-B | 96.5 M | 9.35 M | not published |
| **ViT-L (chosen)** | **329.7 M** | **26.58 M** (8.06 %) | **94.55** (their Table A13) |

⭐ **The argument that decided it is not the +0.67.** Their Table 7 reports **DINOv2 ViT-S = DINOv3
ViT-S = 93.88**, i.e. *at small scale the backbone provably does not matter*. A DINOv3-only
experiment run at ViT-S would therefore have been **uninformative by construction**. At ViT-L the
backbone is where the difference appears.

⚠️ **Consequence, recorded rather than buried:** REFe is then **319.0 M** (316.9 M before the
2026-09-20 fix that took the proposal decoder from 3 to 4 layers, as the paper specifies;
trainable 11.88 M → 13.99 M, 3.75 % → 4.38 %). ⭐ **PI 2026-09-20: "no problem with the
316.9M"** — the figure and its increase are accepted. It exceeds the programme's
**sub-300M** thesis by 16.9 M. REFe is a reproduction on their data, not the deployed product, so the
thesis is not asserted to bind here — but that is a PI judgement and it is written into
`REFeConfig.backbone` where the next reader will meet it.

**Backbone is a named variant** (`REFeConfig.for_backbone("vits16"|"vitb16"|"vitl16")`), never a
hand-edit. All three are on D: and all three load strictly.

## 3 · Architecture fidelity — cross-checked against a number I did not fit

Their published student is **338.46 M / 18.58 M trainable = 5.49 %** (ViT-L, **four** cameras).
Rebuilding REFe in **that** configuration:

| | total | trainable | fraction |
|---|---|---|---|
| my reconstruction, their config | 322.89 M | 17.83 M | **5.52 %** |
| their published | 338.46 M | 18.58 M | **5.49 %** |

⭐ The heads were sized from the paper's prose, **not** fitted to that ratio, so landing within
**0.03 percentage points** is an independent check that the architecture is dimensionally right.
Absolute counts are within 4.6 % / 4.1 %; the residual is unmodelled detail (their exact decoder
geometry, the DriveVFM trunk versus a plain ViT-L).

## 4 · Controls — every one of them earned its place

| control | result |
|---|---|
| LoRA is identity at init (B zeroed) | **max diff 0.000e+00** |
| only LoRA tensors unfrozen in the trunk, **asserted by name** | 48 of 48, 0 non-LoRA |
| output shapes | traj (B, 64, 20, 3), score (B, 64, 6) |
| winner-takes-all routes gradient to one proposal | ✅ |
| scorer is detached | ✅ |
| **deliberate-regression arm: scorer re-attached** | goes **RED** — the detach check is not inert |
| overfit 8 tuples, **synthetic** images, ViT-S, batch 8 | L1 **1.3130 → 0.0388**, 97 % reduction |
| overfit 8 tuples, **REAL camera frames**, ViT-L, batch 2 | L1 **1.0781 → 0.1791**, 83.4 % reduction, **PASS** |
| **deliberate-regression arm: optimizer emptied** | loss **frozen at 1.2460** for 125 steps |
| DINOv3 load: every checkpoint tensor consumed | **0 unconsumed**, 0 unexpected gaps, all 3 sizes |

⭐ **Two of these found real defects before any GPU was rented.**

**The strict loader found a fidelity gap.** DINOv3 carries its **own** `cls_token` and **4
`reg_token`s**, and its attention was pretrained with them present. My first trunk dropped both,
which would have left a "frozen pretrained" backbone silently computing a different function. They
are **not** the same as DriveZero's 16 task registers per camera, which are added *after* the trunk.
A partial load is worse than no load: it still trains, still converges, and quietly forfeits the
reason DINOv3 was chosen.

**The target bank's own check found a halved horizon.** The simulation history logs at **10 Hz**
(MEASURED: dt 0.1000 s, 149 steps over 14.80 s) while their head is **20 steps @ 5 Hz = 4.0 s**.
Twenty consecutive steps therefore span **2.0 s**, producing a completely plausible-looking bank.
Only the check *displacement / horizon must equal measured speed* exposed it, reading exactly **2×**
off. `STRIDE = 2` fixes it and is also the policy's own cadence (`interval_s = 0.2`).
⚠️ The corrected check then read 12 % off, and **that** was the control being wrong — a 4 s mean must
exceed the instantaneous speed at t whenever the ego accelerates. The exact identity (path length vs
∫v dt) holds at **0.41 % median**. The threshold was not moved; the control was fixed.

## 4b · REFe learns from real nuPlan camera frames

**MEASURED 2026-09-20:** ViT-L, real CAM_F0 JPEGs, batch 2 — `traj L1 1.0781 → 0.1791`, **PASS**.
The whole path works end to end: JPEG → resize → frozen DINOv3 ViT-L trunk with LoRA → 64 WTA
proposals → L1 against the teacher's realised 4 s trajectory.

⛔ **This is NOT comparable to the 0.0388 above, and the difference must not be read as "real
images are harder".** Three variables moved at once between those runs:

| run | images | backbone | batch | final L1 |
|---|---|---|---|---|
| first | synthetic | ViT-S | 8 | 0.0388 |
| second | **REAL** | **ViT-L** | **2** | 0.1791 |

That is the programme's own *ten-levers-on-two-axes* error. An isolating arm is running: ViT-S at
batch 8, synthetic vs real, moving **only** the images.

⚠️ **I also over-read the trace mid-run** and called it a plateau at ~0.15; it continued to 0.0766
and 0.0811 before ending at 0.1791. Batch 2 produces a noisy trace, not a flat one. Corrected in
`raw/refe_wta_batch_starvation.txt` rather than left standing.

⭐ **What IS established, by arithmetic rather than inference:** WTA gives gradient to exactly one
proposal per **sample**, so at most `batch` of the 64 proposals are touched per step. MEASURED
winner diversity: **1–5 per step at batch 8**, but **1/2 on every logged step at batch 2** — both
samples choosing the same proposal. **Their training used batch 256.** ⇒ the dev box cannot train
this head properly at any batch it can hold, which is why the pod is *necessary* rather than merely
faster. `train.py` now prints the WTA reach on every run.

## 5 · Weights

`facebook/dinov3-*` returns **HTTP 401** (gated, needs a licence acceptance).
`timm/vit_{small,base,large}_patch16_dinov3.lvd1689m` are **ungated** and are the same `lvd1689m`
pretraining: **84 MB / 328 MB / 1.2 GB**, all on D:.

⚠️ Dependencies were kept out of the eval venv beyond `--no-deps` additions of
`huggingface_hub`, `safetensors`, `truststore`, because installing into it is how this project
previously replaced torch with a wheel the driver could not run. **Torch was re-verified after the
install with a real CUDA `conv2d`, not merely an `import`.**

## 6 · Target bank (Stage 2)

**1,746 tuples** from 8 scenarios × 2 route ranks: front-camera path, ego kinematics, **goal points**
(the augmentation input), and the teacher's realised 4 s trajectory in the ego frame.

* camera pairing: **100 %** of DB rows resolved, mean offset **24 ms**, max 38 ms (camera and history
  are both 10 Hz).
* coverage 73.2 % — the dropped remainder is the final 4 s of each scenario, which cannot form a
  target. By design, not loss.
* **augmentation delivers diversity**: rank 0 vs rank 1 differ by >0.5 m on **71.6 %** of trajectory
  steps and **96.9 %** of goals, at the *same* scene state.

⛔ **Not built yet:** the scorer's six-PDM-component target, which needs a rollout per proposal to
score. The trajectory loss is the main supervision and comes first.

## 7 · What is next

| item | state |
|---|---|
| camera frames for the scenario logs | ✅ **landed** — targeted CAM_F0 fetch; REFe trains on real JPEGs |
| nuPlan planner wrapper | ✅ **built** (`refe/planner.py`) — REFe is scoreable in THEIR harness |
| scale the target bank beyond 8 scenarios | ⏳ the rehearsal is not a training set |
| per-proposal PDM targets for the scorer | ⛔ not built — selection at inference is arbitrary until then |
| **trained weights at scale** | ⛔ **this is what the pod is for** — see `POD_HANDOFF.md` |

⚠️ **Dev-box ceiling:** the 4060 is **8.00 GB**. ViT-L peaks at 3.26 GB (batch 1) and 5.36 GB
(batch 2); a reported "batch 4 at 9.37 GB OK" is **Windows spilling to host RAM rather than raising
OutOfMemoryError** — it succeeds and crawls. Local ceiling is batch 2.

⛔ **And batch 2 is not merely slow, it is the wrong regime**: WTA touches at most `batch` of the 64
proposals per step, so the dev box cannot train this head properly at any batch it can hold. Their
batch was 256. That is why the pod is **necessary**, not just faster.

**Cost, MEASURED:** 0.780 s/sample on this 4060 ⇒ a full reproduction at their 337 K-sample scale is
**~608 A40-hours ≈ 25 days on one card**. `POD_HANDOFF.md` gives four scaled arms; the recommended
first day is **~3.5 GPU-hours** and ends with a REFe that can be scored against the teacher's 97.19.

## 8 · Stage 2b — the scorer's targets, and an estimate I got wrong twice

REFe predicts six PDM components per proposal. Producing their targets needs THEIR six released
reward calculators, and the question was always how much has to be built to feed them.

| version | claim | basis | verdict |
|---|---|---|---|
| v1 | "their builder already emits most of the ScenarioData arrays" | a NAME GREP reporting 14/14 | ⛔ wrong — a name in a file is not an array produced |
| v2 | "only 3 of 14 fields come from it; budget real work" | ran `_empty_map_arrays` | ⛔ wrong — that is the MAP helper, the wrong object |
| v3 | **`DriveRLNuPlanFeatureBuilder.build()` already returns a COMPLETE `ScenarioData`** | ran the import and read the construction site | ✅ **verified** |

⭐ **Self-test green** (`raw/refe_scorer_selftest.txt`): the builder imports, **agent 0 is the ego**
confirmed from source, the polygon cache is present, and **all six calculators import**.

**How a proposal is scored:** the calculators read agent geometry *out of* the `ScenarioData` they
are handed — they take no trajectory argument — so a proposal is scored by appending its poses to
the ego track and re-running them.

⚠️ **The single way to get this silently wrong:** the polygons are **cached**. A mutation without
`clean_polygon_cache()` scores the PREVIOUS state and returns a completely plausible, completely
wrong target bank. `inject_proposal` always clears it.

### 8.1 · Two more wrong versions, and the control that ended it (2026-09-20)

| version | claim | basis | verdict |
|---|---|---|---|
| v4 | "it runs, returns real numbers, and is **INERT** — 0 of 9 signals differ" | an end-to-end run with three candidates | ⛔ wrong — see below |
| v5 | **the scorer DISCRIMINATES**: teacher `OffRoad.info` 0.0000, over-curb 1.0000 (reward −1.6994), **10 of 28** signals separating | `refe/scorer_gate.py`, `raw/refe_scorer_discriminates.txt` | ✅ **MEASURED** |

⛔ **v4 was the most dangerous of the five, because the pipeline genuinely ran.** Three defects,
all mine, each of which alone produces a confident false negative:

1. **The readout was the inert part.** Every calculator writes `{"reward":…, "info":…}`. `reward`
   is the weighted, saturated training signal and is flat across candidates *by design*; `info` is
   the raw event/offset channel. My reader collapsed each dict to `reward`. The discriminating
   value was in the very first run and I was throwing it away.
2. **Endpoint scoring hid a mid-horizon violation.** `extract_recent_agent_polygons` reads only
   timesteps −1 and −2 and these are per-step **crossing** detectors. MEASURED: the identical
   over-curb path reads `OffRoad.info` **0.0000** at its endpoint and **1.0000** step by step.
3. **The deliberate-regression arm never committed its violation.** The "25 m sideways veer" moves
   the ego *away* from the nearest curb — corner distance 10.66 m at 0 m, 19.06 m at +14 m, minimum
   0.87 m at +40 m, never touching. OffRoad was right to stay silent. ⭐ The replacement is an
   **analytic target** aimed at a curb segment read out of the map, so the expected verdict is
   known by construction.

⚠️ **And the fix I had named and queued was not among them.** Populating `lanes_centers_groups` /
`_ids` / `_next_groups` is implemented and **changes nothing** — the with- and without-enrichment
arms are identical on all 28 signals. Diagnosing from structure ("this array is empty, that must be
why") instead of from a control cost the whole detour.

⚠️ **A fourth thing the same run caught, never part of the claim: the targets were not
reproducible.** The builder's `domain_randomization` defaults to `None`, so the `ScenarioData`
**draws** its reward weights — `collision_reward_weight` −1.0415 rather than the configured −1.0,
giving −1.5194 in one process and −1.7086 in the next for identical input, while being stable 5/5
*within* a process. Their released yaml already carries the deterministic values; `scorer_gate.py`
now passes them and the weights read exactly −1.0 across processes.

### 8.2 · Stage 2c — the target bank

`refe/build_scorer_targets.py` produces the six-component targets for a structured candidate set:
the teacher's path, four lateral offsets, two longitudinal scalings, a stationary ego, and an
over-curb control when a curb is in range. **MEASURED on a first bank: 11 of 28 target keys vary
across candidates within a single frame**, and every frame's controls separated (0 aborted).

⛔ **A stated difference from DriveZero.** They score the STUDENT'S OWN 64 proposals online, because
the scorer trains jointly with the proposal head. That is a full calculator rollout per proposal per
optimisation step and this rig cannot pay it. We precompute against a fixed basis, so the scorer
learns to rank trajectories drawn from **our** perturbation family and anything far outside it is
extrapolation. Report this with any scorer number.

**Cost, MEASURED (single core, CPU):** one calculator pass 41.6 ms; rollout 842 ms/candidate at
stride 1 and 444 ms at stride 2; whole-frame 12 s at 9 candidates, of which the `ScenarioData`
build is 2.5–4.9 s. A batched multi-proposal path was tried and gives only **2.5× at N=8** with
three of six calculators still untiled, so it is not the lever; parallel processes are.

⛔ **RETRACTED 2026-09-20 — THAT GUARD WAS INERT AND THIS PARAGRAPH CITED IT AS PROOF.** It read:
*"All six components discriminate, and `train.py` proves it rather than assuming it… none inert."*
The guard pooled **every row in the bank** into one standard deviation per component. A component
that is **constant across the candidates of every single frame**, and merely differs between
frames, shows a healthy pooled spread and passes — while being unable to rank anything. The
scorer chooses among the candidates **of one frame**, so the only measurement that answers the
question is the spread **within** a frame.

**MEASURED with the corrected guard** on 5,573 rows over 620 frames:

| component | mean | pooled sd | varies in |
|---|---|---|---|
| off_road | 0.720 | 0.449 | 99.8 % |
| collision | 0.625 | 0.484 | 75.5 % |
| ttc | 0.614 | 0.475 | 86.8 % |
| **comfort** | 0.969 | **0.120** | **0.0 %  ← CANNOT RANK** |
| goal_reaching | 0.910 | 0.286 | 52.3 % |
| center_line | 0.825 | 0.108 | 76.8 % |

⛔ **`comfort` varies in 0.0 % of frames**, and its pooled sd is *larger* than `center_line`'s,
which does rank in 76.8 % — so the old number did not even order the components correctly. A
component constant within a frame adds the same bias to every candidate and cannot move the
argmax, so a sixth of the scoring head does no work.
⭐ **What caught it was not a re-reading** but an independent conformance review measuring
within-frame variation directly; I then reproduced it with the corrected guard.
Root-cause class: *a check whose question is narrower than the claim hung on it*.
Detail: `raw/refe_variance_guard_was_inert.txt`.
⚠️ **A claim I made and retracted the same hour:** I wrote that `goal_reaching` was unvalidated
because it read 0.0000 for every candidate. That was **one frame**. Across the bank it orders
exactly as it should — `lon x1.5` reaches a goal **45/74**, `teacher` **13/74**, `lat±2` **10/74**,
`stopped` **0/73**. `GoalReaching` fires when a step's swept segment passes within
`goal_reaching_threshold`, so a frame whose goals sit ~12 s ahead reads 0 for everything, correctly.
⇒ *a single-frame observation is not a property of the component* — the same over-generalisation
that produced the INERT verdict two hours earlier.

⛔ **The supervision is an ASSIGNMENT, and the loop reports its error.** The banked targets belong to
CANDIDATE trajectories; the scorer head scores the model's OWN proposals. Each candidate is attached
to the nearest proposal and `assign_d` is printed every log line, because it bounds the whole
approximation: a large distance means the scorer is being taught about paths this model does not
produce. MEASURED on an untrained ViT-S: **6.73 m** at step 0. Coverage is printed too, and a run
that sees **zero** real targets says so in capitals rather than reporting a small loss.

### 8.3 · The supervision lands where it should — and the first version of this table was void

`raw/refe_assignment_lands.txt`. MEASURED 2026-09-20, ViT-S, 400 steps, batch 8, CUDA, bank 2,096
rows over 233 frames, trajectory L1 **6.1725 → 0.4634** (92.5 % reduction), scorer coverage 22.7 %.

| candidate | n | mean assignment distance |
|---|---|---|
| teacher | 727 | **3.54 m** |
| lat −2 / +2 | 727 | 3.85 / 3.90 m |
| lat −4 / +4 | 727 | 4.51 / 4.58 m |
| lon ×0.5 / ×1.5 | 727 | 7.41 / 9.74 m |
| stopped | 727 | 14.62 m |
| over-curb | 727 | 16.46 m |

Monotone and in the predicted order: teacher closest, lateral offsets ordered by their own
magnitude, then longitudinal, then the two deliberately-bad arms furthest.

⛔ **THE FIRST RUN OF THIS TABLE WAS VOID, AND ITS ORDERING WAS ALSO MONOTONE.** `ScorerBank` keyed
its candidates by `(log_name, step)`, but `log_name` is the **drive log** and one drive log holds
several scenarios whose step indices overlap — **50 of 154 keys collided**, one log carrying 3
tokens, putting **27** rows on a key whose candidate set has **9**. Frames were supervised with
another scene's trajectories. It read teacher 5.73 / lat 5.91–6.55 / lon 8.27–10.80 / stopped 15.35
/ over-curb 17.21: *the same clean ordering*.
⭐ **A qualitative result being exactly what you predicted is not evidence that it is sound.** What
refuted it was **arithmetic** — `n = 837` against **605** covered samples, impossible for a
candidate appearing once per sample. After the fix every candidate reads `n = 727` against 727
covered samples, which is the identity that must hold. Detail:
`raw/refe_scorer_key_collision.txt`.

⚠️ **And this table still cannot answer its own question, so `train.py` now prints a second
column.** These are whole-run means over every step **including the untrained ones** (step 0 read
11.89 m pooled, the last steps ~5.4 m), so 3.54 m blends *"the model was bad at the start"* with
*"the supervision is mis-assigned"* — the two things the report exists to separate. A trailing
window over the last quarter of the run is now accumulated alongside it. **Quote the trailing one.**

⚠️ **An uncovered batch now contributes NOTHING rather than a zero target.** A zero label teaches
the scorer that every proposal violates everything, which is worse than no supervision. At 25 %
frame coverage and batch 8 that was **0.75⁸ = 10 %** of all steps. The count is printed.

⛔ **Not done:** the scorer's influence on *selection* has not been measured. Training against real
targets is necessary and not sufficient — until a selection metric moves, any closed-loop number is
a *trajectory* result, not a *selection* result.

## 9 · First REFe training on the real bank (not a memorisation test)

**MEASURED 2026-09-20:** ViT-L, real CAM_F0 frames, **1,964 tuples** over 10 scenario types and
both route ranks, batch 2, 900 steps — `traj L1 6.0726 → 0.6934`, **88.6 % reduction**.

⛔ **What this is NOT.** It is not a driving result and not comparable to anything published. It
says the full path — JPEG → frozen DINOv3 ViT-L + LoRA → 64 WTA proposals → L1 against the
teacher's realised 4 s trajectory — learns on real data at bank scale. The loss is noisy
(0.26 … 0.70 across the last 300 steps) because batch 2 is the dev-box ceiling.

⚠️ **And batch 2 is the wrong REGIME, not merely slow.** The run prints its own limitation:
`WTA reach: at most 2 of 64 proposals updated per step (3.1 %)`. Winner-takes-all gives gradient
to one proposal per SAMPLE, so at batch 2 the head is learning roughly one mode, not 64.
**Their training used batch 256.** No conclusion about proposal diversity can be drawn from this
run, and none is.

⭐ The bank it trained on is the day's whole pipeline end to end: teacher rollouts under two
augmented routes → 4 s targets at the correct stride → front-camera frames fetched by HTTP range
from archives never downloaded → 100 % camera pairing at a 24 ms mean offset.
