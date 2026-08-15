# Alpamayo 2 Super — technical analysis and what TanitAD should do about it

**Date:** 2026-08-05 · **Author:** chief-scientist review · **Subject:** `nvidia/Alpamayo2-Super`,
released **2026-08-04**.

⛔ **EVIDENCE DISCIPLINE.** This model postdates my training data entirely, so **nothing here comes
from memory.** Every fact is `PUBLISHED` from a primary source retrieved 2026-08-05:
the HF model card, the machine-readable `config.json`, the HF API metadata, the `NVlabs/alpamayo2`
README, and arXiv **2511.00088** (the predecessor paper). Where a source is silent I say so rather
than infer. Marketing/secondary coverage is used for nothing load-bearing.

---

## 1. What it is

| | |
|---|---|
| **Total parameters** | **34.3 B** = 32 B VLM backbone + **2.3 B** action expert |
| **Class** | Vision-Language-**Action** (VLA) with a **diffusion** action decoder |
| **Backbone** | **Qwen3-VL** (`vlm_class: Qwen3VLForConditionalGeneration`, `model_type: qwen3_vl`), branded as **Cosmos 3 Super Reasoner** |
| **HF id / arch** | `nvidia/Alpamayo2-Super` · `architectures: ["Alpamayo2Super"]` · `dtype: bfloat16` |
| **On-disk** | **71.65 GB**, 15 safetensors shards, 32 files |
| **Gating** | HF API reports **`gated: false`** — openly downloadable. ⚠️ The GitHub README still says *"Access to the **gated** Alpamayo 2 Super model"*, so the gate was lifted after the README was written. Treat the README as stale on this point. |
| **Licence** | **weights: OpenMDW-1.1** (Linux Foundation permissive — fine-tuning, derivatives, **commercial redistribution**). **Code: Apache-2.0.** |
| **Uptake at review time** | 45 downloads, 69 likes (≈24 h after release) |

### The action expert — the interesting half

`config.json` is far more informative than the prose. The expert is **not** a waypoint regressor:

```
action_space_cfg: UnicycleAccelCurvatureActionSpace
    n_waypoints 64 · dt 0.1
    accel_bounds     [-9.8, +9.8]    accel_mean 0.0290   accel_std 0.6810
    curvature_bounds [-0.33, +0.33]  curv_mean  0.000269 curv_std  0.02615
    ridge/lambda terms on a, kappa, theta, v   (least-squares action fit)
diffusion_cfg: FlowMatching
    int_method euler · train_timestep_sampler beta
    inference_guidance_weight 3.0 · use_classifier_free_guidance false
    train_ignore_guidance_rate 0.1
action_in_proj: PerWaypointActionInProjV2 (Fourier feats: 20, max_freq 100, 2 enc layers, h=512)
expert_update_cfg: hidden 1536 · intermediate 6144 · heads 16 · head_dim 128
expert llm_config: qwen3_vl_text · 64 layers · 16 heads / 8 KV heads (GQA)
                   vocab 155,776 · rope_theta 5e6 · max_pos 262,144
expert_non_causal_attention: true
cotrain_expert_vlm: false
```

Five things follow, and they matter more than the parameter count:

1. **The action space is `(acceleration, curvature)` on a unicycle model**, integrated at `dt = 0.1`
   to 64 waypoints. The model does **not** emit free XY. **Every output is dynamically feasible by
   construction** — no bounds check, no post-hoc smoother, no "physically impossible path" failure
   mode. The bounds `a ∈ [-9.8, 9.8]`, `κ ∈ [-0.33, 0.33]` (≈3 m minimum turn radius) are the
   vehicle envelope, enforced in the parameterisation.
2. **Flow matching, not DDPM.** Euler integration, **10 steps** at inference. This is the
   real-time-ability lever: 10 function evaluations of a **2.3 B** expert, not of the 32 B backbone.
3. **The expert is 64 layers at hidden 1536** — deep and narrow, matching the backbone's layer
   count. That is the π0-style interleaved pattern: the expert attends into the backbone's per-layer
   KV cache rather than consuming a single pooled embedding. `expert_non_causal_attention: true`
   means the expert sees the whole trajectory jointly (it is denoising a full 64-step plan, not
   autoregressing it).
4. **`cotrain_expert_vlm: false`** — in this released configuration the expert is trained against a
   **frozen** backbone. The reasoning and the acting are decoupled at training time.
5. **There is a SECOND, discrete trajectory path alongside the diffusion one.** The config carries a
   full trajectory *tokenizer*: `future_vocab_size 3000`, `history_vocab_size 1000`,
   `traj_vocab_size 4000`, `tokens_per_future_traj 128`, `tokens_per_history_traj 45`, plus special
   tokens `future_start/end/pad`, `history_start/end/pad`. So the **VLM can read and write
   trajectories as text tokens**, while the diffusion expert produces the precise continuous plan.
   The language model reasons *about* trajectories in its own vocabulary.

### Inputs

| | |
|---|---|
| **Cameras** | **6 of 7** per task, **4 synchronised frames each**, ending at `t0` (nominally `t0−0.3, −0.2, −0.1, 0` s) |
| **Camera IDs — trajectory / meta-action / auto-label / grounding** | `[0,1,2,3,5,6]` = cross-left, **front wide**, cross-right, rear-left, rear-right, **front tele** |
| **Camera IDs — VQA** | `[0,1,2,3,4,5]` = cross-left, front wide, cross-right, rear-left, **rear tele**, rear-right |
| **Resolution** | **Dynamic**, not fixed: `min_pixels 163,840` … `max_pixels 196,608` per image (≈ 512×384). Qwen3-VL native dynamic resolution; the packaged processor owns resize/normalise — the card explicitly declines to publish a fixed raw-image contract. |
| **Ego history** | translation `(x,y,z)` + **9-D rotation (3×3)**, multi-timestep, with timestamps |
| **Text** | free-form string (instructions, questions). Navigation instructions e.g. `"Turn right in 30m"` |
| **Token layout** | `token_layout: camera_ts` — ordered by camera then timestep; `include_camera_ids: true`, so **camera identity is an explicit token**, not implicit in position |
| **Sensors NOT used** | ⛔ **no lidar, no radar.** Training sensors are RGB cameras + IMU + GPS. This is a **camera-only** model. |

### Outputs

| | |
|---|---|
| **Trajectory** | **64 waypoints, 0.1 → 6.4 s at 0.1 s**, ego-frame XYZ **+ 3×3 rotation per waypoint** (full SE(3), not just position) |
| **Text** | Chain-of-Causation reasoning traces · meta-actions (longitudinal / lateral / lane) · VQA answers · **2D grounding coordinates** · structured auto-labelling JSON (`critical_components_analysis`, `ego_vehicle_motion_analysis`, `trajectory_analysis`, `chain_of_causation`) |

### The reasoning concept — Chain of Causation

From arXiv 2511.00088 (the R1 predecessor, and the mechanism Alpamayo 2 inherits): **CoC** is a
dataset and output format of *decision-grounded, causally linked* reasoning traces aligned to driving
behaviour, built by **hybrid auto-labelling + human-in-the-loop**. Alpamayo 2 Super was trained on
**≈3.7 million CoC traces**.

The released inference path is: **backbone generates CoC text → action expert samples the
trajectory conditioned on it.** Reasoning is not decorative narration produced after the fact; it is
upstream of the plan.

R1's measured contribution of this design (**PUBLISHED**, on the *predecessor*, not on Super):
**+12 %** planning accuracy on challenging cases vs a trajectory-only baseline; **−35 %** close-encounter
rate closed-loop; RL post-training improved reasoning quality **+45 %** and reasoning–action
consistency **+37 %**; scaling 0.5 B → 7 B improved monotonically; on-vehicle **99 ms** latency.

---

## 2. Training data

| | |
|---|---|
| **Video** | **≈115,000 hours** of multi-camera driving |
| **Images** | **> 1 billion** |
| **Text** | < 1 billion tokens |
| **CoC traces** | **≈3,700,000** |
| **Named datasets** | `nvidia/PhysicalAI-Autonomous-Vehicles` and `…-NuRec` |
| **Sensors** | RGB cameras, IMU, GPS |
| **Labelling** | hybrid automated + manual |

⚠️ **The named public datasets cannot be the bulk of it.** PhysicalAI-AV is on the order of
thousands of 20-second clips — single-digit hours to low hundreds. 115,000 hours is **three to four
orders of magnitude larger**, so the overwhelming majority is **undisclosed internal NVIDIA fleet
data**. The public datasets are the *reproducible slice*, not the training set. Any "open model"
reading should carry that: **the weights are open, the data is not.**

## 3. Published performance

| benchmark | result |
|---|---|
| **LingoQA** (Lingo-Judge) | **79.2** |
| **AlpaSim closed-loop**, 910 NuRec scenarios | **1.50 ± 0.13** |
| **Open-loop**, 1,434 challenging PhysicalAI-AV samples | **minADE₆ @ 6.4 s = 0.911 m** |

⚠️ Read `minADE₆` precisely: **minimum over 6 sampled trajectories** against ground truth — a
best-of-6 oracle-selection metric, not a single-shot error. It is *not* comparable to a
single-trajectory ADE without saying so. This is the same distinction as our own
`oracle_ade` vs shipped ADE, where the gap is **65 %** of our error.

## 4. Compute and deployment envelope

| | |
|---|---|
| **Tested hardware** | 1 × **H100 80 GB HBM3**. "Other GPU architectures have not yet been validated." |
| **Measured peak** | **72,115 MiB** device memory (69.3 GiB peak PyTorch allocation) — 7 cameras, 4 frames, batch 1, 1 trajectory sample, BF16, SDPA, CFG off, 10 diffusion steps |
| **Two-GPU nav-CFG demo** | ≈67 GiB on the VLM GPU **and** ≈71 GiB on the expert GPU |
| **Stack** | PyTorch ≥ 2.8, Transformers ≥ 4.57.1, DeepSpeed ≥ 0.17.4, **Python 3.12**, CUDA 12.x, `flash-attn` built from source, `uv` |

⛔ **This does not fit our hardware.** Both TanitAD pods are **A40 46,068 MiB**. Peak demand is
**72 GB**; weights alone are 71.65 GB on disk (≈68 GB in BF16 resident). Running it here requires
**quantisation** (see §7) or an H100.

---

## 5. Strengths

1. **Dynamically-feasible-by-construction output.** The unicycle accel/curvature action space is the
   single most transferable idea in the release.
2. **Reasoning is upstream of action, and it is measured.** R1 quantified the contribution
   (+12 % / −35 % close encounters) rather than asserting interpretability.
3. **Genuinely permissive licence.** OpenMDW-1.1 on weights + Apache-2.0 on code, **commercial
   redistribution allowed**. This is unusually open for a frontier driving model.
4. **Multi-task from one model** — trajectory, VQA, grounding, meta-actions, auto-labelling. The
   auto-labelling head is a *label factory*, which is worth more to us than the planner (§8).
5. **Cheap sampling.** 10 Euler steps of a 2.3 B expert; the 32 B backbone runs once.
6. **Surround context.** 6 cameras including rear — it can see cut-ins from behind, which a
   front-only model structurally cannot.
7. **Honest engineering disclosure.** Exact memory profile, exact camera IDs per task, an explicit
   statement that the auto-labeller does *not* receive post-`t0` frames even though its offline
   teacher did. That last one is a leak guard stated in public — good practice.

## 6. Weaknesses and open questions

1. **⛔ No paper for Alpamayo 2 Super.** arXiv 2511.00088 covers **R1**. The 34 B model's training
   recipe, data mix, ablations and RL details are **unpublished**. Every architectural inference in
   §1 is mine, read off `config.json`.
2. **⛔ The training data is not open.** ~115,000 h of undisclosed internal fleet data. Not
   reproducible; possibly not auditable for the licence's purposes.
3. **Camera-only.** No lidar/radar — no direct range. Fine as a research position, a constraint in
   deployment.
4. **Enormous inference cost.** 72 GB and an H100 for **one sample, batch 1**. There is no published
   latency for Super (R1's 99 ms was a *much* smaller model on-vehicle). A 34 B model generating CoC
   text *before* planning is very unlikely to close a 10 Hz control loop; expect this to be a cloud /
   offline / teacher model, not an on-vehicle planner. The release language ("integrated into
   autonomous driving software **in the cloud**") supports that reading.
5. **`minADE₆` is best-of-6.** Headline open-loop number is oracle-selected.
6. **Closed-loop score is in-house.** AlpaSim + NuRec are both NVIDIA artifacts; 1.50 ± 0.13 has no
   external calibration. (We have measured the general hazard: on NuRec reconstructions REF-C's
   open-loop ADE is **3.21× worse** than on real footage — sim numbers are within-sim relative.)
7. **`cotrain_expert_vlm: false`** — the released config trains the expert against a frozen backbone,
   so reasoning→action coupling is one-directional. Whether joint training was tried and dropped is
   unpublished.
8. **Unvalidated outside H100.** Explicitly stated.

---

## 7. Can we run it? — the honest answer

**Not as shipped.** 72 GB peak vs 46 GB available. Three routes:

| route | verdict |
|---|---|
| **4-bit NF4 on the 32 B backbone, expert + vision in BF16** | **the only one that fits.** ≈16 GB (backbone) + ≈4.6 GB (expert) + vision + KV over ~24 K vision tokens ≈ **28–34 GB**. Attempted in §9. ⚠️ Quantisation is **not** validated by NVIDIA and **changes the model** — any number carries that label. |
| CPU offload (accelerate/DeepSpeed) | works, minutes per sample, fine for a handful of qualitative samples |
| Rent an H100 80 GB | the clean answer. **PI decision (spend).** |

---

## 8. How TanitAD should leverage it — ranked by value

### ⭐ Tier 1 — do these

**1. Adopt the unicycle accel/curvature action space. (Highest value, lowest risk, no Alpamayo dependency.)**
Our arm emits free XY waypoints and has a **measured systematic over-speed**: `speed_bias +0.484 m/s`,
prediction ahead of the human at 2 s on **71.95 %** of windows, ahead on speed on **75.51 %**.
Re-parameterising the output as `(a, κ)` integrated through a unicycle makes speed a **state you
integrate**, not a coordinate you regress — the error becomes an *acceleration* error, which is
bounded, physically interpretable, and directly penalisable. This is a contained change to the
trajectory head that plausibly attacks our single largest measured defect. **Pre-register it as a
one-lever arm.**

**2. Use Alpamayo 2 as an AUTO-LABELLER, not a competitor.**
Its auto-labelling head emits exactly the four things our corpus lacks:
`critical_components_analysis`, `ego_vehicle_motion_analysis`, `trajectory_analysis`,
`chain_of_causation`. Our strategic brain has **no map, no lane graph, no junction annotation** —
settled at five probes — which is precisely why our route head degenerated to a constant predictor
(`{left 0, straight 1737, right 0}`). **Alpamayo can manufacture the strategic supervision
PhysicalAI-AV does not ship.** OpenMDW-1.1 permits derivative use. This is the single biggest
unlock available to us.

**3. Distil the meta-action head into our 5-way tactical softmax.**
Alpamayo emits **separate longitudinal, lateral and lane actions**. Our known-largest architectural
defect is a **5-way softmax that MIXES lateral and longitudinal classes**. Alpamayo's factorisation
is independent evidence that the axes should be separate — and it can label our corpus in that
factored form, giving us the targets for free.

**4. Reference system on the FOUR FAMILIES, not on ADE.**
Score Alpamayo on our binding block. The question worth money is **not** "who has lower ADE" — that
is settled a priori by 115× parameters and 6× cameras. It is: **does a 34 B surround model trained on
115,000 h also drive systematically too fast?** If yes, our over-speed is a **property of the data or
the label convention**, not our defect, and we would be optimising against a bias in the ground
truth. If no, it is ours to fix. **That single experiment redirects our longitudinal effort.**

### Tier 2 — strong

**5. Adopt flow matching + Euler-10 in REF-C.** Our anchored-diffusion decoder's measured selection
gap is **65 % of its error** (0.4714 vs oracle 0.1640), and reachability is refuted as a fix (a
compute lever, not a selection lever). Flow matching with a small expert is a different, cheaper
sampler worth an arm.

**6. Steal the trajectory tokenizer.** 3000-bin future / 1000-bin history vocab lets a language model
*reason about* trajectories. It is the missing bridge if we ever want a text-reasoning layer above
the hierarchy.

**7. LingoQA as an external, non-NVIDIA benchmark.** Our programme has **no external calibration at
all**. LingoQA is Wayve's. Even a poor score is a fixed point.

**8. Camera-count ablation as a cheap capability bound.** Alpamayo at 6 cameras vs our 1: run
Alpamayo with front-only input. The delta is a **measured price of our sensor choice**, which we
currently do not know.

### Tier 3 — opportunistic

**9. CoC traces as a training signal** for a future reasoning layer (3.7 M exist; the format is public).
**10. `PerWaypointActionInProjV2` Fourier features** — cheap positional encoding for waypoint indices.
**11. Their `AlpaSim` closed-loop harness** — we already use AlpaSim; their 910-scenario protocol is a
ready-made external comparison.
**12. Non-causal joint denoising over the whole horizon** instead of autoregressive rollout — directly
attacks compounding error.

### ⛔ What NOT to do

- **Do not reposition TanitAD as "beating Alpamayo".** 34 B vs 0.3 B, 6 cameras vs 1, 115,000 h vs
  ~9 h. Our thesis is a **sub-300 M hierarchical** model; the interesting axis is
  **capability per parameter and per camera**, and that framing must be explicit in every comparison.
- **Do not fine-tune Alpamayo as our deliverable.** It would abandon the programme's thesis and we
  cannot even hold it in memory.
- **Do not quote its numbers as a target on our corpus** until the contamination question in §9 is
  settled.

---

## 9. The comparison — design, and the confound that decides its meaning

⛔ **FIVE ways the comparison is not like-for-like**, all of which must travel with any number:

| axis | Alpamayo 2 Super | TanitAD flagship-v1arch |
|---|---|---|
| parameters | 34.3 B | < 0.3 B (**≈115×**) |
| cameras | 6 (surround, incl. rear + tele) | **1** (front, 256×256) |
| horizon | 64 wp / **6.4 s** | 20 wp / **2 s** |
| training video | ≈115,000 h | 9,000 clips (≈50 h) |
| output metric | **minADE₆** (best of 6) | single-trajectory ADE |

⛔ **AND THE ONE THAT MATTERS MOST — TRAINING CONTAMINATION.** Alpamayo 2 Super lists
`nvidia/PhysicalAI-Autonomous-Vehicles` as a **training** dataset. **Our clean OOD-val corpus is
PhysicalAI-AV's own official val split.** NVIDIA does not state whether they excluded that split.
If they did not, our corpus is **inside Alpamayo's training set** and any comparison there is
contaminated **in Alpamayo's favour**.

⇒ **This must be resolved or declared before a single number is published.** It is the same class as
our own v1arch canonical-val leak (21 of 40 val episodes inside its training pool), and we retracted
that rather than quote it. Options: ask NVIDIA on the developer forum; or compare only on a corpus
demonstrably outside both training sets.

**Given all of the above, the admissible experiment is not a leaderboard.** It is:

> **Does Alpamayo 2 Super, on our exact windows, exhibit the same longitudinal over-speed bias
> (+0.484 m/s, 71.95 % of windows ahead at 2 s) that our flagship does?**

A shared bias points at the data or the label convention. A divergent one points at our model. Either
answer redirects the programme's largest open work item — and, unlike an ADE ranking, **neither
answer is decided in advance by the parameter count.**

---

## Sources (all retrieved 2026-08-05)

- HF model card + API — `https://huggingface.co/nvidia/Alpamayo2-Super`
- `config.json` (machine-readable architecture) — same repo, `raw/main/config.json`
- `https://github.com/NVlabs/alpamayo2` — README, usage, memory profile
- arXiv **2511.00088**, *Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable
  Autonomous Driving in the Long Tail* — the predecessor's mechanism and ablations
- `https://huggingface.co/nvidia/Alpamayo-R1-10B` — predecessor metadata

---

## 10. ⭐ RUNNING — Alpamayo 2 Super executes on a 46 GB A40 at **25.84 GiB peak**

**MEASURED 2026-08-05, pod4 (A40 46,068 MiB).** NVIDIA's profile is **72,115 MiB on an H100 80 GB**
and they state non-H100 architectures are unvalidated. 4-bit NF4 on the backbone alone brings that
to **25.84 GiB — a 2.79× reduction — and the model produces correct, semantically grounded output.**

```
[quant] NF4 backbone · BF16 skip-list ['visual','lm_head','expert','action_in_proj','action_out_proj'] · attn=sdpa
[quant] Linear4bit modules: 448
[quant] weights resident: 23.58 GiB
Chain-of-Causation: "Nudge left to avoid the cones on the right side."
minADE: 1.4222202 meters
[quant] PEAK device memory: 25.84 GiB (A40 capacity 45.0 GiB)
A2_RUN_RC=0
```

**The output is right, not merely well-formed.** The clip
(`030c760c-ae38-49aa-9ad8-f5650a545d26`, `t0_us 5,100,000`) is a roadworks scene; cones are visible
in the cross-right, front-wide and rear-right views, and the predicted path nudges left. The model
reasoned about the correct object and acted on it.

### What was quantised, and what was deliberately not

The checkpoint splits cleanly — `vlm.*` (1,058 tensors, the 32 B Qwen3-VL) and `expert.*`
(717 tensors, the 2.3 B diffusion head), read from `model.safetensors.index.json`:

| | |
|---|---|
| **NF4 (double-quant, bf16 compute)** | `vlm.model.language_model.*` — 448 `Linear4bit` modules |
| **kept BF16** | `vlm.model.visual.*`, `vlm.lm_head`, **all of `expert.*`** incl. `action_in_proj` / `action_out_proj` |

⛔ **Quantising the action expert would have been the wrong saving.** It is the module that emits the
trajectory; 4-biting 2.3 B to save ~3.5 GB, when the 31 B backbone is where the memory actually sits,
corrupts the measured output to buy almost nothing. The vision tower is likewise small and is the
only thing that sees.

Exactly one line of NVIDIA's code is patched — `inference_smoke.py:133` — so data loading, prompt
construction, CoC generation, expert sampling, minADE and visualisation are all theirs.

### ⛔ How this number may and may not be used

- **`QUANTISED-4BIT-UNVALIDATED`.** NF4 is lossy and is not an NVIDIA-validated configuration.
- **The 1.4222 m minADE here may NOT be compared to their published 0.911 m.** Theirs is
  **minADE₆** (best of 6 samples) over **1,434 curated challenging samples**; this is **one sample**
  of **one** trajectory (`num_trajectory_samples: 1`) on **one** clip, 4-bit. Different estimator,
  different denominator, different precision. Three reasons, any one of which is disqualifying.
- What it **is** good for: a **self-consistent** comparison against our own arm on our own windows,
  where both sides are measured by us under one protocol.

### Practical notes for anyone reproducing

- The env is Python **3.12** via `uv` (the pod ships 3.11); torch 2.8.0+cu128, transformers 4.57.1,
  flash-attn 2.8.3 built from source, bitsandbytes 0.50.0.
- ⚠️ **Do not override `HF_HOME`.** The first attempt did, which pointed token lookup at an empty
  cache and produced a `GatedRepoError 401` on the PhysicalAI-AV dataset — a failure that looks like
  a permissions problem and is actually a path problem.
- ⚠️ The venv on MooseFS costs **~6 minutes of import time per run**. Put it on local disk
  (pod4 has a 500 GB overlay at `/`, 771 MB used) before iterating.
- Load is ~18 s/shard × 15 shards ≈ 4.5 min from MooseFS.

---

## 11. ⭐ THE RESULT — Alpamayo is UNBIASED where our arm is not

**MEASURED 2026-08-06, 39 paired clips**, both truncated to a **2.0 s / 20-waypoint**
horizon on the same PhysicalAI OOD-val clips at the same nominal `t0 = 5.1 s`.

| | Alpamayo 2 Super | TanitAD flagship-v1arch |
|---|---|---|
| ADE @2 s | **0.2703 m** | 0.3303 m |
| **speed bias** | **+0.0569 m/s** | **+0.4245 m/s** |
| **frac faster than human** | **59.0%** | **84.6%** |
| along bias @2 s | +0.1132 m | +0.8176 m |
| frac ahead at 2 s | 59.0% | 79.5% |
| (Alpamayo native 6.4 s ADE) | 2.4026 m | — |

### ⇒ The over-speed is OURS, not the data's

Alpamayo sits at **+0.057 m/s** and **59.0 %** faster-than-human — a coin flip, i.e. essentially
**unbiased**. Our arm is **+0.42 m/s** and faster on **84.6 %** of these clips, consistent with the
corpus-wide **+0.484 m/s / 75.5 %** measured independently over 6,382 windows.

**A 34 B model trained on ~115,000 h of the same-family data does NOT run ahead of the human.**
So the bias does not live in the ground truth or the label convention — it is ours, and it is
ours to fix. That converts leverage idea #1 (the unicycle accel/curvature action space) from a
plausible borrowing into a **motivated** one.

### ⭐ And the second reading, which favours us

The ADE gap is **0.2703 vs 0.3303 — a factor of 1.22** — against **~115× the parameters**
and **~190× the input pixels**. On the axis this programme actually claims — capability per
parameter, per camera — that is a strong showing, and it is the only axis on which the outcome
was not decided in advance.

### How the ground truth was obtained, and why it is trustworthy

The trajectory capture returned **empty GT**: `sample_trajectories_from_data(data=model_inputs)`
receives the model's *inputs*, which carry ego history but no future. Rather than re-run 40
samples, the GT was reconstructed from the staged `egomotion` parquet — and then **proven**:
recomputing ADE between Alpamayo's own captured prediction and the reconstructed GT reproduces
the `min_ade_m` NVIDIA's code printed, to within **0.0220 m (mean 0.0047 m)** over 39 samples.
The check *refuses to write* a GT that does not reproduce their metric — a flipped sign would
have been small, plausible and wrong.

### ⛔ Caveats, carried in the JSON and burned into the video banner

* **n = 39**, unweighted mean over clips — not an episode-cluster bootstrap
* Alpamayo is **NF4-quantised**, which is **not** an NVIDIA-validated configuration
* ⛔ **CONTAMINATION UNRESOLVED** — these clips are PhysicalAI-AV, which Alpamayo lists as a
  **TRAINING** dataset. Any advantage may be contamination rather than capability.
* **not like-for-like**: 34.3 B vs < 0.3 B · 6 cameras @1920×1080 vs ONE 256×256 crop ·
  Alpamayo truncated 6.4 s → 2.0 s · its published headline is minADE₆ (best-of-6), this is 1 sample
* **alignment**: our window origins sit on an 0.8 s stride, so |dt| ≤ 0.4 s. Each model is scored
  against **its own** GT at **its own** t0. The over-speed *rate* is robust to that; a per-clip ADE
  difference is not.

**Artifacts:** `comparison/comparison.json` (all 39 rows + per-clip CoC) ·
`comparison/alpamayo_oodval.jsonl` (40 runs, all CoC text) · `comparison/alpamayo_gt.json`
(reconstructed GT + validation) · video at
`TanitAD Research Hub/Evaluation/Videos/alpamayo2-vs-flagship-2026-08-06/`.

### What Alpamayo was actually fed — its FULL validated trajectory profile

Recorded per sample, not asserted afterwards. Every one of the 40 runs carries
`image_frames_shape: [6, 4, 3, 1080, 1920]` — **6 cameras × 4 frames at native 1920×1080** —
and the sidecar records `camera_indices: [0, 1, 2, 3, 5, 6]`.

That is NVIDIA's own `DRIVING_SIX_CAMERA_FOUR_FRAME` profile, defined at
`src/alpamayo2_super/input_profiles.py:40-41` as `camera_ids=(0, 1, 2, 3, 5, 6)` and asserted by
`assert_task_input(...)` — the batch called `inference_smoke.run_smoke(...)` with **no camera
override**, so the model ran on its default, validated trajectory input.

| | |
|---|---|
| rig cameras present in the source clip | **7** — `[0..6]`, from `camera_cross_left_120fov` to `camera_front_tele_30fov` |
| given to Alpamayo | **6** — `[0, 1, 2, 3, 5, 6]` = cross-left, front-wide, cross-right, rear-left, rear-right, front-tele |
| withheld | **1** — id `4`, `camera_rear_tele_30fov` |

⚠️ **The withheld camera is not a handicap we imposed.** NVIDIA's card splits the profiles by
task: **trajectory / meta-action / auto-labeling / grounding use `[0,1,2,3,5,6]`**, while **VQA uses
`[0,1,2,3,4,5]`**. Rear-tele is a VQA camera; excluding it for trajectory *is* the validated
configuration. Giving it would have been the deviation.

⇒ **The ~190× pixel asymmetry stated in the banner and in `comparison.json` is real and is
Alpamayo's full intended input** — 6 × 4 × 1920 × 1080 against our arm's single 256×256 front
crop. Alpamayo was not run degraded; our arm was not run flattered.

⚠️ The one configuration deviation remains **NF4 quantisation of the backbone**, which is ours and
is not NVIDIA-validated. It is stated with every number and is a separate work item
(bf16 re-run on a 80 GB card) — not a camera question.

## 12. ⛔ THE FOUR BINDING FAMILIES — and only one of them favours the 34 B model

The §11 table is **ADE plus one longitudinal scalar**, which under the binding rule of 2026-08-02
is *one row of four*. Scored properly, per family, never pooled — same 39 clips, same 2.0 s /
20-waypoint dense 0.1 s grid, each arm against **its own** GT at **its own** t0, computed by
`taniteval.four_families` rather than re-implemented:

### LONGITUDINAL — Alpamayo wins decisively, and the margin is bigger than ADE suggested

| | Alpamayo 2 Super | TanitAD flagship-v1arch | ratio |
|---|---|---|---|
| speed MAE (m/s) | **0.3833** | 0.7050 | 1.84× |
| speed **bias** (m/s, + = too fast) | **+0.0569** | +0.4245 | 7.5× |
| **accel MAE (m/s²)** | **0.5077** | **1.7644** | **3.48×** |
| along bias (m) | +0.0315 | +0.1543 | 4.9× |
| along final bias @2 s (m) | +0.1132 | +0.8176 | 7.2× |
| ego-progress ratio (GT = 1.0) | 0.9836 | 1.0566 | — |
| under-progress rate | 0.3714 | 0.1714 | — |
| distance keeping (headway / time-gap / TTC) | **UNAVAILABLE** | **UNAVAILABLE** | — |

⭐ **The new finding is `accel MAE`, not the speed bias.** Our arm's **acceleration profile is
3.48× worse** than Alpamayo's — a metric ADE cannot see at all, and one that no previous report
carried. Combined with the +0.42 m/s bias and the 1.057 progress ratio, the picture is specific:
**our arm accelerates too hard, too often, and ends up ahead** — it is not merely mis-set on a
constant speed offset.

⇒ This sharpens leverage idea #1 from "borrow the unicycle action space" to a **prediction**: a
model that emits **acceleration** and integrates it, rather than emitting free waypoints, is
directly regularised on precisely the quantity we are 3.5× worse on. That is now a
**pre-registerable** experiment with a named metric, not an architectural preference.

⛔ **`distance_keeping` is UNAVAILABLE for both arms and that is a WORK ITEM, not a pass.** The
banked OOD-val lead block is keyed on our 0.8 s rollout window grid; Alpamayo ran at clip
`t0 = 5.1 s`, so the two do not join. Building it needs a `build_lead_block.py` pass over
`obstacle.offline` at these 39 origins. Half of the binding LONGITUDINAL family is therefore
missing from this comparison, and it is stated rather than dropped.

### LATERAL — our arm is COMPETITIVE, and better on curvature

| | Alpamayo 2 Super | TanitAD flagship-v1arch |
|---|---|---|
| heading MAE (deg) | **0.6794** | 0.7800 |
| yaw-rate MAE (deg/s) | 1.8285 | **1.7791** |
| **curvature MAE (1/m)** | 0.009162 | **0.007551** |
| curvature bias (1/m) | +0.001021 | −0.001996 |
| cross-track MAE (m) | **0.0398** | 0.0493 |
| cross-track bias (m, + = left) | +0.0036 | −0.0241 |

⭐ **This is the most important row in the whole comparison and ADE had hidden it.** A sub-0.3 B
front-crop-only model matches a 34.3 B six-camera model on heading and yaw-rate, and **beats it on
curvature MAE (0.00755 vs 0.00916)**. The deficit that produced the ADE gap is **not lateral**.

⇒ Independent confirmation, from an entirely different direction, of the programme's own
"88.7 % of the oracle gap is longitudinal" finding. Two unrelated instruments now agree on where
the defect lives — which is the strongest form of evidence available here.

### TACTICAL — split verdict, and our declared head is only WEAKLY coupled to what we drive

Both arms scored on the **same instrument** — the manoeuvre actually *executed*, derived from net
yaw over the 2 s horizon at the 0.15 rad gate, so the comparison does not depend on either arm's
declared head.

| | Alpamayo 2 Super | TanitAD flagship-v1arch |
|---|---|---|
| executed-manoeuvre accuracy | 0.7949 | **0.8462** |
| executed-manoeuvre **κ** | 0.3333 | **0.4968** |
| left-turn recall (n = 2) | 0.5 | **0.0** |
| declared manoeuvre head | **UNAVAILABLE** | present |
| declared vs **driven** κ | — | **0.3432 — WEAK** |
| declared vs GT direction | — | acc 0.8718 · κ 0.5886 |

⚠️ **Our arm's declared manoeuvre is only WEAKLY coupled to the path it actually drives**
(κ = 0.3432). Not decorative — the earlier "0 of 881 accelerate" pathology is **absent** here, all
five classes are emitted (`lane_keep` 10, `turn_left` 4, `turn_right` 3, `accelerate` 8,
`brake_stop` 14) — but a κ of 0.34 means the decision layer and the trajectory layer are drifting
apart. This is exactly the class of defect a scalar ADE cannot see, which is why the family is
binding.

⚠️ **Our arm drove 0 of 2 left turns** (Alpamayo drove 1 of 2). n = 2, so this is a flag for a
larger turn-subset panel, not a finding.

⛔ Alpamayo's **declared** manoeuvre is UNAVAILABLE because this pass ran the **trajectory** task
only; its meta-action head exists (`text_tasks.py`) and was not invoked. Its Chain-of-Causation
text was captured for all 39 clips, but **free text is not a scored decision** and is not counted
here. WORK ITEM: re-run the same 39 clips under the meta-action task.

### STRATEGIC — UNAVAILABLE for BOTH arms, and *that* is the finding

Neither arm can be scored, for the **same** reason: PhysicalAI-AV ships **no map, lane graph,
junction annotation or route signal** (five independent probes; the card says verbatim *"we do not
include open maps data"*). The only route label derivable is from the ego's own future path, which
cannot separate *"took the left branch"* from *"drifted left on a curving road"* — and scoring
against it is what produced flagship v1's `route_head_eq_logged = 1.0000`, an **echo of its own
nav input read as skill**. Republishing that number would be worse than reporting nothing.

⇒ The admissible instrument is `taniteval.strategic_optionset` over **map-derived option sets**,
which needs AlpaSim or an external corpus. **The programme's central thesis — that the hierarchy
works — remains unmeasured at its top level, for us and for a 34 B reference system alike.**

### The estimator, stated because a number without one is inadmissible

Unweighted mean over 39 paired clips, **one window per clip**. ⛔ That is **not** the decision-grade
estimator: with one window per episode the episode-cluster bootstrap degenerates to the i.i.d.
case, so **no CI is quoted here rather than a wrong one**. A decision-grade read needs many windows
per episode — a further work item, and cheap now that the harness exists.

**Artifacts:** `comparison/a2_four_families.json` (all four families, both arms, with every
UNAVAILABLE reason and n) · `comparison/alpamayo_traj_2s.json.xz` (Alpamayo's 2 s waypoints) ·
`tools/a2_four_families.py` (the scorer).

## 13. ⭐ LEVERAGE IDEA #1 IMPLEMENTED — and the recovered controls give a sharper number than any ADE

The `accel MAE` gap in §12 said *what* is wrong. Recovering the controls each arm's path
**implies** says *how much*, in the units a control head would actually emit.

`stack/tanitad/models/kinematic.py` gains `rollout_unicycle` (the Alpamayo
`(accel, curvature)` action space), `unicycle_controls_from_path` (its inverse) and
`entry_speed_mismatch`. 12 tests, all green.

⚠️ **We already had half of this and it was dead code.** `rollout_bicycle` — a differentiable
kinematic-bicycle integrator with a Kamm-circle penalty — has been in the repo since H14 Track 1,
is exported from `models/__init__`, is NaN-tested, and is imported by **no model and no trainer**
(verified by grep over `stack/`). The flagship emits free waypoints. Wiring a control head is
therefore a **new capability**, not a switch.

### MEASURED — what each arm's path implies it commanded (39 clips, 2 s, dt = 0.1 s)

| | Alpamayo 2 Super | TanitAD flagship | human (GT) |
|---|---|---|---|
| implied accel RMS (m/s²) | 1.027 | **4.1656** | **0.8048** |
| **× the human's accel magnitude** | **1.27×** | **5.18×** | 1.00× |
| implied accel MAE vs human (m/s²) | 0.5090 | 1.7350 | — |
| implied accel **bias** (m/s²) | +0.1076 | **+0.7160** | — |
| implied curvature RMS (1/m) | 0.030829 | 0.032496 | 0.020914–0.024488 |
| implied curvature MAE (1/m) | 0.008497 | **0.008275** | — |
| entry transient MAE (m/s²) | 1.4039 | 1.5367 | **0.4249 (floor)** |

⭐ **Our arm commands 5.18× the acceleration magnitude a human does.** Alpamayo commands 1.27×.
That is the single most specific statement the programme has about the longitudinal defect, and
**no ADE at any horizon can express it** — a path can match position while thrashing the throttle.

⭐ **And the launch is NOT where the two arms differ.** Both sit at ~1.4–1.5 m/s² of entry
transient against the instrument's own floor of **0.4249** (what the *human's* path scores against
the ego's recorded `v0` under the same chord reading). The arms are indistinguishable there. The
defect is in the **sustained acceleration profile**, which is exactly what an integrated
`(accel, curvature)` head constrains and a free-waypoint head does not.

⚠️ **On curvature the two arms are the same** (MAE 0.00828 vs 0.00850) and **both over-command**
relative to the human (0.032/0.031 vs 0.021/0.024 RMS). Consistent with §12's LATERAL read: the
lateral channel is not our problem.

### ⛔ Two traps this work caught, both of which would have shipped a number

1. **An off-by-one in the inverse map.** `rollout_unicycle` advances position on the speed at the
   *start* of each step, so step `k`'s displacement reveals the speed *before* `accel[k]` — the
   inverse is shifted by one. The naive version drifted **1.2233 m** over 2 s and returned every
   control one step late. Caught only because the round-trip test integrates → recovers →
   re-integrates and compares.
2. **`tanh` is not a safe saturating squash.** MEASURED: `1 - tanh(51)**2` is **exactly 0.0** in
   float32, so a control far outside its limit has an *underflowed* gradient — the same silent
   dead-head a hard `clamp` produces, moved out to where nobody tests. Replaced with softsign
   `x / (1 + |x|/limit)`, whose 1/x² decay leaves ~3.7e-4 at the same overshoot.
3. **Curvature at a standstill is undetermined, not large.** `yaw_rate = v·κ`, so at `v ≈ 0` every
   κ gives the same zero heading change. Ungated, the implied-curvature MAE came back as
   **1.6 × 10⁶** and **7.6 × 10³** 1/m — meaningless numbers on their way into this table. Gated at
   `MIN_DS_MPS = 0.5`, the same constant `four_families` uses.

### The pre-registration this makes possible

**Both outcomes committed in advance.** An arm whose operative head emits `(accel, curvature)` and
integrates through `rollout_unicycle`, trained otherwise identically to `flagship-v1arch-v2bal-30k`:

* **PASS** — implied accel RMS drops toward the human's 0.80 (target: ≤ 2.0×, i.e. ≤ 1.6 m/s²) **and**
  `speed_bias_mps` falls below +0.15 m/s, **without** LATERAL regressing (curvature MAE ≤ 0.0090).
* **FAIL** — accel RMS stays above 3× the human's, **or** lateral regresses. Then the action space
  is not the lever and the defect is in the loss or the corpus, not the parameterisation.

⚠️ The comparison must be run at **matched steps** and read through `taniteval.four_families`, not
through ADE — the whole point is that ADE could not see this.

**Artifacts:** `comparison/implied_controls.json` · `stack/tanitad/models/kinematic.py` ·
`stack/tests/test_unicycle_action_space.py`.

## 14. ⛔ CORRECTION to §12's TACTICAL block — two defects, both ours, and the conclusion flips

Running Alpamayo's **meta-action** task (39/39, 1561 s on the A40) closed the
declared-TACTICAL work item — and in checking its result against an obvious confound,
found that **§12's TACTICAL numbers were artifacts of the instrument, not readings of the
models.** Corrected here; the LONGITUDINAL, LATERAL and STRATEGIC blocks are unaffected.

### Defect 1 — net yaw was summed over steps where the ego was not moving

`yaw` has no meaning at `v ≈ 0`: the path tangent flips freely. One stopped window
contributed a net yaw of **π**. Steps below `MIN_DS_MPS = 0.5` are now excluded, exactly
as `four_families._seq_geometry` does. **This alone moved Alpamayo's executed-manoeuvre
κ from 0.3333 to 0.4882.** Third appearance of the same trap in this work — after
`df` on pod disk and curvature-at-standstill: *a quantity that is undefined in a regime,
aggregated over that regime, read as a measurement.*

### Defect 2 — the 0.15 rad direction gate is mis-scaled for 2 s windows

Alpamayo's own Chain-of-Causation says things like *"Nudge left to pass the parked SUV"*.
A **nudge** is nowhere near a 0.15 rad net-heading turn, so a low κ might be a threshold
mismatch rather than incoherence. It was.

**MEASURED, on the human's own paths:** median |net yaw| over 2 s is **0.023 rad**, p90
**0.185**, and only **17.9 %** of windows exceed the 0.15 gate. `hierarchy.DIR_YAW_RAD`
is ~6.5× the typical turn, so nearly every window is "straight" by construction.

| gate (rad) | Alpamayo declared κ | flagship declared κ | Alpamayo executed κ | flagship executed κ |
|---|---|---|---|---|
| **0.15** (as published in §12) | 0.1961 | **0.4402** | 0.4882 | **0.6176** |
| 0.10 | 0.3004 | 0.3743 | **0.7292** | **0.7263** |
| 0.06 | 0.2639 | 0.2835 | 0.7222 | 0.7132 |
| 0.04 | **0.4059** | 0.2390 | 0.6277 | 0.6848 |
| 0.03 | **0.4553** | 0.1986 | 0.6926 | 0.5752 |
| 0.01 | **0.4660** | 0.1159 | 0.8077 | 0.3953 |

### ⇒ What actually changes

1. ⛔ **RETRACTED: "our executed-manoeuvre κ 0.4968 beats Alpamayo's 0.3333".** At the
   gate where the instrument is best matched to the data (0.10, near the human's p90 of
   0.185/2), the two arms are **indistinguishable — 0.7263 vs 0.7292**. The §12 ranking
   was a gate artifact compounded by the stopped-window contamination.
2. ⭐ **NEW, and it is the substantive finding: the two arms' declarations move in
   OPPOSITE directions as the gate tightens.** Alpamayo's rises 0.196 → 0.466; ours falls
   0.440 → 0.116. Its declaration carries **fine** lateral information — the nudges — that
   our gate discards; ours carries only **coarse** information, agreeing on big turns and
   saying nothing about small ones. That is exactly what a vocabulary with severity
   (`Steer Left` vs `Sharp Steer Left`) buys, and our 5-way softmax has no severity axis
   at all.
3. **Gate-free cross-check.** Discarding magnitude entirely and asking only whether the
   driven path leans the declared way: **0.7143 for both arms** — but over **n = 21**
   declared turns for Alpamayo against **n = 7** for ours. It declares 3× as many
   lateral actions, and is right about them just as often.
4. ⚠️ **"Our arm drove 0 of 2 left turns"** from §12 was computed under both defects and
   is withdrawn; the turn subset is gate-defined and n = 2 either way.

### ⭐ The architectural finding the meta-action run was launched for

Alpamayo declares a manoeuvre on **three independent axes**:

```
Longitudinal: Gentle Deceleration.
Lateral:      Steer Left.
Lane:         Lane Keep.
```

| axis | observed vocabulary (n = 39) |
|---|---|
| **Longitudinal** | Gentle Deceleration 15 · Maintain Speed 9 · Gentle Acceleration 9 · Stop 5 · Strong Acceleration 1 |
| **Lateral** | Go Straight 13 · Steer Right 13 · Steer Left 6 · Sharp Steer Left 2 · *(absent 5)* |
| **Lane** | Lane Keep 32 · Turn Left 2 · *(absent 5)* |

Our flagship declares **one 5-way softmax** over
`[lane_keep, turn_left, turn_right, accelerate, brake_stop]`, which **mixes** the lateral
and longitudinal decisions into a mutually-exclusive choice. *"Decelerating **and**
turning left"* is one label there and is **unrepresentable** here. CLAUDE.md names this
mixing as the programme's single largest known defect — and this is a working system that
does not have it, with the measured consequence in row 2 above.

⚠️ **`Stop` short-circuits the axes.** In all 5 rows where the longitudinal action is
`Stop`, generation ends before the Lateral and Lane axes are emitted. The axes are
therefore *not* fully independent in Alpamayo's own scheme — a factorisation we borrow
must decide deliberately whether to reproduce that. Recorded because the parser counts
unparsed rows rather than dropping them.

⚠️ **Sampled, not modal.** `generate_text` runs at temperature 0.6; one draw per clip,
seed 42. Stability across draws is **UNMEASURED** and is a work item — a κ computed on
single samples has a variance floor we have not quantified.

### ⛔ Blast radius beyond this document

`DIR_YAW_RAD = 0.15` is `taniteval/hierarchy.py:164` and feeds
`consistency.maneuver_vs_trajectory`, `commanded_route_vs_maneuver`,
`commanded_route_vs_trajectory` and every `*_turn_subset` in the hierarchy panel — i.e.
**every published manoeuvre-coherence κ in the programme**. Those were computed on the
same 2 s horizon over the same corpus. **They should be re-read at 0.10 and the
sensitivity published**, and any verdict that flipped between 0.15 and 0.10 was never
decision-grade. Logged in `RETRACTION_LOG.md`.

**Artifacts:** `comparison/alpamayo_meta_action.jsonl.xz` (all 39 raw generations) ·
`comparison/a2_meta_action_parsed.json` · `comparison/a2_gate_audit.json` ·
`tools/a2_meta_action.py`, `tools/a2_parse_meta_action.py`, `tools/a2_gate_audit.py`.
