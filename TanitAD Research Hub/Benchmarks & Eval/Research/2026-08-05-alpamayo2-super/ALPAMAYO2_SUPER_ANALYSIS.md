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
