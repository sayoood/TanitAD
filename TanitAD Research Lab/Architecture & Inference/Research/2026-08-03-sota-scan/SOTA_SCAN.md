# SOTA scan — what the last ~6 months publish that TanitAD should actually act on

**Date:** 2026-08-03 · **Agent:** `sota-scan` (PRIORITY 11) · **Hosts touched:** NONE (web + local repo
reads only; thor / tanitad-new / pod2 / pod4 untouched).
**Scope:** papers and releases dated **2026-01 → 2026-08** on: hierarchical & latent world models for
driving; closed-loop AV simulation + neural reconstruction; latent-action / inverse-dynamics from
action-free video; anchored & diffusion trajectory decoders and how they report LONGITUDINAL vs
LATERAL; Jetson-class edge deployment and quantisation that preserves *decision* quality.

**Evidence-class legend (CLAUDE.md operating standard):**
`PUBLISHED` (external, URL-cited) · `MEASURED` (ours, artifact path cited) ·
`INHERITED` (our doc/commit, **not** re-verified this run) · `ESTIMATED` · `HYPOTHESIS`.

⚠️ **Method note, and it cost the last research agent a near-miss:** every external claim below was
read at **HTML depth** (`arxiv.org/html/...`) or on the project's own repo page. Three PDF fetches in
this run returned only compressed streams / structural metadata and were **discarded, not
paraphrased** (IDOL, DriveAnchor and the OpenReview LAM critique all failed at PDF depth first). The
`2026-07-27-latent-action-models` note §8 records an automated PDF hop **inventing a verbatim
quotation**; I did not re-open that path.

---

## 0. The one-paragraph verdict

The field moved decisively toward **closed-loop, reconstruction-backed evaluation** and away from
open-loop displacement error — and it now has the **measurement** to justify it: L2/ADE has
**no significant correlation** with closed-loop driving score (ρ = −0.36, p = 0.43), while a
progress-and-safety composite does (ρ = 0.90, p = 0.002)
([2605.00066](https://arxiv.org/html/2605.00066v1)). That is external, independent corroboration of
Sayed's four-families rule and of our own open→closed gap. Simultaneously, **NVIDIA open-sourced both
halves of the sim stack we were treating as closed** — feed-forward 3DGS reconstruction
([Instant NuRec](https://github.com/NVIDIA/instant-nurec), Apache-2.0) and the generative
closed-loop renderer ([Cosmos-Dreams / OmniDreams](https://github.com/nv-tlabs/omni-dreams),
Apache-2.0, weights on HF) — both keyed to the **same PhysicalAI-AV NuRec corpus we already hold**.
The single highest-value change is therefore *not* an architecture change: it is **owning the
closed-loop measurement**, and the two cheapest items on this list (#2, #3) are 0-GPU and can start
today.

**Ranking principle:** value-to-programme × cheapness × *discriminating* power (would the result
change a decision?). Items 1–4 are the ones I would defend spending on. Items 5–8 are real but
smaller or more speculative. §10 is a **do-NOT-adopt** list, which is also a result.

---

## 1. ⭐ Own the reconstruction: `Instant NuRec` gives us a closed-loop renderer we control

**Source:** *Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation*,
Huang, Ren, Tyszkiewicz, … Gojcic, Fidler (NVIDIA), arXiv **2607.14203**, 2026-07-15 —
[abs](https://arxiv.org/abs/2607.14203) · [html](https://arxiv.org/html/2607.14203v1) ·
[code](https://github.com/NVIDIA/instant-nurec). **Class: PUBLISHED.**

**What it is.** A feed-forward model that turns a **10–20 s multi-camera driving log into a
simulatable 3DGS world in one forward pass**, ~**1.5 s** per scene, **no per-scene optimisation**.
Inputs: **1–5 calibrated cameras** (V ∈ {1,3,5}) with 6-DoF poses. LiDAR depth and semantic labels
are used **in training only** — at inference depth is predicted, so **no LiDAR is required**.
Outputs: static + dynamic 3DGS layers, a sky cubemap, per-camera ISP corrections; non-pinhole camera
support via 3DGUT. Waymo Open: **PSNR 28.26 / SSIM 0.859** (dynamic regions 24.93 / 0.793), claimed
**+2.01 dB** over the strongest baseline.

**Release reality (probed at the repo, second probe after the paper).** Apache-2.0. **Code only** in
the repo; **weights auto-download from HF** on first run (`pa-front`, `pa-multiview`, `pq-front`).
Input format is **NCore V4 sequences** (`.json`) — i.e. the
`nvidia/PhysicalAI-Autonomous-Vehicles-NCore` convention. ⚠️ **The standalone CLI exports only the
STATIC scene gaussians, to PLY** — not `.nurec`, not USD; NVIDIA positions the PLY as an
*initialisation* for downstream NuRec training. Python 3.11 + `uv`; **no aarch64/Jetson support is
documented** (build path unknown — that is an open question, not a refutation).

**Scepticism / comparability.** PSNR/SSIM are *reconstruction* metrics on **held-out frames of the
recorded trajectory**. The paper **does not quantify degradation at lateral offsets from the log** —
which is the only regime that matters for closed-loop counterfactual driving. Stated limitations:
Gaussian-count vs quality tension; rigs unlike the training corpus (low-mounted, fisheye-only)
**need fine-tuning**; **three-keyframe piecewise-linear actor trajectories cannot capture sub-second
non-rigid motion** (pedestrian articulation).

**What WE change.** We already proved the hard half: NuRec's gaussian payload is plain gzip+msgpack
and **renders on Thor with gsplat** (`INHERITED`, commit `63ae826`; artifact
`stack/experiments/nurec-gsplat/`). Instant NuRec closes the other half — it lets us **generate new
scenes from our own logs** instead of consuming NVIDIA's 574 shipped ones. Concretely: (a) add an
`instant-nurec` intake that emits PLY; (b) point our existing `nurec_loader.py` at the PLY path
(one loader branch, static layer only, matching the CLI's actual export); (c) that produces the
**low-OOD closed-loop renderer** the 2026-07-24 note asked for, on hardware we own.

**Cheapest discriminating experiment (~2–4 GPU-h, eval A40, $0 marginal).**
Reconstruct **3 PhysicalAI val episodes** and score with the **negative control we already built**:
`grad-NCC` frame-identification against 5 candidate reference frames.
- **Pre-registered pass:** `argmax_grad_ncc` = the correct frame on ≥ 3/3 episodes, with margin
  ≥ the 0.0806 we measured on the shipped scene (`MEASURED`,
  `stack/experiments/nurec-gsplat/FINDINGS.md`).
- **Pre-registered fail:** margin collapses, or a wrong frame wins ⇒ **feed-forward reconstruction is
  not usable on our rig**, and we stay on NVIDIA's shipped scenes. Write that outcome either way.
- ⛔ **Do not report PSNR on our night clips.** We already MEASURED that PSNR and plain NCC rank a
  **wrong** frame first on this corpus (PSNR 17.457 @ frame 150 > 16.758 @ the correct frame 0).
  Instant NuRec's 28.26 dB is on Waymo daylight and is **not comparable to anything we compute**.

---

## 2. ⭐⭐ 0-GPU, highest decision-value: ADE does **not** predict closed-loop — and *progress* does

**Source:** *Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of
NAVSIM and Bench2Drive*, Yiru Wang et al., arXiv **2605.00066**, 2026-04-30 —
[html](https://arxiv.org/html/2605.00066v1). **Class: PUBLISHED.**

**The numbers.** Eight methods with complete paired data on both benchmarks (UniAD, Hydra-MDP, WoTE,
Hydra-NeXt, VADv2, SafeDrive, SparseDriveV2, DriveSuprim), correlated against Bench2Drive Driving
Score:

| open-loop metric | Spearman ρ vs closed-loop DS | p |
|---|---|---|
| **traditional L2 (ADE/FDE)** | **−0.36** | **0.43 — not significant** |
| PDMS aggregate | 0.90 | 0.002 |
| **Ego Progress (EP) alone** | **0.83** | — |
| No-Collision (NC) alone | 0.45 | — |

Their sharpest example: **UniAD has the lowest L2 (0.73) and the second-worst DS (37.72)**;
Hydra-NeXt has *worse* L2 (0.92) and the **best** DS (65.89).

**Scepticism.** **n = 8** — a Spearman ρ on 8 points has a very wide interval, and they report a
p-value but no CI. The two benchmarks also differ in sensor suite and scenario distribution, so some
of the −0.36 is benchmark mismatch, not a property of ADE. **Treat the direction as established and
the magnitudes as indicative.** Corroborating, independently:
[*Post-Training in End-to-End Autonomous Driving: A Unified View*](https://arxiv.org/html/2607.08072v1)
(Yang et al., arXiv 2607.08072, 2026-07-09) states that PDMS/EPDMS *"now cluster within a narrow
range, making a single headline score less useful for separating real methodological progress from
random variation."*

**What WE change — three things, all free.**
1. **This is external validation of the four-families rule.** Sayed's binding directive
   (`CLAUDE.md`) is now the *published* position, not a local preference. Cite 2605.00066 in the
   gate protocol so nobody re-argues it.
2. **`Ego Progress` is a LONGITUDINAL quantity and it is the strongest single closed-loop
   predictor** (ρ = 0.83 ≫ NC's 0.45). That aligns exactly with our own finding that **88.7 % of the
   oracle gap is longitudinal** (`INHERITED`, task #44 / `CLAUDE.md`). ⇒ Add a **progress-ratio**
   scalar (`along-track distance covered ÷ human's`) to `taniteval/four_families.longitudinal`. We
   already compute `along_bias_m` and `along_final_bias_m` there — progress-ratio is a few lines on
   the same tensors (`MEASURED`, `taniteval/taniteval/four_families.py:97-127`).
3. **Our published corroboration is stronger than theirs and we should say so.** Our open-loop
   0.45 m → closed-loop 1.69 m divergence (`INHERITED`, memory `flagship-closed-loop-gap`) is a
   *within-model* observation, which is cleaner than a cross-method rank correlation at n = 8.

**Cheapest discriminating experiment (0 GPU, hours).** We hold **~27 arms' per-window pred/gt/cv
dumps** on `tanitad-eval` (`INHERITED`, memory `evalpod-banked-window-dumps`). Recompute
progress-ratio and the full four families for all 27, then correlate each family against the
**closed-loop** numbers we already have for the arms that have them.
- **Pre-registered pass:** progress-ratio ranks arms differently from `ade_0_2s` (Kendall τ < 0.7)
  **and** tracks closed-loop better. Then it becomes a **gate emitter**.
- **Pre-registered fail:** τ > 0.9 with ADE ⇒ on *our* corpus progress adds nothing, ADE is a
  sufficient longitudinal proxy, and we say so publicly instead of importing the field's framing.
- Estimator: **paired episode-cluster bootstrap** over the 40 val episodes
  (`taniteval/taniteval/ci.py:261 paired_episode_cluster_bootstrap`) — ⛔ never
  `overlapping_holdout_se` (`ci.py:121`), which biases the point estimate.

---

## 3. ⭐⭐ Close the binding LONGITUDINAL gap: headway/TTC from `obstacle.offline` (0 GPU)

**Sources (PUBLISHED, for the metric definitions and their standing):** criticality metrics —
TTC, Time-to-Brake, **Time Headway (THW)**, Deceleration-to-Safety-Time — are the field's standard
longitudinal safety family; a 2026 treatment adds **False Speed Reduction** and **Maximum
Deceleration Rate** as *effort-based* criticality metrics
([arXiv 2603.28029](https://arxiv.org/html/2603.28029)). Every closed-loop score we benchmark against
(NAVSIM PDMS/EPDMS, Bench2Drive DS) contains **TTC and progress** as first-class sub-scores
([NAVSIM](https://arxiv.org/html/2406.15349v1); EPDMS composition per
[IDOL](https://arxiv.org/html/2605.31476v1)).

**Our gap, MEASURED this run.** `taniteval/taniteval/four_families.py:97-127` — the LONGITUDINAL
family returns:

```
"distance_keeping": {"status": "UNAVAILABLE",
  "reason": "no lead-agent track in the episode cache — PhysicalAI-AV ships obstacle.offline
             (3D agent tracks, 97.44 % of the corpus) but our ingest does not read it.
             Implementing it is a WORK ITEM, not a pass.", "n": 0}
```

So **half of the family Sayed made binding cannot be computed at all**, and the instrument correctly
says so rather than silently passing. `obstacle.offline` covers **97.44 %** of the corpus with
**87,481 cuboids over 10 dynamic classes** (`INHERITED`, `CLAUDE.md`).

**What WE change.** Extend the episode ingest to read `obstacle.offline`, project agent cuboids into
the ego frame, select the **lead agent** (nearest in-corridor agent ahead), and emit
**headway (m), time-gap (s), min-TTC (s)** per window — plus the *n* and the per-window reason when
no lead agent is in frame (the binding rule requires per-family reasons and n, not silent drops).
This is the same ingest surface that already reads `camera_intrinsics` / `sensor_extrinsics`
(`physicalai.py:153-154`), so the pattern exists.

**Cheapest discriminating experiment (0 GPU).** Before scoring any arm, run the **discrimination
control the standard demands**: compute headway/TTC on **ground truth** and on a **hold-v0 constant-
velocity baseline** over the same windows.
- **Pre-registered pass:** the metric separates GT from CV with a paired episode-cluster bootstrap CI
  excluding 0 ⇒ the instrument can discriminate and is admissible.
- **Pre-registered fail:** CI spans 0 ⇒ our val windows are too free-flow for distance-keeping to
  bite; report the family as **NOT-APPLICABLE with n**, and say so, rather than publishing a
  metric that cannot move.

**Why this ranks 3rd and not 5th:** it is 0-GPU, it is *binding*, it is currently *absent*, and
item 2 just showed that the closed-loop score the field trusts is dominated by exactly this family.

---

## 4. ⭐ The generative closed-loop renderer is now Apache-2.0 — and it is keyed to *our* corpus

**Source:** *NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle
Simulation*, ~33 authors (NVIDIA), arXiv **2606.03159**, submitted 2026-06-02, revised 2026-07-23 —
[abs](https://arxiv.org/abs/2606.03159) · [html](https://arxiv.org/html/2606.03159v2) ·
[code](https://github.com/nv-tlabs/omni-dreams) · [weights](https://huggingface.co/nvidia/omni-dreams-models).
**Now renamed `Cosmos-Dreams`.** **Class: PUBLISHED.**

**What it is.** A **2 B-parameter** model mid/post-trained from **Cosmos-Predict 2.5**, distilled to
**2-step** generation via Self Forcing, generating **action-conditioned video autoregressively in
real time**: **68 FPS @ 720p (704×1280) single-view on one GB300** (118 ms/chunk), **105 FPS
multi-view on 16 GB300** (151 ms/chunk); 8 frames/step SV, 16 MV. Conditioning: **future ego
trajectory**, **rendered lane lines + bounding boxes**, text prompt (weather/lighting), and a
streaming KV cache of visual history. Trained on ~21 k hours (~4 M multi-view 20 s clips).

**The result that matters to us.** Deployed **closed-loop with the Alpamayo-1 policy and the AlpaSim
orchestrator** on **574 scenes of the PhysicalAI-AV NuRec dataset**, 20 s rollouts, 10 Hz replanning:
their **2 B** World-Action Model reaches **4.2 % collision rate vs 6.9 %** for **~10 B** Alpamayo-1.5
(front 0.9 vs 1.0; lateral 0.4 vs 0.6; **rear 3.0 vs 5.3**).

**Scepticism — and it is substantial.** (a) These are **within-sim** numbers on a **generative**
simulator, single run, **no CI, no seeds** — the same caveat we already apply to our own AlpaSim
numbers (`INHERITED`, memory `alpasim-runs-bare-on-a40`: sim numbers are within-sim relative).
(b) A generative simulator **evaluated by a policy trained by the same group on the same corpus** is
not an independent instrument. (c) Their own stated limitations are load-bearing: chunk-based
generation **freezes agent behaviour within a chunk**; the LightVAE speed path costs **FVD
24.8 → 45.4**; local attention window of only 6–8 latent frames; **front-wide camera bias**;
*"errors propagate"* from inaccurate world-scenario conditioning.
(d) ⚠️ **Hardware reality check, probed at the repo:** the released repo ships **post-training code
for Cosmos2 SV-HDMap**, tested on **8× H100 80 GB, CUDA 12.8**; interactive inference lives in a
separate **FlashDreams** project; **no Blackwell/aarch64/Jetson support is documented**. The 68 FPS
figure is a **GB300** number. ⛔ We should **not** plan on running this on Thor.

**What WE change — the conditioning contract, not the model.** The actionable extract is the
**input format**: *a single frame + text prompt + **per-frame coarse HD-map image** + trajectory
poses*. We spent months believing PhysicalAI-AV had **no map** (settled at five probes: no lane
graph, no junctions, no traffic lights — `CLAUDE.md`). On **2026-08-02** we then found **`map.xodr`
inside the NuRec scene bundles** and extracted lane centerlines and a lane graph
(`INHERITED`, `…/Research/2026-08-02-nurec-xodr-map/`: `lane_centerlines.json`,
`lane_graph_edges.json`, `map.xodr`). ⇒ **That is precisely the conditioning signal Cosmos-Dreams
consumes — and precisely the strategic-brain topology our hierarchy thesis has been missing.**
Rendering an ego-frame raster of the xodr lane graph is a self-contained, 0-GPU deliverable that
serves **both** the strategic brain **and** any future generative-sim hookup.

**Cheapest discriminating experiment (0 GPU, then ~1 GPU-h).** Render the coarse HD-map raster from
`map.xodr` for the val episodes and add it as a **strategic-level input channel** to a frozen-trunk
probe.
- **Pre-registered pass:** strategic goal/route accuracy improves with a **paired** episode-cluster
  bootstrap CI excluding 0 ⇒ the map is the missing strategic input, and the strategic family stops
  being unmeasurable.
- **Pre-registered fail:** CI spans 0 ⇒ the map adds nothing our latents don't already carry, which
  is itself a publishable answer to H26 and stops the "we need an external corpus" line of spending.
- ⛔ **Do not** let this become a Cosmos-Dreams integration. Integration is a PI spend decision;
  the raster is ours and does not wait.

---

## 5. Quantisation for Thor must be scored on **trajectory divergence**, not on weight error

**Sources.**
(a) *DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models*,
Xu, Wang, Li, Zhu, Shen, arXiv **2604.11572**, 2026-04-13 —
[html](https://arxiv.org/html/2604.11572v1).
(b) *Quantizing Time-Series Models As Dynamical Systems: Trajectory-Based Quantization Sensitivity
Score*, Pavlova, Zhu, Vitanova, Semenova, Li, arXiv **2606.13300v3**, 2026-06-15 —
[html](https://arxiv.org/html/2606.13300v3). **Class: PUBLISHED.**

**The shared claim.** In sequential control, quantisation error is **not** locally bounded: it is
**temporally accumulated and geometrically amplified**, so a model can hold its headline accuracy
while its *executed trajectory* drifts. DA-PTQ demonstrates the failure mode — a W4A8 baseline
(QuantVLA) buys **55.8 % speedup for a 5.4-point success-rate drop**; DA-PTQ recovers to **48.9 %
vs 51.3 % FP** at **42.5 % memory / 54.8 % speedup** by (i) folding a low-rank affine correction
into the weights (zero inference overhead) and (ii) allocating **BF16 to the top-30 % drift-sensitive
layers, W4 to the rest**.

**TQS is the one that transfers to us.** It defines a **finite-time-Lyapunov-style** per-layer score:
perturb one layer by its quantisation residual, roll the model out, and take the growth rate of
trajectory divergence — **forward-pass only, no gradients, no Hessian**. Validated on **113 M–277 M**
foundation models (Aurora-small, TimesFM-2.5, Pangu-Weather) — **the same scale band as our 286 M
WM** — over 30-day / 500-step autoregressive rollouts. Results: **≥ 32× compression at ≤ 1 % MAE
degradation** (Aurora), **~4×** (TimesFM), and **64 of 75 variable-model wins** vs uniform baselines
at matched compression. Their Appendix A.12.1 reports that **TQS ranks different layers as critical
than curvature/Hessian methods**, flagging I/O-boundary modules. The paper explicitly frames the
model as `x_{t+1} = F_θ(x_t)` — **directly applicable to a latent-state autoregressor**, which is
what our world model is.

**Scepticism.** Neither paper is on driving. Neither reports **any** CI or seed variance (DA-PTQ is
explicitly point-estimate-only; TQS reports win-counts). TQS is **not benchmarked against Hessian or
SQNR** head-to-head — only against GPTQ/GPTAQ/QEP/RTN. And it is **not cheap**: **32–58 minutes per
probe point**, 7.7–11.5 h for a 10–16-point sweep.

**What WE change.** Thor deployment is on the roadmap and INT8/FP8/NVFP4 is the obvious lever
(Thor: 2070 FP4 TFLOPS, [NVIDIA](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)).
⛔ **Do not accept a quantisation recipe that is validated on ADE.** Our own programme has the exact
failure this literature describes: **σ dissipates and the belief collapses to an attractor over a
k-step recursive rollout** (`INHERITED`, `…/Research/STATE.md`: cos_rollout → chance by k3;
attractor 0.219 → 0.805). A quantisation that perturbs that recursion is *precisely* the case where
a 1-step metric lies. ⇒ Score any Thor quantisation on the **four families + closed-loop divergence
over k**, and rank layers by a TQS-style rollout-divergence probe.

**Cheapest discriminating experiment (~3–4 GPU-h on the eval A40, before any Thor work).** Take the
deployed arm; quantise **uniformly to W8 and to W4**; run the existing **blind K-step rollout**
harness (`Implementation/belief_rollout_diagnostic/blind_rollout_flagship.py`) plus the four
families, with the paired episode-cluster bootstrap.
- **Pre-registered pass for "ADE is sufficient":** ADE degradation and four-family degradation agree
  in rank at every bit-width ⇒ skip TQS entirely, quantise uniformly, save the sweep.
- **Pre-registered fail (⇒ build TQS):** any bit-width where **ADE moves < 5 % but a family (most
  likely LONGITUDINAL speed-bias, or the tactical decision confusion) moves with a CI excluding 0**
  ⇒ decision quality is degrading invisibly, and the per-layer TQS sweep is justified.
- This is the negative control **first**, the instrument **second** — the order the standard requires.

---

## 6. Keep the inverse-dynamics signal **in the loop**, and feed it as a **displacement**

**Sources.**
(a) *IDOL: Inverse-Dynamics-Guided Future Prediction for End-to-End Autonomous Driving*, Zhang, Li,
Li, arXiv **2605.31476**, 2026-06-01 — [html](https://arxiv.org/html/2605.31476v1).
(b) *WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning*, Yang, Lu, Xia, … Zhang,
arXiv **2512.19133**, 2025-12-22 — [abs](https://arxiv.org/abs/2512.19133).
(c) *Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving*, Wang, Zheng, Chen
et al. (CAS / Tsinghua / Chang'an), arXiv **2603.24581**, 2026-03-25 —
[html](https://arxiv.org/html/2603.24581v1). **Class: PUBLISHED.**

**IDOL's mechanism, and the surprise.** The IDM is **not** a training-time auxiliary that gets thrown
away — it **runs at inference**. It takes **adjacent latent BEV states** and emits a *spatial dynamics
map* plus a *global dynamics feature*, which **refine the ego planning query** in a closed-loop
refinement module. It never predicts raw actions. Ablation (their Table 3): **no IDM 87.3 → IDM only
89.2 → IDM + closed-loop refinement 90.0 PDMS** on NAVSIM v1 navtest; 38.0 EPDMS on NAVSIM v2 navhard
(+10.1 over WoTE).

**Latent-WAM's inconvenient ablation — read it before believing any world-model paper.** Their
104 M-param model hits **89.3 EPDMS** (SOTA), but the component breakdown is:
scene compression **−0.2**, **+ geometric distillation +0.7**, **+ world model +0.1**, **+ ego-status
supervision +0.4**. ⇒ **The latent world model itself contributes +0.1 EPDMS.** The gain is carried by
**geometric distillation from a foundation model** and by **ego-status supervision** — not by
imagination. That is a hard, publicly-reported number that a programme whose thesis is imagination
should sit with.

**Scepticism.** All three are **NAVSIM/nuScenes single-run, no seeds, no CI, no longitudinal/lateral
split** (confirmed per-paper). Per the post-training survey, EPDMS scores **cluster narrowly**, so a
+0.1 or +0.7 delta is inside the range that "does not separate progress from random variation."
IDOL's +2.7 total is larger and more likely real; Latent-WAM's +0.1 world-model term is
**not distinguishable from noise** by their own field's admission — which is the point.

**What WE change.** Our IDM stream is live (task #48 P9) and our IDM is a **training-time auxiliary**.
IDOL argues for a **latent-transition readout that persists at inference and refines the plan query**.
Combine with **backlog A7** (Delta-JEPA: feed the IDM decoder the latent **displacement**
`z_{t+1} − z_t` rather than concatenated endpoints) — same input surface, one-line change, and the two
proposals compose: a displacement *is* the latent-transition feature IDOL refines with.

**Cheapest discriminating experiment (~1–2 GPU-h, frozen trunk, no retrain).** On the deployed arm,
compute `Δz = z_{t+1} − z_t` from the existing predictor and add it as **one extra token** to the
planner query; score with four families + paired bootstrap.
- **Pre-registered pass:** tactical decision quality or LONGITUDINAL speed-bias improves with CI
  excluding 0 ⇒ promote to a trained-config lever (**D-018 escalate**).
- **Pre-registered fail:** CI spans 0 ⇒ record that IDOL's gain does not transfer to a
  camera-only, non-BEV latent, and stop. **Commit both outcomes now.**
- ⛔ Do **not** import IDOL's number as evidence for our architecture: their latent is a **BEV
  feature map** with LiDAR-grade geometry; ours is not. Different substrate, different claim.

---

## 7. Backlog **B5 is still novel** — nobody has run frozen-video-pretrained vs from-scratch

**Sources.**
(a) *Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving*, L. Wang,
Yang, Bai, … C. Lu (Virginia Tech / Purdue / XPENG / Nanjing), arXiv **2601.22032v2**, 2026-07-02 —
[html](https://arxiv.org/html/2601.22032) · [code (promised on acceptance)](https://github.com/linhanwang/Drive-JEPA).
(b) *Latent Video Prediction Learns Better World Models*, Alrasheed, Yazdan Parast, Azam, Bailey,
Akhtar, arXiv **2605.15618**, 2026-05-15 — [html](https://arxiv.org/html/2605.15618v1).
**Class: PUBLISHED.**

**The confirmation that matters.** `BACKLOG.md` B5 calls frozen V-JEPA-2 vs from-scratch on the same
corpus/steps *"the one experiment nobody in the field has run."* **Drive-JEPA is the closest anyone
has come, and it does NOT run it**: it **initialises** from V-JEPA-2 and then **fine-tunes** with the
V-JEPA objective on curated driving video. There is no frozen arm and no matched-corpus from-scratch
arm. ⇒ **B5 stands. Do not retire it.** *(Absence claim, two probes: the paper's own ablation table
and its method section; I did not find a frozen/scratch arm in either.)*

**What their ablation does tell us (Table 6, NAVSIM v2, vs a ResNet34 baseline):**
V-JEPA-2 checkpoints **+1.7 EPDMS** → **driving-video continued pretraining +2.0** →
trajectory distillation +0.4 → momentum-aware selection (+3.7 cumulative).
⇒ **The continued-pretraining step is worth MORE than the initialisation.** That reorders B5's
design: the interesting arm is not "frozen V-JEPA-2 as-is" but **"V-JEPA-2 + continued pretraining on
OUR corpus, then frozen."** Headline: 93.7 PDMS (v1) / 87.8 EPDMS (v2), narrowly over DriveSuprim
(93.5 / 87.1) — **no seeds, no CI**, and a 0.2–0.7-point margin inside the "narrow cluster" the
survey warns about. Inputs: **512×256 front camera only**, 8-frame clips @ 2 Hz.

**The mechanism paper, and its limits.** 2605.15618 controls masking strategy and varies **only the
prediction target** (V-JEPA latent vs VideoMAEv2 pixel), over >1000 A100-h on four matched ~300 M
ViT-L encoders. Latent prediction wins decisively where it matters to us: under maximum
spatiotemporal patch dropout **V-JEPA-2.1 retains 46.1 % accuracy where VideoPrism collapses to
2.7 %** (while keeping 0.98 cosine similarity — a warning that feature similarity is not
task-preservation), and **3–4× higher directional-coherence** on video reversal.
⚠️ **But it is Something-Something-v2 action *classification* with frozen attentive probes — not
action-conditioned dynamics, not planning, not driving.** It supports *"latent target beats pixel
target as a representation"* and **nothing stronger**.

**What WE change.** Two edits to B5's spec: (i) the frozen arm should be **V-JEPA-2 + continued
driving-video pretraining**, per Drive-JEPA's own ablation ordering; (ii) the read-out should include
**occlusion/degraded-visibility robustness**, because that is where latent prediction's published
advantage is largest and it maps onto our **H15/D8 degraded-visibility** work.

**Cheapest discriminating experiment.** Run B5 as written but add a **third, nearly-free arm**: the
frozen V-JEPA-2 encoder **without** continued pretraining. That converts B5 from a 2-way to a 3-way
comparison and directly tests Drive-JEPA's *"+2.0 > +1.7"* ordering on our corpus.
- **Pre-registered fail for the whole direction:** if from-scratch-on-our-corpus ≥ both V-JEPA arms
  with paired CI excluding 0, then our **flagship-v1 encoder remains the stronger substrate** —
  consistent with what we already MEASURED cross-rig (+0.657 speed R² for plain frozen v1
  vs −0.667 for the camera-conditioned own-encoder; `INHERITED`,
  `…/2026-07-22-encoder-strategy-and-vjepa2ac.md`) — and video pretraining is closed for us.

---

## 8. Anchors: build the vocabulary by **FPS over a corpus**, and constrain **kinematically at decode**

**Sources.**
(a) *DriveAnchor: Progressive Anchor-based Flow Learning for Autonomous Driving Planning*, Yan, Tang,
Qiu, Liu, Xu (Meituan Autonomous Driving), arXiv **2606.00519**, 2026-05-30 —
[html](https://arxiv.org/html/2606.00519v1).
(b) *PLAN-S: Bridging Planning with Latent Style Dynamics for Autonomous Driving World Models*, Qiu,
He, Chen, Huang, Wang, Wang, Zheng, arXiv **2606.06014**, 2026-06-04 —
[html](https://arxiv.org/html/2606.06014v1). **Class: PUBLISHED.**

**DriveAnchor — the one number that is our thesis.** A **2,398-entry** anchor vocabulary built by
**farthest-point sampling over 100 M+ real driving frames** (temporally disjoint from train/eval),
all evaluated in a **single forward pass**, **2.06 ms/scene on NVIDIA Drive Orin at fp16**. Stage 2
imposes **six kinematic constraints — speed, acceleration, curvature, jerk, lateral acceleration,
lateral jerk**. Result: **near-range collision 27.2 % → 2.9 % (−89 %)**, mean reward +32 %, and
**`gt_ADE@30` essentially unchanged at 0.23**. ⇒ **A large safety gain with ADE flat** — the clearest
published instance of exactly the effect Sayed's four-families rule exists to catch.

**Scepticism — heavy.** ⚠️ **Internal dataset only. No nuPlan, no NAVSIM, no Waymo.** Their own text
concedes it. No CI, no seeds, no anchor-count ablation, **no longitudinal/lateral decomposition**
(the six constraints are applied to a unified 160-D trajectory space), and **no comparison with
DiffusionDrive**. The 2.06 ms is on **Drive Orin**, not Thor, and for a planner head only.
**Not comparable to our paired episode-cluster bootstrap in any way** — treat the *direction*
(anchors + kinematic constraints ⇒ safety up, ADE flat) as the transferable claim and the magnitudes
as unverifiable.

**PLAN-S — the piece our strategic level can use.** A two-level design where a **style code**
conditions a four-channel semantic cost map (dynamic obstacles / off-road / static obstacles /
drivability) via dual AdaFiLM, and the cost map then guides planning. Crucially, **the style code for
the anchor-score instantiation is an explicit 2-D score of longitudinal and lateral aggressiveness**
— an *architectural* lat/lon separation, not just a reporting one. nuScenes: L2 0.55 m (vs 0.59) and
**collision@3 s 0.25 % vs 0.43 % (−42 %)**; NAVSIM 89.4 PDMS. Their interface ablation shows upstream
prior-cost fusion **and** downstream cost-gated refinement are both needed (0.111 % vs weaker alone),
and — usefully honest — **hand-designed rule cost wins on aggregate (89.36) while learned cost wins
by +17.2 PDMS on hard scenes**. Single seed, no CI, no code.

**What WE change.** We already MEASURED **state-conditioned anchors** (task #32) — DriveAnchor's
anchors are deliberately **not** state-conditioned, so that fork is already ours to compare. The two
transferable pieces are: (i) **build the anchor vocabulary by FPS over our corpus** rather than by
k-means/handcrafting, which guarantees coverage of rare kinematics instead of density-weighting toward
cruising — directly relevant to our **0/881 accelerate** collapse and the **5-way softmax that mixes
lat+lon** (`INHERITED`, memory `longitudinal-blindness-root-cause`); and (ii) apply the **six
kinematic constraints at decode time**, separated into a longitudinal set (speed/accel/jerk) and a
lateral set (curvature/lat-accel/lat-jerk) — which is the *architectural* version of the split
Sayed made binding for reporting, and which PLAN-S independently arrives at.

**Cheapest discriminating experiment (0 GPU for the vocabulary, ~1 GPU-h to score).** Build an FPS
anchor set from our **2,376-episode canonical corpus** (parity-safe: FPS *selects among existing GT
trajectories*, it does **not re-select episodes** — the parity invariant is untouched). Compare
coverage against the current anchor set on the **rare-kinematics strata** (accelerate, decelerate,
junction).
- **Pre-registered pass:** FPS anchors cover the accelerate/decelerate strata materially better, and
  swapping them improves the **TACTICAL** family (selected-vs-executed manoeuvre confusion) with a
  paired CI excluding 0.
- **Pre-registered fail:** coverage is comparable ⇒ the anchor *vocabulary* is not our bottleneck,
  which sharpens the case that the **5-way lat+lon-mixing softmax** is, and we spend there instead.

---

## 9. Protocol scepticism — the standing rule this scan reinforces

Applied uniformly across every paper above:

| paper | benchmark | seeds / CI | lat/lon split | comparable to us? |
|---|---|---|---|---|
| Instant NuRec 2607.14203 | Waymo (recon PSNR/SSIM) | none | n/a | ⛔ PSNR invalid on our night corpus (MEASURED) |
| Cosmos-Dreams 2606.03159 | PhysicalAI-AV NuRec, 574 scenes | none | front/lat/rear collision only | partially — **same corpus**, but within-sim & self-evaluated |
| 2605.00066 (open vs closed) | NAVSIM × Bench2Drive, **n=8** | p-values, no CI | no | direction yes, magnitudes no |
| IDOL 2605.31476 | NAVSIM v1/v2 | none | no | different substrate (BEV+LiDAR) |
| Latent-WAM 2603.24581 | NAVSIM v2, HUGSIM | none | no | world-model term **+0.1 EPDMS** ≈ noise |
| Drive-JEPA 2601.22032 | NAVSIM v1/v2, Bench2Drive | none | no | margin 0.2–0.7 inside the narrow cluster |
| DriveAnchor 2606.00519 | **internal only** | none | no | ⛔ unverifiable magnitudes |
| PLAN-S 2606.06014 | nuScenes, NAVSIM | single seed | **yes, architecturally** | design idea only |
| DA-PTQ 2604.11572 | SimplerEnv | none | n/a | mechanism transfers, numbers don't |
| TQS 2606.13300 | Aurora/TimesFM/Pangu | win-counts | n/a | **scale band matches ours (113–277 M)** |

**Not one of the ten reports a confidence interval on its headline number.** Our
`paired_episode_cluster_bootstrap` (`taniteval/taniteval/ci.py:261`) is, on this evidence, ahead of
published practice. ⛔ That is also the reason **no number in this table may be used as a baseline we
"beat"** — they are point estimates on different corpora with different estimators. They are
directions and mechanisms, not targets.

---

## 10. ⛔ DO NOT ADOPT — findings that close a direction (this is a result, not a gap)

1. **The full latent-action / hierarchical-latent-action stack, again.** *Hierarchical Latent Action
   Model* (Kim, Pinto, Kim — Yonsei/NYU, arXiv **2603.05815**, 2026-03-06,
   [html](https://arxiv.org/html/2603.05815v1)) builds a 2-level hierarchy (frame-interval inverse
   dynamics → variable-length skills via H-Net chunking) and evaluates on **LIBERO manipulation
   only** (94 % vs BAKU 87 % full-data; 45 % vs 23 % at 10 % demos). **No LAPO/Genie/LAPA comparison,
   no driving, no continuous metric-scale action recovery, no code.** *DLAM: Distributional Latent
   Actions with Temporal Constraints* (Tang et al., arXiv **2607.27138**, 2026-07-29,
   [html](https://arxiv.org/html/2607.27138)) is the newest — diagonal-Gaussian latent transitions
   with composition/reversal constraints — and is **MetaWorld / LIBERO / a 6-DoF arm only**, with the
   variance *explicitly not interpreted as calibrated uncertainty*.
   ⇒ **Refutation criteria R1 and R4 from `2026-07-27-latent-action-models` §1 still fire at
   2026-08.** The 2026-07-27 verdict — *stick with a supervised IDM* — **holds unchanged**, now with
   four more months of literature behind it. Re-confirming a live verdict is cheaper than re-running
   the study; this is the re-verification, and it is negative.
   ✅ **One genuinely useful import from DLAM:** its stated failure mode — *reconstruction-trained
   latent codes capture camera motion, background dynamics and appearance change that improve frame
   prediction but are only weakly related to control* — is a **named hypothesis for our own
   σ-dissipation/attractor pathology**, and its **composition + reversal** consistency checks are
   cheap **diagnostics** we could run on our existing latents without adopting the model.

2. **Do not port Cosmos-Dreams to Thor.** Post-training is documented on **8× H100**; inference lives
   in a separate FlashDreams project; **no aarch64/Blackwell-Jetson support is documented** and the
   headline FPS is a **GB300** figure. Our Thor path stays **gsplat + NuRec/PLY** (which we have
   already MEASURED working), not generative video.

3. **Do not treat "hierarchy helps" as settled by the literature.** The 2026 hierarchical-driving
   papers I surfaced either evaluate on non-driving RL benchmarks or report hierarchy gains without
   isolating them, and **Latent-WAM's own ablation puts the latent-world-model term at +0.1 EPDMS**.
   Our **H26 cross-alignment proof (task #15)** remains the discriminating experiment; no external
   result substitutes for it.
   ✅ **But borrow the method.** *What Probing Reveals about Autonomous Driving: Linking Internal
   Prediction Errors to Ego Planning* (Jeon, Kim, Vinitsky, Kim — GIST/NYU, arXiv **2606.31106**,
   2026-06-30, [html](https://arxiv.org/html/2606.31106)) is the closest published template for H26:
   linear probes on early/late attention layers predicting surrounding-vehicle and ego future
   positions (discretised to an **8×8 ego-FOV grid**, 1–4 s ahead), across BC / PPO-RL / SMART on
   ~90 k WOMD scenes. Crucially they go past correlation to **intervention** — *replacing* an
   incorrect surrounding-vehicle representation with the correct one changes the ego-planning probe
   in **36/59 BC, 37/58 RL, 40/64 IL** cases, i.e. a **causal** link from internal prediction error
   to planning error. They also report probing performance **plateauing around 10 k training scenes**
   and models *failing to emphasise safety-critical vehicles early enough*.
   ⇒ **Our H26 proof should be an intervention study, not a correlation study.** We already have the
   probe machinery (`Implementation/orthogonality_verification/`, `strategic_probes.py`); what we
   lack is the *swap-the-latent-and-watch-the-plan-move* step. ⚠️ Note the fraction: even in their
   best case only **~62 %** of interventions moved the plan — so a partial effect is the *expected*
   outcome, and we must pre-register the threshold before running, not after.

---

## 11. What I would do in the next turn, in priority order

Each is executable with the stated resource; **items 1–3 need no GPU at all**, so a gated fleet is no
excuse (`CLAUDE.md`: gated ≠ idle).

| # | action | resource | maps to |
|---|---|---|---|
| 1 | **Implement `obstacle.offline` lead-agent ingest → headway / time-gap / min-TTC** in `four_families.longitudinal`, with the GT-vs-CV discrimination control run **first** | 0 GPU | §3, binding rule |
| 2 | **Add progress-ratio** to the LONGITUDINAL family; recompute all four families over the **27 banked window dumps**; correlate against the closed-loop numbers we hold | 0 GPU (eval pod CPU) | §2 |
| 3 | **Render the ego-frame lane-graph raster from `map.xodr`**; wire as a strategic-level input channel | 0 GPU | §4 |
| 4 | **Instant NuRec on 3 val episodes**, validated by **grad-NCC frame identification**, not PSNR | ~2–4 GPU-h | §1 |
| 5 | **Uniform W8/W4 quantisation × K-step blind rollout × four families** — the negative control that decides whether a TQS sweep is worth 8–11 h | ~3–4 GPU-h | §5 |
| 6 | **`Δz` latent-displacement token into the planner query** on the deployed arm, frozen trunk | ~1–2 GPU-h | §6 |
| 7 | **FPS anchor vocabulary** over the canonical 2,376-episode corpus; coverage diff on rare-kinematics strata | 0 GPU to build | §8 |
| 8 | **Amend B5's spec** to a 3-way arm set (scratch / frozen V-JEPA-2 / frozen V-JEPA-2 + continued driving pretraining) | spec edit, 0 GPU | §7 |

**⚠️ ESCALATION (not a "please merge" line in a doc — `CLAUDE.md` rule 3).** Two integration
decisions belong to the orchestrator/PI and are named here so they do not sit unread:
- **Items 1 and 2 change what a gate reports.** They *add* metrics and remove no ADE row, so they are
  additive under the binding rule — but the **gate emitter list** in `GATE_PROTOCOL.md` must be
  updated in the same change, or the metrics exist and never reach a decision.
- **Item 3 (`map.xodr` → strategic input)** touches the strategic brain's input contract and is a
  **D-018 escalate** before it can become a trained config.

---

## 12. Deliverable manifest

| artifact | where it lives |
|---|---|
| **This report** | `TanitAD Research Hub/Architecture & Inference/Research/2026-08-03-sota-scan/SOTA_SCAN.md` (repo, **staged**) |

**Hosts touched:** none. **GPU spent:** 0. **Nothing is stranded on a pod or in a worktree** — the
single artifact is in the repo working tree and staged for the orchestrator to commit.

**Read-with (for the claims marked `INHERITED` above, none of which I re-verified this run):**
`Project Steering/BACKLOG.md` (A7, B5) · `Project Steering/MODEL_REGISTRY.md` ·
`TanitAD Research Hub/Architecture & Inference/Research/STATE.md` (σ-dissipation, attractor) ·
`…/Research/2026-07-27-latent-action-models/LATENT_ACTION_RESEARCH.md` (R1–R4 criteria) ·
`…/Research/2026-07-22-encoder-strategy-and-vjepa2ac.md` (encoder forks already refuted) ·
`…/Research/2026-08-02-nurec-xodr-map/XODR_MAP.md` (the map we found) ·
`stack/experiments/nurec-gsplat/FINDINGS.md` (grad-NCC control; **MEASURED**, re-read this run) ·
`taniteval/taniteval/four_families.py` + `taniteval/taniteval/ci.py` (**MEASURED**, re-read this run).

---

### Sources

- [Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation (arXiv 2607.14203)](https://arxiv.org/abs/2607.14203) · [HTML](https://arxiv.org/html/2607.14203v1) · [code](https://github.com/NVIDIA/instant-nurec)
- [Do Open-Loop Metrics Predict Closed-Loop Driving? (arXiv 2605.00066)](https://arxiv.org/html/2605.00066v1)
- [NVIDIA OmniDreams / Cosmos-Dreams (arXiv 2606.03159)](https://arxiv.org/abs/2606.03159) · [HTML](https://arxiv.org/html/2606.03159v2) · [code](https://github.com/nv-tlabs/omni-dreams) · [weights](https://huggingface.co/nvidia/omni-dreams-models)
- [Post-Training in End-to-End Autonomous Driving: A Unified View (arXiv 2607.08072)](https://arxiv.org/html/2607.08072v1)
- [DA-PTQ: Drift-Aware Post-Training Quantization for VLA Models (arXiv 2604.11572)](https://arxiv.org/html/2604.11572v1)
- [Quantizing Time-Series Models As Dynamical Systems: TQS (arXiv 2606.13300)](https://arxiv.org/html/2606.13300v3)
- [IDOL: Inverse-Dynamics-Guided Future Prediction (arXiv 2605.31476)](https://arxiv.org/html/2605.31476v1)
- [Latent-WAM: Latent World Action Modeling (arXiv 2603.24581)](https://arxiv.org/html/2603.24581v1)
- [WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning (arXiv 2512.19133)](https://arxiv.org/abs/2512.19133)
- [Latent World Models for Automated Driving: A Unified Taxonomy (arXiv 2603.09086)](https://arxiv.org/abs/2603.09086)
- [Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation (arXiv 2601.22032)](https://arxiv.org/html/2601.22032)
- [Latent Video Prediction Learns Better World Models (arXiv 2605.15618)](https://arxiv.org/html/2605.15618v1)
- [DriveAnchor: Progressive Anchor-based Flow Learning (arXiv 2606.00519)](https://arxiv.org/html/2606.00519v1)
- [PLAN-S: Bridging Planning with Latent Style Dynamics (arXiv 2606.06014)](https://arxiv.org/html/2606.06014v1)
- [Hierarchical Latent Action Model (arXiv 2603.05815)](https://arxiv.org/html/2603.05815v1)
- [DLAM: Distributional Latent Actions with Temporal Constraints (arXiv 2607.27138)](https://arxiv.org/html/2607.27138)
- [What Probing Reveals about Autonomous Driving (arXiv 2606.31106)](https://arxiv.org/html/2606.31106)
- [Effort-Based Criticality Metrics for 3D Perception Errors (arXiv 2603.28029)](https://arxiv.org/html/2603.28029)
- [NAVSIM: Data-Driven Non-Reactive AV Simulation and Benchmarking (arXiv 2406.15349)](https://arxiv.org/html/2406.15349v1)
- [Introducing NVIDIA Jetson Thor (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
