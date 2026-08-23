# Screening: two papers hand-delivered by Sayed (2026-07-11)

Both were MISSED by our literature machinery → root-cause fix shipped as **D-028** in
`agents/_common-protocol.md` (recency-first arXiv listing scan + seam ownership). Analysis below.

## 1. AUTOPILOT-VQA (arXiv 2607.08745, 9 Jul 2026) — incident-centric dashcam VQA benchmark

**What:** VQA benchmark over real dashcam incident/near-incident videos; 9 annotated safety
categories: weather+lighting, traffic environment, road layout, road surface, signage, involved
entities, accident occurrence, impact location, **avoidability reasoning**. Paper CC-BY 4.0;
dataset released via the AUTOPILOT CVPR-2026 Kaggle competition (license = competition terms,
verify before any training use).

**Why it matters to us (we are NOT a VLM shop — three non-VQA uses):**
- **(a) Probe-suite external validity:** the 9-category taxonomy is an independently designed
  checklist of "what a safe driver must know about a scene". Fitting A3-style calibrated probes
  from OUR latents onto THEIR category labels = external evidence for the inherent-safety edge
  ("the latent state contains the safety-relevant scene variables"), on data we never trained on.
- **(b) Eval-lane data (two-lane doctrine):** real incident-centric clips → candidate never-trained
  eval corpus for the scenario DB (SC classes: impact location ↔ SC-04/SC-07 family), IF license
  permits eval use. Heterogeneous dashcams also make it a natural Y-track (YouTube strategy)
  eval anchor.
- **(c) Avoidability = counterfactual rollout:** their "was this avoidable" category is literally
  the question H15 imagine-and-select answers by rollout instead of by language — a future
  demonstrator: world-model avoidability judgments scored against their labels (paper §7 material,
  opponent-differentiating: VLM competitors answer it with words, we answer it with physics).

**Actions:** Benchmarks & Eval BACKLOG P1 item added (license check → probe-transfer experiment);
Opponent Analyzer should map the taxonomy onto SC-01..SC-14 next run.

## 2. ZipDepth (Bologna; zipdepth.github.io) — lightweight zero-shot monocular depth

**What:** 6.1M params (≈50× smaller than depth foundation models), distilled from
Depth-Anything-v2-Large over 14.1M images/17 domains, reparameterizable convs (zero custom ops).
**77 FPS TensorRT-fp16 on a 15 W Jetson Orin NX** (~397 mJ/frame). Open weights. Affine-invariant
(MiDaS objective) → **no metric scale from monocular alone**.

**Sayed's read (adopted):** efficient but with artifacts/quality gaps vs foundation depth models —
NOT a Phase-0 item; a **second-step** candidate for (a) grounding extraction heads from the latent
space, (b) depth information in safety considerations.

**Where it would slot (when its turn comes):**
- **Fallback-layer safety envelope (4B layer 4):** a 6.1M/77FPS depth net is cheap enough to run
  as an INDEPENDENT minimum-clearance monitor beside the world model on Orin — redundant-channel
  material for the UN-ADS safety case. Needs metric scale: pair with camera height + flat-ground
  calibration or a small metric adapter; artifacts argue for envelope/veto use, not planning use.
- **Extraction-head grounding (Phase 1/2):** distill-time auxiliary "latents → coarse depth"
  probe = geometric grounding evidence for D-class gates, without adding inference cost.
- **Y-track:** pseudo-depth teacher for uncalibrated YouTube dashcam data (scale-free targets fit
  the affine-invariant output exactly — no metric problem in that use).

**Actions:** Production & Optimization BACKLOG P1 item added (TRT-fp16 pipeline cousin — their §4
toolchain applies; measure on 4060 as Orin proxy when picked up). Architecture agent: note under
ENCODER_MULTICAM_OPTIMIZATION as a non-blocking side channel, not an encoder change.

**Verdict line:** paper 1 = act this phase (license check + probe transfer is cheap, evidence value
high); paper 2 = logged, second step, agreed with Sayed's placement and quality caveat.
