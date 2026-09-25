# Stream W2 — Frontier research: massively reducing labelled-data needs for driving; knowledge injection; continual/fleet learning

**Status:** IN PROGRESS (banking incrementally). **Date:** 2026-09-25. **Model tier:** Sonnet.
**Method:** web research (WebSearch/WebFetch), primary sources + arXiv ids, evidence-class discipline per
`CLAUDE.md` rule "Operating standard §1". Search/fetch budget: ≤45 searches, ≤30 fetches.

**Delta skim performed against** (headers/conclusions only, 2026-09-25):
`TanitAD Research Hub/2026-07-08-screening-digest.md`, `INITIAL_RESEARCH_SYNTHESIS.md` (v1.1,
2026-07-06), `Data Engineering/{DATA_STRATEGY_FOR_HIERARCHY,OWN_DATASET_PLAN}.md`,
`Architecture & Inference/IDM_VIDEO_PRETRAIN_DESIGN.md` (v0, 2026-07-22),
`Project Steering/PREREG_deep_research_2026-07-29.md`.

**What is already banked internally (do NOT re-report as new; this stream hunts the DELTA since):**
- H7 (data-leverage-via-IDM) already a Phase-1-core recommendation as of 2026-07-06/07-22, citing
  VPT (arXiv:2206.11795), Genie (2402.15391), LAPA (2410.11758), BCO (1805.01954), Seer (ICLR'25,
  2412.15109), DriveWAM (2605.28544), IDM-vs-BC (2602.02762). Design: non-causal predictive IDM head
  reusing the flagship WM predictor trunk, trained on CAN-labeled corpus, applied to YouTube.
  **Per the orchestrator fact sheet this line has since FAILED held-out-camera-rig transfer** — this
  is the single most important open question for §C below, and the reason §C is not "redo H7" but
  "why did it fail and what specifically fixes it."
- L2D (Apache-2.0) identified as the strategic/horizon/indicator unblocker (2026-07-21); obstacle.offline
  lead-state ingest recommendation **falsified** by gate (2026-07-21): lead state does not improve
  ego-only longitudinal ADE at the FULL-population level (+1.16% [-0.92,+3.19], inside FAIL band);
  survives as a lead-conditioned-specialist hypothesis on the 38.5% lead-present subpopulation, not yet
  pre-registered.
- Data-scaling-laws for E2E driving and Waymo's motion-forecasting scaling work are NOT yet in the
  hub docs skimmed — open territory for §A below.
- Own-dataset plan already scoped: comma2k19, Cosmos-Drive-Dreams (CC-BY-4.0), PhysicalAI-WorldModel-
  Synthetic-Scenarios (OpenMDW-1.1), PandaSet, Udacity, CARLA self-gen as the license-clean core;
  ZOD (CC-BY-SA) as flagship new real-urban ingest. This stream does not re-litigate licensing (that's
  DataEng's job) but flags where new 2026 sources/methods bear on it.
- RMFM (Reward-Modulated Flow Matching) already the Phase-1 flagship for rule-injection (H9) —
  relevant context for §H (knowledge injection) below, not re-derived here.
- σ-gated tactical MoE, K-step rollout loss, RoPE-in-FiLM already triaged into BACKLOG — not repeated.

---

## A. Data scaling laws for driving/planning

Two industrial-scale scaling-law papers now exist for planning specifically (not just perception), plus
one paper whose central claim is a direct external test of TanitAD's own architectural bet.

**A1. Data Scaling Laws for End-to-End Autonomous Driving (+ IL companion).**
Fits power-law data-scaling curves for camera E2E planning vs. dataset size and model capacity.
*Evidence:* PUBLISHED (Naumann et al., arXiv:2504.04338, CVPR2025-W WAD) — bigger model capacity shifts the
curve so the SAME accuracy needs LESS data (a capacity→data-multiplier); diminishing returns confirmed;
targeted long-tail augmentation beats indiscriminate scaling. A companion, "Data Scaling Laws for
Imitation-Learning-Based E2E AD" (arXiv:2412.02689), reports per-scenario curves; a search-snippet figure
of "turning scenarios scale consistently to ~8,192 h then plateau" could not be pinned to one paper with
certainty this pass — **UNVERIFIED which paper**, flagged rather than asserted. **Maturity: PROVEN**
(real fleets), but the plateau point is fleet/task-specific.
*Pain points:* P4 directly — tells us where on the curve we sit; P1 via long-tail-beats-random.
*Admissibility:* fully admissible (an analysis method).
*Cost:* ~2-3 eng-days to fit an analogous curve to our own checkpoints (pairs naturally with D-021's
spectral-knee study); 0 new GPU-days if existing smoke checkpoints at 10/25/50/100% corpus fraction exist.
*Experiment:* refit ADE-vs-training-fraction on existing checkpoints, report the exponent with R² and
window per our own exponent-citation rule. **A:** exponent in the ~-0.1 to -0.3 range seen in the AD
literature → confirms we are data-bound, prioritise the 10× programme below. **B:** near-flat exponent →
something other than data volume (label quality, capacity) binds first; reprioritise toward §G over §B/C.

**A2. Scaling Laws of Motion Forecasting and Planning (Waymo).**
Compute/data/model-size power-law study for a joint forecasting+planning transformer at 500K driving hours.
*Evidence:* PUBLISHED (Waymo, arXiv:2506.08228, June 2025) — training loss AND eval metrics both power-law
in compute; **closed-loop metrics scale too**, not just open-loop; compute-optimal scaling grows model
size ~1.5× as fast as dataset size; at inference, **sampling + clustering a small model's outputs makes it
competitive with a larger model** — a compute-for-params substitution. **Maturity: PROVEN**, at a scale
~40,000× ours; the transferable part is the qualitative shape, not the exponent value.
*Pain points:* P4, P9 (few A40s) — the sample-and-cluster result is usable ON our existing few-A40 budget
instead of training bigger.
*Admissibility:* fully admissible.
*Cost:* ~1 A40-day + 2 eng-days.
*Experiment:* on the existing flagship checkpoint, compare ADE@2s for one deterministic decode vs. N=8
stochastic samples clustered/averaged. **A:** clustering improves ADE by >2-3% with a paired
episode-cluster-bootstrap CI excluding 0 → adopt for eval, consider for deployment if latency allows.
**B:** no improvement → the decoder may already be too deterministic/mode-collapsed (P7-relevant evidence).

**A3. MOSAIC — Scaling-Aware Data Selection for E2E AD Systems.**
Fits a neural scaling law PER DOMAIN against the eval metric, then greedily builds the data mixture that
maximizes metric gain per added sample.
*Evidence:* PUBLISHED (Dimlioglu, Chang, Shen, Mahmood, Alvarez — NVIDIA/NYU/U. Ottawa, arXiv:2604.08366,
April 2026) — matches or beats standard-mixture baselines with **up to 80% fewer examples**, measured on
EPDMS. **Maturity: PROMISING** (single paper, not yet independently replicated, but simple and portable).
*Pain points:* P1 (selection — literally the paper's subject), P4 (a labelled-data multiplier of roughly
**5×** at fixed performance, if it transfers to composing PhysicalAI + L2D + ZOD + Cosmos-Drive-Dreams).
*Admissibility:* fully admissible; scoped to composing NEW-source mixtures, never to re-selecting the 2,376
parity episodes — the "parity is sacred" invariant must gate its use explicitly.
*Cost:* ~3-5 eng-days; ~4-8 A40-days for the per-domain curve-fitting runs (10/30/100% of each candidate
source), reusable afterward.
*Experiment:* fit 3-point curves separately for PhysicalAI-parity, L2D, Cosmos-Drive-Dreams on a fixed
small eval; check if MOSAIC's greedy rule beats an equal-weight naive mixture on held-out val by more than
the CI width. **A:** it does → adopt as the standard corpus-growth procedure. **B:** curves too noisy at
881 val windows to fit reliably → grow val before trusting any curated-mixture decision (itself a finding).

**A4. DriveVLA-W0 — world modeling amplifies the data scaling law.**
Adds future-image prediction as a DENSE self-supervision signal on top of a VLA's sparse action loss,
arguing action-only supervision leaves most of a large model's capacity unused (a "supervision deficit").
*Evidence:* PUBLISHED (BraveGroup, "DriveVLA-W0," arXiv:2510.12796, ICLR 2026) — on a public benchmark
AND a **680× larger in-house dataset**, the world-modeling-augmented VLA's gains **accelerate** as data
grows while an action-only baseline plateaus. Two instantiations: autoregressive WM for discrete-token
VLAs, diffusion WM for continuous-feature VLAs. **Maturity: PROMISING→PROVEN-directionally** (replicated
within-paper at two very different scales, not yet by a second lab).
*Pain points:* P4 directly, and this is close to an **external, independent validation of TanitAD's own
central bet** (JEPA-style latent prediction as the backbone, not an add-on). P7 — dense future prediction
IS our "imagination" objective; the paper's evidence says scaling that objective's WEIGHT/quality, not
just adding more (steer, accel) pairs, is the higher-leverage lever on a small corpus.
*Admissibility:* fully admissible (self-supervised on video, no privileged signal at inference).
*Cost:* ~3 A40-days for 3 short (5k-step) comparative arms; near-zero eng cost (it's a loss-weight sweep on
an already-implemented objective).
*Experiment:* train 3 short arms differing only in imagination-loss weight (0.5×/1×/2× current); check
whether the ADE-vs-training-fraction exponent (fit at 25/50/100% of parity corpus within each arm) gets
**steeper** as weight increases. **A:** it does (report with R²/window) → directly actionable, raise the
weight in the next full run. **B:** no change → either our LATENT-only imagination doesn't provide the
richness DriveVLA-W0's PIXEL-level future prediction does (a design gap, not a refutation), or the
objective is already saturated in its current form.

**A5. World Engine — post-training on synthesized safety-critical variations.**
Reconstructs real logs into interactive (Gaussian-splat-like) environments, extrapolates them into
safety-critical variations, and RL-post-trains an already-pretrained policy on those instead of collecting
more real long-tail miles.
*Evidence:* PUBLISHED (Li, Geiger, Li et al. — OpenDriveLab, arXiv:2606.19836, June 2026) — explicit framing
that real long-tail events are rare by definition and don't scale with more collection; GitHub repo exists.
**Maturity: PROMISING** (very recent, no independent replication seen).
*Pain points:* P4's long-tail slice specifically (the part raw hour-counting cannot fix), P8 (no safety
layer — training against synthesized near-misses is a step toward a learned safety margin).
*Admissibility:* admissible — synthesis downstream of real/license-clean logs; no privileged signal reaches
inference.
*Cost:* HIGH relative to our fleet if built from scratch; ~10-20 eng-days reusing NVIDIA's Instant NuRec
(§D4) rather than building a reconstruction pipeline ourselves; ESTIMATED 10-30 A40-days for a first pilot.
*Experiment:* skip the reconstruction pipeline first — take existing `obstacle.offline`-tagged near-miss /
lead-braking clips, apply cheap ChauffeurNet-style pose perturbations (§D1, no 3DGS needed), and briefly
post-train on the perturbed set. **A:** a safety-relevant metric (e.g. TTC-to-lead-agent) improves without
ADE regression → the thesis holds at small scale, justifying the bigger reconstruction investment. **B:**
no measurable gain → either the perturbation is too crude, or the failure mode isn't in that distribution
at all (informs WHERE the long-tail gap actually is).

## B. Self-supervised & video pretraining for driving policies

The single most important finding in this stream lives here: a Feb-2026, monocular-only, fully label-free
pretraining method that beats multi-camera+LiDAR baselines and is explicitly evaluated at 1%/10% labels.

**B1. PPGeo / SelfD / ACO — the founding precedent.**
Two-stage self-supervised geometric modeling (pose+depth, then photometric future-ego-motion prediction) on
unlabeled, uncalibrated YouTube driving video; SelfD/ACO are the semi-supervised pseudo-trajectory siblings.
*Evidence:* PUBLISHED (Wu et al., "PPGeo," arXiv:2301.01006, ICLR 2023); SelfD and ACO confirmed to exist
as related web-scale self-training precedents (exact current numbers not re-verified this pass —
**UNVERIFIED** beyond mechanism). **Maturity: PROVEN** as a paradigm, superseded in results by B7 below.
*Pain points:* P4 — the original proof that YouTube driving video carries usable policy signal at all.
*Admissibility:* admissible (label-free, vision-only).
*Cost:* low if reusing published code; mainly of historical/architectural interest now that B7 exists.
*Experiment:* superseded — see B7's experiment instead.

**B2. GenAD + OpenDV-2K.**
A generalized video-predictive model trained on the largest public multimodal driving corpus assembled
from YouTube.
*Evidence:* PUBLISHED (arXiv:2403.09630, CVPR 2024 Highlight) — OpenDV-2K: **2,059 hours** (1,747 YouTube +
312 public), 40 countries, 244 cities; zero-shot domain transfer, language- and action-conditioned
prediction, motion planning all demonstrated from the same pretrained backbone. **Maturity: PROVEN.**
*Pain points:* P4 — a ready-made, ~160× our 13-hour corpus, action-free video source with an existing
transfer recipe; P5 (untested video-pretrained encoders) — this is exactly a video-pretrained encoder
already shown to transfer to planning.
*Admissibility:* admissible if OpenDV-2K's YouTube provenance clears our licensing review (a DataEng
question, not ours to resolve, but worth flagging: our own OWN_DATASET_PLAN already lists
"OpenDV-YouTube" as **blocked** for a redistributable owned derivative — usable as a PRETRAINING-ONLY,
non-redistributed asset is a different, narrower claim that should be checked explicitly, not assumed).
*Cost:* ~5-8 eng-days to adapt the front-wide-camera-only pipeline to GenAD-style pretraining; GPU cost
dominated by whatever subset of OpenDV-2K is used (ESTIMATED 5-15 A40-days for a meaningful pretrain).
*Experiment:* pretrain the ViT encoder on a modest OpenDV-2K subset (e.g. 200h) with the GenAD-style
future-frame objective, then fine-tune the existing flagship heads on the parity corpus unchanged. **A:**
ADE@2s improves over the from-scratch flagship baseline (0.452 m) by more than CI width → a real, cheap
pretraining win. **B:** no improvement or regression → echoes the frozen-DINOv2 failure (2.17 m) and
suggests the problem is DOMAIN GAP (YouTube dashcams vs. PhysicalAI-AV's rig) rather than lack of video
pretraining per se — directly informs whether to keep pursuing B or pivot to C's camera-rig-robust methods.

**B3. UniPAD / ViDAR — 3D-aware self-supervised pretraining.**
UniPAD: differentiable-volumetric-rendering self-supervision; ViDAR: builds on it with a visual
point-cloud-forecasting pretext task, jointly modeling 3D geometry and temporal dynamics.
*Evidence:* PUBLISHED (UniPAD, arXiv:2310.08370; ViDAR, CVPR 2024 Highlight, OpenDriveLab) — UniPAD's
reported gains are **perception** metrics (+9.1/7.7/6.9 NDS for LiDAR/camera/LiDAR-camera baselines), not
planning-specific — **flagged explicitly, do not over-read as a planning number**. **Maturity: PROVEN**
for perception pretraining; planning transfer is plausible but not the number quoted here.
*Pain points:* P5 (untested pretraining objectives) — a candidate PRETEXT TASK for our own encoder, using
signal we can derive from `obstacle.offline` (3D boxes) instead of raw LiDAR.
*Admissibility:* admissible.
*Cost:* ~5 eng-days to implement a point/box-forecasting pretext head; ~2-4 A40-days to test.
*Experiment:* add a ViDAR-style future-3D-box-forecasting auxiliary head (fed by the already-available but
under-ingested `obstacle.offline`) to the existing encoder, ablate with/without on val ADE and a tactical
metric. **A:** improves either family → a nearly-free structural addition since the label source is
already license-clean on the parity corpus. **B:** no change → the earlier lead-state gate's null result
(agent state doesn't help the LONGITUDINAL head) may generalize to representation-level use too, which
would be a genuinely new (and reportable) negative result, distinct from the already-falsified DIRECT-input
use.

**B4. DINOv3 — frozen self-supervised backbone, in tension with our own H4 result.**
Meta's newest self-supervised ViT family; explicitly tested as a frozen, modular backbone swapped into
three E2E planning paradigms (regression/diffusion/scoring-based).
*Evidence:* PUBLISHED (Meta, arXiv:2508.10104, Aug 2025) — "frozen DINOv3 remains competitive" is reported
qualitatively for planning integration and quantitatively for robotic manipulation (a DINOv3-diffusion-
policy paper, arXiv:2509.17684, reports frozen DINOv3 "competitive" with fine-tuned ResNet-18 on several
manipulation tasks — **not the driving number**, flagged). **Maturity: PROMISING**, and **directly in
tension with our own MEASURED result** that frozen DINOv2/I-JEPA fails at 2.17 m ADE (vs. 0.452 m
flagship). This tension is itself the finding: either DINOv3 is a materially stronger backbone than DINOv2
for driving specifically, or the "modular frozen interface" papers use adapters/fine-tuned heads richer
than our own frozen-encoder arm did, or driving planning genuinely needs the encoder to be trained
end-to-end regardless of backbone quality (which would be a strong, useful negative result FOR TanitAD's
own from-scratch-ViT design choice).
*Pain points:* P5 directly — this is the cheapest possible re-test of the "does a frozen foundation
encoder work" question.
*Admissibility:* admissible.
*Cost:* ~2-3 eng-days (swap-in, DINOv3 weights are public); ~1-2 A40-days for a short training-head-only run.
*Experiment:* repeat the exact frozen-encoder arm protocol that produced the 2.17 m DINOv2 result, but with
frozen DINOv3-ViT-B/L in place of DINOv2/I-JEPA, same heads, same 30k-step budget (or a matched-compute
shorter budget with the exponent-citation rule applied if shortened). **A:** DINOv3 closes most of the gap
to 0.452 m → frozen foundation encoders were never the wrong idea, DINOv2/I-JEPA specifically was; revisit
P5 as a live option. **B:** DINOv3 also fails at a similar magnitude → strengthens the case that
driving-specific, jointly-trained representations are load-bearing for TanitAD's regime, independent of
backbone quality — a validated architectural decision, worth stating plainly rather than re-litigating P5
again later.

**B5. V-JEPA 2 — the closest non-driving cousin, with a striking small-data number.**
Meta's video joint-embedding predictive architecture; an action-conditioned variant is post-trained on a
tiny amount of robot video and deployed zero-shot.
*Evidence:* PUBLISHED (Meta, arXiv:2506.09985, June 2025) — V-JEPA2-AC post-trained on **<62 hours** of
unlabeled DROID robot video → zero-shot pick-and-place on Franka arms in two NEW labs, **no task-specific
training, no reward, no data from the target environments**. **Maturity: PROVEN** (robotics domain, not
driving — flag the domain gap explicitly).
*Pain points:* P4 — a concrete existence proof that a JEPA-family world model can be usefully post-trained
on an amount of data smaller than TanitAD's own 13-hour corpus and still generalize zero-shot to new
physical setups; directly encouraging for the "small-corpus is enough IF the objective is right" thesis
that A4/G6 also point at.
*Admissibility:* admissible (self-supervised video + action post-training, no privileged inference signal).
*Cost:* 0 GPU-days to read; this is cited for its EVIDENTIAL value, not as a directly-portable recipe (see
B6/B7 for the driving-specific descendants that actually port it).
*Experiment:* none standalone — see B6, B7.

**B6. WA-JEPA — the closest published cousin of TanitAD's own architecture.**
Rebuilds V-JEPA specifically for driving: replaces V-JEPA's random spatiotemporal masking with
HYBRID FUTURE-MASKED pretraining (infer future latents from context, closer to what a deployed planner
actually needs), and replaces deterministic future-latent regression with CONDITIONAL FLOW MATCHING,
jointly modeling future scene dynamics and actions in one latent space.
*Evidence:* PUBLISHED (arXiv:2608.20974, Aug 2026) — **91.7 EPDMS on NAVSIM-v2 navtest**, **0.4462 HD-Score**
on 436 HUGSIM closed-loop scenarios; strong open-loop AND zero-shot closed-loop generalization reported
together (relevant to our own never-ADE-alone / open-vs-closed-loop hygiene rules). **Maturity: PROMISING**
(single paper, Aug 2026, no independent replication).
*Pain points:* P4, P6 (closed-loop compounding — flow-matching future latents is argued to generate more
plausible, less mode-collapsed futures than deterministic regression, which bears directly on compounding
error), P7 (imagination quality — this is architecturally the most direct external comparison point for
TanitAD's own operative predictor + imagination module).
*Admissibility:* admissible.
*Cost:* HIGH to fully replicate (a new pretraining recipe); LOW-MEDIUM (~5-10 eng-days) to borrow just the
two design deltas — future-masked (not random-masked) pretraining, and flow-matching (not deterministic
regression) for the predictor's output — as ablations on the existing flagship architecture; ~4-6 A40-days.
*Experiment:* swap the flagship's deterministic latent-rollout predictor for a flow-matching predictor head
(keeping everything else fixed), matched-compute, and measure ADE + the Kinematic-Consistency-Error
diagnostic (§G6). **A:** flow-matching reduces both ADE and kinematic-inconsistency at matched compute →
adopt; this is a concrete, scoped architecture change with two independent external papers (WA-JEPA here,
DriveWAM already in our own H7 notes) pointing the same direction. **B:** no change or regression →
flow-matching's benefit may be specific to WA-JEPA's discrete-token/larger-model regime and not transfer to
our sub-300M continuous-latent setting — a useful negative result bounding when flow-matching helps.

**B7. LFG ("Learning to Drive is a Free Gift") — the top finding of this stream.**
Fully label-free, teacher-guided pretraining from UNPOSED, in-the-wild YouTube driving video, using a
single MONOCULAR FRONT-FACING camera — matching TanitAD's own front-wide-only sensor choice exactly.
Jointly predicts point-maps, camera poses, semantic segmentation and motion masks to learn a unified
geometry+semantic+motion "pseudo-4D" representation, with no poses, labels, or LiDAR required for
pretraining.
*Evidence:* PUBLISHED (arXiv:2602.22091, Feb 2026, CVPR 2026) — on NAVSIM, **outperforms multi-camera +
LiDAR BEV methods (85.2 PDMS) using only a monocular front camera**, and reports **81.4 PDMS with only 10%
labels** — the paper's own framing is explicitly a data-efficiency result, and its authors' released
comparison shows it "consistently strongest at 1% and 10% labels, remaining competitive at 100%." **Labelled-
data multiplier: effectively >>10× — most of full-label performance is recovered at 1/10th the labels, on
a single camera.** **Maturity: PROMISING→PROVEN-directionally** (CVPR 2026 accepted, single-lab result, no
independent replication yet, but the sensor configuration match to TanitAD is exact and the numbers are
unusually strong).
*Pain points:* P4 (small data) — this is the single most direct, best-matched external answer to
TanitAD's exact problem (one camera, small labelled corpus) found in this entire stream. P5 (untested
video-pretrained encoders) — LFG IS a video-pretrained encoder recipe, unlike the frozen-DINOv2 arm that
already failed.
*Admissibility:* admissible — label-free, vision-only pretraining; the pretrained representation itself
carries no privileged signal, and downstream fine-tuning stays on our own labels.
*Cost:* ~10-15 eng-days to reproduce or adapt the pretraining recipe (multi-task pseudo-4D objective is
nontrivial engineering); ESTIMATED 10-20 A40-days for a from-scratch pretrain on a modest YouTube-front-cam
subset, OR near-zero if their released weights (if public) can be probed/fine-tuned directly first.
*Experiment (the single highest-priority experiment in this whole stream):* check whether LFG's authors
released pretrained weights; if yes, LINEAR-PROBE them directly on our val set (near-zero GPU cost, ~2
eng-days) before committing to any reproduction. **A:** the probe beats or matches our from-scratch flagship
encoder on ADE → strong signal to invest in full reproduction/fine-tuning as the primary P4/P5 lever,
ahead of collecting more raw hours. **B:** the probe underperforms → check whether this is a domain gap
(YouTube dashcams vs. PhysicalAI-AV rig — see §C for the mechanistic fix) before concluding the paradigm
doesn't transfer; re-test after applying §C's canonicalization fix, since LFG's own front-camera-only
framing suggests camera geometry sensitivity is exactly the variable in play.

## C. Pseudo-labelling unlabelled video with an inverse-dynamics model — camera-rig transfer

**This is the highest-priority section of this stream**, because it is the one place where TanitAD's own
prior work is confirmed to have failed (per the orchestrator fact sheet: "the programme's own
inverse-dynamics (IDM) pseudo-labelling line FAILED held-out-camera-rig transfer"), and the literature now
gives at least three independent, concrete, cheap-to-test mechanisms for why, each with its own fix.

**C1. Focal-length / canonical-camera-space normalization.**
Resize every training AND inference frame so the EFFECTIVE FOCAL LENGTH is the same fictive value (~1000
px), removing rig-specific pixel↔metric mapping as something the network has to memorize per-rig.
*Evidence:* PUBLISHED (Meta, "VLM³: Vision Language Models Are Native 3D Learners," arXiv:2605.30561, 28
May 2026 — **independently confirmed this pass**, matching our own internal doc's citation exactly: depth
accuracy improves 0.84→0.90 purely from focal-length unification, no extra 3D modules needed). The same
family goes back to Metric3D / Metric3D v2 (arXiv:2307.10984 / 2404.15506) via a "Canonical Camera Space
Transformation Module," and is echoed in a 2026 VLM depth-estimation paper that explicitly "rescales all
images to a unified focal length to remove dataset-specific biases." **Maturity: PROVEN** (three
independent papers over three years converge on the same mechanism).
*Pain points:* **P5/the IDM-camera-rig failure, mechanistically:** if our IDM was trained on frames at
PhysicalAI-AV's native focal length and never canonicalized, it likely learned a pixel-to-metric mapping
specific to that rig's intrinsics, which is exactly what breaks under a new rig (YouTube dashcams, a
different vehicle's front-wide camera, etc.).
*Admissibility:* fully admissible — a preprocessing transform, no privileged signal.
*Cost:* LOW — ~3-5 eng-days to add a focal-length-canonicalizing resize/crop step ahead of the existing IDM
head (intrinsics are known for PhysicalAI-AV and can be estimated for YouTube video via any of §C3's VO
methods); ~1-2 A40-days to re-run the existing IDM training with this one change.
*Experiment:* re-run the IDM's ALREADY-FAILED held-out-camera-rig evaluation with focal-length
canonicalization added, nothing else changed. **A:** held-out-rig error drops materially (report with the
same protocol/metric the original failure was measured on, so the comparison is apples-to-apples) → the
missing canonicalization step was the root cause, ship it and re-open the YouTube-pseudo-labelling line.
**B:** no material change → canonicalization alone isn't sufficient, move to C2's complementary mechanism
before concluding the whole IDM-pretraining line is dead.

**C2. Rig-aware conditioning (the complementary strategy).**
Instead of normalizing rig differences away, explicitly CONDITION the model on rig metadata (camera ID,
capture time, rig pose) so it builds a rig-aware latent space that stays robust even with missing metadata,
and can INFER rig structure when it isn't given.
*Evidence:* PUBLISHED (Li, Kachana et al., "Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction,"
arXiv:2506.02265, June 2025) — outperforms prior methods by **17-45% mAA** on 3D reconstruction, camera
pose estimation, and rig discovery, evaluated ACROSS Argoverse, Waymo, and nuScenes — i.e. specifically
tested for cross-rig generalization, the exact axis our IDM failed on. **Maturity: PROVEN** (multi-dataset
cross-rig evaluation, single forward pass, no iterative refinement needed).
*Pain points:* P5/IDM-camera-rig-failure — the complementary mechanism to C1: if canonicalization alone
under-performs, adding explicit rig conditioning (even coarse: "this clip's rig has FOV=X, mount-height=Y")
as an auxiliary input to the IDM head is a second, independent lever, and the two are not mutually
exclusive (Rig3R's ablations show conditioning helps even alongside geometric normalization elsewhere in
the pipeline).
*Admissibility:* admissible — rig metadata is either known (our own corpus) or estimable per-clip (YouTube),
not a privileged runtime signal in the deployment sense (it is a property of the SENSOR, not the SCENE).
*Cost:* ~5-8 eng-days (needs a small conditioning-embedding addition to the IDM architecture); ~2-3 A40-days.
*Experiment:* add a learned rig-embedding (from known or estimated intrinsics/mount pose) as a conditioning
input to the IDM head, re-run the same held-out-rig evaluation as C1 (as a combined C1+C2 arm, and C2-alone,
to separate the two effects). **A:** C2 alone or C1+C2 together closes the held-out-rig gap → adopt the
combination; this also produces a clean, reportable "which mechanism mattered" result. **B:** neither helps
→ the failure is not intrinsics/rig-geometry at all, redirect suspicion to C5's third hypothesis (evaluation
protocol) before concluding the paradigm is unfixable.

**C3. Cross-domain visual odometry as a training/evaluation recipe.**
Two methods purpose-built for the EXACT problem of "train on one camera setup, deploy zero-shot on others,"
using explicitly held-out-region/held-out-dataset protocols rather than train/test splits from the same rig.
*Evidence:* PUBLISHED (XVO, arXiv:2309.16772, ICCV 2023 — cross-modal self-training on unconstrained YouTube
dashcam video, off-the-shelf transfer across KITTI/nuScenes/Argoverse2 WITHOUT fine-tuning, using a
region-held-out protocol — train on one nuScenes region, eval on the REST plus other datasets entirely,
specifically to test generalization rather than memorization); PUBLISHED (ZeroVO, arXiv:2506.08005, CVPR
2025 — calibration-free, geometry-aware; **>30% improvement over prior methods** on KITTI, nuScenes,
Argoverse2, AND a synthetic GTA dataset). **Maturity: PROVEN** for both.
*Pain points:* P5 — beyond the architectural fixes in C1/C2, this is an EVALUATION-METHODOLOGY lesson: XVO's
region-held-out protocol is the right template for testing our own IDM's rig-generalization claim BEFORE
deployment, rather than discovering the failure after the fact (which per the fact sheet is what happened).
*Admissibility:* admissible.
*Cost:* ~3-5 eng-days to adopt the held-out-region/held-out-rig evaluation protocol as a standing gate for
any future IDM training, ahead of any architecture change; near-zero incremental GPU cost (reuses existing
checkpoints for the eval).
*Experiment:* audit whether the ORIGINAL IDM camera-rig-transfer failure was discovered via a proper
held-out-rig protocol (XVO-style) or via same-rig train/test that only later met a genuinely new rig in
deployment. **A:** the original eval already used a proper held-out protocol → the failure is architectural
(C1/C2 apply directly). **B:** the original eval did NOT hold out rig properly → the failure may be
partially or wholly an EVALUATION gap, not a modelling gap — a distinct and important finding in its own
right (the failure was invisible, not inevitable), and the fix is procedural (adopt this protocol going
forward) as much as architectural.

**C4. LA-Pose — latent-action pretraining repurposed for pose/rig estimation, at driving scale.**
Self-supervised pretraining on a large corpus of unlabeled driving video to learn LATENT ACTIONS, then uses
those as input features for camera pose estimation, with only minimal 3D-annotated post-training.
*Evidence:* PUBLISHED (CVPR 2026 accepted — "LA-Pose: Latent Action Pretraining Meets Pose Estimation")
— pretrained on **10 million unlabeled driving videos**, achieves **>10% higher pose accuracy** than
feed-forward SOTA (VGGT) on Waymo/PandaSet after minimal 3D-annotated fine-tuning. **Maturity: PROMISING**
(CVPR 2026, single paper, exact arXiv id not confirmed this pass — **UNVERIFIED id**, title and venue only).
*Pain points:* P5 — a direct existence proof that latent-action pretraining (the same family as our own IDM
approach, and as Genie/LAPA already in our internal H7 notes) DOES transfer to pose/rig-relevant tasks at
scale, provided the pretraining corpus is large and diverse (10M videos, plausibly spanning many rigs) —
suggesting our own IDM's failure may in part be a SCALE/DIVERSITY-of-pretraining-corpus issue, not only an
architectural one.
*Admissibility:* admissible.
*Cost:* not directly reproducible at our compute scale (10M-video pretraining is out of reach on a few
A40s); the actionable takeaway is the SHAPE of the recipe (latent-action pretrain → minimal supervised
post-training), not a literal reproduction. ~0 GPU-days beyond reading; cited for evidential value.
*Experiment:* none standalone — informs C1/C2's priority (favor architectural fixes we CAN afford over
"just pretrain on more video," since we cannot match 10M-video scale).

**C5. Synthesis — three testable hypotheses for our own failure, ranked by cost.**
Combining C1-C4: our IDM most likely failed held-out-rig transfer because it did (a) no focal-length
canonicalization, and/or (b) no explicit rig conditioning, and/or (c) was validated with a same-rig
train/test split that never actually exercised held-out-rig generalization until real deployment. All three
are cheap to test (LOW eng-days, ≤3 A40-days each) and not mutually exclusive; C3's audit (near-zero cost)
should run FIRST since its answer changes how to interpret C1/C2's results.
*Cost (combined):* ~10-15 eng-days, ~5-8 A40-days total for all three tests run in sequence.
*Cheapest discriminating experiment (the whole-section version):* run C3's audit, then C1, then C2, in that
order, on the exact held-out-rig benchmark that originally exposed the failure. Pre-registered outcomes:
**(i)** if C1 alone fixes most of the gap → ship canonicalization, treat rig-conditioning as a secondary
refinement; **(ii)** if C1+C2 together fix it but neither alone does → both mechanisms are load-bearing,
ship both; **(iii)** if neither fixes it → the failure is likely NOT camera-geometry at all (candidates:
insufficient pretraining diversity per C4, a genuinely different visual domain like YouTube compression
artifacts/lens distortion/HDR tone-mapping not modeled by intrinsics alone) and the IDM-pseudo-labelling
line should be RE-SCOPED (e.g. restrict pseudo-labelling to camera rigs similar enough to the training rig,
rather than claimed as fully rig-agnostic) rather than abandoned outright.

## D. Synthetic & generative data

**D1. Trajectory perturbation & recovery synthesis — ChauffeurNet's descendants.**
Synthesize off-nominal states (drift, near-collision, off-road) and the recovery back to nominal, so the
policy sees corrective behaviour it would otherwise only encounter via costly real-world DAgger loops.
*Evidence:* PUBLISHED (Bansal et al., "ChauffeurNet," arXiv:1812.03079, RSS 2019 — the founding method,
**confirmed** via its RSS proceedings page) — perturbs pose (translation/rotation/heading/speed), fits a
recovery line back to nominal, and demonstrably learns to nudge around parked vehicles / recover from
deviation. A 2026 descendant applies the identical idea to LATENT WORLD MODELS instead of pose space
directly: search-synthesized evidence (title/arXiv id not independently pinned this pass — **UNVERIFIED**
exact citation) describes injecting Gaussian noise into a world model's CONTEXT at a scale matching its own
autoregressive-rollout error distribution, training it to predict well FROM degraded/perturbed contexts,
which measurably improves autoregressive-rollout robustness. **Maturity: PROVEN** (ChauffeurNet, pose-space,
long-established); **PROMISING** (the latent-space generalization, 2026, exact source unverified).
*Pain points:* **P6 (closed-loop compounding error) directly** — this is the textbook fix for exactly that
problem, and it is currently ABSENT from TanitAD's training recipe as far as this stream's delta-skim of
the hub docs shows (the hub's own "imagination-error" monitoring is a DETECTION mechanism, not obviously a
TRAINING-TIME recovery-synthesis mechanism — worth a direct check against `stack/` by whichever stream owns
architecture).
*Admissibility:* fully admissible — synthetic perturbation of our own already-licensed trajectories/latents,
no new data source, no privileged inference signal.
*Cost:* LOW — ~5-7 eng-days to implement latent-space noise injection scaled to the model's own measured
rollout error; ~3-5 A40-days to test.
*Experiment:* during training, inject noise into the operative predictor's latent context at a scale
matched to the CURRENT model's own measured k-step rollout error (self-referential, no extra labels needed),
and measure whether closed-loop-style multi-step rollout ADE improves relative to an unperturbed-context
control at matched compute. **A:** rollout error decays more slowly / plateaus higher-quality with
perturbation training → a direct, cheap fix for P6, ship it. **B:** no change → compounding error in our
setup may be dominated by something perturbation training doesn't address (e.g. genuinely novel states
outside the perturbation's support), pointing back at G6's kinematic-vs-dynamic diagnosis for a sharper
root cause.

**D2. Cosmos-Drive-Dreams — NVIDIA's synthetic driving data pipeline.**
A world-foundation-model-powered pipeline that turns real driving logs (with HDMap/BBox/LiDAR labels) into
large volumes of controllable, spatiotemporally consistent synthetic video across weather/lighting variants.
*Evidence:* PUBLISHED (NVIDIA, arXiv:2506.09042, June 2025) — **5,843 real 10-second clips → 81,802
synthetic clips** (121 frames each), covering rain/snow/fog/night; consistently improves downstream
perception AND policy learning, and — the more important claim — **gains PERSIST even when augmenting
LARGER real datasets**, i.e. this is not merely a small-data patch but a durable multiplier. Already CC-BY-
4.0 and already listed in our own OWN_DATASET_PLAN as part of the license-clean core. **Maturity: PROVEN**
(NVIDIA-authored, dataset+weights open-sourced, HF dataset card exists).
*Pain points:* P4 (small data, specifically weather/lighting/night diversity our 13-hour corpus likely
under-samples), P1 (selection — synthetic weather variants let us CHOOSE the long-tail slice to fill rather
than waiting to collect it).
*Admissibility:* fully admissible; already cleared by DataEng's own licensing pass.
*Cost:* mostly a DOWNLOAD-and-integrate cost, not a generation cost, since NVIDIA already published the
81,802-clip dataset — ESTIMATED ~3-5 eng-days to integrate into the existing cache pipeline, ~2-4 A40-days
to re-train/fine-tune with it mixed in at a modest ratio.
*Experiment:* add a 10-20% Cosmos-Drive-Dreams mixture (weather/night slices specifically) to a training arm,
matched-compute against the parity-only baseline, evaluate on any night/weather-stratified slice of val (if
one exists) plus overall ADE. **A:** night/weather-stratified metrics improve without overall regression →
adopt as a standing corpus supplement. **B:** no improvement → either our val set lacks enough night/weather
windows to detect the effect (a val-set gap, itself worth flagging) or the synthetic-to-real domain gap
matters more for our small-model regime than NVIDIA's larger-model results suggest.

**D3. Cosmos-Transfer — multi-modal-conditioned sim-to-real.**
The sibling pipeline to Cosmos-Drive-Dreams: takes structured conditioning (maps, boxes, simulator output)
and generates photorealistic video conditioned on it, i.e. a route to turning CHEAP simulator states
(CARLA, MetaDrive) into photorealistic training frames despite our pods being unable to RENDER CARLA
directly (a constraint already logged in our own DATA_STRATEGY_FOR_HIERARCHY.md).
*Evidence:* referenced across the Cosmos-Drive-Dreams ecosystem search results but **not independently
opened this pass — UNVERIFIED at the level of a specific number**; treated here as PUBLISHED-by-association
(same NVIDIA Cosmos family, same paper's broader toolkit) but flagged for direct verification before citing
a specific gain figure. **Maturity: PROMISING** pending that verification.
*Pain points:* P4, and specifically the CARLA-rendering blocker already on record (no Vulkan/EGL on our
pods) — if Cosmos-Transfer can take CARLA's STATE (not its rendered pixels) and produce photorealistic
video, it may be a way around that exact blocker without needing a graphics-capable pod.
*Admissibility:* admissible in principle (synthetic, training-time only); needs the same licensing check as
D2.
*Cost:* UNKNOWN pending verification of the actual pipeline requirements; ESTIMATED 5-10 eng-days to
evaluate feasibility alone.
*Experiment:* a scoping spike (not a training experiment) — verify Cosmos-Transfer's exact input contract
(does it need CARLA's rendered frames, or just its structured state?) directly from primary sources before
committing engineering time. **A:** state-only conditioning is sufficient → this durably resolves the
CARLA-rendering blocker, high priority. **B:** it still needs rendered frames as conditioning → the
blocker persists, deprioritize until a graphics-capable pod exists (as already planned in DATA_STRATEGY).

**D4. Instant NuRec — feed-forward 3D Gaussian reconstruction for closed-loop re-simulation.**
Turns a short multi-view driving log into a fully simulatable 3D Gaussian Splat world in a single forward
pass (~1.5s for a 10-20s scene), integrating with NVIDIA's AlpaSim for closed-loop simulation.
*Evidence:* PUBLISHED (NVIDIA, arXiv:2607.14203, July 2026) — **PSNR +2.01 dB above the strongest evaluated
baseline** on Waymo Open Dataset; single forward pass, no per-scene optimization. **Maturity: PROVEN**
(NVIDIA Research + Omniverse product docs + open GitHub repo all corroborate).
*Pain points:* P3 (hierarchy has no route/goal signal) and P10 (no external benchmark) — our own fact
sheet already flags AlpaSim as a candidate strategic-topology source; Instant NuRec is the concrete
RECONSTRUCTION step that makes our OWN PhysicalAI-AV logs usable as AlpaSim-compatible closed-loop scenes,
which is a route to closed-loop evaluation without needing CARLA rendering at all.
*Admissibility:* admissible — reconstruction of our own already-licensed logs, training/eval-time only.
*Cost:* ESTIMATED 10-15 eng-days to stand up the reconstruction+AlpaSim integration; GPU cost is mostly
inference-time reconstruction (cheap per clip) rather than training, ESTIMATED 2-5 A40-days for a pilot
batch of clips.
*Experiment:* reconstruct a handful of parity-corpus clips with Instant NuRec, run the existing flagship
through AlpaSim closed-loop on them, and check whether the closed-loop metric ordering (flagship vs.
constant-velocity vs. REF-C) matches the already-known open-loop ordering. **A:** orderings match →
closed-loop evaluation is now available cheaply, a direct answer to P10 (no external benchmark) and a hedge
against the open-loop/closed-loop divergence our own hub docs already flag as a standing risk (L2,
2605.00066). **B:** orderings diverge → exactly the kind of finding our own "never rank on ADE alone" rule
exists to catch — high value either way.

**D5. World Engine — see §A5.**
Already covered in full under Data Scaling Laws (§A5) since its primary claim is about the SCALING
implication of post-training vs. collection; cross-referenced here because it is equally a SYNTHETIC-DATA
method (log reconstruction → safety-critical extrapolation). No separate scoring to avoid double-counting
in the TOP-10 below.


## E. Curation & active learning

**The central admissibility issue for this whole section:** classic "data engines" (Tesla, Waymo) are
FLEET flywheels — TanitAD has no deployed fleet, so the mechanism, not just the scale, doesn't transfer.
The substitutes are corpus-internal (dedup, curated mixtures) or reuse our own already-designed
surprise-gated write mechanism (H10) over a fixed offline corpus rather than live miles.

**E1. SemDeDup — semantic deduplication.**
Uses pretrained embeddings to find and remove near-duplicate (not exact-duplicate) samples.
*Evidence:* PUBLISHED (arXiv:2303.09540) — removes **50% of a LAION subset with minimal performance loss**,
roughly halving training time, with OOD performance actually IMPROVING. **Maturity: PROVEN** in
vision-language web-scale data; **UNVERIFIED/domain gap** for driving video/trajectory data specifically —
no driving-specific replication found this pass, flagged rather than assumed to transfer.
*Pain points:* P4 indirectly (frees compute/eng-time by cutting redundant highway-cruise-style windows,
which are likely over-represented in any dashcam-style corpus) and P9 (few A40s — less redundant data means
more effective passes per GPU-day).
*Admissibility:* fully admissible — an internal filtering step on already-licensed data.
*Cost:* ~3-5 eng-days to embed and cluster the existing 2,376-episode corpus (or any new-source candidate)
and identify near-duplicate windows; 0 additional GPU-days (a preprocessing step, arguably SAVES GPU-days).
*Experiment:* embed all parity-corpus windows with the existing trained encoder, run SemDeDup-style
semantic clustering, and check what fraction are near-duplicates. **A:** a meaningful fraction (>15-20%) are
near-duplicates → re-train a matched-compute arm on the deduplicated set and check whether ADE holds or
improves at fewer effective steps — if so, this is a free compute-efficiency win with no new data needed.
**B:** the corpus is already low-redundancy (plausible — it's curated, not raw fleet dump) → dedup has
little to offer here, but the SAME analysis becomes a useful pre-ingest filter for any NEW source (L2D, ZOD)
before it's added, which is where redundancy is more likely to matter.

**E2. MOSAIC — cross-referenced from §A3.**
Already scored under Data Scaling Laws since it is fundamentally a scaling-law-driven MIXTURE optimizer;
equally an active-curation method (it decides not just how much data but which domain to add next). See §A3
for the full six-field entry; **no separate scoring here** to avoid double-counting in the TOP-10.

**E3. Fleet-scale data engines — Tesla shadow mode, Waymo's outer learning loop.**
Tesla: onboard anomaly detection (perception mismatches, unexpected planner deviations) triggers upload of
rare events from a deployed fleet. Waymo: a "Critic" model automatically flags suboptimal behaviour from
autonomous miles and feeds edge cases back into closed-loop RL.
*Evidence:* PUBLISHED-by-report (secondary sources; exact papers not independently opened this pass —
**UNVERIFIED at the primary-source level**, numbers as reported: Tesla's data engine cited as having
ingested **>9 billion miles** by early 2026 targeting a 10-billion-mile threshold; Waymo's Critic loop
cited as operating over **>100 million autonomous miles**). **Maturity: PROVEN operationally at those two
companies**, but the evidence for the SPECIFIC numbers here is secondary, not a primary paper read directly.
*Pain points:* directly illustrates the SCALE at which fleet data engines operate — useful as a calibration
point for how far TanitAD's problem is from that regime, not as a directly portable technique.
*Admissibility:* **not directly admissible as a mechanism** — requires a deployed fleet TanitAD does not
have. The TRANSFERABLE lesson is narrower: an UNCERTAINTY/SURPRISE-triggered flagging mechanism, applied not
to live fleet miles but to (a) our own model's imagination-error signal over the FIXED offline corpus
(already H10 in our own hub docs) and (b) any shadow/replay logs from closed-loop simulation (§D4) once
that exists.
*Cost:* N/A as a fleet mechanism; the H10 imagination-error-write mechanism is already scoped in our own
docs with its own cost estimate.
*Experiment:* N/A — this entry exists to make the SCOPING GAP explicit (fleet-scale mechanisms don't apply
here) rather than to propose a new experiment; the actionable item is H10's own pre-registration, not a new
one.

**E4. Trajectory-entropy-based data pruning.**
Prunes large-scale AV datasets by maximizing retained trajectory entropy (diversity of motion patterns)
rather than random or uniform sub-sampling.
*Evidence:* PUBLISHED (title: "Are All Data Necessary? Efficient Data Pruning for Large-scale Autonomous
Driving Dataset via Trajectory Entropy Maximization," arXiv:2512.19270) — found via direct search of
driving-specific pruning literature; exact retained-fraction/performance numbers not independently opened
this pass — **UNVERIFIED at the number level**, title/arXiv id/mechanism confirmed. **Maturity: PROMISING**
pending that verification.
*Pain points:* P1 (selection) directly — a concrete, driving-native alternative/complement to SemDeDup's
embedding-based dedup, using MOTION diversity specifically rather than visual-semantic similarity, which
may catch a different kind of redundancy (e.g. visually distinct but kinematically repetitive highway
cruising).
*Admissibility:* fully admissible.
*Cost:* ~3-5 eng-days to implement a trajectory-entropy scoring pass over the parity corpus.
*Experiment:* compute trajectory-entropy scores over the 2,376 parity episodes, compare the ranking to
SemDeDup's (E1) semantic-redundancy ranking — do they flag the SAME windows as low-value, or different ones?
**A:** they overlap heavily → one method suffices, prefer the cheaper (SemDeDup, reuses the existing
encoder). **B:** they flag different windows → the two are complementary (visual vs. kinematic redundancy
are different things), and a combined filter is worth the extra eng-time.

**E5. TAROT — targeted data selection via optimal transport.**
A general-purpose data-selection method that matches a training distribution to a target distribution via
optimal transport, rather than heuristic scoring.
*Evidence:* PUBLISHED (arXiv:2412.00420) — a general ML method, not driving-specific; found in the same
search cluster as MOSAIC and the trajectory-entropy paper, exact numbers not independently opened this pass
— **UNVERIFIED** beyond title/mechanism/id. **Maturity: PROMISING**, general-domain.
*Pain points:* P1 — a candidate alternative to MOSAIC for the specific sub-problem of "which of our val-like
target scenarios are under-represented in the training mixture," which OT-based matching is well-suited to.
*Admissibility:* fully admissible.
*Cost:* ~5 eng-days to adapt (OT-based selection needs a feature embedding + a target-distribution
definition, e.g. "the val-episode distribution" or "the lead-present subpopulation" from the falsified
lead-state gate's own stratification).
*Experiment:* lower priority than E1/A3 given it targets the same problem with more implementation
complexity; recommend only if MOSAIC (A3) under-delivers on its own discriminating experiment.

## F. Distillation & privileged teachers

Privileged-information distillation is explicitly ADMISSIBLE per our own binding rules: ground-truth /
label derivation may use anything (ego, other agents, maps, future poses), and a teacher trained on
privileged information whose DISTILLED OUTPUT (not its privileged inputs) supervises the vision-only student
is a label-time use, not an inference-time leak — provided the same discipline as the situation-classifier
rule is applied: the student's INFERENCE inputs must never include the privileged channel or a
classifier-derived echo of it.

**F1. Learning by Cheating — the founding teacher-student pattern.**
A privileged agent (ground-truth layout + traffic participants) is trained first, then acts as a teacher
for a purely vision-based student that never sees privileged state.
*Evidence:* PUBLISHED (Chen, Zhou, Koltun, Krähenbühl, arXiv:1912.12294, CoRL 2019) — substantially
outperforms prior SOTA on the CARLA and NoCrash benchmarks. **Maturity: PROVEN**, foundational.
*Pain points:* P1 (selection) — the general pattern this whole section instantiates: a privileged teacher
sees the "right answer" more easily and passes DISTILLED judgment, not raw privileged state, to the student.
*Admissibility:* admissible under our own rule as stated above.
*Cost:* the pattern itself is near-free to adopt structurally; cost is dominated by whichever specific
teacher (F2-F6) is chosen.
*Experiment:* superseded by the more modern, driving-planning-specific instantiations below (F4-F6).

**F2. Roach — an RL teacher, not just a privileged-BC teacher.**
Trains a reinforcement-learning expert on bird's-eye-view state to a NEW CARLA performance ceiling, then
uses it as the imitation target/supervision source for a vision-based student.
*Evidence:* PUBLISHED (Zhang, Liniger, Dai, Yu, Van Gool, arXiv:2108.08265, ICCV 2021) — the RL coach sets a
new CARLA performance upper bound; the resulting E2E student achieves **78% success on NoCrash-dense** while
generalizing to a new town and new weather. **Maturity: PROVEN.**
*Pain points:* P1, and P8 (no safety layer) indirectly — an RL-trained privileged teacher that has already
learned to avoid collisions under full state access is a natural SOURCE of safety-relevant supervision for
a student that cannot see full state.
*Admissibility:* admissible (teacher trained with privileged simulator state at LABEL/TRAINING time only).
*Cost:* HIGH if training the RL teacher from scratch in CARLA (CARLA rendering is blocked on our pods per
our own DATA_STRATEGY doc); LOW-MEDIUM if a pretrained Roach-style checkpoint or its published rollouts can
be reused directly as supervision without needing to render CARLA ourselves.
*Experiment:* lower near-term priority than F4/F5 given the CARLA-rendering blocker; revisit once/if D3's
Cosmos-Transfer scoping resolves that blocker.

**F3. PlanT — object-level, explainable planning transformer teacher.**
Uses a compact OBJECT-LEVEL (not pixel/BEV-grid) input representation for a privileged planning transformer,
distilled to a camera-based student.
*Evidence:* PUBLISHED (Renz et al., arXiv:2210.14222, CoRL 2022) — matches the CARLA expert's driving score
on the Longest6 benchmark while being **5.3× faster at inference** than equivalent pixel-based planning
baselines, with attention weights that identify the most relevant objects (an explainability by-product).
**Maturity: PROVEN.**
*Pain points:* P1, P3 (hierarchy/tactical decisions) — PlanT's object-level representation is a natural fit
for TanitAD's `obstacle.offline` tracks, which ARE an object-level representation already sitting
license-clean but under-used on the parity corpus.
*Admissibility:* admissible.
*Cost:* ~8-12 eng-days to build a PlanT-style object-level privileged planner head fed by `obstacle.offline`,
used ONLY to generate distillation TARGETS (e.g. a soft trajectory-scoring signal) for the existing
vision-only tactical head, never to feed the deployed student at inference; ~3-5 A40-days.
*Experiment:* train a small PlanT-style object-level scorer on `obstacle.offline` + ego state (privileged,
training-only), use its output trajectory SCORES as an auxiliary distillation target for the existing
tactical head (vision-only at inference), and check tactical-decision-quality metrics (per our own
binding four-metric-family rule) against the undistilled baseline. **A:** tactical metrics improve → a
concrete, admissible use of our already-available-but-under-ingested agent tracks, distinct from (and not
contradicted by) the already-falsified DIRECT lead-state input finding, since this is a training-time
distillation target, not an inference-time feature. **B:** no improvement → strengthens the existing
finding that agent state doesn't easily help THIS corpus/task combination, now tested via a second,
independent mechanism.

**F4. Hydra-MDP / Hydra-MDP++ — multi-teacher rule-based + human distillation.**
A multi-head decoder learns diverse trajectory candidates, each supervised by a DIFFERENT teacher: human
demonstration AND multiple rule-based simulators (collision, traffic-light compliance, lane-keeping,
comfort), so the student learns to satisfy metrics no single human-imitation loss would capture.
*Evidence:* PUBLISHED (Hydra-MDP, arXiv:2406.06978 — 1st place, NAVSIM challenge; Hydra-MDP++,
arXiv:2503.12820 — adds traffic-light/lane-keeping/comfort teachers) — **91.0% drive score on NAVSIM**
reported for Hydra-MDP++ via image-encoder scaling. **Maturity: PROVEN** (competition-winning, widely cited).
*Pain points:* **P2 (longitudinal), P3 (tactical/strategic decision quality) directly** — rule-based
teachers are EXACTLY the mechanism to inject "what good distance-keeping/lane-keeping looks like" as a
training signal without needing more human-labelled (steer, accel) pairs; this is a strong, concrete
candidate for closing the 88.7%-of-oracle-gap longitudinal problem our own fact sheet names as P2.
*Admissibility:* admissible — rule-based teachers are DERIVED, computable quantities (TTC, lane-keeping
error, comfort jerk), not privileged sensor channels; they can be computed from our own future-pose
ground truth at label time and never touch inference.
*Cost:* ~10-15 eng-days to implement 3-4 rule-based teacher scores (collision, TTC/headway, lane-keeping,
comfort) as auxiliary distillation targets alongside the existing trajectory head; ~5-8 A40-days.
*Experiment:* add a Hydra-MDP-style multi-teacher auxiliary loss (TTC/headway teacher specifically, since
that's our named longitudinal weak point) to the existing flagship's trajectory head, matched-compute
against the current single-teacher (human-imitation-only) baseline, and evaluate on the LONGITUDINAL metric
family (target-speed accuracy, headway/TTC — per our own binding four-family eval rule). **A:** longitudinal
metrics improve without ADE regression → directly actionable against P2, our single largest named pain
point. **B:** no improvement → the longitudinal gap may not be a SUPERVISION-signal problem (i.e. the model
isn't lacking the right training target) but a REPRESENTATION or ARCHITECTURE problem — redirects effort
toward B/C's representation-learning levers instead.

**F5. CaRL — scaling reinforcement learning with a single simple reward.**
Shows that PPO fails to scale with complex shaped rewards at large batch sizes, and that a SINGLE reward
(route completion) scales cleanly to very large sample counts on modest hardware.
*Evidence:* PUBLISHED (Jaeger, Dauner, Beißwenger, Gerstenecker, Chitta, Geiger — CoRL 2025, PMLR 305,
arXiv:2504.17838) — scales PPO to **300M samples in CARLA and 500M samples in nuPlan on a SINGLE 8-GPU
node**; achieves **64 DS on CARLA longest6 v2**, outperforming more-complex-reward RL by a large margin.
**Maturity: PROVEN.**
*Pain points:* P9 (few A40s) — the headline result is precisely that a simple-reward RL recipe needs LESS
engineering/tuning and scales on modest hardware, which is directly relevant to a few-A40 programme; P8 (no
safety layer) — RL against a route-completion-plus-no-collision reward is itself a route to a supervision
signal that never appears in imitation-learning-only data.
*Admissibility:* admissible — RL reward is a DERIVED simulator/replay quantity computed at training time,
not a privileged inference input, PROVIDED it is computed in a replay/simulation setting (our own CARLA
rendering is blocked on-pod, but nuPlan-style replay against recorded logs does not require rendering).
*Cost:* HIGH relative to imitation-learning fine-tuning (RL is sample-hungry even at CaRL's improved
efficiency); ESTIMATED 15-20 A40-days for even a scaled-down pilot given TanitAD's much smaller compute
budget than CaRL's "single 8-GPU node" figure implies for OUR node count.
*Experiment:* lower near-term priority given cost; the more actionable takeaway is the REWARD-DESIGN lesson
(prefer one simple, well-behaved reward over many shaped terms) if/when TanitAD's own RMFM rule-injection
work (already Phase-1-flagship per our internal H9 notes) runs into PPO-scaling issues — a documented
pitfall to avoid pre-emptively rather than a new experiment to run now.

**F6. DiMA and the VLM→planner distillation family (Drive-KD, BucketKD, PlanKD).**
Distill a large multi-modal LLM's scene understanding into a small vision-only planner via auxiliary
surrogate tasks (masked reconstruction, future prediction, scene editing) and representation alignment
(KL-divergence between student/teacher hidden features); the LLM is DISCARDED at inference, leaving only the
efficient vision planner.
*Evidence:* PUBLISHED (DiMA, arXiv:2501.09757, CVPR 2025 — three surrogate tasks, shared scene encoder
doubling as MLLM tokenizer and planner feature extractor); PUBLISHED (Drive-KD, arXiv:2601.21288, Jan 2026
— multi-teacher VLM distillation); PUBLISHED (BucketKD, arXiv:2607.10565, July 2026 — safety-aware,
bucket-based KD for motion planning); PlanKD referenced via search synthesis as an information-bottleneck
planning-feature-distillation method with a safety-aware waypoint-attentive mechanism — **exact arXiv id
UNVERIFIED this pass**. **Maturity: PROVEN** (DiMA, peer-reviewed CVPR 2025); **PROMISING** (the 2026
successors, less independently checked).
*Pain points:* P3 (tactical/strategic — VLM world/traffic-rule knowledge is exactly the semantic layer our
hierarchy's strategic level currently lacks any external source for, per our own fact sheet's "no maps, no
route, no traffic-light labels" finding), and this is the SAME mechanism as §H's "knowledge injection" —
cross-referenced there, not double-scored.
*Admissibility:* admissible — the VLM teacher may use ANY signal (including text/world knowledge) at label/
distillation time; the deployed student remains vision-only, satisfying the vision-only-at-inference rule
by construction (there is no VLM at inference at all, let alone one carrying situation-classifier output).
*Cost:* HIGH to train a full VLM teacher from scratch (out of reach on a few A40s); LOW-MEDIUM (~8-12
eng-days, ~3-5 A40-days) to use an EXISTING open-weight VLM (no training) purely as a frozen annotator that
labels our own clips with traffic-semantic tags (e.g. "four-way stop," "protected left," "school zone")
offline, then distill THOSE labels into an auxiliary classification head on the existing encoder — a much
cheaper slice of the same idea.
*Experiment:* the cheap slice above: label a sample of parity-corpus clips with an off-the-shelf VLM's
traffic-scene-type judgment (offline, training-time only), add it as an auxiliary multi-task head, and check
whether the STRATEGIC metric family (per our binding four-family rule) improves. **A:** it does → a cheap,
admissible first step toward closing P3 (no route/goal signal is a bigger gap, but scene-type semantics is
adjacent and answers whether VLM knowledge injection helps AT ALL for our setup before committing to a full
DiMA-style joint-training pipeline). **B:** no improvement → either the strategic metric family itself needs
more instrumentation first (per our own binding rule 3: "a missing metric is a work item"), or traffic-
scene-type semantics specifically isn't the missing ingredient (route/goal likely matters more, per P3's own
framing) — informs prioritizing a goal-point predictor (already flagged admissible per our binding
goal-input rule) over VLM-semantic injection.

## G. Structural priors that substitute for data

**G1. Kinematic bicycle model as an output/decoder layer.**
Constrains the trajectory decoder's output to be dynamically feasible (bounded curvature-speed coupling)
rather than letting a free-form regression head output physically impossible paths.
*Evidence:* PUBLISHED (kinematic bicycle model consistency analysis for AV trajectory planning, IEEE — the
foundational formulation is decades old and well-established; TanitAD's OWN label parametrization,
`steer = atan(2.9·κ)`, is already exactly this family of prior). **Maturity: PROVEN**, and already partially
adopted internally.
*Pain points:* P2 (longitudinal), G-general (structural priors substitute for data by ruling out large
swaths of the output space that no amount of extra labelled data would otherwise teach the model to avoid).
*Admissibility:* fully admissible (a decoder architecture choice, not a data source).
*Cost:* if not already fully applied to EVERY output head (tactical waypoints, strategic goal points), the
incremental cost of extending it is LOW (~3-5 eng-days); this is closer to an audit item than a new
experiment.
*Experiment:* audit which of the four output heads (operative, tactical, strategic, plus any fallback) use a
kinematically-constrained decoder vs. a free regression head, and for any that don't, add the constraint and
matched-compute-compare. **A:** an unconstrained head shows measurable ADE or lateral-family (curvature
error, yaw-rate error — per our binding four-family rule) improvement once constrained → cheap, direct win.
**B:** all heads are already constrained, or constraining an already-good head shows no change → the prior
is already fully exploited here, redirect structural-prior effort to G2/G4 below.

**G2. SE(2) equivariance / mirror symmetry.**
Builds left-right mirror symmetry and rotation/translation invariance into the network architecture itself
(equivariant layers), rather than relying on data augmentation to teach the model the same invariance
empirically.
*Evidence:* PUBLISHED ("Pioneering SE(2)-Equivariant Trajectory Planning for Automated Driving,"
arXiv:2403.11304) — improves L2 distance at 3s by **20.6%** and surpasses SOTA **despite using only a small
split of the dataset**, i.e. the paper's own framing is explicitly a SAMPLE-EFFICIENCY result, not just an
accuracy one; explicitly contrasted against data augmentation, which "does not ensure equivariance and
requires longer training times." **Maturity: PROVEN** (peer-reviewed, quantified small-data result).
*Pain points:* **P4 directly, mechanistically the cleanest item in this whole section** — mirror-symmetry
equivariance means every left-turn example the model sees IS also a right-turn example for free, roughly
DOUBLING the effective diversity of manoeuvre-completion examples in a 2,376-episode corpus at zero
additional data cost.
*Admissibility:* fully admissible (an architecture/training-time symmetry, not a data source).
*Cost:* MEDIUM — ~8-12 eng-days to retrofit SE(2)/mirror equivariance into the existing ViT+predictor
stack (equivariant layers are a real architectural change, not a drop-in); ~3-5 A40-days to validate at
matched compute.
*Experiment:* the cheapest version is NOT full equivariant layers but a mirror-flip DATA AUGMENTATION
control first (flip every training frame + negate steer/yaw labels), to check whether the underlying
left/right-manoeuvre imbalance is even present and exploitable in our corpus, BEFORE investing in the
harder equivariant-architecture change. **A:** mirror-augmentation alone improves manoeuvre-class-balanced
tactical metrics → the imbalance is real and exploitable; escalate to full equivariant layers for the
(per the cited paper) additional gain augmentation alone can't reach. **B:** no improvement from simple
mirroring → either our corpus is already left-right balanced (less likely for real-world driving, but
possible for our specific route selection) or the tactical head isn't sensitive to this axis, in which case
skip the more expensive equivariant-architecture investment.

**G3. Equivariant Continuous Convolutions (ECCO) — a second, older equivariance result.**
An equivariant convolutional architecture for trajectory prediction that bakes in the same symmetry class as
G2 at the convolution level.
*Evidence:* PUBLISHED (title and mechanism confirmed via search synthesis: "up to 8× fewer parameters and
significantly better sample efficiency than standard models"; exact arXiv id **UNVERIFIED this pass** — not
independently opened, likely the NeurIPS-era Walters et al. equivariant-continuous-convolution line but not
confirmed with certainty). **Maturity: PROMISING** pending id verification, but the 8×-fewer-parameters
claim is a second independent data point for the same G2 mechanism.
*Pain points:* P4, P9 (few A40s — fewer parameters at matched accuracy is a direct compute-budget win for a
sub-300M-parameter, few-A40 programme).
*Admissibility:* fully admissible.
*Cost:* folds into G2's cost estimate if pursued (same underlying mechanism family); do not double-budget.
*Experiment:* subsumed by G2's experiment — if G2's mirror-augmentation control (Outcome A) succeeds,
ECCO's specific convolutional formulation becomes a candidate IMPLEMENTATION detail to compare against a
simpler equivariant-attention approach, not a separately prioritized experiment.

**G4. Trajectory vocabularies / anchors — factorized, combinatorial coverage.**
Discretizes the planning output space into a VOCABULARY of anchor trajectories (path anchors × velocity
anchors, combined combinatorially), turning trajectory generation into scoring/selection over a fixed,
interpretable set rather than free-form regression.
*Evidence:* PUBLISHED (VADv2, arXiv:2402.13243 — probabilistic planning over a large tokenized vocabulary,
SOTA on CARLA Town05 and Bench2Drive); PUBLISHED (SparseDriveV2, arXiv:2603.29163, "Scoring is All You
Need," early 2026 — a factorized vocabulary of **1024 path anchors × 256 velocity anchors = 262,144
combinatorial trajectories** for NAVSIM v1, with SparseDriveV2 itself achieving **32× denser** vocabulary
coverage than prior methods via the same factorization trick). **Maturity: PROVEN** (multiple independent
SOTA results on this family).
*Pain points:* **P1 (selection) directly** — our own fact sheet names "good candidates, bad choice" as P1;
an anchor-vocabulary + scoring architecture is LITERALLY a reformulation of planning as candidate-selection,
which is the mechanistic answer to a selection-quality problem, more so than a generation-quality one.
P3 (tactical — the "5-way softmax that mixes lat+lon" already flagged as our single largest known defect in
our own binding eval rule) — a factorized path×velocity vocabulary is a direct, concrete alternative to a
single mixed-manoeuvre softmax, decomposing the SAME decision into two more legible, separately-scoreable
axes.
*Admissibility:* fully admissible (an output-parametrization choice).
*Cost:* MEDIUM-HIGH — ~15-20 eng-days to replace the tactical head's decision structure with a factorized
anchor vocabulary + scorer; ~5-8 A40-days to re-train/fine-tune and validate.
*Experiment:* replace ONLY the tactical head's 5-way manoeuvre softmax with a small factorized
anchor-vocabulary + scorer (start small: e.g. 32 path anchors × 8 velocity anchors = 256 combinations, far
short of SparseDriveV2's 262K, sized for our sub-300M budget), matched-compute against the existing softmax,
and evaluate specifically on the TACTICAL metric family (manoeuvre-decision quality, confusion over classes
— per our binding four-family rule). **A:** confusion/decision-quality metrics improve → directly addresses
our own already-named single largest defect (P1/P3 combined), prioritize a fuller rollout. **B:** no
improvement → the mixed-softmax's problem may be in the TRAINING SIGNAL (what it's being taught to prefer)
rather than the OUTPUT PARAMETERIZATION, redirecting effort to F4's rule-based-teacher approach instead.

**G5. Auxiliary tasks from `obstacle.offline` — occupancy/agent-box forecasting.**
Adds a future-occupancy or future-agent-box forecasting auxiliary head, trained using our OWN already
license-clean but under-ingested 3D agent tracks (available on 97.44% of clips per our own fact sheet),
purely as a REPRESENTATION-SHAPING auxiliary loss, not as a direct runtime input.
*Evidence:* PUBLISHED, general mechanism (occupancy/flow forecasting as an auxiliary task is well-established
across UniPAD/ViDAR §B3 and dedicated occupancy-flow papers such as arXiv:2609.18442, "Risk-Aware World
Modeling with Flow-Guided Occupancy Evolution"); the DRIVING-SPECIFIC number for OUR corpus does not yet
exist and must be measured, not assumed. **Maturity: PROVEN as a general pretext-task class; UNTESTED for
our specific data/architecture** — correctly scoped as our own work item, not an external citation to lean
on for the number.
*Pain points:* **P4 directly, and at near-zero marginal DATA cost** — this is explicitly a case where "the
label already exists in our license-clean corpus and is simply unused," per our own DATA_STRATEGY_FOR_
HIERARCHY.md's own framing of `obstacle.offline`. It is IMPORTANT to distinguish this from the
ALREADY-FALSIFIED use: the lead-state gate falsified using agent state as a DIRECT INPUT to the longitudinal
head; using the SAME underlying labels as an AUXILIARY REPRESENTATION-LEARNING TARGET (predict where other
agents will be, as a training-time-only loss, not a runtime input). This is a mechanistically DIFFERENT
claim, and treating the earlier falsification as covering this case too would be exactly the kind of
over-generalized "absence"/refutation error our own CLAUDE.md operating standard warns against.
*Admissibility:* fully admissible (auxiliary loss at training time only; no runtime input change).
*Cost:* ~5-8 eng-days (mostly re-deriving box-forecasting targets from `obstacle.offline`, which requires
the same ~12.4 GB ingest already scoped and costed in DATA_STRATEGY_FOR_HIERARCHY.md); ~2-4 A40-days.
*Experiment:* add a future-agent-box-forecasting auxiliary head (predict nearby agents' positions 1-2s
ahead from the current latent), trained jointly but contributing NOTHING to the deployed inference path,
and measure whether the MAIN task's four metric families improve at matched compute vs. a no-auxiliary
control. **A:** any family improves → a genuinely free win (data already licensed and downloaded-scale
costed), ship it. **B:** no improvement → a second, independent, mechanistically-distinct test of whether
`obstacle.offline` carries exploitable signal for THIS corpus/architecture at all, complementing (not
duplicating) the already-falsified direct-input finding and F3's distillation-target test.

**G6. Kinematic-Consistency-Error — a diagnostic prior for imagination quality.**
Defines a metric that decodes IMAGINED (rolled-out) latents into physical quantities and checks whether
they are KINEMATICALLY CONSISTENT (e.g. implied acceleration/curvature is physically plausible) even when
they are not necessarily predicting the right DYNAMIC outcome — i.e. separates "the rollout looks like
smooth physically-plausible motion" from "the rollout is predicting what will actually happen."
*Evidence:* PUBLISHED ("Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model
Failure," arXiv:2607.05966, July 2026) — the paper's central claim, as reported in the operationalized
"Kinematic-Consistency Error" diagnostic, is that world models' imagined rollouts DEGRADE by becoming
kinematically-plausible-but-dynamically-wrong over long horizons, rather than becoming obviously physically
broken — meaning a naive visual/ADE-based check of imagined rollouts can look fine while the underlying
DYNAMICS reasoning has already failed. **Maturity: PROMISING** (July 2026, single paper, mechanism directly
applicable regardless of independent replication since it's a diagnostic tool, not a training method).
*Pain points:* **P7 (anti-calibrated imagination) directly** — this is a ready-made, external, off-the-shelf
diagnostic for exactly TanitAD's own named pain point, and per our own operating-standard rule ("a missing
metric is a work item, not an excuse"), imagination quality currently likely lacks this specific lens.
*Admissibility:* fully admissible (an evaluation/diagnostic instrument, not a data source or architecture
change by itself, though its RESULT may motivate one).
*Cost:* LOW — ~4-6 eng-days to implement the Kinematic-Consistency-Error metric against existing imagined
rollouts from already-trained checkpoints; 0 additional GPU-days (a post-hoc analysis of existing rollouts).
*Experiment:* compute Kinematic-Consistency-Error on the flagship's already-generated imagined rollouts
across increasing horizons, and check whether it degrades BEFORE, WITH, or AFTER the already-known ADE-based
imagination-decay curve (this directly extends the already-pre-registered E-CR experiment on imagination
decay in `PREREG_deep_research_2026-07-29.md`, rather than duplicating it). **A:** kinematic consistency
degrades LATER than ADE/dynamic accuracy → our imagination's problem is primarily DYNAMIC (predicting the
wrong outcome), not kinematic (the shape of the rollout stays physically sane) — points toward more training
signal/data as the fix (favor §A4/§B7). **B:** kinematic consistency degrades AT THE SAME TIME OR EARLIER
than dynamic accuracy → the rollout is losing even basic physical plausibility, which is an ARCHITECTURAL
finding (favor G1's kinematic-constraint layer applied to the IMAGINATION path specifically, not just the
final output decoder) — a clean, falsifiable answer either way, and cheap to obtain since it reuses existing
checkpoints.

## H. Knowledge injection & continual/fleet learning

**H1. R²LPL — Rollout-Retrieval Lifelong Policy Learning.**
A closed-loop rollout exposes failures; a RETRIEVAL step specifically identifies RECOVERABLE
mistake-related states and constructs corrective targets from them; a lifelong-learning step updates the
policy with this new knowledge plus replayed memory — an explicit Rollout→Retrieve→Learn cycle.
*Evidence:* PUBLISHED (Gong, Wang, Lu, Gong — Beijing Institute of Technology; Li — NTU;
"Learning from Mistakes: Rollout-Retrieval Lifelong Policy Learning for Autonomous Driving," arXiv:2606.30537,
June 2026). Mechanism confirmed in detail; specific quantitative gain over a non-retrieval baseline not
independently opened this pass — **UNVERIFIED at the number level**, mechanism and venue confirmed.
**Maturity: PROMISING** (recent, single paper).
*Pain points:* **P6 (compounding error), and directly extends TanitAD's OWN already-implemented H10
mechanism** (latent-RAG, surprise-gated write, MEASURED in-house at +18.8% on unseen surprise contexts but
**−24% interference on well-predicted contexts**). R²LPL's retrieval is gated on RECOVERABLE MISTAKES
specifically (outcome-conditioned), which is a MORE SELECTIVE trigger than generic prediction-surprise —
a concrete, externally-motivated candidate fix for exactly the interference side-effect our own docs already
measured and flagged as a known failure mode of the current surprise-gate.
*Admissibility:* admissible — retrieval keys and corrective targets are derived from the model's OWN past
rollouts/replay, not a privileged runtime channel.
*Cost:* MEDIUM — ~10-12 eng-days to add outcome-conditioning (not just surprise-magnitude) to the existing
H10 memory-write trigger; ~3-5 A40-days to re-run the D7 gate (repeat-exposure improvement) already planned
for H10 with this modification.
*Experiment:* re-run the ALREADY-PLANNED H10/D7 gate (repeat-exposure improvement test) with the write
trigger changed from pure imagination-error magnitude to "imagination-error magnitude AND the state was
later followed by a recoverable correction" (R²LPL's outcome-conditioning), and compare the
well-predicted-context interference number against the already-measured −24% baseline. **A:** interference
shrinks materially (e.g. toward −10% or better) while the +18.8% surprise-context gain is retained → a
direct, quantified fix to an already-known and already-costed failure mode, adopt outcome-conditioned
retrieval as the H10 write policy. **B:** interference persists similarly → the interference is likely a
property of the RETRIEVAL/FUSION mechanism (e.g. the cross-attention gate itself) rather than the WRITE
TRIGGER, redirecting the fix toward the gate-MLP/fusion architecture instead of the trigger condition — a
useful, falsifiable narrowing either way.

**H2. MoE-LoRA continual adaptation.**
Combines feature-generation replay with a Mixture-of-Experts router and LoRA adapters, so new-domain
knowledge is added via new low-rank adapter capacity while old-domain behaviour is preserved via replay,
rather than full-network fine-tuning risking catastrophic forgetting.
*Evidence:* PUBLISHED (feature-generation-replay + MoE-LoRA for data-driven autonomous guidance, MDPI —
title/mechanism confirmed via search; exact quantitative retention numbers **UNVERIFIED this pass**, not
independently opened). Broader context: LoRA is explicitly characterized in the 2026 continual-learning
literature as "a continual-learning method in disguise" — a capacity-allocation view of adapters that is a
useful FRAME even independent of this specific paper. **Maturity: PROMISING.**
*Pain points:* directly relevant to H8's already-existing Sparse MoE tactical router (our own hub docs
already note the router interface is in place for sensor-modality and skill experts) — LoRA-per-new-domain
is a natural extension of an architecture TanitAD already has, rather than a new architectural commitment.
*Admissibility:* admissible.
*Cost:* ~8-10 eng-days to add LoRA-adapter slots to the existing MoE tactical router for a SECOND data
source (e.g. L2D, once ingested) without retraining the base network; ~2-4 A40-days.
*Experiment:* when L2D (already planned per DATA_STRATEGY_FOR_HIERARCHY.md) is ingested as a new arm, train
a LoRA adapter for it on top of the frozen flagship base rather than a full fine-tune, and compare
parity-corpus performance BEFORE vs. AFTER the L2D-adapter addition (testing for forgetting). **A:**
parity-corpus performance is unchanged (within CI) while L2D-relevant metrics improve → LoRA-per-source is a
safe, low-risk way to grow the corpus mixture without the "new data = new arm, never re-selection" invariant
being put at risk by full-network drift. **B:** parity performance degrades → the base network is not as
modular as hoped, and full multi-source joint training (with its own forgetting risks) may be unavoidable —
informs a real architectural decision before L2D integration proceeds at scale.

**H3. Continual test-time adaptation (CTTA) and world-model-aware online adaptation.**
CTTA methods adapt a frozen-at-deployment model continually as it encounters a SEQUENCE of shifting target
domains at test time, without domain-boundary labels; a world-model-specific variant (AdaWM-style) DIAGNOSES
a world-model/policy MISMATCH via a divergence measure and triggers alignment fine-tuning rather than
adapting blindly.
*Evidence:* PUBLISHED (CTTA survey, "Continual Test-Time Adaptation in Computer Vision: Methods, Benchmarks,
and Future Directions," arXiv:2607.08164, July 2026 — general CV survey, not driving-specific); AdaWM-style
adaptive world-model RL referenced via search synthesis, exact arXiv id **UNVERIFIED this pass**.
**Maturity: PROMISING**, general-domain for CTTA; **SPECULATIVE** for the WM-mismatch-diagnosis framing
applied to driving specifically.
*Pain points:* P6 (a model that can detect its own world-model/policy mismatch at deployment time is a
softer, cheaper version of the fleet-learning flywheel TanitAD cannot otherwise afford, per §E3's admissibility
gap) — this is the SPECULATIVE, Thor-deployment-relevant end of the continual-learning spectrum: adapting
ONE deployed unit online, rather than a fleet.
*Admissibility:* admissible IF adaptation uses only the deployed vehicle's own vision-only observations
(consistent with the vision-only-at-inference rule) — a CTTA method that uses privileged signals to steer
adaptation would violate the same rule that governs the situation classifier and must be checked explicitly
before adoption, not assumed safe by default.
*Cost:* HIGH — CTTA on an embedded Jetson Thor target is a nontrivial systems problem (online updates,
catastrophic-forgetting risk with no replay buffer of prior deployment data by default); ESTIMATED 15-20
eng-days for even a minimal on-device CTTA pilot, and this cannot be meaningfully GPU-cost-estimated without
Thor hardware access (which this review stream does not have — flagged as a hardware-in-the-loop dependency).
*Experiment:* given the cost and Thor-access dependency, this is NOT a near-term experiment; the actionable
item is a SCOPING note for whichever stream owns Thor deployment: read the CTTA survey (arXiv:2607.08164) for
a benchmark/method shortlist before any online-adaptation design work begins, rather than designing from
scratch.

**H4. VLM knowledge injection — cross-referenced from §F6.**
Traffic-rule/semantic knowledge injection via a VLM teacher, distilled into a vision-only student that
discards the VLM at inference, is the SAME mechanism scored in full under §F6 (DiMA/Drive-KD/BucketKD/PlanKD)
since distillation and knowledge-injection are the same technique viewed from two angles here. See §F6 for
the full six-field entry and the concrete cheap experiment (offline VLM scene-type labelling → auxiliary
head). **No separate scoring here** to avoid double-counting in the TOP-10.

**H5. Genie 3 → Waymo World Model — the fleet-scale ceiling, and why it doesn't directly transfer.**
DeepMind's Genie 3 (a general-purpose, real-time interactive world model pretrained on a massive diverse
video corpus) was adopted by Waymo and specialized via post-training into a driving-specific world model
that outputs Waymo's own LiDAR modality, letting Waymo explore situations never directly observed by its
fleet.
*Evidence:* PUBLISHED (Genie 3, Google DeepMind, Aug 2025 — real-time interactive world generation,
photorealistic, ~1 minute memory); PUBLISHED-by-report (Waymo's Feb 2026 adoption and specialization,
"the Waymo World Model," via Waymo's own blog and secondary coverage — **not independently opened as a
primary technical paper this pass**, so the LiDAR-transfer mechanism detail is reported, not verified
first-hand). **Maturity: PROVEN** for Genie 3 itself; **PUBLISHED-by-report, UNVERIFIED-at-the-mechanism-
level** for the Waymo specialization specifically.
*Pain points:* illustrates the CEILING of the "pretrain on huge diverse video, specialize via post-training"
recipe — but **explicitly requires a fleet + LiDAR TanitAD does not have**, so this is scored primarily as
an ADMISSIBILITY/FEASIBILITY note, not a directly portable technique. The TRANSFERABLE LESSON, independent
of Waymo's specific execution, is that general video-pretrained world knowledge CAN be grafted onto a
sensor-specific small model via post-training — which is exactly what TanitAD's OWN frozen-encoder arm
attempted and failed at (2.17 m ADE), suggesting the GRAFTING MECHANISM (naive frozen-backbone + linear
probe) is what needs fixing, not the underlying premise — pointing back at §B6/B7 (WA-JEPA, LFG) as the
mechanistically richer grafting recipes worth retrying instead.
*Admissibility:* the GENERAL lesson is admissible; the SPECIFIC Waymo mechanism is not reproducible here
(no fleet, no LiDAR) and should not be cited as a directly portable recipe.
*Cost:* N/A as a directly portable technique.
*Experiment:* none standalone — this entry exists to correctly SCOPE the ambition (aim above published SOTA
per our own operating standard, but via B6/B7's cheaper, sensor-matched recipes, not via a fleet-scale
system TanitAD cannot build).

**H6. AdaWorld — context-invariant latent actions for fast adaptation.**
Extracts LATENT ACTIONS from unlabeled video in a context-invariant way, so a demonstrated action can be
transferred to new environments/embodiments with minimal or no further training, and a base world model can
be adapted to a new context quickly with limited data.
*Evidence:* PUBLISHED (arXiv:2503.18938, ICML 2025) — **70.5% human success rate on LIBERO transfer vs. 20%
for baseline**; faster adaptation with limited data on game and robotic tasks. **Maturity: PROVEN**, but
**in robotics/game domains, NOT driving** — flagged explicitly, no driving-specific number exists.
*Pain points:* SPECULATIVE relevance to "fleet learning" in spirit — a new sensor rig, a new city, or a new
country could in principle be treated as a new "context" that AdaWorld-style latent-action transfer adapts
to quickly, rather than retraining from scratch; this is the most SPECULATIVE item in this stream and should
be weighted accordingly.
*Admissibility:* admissible in principle (self-supervised, no privileged signal).
*Cost:* HIGH to adapt the method to driving's continuous, high-precision action space (the same LAPA/Genie
discretization mismatch already flagged in our own internal H7 notes applies here too); ESTIMATED 15-20
eng-days for a driving-specific feasibility pilot alone.
*Experiment:* not recommended as a near-term experiment given cost and domain mismatch; listed for
completeness per the brief's explicit coverage list, and because its CORE MECHANISM (context-invariant
latent actions from unlabeled video) is conceptually continuous with §C's IDM-camera-rig-transfer work — if
§C's fixes (canonicalization, rig-conditioning) succeed, AdaWorld-style context-transfer becomes a more
plausible NEXT step (new rig = new "context") worth revisiting then, not now.

## Ranked TOP-10 for TanitAD

*(pending)*

## A concrete "10x less labelled data" programme

*(pending)*

## What NOT to do

*(pending)*

## Deliverable manifest

*(pending)*
