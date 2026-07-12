# TanitAD: A Data-Efficient, Hierarchically-Imagining, Self-Supervised Driving Stack with Built-In Self-Knowledge

**Status:** living paper, v0.1 (2026-07-08). Maintained per D-020: every gate evaluation appends
results; every accepted decision that changes the method updates §3–§5. Source of truth is this
Markdown; LaTeX export is a release step. Honesty rule (P8): no number appears here without its
instrument rows in the referenced experiment record.

**Authors:** Sayed Bouzouraa; TanitAD autonomous research system.

---

## Abstract

End-to-end driving models have won the architecture argument but retain four structural weaknesses:
they are data-hungry, opaque, unaware of their own limits, and expensive at inference time. We
present TanitAD, a driving stack built around a from-scratch, fully self-supervised hierarchical
latent world model (the *4B architecture*: operative, tactical, strategic, and fallback brains) that
attacks all four weaknesses simultaneously. The model (261 M parameters) trains on tens of hours of
unlabeled front-camera video plus free proprioception — no perception labels, no HD maps, no reward,
no pretrained foundation encoder — using a single provable anti-collapse mechanism (SIGReg/LeJEPA).
Maneuver selection is performed by *imagine-and-select*: the operative predictor imagines the latent
consequence of each candidate action, and frozen probes calibrated on the predictor's own imagined
latents decode those consequences into metric space at millisecond cost. Preliminary results at 17 %
of the first training run: imagined action-conditioned latents rank real driving actions with 0.87
direction accuracy (0.94 via a forward-dynamics readout) against a 0.5 chance floor — while raw
imagination fidelity is still far from perfect, replicating on real data the finding that *action
discrimination, not imagination fidelity, bounds control*. A spectral analysis of the learned
transition operator (fit R² = 0.997) locates the task-relevant dynamics of highway-dominated driving
in ≈ 22–35 latent dimensions, consistent with the generalization theory that motivates latent world
models' sample-efficiency advantage. We describe the architecture, its mathematical grounding, an
instrument doctrine for honest measurement that caught three silent measurement hazards in its first
week, and the falsifiable gate program by which every claimed edge will stand or fall.

---

## 1. Introduction

The autonomous-driving industry has converged on end-to-end learned stacks and, increasingly, on
world models (Wayve GAIA-class; NVIDIA Alpamayo; latent-world-model lines such as LAW and
World4Drive). Yet the dominant recipes pay for capability with scale: internet-scale pretraining
(V-JEPA-2-class, ~10⁶ h), billions of parameters of pixel-space generation (GAIA-class), or dense
human annotation (UniAD-class). Independently of scale, deployed systems retain characteristic
failure classes — construction zones, unprotected turns, occlusion amnesia, gesture blindness — that
recalls and regulator probes continue to document.

This work starts from a different premise, distilled from two decades of industrial AD experience
and a sequence of controlled world-model experiments (ALPS-4B): **the structure of the driving task,
not the scale of the training corpus, is the primary unexploited resource.** Driving decomposes
naturally (Michon) into operational, tactical, and strategic levels with distinct time scales;
egocentric observation makes every action's consequence dominate the visual field; ego-motion is
free supervision on every vehicle; and the task-relevant dynamics are intrinsically low-dimensional
relative to the observation stream. Each of these structural facts converts directly into data- and
compute-efficiency, and — through explicit hierarchy, decoded plans, and imagination-error
monitoring — into the transparency and self-knowledge that the new UN ADS regulation (June 2026)
demands.

Contributions (each tied to a falsifiable gate, §6):

1. **The 4B architecture** — a from-scratch self-supervised hierarchical latent world model with
   operative/tactical/strategic/fallback decomposition, trained end-to-end with a single
   anti-collapse mechanism and no teacher-student heuristics (§3).
2. **Imagine-and-select with calibrated frozen decoding** — planning as argmin over a discrete
   maneuver vocabulary in latent space, with metric readout through probes fitted on the
   predictor's *own imagined latents*; and a cheaper forward-dynamics readout (P4) that operates in
   decoded state space without imagination in the loop (§3.4).
3. **Imagination in unobserved areas** — sector-masked training, semi-Lagrangian latent advection,
   and heteroscedastic epistemic gating as an integral training objective, providing the principled
   trigger for sensor/modality gating (§3.5).
4. **An instrument doctrine** for honest measurement (oracle rows, batch-consistency under pinned
   numerics, route-level splits, persistence baselines, task-identity fingerprints), presented not
   as methodology garnish but as a first-class system component that repeatedly changed our
   conclusions (§5).
5. **Preliminary evidence** from the first learning-valid training run on real data (§7), and a
   theoretically grounded account of *why* the approach is sample-efficient (§4).

## 2. Related work

**Generative pixel world models** (GAIA-1/2, Vista, DriveDreamer, Cosmos-class) excel at controllable
synthesis but plan by rendering pixels — a per-decision cost of seconds to minutes and a parameter
budget spent on appearance. **Latent world models for driving** (LAW, World4Drive, WorldRFT,
ResWorld, IDOL) are our family: prediction and evaluation in latent space, annotation-free or
lightly supervised; but published systems are flat or two-level, typically ride pretrained
perception backbones, and none combine hierarchy, from-scratch SSL, and calibrated imagination
readout. **JEPA-family control** (V-JEPA-2-AC, Drive-JEPA) demonstrates the architecture direction
at foundation scale while presupposing the giant encoder we deliberately refuse; its 62-hour
action-conditioning stage is nonetheless the key external datum that action grounding is cheap once
representation exists. **Hierarchical JEPA** appears in adjacent domains (HiT-JEPA for trajectory
similarity) with independent evidence that hierarchy buys cross-domain generalization.
**Theory:** LeJEPA gives the isotropic-Gaussian optimality argument and the SIGReg mechanism; the
JEPA generalization theory (Cui et al., 2026) supplies finite-sample planning-regret bounds and the
latent-dimension trade-off we exploit in §4. Our position is the empty cell in this matrix: ≤ 10² h
data, ~10⁸ parameters, zero labels, hierarchical latent planning, calibrated frozen decode, built-in
OOD self-monitoring.

## 3. The 4B architecture

### 3.1 Overview and notation

Observations are egocentric camera stacks x_t ∈ ℝ^{9×256×256} (three RGB frames at 100 ms spacing,
D-015), actions a_t = (steering, acceleration) ∈ ℝ² from CAN/ego-motion, poses p_t = (x, y, ψ, v)
from odometry. An encoder f_θ maps x_t to a token grid; a spatial grid readout (never global
pooling; A7) produces the compact state z_t = r(f_θ(x_t)) ∈ ℝ^{2048}. The instantiated budget is
261 M parameters: encoder 99.5 M (ViT, d = 768, 14 blocks, patch 16, LayerNorm only — batch-free
norms are a *correctness* requirement, §5), operative predictor 107.7 M (causal transformer,
window 8, FiLM action conditioning, residual multi-horizon heads k ∈ {1, 2, 4}), tactical predictor
26.5 M (d = 512, horizons k ∈ {8, 16}), imagination field 22.1 M (§3.5), inverse-dynamics head
5.2 M; the strategic layer is deliberately non-parametric (VQ codes over pooled latents + a latent
transition graph), and the fallback brain is out-of-gradient monitoring logic.

### 3.2 Self-supervised objective

Training minimizes, with no EMA, no stop-gradient, and no teacher network (A1):

L = L_pred + λ_tac L_tac + λ_sig (S(Z) + S(Ẑ)) + λ_inv L_inv + λ_H15 L_imag

- **Residual multi-horizon prediction** L_pred = Σ_k w-MSE(ẑ_{t+k}, z_{t+k}), where
  ẑ_{t+k} = z_t + Δ_k(z_{t−W+1:t}, a_{t−W+1:t}) and w-MSE is change-weighted: per-dimension weights
  ∝ |z_{t+k} − z_{t+k−1}| normalized to unit mean. Residual + change-weighting beat plain MSE and
  flow objectives decisively in controlled bake-offs (0.97 vs 0.71 vs 0.44 direction accuracy; A4)
  and change-weighting is justified on real data by measured consequence-dominance (A8: per-step
  frame-change fraction ≈ 0.05–0.11 on our corpora).
- **SIGReg** S(·) is the sliced Epps–Pulley statistic of LeJEPA: embeddings are projected onto
  M = 512 freshly-sampled unit directions and each projection is scored against N(0,1) via
  T_{n,β} = (1/n) Σ_{j,k} e^{−β²(Y_j−Y_k)²/2} − 2(1+β²)^{−1/2} Σ_j e^{−β²Y_j²/(2(1+β²))} + n(1+2β²)^{−1/2},
  applied to both encoder outputs and predictions at all levels. By Cramér–Wold, driving all 1-D
  projections to normality drives the joint toward the isotropic Gaussian — the embedding
  distribution LeJEPA proves optimal — with uniformly bounded gradients. Two practical laws we
  learned by violating them: the statistic's built-in batch scale must not be normalized away, and
  the test is statistically starved below ≈ 256 samples per step (F-2: at 32 samples/step the
  representation collapsed to effective rank 23/2048 while the prediction loss kept falling).
- **Inverse dynamics** L_inv = ‖g(z_t, z_{t+1}) − a_t‖²: proprioception as free supervision forces
  controllable state into the latent (A5) and seeds the pseudo-labeling model for action-free video
  (H7).
- **Tactical horizon loss** L_tac: the same residual objective at maneuver horizons (0.8/1.6 s),
  training the tactical predictor that imagine-and-select queries.
- **Imagination loss** L_imag: §3.5.

### 3.3 Hierarchy

The four brains operate at separated rates and interfaces: the operative predictor imagines at
10–20 Hz over 0.1–0.5 s; the tactical layer selects maneuvers at 1–2 Hz over 0.8–5 s by imagining
each candidate's post-maneuver latent; the strategic layer routes at ≤ 0.5 Hz over a latent
transition graph whose nodes are VQ place-situation codes and whose edges carry empirical costs
(topological memory of the driven network — routing and re-routing without HD maps; +58 % over
greedy on topology tasks in the toy program, A6); the fallback brain monitors continuously and owns
the minimal-risk manoeuvre. Two theoretical arguments support the decomposition: (i) planning
regret grows linearly with horizon T (Cui et al. Thm 4.2), so factorizing one long horizon into
per-level short horizons pays a strictly smaller bound at each level; (ii) hierarchical
recombination of maneuvers and places converts sample complexity from the product to the sum of
level cardinalities. Inference-rate layering also converts the hierarchy into an efficiency device:
the 207 M operative path is the only component at full rate.

### 3.4 Planning: imagine-and-select with calibrated decoding

Given the current window and a discrete maneuver vocabulary {m_i} (9–15 parameterized action
primitives), the tactical layer computes ẑ^{(i)} = imagine(z, m_i) for all i in one batched pass and
scores s_i = d(readout(ẑ^{(i)}), goal) + costs. Two properties make this practical:

- **Calibrated frozen decoding (A3).** Probes that read imagination are ridge regressors fitted on
  (imagined latent → realized future) pairs — not on real-frame encodings. Imagined latents live
  slightly off the encoding manifold; fitting the probe on them absorbs the systematic shift
  (measured: 0.97 vs 0.66 direction accuracy for identical predictions). Probes are fitted offline
  in closed form, are frozen at deployment, carry zero training burden, and cannot couple the
  planner to a learned metric head.
- **The P4 forward-dynamics readout.** A second frozen probe g: (decoded low-D state, action) →
  next decoded state ranks candidate actions *without imagination in the loop* — the cheapest
  readout and, in both the toy program (0.76) and our first real-data evaluation (0.94), the
  strongest. P4 doubles as a redundancy channel for the safety case.

Selection costs K predictor passes plus probe matmuls — milliseconds at this scale, versus seconds
to minutes for pixel-rendering or CEM-population planners.

### 3.5 Imagination in unobserved areas (H15)

Human drivers maintain beliefs about traffic they cannot currently see. TanitAD trains this
explicitly: whole spatial sectors of the input are masked (simulating occlusion or a gated sensor);
a dedicated field must predict the *future latent content* of hidden cells. Hidden cells evolve by
semi-Lagrangian advection ẑ_{t+1}(u) = z_t(u − v_θ(u)Δt) under a learned token-grid flow — object
permanence by construction — refined by attention over visible context, with a per-cell
log-variance head trained under heteroscedastic NLL:

L_imag = Σ_cells w(u) · [ e^{−s(u)} ‖ẑ(u) − z(u)‖²/2 + s(u)/2 ],  w = 1 on hidden, 0.1 on visible.

The model must therefore *know where it cannot know*. The variance field is the principled trigger
for attention-based modality steering (H2): a sensor may be powered down exactly when imagined
uncertainty in its field of view is low — replacing heuristic sensor schedulers with a
world-model-native criterion. Gate D9 measures hidden-sector imagination against a shuffled-cell
floor and requires calibration (higher variance where blind).

### 3.6 Fallback and self-knowledge

The imagination error ‖ẑ_t − z_t‖ (relative to step scale) is a free, always-on familiarity signal
(A9); layered monitors (imagination error at 10 Hz; routing/checker statistics at 1 Hz; Mahalanobis
drift at 0.1 Hz) arm a deterministic minimal-risk manoeuvre and constitute the substrate for the UN
regulation's ISMR/DSSAD requirements — self-monitoring logs *are* the incident-reporting feed.

### 3.7 Action grounding across abstraction levels

**Actions are inputs whose consequences the model predicts.** The residual predictor (§3.2) is
action-conditioned by construction: ẑ_{t+k} = z_t + Δ_k(z_{t−W+1:t}, a_{t−W+1:t}), where each action
a = (steer, accel) enters every predictor layer as a FiLM scale/shift. Accelerate, steer and
decelerate are therefore fed to the predictor and it estimates the resulting latent state — the
world-model contract: *simulate the effect of your own control*. Nothing here is a target the model
regresses; the action is an exogenous input and the future latent is its predicted effect.

**The grounding gap.** JEPA prediction makes future latents predictable and rankable but does not
force the latent to encode *metric* ego-motion. Our diagnostic (§7) showed the frozen state decodes
ego-position 10–15× worse than a constant-velocity baseline *even in-distribution* (oracle ADE@1s
1.65 m vs 0.28 m): the representation was under-grounded because the objective never demanded metric
consistency, only latent predictability and within-window rank (which is why the selection gate D2
passed while the position gate D1 failed). We close the gap with two *proprioceptive* losses that
leave the self-supervised core untouched.

**(i) Metric inverse dynamics.** A head h_ν maps a latent pair to the metric ego-displacement in
SE(2):  Δp̂_{t→t+k} = h_ν(z_t, z_{t+k}) = (Δx, Δy, Δψ),  with
L_mid = Σ_k ρ( Δp̂_{t→t+k} ⊖ Δp_{t→t+k} ),
where Δp is the true relative ego-pose from odometry, ⊖ the SE(2) residual, ρ a Huber penalty. This
strengthens the action-inverse-dynamics L_inv (§3.2, which recovers a_t): rather than the control,
the latent pair must now recover *where the vehicle physically went*.

**(ii) Forward metric consistency.** Rolling the predictor K steps under the *true* action sequence
and decoding each predicted transition to a per-step displacement r_ξ(z_t, ẑ_{t+1}), the accumulated
SE(2) trajectory must match odometry:
ẑ_{t+j} = z_t ⊕ Δ_{≤j}(ẑ, a)  (recursive rollout under fed actions, §3.2, D-027);
τ̂_{t+j} = ⊕_{i≤j} r_ξ(ẑ_{t+i−1}, ẑ_{t+i});  L_fmc = Σ_j ρ( τ̂_{t+j} ⊖ τ_{t+j} ),
with τ the odometry ego-trajectory and ⊕ SE(2) composition. The displacement is decoded from the
*predicted* (imagined) latent under the fed action — so **imagination itself is forced to be
metrically correct**, exactly the quantity imagine-and-select (§3.4) consumes: the trajectory
*emerges by rolling out grounded dynamics under actions*, never regressed in one shot. The grounded
objective is L + λ_mid L_mid + λ_fmc L_fmc, with L (JEPA + SIGReg + imagination) unchanged.

**Why the unsupervised character is preserved.** Both new targets — actions a_t (CAN bus) and
relative poses Δp, τ (IMU / GNSS / wheel odometry) — are *proprioceptive sensor readings*, generated
for free on every driven metre, never human annotations. Using them incurs no labelling cost and
scales with mileage; it is the "S" of self-supervision realised through the vehicle's own motion
senses — as vestibular and proprioceptive feedback ground egomotion in animals — not supervised
imitation of an annotated trajectory. The 1000×-efficiency claim (§4) rests on the absence of
*annotation*, which is untouched: no term requires a human label, the encoder is still shaped by
SIGReg-regularised latent prediction, and the metric heads are auxiliary readouts whose gradients
merely *orient* the same self-supervised representation toward metric ego-motion. Grounding on
proprioception ≠ supervising on labels.

**Hierarchical grounding (the unifying principle).** Grounding is not confined to low-level control.
Each abstraction level ℓ has its own action a^ℓ, forward model g_φ^ℓ, and consequence horizon H_ℓ,
and each is grounded to the *metric consequence at its own scale* under one common contract —
**predict-the-metric-consequence-of-your-action**:
ẑ^ℓ_{t+H_ℓ} = g_φ^ℓ(z^ℓ_t, a^ℓ),   L_ground^ℓ = ρ( r_ξ^ℓ(z^ℓ_t, ẑ^ℓ_{t+H_ℓ}) ⊖ Δp^ℓ_gt ).

| level ℓ | action a^ℓ | horizon H_ℓ | grounded consequence |
|---|---|---|---|
| operative | continuous (steer, accel) ∈ ℝ² | 0.1–0.5 s | per-step Δp (SE(2)) |
| tactical | maneuver / intent primitive m | 0.8–2 s | post-maneuver latent + metric trajectory shape |
| strategic | route decision g (branch/exit at a place-node) | 5–30 s | place-graph transition + coarse metric route |

A maneuver is thus grounded to the 2-s trajectory it *produces*, and a route decision to the
place-to-place displacement it *produces*, each from proprioception at that horizon. This turns the
4B hierarchy into a **stack of grounded forward models**: imagine-and-select (§3.4) ranks maneuvers
by their grounded metric consequence, and strategic routing ranks route actions by their grounded
place-graph consequence — one principle at three scales. Operative grounding ships now (this run);
tactical and strategic grounding are the immediate hierarchical extension (H-ledger **H18**), and
they compose with counterfactual action augmentation (regenerate a scene under varied actions,
learn each level's consequence — H-ledger, `COUNTERFACTUAL_ACTION_AUGMENTATION.md`).

### 3.8 Reference architectures (H1 / H4 controls)

Two budget-matched (~261 M) references isolate a single axis each; both train on the *identical*
corrected-geometry corpus (§6.1) so only the tested axis differs.

- **REF-A — frozen-DINO world model (H4, encoder axis).** Identical predictor, imagination and
  action grounding (§3.7); the from-scratch encoder f_θ is replaced by a *frozen* DINOv2 encoder plus
  a trainable spatial-grid adapter r_A, so z_t = r_A(DINO(x_t)). Every loss including L_mid, L_fmc
  applies to the adapter + predictor; the frozen features carry no gradient (the inherent, correct
  asymmetry: from-scratch encoder can be reshaped by grounding, a frozen encoder can only be adapted).
  Isolates *learned-from-scratch vs web-pretrained* representation at matched predictor and matched
  grounding.
- **REF-B — vision-action E2E, no world model (H1, world-model-earns-its-parameters axis).** The same
  encoder trunk feeds direct operative (10–20 Hz (steer, accel) + 0.5 s sequence) and tactical
  (maneuver distribution + 2 s waypoints) heads trained by behavior cloning; no predictor, no
  imagination, no SIGReg-latent-prediction, and the freed ~130 M go into a deeper encoder + wider
  heads (budget-matched). Its "grounding" is *direct* trajectory/action supervision — the strongest
  fair imitation baseline and gate D4's learned opponent. By construction it lacks imagination-error
  self-knowledge (§3.6), hidden-actor rollout (H15) and counterfactual imagine-and-select (§3.4) —
  precisely where the world model is pre-registered to win; if REF-B matches it there, the
  world-model premise is wounded and we report that.

## 4. Why less data suffices: theoretical grounding

Three independent arguments, one measured corroboration.

1. **Latent targets discard nuisance.** The JEPA generalization theory formalizes the trade-off: with
   a spectral-contrastive objective, pretraining risk equals a rank-k factorization error of the
   action-conditioned co-occurrence operator; downstream planning regret is bounded by
   O(T·√(Σ_{i>k} σ_i² + C(k)/√n)) with approximation error falling and estimation error rising in
   the latent dimension k. Latent models win precisely when moderate k captures task dynamics while
   pixels carry nuisance — the driving regime by construction. Pixel-level prediction is the
   degenerate k = n endpoint paying maximal sample complexity.
2. **Consequence dominance concentrates gradient.** In egocentric video every action moves every
   pixel; the action's consequence is a dominant, predictable fraction of frame change (measured
   0.05–0.11 per step on our corpora). The toy program demonstrated a 0.19 → 0.69/0.76 control jump
   from the observation model alone (A11) — driving's default observation model is the one in which
   action-conditioned dynamics are cheap to learn.
3. **Hierarchy recombines instead of memorizing.** Maneuvers learned once recombine across routes
   via graph routing; the sample space factorizes.

**Measured corroboration (step-3000 checkpoint):** the transition operator (z_t, a_t) → z_{t+1} fits
linearly with R² = 0.997; its spectrum concentrates 99 % of energy in ≈ 22 dimensions (effective
rank ≈ 35). Task-relevant driving dynamics are low-rank in situ; the observation stream is ~10⁵
dimensional. The gap between those two numbers is the sample-efficiency budget this program spends.

## 5. The instrument doctrine

Every result in this paper ships with instrument rows assembled *before* the claim, and a gate whose
instruments fail is BLOCKED — reported as unmeasurable, never as a model failure. The rows: I1
oracle decode (the harness must rank real futures ~perfectly before imagination is graded); I2
batch-consistency under pinned numerics (deployment is batch-1 streaming; TF32/cuDNN kernel
selection alone produced 4× the tolerated deviation at 261 M scale — F-1); I3 route-level splits
(random-frame splits measured 4× optimistic); I4 persistence baselines — *demoted from gate to
diagnostic* for control claims after A13 (control measured usable at imagination-error ratio 1.27;
what bounds control is action discrimination in decoded space, which is exactly what gate D2
measures); I7 task-identity fingerprints (probe-fit corpus ≡ evaluation stream, checked
mechanically — camera intrinsics canonicalized to a common effective focal length across corpora,
§6.1). In the program's first week the doctrine caught three silent hazards (numerics, collapse
masked by a falling loss, a data-selection bug that chose parked cars) — each invisible in the
happy-path training curves.

## 6. Experimental program

### 6.1 Data

Phase 0 trains on ~44 h of real driving: comma2k19 (33 h highway commute; real CAN actions; MIT
license; the public-claims anchor) and a scenario-filtered urban subset of a large multi-country
corpus (500 clips selected by motion statistics from 25 countries; usage license under review, so
all public numbers are reported on the open corpora), plus the CC-BY-4.0 Cosmos-Drive-Dreams
synthetic long-tail corpus (weather/night variants; verified consequence-dominant at A8 = 0.109).
All cameras are canonicalized to a common effective focal length (266 px at 256² input) so metric
motion maps to consistent pixel motion across corpora; corpus fingerprints are enforced at
evaluation (I7). Validation splits are by route/drive, never by frame.

### 6.2 Gates

The falsifiable ladder (thresholds fixed before runs): D1 metric decodability (< 1 m ADE@1s,
camera); D2 action ranking (direction accuracy > 0.7 via calibrated probe OR forward-dynamics
readout); D3 imagined-vs-oracle trajectory decode (ratio ≤ 1.5); D4 tactical > greedy; D5 strategic
routing on topology; D6 simple→complex generalization at matched parameters; D7 episodic memory;
D8 OOD monitoring (AUROC > 0.85); D9 hidden-sector imagination. D1–D3 are decode gates — necessary,
not sufficient (decode quality does not imply planning success); closed-loop D4–D6 arbitrate.

### 6.3 Training configuration

261 M parameters; bf16 autocast with SIGReg computed in fp32; gradient accumulation (micro 32 × 2,
effective 64 — keeping SIGReg above its statistical floor); activation checkpointing; ~30 k
optimizer steps on a single 48 GB GPU; total training cost of the first run ≈ $40 of commodity
cloud compute. (That figure is itself part of the thesis.)

## 7. Preliminary results (first run, in progress)

At step 5,000 of 30,000 (17 %), on 48 held-out route-level validation episodes, instruments first:

| Gate | Result | Detail |
|---|---|---|
| D2 | **PASS** | direction accuracy 0.872 (calibrated) / **0.940 (P4)** vs 0.7 bar, 0.5 chance; imagination-error ratio 9.7 — the A13 pattern (discrimination ≫ fidelity) on real data |
| D1 | FAIL (at 17 % training) | waypoint ADE@1s 10.9 m vs 1 m bar — the trend metric for the remainder of the run |
| D3 | BLOCKED | multi-step imagination below persistence at this stage; the doctrine refuses the ratio |

Spectral diagnostics as in §4. Interpretation, honestly bounded: the core planning mechanism —
action-conditioned imagination that separates candidate actions — is established at a fraction of
training on real data; metric decoding and multi-step imagination remain open and are exactly what
the remaining 83 % of training and gates D1/D3 will decide. No driving-competence claim is made or
implied by decode gates.

**Inference efficiency (measured, step-6500 weights; commodity RTX 4060, fp32, batch 1, pinned
numerics):** one full decision tick — encoding the current frame stack plus a batched K = 9
imagine-and-select tactical pass — costs **15.1 ms p50 (17.2 ms p95) at 1.08 GB peak VRAM**,
i.e. ≈ 66 Hz *before any* TensorRT or quantization work, against a 10–20 Hz operative
requirement. Batching amortizes candidate evaluation almost entirely (K = 9 select 5.7 ms vs
6.1 ms for a single predictor pass) — the millisecond-planning property of §3.4 measured rather
than asserted.

**Self-knowledge, first controlled measurement (step 6500):** on 23 matched pairs of the *same*
synthetic scene rendered under clear vs degraded weather, one-step imagination error is higher
under degradation in 16/23 scenes (median paired shift +1.6; sign test p ≈ 0.047) — while every
*unpaired* comparison sits at chance and a diagonal-Mahalanobis latent detector is dominated by
within-corpus route shift. The familiarity signal exists but is weak and confounded this early;
the paired protocol is pre-registered for re-measurement at 50 % and 100 % of training. Gate D8
proper runs on real never-trained OOD probes.

### 7.1 Decision-grade diagnosis at ≈30 k, and two corrected errors

At step 27 000 (≈30 k, 90 %; a memory-cap crawl on the record pod was stopped at 90 % and the
checkpoint frozen), on the exact training val corpora with the *route-resampled* D1 protocol
(mean ± 95 % CI over 8 episode splits — a single split's ADE swung 5.2→11.5 m on identical
information, so single-split D1 is split-luck): **D1 6.44 ± 0.55 m FAIL** (camera unit,
necessary-not-sufficient), **D2 0.864 dir-acc / 0.971 P4 PASS**, **D3 imagined-vs-oracle 1.30
(K-step-improved from ~4×)**.

**The decisive diagnostic (proofs, not narrative).** A baseline + decode-ladder + error-localization
probe established, on real data: (i) the model is **10–15× worse than constant-velocity everywhere,
including straight highway** (2.75 m vs 0.18 m ADE@1s) — not a curve-only capability gap but a
failure to read its own trajectory; (ii) the deficit splits into a **2.4× readout/route-generalization
gap** (held-out MLP 3.89 m vs oracle-in-distribution 1.65 m) *and* a **representation floor** (oracle
1.65 m ≫ CV 0.28 m — even perfect decode cannot recover metric trajectory from the frozen latent);
(iii) mechanistically, the JEPA objective made latents predictable and rankable (D2 passes) but never
demanded metric ego-motion — motivating the action grounding of §3.7 as the pre-registered fix
(target: oracle < 1.0 m, beat CV on straights, D2 ≥ 0.80).

**Two basic errors, found and excluded/corrected by evidence.** (a) *Camera-geometry integrity audit.*
Every candidate corruption of the mixed corpus was tested; all excluded (pose↔frame lag 0, action
units/handedness consistent, no representation collapse) **except one, confirmed**: PhysicalAI's
120° front camera is an **f-theta fisheye** (real focal ≈ 926 px @1920) that `calib.py` canonicalized
with the *nominal rectilinear* focal (554 px), so PhysicalAI frames trained at **~1.6× the intended
zoom** (achieved f_eff ≈ 431 vs the intended 266 shared with comma2k19) — and PhysicalAI is **60 %
of the mix**, with no corpus conditioning, so identical ego-motion mapped to ~1.6× different pixel
motion. The corrected f-theta canonicalization restores f_eff = 266 (verified 437→266; actions/poses
unchanged, only frames), and all three arms retrain on the corrected geometry. (b) *Resolution.* A
degradation sweep (256→64 px) leaves ADE flat, so input resolution is **not** the binding constraint
on the current model (the bottleneck is grounding, per the diagnostic) — re-tested after grounding.

**Reference arms (encoder axis, comma-val probe, pre-correction).** The frozen-DINO REF-A decoded
ego-geometry *worse* than the from-scratch encoder (pool-adapter ADE@1s 17.0, grid-adapter 20.2 vs
the main encoder's 7.0–8.5), first evidence that task-specific from-scratch SSL out-grounds web
pretraining at this scale/task; the grounded, corrected-geometry three-way (main vs REF-A vs REF-B)
is the H1/H4 evidence table, in progress.

## 8. Roadmap

Completion of the first run and the full gate ladder; closed-loop evaluation (CARLA harness) for
D4–D6 including opponent-derived weak-spot scenarios (a scenario database built from documented
competitor failures, with per-scenario excellence as an explicit leaderboard section); the
frozen-encoder comparison arm (H4); data-efficiency slope experiments toward the 1000× thesis (H7);
modality steering driven by imagination uncertainty (H2); and NAVSIM/Bench2Drive entries once
closed-loop gates pass.

## References

(Formal bibliography at LaTeX export; the working citations live in
`TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` and the dated research notes: LeJEPA
arXiv:2511.08544; JEPA generalization theory arXiv:2606.27014; V-JEPA-2 arXiv:2506.09985; LAW
arXiv:2406.08481; World4Drive arXiv:2507.00603; HiT-JEPA arXiv:2507.00028; GAIA-2 arXiv:2503.20523;
Drive-JEPA arXiv:2601.22032; ALPS-4B transfer study `Ressources/AD_TRANSFER_RESEARCH.md` v1.1.)

---

### Changelog
- v0.1 (2026-07-08): initial living version — architecture, math, doctrine, theory grounding,
  step-5000 preliminary results.
- v0.2 (2026-07-12): added §3.7 (action grounding across abstraction levels — metric inverse
  dynamics + forward metric consistency as proprioceptive self-supervision, with the operative/
  tactical/strategic hierarchical extension, H18) and §3.8 (REF-A/REF-B reference architectures);
  §7.1 decision-grade ≈30 k diagnosis (metric-grounding gap: 10–15× worse than constant-velocity,
  oracle ceiling 1.65 m), the confirmed f-theta focal error (PhysicalAI 1.6× over-zoom, corrected)
  and the resolution-not-binding sweep. Actions confirmed to enter the predictor as FiLM-conditioned
  inputs whose latent consequence is predicted (§3.7).
