# TanitAD: A Data-Efficient, Hierarchically-Imagining, Self-Supervised Driving Stack with Built-In Self-Knowledge

**Status:** living paper, v0.6 (2026-07-25). Maintained per D-020: every gate evaluation appends
results; every accepted decision that changes the method updates §3–§5. Source of truth is this
Markdown; LaTeX export is a release step. Honesty rule (P8): no number appears here without its
instrument rows in the referenced experiment record. *(Version note: the status line was left at
v0.4 while the §9 design round was already logged as v0.5 in the changelog; this results round is
therefore v0.6, and the numbering is not silently reused.)*

**Authors:** Sayed Bouzouraa; TanitAD autonomous research system.

---

## Abstract

End-to-end driving models have won the architecture argument but retain four structural weaknesses:
they are data-hungry, opaque, unaware of their own limits, and expensive at inference time. We
present TanitAD, a driving stack built around a from-scratch, fully self-supervised hierarchical
latent world model (the *4B architecture*: operative, tactical, strategic, and fallback brains) that
attacks all four weaknesses simultaneously. The model (**263.4 M parameters measured** — `total_model`
263,442,838, `trainable` 277,404,073, `MODEL_REGISTRY §1.2`; the *design budget* every arm is matched to
is 261 M, D-008) trains on tens of hours of
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
models' sample-efficiency advantage. *Second-round update (v0.3):* after a causal speed-grounding
fix — feeding the measured ego-speed v₀ as a third action channel — the flagship arm became the
program's first to beat constant-velocity extrapolation on every held-out open-loop metric
(ADE@2s 0.628 ± 0.055 vs 0.825), the gain causally attributed by paired A/B (+2.21 m, win-rate
83.8 %); yet a two-parameter kinematic oracle (CTRV, **0.523** — see §7.2 on the superseded 0.544) still tops the open-loop table — an
ego-status shortcut we replicate from the literature and adopt as the honest bar, all open-loop
numbers remaining weak claims under the program's standing rule (arXiv 2605.00066, §7.2).
*Third-round update (v0.4):* at the completed 30 k checkpoint the flagship is the program's **first
sub-floor arm** — ADE@2s 0.452 ± 0.031, below the best-of-3 kinematic floor (0.500), CTRV (0.523)
and a learned ego-status ceiling (0.574) — resolving the v0.3 pending verdict in its favour. More
importantly, a causal panel establishes that the model **genuinely predicts scene physics on the
windows that require it**: on windows where a kinematic oracle diverges from the realized future
(upcoming turns/brakes), the model beats CTRV by +0.80 m and that *entire* advantage is vision —
mean-replacing the scene inverts it to −0.53 m (vision effect **+1.32 m, CI [+1.04, +1.64]**,
separated), the advantage grows monotonically with divergence, upcoming road curvature linearly
decodes from the latent (R² 0.25 vs 0.03 ego-only), and occluding the road ahead perturbs the
prediction 1.6× more than the periphery. The honest counterweight, reported with equal weight: this
in-distribution prediction does **not** transfer to beating the kinematic floor on *unseen* corpora
(comma2k19 real-highway ADE@2s 0.849 vs floor 0.372; Cosmos synthetic 0.583 vs 0.358) — error
roughly doubles and the anticipation advantage collapses (0.80 → 0.15 m) — though vision-ablation
*still* degrades the out-of-distribution prediction (comma +0.27 m, CI-separated), so the model
reads the unseen scene but cannot net-beat a very strong highway floor: it is partly
distribution-fit, not yet corpus-general. *Fourth-round update (v0.6):* the program's crux question
this round was whether a planner can be coupled to the world model at all. Four arms that warm-started
a *new* anchored-diffusion planner onto the already-converged 30 k world model all failed — the best
reached ADE@2s 0.852 [0.75, 0.98] against a 0.60 bar while its WM-integrity canary rose from 0.452 to
0.70+ — and a ~0-GPU **gradient-geometry pre-probe** refuted the obvious repair: at the coupling seam
cos(g_wm, g_plan) = **+0.0043** over 512 windows, so gradient surgery would strip ~2 % of the planner
gradient and is a no-op. That measurement redirected the program to **co-evolution from random
initialisation** — v1's own recipe — and at full coupling (λ_plan = 1.0) the world-model canary
*descends* (15.674 at random init → ≈1.4) while ~~held-out ADE@2s falls to ≈0.48 at 40 % of training~~
— **corrected 2026-07-25**: the ≈0.48 was the trainer's dense-20 statistic; the first decision-grade eval
(step 15 k, episode-cluster bootstrap) reads **0.5839 [0.4962, 0.6821]**, paired **Δ +0.1568
[+0.0630, +0.2504] CI-separated *behind* v1** while CI-separated *ahead* of both trivial floors —
its 10 k gate returning CONTINUE. **The scientific point is that planner–world-model interference was a
warm-start artifact, not an intrinsic conflict between prediction and control** — reported as
in-progress, on the trainer's in-loop evaluation, with the formal gate deferred. Two additive
directions closed cleanly: a 3.77 M planner trained by analytic gradients **through the frozen** world
model reaches 0.599 m — statistically indistinguishable from that model's own oracle-action ceiling
(+0.194 [−0.045, +0.448]) and flat under an 11× capacity scaling — so its residual is *aleatoric*, a
degradation-free fallback rather than a contender; and a closed-loop recovery-augmentation lever that
looked promising at n = 12 **reversed sign at n = 40** and was withdrawn. A new **real-footage
log-replay** closed-loop instrument (on-policy observation-OOD 1.02–1.19×, against a photoreal
reconstruction's 3.2–3.75×) confirms across three independent instruments that a 104 M anchored-diffusion
reference arm **out-drives the flagship's own supervised tactical head** closed-loop (ADE@2s 0.564
[0.452, 0.676] vs 1.488 [1.329, 1.647]; departure rate 0.0134 vs 0.0318) and decomposes that deficit as
**longitudinal, not lane-keeping**. Finally the data thesis gained its first evidence: our own frozen
encoder plus a multi-domain inverse-dynamics head pseudo-labels action-free video well enough that
downstream world-model pretraining captures **96 %** of the value of real labels on proxy corpora and
**109 % speed / 71 % yaw** on the actual target, and an 80-clip Creative-Commons YouTube pilot lifts
downstream parity-validation speed R² from −0.520 to +0.563 (≈92 % of the real-label ceiling) —
directional, not decision-grade. We
describe the architecture, its mathematical grounding, an
instrument doctrine for honest measurement that caught three silent measurement hazards in its first
week and has since been extended by three more failure classes it learned the hard way, and the
falsifiable gate program by which every claimed edge will stand or fall.

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
pooling; A7) produces the compact state z_t = r(f_θ(x_t)) ∈ ℝ^{2048}. The **design budget** is
261 M parameters (D-008); the **instantiated** flagship measures **263,442,838** (`MODEL_REGISTRY §1.2`;
the per-module split below is the design allocation, which differs from the measured per-module counts —
e.g. encoder 87,121,280 measured vs 99.5 M allocated): encoder 99.5 M (ViT, d = 768, 14 blocks, patch 16, LayerNorm only — batch-free
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

**Making the top-down seams load-bearing (H26).** A hierarchy only earns its parameters if the
upper level's decision measurably steers the lower one. We instrument every seam with a
conditioning ablation (FiLM-off vs on) plus a per-window content test (real upper-level signal vs a
constant/mean surrogate) — a standing "hierarchy panel" (§7.5). The first verdict exposed a
concrete failure mode: the intent→operative FiLM injection was *magnitude-swamping* the action
signal (`intent_proj` norm ‖31.4‖ vs action-embedding ‖28.3‖), so an ungated intent term either
corrupted the operative rollout or, by design, had to be zeroed to be harmless. The architectural
fix is a **ReZero-style learnable scalar gate on the intent term, initialised at 0.1**
(`predictor.py`), so the seam starts near-inert and training decides how much intent to admit rather
than fighting a swamped FiLM. This is the same "a conditioning term must not swamp the base signal"
lesson applied to the REF-C anchored decoder's maneuver→anchor priors (§3.8).

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

*Measured limitation (2026-07-17, §7.2 — σ-dissipation):* the calibration property is currently
established at **one imagination step only**. Under blind autoregressive rollout the trained field
grows *more* confident as fidelity decays to chance; every deployed use of the variance field
(including the §3.6 self-monitor) is therefore capped at 1-step until multi-step belief-rollout
training or a parallel-horizon decode closes the gap.

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

**The loss-balance tension (why the encoder became an odometer).** Grounding buys the metric at a
cost we now measure and name. In the joint objective the *supervised* metric-motion terms —
hierarchical grounding (`invdyn` 2.0×3 + forward-consistency `fwd` 1.0×3 = 9.0) plus the
waypoint/maneuver/route/inverse heads — outweigh the *self-supervised world-model core* (JEPA
prediction 1.0 + K-step rollout 0.5 + goal-latent 0.5 + SIGReg 0.1 ≈ 2.1) by **roughly 5 : 1**, and
per §3.2 there is *no stop-gradient*: the real-pair inverse-dynamics term reads the encoder latent
directly (no `.detach()`), so its gradient reshapes the trunk into a metric odometer. This is the
mechanistic root of a measured symptom — the trained encoder *redundantly re-encodes* the fed
proprioception (in-latent yaw R² 0.89) and spends little capacity on scene, so vision contributes
only ≈ 12 % of the prediction (§7.5) and cross-corpus transfer is fragile (§7.4). One honest caveat
keeps this from being over-read: loss *weights* are not steady-state gradient *magnitudes* — the
MSE-type grounding terms shrink toward zero as they fit (`fwd`-ADE reaches 0.033 m), so the 5 : 1
ratio characterises the *early trajectory that set the encoder's character*, not the late-training
budget; a bare re-weight is therefore a weak lever, and the coupling itself is the thing to cut.
The surgical rebalance (v2.x candidate, gated, test-pinned) is a **straight-through gradient-decouple
(scale α = 0.25)** applied to the latents *only where they feed the real-pair "odometer" term*
(`grounding.invdyn`, mass 6.0), leaving the forward-consistency rollout term — the one that
*produces* the protected 0.033 m operative metric — together with JEPA and SIGReg fully attached.
It removes the dominant trunk-reshaping pressure without touching the term that owns the metric, adds
no learned parameter, is forward-pass-identical, and is gated at a 5 k mid-checkpoint against a
`fwd`-ADE non-regression and a rising vision-use target. Its deeper motivation — that supervised
heads *distort* a good representation and hurt out-of-distribution — is the two-stage question taken
up in §8.

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
- **REF-C — anchored multi-modal decoder (tactical-head axis, H19).** Our first tactical head was a
  single unimodal waypoint regressor; it reproduced the known failure mode of that design — a 3.4 m
  tactical error (§7.5) that is *mode-averaging* over a multi-modal future. REF-C replaces it with a
  **DiffusionDrive-style anchored decoder** (arXiv 2411.15139): a fixed vocabulary of trajectory
  *anchors* — furthest-point-sampled over our own ego-frame future trajectories, not k-means, so the
  74 %-straight skew does not collapse the vocabulary — whose queries cross-attend the feature map and
  emit a per-anchor confidence (nearest-anchor cross-entropy, winner-takes-all offset regression)
  refined by truncated diffusion off one weight set (`steps = 0` reproduces the classifier
  byte-for-byte). It is **multi-modal by construction**, validated live on REF-B v2 (n_modes ≈ 22–28),
  and it is the graft point for our hierarchy: the tactical maneuver-logits reweight the anchor
  confidence priors through a learned maneuver→anchor projection (**validating H19** — our maneuver
  vocabulary *is* the anchor set, propose-and-refine), with metric grounding replacing top-1 selection
  by scoring anchors on logged data. It is run as a 3-size scaling study (55 M / 104 M / 252 M) on the
  identical corpus.

**A label-integrity fix underneath the tactical head.** The strategic/tactical pseudo-labels were
first reverse-engineered from *future dynamics* — the route class thresholded net heading change over
a 15–25 s window (|Δψ| > 45°) — which is circular (the fed nav command *is* the derivation),
degenerate on 74 %-straight highway, and conflates a gentle road curve taken at speed with a tight
junction turn (Δψ = κ·v·t mixes curvature, speed and time). A label-quality harness measured the
conflation: v1 **mislabels 24.5 % of road curves as route-turns**. The v2 derivation is
*curvature-relative* (decide on κ = Δψ/Δs = 1/R, speed-invariant), which drives that conflation to
**0.0 %** on the known-semantics corpus and explicitly flags ambiguous junctions/forks rather than
forcing a binary — a prerequisite for any honest maneuver-accuracy or route claim.

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

**The estimator correction (2026-07-20), and what it does and does not invalidate.** Every `± CI95`
in §7.1–§7.5 above — and in v0.1–v0.4 of this paper — was produced by a block historically labelled
*"8-split episode-disjoint jackknife."* It is neither a jackknife nor a valid standard error: it draws
eight independent random 20 % holdouts from the same 40 validation episodes and reports 1.96·std/√8
over overlapping estimates, so it measures **split-selection noise, not model uncertainty**. Measured
across ten arms it is **1.28–2.06× too narrow** (median 1.51×); a coverage simulation gives **62.3 %**
against a cluster bootstrap's 93.8 % (target 93–97 %). The **decision-grade interval is the
episode-cluster bootstrap** over the 40 validation episodes (2000 resamples), and for two arms scored on
the same windows the **paired** bootstrap — never a combination in quadrature. All numbers from §7.6
onward carry the corrected estimator, named inline.

> ⚠️ **CORRECTION (2026-07-25) — this paragraph previously asserted that "the point estimates and every
> qualitative verdict in §7.1–§7.5 stand; the *widths* quoted there are the deprecated statistic." That
> assurance is WITHDRAWN: it was wrong, and it was load-bearing.** The deprecated block does not only
> mis-state widths — **its central value is a mean-of-split-means, so it moves the POINT ESTIMATE too.**
> Measured 2026-07-25 by recomputing **27 arms** from the raw per-window dumps (`windows_*.pt`, 881
> windows / 40 episodes; pipeline validated bit-for-bit against `CI_RECOMPUTE_2026-07-20.json`, 10/10
> arms exact): headline `ade_0_2s` shifts **−6.67 % to +11.69 %**, and the shift is **bidirectional — 11
> arms inflated, 16 deflated, none unchanged** — so no legacy point estimate may be assumed conservative.
> On paired deltas the distortion reaches **×−4.15, including a sign flip**, and on hierarchy seams
> **×3.3**. Widths are **1.107–3.100× too narrow (median 1.499×)** over 27 arms; the "1.28–2.06×" above
> came from only ten and was under-sampled. **Under the corrected estimator the cross-arm ranking changes
> in 10 of 27 positions.** Note the internal tension this correction resolves: the sentence immediately
> following already conceded that the split-*mean* compresses between-arm gaps — that concession was
> right, and it was incompatible with "the point estimates stand."
> **Consequently every `± CI95` *and every central value* in §7.1–§7.5 is provisional pending
> re-derivation** (per-arm corrections: `…/incoming/2026-07-25-jack-blast-radius/`). What survives
> unchanged, checked explicitly: **no gate verdict flips** — the v3enc RESTART, the v4.1 FAIL and the
> v4.2 kill all clear the bias by 10–100× — and the AlpaSim n=12, n=40 low-OOD, departure cross-fit and
> `IMAGINATION_HELPS` panels never used the deprecated path at all.

A second consequence, which the correction above generalises: the split-*mean* compresses between-arm
gaps (a two-arm difference read 0.006 m under the split-mean and 0.044 m on the full set), so ranking
claims must come from the paired test, not from comparing two split-means.

**Three failure classes added by measurement, 2026-07-21 → 07-24.** Each is a rule we now run
*before* a claim, and each was earned by a wrong claim that a cheap check would have caught. They are
reported here as method, not as errata, because each generalizes beyond the experiment that produced it.

- **I8 — power before closure.** An effect measured on one underpowered held-out split does not
  survive its own sign. A closed-loop recovery-augmentation lever measured **+0.0089** (a departure-rate
  *reduction*) on a 12-episode held-out set and **−0.0302 [−0.0595, −0.0088]** — a 3.3× *increase* —
  under a 2-fold cross-fit that puts all 40 episodes held-out (§7.7). Consequence: **every closed-loop
  effect at the ~1 pp scale goes through a full-corpus cross-fit**, and a claim that a *direction is
  closed* gets the cheapest metric-or-power check first — in one session four separate "this direction
  is closed" claims were reopened by a follow-up costing zero GPU-hours.
- **I9 — a privileged-input arm is not a headroom estimate.** A CEM search over the frozen world model
  scored 0.132 m against a feedforward planner's 0.599 m and was quoted as "4.5× of planning headroom."
  The search arm *optimizes against the expert's realized future*; the deployable planner cannot see it.
  The contrast therefore varies two things — planner quality **and** access to the future — and the
  deployable version of the same search (a learned value model, no ground-truth future) scores **1.016**,
  CI-separated *worse* than feedforward (§7.7). Rule: **name the input asymmetry before quoting a gap as
  headroom**; an oracle arm bounds the instrument, not the achievable policy.
- **I10 — verify presence with the tool that owns the fact.** Two high-value modules were declared
  "stranded outside the main tree" on the strength of a file listing that is sorted by modification time
  and truncated at 100 results; the main-tree copies sorted past the cut. Both were present, newer, and
  more complete. This is the two-probe rule (§Operating standard) applied to *presence*: absence, and
  equally the presence of a problem, must be established with the tool that owns the fact — `git
  ls-files` over a listing, a real write test over a filesystem report, the process table's owner over a
  name match.

**Two instrument definitions this round pinned down, because both had already produced a wrong number.**
(i) **A latency figure without its definition, hardware, checkpoint and corpus is not a figure.** Two
"ticks" coexist in this system — a *decision tick* (encode one frame + a batched K = 9 imagine-and-select)
and a *planning tick* (encode an 8-frame window + 20 sequential predictor steps + metric accumulation,
the path that actually produces the scored trajectory). They differ in five dimensions at once and by ~9×;
quoting the first as the system's latency propagated for two days (§7.10). (ii) **Exact-path L2 ADE
mis-scores benign closed-loop recovery.** A tolerance-band variant, `band_ade2d(b) = mean over waypoints
of max(0, ‖pred − gt‖ − b)` at b = 1.0 m (half a lane half-width), showed that a measured "ADE cost"
vanished (CI ∋ 0) for three of four configurations and shrank 74 % for the fourth — the exact-path metric
overstated the trade ~4× by charging in-lane wiggle as error (§7.7). Closed-loop levers are now gated on
the band metric alongside the departure rate. **A third, data-side:** the validation split
`physicalai-val-f1b378f295ae` was found to share **78 % of its episodes (62/79) with the parity training
set**; the evaluation harness now **refuses it in code** rather than documenting the hazard, and points
the caller at the clean 40-episode split. Instrument rule I7 (task-identity fingerprints) is thereby
extended from *corpus identity* to *train/eval disjointness*, enforced mechanically.

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

263.4 M parameters measured (261 M design budget); bf16 autocast with SIGReg computed in fp32; gradient accumulation (micro 32 × 2,
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

> **Supersession note (v0.3, 2026-07-14 reset).** §7.1 stands as the record of the *speed-blind*
> first round. Its headline deficit ("10–15× worse than constant velocity", oracle ceiling 1.65 m)
> was subsequently localized primarily to a **missing input** — no arm was fed the current
> ego-speed v₀, so each was asked to infer absolute scale from monocular appearance (barely
> decodable from a frozen encoder, probe R² = 0.61) — and is superseded by the speed-grounded
> second round (§7.2), whose flagship beats CV on every held-out metric at 19 k. The
> pre-correction REF-A probe numbers above are likewise void under the reset protocol (post-reset
> REF-A: 2.14 m fwd-ADE at 30 k, plateaued — the frozen-encoder ceiling). §3.7's pre-registered
> target "beat CV on straights" is met — on all strata and metrics — and the program's bar has
> since been *raised* to the stricter CTRV kinematic oracle (§7.2).

### 7.2 Second round: speed grounding, the first CV-beating arm, and the kinematic-oracle bar (reset 2026-07-14; results 2026-07-16/17)

All numbers in this section are open-loop and therefore **weak claims** under the program's
standing rule (open-loop ⊥ closed-loop, arXiv 2605.00066); the capability arbiter remains
closed-loop D4–D6. Records: TanitEval (the canonical eval substrate on the eval pod,
`/root/taniteval/results/`), `Project Steering/FLEET_REVIEW_2026-07-17.md`, intakes
`2026-07-15-baseline-floor` and `2026-07-17-openloop-l2-egostatus-shortcut`, and
`Benchmarks & Eval/LEADERBOARD.md`.

**The missing input, found and fixed.** A four-ablation localization on the plateaued
frozen-encoder arm attributed 71–83 % of its residual trajectory error to speed/scale *magnitude* —
not rotation (trajectory shape was good) and not imagination. The cause was architectural, not
representational: no arm received the measured ego-speed v₀ as input. The fix is proprioceptive in
the sense of §3.7 — v₀ enters as a third action channel (with jerk continuity and auxiliary
ego-motion supervision), no new labels — and was validated in isolation before committing the
retrain (fwd-ADE 3.73 → 0.83 m, speed decodability R² 0.61 → 0.965). All three arms restarted
2026-07-14 from scratch on the identical canonical corpus.

**First arm past constant velocity.** At step 19 k of 30 k the speed-grounded flagship measures
**ADE@2s = 0.628 ± 0.055 m** on the held-out canonical validation set (TanitEval grounded-rollout
protocol) — the program's **first checkpoint to beat the constant-velocity baseline on every
reported metric**: ADE 0.628 vs 0.825, FDE 1.317 vs 1.708, RMSE 0.942 vs 1.541, miss@2 m 0.180 vs
0.313. The gain is *causally* attributed, not narrated: a paired A/B against a same-architecture
arm trained without the speed channel gives **+2.21 m mean improvement [2.04, 2.39] with an
83.8 % per-window win rate** (paired bootstrap). No driving-competence claim follows.

**The kinematic oracle tops the table — and that is itself the finding.** A two-parameter physics
extrapolation, CTRV (constant turn rate and velocity from v₀ + ψ̇₀ alone, zero pixels), scores
**0.523** on the same protocol — above every learned arm at this round. This replicates on our corpus the
ego-status-shortcut result (AD-MLP / BEV-Planner line, arXiv 2312.03031): our validation is ≈74 %
straight driving (73.9 % measured — coincidentally identical to nuScenes' 73.9 %), so open-loop
L2-class metrics are dominated by kinematic extrapolation. A second, independently implemented
instrument line inside the program converged on the same oracle (CTRV 0.545, different corpus and
stratification) — replication, not coincidence. Consequence, pre-registered: **beat-CTRV is the
honest open-loop bar.** The flagship stands **0.092 above it at 19 k** on the full set (0.6152 vs 0.523)
and closing; whether it crosses by 30 k is the pending verdict. *(Resolved in §7.3: at the completed
30 k checkpoint the flagship crosses — full-set 0.4271, below both CTRV and the best-of-3 floor.)*

> 🔧 **RECONCILIATION (2026-07-26) — this paragraph and the abstract previously read CTRV = 0.544 while
> §7.3 and `MODEL_REGISTRY §0.3` read 0.523. One quantity, two published values.** Both are the same
> oracle on the same 881-window canonical val. **0.523 is the one to quote**: it is what
> `MODEL_REGISTRY.md:64` — the program's only quotable source for model facts — has carried since
> 2026-07-18, and it is what §7.3, the OOD table and every downstream document use. **0.544 is a
> superseded 2026-07-17 reading** (`Project Steering/FLEET_REVIEW_2026-07-17.md:17,50`), kept named here
> rather than deleted so the earlier round stays traceable. ⚠️ **What changed between the two readings
> could NOT be determined** — TanitEval did not enter git until 2026-07-20 (`a91bef8`), so no code
> timeline exists for either measurement. *Settled by:* re-running `kinematic_floor` on the canonical
> val and persisting the CTRV per-window array (≈ 0 GPU; the emitter is `bench.py:485-511`/`:558`).
> ⚠️ Do **not** confuse either with the **0.545** cited two sentences above: that is the *comma2k19*
> CTRV (`…/incoming/2026-07-17-openloop-l2-egostatus-shortcut/results_openloop_l2.json`,
> `comma_highway.L2_pointwise.ctrv.2s`) — a different corpus, which is exactly what makes it a
> replication rather than a restatement. **All three trivial bars (best-of-3 0.5005, CTRV 0.523,
> ego-status ceiling 0.5735) are full-set means by construction** (`bench.py:511`, `:558`), so unlike
> the arm numbers they are untouched by the §5 estimator correction — but for the same reason, every
> legacy "beats the floor" verdict compared a *split-mean* arm against a *full-set* bar. Re-checked
> like-for-like, the flagship's crossing survives (0.4271 vs 0.5005/0.523/0.5735).

**How much of it is vision? (imagination panel).** A 2×2 vision-ablation plus latent-fidelity
panel separates what each arm's predictions owe to pixels versus integrated dynamics. The
trained-encoder flagship: **vision_use 12.9 %, imagination 8.7 %, latent-fidelity gain +0.054** —
genuine but modest visual world-modelling on a corpus this kinematic. The frozen-encoder REF-A
arms: 3.4 % / 1.5 % — functionally, dynamics integrators. The split variable is
**trained-vs-frozen encoder**: the H4 axis measured functionally rather than through decode
probes alone.

**Planning brains are speed-starved; decode, not imagination, is the tactical bottleneck.** First
tactical/strategic evaluation: maneuver accuracy 0.61 held-out, turn recall 0.75, tactical
waypoints 3.38 m. The v₀ channel currently reaches only the operative predictor, so the upper
brains still carry exactly the deficiency the reset fixed below them. Meanwhile goal-latent
imagination is strong — cosine 0.885 between imagined and realized post-maneuver latents: the
imagination is close; *decoding it to metric space* is where the 3.38 m is lost. This sharpens,
and is consistent with, the A13 pattern of §7: latent-space selection quality keeps outrunning
metric readout.

**σ-dissipation: recursive imagination collapses with false confidence.** Rolling the
1-step-trained H15 field blind (autoregressively) on real held-out data: fidelity falls
0.357 → 0.011 — chance — by k = 4 while the predicted log-variance *shrinks* (−7.79 → −8.55) and
beliefs collapse toward an attractor (inter-sample cosine 0.21 → 0.57). The field becomes more
confident as it decays; the §3.5 calibration property is established at one step and measurably
false under recursion. Freezing the k = 1 imagination instead holds ≈0.25 fidelity flat across 8
horizons — **the defect is the recursion, not capacity**. Conservative corrections adopted: the
deployed self-monitor (§3.6) consumes 1-step imagination error only; multi-step belief-rollout
training and a parallel-horizon (non-autoregressive) decode are the pre-registered fixes.
(Record: `Architecture & Inference/Implementation/belief_rollout_diagnostic/`.)

**Frozen-encoder family is second-order; data scale dominates.** A matched comparison on 320
episodes: I-JEPA features beat DINOv2 at every horizon (fwd-ADE 3.194 vs 3.796 at 15 k steps) at
5.5× the encode cost — and both overfit at this corpus size. In the frozen regime, encoder family
matters less than data scale; this refines, and does not overturn, §7.1's REF-A finding.

> 🟥 **DISCLOSURE (2026-07-26) — this comparison is INADMISSIBLE as published, and the source of truth
> says so.** `MODEL_REGISTRY §2.2` marks the I-JEPA arm's canonical-val number **UNUSABLE**: *"320-ep
> variant … Canonical val **80 % LEAKED** into its train set → guard excludes; clean number lives on the
> f1b378 val"* (`taniteval/registry.py`, carried as open risk **R8**, `MODEL_REGISTRY:1556`). The
> paragraph above discloses the **overfitting** but not the **leak**, and the leak is the disqualifying
> defect: 3.194 is a partly-in-sample number, so **"I-JEPA beats DINOv2" is an overfit-regime ranking
> with data binding, not a feature-quality verdict.** The DINOv2 control arm
> (`refa-dino320-4brain-speed-15k`) is matched on episodes but the leak is not symmetric evidence — it
> inflates whichever arm saw the val. **The claim is withdrawn pending re-evaluation on a clean split;
> until then it must not be quoted in either direction.**
>
> ⛔ **THE REMEDY THIS PARAGRAPH USED TO PRESCRIBE WAS THE DISEASE — corrected 2026-07-28 (C48).** It
> read *"pending re-evaluation on the clean `f1b378` val (pod3 gates)"*. **`f1b378` is not clean: 62 of
> its 80 episodes (77.5 %) are bit-identical to parity-train episodes** by content (sha256 of raw
> `poses` and `frames_u8`). ⚠️ **This paper already said so 200 lines earlier** — §5 records the
> 78 % overlap AND that the harness *"refuses it in code"* — so the same document both **refused** the
> split and **prescribed** it. §5 also already names the answer: the **clean 40-episode split**,
> `physicalai-val-0c5f7dac3b11` (0/40 and 0/600 by content). The quoted `registry.py` sentence above is
> left verbatim **because it is a quote of the defective source**; that source is fixed separately.
> *(This does not
> touch §7.1's REF-A finding, which rests on the canonical-corpus `refa-dynin-30k` / `refa-dinov2` arms
> — 3.0471 and 2.1675 full-set, both clean.)*

**REF-B at 6 k: the E2E baseline is strong and yaw-blind.** Direct waypoint regression reaches
0.868 (its RMSE already beats CV) — but its rotation-gain in curves is 0.03: the encoder is
yaw-blind (yaw-rate probe R² = 0.11), so the arm rides the same ego-status shortcut as the
oracles. The patch now training (refbpatch): a fixed-**distance** path head (TF++-style
path/speed decoupling), v₀-dropout 0.5, an auxiliary yaw head, and turn-weighted loss — keeping
REF-B the strongest fair imitation opponent for gate D4 rather than a strawman.

### 7.3 The flagship at 30 k: the first sub-floor arm, and a causal proof of genuine scene prediction (2026-07-18)

The flagship's first training run completed at 30 k. On the canonical held-out validation set
(TanitEval, route-resampled protocol, n = 881 windows / 40 episodes) it is the program's **first
arm below every trivial bar**: ADE@2s **0.452 ± 0.031 m** (plain mean 0.427), under the best-of-3
kinematic floor (0.500), the CTRV oracle (0.523) and a learned ridge ego-status ceiling (0.574).
Against the 19 k checkpoint the gain is +0.188 m at an 81.2 % per-window win rate, with miss@2 m
0.180 → 0.060 (3×); every curvature stratum improved and the turn strata now *beat* the floor
(skill-vs-floor straights 1.03 at-floor, gentle 0.68, sharp 0.60). One stratum remains above floor —
top-decile high speed (1.785) — the open weakness dissected in §7.5. These are open-loop numbers and
therefore weak claims (arXiv 2605.00066); the checkpoint is pushed to a gated model repo.

The decisive result is not the aggregate but a **causal panel that isolates what the model owes to
seeing** (TanitEval generalization panel, `taniteval/generalization.py`; ledger 2026-07-18). It
stratifies windows by how far a kinematic oracle (CTRV) diverges from the realized future — the
windows where anticipation, not extrapolation, is required — and ablates vision by mean-replacing the
scene:

- **Vision-tied anticipation (headline).** On high-divergence windows (upcoming turns/brakes CTRV
  cannot extrapolate) the model beats the CTRV oracle by **+0.796 m on 94 %** of them, and that
  *entire* advantage is vision: mean-replacing the scene **inverts** it to −0.529 m (worse than
  CTRV). The vision effect is **+1.325 m, CI [+1.04, +1.64]** (separated from zero). The anticipation
  is *read from the scene*, not extrapolated from dynamics.
- **Monotone in divergence.** The advantage over CTRV rises monotonically across divergence quartiles
  (Q1 −0.372 → Q4 +0.796) — the model beats the oracle *most* exactly where dynamics fails, and
  correctly defers to near-inertial extrapolation where CTRV is already optimal.
- **The latent encodes the road, not just the ego-state.** Upcoming road curvature linearly decodes
  from the pooled latent at **R² 0.254 vs 0.031** for an ego-kinematics-only control (+0.223) — the
  representation carries scene geometry beyond the fed proprioception.
- **Learned physics, not memorized paths.** Predicted trajectories are physically feasible 95.9 % of
  the time vs 97.1 % for ground truth.
- **It reads the road ahead.** Occluding the road-ahead pixels shifts the prediction **1.60× more**
  than occluding an equal-area periphery patch, with dynamics inputs held fixed.

*Honest limits, reported with the result:* on low-divergence / near-inertial windows the model
dynamics-guesses (correct — CTRV is optimal there); it **loses to CTRV on the top-10 % high-speed
windows (−0.617 m)**, the §7.5 weakness; and the anticipation lead-time test is inconclusive at this
stage (an instrument limit under action-grounded rollout, not a null result). The claim this panel
supports is precise and bounded: *in-distribution, the model genuinely predicts scene physics on the
windows that require it* — a causal, vision-attributed claim, distinct from and stronger than the
aggregate ADE, and distinct from any closed-loop driving-competence claim (still unmade).

### 7.4 Out-of-distribution: the honest limit (2026-07-19)

With comma2k19 and Cosmos pixels staged on the eval pod, the same 30 k flagship was run on **unseen
corpora** (ledger 2026-07-19). The in-distribution proof does **not** transfer to beating the
kinematic floor out-of-distribution:

| corpus | regime | n | model ADE@2s | best-of-3 floor | aggregate verdict | per-window win-rate vs floor |
|---|---|---|---|---|---|---|
| PhysicalAI | in-distribution | 881 | **0.427** | 0.523 | **beats** floor | 49.7 % |
| comma2k19 | OOD, real highway | 2176 | **0.849** | 0.372 | **loses** | 17.5 % |
| Cosmos | OOD, synthetic | 92 | **0.583** | 0.358 | **loses** | 29.4 % |

Two things drive the gap, and both are honest. First, comma2k19 and Cosmos are highway-dominated, so
their CTRV floor is *very* strong (≈ 0.37 m) — a hard bar precisely because the ego-status shortcut
(§7.2) is near-optimal there. Second, the model's own error roughly **doubles** out of distribution,
and its high-divergence anticipation advantage — the §7.3 headline — **collapses from 0.80 m to
0.15 m**. Path feasibility on sharp-curvature windows falls from **97.8 % in-distribution to 62.8 %
OOD**.

The nuance that keeps this from being a flat negative: **vision-ablation still hurts out of
distribution** — mean-replacing the scene degrades the comma2k19 prediction by +0.27 m (CI-separated).
So the model *reads* the unseen scene; it simply cannot convert that reading into a net win over a
highway floor this strong. The honest verdict: **the in-distribution genuine-prediction is real, but
the model is partly distribution-fit — not yet a corpus-general world model.** This is exactly the
target of the v2/v3 vision-reliance and loss-rebalance levers (§3.7, §8), for which these numbers are
now the pre-registered OOD baseline; 24 clear/degraded Cosmos weather *pairs* are also staged, making
a true weather counterfactual a modest panel addition.

### 7.5 Hierarchy, tactical decoding, and the longitudinal weakness

**Hierarchy panel (H26).** The standing seam instrumentation (§3.3) returns a mixed, honest verdict
that sharpened with training (ledger 2026-07-18). *Grounding dominance (H18) is confirmed and grew:*
the grounded operative rollout (0.615 m) beats the ungrounded tactical head (3.43 m) by 5× at 19 k,
and the gap widened to Δ **+2.9568 m** at 30 k — ~~Δ 2.70 m~~ **corrected UPWARD 2026-07-25** by the same
estimator migration that retracted the ctx→tactical seam below; this leg **survives and strengthens**,
needing an **8.65× interval widening** to un-separate against a worst-measured 3.10×. *Cross-layer
consistency holds:* maneuver and trajectory
agree at 0.872 (κ 0.612). *But top-down conditioning was initially inert or harmful:* at 19 k the
intent→operative seam was magnitude-swamped (§3.3; ungated `intent_proj` ‖31.4‖ vs action ‖28.3‖ —
the deployed rollout was intent-free by design), and per-window intent *content* was inert on both a
trained-encoder (flagship) and a frozen-encoder (REF-A) arm. At 30 k ~~**one seam flipped to
load-bearing** — ctx→tactical now shows `content_matters = true` (vs-mean maneuver Δ +0.044,
CI-separated)~~ — **RETRACTED 2026-07-25, see below**; intent→operative remains harmful when ungated
(confirming the ReZero fix of §3.3 is the right lever), and nav→strategic is still a pure command-echo
(route-from-vision skill 0.0, an open v3 target).

> ⚠️ **RETRACTION (2026-07-25): the ctx→tactical seam was NOT load-bearing — the estimator manufactured
> it.** Migrating the hierarchy panel off the deprecated statistic (§5 correction) shows the published
> **Δ +0.044 is an artifact of the mean-of-split-means; the true full-set paired delta is +0.0148**
> (×2.97; independently reproduced at ×3.28 on the v4.2b arm and ×1.76 on the v1 artifact). Under the
> correct estimator the seam **fails all three gates on the point estimate alone** — 0.0148 < 0.02
> (maneuver-acc), 0.0050 < 0.01 (cosine), 0.0437 < 0.05 (ADE) — so no widening of intervals is required
> to reject it. **The honest count is therefore 0 of 3 seams load-bearing at 30 k, not 1 of 3:**
> intent→operative harmful, nav→strategic a command echo, ctx→tactical retracted. *(Reading note for
> anyone re-deriving this: the deprecated predicate was ONE-SIDED — a naive two-sided port flips the
> **harmful** intent seam to "load-bearing". Read `separated_positive`, not `separated`.)*
> **What this does and does not license.** It removes the only measured leg that supported the
> hierarchy claim as published, and that claim is withdrawn here. It is **not** evidence that hierarchy
> does not help, because all three seams were measured under conditions that make hierarchical value
> undetectable: the strategic route *target* is a lookup of the route *input* (`route_target =
> _NAV_TO_ROUTE[nav_cmd]`, so route CE reaches exactly 0.0 and `route_skill` is 0.0 **by construction**,
> at 27 % coverage), and the scored 0.45 m path does not traverse the hierarchy at all. A separate
> hierarchy-supporting result **survives the same correction and strengthens**: grounding dominance
> corrects **up** to Δ **+2.9568 m** and would need an 8.65× widening to un-separate, against a
> worst-measured 3.10×. The claim is re-opened as untested, not closed as refuted; the pre-registered
> conditions and the six discriminating predictions that would settle it are in
> `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/01_EXECUTION_PLAN.md` Part A.

Read plainly: at 30 k the 0.45 m comes from the operative predictor plus grounded
step-readout; the upper brains *cohere* but do not yet *drive* the operative — making the seam-scaling
levers, not more capacity, the path to the "hierarchy is dominant" claim.

**The tactical decoder and the high-speed longitudinal weakness.** The unimodal tactical head's 3.38 m
error (§7.2) motivated the REF-C anchored-decoder replacement (§3.8). Decomposing the flagship's one
remaining above-floor stratum with a decoupled long/lateral panel (`taniteval/pathspeed.py`,
2026-07-18) localizes it precisely: **89 % of the 2 s squared error is along-track (speed), not
lateral.** At high speed the model **over-predicts speed by +0.66 m/s** (longitudinal RMSE 1.38 m vs
CTRV 0.077 m; lateral only 0.63 m), and the error compounds over the horizon (per-step displacement
error 0.07 → 0.22 → 0.51 → 0.91 m). The mechanism: the model applies the expected-speed-*change*
behaviour learned from common low-speed accelerate/brake events to high-speed cruise, where constant
velocity is near-optimal — **it plans the path well but the speed poorly**, a fault hidden inside
aggregate ADE until the decoupled metric exposed it. A targeted longitudinal fix (along-track
up-weighting + speed-stratified sampling + an anti-overshoot term) is measurement-gated on this
panel, deferred behind a check of whether the v2 rebalance already relieves it.

### 7.6 Coupling a planner to the world model: a warm-start artifact, not an intrinsic conflict (2026-07-22 → 07-25)

The §9 design — planning over the world model rather than reading it through supervised heads — was
built. **flagship-v4** keeps v1's trunk verbatim (from-scratch ViT encoder + readout 87.1 M,
action-conditioned operative predictor 96.6 M, H15 imagination field 22.1 M), deletes v1's three
supervised policy heads, and adds three *planners*: a **strategic planner** with its own predictor
operating in a compressed 128-d subspace of the state (5.15 M), and **tactical and operative
anchored-diffusion planners** (DiffusionDrive-style, 256 anchors, 9.77 M each — the §3.8 decoder
promoted from a reference arm to the flagship's own proposal mechanism), plus factorised
lateral/longitudinal/distance heads. A λ_plan curriculum couples the planner gradient into the shared
trunk. Measured by instantiation the trainable total is **≈ 247.9 M — about 30 M *smaller* than v1**,
because three regression heads cost more than three planners.

**The instrument that makes this line legible is a world-model-integrity canary:** the *plan-free*
operative rollout ADE@2s — the world model scored exactly as §7.3 scores it, with the planner removed
from the path. v1's value is 0.452. It is the only quantity that separates "the planner is bad" from
"the planner is *making* the world model bad," and it is the reason the failure below is attributable.

**Four warm-started arms, one lever each, all fail — and they fail in two distinct ways.**

| arm | the one lever | held-out ADE@2s (episode-cluster bootstrap) | WM canary | reading |
|---|---|---|---|---|
| v4 | hot trunk, lr_trunk 3e-4 | killed at ~3,500, never gated | 0.452 → ≈1.3 *(in-loop)* | the LP-FT learning rate alone degrades the WM, with the planner gradient clamped to zero |
| v4.1 | lr_trunk 3e-5 | **0.8522 [0.7468, 0.9800]** vs a 0.60 bar | **0.4599 (PASS)** | WM healthy, **planner starved** — a controller bug decayed the planner gradient to ≈off by step 2,000 |
| v4.2 | canary controller floored at 0.25 | 0.9869 [0.8795, 1.1088] at step 4,000 | **0.7222 (FAIL)** | protecting the planner costs the WM |
| v4.2b | floor lowered to 0.15 | not gated | 0.697 at step 4,000, held at floor | floor-tuning exhausted between 0.15 and 0.25 |

v4.1 is the informative one. Its primary fails outright — the interval sits entirely above the bar, at
roughly twice v1's 0.427 — while its canary passes, and the paired decomposition says exactly where the <!-- lint-ok: "v1's 0.427" here is v4.1's ratio to the deployed full-set number, NOT the retracted 07-25 C1 claim (which was flagship-v4-fromscratch's trainer-log "~0.48 descending to v1's 0.427"); that one is retracted in the §7.6 block below. -->
loss lives: speed error is CI-separated *worse* than constant velocity (Δ −0.366 [−0.491, −0.245]) and
worse than hold-v0 on steady cruise (Δ −0.559 [−0.648, −0.469]), whereas **speed-decoupled path geometry
CI-separatedly beats CV** (+0.115 [+0.017, +0.240]). The fault is the planner's longitudinal selection,
not the world model — the same longitudinal signature as §7.5, now inside a planner. *(Formally the gate
renders `INCOMPLETE`: three of eight pre-registered kill secondaries have no emitter anywhere in the
codebase. Substantively the primary fails without ambiguity. Both readings are recorded; neither is
allowed to stand in for the other.)*

**The ~0-GPU experiment that redirected the program.** The obvious repair for "two objectives fighting
over one trunk" is gradient surgery (PCGrad-style projection at the seam). Before spending a GPU-day on
it we measured the seam's *geometry*. On the v4.2b checkpoint, over **512 windows** of the clean
validation split, at the single `states` seam where the planner loss enters the shared trunk:

| quantity | value | reading |
|---|---|---|
| mean cos(g_wm, g_plan) | **+0.0043** (sd 0.064) | the two gradients are **orthogonal**, not opposed |
| fraction of windows with cos < 0 | 0.479 | half-and-half — the sign is noise, not conflict |
| mean fraction of ‖g_plan‖ removed by one-sided PCGrad | **0.0224** (median 0.0) | surgery would strip ~2 % — a no-op |
| ‖g_wm‖ / ‖g_plan‖ | 0.125 / 0.030 | the planner gradient is already ~4× the weaker of the two |

Two independent arguments then close the door on the whole coupling-attenuation family: the removable
conflicting component is negligible (2.2 %), *and* the scalar floor had already attenuated the **entire**
planner gradient to 15 % while the canary still degraded to 0.70 — surgery passes 98 % of it, i.e. *more*
coupling than the floor, so it cannot possibly hold the world model better. **Whatever degrades the
canary is neither the planner gradient's direction nor its magnitude.** The remaining candidate is what
the four arms share and v1 did not: a trunk that was **already prediction-converged** when a randomly
initialised planner was attached to it.

**Co-evolution from random initialisation.** `flagship-v4-fromscratch-30k` is the same architecture and
the same command with the trunk randomly initialised — v1's own recipe, in which world model and decision
layer were never separately converged. The canary baseline has to be recalibrated first, and this is the
kind of detail that silently invalidates a gate: from random init the plan-free rollout error starts at
**15.674**, not 0.42, so the warm-start arms' ≤ 0.55 bar is unreachable in 10 k steps and meaningless
here. The pre-registered read is the **descent trajectory**, not the level. At full coupling
(λ_plan = 1.0) the canary descends 15.674 → 2.59 (step 7,000) → **1.371** (step 9,000), where every
warm-start arm instead rose; ~~held-out ADE@2s falls 0.531 (9 k) → 0.4825 (10.5 k) → 0.4788 (11.5 k)~~
with miss@2 m 0.169 and best-in-fan oracle 0.242, on the clean 881-window split. The 10 k gate returned
**CONTINUE**, restarts 0.

> 🔧 **SUPERSEDED 2026-07-25 by the arm's first decision-grade eval** (`RETRACTION_LOG` 07-25, class
> **C1**). The `0.4788`-style figures above are the **trainer's dense-20 in-loop statistic**; the
> v1-comparable **4-gate-waypoint** reduction on the *same forward pass* reads differently, and at step
> 15,000 the harness (episode-cluster bootstrap, B = 2000, pinned by recomputing v1 to **0.4271**
> exactly) measures **ADE@2s 0.5839 m [0.4962, 0.6821]** — paired against v1, **Δ +0.1568 m
> [+0.0630, +0.2504], CI-SEPARATED *BEHIND* v1**, not ≈0.05 m short of it. **A metric NAME is not a
> metric DEFINITION**: both numbers are "val ADE@2s" on the same checkpoint and the same clean split and
> differ only in the waypoint set averaged over. **What stands on that same eval:** the arm beats both
> trivial floors CI-separated (CV 0.8377 by +0.2538; hold-v0 0.7876), and the WM canary at 2.0739 against
> the planner's 0.5839 is the expected from-scratch co-evolution signature — **a descent position at 15 k
> of 30 k, not a verdict.** The §7.6 architectural conclusion (coupling failure was a warm-start artifact)
> is unaffected; only the *level* is.

*Honest limits, carried with the result and load-bearing:* **(i)** these from-scratch numbers are the
**trainer's in-loop evaluation** on the clean held-out split, not the canonical `eval_flagship_v4.py`
harness — the formal eight-metric gate is deferred behind an artifact-relay quota block. Under the
program's own C1 rule a trainer number is not an eval number, and the one time we forgot, an in-loop
value read ~10 % optimistic against the harness. **Treat the level as provisional and the trend as the
claim.** **(ii)** the run is at roughly 40 % of its 30 k schedule; no final verdict is claimed here.
**(iii)** the arm differs from v4.2b in two flags rather than one (initialisation and the λ floor),
though the floor is *measured* inert from random init. **(iv)** an earlier reading of this same curve —
"canary descending, co-evolution confirmed" — was retracted the same day: it rested on a single
evaluation delta, and the next point bounced. The claim above rests on the shape of the whole descent
through the coupling ramp, which is the only thing a noisy per-point canary supports.

**What the line establishes, stated precisely.** Planner–world-model interference in this architecture is
a **warm-start artifact** — the cost of attaching an untrained decision layer to a converged predictive
trunk — and not an intrinsic conflict between the prediction objective and the control objective. The
generalizable instrument is the pre-probe itself: **measure the seam's gradient geometry before buying a
surgery**, because a near-orthogonal seam makes the entire projection family a no-op for the price of a
few minutes of a free GPU. At step 15 k of 30 k the co-evolved arm measures **0.5839 [0.4962, 0.6821]**,
**CI-separated behind** v1's 0.4271 (paired Δ +0.1568 [+0.0630, +0.2504]) and CI-separated ahead of both
trivial floors — so whether it *closes* that gap by 30 k is the open question and the next gate; the
architectural question — can a planner be coupled at all — is answered yes.

### 7.7 Two additive directions: a frozen world model's ceiling is aleatoric, and a closed-loop lever that closed

Two directions were run alongside the flagship line to bound the design space rather than to win it. Both
returned clean negatives, and both produced a measurement lesson worth more than the result.

**D1 — the frozen world model with a learned planner.** v1 is frozen end-to-end (encoder, readout,
predictor, step-readout, all `requires_grad = False`) and a **3.77 M** planner is trained *only* by
backpropagating open-loop ADE **through** the frozen world model — the Dreamer/SHAC analytic-gradient
pattern, with our own model as the differentiable simulator. By construction the world model cannot
degrade. On 12 held-out episodes / 265 windows, episode-cluster bootstrap:

| arm | what it is | ADE@2s | CI95 |
|---|---|---:|---|
| oracle-action ceiling | frozen WM rolled under **ground-truth** actions | **0.4045** | [0.310, 0.514] |
| **W — analytic gradient through the frozen WM** | planner → actions → frozen rollout → ADE | **0.5989** | [0.374, 0.854] |
| hold-v0 / CV | trivial floors | 0.7883 / 0.8463 | — |
| B — action behaviour-cloning (no WM in the loss) | planner → actions, MSE to ground-truth actions | 1.0001 | [0.697, 1.354] |
| F — direct decode off the frozen state | planner → 20 waypoints, no predictor | 3.649 | [2.632, 4.723] |

Paired on the same windows: **W − CV −0.2474 [−0.505, −0.034]** (separated), **W − B −0.4012 [−0.717,
−0.128]** (separated), **W − F −3.0501** (separated), and **W − oracle +0.1944 [−0.045, +0.448] — not
separated.** A 3.77 M planner driving a frozen world model lands within bootstrap noise of feeding that
world model perfect actions.

Two mechanisms fall out. First, **arm F relocates the frozen-encoder ceiling**. F = 3.65 m reproduces the
program's static-latent probe (3.89 m, §7.1) and sits squarely in the REF-A band — so the ceiling
documented in §7.1/§7.2 and closed as H4 is a ceiling of **static decode off a JEPA latent**, not of
freezing as such: the metric information the static latent lacks is present in the *action-conditioned
rollout*, and routing through the frozen dynamics recovers it 6.1×. Second, **capacity is not the lever**:
scaling W's own head family 11× (3.77 M → 30.8 M → 42.6 M) gives 0.599 → 0.601 → 0.599, none
paired-separated, while bigger query-decoder planners *overfit* to 0.82–0.86 on 8,803 training windows.
The residual above 0.404 is therefore **aleatoric** — the driver's future intent is not determined by the
past — and no feedforward capacity can reduce what is unknowable.

*The retraction that is the point of the direction (I9).* A CEM search over the same frozen model scored
**0.132** and was quoted as "4.5× of planning headroom." That search **peeks at the expert's realized
future** as its cost; the deployable planner cannot. The discriminating experiment was to build the
deployable version — a learned value model trained on the true cost of explored candidates, then CEM
ranking by that value with no ground-truth future. It scores **1.0162 [0.809, 1.273]**, +0.4173 [+0.237,
+0.605] CI-separated **worse** than plain feedforward, with a within-window rank correlation of 0.613
against the true cost. The mechanism is clean: a value model learns E[cost | state], whose minimiser is
the mean trajectory the feedforward planner already produces, and CEM then adversarially exploits its
errors. Every deployable route hits the same ~0.60 wall (feedforward 0.599, scaled 0.601, distilled 1.40,
learned-value 1.02). **Verdict: a ~0.60 m, degradation-free fallback — not a contender.** *(A learned
value may still pay in a genuine closed-loop setting where the ego controls the future; that is a
different evaluation and needs a simulator.)*

**D2 — closed-loop recovery augmentation, and why it did not survive its own metric or its own n.** The
lever fine-tunes the REF-C anchored decoder on renderer-free recovery scenarios (the ego is perturbed off
the recorded path and must return), gated by an encoder-integrity canary. Two cheap follow-ups settled it
in opposite directions on the same day, and both are instrument findings.

*(i) Most of the measured cost was a metric artifact.* Re-scoring the **existing** rollouts under the
tolerance band `band_ade2d(1.0)` instead of exact-path L2 makes the ADE penalty **vanish (CI ∋ 0) for
three of four configurations** and shrink 74 % for the fourth — the exact-path metric overstated the
trade roughly 4× by charging benign in-lane recovery as error. The band is not vacuous: the un-fine-tuned
base itself scores 0.1997 > 0 under it.

*(ii) The benefit did not survive full power.* The departure-rate improvement measured **+0.0089**
on a 12-episode held-out set. Under a 2-fold cross-fit that puts **all 40 episodes** held-out — each
scored by a model that never trained on it — it becomes **−0.0302 [−0.0595, −0.0088]**, CI-separated in
the *wrong* direction: the fine-tuned arm departs the lane **3.3× more often** (0.0436 vs 0.0134), and is
worse under the fair band metric too (−0.3655 [−0.482, −0.262]). *Stated confound:* each cross-fit fold
trains on 20 episodes against the original's 28, so part of the reversal is reduced training data rather
than pure statistical power. It does not rescue the lever — the cross-fit is the standard unbiased
full-corpus estimator, and a lever whose benefit requires both a larger training fold *and* a favourable
single split to appear at all is not robustly promotable.

**Not promotable — and three things that survive it.** The **machinery** (renderer-free recovery
augmentation, an on-policy low-OOD training harness, the tolerance-band metric, the encoder-integrity
canary) is sound and reusable. **REF-C's encoder is safely fine-tunable**: at a *material* move
(feature cosine 0.9658, relative L2 drift 0.263 — ~3× the previous probe) the canary still holds
(maneuver agreement 0.9861, route agreement 0.9167), which de-risks encoder-in-the-loop work generally —
notably it is *not* the v4 world-model degradation hazard. And the two measurement lessons (I8, and the
band metric) now gate every closed-loop claim the program makes.

### 7.8 Closed-loop benchmarking: a low-OOD instrument, and a reference arm that out-drives the flagship

Open-loop does not predict closed-loop (§7.5, and the standing rule of arXiv 2605.00066), and until this
round our only closed-loop numbers came from an imagination-in-the-loop harness that is self-referential
and from photoreal scene reconstructions whose own fidelity confounds the result. Both gaps were closed
by instruments, and the result they agree on is uncomfortable.

**The confound that had to be removed first.** Running the reference arm **open-loop on the
reconstructions** — a control, not a result — measured ADE@2s **1.52 m against its own real-footage
0.473 m, a 3.21× shift** over 4 scenes / 288 predictions. Any closed-loop failure rate on those scenes
therefore measures *model × reconstruction fidelity*, not the model. That control retracted a headline
("REF-C fails half of closed-loop") and is now a standing prerequisite: **run the open-loop-on-the-same-
input control before attributing a closed-loop failure to a model.**

**The new instrument: real-footage log-replay.** The ego is driven on-policy through *recorded* frames,
each tick's observation synthesized by warping the real frame to the ego's actual deviation from the
recorded path. It has no map and no agents, so it cannot emit off-road or collision — but its
**observation-OOD is measured, small, and bounded**. On a 12-episode envelope with paired
episode-cluster bootstrap, the flagship's prediction is statistically *flat* out to a **2.0 m lateral**
excursion (3 m: +0.066 [+0.010, +0.138] is the first separated rise) and separates on yaw only at **3°**
(+0.017 [+0.001, +0.034]), rising monotonically to +0.055 at 12°. Every separated rise is ~17–20× smaller
than the gap to the reconstruction source. On-policy through the actual rollouts, both arms sit at
**1.02–1.20×** OOD (longitudinal stratum 1.018 / 1.004 — effectively OOD-free; junction 1.196 / 1.152)
against the reconstruction's flat **3.75×**.

**The comparison, at n = 40 episodes / 881 windows, identical windows, paired bootstrap:**

| | flagship v1 (deployed strategic→tactical head) | REF-C base (104.2 M anchored diffusion) | paired Δ | separated |
|---|---|---|---|---|
| closed-loop ADE@2s (m) | **1.488** [1.329, 1.647] | **0.564** [0.452, 0.676] | **+0.924** [+0.781, +1.065] | yes |
| corridor departure rate @1.75 m | **0.0318** [0.0152, 0.0531] | **0.0134** [0.0059, 0.0223] | **+0.0184** [+0.0077, +0.0328] | yes |
| peak cross-track error (m) | 0.764 [0.530, 1.060] | 0.442 [0.314, 0.585] | +0.321 [+0.193, +0.495] | yes |

This is the **third independent confirmation of the same ordering**, and the first that is neither
underpowered nor confounded: n = 1 (retracted as a lucky scene) → a paired 12-scene reconstruction suite
(pass 8/12 vs 2/12, mean score 0.496 vs 0.066, paired Δ −0.430 [−0.646, −0.215], sign test 8–0, p = 0.008;
**collisions tied 1–1**) → this n = 40 real-footage run. Measured through a *completely different*
instrument at 3× lower observation-OOD, the ordering is not an artifact of reconstruction fidelity.

**What the lane metric buys is a decomposition, and it is the paper's own §7.5 signature seen from the
other side.** In the longitudinal stratum (374 windows / 24 episodes) **both arms keep the lane nearly
perfectly** — departure rates 0.4 % and 0.04 % — yet the flagship's ADE is **4× the reference's** (1.455
vs 0.354, paired +1.101 [+0.906, +1.284]). The deficit is **longitudinal, not lane-keeping**, exactly the
89 %-along-track failure signature of §7.5, now confirmed on-policy. In the junction stratum (182 windows
/ 22 episodes) the flagship departs the corridor **~2.3× more often** with a peak cross-track error of
2.372 m against 1.458 m — clearing a full lane. Its tactical head is a **high-deviation planner** whose
failure mode is leaving the road, not hitting something (an independent per-plan deviation measure reads
1.12 vs 0.34).

*Honest limits.* This measures **lane-keeping drift at low observation-OOD, explicitly not off-road
departure or collision** — the map-free, agent-free source is structurally unable to emit those. It is a
within-source **relative** comparison of two deployed decoders through a shared pure-pursuit + kinematic-
bicycle controller, at the 256²/f_eff = 266 cache resolution, and a different controller changes the
absolute deviations. And it exposes a gap we now believe is close to fundamental with the assets we have:
a real off-road/collision rate needs a **map plus reactive agents**, which means a renderer, and every
renderer available to us sits at ~3.2× OOD — while low OOD requires real footage, which has no agents.
Resolving both at once needs a lower-OOD reactive renderer; until then it is a genuine trade-off, and the
honest move is to name which half a given number measures. *(An on-policy training attempt on this
instrument was run and came back bound: the base arm rarely departs, so the objective starves. The
instrument is a good measuring device and a poor training signal.)*

The reading that matters for the architecture: **a deployed anchored-diffusion planner out-drives the
flagship's own supervised tactical head closed-loop**, which is the same verdict §7.5 and §9.4 reached
open-loop — the heads are a lossy readout of a good world model — now confirmed in the regime that
arbitrates. It is a *planner* comparison, not a world-model comparison; the flagship's operative rollout
is not in it. That is precisely why §7.6 exists.

### 7.9 Data: the corpus we actually trained on, and the first evidence for the scaling thesis

§4 argues that the *structure* of driving, not corpus scale, is the primary unexploited resource. That
argument is only honest if we can say what our corpus contains and can show that the cheap-data path
(H7 — mining action-free video with inverse dynamics) is real rather than assumed. Both were measured
this round.

**The corpus, exactly (MEASURED by loading every cached episode).** The canonical parity training set is
**13.13 driving hours** — 2,376 clips × 19.9 s at 10 Hz, **472,627 frames**, **406,099** unique windows
(a count that independently reproduces the training run's own log). At 30 k optimizer steps and an
effective batch of 64 that is 1,920,000 presentations = **4.73 epochs**: the deployed flagship saw its
corpus fewer than five times. The kinematic scenario mix is **lane_keep 59.6 % · accel 13.2 ·
brake_stop 12.9 · turn_right 7.4 · turn_left 6.9** (14.25 % turns), speed strata ~46 % highway / ~46 %
city / ~8 % stopped. Two structural gaps matter more than the aggregate: only **42.6 % of clips contain
any turn at all**, and **semantic scenarios — traffic lights, roundabouts, merges — are 0 %-labeled**,
i.e. invisible to every loss in §3. This is the concrete content behind §7.4's "partly distribution-fit"
and behind the strategic layer's command-echo behaviour (§7.5): a route head cannot learn junction
topology from a corpus in which most clips never turn.

**A balanced 50 h corpus, built by selection.** A v2 corpus was designed and built entirely *within* the
same NVIDIA PhysicalAI-AV source — 197 egomotion chunks already on hand yield **18,731 moving clips /
104.6 h**, a 2× headroom — with "augmentation" defined as **distribution-balancing by selection**
(oversampling rare classes), never synthetic perturbation. The balanced 9,000-clip selection hits its
targets: turns **14.25 → 28.0 %**, lane_keep 59.6 → 45.0 %, junction-clip presence **37.7 → 61.3 %**. It
is stored as a JPEG-compressed 256 px cache — **982 GB → ~25 GB** at full parity resolution, with frames
verified **bit-identical** to the parity decode path (only JPEG quantization differs). It **breaks parity
with the sacred key by design** and is for the next generation; the running arm finishes on the 13 h set,
so no cross-arm comparison is contaminated. *Honest limit: kinematic selection cannot buy semantic
scenarios — they remain 0 %, and closing them needs the separate vision-language labeling track.*

**H7 — pseudo-labeling action-free video, and the first end-to-end evidence.** The mechanism is v1's
**frozen** encoder plus a small **multi-domain inverse-dynamics head** that reads a latent window and
emits ego-motion; unlabeled video is pseudo-labeled by it, a world model is pretrained on the
pseudo-labels, and the benefit is measured downstream against real labels. Three measurements, in
increasing order of what they license:

1. **Direct label accuracy, stated plainly and unflatteringly.** Extracted-vs-ground-truth speed R²
   **0.62–0.66** cross-domain, longitudinal trajectory R² **0.60**, **yaw ≈ 0 cross-class (weak)**,
   acceleration unusable and dropped. On its own this reads as a "no."
   > 🔴 **Label-protocol correction, 2026-07-27 (C29) — the "yaw ≈ 0 cross-class" is a fact about the
   > LABEL, not the transfer.** The cross-class corpus is comma2k19, whose heading is `arctan2` of the
   > ENU velocity and is **undefined at standstill**: MEASURED, **26.27 %** of comma frames below
   > 0.5 m/s are physically impossible and **0.000 %** above it (PhysicalAI: **zero in every bin**, so
   > every PhysicalAI/rig number in this section is unaffected). The `≈ 0` above was scored with
   > `heading_repair` **OFF** and **no `v_min` gate**; it is **left in place and marked STALE-PENDING**
   > because no repaired measurement exists on that substrate. On the substrate where the repair *was*
   > measured (v3 val split, `heading_repair` ON, `v_min` 0.5, 2,992 comma windows, **nothing
   > retrained**) the same deployed head reads comma `yaw_rate` **R² +0.3308** — up from **+0.0114** on
   > the identical windows with the repair off — and a *retrained* head reads **+0.679**.
   > ⭐ **Stated against interest:** comma-only, MAE falls **42.5 %** but **medAE moves only −1.1 % and
   > nMedAE gets 8.0 % WORSE** (Spearman ρ flat, +0.001). **The repair fixes the tail and the summary
   > statistic, not typical accuracy.** The three-measurement argument of this paragraph is unchanged —
   > it never rested on the yaw channel. Inventory: `TanitAD Research Hub/Benchmarks & Eval/
   > Implementation/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`.
   > 🔴 **AMENDED 2026-07-27 (`anchor-settlement`, class C43): `+0.3308` is WITHDRAWN.** *(Block
   > above keeps its text and date.)* Settled BY CONTENT — sha256 of the raw `poses` float32 bytes
   > **and** of the raw `frames_u8` sensor bytes, never filenames: **2 of the 22 comma val episodes
   > it was measured on are bit-identical to 2 of that deployed head's own 40 comma TRAINING
   > clips**. Without them it reads comma yaw **R² −0.746 (CI [−1.574, −0.177])**, and its
   > published interval **[−1.2982, +0.7047]** already spanned zero. ✅ `+0.679` is **not**
   > withdrawn (no leak) but reads **+0.3038 (CI [+0.054, +0.479])** on the 20 content-clean
   > episodes. ⇒ **comma2k19 yaw is TESTABLE and this deployed head does not do it** — on its own
   > held-out comma clips with a fully admissible label it reads **R² −0.288**, ρ 0.211,
   > nMedAE 2.36. **The `≈ 0` cell stays STALE-PENDING**, the three-measurement argument is still
   > unaffected, and PhysicalAI is unaffected (re-measured: `n_pai_changed = 0`, +0.903482
   > bit-identical). Record: `…/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md`.
2. **The downstream ablation overturns the proxy.** In the regime that actually matters — abundant
   pseudo-labeled data, scarce real labels — pseudo-label pretraining captures **~96 % of the real-label
   pretraining benefit**: fraction-of-ceiling 0.965 (speed) / 0.984 (trajectory) on a cross-*class*
   corpus and 0.960 / 0.917 on a same-class held-out rig, beating the random-init floor on **all 8 seeds
   across both domains**. On the **actual parity target** with 4 seeds and readable yaw it is stronger
   still: **109 % of ceiling on speed, 107 % on trajectory, 71 % on yaw**, every seed CI-separated from
   the floor. The mechanism is not mysterious — pretraining needs labels that convey *structure*, and
   tolerates noise that a direct-accuracy proxy penalizes.
3. **The real-video pilot — the one read our own data cannot give.** 80 **Creative-Commons** dashcam
   clips (8,960 windows), face/plate/body blurred at full resolution before downscaling, only latents,
   pseudo-labels and pointers persisted. Pretraining on them lifts downstream parity-validation
   **speed R² from −0.520 ± 0.200 to +0.563 ± 0.047** (3 seeds; clip-cluster bootstrap CI excludes zero
   on *every* seed, gaps +1.37 / +0.88 / +1.05), **yaw R² 0.55 → 0.75**, and **halves trajectory ADE,
   12.82 → 6.31 m** — **≈92 % of the real-label pretraining ceiling. The YouTube domain transfers.**

*Honest counterweight, with equal weight.* The pilot is **directional, not decision-grade**: 80 clips
against a planned ~300, 3 seeds, unknown camera intrinsics (a nominal-HFOV crop is a *named* domain-shift
source that biases apparent-motion scale), and no ground truth of any kind on the source. Because the
floor is negative, the raw gap is inflated — **the substantive claim is the ≈0.92 fraction of ceiling**,
not the R² delta. Speed and trajectory are trustworthy; yaw rides a 15-clip real fine-tune. Operationally,
clean continuous Creative-Commons forward dashcam is **scarce** (80 clips from 31 producing channels, 63
tried, ~339 candidates), so scale is a licensing question, not a technical one.

**And the negative that constrains the design.** A dedicated from-scratch encoder with explicit GAIA-2
camera conditioning, built precisely to give the inverse-dynamics head a rig-robust latent, **failed**:
held-out cross-rig speed R² **−0.667** against a pre-registered +0.9 bar, and — the paired, same-regime
contrast that matters — **worse than simply freezing flagship v1's own encoder (+0.657)**, CI-separated
on 3 of 4 arms. Explicit camera conditioning at this scale does not close the cross-rig problem; the
deficit is upstream of rig-invariance, in representation quality. The positive discovery inside the
negative is that **v1's own trained encoder is the stronger cross-rig substrate** — though not uniformly
(−1.169 on a single-domain rig arm), so the problem is narrowed, not solved.

### 7.10 Beyond-ADE metrics, a traffic-light gap, and the deployment envelope

**The metric suite meets real models, partially.** The beyond-ADE suite (LAL anticipation lead, TMS motion
smoothness, OKRI kinematic-risk, CNCE compute-normalized capability, LOPS latent-planning stability) had
been validated only on synthetic fixtures. It now has **first real numbers**, measured over 30 comma2k19
validation episodes on a commodity RTX 4060: **decision-tick latency p50 14.331 ms** (encode 9.273 +
K = 9 select 5.058), **TMS median 0.0435** and **CNCE median 210,551**. Two disclosures travel with them:
the TMS figure scores the **expert log**, establishing a
reference band, *not our policy*; and CNCE's collision term is zero by log-replay construction while its
latency and parameter terms are weight-independent — so it is a real *architecture*-efficiency number,
not a driving number. The remaining suite (LAL/OKRI/LOPS, and TLC below) stays **renderer-gated**: the
simulator needed for rendered occlusion and signal geometry was confirmed absent at three separate probes.

> 🔧 **CONDITIONS, ADDED 2026-07-26 — and a correction to what was measured.** This paragraph previously
> attributed these numbers to *"the deployed 262.8 M architecture."* **Neither half was right.** Traced
> to its artifact,
> `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-24-traffic-light-scenario-metric/real_tms_cnce.json`
> (`exp: sc14-P2-real-tms-cnce-log-replay`; generator `real_telemetry_tms_cnce.py:109`), the measurement
> is on a **`base250cam` WorldModel, `params_billions` 0.2628 → 262.8 M**, instantiated fresh — **not the
> deployed flagship v1**, which measures **263,442,838 (263.4 M)**. Latency and the CNCE parameter term
> are weight-independent, so the number is a valid *architecture* read; it is not a read of the deployed
> checkpoint, and the coincidence that 262.8 M is also REF-B's count (`MODEL_REGISTRY:1481`) is exactly
> the kind of collision that makes an unlabelled param count unquotable. **Full conditions, none of which
> were stated:** RTX 4060 · **fp32 eager** (`WorldModel(...).to(device).eval()` — no autocast, no CUDA
> graph) · comma2k19 val, **n = 30 episodes** · log-replay · random-init weights.
> ⚠️ **This is the program's FOURTH live value for the decision tick** and it is the only one that did not
> appear in `MODEL_REGISTRY` — the others are **11.16 ms** (fp16 + CUDA-graph) and **17.75 ms** (fp32),
> both step-6,500/comma2k19/RTX 4060 (`MODEL_REGISTRY §1.2`), and the §7.4-era **15.1 / 17.2 ms** p50/p95
> at line 549. Against its own fp32 sibling the gap is the *architecture config*, not a regression:
> 14.331 (base250cam) vs 17.75 (step-6,500 model), same tick definition, same GPU, same corpus.
> The registry has been given the row (`§1.2`, decision-tick table note). **The program already retracted
> one latency figure for precisely this defect — a bare tick with no hardware, checkpoint, corpus or
> precision — and then published a second one the same way.**

**A traffic-light scenario and metric, because the corpus has none.** §7.9 measured that signalized
intersections are 0 %-labeled in our corpus; the evaluation side had the mirror gap — no traffic-light
scenario and no metric for handling one. Both are now built. SC-14 is a signalized approach with an
explicit per-step signal phase, and **TLC (Traffic-Light Compliance) = red_entry_gate × stop_quality ×
green_flow ∈ [0, 1]**, where `red_entry_gate ∈ {0,1}` is a hard legal barrier — **a single red-light
entry zeroes the entire score** — `stop_quality` combines stop-line margin with deceleration smoothness,
and `green_flow` penalizes phantom braking on a genuine green. It is discriminative by construction: a
design oracle implementing the §3.6/H9 hard rule barrier scores **TLC = 1.0**, while a soft-cost prior
that treats the signal as one term among many runs the red and scores **0.0** (peak deceleration 3.0 m/s²
against the barrier policy's 0.986). This is the H9 rule-compliance claim expressed as a scored scenario
rather than an assertion — but it is a **design oracle, not our model** (P8), and the model-side number is
renderer-gated with the rest.

**Deployment precision: FP16, and INT8 rejected on evidence.** A per-layer TensorRT benchmark on real
deployed weights settles the quantization question against the published ViT-INT8 folklore. On **latency**,
calibrated weight+activation INT8 is **2.1 % faster on the encoder and 2.1 % slower on the predictor** —
no win — and the per-layer profile shows why: INT8 adds a real reformatting/re-quantization tax at the
network boundary and redistributes time inside attention without shrinking it, reproducing the documented
TensorRT trap on our own architecture. On **accuracy**, every transformer block in both the encoder and
the predictor tolerates weight-only INT8 at isolated cosine ≥ 0.999999 — the blocks are essentially
immune — but the encoder's **un-normalized post-pool `readout_head`** collapses to cosine **0.566** under
weight+activation INT8 and accounts for nearly the whole blanket failure. And the failure compounds: per
single predictor call weight+activation INT8 looks near-perfect, yet **rolled out 20 steps on 880 held-out
windows it costs +0.0215 m ADE@2s, past the pre-registered 0.02 m falsifier**, with the degradation ratio
growing 27× from 0.5 s to 2 s. Weight-only INT8 passes cleanly (+0.0065 m). **FP16 is the deployment
precision.** The exported FP16 engine reaches encoder **1.205 ms** / predictor **0.666 ms** on an Ampere
proxy, with torch-vs-runtime parity ≤ 1.9e-6 — but *the proxy is a proxy*: TensorRT engines are not
portable across GPU architectures, and every real Orin/Thor throughput number is hardware-blocked, not
estimated.

**The tick, defined.** Composing four levers — capturing the 20-step rollout as a CUDA graph, caching the
encoder, FP16 weights, and dropping two unused horizon heads — takes the *planning* tick from 100.29 ms
to **18.75 ms p50 (5.35×)**, clearing the 10 Hz budget at p99 with ~5.3× headroom, at a maximum absolute
trajectory deviation of 0.024 m. Two cautions are part of the result: **the levers are sequenced, not
additive** (capture first — the other three are worth ~1.0× before it), and the earlier "levers compose
additively" finding from a 1-step select does not generalize to a 20-step rollout. Separately, an
8-candidate imagine-and-select fan costs **20.82 ms p50**, ~0.3 ms per marginal candidate, provided the
encoder runs once and broadcasts — which **refutes** the per-candidate-re-encoding arithmetic that had
projected 723 ms and nearly retired the planning thesis on a calculation.

### 7.11 The four-family doctrine pays for itself: a corpus arm falsified, compounding proven, rollout recovery measured, and the flagship's missing imagination found (2026-07-29 → 2026-08-02)

This round is dominated by a change in **what we measure**, ordered by the PI as binding: every
evaluation now reports four metric families — LONGITUDINAL (speed setting + distance keeping),
LATERAL (heading, curvature, yaw-rate, cross-track), TACTICAL (manoeuvre decision + goal), and
STRATEGIC (route/goal) — **in addition to** ADE, per family, never pooled, each with the paired
episode-cluster bootstrap. The instrument is `taniteval/four_families.py`. Within days it exposed a
sign flip, a decorative tactical brain, and a trade-off that an ADE column is structurally unable to
represent. ADE is retained; it is simply no longer allowed to stand alone.

**(a) The v2-corpus arm is falsified — and the experiment could not have said what it was built to
say.** The 50 h manoeuvre-balanced corpus (`physicalai-v2bal`, 9,000 clips) was trained to 30 k as
`flagship-v2corpus-30k`. Two audit findings preceded any score. First (**C64**), the corpus
re-selection did not inherit the previous generation's validation exclusion: **21 of the 40**
canonical validation episodes sit inside the new arm's training corpus, so the canonical surface is
void for it; scoring moved to the 19 leak-free episodes with v1 **re-scored there** (v1's published
0.4271 is the full-40 statistic and is not comparable). Second, the launch had passed both
`--v2-cache` (a data-loader flag) **and** `--v2` (a ten-lever architecture pack that also silently
forces `rollout_k` 12 vs v1's 4) — so the arm differs from v1 in corpus *and* architecture *and*
rollout horizon, and **no outcome is attributable to the corpus**. The state dict itself carries the
proof (`goal_traj_head`, `intent_gate`, an `anchor_decoder`; 286.3 M vs 276.9 M parameters). The
admissible claim is only: *the v2-line arm is worse*. On the 19 leak-free episodes (n = 418 windows,
paired episode-cluster bootstrap, B = 2000): v1 **0.393** [0.307, 0.493] vs v2corpus **0.575**
[0.429, 0.752]; paired Δ CI **[−0.221, −0.145]**, separated. A trainer preflight
(`_preflight_banner`) now prints the DATA axis and the ARCHITECTURE axis separately at every launch
and records them in `config.json`, so this conflation class cannot recur silently.

**(b) The four families say *why* it is worse, and ADE could not have.** Longitudinal: the two arms
fail in **opposite directions** — v1's signed speed bias is **+1.465 m/s** (runs ahead of the
human), v2corpus's is **−1.260 m/s** (lags). An absolute-distance scalar cannot represent a sign
flip. Lateral: v2corpus is uniformly ≈2× worse (heading 0.529° → 1.072°; curvature 0.00168 →
0.00313 m⁻¹; yaw-rate 4.99 → 7.88 °/s; cross-track 0.114 → 0.197 m). Tactical: the
manoeuvre-vs-trajectory agreement collapses from κ = **0.253** (v1, weak) to κ = **0.0072**
(v2corpus) — the declared manoeuvre is unrelated to the driven path; the `--v2` pack's tactical
machinery is **decorative**. Strategic: under nav-dropout 0.5 the arm *learned the dropout* —
route-following under a given command fell 1.0 → **0.5351**. And a correction to our own instrument,
logged before any number shipped: `route_acc_nav` feeds the model the command, so its 1.0 for v1
measures **copying, not route skill**; the honest test is vision-only route accuracy against the
majority-straight baseline, where v1 scores 0.9474 — **exactly** the majority rate. Neither arm
demonstrably does route work, and the hierarchy seam panel agrees: **0/3 seams beneficial on both
arms**. That is the current, honest state of the hierarchy thesis — and it motivated (f).

**(c) Compounding is real (C61 resolved: H-COMPOUND).** The earlier "decay accelerates" reading
from a rising ADE-vs-horizon exponent confounded task difficulty with error feedback. The missing
control — a **teacher-forced arm at matched steps** (identical roll, one line changed: the window
advances with the true latent) — separates them: the recursive rollout's compounding ratio rises
**3.50 → 80.77** across the horizon while the teacher-forced arm stays flat. Imagination error is
dominated by the rollout eating its own error, not by far horizons being intrinsically hard.
(E-DPSI, the direction-sensitivity probe, is null below 12° — no anticipation claim is made below
that threshold.)

**(d) Rollout-recovery training erases the speed bias — and pays in curvature.** Prescribed by (c):
fine-tune v1 for 2,000 steps with `--rollout-k 20` (RR-20) against an identical-seed control at
`--rollout-k 4` (RR-CTL); one flag apart, same pod; RR-20 is compared **only** to RR-CTL (against
v1 it would confound the horizon with 2,000 extra steps). On the canonical 40 episodes (n = 881,
paired B = 2000): ADE 0.424 → **0.348**, Δ CI [0.0613, 0.0906], separated. The four families turn
the headline into a trade: the longitudinal speed bias is **erased** (+0.9397 → **−0.0092 m/s**;
speed MAE −27 %) — the first intervention to move the programme's largest measured defect — but
curvature error is **2.2× worse** (0.0218 → 0.0483 m⁻¹), yaw-rate degrades, and miss@2m rises
0.043 → 0.056 *while ADE improves*. Under an ADE-only doctrine this would have been recorded as an
unambiguous win. Rollout recovery therefore enters the flagship plan as a **post-training phase**
whose gate is the family panel, not a bake-in; the mitigation hypothesis (an RR phase under a loss
that keeps curvature weighted — v1's loss had no curvature term) is pre-registered.

**(e) The situation-classifier result cannot answer the question it was built for.** The
lane-change/intersection labels are derived **from the ego trajectory alone** (`situations.py`:
yaw-rate lobes; heading change) — a frame is labelled "intersection" iff the ego turned. The
ego-kinematics head is thus a partial tautology (0.08858 CV-AP decodes its own input's labelling
rule), and the camera head is **penalised for correctly seeing** a junction driven straight through.
What survives: camera-only anticipation is real (0.04869 ≈ 2.1× the shuffle null, which lands at
the base rate), and under a *linear* ridge probe the camera reaches 0.836 of ego where the neural
head reaches 0.549 — the signal is present; the extraction, and above all the **label**, are the
bottlenecks. A scene-truth gold set (300–500 hand-labelled frames) is the pre-registered L0; no
camera-count decision may cite the current numbers.

**(f) The flagship's imagination was off — found, fixed, guarded, and re-launched as v5f.** The
running v5 (176×624 @ 120° cylindrical, 256-anchor two-step diffusion head, goal conditioning,
separated loss families) had `cond_imagination` — the mechanism §3.4 calls the novel part, the
decoder seeing the consequences of candidate controls before it denoises — **hard-wired off** in
the trainer since a smoke experiment: zero imagination tokens for the whole run, a reactive planner
over goal tokens. The fix is a launch flag plus the missing feed (no-grad probe rolls under an
FPS vocabulary of eight real action sequences, cached for resume parity), validated by tests
including a dead-conditioning assert (different imagination must change the decode). Before
relaunch, a pre-registered guard on the step-5k checkpoint scored the anchor head's tactical and
strategic families for the first time (a permutation-null instrument — command-following against
shuffled commands — that needs no knowledge of the dropped embedding row): route conditioning
**live** (Δ 0.067, CI [0.032, 0.103]), the tactical set-speed goal **live** (+0.895 m/s output-speed
movement), anchor-vs-refined κ **0.519** — the most tactically coherent arm the programme has
measured. Goal-dropout 0.5 was therefore *kept* (the v2corpus collapse was the `--v2` pack, not
dropout per se). Measured relaunch cost of imagination: **≈1.01×** step time (12.10–12.16 s/step vs
12.0 baseline) — the no-grad rolls hide inside data loading; the 1.2–1.5× estimate is retired.
`flagship-v5f` trains from scratch on the parity corpus with this configuration; the v2bal corpus
is deferred until a leak-free v2-line validation split exists (C64, option B).

**Retraction discipline, this round (C61–C67, by root-cause class):** C61 — a mechanism claimed
where only a magnitude was measured (resolved by the teacher-forced control). C62 — fleet state
recited from working memory instead of the stated enumeration. C63 — a metric imported without
measuring its precondition (decoded-displacement CR; the readout cannot decode true latents).
C64 — a missing constraint in a build spec (validation exclusion not inherited). C65/C66 — a
stale tree force-loaded past `PYTHONPATH` by a hard-derived `sys.path` insert, and a fix "verified"
along a path the failing code never took. C67 — a running job killed before its replacement was
verified, compounded by a scheduler that silently failed to re-arm. Every number in this section is
MEASURED, with raw artifacts under `TanitAD Research Hub/Evaluation/Implementation/incoming/`
(2026-08-02 directories) and estimators named inline.

## 8. Discussion: self-supervision, the two-stage question, and what the honest results demand

**What the 30 k results jointly say.** The causal panel (§7.3) and the OOD gap (§7.4) are not in
tension — together they locate the model precisely. In-distribution, the representation genuinely
encodes and predicts scene physics on the windows that need it (vision effect +1.32 m, CI-separated;
curvature R² 0.25; reads-road-ahead 1.6×). Out-of-distribution, that same vision signal survives
(comma2k19 ablation +0.27 m, CI-separated) but is not strong enough to net-beat a highway kinematic
floor, and the anticipation advantage collapses. The system has learned a *real but
distribution-fit* world model. The mechanistic account of *why* is the loss-balance reading of §3.7:
a ≈ 5 : 1 supervised-metric-to-SSL mass with no stop-gradient shaped the encoder into a metric
odometer (yaw R² 0.89 re-encode of fed proprioception), leaving vision at ≈ 12 % of the prediction
and the trunk under-invested in transferable scene structure.

**The two-stage question (LeWM/JEPA train-then-decode).** The JEPA/latent-world-model lineage
consistently *separates* self-supervised representation learning from action-conditioned prediction,
and there is convergent evidence this separation is what buys out-of-distribution robustness:
V-JEPA 2-AC freezes a self-supervised video encoder and post-trains a compact action-conditioned
predictor for zero-shot latent MPC (arXiv 2506.09985); DINO-WM couples frozen perceptual features to
a learned latent dynamics model and beats task-specific latents for planning (arXiv 2411.04983);
I-JEPA/LeJEPA evaluate through a frozen encoder + attentive probe (arXiv 2511.08544). The *theorem*
underneath is Kumar et al. (ICLR 2022, arXiv 2202.10054): full fine-tuning **distorts** good
pretrained features and underperforms a linear probe out-of-distribution, while probe-then-gentle-tune
(LP-FT) recovers ≈ 1 % in-distribution and ≈ 10 % OOD — which is exactly the pathology our joint
objective exhibits, the supervised heads distorting the trunk. A **v3 that separates encoder training
from action-conditioned prediction is therefore warranted** — but the evidence is explicit that it is
*not* simply "freeze a generic encoder": that is REF-A (frozen DINOv2), already run and plateaued at
2.14 m, because web-image features do not contain driving ego-motion. Our own data adds a hard warning
— the pre-grounding JEPA latent had a **1.65 m oracle in-distribution decode ceiling** (vs the 0.033 m
the grounded encoder reaches), so a frozen trunk risks capping the metric far above what end-to-end
grounding achieves *unless* stage-1 self-supervision is on our own driving corpus and the
action-conditioned predictor carries the metric. The clean de-risking move is that the §3.7
gradient-decouple **is the ≈ 1 %-cost ablation of this v3**: if relieving the encoder's static-probe
grounding raises vision-use while holding the 0.033 m metric, the two-phase freeze is worth building;
if the metric collapses the moment the encoder is unshackled, a frozen v3 would hit the same 1.65 m
ceiling and stage-1 must first be solved. Either way the decision is measurement-gated, not assumed
(citations: Kumar 2202.10054; V-JEPA 2 2506.09985; DINO-WM 2411.04983; LeJEPA 2511.08544; LeJEPA
identifiability 2605.26379).

**What §7.6–§7.7 add to that question, and where it now stands.** Three measurements have since
sharpened the two-stage argument in a way that partly overturns its framing. **(a) The frozen-encoder
ceiling is a ceiling of *static decode*, not of freezing.** A planner reading the frozen v1 through its
*dynamics* reaches 0.599 m and is statistically indistinguishable from that model's oracle-action
ceiling, while decoding waypoints off the same frozen state lands at 3.65 m — squarely in the REF-A band
(§7.7). The 1.65 m "frozen trunk risks capping the metric" warning above therefore applies to the
*probe*, not to the architecture: freeze-and-plan is viable; freeze-and-regress is not. **(b) The
two-stage separation has a measured price, and it is aleatoric.** Every deployable route over the frozen
model saturates at ~0.60 m because the residual is future intent that the past does not determine — so
staging buys robustness at a floor that no amount of stage-2 capacity or search can lower, and that floor
sits above the 0.427 the jointly-trained arm already reaches in-distribution. **(c) The failure of joint
training was not the joint objective.** Four arms that coupled a planner into an *already-converged*
trunk degraded it; the seam's gradient geometry is near-orthogonal (cos = +0.0043), and co-evolving both
from random initialisation reproduces v1's behaviour with the canary *descending* under full coupling
(§7.6). Read together: the LP-FT literature's "fine-tuning distorts features" result is reproduced here
as a **warm-start** pathology specifically, and the escape it recommends (stage, then gently tune) is one
of two escapes — the other is never to converge the stages separately in the first place. Which of the
two wins on out-of-distribution transfer is the open question, and the OOD panel of §7.4 is its
pre-registered arbiter.

## 9. v3 design: hierarchical goal-vocabulary planning over the world model

*Status: design contribution (2026-07-19), measurement-motivated; implementation staged (P0–P5), first
evidence gate = the training-free planner-over-v1 experiment (§9.4). **Updated 2026-07-25:** the design
has since been built as **flagship-v4** (three planners over the world model, ≈247.9 M trainable) and
trained through five arms; its first results — four warm-start failures, the gradient-geometry pre-probe
that redirected them, and the co-evolved arm now in training — are in **§7.6**. §9 below remains the
design of record; §7.6 is its empirical status.*

### 9.1 Motivation from the measured failures
Three independent measurements converge on one diagnosis. (i) The **frozen-encoder reference** (REF-A
dyn-in, §3.8): full-val ADE@2s 2.92 m vs the trained-encoder twin's 0.452 m, losing to constant-velocity
in every stratum; the failure is 94–99 % longitudinal — the model **regresses toward a mean speed even
with v₀ fed as input** (over-predicts +1.72 m/s when slow, under-predicts −0.58 m/s when fast), while
path geometry stays nearly intact (0.27 m cross-track). Monotone 5k→30k improvement rules out
overfitting: it is a capacity ceiling of supervised regression on static features. (ii) The **hierarchy
panel** (H26, §7.5): the strategic route head is a command echo (100 % "straight" under follow),
the tactical head is 8× worse than the operative rollout (3.38 vs 0.43 m), and un-gated intent
conditioning *hurts*. The hierarchy exists as structure, not as functioning planning. (iii) The
**closed-loop probe**: open-loop 0.45 m → closed-loop 1.69 m with 22 % divergence, and the open-loop-
decisive speed channel does not transfer. All three are the same disease: **supervised single-output
heads collapse to conditional means and echoes**, and nothing in the stack represents *what should be
achieved* — only what was done.

### 9.2 The design: goals → options → consequences → cost → choice
v3 restructures the hierarchy as planning over the world model (Michon's strategical/tactical/
operational hierarchy and Donges' Navigation/Bahnführung/Stabilisierung, instantiated in a learned WM).
A **frozen, tokenizable goal vocabulary** (110 tokens / 17 slots, v1 spec) defines: the strategic tuple
⟨MISSION, ROUTE, VTARGET, VSOURCE, LANEOBJ, STYLE, RISK, ODD⟩ and the tactical tuple ⟨LONMODE,
LATMANEUVER, HEADWAY, DYN, RULECTX, SIGNAL, INTERACT, TACPOINT, LIGHTSTATE⟩. The strategic module
*predicts* its tuple from scene+navigation (target speed becomes an inferred, banded goal — "extract the
set speed from signs" is a trained task, never a raw future-speed input, avoiding the ego-status
shortcut). The tactical layer *proposes* options (vocabulary enumeration × a learned multi-modal
proposal prior — the anchored decoder validated by REF-B v2's 0.646 m, the first reference arm to beat
the CV floor in every speed stratum). The operative world model *predicts each option's consequences*
(feature-space rollout for the frozen arms, per DINO-WM; grounded latent rollout for the flagship arm).
A **planner** (CEM; Diffusion-ES staged) scores options on a lexicographic Rulebooks-style cost —
safety ≻ rules ≻ mission ≻ comfort — and picks the plan; heads never decide.

Three properties fall out rather than being engineered in: **longitudinal mode-switching** (free-flow ↔
following emerges from the target-speed term competing with the gap barrier, the IDM insight — no mode
classifier at execution); **driving style and risk degradation as cost presets** (STYLE/RISK tokens
re-parameterize planner weights and the risk budget — behavioral degradation under weather/visibility/
anomaly without retraining, an ISO 21448 SOTIF mitigation path); and **deliberation-on-demand**
(`elevated_anomaly` raises the planner's compute budget — more samples, longer horizon — while `creep`/
`hold_stop` buys time, escalating to pull-over/MRM per SAE J3016; a capability head-based E2E stacks
cannot express, since a head spends identical compute on easy and hard scenes). Rule-conformity is a
priority lattice (Rulebooks), so "cross the solid line to avoid the obstacle" is lattice-consistent and
tokenized as a justified deviation with provenance rather than an exception.

### 9.3 Both arms under one spine
flagship-v3 (trained encoder, v2 lineage with measurement-gated lever staging) and refa-v3 (a
three-arm frozen-encoder matrix: frozen generic DINOv2 as the faithful DINO-WM control; own-driving-SSL-
then-freeze, the V-JEPA-2-AC pattern; LoRA/LP-FT partial adaptation per Kumar et al.) share the
vocabulary, predictors, cost, and planner — turning the frozen-vs-trained encoder question (H4) into a
controlled comparison *inside* the same planning architecture, which REF-A could not provide (it
confounded the frozen encoder with supervised-regression heads).

### 9.4 Falsifiable gates
G1 counterfactual plan-ranking: the chosen plan must beat non-chosen options on realized outcome,
CI-separated (the direct test of "evaluating alternatives" — replaces head-consistency as the H26
instrument). G2 goal-causality: swapping a goal token must change the conditioned output (kills echo
pathologies). G3 |v̂−VTARGET| tracking. G4 closed-loop drift below the 1.69 m head baseline. G5
open-loop non-regression vs 0.452 m. The staging de-risks the thesis cheaply: the planner (cost+CEM)
runs over the **already-trained v1 world model** with offline-minted VTARGET labels — if planning over
frozen v1 already beats the 3.38 m tactical head and the 1.69 m closed-loop drift, the architecture is
validated before any v3 training.

## 10. Roadmap

The standing programme is unchanged: the full gate ladder; closed-loop evaluation for D4–D6 including
opponent-derived weak-spot scenarios (a scenario database built from documented competitor failures,
with per-scenario excellence as an explicit leaderboard section); the frozen-encoder comparison arm
(H4, now closed negative); data-efficiency slope experiments toward the 1000× thesis (H7); modality
steering driven by imagination uncertainty (H2); and NAVSIM/Bench2Drive entries once closed-loop gates
pass. Standing method levers from the 30 k results also remain open: the §3.7 loss-rebalance and its
two-stage extension (§8) for vision-reliance and OOD transfer, the ReZero-gated intent seam (§3.3), and
the targeted longitudinal fix (§7.5). The v0.6 round re-orders what comes first:

1. **Finish the co-evolved coupled arm to 30 k and gate it on the formal eight-metric card** (§7.6).
   Two things are required for the verdict to be quotable at all: the canonical held-out harness rather
   than the trainer's in-loop evaluation (I-class C1), and the three pre-registered kill secondaries that
   currently have **no emitter anywhere in the codebase** — a gate that cannot render a complete verdict
   is an instrument failure, not a model result. The acceptance criterion is the **out-of-distribution**
   panel of §7.4, not the in-distribution one we already pass.
2. **Longitudinal control, now the single most-confirmed weakness.** It is the above-floor stratum
   open-loop (§7.5), the failure axis of every v4 planner arm (§7.6), and — measured on-policy through an
   independent instrument at n = 40 — the whole of the flagship's closed-loop deficit while lane-keeping
   is nearly perfect (§7.8). Three independent instruments now point at one mechanism.
3. **A lower-OOD reactive-agent instrument.** The measured trade — low observation-OOD requires real
   footage, which has no agents; agents require a renderer, which sits at ~3.2× OOD — is the binding
   constraint on every safety-grade closed-loop claim, and on the renderer-gated half of the beyond-ADE
   suite including TLC (§7.8, §7.10). This is a build, not an experiment.
4. **Train the next generation on the balanced 50 h corpus** (§7.9), whose turn and junction coverage
   directly target the strategic seam's command-echo behaviour (§7.5) and the 42.6 %-of-clips-contain-no-
   turn structural gap; and close the semantic-scenario 0 % with the separate labeling track.
5. **Scale the inverse-dynamics video pipeline from the 80-clip pilot to a decision-grade harvest**
   (~300+ clips, 4+ seeds, plus a per-video intrinsics estimator to remove the nominal-HFOV
   approximation) — the direct path to the C2 data-efficiency slope and the first quantitative test of
   the 1000× thesis (§7.9).
6. **Retire the deprecated interval estimator from the historical tables** and re-publish §7.1–§7.5's
   widths under the episode-cluster bootstrap (§5), so the paper carries one estimator throughout.

## References

(Formal bibliography at LaTeX export; the working citations live in
`TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` and the dated research notes: LeJEPA
arXiv:2511.08544; LeJEPA-identifiability (Klindt et al.) arXiv:2605.26379; JEPA generalization theory
arXiv:2606.27014; V-JEPA-2 / V-JEPA-2-AC arXiv:2506.09985; DINO-WM arXiv:2411.04983; fine-tuning
distorts features / LP-FT (Kumar et al., ICLR 2022) arXiv:2202.10054; DiffusionDrive
arXiv:2411.15139; LAW arXiv:2406.08481; World4Drive arXiv:2507.00603; HiT-JEPA arXiv:2507.00028;
GAIA-2 arXiv:2503.20523; Drive-JEPA arXiv:2601.22032; ego-status open-loop shortcut (AD-MLP /
BEV-Planner) arXiv:2312.03031; open-loop⊥closed-loop arXiv:2605.00066; ALPS-4B transfer study
`Ressources/AD_TRANSFER_RESEARCH.md` v1.1.)

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
- v0.3 (2026-07-17): §7.2 second-round results — the 2026-07-14 speed-grounding reset (v₀ as third
  action channel); flagship-speed @19 k is the first CV-beater on every held-out metric
  (0.628 ± 0.055 ADE@2s; causal A/B +2.21 m [2.04, 2.39], win-rate 83.8 %); the CTRV
  kinematic-oracle bar (**0.523** — the 0.544 quoted in this round was superseded 2026-07-18, §7.2;
  ego-status shortcut replicated on our 73.9 %-straight corpus, independently converged at 0.545 on
  *comma2k19*); imagination panel (vision_use 12.9 % / imagination 8.7 % /
  latent-gain +0.054; frozen-encoder arms are dynamics integrators); planning-brain eval (tactical
  3.38 m, speed-starved; goal-latent cos 0.885 — decode, not imagination, is the bottleneck);
  σ-dissipation under recursive rollout (false confidence; freeze-1 holds ≈0.25) with the 1-step
  self-monitor cap and the §3.5 calibration caveat; matched frozen I-JEPA-vs-DINOv2 comparison
  (⚠️ **WITHDRAWN 2026-07-26 — ~80 % val leak, `MODEL_REGISTRY §2.2`/R8; see §7.2**);
  REF-B @6 k yaw-blindness and refbpatch. §7.1 marked superseded-as-diagnosis by the reset;
  abstract updated. All §7.2 numbers open-loop = weak claims (arXiv 2605.00066).
- v0.4 (2026-07-19): 30 k first-run completion and the honest in-distribution/OOD split.
  §7.3 flagship-30k FINAL — first sub-floor arm (ADE@2s 0.452 ± 0.031, below floor 0.500 / CTRV
  0.523 / ego-status 0.574; resolves the v0.3 pending verdict) + the **genuine-prediction causal
  panel** (vision effect +1.32 m CI [+1.04,+1.64] on high-CTRV-divergence windows; advantage
  +0.80 m→inverts −0.53 m under scene mean-replace; monotone in divergence; curvature-decode R²
  0.25 vs 0.03; reads-road-ahead 1.6×; ledger 2026-07-18). §7.4 **OOD** — does not beat the strong
  highway floor on unseen corpora (comma2k19 0.849 vs 0.372; Cosmos 0.583 vs 0.358; error ~doubles;
  anticipation collapses 0.80→0.15 m; path feasibility 97.8 %→62.8 %) yet vision-ablation still
  hurts OOD (+0.27 m CI-sep) → real-but-distribution-fit, not corpus-general (ledger 2026-07-19).
  §7.5 hierarchy (H26: ctx→tactical flipped load-bearing at 30 k, intent→operative still swamped),
  REF-C tactical decoder, and the high-speed **longitudinal** weakness (89 % of 2 s error along-track;
  +0.66 m/s speed over-prediction). Method: §3.3 ReZero gated-intent (init 0.1); §3.7 loss-balance
  (≈5:1 supervised:SSL, no stop-gradient → odometer encoder; gradient-decouple α=0.25 rebalance);
  §3.8 REF-C DiffusionDrive anchored decoder + curvature-relative label fix (24.5 %→0.0 % conflation).
  New §8 Discussion (self-supervision reading + the two-stage LeWM/JEPA v3 question, Kumar 2202.10054
  / V-JEPA-2-AC / DINO-WM); Roadmap renumbered §9; references extended. All open-loop = weak claims.
- v0.5 (2026-07-19): new **§9 v3 design — hierarchical goal-vocabulary planning over the world model**
  (Roadmap renumbered §10). Motivation triangulated from three same-day measurements: REF-A full-val
  frozen-encoder ceiling (2.92 m, 94–99 % longitudinal, mean-speed regression despite v₀; monotone
  curve = capacity not overfitting), H26 head degeneracy (strategic echo, tactical 8× worse than
  operative, un-gated intent harmful), and the closed-loop gap (0.45→1.69 m, speed channel does not
  transfer). Design: frozen goal vocabulary v1 (110 tokens / 17 slots; strategic ⟨MISSION…ODD⟩ +
  tactical ⟨LONMODE…LIGHTSTATE⟩, Michon/Donges/J3016/SOTIF/Rulebooks-anchored), planner-based hierarchy
  (options × WM-consequences × lexicographic cost, CEM→Diffusion-ES), target speed as inferred goal +
  planning cost (never a raw input; anti-shortcut), style/risk as cost presets, deliberation-on-demand,
  and the two-arm spine (flagship-v3 trained encoder vs refa-v3 frozen matrix incl. own-SSL-then-freeze
  and LoRA/LP-FT) making H4 a controlled within-architecture comparison. Falsifiable gates G1–G5 incl.
  the training-free planner-over-v1 de-risk. Design docs: V3_HIERARCHICAL_PLANNING_DESIGN.md +
  V3_GOAL_VOCABULARY_V1.md. Mid-training context: REF-B v2 @20k = 0.646 (first REF-B to beat the CV
  floor in every speed stratum; validates the time-anchored proposal decoder); flagship-v2 @6k behind
  v1's trajectory (6.18 m), mechanism diagnostic + 10k gate pending — lever staging feeds §9.3.
- v0.6 (2026-07-25): the coupled-planner round, two additive directions closed, the first confound-free
  closed-loop comparison, and the data thesis's first evidence. *(Status line corrected: it had been left
  at v0.4 while the §9 design round was already logged as v0.5; this results round is v0.6 so no number
  is reused.)*
  **§7.6 flagship-v4 — planner–WM coupling.** The §9 design built (≈247.9 M trainable, ~30 M *smaller*
  than v1: three planners replace three supervised heads; strategic planner in a 128-d subspace +
  tactical/operative anchored-diffusion at 256 anchors). Four **warm-start** arms fail: v4 (hot trunk)
  killed ~3.5k with the canary rising under a clamped planner gradient; **v4.1 10k `ade_0_2s` 0.8522
  [0.7468, 0.9800]** vs a 0.60 bar with canary 0.4599 PASS (WM healthy, planner starved; loss is
  longitudinal — speed vs CV Δ −0.366 [−0.491, −0.245] separated, while speed-decoupled path geometry
  *beats* CV +0.115 [+0.017, +0.240]); v4.2 (floor 0.25) 0.9869 with canary 0.7222 FAIL; v4.2b
  (floor 0.15) canary 0.697. **Cosine pre-probe (~0 GPU, n=512 windows, the `states` seam):
  cos(g_wm, g_plan) = +0.0043 (sd 0.064), 47.9 % of windows negative, PCGrad `frac_removed` 0.0224 →
  gradient surgery REFUTED as a no-op**, and the floor had already attenuated g_plan to 15 % with the
  canary still degrading ⇒ neither direction nor magnitude is the cause. **From-scratch co-evolution**
  (`flagship-v4-fromscratch-30k`): canary baseline recalibrated to **15.674** at random init; at full
  coupling λ_plan = 1.0 it **descends** 15.674 → 2.59@7k → 1.371@9k while held-out ADE@2s falls
  0.531@9k → 0.4788@11.5k (miss@2m 0.169, oracle-in-fan 0.242); 10k gate **CONTINUE**. **Claim: the
  coupling failure is a warm-start artifact, not an intrinsic prediction-vs-control conflict.**
  ⚠️ from-scratch numbers are the **trainer's in-loop** clean-split evaluation, formal 8-metric gate
  **deferred**; run is at ~40 % of 30k; an earlier one-point "canary descending confirmed" was retracted.
  **§7.7 two additive directions.** *D1 frozen-WM + learned planner:* a 3.77 M planner trained by
  analytic gradients through the frozen v1 reaches **0.5989 [0.374, 0.854]** — paired-beats CV
  (−0.2474 [−0.505, −0.034]), hold-v0 and action-BC (−0.4012 [−0.717, −0.128]), and is **not separated
  from the WM's own oracle-action ceiling 0.4045** (+0.1944 [−0.045, +0.448]); static decode off the
  same frozen state is 3.649 (the REF-A regime) ⇒ **the frozen ceiling is a ceiling of static decode,
  not of freezing.** Capacity is flat under 11× scaling (0.599/0.601/0.599, none separated); the wall is
  **aleatoric**. 🔴 RETRACTED (I9): the CEM "0.132, 4.5× headroom" is **hindsight-privileged**; the
  deployable learned-value search scores **1.0162**, +0.4173 [+0.237, +0.605] separated *worse* than
  feedforward. Verdict: ~0.60 m degradation-free **fallback**, not a contender. *D2 closed-loop
  recovery augmentation:* the ADE "cost" was largely a knife-edge-L2 artifact (`band_ade2d(1.0)` makes
  it vanish CI∋0 for 3/4 configs, −74 % for the fourth), but the departure **benefit reverses at power**
  — n=12 +0.0089 → **n=40 2-fold cross-fit −0.0302 [−0.0595, −0.0088]**, departs 3.3× more. Not
  promotable; durable = the machinery, **REF-C's encoder is safely fine-tunable** (feat_cos 0.9658 at a
  material move, canary holds), and two measurement lessons.
  **§7.8 closed-loop benchmarking.** New **real-footage log-replay** instrument: on-policy observation-OOD
  **1.02–1.20×** (longitudinal 1.018/1.004; junction 1.196/1.152) vs a photoreal reconstruction's flat
  **3.75×**; flagship envelope flat to **2.0 m** lateral (3 m: +0.066 [+0.010, +0.138]), yaw separates at
  **3°**. At **n = 40 eps / 881 windows**, paired: **REF-C base ADE@2s 0.564 [0.452, 0.676] vs flagship v1
  1.488 [1.329, 1.647]** (Δ +0.924 [+0.781, +1.065]); departure@1.75 m 0.0134 vs 0.0318 (Δ +0.0184
  [+0.0077, +0.0328]) — the **third independent confirmation** (n=1 retracted → n=12 reconstruction suite
  pass 8/12 vs 2/12, Δ −0.430, sign-test p=0.008, collisions tied → n=40 real footage). Decomposition:
  in longitudinal scenes both arms keep the lane (0.4 % / 0.04 % departure) yet the flagship's ADE is
  **4×** — the deficit is **longitudinal, not lane-keeping**; in junctions the flagship departs ~2.3×
  more (peak XTE 2.372 vs 1.458 m). ⚠️ map-free/agent-free ⇒ **lane-keeping drift, not off-road or
  collision**; the low-OOD-vs-safety-metric gap is ~fundamental without a lower-OOD reactive renderer.
  **§7.9 data.** Corpus MEASURED: **13.13 h / 472,627 frames / 2,376 clips × 19.9 s / 406,099 windows;
  30 k steps = 4.73 epochs**; lane_keep 59.6 % · accel 13.2 · brake_stop 12.9 · turn_right 7.4 ·
  turn_left 6.9; **only 42.6 % of clips contain any turn; semantic scenarios 0 %-labeled.** A balanced
  **50 h v2 corpus** designed+built inside the same source (turns 14.25→28.0 %, junction-clip presence
  37.7→61.3 %, key `physicalai-v2bal-4b7eeeac222d`), stored JPEG-compressed **982 GB → ~25 GB** with
  frames bit-identical to the parity decode path; breaks parity **by design**. **H7:** direct
  extracted-vs-GT accuracy is modest (speed R² 0.62–0.66, longitudinal-traj 0.60, **yaw ≈ 0**
  ⚠️*STALE-PENDING: `heading_repair` OFF — comma's heading is undefined at standstill; see the C29
  note in §H7. On repaired labels the same head reads comma yaw **+0.3308**, retrained **+0.679** —
  🔴 AMENDED 2026-07-27 (C43): **+0.3308 WITHDRAWN** (2 of its 22 comma val episodes are, BY CONTENT,
  in that head's own comma training set; without them **−0.746**); **+0.679** stands but is **+0.3038
  [+0.054, +0.479]** on the 20 content-clean episodes. comma yaw is testable; this head does not do
  it. See `…/incoming/2026-07-27-anchor-settlement/`*, accel
  dropped) but downstream pseudo-label WM pretraining captures **~96 %** of real-label value (8 seeds,
  two domains) and **109 % speed / 107 % traj / 71 % yaw** on the actual parity target (4 seeds); an
  **80-clip CC YouTube pilot** lifts parity-val speed R² **−0.520 → +0.563** (3 seeds, clip-cluster CI
  excludes 0 every seed), yaw 0.55→0.75, ADE **halved** 12.82→6.31 m ⇒ **≈92 % of the real-label
  ceiling — the YouTube domain transfers.** ⚠️ DIRECTIONAL (80 clips, 3 seeds, unknown intrinsics; the
  fraction-of-ceiling is the substantive claim, not the R² delta). Negative that constrains the design:
  a from-scratch GAIA-2 camera-conditioned encoder **FAILS** cross-rig (speed R² **−0.667** vs frozen
  v1's **+0.657**, paired CI excludes 0 on 3/4 arms).
  **§7.10 metrics + deployment.** Traffic-light scenario **SC-14** + **TLC = red_entry_gate ×
  stop_quality × green_flow** (a single red-run zeroes it; design oracle rule_barrier 1.0 vs soft_prior
  0.0). First **real** beyond-ADE numbers on a **`base250cam` 262.8 M WorldModel** (RTX 4060, fp32 eager,
  comma2k19 n = 30 — **not** the deployed 263.4 M flagship; conditions added 2026-07-26, §7.10):
  decision-tick **p50 14.331 ms**,
  TMS median **0.0435** (expert-log reference band, not our policy), CNCE median **210,551**;
  TLC/LAL/OKRI/LOPS remain **renderer-gated** (absence confirmed at 3 probes). **FP16 is the deployment
  precision; INT8 rejected** — no latency win (encoder +2.1 %, predictor −2.1 %) and weight+activation
  INT8 collapses the un-normalized `readout_head` to cosine 0.566, costing **+0.0215 m** ADE@2s over a
  20-step rollout past the 0.02 m falsifier (weight-only passes at +0.0065). Composed planning tick
  **100.29 → 18.75 ms p50 (5.35×)**, 10 Hz at p99 with 5.3× headroom, levers **sequenced not additive**;
  an 8-candidate select fan is **20.82 ms** (~0.3 ms/candidate), refuting the 723 ms projection.
  **§5 doctrine extended** with three method-grade failure classes — **I8** power before closure (an
  effect on one underpowered split reversed sign at full power), **I9** a privileged-input arm is not a
  headroom estimate, **I10** verify presence/absence with the tool that owns the fact (a truncated
  mtime-sorted listing produced a false "stranded" claim) — plus the **estimator correction** (the
  historical `± CI95` is `overlapping_holdout_se`, 1.28–2.06× too narrow, coverage 62.3 % vs 93.8 %;
  decision-grade = episode-cluster bootstrap, paired for two arms), the **two-tick latency definition**
  trap, the **tolerance-band closed-loop metric**, and a leaking validation split (**78 %** overlap) now
  **refused in code**. §8 extended with what §7.6–§7.7 do to the two-stage question; §9 status updated to
  point at §7.6; §10 roadmap re-ordered. All open-loop numbers remain weak claims (arXiv 2605.00066).
- v0.7 (2026-08-02): §7.11 — the binding four-family doctrine (LON/LAT/TAC/STR added to ADE,
  per-family, paired episode-cluster bootstrap) and what it exposed in its first days: the
  v2-corpus arm falsified on the 19 leak-free episodes (v1 re-scored 0.393 [0.307, 0.493] vs
  0.575 [0.429, 0.752], paired ΔCI [−0.221, −0.145]) with the C64 leak (21/40) and the
  `--v2-cache`/`--v2` conflation making it a corpus+architecture+rollout_k change, hence not a
  corpus result; the longitudinal **sign flip** (+1.465 vs −1.260 m/s) that ADE cannot represent;
  tactical κ 0.253 → 0.0072 (decorative) and the `route_acc_nav`-is-copying correction (vision
  route = exactly the majority-straight rate; 0/3 hierarchy seams on both arms); **H-COMPOUND**
  (CR 3.50 → 80.77, teacher-forced flat; C61 resolved) and E-DPSI null < 12°; rollout recovery
  RR-20 vs RR-CTL (ADE 0.424 → 0.348, ΔCI [0.0613, 0.0906]; speed bias erased +0.9397 → −0.0092;
  curvature ×2.2, miss@2m up — a trade, not a win); the situation-classifier **label defect**
  (ego-derived labels; camera penalised for being right); and the v5 → v5f fix (`cond_imagination`
  hard-wired off → flag + feed + dead-conditioning test; 5k guard ROUTE_LIVE Δ0.067 [0.032, 0.103],
  VT_LIVE +0.895 m/s, κ 0.519; imagination cost measured ≈1.01×). Retractions C61–C67 by
  root-cause class. All §7.11 numbers MEASURED with named estimators.
