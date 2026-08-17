# TanitAD: A Data-Efficient, Hierarchically-Imagining, Self-Supervised Driving Stack with Built-In Self-Knowledge

**Status:** living paper, **v1.0 (2026-08-12)**. Maintained per D-020: every gate evaluation appends
results; every accepted decision that changes the method updates §3–§5. Source of truth is this
Markdown; LaTeX export is a release step. Honesty rule (P8): no number appears here without its
instrument rows in the referenced experiment record. *(Version note: the status line was left at
v0.4 while the §9 design round was already logged as v0.5 in the changelog; that results round was
therefore v0.6. The same drift then recurred: the status line sat at v0.6 while the changelog
already carried v0.7 (2026-08-02) and v0.8 (2026-08-03). This round was requested as "v0.7" but
that number is taken — it is therefore v0.9, and the numbering is again not silently reused.
**v1.0 (2026-08-12)** lands the four verdicts of the night of 08-11/08-12 — W7-FULL, H-COTRAIN,
P8 attempt-2 and I4a — with their mathematics (§3.10–§3.13), the tier and four-family doctrines
promoted to named methodological sections (§5.4–§5.5), and the v6 future-work ladder with its
open, costed decisions (§10). It is v1.0 rather than v0.10 because it is the first version in
which a hypothesis this paper itself carried is **retracted by our own measurement** in the
paper's own voice, §7.17.)*

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
*Fifth-round update (v0.9, 2026-08-11):* the round is dominated by two methodology corrections now
binding on every number — an **eval-tier doctrine** (T0 teacher-forced = WM diagnostic only, T1
action-closed loop = the primary capability tier), forced by the measured **action echo**: the
open-loop lateral "skill" of the best decoder heads is 97.9 % S-curve reproduction under the true
action transcript and **~5 % closed-loop / 0.0 % hold-action** (`MODEL_REGISTRY §1.12`, T1) — and the
completed retirement of the historical `overlapping_holdout_se` estimator, which biased *point
estimates* (−6.67 % to +11.69 %, sign-flips on paired deltas) and not only intervals (§5). On the
science: the co-trained **v5f** flagship (from-scratch joint WM+planner, conditional imagination,
120° cylindrical) completed 30 k with T0 selected ADE@2s 0.4011 m over a fan whose oracle is
0.1975 m (`MODEL_REGISTRY §1.8`); a physics-proof battery (§7.14) then located **one root defect
under four separate failures** — the trunk responds to counterfactual actions with the right sign
(99.5 % lateral) inside a 3-dimensional action subspace but at **~¼ the physical gain** (0.27,
`w3_gate.json`, T1-diagnostic) — and a predictor-only **stage-A post-training** with a
control-response loss repaired it (gain 0.27 → 0.97, longitudinal sign 0.74–0.79 → 1.0,
subspace preserved, `stage_a_gate.json`), after which the world-model-roll selection cost became
the programme's best-calibrated selection signal (Spearman ρ 0.716 [0.585, 0.770] vs ≤ 0.26 for
every learned scorer; `w7_gate_repaired_k32.json`, `p7_regrade.json`). Three learned fast-selector
surfaces failed one after another by pre-registration and were retired — the selection information
is in rolled consequences, not in light readouts of the trunk — which is the programme's own thesis
arriving by elimination. A P1 probe battery also measured the latent's sharpest absence: **no
readable lead-vehicle distance at any probe capacity** (`p1_lead_transforms.json`), answered by a
label-free lever program rather than label injection, per the PI's standing rule that labels into
the trunk would break the self-supervised thesis.
*Sixth-round update (v1.0, 2026-08-12):* four measured verdicts, two of them negative and both
reported as results. **(1) The selector-free planner fails, and the mechanism is the winner's
curse.** With every stale component removed — repaired trunk, head refit on it, and **no
shortlist at all** (256 of 256 candidates, so the true best is always available) — selection by
argmin of a world-model roll-consistency cost plus a kinematic cost scores **3.3348 m against a
0.4505 m gate over a fan whose oracle is 0.1273 m** (`w7_full_gate.json`, T0). The cost's
*within-window* rank correlation is 0.445/0.497 and its across-window calibration ρ 0.3185
[0.2064, 0.4086] — yet **the argmin's error-rank is 132 of 256, the median**, and the mean error
inside its top-m set is flat at the fan's own mean for every m. Rank correlation is a bulk
statistic; argmin is an extreme one, and the quantity that governs selection is lower-tail
dependence, which is zero here (§3.12). A self-consistency cost has a **degenerate minimiser** —
a near-stationary candidate — so deepening the fan makes a minimiser monotonically worse while
the oracle improves; the published loops we copied (V-JEPA 2-AC, DINO-WM) minimise distance to a
*goal*, which inaction cannot minimise, and we had dropped that term. **(2) The standing
hypothesis that planner co-training erodes physical representation is REJECTED within the
measured range and retracted in this paper's own voice**: across the λ_plan ramp every probed
physical variable became *more* decodable (curvature 0.213 → 0.551 encoded, yaw-rate
0.583 → 0.869), the latent's participation ratio *expanded* 53 % (4.53 → 6.94 of 2048) and the
P1 battery went FAIL → PASS — which also **validates SIGReg** under a full planner gradient
(retention 1.53× against a ≥ 0.8× gate), with three scope bounds stated rather than buried
(§7.17). **(3) The predicted latent retains the environment**: a frozen-latent BEV occupancy
readout gives **retention 0.932 at k = 10** (gate ≥ 0.80, τ\* chosen on the encoded arm so the
gate can only harden), and occluded-agent recall is **not worse** than visible — the latent
carries agents the camera cannot see (§7.18; absolute IoU ≈ 0.02, so the ratio is the quotable
claim). **(4) Imagination is load-bearing**: zeroing it collapses selected ADE 0.4011 → 7.6493 m
and shuffling it — which preserves marginals and destroys only correspondence — gives 1.2492 m,
so the planner reads imagination as content, not as a bias term. Together with the measured
**consumer-invalidation** result (repairing a trunk moves the frozen selector 0.7933 → 4.4159 m,
which is why the shipped assembly is the frozen-trunk one at 0.4815 m [0.3928, 0.5771]), these
define the programme's next generation as a **staged** ladder in which every consumer is trained
on the trunk it consumes (§10).

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

> **Figure 1 — `Paper/figures/v6_architecture.svg`** (also rendered as `.png`): the v6
> instantiation of the 4B architecture — the four layers with their own predictors and
> action spaces, goals conditioning downward, latents flowing upward through tested
> gradient-isolation barriers, the single 6 s trajectory spanning the operative and
> tactical bands, the frozen-latent interpretation heads, and the staged S-W → S-T →
> S-S → (optional) S-J training protocol with its per-stage gates. §3.9 gives the
> formal objects; §10 gives the staging rationale and its evidence.

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

### 3.9 The v5f generation: the objective family stated, hierarchy as per-layer predictors, and control-response post-training (added v0.9)

**The JEPA objective, general form.** The family this architecture instantiates is: an encoder
f_θ, a predictor P_φ, and a (possibly EMA) target encoder f_θ̄, trained on

L_JEPA = ‖ P_φ( f_θ(x_t), a_t ) − sg[ f_θ̄(x_{t+k}) ] ‖ ,

where sg[·] is a stop-gradient and the EMA/stop-gradient pair is the usual anti-collapse
mechanism (V-JEPA 2, arXiv 2506.09985; I-JEPA lineage). **Our instantiation removes both**: no
EMA, no stop-gradient, no teacher network (A1, §3.2) — collapse is prevented *provably* by
SIGReg alone, driving every 1-D projection of both encoder outputs and predictions toward N(0,1)
(Cramér–Wold ⇒ joint isotropic Gaussian, the LeJEPA-optimal embedding law). The deployed v4/v5
trainer uses the **`full_relaxed` SIGReg variant** — the sliced Epps–Pulley statistic of §3.2
applied to the full state with a configured number of *free dimensions* exempted from the
normality pressure (`train_flagship_v4.py:101`, `sigreg_variant="full_relaxed"`), so a small
subspace may carry non-Gaussian task structure while the bulk is regularised. The residual
multi-horizon form ẑ_{t+k} = z_t + Δ_k(z_{t−W+1:t}, a_{t−W+1:t}) of §3.2 is the concrete P_φ.

**The hierarchical 4B formulation as per-layer predictors (design of record 2026-08-07,
`incoming/2026-08-07-hierarchical-wm-redesign/HIERARCHICAL_WM_REDESIGN.md`).** Each level ℓ ∈
{op, tac, str} owns a state space, a clock, and its **own predictor in its own space**, with
temporal down-sampling as the abstraction mechanism (slow features; dimensionality decreases
upward by design so the level is *forced* to abstract):

| level | state | clock | horizon | "action" |
|---|---|---|---|---|
| operative | z_op ∈ ℝ^2048 | 10 Hz | 0–2 s dense | continuous (a, κ) controls |
| tactical | z_tac ∈ ℝ^{d_t}, d_t ≈ 512 = φ_tac(z_op(t−3..t)) | 1 Hz | 2–6 s | the tactical **goal** g_tac |
| strategic | z_str ∈ ℝ^{d_s}, d_s ≈ 256 = φ_str(z_tac window) | 0.2 Hz | 6–15 s | the strategic goal g_str |

Level dynamics: ẑ_ℓ(t+τ_ℓ) = f_ℓ(z_ℓ(t), g_ℓ) — **the upper levels' action *is* the goal**, and
each level proposes a fan of N candidate goals, rolls *its own* predictor (cheap: small state,
slow clock — a 15 s strategic roll is 3 steps of a 256-d model), and scores with a selector.
Goals condition downward: the operative decode consumes the selected g\*_tac as an input token,
the tactical consumes g\*_str. g_tac is a predicted ego-frame goal point + heading + target speed
at τ ∈ [2, 6] s plus a manoeuvre class on **three independent axes with severity** (the measured
5-way lat/lon-mixing defect is the reason for the factorisation); g_str is a goal point +
corridor at route scale. ⛔ Neither goal may carry the situation classifier's output (the
2026-08-03 admissibility rule), and both are geometric, predicted from vision. `sel_gap` —
oracle-vs-selected error at each level — is a first-class metric of the hierarchy, which §7.13
shows was prescient: at both levels measured so far, the fans are adequate and the *selectors*
are the defect.

**Control-response post-training (stage A), the mathematics.** Logged data supplies triples
(o_t, a^human_t, o_{t+1}) only on the human action manifold; under a counterfactual action ã
there is no logged future, so any loss ‖f(z, ã) − z'_logged‖ teaches the predictor that the
world ignores actions. What *is* supervisable off-policy is the ego-motion component, which is
fully determined by the commanded actions. Stage A therefore trains **the predictor only**
(encoder, readout and heads frozen, md5-proved) on three losses (`train_stage_a.py`,
`stage_a_gate.json`): for a counterfactual perturbation δa of the logged actions a (channels:
δκ = ±0.02 m⁻¹ through the steer encoding, δa = ±2 m/s²; plus random draws within an
a_max = 4 m/s², κ_max = 0.2 m⁻¹ envelope), define the decoded **response**

Δ_probe = decode(roll(z₀, a+δa)) − decode(roll(z₀, a)),  Δ_analytic = U(a+δa, v₀) − U(a, v₀),

with U(·, v₀) the unicycle integral of the action sequence from the observed speed v₀ and
decode(·) the *frozen* step-readout (so the gradient cannot cheat by re-learning the decode).
Then, in **response form** (`ctrl_form: "response"`):

L = L_ctrl + L_factual + 0.3·L_scene,  L_ctrl = ‖Δ_probe − Δ_analytic‖₁,

whose gated summary statistic is the **response gain g = ‖Δ_probe‖ / ‖Δ_analytic‖ at 1 s**,
required in [0.5, 2.0] per channel with sign-correctness ≥ 0.95; L_factual is the ordinary
teacher-forced roll on the human actions (the catastrophic-forgetting anchor, gated at ≤ +10 %
ADE); and L_scene penalises latent change in the subspace **orthogonal** to the measured
action-response subspace (batch-local top-8 uncentered PCA approximating W3's corpus-level
3-dim subspace) — actions must move the ego *through* the world, not repaint the world. The
measured result of this loss is §7.13's repair. Its theoretical significance: it is a
*self-supervised* grounding of the action interface — the analytic target is geometry, not a
label — so it composes with the label-free thesis.

**The H-COTRAIN hypothesis and its curriculum-based test (pre-registered 2026-08-11,
`PREREG_H_COTRAIN.md`).** PI hypothesis: joint WM+planner training forces the trunk toward an
ego-action/trajectory feature extractor rather than a physical-world model. It is testable at
~0 marginal cost because v5f's planner coupling followed a measured **λ_plan curriculum — 0
until step 2,000, linear ramp to 8,000, then 1.0 to 29,999** (`--lambda-plan sched
--phase-a-steps 2000 --phase-b-steps 8000`, `MODEL_REGISTRY §1.5/§1.8`) — and milestone
checkpoints (5 k/10 k/15 k/20 k/30 k) sample the trunk along the ramp. The frozen probe battery
runs per milestone; outcomes bound in advance: a scene variable readable at 5 k (R²(enc) > 0.15)
that declines ≥ 0.15 absolute by 30 k while ego-dynamics probes hold ⇒ CONFIRMED (consequence:
gradient isolation / λ ceiling becomes a named v6 lever); lead-gap unreadable at *every*
milestone including 5 k ⇒ REJECTED for that variable (it never formed — joint training cannot
have destroyed it; the cause reverts to objective/data pressure and the label-free levers keep
priority). An independent SIGReg verdict rides the same runs: effective rank of z_enc at 30 k
≥ 0.8× its 5 k value validates the anti-collapse mechanism under a full planner gradient.
Evidence already in hand cuts both ways and is stated with the prereg: FOR — the predicted
latent decodes ego state *better* than the encoded one (speed R² 0.99 vs 0.74, §7.14) and
the action interface is 3-dimensional; AGAINST sufficiency — PUBLISHED, pure-SSL video models
without any planner also fail to form dense spatial features reliably (V-JEPA 2.1,
arXiv 2603.14482), so a missing lead variable does not *require* a planner-gradient explanation.

⚠️ **OUTCOME (2026-08-11): H-COTRAIN is REJECTED within the measured range, and the SIGReg
verdict is VALIDATED.** Neither CONFIRM condition fired: every probed physical variable became
*more* decodable along the ramp (curvature 0.213 → 0.551 encoded; yaw-rate 0.583 → 0.869) and
the latent's participation ratio *expanded* 4.53 → 6.94 of 2048 (+53 %, retention 1.53× against
a ≥ 0.8× gate). §7.17 carries the full curve, the three scope bounds that limit the claim (the
lowest available λ is 0.5, not 0; the spectrum series ends at 20 k, not 30 k; a real transient
dips the predicted-latent readouts at 10 k before they recover and overshoot), and the retraction
of the erosion premise this paper previously carried. **The staged v6 recipe below is unaffected,
because it never rested on erosion** — it rests on the field's convergent staging evidence and on
the measured consumer-invalidation result (§7.16).

**The staged-training evidence (PUBLISHED + our own gates).** The frontier systems separate
representation learning from planning: V-JEPA 2 → 2-AC pretrains on ~10⁶ h then post-trains an
action-conditioned predictor on 62 h of *unlabeled* robot video for zero-shot latent MPC
(arXiv 2506.09985); DINO-WM learns dynamics over frozen DINOv2 **patch** features — never a
pooled global vector — and plans by MPC with a latent goal cost (arXiv 2411.04983); Drive-JEPA
V-JEPA-pretrains a driving encoder and post-trains a proposal-centric planner over the frozen
predictive features, reaching NAVSIM v1 93.7 PDMS / v2 87.8 EPDMS / Bench2Drive 64.52
(arXiv 2601.22032); AD-L-JEPA shows occupancy forecasting *emerging* from self-supervised BEV
latent prediction, read out by light decoders (arXiv 2501.04969) — the P8 pattern; and
V-JEPA 2.1 treats missing dense spatial features with **loss shaping, not labels**
(arXiv 2603.14482) — the LF-program pattern. Our own gate history is consistent: every *staged*
component passed its gates (W4 head retrofit, stage-A repair, W4r refit), while the co-trained
path produced the muffled action interface, three selector failures and the missing lead
variable (§7.13–§7.14). **Nobody at the frontier co-trains the planning gradient into the
encoder from step 0** — the v6 staged recipe (§10) is the programme's response, with H-COTRAIN
as its cheapest discriminating measurement *(and it has since returned: erosion is rejected in
range, §7.17 — the staging argument stands on the field evidence and on consumer invalidation,
§7.16, not on erosion)*.

### 3.10 The label-free commitment, stated as an admissibility algebra (added v1.0)

The programme's binding design commitment is not "we use few labels" but a statement about
*where in the computational graph a label may appear*. Making it formal removes the recurring
argument about whether a given lever is admissible.

Let 𝒮 = {x (frames), a (CAN/ego actions), p (odometry poses)} be the **self-supervised** signal
set — every element is a sensor reading generated for free on every driven metre — and let
ℒ = {obstacle cuboids, VLM fields, SAM masks, scenario tags, hindsight route geometry} be the
**label** set, produced offline by an external annotator, engine, or privileged hindsight. Write
θ_trunk for the parameters of the encoder f_θ, the readout r and every level predictor P_ℓ, and
θ_read for the parameters of any frozen-latent readout, probe, or goal head. The commitment is:

> **(L1) No element of ℒ may enter any loss whose gradient reaches θ_trunk.**
> **(L2) No element of ℒ may be an input at inference, at any level.**
> **(L3) Elements of ℒ *are* admissible as (a) targets of readouts trained with
> ∂/∂θ_trunk ≡ 0 (frozen probes, the P1/P4/P8 battery), (b) evaluation strata, and
> (c) supervision of the strategic *goal head*, which is a planner-side output and never a
> trunk loss.**

Three consequences we have had to enforce in practice. **(i)** The auxiliary lead-readout loss
proposed when P1 found no lead-distance variable was inadmissible under (L1) and was retracted
the day it was proposed; the replacement is the label-free lever program LF0–LF4, which shapes
the *objective* (masking, sampling, near-field weighting) rather than adding a target (§7.14,
`JEPA_PHYSICS_SURVEY.md`). **(ii)** The P8 occupancy decoder trains on label rasters and is
admissible **only** because the trunk's md5 is proved identical before and after (§7.18) — the
freeze is the admissibility certificate, not a courtesy. **(iii)** Under (L2), the deployable
scenario classifier is the image-only arm, even though the ego-conditioned arm scores better —
because the situation labels are themselves derived from ego dynamics, so an ego input at
inference is partly reading the label's own source. That is the general test this algebra
encodes and which we now run on every head: **does an input at inference contain something the
label was derived from?** It is the same test that caught the nav-echo (a route head scoring
1.0000 as an exact bijection of its own input) and the ~80 % train/val leak in the REF-A I-JEPA
arm.

### 3.11 The 4-brain hierarchy, formally: per-layer predictors, goal conditioning, and the gradient-isolation matrix (added v1.0)

Let ℓ ∈ {O, T, S} index the operative, tactical and strategic levels (the fallback brain C is
out-of-gradient monitoring logic and carries no predictor). Each level owns a state space, a
clock, a horizon, an action space, and **its own predictor in its own space**:

| ℓ | state | adapter | clock | horizon | action a^ℓ |
|---|---|---|---|---|---|
| O | z_O ∈ ℝ^{2048} = r(f_θ(x)) | — | 10 Hz | 0–2 s | continuous controls (a, κ) |
| T | z_T ∈ ℝ^{d_T}, d_T ≈ 512 | z_T = φ_T(sg[z_O(t−3..t)]) | ≈ 1–2 Hz | 2–6 s | tactical goal token g_tac |
| S | z_S ∈ ℝ^{d_S}, d_S ≈ 256 | z_S = φ_S(sg[z_T window]) | ≈ 0.2–0.5 Hz | 6–30 s | strategic goal token g_str |

with dynamics, **goals conditioning strictly downward**:

  ẑ_S(t+K) = P_S( z_S(t), a_S(t) )
  ẑ_T(t+k) = P_T( z_T(t), a_T(t) │ g_str )
  ẑ_O(t+j) = P_O( z_O(t), (a, κ)_t │ g_tac )

Two structural properties are asserted by construction and then *checked*, not assumed.

**(1) Abstraction by temporal down-sampling under a shrinking state.** d_S < d_T < d_O and
τ_S > τ_T > τ_O, so an upper level cannot represent the lower level's detail even if the
objective would reward it: the level is *forced* to abstract. A strategic roll over 15 s is
three steps of a 256-dimensional model — the hierarchy is an efficiency device as much as a
semantic one, since only the operative path runs at full rate.

**(2) The gradient-isolation matrix G.** Let 𝒫 be the set of planner/goal-head parameters and
ℰ the set of encoder parameters. Define G[u, v] = 1 iff a gradient path from module u to module
v is *permitted*. The v6 contract is

  G[𝒫, ℰ] = 0  (no planner or goal head may backpropagate into any encoder)
  G[ℓ', ℓ] = 0 for ℓ' above ℓ  (upward latents pass through sg[·] or an EMA-slow copy)
  G[ℓ, ℓ] = 1  (each level trains its own predictor)

Uplinks are `sg[·]` or an EMA target with decay 0.996 (the V-JEPA teacher pattern) — a choice,
not a default, and it is an arm. The contract is enforced by an **autograd probe** rather than a
comment: `V6Stack.assert_isolation()` backprops the module's declared planner-side surface and
reads which parameters actually received gradient via `torch.autograd.grad(..., allow_unused=True)`
(it never touches `.grad`, so it is safe mid-training), probing the three edges planner→encoder,
tactical→below, strategic→below. Two traps the check itself must avoid, both of which we hit in
design: a **frozen** parameter records no autograd edge, so a check run mid-stage would report
"isolated" vacuously — the probe therefore makes every parameter temporarily differentiable,
because isolation is an *architecture* property, not a training-state property; and a zero-init
head produces a zero gradient that also looks like isolation, which a dedicated test pins.

**Why goals, not latents, are the downward interface.** A goal token is a *finite discrete type
with typed continuous slots* — `(TYPE, args)` in physical units — so the seam is inspectable,
loggable, and testable for the disjointness rule (§3.10, (L2)): no goal token may be derivable
from the situation classifier's output, and every supervised instance carries its derivation
source (`path | signage | vlm-fused`). A latent seam would be none of those things. The
vocabulary itself (strategic `KEEP_CORRIDOR`/`LANE_TARGET`/`EXIT_*`/`TURN_*`/`ROUTE_TO`/`STOP_AT`
with `FOLLOW_MAIN_ROAD` as the no-route default; tactical `ANCHOR_GOAL`/`CORRIDOR_OFFSET`/
`GAP_TARGET`/`SPEED_BAND`/`YIELD_AT`/`STOP_POINT`/`WAIT_FOR_ONCOMING`/`EVADE_IN_CORRIDOR`/
`TRAFFIC_LIGHT_REACT`, each with optional `within_m`/`by_time_s`/`at_arc_m`/`hold_for_s`
constraint slots) is specified in `HIERARCHY_VOCABULARY.md` and is a v6 deliverable, not a claim
of this paper.

**The horizon spec, and why it is one trajectory and not two.** A plan is a **single 60-step
(a, κ) control sequence at 10 Hz integrated through one unicycle rollout, 0 → 6 s**. The 0–2 s
band carries operative control authority and the 2–6 s band is shaped by g_tac conditioning, but
they share one integrator, so the 2 s seam is **discontinuity-free by construction** and the seam
metrics verify rather than repair it. Stitching two trajectories would introduce exactly the
artefact the hierarchy is supposed to avoid. The eval consequence is binding: four families and
oracle/selected are reported at **both** 0–2 s and 0–6 s.

### 3.12 Selection under a surrogate cost: the winner's curse, formally (added v1.0)

Imagine-and-select is `argmin` over a candidate set. §7.16 measures that this fails badly at
N = 256 candidates even when the true best candidate is guaranteed present. The mathematics
below is what makes that a *result* rather than a disappointment, and it discriminates between
two hypotheses using our own numbers.

**Setup.** Fix a window. Let candidates i = 1…N have realised errors e_i (the quantity we would
minimise if we could see the future) and surrogate costs c_i (the quantity we can compute).
Write e_{(1)} = min_i e_i for the **oracle**, and i\* = argmin_i c_i for the **realised choice**.
The object of interest is E[e_{i\*}] as a function of N and of the dependence between e and c.

**Model A — the honest-but-noisy cost (Gaussian copula).** Suppose, after marginal
standardisation, (e_i, c_i) are i.i.d. across i with

  c_i = ρ·e_i + √(1 − ρ²)·ξ_i,  e_i, ξ_i ~ 𝒩(0, 1) independent.

Each c_i is standard normal, (e_i, c_i) is bivariate normal with correlation ρ, and i\* is a
function of c alone — so E[e_{i\*}] = E[ E[e_{i\*} │ c] ] = ρ·E[min_i c_i], giving the **exact**
identities

  E[e_{i\*}] = −ρ · a_N,  E[e_{(1)}] = −a_N,  E[e_{random}] = 0,
  with a_N = E[max of N i.i.d. 𝒩(0,1)] ≍ √(2 ln N).

Two readings. First, since (random − selected)/(random − oracle) = ρ exactly,
**the fraction of the oracle's advantage a minimiser captures is exactly ρ**
— a cost with ρ = 0.45 buys 45 % of the available gain, no matter how good the fan is; this alone
is a strong argument against argmin as an architecture. Second, and decisively for us,
E[e_{i\*}] under Model A **improves** with N like −ρ√(2 ln N): a noisy-but-honest cost gets
*better* on a deeper fan. **Our measurement therefore refutes Model A.** Selection degrades as
the fan deepens (frozen trunk K = 8/32/64 → 0.5772/0.5173/0.5319; repaired trunk N = 256 →
3.3348; different trunks, same direction, §7.16), and the argmin's mean error-rank is 132.3 of
256 against 128 expected under *independence* — that is not a ρ = 0.45 signal in the tail, it is
no signal at all in the tail.

**The reconciliation, and the statistic that matters.** Spearman's ρ is a **bulk** functional: it
constrains average pairwise concordance and places no constraint on the joint law in the lower
tail of c. The quantity selection actually depends on is the **lower-tail dependence
coefficient**

  λ_L = lim_{u↓0} P( F_e(e) < u │ F_c(c) < u ),

i.e. "given the cost is in its lowest quantile, how likely is the error to be too". A cost can
have ρ = 0.45 and λ_L = 0. Our measurement estimates the **finite-m analogue** of λ_L — the
empirical low-cost stratum at m/N between 0.008 and 0.125 — and finds it indistinguishable from
zero across that whole range: conditioning on the m lowest-cost candidates leaves the mean error
at the fan's own mean (5.408/5.327/5.313/5.321/5.321/5.321 for m = 2…32 against a fan mean
≈ 5.32), i.e. **the low-cost stratum is distributed like a random m-subset**. The falling
best-in-top-m *ceiling* (3.380 → 0.356 m over the same m) is consistent with the same reading —
it is what the minimum of a *random* subset does — but the flat conditional mean is the direct
evidence, and the ceiling on its own would not distinguish the two models. **ρ is not a
sufficient statistic for a selection rule; report λ_L (or its finite-m analogue, or the argmin's
error-rank), or do not claim a cost selects.**

**Model B — the degenerate minimiser, which the data supports.** Let the cost decompose as
c(u) = c_roll(u) + λ c_kin(u) over control sequences u. Both terms are *self-consistency /
smoothness* functionals: c_roll(u) = ‖waypoints(u) − decode(roll(z₀, u))‖ measures agreement
between a candidate and the world model's imagination *of that same candidate*, and c_kin
penalises |a| and |jerk|. Let 𝒟_δ = {u : ‖displacement(u)‖ < δ} be the near-stationary set. Both
terms → 0 on 𝒟_δ as δ → 0 (nothing moves, so nothing can disagree and nothing accelerates),
while the realised error is bounded **away** from zero there, because the ego does move:
E[e │ u ∈ 𝒟_δ] → ‖true displacement‖ > 0. Hence if fan candidates are drawn i.i.d. from a law
with positive mass near 𝒟_δ,

  P( i\* ∈ 𝒟_δ ) → 1 as N → ∞,   so  E[e_{i\*}] → E[e │ 𝒟_δ] ≫ E[e_{(1)}] → 0.

**The minimiser converges to the degenerate set while the oracle converges to zero.** This is the
predicted monotone divergence, and it is what we observe. It also predicts the deployed
decomposition exactly: argmin on c_roll **alone** scores 5.2898 m ≈ the fan mean (pure
degeneracy), and the whole of W7-FULL's 3.3348 m comes from the λ = 0.2 kinematic term partially
opposing it.

**The remedy the formalism prescribes.** A selection cost must contain a term that **inaction
cannot minimise**. V-JEPA 2-AC minimises latent distance to a *goal* state and DINO-WM minimises
a latent goal cost — standing still leaves you far from the goal, so 𝒟_δ is not a minimiser
there. We copied the planning loop and dropped the goal term; W7's own anti-degeneracy progress
term (minus arc length) has been at weight 0.0 in every run to date. Beyond adding it, the
formalism points at two structural alternatives, both pre-registrable: **top-m aggregation**
(average or median over a low-cost set rather than its argmin — an M-estimator instead of an
extremum, which is exactly the classical remedy for a winner's curse) and **sharpening**
(learning a monotone recalibration of c against realised error on held-out windows, which raises
λ_L directly and is measurable as such). Both must be pre-registered before use; neither may be
selected on the same windows that measured the failure.

**Why this generalises past our stack.** Every "generate a fan, score it, take the best" planner
— anchored-diffusion decoders, MPC over learned dynamics, best-of-N sampling from a generative
policy — is an argmin over a surrogate. The scaling instinct in that family is to *widen the
fan*, and widening is exactly what makes a degenerate or tail-uninformative cost worse while the
reported oracle improves. **Publishing a fan's oracle alongside its argmin is therefore not a
courtesy; it is the diagnostic**, and the gap between them is the quantity a selection paper owes
its reader.

### 3.13 SIGReg, effective rank, and what "retention" measures (added v1.0)

The anti-collapse mechanism (§3.2) is a distributional constraint, so the honest way to audit it
is a spectral statistic of the representation, tracked as a **series**. Let Z ∈ ℝ^{n×D} be a
batch of encoder states (D = 2048) and Σ = Cov(Z) with eigenvalues λ₁ ≥ … ≥ λ_D ≥ 0. Define

  **participation ratio**  PR(Σ) = (Σ_i λ_i)² / Σ_i λ_i² ∈ [1, D],
  **effective rank**  erank(Σ) = exp( −Σ_i p_i ln p_i ),  p_i = λ_i / Σ_j λ_j,
  **top-k share**  s_k = Σ_{i≤k} λ_i / Σ_i λ_i.

PR = 1 for a rank-one (fully collapsed) representation and D for an isotropic one — which is the
distribution LeJEPA proves optimal and which SIGReg drives toward by forcing every 1-D projection
to 𝒩(0,1) (Cramér–Wold). **"Retention" is a ratio, never a level**: retention(t₁ : t₀) =
PR(Σ_{t₁}) / PR(Σ_{t₀}), gated at ≥ 0.8 across any curriculum phase. Stating it as a ratio is
deliberate — PR's absolute value depends on batch size, centring convention and the free-dimension
count of the `full_relaxed` variant, so a level is not comparable across arms while a
within-arm ratio is. Measured on v5f's own λ_plan ramp: PR 4.53 → 6.94 of 2048 with s₈ falling
0.9903 → 0.9232, retention **1.532** over 5 k → 20 k — expansion, not collapse (§7.17). ⚠️ Two
disciplines travel with it: the ratio's **endpoints must be named** (this one is 20 k over 5 k,
not the 30 k the pre-registration wrote), and the same code must compute the training-time
monitor and the offline instrument, or the monitor and the audit will drift apart — in v6 both
call `tanitad/eval/spectral.py`.

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

*(Structured into named subsections at v1.0; the content of §5.1–§5.2 and §5.6–§5.7 is unchanged
from v0.9, §5.3–§5.4 were added at v0.9, and §5.5 is new.)*

### 5.1 Instrument rows (I1–I7)

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

### 5.2 The estimator correction, and its measured blast radius

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

### 5.3 The decision-grade estimator, stated mathematically

**The decision-grade estimator, stated mathematically (added v0.9).** Windows are not independent —
they are strided samples of episodes, and the correlation lives at the episode level. Let the
validation set be episodes E = {e₁…e_M} (M = 40 on the canonical split), episode e contributing
windows W_e, and let T({w}) be the statistic of interest (a mean error, a paired delta, a rank
correlation). The **episode-cluster bootstrap** draws, for b = 1…B (B = 2000),
M episodes *with replacement* from E, recomputes T on the union of the drawn episodes' windows
(an episode drawn twice contributes its windows twice), and reports the percentile interval
[T*₍.025₎, T*₍.975₎] of {T^(b)}. The point estimate is always the **full-set** T over all windows,
never a mean of split-means. For two arms scored on the same windows the statistic is the
**paired** per-window difference resampled by the same episode draws — never two marginal
intervals combined in quadrature (quadrature assumes independence the pairing exists to exploit).
Implementation: `taniteval/ci.py`; every artifact JSON cited in §7.13–§7.15 stamps
`estimator: episode_cluster_bootstrap` or states explicitly that it is a corpus-grid point
estimate pending the pod-side rescore. The deprecated block is not merely a wrong width: because
its central value is a mean of overlapping split-means, **it moves the point estimate too** —
the correction box above carries the measured blast radius.

**Why the deprecated block biases the *point estimate*, algebraically (added v1.0).** Let the
statistic be a mean error over windows and let split s ∈ {1…8} hold out a random 20 % subset
H_s ⊂ W of the same window pool, with |H_s| = h. The deprecated block reports

  T_heldout = (1/8) Σ_s ( (1/h) Σ_{w ∈ H_s} e_w )   versus   T_full = (1/|W|) Σ_{w ∈ W} e_w.

If every window were drawn into the same number of splits, the two would coincide; they do not,
because the eight holdouts are drawn **independently and therefore overlap unevenly**. Write
m_w = #{s : w ∈ H_s} for the multiplicity of window w. Then

  T_heldout = Σ_w (m_w / 8h) · e_w = Σ_w ω_w e_w  with  Σ_w ω_w = 1, ω_w ∝ m_w,

i.e. **T_heldout is a *randomly re-weighted* mean whose weights are the sampling multiplicities**,
and it equals T_full only if m_w is constant. Its bias is therefore Cov_w(ω_w, e_w)·|W|, which is
zero in expectation over draws but **not** for the single realised draw a published number comes
from — and the realised draw is fixed, so the error does not average away over arms, it varies
per arm. That is exactly the measured signature: headline `ade_0_2s` shifts **−6.67 % to
+11.69 %** across 27 arms, **bidirectional** (11 inflated, 16 deflated, none unchanged), up to
**×3.3** on hierarchy seams and **×−4.15 including a sign flip** on paired deltas, with widths
1.107–3.100× too narrow (median 1.499×). Two rules follow and are enforced: the point estimate is
**always** the full-set mean, and any pre-2026-07-25 number must be checked for which of the two
it is (`MODEL_REGISTRY` publishes both, and they differ).

### 5.4 The eval-tier doctrine (T0 / T1 / T2)

**The eval-tier doctrine (BINDING 2026-08-09; `Project Steering/EVAL_DOCTRINE.md`).** The PI's
critique — *"if the model is consuming at eval the future gt data, then it's not really an eval"* —
was measured before it was adopted, and the measurement re-frames every open-loop number in this
paper. Three tiers:

| tier | condition | may be quoted as |
|---|---|---|
| **T0** | teacher-forced: the predictor consumes the *recorded future actions* | "prediction quality" / WM diagnostic — ⛔ **never "driving performance"** |
| **T1** | action-closed loop: the predictor consumes the decoder/planner's **own** actions; perception context fixed at t₀ | "closed-loop (imagination) driving" — **the primary capability tier** |
| **T2** | perception-closed loop (re-render what the ego would now see) | "closed-loop driving" — **not provisioned** |

The measured basis (`MODEL_REGISTRY §1.12`, T1, 6,834-window grid): removing the recorded future
actions degrades the v1.6/v1.7 unicycle readouts from ADE 0.340/0.285 to 0.471/0.462 (still under
the 0.535 CV floor), net-yaw error ×6.7 — and **S-curve reproduction collapses 0.9785 → 0.0538
(v1.6) / 0.0430 (v1.7), with the hold-action control at 0.0 %**. The open-loop counter-steer was an
**action echo**: the eval fed the model the answer's own action transcript and scored it on
reproducing the answer. Consequences, binding: every registry results block and every number in
this paper from §7.13 on carries its tier stamp; pre-doctrine blocks are stamped retroactively
(§1.10/§1.11 = T0, §1.12 = T1); a capability claim requires T1 or better; T0 keeps its role as the
attribution instrument (it is how the decel-ramp defect was assigned to the readout rather than
the roll). Cross-tier comparisons are invalid. The four-metric-families rule (§5.5) applies at
every tier.

### 5.5 The four metric families: the evaluation contract (added v1.0)

**ADE is one row of four, and an evaluation that reports it alone is incomplete.** This is a
binding contract in this programme (PI, 2026-08-02, after three reports went out with ADE-only
tables), and it is stated here as method because the reason is structural rather than
stylistic: **ADE is the cheapest number to produce and the easiest to compare, so it crowds out
the metrics that decide whether the car drives well.** An arm can win ADE while setting the
wrong speed, and a scalar path error cannot see a decision error at all.

| family | what must be reported | why it is not optional, measured |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy **and distance-keeping** (headway / time-gap / TTC to the lead agent) | **88.7 % of our oracle gap is longitudinal**; and the state variable longitudinal control most needs — lead distance — is the one the latent provably does not carry (§7.14) |
| **LATERAL** | heading error, **curvature error, yaw-rate error**, cross-track | "lateral is fine" has been asserted from cross-track alone; curvature and yaw are where a smooth-but-wrong path shows up — and where the action echo hid (§5.4) |
| **TACTICAL** | manoeuvre-decision quality **and tactical goal-setting** (selected vs executed manoeuvre, class confusion, goal/anchor selection) | the 5-way softmax that **mixes lateral and longitudinal** is our largest known head defect; and selection, not generation, is where both measured hierarchy levels fail (§7.13, §7.16) |
| **STRATEGIC** | strategic decision + goal/route setting quality | the hierarchy is this programme's thesis; if the strategic level is never measured, it cannot be claimed to work |

**Five rules travel with the families.** (1) **Per-family, never pooled** — a single composite
hides exactly the trade-off the decomposition exists to expose; v5.8f's headline trade (+0.08 m
selected ADE for a 16× kinematic improvement, §7.13) is invisible in any pooled score.
(2) Each family carries its **estimator** (paired episode-cluster bootstrap, §5.3) and its CI, on
the *same* windows as the ADE it accompanies. (3) **A missing metric is a work item, not an
excuse** — if a family has no instrument, the instrument is built. (4) A horizon sweep of ADE is
never "the result". (5) Where a family genuinely cannot be computed, it is reported **per family
with its reason and its n**, never silently dropped.

**Where we stand against our own contract, stated honestly.** LONGITUDINAL and LATERAL are
computed on every arm from §7.11 onward. TACTICAL is computed where a fan and a selector exist
(shortlist coverage, winner-hit fraction, selected-rank percentile — e.g. W7-FULL's
winner-hit 3.2 % at sel-rank 34.6 %, §7.16). **STRATEGIC is currently uncomputable on our corpus,
and the reason is a property of the dataset, not of the model**: PhysicalAI-AV contains **no map,
no lane graph, no junction annotation, no traffic-light feature and no route or goal signal** —
its card says verbatim *"we do not include open maps data"* — and its `obstacle.offline` enum
over 87,481 cuboids is 10 classes, **all dynamic agents**; the `egomotion` stream carries no
lat/lon/GNSS (coordinates are clip-local metres), so external map-matching on our traces is not
possible either. This was established at five independent probes and is settled. Every artifact
therefore stamps `STRATEGIC: n/a — no route/goal label exists on PhysicalAI-AV`, with the
`n` it would have had.

**What would close it**, in the order it is planned (§10): a strategic goal-label stream produced
offline by the two-engine labeling pipeline — Engine A (deterministic hindsight geometry over
integrated ego trajectories: corridor continuation, lane-displacement events, junction turns),
Engine B (a VLM reading past+future clips for scenario/domain/signage fields under a strict
schema with OCR evidence required for any `route_to`), and Engine C (SAM-family video
segmentation supplying drivable-surface and instance masks — the closest admissible substitute
for the missing map), fused adversarially with disagreements banked as `disputed`. Under §3.10
those labels supervise the **strategic goal head only**, and are never inputs at inference and
never trunk losses. Until that stream exists, "the hierarchy works" remains an unmeasured claim
at its top level, and we say so rather than substituting a proxy.

### 5.6 Three failure classes added by measurement

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

### 5.7 Two instrument definitions pinned down

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
manoeuvre-vs-trajectory agreement collapses from κ = **0.253** (v1) to κ = **0.0072**
(v2corpus), both at the published direction gate `DIR_YAW_RAD = 0.15` — the declared manoeuvre is
unrelated to the driven path; the `--v2` pack's tactical machinery is **decorative**.

> ⚠️ **ANNOTATION (2026-08-17, PI-directed) — THIS COMPARISON IS CONFOUNDED WITH A LABEL-DEFINITION
> CHANGE, AND THE TWO CAUSES ARE NOT SEPARABLE FROM BANKED DATA.**
> `DIR_YAW_RAD` is **not an eval constant**: `hierarchy.py:169` aliases `refb_labels.YAW_TURN_RAD`
> — the **training-label** threshold, same horizon, same wrap, same number. And for `--v2` arms **no
> gate value applies at all**, because `classify_maneuver_v2` gates **CURVATURE (1/m)** where v1
> gates **NET YAW**; the registry (§1.7/§1.6) confirms `--v2` implies `--labels-v2`. ⇒ *"both at the
> published direction gate 0.15"* describes **two different quantities**, not one quantity measured
> twice.
>
> ⚠️ **The claim is NOT withdrawn**, and the distinction matters: v2corpus's manoeuvre head **is**
> genuinely degenerate (**404/418 windows one class**, 96.7 %), and that degeneracy is **gate-free**
> — it survives every re-read below. **What cannot be attributed is the MAGNITUDE of the drop**: how
> much of 0.253 → 0.0072 is the corpus and how much is the label definition is unrecoverable
> without a re-score, and `man_pred` is on disk **nowhere** (four independent probes: 1,929 JSONs →
> 0 hits; all 60 window dumps carry no manoeuvre key). A GPU pass is the only path.
>
> ⇒ **Admissible reading:** *"v2corpus's tactical head is degenerate and its manoeuvre declaration
> is unrelated to the driven path."* ⛔ **Inadmissible:** *"the v2 corpus caused a κ collapse of this
> size."* A separate, unrelated scoring defect was also found and fixed on this path
> (`hierarchy.py:592` hardcoded **v1** labels for every `--v2` arm — see `…/incoming/
> 2026-08-17-maneuver-label-mismatch/`, landed in `c98aadb`); it does **not** clear this confound,
> which runs through `man_pred` vs `traj_dir` and never touches `man_tgt`. **Keep the two
> retractions separate.**
> Provenance: `…/incoming/2026-08-17-diryaw-reread/DIRYAW_REREAD.md`. ⚠️ Re-read at a
**0.10** gate (0 GPU; exact envelope over the banked marginals, which reproduce both published κ to
4 dp): the word **decorative holds unconditionally** — v2corpus's κ stays within [−0.040, +0.057]
for *any* crossing rate up to 33 % of windows and never reaches the 0.1 line, because the manoeuvre
head is degenerate (404/418 windows one class) and that degeneracy is **gate-free**. The **collapse
itself also holds**: it could only fail at a 14.6 % crossing rate, ~3× the largest band mass ever
measured here (3.97 % canonical val, 5.34 % OOD-val). ⛔ What does *not* survive is v1's own verdict
word: at 0.10 a crossing rate of just 0.96 % already admits **SUBSTANTIAL**, so v1 is quoted here
without one — it is bounded away from DECORATIVE (which needs 7.4 %), and nothing more is
established. Strategic: under nav-dropout 0.5 the arm *learned the dropout* —
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
is deferred until a leak-free v2-line validation split exists (C64, option B). *(Status update
v0.9: v5f completed its 30 k on 2026-08-09 — its final numbers, the wedge ladder built on it,
and the stage-A repair arc are §7.13.)*

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

### 7.12 Closed-loop on a neural reconstruction, on the edge device: REF-C beats the flagship and the whole separation is LATERAL (2026-08-03)

**The strongest test of the four-family doctrine so far, and it is the one where ADE sees nothing.**

**Setting.** Nine rollout starts x 50 ticks in a NuRec 3D-Gaussian reconstruction, rendered **on the
Jetson Thor** (aarch64 Blackwell sm_110). Estimator: episode-cluster bootstrap, clusters = rollout
starts, n = 9; paired comparison over the **437 shared windows**.

| family | flagship v1 | REF-C base |
|---|---|---|
| **ADE** `ade_0_2s` | 3.6432 [2.400, 5.267] | **2.7341** [1.711, 3.787] |
| `dist_to_gt_traj_m` | 2.2393 | **0.8856** |
| **LON** target-speed error (signed) | **-2.0412** [-3.751, -0.432] | **+1.3307** [0.288, 2.371] |
| **LAT** heading error (rad) | 0.1368 | **0.0422** |
| **LAT** curvature error (1/m) | 0.0074 | **0.0018** |
| **LAT** yaw-rate error (rad/s) | 0.0555 | **0.0152** |
| **TAC** executed == planned | **0.4481** | **0.8741** |
| **STR** corridor departure | **0.3533** | **0.1304** |

**Paired flagship - REF-C, separated:** `dist_to_gt` **+1.171 [0.030, 2.244]**, heading
**+0.084 [0.028, 0.175]**, curvature **+0.0050 [0.0008, 0.0130]**, yaw-rate **+0.038 [0.020, 0.057]**.
⛔ **ADE is NOT separated: +0.789 [-0.865, +2.728].**

⛔ **RETRACTED THE SAME DAY (R-2026-08-03-C) — the numbers above are the OLD render.** Re-measured on
the improved render the shipped videos use (+23.4 % grad-NCC, same scene, checkpoints, starts, scorer
and 437 paired windows): **ADE separates at +7.164 [+5.265, +8.966]**, `abs_target_speed_err_ms` at
**+6.397 [+5.000, +7.801]**, `along_track_ade_m` at **+7.153 [+5.240, +8.953]** and corridor
departure at **+0.506 [+0.382, +0.629]**; all four lateral separations survive and widen.
**REF-C still wins — "entirely lateral" does not.** ⚠️ Note also that `dist_to_gt` and
`cross_track_abs` are the SAME measurement (`cl_metrics.py`: `"dist_to_gt": abs(ct)`), so this is
four separated lateral metrics, not five. The mechanism is a **21× render-sensitivity ratio**
(flagship's driven path moves 9.05 m mean under the render change, REF-C's 0.43 m; flagship's
commanded speed falls 12.96 → 7.05 m/s against a logged ~15.0), isolated by a 2×2 to the **scale
cull**, and licensed by a determinism control that returned **exactly 0.0** on 450/450 windows.
Current panel: `stack/experiments/alpasim-gsplat/results/closedloop-hq-render/`.


> ⛔ **SUPERSEDED 2026-08-03 — the reference video is offset from the rig by a per-scene
> constant** (`+6` on `00040136`, `+5` on `7c72937c`; rule: `video_idx = rig_idx +
> (n_mp4_decodable − n_rig_frames)`, measured by the renderer, unanimous over 12 frames each).
> Re-baselined against the **aligned** reference the improvement is **roughly half the size and
> does not replicate**: `00040136` n=5 **+13.5 %** (was +23.4 %), n=12 **+8.0 %**, and
> `7c72937c` n=12 **+4.4 % — NOT SEPARATED** [−0.0097, +0.0521]. Absolutes move too:
> BEFORE 0.2774 → **0.4228**, AFTER 0.3424 → **0.4800**. The render is still better; the
> magnitude quoted here is not. Corrected table + estimator:
> `TanitAD Research Hub/Evaluation/Implementation/incoming/2026-08-03-render-rebaseline/`;
> `RETRACTION_LOG.md` R-2026-08-03-align. ⚠️ No closed-loop conclusion moves — `cl_metrics.py`
> never opens the reference video.


⇒ **An ADE-only table would have reported "no difference" on a comparison where four lateral
measures separate cleanly and in the same direction.** This reproduces the 2026-07-23 native-1080
n = 12 suite on **different hardware, a different renderer and a different scene** — the doctrine is
not an artifact of one harness. ⚠️ **On the improved render this panel no longer demonstrates that**
(ADE separates there); the doctrine's claim — that ADE *can* hide a real gap — is what it showed,
and the 07-23 suite remains the un-retracted instance.

**Three defects only the families expose.**
1. **The arms fail longitudinally in OPPOSITE directions** — flagship 2.04 m/s too slow, REF-C
   1.33 m/s too fast. Any pooled longitudinal score cancels this to near zero.
2. **The flagship does not execute what it selects** (0.4481 vs REF-C's 0.8741) — a decision-quality
   failure invisible to any displacement metric.
3. ⭐ **REF-C's 5-way manoeuvre head never emits a longitudinal class.** Head share
   `lane_keep 0.63 / turn_right 0.37 / accelerate 0 / brake_stop 0`, while **42 % of logged windows
   are `brake_stop`**. This is the programme's documented "5-way softmax mixes lat+lon" defect,
   observed for the first time **in closed loop** rather than inferred from open-loop statistics.

**Objects vs empty road: NULL for both arms.** 19 of 20 paired deltas have CIs containing zero;
headway is statistically identical with and without the other vehicles rendered. **Neither arm
changes its driving when traffic becomes visible.** Bounded honestly: the agents cover 0.02-0.4 % of
frame at 40-45 m (~2.8 s gap), so this is evidence about *distant* traffic only — a cut-in /
close-following scene is the discriminating follow-up, not a claim that vision is ignored.

**An instrument property that constrains every future sim number.** The renderer is a **step function
of pose**: identical pose reproduces bit-exactly, but perturbations from 1e-9 to 1e-4 rad all produce
the *same* mean pixel delta (8.7-11.0/255) with no growth in between — discrete blend-order ties
among 3.1 M overlapping semi-transparent gaussians, not floating-point precision. The decision-level
cost is large: on one identical 10-frame window the flagship's 2 s waypoint moves **0.0000 m** on an
in-process repeat, **4.59 m** through the gRPC float32 pose round-trip, and **6.65 m** under a
0.1 px camera rotation. ⇒ **All production numbers must come from ONE numerical path, and
"bit-identical" is the wrong acceptance criterion for a splat renderer.** It is also an independent
measure of how far OOD these arms are on reconstructions.

**Scope, stated rather than implied.** This satisfies AlpaSim's renderer **wire contract** with a
gsplat backend and drives it from a TanitAD closed-loop harness; it is **not**
`alpasim_runtime.simulate`, so there is **no AlpaSim collision/offroad/scene score here**. The
strategic family is reported but **degenerate on this clip** — `route_head_eq_logged` = 1.0000 is a
constant-predictor tie, and the 20 s scene contains no junction-scale decision; **a junction scene is
required before any strategic-accuracy claim**. TTC is reported as n = 0 with its reason rather than
dropped. ⚠️ And as established at n = 4 scenes previously, closed-loop rates on reconstructions are
**within-sim relative** (REF-C open-loop ADE 1.5157 here vs 0.4728 on real footage = **3.21x OOD**):
orderings survive, absolute rates do not.

⭐ **A capability result alongside the science.** NVIDIA's NRE renderer is amd64-only, but the
reconstruction format is not closed: `volume.nurec` is gzip + MessagePack, and **gsplat renders it
natively on aarch64 including the f-theta camera model at 16-28 ms per 1920x1080 frame**, whole
closed loop **0.09-0.21 s/step (5-11 Hz)**. Validation is by **gradient-NCC** against the scene's own
shipped reference video (0.3664 correct vs 0.2052 best-wrong over the wire, argmax 0). ⛔ PSNR and
plain NCC are **inadmissible on this clip** — a *wrong* reference frame outranks the correct one
under both, because every frame is a dark night street. **On low-dynamic-range corpora the negative
control must choose the metric before the metric is quoted.**

### 7.12b The v1arch/v1.6/v1.7 line: the first complete four-family block, a latents-only unicycle readout, and the action-echo measurement that forced the tier doctrine (2026-08-05 → 08-06)

**v1arch — the data axis, isolated for once (`MODEL_REGISTRY §1.9`).** `flagship-v1arch-v2bal-30k`
trains the *v1 architecture unchanged* (every `v2_*` lever measured `false` in its own
`config.json`) on the 9,000-clip balanced corpus — so, unlike the falsified §7.11(a) arm, the
data axis is attributable. Its corpus breaks canonical-val disjointness (21/40 episodes inside
the training pool — canonical-val numbers inadmissible for it), so evaluation moved to
PhysicalAI-AV's **own official eval split** (`physicalai-oodval-6f4b94e4c7ce-q90`: 290 clips,
zero overlap, 6,382 windows / 290 episode clusters). There it produced **the programme's first
complete four-family block** (`_complete: true`; T0, retroactively stamped; episode-cluster
bootstrap): ADE(4wp) 0.5752 [0.5370, 0.6142]; LONGITUDINAL — a systematic **over-speed prior,
not a tail** (speed bias +0.484 m/s; 71.95 % of windows ahead at 2 s, 75.51 % faster than the
human); LATERAL tight (cross-track MAE 0.0552 m [0.0500, 0.0611]) — not where effort belongs;
TACTICAL — κ 0.6033 (substantial; the manoeuvre label is honest) but **`seams_beneficial` = 0 of
3, the hierarchy seam falsified at this checkpoint**; STRATEGIC — `route_acc_follow` 0.8031 ==
the majority-straight rate with a follow-prediction distribution of {left 0, straight 1737,
right 0}: **a constant predictor, confirmed off-leak**, and `route_acc_nav` 1.0000 is an echo of
its own input. Two harnesses disagree by 0.8 % on this corpus (0.5705 vs 0.5752) — recorded, not
smoothed over, unresolved.

**v1.6 — the unicycle step-readout: what a control parameterisation alone buys
(`MODEL_REGISTRY §1.10`, T0, paired episode-cluster bootstrap over 40 episodes, 6,834 windows).**
A **2.11 M** trainable `UnicycleStepReadout` on the *entirely frozen* v1arch trunk (md5-proved)
maps latent transitions to per-step (accel, yaw-rate), integrated non-holonomically — and,
crucially, **latents-only**: the v₀/feedback shortcut surface was removed after run 4 failed the
WM-reliance gate at 0.0891. Against the trunk's own displacement readout on identical latent
rolls (a decoder-only contrast by construction): ADE parity (0.3398 vs 0.3584, not separated) —
but **speed bias +0.3793 → −0.0265 m/s, accel RMS 2.9465 → 0.7172 m/s² (human ≈ 0.91), jerk RMS
36.17 → 1.13 m/s³ (human ≈ 1.71), net-yaw error −65 %, cross-track −28 %, all CI-separated**, and
replan accel-jump 11× lower. WM-reliance 0.6233 (gate ≥ 0.5 PASS; the frozen-latents control
collapses to the CV floor — what the decoder consumes is the predictor's rolled prediction).
Against real traffic (paired lead-block join): v1arch is CI-separated *more aggressive* on all
three distance-keeping metrics (min-TTC 2.8 s lower), while **v1.6 is statistically
indistinguishable from ground truth on every distance-keeping metric** at this n; its
executed-manoeuvre agreement with GT rises 0.5016 → 0.7694 and its toggle rate matches GT
(0.0309 vs 0.0318, Δ not separated). A physically-parameterised emission surface, at 0.8 % of
the trunk's size, removed most of the longitudinal-defect *expression* — the lesson W4 (§7.13)
carries into v5.8f. **v1.7** (run 6, one change: a speed-profile L1) is the discipline example:
its pre-registered decel-lag hypothesis was **refuted — outcome B binds** (P1 response ratio
0.1547 vs ≥ 0.40, P2 lag +0.173 s vs ≤ +0.15; `MODEL_REGISTRY §1.11`), even though the arm is
CI-better on ADE (0.2849 vs 0.3398, −16 %) with every non-regression gate green — registered as
the best open-loop head in the lineage, **not** as the lag fix.

**§1.12 — the measurement that re-framed every open-loop number in this paper
(`MODEL_REGISTRY §1.12`, T1).** Rolling the predictor on the *decoder's own* actions (no recorded
future anywhere; perception context fixed — imagination-closed loop): v1.6 ADE 0.3398 → 0.4714
(+0.132 [0.112, 0.152]), v1.7 0.2849 → 0.4616 — both still under the 0.5352 CV floor, retaining
~33 % of the open-loop advantage — but net-yaw error ×6.7 and **S-curve reproduction 0.9785 →
0.0538/0.0430, with the hold-action arm at 0.0 %**. The open-loop counter-steer came from the
true-action conditioning, not from vision: **open-loop lateral "skill" was an action echo.** The
consequence is the binding tier doctrine of §5; the numbers are repeated here because this is
the experiment that produced it.

### 7.13 The v5f flagship and the repair arc: one measured defect under four failures, and its predictor-scale repair (2026-08-09 → 08-11)


> **Figure 2 — `Paper/figures/v58f_results.svg`** (generated by
> `Paper/figures/make_v58f_results.py` directly from the registry literals, so the plot
> cannot drift from the source of truth): the repair arc in five measured panels —
> action-response gain against its pre-registered band, fan oracle ADE, selected-trajectory
> acceleration MAE, selector calibration against the P7 gate, and the imagination ablation.
> The figure's footer carries the open item: selection over a frozen-selector shortlist
> still fails its gate.

This is the round in which the programme's diagnostic machinery earned its cost: a ladder of
pre-registered wedges converged four independent failures onto **one measured root defect in the
action interface**, a predictor-only post-training repaired it, and a chain of eliminations then
isolated the last stale component of the selection pipeline. Every number carries its tier; unless
an interval is printed, values are corpus-grid point estimates over the fixed 881-window /
40-episode held-out grid with the episode-cluster bootstrap named as the decision-grade rescore in
the artifact itself.

**The arm.** `flagship-v5f-w120-30k` (`MODEL_REGISTRY §1.8`) is the §7.11(f) relaunch completed:
from-scratch **joint WM+planner co-training** (v1's own co-evolution recipe, §7.6) under the
measured λ_plan curriculum (0 → ramp(2 k..8 k) → 1.0, §3.9), with conditional imagination live
(the §3.4 mechanism, restored in §7.11(f)), on the 120° cylindrical wide-FOV geometry (256×640,
f_ref 305.5775, subframe 176×624) over the w120 sibling of the parity corpus (2,400 clips,
sha-verified against the committed manifest, skip-hash lineage `f09e44db`). Completed at step
30,000 on 2026-08-09. Final eval **[T0 — teacher-forced WM diagnostic, never driving
performance]** on the full 600-episode w120 val corpus, 881 windows (`MODEL_REGISTRY §1.8`):
**selected ADE@2s 0.4011 m** against a 256-candidate fan whose **oracle (best-in-fan) is
0.1975 m** — a selection gap of 0.2036 m, i.e. *the arm would be ~2× better if it merely chose
correctly among candidates it already generates*; miss@2 m 0.1487. ⚠️ w120 geometry ⇒ **not
comparable to any 256²-pinhole number** in §7.1–§7.8 (cross-frame). The four families
(`v5f_four_families_30k.json`, T0, episode-cluster bootstrap): LONGITUDINAL speed MAE 0.7024 m/s
[0.6311, 0.7829], bias +0.1009 — and **accel MAE 8.1075 m/s²**, the number that names the
disease; LATERAL heading MAE 4.08°, yaw-rate MAE 49.7 °/s, cross-track MAE 0.1819 m
[0.1274, 0.2471]; TACTICAL and STRATEGIC honestly UNAVAILABLE with reasons stamped (a WM-fidelity
pass does not traverse the hierarchy — recorded as a work item per the binding rule, not a pass).

**The wedge ladder, and what each rung eliminated (`MODEL_REGISTRY §1.13`).** **W1** — the
pre-registered hypothesis that a kinematic-cost re-rank of the selector's top-8 closes ≥ 30 % of
the selection gap — was **refuted**: re-ranking *worsens* selection by −16.7 % (0.4011 → 0.4351;
`x0_lite_f32.json`, T0). **W2**, the fan-feasibility census, explains why: **97.6 % of all
256×20×881 fan steps violate |a| ≤ 4 m/s² ∨ |yaw-rate| ≤ 0.33·v+0.05, 100 % of candidates are
infeasible including the oracle**, and mean |accel| over all candidates is 252.1 m/s² — any
waypoint-space kinematic cost ranks jitter, not manoeuvre quality. **W2b** (exploratory): a free
3-tap smoother improves *both* selected and oracle ADE (0.4011→0.3975, 0.1975→0.1879) while
cutting selected accel MAE 8.10→3.09 — the signature of truncated-denoise **noise around signal**,
not signal content. The waypoint jitter was hiding coverage, not providing it.

**W4 — the unicycle emission head: feasibility by construction.** Instead of emitting waypoints,
a 109,096-parameter `UnicycleEmission` MLP off the (frozen) offset-head query emits per-step
controls squashed into the physical envelope — **a = a_max·tanh(·), κ = κ_max·tanh(·)** (a_max =
4 m/s², κ_max = 0.2 m⁻¹) — integrated non-holonomically to waypoints; trunk and every other head
frozen (md5-identical before/after). Result (`w4_gate.json`, T0, same grid): both pre-registered
gates **pass** — selected-candidate accel MAE **9.297 → 0.774 m/s²** (winner candidate 0.261;
census-violation fraction 0.0) *and* the fan oracle nearly halves, **0.1991 → 0.1077 m** — the
jitter was masking coverage. The one regression is diagnostic gold: the **frozen selector**,
trained against the old fan's geometry, is near-uninformed on the new fan (selected 0.7933 ≈
CV-floor territory) — a selector-calibration defect, not a fan defect.

**The fast-selector elimination chain (three surfaces, one failure mode, retired by
pre-registration).** **W4b-feat** (rescorer off the offset-head query): held-out selected ADE
**0.5600** vs gate ≤ 0.45, with the train monitor at 0.21–0.33 — the scorer **memorises
train-window selection** rather than generalising it (`w4b_gate_feat.json`). **W4b-kin** (adds
the candidates' own (a, κ) to the input): **0.5637** — a < 0.004 move; the failure is not the
input surface (`w4b_gate_kin.json`). **W4c** (spatial cross-attention scoring, the REF-C conf
mechanism): **0.6609**, near-uniform entropy 5.37, the same train-vs-held-out memorisation gap
0.139 (`w4c_gate.json`). Per the bound G-null, **fast per-candidate scoring on this trunk is
retired** — no fourth attempt without new evidence; a fast selector may return only as a
distillation of the roll-based mechanism. The same pattern reproduces **one hierarchy level up**:
the first trained tactical stage-0 layer (E4.4, 6.39 M on the frozen trunk) fails its goal-FDE
gate **by selection, not generation** — selected goal FDE@4s 12.86 m vs CV 6.09, while the
8-candidate goal fan's *oracle* is 5.28 m and beats CV at both 4 s and 6 s (9.92 vs 12.46)
(`e44_gate.json`, T0). **Fans generate adequate hypotheses at both measured levels; learned
pooled-feature selectors fail to find them.**

**W7 — selection by rolled consequence, and the convergence.** The world-model-roll re-rank
(roll the top-K candidates through the WM, score by explicit costs — structurally the V-JEPA-2-AC
planning loop) failed its gate at every K on the frozen trunk, but the K-sweep is the finding
(`w7_gate_k{8,32,64}.json`, T0): at K = 8 the roll-cost is the programme's **first calibrated
selection signal** (Spearman ρ 0.399 against realised error, vs 0.05–0.26 for every learned
scorer), yet at K = 32/64 the shortlist oracle improves hugely (0.182/0.142) while selection
stalls and the calibration **collapses** (ρ 0.106/0.047) — with many similar good candidates the
muffled rolls barely differ, and the cost drowns. **Why muffled: W3.** The counterfactual
action-response probe pack (`w3_gate.json`, T1-diagnostic — hypothetical actions probe
controllability, never driving performance) had measured, on the same grid: lateral
sign-correctness **99.5 %/99.2 %** (left/right, gate ≥ 0.95 ✓) but **median response gain
0.271/0.233 against the unicycle-analytic reference at 1 s** — the WM turns the right way at
**~¼ the physical magnitude**; longitudinal sign only 74.5 %/78.7 % ✗; and (P6) the
action-induced latent change lives in a **3-dimensional subspace** of the 2048-d state (gate
≤ 32) carrying 0.91–0.94 of the lateral energy. One defect — a correctly-signed, low-rank,
**muffled** action interface — now explains four separately-observed failures: the §1.12 action
echo (closed-loop near-straight driving), the W4b/W4c scoring failures (features cannot separate
candidates whose rolled consequences barely differ), and W7's dense-K collapse.

**Stage-A post-training: the repair (`stage_a_gate.json`, T1-diagnostic, all gates pre-registered
with both outcomes bound).** Predictor-only training under the §3.9 control-response loss
(3,000 steps, lr 1e-5, 91.36 M trainable predictor, 185.6 M frozen; ~4 h): **lateral gain
0.271/0.233 → 0.971/0.966** (gate [0.5, 2.0] ✓✓), lateral sign stays 1.0, **longitudinal sign
0.745/0.787 → 1.0/1.0** (gate ≥ 0.95 ✓✓, gains 0.972 reported), **the action subspace stays
exactly 3-dimensional** (factorisation preserved), and the no-harm gate passes with the factual
roll *improving* (ADE 0.176 → 0.119 vs a ≤ +10 % cap). The single root defect is closed at
head-scale cost — without touching the encoder and without a label anywhere.

**The elimination chain closes on one stale part.** W7 on the repaired trunk first returned a
gate FAIL that is an **instrument-composition failure, not a repair verdict**
(`w7_gate_repaired_k32.json`, T0): stage-A necessarily moved the trunk's feature distribution,
and the W4 emission head + frozen selector — both trained on frozen-trunk features — no longer
compose with it (in-run frozen selector 3.448 vs its banked 0.7933; fan oracle degraded to
0.289). The repair's signal survives inside the same artifact: across-window roll-cost
calibration **ρ 0.7164 [0.5847, 0.7696]** (episode-cluster bootstrap, n = 881; P7 gate PASS —
~1.8× the frozen trunk's 0.399, **the strongest calibration measured in the programme**), and
W7's pick beats or ties the frozen pick on 71.1 % of windows. **W4r** — the cheap refit of the
unicycle head on the repaired trunk — then **passes** (`w4r_gate.json`, T0): fan oracle
**0.1273 m** (cap 0.2173 ✓), winner accel MAE **0.276 m/s²**, violations 0.0 — the fan on the
repaired trunk is healthy — while the same fan through the still-frozen selector reads 4.416,
and W7-w4r (K = 32) fails at 3.614 **because the shortlist pruner is the frozen selector**
(`w7-repaired-w4r-k32/w7_gate.json`). Chain of eliminations complete: trunk repaired, head refit
PASS, roll-cost calibrated — **the frozen selector is the last stale component, and it sits in
W7's pruner, not its cost**. W7-FULL (top-K = 256, selector-free: roll-cost + kinematic cost over
the whole healthy fan — the first selection read with no stale part anywhere) **has since run:
it fails at 3.3348 m against a 0.4505 m gate over a fan whose oracle is 0.1273 m, and the failure
is diagnosed rather than merely recorded — the argmin's error-rank is the median of the fan. That
is §7.16, and it closes the selection question with a mechanism (the winner's curse) rather than
another confound.**

**v5.8f assembled — first T0 numbers, honestly not yet a release row (`MODEL_REGISTRY §1.14`).**
The assembly (frozen v5f-30k trunk + W4 fan + gate-decided selector), two arms on the same 881
windows, episode-cluster bootstrap (`v58f_rescore_ci.json`, T0): **rescorer-top8-kincost 0.4815 m
[0.3928, 0.5771]** vs frozen-argmax control 0.7933 [0.6414, 0.9757] (CI-separated), selected
accel MAE **0.515 vs the v5f baseline's 8.10 m/s² (16×)**, oracle 0.1077 (1.8× better than
v5f's 0.1975). The honest reading: v5.8f currently trades **+0.08 m selected ADE for a 16×
kinematic improvement and a fan whose oracle nearly halves**; the whole deficit is selection
(sel_gap 0.374 [0.3014, 0.4489], `p7_regrade.json`) — and unlike v5f's, this gap sits over a
*feasible* fan with 0.37 m of recoverable headroom. Notably, on the clean fan the W1-refuted
kinematic cost becomes *useful* as a tie-breaker (top8+kincost 0.4815 beats the trained
rescorer's argmax 0.560) — exactly as the fusion design predicted. **Missing before any release
row or capability claim: the four families + cluster CIs on the banked windows, and the T1
(action-closed-loop) rows** — T0 supports none of the above as driving performance.

**What §7.13 establishes, stated precisely.** (i) The failure of imagine-and-select in this
generation was never the imagination: fans are adequate at both hierarchy levels (oracle 0.108 m
operative, 5.28 m tactical goals) and the fan's uncertainty is calibrated (§7.14 P7). (ii) Three
learned per-candidate scoring surfaces fail identically by memorisation, and selection by
**rolled consequence** is the only mechanism whose score correlates with realised error — the
programme's central thesis (plan by imagining consequences) arriving **by elimination**.
(iii) The action interface of a jointly co-trained trunk was correctly signed, low-rank, and
muffled at ~¼ gain — and this is *post-trainable at predictor scale* with a self-supervised
geometric loss, without labels and without touching the encoder. (iv) A repair that moves a
trunk's feature distribution silently invalidates every consumer trained on the old features —
instrument-composition is a failure class of its own, distinct from model failure, and the gate
that "failed" is the instrument saying so.

### 7.14 The WM-physics proof battery: P1–P9 + I4 (2026-08-10 → 08-11)

The PI's question — *"how can [we] prove that the WM, encoder and predictor, is learning
correctly the physical world and predicting the right relevant part of it?"* — was answered with
a battery of pre-registered probes rather than a narrative
(`incoming/2026-08-07-hierarchical-wm-redesign/WM_PHYSICS_PROOF.md`). It decomposes the question
into three falsifiable properties: **(A) physical correctness** (predicted futures obey the
physics that generated the data), **(B) relevance** (the latent carries the variables driving
needs and *not* nuisance detail — the information-bottleneck signature: driving-state
decodability up, clip-identity decodability down with horizon), and **(C) causal grounding**
(the prediction responds to actions the way the physical system would — the property T0 evals
are structurally blind to, and §1.12 measured how expensive that blindness is). Nine probes plus
an imagination-validation triplet, each with its gate fixed before running; every probe stamps
its tier (P1/P2/P4/P6/P7 are T0-diagnostic by design — they interrogate representations; P3/P5
are T1), and a probe that cannot run reports its reason per-probe rather than being dropped. The
battery is offered as a methods contribution: it is cheap (largely 0-GPU off banked dumps), it
runs on existing corpora, and in its first two days it changed the programme's direction twice.

- **P1 — decodability curves (A).** Linear probes on the *predicted* latent ẑ_{t+k} vs the same
  probe on the *encoded* true frame z_{t+k}. Result (`p12_gate_clsfilter.json`, T0-diagnostic,
  episode-disjoint out-of-fold): gate **PASS on all three computable driving targets** — at
  k = 10, speed R² **0.9934 (pred) vs 0.7441 (enc)**, curvature **0.7595 vs 0.5512**, yaw-rate
  similar, no cliffs. The predicted latent decodes driving state *better* than the encoded one —
  the rollout carries the action-implied state strongly (caveat stamped in-artifact: partly the
  action conditioning itself; the T1 lens applies).
- **P1 lead-gap — the battery's sharpest negative, and an instrument-first resolution.** The
  lead-vehicle gap probe first failed uninterpretably (R²(enc) ≤ 0 — "fix the probe before
  gating the predictor", stamped in-artifact). The class-agnostic label join was a real defect;
  fixing it (vehicle-only lead candidates, n = 266 lead windows per horizon) **did not dissolve
  the failure**. The pre-registered follow-up then excluded parameterization and probe capacity:
  linear probes on log1p-gap, inverse-gap and a TTC proxy all fail (R² −5.7 to −154, overfit
  symptoms at this n, direction unambiguous), and the **2-layer-MLP capability ceiling reads
  R² −0.334** on the encoded latent (`p1_lead_transforms.json`; small-n caveat stamped). Verdict,
  model-class: **v5f's latent does not carry a readable lead-distance variable, in any
  parameterization, at any probe capacity tested** — load-bearing because 88.7 % of the oracle
  gap is longitudinal, and this is the missing longitudinal state variable. The label-free
  interpretation (§3.9, PUBLISHED anchors): JEPA risk behaves like a low-rank factorization of
  the action-conditioned co-occurrence operator, and **rare-event variables are exactly what a
  fixed-rank latent sacrifices first** — a free-flow-dominated corpus makes lead distance a
  rare-event variable, so the objective *can* be minimised while ignoring it. ⚠️ The aux
  lead-readout loss originally proposed as the fix was **retracted the same day by the PI**
  (labels into the trunk break the self-supervised thesis); the response is the pre-registered
  **label-free lever program** (`JEPA_PHYSICS_SURVEY.md`): LF0 *locate first* — probe the
  pre-pool spatial tokens and the P8 decoded-BEV read-off (pooling is where geometry goes to
  die, the DINO-WM lesson; probe-only, admissible) — then LF1 interaction-weighted sampling
  (ego-kinematic saliency, from actions alone), LF2 masked-latent prediction over the readout
  grid (I-JEPA pattern), LF3 dense distance-weighted near-field loss (V-JEPA 2.1 pattern), LF4
  longer-horizon rollout targets — each gated on the *same frozen* P1 lead battery.
  Headway/TTC remain GT-join instruments, never latent probes.
- **P3/P6 — counterfactual sign/gain maps and ego/scene factorisation (C).** Reported in §7.13
  (they are the W3 pack): sign right, gain ~¼, subspace 3-dim; P6 gate PASS decisively, P3 FAIL
  with the load-bearing structure that stage-A then repaired. A WM failing P3 does not model the
  world — it replays it.
- **P7 — uncertainty calibration of the fan (B).** On the v5f fan+selector pairing: endpoint
  dispersion **Spearman ρ 0.4915**, selector entropy 0.3954 (gate ≥ 0.3; permutation p ≈ 0,
  n = 881; `p7_calibration.json`) — where the fan spreads, it errs; the model knows what it
  cannot predict. The registry-grade eid-clustered rerun on the v5.8f arms **fails both**:
  frozen-argmax 0.2622 [0.0909, 0.4098] (miscalibrated by the re-parameterisation, not dead),
  the W4b rescorer **0.0542 [−0.1395, 0.2387]** at near-uniform entropy 5.41 — no uncertainty
  information at all, consistent with the memorisation verdict (`p7_regrade.json`). Calibration
  is restored by W7's roll-cost on the repaired trunk (ρ 0.716, §7.13).
- **P5 — compounding-error boundedness (A, T1).** The T1 instrument itself was validated by a
  byte-close control: the E1.4 pipeline reproduces the banked §1.12 dump byte-exactly
  (max |Δ| = 0.0; `e14_t1_v16_bytecheck.json`). The v5.8f T1 rows ride it next.
- **P8 — BEV occupancy readout, attempt 1: an instrument lesson worth keeping (B).** A ~1 M-param
  frozen-latent decoder z → ego-frame BEV occupancy (60×32 m, agents rasterised from the
  `obstacle.offline` join — 26,084 records / 137 clips, occlusion flags) was trained with plain
  BCE and **collapsed to the all-empty prediction**: occupancy rasters are overwhelmingly empty,
  so unweighted BCE *rewards* the collapse, and measuring IoU only at sigmoid 0.5 on a collapsed
  readout produced 0.0003 vs 0.0001 — "a ratio of noise" (gate ratio 0.412 vs ≥ 0.8, recorded as
  FAIL; `p8_gate_attempt1.json`, T0-diagnostic). The attempt-2 fix is in code and pre-registered
  (`stack/scripts/train_p8_occupancy.py`): **auto pos-weight** (neg/pos ratio measured on a
  sample of training rasters, capped at 200), a **soft-Dice term** (an all-empty prediction
  scores ~1.0 loss there regardless of class rarity — exactly the collapse BCE rewarded), and a
  **τ\* operating-point sweep chosen on the encoded arm** — which makes the pred/enc retention
  gate strictly *harder*, never easier. P8 matters beyond itself: its decoded-BEV lead read-off
  is the convergent test for the P1 lead absence (if the occupancy decode shows the lead
  vehicle, the information enters the latent and dies in the pooled readout — localising the
  defect to routing, LF0's RC1). **Attempt 2 has since run and PASSES its gate (retention 0.932
  at k = 10, a 74× lift over attempt 1) — reported in full, with its limitations, in §7.18.**
  The instrument lesson is kept here rather than deleted: the fix was entirely in the *readout's*
  loss, and attempt 1 would have been reportable as "the predicted latent has lost the scene" by
  anyone who did not check whether the instrument could see a scene at all.
- **P4 (occlusion permanence)** — **measured on the attempt-2 run: occluded-agent recall ≥
  visible at every k** (§7.18), reported as *consistent with* permanence pending a diffuseness
  control. **I4a (imagination attribution)** — **measured: the channel is load-bearing, zeroing
  it collapses selected ADE 19× and shuffling it 3.1×** (§7.18). **P2 (nuisance non-retention)**
  — **FAILS at every milestone of the H-COTRAIN curve** (§7.17) and is a standing open defect.
  **P9 (probe-gradient saliency)** and **I4b–c** (the occluded-split stratification and the
  occlusion-stress windows) remain built and pre-registered with bound outcomes (P4's obstacle
  join: 39/40 episodes, 195,805 boxes); no claim is made for them here.

### 7.15 Data and instruments shipped this round (2026-08-10 → 08-11)

- **Alpamayo-2-Super augmentation dataset** — completed: **4,729 clips · 23,644 rows across 5
  tasks** (`trajectory` / `meta_action` / `auto_labeling` / `vqa` 4,729 each, `grounding_via_vqa`
  4,728), published as an HF public gated dataset with a road-class-coverage and provenance card.
  **Counts MEASURED and registry-anchored** — `Project Steering/MODEL_REGISTRY.md` §11.1, derived
  by direct aggregation over `records.parquet` (sha256-matched to the published artifact) and
  independently re-verified 2026-08-16. ⛔ **The earlier "4,800/4,800 clips, 23,999 rows" was
  INHERITED from `MORNING_REPORT_2026-08-11.md`, which repeated the dataset card; both are wrong.**
  The card understates the completeness hole by **356×** — it reports "one" missing row of 24,000,
  where **356 are missing (1.48 %)**: 81 selected clips produced zero rows (−405), 10 delivered
  clips were never in the manifest (+50), and one clip is missing its `grounding_via_vqa` row (−1).
  ⭐ The **stratification survives intact** — urban 1,884 / intersection-rich 1,241 / highway 384 /
  unstructured 83 are delivered 100 % complete, and the entire hole falls in the unstratified
  remainder. Its meta-actions (3-axis, with severity) and reasoning traces are exactly
  tactical-level auxiliary supervision — a second, independent teacher for the hierarchy's stage 0,
  with the contamination caveat carried. 🟥 **Public use of this dataset is an OPEN PI DECISION**
  (two in-repo documents disagree on whether a PhysicalAI-AV derivative may leave the firewall —
  recorded in registry §11.1, deliberately unresolved).
- **PH1-fused hierarchy label layer over the aug120 slice** — **201/201 clips fused**, Alpamayo
  layer on 201/201, ⛔ **but the SAM3 perception leg covers only 86/201 (42.8 %)**: 115 clips carry
  a named `AUG120_SAM3_STAGE_GAP` marker rather than a silent empty layer. The gap was first
  recorded as "8 clips" and is **14.4× larger** than that; root cause is a `--n` flag omitted for
  the SAM3 stage (default 4), so every batch got SAM3 on its first 4 clips **while printing
  `SAM3_RC=0`**. MEASURED, registry §11.2. *(Root-cause class C18: a listing probe sees a MISSING
  file, never a SHORT one.)*
- **The `obstacle.offline` join** — 3D agent tracks (97.44 % of the corpus) joined to the eval
  grid: 26,084 records over 137 clips **with occlusion flags** (`p8_gate_attempt1.json`
  `raster_source`; MEASURED), feeding P4/P8, the lead-gap battery, and the distance-keeping
  family. Labels from it are admissible as frozen probe targets and eval strata only — never
  trunk losses, never inputs (§3.9 standing rule).
- **VLM strategic labeling — design + PH0 pre-registration (future work, 0-GPU so far).** The
  strategic layer's missing goal supervision (g_str) is designed as a two-engine, adversarially
  fused pipeline (`VLM_STRATEGIC_LABELING.md`): Engine A, a deterministic geometric analysis of
  long-horizon integrated ego trajectories (hindsight); Engine B, a VLM fed past+future clips
  that extracts scenario/domain/sign (incl. OCR of navigation signage) and proposed strategic
  goals/actions under a strict schema with a two-pass extract-then-self-verify protocol; a
  deterministic fusion gate emits a strategic action only when the VLM's claim is consistent
  with the hindsight geometry, banking disagreements as `disputed`. The PH0 pilot
  (`PREREG_PH0_VLM.md`) binds three arms (Qwen3.5-9B bf16; Qwen3.5-27B-FP8;
  Gemma-4-31B QAT-w4a16) on the same 50 stratified clips with gates fixed in advance —
  **G1 sign-OCR precision ≥ 0.9** on a human-checked sample (fail ⇒ that arm's signage text and
  every signage-derived `route_to` goal is excluded; all-fail ⇒ the strategic goal degrades
  honestly to hindsight-geometry corridor intent), G2 schema compliance ≥ 0.9, G3 geometric
  consistency reported with CI as a baseline. The admissibility rules travel into the schema:
  **labels may use ego/future** (offline hindsight is the design), **inference stays
  vision-only** (nothing from this pipeline is ever an inference-time input), and **goal/
  situation disjointness** is enforced at label time — every strategic field carries its
  derivation source (`path|signage|vlm-fused`), and any field derivable only from the scenario
  classification is inadmissible as goal supervision. **Sequencing (revised by the PI
  2026-08-11, `HIERARCHY_VOCABULARY.md` preamble): the VLM/algorithmic labeling pipeline moves
  *ahead of* v6 training rather than behind the v5.8f release row** — it is the supervision
  source for v6's strategic goal head and for the hierarchy vocabulary (§10), so it must
  produce labels before the stage that consumes them, and it costs no flagship GPU (a separate
  pod). *(This paragraph previously read "PH0 runs only after the v5.8f release row, T1 rows and
  P8/P9 close"; that ordering is superseded.)*

### 7.16 Selection settled: a selector-free planner over a healthy fan fails, and the mechanism is the winner's curse (2026-08-12)

> **Figure 3 — `Paper/figures/winners_curse.svg`** (also rendered as `.png`; generated by
> `Paper/figures/make_winners_curse.py`, which **reads the gate JSONs at generation time and
> exits if one is missing** — the figure has no hand-typed fallback by design, and a
> cross-artifact tripwire refuses to draw if the fan oracle disagrees between `w4r_gate.json`
> and `w7_selection_rules.json`): the mechanism in three panels — the realised error inside the
> cost's own top-m (flat at the fan's mean while the top-m *ceiling* falls, i.e. the low-cost
> stratum is a random subset), the rank ruler showing the argmin landing at 132.3 of 256 against
> a chance rank of 128, and what each selection rule scores on the same fan.

This is the round's headline negative result, and it is reported as a result rather than as a
setback because its mechanism is general: **a plan chosen as the argmin of a noisily-correlated
surrogate cost over a large candidate set is not a good plan, and gets worse as the candidate set
grows — while the oracle over the same set gets better.** We measured this in a configuration
built specifically to have no alibi left.

**The run, and why it has no confound (`MODEL_REGISTRY §1.14`, artifact `w7_full_gate.json`,
[T0 — WM diagnostic, 881-window / 40-episode held-out grid, MEASURED]).** Every previously
implicated stale component was removed at once: the **repaired stage-A trunk** (§7.13; the
action-response gain in band, `frozen_proof.identical: true`, md5 `06019658…`), the **W4r
unicycle emission head refit on that trunk** (`w4r_gate.json`, gate PASS: fan oracle 0.1273 m
against a 0.2173 m cap, winner accel MAE 0.276 m/s², census violations 0.0), and **no shortlist
at all** — `topk = 256`, so `winner_in_shortlist_frac = 1.0` and the true best candidate is
available in *every* window by construction. Selection is `argmin_i [ 1.0·c_roll(i) +
0.2·c_kin(i) ]` with `c_roll` = ADE between the candidate's own unicycle waypoints and the
world-model's imagination-closed roll of that candidate decoded to waypoints (k = 10, 1.0 s),
and `c_kin` = mean|a| + 0.5·mean|jerk| from the controls. W7 trains nothing: it is a re-rank
instrument over frozen artifacts (`n_trainable: 0`).

| quantity | value | note |
|---|---|---|
| fan **oracle** ADE | **0.1273 m** | healthy fan; the answer is always in the set |
| **W7-FULL selected** ADE | **3.3348 m** | pre-registered gate ≤ **0.4505 m** (= 50 % of the frozen-argmax sel_gap) — **FAIL**, `frac_selgap_closed` −3.71 |
| frozen selector on the *same* fan (in-run) | 4.4159 m | W7's pick beats **or ties** it on **67.5 %** of windows (`w7_pick_le_frozen_pick_frac`) and still fails absolutely |
| sel_gap | 3.2075 m | 26× the oracle |
| within-window rank corr. (cost vs realised error, over the 256 candidates) | ρ_mean **0.4453** / ρ_median **0.4969** | the cost ranks *broadly* correctly |
| across-window calibration (P7 analogue) | ρ **0.3185** [0.2064, 0.4086] | **episode-cluster bootstrap**, n = 881 windows / 40 episodes, B = 2000 |
| **error-rank of the argmin** | **132.3 of 256** | the median rank is 128 — *the winner is a coin flip* |

**The four families, per the §5.5 contract** (same run, same windows; the artifact's own
`mini_eval.families` block):

| family | measured | note |
|---|---|---|
| **LONGITUDINAL** | speed MAE **0.4739 m/s**, accel MAE **0.5105 m/s²** | headway / time-gap / TTC **not computable in this instrument** — it has no lead-agent channel; they remain GT-join instruments on the pod-side harness (stated rather than dropped) |
| **LATERAL** | heading MAE **0.2908 rad**, yaw-rate MAE **0.3677 rad/s**, curvature MAE 6.06 (artifact field `curvature_mae_1pm`) | waypoint-derived adjuncts — finite-difference `atan2`, noisy near standstill; stamped as such in-artifact |
| **TACTICAL** | winner-hit **3.2 %**, selected-rank percentile **34.6 %**, winner-in-shortlist **100 %** | for a re-rank instrument, decision quality *is* the tactical family: the true winner is always available and is found 3 % of the time |
| **STRATEGIC** | **n/a** | no route/goal label exists on PhysicalAI-AV (§5.5; settled at five probes). Stated per family with its reason, never silently dropped |

Note that the LONGITUDINAL and LATERAL numbers look *reasonable* while the arm's selected ADE is
3.33 m — a direct demonstration of why the families are reported per-family and never pooled: the
chosen trajectory is smooth and plausibly-paced, and it is the wrong one.

**The diagnostic that names the mechanism (0-GPU follow-up sweep over the banked scoring
windows, `w7_selection_rules.json`, EXPLORATORY class — a winning rule must be re-measured on a
fresh grid).** Two facts, taken together, are the whole finding:

1. **Inside the cost's own top-m set, the mean realised error is *flat at the fan's own mean*.**
   `top_m_mean_error_ade` = 5.408 / 5.327 / 5.313 / 5.321 / 5.321 / 5.321 for m = 2/3/5/8/16/32,
   against a fan-wide mean of ≈ 5.32 m. Conditioning on "the cost likes this candidate" changes
   the error distribution by nothing measurable. Equivalently: **the top-m set is statistically
   indistinguishable from a random m-subset of the fan.**
2. **The top-m *ceilings* do fall**: the best-in-top-m error is 3.380 / 2.384 / 1.561 / 1.067 /
   0.598 / 0.356 m for the same m — which is exactly what random m-subsets of a fan whose full
   oracle is 0.127 m would give. So *headroom* exists inside a shortlist; it is not the argmin
   that will find it.

And the decomposition of the deployed number: **argmin on the roll-consistency term alone scores
5.2898 m** — the fan's mean, i.e. *nothing* — so the entire difference between 5.29 and the
deployed 3.3348 comes from the **kinematic term at weight 0.2**, not from world-model roll
consistency. A rank-blend of the two terms scores 5.4615 m, no better.

**Why the fan gets *worse* to minimise over as it deepens, and the model this refutes.** §3.12
gives the formalism; the empirical shape is: on the frozen trunk the K-sweep read 0.5772 /
0.5173 / 0.5319 m at K = 8/32/64 (`w7_gate_k{8,32,64}.json`, T0) and W7-FULL reads 3.3348 at
K = 256 on the **repaired** trunk. ⚠️ Those are two different trunks and therefore **not one
curve** — the direction agrees, the series does not, and we do not draw it as one. What the
numbers *do* settle is a model comparison stated in §3.12: under a homoscedastic Gaussian-copula
model of a noisy-but-honest cost, deepening the fan would make the argmin *better*
(E[error | argmin] scales with −ρ·√(2 ln N)); the measured degradation therefore **refutes that
model** and requires the alternative — a cost with a **degenerate minimiser**. The
roll-consistency term is a *self-consistency* functional: it is minimised by a candidate the
world model can trivially reproduce, and a near-stationary candidate is exactly that. As the fan
deepens, the probability that it contains such a candidate goes to one, so the argmin converges
onto the degenerate set while the oracle keeps improving. The artifact names the untried
antidote itself: W7's anti-degeneracy **progress term** (`--w-prog`, minus candidate arc length)
has been at **weight 0.0 in every W7 run to date**.

**The published contrast this exposes, and it is the reusable lesson.** V-JEPA 2-AC plans by
minimising **latent distance to a goal state** (arXiv 2506.09985); DINO-WM likewise uses a latent
*goal* cost (arXiv 2411.04983). A goal-distance cost is not minimised by inaction — standing
still leaves you far from the goal. Our roll cost measured **agreement between a candidate and
the world model's own imagination of it**, which is minimised by inaction. Structurally we
copied the planning *loop* and dropped the term that makes it a proper score. **A selection cost
for MPC over a learned world model must contain a term that inaction cannot minimise.**

**The second load-bearing consequence: repairing a trunk invalidates every frozen consumer.**
The same artifact measures it directly: the frozen selector reads **0.7933 m** on the frozen
trunk's W4 fan and **4.4159 m** on the repaired trunk's W4r fan — a 5.6× degradation from
changing *nothing but the features it scores*, with `winner_in_shortlist` for the earlier
shortlisted variant at 19.6 % (`w7_gate_repaired_k32.json`). The head refit (W4r) recovered the
fan; the selector was never refit, and it cannot be, because it was trained against a feature
distribution that no longer exists. **You cannot repair a trunk and keep its planner.** Two
things follow, and both are decisions rather than observations:

- **v5.8f ships the FROZEN-trunk assembly** — `rescorer-top8-kincost`, selected ADE **0.4815 m
  [0.3928, 0.5771]** (episode-cluster bootstrap, `v58f_rescore_ci.json`, T0), selected accel MAE
  0.515 m/s², oracle 0.1077 m (`MODEL_REGISTRY §1.14`). The stage-A trunk has demonstrably
  better physics (§7.13) and the assembly built on it is worse, because its consumers are stale.
  This is instrument composition deciding a release, and we state it that way.
- **That sentence *is* the staged-training argument for v6** (§10), and it is stronger than the
  argument from the literature because it is ours and it is measured: consumers must be trained
  **on the trunk they consume**, which is what a staged ladder (S-W → S-T → S-S) enforces and
  what co-training cannot guarantee across a repair.

**Honest scope.** (i) T0 throughout: none of these numbers is driving performance, and the T1
confirmation of the shipped assembly is still open (§10 item 7). (ii) The gate threshold
0.4505 m is a *pre-registered function of the frozen-argmax reference*, not the deployed arm's
score; the deployed v5.8f interim arm is 0.4815 m and the deployed flagship v1 is 0.452 m
(`MODEL_REGISTRY §1.2`) on a different, non-comparable geometry — cross-frame, never compared.
(iii) The point estimates are corpus-grid means over the fixed 881-window grid; only the two
correlations carry the episode-cluster bootstrap, as their artifact states. (iv) The selection-
rule sweep re-uses the W7 scoring windows and is stamped EXPLORATORY by its own artifact: any
rule chosen from it must be re-measured on a fresh grid before it is quoted.

### 7.17 Does the planner gradient erode the world model? H-COTRAIN measured, the hypothesis rejected within range, and SIGReg validated (2026-08-11)

**A standing hypothesis of this programme — carried in §3.9 of the previous version of this
paper — is now measured false in the range we can measure, and we retract it in our own voice.**

**The hypothesis (PI, pre-registered before any number was seen, `PREREG_H_COTRAIN.md`).** Joint
world-model + planner training forces the trunk to become an ego-action/trajectory feature
extractor rather than a model of the physical world. It was cheap to test because v5f's coupling
followed a *measured curriculum* — λ_plan = 0 until step 2,000, linear ramp to 8,000, then 1.0 to
29,999 — and restored milestone checkpoints (5 k/10 k/15 k/20 k, plus the banked 30 k row) sample
the trunk at increasing planner-gradient exposure. The same frozen probe battery runs at every
milestone: a **within-run dose–response curve** with the instrument held fixed.

**Bound outcomes, fixed in advance.** CONFIRM required either (a) a scene variable reading
R²(enc) > 0.15 at 5 k that **declines by ≥ 0.15 absolute** by 30 k while ego-dynamics probes
hold, or (b) the latent spectrum's effective rank shrinking **> 20 %** across the ramp while ego
probes hold. REJECT-for-the-lead-variable required lead-gap to be unreadable at *every*
milestone including 5 k. A separate, independent SIGReg verdict rode the same runs: participation
ratio retention ≥ 0.8× validates the anti-collapse mechanism under a full planner gradient.

**Result: neither CONFIRM condition fired, in the opposite direction
(`h_cotrain_curve.json`, MEASURED, [T0-diagnostic — representation interrogation], pooled
out-of-fold R² per milestone, episode-disjoint folds).** Every probed physical variable became
**more** decodable as planner exposure increased:

| probe (R² at k = 10) | 5 k | 10 k | 15 k | 20 k | 30 k |
|---|---|---|---|---|---|
| speed — **encoded** latent | 0.649 | 0.654 | 0.688 | 0.717 | **0.744** |
| yaw-rate — encoded | 0.583 | 0.670 | 0.734 | **0.869** | — |
| curvature — encoded | 0.213 | 0.173 | 0.371 | 0.513 | **0.551** |
| speed — **predicted** latent | 0.993 | 0.994 | 0.995 | 0.995 | — |
| yaw-rate — predicted | 0.370 | **0.177** | 0.572 | 0.758 | — |
| curvature — predicted | 0.225 | **0.050** | 0.485 | 0.704 | — |
| P1 battery gate | FAIL | FAIL | FAIL | **PASS** | — |

and the latent spectrum **expanded**: participation ratio of z_enc 4.53 → 4.96 → 5.96 → **6.94**
(of 2048 dimensions), a **+53 %** increase, with the top-8 eigenvalue share *falling* 0.9903 →
0.9232 — energy spreading out of the leading directions, which is the opposite of collapse.

**Three things must be said plainly about scope, because they bound the claim.**

1. **The lowest available reference is λ = 0.5, not 0.** All four milestones sit **at or past**
   the ramp (λ ≈ 0.5 already at 5 k, 1.0 from 8 k). The measurement therefore rejects **erosion
   *under* joint training within 0.5 ≤ λ ≤ 1.0**; it does *not* establish that joint training is
   optimal, and it cannot, because the clean λ = 0 control was never run. That control is not
   hypothetical — it is **v6's S-W stage** (§10), and it is the confirmatory arm the
   pre-registration already named (mode-0 vs scheduled at matched steps, both probed with the
   same frozen battery).
2. **The SIGReg retention gate is evaluated at 20 k, not 30 k.** The pre-registration wrote the
   gate as "participation ratio at 30 k ≥ 0.8× its 5 k value"; the spectrum series in the
   artifact stops at 20 k, so the **measured retention 1.532** is 6.94/4.53 = 20 k over 5 k.
   The verdict — **SIGReg VALIDATED, no progressive collapse under a full planner gradient,
   effective rank expands 53 %** — is stated on that window and on no other. Closing the 30 k
   spectrum row is a ~0-GPU item.
3. **There is a real transient, and it is not noise.** Predicted-latent yaw-rate and curvature
   **dip sharply at 10 k** (0.370 → 0.177 and 0.225 → 0.050) — immediately after λ reaches 1.0 at
   step 8,000 — then recover and *overshoot* by 20 k. The planner gradient visibly **disturbs**
   the predictive readout when it arrives; the system re-organises rather than degrading. A run
   stopped at 10 k would have produced the CONFIRM verdict. That is a methodological point about
   dose–response curves in general: **a single post-intervention checkpoint cannot distinguish
   damage from transient re-organisation.**

**What we retract, and what replaces it.** The previous version of this paper carried the
erosion hypothesis as a live explanation for the muffled action interface and the missing lead
variable, and §10 listed gradient isolation as a lever motivated by it. **The erosion premise is
withdrawn within the measured range** (root-cause class: *a mechanism inferred from a coincidence
of symptoms, never dose-tested*). The staged v6 recipe **does not lose its justification** — it
never rested on erosion. It rests on (i) the field's convergent recipe (V-JEPA 2 → 2-AC,
DINO-WM, Drive-JEPA all post-train consumers on a representation trained without them; §3.9),
and (ii) the **consumer-invalidation** result of §7.16, which is ours and is measured: a repaired
trunk breaks every consumer trained on the old features. Gradient isolation survives as a
**preventive design property** with an autograd-level check (§3.11), not as a treatment for a
disease we have now failed to find. One defect *does* survive the milestone curve untouched:
**P2 (nuisance non-retention) fails at every milestone** and remains a standing open item.

### 7.18 The environment survives prediction: occupancy retention, occlusion permanence, and a load-bearing imagination channel (2026-08-11 → 08-12)

Three probes of the physics battery (§7.14) closed this round, and together they answer the half
of the PI's question that the ego-dynamics probes cannot reach: **does the predicted latent still
contain the *world*, or only the ego?**

**P8 — BEV occupancy retention: GATE PASS (`p8_gate_attempt2.json`, MEASURED,
[T0-diagnostic — a representation probe, never a driving number], 881-window grid).** A
984,817-parameter readout `z → ego-frame BEV occupancy` (120×64 grid, agents rasterised from the
`obstacle.offline` join: 26,084 records / 137 clips with occlusion flags) is trained **on encoded
latents only**, with the trunk md5-frozen before and after; the predicted-latent path is
**eval-only by design**, so the readout never sees the arm it is used to grade. Attempt 1's
all-empty collapse was diagnosed as an instrument failure — unweighted BCE on rasters that are
98.76 % empty *rewards* the collapse (§7.14). Attempt 2 fixes the instrument and nothing else:
measured `pos_weight` **79.7** (from **1.239 %** positive cells), a soft-Dice term (an all-empty
prediction scores ≈ 1.0 loss there regardless of class rarity), and a **9-point threshold sweep
whose τ\* is chosen on the ENCODED arm** — the conservative side, since the gate is a
predicted-over-encoded ratio and optimising the denominator can only make it harder.

| k = 10, τ\* = 0.7 | value |
|---|---|
| IoU, decode(**encoded** z_{t+10}) | 0.02005 (n = 797) |
| IoU, decode(**predicted** ẑ_{t+10}) | 0.01869 (n = 823) |
| **retention ratio** | **0.932** — gate ≥ 0.80 **PASS** |
| lift vs attempt 1 | **74×** (attempt 1: IoU 2.7 × 10⁻⁴) |
| τ\* sweep (encoded, pooled over k) | 0.0149 → 0.0191 → 0.0180 across τ = 0.05 → 0.7 → 0.8 — an interior maximum, not an edge artifact |

⇒ **rolling the predictor a full second forward loses only ≈ 7 % of the scene content the encoder
itself exposes.** That is the property P8 exists to test, and it is the first direct evidence in
this programme that the predicted latent is a model of the *world* and not merely an ego
integrator — which matters precisely because the ego-state probes (P1: speed R² 0.993 predicted
vs 0.744 encoded) are the ones that could be read as an ego-only story.

⚠️ **The limitation travels with the number, in the same breath.** The **absolute** IoU is low
(≈ 0.02) — a 1 M-parameter readout on frozen latents against sparse rasters — so the admissible
claim is the **retention ratio** (one instrument, two inputs, everything else held fixed), never
the absolute occupancy quality. Anyone quoting 0.02 as "the model's occupancy accuracy" would be
quoting the readout's capacity, not the latent's content.

**P4 — object permanence, now visualised on the same run.** Occluded-agent cell recall is **not
worse** than visible-agent recall at k = 10: **encoded 0.2178 occluded vs 0.1881 visible**
(n = 194 / 548), **predicted 0.1743 vs 0.1717**. For the **encoded** arm the ordering holds at
k = 5/15/20 as well; ⚠️ **for the predicted arm it does NOT** — corrected 2026-08-16 by re-reading
the banked artifact, the predicted gap reverses at k = 5 / 15 / 20 (−0.0035 / −0.0034 / −0.0057)
and is positive at k = 10 alone. k = 10 is the pre-registered gate k, so the quotation is
principled rather than selected after the fact, but the predicted half of this result is
k-specific and must not be stated as holding across k. The latent carries agents the camera
cannot see, which is the sharpest available test that the model predicts the world rather than
the image. ⚠️ **This needs a diffuseness control before it stands alone**: a decoder that paints
broad blobs would score occluded and visible cells alike, so occluded ≥ visible is consistent
with permanence *and* with an uninformative readout. That control is the next cheap probe and is
named as such; until it runs, P4 is reported as *consistent with* permanence, not as proof of it.

⛔⭐ **And the reason that control is not optional — the P4 label and the camera-field mask are
the SAME PREDICATE (MEASURED 2026-08-16).** The join's `occ` flag and `bev_raster.fov_mask`
select identical cell sets (0 / 7 680 disagreeing at every half-angle tested; bit-identical
defaults), differing only in granularity. So (a) the diffuseness control has a *concrete* form:
the occluded arm is scored entirely inside a 590-cell near wedge (7.68 % of the grid, all at
x < 9.24 m) while the visible arm ranges over the other 92 %, so the two arms differ in
**position** as well as in visibility, and the discriminating comparison is against
**same-region, same-visibility** agents (`--p4-region-control`: `occluded_over_visible_near`
≈ 1.0 ⇒ regional, > 1.0 ⇒ survives). And (b) the "obvious" correctness fix applied to every
other consumer of this raster — masking to the camera field — **would delete this result rather
than harden it**, because the occluded population *is* the masked-out set; measured, a masked
occluded arm keeps 0–27 % of its cells with the survival fraction rising with vehicle length.
That asymmetry is itself a methodological point: a visibility mask is a correction for a metric
scored *over the field* and a deletion for one scored *about its complement*.

**I4a — the imagination channel is load-bearing, not decorative (`MODEL_REGISTRY §1.14`;
artifacts `i4a/flagship-v5f-w120-30k-i4a-{none,zero,shuffle}.json`, MEASURED, T0, same 881
grid, same checkpoint, only the imagination input changed).**

| arm | selected ADE | fan oracle | miss@2 m |
|---|---|---|---|
| intact | **0.4011 m** (byte-matches the banked v5f baseline — an in-run instrument-parity proof) | 0.1975 | 0.149 |
| **shuffled** across windows | **1.2492 m** (3.1×) | 0.426 | — |
| **zeroed** | **7.6493 m** (19×) | 1.457 | 0.805 |

*Four-family status for this triplet, per §5.5 rule 5:* the banked I4a artifacts and the
registry row carry selected/oracle ADE and miss@2 m; the **per-family stratification of the
ablation is not yet computed** — it is a rescore off the same banked runs (~0 GPU) and is a work
item, not an omission. It is named here rather than left implicit.

The **ordering** is the result, not the magnitudes: shuffling preserves the imagination input's
marginal statistics and destroys only the window↔consequence correspondence, so
`zero ≫ shuffle ≫ intact` shows the planner reads imagination as **content**, not as a bias term
or a magnitude cue. (A channel used as a bias would be indifferent to shuffling; a dead channel
would be indifferent to both.) ⚠️ Caveat stamped in the artifact and repeated here: the head was
**trained with imagination present**, so this measures the dependence of *this* architecture, not
the value of retraining without it. I4b — the same triplet stratified on the P4/P8 occluded
split, where imagination should matter most — is the next refinement and costs no extra GPU.

**Taken together with §7.16**, these three results locate the generation's failure precisely: the
world model carries the scene (P8 0.932 retention), carries hidden agents (P4), responds to
actions correctly after repair (§7.13), and supplies a fan containing 0.127 m answers — and the
programme still cannot *choose*. Every remaining metre is in selection.

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

**Where the two-stage question stands after v1.0's measurements, and what actually decided it.**
Three results this round move the argument, and the honest accounting is that **the one we
expected to decide it did not**. (a) **The erosion account is dead in range.** We had a standing
hypothesis that the planner gradient was overwriting physical state — the natural mechanistic
story for why a jointly-trained trunk has a muffled action interface and no lead variable. The
dose–response curve says the opposite: physical decodability *rises* monotonically with planner
exposure and the latent's effective rank *expands* (§7.17). Anyone building a staged pipeline
because "co-training erodes the representation" is, on this evidence, building it for a reason
that is not there — within 0.5 ≤ λ_plan ≤ 1.0, which is the range we can speak about. (b) **What
does decide it is composition, not erosion.** Repairing the trunk improved its physics and
simultaneously destroyed every consumer trained on the old features (frozen selector
0.7933 → 4.4159 m, §7.16). That is a *compositional* property of any pipeline whose stages are
trained against each other's outputs, and it is precisely what a staged ladder makes explicit and
gateable: each consumer is trained on the trunk it consumes, and the frozen battery says whether
the trunk moved. It also reframes LP-FT: the literature's "fine-tuning distorts features" is a
statement about the *representation*; ours is a statement about the *consumers*, and the second
bites even when the first does not. (c) **A transient is not damage.** The predicted-latent
readouts dip hard at 10 k and recover by 20 k (§7.17). A programme that had probed once, shortly
after coupling, would have concluded erosion and restarted. That is a general hazard for anyone
evaluating an intervention at a single post-intervention checkpoint, and it is an argument for
dose–response curves as a standard instrument rather than a luxury.

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
pathologies). G3 |v̂−VTARGET| tracking. G4 closed-loop drift below the **1.7318 m** head baseline. G5
open-loop non-regression vs **0.4271 m**. The staging de-risks the thesis cheaply: the planner (cost+CEM)
runs over the **already-trained v1 world model** with offline-minted VTARGET labels — if planning over
frozen v1 already beats the **3.3839 m** tactical head and the **1.7318 m** closed-loop drift, the
architecture is validated before any v3 training.

> ⛔ **GATE BARS CORRECTED 2026-08-17 — these were pre-registered against BANNED-ESTIMATOR numbers.**
> G4's bar read **1.69 m** and G5's **0.452 m**; both were legacy `heldout` split-means
> (`overlapping_holdout_se`), and the 3.38 m head figure likewise. Decision-grade `full_set` values with
> episode-cluster bootstrap intervals: head baseline **1.7318** [1.5707, 1.9070], open-loop
> **0.4271** [0.3675, 0.4871], tactical head **3.3839** [2.8336, 3.9722]. ⚠️ **This matters more for a
> gate than for a result: the old G4 bar was 2.69 % LOWER than the honest one, i.e. it held every future
> arm to a bar that is HARDER than the measurement justifies** — a pre-registered gate inheriting a
> defective statistic. *(Superseded, kept visible: 1.69 · 0.452 · 3.38.)* MEASURED —
> `…/incoming/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json` and
> `…/incoming/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json`; see `MODEL_REGISTRY.md` §5.

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

**The v0.9 round re-orders again; the near-term ladder is now (in priority order):**

7. **Close the v5.8f repair arc and mint its first quotable rows.** ~~W7-FULL~~ **done — it
   failed, with a mechanism (§7.16), and the release arm is consequently the FROZEN-trunk
   assembly at 0.4815 m [0.3928, 0.5771].** What remains for a release row: the **four families
   + episode-cluster CIs on the banked windows and the T1 (action-closed-loop) rows** — the row
   exists only when both do, and no capability claim is made from T0 (§5.4). Add to it the two
   selection rules the winner's-curse formalism prescribes (§3.12), **pre-registered before use
   and measured on a fresh grid**: enabling W7's dormant anti-degeneracy progress term
   (`--w-prog`, at weight 0.0 in every run to date), and replacing argmin with a **top-m
   aggregate** or a **monotone cost recalibration** whose success criterion is a rise in
   lower-tail dependence λ_L (or equivalently a fall in the argmin's error-rank from 132/256),
   not a fall in ADE alone.
8. **The v6 staged-training direction, decided by measurement, not fashion.** The staged ladder
   is now the programme's main line, and its justification is explicitly *not* the erosion
   hypothesis (rejected in range, §7.17) but (a) the field's convergent recipe and (b) the
   measured **consumer-invalidation** result: a repaired trunk breaks every consumer trained on
   its old features (§7.16), so consumers must be trained **on the trunk they consume**.

   | stage | what trains | what is frozen | gate before the next stage begins |
   |---|---|---|---|
   | **S-W** (world) | encoder + operative predictor, WM-only, λ_plan ≡ 0, planner absent | — | **P1** retention ≥ 0.85× R²(z) at k = 10 per target · **P3** sign ≥ 0.95 both channels **and** gain median ∈ [0.5, 2.0] **without post-training** · **P6** action subspace ≤ 32 dims (reported: P2, P5, P8, O6 spectrum) |
   | **S-T** (tactical) | tactical layer + goal head + selector, on the frozen S-W trunk | trunk | TACTICAL family · `sel_gap` ≤ 0.5× the fan oracle **at T1** · confusion improves on the E4.1 strata (reported: P7 per stratum, LATERAL, X2 seam) |
   | **S-S** (strategic) | strategic layer + g_str goal head | trunk + tactical | **STRATEGIC family computable at all** (measured vs today's `n/a`) · ADE(8–30 s) beats CV/corridor at T1 |
   | **S-J** (optional) | brief joint polish, **isolation still on** | nothing | zero live forbidden edges (X3 autograd probe) · the frozen battery **FLAT** across the joint phase |

   The stage-0 measures are the already-gated parts promoted from retrofit to design: **O1** =
   stage-A's response-form `L_ctrl` **from step 0** rather than as a post-hoc repair (§3.9);
   **O2** a near-field distance-weighted latent loss, *time*-scaled not metre-scaled (a fixed
   40 m band cannot cover 6 s — 180 m at 30 m/s); **O3** contiguous masked spatial-latent
   prediction (scattered per-cell dropout is trivially inpainted and teaches nothing about
   permanence); **O4** interaction-weighted sampling from **actions only** (|jerk|, |decel|,
   steering reversals — label-free, and it *reweights the draw, never removes a window*, so
   parity holds and α = 0 reproduces uniform exactly); **O5** rollout consistency at **every**
   step, not endpoint-only; **O6** per-layer SIGReg with the spectrum monitor of §3.13 as a
   standing training-time gate. The emission head is W4's unicycle parameterisation, feasible by
   construction, scaled k = 20 → **60** for the 6 s horizon.
9. **The vocabulary and labeling pipeline, which is now a PRECURSOR to v6 rather than a
   successor to the release row.** v6's strategic stage cannot be gated at all until the
   STRATEGIC family is computable (§5.5), and that requires the label stream to exist *before*
   S-S trains: **PH0** (three VLM arms — Qwen3.5-9B / 27B-FP8 / Gemma-4-31B-QAT — on 50
   stratified clips, gates G1 sign-OCR ≥ 0.9 and G2 schema ≥ 0.9 bound in advance, plus an
   optional Engine-C/SAM column decided on measured mask-vs-join agreement), then **PH1** (the
   full **4,729**-clip run — the delivered A2 population, MEASURED, `MODEL_REGISTRY.md` §11.1, not
   the card's 4,800 — spend decided from PH0's measured s/clip), then **PH2** (the g_str
   supervision stream, the nuisance/domain strata for P2, and domain-stratified four-family
   evals). Every semantic claim any engine emits must carry its **image-space grounding**
   (bounding box, and a mask/contour reference where available, with a frame index); ungrounded
   claims are `disputed` by default. The Alpamayo-2-Super meta-action distribution (**4,729
   clips** — MEASURED, `MODEL_REGISTRY.md` §11.1, **not** the card's 4,800) supplies the empirical
   phrase distribution the token set must cover, and **vocabulary coverage is measured**, with
   unmappable phrases logged rather than silently dropped.
10. **The label-free lever program LF0–LF4 for the missing lead variable** (§7.14): LF0 first —
   probe pre-pool spatial tokens and the P8 attempt-2 decoded-BEV read-off (~0.5 GPU-h, decides
   whether the defect is routing or learning) — then LF1–LF4 in cost order, each gated on the
   same frozen P1 lead battery. Labels stay probes and strata, never trunk losses.
11. **The scaling ladder S1–S4** (`PREREG_SCALING_LADDER.md`): S1 data volume ({1×, 3×, 10×} the
   13 h canonical corpus — the programme currently consumes < 1 % of PhysicalAI-AV's ~1,701 h;
   the "build the 30× corpus" decision binds to the measured 1×→10× T1 slope, with the
   learning-curve window/R² rules applying to data curves too, and additive supersets only so
   parity is never re-selected); S2 distribution at matched hours (tail metrics per family, not
   means); S3 resolution/context; S4 encoder capacity within the binding **sub-300 M envelope**
   — "dominance of 4-brain TanitAD" is defined as beating bigger published baselines at ≤ 300 M
   total, and needs curves, not points. S1 runs only after the v5.8f release row exists as the
   fixed yardstick.
12. **Complete the physics battery.** P8 attempt-2, P4 permanence and I4a **have landed**
   (§7.18); what remains is the **P4 diffuseness control** (without which occluded ≥ visible is
   consistent with permanence but does not prove it), **P2 nuisance non-retention** — which
   *fails at every milestone* of the H-COTRAIN curve (§7.17) and is the battery's one
   unaddressed standing defect — **P9** probe-gradient saliency, **I4b/I4c** (the occluded-split
   stratification and occlusion-stress windows, both ~0 extra GPU off the banked runs), and the
   **30 k spectrum row** that would let the SIGReg retention gate be quoted at its
   pre-registered endpoint rather than at 20 k (§3.13).

### 10.1 Open decisions this paper cannot close (they are the PI's, and they are costed)

These are stated with their measured or estimated inputs so the decision is made against
numbers, not against a narrative. None of them is an experiment we can run around.

1. **S-W's cost is the ladder's decision.** **ESTIMATED 175–290 A40-hours** for 30 k steps
   (7–12 A40-days), against a full ladder of **≈ 220–370 A40-hours** (S-T 17–28, S-S 11–20,
   S-J 18–29). The estimate is anchored on two MEASURED baselines — flagship v1
   `flagship4b-speedjerk-30k` at **6.374 s/step** (`wallclock_s` 191,206.2 for 30 k,
   `MODEL_REGISTRY §1.2`) and **`flagship-v4-fromscratch`** at **7.085 s/step** (`wallclock_s`
   212,544.6 = 59.04 h, `MODEL_REGISTRY §1.5.5`) — inflated because S-W encodes 26 frames per
   sample (window 6 + 20 futures, in one batched pass) and rolls the 60.3 M predictor ~80 times
   per sample. ⚠️ *Attribution corrected at v1.0: this 59.04 h basis was carried in the design
   note as "v4.2". The registry says it belongs to `flagship-v4-fromscratch` (§1.5.5);
   `flagship-v4.2-30k` (§1.5.3) is a different arm, killed at ~step 5 k, and has no 30 k cost.
   The registry wins; the estimate itself is unaffected because the s/step basis is the same two
   measured runs.* It is an **ESTIMATE and must not be spent as if it were measured** — and this
   programme has the receipt: the same §1.5.5 row's superseded estimate of "~53 h" was **~11 %
   low**, understating the real spend by **≈ 2.5 GPU-days**. The honest move is to launch, read
   the trainer's own already-divided `step_s` at step 500 (the divisor is named in
   `step_s_note` precisely so nobody re-derives the false "430 s/step" alarm from an accumulated
   counter), and re-cost before committing the ladder.
2. **The 6 s horizon costs ≈ 43 % of the training windows.** MEASURED: the horizon plan every
   existing flagship trainer inherits returns `max_horizon = 20` (2 s of future per window), so
   a v6 trainer that inherited it would make the 6 s spec **structurally untrainable** while
   failing in a way that looks like a corpus limitation. `max_horizon` is a *windowing*
   parameter, not a property of the cache, so v6 derives its own — but a 120-frame episode then
   yields `120 − 6 − 60 = 54` windows instead of `120 − 6 − 20 = 94`. **Parity is untouched**
   (parity is *episode* selection: `physicalai-train-e438721ae894`, 2,376 episodes, skip-hash
   `f09e44db`), but the window distribution genuinely changes versus v5f and that belongs in the
   run row. The decision: accept the change, or rebuild the cache with longer episodes.
3. **E-ENC — one encoder or one per layer — decided at MATCHED TOTAL PARAMS.** MEASURED at
   instantiation: shared encoder + per-layer adapters **87.89 M**; per-layer encoders
   **120.74 M**; the matched-parameter counterpart of the latter is the shared arm at
   `--pred-dim 960` = **118.11 M**, a **2.2 % residual gap** that is quoted rather than rounded
   away. Both sit inside the sub-300 M invariant, which the builder refuses to violate before
   any GPU time is spent. Deciding at matched *per-layer widths* instead would let an arm win on
   capacity and be read as winning on architecture — the same confound class as comparing a
   decoder on its marginal. Prior from the field: every frontier system (V-JEPA 2, DINO-WM,
   Drive-JEPA) uses **one** encoder with downstream consumers, so separate encoders must earn
   their parameters and a tie goes to the shared arm.
4. **W5/E-H1 is a REQUIRED precursor, not a queued nicety**: v5.8f must be baselined at 6 s
   (gate ADE(6 s) ≤ 3× ADE(2 s)) *before* v6 trains against it, or the thing v6 changes has no
   yardstick.
5. **S2 (`g_str` supervision) is not wired and must not be improvised.** It comes from
   PH0 → PH1 → PH2 (item 9). Until it exists, the STRATEGIC family stays `n/a` **with its reason
   and its n** (§5.5) — an honest absence, never a proxy.

## References

(Formal bibliography at LaTeX export; the working citations live in
`TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` and the dated research notes: LeJEPA
arXiv:2511.08544; LeJEPA-identifiability (Klindt et al.) arXiv:2605.26379; JEPA generalization theory
arXiv:2606.27014; V-JEPA-2 / V-JEPA-2-AC arXiv:2506.09985; DINO-WM arXiv:2411.04983; fine-tuning
distorts features / LP-FT (Kumar et al., ICLR 2022) arXiv:2202.10054; DiffusionDrive
arXiv:2411.15139; LAW arXiv:2406.08481; World4Drive arXiv:2507.00603; HiT-JEPA arXiv:2507.00028;
GAIA-2 arXiv:2503.20523; Drive-JEPA arXiv:2601.22032; ego-status open-loop shortcut (AD-MLP /
BEV-Planner) arXiv:2312.03031; open-loop⊥closed-loop arXiv:2605.00066; ALPS-4B transfer study
`Ressources/AD_TRANSFER_RESEARCH.md` v1.1. Added v0.9 (all PUBLISHED, §3.9/§7.14): AD-L-JEPA
arXiv:2501.04969; V-JEPA 2.1 dense features arXiv:2603.14482; JEPA LiDAR occupancy WM
arXiv:2602.12540; geographic diversity vs data volume for zero-label JEPA driving WMs
arXiv:2607.04500.)

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
  tactical κ 0.253 → 0.0072 @ `DIR_YAW_RAD` 0.15 (**decorative** — the word and the collapse both
  survive a 0.10 re-read; v1's own word does not, and is not quoted) and the
  `route_acc_nav`-is-copying correction (vision
  route = exactly the majority-straight rate; 0/3 hierarchy seams on both arms); **H-COMPOUND**
  (CR 3.50 → 80.77, teacher-forced flat; C61 resolved) and E-DPSI null < 12°; rollout recovery
  RR-20 vs RR-CTL (ADE 0.424 → 0.348, ΔCI [0.0613, 0.0906]; speed bias erased +0.9397 → −0.0092;
  curvature ×2.2, miss@2m up — a trade, not a win); the situation-classifier **label defect**
  (ego-derived labels; camera penalised for being right); and the v5 → v5f fix (`cond_imagination`
  hard-wired off → flag + feed + dead-conditioning test; 5k guard ROUTE_LIVE Δ0.067 [0.032, 0.103],
  VT_LIVE +0.895 m/s, κ 0.519; imagination cost measured ≈1.01×). Retractions C61–C67 by
  root-cause class. All §7.11 numbers MEASURED with named estimators.

- v0.8 (2026-08-03): added **§7.12 — closed-loop on a neural reconstruction, on the edge device**.
  REF-C beats flagship v1 closed-loop. ⛔ **The "whole separation is LATERAL / ADE is not separated"
  reading is RETRACTED the same day (R-2026-08-03-C)**: re-measured on the improved render the
  shipped videos use, **ADE separates at +7.164 [+5.265, +8.966]** along with both longitudinal
  metrics and strategic corridor departure, while all four lateral separations survive and widen.
  The mechanism is a **21× render-sensitivity ratio** between the arms, isolated by a 2×2 to the
  scale cull and licensed by an exactly-zero determinism control.
  Reproduces the 07-23 native-1080 suite on different hardware, renderer and scene. Three defects only the families expose: the arms fail
  longitudinally in OPPOSITE directions (-2.04 vs +1.33 m/s, a pooled score cancels them); the
  flagship executes what it selects only 0.4481 of the time; and **REF-C's 5-way head never emits a
  longitudinal class** (accelerate 0, brake_stop 0) while 42 % of windows are brake_stop — the
  lat+lon softmax defect seen in closed loop for the first time. Objects-vs-empty-road is **NULL**
  for both arms (19/20 CIs contain zero), bounded to *distant* traffic (0.02-0.4 % of frame at
  40-45 m). Instrument property that constrains all future sim numbers: the renderer is a **step
  function of pose** (discrete blend-order ties among 3.1 M gaussians) — a 0.1 px camera rotation
  moves the 2 s waypoint 6.65 m and the gRPC float32 round-trip alone costs 4.59 m, so every
  production number must come from ONE numerical path. Capability: `volume.nurec` is gzip+MessagePack
  and **gsplat renders it natively on aarch64 incl. f-theta at 16-28 ms/frame** (closed loop
  5-11 Hz) — the amd64-only NRE renderer is not required. Validation by **gradient-NCC** against the
  scene's shipped reference; ⛔ PSNR and NCC are inadmissible on this clip (a wrong frame outranks
  the correct one under both). Scope stated: renderer **wire contract**, not `alpasim_runtime`, so no
  AlpaSim collision/offroad score; strategic family degenerate on a junction-free 20 s clip; rates
  are within-sim relative (3.21x OOD).

- v0.9 (2026-08-11): the eval-tier and repair round. *(Version note: the status line had again
  drifted — it read v0.6 while the changelog carried v0.7 and v0.8; this round was requested as
  "v0.7" but that number is taken, so it is v0.9 — numbering not silently reused, twice.)*
  **§5 extended** with (i) the episode-cluster bootstrap stated mathematically (cluster resample
  over the 40 val episodes, statistic recomputed per draw, percentile interval, paired form for
  two arms; full-set point estimate, never a mean of split-means) and (ii) the **binding
  T0/T1/T2 eval-tier doctrine** with its measured basis — the §1.12 action echo: S-curve
  reproduction 0.9785 open-loop vs 0.0538/0.0430 closed-loop and **0.0 % hold-action**; T0 is a
  WM diagnostic, never driving performance; T1 (action-closed loop) is the primary capability
  tier; cross-tier comparison invalid. **New §3.9** (theory): the JEPA objective in
  predictor/target-encoder form with our no-EMA/no-stop-gradient SIGReg-only instantiation and
  the deployed `full_relaxed` variant; the hierarchical 4B formulation as per-layer predictors at
  separated clocks with goals as upper-level actions conditioning downward; the stage-A
  control-response loss written out (counterfactual δa, response gain g = ‖Δ_probe‖/‖Δ_analytic‖
  gated into [0.5, 2], L_factual anchor, L_scene complement-subspace stability); H-COTRAIN
  pre-registered on the λ_plan curriculum milestones with both outcomes bound; the
  staged-training evidence (Drive-JEPA 2601.22032, V-JEPA 2 2506.09985, DINO-WM 2411.04983,
  AD-L-JEPA 2501.04969, V-JEPA 2.1 2603.14482 — all PUBLISHED). **New §7.12b** — the
  v1arch/v1.6/v1.7 line: v1arch as the attributable data-axis arm with the programme's first
  complete four-family block on the leak-free official OOD-val split (over-speed is a prior:
  71.95 % ahead; seams 0/3 falsified; route head a constant predictor confirmed off-leak); the
  v1.6 latents-only unicycle readout (jerk 36.17 → 1.13, distance-keeping indistinguishable from
  GT, reliance-gated); v1.7's pre-registered refutation (outcome B) despite a −16 % ADE win; and
  the §1.12 T1 action-echo measurement restated as the experiment that produced the tier
  doctrine. **New §7.13** — v5f completed
  30 k (T0 selected 0.4011 / fan oracle 0.1975 / accel MAE 8.11, four families with
  TAC/STR honestly UNAVAILABLE); the wedge ladder (W1 refuted −16.7 %; W2 97.6 % of fan steps
  infeasible; W2b smoother ⇒ jitter is denoise residue); **W4 unicycle emission head**
  (a = a_max·tanh, κ = κ_max·tanh; accel MAE → 0.774, oracle 0.1991 → 0.1077, violations 0.0);
  the three-surface fast-selector elimination (W4b feat 0.5600 / kin 0.5637 / W4c 0.6609, all
  memorisation — retired by pre-registration) reproduced at the tactical level (E4.4: goal fan
  beats CV at 4/6 s, selector throws it away); W7 K-sweep (roll-cost the first calibrated
  selector, ρ 0.399, collapsing at dense K); **W3's unifying defect** (lateral sign 99.5 % at
  gain 0.27, 3-dim action subspace) and the **stage-A predictor-only repair — all gates PASS**
  (gain → 0.971/0.966, lon sign → 1.0, subspace preserved, factual roll improves); W7-on-repaired
  = instrument-composition failure with ρ 0.716 [0.585, 0.770] calibration; W4r PASS (oracle
  0.1273, winner accel 0.276) isolating the frozen selector as the last stale part; v5.8f
  assembled T0 rows (0.4815 [0.393, 0.577] vs 0.7933 [0.641, 0.976]; +0.08 m ADE traded for 16×
  kinematics; NOT a release row — families + T1 pending). **New §7.14** — the WM-physics proof
  battery P1–P9 + I4 as a methods contribution: P1 decodability PASS (pred > enc), the P1
  lead-gap resolution (class-filter fix did not dissolve it; transforms + MLP ceiling −0.334 ⇒
  **the latent lacks a readable lead-distance variable**; aux-label lever retracted by the PI;
  label-free LF0–LF4 program instead), P7 calibration (v5f ρ 0.49 pass; both v5.8f learned arms
  fail), P5 instrument byte-close, and the P8 attempt-1 instrument lesson (unweighted BCE on
  overwhelmingly-empty rasters collapses to all-empty; IoU-at-0.5 on a collapsed readout is a
  ratio of noise; fix = auto pos-weight + soft-Dice + τ\* swept on the encoded arm). **New
  §7.15** — Alpamayo-2-Super augmentation (**counts now MEASURED and registry-anchored at
  `MODEL_REGISTRY.md` §11.1 — 4,729 clips / 23,644 rows, correcting the card-derived 4,800 /
  23,999 that stood here as INHERITED**; plus the PH1-fused layer at §11.2 with its named 57.2 %
  SAM3 hole), the obstacle.offline join
  (26,084 records / 137 clips, occlusion flags), and the VLM strategic-labeling design + PH0
  three-arm prereg with its admissibility rules. §7.11(f) status-noted; §10 extended with the
  W7-FULL/T1/release-row ladder, the v6 staged direction (S-W/S-P/S-J), LF0–LF4, and the S1–S4
  scaling ladder; references extended. Every §7.13–§7.15 number is MEASURED with artifact
  filename or registry section inline, tier-stamped, with corpus-grid point estimates named as
  such where the cluster-CI rescore is still pending.
- **v1.0 (2026-08-12): the four verdicts, their mathematics, the doctrines as method, and the v6
  ladder.** **New §7.16** — W7-FULL, the selector-free planner: gate FAIL 3.3348 m vs 0.4505 m
  over a 0.1273 m-oracle fan with **no shortlist** (256/256, `winner_in_shortlist_frac` 1.0) on
  the repaired trunk with a refit head; within-window rank correlation 0.445/0.497, across-window
  ρ 0.3185 [0.2064, 0.4086] (episode-cluster bootstrap), **argmin error-rank 132 of 256**, top-m
  mean error flat at the fan mean (~5.32) for every m while top-m ceilings fall (0.356 at m = 32),
  roll-cost alone 5.2898 ⇒ the deployed number comes from the kinematic term; plus the
  **consumer-invalidation** result (frozen selector 0.7933 → 4.4159 on repaired features) and the
  consequent decision that v5.8f ships the **frozen-trunk** assembly at 0.4815 [0.3928, 0.5771].
  **New §7.17** — H-COTRAIN **REJECTED within the measured range** on the λ_plan dose–response
  curve (curvature 0.213 → 0.551 enc / 0.225 → 0.704 pred; yaw 0.583 → 0.869; P1 FAIL → PASS at
  20 k) and **SIGReg VALIDATED** (participation ratio 4.53 → 6.94 of 2048, +53 %, retention 1.532
  vs a ≥ 0.8× gate; top-8 share 0.9903 → 0.9232), with three scope bounds stated in the section
  rather than buried — lowest available λ is 0.5 not 0, the spectrum series ends at 20 k not the
  pre-registered 30 k, and a **real transient** dips the predicted-latent readouts at 10 k right
  after λ hits 1.0 before they recover and overshoot — and the **retraction** of the erosion
  premise this paper carried at v0.9 (root-cause class: a mechanism inferred from a coincidence
  of symptoms, never dose-tested). **New §7.18** — P8 attempt-2 **GATE PASS** (retention 0.932 at
  k = 10; IoU 0.01869 pred / 0.02005 enc; τ\* = 0.7 chosen on the *encoded* arm; 74× lift over
  attempt 1; pos_weight 79.7 from 1.239 % positive cells), P4 permanence (occluded recall ≥
  visible: enc 0.2178 vs 0.1881, pred 0.1743 vs 0.1717, n 194/548) with the **diffuseness
  control** named as required before it stands alone, and **I4a** (intact 0.4011 / shuffled
  1.2492 / zeroed 7.6493 — the ordering, not the magnitudes, is the result). **New mathematics:**
  §3.10 the label-free commitment as an admissibility algebra (L1–L3); §3.11 the 4-brain
  hierarchy formally — per-layer predictors, goal-token conditioning downward, latents upward
  through sg/EMA, and the **gradient-isolation matrix** with its autograd probe and the two
  vacuous-pass traps; §3.12 **the winner's curse** — the Gaussian-copula model (capture fraction
  = ρ; E[error│argmin] ≍ −ρ√(2 ln N), which our data *refutes*), lower-tail dependence λ_L as the
  statistic that actually governs selection, and the **degenerate-minimiser** model that the data
  supports, with its prescribed remedies (a term inaction cannot minimise; top-m aggregation;
  monotone recalibration); §3.13 SIGReg, participation ratio / effective rank, and what
  "retention" measures (a ratio, with named endpoints). **Doctrine promoted to method:** §5 is
  now sectioned, with **§5.4 the eval-tier doctrine** (T0 WM-diagnostic / T1 primary / T2 not
  provisioned, on the measured action-echo basis) and **§5.5 the four metric families** as the
  evaluation contract — including the honest statement that **STRATEGIC is uncomputable on
  PhysicalAI-AV** (no map, no lane graph, no route signal; the card's verbatim "we do not include
  open maps data"; `obstacle.offline`'s 87,481 cuboids are 10 all-dynamic classes; `egomotion`
  has no GNSS) and exactly what would close it; §5.3 gains the **algebraic** account of why
  `overlapping_holdout_se` moves the point estimate (a multiplicity-weighted mean, bias
  Cov(ω, e)). **§10 rewritten as future work:** the v6 staged ladder S-W → S-T → S-S → (optional)
  S-J with each stage's pre-registered gate and the O1–O6 measures; the VLM/SAM/Alpamayo
  vocabulary pipeline **promoted to a precursor** of v6 (PH0 → PH1 → PH2) because S-S cannot be
  gated until STRATEGIC is computable; the 6 s single-rollout horizon; and **§10.1, the open PI
  decisions with their costed inputs** — S-W ESTIMATED 175–290 A40-hours (ladder ≈ 220–370), the
  6 s windowing's MEASURED ≈ 43 % window loss at untouched parity, and the E-ENC matched pair
  (per-layer 120.74 M vs shared `--pred-dim 960` 118.11 M, 2.2 % residual gap). **New Figure 3**
  (winner's curse), generated by a script that reads the gate JSONs at generation time and
  refuses to draw if one is missing or if two artifacts disagree on the fan oracle — palette
  re-validated (`#2a78d6,#eb6834,#4a3aa7`, all pairs, light: ALL CHECKS PASS). **Figure 2
  re-rendered**: its bar-value labels were *invisible in the PNG* because
  `paint-order="stroke"` is not honoured by the rasteriser, so every label was erased by its own
  white halo — present in the file, absent from the render; the halo is now a separate underlay
  element, the takeaway cell no longer overflows into the footer, and its footer (written while
  W7-FULL was queued) now states the outcome. Corrections: the PH0 sequencing sentence in §7.15
  (the pipeline now runs *ahead of* v6, per the PI's 2026-08-11 resequencing) and the
  clarification that 0.4505 m is the pre-registered **gate threshold**, not a deployed arm's
  score. A third correction is registry-vs-doc: §10.1's S-W cost basis had inherited the design
  note's attribution of the MEASURED 59.04 h / 7.085 s-per-step row to **"v4.2"**; the registry
  assigns it to **`flagship-v4-fromscratch` (§1.5.5)**, while `flagship-v4.2-30k` (§1.5.3) was
  killed at ~step 5 k and has no 30 k cost. The registry wins and the paper now names the right
  arm; the estimate is unaffected because the basis runs are the same two.

