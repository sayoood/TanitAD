# Research — Frozen World-Model + Learned End-to-End Planner

**Author:** frozenwm-planner subagent · **Date:** 2026-07-23 (Europe/Berlin) ·
**Status:** research bank (P0). Additional research direction — modifies nothing that is running.

**Evidence-class convention (binding, CLAUDE.md §Operating standard 1).** Every claim is tagged
`[PUB:<cite>]` published & cited · `[MEAS:<artifact>]` measured by us · `[INH:<src>]` inherited from
another agent/doc, not re-verified by me · `[EST]` estimated · `[HYP]` hypothesis. No GPU-day decision
rests on an `[INH]`.

---

## 0. The question, and why it is worth a research bank

Two of the program's own measurements point in opposite directions, and the gap between them is an
unexplored design axis:

- **Joint planner+WM training DEGRADES the world model.** `[INH: V4_FLAGSHIP_DESIGN.md §5.1;
  planner-wm-gradient-coupling/DESIGN.md; LOOP_STATE]` v1.6 unfroze 4 ViT blocks + the predictor under a
  planner objective → the plan-free WM canary rose **0.452 → 1.1022 (+144 %)** `[MEAS: eval_v16.json,
  REGISTRY §1.4b]`; the v4 saga's warm-start coupling drove the operative-rollout canary **0.42 → 1.30+**
  (v4, runaway) and **0.72 @ 4k** (v4.2) `[INH: planner-wm-gradient-coupling/DESIGN.md §arm-table]`.
  Critically, the coupling stream MEASURED that the trunk **kept degrading with the planner gradient
  clamped to zero** — *"the trunk fine-tuning, not the planner, is the culprit"* — i.e. a **warm-start
  single-task effect**, the warm world model re-optimising under its own loss at a warm-start-incompatible
  LR `[INH: same, LOOP_STATE §hist]`.
- **A frozen ENCODER bottlenecks.** `[MEAS: REGISTRY §2 REF-A]` REF-A froze a DINOv2 encoder and, even
  with a fully trained 4-brain predictor on top, plateaued at **2.13–2.92 m** ADE@2s, speed-blind (root
  cause: the frozen features discarded speed/scale magnitude) `[MEAS: refa-bottleneck memory; REGISTRY]`.

**Freezing the whole world model resolves the first tension trivially — a frozen trunk cannot
re-optimise, so the degradation the coupling stream fights with LR controllers and gradient surgery
(PCGrad/GradVaccine) simply cannot occur.** The open question is whether it re-introduces the second:
**does a frozen v1 WM bottleneck a learned planner the way a frozen DINO encoder bottlenecked REF-A?**
This document surveys how the field freezes world models and learns planners against them, ranks the
mechanism options for our exact setting (offline logs, safety-critical, a *differentiable* WM in hand),
and states the theory that predicts which regime we land in.

---

## 1. Lineage — systems that FREEZE the model and learn the planner, and HOW

The through-line: a world model is trained (by prediction / value-equivalence), then the **policy/planner
is optimised against the model with the model's parameters held fixed** during that phase. They differ in
*how the planner extracts a decision from the fixed model* — and that difference is the whole design space.

### 1.1 Dreamer / "Dream to Control" (Hafner et al., 2019) — ANALYTIC value gradients through the fixed dynamics
`[PUB: arXiv:1912.01603]`

- **Model:** an RSSM latent world model (deterministic GRU state + stochastic latent), trained by
  reconstruction + reward + KL. Decoder-based.
- **How the planner uses it — the mechanism I am building on.** Behaviour is learned *purely in latent
  imagination*: from each replayed state, roll the action model forward under the **fixed** dynamics to
  produce an imagined latent trajectory, estimate the value, and **backpropagate the analytic gradient of
  the value back through the chain of imagined states into the action model.** `[PUB: arXiv:1912.01603 §3,
  "propagating analytic gradients of learned state values back through trajectories imagined in the
  compact latent space"]` Continuous actions are made differentiable by a **tanh-transformed Gaussian with
  reparameterised sampling**, so the gradient passes through the sampling op. `[PUB: same]`
- **Frozen?** During behaviour learning, **yes** — the world-model parameters are fixed; only the
  actor/critic update. The model is refreshed on a separate cadence. This is exactly the "fixed
  differentiable simulator, learn the policy against it" regime.

### 1.2 DreamerV2 / V3 (2020 / 2023) — the SAME freeze, a DIFFERENT gradient by default
`[PUB: arXiv:2010.02193; arXiv:2301.04104; Nature 2025 s41586-025-08744-2]`

- V2/V3 keep the freeze (actor-critic trains on **detached** latent states — no gradient into the world
  model) but for the **actor gradient** default to **REINFORCE / score-function** returns (normalised
  λ-returns + entropy) rather than dynamics-backprop, because straight-through analytic gradients proved
  **less robust across the full domain suite** `[PUB: arXiv:2301.04104; corroborated by the web synthesis
  "the actor is trained using REINFORCE … detached world model states"]`.
- **The nuance that matters for us:** "Dreamer freezes the WM" is true throughout, but **only DreamerV1
  (and V2/V3's optional dynamics-backprop path) uses the analytic-gradient-through-the-model mechanism our
  brief names.** DreamerV3's *default* does not backprop the driving cost through the model — it estimates
  a return and uses a score-function gradient. Our setting has a **differentiable metric readout on the
  trajectory** (`step_readout` → SE(2) accumulate, all differentiable), so the analytic path is available
  to us where DreamerV3 chose the score-function path for cross-domain robustness. `[MEAS: metric_dynamics.py
  rollout_decode is fully differentiable — smoke: grad reaches fed actions, norm 0.85]`

### 1.3 TD-MPC / TD-MPC2 (Hansen et al., 2022 / 2024) — MPC over the model + a LEARNED policy PRIOR
`[PUB: arXiv:2203.04955; arXiv:2310.16828]`

- **Model:** a decoder-free (implicit) latent world model trained by a **value-equivalent / TD** objective
  — it need not reconstruct pixels, only predict reward + value + next latent consistently. `[PUB:
  arXiv:2310.16828]`
- **How the planner uses it:** at each step, **local trajectory optimisation in the latent space (MPPI /
  CEM)** — sample action sequences, roll them through the model, score by predicted return, iteratively
  refit the sampling distribution. A **learned policy prior** (a small amortised actor) **seeds** the
  sampler and is trained to imitate the planner's output, so the expensive search is warm-started and
  distillable to a fast policy. `[PUB: arXiv:2310.16828 §3]`
- **Frozen?** The planner (MPPI) is *non-parametric* — it uses the model at decision time with no learning.
  The **policy prior is the learned amortisation** and is trained against the fixed model's rollouts.
  Scales to **317 M params / 80 tasks, one hyperparameter set.** `[PUB: same]`
- **Why it is central for us:** TD-MPC2 is the reference design for **(c) amortised/learned MPC** — exactly
  the pairing of our already-measured P2 CEM planner over the frozen WM (20.8 ms @ K=8 `[MEAS: REGISTRY
  §1.2]`) with a learned prior distilled from it.

### 1.4 MuZero (Schrittwieser et al., 2020) — value-equivalent model + MCTS, policy DISTILLED from search
`[PUB: arXiv:1911.08265; corroborated arXiv:2102.12924 "Visualizing MuZero Models"]`

- **Model:** representation + dynamics + prediction networks, trained **value-equivalently** (the model is
  optimised so that its *predicted policy/value/reward* match search-improved targets — it explicitly does
  **not** model observations). `[PUB: arXiv:2306.00840 "What model does MuZero learn?"]`
- **How the planner uses it:** **MCTS** unrolls the learned model to produce an improved action-visit
  distribution at the root; that distribution is the **training target for the policy network** — planning
  improves the policy, the model only *evaluates*. `[PUB: web synthesis: "uses the output of the search as
  training targets for a learned policy network"]`
- **Frozen?** Within an update the model is fixed and the policy is trained toward search targets. The
  discrete-action + reward-model assumptions make MuZero-proper a **poor direct fit** for our
  continuous-control, reward-free, offline-logs setting (Sampled-MuZero/EfficientZero relax the action
  space but still need a reward/return signal we don't natively have from logs).

### 1.5 IRIS / transformer world models (Micheli et al., 2023) and PlaNet (2019)
`[PUB: arXiv:2209.00588 (IRIS); arXiv:1811.04551 (PlaNet)]`

- **PlaNet:** learn an RSSM, then **plan by CEM at decision time** over the fixed model (no learned actor
  at all) — the purest "model as simulator, search the plan." `[PUB: arXiv:1811.04551]`
- **IRIS:** a discrete-token autoencoder + a **Transformer world model**, with the **policy trained inside
  imagined rollouts** of the (per-phase fixed) transformer model — the Dreamer recipe with an
  autoregressive-transformer backbone. Establishes that a transformer WM supports imagination-based policy
  learning. `[PUB: arXiv:2209.00588]` `[EST: relevance — our operative predictor is a causal transformer,
  so IRIS is the closest backbone analogue to ours.]`

**Summary table — who freezes the WM, and how the planner extracts the decision:**

| System | Model objective | Planner extracts decision by | WM frozen during policy learning | Gradient into policy |
|---|---|---|---|---|
| Dreamer v1 `[PUB:1912.01603]` | recon+reward | **analytic value grad through fixed dynamics** | yes | **backprop through model** |
| DreamerV3 `[PUB:2301.04104]` | recon+reward | actor-critic on **detached** imagined states | yes | REINFORCE (default) |
| TD-MPC2 `[PUB:2310.16828]` | value-equivalent/TD | **MPPI/CEM** + learned **policy prior** | prior trained vs fixed model | distillation from search |
| MuZero `[PUB:1911.08265]` | value-equivalent | **MCTS**, policy ← search targets | yes | distillation from search |
| PlaNet `[PUB:1811.04551]` | recon | **CEM at decision time** | n/a (no learned actor) | — |
| IRIS `[PUB:2209.00588]` | token recon | policy in transformer-imagination | yes | actor-critic in imagination |

---

## 2. Mechanism options for "learned e2e planner on a frozen WM," RANKED for OUR setting

Our setting is specific and it prunes the field hard: **offline expert logs only** (no online interaction,
no simulator reward), **safety-critical** (interpretable failure modes preferred), and — decisively — **a
differentiable world model already in hand** whose trajectory readout is metric and differentiable
end-to-end `[MEAS: metric_dynamics.rollout_decode; smoke grad-check]`.

**(a) ANALYTIC GRADIENT through the frozen differentiable WM — RANK 1 to prototype.**
Backprop the driving cost (ADE / smoothness) through the frozen `rollout_decode` chain into the planner —
DreamerV1's actor mechanism with the WM frozen. **Why rank 1 here:** the gradient already flows in our
stack (verified); it is the **cheapest to prototype** (no reward model, no search loop, no online rollouts);
and offline expert logs give a *dense, well-posed* target (the expert trajectory), which is exactly the
regime where analytic gradients are best-behaved. **Known caveat `[PUB: SHAC arXiv:2204.07137; "Do
Differentiable Simulators Give Better Policy Gradients?" arXiv:2202.00817]`:** BPTT through many steps can
give **biased/high-variance** gradients on stiff or chaotic landscapes (exploding/vanishing, noisy loss
surface). Mitigations the literature validates: **short horizons / truncation** (SHAC cuts the graph and
uses a critic for the tail), gradient clipping, and comparing against a score-function baseline. Our
horizon is a modest **20 steps (2 s)**; clip + (if needed) truncate.

**(b) MODEL-BASED RL, WM as neural simulator — RANK 3.** Train the planner by RL against the frozen WM as
an environment (Think2Drive's recipe, §3). **Why lower for a first proof:** it needs a **reward model**
(we have expert logs, not rewards — reward design is its own project) and an **on-policy rollout loop**
(more moving parts, slower to a first number). It is the right *second* step once a reward is defined, and
it inherits the exact same frozen-WM ceiling question.

**(c) AMORTISED / LEARNED MPC — RANK 2, and the strongest *product* path.** Run CEM/MPPI over the frozen WM
(our P2 planner, already **20.82 ms p50 @ K=8** `[MEAS: REGISTRY §1.2]`) and **distil the search into a fast
learned prior** (TD-MPC2's policy prior; MuZero's policy-from-search). **Why rank 2:** it reuses an asset we
have measured to be latency-feasible, degrades gracefully (the search is a safety net if the prior is
weak), and is interpretable. Slightly more infrastructure than (a) for a first number, but the natural
destination if (a) shows the frozen WM is a faithful simulator.

**(d) IMAGINATION-ROLLOUT IMITATION — RANK 4 / baseline.** Behaviour-clone the expert *actions*, then let
the frozen WM convert actions→trajectory. Cheapest of all, but it never uses the WM as a *cost model* — so
it is the **control** that isolates what the analytic gradient buys, not a candidate mechanism. (This is
our experiment's Arm B.)

**Verdict for the cheapest discriminating proof: (a).** It is the one that (i) is already differentiable in
our code, (ii) needs no reward model or online loop, and (iii) directly answers the ceiling question. (c)
is the product path; (b) is deferred behind reward design.

---

## 3. Driving-specific — world models used to train/条件 a driving planner

- **MILE (Wayve, NeurIPS 2022)** `[PUB: arXiv:2210.07729]` — **model-based imitation learning** for urban
  driving: **jointly** learn a BEV latent world model + a driving policy from **offline expert video**, no
  online interaction; executes maneuvers "entirely predicted in imagination"; **+31 % driving score** in a
  new town/weather on CARLA. **Directly relevant but it is the COUPLED regime** (world model + policy
  trained together) — the very regime our program measured to degrade the WM. MILE succeeds because it
  co-designs the objective from scratch; our tension is specifically about *warm-starting* a
  prediction-converged WM and then coupling — which MILE does not do.
- **Think2Drive (ECCV 2024)** `[PUB: arXiv:2402.16720]` — **DreamerV3 as a neural simulator** for CARLA-v2:
  the world model learns transitions/reward/termination and **the planner is trained to maximise
  WM-predicted reward**, reaching **expert-level in 3 days on a single A6000**. This is mechanism **(b)** in
  a driving setting and a proof that a learned latent WM can train a competent driving planner — but online,
  with a reward, in sim.
- **Recent AD world models** `[PUB: arXiv:2403.02622 survey; arXiv:2501.13072 AdaWM; arXiv:2510.12560
  CoIRL-AD]` — the 2024–2026 line increasingly **separates a (pre)trained world model from a policy** and
  studies adaptation/imitation-RL hybrids in latent WMs; the recurring finding is that the **world model's
  representation quality caps the planner**, which is precisely our ceiling question.

**Reading for us:** the driving field's *joint* successes (MILE) are from-scratch co-training; its
*model-as-simulator* successes (Think2Drive) are online-with-reward. **The specific cell we occupy —
offline logs + a *warm, frozen, differentiable* WM + a learned planner via analytic gradient — is
under-explored, and is exactly the additional direction this bank scopes.**

---

## 4. Theory — frozen-WM ceiling vs freedom-from-degradation: when does frozen win?

The RL-representation literature gives a sharp predictor `[PUB: "Pretrained Visual Representations in RL"
aimodels/researchgate 2024; arXiv:2203.03580; kayburns segmentingfeatures]`:

> **Frozen features are "sufficient to learn an (near-)optimal policy in almost all tasks"; they bottleneck
> exactly on the tasks where the frozen representation *discards task-relevant information.*** Fine-tuning
> then helps most precisely there (the cited Reacher-hard case: the frozen encoder collapsed a
> goal-size cue the task needed).

Mapped to us, this makes a **falsifiable prediction**, not a vibe:

- **REF-A bottlenecked because its frozen DINO features discarded the task-relevant variable** — speed/scale
  magnitude `[MEAS: refa-bottleneck memory]`. That is the "discards task-relevant info" failure mode, on
  the nose.
- **The frozen v1 WM is a different object.** It was trained *with* the speed channel and *with* metric
  ego-motion grounding, and it demonstrably retains the metric information **in its action-conditioned
  dynamics** — the operative rollout under true actions scores **0.4045–0.4271** `[MEAS: REGISTRY §1.2;
  this experiment refs]`. The theory therefore predicts **frozen v1-WM should NOT bottleneck a planner the
  way frozen DINO did — provided the planner reads the information through the *dynamics*, not by static
  decode of the latent.**
- **The static-decode caveat is itself measured:** v1's JEPA latent does **not** linearly hold metric
  ego-trajectory (held-out MLP probe **3.89 m** ADE@1s) `[MEAS: metric_dynamics.py docstring]`. So a planner
  that decodes the trajectory *directly off the frozen state* should hit ~REF-A-grade numbers, while one
  that *drives the frozen dynamics* should approach the 0.40 rollout ceiling. **This split is the
  experiment (P2): Arm F (direct decode) vs Arm W (through dynamics).**

**The trade, stated cleanly.** Freezing buys **freedom from degradation for free** (a frozen trunk cannot
re-optimise — it removes the *single-task warm-start effect* the coupling stream MEASURED even at
λ_plan = 0, and it removes any planner→WM gradient conflict by construction). It **spends** the ability to
adapt the representation to the planner. **The theory says that spend is cheap when the frozen model already
contains what the planner needs** — and our WM, unlike REF-A's encoder, was built to contain it. Whether
that holds *through a frozen planner-driven rollout* is an empirical question with a cheap, decisive answer.

---

## 5. What this implies for the design (feeds P1)

1. **Prototype with mechanism (a)** — analytic gradient through the frozen WM — because it is
   already-differentiable, reward-free, and cheapest to a first number.
2. **Read the ceiling through the dynamics, not the static latent** — the planner must output *actions* the
   frozen predictor rolls, not decode waypoints off the frozen state (the latter is the measured 3.89 m
   trap). Include the static-decode arm as the ceiling control.
3. **Guard the analytic gradient** — clip; be ready to truncate the 20-step BPTT (SHAC) or fall back to the
   score-function/BC baseline if the gradient is pathological. Ablate (a) vs (d) to quantify what the WM-as-
   cost-model buys.
4. **The product destination is (c)** — distil the measured-feasible P2 CEM search over the frozen WM into a
   learned prior (TD-MPC2). (a) settles whether the frozen WM is a good enough simulator to make (c) worth
   building.

---

## 6. Sources (primary unless noted)

- Hafner et al., **Dream to Control: Learning Behaviors by Latent Imagination**, arXiv:1912.01603 —
  https://arxiv.org/abs/1912.01603
- Hafner et al., **Mastering Diverse Domains through World Models (DreamerV3)**, arXiv:2301.04104; Nature
  2025, https://www.nature.com/articles/s41586-025-08744-2
- Hansen et al., **TD-MPC2: Scalable, Robust World Models for Continuous Control**, arXiv:2310.16828 —
  https://arxiv.org/abs/2310.16828 ; **TD-MPC**, arXiv:2203.04955
- Schrittwieser et al., **MuZero**, arXiv:1911.08265 ; **What model does MuZero learn?**, arXiv:2306.00840 ;
  **Visualizing MuZero Models**, arXiv:2102.12924
- Hafner et al., **PlaNet**, arXiv:1811.04551 ; Micheli et al., **IRIS**, arXiv:2209.00588
- Xu et al., **Accelerated Policy Learning with Parallel Differentiable Simulation (SHAC)**,
  arXiv:2204.07137 ; Suh et al., **Do Differentiable Simulators Give Better Policy Gradients?**,
  arXiv:2202.00817
- Hu et al., **Model-Based Imitation Learning for Urban Driving (MILE, Wayve)**, arXiv:2210.07729 —
  https://anthonyhu.github.io/mile
- Li et al., **Think2Drive**, arXiv:2402.16720 — https://thinklab-sjtu.github.io/CornerCaseRepo/
- **World Models for Autonomous Driving: survey**, arXiv:2403.02622 ; **AdaWM**, arXiv:2501.13072 ;
  **CoIRL-AD**, arXiv:2510.12560
- **Pretrained Visual Representations in RL**, aimodels.fyi/researchgate 2024 ; Parisi et al.,
  **The (Un)surprising Effectiveness of Pre-Trained Vision Models for Control**, arXiv:2203.03580
