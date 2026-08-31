<title>raw/QUOTES — tiered primary passages, 2026-08-31</title>

# raw/QUOTES.md — the quotable layer, with tiers

`Three tiers per SPEC §3. ⛔ A ⚠️RELAYED item may not be quoted in a claim, may not carry a`
`number, and may not settle a novelty question. It is listed so the verification is schedulable.`

---

# TIER ⭐ VERIFIED — I opened these myself on 2026-08-31

## V1 · `2606.12987` — *Diffusion Transformer World-Action Model for AV Scene Prediction*

**Authors:** Ruslan Sharifullin, Benjamin Jiang, Kai Xi Chew · **Submitted:** 2026-06-11
**Retrieval:** `arxiv.org/abs/2606.12987` (abstract) **and** `arxiv.org/html/2606.12987v1`
(full text, the action-definition section). Corpus: **150 held-out nuScenes scenes.**

> "Ego-vehicle actions are extracted from CAN-bus data as 2D vectors aₜ=(steerₜ, accelₜ),
> z-score normalized using training-set statistics only." — **full text**

> "The model is genuinely action-controllable (steering drives scene displacement,
> Spearman ρ = 0.81, vs −0.18 for regression)." — abstract

> "We trace limited single-pass motion to a shared-present anchor and engineer a compact
> 1.7M-parameter 'jump' model that recovers full ground-truth motion magnitude (1.02× GT),
> where single-pass models capture less than half." — abstract

> "a compact (1.7 M-parameter, n_blocks=2, dim 192) network predicts a single Δt=4 transition
> z_t → z_{t+4}, conditioned on the four intervening actions." — **full text**

> "At inference, this model is applied as a 4-step open-loop chain, re-anchoring on its own
> output at each step" — **full text** (to reach 8 s at 2 Hz)

> "across six frozen encoders spanning four representation families, V-JEPA2 with temporal
> context reduces steering RMSE by 40% over the best single-frame encoder" — abstract

> "distortion metrics (cosine similarity, SSIM) favor the blurry mean, masking that the
> diffusion model is far closer to the real frame distribution" — abstract

> "diffusion attains KID 0.078 versus 0.375 for regression (4.8× better)" — abstract

## V2 · `2607.22535` — *Robot-Factored World Models via Robot Rendering*

**Authors:** Byungjun Kim, Taeksoo Kim, Hyunsoo Cha, Hanbyul Joo · **Submitted:** 2026-07-24
**Retrieval:** `arxiv.org/abs/2607.22535`, abstract. ⛔ Robotics, not driving.

> "Conditioning directly on action commands asks the world model to learn the realization
> process itself, while conditioning on logged future states leaks the interaction outcomes
> it is meant to predict."

> "each command is rolled through the robot's own controller and kinematics into a
> deployment-available nominal trajectory, a middle signal that avoids both action-realization
> learning and future-state leakage"

## V3 · `2608.04896` — *When Shared Rollouts Fail in Defensive Driving Evaluation: A NAVSIM Score Basis Audit*

**Authors:** Ziang Wei, Minjun Yu, Zheyuan Lai, Mingjie Pang, Wei Li · **Submitted:** 2026-08-05
**Retrieval:** `arxiv.org/abs/2608.04896`, **abstract only.**

> "Re-simulation benchmarks may use reference-conditioned forgiveness, under which an agent
> receives credit when the logged human reference fails a compliance channel."

> "When agent and reference share an unstable rollout transformation, this rule can propagate
> shared reference failures into broad compliance credit."

> "We audit this risk in NAVSIM v2.2 original scene single-stage scoring. Under the affected
> documented-stack condition on the audited numerical backend, the route-blind Ignore-All
> probe and a route-aware actor-blind probe outrank human replay and PDM-Closed over the
> complete 12,146-token navtest split."

⚠️ **Scope conditions are in the sentence and must travel with any citation:** *NAVSIM v2.2*,
*original scene*, ***single-stage*** *scoring*, *the affected documented-stack condition*,
*the audited numerical backend*. ⛔ **Specific score values (e.g. a probe at 79.6 vs human
74.0) are NOT in the abstract and I did not open the full text — UNVERIFIED, do not quote.**

## V4 · NAVSIM official repository README — the changelog

**Retrieval:** `raw.githubusercontent.com/autonomousvision/navsim/main/README.md`, fetched
2026-08-31. Official project documentation, **VERIFIED**; not bankable as a PDF.

> **2025-04-28 — NAVSIM v2.2:** "Fixed bug in `openscene_meta_datas` for `navhard` and
> `warmup`—If you used `navhard_two_stage/openscene_meta_datas` or
> `warmup_two_stage/openscene_meta_datas` to evaluate your model, please re-download and use
> the new data."

> **2025-09-29 — Bugfix:** "Fixed a bug in metric filtering where `multiplicative_metrics_prod`
> and `weighted_metrics` were not correctly excluded by the human filter"

> **2025-07-16 — ICCV Warmup Leaderboard:** "This release introduces a registration system…"

**Latest release named on the page: NAVSIM v2.2 (2025-04-28).** ⇒ no v3 as of this retrieval.

## V5 · `2503.20523` — GAIA-2 (Wayve)

**Title:** *GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving*
**Submitted:** 2025-03-26 · **Retrieval:** `arxiv.org/abs/2503.20523`, **abstract only.**

> "conditioned on a rich set of structured inputs: ego-vehicle dynamics, agent configurations,
> environmental factors, and road semantics"

⛔ **What the abstract does NOT contain, checked for:** (i) any statement that the ego action
is *speed and curvature*; (ii) **any quantitative metric of ego-action adherence or
controllability.** ⇒ a "GAIA-2 uses speed+curvature" attribution is **NOT supported by what I
read** and must not be written.

---

# TIER **BANKED** — existence, title and date confirmed by `kb_add.py` fetching the PDF

`Citable as pointers. Their internals are NOT verified by me.`

| key | title (as banked) |
|---|---|
| `2607.15898` | Orbis 2: A Hierarchical World Model for Driving |
| `2506.09981` | ReSim: Reliable World Simulation for Autonomous Driving |
| `2607.22430` | On the Identifiability of Controlled World Models |
| `2608.06706` | Dueling World Models: Advantage-Style Action Channels for Common-Mode … |
| `2511.23369` | SimScale: Learning to Drive via Real-World Simulation at Scale |
| `2607.05133` | UNIVERSE: Unified Video Action Models for Autonomous Driving with Flex… |

---

# TIER ⚠️ RELAYED — reported by a delegated sweep, NOT read by me

⛔ **None of the following may be quoted, may carry a number into a claim, or may settle the
novelty question. They are listed as VERIFICATION TARGETS, ranked by what they would change.**

| # | relayed content | what it would change if true | priority |
|---|---|---|---|
| R1 | **Orbis 2** runs counterfactual trajectory scaling (×0.5 / ×1.5) against an **unaltered-ground-truth arm** | ⛔ would put our 2026-08-30 novelty claim — *"a hold-action floor a model must beat: not found in any driving world model"* — **at risk**. A null/unaltered arm is not identical to a floor the model must BEAT, and only reading the paper can tell which it is | ⭐⭐⭐ **highest** |
| R2 | **Orbis 2** uses a two-rate hierarchy: an abstract high-level latent at ~2 Hz over a ~10 Hz low level (≈5× stride) | would give P4 its **first driving-domain** temporal-abstraction precedent, and a stride to compare against MM-E16's 1 : ~3 : ~15 | ⭐⭐ high |
| R3 | **ReSim**: expert-only real data restricts the state-action space, causing hallucination under unseen non-expert actions; importing non-expert behaviour improves controllability substantially | would add a **third P2 lever — action-space coverage** — that is neither the channel nor the target, and connects to L2D's 13.8 % STUDENT split | ⭐⭐ high |
| R4 | **Dueling World Models** describes models that *"quietly go action-blind"* while training loss improves | a name for our failure mode; a common-mode-cancellation mechanism | ⭐ medium |
| R5 | **On the Identifiability of Controlled World Models** requires non-degenerate conditional action variation for transition identifiability | would make our near-degenerate action channel a *provable* identifiability problem rather than an empirical one — ⚠️ and MM-E12 already measured that the action is NOT predictable from the scene, so the transfer is not obvious | ⭐ medium |
| R6 | **UNIVERSE** §3.2 describes trajectory/video token leakage and a Modality-Decoupling Visibility Mask | a masking-side treatment of the leakage horn named in V2 | medium |
| R7 | **SimScale**: the same method gains +8.6 EPDMS on navhard vs +2.9 on navtest | would quantify how split-dependent an EPDMS delta is — directly relevant to our pinned target | ⭐⭐ high |
| R8 | The wider claim that **nearly all** frontier driving WMs derive the action from the ego log | the "field norm" half of F1. ⚠️ V5 shows the check is non-trivial: GAIA-2's abstract does **not** state its action parameterisation at all | ⭐⭐⭐ **highest** |
