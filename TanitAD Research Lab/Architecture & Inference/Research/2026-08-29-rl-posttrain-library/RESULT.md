# RL post-training for the refcv3 planner — the method library, Phase 1 (research + banking)

**Date:** 2026-08-29 · **Stream:** TanitAD_TrainingFlyWheel delegate on PI commission **D-RL-REFCV3**
(`Project Steering/GOALS_AND_CLAIMS.md:1571`) · **Scope:** research + banking **ONLY**. No code was
written, no library was created, **0 GPU**. Thor was not touched.

**What this document is NOT.** It does not repeat
`…/Research/2026-08-29-refc-origins-successors/RESULT.md` (the DiffusionDrive/DDv2 lineage read — its
findings are taken as given), and it does not repeat `products/P4-training-pipelines/METHOD_LIBRARY.md`
(the RL/GRPO/DPO family survey for our stack — its **§2.1** and **§2.2** structural results are treated
as established and are used, not re-derived). This is the **refcv3-planner-specific layer neither
covers**: what DDv2's method actually is at equation level, what a reward can be built from *here*, and
which estimator survives contact with our signal ledger.

**Evidence classes** (`CLAUDE.md` operating standard §1):
`PUBLISHED-PRIMARY (banked, sha-verified)` = read from a Library PDF whose sha256 I recomputed against
the banked entry **this session** · `MEASURED (ours)` = repo artifact / registry / raw JSON ·
`MEASURED-ABSENCE` = probed twice and not found · `DERIVATION` = algebra over a primary or over code at
HEAD, checkable by inspection, **not** an experiment · `INHERITED` · `HYPOTHESIS`.

**The four primaries I read end-to-end this session, with the sha256 of the bytes I read** (each equals
its `library.json` entry — verified by content, not by presence):

| key | paper | sha256 (first 24) | pages read |
|---|---|---|---|
| **2512.07745** | DiffusionDriveV2 | `076ce47e0b0a323c9bb00724` | 17/17 (incl. suppl.) |
| **2507.04049** | DIVER | `bf2c358071fe3e5a83cb2196` | 17/17 |
| **2409.00588** | DPPO — Diffusion Policy Policy Optimization | `4b81907b0a8ea7118e492272` | 30/46 (main + best-practices + App. C) |
| **2305.13301** | DDPO — Training Diffusion Models with RL | `fea5c60166a3718d5f435e51` | 16/16 (main + App. A) |

---

## 0. FINDINGS FIRST

1. ⛔⛔ **DDv2 — the lever this commission is built on — DOES NOT PUBLISH ITS REWARD.**
   `MEASURED-ABSENCE (full read, 17 pp, two probes)`. `R(τ_0^{k,i})` appears in Eq. 8 and is **never
   defined anywhere in the paper**; the word "reward" occurs **4 times in 17 pages** and not one
   occurrence is definitional; the supplementary hyperparameter table (Tab. 7) lists **10 knobs and no
   reward weight**; there is **no reward ablation**. The `collision` predicate of Eq. 10 is likewise
   never sourced. ⇒ **The PI's ask — "the precise reward composition" — is unanswerable from the
   primary, and any document that answers it is inventing.** What DDv2 *does* fix is the **advantage
   algebra and the exploration mechanics**, which are fully specified (§A.1).
   ⭐ The reward-design work is therefore **ours**, and the paper that *does* publish a composite
   trajectory reward with per-term weights and a per-term ablation is **DIVER (2507.04049)** — the
   template to copy the **form** from, never the terms (§B).

2. **DDv2's "GRPO" is not GRPO — it is DDPO_SF wearing GRPO's advantage.** `DERIVATION over the
   primary.` Eq. 7 carries **no importance ratio, no clip, no reference model and no KL**. §4.4 states
   the substitution in its own words: *"analogous to how the original GRPO provides regularization by
   adding a KL divergence between the policy model and the reference model, we incorporate an
   additional imitation learning loss"*. So: **the IL loss replaces the KL**, and the surrogate is the
   plain score-function estimator that DDPO (2305.13301 §4) calls `DDPO_SF` and states *"only allows
   for one step of optimization per round of data collection"*. Everything else in DDv2's RL stage —
   `η=1` train / `η=0` eval, the σ≥0.1 likelihood floor, the denoising discount — is **DPPO's
   §4.3 best-practice list, uncited at those three points** (§A.6).

3. **DDv2's headline +3.1 PDMS decomposes into +1.0 selector and ≈+2.1 RL, by their own control.**
   Tab. 9: DiffusionDrive 88.1 → DiffusionDrive **with DDv2's selector** 89.1 → DDv2 91.2. Anyone
   sizing an RL post-train on "+3.1" is over-sizing it by ~48 %.

4. ⚠️ **DDv2's ablation deltas are on a 90.1 base, not on the 91.2 headline, and must never be added to
   88.1.** §5.4: *"all ablation studies … are trained for fewer epochs for rapid validation"*. The
   deltas are intra-anchor **+0.9** (89.2→90.1), inter-anchor truncation **+0.6** (89.5→90.1),
   multiplicative-vs-additive noise **+0.4** (89.7→90.1), coarse-to-fine selector **+0.2**, rank loss
   **+0.2**.

5. ⭐ **Exactly one DDv2 mechanism ports to us at zero signal cost — the truncation rule's SHAPE
   (Eq. 10) — and we already own both of its ingredients in a stronger form.** "Only penalise absolute
   failures, never relative ones" is the same instinct as our reachability prefilter (**72.08 %** of
   REF-C-XL's fan deleted for a paired ΔADE of **exactly 0.0000**, MEASURED) and the refcv3 σ≤0.8 m
   admission gate. What does **not** port is the `collision` predicate that fills the `−1` branch.

6. ⛔ **We cannot build DDv2's or DIVER's reward, and the reason is data, not effort.** DIVER's
   `r_safe` needs an obstacle-clearance map `D_safe(x)`; its `r_LK` needs a **lane centerline**;
   DDv2's `−1` branch needs a **simulator collision verdict**. PhysicalAI-AV ships *"no map, lane
   graph, junction annotation, roundabout label, traffic-light feature or route/goal signal"*
   (`CLAUDE.md`, settled at five independent probes), and `obstacle.offline` — the only agent source —
   **is not in the trainer's batch** (`METHOD_LIBRARY.md` R6). A clean REJECT for both terms.

7. ⛔ **refcv3 has no registry row and no checkpoint.** Two instruments (Grep tool + PowerShell
   `Select-String` over `Project Steering/MODEL_REGISTRY.md`, which carries **78** "REF-C" mentions,
   **0** for refcv3/`REF-C v3`). RL post-training is a **phase on top of an IL cold start** — DDv2 §4.1
   is explicit that the generator is DiffusionDrive's pre-trained weights — so **the IL arm is a hard
   prerequisite**, and it is now scheduled on the **B1 Alpamayo corpus** (`D-CORPUS-B1`), not the parity
   corpus. ⇒ the RL library must be **corpus-agnostic** and must never same-data-compare across that
   boundary.

8. ⭐ **The honest target for RL here is the fan's FLOOR, not its top-1 — and DDv2's own Tab. 3 says
   so.** Raw pre-selector quality: DD PDMS@1/5/10 = 93.5 / 84.3 / **75.3** → DDv2 94.9 / 91.1 /
   **84.4**. The top-1 moved **+1.4**; the 10th-best moved **+9.1**. That is the *same* quantity our
   `frac_sel_2x_worse` **0.454** (XL) / **0.4109** (base) measures from the other side, and it is
   compatible with the registry's standing **"the oracle gap is ~92 % irreducible"** caveat in a way
   that a top-1 target is not.

9. ⚠️ **A CONCURRENT STREAM IMPLEMENTED THE LIBRARY WHILE THIS RESEARCH RAN.** `stack/tanitad/rl/`
   (7 modules + 6 test files) exists and is **already staged**, written 2026-08-29 **16:26–16:44** — i.e.
   during this session. Much of it converges independently with §B/§D (bounded terms with a named
   degenerate policy each, a `FORBIDDEN_REWARD_INPUTS` set, a shipped hackable-reward arm, Dr. GRPO
   centre-only advantage, `freeze_trunk`, DPO refused at `validate()`). **Four substantive
   disagreements remain, and two of them are reward-hacking exposures — §E.** This stream wrote no code
   and did not touch those files.

> ## 🔴 ESCALATION — ONE DECISION, AND IT IS NOT "WHICH ESTIMATOR"
>
> **Every estimator in §C is implementable in days; none of them can run until two things exist that
> this stream cannot produce: (a) a refcv3 IL checkpoint to cold-start from, and (b) a decision on
> whether the reward may read `obstacle.offline`.** Without (b) the reward reduces to
> *distance-to-expert + own-kinematics*, which is **`METHOD_LIBRARY.md` §1.5's C101 cost** — the cost a
> near-oracle CEM optimiser already MEASURED **+0.2585 m [+0.0869, +0.4309] WORSE than constant
> velocity, closed-loop**. ⇒ **Post-training refcv3 against that cost is optimising a cost we have
> already measured into a loss.** The three components in §B that are *not* that cost — fan-floor
> safety, kinematic admissibility, tactical-token agreement — are what make an RL phase worth running,
> and two of the three depend on decision (b).
>
> ⇒ **Requested of the Master Mind / PI:** rule on `obstacle.offline` ingest for the B1 corpus
> (`METHOD_LIBRARY.md` B5) **before** the refcv3 IL arm finishes, so the RL phase does not have to
> choose between waiting and shipping a reward we know is misspecified.
>
> ⛔ **AND THIS IS NOW URGENT RATHER THAN THEORETICAL.** The concurrently-written library's
> **default** reward vector weights `collision` at **1.00** and `headway` at **0.30** — and on our
> corpus **both evaluate to exactly 0 for every candidate**, because nothing populates
> `ctx["obstacles"]` / `ctx["lead_path"]`. A constant cancels identically in the group-relative
> advantage, so the shipped default reduces to a **progress-weighted reward whose two named safety
> checks are inert**. ⭐ **The implementing stream found this independently and made it its blocking
> pre-arm** (`LAUNCH_PLAN.md` §0, arm A0) — so the finding is corroborated from two directions, and
> what is left is a decision, not a measurement. **§E.1.** The ingest ruling is what turns that default
> from a hazard into the design it is described as.

---

## A. DDv2's METHOD, EXACTLY

`All of §A is PUBLISHED-PRIMARY (banked, sha 076ce47e0b0a…), page/section cited per claim.`
arXiv 2512.07745**v1**, 8 Dec 2025, Zou · Chen · Liao · Zheng · Song · Zhang · Zhang · Liu · Wang
(HUST + Horizon Robotics + Wuhan U). Code: *"will be available"* — **not released at v1**, so nothing
below is checkable against an implementation.

### A.1 The objective, written out

**Setting (§3, p.3–4).** `N_anchor` k-means anchors `{a_k}` partition the trajectory space; the
generator is a GMM over anchored Gaussians.

```
(1)   p(τ_k | a_k, z)  =  N( τ_k ;  a_k + μ_k(z),  Σ_k(z) )
(2)   p(τ  | z)        =  Σ_{k=1..N_anchor}  s(a_k | z) · p(τ_k | a_k, z)
(3)   τ_t^k            =  sqrt(ᾱ_t)·a_k + sqrt(1−ᾱ_t)·ε ,   ε ~ N(0, I),  t ∈ [1, T_trunc],  T_trunc ≪ T
(4)   L_IL             =  Σ_k [ y_k · L_rec(τ̂_k, τ_gt)  +  L_BCE(ŝ_k, y_k) ]
                          y_k = 1 for the anchor nearest τ_gt, 0 for every other anchor
```
Eq. 4 is the defect statement: **only `y_k=1` gets a reconstruction gradient**, so "the negative modes,
which constitute the vast majority" are unconstrained (§1, p.2).

**The denoising policy (§4.2, Eq. 5, p.5).** Each conditional denoise step is a Gaussian policy:

```
(5)   π_θ( τ_{t−1}^k | τ_t^k , z, a_k )  =  N( τ_{t−1}^k ;  μ_θ(τ_t^k, t, z, a_k),  η(1−α_t)·I )
```
with **η = 1 during RL training** (i.e. DDPM), **η = 0 at evaluation** (deterministic DDIM). §4.3
gives the reason verbatim: to *"enable broader exploration and prevent the issue of calculating a
likelihood over a Dirac distribution"*. Eq. 6 is then plain REINFORCE over the truncated chain.

**Exploration noise (§4.3, p.5) — multiplicative, and it is a FLOOR, not a schedule.**

```
      τ'  =  (1 + ε_mul) ⊙ τ ,     ε_mul = (ε_long, ε_lat)          # TWO scalars per trajectory
```
Not two per waypoint: one longitudinal and one lateral scale applied to the whole trajectory, so the
explored path stays smooth. Their Fig. 3 contrasts this with per-point additive noise, which *"produces
a jagged exploratory path that resembles a broken line"*.
⛔ **`MEASURED-ABSENCE`: there is no noise SCHEDULE.** The only published quantity is a floor —
Tab. 7 *"Minimum denoising std 0.04"*, set *"to push the model towards sufficient exploration and
prevent entropy collapse"* (suppl. §7). No annealing, no decay, no per-epoch profile is given. A
second, separate floor exists for the *likelihood*: *"we set the standard deviation to be at least 0.1
when evaluating the Gaussian likelihood log π_θ(...), which improves training stability by avoiding
large magnitude gradients"* — that is DPPO's `σ^prob_min` (§A.6).

**Intra-anchor advantage (§4.4, Eq. 8, p.6) — the group is ONE anchor's G samples.**

```
(8)   A^{k,i}  =  ( r^{k,i} − mean_j{ r^{k,1..G} } ) / std_j{ r^{k,1..G} } ,     r^{k,i} = R(τ_0^{k,i})
```
The motivating claim (§4.4): pooling across anchors *"would even lead to mode collapse … if samples
from anchors representing 'turn right' and 'go straight' were optimized relative to each other, the
policy would likely collapse to the single, more common 'go straight' mode."*
⚠️ **Note the `std` divisor is PRESENT.** That is exactly the normaliser **Dr. GRPO (2503.20783,
banked) argues is a bias** toward low-variance groups, and which our own `softade` does not have
(`METHOD_LIBRARY.md` §2.1). DDv2 does not discuss it. **Do not port the divisor without deciding it.**

**Inter-anchor truncation (§4.5, Eq. 10, p.6) — "reward relative improvements, but only penalize
absolute failures".**

```
(10)  A^{k,i}_trunc  =  −1                       if  collision(τ_0^{k,i})
                        max( 0, A^{k,i} )        otherwise
```
The stated failure it repairs: with fully isolated groups *"a dangerous, colliding trajectory in another
mode might get a positive advantage if it's the 'best' sample within its own group."*

**The RL loss (§4.4, Eq. 7, p.6), with Eq. 10 substituted for `A`:**

```
(7)   L_RL = − (1/N_anchor) Σ_k (1/G) Σ_i (1/T_trunc) Σ_{t=1..T_trunc}
                  γ^{t−1} · log π_θ( τ_{t−1}^{k,i} | τ_t^{k,i} ) · A^{k,i}_trunc
```
`γ = 0.8` (Tab. 7). Because `t=1` is the last (cleanest) denoise step, `γ^{t−1}` weights the clean step
at **1.0** and decays toward the noisy end — suppl. §7: *"to down-weight the contribution of earlier
noisy denoising steps in the policy gradient"*. One scalar reward, computed on the **final clean
trajectory τ_0**, is broadcast to every step of that chain.

**The total objective (§4.4, Eq. 9, p.6):**

```
(9)   L  =  L_RL  +  λ · L_IL ,      λ ∈ (0,1) ;   Tab. 7 "BC loss weight" = 0.1
```

**The selector, stage II (§4.6, Eq. 11, p.6).** Trajectory coords as queries → deformable spatial
cross-attn on BEV → cross-attn with agent and map queries → MLP score; **two-stage coarse-to-fine**
after DriveSuprim; BCE for the score plus a margin-rank term:

```
(11)  L_rank = (1/N) Σ_{i,j} max( 0,  −sign(s_i − s_j)·(ŝ_i − ŝ_j)  +  m )
```
`s` = ground-truth continuous metric, `ŝ` = predicted, `m` a positive margin (**value not published**).

### A.2 ⛔ THE REWARD — what is stated, and what is absent

| question the commission asks | answer from the primary |
|---|---|
| the reward's **components** | ⛔ **MEASURED-ABSENCE.** Never stated. |
| the reward's **weights** | ⛔ **MEASURED-ABSENCE.** Tab. 7 has no reward row. |
| a reward **ablation** | ⛔ **MEASURED-ABSENCE.** Tab. 4/5/6/8/9 ablate noise type, intra-anchor, inter-truncation, selector design, selector impact — never the reward. |
| the **`collision` predicate** of Eq. 10 | ⛔ **MEASURED-ABSENCE.** Neither defined nor sourced. |
| what **is** stated | it is **trajectory-level**; it is *"a single reward estimate R(τ_0^{k,i}), calculated from the final clean trajectory"* broadcast across the chain (§4.4); it optimises *"non-differentiable objectives"* (§4.4); the framing is *"goal-alignment constraints to all modes"* (§1). |
| the nearest **inference**, stated as such | `HYPOTHESIS, NOT ASSERTED`: the reward is the NAVSIM **PDMS** simulator score (or its sub-metrics), because (a) that is the only trajectory-level scorer in their benchmark, (b) their sibling DIVER *does* say exactly this — *"we reuse the PDMS from Hydra-MDP … and provide online reinforcement guidance within the GRPO optimization loop"* (2507.04049 §4.3.2) — and (c) a `collision` flag is NAVSIM's `NC` sub-metric. **This is an inference from a sibling paper and is inadmissible as a DDv2 fact.** |

**Probe record for the absence** (the absence rule requires two): (1) full text extraction of all 17
pages + exhaustive term search for `reward / Reward / R( / KL / clip / ratio / entropy / collision /
goal-align / PDM`; (2) line-by-line read of suppl. §7 "Further Experiment Settings" and Tab. 7. Both
negative.

### A.3 Everything else the paper does not publish

`MEASURED-ABSENCE (full read)` for each: the group size **G** — the central GRPO knob; `N_anchor` at RL
time; `T_trunc`; `N_infer` during RL rollouts; the margin `m` of Eq. 11; whether the perception
backbone is frozen during the RL stage; the `top-k` of the coarse selector; the seed count / variance of
any reported number (**every DDv2 number is single-run and carries no interval**).

### A.4 The published budget (suppl. §7, Tab. 7, p.12)

| stage | epochs | batch | lr | wd | schedule | warmup | other |
|---|---|---|---|---|---|---|---|
| **I — RL** | **10** | 512 | 2e−4 | 1e−4 | cosine | 0.10 | BC loss weight **0.1** · min denoising std **0.04** · min log-variance std **0.10** · discount **0.8** |
| **II — mode selector** | **20** | 512 | 2e−4 | 1e−4 | cosine | 0.10 | 2 augmented trajs · aug noise std **(0.1, 0.2)** · **1 %** of the GTRS fixed vocabulary mixed in |

Hardware **8 × NVIDIA L20**; cold start = DiffusionDrive's pre-trained IL weights (§4.1, §5.2);
inference stays at **2 denoising steps** (§5.2). Selector-stage augmentation is worth noting on its own:
they harden the selector by **perturbing its own generator's outputs and injecting 1 % foreign
vocabulary** — an OOD-robustness measure for the exact component they argue is the system's weak point.

### A.5 The results, with the caveats that travel with them

| table | number | caveat that must travel |
|---|---|---|
| Tab. 1 (p.7) | **91.2 PDMS**, NC 98.3 / DAC 97.9 / TTC 94.8 / Comf 99.9 / **EP 87.5**, ResNet-34 21.8 M | protocol **P1** (NAVSIM v1 navtest PDMS). Never merged with P4. |
| Tab. 2 (p.8) | **85.5 EPDMS** | protocol **P4** ("EPDMS on navtest", single-stage). ⛔ incomparable with P1. |
| Tab. 3 (p.8) | raw fan, 20 trajectories, **pre-selector**: Div 42.3→30.3; PDMS@1 93.5→94.9; @5 84.3→91.1; **@10 75.3→84.4** | the floor moved 6.5× more than the top. **Diversity FELL** 42.3→30.3 — the method buys quality by *narrowing* the fan, which is the trade any fan-floor reward will also make (see §B.4). |
| Tab. 9 (p.13) | DD 88.1 · DD+DDv2-selector **89.1** · DDv2 **91.2** | ⭐ the selector alone is +1.0. ⚠️ And it is bought **in the hackish direction**: DD+selector moves NC **98.2→97.2** and TTC **94.7→92.2** while EP jumps **82.2→86.8** — more progress, more collisions. Only the RL arm holds NC at 98.3 *and* takes EP to 87.5. **A selection-side fix alone traded safety for progress; the generation-side fix did not.** |
| Tab. 4/5/6/8 | +0.4 / +0.9 / +0.6 / +0.2+0.2 | all on the reduced-epoch 90.1 base (§5.4). |

### A.6 ⭐ Where DDv2's knobs actually come from — the DPPO concordance

DDv2 cites DPPO (2409.00588) once, for the MDP formulation (§4.2, ref [38]). **Three further knobs are
DPPO's §4.3 "Best Practices for DPPO" verbatim in substance, uncited at the point of use** — which
matters because DPPO also publishes the *reasons* and *one more lever DDv2 did not take*.

| knob | DDv2 | DPPO §4.3 (2409.00588, p.7) |
|---|---|---|
| stochasticity for a deterministic sampler | η=1 train / η=0 eval (§4.3) | *"we set η = 1 for training (equivalent to applying DDPM) and then η = 0 for evaluation"* — same sentence structure, same reason (avoid a Dirac likelihood) |
| likelihood variance floor | *"at least 0.1 when evaluating the Gaussian likelihood"* (suppl. §7) | *"we clip σ_k to be at least 0.1 (denoted σ^prob_min) when evaluating the Gaussian likelihood … avoiding large magnitude"* |
| exploration variance floor | min denoising std **0.04** | *"clipping σ_k to a higher minimum value (denoted σ^exp_min, e.g. 0.01 − 0.1) when sampling actions helps exploration"* |
| denoising discount | `γ^{t−1}`, γ=0.8 | `γ_DENOISE^k` — *"downweighting the contribution of noisier steps"* |
| **the surrogate** | ⛔ **REINFORCE, no ratio, no clip** | ⭐ **PPO clipping variant, ε tuned to a 10–20 % clip fraction, higher ε for earlier denoising steps** |
| ⭐ **the lever DDv2 did not take** | — | **fine-tune only the last K′ denoising steps**: freeze θ, keep a copy θ_FT for the last K′ steps only — *"speeds up DPPO training and reduces GPU memory usage without sacrificing the final performance"* |

⇒ **DERIVATION:** DDv2 ≈ *DPPO's diffusion-RL mechanics* + *GRPO's group-relative advantage* +
*a hand-built truncation rule* + *an IL loss where the KL goes* − *the PPO surrogate*. Stating it that
way is what makes each piece separately adoptable or rejectable in §C.
---

## B. ⭐ THE REWARD-DESIGN QUESTION FOR US — the load-bearing section

We have **no PDMS oracle in-loop**, so §A.2's blank cannot be filled by copying. This section builds the
reward from what we actually own, and it is written **against reward hacking first and capability
second**, because every component below has a degenerate policy that maximises it.

### B.0 The five laws this section is written under — each with the measurement that earned it

| # | law | earned by |
|---|---|---|
| **L1** | **Aim at what you are paid for, not at its legible correlate.** | **C20** (`RETRACTION_LOG.md:220-230`, MEASURED, same features/folds): gating on the payoff recovers **−0.1397**; gating on the interpretable correlate recovers **−0.0383** — **aiming at the correlate costs 3.6×**. |
| **L2** | **Every rule-based term reports its FIRING FRACTION beside its effect. A term that fires on everything or nothing is degenerate and must be visible as such.** | **C19** (`RETRACTION_LOG.md:208-219`, MEASURED): a stratum result of −0.3754 firing on 22.7 % of windows is worth **−0.0852** as a policy — **4.4× over-stated bare** — and in that same stream *"two PURE-NOISE gates would have been written up as separated wins"* without the column. |
| **L3** | **Every panel carries a constant-only control that must read the no-information value exactly, a raw-input floor, and its `n` and `d`.** | `CLAUDE.md` traps preflight (MEASURED 2026-08-22): **four** distinct probe failures in one afternoon, each producing a confident publishable number, and **three of the four were caught ONLY because a control read the same value as the thing being measured.** |
| **L4** | **An echo is identified by MANIPULATION, never by an observational table.** Hold the scene fixed and move the suspect input; if the quantity does not move, it was not reading it. | **R-2026-08-03-l** (`RETRACTION_LOG.md:2269-2275`, MEASURED): the flagship's route head scored **1.0000** as a bijection of its own nav input; the label-side control passed at +0.5641 and *"was never capable of saying anything about an arm"*. Enforced in code at `taniteval.strategic_optionset.conditioning_echo_control`; **with no sweep supplied the verdict is `None` (UNTESTED), never a pass.** |
| **L5** | **A reward optimised past a point stops tracking the thing it proxies, and no regulariser is known to prevent it.** | **PUBLISHED-PRIMARY (banked, sha fea5c601…)** — DDPO §6.2 (2305.13301, p.9): optimising incompressibility → *"the model eventually stops producing semantically meaningful content, degenerating into high-frequency noise"*; optimising VLM alignment → a **typographic attack** (*"sixx ttutttas" above a picture of eight turtles*). Their conclusion: *"There is currently no general-purpose method for preventing overoptimization … existing solutions, including KL-regularization, may be empirically equivalent to early stopping. As a result … we manually identified the last checkpoint before a model began to deteriorate."* ⇒ **the RL phase MUST be stopped by a held-out gold read, never by the reward curve.** |

### B.1 The candidate reward components, one row each

`Instrument column: MEASURED (code at HEAD) unless marked. "Degenerate policy" = the policy that
maximises this term ALONE and is worse at driving.`

| # | component | what it measures | instrument that computes it today | rule-based / learned | firing fraction | **degenerate policy that maximises it** | hackability | verdict |
|---|---|---|---|---|---|---|---|---|
| **R-A** | **−distance-to-expert** (the fan's per-candidate `err`) | imitation of the single logged realisation | already computed per candidate in the live trainer (`stack/scripts/train_v6_staged.py:2423`), and on banked fans by the D-SEL probe path | rule-based | 100 % | ⛔ **collapse the fan onto the conditional mean of the expert** — a unimodal planner scores best. This is precisely the mode collapse DD's anchors and DDv2's intra-anchor grouping exist to prevent | **HIGH for a generator** (low for a scorer) | ⛔ **REJECT as `R`. ADOPT as `L_IL` (Eq. 9) only.** DDv2 puts it exactly there, and that is not an accident: a distance-to-a-single-expert reward and a multimodal generator are in direct opposition. **Also disjointness-blocked** (§B.2). |
| **R-B** | **kinematic admissibility** — is the candidate flyable from `v0` | physical realisability in the (a, κ) space | `refc_select.reachability_mask(fan, v0, accel_max=2.5, horizon_s=2.0)` (`stack/tanitad/refs/refc_select.py:178`), a re-export of `flagship_v15.reachability_mask:299`; band `v_mean ∈ [max(0, v0−a·T), v0+a·T]`, **strictly conservative — 2× wider than the kinematic bound by construction** | rule-based, label-free, deterministic | ⚠️ **MEASURED 72.08 % of REF-C-XL's fan fails it**, and deleting those is **exactly inert** (paired ΔADE **0.0000**, oracle survives 100 %, 0.00 % empty survivor sets) | ⛔ **emit constant-velocity straight-ahead** — perfectly admissible, `a=0`, `κ=0`. Which is the baseline `hold_v0_path` that our closed-loop arms already fail to beat (**C101: the CEM planner is +0.2585 m [+0.0869, +0.4309] WORSE than CV**) | **HIGH if maximisable, NIL as a gate** | ⭐ **ADOPT as the GATE — this is our replacement for DDv2's `collision` in Eq. 10's `−1` branch.** Never as a maximand. ⚠️ Its value is **not** the 72.08 %: that number says the filter is *compute, not selection* (`Project Steering/REACHABILITY_IS_COMPUTE_NOT_SELECTION_2026-08-05.md`). Its value in RL is different and untested — it constrains **generation**, and nothing in our stack has ever constrained generation. |
| **R-C** | **comfort / smoothness** — accel², jerk², yaw-rate², curvature-rate | ride quality | own kinematics, in-graph and differentiable: `four_families._seq_geometry` (`taniteval/taniteval/four_families.py:149`) gives speed/accel/along; `lateral.py` gives the lateral decomposition | rule-based, **differentiable** | 100 % | ⛔ **stand still, or drive perfectly straight at constant speed.** Comfort is maximised by not driving | **HIGH** | ⚠️ **ADOPT ONLY BOUNDED AND SATURATING.** Note how PDMS treats it: `Comf.` enters `NC×DAC×(5EP+5TTC+2C)/12` as a small **bounded** factor, and every published Comf. column sits at **99.9–100** — i.e. in the literature it functions as a *constraint that is nearly always satisfied*, **not as a lever**. Copy that role, not the term. |
| **R-D** | **progress / along-track advance** | did the car get anywhere | computable from the candidate alone; `taniteval/taniteval/progress.py:87 progress_per_window(pred, gt)` — ⚠️ note ours is **relative to GT**, not a free progress term | rule-based | 100 % | ⛔⛔ **drive as fast as possible in a straight line through everything.** The canonical hacking example | **HIGHEST** | ⛔ **REJECT for v1.** Admissible only inside a multiplicative envelope a safety factor can zero — PDMS's own structure. **We have no `NC` and no `DAC`, so we cannot build that envelope.** ⭐ And DDv2's own Tab. 9 shows the hack in miniature: their selector-only arm bought **EP +4.6** while losing **NC −1.0** and **TTC −2.5**. |
| **R-E** | **lead-gap / time-gap / min-TTC** (distance keeping) | the interaction half of LONGITUDINAL — the family Sayed made binding | ⭐ **WIRED AND ADMITTED**: `taniteval/taniteval/lead_metrics.py` (`per_step_gap:100`, `distance_keeping:125`, `distance_keeping_by_speed:255`, `paired_distance_keeping:350`), consumed by `four_families.longitudinal(..., lead=…)` (`:428`), lead block built by `lead_source.lead_block:378` / `select_lead_causal:241` from **`obstacle.offline`**. ⚠️ **`driving.py:65` records "headway / distance-keeping / TTC — REFUSAL RETIRED 2026-08-18"** — this corrects `METHOD_LIBRARY.md` R6, which still reads "offline join only" | rule-based **given the join**; needs a GT channel at TRAIN time only | ⚠️ **MUST be reported** — `four_families` already emits `DK_HAVE` / `n_time_gap` availability masks and **refuses speed-pooled reads** (`distance_keeping_by_speed`: the crawling regime dominates the window count and is where a time gap is least meaningful) | ⛔ **brake / fall back** — infinite headway is achieved by stopping. So a *maximisable clearance* is a stopping reward | **MEDIUM — controllable** | ⭐ **ADOPT-LATER (blocked on the ingest decision), and it is the ONLY component that makes this exercise more than a re-implementation.** Form: **one-sided, bounded, saturating** — DIVER's exact shape, `r_safe(τ) = −(1/T)·Σ_t 1[d_t < d_thresh] ∈ [−1, 0]` — never a clearance to maximise. **Leak status: ADMISSIBLE.** `obstacle.offline` is read at TRAIN only to build a label; the PI's ruling is *"for ground truth data … you can use both ego and other label, for inference only vision"*. ⛔ It must never reach the model's condition, and the disjointness check of §B.2 must be run on it. |
| **R-F** | **tactical-token agreement** (v7 vocab) | does the candidate execute the labelled manoeuvre | `D-LABEL-GT` v7 set (`GOALS_AND_CLAIMS.md:1566`, blob `121a8d93`, 4,719 clips, vocab v7 **52 tokens frozen, 44 emitting, 8 `NOT_YET_EXTRACTABLE`**, bands OPERATIVE [0,2] / TACTICAL [2,6] / STRATEGIC [8,30]); scoring side `four_families.tactical_from_trajectory:1200`, `tactical_goal:1110` | rule-based label | 44/52 tokens; **the 8 must be masked** or the panel reports a full-vocab number it cannot support | ⛔ same collapse as R-A, one quantisation coarser | **HIGH, and it is a NON-OBVIOUS DUPLICATE** | ⛔ **REJECT as a reward term** — the v7 tactical tokens are **derived from ego geometry**, so a reward on them is a *coarse relabelling of R-A*: it adds no information over distance-to-expert and inherits its mode-collapse hazard. ⭐ **Its correct use is the GROUPING VARIABLE.** DDv2 partitions groups by *anchor-as-intent* because that is the only intent signal they have; **we have an actual intent label**. Partitioning the intra-group advantage by v7 tactical token — not by anchor index — is DDv2's insight ported with information they did not have. That is a design contribution, and it is free of the reward hazard because a partition is not a maximand. |
| **R-G** | **the selector's own score** (`GoalDistanceScorer`) | — | `stack/tanitad/refs/refc_v3.py:340` (`d_goal_embed=cfg.d_tac`, `n_candidates=cfg.core.anchors.n_anchors=128`, `scorer_tau_m=1.0`), behind `admission_sigma_m = 0.8` (`:153`) | — | — | ⛔ **move every candidate onto the predicted goal point ĝ** — total fan collapse, i.e. the destruction of exactly the multimodality DDv2's method exists to preserve | ⛔ **CATASTROPHIC** | ⛔⛔ **FORBIDDEN BY THE COMMISSION, and independently by three of our own measurements** — see §B.2. |
| **R-H** | **a learned reward / value head on our latent** | anything | ⛔ nothing to read | learned | — | — | ⛔ **UNBOUNDED** | ⛔ **REJECT.** **LF0 MEASURED**: in **81.4 %** (encoded) / **92.3 %** (predicted) of windows where GT has a lead vehicle in the ego corridor, the decoded BEV has **no occupied cell there at all**; when it fires the read is off by **26.85 m / 42.65 m** on a 60 m grid (n = 129 labelled). A learned safety reward on `z` has nothing to read. And **L5**: DDPO's overoptimization demonstration is on a *learned* reward, and it has no known fix. |
| **R-I** | **VLM / AI judge** | semantic quality | `Sayood/tanitad-alpamayo2-augmentation` (23,644 rows / 4,729 clips × 5 tasks) | learned | — | — | ⛔ HIGH | ⛔ **REJECT-NOW.** The fused hierarchical layer ships a named **57.2 % perception hole** and G1 sign-OCR closed at **0/31 verifiable**. An AI judge built on that optimises the labeller's errors. |
| **R-J** | **human preference over trajectory pairs** | style/comfort — the one thing R-A provably cannot express | ⛔ **none exist** (R8) | — | — | — | — | **ADOPT-LATER**, blocked on `METHOD_LIBRARY.md` **B4**. ⚠️ **B4's sizing needs one correction**: it says *"there is no `corpus_overlay.py` in this repo"* — **there is**, `taniteval/taniteval/corpus_overlay.py`, **463 lines, committed** (`37ccfea`), and it is the standing 3-panel viz standard (camera projection + metric BEV + tactical/strategic HUD). B4 is therefore **cheaper** than it was priced (§E.3). |

### B.2 ⛔ THE DISJOINTNESS RULE — the reward is never the selector's score, and here is why per component

**The commission's constraint, stated as a mechanism rather than a prohibition:** if `R` is a function
of the selector's output, then the RL objective is *"please the selector"*, so (i) the selector's
errors become the generator's target and the loop is **unfalsifiable** — a planner improvement can no
longer be attributed to the generator or to the selector, which is the `--v2` conflation failure and
the C6 confound again; (ii) the selector stops being independently evaluable; (iii) it re-runs
**E-S1-0**, where scoring an object off the distribution it was trained on cost **2.8×**
(0.4728 → 1.3100; reproduced on the XL fan at 0.4714 → 1.3901, **2.95×**).

| component | could this have been computed from the selector's score? | evidence |
|---|---|---|
| R-A `−err` | ⛔ **YES, in the direction that matters.** `GoalDistanceScorer` is candidate-**independent** (it scores a candidate against the predicted goal ĝ, not against other candidates), but the *selector's supervision* — our `w_select`/`softade` term — **is** `err`. Using `−err` as `R` makes the generator's reward and the selector's training target the same object. And `METHOD_LIBRARY.md` §2.1 shows our `softade` is already the **exact-expectation GRPO gradient** over `err`: an RL term on the same quantity is a strictly higher-variance re-implementation of a term we already ship | DERIVATION over code at HEAD + §2.1 |
| R-B admissibility | ✅ **NO.** Computed from `(fan, v0)` only; the selector reads `(candidate, ĝ_tac)`. `v0` is a measured input, not a selector output | source |
| R-C comfort | ✅ **NO.** Own kinematics of the candidate | source |
| R-E lead gap | ✅ **NO** — it is computed from `obstacle.offline`, a channel the selector never sees. ⚠️ **BUT the check must still be RUN**, because refcv3's goal head and the tactical head read the same `pooled` (declared at `PREREG_D-SEL…md` §4b.3), so a *future* arm that supervised the goal head from a lead-derived label would open the back door. `RefCModel.goal_provenance()` exists precisely to be read at the point of change | source + PREREG §4b.3 |
| R-F tactical token | ⚠️ **PARTIALLY** — ĝ_tac (the selector's condition) and the v7 tactical token are both functions of the ego's future geometry. Not identical, but not disjoint either. **Second reason to reject it as a reward and keep it as a partition** | D-LABEL-GT + refc_v3 |
| R-G selector score | ⛔ **IT IS THE SCORE** | — |

⭐ **How the check is run — by MANIPULATION (L4), not by correlation.** For each reward term: hold the
window fixed, perturb / permute the selector's score vector, and assert the reward vector is
**bit-unchanged**. A correlation table cannot separate a competent reward from an echo, because *"a
competent head and an echo agree whenever the command is correct"* (R-2026-08-03-l). The pattern
already exists in code — `taniteval.strategic_optionset.conditioning_echo_control` — and its rule that
**an unsupplied sweep returns `None` (UNTESTED), never a pass**, is the rule to copy.
⛔ **And it must be a config-time refusal, not a report.** The precedent is the **C6 guard**:
`refc_train.train` refuses `--graft-route --labels v1` **at parse time** because that route target is a
deterministic function of a model input (`PREREG_D-SEL…md` §4.2). A reward whose declared provenance
set intersects the selector's input set must fail the same way.

### B.3 The composition — PDMS's STRUCTURE, none of its terms

```
                 ⎧ −1                                                    if ¬admissible(τ)      # DDv2 Eq.10 branch, R-B
   R(τ)   =      ⎨
                 ⎩ w_gap · r_gap(τ)  +  w_comf · r_comf(τ)                otherwise

   r_gap (τ)  = −(1/T) Σ_t 1[ d_lead(τ_t) < d_thresh(v0) ]   ∈ [−1, 0]    # DIVER's form; 0 where no lead — report the mask
   r_comf(τ)  = −min( 1, (jerk_rms/ j_ref)² + (yawrate_rms/ ω_ref)² )     ∈ [−1, 0]    # SATURATING by construction
   L          = L_RL(R)  +  λ · L_IL ,      λ = 0.1                        # DDv2 Eq.9; the IL loss is where R-A lives
```

**Four construction rules, each traceable to a measurement:**

1. **Every term is BOUNDED, and its bound is printed.** An unbounded term is an invitation (L5 — DDPO's
   incompressibility reward drove the model to pure noise). `r_gap` and `r_comf` are both closed in
   `[−1, 0]`; the failure branch is exactly `−1`, so no combination can exceed `0` or fall below `−1`.
2. **Feasibility is GATED, never rewarded** (R-B's degenerate policy is hold-v0, and C101 measured that
   hold-v0 already beats our search).
3. **No progress term** until an `NC`-analogue exists to zero it (R-D).
4. **The IL term is the anchor to the data manifold, and it is where distance-to-expert belongs.** DPPO's
   §5.4 finding is the mechanism: diffusion RL gives *"structured, on-manifold exploration"* — the
   multiplicative-noise trick plus the IL regulariser are the two things keeping the fan on the manifold,
   and removing either one is what turns exploration into drift.

⚠️ **`d_thresh(v0)` is the one free parameter and it is a THRESHOLD ON A TWO-ENDED QUANTITY.** Fix it
before any run, apply it identically to the arm **and to the GT**, and report the headline as a
**difference** — the rule `taniteval/taniteval/v0_antiecho.py:136-140` already states for its own
thresholds: *"Every one is PROPOSED and every one is applied to the ARM AND TO THE GT ALIKE … An
unmarked threshold in this file is a bug."*

### B.4 ⚠️ The trade this reward WILL make, stated in advance so it is not discovered as a surprise

DDv2's own Tab. 3 shows the fan **narrowing** while its floor rises: diversity **42.3 → 30.3** for
PDMS@10 **75.3 → 84.4**. Any fan-floor reward buys quality by deleting modes. ⇒ **`Div.` — or our
equivalent, the fan's pairwise spread — is a MANDATORY reported column, not an optional one**, and a
pre-committed floor on it belongs in the outcome table (§D.4). A reward that raises the floor by
collapsing the fan to one mode has reproduced vanilla diffusion (DDv2 Tab. 3: TransfuserTD, Div **0.1**,
flat 85.7 at every K) and has undone the entire reason refcv3 is a diffusion planner.

---

## C. METHOD COMPARISON FOR **OUR** SITE

**Our site, stated once:** the **generator** of refcv3 — the anchored truncated-diffusion decoder,
**128 anchors** (`pool_size` 4096), **2 denoise steps**, emitting a 128-candidate fan, selected by the
candidate-independent `GoalDistanceScorer` behind a **σ ≤ 0.8 m @ 2 s** admission gate
(`stack/tanitad/refs/refc_v3.py:153,175,340`). Sizes: small **62,930,419** params, XL **217,760,775**.
**Frozen-trunk option is mandatory** in the comparison below.

⭐ **The site distinction that decides half the table.** `METHOD_LIBRARY.md` §2.1 rejects GRPO — correctly
— **at the SELECTOR (S2)**, where our `softade` is already the exact-expectation form of the same
gradient over an enumerable fan. **DDv2 puts GRPO on the GENERATOR**, which is a site this programme has
never touched: nothing in refcv3's loss constrains what the *negative* modes emit. ⇒ *"we already have
GRPO"* is true of the selector and **false of the generator**, and conflating them would reject the one
lever DDv2 actually demonstrates.

| # | method (banked primary) | what it needs | can we supply it | integration cost | **verdict** |
|---|---|---|---|---|---|
| C1 | **Intra-anchor GRPO advantage** (DDv2 2512.07745 §4.4) | a per-candidate trajectory-level reward; a group per intent; a stochastic sampler | ⚠️ reward = §B (constrained); groups ✅ (128 anchors, or ⭐ the v7 tactical token); sampler ✅ (η=1) | **S** | ⭐ **ADOPT-NOW as the library's default advantage** — ⛔ **with `std` normalisation OFF by default.** Dr. GRPO (2503.20783, banked) argues the `std` divisor biases toward low-variance groups; our `softade` has never had one; DDv2 keeps it and does not discuss it. Ship it as a knob defaulting to `False` with a pinned test. |
| C2 | **Vanilla GRPO pooled across the whole fan** | same | ✅ | XS | ⛔ **REJECT.** DDv2 Tab. 5 MEASURED **−0.9 PDMS** for pooling (89.2 vs 90.1), and the mechanism is our own hazard: comparing a left-turn candidate against a straight candidate collapses to the majority mode. Our strategic layer is *already* a constant predictor (`route_acc_follow` 0.8031 **==** `majority_straight_rate` 0.8031, distribution {left 0, straight 1737, right 0}) — pooled advantage would push the tactical layer the same way. |
| C3 | **GRPO / RLOO at the SELECTOR (S2)** | a group of scored candidates | ✅ (R3) | S | ⛔ **REJECT — already present in exact form** (`METHOD_LIBRARY.md` §2.1: the baseline cancels identically under exact enumeration; sampling `G` only adds variance). Unchanged by this document. |
| C4 | **DDPO_SF** — score-function PG over the denoising chain (2305.13301 §4) | one gradient step per rollout batch; an analytic Gaussian likelihood per denoise step | ✅ — Eq. 5's policy is Gaussian and our fan is re-emitted every step, so there are **no stale samples** | **S** | ⭐ **ADOPT-NOW — this IS DDv2's surrogate** (§A.1 note 2). It is also the honest default *because* our fan is on-policy every step: the machinery PPO exists to justify (multiple epochs on stale data) has nothing to do here. |
| C5 | **DDPO_IS / PPO-clipped on the denoising MDP** (2305.13301; DPPO 2409.00588 §4.2) | importance ratios + clipping (+ a value head for GAE) | ✅ computable, but it buys multiple inner epochs we have not shown we need | **M** | **ADOPT-LATER (blocked on a measured need).** Take it only if rollout cost dominates. DPPO's own tuning target is a **10–20 % clip fraction**, with a **higher ε for earlier denoising steps** — record that so it is not re-derived. |
| C6 | ⭐ **DRaFT / pathwise gradient through the sampler** (2309.17400, banked) | a reward that is **differentiable in the emitted trajectory** | ✅ **for `r_comf` and for `L_IL`** — `METHOD_LIBRARY.md` §2.2: `UnicycleEmission.forward` integrates `(a, κ)` in-graph (`a = 4.0·tanh`, `κ = 0.2·tanh`, `stack/tanitad/models/v6.py:3111-3116,3735-3736`), so waypoints carry gradient. ⛔ **NOT for the indicators** (`1[¬admissible]`, `1[d<d_thresh]`) | **M** | ⭐⭐ **ADOPT-NOW as a DESIGN RULE, and it is this document's main original recommendation: SPLIT THE OBJECTIVE BY DIFFERENTIABILITY.** Pathwise for the differentiable terms (it is the exact derivative, not an estimator); score-function **only** for the indicator terms. ⚠️ **DIVER does not do this and wastes the gradient**: it calls `r_div` *"differentiable"* and `r_safe` *"a differentiable distance-based cost function"* (2507.04049 §4.3.2) and then optimises both through GRPO's score-function estimator anyway. DDv2 cannot do it — its reward is non-differentiable by construction. **We can, and it is free variance reduction.** |
| C7 | **RWR / reward-weighted regression** (AWR 1910.00177, banked) | a reward; no rollouts, no critic. `L_IL × exp(R/λ)` | ✅ | **XS** | ⭐ **ADOPT-NOW as the MANDATORY BASELINE ARM.** It is the control that says whether the policy-gradient machinery earns its place, and DDPO ran exactly this comparison and reports DDPO ahead of RWR (§6.1). If RWR matches DDPO_SF on our tiny rig, the machinery is unnecessary and we ship 3 lines instead of a module. `METHOD_LIBRARY.md` **B7** already asks for the plan loss to be kept in this shape. |
| C8 | **KL-to-reference regularisation** (DPOK 2305.16381, banked; InstructGPT 2203.02155) | a frozen reference copy of the policy | ✅ but it costs a second model in memory | S | ⛔ **REJECT in favour of the IL regulariser** — DDv2's own substitution (§A.1), and we already compute `L_IL`. ⚠️ Carry DDPO's caveat: KL regularisation *"may be empirically equivalent to early stopping"*, so neither form removes the need for the gold-metric stop (L5). |
| C9 | **Diffusion-DPO on trajectory pairs** (2311.12908, banked) | preference pairs | ⛔ **R8 absent.** A pair can be *manufactured* from the fan by thresholding `R` — but **E-OBJ-1 MEASURED that reducing our cardinal cost to a binary is separated WORSE (+0.0974 m base / +0.1670 m XL), and softening the target was worse at every τ** | M | ⛔ **REJECT-NOW (the pairwise reduction is strictly lossy on the one signal we own).** Re-opens **only** with genuine style/comfort preferences (B4), where R-A provably cannot substitute. |
| C10 | **TrajHF-style RLHF on a generative trajectory model** (2503.10434, banked) | human preference + a reward model | ⛔ R8 | XL | **ADOPT-LATER (blocked on B4).** ⚠️ And its ~94-PDMS rows exist **only with a PDMS-aligned selector** — metric-informed selection, not a deployable number. |
| C11 | **Flow-GRPO / ODE→SDE** (2505.05470, banked) | a flow-matching sampler | ⛔ we are truncated diffusion, and flow matching **loses at matched backbone** (GoalFlow 85.7 vs DD 88.1 @R34) | — | ⛔ **REJECT the method; KEEP the idea as the explanation of `η=1`** — that a deterministic sampler must be made stochastic before a policy gradient exists at all. |
| C12 | **Diffusion-QL / value-guided diffusion policy** (2208.06193, banked) | `(s, a, r, s′)` + a critic over a 60×2 action | ⛔ no `r`, no `s′`; and R-H says the latent cannot read agents | L | ⛔ **REJECT** (same block as the whole offline-RL family, `METHOD_LIBRARY.md` §3.2). |
| C13 | **Fine-tune only the last K′ denoise steps** (DPPO 2409.00588 §4.3) | a frozen θ + a trainable copy θ_FT | ✅ trivially | **XS** | ⚠️ **REJECT AS WRITTEN, KEEP THE SHAPE.** With `T_trunc = 2` there is almost nothing to subset — K′=1 is the only non-trivial choice. ⭐ **But its FROZEN/TRAINABLE split is exactly the frozen-trunk requirement:** freeze the encoder and the anchor prior, train the decoder's last denoise pass + the offset head. Record it as the default partition, not as a step-subsetting trick. |
| C14 | **Multiplicative scale-adaptive exploration noise** (DDv2 §4.3) | 2 scalars per trajectory | ✅ | XS | ⭐ **ADOPT-NOW.** MEASURED +0.4 PDMS over additive in their Tab. 4, and the mechanism transfers exactly: our waypoints share the same proximal/distal scale inconsistency, and additive per-point noise would push candidates **out of the reachability band** we are about to gate on — i.e. additive noise and R-B fight each other, multiplicative noise does not. |
| C15 | **Inter-anchor truncation** (DDv2 §4.5, Eq. 10) | a hard failure predicate | ⚠️ **the shape ports, the predicate does not.** Their `collision` needs a simulator; ours becomes `¬admissible` (R-B) and, once ingested, `d < d_thresh` (R-E) | XS | ⭐ **ADOPT-NOW (shape), with our predicate.** MEASURED +0.6 in their Tab. 6. |
| C16 | **DIVER's reward terms** (2507.04049 §4.3.2) | `D_safe(x)` obstacle-clearance map; **lane centrelines** | ⛔ **`r_LK` is dead here**: `taniteval/taniteval/driving.py:626` refuses `lane_centre_deviation` — *"no lane geometry exists; LANE_HALF_M is an assumed constant"*, and PhysicalAI-AV ships no map or lane graph (settled at five probes). `r_safe`'s *form* survives via R-E; `r_div` see below | — | ⛔ **REJECT `r_LK` and `r_PDMS`. ADOPT `r_safe`'s FORM only.** ⚠️ **REJECT `r_div` outright**: an unbounded pairwise-spread maximand (their Eq. 15) is maximised by scattering candidates arbitrarily far apart, and their own Tab. 10 shows the extra terms *reducing* Bench2Drive diversity (0.35 → 0.33 → 0.32) and DS (49.21 → 48.71). Diversity is a **reported column**, never a reward. |

---

## D. THE RECOMMENDED DEFAULT RECIPE FOR `stack/tanitad/rl/`

`⛔ NO CODE WAS WRITTEN. This is the spec Phase 2 implements, and the controls it must not ship without.`

### D.1 Module shape (names and responsibilities only)

| module | responsibility | why it is its own file |
|---|---|---|
| `rewards.py` | a **registry** of bounded terms. Each declares: `name`, `unit`, `bound` (lo, hi), `firing_predicate`, `differentiable: bool`, `provenance` (the exact set of quantities it reads) | L2 needs the firing predicate to be a first-class field; C6 needs `differentiable` to route the gradient |
| `provenance.py` | the machine-readable reward declaration, modelled on `RefCModel.goal_provenance()` — written into every run's `config.json`, asserted by tests, and **checked against the selector's input set at construction time** | §B.2; the C6 guard's parse-time refusal is the precedent |
| `advantage.py` | intra-group standardisation (`std_divisor: bool = False`), the Eq. 10 truncation, and the **group partition strategy** (`anchor` \| `tactical_token`) | C1 + R-F |
| `explore.py` | multiplicative 2-scalar noise, the `η` train/eval switch, `σ^exp_min` / `σ^prob_min` floors | C14 + §A.6 |
| `objective.py` | the **differentiability split** — pathwise for differentiable terms, score-function for indicators — plus `L = L_RL + λ L_IL` | C6 |
| `controls.py` | the mandatory controls of §D.3, importable by any arm | L3 — a control that lives in a runbook is not a control |
| `audit.py` | the reward audit that must **FLAG** a hackable reward and **REFUSE** a disjointness violation | §D.3.4 |

### D.2 The knobs, with defaults and the reason for each value

| knob | default | why this value |
|---|---|---|
| `estimator` | `ddpo_sf` | C4 — on-policy fan, no stale samples |
| `group_by` | `anchor` (v1) → `tactical_token` (v2, after the v7 masking lands) | C1/R-F |
| `advantage_std_divisor` | **`False`** | Dr. GRPO; our `softade` has never had one; **differs from DDv2 deliberately — record it as a deviation** |
| `advantage_truncate` | `True` (negatives → 0, failure → −1) | DDv2 Eq. 10, +0.6 measured |
| `failure_predicate` | `¬reachability_mask(fan, v0, accel_max=2.5, horizon_s=2.0)` | R-B; the identical function the 72.08 % was measured with — **never re-implement it** |
| `denoise_discount γ` | `0.8` | DDv2 Tab. 7 = DPPO's `γ_DENOISE`; with `T_trunc=2` it weights the clean step 1.0 and the noisy step 0.8 |
| `eta` | `1.0` train / `0.0` eval | DPPO §4.3; without it the likelihood is a Dirac |
| `sigma_exp_min` | `0.04` | DDv2 Tab. 7 (entropy-collapse floor) |
| `sigma_prob_min` | `0.10` | DPPO §4.3 `σ^prob_min` — gradient-magnitude stability |
| `lambda_il` | `0.1` | DDv2 Tab. 7 "BC loss weight"; this is where R-A lives |
| `frozen` | encoder + anchor prior frozen; decoder last denoise pass + offset head trainable | C13; the frozen-trunk requirement |
| `epochs` | tiny-rig first; **never 10 epochs at batch 512 on the first arm** | DDv2's budget is 8×L20; ours must be sized to the dev box, and L5 says the stop is a gold read, not an epoch count |
| `w_gap`, `w_comf` | **unset — pre-register per arm** | a weight chosen after seeing the result is not a weight |

### D.3 ⛔ THE CONTROLS THAT MUST SHIP WITH IT — no arm runs without all five

**D.3.1 Known-value reward tests** (unit tests, 0 GPU). Every one asserts an **exact stated value**:

| input | required reading | rationale |
|---|---|---|
| the **logged expert trajectory** | the maximum attainable `R` **on windows where it is admissible** — and the test asserts `admissible(τ_gt)` first | if the human's own trajectory does not score well under `R`, `R` is wrong. There is a precedent read for the lead half: **D-LEAD-1** (pre-registered GT-vs-CV control, 2026-08-03) measured GT beating CV on all three — Δ min-TTC **+1.7474 s** [1.5813, 1.9218], Δ headway **+0.9769 m** [0.883, 1.0758], Δ time-gap **+0.1641 s** [0.1499, 0.1786], paired episode-cluster bootstrap, **14,027 windows / 1,431 clip clusters** |
| a **constant / degenerate policy** — `taniteval/taniteval/v0_antiecho.py:202 hold_v0_path(v0, n, dt)` | a **stated** value, and **strictly below the expert's** | L3. ⚠️ It will score *well* on `r_comf` and pass `admissible` by construction — that is the point: the test documents that a stopped car is not the maximum |
| a **shuffled candidate axis** | the no-information value | L3; the pattern exists at `rank_metrics.py:112 random_ranking_ap` / `:153 assert_chance_comparator`, which raises `ComparatorNotChance` |
| **permutation equivariance** | permuting the fan permutes `R` identically | catches an index bug that would otherwise look like a learning signal |

**D.3.2 The reward-vs-selector disjointness check** — by MANIPULATION (§B.2), plus a **construction-time
refusal** when the declared `provenance` set intersects the selector's input set. Modelled on the C6
guard's parse-time refusal. **With no manipulation sweep supplied the verdict is `UNTESTED`, never a
pass** (the `conditioning_echo_control` rule).

**D.3.3 The firing-fraction column** (L2) on every rule-based term, in every panel, beside its effect —
plus the whole-set policy value where a term is conditional. `four_families` already does this for the
lead half (`DK_HAVE`, `n_time_gap`, per-speed-band `n` with a reason when unavailable); the reward
module must match it, not re-invent it.

**D.3.4 ⭐ THE DELIBERATE-REGRESSION HACKABLE REWARD — the audit's own test**

Ship a reward the audit **must** flag, and a test that fails if it does not:

```
R_hack(τ)  =  along_track_displacement(τ)          # unbounded above, no gate, fires on 100 % of windows
```
Pre-committed: `audit.py` flags it on **three** independent grounds — (a) unbounded above, (b) no
admissibility gate, (c) firing fraction 1.000 with no complement. `test_the_audit_refuses_an_unbounded_progress_reward`.
**If the audit does not flag it, the audit is broken and no arm may run.**
A **second** deliberate regression: `R = selector_score(τ)` must be **refused at construction**, not
merely flagged (`test_a_reward_equal_to_the_selector_score_is_refused_at_construction`).
And per `TanitAD_ValidateAIDesign`, the *training* ladder carries its own deliberate-regression arm — the
recommended one is **`advantage_truncate = False` with the collision branch removed**, which DDv2 Tab. 6
predicts should be **worse**; if it is not, our truncation is not doing what theirs does.

**D.3.5 The gold-metric early stop** (L5). Checkpoint frequently; select the checkpoint by a **held-out
four-family read at T1**, never by the reward curve. DDPO's authors did this by hand and said so; we
should do it by rule.

### D.4 The pre-committed outcome table — **THREE-SIDED**, per the D-SEL lesson

⛔ `PREREG_D-SEL_REFC_SELECTION_SURFACE.md` §6.3 records, in its own words, that its registered table
*"had no branch for 'separated adversely'"* and the experiment then produced exactly that
(**+0.8372 m** base / **+0.9187 m** XL, separated **worse**), so the adjudication had to be labelled
post-hoc. **Every row below is three-sided by construction.**

| endpoint | better | not separated | **worse** |
|---|---|---|---|
| ⭐ **PRIMARY — the fan FLOOR**: ADE of the K-th ranked candidate, K = 1/5/10, plus `frac_sel_2x_worse` (control: XL **0.454**, base **0.4109**) | RL is doing what DDv2 says it does (their @10: 75.3→84.4) | the reward carries no generation-side signal; **report and stop** — do not reframe as "needs more epochs" | the fan got worse where it was already worst ⇒ the reward is mis-signed or the truncation is inverted; **audit before any retry** |
| **fan diversity** (pairwise spread, DDv2 Eq. 12–13) | — | ✅ the required state | ⛔ **collapse — fails the arm even if the floor improves** (§B.4). A floor bought by collapsing to one mode has reproduced vanilla diffusion |
| **LATERAL family** (`cross_mae_m`, `heading_mae_deg`, `curvature_mae_1pm`, yaw-rate) | — | ✅ required | ⛔ **GUARD-RAIL — a separated lateral regression fails the arm** (the D-SEL rule, unchanged) |
| **LONGITUDINAL family** incl. distance-keeping, **speed-stratified**, with `anti_echo` attached | the target | — | fails |
| **TACTICAL** (`rank_acc`, `sel_gap`, per-class recall AND precision AND F1) | — | — | — |
| **STRATEGIC** | reported with its majority baseline printed beside it, always; `UNAVAILABLE` with `n` and a reason is an acceptable state, a silent drop is not | | |
| ⛔ **RED FLAG** | ADE separated better by **> 0.10 m** ⇒ **stop and audit, do not publish.** That is ~1/3 of the "92 %-irreducible" gap from a post-train; the first hypothesis is a leak (the reachability band reading a `v0` that ego-dropout withheld; the reward touching the selector), not a win | | |

**Estimator, declared before any number:** `taniteval/taniteval/ci.py:275 paired_episode_cluster_bootstrap`,
resampling unit = **episode**, `n_boot = 2000`; unpaired form `:225 episode_cluster_bootstrap`.
⛔ `overlapping_holdout_se` (`ci.py:121`) is never called. **Tier: the arm is judged at T1**; anything
computed teacher-forced is T0 and may not be called driving performance.

### D.5 Sequencing — and the two things that gate it

```
 (0) refcv3 IL cold start exists           ⛔ BLOCKER — no registry row, no checkpoint (two probes)
 (1) 0-GPU reward audit on BANKED fans     ← runs TODAY: fan_refc-{base,xl}-30k.pt + the controls of D.3
 (2) PI ruling on obstacle.offline ingest  ⛔ BLOCKER for R-E — without it R reduces to the C101 cost
 (3) v7-tiny RL arm + RWR baseline + deliberate-regression arm      (TanitAD_ValidateAIDesign)
 (4) refcv3 RL post-train arm
```

⭐ **Step (1) is unblocked and should be Phase 2's first action, not step (3).** Every control in §D.3
is computable on the already-banked REF-C fans with zero training: the known-value tests, the constant
policy (`hold_v0_path`), the shuffled control, the firing fractions of R-B and R-C, and the reward's
whole distribution over a real 128-candidate fan. **A reward that fails its own known-value tests on
banked data must never reach a GPU** — and finding that out costs nothing.
---

## E. 🔴 RECONCILIATION WITH THE CONCURRENTLY-WRITTEN `stack/tanitad/rl/`

⚠️ **This section reviews a MOVING TARGET.** While this research ran, a **concurrent stream implemented
the library** — `stack/tanitad/rl/`, **already staged** (`git status --short` reports `A ` on 6 modules
plus 6 test files). A 7th module (`refcv3_adapter.py`) appeared **mid-review**. Every claim below is
anchored to the exact bytes I read:

| file | sha256 (12) | mtime |
|---|---|---|
| `stack/tanitad/rl/advantage.py` | `12d750e4c40d` | 2026-08-29 16:26:23 |
| `stack/tanitad/rl/__init__.py` | `a1d3dd5e4929` | 16:29:09 |
| `stack/tanitad/rl/rewards.py` | `2ce3852e6699` | 16:33:57 |
| `stack/tanitad/rl/audit.py` | `b72a223a6222` | 16:34:36 |
| `stack/tanitad/rl/config.py` | `b06ff4a6ea3b` | 16:40:32 |
| `stack/tanitad/rl/posttrain.py` | `9da3ada0e273` | 16:40:47 |
| `stack/tanitad/rl/refcv3_adapter.py` | `8a42a89363ef` | 16:44:52 |

**Where it and this research independently converge** (worth recording, because independent convergence
is evidence): bounded components with a **named degenerate policy per term**; a
`FORBIDDEN_REWARD_INPUTS` frozenset enforced by `audit.assert_selector_disjoint`; `HACKABLE_WEIGHTS =
{"progress": 1.0}` as a shipped deliberate-regression arm; **Dr. GRPO-correct centre-only advantage by
default** (`advantage.grpo_advantage`, `normalize` off); DDv2's truncated inter-anchor advantage;
multiplicative exploration noise; `freeze_trunk=True`; `dpo` **refused at validate()** for want of
preference pairs. Its `_motion_gate` (`rewards.py:221-235`) is a genuinely good catch its own audit
made: without it `feasibility` + `comfort` both read **1.0** for a stationary path, handing the frozen
policy **0.70 of free reward**.

**Four review points, in severity order. Each is a claim about the code I read, verified by content.**
⭐ **E.1 was found by BOTH streams independently** and is already the implementing stream's blocking
pre-arm — it is recorded here as corroboration, not as a catch. **E.2, E.3 and E.4 are NOT addressed
anywhere in `LAUNCH_PLAN.md`** — probed by term search over that file for
`imitation / w_imitation / differenti / pathwise / DRaFT`, **0 hits each**.
⚠️ **A note on the work-package directory:** `LAUNCH_PLAN.md` and this `RESULT.md` now share
`…/Research/2026-08-29-rl-posttrain-library/`. Both streams were commissioned under D-RL-REFCV3; if the
implementing stream also writes a `RESULT.md` there, one will overwrite the other. **Flagged to the
Master Mind for a naming call.**

### E.1 ✅ CORROBORATED INDEPENDENTLY — the default reward's two safety terms are identically zero on our data

⭐ **This was found independently by BOTH streams, and the implementing stream got there first and made
it its blocking pre-arm.** `…/2026-08-29-rl-posttrain-library/LAUNCH_PLAN.md` §0 (written 16:39) opens
with **A0 — reward coverage on real refcv3 windows, 0 training steps**, and states it plainly:
`audit_reward` with no scene context returns **INCONCLUSIVE**, naming `collision`, `headway` **and**
`gt_similarity` as CONSTANT across the whole degenerate panel; its risk table records that the reward
is then *"effectively `progress + feasibility + comfort + gt_similarity`"*. Their formulation —
*"a reward whose safety terms never fire is not a safe reward — it is an absent measurement wearing
safety's name"* — is the same conclusion this research reached from C19. **I record the convergence
rather than claim the finding**, and everything below is only the residue that their plan does not
already cover.

`DEFAULT_WEIGHTS` (`rewards.py:312-319`) is `progress 0.30 · collision 1.00 · headway 0.30 ·
feasibility 0.50 · comfort 0.20 · gt_similarity 0.40`, and its own comment says *"progress is checked by
collision + headway"*.

**Both checks are inert on every real batch today:**
- `_collision` (`rewards.py:190-192`) returns **exactly 0 for every candidate** when `ctx["obstacles"]`
  is absent; `_headway` (`:210-211`) likewise when `ctx["lead_path"]` is absent.
- The only context builder, `refcv3_adapter.gt_context` (`:110-121`), copies those keys **only if they
  are in the batch**.
- They are never in the batch: the episode contract is *frames / actions / poses / episode_id /
  maneuvers* only, and `obstacle.offline` is **not ingested** — `Select-String obstacle` over
  `stack/tanitad/data/physicalai.py` returns **0**, independently recorded at
  `stack/scripts/build_obstacle_join.py:41-42`.

⇒ **On our corpus the default reward reduces to `0.30·progress + 0.50·feasibility + 0.20·comfort +
0.40·gt_similarity`, and a constant term cancels EXACTLY in the group-relative advantage** (Eq. 8
subtracts the group mean). **`progress` — the term the library itself labels "THE CANONICAL HACKABLE
REWARD" (`rewards.py:172`) — is left checked only by the motion-gated feasibility/comfort pair**, and
with `progress` bounded to `hi = 1.5` it contributes up to **0.45 of a 1.55 achievable maximum, ≈ 29 %**.

**What the plan covers:** the audit *does* see it at runtime (`audit.report_component_coverage:164`
flags components constant across candidates; `audit.py:194-195` warns to supply a context), and A0
turns that into a gate. **This is C19 in its healthy form** (`RETRACTION_LOG.md:208-219`): *a gate that
fires on everything or nothing is degenerate and must be visible as such* — in that stream **two
pure-noise gates would have been written up as separated wins** without the firing-fraction column.

**The residue, and it is a shipped-default question rather than a measurement one** *(offered to the
implementing stream — this stream writes no code)*:
1. `DEFAULT_WEIGHTS`' own comment claims *"progress is checked by collision + headway"*
   (`rewards.py:308-311`). On our corpus that sentence is **false today**, and it is the sentence a
   future reader will quote. Either the comment states the corpus condition, or the default changes.
2. **Until `obstacles` / `lead_path` are actually supplied, `progress` should ship at `0.0`**, reachable
   only by explicit override. A term whose two named checks are absent is not "weighted low" — it is
   **unchecked**, and with `progress`'s bound `hi = 1.5` it is **≈ 29 %** of the achievable reward
   (0.30×1.5 = 0.45 of a 1.55 maximum).
3. The audit's `INCONCLUSIVE` is a **verdict**, not a **refusal**. Given A0 exists, the cheap hardening
   is to make `run_posttrain` refuse to start when the reward's coverage report on the first real batch
   shows a weighted term with zero across-candidate variance — the same shape as the C6 guard's
   parse-time refusal, and the same rule the `conditioning_echo_control` already applies (**an
   unsupplied sweep is `UNTESTED`, never a pass**).

### E.2 ⛔ `gt_similarity` IS INSIDE THE REWARD *AND* THE IL LOSS IS ADDED SEPARATELY — the imitation signal is counted twice, and once in the wrong place

`rl_objective` (`posttrain.py:183-185`) adds `cfg.w_imitation * imitation_loss.mean()` **on top of** the
policy-gradient loss, while `gt_similarity` (weight **0.40**) is simultaneously inside the reward that
forms the advantage.

**DDv2 does not do this**, and its own text is the argument (§A.1, Eq. 9): `L = L_RL + λ·L_IL`, with
`λ = 0.1` — the imitation term is a **separate loss**, deliberately outside the advantage, put there
*where classic GRPO puts the KL*. Its whole §1 thesis is that IL *"is simplified in practice to
optimizing only the parameters of the single positive mode"*, which is the defect RL is supposed to
repair. **Putting distance-to-the-single-expert inside the group-relative advantage rewards the fan for
collapsing onto that expert — the exact failure the anchors and the intra-anchor grouping exist to
prevent** (§B.1 R-A, and DDv2 Tab. 3 where diversity falls 42.3 → 30.3 even *without* such a term).

⚠️ Second, independent reason: `gt_similarity` is a monotone function of `err`, which **is the
selector's own training target** in our stack, and `METHOD_LIBRARY.md` §2.1 shows our `softade` term is
already the **exact-expectation GRPO gradient over that quantity**. The library's own
`FORBIDDEN_REWARD_INPUTS` blocks the selector's *outputs*; it does not block the selector's *target*.

**Recommended:** move `gt_similarity` out of `COMPONENTS`' default vector and into the `imitation_loss`
path at `w_imitation ≈ 0.1` (DDv2's λ), keeping the component available for diagnostics. The library's
own docstring already states the correct design — *"DDv2 keeps an IL regulariser for exactly this
reason"* (`rewards.py:268-269`) — it is the **placement** that differs from what DDv2 does.

### E.3 ⚠️ ABSENCE IS SCORED INCONSISTENTLY BETWEEN THE TWO GT-DERIVED TERMS

`_collision` is bounded `[−1, 0]`, so its absent-data value **0 is its MAXIMUM** ("no information"
reads as safe). `_headway` is bounded `[0, 1]`, so its absent-data value **0 is its MINIMUM** — a window
with **no lead vehicle** is scored **exactly as if the ego were tailgating**. Today this cancels (it is
constant within every group), so the practical impact is nil; it stops cancelling the moment leads are
partially present, at which point no-lead windows carry a silent penalty. ⇒ **make absence an explicit
third state with its own mask**, the way `four_families` does (`DK_HAVE` / `n_time_gap` / a per-band
reason string), rather than a magic value inside the bound.

### E.4 ⚠️ THE OBJECTIVE DOES NOT SPLIT BY DIFFERENTIABILITY — free variance reduction is being left on the table

`rl_objective` routes **every** component through the score-function estimator. Four of the six
(`progress`, `feasibility`, `comfort`, `gt_similarity`) are **differentiable functions of the emitted
waypoints**, and our emission is differentiable to them (`METHOD_LIBRARY.md` §2.2:
`UnicycleEmission.forward` integrates `(a, κ)` in-graph, `a = 4.0·tanh`, `κ = 0.2·tanh`). For those
terms the **pathwise gradient is the exact derivative, not an estimator** (DRaFT, 2309.17400, banked).
Only `collision` and a thresholded `headway` genuinely require REINFORCE. See §C row C6 — this is the
one place where our stack can do something neither DDv2 (non-differentiable reward by construction) nor
DIVER (calls its terms differentiable, then optimises them through GRPO anyway) does.
`RewardComponent` already carries the fields to express it; it needs a `differentiable: bool` flag and
two accumulators.

### E.5 Two corrections this research owes to documents it consumed

1. ⚠️ **`METHOD_LIBRARY.md` R6 is stale.** It reads *"PARTIAL — offline join only, not in the trainer's
   batch"*. The **eval-side refusal was RETIRED on 2026-08-18** — `taniteval/taniteval/driving.py:65`:
   *"headway / distance-keeping / TTC — REFUSAL RETIRED 2026-08-18"* — and the family is wired
   (`four_families.longitudinal(..., lead=…)`, `lead_metrics.distance_keeping`,
   `lead_source.lead_block`), admitted by the pre-registered **D-LEAD-1** control, and attachable to an
   **already-banked window dump with no re-inference** via `taniteval/taniteval/dump_lead_join.py`. The
   half that is genuinely still absent is the trainer-side batch and the `obstacle.offline` **chunk
   download** (`taniteval/taniteval/pseudosim.py:691-697` names it as the one remaining blocker for
   collision).
2. ⚠️ **`METHOD_LIBRARY.md` §3.5 / §5.6 and the standing memory both assert "there is no
   `corpus_overlay.py` in this repo". There is.** `taniteval/taniteval/corpus_overlay.py`, **463 lines,
   committed** (last touched in `37ccfea`), and it is the 3-panel viz standard (camera projection +
   metric BEV inset + tactical/strategic HUD). This is the *"absence found at ONE location is not
   absence"* rule firing again, compounded by the documented Drive-mount grep under-reporting. ⇒
   **`METHOD_LIBRARY.md` B4 (the preference-collection instrument) is cheaper than it was priced**: what
   is missing is the *fan-comparison* renderer and the ranking UI, not the overlay.

---

## F. WHAT THIS DOCUMENT DOES **NOT** CLAIM

1. ⛔ **Nothing here is measured on a refcv3 fan** — no refcv3 arm exists (registry probed twice). Every
   fan number quoted is **REF-C base/XL at 2 s / 4 waypoints**; refcv3 emits **8 waypoints to 6 s**
   (`refc_v3.py:104 V3_HORIZONS = (5,10,15,20,30,40,50,60)`), and `refc_v3.py:68-71` pins the caveat in
   source: **the 72.08 % / 3.58× reachability numbers are 2 s figures and must not be quoted for the 6 s
   band.** Every absolute in §B must be re-measured at 6 s, never extrapolated.
2. ⛔ **The DDv2 reward is ABSENT, not inferred.** §A.2's PDMS reading is labelled `HYPOTHESIS` and comes
   from a sibling paper. It must not enter the registry or the paper as a DDv2 fact.
3. ⛔ **§A.6's "DDv2 = DPPO + GRPO's advantage − the surrogate" is a DERIVATION over two primaries**, not
   an experiment and not a claim the authors make.
4. ⛔ **§B's composition is a DESIGN PROPOSAL. No reward has been computed on any fan in this session.**
   The known-value tests of §D.3.1 are exactly the reason: a reward that has not read its controls is a
   hypothesis with units.
5. ⛔ **§E is a review of files that changed during the review** (a 7th module appeared mid-read). It is
   anchored by sha256 + mtime and must be re-checked against current bytes before anything is acted on.
6. ⚠️ **I did not run the test suite** and make no claim about it. `pytest -q` must be green before any
   commit, per `CLAUDE.md`.
7. ⚠️ **The two-probe absences** are: refcv3 in the registry (Grep tool + PowerShell `Select-String`);
   `obstacle` in `physicalai.py` (PowerShell `Select-String` + the independent record in
   `build_obstacle_join.py:41-42`); collision in `taniteval/` (Grep tool + PowerShell, plus the dated
   refusal string in `pseudosim.py:691-697`); the DDv2 reward (full-text term search + line-by-line
   supplementary read).
---

## G. BANKING MANIFEST

**Verification, by content:** `tools/kb_add.py --verify` →
**`verified 144 entries, 0 orphan(s), 0 problem(s)`** (2026-08-29, run under
`C:\Users\Admin\venvs\tanitad\Scripts\python.exe` — the system Python 3.14 lacks `truststore` and dies
with `CERTIFICATE_VERIFY_FAILED`). The library held **133** entries at the start of this session and
**144** at the end: **11 banked here**, 0 re-banked.

**Tool flags — as they actually are.** `kb_add.py --help` accepts
`[--local] [--key] [--title] [--url] [--tag] [--note] [--cited-by] [--reindex] [--verify]
[--reindex-orphans] [ident]`. The brief's form was correct; `--tag` takes **one** tag per invocation
(repeat the flag for more), and `--reindex-orphans` exists for the documented concurrent-write race —
it was **not needed** (0 orphans). Each call takes ~4–6 min here: the arXiv fetch runs through the TLS
proxy and the reindex rewrites a ~100 KB `LIBRARY.md` over the flapping Drive mount. **Bank
sequentially** — concurrent calls race the index.

### G.1 Banked in this session (all `--tag rl-posttrain`, all `--cited-by` this file)

| arXiv | title | size | why it is load-bearing here |
|---|---|---|---|
| **2409.00588** | Diffusion Policy Policy Optimization (DPPO) | 19,307 KB | the MDP DDv2 cites (its ref [38]) **and** the uncited source of three of DDv2's four RL hyper-parameters (§A.6) |
| **2305.13301** | Training Diffusion Models with Reinforcement Learning (DDPO) | 5,253 KB | `DDPO_SF` vs `DDPO_IS`; **L5** — the reward-overoptimization demonstration and the "no general-purpose fix; we hand-picked the last good checkpoint" statement |
| **2305.16381** | DPOK: RL for Fine-tuning Text-to-Image Diffusion | 25,029 KB | the KL-to-pretrained regulariser DDv2 replaced with an IL loss (§C row C8) |
| **2311.12908** | Diffusion Model Alignment Using DPO (Diffusion-DPO) | 12,694 KB | preference optimisation on a diffusion policy without a reward model (§C row C9) |
| **2309.17400** | Directly Fine-Tuning Diffusion Models on Differentiable Rewards (DRaFT) | 28,553 KB | ⭐ the pathwise alternative that dominates the score-function estimator on our differentiable terms — §C row C6, §E.4 |
| **2505.05470** | Flow-GRPO: Training Flow Matching Models via Online RL | 9,175 KB | the ODE→SDE conversion that explains `η = 1` (§C row C11) |
| **2210.10760** | Scaling Laws for Reward Model Overoptimization | 3,111 KB | the proxy-vs-gold divergence curve any learned reward must be sized against |
| **2209.13085** | Defining and Characterizing Reward Hacking | 829 KB | the formal statement of which reward pairs can diverge under optimisation |
| **2201.03544** | The Effects of Reward Misspecification | 2,183 KB | **phase transitions**: a *more capable* policy suddenly starts hacking a proxy — the argument for a capability-scaled reward audit |
| **2506.08052** | ReCogDrive | 11,300 KB | the other GRPO-on-diffusion E2E-AD work DDv2 positions against (its ref [29]) |
| **2208.06193** | Diffusion Policies as an Expressive Policy Class for Offline RL (Diffusion-QL) | 1,267 KB | the value-guided alternative to policy-gradient post-training (§C row C12) |

### G.2 Already banked, re-read or re-verified here (NOT re-banked)

| arXiv | role | sha256 (12) — recomputed against the banked entry this session |
|---|---|---|
| **2512.07745** | DiffusionDriveV2 — §A is a full read of these bytes | `076ce47e0b0a` ✅ (matches the origins stream's local sha, so the two documents read the same file) |
| **2507.04049** | DIVER — the reward-composition template of §B | `bf2c358071fe` ✅ |
| 2411.15139 · 2503.20783 · 2503.14476 · 2402.03300 · 2503.10434 · 2504.19580 · 1910.00177 · 2507.18071 · 2402.14740 | DiffusionDrive · Dr. GRPO · DAPO · DeepSeekMath(GRPO) · TrajHF · ARTEMIS · AWR · GSPO · RLOO | checked present in `library.json` before banking; **not re-banked** |

⭐ **Resolved from the origins stream's `bank-pending` list:** **2512.07745, 2507.04049 and 2504.19580
were all already banked** by the time this stream started (the origins agent's queued commands
completed). Nothing was left pending. `2206.08129` (TCP) is also present.

### G.3 Deliverable manifest

| artifact | path | state |
|---|---|---|
| **this document** | `TanitAD Research Lab/Architecture & Inference/Research/2026-08-29-rl-posttrain-library/RESULT.md` | repo, **staged** |
| 11 banked PDFs | `TanitAD Research Lab/Library/papers/*.pdf` | repo (**Drive mount only — one place**), sha256 in `library.json` |
| library index + metadata | `TanitAD Research Lab/Library/library.json`, `LIBRARY.md` | repo, **staged** |

⛔ **Nothing was committed and nothing was pushed.** No code was written and the library was not
created — `stack/tanitad/rl/` is a **concurrent stream's** work (§E) and this stream did not touch it.
Nothing lives only on a pod or only in a worktree. **0 GPU**; Thor was not contacted.
