# R3 — DESIGNING THE FIX BEFORE FUNDING IT: action oscillation, compounding rollout error, and whether a 59-hour run is warranted

**Date:** 2026-07-27 (Europe/Berlin; the dev box clock read 2026-07-26 23:00 UTC when this began —
the program's narrative clock runs ahead, flagged per standing practice).
**Stream:** Architecture & Inference · T_blind ladder · R3.
**Mode:** ⛔ **CPU / web only. NO POD WAS CONTACTED.** pod1 trains, pod2 runs an owed-controls job,
pod3 builds a classifier, the eval pod runs trafficsim. Nothing here was computed on a GPU; every
number of ours is read from a committed artifact or a committed log.
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written **before** the literature
synthesis and **not edited**. Its three admissible outcomes (DESIGN / DO-NOT-RUN / RE-AIM) and its
five refusal conditions S1–S5 were fixed in advance.

**Evidence classes:** `PUBLISHED (cited)` · `MEASURED (ours + path)` · `INHERITED` · `ESTIMATED` ·
`HYPOTHESIS`. **Tiers:** `PROVISIONAL` · `CONFIRMED` · `DECISION-GRADE`.

---

# 0. VERDICT

> ## ⛔⛔ **R3 AS SPECIFIED — "scheduled sampling / student forcing on the action channel" — SHOULD NOT BE RUN. Refusal condition S1 fires on evidence already in hand, and S2 and S4 fire on the literature. This is pre-registered outcome B/C, and it was declared admissible before any of it was known.**
>
> ## 🔴 **S1 — THE METRIC IS SATURATED, AND NOT BY 1 %. R3's target lever is the own→hold action-channel gap, `MEASURED` at +90 steps. A zero-training filter recovers `frac_of_ceiling_recovered = 1.011` of it. There is nothing left on that axis for a 59-hour run to win. The stale 3.2 s bar is not the problem; **re-deriving it to 11.6 s does not save R3, it kills it**, because the free filter is already AT 11.6 s.**
>
> ## 🔴 **S4 — AND THE GRADIENT R3 WOULD CREATE IS DEAD EXACTLY WHERE THE PATHOLOGY LIVES. The fed acceleration is produced by `((v − v_prev)/DT).clamp(−3, 3)` (`blindimag.py:204`). `torch.clamp` passes **zero gradient** in saturation, and saturation is **46.4 %** of the first 0.5 s. A naive R3 would backpropagate through a channel that is gradient-silent on nearly half the steps that matter. `MEASURED` (code) × `MEASURED` (`rung1_mechanism.json`).**
>
> ## ⭐⭐ **BUT THERE IS A REAL TRAINING-TIME FIX, IT IS NOT SCHEDULED SAMPLING, AND IT COSTS ≈ 0 MARGINAL GPU. Our own `inv_dyn` — a LEARNED inverse-dynamics map — is trained ONLY on REAL consecutive latents at one window position (`flagship_losses.py:379-380`) and is NEVER applied to the imagined transitions the rollout already produces. Requiring `inv_dyn(ẑ_j, ẑ_{j+1}) ≈ a_j` on those imagined transitions is the published **cycle-action-consistency** idea (ACID, arXiv 2607.02403) moved from decision time to training time. It attacks the measured pathology, it is one small MLP call per rollout step, and — unlike R3 — it **strengthens** action-conditioning instead of risking its loss.**
>
> ## ⛔ **AND THE ONE EXPERIMENT THAT MUST RUN FIRST COSTS ~20 GPU-MIN, NOT 59 HOURS: measure the inverse-dynamics cycle residual on v1's IMAGINED transitions. It separates the two hypotheses that imply OPPOSITE decisions — "our hand-written inverse map is wrong" (free fix, no run) versus "the model's imagined dynamics are wrong" (a run is justified). Today we cannot tell them apart, and the 59 hours were about to be spent on the second without checking.**

| what the brief asked | the answer |
|---|---|
| *Why do learned controllers oscillate?* | **In the literature: bang-bang optima, high actor Lipschitz constant, tanh/clamp parameterisation. In OUR loop: none of those. There is no policy network in the blind action path at all — the command is a one-tick finite difference of the model's own decoded speed, i.e. a hand-written differentiator with gain `1/DT = 10`.** The literature's *diagnosis* transfers; its *remedies* regularise a network we do not have in that position. |
| *What are the published working remedies?* | CAPS, Grad-CAPS, LipsNet, LCP, action-rate reward penalties, low-pass filters, spectral norm — §3. **All operate on a state→action network.** For us they apply to the **tactical planner** (R1's arm), not to the blind kinematic inverse (R3's arm). |
| *Which methods attack an oscillation, which a drift?* | §4 and `CITATION_TABLE.md`. **DAgger, DART, ChauffeurNet, scheduled sampling, professor forcing — all DRIFT.** Only the smoothness/Lipschitz family attacks oscillation, and only ACID-style cycle-consistency attacks *our* specific version of it. |
| *What does a training fix buy that a filter does not?* | **Command authority — and it is the ONLY thing.** `MEASURED`: the filter's 11.6 s costs the model 75 % of its command; at 75 % authority it is worth 8.5 s. `PUBLISHED` support both ways: LCP measures a filtered policy as worse than a trained-smooth one on jitter, energy **and** task return; LP-MPPI measures filtering as comparable-or-better. §6. |
| *R3 design, or don't run it?* | **Don't run R3. Run R1 (30 GPU-min) and F1 (20 GPU-min) first, and if a training-time term is added, add `R3′` (§7) to an already-planned run at ≈ 0 marginal cost — never as its own 59-hour run.** |

**Tier.** The **refusal is DECISION-GRADE** on S1: it rests on pre-registered, matched-comparator,
bootstrap-intervalled numbers from `rung1_blend_curve.json` and on nothing I derived. **The `R3′`
design is PROVISIONAL** — it is a design, not a measurement, and §8's F1 is the experiment that
would promote or kill it.

---

# 1. Pre-registration — what it fixed, and which branch fired

`PRE_REGISTRATION.md` committed three admissible outcomes and five refusal conditions **before** the
literature was read. For the record, which fired:

| condition | fired? | on what |
|---|---|---|
| **S1** metric saturation | ⛔ **YES** | `frac_of_ceiling_recovered = 1.011`, `rung1_verdict.json`. `MEASURED`, DECISION-GRADE. |
| **S2** wrong pathology | ⛔ **YES** | Every method in R3's family (scheduled sampling / DAgger line) is aimed at covariate shift on the **state** distribution. §4. `PUBLISHED`. |
| **S3** cheaper elsewhere | ⛔ **YES** | The cycle-consistency term (§7) is ≈ 0 marginal cost; the hand-written inverse map is *our code* and free to change. |
| **S4** gradient cannot reach it | ⛔ **YES** | `torch.clamp` at 46.4 % saturation. §2.4. |
| **S5** installs a capability loss | ⚠️ **UNRESOLVED, and that is itself a reason to wait** | The literature has a named, measured version of this failure ("background collapse", arXiv 2605.07514). §5.3. |

**Outcome: B (DO NOT RUN) for R3-as-named, with C (RE-AIM) supplying the replacement.** Both were
declared admissible in advance; neither is being dressed up as the other.

⚠️ **One thing changed between pre-registration and now** and it is recorded rather than folded in:
in §2 I found **four MEASURED facts about our own system that were not in the brief**, three of them
from files nobody had read for this question. They strengthen the refusal rather than weaken it, but
they were not known when the outcomes were fixed.

---

# 2. Our pathology, restated from primary sources — including four facts the brief did not carry

Everything in this section is `MEASURED` from a committed artifact or committed source. Nothing is
re-derived from a summary.

## 2.1 The brief's numbers, confirmed at their source

`…/2026-07-26-tblind-rung1/artifacts/rung1_blend_curve.json`, `rung1_interventions.json`,
`rung1_mechanism.json`, `rung1_action_signature.json`. v1 = `flagship4b-speedjerk-30k` @ 29999,
calibrated `str` (k = 20) readout, 599 windows / 596 episode clusters, paired episode-cluster
bootstrap B = 2000 seed 0.

| arm | `T_blind` | `de@2s` | `ade_0_2s` | beats CV | `T_useful@1m` | model's share of command |
|---|---:|---:|---:|---|---:|---:|
| **own kinematic** | **25** (2.5 s) | 1.8165 | 0.8710 | ⛔ 0/185 | 1.4 s | **100 %** |
| `blend0.25` | 85 (8.5 s) | 1.0736 | 0.5440 | ✅ 43/185 | 1.9 s | 75 % |
| `blend0.5` | 111 (11.1 s) | 0.7924 | 0.4010 | ✅ 72/185 | 2.2 s | 50 % |
| ⭐ `blend0.75` **best free filter** | **116** (11.6 s) | 0.6842 | 0.3437 | ✅ 81/185 | 2.3 s | **25 %** |
| `hold` (α = 1, no policy) | 115 (11.5 s) | 0.6718 | 0.3351 | ✅ 83/185 | 2.3 s | **0 %** |
| `accelclip0.3` | 62 | 1.2053 | 0.5774 | ✅ 14/185 | 1.8 s | (clipped) |
| `ema0.95` | 111 | 0.6966 | 0.3464 | ✅ 76/185 | 2.3 s | (smoothed) |
| `every2/5/20` ⛔ | **9 / 9 / 9** | 2.81 / 3.99 / 4.63 | — | 0/185 | ~1.0 s | 100 % (held) |
| `gtkin` ⚠️ privileged | 185 ⚠️ saturated | **0.4361** | **0.2552** | ✅ **179/185** | **3.0 s** | — |

Action statistics (`rung1_mechanism.json`, cumulative to the step shown, 599 windows):

| arm | step | mean \|accel\| | frac accel at ±3 clamp | jitter(accel) |
|---|---:|---:|---:|---:|
| own | 5 | **2.0582** | ⛔ **0.4641** | 3.1524 |
| own | 20 | 2.0057 | 0.4580 | 2.8091 |
| `gtkin` (same map, TRUE motion) | 5 | ✅ 0.5387 | ✅ 0.0053 | 0.2976 |

⇒ **3.8× the magnitude, 87× the saturation, from the same inverse map.** Confirmed at source.

## 2.2 ⭐ NEW MEASURED FACT #1 — the model cannot predict its own longitudinal acceleration

`MEASURED`, `taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl`, **final line, step 29999**
(620 log lines, read on the dev box, zero GPU):

| logged key | step 0 | **step 29999** |
|---|---:|---:|
| `aux_accel` (loss) | 0.4649 | **0.3869** |
| ⭐ **`aux_accel_r2`** | 0.0031 | ⭐ **0.1608** |
| `inv` (inverse-dynamics loss, REAL pairs) | 1.3067 | 0.0667 |
| `jerk` | 0.000384 | 0.000511 |

> ### 🔴 **v1 carries a dedicated 528,897-parameter longitudinal-acceleration head (`MODEL_REGISTRY.md` §1.2 `param_breakdown`; confirmed in `taniteval/results/trainlogs/v1-speedjerk_config.json`), and after 30,000 steps it explains 16.1 % of the variance of the true acceleration. The quantity the blind loop obtains by DIFFERENTIATING the decoded speed is a quantity the model demonstrably barely represents.**

**Why this matters for R3.** R3's premise is exposure bias: *the predictor has not seen its own
actions*. This number points at a different premise: *the acceleration signal is not in the latent to
begin with*. A differentiator applied to a weakly-represented quantity produces noise — which is
precisely the measured signature (mean \|accel\| 2.06 against a corpus maximum of 1.9,
`INHERITED` `closedloop.py:155`). **Exposure to that noise during training does not create the missing
signal.** `MEASURED` premises, `HYPOTHESIS` conclusion — and §8's F1 is what would settle it.

⚠️ **Two ways this could be wrong, stated because I cannot exclude them here:** (i) `aux_accel_r2` is
a **training-set, single-state** probe — accel is a two-state quantity, so a single-state head is the
wrong instrument and 0.16 may understate what a *pair* of latents carries (indeed `inv` on real
**pairs** falls to 0.0667); (ii) it is logged by the trainer, and this program has a standing rule
that **trainer logs are not eval-grade** (`operating standard` rule 1 — "v1.6 is best-in-program" was
a trainer log, ~10 % optimistic). It is quoted here as a **direction**, not as a decision input, and
nothing in §7 depends on its exact value.

## 2.3 ⭐ NEW MEASURED FACT #2 — v1's existing smoothness penalty was numerically inert

`MEASURED`, same log + `stack/tanitad/train/v4_curriculum.py:17-20`.

v1's run command carries `--jerk-weight 0.02` (`MODEL_REGISTRY.md` §1.2, exact command). Its logged
loss at step 29999 is **5.11e-4**, so its contribution to the objective is
**0.02 × 5.11e-4 = 1.02e-5** against a total loss of order 4.0 — a relative weight of **≈ 2.6e-6**.
`v4_curriculum.py`'s docstring records the same finding independently: *v1's `--jerk-weight` acted on
a 4-point, non-scored head and contributed ≤ 1e-4 of a ~4.0 loss*.

> ### ⭐ **The program has therefore NEVER tested a smoothness regulariser on the flagship. The one it nominally had was numerically zero. "We already tried smoothness" is not an available objection — and any new smoothness term must be sized by its MEASURED share of the total loss and logged as such, or it will reproduce this exact failure.**

## 2.4 ⭐ NEW MEASURED FACT #3 — the action channel is gradient-dead where it matters most

`MEASURED`, `taniteval/taniteval/blindimag.py:204` and `:498`:

```python
accel = ((v - v_prev) / DT).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)     # DT = 0.1, ACCEL_CLAMP = 3.0
```

`torch.clamp` propagates **exactly zero gradient** outside its range. The measured saturation
fraction over the first 0.5 s is **0.4641** (`rung1_mechanism.json`).

> ### 🔴 **A student-forcing objective that backpropagates through this channel receives NO gradient on 46 % of the early steps — the steps where the pathology is largest. Any R3 implementation must replace the hard clamp with a soft saturation (e.g. `A·tanh(a/A)`) or penalise the PRE-clamp magnitude. This is a design defect a naive implementation would ship with, and it is refusal condition S4.**

⚠️ **Bounded honestly:** the model's decoded speed still receives gradient from the JEPA, grounding
and rollout terms. What is dead is specifically the **action-channel feedback path R3 would create** —
the one path R3 exists to build.

## 2.5 ⭐ NEW MEASURED FACT #4 — a partial R3 already exists in committed code, and the learned inverse map is never applied to imagination

`MEASURED`, `stack/tanitad/train/flagship_losses.py`:

| line | what is there | bearing on R3 |
|---|---|---|
| `262-269` | `v2_fa_dropout`: with probability *p*, the K-step rollout's future actions are replaced by a **zero-order hold of the last observed action** | **A partial R3 is already implemented.** It substitutes `hold` — the α = 1 endpoint of the very filter that reaches the ceiling. Default off; v1's run command carries no such flag. |
| `303-312` | `v2_traj_jerk`: 3rd-difference penalty on predicted **waypoint paths**, default `0.0` | The smoothness hook exists but acts on the tactical waypoint head, not on the operative decoded motion. |
| ⭐ `379-380` | `a_hat = model.inv_dyn(states[:, -2], states[:, -1])` ; `loss_inv = (a_hat − actions[:, -2])²` | ⭐ **The learned inverse-dynamics map is trained on ONE REAL consecutive pair at ONE window position. It is NEVER applied to the K imagined transitions the rollout already computes.** This is the hole §7 fills. |

## 2.6 ⚠️ Two things in the brief's framing that the artifacts do NOT support as stated

**(a) The exponent does not discriminate drift from oscillation.** The brief offers log-log exponents
**2.098** (R² 0.995, n 19, steps 2–20) and **1.346** (R² 0.997, n 166, 20–185) as *"compounding
confirmed"*. They are admissible under the registry's R² ≥ 0.80 rule and I do not dispute them. But
an exponent of 2 is the signature of **double integration of a sustained acceleration error** just as
much as it is of Ross & Bagnell's **T²ε** compounding-error bound (`PUBLISHED`, AISTATS 2010).
⚠️ **The same number is predicted by the drift story and the oscillation story. It cannot be cited as
evidence for either.** Recorded because this program logged a retraction today for a plausible
mechanism becoming a finding.

**(b) "Near-zero-mean" is a POOLED statistic, and one arm's numbers cut against the amplitude story.**
From `rung1_action_signature.json` (all six arms, same 599-window reconstruction):

| arm | mean \|accel\| | mean \|rolling-20 mean accel\| | **LF share** | sign-flip/tick | `T_blind` |
|---|---:|---:|---:|---:|---:|
| own | 1.0917 | 0.3253 | **0.298** | 0.2886 | 25 |
| `blend0.5` | 0.4956 | 0.1960 | 0.396 | 0.1270 | **111** |
| `ema0.8` | **0.3858** | 0.3093 | 0.802 | 0.0403 | **64** |
| hold | 0.2727 | 0.2048 | 0.751 | 0.0335 | 115 |
| `gtkin` | 0.5032 | **0.4661** | **0.926** | 0.0933 | **185** |

Two readings, and both belong in the record:

* ✅ **Supporting the report's mechanism:** `gtkin` — the best arm — carries the **largest** 2-second-
  averaged acceleration (0.4661) of any arm. **A large low-frequency command is therefore not the
  damage.** What distinguishes `own` is that only **29.8 %** of its command survives 2 s of averaging,
  against 92.6 % for `gtkin`. That is the definition of a high-frequency oscillation, and it is
  measured.
* ⚠️ **Cutting against "amplitude, and nothing else":** `ema0.8` has **lower** mean \|accel\| (0.3858)
  than `blend0.5` (0.4956) and a **worse** `T_blind` (64 vs 111). ⇒ **Amplitude alone is not monotone
  with `T_blind` across families.** A second variable is doing work. `ema0.8`'s `bias_over_amplitude`
  is **0.2227** — **14× the own arm's 0.0162** — so the candidate second variable is **the bias an
  EMA introduces by lagging a signal**, which is the classical cost of a filter (phase lag) and is
  exactly what `every` does in the extreme.
* ⛔ **Escalation E-2** (§10): I cannot verify from the artifacts whether the signature file's
  per-arm rows are **fed** (post-filter) or **reconstructed raw** actions — `reconstruct_kinematic_
  actions`' docstring says *"the RAW own-kinematic action per step"*, while §4.3 of `TBLIND_RUNG1.md`
  quotes `ema0.8`'s **fed** \|accel\| as 0.344 on a different (41-window) subset. **The owning agent
  must confirm which.** Until then this reading is **PROVISIONAL** and I am not building the design
  on it.

---

# 3. THREAD 1 (priority) — action oscillation in learned controllers: causes and published remedies

## 3.1 The published causes — and which of them we have

| published cause | mechanism | do we have it? |
|---|---|---|
| **Bang-bang optima** — RL agents prefer action-space boundaries; Seyde et al., *Is Bang-Bang Control All You Need?*, NeurIPS 2021 ([arXiv 2111.02552](https://arxiv.org/abs/2111.02552)) | for many continuous-control objectives the optimal policy is genuinely at the extremes, so a Gaussian-plus-`tanh` policy collapses onto the bounds | ⛔ **NO.** There is **no reward, no policy and no optimisation** in the blind action path. The command is not chosen; it is computed. **Wrong pathology — do not cite this for us.** |
| **High Lipschitz constant of the actor** — Song et al., *LipsNet*, ICML 2023 ([PMLR v202](https://proceedings.mlr.press/v202/song23b.html)) | consecutive actions differ sharply under slight state variation because the state→action network has a large gradient norm | ⚠️ **The DIAGNOSIS transfers exactly; the OBJECT does not.** Our state→action map is `a = ((v(z_j) − v(z_{j−1}))/DT)`, a hand-written differentiator whose Lipschitz constant is **`1/DT = 10` by construction**. It is not a network and cannot be regularised as one. |
| **Discretisation / finite differencing of a continuous command** | differentiating a noisy signal amplifies its high-frequency content by ∝ f | ✅ ⭐ **THIS IS OURS**, and it is the only one of the four that is. |
| **Losses indifferent to command smoothness** — the standing motivation of the whole CAPS line | nothing in the objective penalises a jagged action sequence | ✅ **OURS**, and §2.3 shows the one penalty we nominally had contributed ≈ 2.6e-6 of the loss. |

> ### 🔴 **THE LOAD-BEARING NEGATIVE RESULT OF THIS THREAD: the entire published action-smoothness toolkit — CAPS, Grad-CAPS, LipsNet, LCP, spectral norm, action-rate rewards — regularises a *policy network* that maps state to action. In TanitAD's blind rollout there is no such network. These methods are NOT applicable to R3's arm as written. They ARE applicable to the tactical planner, which is R1's arm — and R1 costs 30 GPU-min and has not been run.**

## 3.2 The remedies, and what each actually demonstrated

**CAPS** — Mysore, Mabsout, Mancuso, Saenko, *Regularizing Action Policies for Smooth Control with
Reinforcement Learning*, ICRA 2021 ([arXiv 2012.06644](https://arxiv.org/abs/2012.06644)).
Two additive terms on the actor loss, with `D(a₁,a₂) = ‖a₁ − a₂‖₂`:

* **temporal:** `L_T = D(π_θ(s_t), π_θ(s_{t+1}))`
* **spatial:** `L_S = D(π_θ(s_t), π_θ(s̄_t))`, `s̄ ~ φ(s) = N(s, σ)`, σ set from expected measurement noise

Smoothness is scored in the **frequency domain**: `Sm = (2/n f_s) Σᵢ Mᵢ fᵢ` — amplitude-weighted mean
normalised frequency.
**Demonstrated:** ~80 % power reduction and ~96 % smoothness improvement on a real quadrotor; Gym
benchmarks (Pendulum, LunarLanderContinuous, Reacher, Ant).
**Cost, stated by the authors and worth quoting because it is the trade the brief asks about:** reward
is *"marginally worse"* on some benchmarks — a *"nominal performance hit"*.
**Asserted rather than demonstrated:** that σ should be set from measurement noise; per-environment
λ_T/λ_S are relegated to a website ablation and no universal values are given.

**Grad-CAPS** — *Gradient-based Regularization for Action Smoothness in Robotic Control with RL*, 2024
([arXiv 2407.04315](https://arxiv.org/abs/2407.04315)).
**Its critique of CAPS is the most useful single sentence in this thread for us:** CAPS makes the
policy *excessively* smooth by minimising action **changes**, enforcing a small Lipschitz constant and
**losing the ability to change action with agility**. Grad-CAPS instead penalises the **difference of
action differences** (the first derivative of the action sequence) plus a **displacement
normalisation** so the penalty is invariant to action scale.
**Why this is decisive for our design:** our command *is already a first difference of speed*.
Penalising it directly (CAPS-style) is penalising acceleration — which forbids the vehicle from
accelerating. Penalising its difference (Grad-CAPS-style) is penalising **jerk**, which does not.

**LipsNet** — Song et al., ICML 2023. Architectural rather than loss-based: Multi-dimensional Gradient
Normalization gives the actor an **adaptive** Lipschitz constant, so smoothness is bought without a
globally small `K`. Demonstrated on control benchmarks with preserved control performance.

**LCP** — *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies*, 2024
([arXiv 2410.11825](https://arxiv.org/abs/2410.11825)). Penalty
`max_π J(π) − λ_gp·E[‖∇_s log π(a|s)‖²]`. Its Table I-a is the single most useful **measured**
comparison in this whole survey and is carried into §6.

**Low-pass filtering the policy output** — standard practice in sim-to-real legged locomotion
(Butterworth, cut-off ≈ 4 Hz is the commonly reported setting). Demonstrated to work in production;
criticised in the same literature as non-differentiable, exploration-dampening and lag-inducing.

**LP-MPPI** — *Low-Pass Filtering for Efficient Model Predictive Path Integral Control*, 2025
([arXiv 2503.11717](https://arxiv.org/abs/2503.11717)). Filters the **sampled control sequences**
inside MPPI. Reports comparable-or-better task performance with lower compute, and — directly relevant
to us — states that high-frequency control noise **interacts badly with learned/approximate dynamics
models**, degrading prediction accuracy during planning. ⚠️ That last point is an assertion in the
framing, supported indirectly by their results, not a controlled measurement of the interaction.

**L2C2** — Kobayashi, *Locally Lipschitz Continuous Constraint towards Stable and Smooth RL*, 2022 —
same family, constraining local Lipschitz continuity in state **and** time.

## 3.3 ⚠️ Where the literature disagrees, stated as disagreement

| question | side A | side B |
|---|---|---|
| **Does smoothing cost task performance?** | **Yes, measurably.** LCP Table I-a: task return **26.03 ± 1.51** with LCP vs **28.87 ± 0.85** with no smoothing — a ~10 % cost. CAPS: *"marginally worse"* reward, a *"nominal performance hit"*. | **Not necessarily.** LipsNet reports smooth actions **while preserving** control performance via an adaptive rather than global Lipschitz constant. LP-MPPI reports comparable-or-better performance. |
| **Filter or train?** | **Train.** LCP Table I-a measures the low-pass-filtered policy as worse than LCP on **all three** of jitter (7.86 ± 3.00 vs 3.21 ± 0.11), energy (32.83 ± 5.50 vs 24.57 ± 1.17) and task return (24.98 ± 1.29 vs 26.03 ± 1.51). | **Filter.** LP-MPPI filters and wins on compute. The legged sim-to-real literature ships Butterworth filters in production. |
| **Penalise the action or its rate?** | **The action difference** (CAPS). | **The difference of differences** (Grad-CAPS), because CAPS over-smooths and loses agility. |

---

# 4. THREAD 2 — compounding error in learned rollouts, and the pathology each method actually attacks

## 4.1 The methods

| method | what it does | **pathology attacked** |
|---|---|---|
| **Ross & Bagnell 2010**, *Efficient Reductions for Imitation Learning* ([PMLR v9](https://proceedings.mlr.press/v9/ross10a.html)) | proves behaviour cloning's regret grows as `min{H, εH²}` — the learner errs, lands in states the expert never visited, and pays maximal cost thereafter | **DRIFT** (theory of it) |
| **DAgger** — Ross, Gordon & Bagnell 2011 | iteratively query the expert **on states the learner visits** | **DRIFT** |
| **DART** — Laskey et al., CoRL 2017 ([PMLR v78](https://proceedings.mlr.press/v78/laskey17a.html)) | inject optimised noise into the **supervisor's** demonstrations so the demonstration distribution covers the learner's; up to 280 % cheaper than DAgger and costs the supervisor only ~5 % reward during collection | **DRIFT** |
| **ChauffeurNet** — Bansal, Krizhevsky & Ogale 2018 ([arXiv 1812.03079](https://arxiv.org/abs/1812.03079)) | synthesise **perturbations to the expert's trajectory** (including collisions / off-road) plus explicit penalty losses; the canonical driving precedent | **DRIFT** |
| **Scheduled sampling** — Bengio et al. 2015 ([arXiv 1506.03099](https://arxiv.org/abs/1506.03099)) | with a ramped probability, feed the model its **own previous output** instead of ground truth | **DRIFT (exposure bias)** |
| **Huszár 2015** ([arXiv 1511.05101](https://arxiv.org/abs/1511.05101)) | shows scheduled sampling's objective is **improper** and the learning algorithm **inconsistent** — it does not recover the data distribution even in the infinite-data limit | *(the known bias problem the brief asked for)* |
| **Professor forcing** — Lamb et al., NeurIPS 2016 ([arXiv 1610.09038](https://arxiv.org/abs/1610.09038)) | adversarially match the **hidden-state distributions** of teacher-forced and free-running modes | **DRIFT** |

## 4.2 ⭐ The adjudication the brief asked for, stated plainly

> ### 🔴 **EVERY method in R3's family attacks DRIFT — a covariate shift on the STATE distribution. Not one of them attacks an oscillation. Their common premise is "the learner reaches states the expert never demonstrated and does not know what to do there." Our measured premise is different: the learner reaches states just fine; what it emits is a ±3 m/s² zero-mean oscillation produced by DIFFERENTIATING a quantity it represents at R² 0.16.**

**And in a BLIND rollout the analogy breaks a second time, structurally.** Scheduled sampling
substitutes the model's own **observation/token** for the ground-truth one. In our blind rollout the
percept is **frozen by construction** — there is no observation history to corrupt and therefore no
observation-exposure bias to fix. The only thing R3 could substitute is the **action**, and the action
is not something the model produces; it is something *our inverse map computes from what the model
produces*. **R3 is a method for a channel we do not have, in a regime where its usual object does not
exist.**

## 4.3 ⚠️ The strongest case FOR R3, given fairly

I owe the other side, and it is not nothing:

* If R3 backpropagates through the action-generation path, the gradient says *"produce a decoded speed
  sequence whose one-tick difference is a plausible command."* **That does attack the right object** —
  the decoded motion, not the predictor's tolerance. It is a genuine mechanism and I am not dismissing
  it.
* **HorizonDrive** (2026, [arXiv 2605.11596](https://arxiv.org/abs/2605.11596)) is a *driving*
  world model whose **Scheduled Rollout Recovery** is exactly this family — generate N autoregressive
  steps, then train on segments of the degraded rollout as conditioning history, with a pred-to-GT
  transition and a boundary-decay curriculum from late (drifted) positions toward early ones. It
  reports FID 13.82 / FVD 92.99 at ~20 s on nuScenes and **no reported cost** to short-clip quality or
  control fidelity. **The family works, in driving, at long horizon.**
* ⚠️ **But what it corrupts is the FRAME HISTORY, not the action.** Its exposure bias is on the
  observation channel — the channel our blind rollout has already frozen. So even the closest
  published analogue of R3 is attacking a different tensor than R3 would.

**Net:** the mechanism in the first bullet is real, and it survives — but it is a statement about the
**decoded motion**, and §7 buys it far more cheaply and without the two defects R3 carries (S4's dead
gradient and S5's capability risk).

---

# 5. THREAD 3 — world-model and latent-rollout specifics

## 5.1 What the Dreamer / TD-MPC line does about long imagination

* **Dreamer** optimises policies entirely in latent imagination and is explicitly limited by model
  error over long horizons; **TD-MPC2** is reported to reduce compounding model error and improve
  planning stability by learning a task-oriented latent rather than a reconstruction-oriented one.
* **Koopman Dreamer** (2026, [arXiv 2607.19719](https://arxiv.org/abs/2607.19719)) puts a
  **spectrally constrained** deterministic latent-dynamics core in a Dreamer-style model and reports
  improved long-horizon rollout stability and stronger closed-loop control. **This is the closest
  published thing to a principled fix for "the latent rollout is unstable": constrain the SPECTRUM of
  the dynamics operator.** ⚠️ For us this is an architecture change and is out of scope per §12.
* **ELVIS** (2026, [arXiv 2605.04709](https://arxiv.org/abs/2605.04709)) limits compounding error at
  *planning* time with an ensemble-calibrated, uncertainty-gated λ-return — an **inference-time**
  remedy, and therefore in the same class as our free filter.
* **Gradient-penalised latent dynamics** (Sonigra & Kumar, 2026,
  [arXiv 2605.23089](https://arxiv.org/abs/2605.23089)) penalises the gradient of the latent-dynamics
  network **with respect to its latent input**, to smooth imagined transitions. ⚠️ I could not confirm
  from the PDF whether it reports action-sequence smoothness separately; recorded as **UNVERIFIED** on
  that specific point.

## 5.2 ⚠️ What none of them report

**I found no work in the Dreamer/TD-MPC line that reports the smoothness of the ACTION SEQUENCE in
imagination as a diagnostic.** They report return, model error, and rollout-horizon stability. This is
a genuine gap, and it means our `rung1_action_signature.json` — sign-flip rate, clamp-saturation
fraction, rolling-mean-over-amplitude — is **instrumentation the published literature does not
standardly carry**. That is worth saying out loud: it is a small piece of novelty this program already
owns. `PUBLISHED (absence)`, PROVISIONAL — an absence claim from a literature search is exactly the
kind this program's rule 2 says to distrust, and I probed only three formulations.

## 5.3 🔴 The published version of our refusal condition S5

*Is the Future Compatible? Diagnosing Dynamic Consistency in World Action Models* (2026,
[arXiv 2605.07514](https://arxiv.org/abs/2605.07514)) defines **action–state consistency** — the
alignment between predicted actions and induced state transitions, scored in latent space — and
separates successful from failed rollouts at **AUC 0.77–0.88**.

Its central warning is ours:

> **"Background collapse"** — low-dynamics failed trajectories stay visually static or collapse toward
> background-like predictions, which are **easy to predict and therefore deceptively consistent**.
> Failed episodes receive **high** consistency scores, correlating with **reduced latent transition
> magnitude**.

> ### 🔴 **This is a PUBLISHED, MEASURED instance of the exact confound in our best arm. `hold` and `blend0.75` win by making the imagined motion nearly constant. The `TBLIND_LADDER` limitation 4 already says this in our own words — *"blending toward a constant means the arm degenerates toward the no-policy ceiling"*. A training-time objective that rewards the same thing would be rewarded by the metric and would be a capability loss. THIS is why R3 needs an action-sensitivity guard before it is ever run, and why §7 chooses a term that CANNOT be satisfied by collapse.**

## 5.4 ⭐ The paper that supplies the design

**ACID — *Action Consistency via Inverse Dynamics for Planning with World Models*** (2026,
[arXiv 2607.02403](https://arxiv.org/abs/2607.02403)).

Mechanism: use an **inverse-dynamics model as a verifier**. For each predicted transition, compare the
**conditioning action** against the action the IDM infers **from that transition**; the per-step
residual measures whether the transition is *realizable by the action that claims to have produced
it*. Their planning cost adds this to goal proximity, adaptively weighted.
**Demonstrated:** +4–14 % success across manipulation tasks; baseline performance at 10× fewer CEM
samples; 8.9–39.4 % per-step overhead.
**Its own scope statement, quoted because it bounds what I may claim:** the paper *"does not directly
address implausible or off-distribution actions."* It targets action–trajectory consistency, and it
detects action-ignoring behaviour only implicitly.
**They use it at DECISION time. §7 proposes it at TRAINING time — that is our extrapolation, not
theirs**, and it is labelled accordingly.

---

# 6. THREAD 4 — inference-time filter vs training-time fix: what does the run buy?

## 6.1 What our filter costs, MEASURED

The filter's price is **command authority**, and the dose–response measures it:

| model's share of the command | `T_blind` | `de@2s` |
|---:|---:|---:|
| 100 % (α = 0) | 2.5 s | 1.8165 |
| 75 % (α = 0.25) | 8.5 s | 1.0736 |
| 50 % (α = 0.5) | 11.1 s | 0.7924 |
| **25 % (α = 0.75)** | **11.6 s** | 0.6842 |
| 0 % (α = 1, no policy) | 11.5 s | 0.6718 |

> ### ⭐ **The 11.6 s headline costs the model 75 % of its command. Over α ∈ [0.25, 0.75] the exchange rate is ≈ 3.1 s of `T_blind` per 50 % of authority surrendered. THAT — and, on the evidence, only that — is what a training-time fix could buy: the same horizon at α = 0.**

## 6.2 What the literature says the trade is

**FOR the trained fix — and this is a measurement, not a claim.** LCP Table I-a compares a low-pass-
filtered policy against a trained-smooth one on the same task:

| | action jitter | energy | task return |
|---|---:|---:|---:|
| low-pass filtered | 7.86 ± 3.00 | 32.83 ± 5.50 | 24.98 ± 1.29 |
| **LCP (trained)** | **3.21 ± 0.11** | **24.57 ± 1.17** | **26.03 ± 1.51** |
| no smoothing | — | — | **28.87 ± 0.85** |

**Demonstrated:** the filter is worse than the trained fix on all three axes. **Asserted:** the
explanation (filters *"dampen or limit exploration"*, are *"not directly differentiable"*).
⚠️ **And the honest reading of the third row: ANY smoothing costs ~10 % task return.** The trained fix
is the cheaper smoothing, not a free one.

**AGAINST — filters are often enough.** LP-MPPI reports filtered MPPI at comparable-or-better task
performance and lower compute. Production legged sim-to-real ships fixed-cut-off Butterworth filters.

**A caveat that applies to both and that neither side controls for:** in LCP, as in our sweep, the
filter is applied to a policy **trained without knowledge that it would be filtered**. Neither
literature nor we have measured a policy *trained under its own filter*. `PUBLISHED (gap)`.

## 6.3 ⚠️ And our own measured counter-example to "filters are benign"

`ema0.8` — a filter — has **lower** amplitude than `blend0.5` yet a **worse** `T_blind` (64 vs 111),
and its `bias_over_amplitude` is **0.2227** against the own arm's **0.0162** (§2.6b). The likeliest
reading is the classical one: **an EMA buys smoothness with phase lag, and lag is a bias.** `every`
is the same trade taken to its limit and is catastrophic (9 steps). ⇒ **Not all inference-time
smoothing is equal, and our own data already shows a filter that hurts.** PROVISIONAL pending E-2.

---

# 7. ⭐⭐ THE RECOMMENDATION — refuse R3 as specified; here is what to do instead

## 7.1 The refusal, in one paragraph

**R3 = "scheduled sampling / student forcing on the action channel" should not be funded as a 59-hour
run.** Its target lever — the own→hold action-channel gap — is **fully recovered by a zero-training
filter** (`frac_of_ceiling_recovered = 1.011`). Its method family is aimed at **drift**, and our
measured pathology is an **oscillation** produced by a hand-written differentiator. Its gradient path
is **dead in 46 % of the early steps** because of a hard clamp. And the way it could most easily
"succeed" — teaching the predictor to discount its action channel — is a **published, measured
failure mode** ("background collapse") that would show up as a metric win and a capability loss.
⚠️ **Note carefully what this argument does NOT rest on:** it does not rest on the stale 3.2 s bar. The
brief is right that the bar is stale by 8.4 s. **Re-deriving it to ≥ 11.6 s does not rescue R3 — it
condemns it, because the free filter is already at 11.6 s.**

## 7.2 ⭐ `R3′` — the replacement: **imagined-transition inverse-dynamics consistency**

*Design, `PROVISIONAL`. Precise enough to implement; §8's F1 is what would promote or kill it.*

**The one-line statement.** *The learned inverse-dynamics map is already trained to recover the action
from a REAL transition. Require it to recover the action from an IMAGINED one.*

**Where it goes.** `stack/tanitad/train/flagship_losses.py`, beside `loss_inv` (line 379-380) and
inside the existing K-step rollout (`_rollout_loss`, line 270). No new module, no new parameters.

**The term.** During the K-step rollout the model already produces imagined latents
`ẑ_1 … ẑ_K` under the known conditioning actions `a_1 … a_K` (`fut_actions`, or the held-last
substitution when `v2_fa_dropout` fires):

```
L_cyc = (1/K) Σ_{j=1..K}  w_j · || inv_dyn(ẑ_{j-1}, ẑ_j) − a_j ||²_Σ        (ẑ_0 := z_t)
```

with three details that are the whole design and must not be dropped:

1. **`Σ` is a per-channel scale normaliser, not identity.** The action channels are
   `(steer_rad, accel_mps2, v/10)` and their natural scales differ by ~100× (corpus max |steer| 0.016
   rad vs max |accel| 1.9 m/s², `INHERITED` `closedloop.py:154-155`). An unweighted MSE is an
   accel-only loss by accident. Normalise each channel by its **corpus std**, measured once and
   recorded. *(This is Grad-CAPS's "displacement normalisation" idea, applied for the same reason.)*
2. **`w_j` grows with `j`.** The damage is proportional to time-in-loop (`TBLIND_RUNG1` §4.4: monotone
   in switch time, no knee), so weight late imagined transitions at least as much as early ones. A
   linear ramp `w_j = j/K` is the minimal version.
3. ⛔ **`inv_dyn` must NOT be allowed to solve this by degrading itself.** Detach the *inv_dyn
   parameters* from `L_cyc`'s gradient (or use a small separate weight), so the term pushes the
   **predictor's imagined transitions** toward action-realisability rather than pushing the inverse
   map toward laziness. Without this, the cheapest descent direction is to make `inv_dyn` constant.

**Optional second term, and I recommend it be OFF for the first run** — a **hinge on the pre-clamp
command envelope**, which is gradient-alive exactly where `torch.clamp` is dead (§2.4):

```
a_j^raw = (v̂_j − v̂_{j-1}) / DT
L_env   = mean_j ( relu(|a_j^raw| − A_corpus) )²          # A_corpus ≈ 1.9 m/s², RE-MEASURE before use
```

It penalises **nothing inside the corpus envelope**, so it cannot cost agility — the specific defect
Grad-CAPS identifies in CAPS. It is a *bound*, not a smoother.

**Weighting — the rule, not a number.** ⛔ **Do not ship a fixed λ.** §2.3 measured that v1's existing
smoothness weight contributed **2.6e-6** of the objective and was therefore untested rather than
disproven. **Set λ so each new term contributes a logged 1–5 % of total loss at step 0, and log
`term/total` every `--log-every` as a first-class metric.** A regulariser whose share is not logged
cannot be shown to have been tested.

**Cost.** ⚠️ **Not zero, and not costed as zero.** The rollout and `inv_dyn` already exist; the
marginal work is **K extra `inv_dyn` calls** (one small MLP) plus elementwise ops. `ESTIMATED` at the
same order as the ladder's R2 (~1–2 % step time). **It rides on an already-planned run. It is
explicitly NOT a reason to schedule a 59-hour run of its own.**

## 7.3 Why `R3′` and not the alternatives

| candidate | why not |
|---|---|
| CAPS temporal smoothness on the command | our command **is** the first difference of speed; penalising it penalises accelerating. Grad-CAPS's own critique. |
| Grad-CAPS on the command (= a jerk penalty) | plausible and cheap — but §2.6b shows amplitude is **not** monotone with `T_blind` across families, so a pure smoothness term may be aimed at a variable that is not the binding one. **F1 (§8) decides this.** |
| LipsNet / LCP / spectral norm on the actor | **there is no actor in this loop.** Applies to the tactical planner (R1's arm). |
| Scheduled sampling on the action (R3 as named) | §7.1. |
| `v2_fa_dropout` turned up | **already implemented** (`flagship_losses.py:262-269`) and it substitutes exactly the `hold` action the free filter already reaches. Cheap to try; **cannot exceed the filter's ceiling by construction.** |

> ### ⭐ **The structural reason `R3′` is the right shape: a filter shrinks the command, and pays in AUTHORITY. Cycle-consistency makes the imagined dynamics OBEY the command, and pays in nothing — a model that ignores its action channel CANNOT satisfy it. It is the only candidate here whose failure mode is not "background collapse".**

## 7.4 What `R3′` must beat, and by how much

⛔ **NOT `T_blind`.** Setting a `T_blind` bar for a training-time fix repeats the ladder's own
**C-STALE-BAR** class one level up: the metric is saturated by a free filter, so any bar on it is
either already cleared or unreachable.

✅ **The bar, comparator-free and filter-proof — three conditions, all required:**

| # | condition | value | why this one |
|---|---|---|---|
| **B1** | at **α = 0** (no filter, **full command authority**), `beats-CV` | **≥ 43/185** | `blend0.25`'s value — the trained model at 100 % authority must match today's model at 75 % authority. Comparator-free: the CV floor is pure kinematics and has no readout. |
| **B2** | at **α = 0**, `T_useful@1m` | **≥ 1.9 s** | same row, same reason. |
| **B3** | **`R3′` + best filter** vs **v1 + best filter**, `de@2s` | separated improvement on **0.6842** | ⛔ **the anti-self-deception bar.** You can filter the new model too. If R3′ only matches the filter, the run bought a number we already had. |
| **KILL** | **action-sensitivity** `S` (§7.5) | must not fall > 30 % | §5.3 background collapse. |

**Headroom check — is the bar reachable?** `gtkin` (the same map fed TRUE motion) reaches
`beats-CV 179/185`, `T_useful@1m 3.0 s`, `de@2s 0.4361`. The **entire filter family** tops out at
**83/185** and **2.3 s** (its α = 1 endpoint — a no-policy controller). ⇒ **44–55 % of the achievable
capability gap is provably out of reach of ANY action filter, and that is the target.** `MEASURED`.

## 7.5 The action-sensitivity guard (pre-register it, don't add it later)

```
S = mean_windows | de(rollout | a) − de(rollout | a with accel channel × 0.5) |  at 2 s
```

Two arms, one sweep, computable from the existing per-window machinery. **If `S` falls by more than
30 % against v1, the model has learned to discount its action channel and the run is a capability
loss whatever `T_blind` says.** This is the direct operationalisation of §5.3 and of the ACID
verifier, and it is the guard that R3-as-named did not have.

---

# 8. ⭐ THE CHEAP PRE-RUN FALSIFIERS — hours, on the existing checkpoint

**All three are eval-only on v1 `flagship4b-speedjerk-30k` @ 29999. None requires training. F3 needs
no GPU at all.** ⚠️ None may run today — every pod is occupied.

## F1 ⭐⭐ — the inverse-dynamics cycle residual on IMAGINED transitions · **~20 GPU-min** · **THE DISCRIMINATING ONE**

Roll v1 blind for K = 185 under the `own_kinematic` action, and at every step compute
`r_j = inv_dyn(ẑ_{j−1}, ẑ_j) − a_j` — the learned inverse map against the action actually fed.
Report `|r_j|` per channel vs `j`, and the same on the `hold` and `gtkin` arms as controls.

**Pre-registered predictions, committed now, all three of which are informative:**

| outcome | reading | decision |
|---|---|---|
| `\|r_j\|` is **small and flat** on the own arm | the model's imagined transitions **already agree** with the actions that produced them — the fault is entirely in the **hand-written** inverse map, which is *our code* | ⛔ **`R3′` is inert. Do not add it. Fix the map (free) and run R1.** |
| `\|r_j\|` is **large and GROWS with `j`** on the own arm but not on `gtkin` | the model's imagined dynamics **drift away from action-realisability as the rollout proceeds** — exactly what `L_cyc` penalises, and the term has a real target | ✅ **`R3′` is justified; add it to the next planned run.** |
| `\|r_j\|` is large on **every** arm including `gtkin` | `inv_dyn` itself is unreliable off the real-latent manifold — the 9.4× manifold gap the ladder's **R4** names | ⇒ **R4 first.** `R3′` would be optimising against a broken verifier. |

**Why this is the right experiment:** the first and second outcomes imply **opposite decisions** — a
free code fix versus a funded training term — and today we cannot tell them apart. That is exactly
the situation the ladder exists to resolve before spending GPU-days.

## F2 — the stranded aux-accel head · **~20 GPU-min** · zero training

v1 carries a trained **528,897-parameter** longitudinal-acceleration head
(`MODEL_REGISTRY.md` §1.2 `param_breakdown`; `v1-speedjerk_config.json`). Feed **its** output as the
accel command instead of the finite difference. A **direct regression** has no differentiator gain by
construction.

| outcome | reading |
|---|---|
| ≥ **62 steps** (`accelclip0.3`'s value) | the "differentiator, not the model" diagnosis is confirmed **and a zero-training deployable improvement exists** |
| ≈ the `hold` value (115) | the head is a mean-predictor; consistent with `aux_accel_r2 = 0.1608` |
| ≤ **25** | the diagnosis is wrong — the speed decode is **biased**, not noisy. **This would reinstate a training-time fix.** |

🔴 ⚠️ **BLOCKER, and it is an escalation (E-1):** the head's **construction code is not in the
committed repo.** `stack/scripts/train_flagship4b.py` has no `aux_accel` and no `jerk_weight` argument
(`MODEL_REGISTRY.md` §1.2 *Code state*: the v1 run used a pod-side trainer that was never committed).
The nearest committed implementation is REF-B's (`stack/tanitad/refs/refb.py:127,415`). **Whether v1's
weights are loadable, and whether the head applies to an *imagined* latent, must be checked before F2
is costed.** UNVERIFIED.

## F3 — zero GPU, minutes · the band-split arithmetic

From the already-committed `perwindow/action_audit_K185.pt` (dense `fed_actions` for 10 arms) and
`rung1_perwindow_compact.pt` (dense `de` [599 × 185] for 58 arms), compute the **power spectrum** of
each arm's fed accel and its **band-limited coherence** with the resulting position error.

**The arithmetic that motivates it, and its self-refutation — both recorded.** A zero-mean sinusoidal
acceleration of amplitude `A` at frequency `f` double-integrates to a displacement of only
`A/(2πf)²`: with `A = 1.09 m/s²` and the measured sign-flip rate 0.2886/tick (⇒ `f ≈ 1.4 Hz`), that is
**≈ 0.013 m** — three orders of magnitude below the measured 2 s penalty of **1.145 m**. `ESTIMATED`.
⇒ **The oscillation cannot be damaging us through kinematic integration; it must be damaging us
through the PREDICTOR's nonlinear response to an off-distribution input.**
⚠️ **And the obvious follow-on hypothesis — "then it is the low-frequency content that hurts" — is
already REFUTED by our own artifact:** `gtkin`, the *best* arm, carries the **largest** 2 s-averaged
acceleration of any arm (0.4661 vs own's 0.3253). §2.6b. **A hypothesis that survived one arithmetic
check and died on the second is exactly why F3 is listed as an arithmetic exercise and not as a
finding.**

---

# 9. 🔴 WHAT WOULD MAKE R3 NOT WORTH RUNNING — the explicit statement the brief asked for

**Already true, on evidence in hand:**

1. ⛔ **The lever is exhausted.** The action-channel gap R3 targets (own 25 → hold 115) is recovered at
   **101.1 %** by one line of inference code. `MEASURED`, DECISION-GRADE. **This alone is sufficient.**
2. ⛔ **The remaining headroom is not R3's to take.** The 115 → 185 gap is the difference between a
   *held* action and an action derived from **true motion** — information the model does not have. It
   is a motion-prediction gap, and the ladder already has two ≈ 0-cost rungs aimed at it (**R2**: train
   the readout at the horizon it is read at; **R4**: train the step readout on real latent pairs). **A
   59-hour run should not precede two free ones.**
3. ⛔ **The method family is aimed at drift.** §4.2.
4. ⛔ **The gradient is dead where the pathology is worst.** §2.4.

**Would become true on any of these:**

5. **F1 returns "small and flat"** → the fault is our inverse map, a free fix, and no training term is
   warranted.
6. **F2 returns ≥ 62 steps** → a zero-training improvement exists and the marginal value of a trained
   one collapses.
7. **The action-sensitivity guard cannot be held** (§7.5) → any `T_blind` gain is a controllability
   loss.
8. **R1 lands inside its predicted 6–12 s band** → the deployed tactical planner does not have this
   pathology at all, and the whole line was measuring a property of the *kinematic-inverse ablation*
   rather than of anything we ship. ⭐ **R1 costs 30 GPU-min and is on the record as a pre-registered
   prediction (`TBLIND_RUNG1` §7.2). It should run before anything else in this stream.**

**And the converse — what would make R3-proper worth reconsidering.** If F3's band-split (or F1)
returned *"neither the low-frequency nor the high-frequency component alone reproduces the damage;
only the joint off-distribution action vector does"*, then the pathology genuinely is a
**predictor-input-distribution** problem, exposure is the right instrument, and **HorizonDrive's
Scheduled Rollout Recovery becomes the design to copy** — with a soft saturation replacing the clamp
(§2.4) and the §7.5 guard attached. **I am recording that path so a future agent does not have to
re-derive it, and so this refusal is falsifiable.**

---

# 10. 🔴 ESCALATIONS — in the headline, not written into a README

**E-1. A 528,897-parameter trained head in the DEPLOYED checkpoint has no construction code in git.**
v1's `aux_accel` head (and its `--jerk-weight` path) came from a pod-side trainer that was never
committed (`MODEL_REGISTRY.md` §1.2 *Code state*). This is the operating standard's rule-3 failure
class — *an artifact that exists in only one place* — except the place is a **checkpoint**, so it has
been invisible to the pod-drift check. **It also blocks F2.** A decision is owed: reconstruct the head
from REF-B's implementation and verify a strict load, or record the head as unreachable.

**E-2. `rung1_action_signature.json` is ambiguous about fed-vs-raw actions, and a headline depends on
it.** `TBLIND_RUNG1` §0 states *"what binds is the oscillation's AMPLITUDE, and nothing else"*. §2.6b
here shows `ema0.8` has **lower** amplitude than `blend0.5` and a **worse** `T_blind` (64 vs 111) in
that same file. If the file's rows are **raw reconstructed** rather than **fed** actions, the
comparison is not like-for-like and my caveat dissolves; if they are **fed**, the headline needs the
qualifier *"amplitude, at matched lag"*. **The owning agent (tblind-rung1) must state which.** Cheap
to answer, and it is load-bearing for whether a smoothness term is the right instrument.

**E-3. R3's bar is stale — and re-deriving it does not save R3.** The brief already carries this. The
correction this document adds: **the re-derived bar of ≥ 11.6 s is one a free filter already meets, so
the correct action is not to re-aim R3's bar but to re-aim R3.** The bar that *is* winnable is the
comparator-free one at α = 0 (§7.4), and it is a different metric.

**E-4. R1 should run before anything else in this stream, and it is 30 GPU-min.** Its pre-registered
prediction (6–12 s) is on the record and unscored. If the deployed tactical planner already lands in
that band, most of this analysis describes an ablation rather than a product. **No agent currently
owns it.**

---

# 11. Limitations, stated plainly

1. ⛔ **Nothing here was measured by me on a GPU.** Every one of our numbers is read from a committed
   artifact or a committed log. Four are new *readings* (§2.2–2.5), not new *experiments*.
2. ⚠️ **`aux_accel_r2 = 0.1608` is a TRAINER-LOG number** and this program's rule 1 says trainer logs
   are not eval-grade. It is used as a direction, never as a decision input.
3. ⚠️ **§2.6b's amplitude caveat is PROVISIONAL** pending E-2. I could not resolve fed-vs-raw from the
   artifacts.
4. ⚠️ **The `R3′` design is a design, not a result.** It has never been run. Its own falsifier (F1's
   first outcome) would make it inert.
5. ⚠️ **The literature survey is a survey.** I read abstracts, HTML full texts and one PMLR table
   directly; two PDFs (CAPS, Grad-CAPS, the gradient-penalty paper) resisted extraction and their
   details come from the ar5iv rendering or from secondary summaries — flagged inline. **No λ values
   from CAPS are quoted, because the paper does not publish them in the body.**
6. ⚠️ **One absence claim (§5.2) rests on three search formulations**, which is below this program's
   two-probe bar for a *strong* absence claim. It is marked PROVISIONAL for that reason.
7. ⛔ **Everything about `T_blind` inherits the ladder's own limitations** — episode-initial windows,
   extrapolation past 0.4 s, `gtkin` saturating at the sweep terminus (a lower bound), no safety
   metric. Those are not re-litigated here.
8. ⛔ **This is a drift analysis.** PhysicalAI-AV ships no map, lane graph or agent boxes.

---

# 12. Deliverable manifest

**Everything is in the repo working tree and STAGED (`git add`). Nothing was committed or pushed.**
Path: `TanitAD Research Hub/Architecture & Inference/Research/2026-07-27-action-oscillation-r3-design/`

| artifact | what it is | where it lives | only one place? |
|---|---|---|---|
| `PRE_REGISTRATION.md` | the three admissible outcomes (DESIGN / DO-NOT-RUN / RE-AIM) and the five refusal conditions S1–S5, **written before the literature synthesis**, unedited | **repo** | no |
| `R3_DESIGN_RESEARCH.md` | this document — pre-registration → four threads → the refusal + the `R3′` design → the falsifiers | **repo** | no |
| `CITATION_TABLE.md` | every cited work with its link, mapped to **which pathology it attacks** (oscillation / drift / both / neither), what it demonstrated vs asserted, and its applicability to us | **repo** | no |

**Nothing lives in only one place.** No pod was contacted; no file was produced on a pod; no
checkpoint was read. **No file outside this folder was created or modified**, so neither `stack` nor
`taniteval` can change behaviour and neither suite was re-run (nothing to re-run).

**Cost.** Dev box only. ~20 web fetches/searches, ~12 repo reads, 2 CPU-seconds of Python to read two
committed logs. **Zero GPU.**

---

# 13. Reproduction

```
# The two new MEASURED facts, on the dev box, zero GPU, ~2 s:
python -c "import json; d=json.load(open('taniteval/results/trainlogs/v1-speedjerk_config.json',encoding='utf-8')); print(d['jerk_weight'], d['aux_accel'], d['param_breakdown']['aux_accel'])"
python -c "import json; L=open('taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl',encoding='utf-8').read().strip().split('\n'); r=json.loads(L[-1]); print({k:r[k] for k in ('step','aux_accel','aux_accel_r2','jerk','inv')})"

# The code facts:
#   taniteval/taniteval/blindimag.py:204,498   accel = ((v - v_prev)/DT).clamp(-3, 3)   <- zero grad in saturation
#   stack/tanitad/train/flagship_losses.py:262-269   v2_fa_dropout   (hold-last substitution, default off)
#   stack/tanitad/train/flagship_losses.py:303-312   v2_traj_jerk    (waypoint 3rd-diff, default 0.0)
#   stack/tanitad/train/flagship_losses.py:379-380   loss_inv        (REAL pairs only, window position -2)
#   stack/scripts/train_flagship4b.py                NO aux_accel, NO jerk_weight  <- E-1
```
