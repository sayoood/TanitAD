<title>RESULT — the leaders kept our action channel and fixed the objective instead; and 34B buys nothing open-loop</title>

# RESULT — FRONTIER SCAN 2026-08-31

`TanitAD Research Lab · Frontier Scan · first pass under DAILY_RESEARCH_CHARTER.md`
`0 GPU · 0 spend · Thor untouched · no pod touched.`
`Every query and every EMPTY search: raw/search_log.md. Nothing is claimed here that is not sourced there.`
`⚠️ Executed by the Master Mind, not the Lab agent — the 2026-08-31 Lab pass had already run and produced no Band-B or Band-C coverage at all (charter §1, D1–D4).`

---

## 0. ⭐⭐⭐ THE TWO ANSWERS

### A. The frontier kept our "defective" action channel and attacked the OBJECTIVE instead.

Today's Architecture package concluded from mechanism (`2605.25313`) that **P2(b) is aimed at the
channel while the binding constraint is the target**. That conclusion now has independent support
from a completely different direction — **what the two current NAVSIM leaders actually built**:

| system | its action channel | is it realised motion (our defect)? |
|---|---|---|
| **DriveFuture** (`2605.09701`) | *"a normalized differential representation (Δx, Δy, sin θ, cos θ), a linear projection, and a temporal positional embedding"* | ⛔ **YES — our exact channel** |
| **DriveWorld-VLA** (`2602.06521`) | *"Historical ego actions are serialized into natural language prompts and concatenated with textual instructions"* | ⛔ **YES** |

And DriveWorld-VLA's target is teacher-forced in precisely the sense that makes the
action-invariant solution available: *"we reuse the Stage 1 encoding pipeline to obtain the
corresponding ground truth (GT) BEV latent representation."*

⇒ **Neither leader solved this with a better command.** Both kept realised motion and spent their
design budget on **how the future latent is conditioned and scheduled**. That is the same verdict
our own mechanism paper reached, arrived at by building rather than by proving.

### B. Scaling 10B → 34B buys **~0.5 %** open-loop, and is not cleanly separated closed-loop either.

NVIDIA's own two Alpamayo cards, same task, same protocol (MEASURED by them, `PUBLISHED` model cards):

| model | params | LingoQA Lingo-Judge | AlpaSim closed-loop | open-loop minADE₆ @ 6.4 s |
|---|---:|---:|---:|---:|
| **Alpamayo-1.5** | 8.2 B backbone + 2.3 B expert ≈ **10.5 B** | 74.2 | **1.37 ± 0.10** | **0.916 m** |
| **Alpamayo2-Super** | 32 B backbone + 2.3 B expert ≈ **34.3 B** | 79.2 | **1.50 ± 0.13** (913 scenarios) | **0.911 m** (1434 samples) |

- **Open-loop trajectory: 0.916 → 0.911 m for 3.3× the parameters — 0.5 %.**
- **Closed-loop: the stated intervals OVERLAP** (1.37 + 0.10 = 1.47 vs 1.50 − 0.13 = 1.37). On
  their own published uncertainties the two models are **not cleanly separated on either
  trajectory metric.**
- **The language head is what moves: 74.2 → 79.2 LingoQA.**

⇒ **The reasoning scales; the driving does not.** This is the strongest external evidence the
sub-300M thesis has received, and it independently corroborates `EVAL_DOCTRINE`'s T0/T1 split from
the opposite direction: the open-loop number is so insensitive it cannot even separate a 3.3×
parameter gap.

⛔ **THE CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS: `minADE₆` is BEST-OF-6, not ADE.** It is
**not comparable** to our flagship's 0.452 m `fwd_ade`, and any table that puts them in one column
is wrong. Quoted here only to compare Alpamayo **to Alpamayo**.

---

## 1. ⭐⭐⭐ F1 — LatentAlign: a published, named cure for the train/inference latent gap

**MEASURED-by-them / PUBLISHED (lib `2605.09701`, full text).** DriveFuture conditions its planner
on the **ground-truth** future latent during training and on a **predicted** one at inference. Their
fix, verbatim:

> *"a sigmoid-based annealing schedule that smoothly transitions the planning condition from the
> grounded future latent to the self-predicted forecast over the course of training"*

with `α(e) = 1 − σ(β(e − e₀))`. Their own limitation statement is honest about the residual:
*"DriveFuture depends on predicted future latents during inference, so inaccurate future prediction
may affect planning in uncertain scenarios."*

**Ablation (their Table 4, NAVSIM-v2 navhard ablation setting):** adding the GT-grounding term moves
EPDMS **32.1 → 34.6 (+2.5)**; the hyper-parameter sweep (Table 5) shows the schedule's start point
is sharp — `e₀` 0.75 / **0.83** / 0.95 → 29.2 / **34.6** / 28.9 EPDMS, i.e. **±0.12 in the schedule
costs ~5.5 EPDMS.**

⚠️ ⛔ **UNRESOLVED DISCREPANCY, FLAGGED NOT PAPERED OVER.** The paper's headline navhard number is
**EPDMS 55.5** (*"1st on the NAVSIM-v2 navhard leaderboard"*, April 2026) while **every row of its
own navhard ablation table reads 30.9–34.6**. I could not reconcile these from the text. ⇒ **No
DriveFuture navhard number may be quoted in a comparability table until this is resolved.** The
ablation *deltas* are usable (same table, same setting); the *levels* are not.

| dimension | |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ direct lever — V7 gate **P2/P5**, register **MM-E10** (our h=1 action/scene ratio 0.00408/0.00416/0.00595) |
| **CONSEQUENCE** | We have a measured action-invariance defect **5–7× deeper than the published one** and **no scheduled-sampling mechanism at all**. This is a named, ablated, cheap mechanism aimed exactly at it. |
| **COMBINATION** | Composes with today's Architecture finding rather than competing: that package says *the target is the problem*; LatentAlign is *a way to stop depending on the grounded target without changing the encoder*. It also composes with **F2** below — the two leaders used **different** mitigations, so they are separable arms, not one recipe. |
| **CHANCES / RISKS** | **Chance:** a schedule is a training-knob, not an architecture change — it fits the v7-tiny ladder and needs no retrain of the trunk. **Risk:** the schedule is **sharply tuned** (±0.12 → 5.5 EPDMS) and their setting is a diffusion planner on BEV anchors, not our latent predictor — the operating point almost certainly does not transfer, so it must be swept, not copied. **Risk 2:** the headline/ablation discrepancy means the magnitude of the win is not established. |
| **EXPERIMENT** | **v7-tiny ladder, 2 arms + 2 controls.** Arm A: fixed teacher-forced target (current). Arm B: sigmoid anneal of the target from grounded → self-predicted, `e₀` swept over 3 values. **Pre-committed read: the action/scene ratio at h=1.** SUPPORTED if Arm B raises it ≥3× (0.004 → ≥0.012) at equal or better ADE. **Controls that must read known values:** a constant-predictor control (must read the no-information value) and a raw-pixel floor, per the four-failure ridge-probe rule. |

---

## 2. ⭐⭐⭐ F2 — Action-source MIXTURE with a learned NULL token: a second, independent mitigation

**PUBLISHED (lib `2605.09701`, full text).** DriveFuture does *not* feed the ground-truth trajectory
every step. Training draws the ego-trajectory input from three sources with probabilities
`{p_gt, p_kin, p_∅}`:

1. **ground-truth** trajectory,
2. **kinematic extrapolation** — constant-acceleration from `(vx, vy, ax, ay)`,
3. **a learned NULL token.**

⭐ **This is action-dropout with a physics-based decoy, and it is structurally an anti-echo device.**
A model that can be handed a null or a *wrong-but-plausible* action cannot learn to treat the action
as a free copy of the answer.

| dimension | |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ direct lever — same gate rows as F1, plus the **hold-action floor** doctrine our 2026-08-30 novelty claim rests on |
| **CONSEQUENCE** | We have argued the hold-action floor is *our* instrument. **A published system already trains against a null-action arm.** That does not refute our novelty claim — theirs is a *training* mixture, ours an *evaluation* floor — but it is adjacent enough that the paper must state the distinction rather than claim the space empty. ⇒ **Feeds directly into the at-risk novelty claim the Opponent package flagged today.** |
| **COMBINATION** | The kinematic decoy is the sharper half for us: our action channel correlates with realised motion at **r = 0.9988**, so a constant-acceleration surrogate is *almost* the true action — it is a **hard negative**, exactly the regime where an echo-solution breaks and a genuine dynamics model does not. |
| **CHANCES / RISKS** | **Chance:** implementable as a dataloader change; zero architecture risk; testable at tiny scale. **Risk:** with `p_∅` too high the planner degrades into an unconditional prior — their sweep does not report `p` values, so **we have the mechanism but not the operating point**, and must find it ourselves. **Risk 2:** on a corpus where the action is nearly deterministic in the state, the kinematic decoy may be *too* close to GT to teach anything — measure the decoy/GT distance before concluding. |
| **EXPERIMENT** | Add `{p_gt, p_kin, p_∅}` sampling to the v7-tiny trainer; sweep `p_∅ ∈ {0, 0.1, 0.3}`. **Pre-committed read:** hold-action ADE **must degrade** relative to true-action ADE — if the two stay equal, the arm has not removed the echo and the change is refuted. Report the decoy/GT action distance alongside, or the null result is uninterpretable. |

---

## 3. ⭐⭐⭐ F3 — NVIDIA Alpamayo: an open reasoning-VLA family trained on OUR corpus, unregistered by this programme

**PUBLISHED (NVIDIA newsroom + HF model cards, fetched today).**

| | Alpamayo-1.5-10B | Alpamayo2-Super |
|---|---|---|
| architecture | Cosmos-Reason2 **8.2 B** backbone + **2.3 B** action expert, *"diffusion-based trajectory decoder"* | **32 B** VLM backbone + **2.3 B** action expert |
| licence | **OpenMDW-1.1** — *"ready for non-commercial use. Commercial licensing available upon request."* | **OpenMDW-1.1**, *"permits commercial deployment globally"*; source Apache-2.0 |
| inputs | image/video (multi-camera, multi-timestep) · text (*"user commands and navigation guidance"*) · **egomotion history** | same + 3D translation, 9D rotation |
| training data | ⭐ **`nvidia/PhysicalAI-Autonomous-Vehicles` AND `-NuRec`** + 16 named public QA/driving sets + internal data; **3.0 M** Chain-of-Causation traces; **RL post-trained** | ~**115 000 h** video; **3.7 M** CoC traces |

⭐⭐ **`AlpaSim` — which this programme already runs — is part of this release.** We adopted the
simulator and never registered the model family shipped beside it, trained on the same corpus we
train on.

⚠️ **Two corrections this finding forced, both recorded:**
- The search summary dated the announcement **2026-08-10**; the newsroom primary says **CES,
  2026-01-05**. The primary wins — this is a **7-month** error that would have read as breaking news.
- The two cards state **different commercial terms under the same licence name**. Immaterial for us
  (research use is sanctioned per the PI's binding rule) but **material the moment anything ships**.

| dimension | |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — Opponent Analysis, Benchmarks (G3), **and** today's Architecture package: *"navigation guidance"* is a **text-channel command input**, the exact question that package was asking |
| **CONSEQUENCE** | Three at once: **(a)** a directly-comparable opponent trained on our corpus, so a like-for-like comparison is possible in a way NAVSIM leaders never allowed; **(b)** a candidate **auto-labeller** — 3.0 M CoC traces were themselves *"VLM-generated"*, which is the untimed-CoT problem our `D-LABEL-GT` row is stuck on; **(c)** their command is **natural-language navigation guidance**, i.e. a *declared* instruction, not recoverable from the path — the property our Data package prized in L2D's `turn_signal`. |
| **COMBINATION** | ⛔ **The admissibility check binds here.** Alpamayo consumes **egomotion history at inference**. Under the binding vision-only rule, an Alpamayo-style ego-input arm is **inadmissible as our deployable arm** — it is usable as an **opponent** and as an **offline labeller** (where ego is explicitly allowed), never as our inference recipe. The two-arm design already in the backlog (`P1 ⛔PI`) is exactly the right frame. |
| **CHANCES / RISKS** | **Chance:** a same-corpus opponent removes the corpus confound from every comparison we have wanted to make. **Risk:** ⛔ **10.5 B and 34 B do not fit Thor's budget** — today's own measurement (latency linear in K, 203 ms at 37.8 M) says a 10 B rollout is orders outside the 100 ms loop; this is a **reference and a teacher, never a deployment target**. **Risk 2:** using their CoC traces as labels imports their errors as ground truth — needs the same alignability gating the `D-LABEL-GT` row is already arguing about. |
| **EXPERIMENT** | **0-GPU first:** pull the two model cards' eval configs and determine whether their `minADE₆ @ 6.4 s` split intersects our parity split (`e438721ae894`, skip-hash `f09e44db`). **Pre-committed read:** if the episode sets are disjoint, **no comparison is admissible** and we say so; if they intersect, we have our first same-corpus external baseline. |

---

## 4. ⭐⭐ F4 — Band-B transfer: the Transformer→hybrid distillation budget

**⚠️ RELAYED — search-summary level, primary `2601.22156` NOT fetched, NOT banked.** The claim:
converting a pretrained transformer into a hybrid linear-attention model *"reduc[es] the total
training budget to approximately 25 % relative to training a comparable model from scratch"*, via
staged long-context training plus teacher-guided distillation.

| dimension | |
|---|---|
| **RELEVANCE** | ⭐⭐ informs a live decision — our fleet is fixed (Thor + one 4060 + gated pods); **arms-per-GPU-day is the binding constraint on the whole programme**, not model quality |
| **CONSEQUENCE** | If a *conversion* recipe reaches parity at 25 % of from-scratch cost, then our ladder economics change: we could afford ~4× the arms per calendar week at fixed spend. |
| **COMBINATION** | Composes with today's rollout-depth measurement (latency **linear in K**, no amortisation): a linear-attention / SSM predictor has **O(1) state** rather than a growing KV cache, which is the one architectural family that could make a **deep strategic rollout (K≈300)** fit a budget that today's measurement says nothing else fits. |
| **CHANCES / RISKS** | ⛔ **Risk first: this is RELAYED and may not decide a GPU-day** (charter §5). It is an LLM-text result; our sequence is 6–60 latent steps, not 128 k tokens — the regime where linear attention *wins* may be entirely absent at our length. **Chance:** even a partial transfer attacks the one constraint (rollout depth) that today's Deployment package says is currently unsatisfiable at any scale we measured. |
| **EXPERIMENT** | **Verification first, then a probe.** (1) Fetch and bank `2601.22156`; re-read the 25 % claim in its own table. (2) Only then: on the 4060, benchmark an SSM/linear-attention predictor block against our transformer block at K ∈ {8, 60, 300}, batch 1, fp16 — the same harness as today's `rollout_depth_bench.py`, whose controls already read their known values. **Pre-committed read:** the SSM wins only if ms/step is **flat in K AND lower in absolute terms at K=60**; flat-but-slower is a refutation. |

---

## 5. ⭐⭐ F5 — EB-JEPA: a single-GPU, action-conditioned JEPA rig the Lab can actually run

**PUBLISHED (lib `2602.03604`, abstract-only — declared).** FAIR's `eb_jepa`
(`github.com/facebookresearch/eb_jepa`, CC BY 4.0) covers image SSL, video temporal modelling, and
**action-conditioned world models**; *"Each example is designed for single-GPU training within a few
hours"*; the action-conditioned example is a Two-Rooms navigation task at **97 % planning success**;
and *"Comprehensive ablations reveal the critical importance of each regularization component for
preventing representation collapse."*

| dimension | |
|---|---|
| **RELEVANCE** | ⭐⭐ — Lab instrument, and it touches the live **VICReg-placement / anti-collapse** backlog row (#9) |
| **CONSEQUENCE** | Our anti-collapse design space is currently explored **only inside our own trainer**, where every arm costs a v7-tiny run. An external, published rig with published ablations is a **control** — the thing the ridge-probe post-mortem said we lack. |
| **COMBINATION** | Directly serves backlog row 9 (VICReg placement) and row 4 (EMA τ-ramp): if their ablation isolates each regulariser on a task where the answer is known, we can check **our** instrument reproduces **their** result before trusting it on our data. |
| **CHANCES / RISKS** | **Chance:** runs on the dev-box 4060, zero spend, zero Thor contention. **Risk:** ⚠️ **abstract-only — I have not confirmed which regularisers it implements.** The search summary named SIGReg, but that came from a *different* result in the same query and **must not be attributed to this library**. Two-Rooms is a gridworld; a collapse result there may not transfer to 256×640 driving video at all. |
| **EXPERIMENT** | Clone, run the action-conditioned example on the 4060 **as-is**, and check the published 97 % reproduces. **Pre-committed read:** if it does not reproduce, the rig is not usable as a control and we stop there — that is a cheap, decisive gate before any port to our data. |

---

## 6. ⭐ F6 — GoalFlow: goal point + flow matching at ONE denoising step

**⚠️ RELAYED / abstract-level (`2503.05689`, CVPR 2025).** Goal-point-conditioned flow matching for
multimodal trajectories; **PDMS 90.3**; *"requires only a single denoising step"*.

| dimension | |
|---|---|
| **RELEVANCE** | ⭐ context, ⭐⭐ for the goal path — the binding **goal-input** directive prefers a *predicted geometric goal point*, which is exactly this input |
| **CONSEQUENCE** | Supplies a published architecture for the goal-point lever our own directive already sanctioned, and its **one-step** decoding is the only generative planner shape compatible with today's measured latency budget. |
| **COMBINATION** | ⛔ **The goal-input admissibility check applies unchanged:** whatever goal signal we adopt must be **information-disjoint from the situation classifier's output**. A flow-matching planner does not change that test — it only changes what consumes the goal. |
| **CHANCES / RISKS** | **Chance:** multimodal futures without the blurry-mean failure a distortion metric rewards (the Opponent package's `2606.12987` finding today). **Risk:** RELAYED and pre-2026; the PDMS 90.3 is NAVSIM-v1, which the same Opponent package showed has **moved scoring basis twice** — the number may not be comparable to anything current. |
| **EXPERIMENT** | Defer. Bank and read the primary first; it is behind F1/F2 in priority because it changes the *planner*, and our measured blocker is upstream in the *objective*. |

---

## 7. ⛔ What this pass did NOT establish

1. **`2601.22156` (25 % budget) and `2503.05689` (GoalFlow) are RELAYED.** Neither may decide a GPU-day.
2. **`2604.01349` PI-JEPA is WITHDRAWN** — its numbers are inadmissible and are recorded in the search log purely so a later pass does not re-find and re-cite them.
3. **The DriveFuture navhard level discrepancy (55.5 vs 30.9–34.6) is unresolved.**
4. **Four named empty searches** (E1–E4 in `raw/search_log.md`), each **single-probe** — none licenses an absence claim.
5. **Band-B tracks B10 (semantic search), B12 (memory/long-context) and B13 (3D/occupancy) were NOT scanned** — this pass covered 19 of 22 tracks. Recorded in `TRACKS.md` as stale, first in the next rotation.

## 8. Recommendations (≤3, per contract)

| # | recommendation | why it is first |
|---|---|---|
| **R1** | **Register a two-arm anti-echo training experiment on the v7-tiny ladder: LatentAlign annealing (F1) × action-source mixture with null token (F2).** Both are training-knobs, both are published-and-ablated, both aim at the deepest measured defect we have. | Our action/scene ratio is **5–7× worse** than the published defect and we run **neither** mitigation. This is the cheapest large lever on the board. |
| **R2** | **Open an Alpamayo opponent row: same-corpus comparability check first (0 GPU), then decide labeller-vs-baseline.** | It trains on **our corpus**, we already run its simulator, and we never registered it. The comparability check is free and gates everything downstream. |
| **R3** | **Clone EB-JEPA on the 4060 and reproduce its published action-conditioned result as an external control** before spending another v7-tiny arm on anti-collapse. | The ridge-probe post-mortem's rule: a panel without a control that reads a known value manufactures results. This is a ready-made control. |

`Escalations, backlog motions and the deliverable manifest are in the LAB-RUN summary.`
