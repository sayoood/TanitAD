<title>Frontier scan 2026-09-09 - findings</title>

# FRONTIER SCAN - 2026-09-09 (sixth pass under the charter)

`TanitAD Research Lab - daily pass under DAILY_RESEARCH_CHARTER.md and its v2 AMENDMENT.`
`Evidence class on every claim. Retrieval date 2026-09-09 for every external source.`

---

## 0. The scheduling gap this pass did not cause and does not backfill

**MEASURED from the directory tree:** `Frontier Scan/Daily/` holds `2026-08-31`, `2026-09-01`,
`2026-09-02`, `2026-09-05` and nothing else. **2026-09-03, 09-04, 09-06, 09-07 and 09-08 produced no
frontier scan at all.** The 2026-09-05 pass escalated the first two; **the count is now five days**,
and 09-06 to 09-08 are new. 2026-09-07 did produce eleven domain packages, so the Lab was working -
what is missing is specifically the Band-D / Band-B / Band-C coverage this artifact carries.

**Escalated to the Master Mind as a scheduling failure**, distinct from the GS-7 budget tension.

---

## 1. Findings, most consequential first

### F1 - The first action-identifiability instrument our FROZEN TRUNK does not block

`Track A2. lib 2608.12939 "Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency". FULL TEXT. Class PUBLISHED.`

Three consecutive passes found an action-identifiability method blocked by the same structural
mismatch: Delta-JEPA's LDAD is **encoder-shaping** (2026-09-02, GS-2 REVISED), ATM's AITS is
**encoder-side** (2026-09-05, recorded as *"SECOND consecutive action-identifiability objective
blocked by our frozen trunk"*). **This one is different, and the difference is the finding.**

ACPC is a **diagnostic computed on an already-trained, frozen model**, not a training objective.
Verbatim from the paper: *"Let F-theta denote the frozen action-conditioned predictor"*, and it is
explicitly contrasted with MWM, which *"enforces action-conditioned rollout consistency during
training"*. The encoder is frozen throughout the diagnostic computation.

Its criterion is bisimulation, stated verbatim as: *"two observations should be treated as the same
state only when their action-conditioned consequences agree."* That is a formal statement of exactly
what our `actdiv` probe gropes at informally.

| quantity | definition (verbatim, transcribed to ASCII) | what it reads |
|---|---|---|
| **ACPC** | `ACPC_H(h, h~, a) = || G_a(E(h)) - G_a(E(h~)) ||_2` over a weighted H-step rollout | how far a clean history and a **perturbed** view of it diverge **under the same actions** |
| **IR** (Invariance Radius) | `IR_q(theta) = Q_q({R_i})`, the q-quantile of normalised ACPC over anchors | clean-vs-perturbed rollout spread |
| **SR** (Separation Rate) | `SR_q,delta(theta) = mean over pairs of 1[ D_diff > IR_q(theta) + delta ]` | whether **different states stay distinguishable after rollout** |

They **prove** the divergence bounds the perturbation-induced change in multi-step prediction error
**and planner cost** - a latent-space diagnostic with a proved link to planner regret.

**The measured collapse signature is the part we can use immediately.** Verbatim: *"we train LeWM on
TwoRoom with four SIGReg weights... the representation collapses... its SR falls to **0.066**, compared
with **0.967-0.984** for nonzero SIGReg."* That is a published healthy band and a published collapse
value for a separation statistic - which is precisely the quantity our rank/drift instruments lack an
operating point for.

**Protocol, MEASURED from the paper:** rollout horizon H=8 for diagnostics (H=5 for the CEM planning
analysis), 100 logged histories as IR anchors, 5 Gaussian-noise draws per anchor, 100 episodes per
evaluation seed across 3 seeds. Reported effects: 55.9 +/- 4.7 % reduction in prediction-error MAE and
15.2 +/- 2.0 % reduction in CEM selection regret.

**It is also a THIRD independent critic of LeWM**, after Delta-JEPA and ATM. Debt D-9 says we cite
LeWM's action-insensitivity only through its critics. **That is now three critics and still no primary
read** - the debt does not weaken with repetition, it hardens.

**What it costs us to adopt:** *"No random baselines or chance-level controls provided."* Their
diagnostic ships without a constant-only control and without a raw-input floor. Under `CLAUDE.md`'s
probe rule that is exactly the shape that manufactured four false results in one afternoon on our own
ridge probe. **If we port IR/SR we supply the controls ourselves, or the port inherits their gap.**

### F2 - Debt D-10 resolved in substance: navhard and navtest EPDMS are TWO DISJOINT POPULATIONS

`Tracks A5 / C4. lib 2603.24581 Latent-WAM, FULL TEXT results section, plus the NAVSIM maintainers' own guidance. Class PUBLISHED + PUBLISHED-BLOG.`

D-10 asked why three independent 2026 tables cap navhard at **<= 45.0 EPDMS** while our records carry
**55.5 / 56.3**. Today's read supplies the mechanism and a third cluster.

| cluster | values (EPDMS) | source |
|---|---|---|
| navhard | 23.1-45.0 (GuideFlow), 38.0 (IDOL), 36.9 (RAP-DINO) | three independent 2026 tables, read 2026-09-05 |
| **"NAVSIM v2", 12k evaluation scenarios** | **89.3** Latent-WAM, **86.1** DriveVLA-W0, **85.1** Epona, **84.8** World4Drive | `2603.24581` Table 1, read today |
| our recorded numbers | 55.5 (DriveFuture), 56.3 (DrivoR) | `LEADERBOARD.md` / register, provenance unstamped |

Two facts settle it. First, Latent-WAM evaluates on *"12k evaluation scenarios"* - **navhard is 450
Stage-1 plus 5,462 Stage-2 observations**, so this is not navhard. Second, and decisively, **the paper
never states a split and never claims the official leaderboard**; the retrieval found no
leaderboard reference and the numbers read as self-evaluated.

The benchmark's own maintainers name this exact failure: they *"discourage the use of self-reported and
unofficial 'NAVSIM v2' benchmark splits"* and direct submitters to the leaderboard *"to ensure both
consistency and visibility"*.

**Verdict: the <=45 vs 84-89 spread is not a capability ranking, it is two different measurement
bases wearing one metric name.** Our 55.5/56.3 sit between the clusters and match neither, which means
their provenance is unknown rather than merely disputed.

**This is rule V-5's named corruption path, caught before it entered a table.**

### F3 - The efficiency wedge has a unit error waiting in it (row 32)

`Track A1. lib 2603.24581, verbatim. Class PUBLISHED.`

*"The model contains **104M parameters at inference time**. During training, an additional EMA encoder
is introduced for self-supervised learning, **bringing the total to 191M, of which only 104M are
trainable**."*

Backlog row 32 restates our efficiency wedge as needing to beat *"~40 M, not 32 B"*. **Both the 40M
DrivoR figure (unverified - backlog L-15) and this 104M figure are inference-time counts on models
whose training-time footprint is larger.** Comparing either against a TanitAD parameter count that is
quoted training-inclusive is the same class of error as `minADE_6` versus our `fwd_ade`.

**Rule proposed: every parameter count entering a comparability table carries INFERENCE-TIME or
TRAINING-TIME as an explicit stamp, exactly as every metric carries its eval tier.**

### F4 - A zero-label clip-value signal, with the controls we demand, and it corrects our own record on EMA

`Tracks B10 / B9. lib 2606.28383 "Zero-Label Driving Scenario Complexity Detection via JEPA". FULL TEXT. Class PUBLISHED.`

This is the day's transfer item and it lands directly on backlog row 22 (H-DATA-1: *does frozen-feature
leverage predict a clip's value for WM training?*).

**The score is the JEPA's own temporal prediction error**: `s = || z_hat_tgt - z_tgt ||_2`. No labels at
training time and none in defining the score. Validated against nuPlan's 67 ground-truth scenario tags;
downstream anomaly detection reaches **AP 0.512 against a 0.436 chance baseline**. Highest-surprise
tags are *"stopping at traffic light without lead"* (2.008) and *"starting unprotected cross turn"*
(1.978); lowest are lane-following and stationary traffic.

**Their four ablations are the control discipline `CLAUDE.md` mandates, and two of them are load-bearing
for us:**

| ablation | Spearman rho | what it tells us |
|---|---|---|
| shuffled scores | -0.126 | the no-information value, read correctly |
| **random encoder** | **0.024** | the untrained floor reads ~zero - the control our ridge probe lacked |
| **constant-velocity baseline** | **0.314** | **a large fraction of "complexity" is recoverable by CONSTANT VELOCITY** |
| **no-EMA training** | **44-fold collapse** in discriminative power | EMA is not a refinement here, it is the mechanism |

**The constant-velocity row is the warning.** 0.314 from constant velocity is our CTRV-floor problem in
a new costume: a signal that looks like learned scene understanding is substantially kinematic.
**Any TanitAD clip-value score must be reported against a constant-velocity floor or it is not
interpretable.**

**The no-EMA row corrects one of our own records.** Backlog row 4 (MM-E1) states that *"our fixed
tau=0.996 has no published operating point - every EMA-teacher line ramps tau."* **This paper uses a
FIXED alpha = 0.996, does not ramp it, and measures a 44-fold collapse without it.** So a published
fixed-0.996 operating point does exist, in a JEPA, on driving data.

**Scope it honestly - this does NOT license adopting fixed tau.** It is a 1,289,130-parameter model on
**structured agent-state vectors** from nuPlan mini (1,322 scenarios, 50 timesteps at 10 Hz), not on
pixels and not at our scale. What dies is only the sentence *"no published operating point exists"*.
The design question row 4 asks - ramp or not, at our scale, on our modality - is untouched.

### F5 - RLIR would optimise the exact leak GS-1 tells us to audit first

`Track B5. lib 2509.23958 "Reinforcement Learning with Inverse Rewards for World Model Post-training". ABSTRACT-ONLY, declared. Class PUBLISHED abstract-only.`

RLIR *"derives verifiable reward signals by **recovering input actions from generated videos using an
Inverse Dynamics Model**"*, mapping video to a low-dimensional action space so GRPO has an objective
reward. Reported: **5-10 % gains in action-following**, up to 10 % on visual quality.

It is genuinely attractive for our RL fan-safety regression, and **we should not take it yet.**

**GS-1 (from Delta-JEPA, 2026-09-02) says an inverse decoder reading endpoint pairs can succeed on
*"action-correlated cues"* absorbed into the endpoint, *"without requiring the model to represent the
actual transition"*.** RLIR's reward is precisely an inverse decoder scored on the model's own output.
On our action channel - realised motion, `r = 0.9988` - that reward is **plausibly satisfiable by the
leak rather than by action-following**. Adopting it before the GS-1 audit would spend RL compute
optimising our best-documented confound.

**Contrast that sharpens F1:** ACPC compares **two rollouts under the same actions** and never reads the
real `z_{t+1}` endpoint, so it does **not** carry the GS-1 leak by construction. **One instrument is
leak-resistant and one is leak-prone, and they arrived on the same day.**

### F6 - Band D: Alpamayo 1.5 adjudicated (claims A15-1 to A15-4)

Full seven-step adjudication in `../../../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`. Headline:
**the teacher-student concession is theirs, unprompted, and it is our thesis** - while the counter-search
found two independent sources holding that **verbose natural-language chain-of-thought, Alpamayo's
exact output form, is the wrong representation for real-time control.**

### F7 - D-4: routes four and five failed, and the failure changed character

`Track C3. Class: retrieval failure, not evidence.`

Route 4 was a **direct primary PDF URL on unece.org**, which returned **HTTP 403**. Route 5 (jasic.org,
a different domain) fetched 2.2 MB that the extractor returned as a **CIDFont stream with no text
layer**. **Five routes, still unread.** The distinction that matters for the PI: this is no longer
*"we cannot find the document"* - we can, and our fetcher is blocked from it. **D-4 is a
retrieval-channel problem.** Someone with a browser can settle it in two minutes.

### F8 - B3 is empty at a term-varied second probe

Recorded under FS5-6's stricter rule: the second probe **varied the term**, not the phrasing, and B3
still returns only LLM token-serving work. Detail in `raw/search_log.md`.

---

## 2. Coverage this pass - stated, not smoothed

**22 of 22 tracks SCANNED. 6 DEEP, of which 3 FULL TEXT. Band D: 1 item, 4 claims adjudicated.**

**Breadth is restored.** The pre-committed rotation item 1 from 2026-09-05 read: *"B3, B1, B2, B4, B6,
B8, B9, B10, B11, C3 SCAN - breadth restoration, and it stays item 1 until a Master-Mind ruling on GS-7
changes the mandate. Four days is enough."* **All ten were scanned today**, ending a four-pass breadth
failure, and the Band-B minimum was met with deep reads on B10/B9 (`2606.28383`) and B5 (`2509.23958`)
plus the B13/A3 sweep.

**GS-7 is therefore answered empirically rather than by ruling:** breadth and depth coexisted in one
budget this pass because **two of the three full-text reads came from SCAN hits on stale tracks**
(B10 and A5), not from the debt list. **The tension in amendment 8.1 is real but it is not
unconditional** - depth-first ordering starves breadth only when depth is spent on standing debts.
**Recommendation to the Master Mind: keep 8.1's priority order and add one clause - the day's
full-text budget is spent on the best hit AVAILABLE, whether it came from a debt or from a scan.**

**What this pass did NOT do, plainly:** it did not read the Kairos curation section at three routes,
and it left D-4, D-7, D-8 and D-9 standing.

⭐ **Banking DID complete.** The mount recovered late in the pass; eight primaries were banked or
citation-updated (Library 487 -> 491 entries) and **debt D-12 is discharged**. ⛔ **The V-1 number is the
uncomfortable part: 4 of 8 were already banked, and `2603.24581` Latent-WAM - the paper carrying this
pass's single most consequential finding - was one of them, unread.** Full detail in `raw/search_log.md`.

---

## 3. Ledger appends made this pass

`LEDGER_A2_jepa.md` (ACPC) - `LEDGER_A5_benchmarks.md` (D-10 resolution) - `LEDGER_A1_world_models.md`
(Latent-WAM) - `LEDGER_B5_post_training.md` (RLIR + the GS-1 caution) - `LEDGER_B10_retrieval.md`
(created, zero-label complexity scorer) - `LEDGER_C1_releases.md` (C1/C2/C4 entries) -
`LEDGER_C3_regulatory.md` (D-4 routes 4-5).
