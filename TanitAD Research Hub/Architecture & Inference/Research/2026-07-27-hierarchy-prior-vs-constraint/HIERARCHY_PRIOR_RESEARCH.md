# Hierarchy as a PRIOR, not a GATE — literature research + design

**Stream:** Architecture & Inference research, 2026-07-27.
**Trigger:** the E-V5-2 measurement (`…/Implementation/incoming/2026-07-26-v5-imagination-selection/`,
`raw/v5_hier.json`) — *hierarchical SELECTION refuted, hierarchical BIASING confirmed.*
**Mode:** CPU / web only. **No pod was touched by this stream.**
**Author:** research subagent, under `Project Steering/AGENT_OPERATING_STANDARD.md`.

---

## 0. PRE-REGISTRATION — written BEFORE any literature was read

> ⛔ Everything in §0 was written and staged **before** the first paper was opened. The only
> prior input was our own measured result (§0.1) and the operating standard.

### 0.1 The measured result this stream is built on — quoted from the primary artifact

MEASURED, DECISION-GRADE for the negative half. Source:
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md` §4.2/§4.3 → `raw/v5_hier.json`.
Estimator: **paired episode-cluster bootstrap**, B = 2000, 40 clusters / 881 windows.
Surface: **produced (deployable)** unless marked oracle.

| structure over the SAME 256 frozen candidates | `ade_0_2s` | paired Δ vs flat | distinct picks |
|---|---:|---|---:|
| `F_flat` — as-trained, grafts present as an **additive bias**, 1-of-256 | **0.8563** | — | 128 |
| `F_base_only` — **grafts removed** from the flat score | 0.8781 | **+0.0218 [+0.0009, +0.0491]** ✅sep | 163 |
| `H_graft q=64` — **commit** to the class, search inside | 1.0621 | +0.2059 [+0.1290, +0.2975] ✅sep | 89 |
| `H_graft q=32` | 1.2000 | +0.3437 [+0.2284, +0.4855] ✅sep | 61 |
| `H_graft q=16` | 2.6510 | +1.7948 [+1.2345, +2.4488] ✅sep | 34 |
| `H_graft q=8` | 6.6752 | +5.8190 [+4.4381, +7.1956] ✅sep | 18 |

Mechanism check first (this is what makes it a result and not a statement about noise):
the three class→anchor grafts are **ReZero zero-init by construction** and training moved all
three — ‖W‖_F = 0.6457 / 0.6592 / 0.7260, max|w| ≈ 0.10.

Three structural facts about our numbers that constrain what the literature can and cannot say
about them:

1. **The prior and the gate are the SAME learned object.** `H_graft(q)` builds its admissible set
   from `lat_to_anchor.weight[:, cls] + lon_to_anchor.weight[:, cls] + dist_to_anchor.weight[:, cls]`
   — the identical tensors that, added to the flat logits, are worth −0.0218 m. This is a
   **prior-strength** experiment with the *information content held fixed*, which is rarer in the
   literature than "hierarchical vs flat architecture".
2. **`q` is a commitment-tightness knob, and the damage is monotone in it** (+0.21 → +0.34 → +1.79
   → +5.82 as q goes 64 → 32 → 16 → 8). `q = 256` is exactly `F_base_only`; `q = 1` would be pure
   tactical dictation. So the whole soft↔hard axis is one parameter, and we measured 5 points on it.
3. **Direction of the effect at the soft end is POSITIVE.** −0.0218 m for having the prior at all.
   The optimum is therefore interior: neither `no prior` (0.8781) nor `hard prior` (1.06–6.68) wins.

Separately MEASURED (§4.3, same artifact, same estimator): conditioning on the **produced** goal is
**worse than no goal**, −0.0943 [−0.1302, −0.0589] for turning it off; the **oracle** goal is worth
−0.2140 [−0.2778, −0.1573]. 87–92 % of that is selection-attributable, not fan quality.

### 0.2 What I am looking for, stated as questions with falsifiable answers

| # | question | what a YES looks like | what a NO looks like |
|---|---|---|---|
| Q1 | Is "harm monotone in commitment tightness" a **known** result? | a paper that sweeps a commitment/termination parameter and reports monotone degradation | the literature reports an **interior optimum** (an intermediate commitment beats both extremes), which would make our monotonicity the special case |
| Q2 | Does the options literature already say interruption dominates commitment? | a theorem or theorem-grade result: interrupting an option can only improve the value | only empirical/mixed evidence, or a stated benefit of commitment |
| Q3 | Do strong driving planners hard-gate on a mode, or soft-bias? | production/SOTA systems that keep a **shared** trajectory space and add mode as a bias/token | systems that genuinely prune to a mode and win |
| Q4 | Has the field characterised a **guidance-strength curve** with an interior optimum? | CFG papers reporting a non-monotone quality-vs-scale curve, and a planning-specific version | guidance monotone-better, or no such measurement |
| Q5 | Is "a wrong goal is worse than no goal" a studied robustness property? | papers measuring performance under corrupted/absent route or command input | nobody measures it |

### 0.3 ⚠️ PRE-REGISTERED FALSIFIER — what would make me conclude HARD COMMITMENT IS RIGHT and our measurement is the special case

I commit to writing **"hard commitment is right and our measurement is the special case"** — not a
scoped-down version — if the literature supplies **any two** of the following, at
PUBLISHED-DEMONSTRATED (not asserted) grade:

- **F1 — the classifier-quality threshold.** A published result showing hard mode commitment wins
  **when the mode classifier exceeds an accuracy/calibration threshold**, and loses below it, with
  the threshold quantified. Our grafts have max|w| ≈ 0.10 on a ReZero residual — they are plausibly
  a *weak* classifier, which would make ours the low-accuracy special case rather than a general law.
- **F2 — the candidate-count/coverage confound.** A published result showing that the harm of hard
  gating is driven by **losing coverage of the candidate set**, not by the commitment itself — i.e.
  hard gating over a *mode-complete* proposal set (each mode independently well-covered) is neutral
  or positive. Our fan is a **single shared 256-anchor vocabulary**, so a q = 8 gate keeps 8 anchors,
  not 8 anchors *per mode*. If the literature says that is the whole story, our monotone curve is a
  coverage curve wearing a commitment costume.
- **F3 — the horizon/frequency condition.** A published result that commitment pays at **long
  horizons or low replanning frequency** and costs at short ones. We measure a **2 s** horizon with
  **per-window (0.1 s-grid) re-selection** — the maximum-replanning, minimum-horizon corner, where
  commitment has the least to offer by construction.
- **F4 — a closed-loop inversion.** A published result where hard mode commitment is worse
  **open-loop** and better **closed-loop** (stability/consistency/no mode-flapping). Every number
  above is **open-loop 2 s ADE**. Our own program has already measured that open-loop does not
  predict closed-loop (`flagship-closed-loop-gap`: 0.45 m open → 1.69 m closed).
- **F5 — the metric-choice inversion.** A published result where hard commitment loses on
  **average displacement** and wins on **collision / rule-compliance / worst-case**, i.e. our ADE
  metric is the thing that punishes it.

**F3 and F4 are the two I consider most likely to fire**, and I am saying so in advance so that a
later "well, it's a special case" reads as a pre-registered outcome and not as a rescue.

**Conversely — what would CONFIRM our reading as general:** an independent published measurement
that sweeps a soft→hard commitment knob **with information content held fixed** and finds the same
monotone degradation, in any domain.

### 0.4 What would make me report NO USABLE ANSWER

If the literature only contains "hierarchical architecture A beat flat architecture B" comparisons —
where the hierarchy also changes capacity, training signal, or candidate coverage — then it does
**not** speak to our question, and I will say the field has not isolated this variable rather than
borrowing a nearby result. **A mechanism that sounds right is not a finding** (this program logged a
retraction on 2026-07-26 where two numbers moving together at 43.5 %/43.6 % looked like a mechanism
and the counterfactual inverted the sign).

### 0.5 Evidence-class discipline for everything below §1

Every claim carries `PUBLISHED (cited — specific paper)` + tier, and **separates what a paper
DEMONSTRATED (ran and measured) from what it ASSERTED (claimed in prose/related work)**.
`INHERITED` = quoted from another agent/doc and not re-verified here. Nothing in this file is
MEASURED by this stream — it is a literature read plus a design.

---

*(Everything below §0 was written after the corresponding source was read.)*

---

## 1. SOFT vs HARD COMMITMENT — what the literature actually demonstrates

### 1.1 In the options framework, INTERRUPTION is theorem-backed; COMMITMENT is a tuned hyperparameter

**PUBLISHED (cited — Sutton, Precup & Singh 1999, *Between MDPs and semi-MDPs*, Artificial
Intelligence 112:181–211), tier CONFIRMED, DEMONSTRATED (proved).**
The **Interruption Theorem** (their Theorem 2) constructs, from any option set `O` and Markov policy
`μ`, a new set `O′` in which each option `o = ⟨I, π, β⟩` is replaced by `o′ = ⟨I, π, β′⟩` with
`β′ = β` **except** that at any history ending in a state `s` where `Q^μ(h, o) < V^μ(s)` one may set
`β′(h) = 1` — terminate early wherever continuing is worse than re-deciding. The guarantee is
one-directional: `V^{μ′}(s) ≥ V^μ(s)` for all `s`, strict wherever an interrupted history is
reachable with non-zero probability.

⇒ **The general statement is: letting the lower level override the higher level's commitment can
never hurt, and generically helps.** That is *precisely* the mechanism our §4.2 identified — *"the
flat score's ability to OVERRIDE the tactical class is doing essential work"* — so our finding is
**not** an anomaly needing special pleading. It is the empirical shadow of a 27-year-old theorem.

⚠️ **What the theorem does NOT say.** It is stated for the *true* `Q^μ`/`V^μ`; it says nothing about
interrupting on a **learned, mis-calibrated** value estimate, and it compares *termination* policies
over a fixed option set — it is **not** literally a statement about pruning a candidate set. Treat it
as the correct **prior on the sign** of the effect, not as a proof of our number.

**The option-critic line** (Bacon, Harb & Precup, AAAI 2017) makes `β` learnable via a termination
gradient. **PUBLISHED, PROVISIONAL — secondary-source level only; full text not read in this
stream.** Its well-known empirical consequence is the next item.

### 1.2 ⭐ Commitment strength has an INTERIOR OPTIMUM — demonstrated independently in three domains

This is the most important thing the literature says about our result, and it reframes it.

| domain | knob | what was DEMONSTRATED | source |
|---|---|---|---|
| **HRL / options** | deliberation cost `η` — a margin an option must be beaten by before being replaced | sweeping `η` ∈ [0, 0.03] in 0.005 steps gives a **non-monotone** curve peaking at intermediate `η` (≈ 0.020–0.025). At `η = 0` options **collapse to one step** (termination prob → 100 %) and score worse: Amidar 512 → **880**, Asterix 1950 → **8700**, Hero 2625 → **20100** | **Harb, Bacon, Klissarov & Precup, AAAI 2018**, *When Waiting is not an Option* |
| **Diffusion generation** | classifier-free guidance strength `w` | **FID is U-shaped in `w`**, IS monotone increasing ⇒ explicit interior optimum at **low** guidance. ImageNet 64²: `w=0` FID 1.80 → **`w=0.1` 1.55 (best)** → `w=0.3` 3.03 → `w=1.0` 12.6 → `w=4.0` 26.22. ImageNet 128²: 7.27 → 4.53 → **2.43 (best, w=0.3)** → 7.86 → 21.53. Authors: *"best FID … with a small amount of guidance (w=0.1 or w=0.3 …) and the best IS … with strong guidance (w≥4)"* | **Ho & Salimans 2022**, *Classifier-Free Diffusion Guidance* |
| **Driving mode assignment** | WTA temperature `T(t) = T₀ρᵗ`, annealed soft (all modes weighted) → hard (winner only) | annealing beats going hard: Argoverse 2, MTR, 6 hypotheses — WTA minADE **0.85** → **aWTA 0.77 (−9.41 %)**, MissRate **−36.67 %**. Hard WTA is *"known to be unstable"*, with *"initialization sensitivity"* and *"mode collapse during training"* | **Annealed Winner-Takes-All for Motion Forecasting**, arXiv 2409.11172 |

A fourth, architectural: **Soft MoE** (Puigcerver et al., ICLR 2024) replaces hard token→expert
routing with a soft weighted combination *specifically because* hard discrete assignment causes
"training instabilities, dropping of tokens, and difficulties in scaling the number of experts", and
DEMONSTRATES it outperforms both dense ViTs and the hard-routed MoEs (Tokens Choice, Experts Choice).
**PUBLISHED, CONFIRMED for its own domain; transfer to planner mode selection is an analogy, not a
result.**

### 1.3 ⚠️ THE HONEST RESTATEMENT OF OUR OWN FINDING — "monotone" needs a qualifier

The brief asked whether *"harm is MONOTONE in commitment tightness"* is known or novel. The
literature forced me to re-read our own table, and it does **not** support an unqualified monotone
claim:

```
λ = 0  (no prior)      F_base_only   0.8781      ← WORSE than having the prior
λ = 1  (soft prior)    F_flat        0.8563  ★ best of the measured points
q = 64 (hard commit)   H_graft       1.0621
q = 32                               1.2000
q = 16                               2.6510
q =  8                               6.6752
```

**Our curve is ALSO interior-optimal.** It is monotone *only along the hard branch* (q = 64 → 8).
Going the other way — removing the prior entirely — is separated-worse
(**+0.0218 [+0.0009, +0.0491]**). The correct statement is therefore:

> **Prior strength has an interior optimum, and ours sits at or near the softest non-zero setting we
> have measured.** Monotone degradation appears only *beyond* that optimum, on the truncation branch.

This matters three ways: (a) it makes our result **consistent with**, not exceptional to, the
CFG / deliberation-cost / aWTA picture; (b) it kills the framing *"hierarchy is useless"* — the
optimum is **not** at zero prior; (c) **we have measured only two points on the soft side (λ = 0,
λ = 1) and therefore cannot say where the optimum is.** §6 exists for that.

What is plausibly novel is not monotonicity but **steepness with provenance held fixed**: a 5.82 m
penalty from tightening a prior whose *information content is identical*. I found **no published
sweep that holds the prior's information fixed while varying only its hardness over a shared
candidate set.** I claim that as a gap, not as a discovery.

### 1.4 The confound the literature makes me worry about most: COVERAGE, not commitment

`H_graft(q)` does not separate two things:

1. **commitment** — deciding the class first, so a good candidate outside it can no longer win;
2. **coverage loss** — going from 256 candidates to `q`.

Our fan is a **single shared 256-anchor vocabulary**, not `q` anchors *per mode*. A `q = 8` gate
leaves 8 trajectories to cover the whole reachable set, and V5 §2.2 measured the fan spanning
**108.7 m** of 2 s along-track displacement per window — so a small subset will miss the true motion
almost regardless of *which* subset it is.

This is pre-registered falsifier **F2**, and it is **testable today, on CPU, from staged artifacts**
(§6.2). I flag it before testing because it is the reading that would most change the design.

---

## 2. DRIVING — do mode-conditioned planners gate, or bias?

### 2.1 The strongest learned planners decode ALL modes and rank flat. None of them gates.

**PLUTO** (Cheng et al., arXiv 2404.14327) — *"surpassing the current top-performed rule-based
planner [PDM] for the first time"* on nuPlan. **DEMONSTRATED mechanism, quoted from the paper:** it
builds `N_R` lateral queries (one per reference line) × `N_L` longitudinal queries, combines them into
`Q₀ ∈ ℝ^{N_R × N_L × D}`, decodes the **full Cartesian product** `T₀ ∈ ℝ^{N_R·N_L × T_F × 6}` with
scores `π₀ ∈ ℝ^{N_R·N_L}`, and only then *"retains … the top K trajectories, ranked by their
confidence scores."* **The lateral mode is a query — a bias on what gets generated. It is never a
gate on what may be chosen.** Tier: **CONFIRMED** (mechanism quoted).

**VADv2 / Hydra-MDP** (Hydra-MDP: arXiv 2406.06978; VADv2 cited within it) — a **fixed planning
vocabulary** from K-means over trajectories, selecting *"the most optimal trajectory from [the] fixed
trajectories dictionary based on the cost function score"*, flat over the vocabulary. Hydra-MDP's
contribution is on the **scoring** side: run the rule-based simulator offline over the whole
vocabulary for the whole training set, distil those simulation scores into the student.
**PUBLISHED, PROVISIONAL** (challenge report + secondary source; full text not read).
⇒ **The published architecture closest to ours — fixed anchor vocabulary + learned score — fixed its
selector by making the SCORE simulator-grounded, not by gating the vocabulary.**

**PDM-Closed** (nuPlan 2023 winner) generates trajectory proposals, **simulates and scores** them,
then selects. **PUBLISHED, PROVISIONAL** — search-level only. Same shape: propose broadly, score by
simulation, select flat.

### 2.2 Where hard staging DOES appear, and what makes it survivable

**TNT / DenseTNT** (Zhao et al. CoRL 2020, arXiv 2008.08294; Gu et al. arXiv 2108.09640) genuinely
stage: predict target states, complete trajectories **conditioned on** them, then score and select a
compact set. **PUBLISHED, PROVISIONAL** (abstracts + secondary sources). Two features distinguish
this from `H_graft(q)`, and both are load-bearing:

- the goal stage keeps a **large** candidate set (dense goal candidates — hundreds to thousands), so
  stage 1 is a *coarse* restriction, not an 8-of-256 cut;
- **a final selection stage re-ranks across goals** — the commitment is not terminal.

**GoalFlow** (Xing et al., CVPR 2025, arXiv 2503.05689) is the sharpest case. It calls the goal point
*"a precise description of the short-term future position, which imposes a strong constraint on the
generation model"* — a genuinely hard conditioning signal, chosen from a **4,096–8,192**-element
endpoint vocabulary — and reaches **90.3 PDMS** on NAVSIM. **And it explicitly builds an escape
hatch:** the **shadow trajectory** is generated with *"the goal point masked during inference"*, and
*"if the shadow trajectory deviates significantly from the main trajectory, we treat the goal point
as unreliable and use the shadow as the output."* Tier: **CONFIRMED** (quoted).

⇒ **The pattern across all three: hard commitment appears only where (i) the committed-to set is
large, and (ii) an unconditioned fallback or a cross-mode re-rank can override it.** Our
`H_graft(q=8)` has neither.

### 2.3 The "hard assignment" in driving prediction is a TRAINING device, not an inference gate

This is where a careless literature read would produce exactly the wrong recommendation.

**MTR** (Shi et al., NeurIPS 2022, arXiv 2209.13508) is routinely described as using *hard
assignment*. That hard assignment is **which of the 64 motion query pairs receives the regression
loss** — the query nearest the GT endpoint — a training-time credit-assignment rule. At inference
**all** query pairs decode and are ranked. The stated purpose is that each query *"takes charge of …
a specific motion mode, which stabilizes the training process"* (abstract, quoted). Tier:
**CONFIRMED** for the abstract claim; **PROVISIONAL** for the inference detail (the fetched abstract
does not spell inference out; aWTA's framing of MTR as a WTA-trained multi-hypothesis model
corroborates it).

And even at training time **the field is moving away from hard**: aWTA's annealed soft→hard
assignment improves MTR by **−9.41 % minADE / −36.67 % MissRate** on Argoverse 2 (§1.2).

> ⇒ **PUBLISHED, CONFIRMED, and directly on the brief's question:** in driving, **mode is a bias on
> generation and a label for training credit; it is not a gate on what may be executed.** Our
> `H_graft(q)` arms implemented the one thing the field does not do.

---

## 3. GOAL / ROUTE CONDITIONING — injection, and what happens when it is wrong

### 3.1 How the strong systems inject it

| system | injection mechanism | evidence class |
|---|---|---|
| TransFuser++ / CARLA line | **target point** = GNSS waypoints ~30 m apart along the route, as a low-dimensional conditioning vector | PUBLISHED, CONFIRMED (Jaeger et al. ICCV 2023, quoted) |
| CIL / CILRS | discrete **navigational command** selecting a branch head | PUBLISHED, PROVISIONAL (Codevilla et al. ICCV 2019, abstract-level) |
| PLUTO | route as **reference-line geometry** → lateral queries (a structured prior on generation) | PUBLISHED, CONFIRMED (quoted) |
| Diffusion Planner | route via **MLP-Mixer → adaptive layer-norm** on the diffusion timestep condition, across all trajectory tokens | PUBLISHED, CONFIRMED (quoted, arXiv 2501.15564) |
| GoalFlow | **goal point** sinusoidally encoded, concatenated with scene + trajectory features, guiding flow matching | PUBLISHED, CONFIRMED (quoted) |

⇒ Two families: **a token/condition on generation** (all of the above) and **a cost/guidance term at
inference** (Diffusion Planner's classifier guidance). **Nobody in this set uses the goal to prune
the candidate set.** Our `route`/`route_graded`/`vt_band` are family one; our `vt_speed`→`sel_gate`
term is family two.

### 3.2 A wrong or absent goal: what is measured

**PUBLISHED, CONFIRMED — Jaeger, Chitta & Geiger, ICCV 2023, *Hidden Biases of End-to-End Driving
Models*.** DEMONSTRATED: target-point conditioning creates a **shortcut** — models steer toward the
nearest TP, which silently supplies the lateral recovery imitation learning otherwise lacks. Their
ablation: NC-conditioned (discrete commands) **32 DS / 56 % RC**; TP-conditioned **39 DS / 84 % RC**;
route deviations **0.00/km** (TP) vs **0.86/km** (NC). **And the failure mode is ours in mirror
image:** when the TP is far away the shortcut *"produces catastrophic failures — the model steers
directly toward a TP behind a turn, cutting the turn and driving into opposing lanes."* Their fix is
representational (transformer cross-attention instead of global average pooling, **+9 RC**) — *make
the decoder able to use perception, so the goal stops being the only usable signal.*

**PUBLISHED, PROVISIONAL — *Closing the Navigation Compliance Gap in End-to-end Autonomous Driving*
(arXiv 2512.10660).** Per the fetched text: planners *"frequently ignore route instructions"*,
*"incorrect navigation commands sometimes receive similar compliance rates as correct ones"*, and
*"absent navigation leads to unpredictable behavior rather than safe fallback"*; their remedy is a
**soft** one — loss weighting that *"encourages but doesn't absolutely mandate compliance."*
⚠️ **I could not extract the paper's own numbers from the PDF. The qualitative claims are
PROVISIONAL and any quantitative claim from it is UNVERIFIED.**

### 3.3 ⭐ GoalFlow supplies the exact three-way ladder our §4.3 measured — with the OPPOSITE ordering

**PUBLISHED, CONFIRMED** (numbers quoted from the paper's ablation). NAVSIM PDMS ↑ vs our
`ade_0_2s` ↓ (MEASURED, `raw/v5_hier_windows.pt`, 881 windows):

| conditioning | GoalFlow PDMS ↑ | TanitAD v4 `ade_0_2s` ↓ |
|---|---:|---:|
| **no goal** | 85.6 | **0.7620** ← *our best* |
| **produced / predicted goal** | **90.3** ← *their deployable* | 0.8563 ← *our worst* |
| **oracle / GT goal** | 92.1 | 0.6423 |

> ⛔ **This is the most decision-relevant single comparison in this document.** GoalFlow's *predicted*
> goal recovers **73 %** of the oracle headroom (90.3 within the 85.6 → 92.1 span). Our produced goal
> recovers **−147 %** — it is worse than not having one. Same three-rung ladder, opposite sign.
>
> ⇒ **"A produced goal is worse than no goal" is NOT a property of goal conditioning. It is a defect
> of our goal producer.** The V5 artifact reached *"a producer problem, not a consumer problem"* from
> our data alone; GoalFlow makes it PUBLISHED-CONFIRMED rather than inferred, and removes the
> tempting alternative reading (*"goal conditioning is a mirage"*) from the table.

**And GoalFlow supplies the mitigation to copy rather than invent:** the **shadow trajectory** — run
with the goal masked, compare, fall back to the unconditioned branch on disagreement. That is
strictly better than our currently-available free win (*"turn the produced goal off"*, −0.0943 m),
because it keeps the upside on windows where the goal is right. It is **D2** in §5.

---

## 4. SOFT-PRIOR MECHANISMS — the menu, and what each is known to buy

| mechanism | form | what is DEMONSTRATED | fit to us |
|---|---|---|---|
| **Additive logit bias / product of experts** | `score = base + λ·log p_prior` | exactly what v4's `_factor_grafts` already computes (`log_softmax` → linear → add); worth a separated **−0.0218 m** *jointly with* the vt term | **already deployed**; only its *strength* is unparameterised |
| **Classifier-free guidance** | `ŝ = s_uncond + w·(s_cond − s_uncond)` | interior optimum, U-shaped FID, best at **w ≈ 0.1–0.3** (Ho & Salimans) | the natural parameterisation of our **goal** channel (D2) |
| **Classifier guidance at inference** | `s̃ = s − ∇_x E_φ(x, t)` | Diffusion Planner adds safety/comfort/speed costs **at inference with no retraining**; ⚠️ **no guidance-strength sweep reported** — case studies only | the natural home for a longitudinal-admissibility cost (V5 §7.2) |
| **Temperature on the prior's posterior** | `log_softmax(logits/τ)` | aWTA: annealing `T` soft→hard beats fixed-hard (**−9.41 % minADE**) | **one-line change** in `_factor_grafts`; decouples *sharpness* from *truncation* |
| **Soft routing (Soft MoE)** | weighted combination of all experts | beats hard routing; hard routing → instability + dropped tokens | architectural analogue; supports the direction |
| **Deliberation cost / switching margin** | `A(s,o) + η` | non-monotone in `η`, interior optimum ≈ 0.02 | the **temporal** analogue → D3 |
| **Hard top-`q` truncation** | mask all but `q` | **no published planner does this at inference**; our own measurement: **+0.21 … +5.82 m** | **refused** |

⚠️ **A gap worth stating plainly:** the guidance-strength curve is carefully characterised **for image
generation** (Ho & Salimans) and **not for planning**. Diffusion Planner — the ICLR 2025 Oral whose
headline feature is *flexible guidance* — provides qualitative case studies and **no sensitivity
analysis over guidance magnitude**. So the brief's question *"what is known about its optimal
strength for planning?"* has the answer: **essentially nothing is published.** That makes §6.3 a
contribution rather than a replication.
