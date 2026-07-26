# Hierarchy as a PRIOR, not a GATE — literature research + design

**Stream:** Architecture & Inference research, 2026-07-27.
**Trigger:** the E-V5-2 measurement (`…/Implementation/incoming/2026-07-26-v5-imagination-selection/`,
`raw/v5_hier.json`) — *hierarchical SELECTION refuted, hierarchical BIASING confirmed.*
**Mode:** CPU / web only. **No pod was touched by this stream.**
**Author:** research subagent, under `Project Steering/AGENT_OPERATING_STANDARD.md`.

---

## HEADLINE — six findings, in the order they should change what we do

1. **Soft-over-hard is well supported and theorem-backed.** The **Interruption Theorem** (Sutton,
   Precup & Singh 1999) says letting the lower level override the higher level's commitment can never
   hurt. Every SOTA driving planner I could read the mechanism of — **PLUTO, VADv2/Hydra-MDP,
   PDM** — decodes *all* modes and ranks **flat** over the whole set. **None of them gates.** (§1.1, §2.1)
2. ⚠️ **"Monotone harm in commitment tightness" is not the field's picture, and re-reading our own
   table, it is not ours either.** In three independent domains — deliberation cost `η`,
   classifier-free guidance `w`, annealed WTA temperature — prior strength has an **INTERIOR
   OPTIMUM**. Our curve does too: 0.8781 (no prior) → **0.8563 (soft)** → 1.06 → 6.68 (hard).
   Monotone degradation is only the *hard branch*. **We have measured two points on the soft side and
   cannot say where our optimum is.** (§1.2, §1.3)
3. ⛔ **HEADLINE CORRECTION: the "+0.0218 m useful as a soft prior" is not the hierarchy alone.**
   `F_base_only` removes the grafts **and** the longitudinal `sel_gate` term. Decomposing two staged
   artifacts on the oracle surface: **graft ≈ 0.0092 m, constant-velocity term ≈ 0.0100 m.** The
   hierarchical prior is worth about **half** the quoted figure. (§5.2)
4. ⭐ **MEASURED HERE (CPU, staged artifacts, no pod): the coverage confound does NOT carry our
   result.** At **q = 64** a *random* 64-subset contains a candidate **0.3750 m better than flat**,
   separated — yet committing to the graft's 64 is 0.2058 m **worse**. Coverage explains **0 %** of
   the harm at q ≥ 32 and **~30 %** at q ≤ 16. (§9.3)
5. ⛔ **I retracted one of my own pre-registered arms mid-run**, and in doing so found that
   **`base_rank` in a staged V5 artifact is not a rank** — it is `[pick] ++ [anchor index order]`,
   verified 881/881. One V5 §5.2 label is wrong because of it; its conclusions survive. (§9.2, §9.5)
6. ⭐ **"A produced goal is worse than no goal" is a defect of OUR producer, not a property of goal
   conditioning.** GoalFlow's identical three-rung ladder runs **85.6 (none) → 90.3 (predicted) →
   92.1 (oracle) PDMS** — predicted recovers **73 %** of the oracle headroom. Ours recovers **−147 %**.
   And GoalFlow supplies the fix to copy: the **shadow trajectory**, an unconditioned branch you fall
   back to when the goal looks unreliable. (§3.3)

**The design (§5):** keep the additive product-of-experts prior v4 *already* has, **expose its two
strength knobs (`λ` gain, `τ` class-posterior temperature), and delete `q` from the deployment path.**
`(λ, τ)` reach arbitrarily hard commitment **without truncating the candidate set** — the axis our
measurement never had and the literature has never published. **The whole 42-cell sweep is ONE
forward pass plus seconds of CPU per cell** (§6.3).

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

---

## 5. THE DESIGN — how our hierarchy should bias rather than gate

### 5.1 What v4 already does, read out of the code (not from a doc)

**MEASURED-by-reading**, `stack/tanitad/models/flagship_v4.py:161-195` and
`stack/tanitad/models/flagship_v15.py:451-466`. The deployed selector is:

```
g_m   = W_m · log_softmax(logits_m)                m ∈ {lat, lon, dist}      # flagship_v4.py:173-176
graft = g_lat + g_lon + g_dist
ratio = ‖graft‖₂ / ‖refined‖₂                      per-sample                # :183
  if ratio.max() > seam_fail (=1.5):  raise                                  # :185-189
graft ← graft · min(1, seam_clamp / ratio)         seam_clamp = 1.0          # :191-192
refined ← refined + graft                                                    # :195
score  = refined + sel_gate · ( −|v_term − v_goal| ) · vt_keep               # flagship_v15.py:451-461
pick   = argmax_c score[c]                          FLAT over all 256        # :465
```

> ⭐ **v4's selector is ALREADY the "bias, don't gate" architecture** — a base score plus a
> **norm-capped product-of-experts class prior** plus a learned-scale longitudinal preference,
> resolved by a **flat argmax over the entire candidate set**. It consumes the full `log_softmax`
> class posterior, not the argmax class.
>
> ⇒ **The design work is not to build the soft prior. It is to PARAMETERISE A STRENGTH THAT IS
> CURRENTLY HARD-WIRED** — implicit gain λ = 1, implicit temperature τ = 1, capped at
> `seam_clamp = 1.0` — **and to delete the `q` gate from the deployment path permanently.**

### 5.2 ⚠️ AN ATTRIBUTION CONFOUND IN OUR OWN HEADLINE — the +0.0218 m is not the hierarchy alone

`F_base_only` is `refined_pre.argmax` (`code/v5_hierarchical_select.py:293`), where `refined_pre` is
`dec["refined_logits"]` — the decoder output **before** `_factor_grafts` **and** before `sel_gate`.
So `F_flat − F_base_only` = **graft + longitudinal term**, and the headline
*"+0.0218 m useful as a soft prior"* bundles two different priors. The V5 file says so in its own
arm description (*"grafts + vt penalty off"*); the headline does not.

**The decomposition, from two independently staged artifacts on the same 881 windows** (both
reproduce `F_flat` = 0.6423 exactly, which is the consistency check that makes the arithmetic
admissible at all):

| oracle surface | `ade_0_2s` | source |
|---|---:|---|
| `F_flat` — graft ON, gate ON | **0.6423** | `…/2026-07-26-v5-imagination-selection/raw/v5_hier_windows.pt` and `…/2026-07-26-v4-restart-lever/raw/v4_selgate_ablation.json` (agree exactly) |
| `sel_gate := 0` — graft ON, gate OFF | **0.6523** | `…/2026-07-26-v4-restart-lever/`, paired Δ **−0.0100 [−0.0191, −0.0020]**, ✅sep |
| `F_base_only` — graft OFF, gate OFF | **0.6615** | `…/2026-07-26-v5-imagination-selection/raw/v5_hier_windows.pt` → key `oracle` + `F_base_only` |

⇒ **longitudinal term ≈ 0.0100 m · class→anchor graft ≈ 0.0092 m · total 0.0192 m** (= 0.6615 −
0.6423 ✓).

> ⛔ **HEADLINE CORRECTION, and it needs to travel:** on the oracle surface the **hierarchical
> graft is worth ≈ 0.0092 m — about HALF of the quoted +0.0218 m.** The other half is a
> **constant-velocity plausibility preference**, not a hierarchy at all: `_goal_inputs` sets
> `vt_speed = v0` (`train_flagship_v4.py:172`) with a ±5.0 m/s reachable clamp, so `v_goal ≡ v0` in
> every v4 run and the "target-speed-aware" term is a pure constant-velocity bias
> (MEASURED, `…/2026-07-26-v4-restart-lever/` §4.2).
>
> **Tier: DERIVED, PROVISIONAL.** This is cross-artifact arithmetic, not one paired measurement — no
> CI is admissible for the graft-alone term, and the **produced-surface split is UNKNOWN** because
> the `sel_gate` ablation was only ever run on the oracle surface. Making it decision-grade needs one
> 3-arm paired run (§6.1, E-H0b) that costs one forward pass.
>
> ⚠️ **Disclosure:** I computed `oracle|F_base_only` = 0.6615 from the staged tensor *before*
> pre-registering it, because it is a decomposition of two already-published numbers rather than a
> new experiment. Recording that here rather than presenting it as pre-registered.

**And note where the other half points.** A constant-velocity plausibility preference being worth as
much as the entire learned hierarchy is the *same* mechanism that made E-V5-1's only winning arm work
(A4's weight mass sat on **C1**, an analytic bicycle model, and **C2**, one reference roll — V5 §2.4)
and the *same* mechanism V5 §7.2 nominated as the cheapest next lever (clip the fan to a
kinematically admissible longitudinal band). **Three independent measurements now point at
longitudinal plausibility as the live lever.** That convergence is the strongest signal in this file
and it is not a hierarchy signal.

### 5.3 D1 — expose (λ, τ). Delete `q`.

```python
# stack/tanitad/models/flagship_v4.py  ::  _factor_grafts
tau = self.cfg.graft_tau        # NEW, default 1.0
lam = self.cfg.graft_lambda     # NEW, default 1.0
lsm = torch.log_softmax
g_lat  = self.lat_to_anchor (lsm(lat_logits  / tau, dim=-1))
g_lon  = self.lon_to_anchor (lsm(lon_logits  / tau, dim=-1))
g_dist = self.dist_to_anchor(lsm(dist_logits / tau, dim=-1))
graft  = lam * (g_lat + g_lon + g_dist)
# ... norm clamp and fail-loud unchanged ...
```

Two config fields, three divisions, one multiply. **`λ = 1, τ = 1` is forward-bit-identical to the
shipped path**, which is the same attributability discipline the zero-init grafts were built with
(`stack/tests/test_flagship_v4.py::test_zero_init_grafts_leave_the_ranked_score_bit_identical`) —
so the delta is testable by an equality assertion, not by a benchmark.

**Why (λ, τ) and not `q`:**

| knob | what it changes | reaches hard commitment? | costs candidates? |
|---|---|---|---|
| **λ** (gain) | how loud the prior is, shape unchanged — the CFG `w` analogue | asymptotically | **no** |
| **τ** (temperature) | how *peaked* the class posterior is; τ→0 ⇒ one-hot class | **yes, exactly** | **no** |
| `q` (truncation) | how many candidates survive | yes | **yes — this is the confound** |

> ⭐ **(λ, τ) reach arbitrarily hard commitment WITHOUT truncating the candidate set.** That is the
> axis our measurement never had, and it is the one that decides whether the 0.21–5.82 m penalty is
> **commitment** or **coverage** (§1.4, falsifier F2). No published sweep does this either (§1.3).

⚠️ **A trap that would silently flatten the λ axis.** `seam_clamp = 1.0` rescales the graft whenever
`‖graft‖ > ‖refined‖`, so **beyond that point λ is a no-op**. A λ sweep run with the clamp at its
default would produce a flat curve above the saturation point and be misread as *"λ does not
matter."* Every cell must therefore report `seam_norm_ratio_preclamp_max`, and the sweep must be run
as **two sheets** — clamp at 1.0 (deployable) and clamp effectively off (diagnostic).

### 5.4 D2 — the goal channel becomes a guidance scale, plus GoalFlow's shadow branch

```
score(λ_g) = score_neutral + λ_g · ( score_produced − score_neutral )
```

This is **literally classifier-free guidance** with the goal as the condition. Two of its points are
already MEASURED on the deployable surface: `λ_g = 1` ≡ today (**0.8563**), `λ_g = 0` ≡ goal off
(**0.7620**, paired **−0.0943 [−0.1302, −0.0589]**, ✅sep, free). **Our two measured points already
say the optimum is below 1**; the CFG picture (§1.2) says the curve has an interior optimum; the
sweep locates it.

Plus the **shadow rule** (GoalFlow, §2.2, published + SOTA): compute both picks; if the two
trajectories disagree by more than `d*`, emit the **neutral** one. This dominates "turn the goal
off" because it keeps the upside on windows where the goal is right — and `d*` is a 1-D sweep over
cached data.

⚠️ Negative `λ_g` is admissible **as a measurement** and never as a deployable: a producer that is
reliably anti-informative is a producer to fix (§3.3), not a sign to flip.

### 5.5 D3 — a TEMPORAL commitment prior, named so it does not get lost

The one commitment the literature does support is **temporal**, not hierarchical (§7, MomAD):

```
score ← score − λ_t · d_Hausdorff( candidate , previous_pick )
```

Soft, weighted, never a gate. **It is invisible to our current metric** — open-loop 2 s ADE at
per-window re-selection cannot see plan-to-plan consistency — so it must not be evaluated on this
surface. Deferred explicitly to the first closed-loop harness, and named here so the pre-registered
falsifier F4 has an owner instead of a footnote.

### 5.6 The implementation delta, file by file

| file | change | size | risk |
|---|---|---|---|
| `stack/tanitad/models/flagship_v4.py` | `graft_lambda`, `graft_tau` in the config dataclass; use them in `_factor_grafts` | ~6 lines | **none at defaults** — bit-identical, assertable |
| `stack/tests/test_flagship_v4.py` | one test: `λ=1, τ=1` reproduces the shipped score bit-for-bit; one: `τ→0` gives a one-hot class posterior | ~20 lines | none |
| `…/2026-07-26-v5-imagination-selection/code/v5_hierarchical_select.py` | dump `prior [W, 256]`, `refined_pre [W, 256]`, `lat/lon/dist logits`, `score_neutral`, `score_produced` into the reduced `.pt` | ~10 lines | none — additive |
| deployment path | **remove `q` entirely**; `hierarchical_pick` stays as a measurement arm only | — | this is the recommendation |

**Nothing here trains anything.** Every arm is a re-scoring of a frozen fan.

---

## 6. THE PRE-REGISTERED COMMITMENT-STRENGTH SWEEP

Estimator for every arm: **paired episode-cluster bootstrap**, B = 2000, unit = episode cluster,
40 clusters / 881 windows, on identical windows (`taniteval/ci.py`). **`overlapping_holdout_se` is
never used.** Every number decomposed into along-track / cross-track.
⚠️ **A null at n = 40 is UNPOWERED, not refuted** (`MODEL_REGISTRY.md §1.2a`: half-widths shrink
×2.8–3.9 at n = 600). Any winning cell must be re-run at 600 before it steers GPU-days.

### 6.1 E-H0b — close the attribution confound. ONE forward pass.

Three arms in **one** harness on the **produced** surface, paired: `graft ON/gate ON` (0.8563),
`graft ON/gate OFF`, `graft OFF/gate OFF` (0.8781). Settles §5.2's PROVISIONAL split with a CI.
**Bar:** if the graft-alone term is **not separated** from zero on the produced surface, then
*"hierarchical biasing is confirmed"* is **UNPOWERED, not confirmed**, and must be restated as such
in `MODEL_REGISTRY.md`. I commit to writing that if it happens.

### 6.2 E-H1 — the COVERAGE CONTROL. CPU only, minutes, staged artifacts only. ⭐ RUN IN THIS STREAM.

**This is falsifier F2 made decidable.** Inputs: `raw/v5_v4_windows_reduced.pt`
(`fan_err4 [881,256]`, `base_rank [881,256]`, `ep [881]`) — nothing else, no GPU, no pod.

*(Verified before pre-registering, and stated because it is load-bearing: `base_rank[w,:]` is a
per-window permutation of 0..255, and `fan_err4.gather(1, base_rank[:, :1]).mean()` = **0.8563** =
`F_flat` exactly, so `base_rank` is the argsort of the **deployed** grafted score, best first.)*

| arm | rule |
|---|---|
| `H_graft(q)` | **measured, staged**: 1.0621 / 1.2000 / 2.6510 / 6.6752 for q = 64/32/16/8 |
| **`R_rand(q)`** | draw a uniform random q-subset `A` per window; pick the member of `A` best-ranked under the **deployed** score: `pick = base_rank[w, min{j : base_rank[w,j] ∈ A}]`. S = 200 seeds, report mean and seed-spread |
| **`O_rand(q)`** | oracle within the same random subset — the pure coverage bound |
| `O_graft(q)` | oracle within the graft-admissible subset — **NOT computable from staged artifacts** (needs `prior`); flagged, deferred to E-H2 |

**Outcomes, committed before running:**

| verdict | condition | what I will write |
|---|---|---|
| **COVERAGE** | `R_rand(q)` within the paired CI of `H_graft(q)` at every q | **F2 FIRES.** Restate the finding as *"truncating a shared anchor vocabulary is what costs; the graft's choice of subset is no better and no worse than chance."* The design conclusion (never truncate) is unchanged, but **the word "commitment" is retracted in favour of "coverage"** and `RETRACTION_LOG.md` gets a C6-class entry |
| **COMMITMENT-INFORMATIVE** | `R_rand(q)` separated-**worse** than `H_graft(q)` | the graft picks a better-than-random subset; the harm is coverage, partly offset by an informative prior. F2 **partially** fires; the headline becomes *"restricting the candidate set is what costs — our prior is good, our gate is the problem"* |
| **COMMITMENT-HARMFUL** | `R_rand(q)` separated-**better** than `H_graft(q)` | F2 does **not** fire. *"Committing to the tactical class is worse than committing at random"* survives exactly as stated, and is a much stronger claim than the one currently in the registry |

⭐ **Stated before running: I expect COMMITMENT-INFORMATIVE**, because the same tensors are worth
≈ +0.009 m as a bias (§5.2) and an informative prior should choose a better-than-random subset.
If the result is COMMITMENT-HARMFUL I will flag it as a *surprise* rather than absorbing it.

> ⛔ **WHAT ACTUALLY HAPPENED — read §9.2 before §9.3.** The `R_rand(q)` arm above rests on a premise
> about `base_rank` that a post-run self-test **refuted on 881/881 rows**: `base_rank` is not a score
> ranking. **The arm is retracted and its numbers are not quoted.** `O_rand(q)` is unaffected and is
> what adjudicated F2 (§9.3). The pre-registration is left here **unedited** so the retraction is
> visible rather than absorbed.

### 6.3 E-H2 — the (λ, τ) surface. ONE forward pass + seconds of CPU per cell.

λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8} × τ ∈ {0.1, 0.25, 0.5, 1, 2, 4}, **q = 256 always**. 42 cells × 2
clamp sheets.

**Why this is cheap and not a GPU-week:** `fan`, `refined_pre`, and the three class logit vectors are
**independent of λ and τ**. One forward pass over the 881 windows caches them; every cell is then a
CPU argmax over `[881, 256]`. **The whole surface costs one v5-style build (~minutes of GPU) plus
seconds of CPU per cell.**

| verdict | condition | reading |
|---|---|---|
| **CONFIRM-INTERIOR** | some (λ, τ) beats **0.8563** by more than the paired CI half-width | the shipped λ=1/τ=1 is **not** the optimum; free ADE on the table at zero training cost |
| **CONFIRM-SOFT** | the argmin has **λ ≤ 1 and τ ≥ 1** | softer is better, exactly as CFG/aWTA predict |
| ⭐ **REFUTE-SHARPNESS** *(the F2 discriminator)* | **τ → 0.1 at λ = 1** (maximally hard commitment, **zero truncation**) is within CI of 0.8563 | the 0.21–5.82 m penalty was **truncation**, not commitment. Our headline is a coverage result |
| **CONFIRM-SHARPNESS** | τ → 0.1 at λ = 1 degrades toward the `H_graft` numbers | **commitment sharpness itself is the cost**, independent of coverage. This is the outcome that makes the original claim general, and it is the one I would most like to be true — which is exactly why it is written down before the run |
| **SATURATED** | every λ ≥ some value gives an identical score AND `seam_norm_ratio_preclamp_max` is pinned at 1.0 | the clamp ate the axis; re-run the diagnostic sheet. **Not** a finding about λ |

### 6.4 E-H3 — the goal guidance scale. Same one-pass trick.

λ_g ∈ {−1, −0.5, −0.25, 0, 0.25, 0.5, 0.75, 1, 1.5} over cached `score_neutral` / `score_produced`,
plus a 1-D sweep of the shadow threshold `d*`.
**Bar: CONFIRM** if any λ_g or `d*` beats the neutral **0.7620** on the deployable surface, paired
and separated ⇒ we recover part of the oracle's −0.2140 m without fixing the producer.
**REFUTE** if the argmin is at λ_g = 0 with no `d*` improving on it ⇒ *turn the goal off and fix the
producer* is the whole answer, and the shadow-branch idea does not transfer.

### 6.5 Priority order (so a killed agent still yields value)

1. **E-H1** — CPU, minutes, staged artifacts, settles F2. **Run first, run here.**
2. **E-H0b** — one forward pass, fixes a number already circulating in the program.
3. **E-H2** — one forward pass, the contribution the literature does not have.
4. **E-H3** — same pass as E-H2 if the neutral/produced scores are cached together.
5. **D3 / closed-loop** — deferred, needs a harness that does not exist yet.

---

## 7. CONTRADICTING EVIDENCE — actively sought, reported fairly

A one-sided read is how this program has been burned. Six items that cut against us, strongest first.

**C1 — MomAD (Song et al., CVPR 2025, *Don't Shake the Wheel*), the strongest.**
**PUBLISHED, CONFIRMED, DEMONSTRATED.** Topological Trajectory Matching computes the Hausdorff
distance from each candidate to the **previous** plan and selects by **argmin** — a hard selection
rule biased entirely toward commitment. Results: Bench2Drive **closed-loop** success 16.71 %
(SparseDrive) → **18.11 %**; comfort 48.63 → **51.20**; and a new consistency metric TPC 0.81 →
**0.65 m** with one frame of history. **This is falsifier F4's direction, demonstrated.**
Three caveats that keep it from firing F4 outright: (i) it is **temporal** commitment (to your own
last plan), not **hierarchical** commitment (to a class); (ii) their own ablation shows the benefit
**plateaus immediately** — TPC 0.65 at t=2 → 0.66 at t=3, i.e. *more* commitment is not better; and
(iii) per the fetched text there is **no strength hyperparameter**, so it is not a commitment-strength
curve. ⇒ It is a strong argument for **D3**, and not an argument for `q`.

**C2 — Harb et al. 2018, the deliberation cost.** **PUBLISHED, CONFIRMED, DEMONSTRATED.** At η = 0,
options *"terminate at every step"* and performance is much worse (Amidar 512 vs **880**; Asterix
1950 vs **8700**; Hero 2625 vs **20100**). **Zero commitment is a real failure mode.** Caveat: it is a
*degenerate-option* failure — an option with no temporal extent is not an option — whereas our
`F_flat` is a working selector, not a degenerate one. And their curve is non-monotone with an
**interior optimum**, matching §1.3 rather than contradicting it.

**C3 — DARPA Urban Challenge FSM planners.** **PUBLISHED, PROVISIONAL (search-level only.)**
Hierarchical finite-state-machine manoeuvre selection with genuinely hard mode switching finished and
won the 2007 event; the cited benefit is *"resilience to perception noise"*. Caveat, and it is F1's
condition met by construction: those mode classifiers were **hand-designed and verified**, with a
full prior map — nothing like a ReZero graft with max|w| ≈ 0.10. Note also that the field replaced
them with flat-scoring learned planners as soon as those worked.

**C4 — GoalFlow.** **PUBLISHED, CONFIRMED.** A goal point that *"imposes a strong constraint on the
generation model"* reaches **90.3 PDMS**, SOTA on NAVSIM — hard conditioning that works. Caveats:
the committed-to vocabulary is **4,096–8,192** elements (not 8), and the system carries an explicit
**unconditioned fallback** (§2.2). It is evidence for *large-set, override-able* commitment.

**C5 — ez-greedy (Dabney, Ostrovski & Barreto 2020).** **PUBLISHED, PROVISIONAL (abstract-level.)**
Persisting a *random* action for a random duration beats step-wise dithering — persistence per se has
value. Caveat: an **exploration**, training-time result; it says nothing about inference-time
candidate selection.

**C6 — cuts the other way, and belongs here for honesty. Nachum et al., ICLR 2020,
*Why Does Hierarchy (Sometimes) Work So Well in RL?*** **PUBLISHED, CONFIRMED, DEMONSTRATED.** They
decouple `c_train` from `c_expl` in HIRO and add a non-hierarchical **"shadow agent"** trained on the
HRL agent's own collected experience; it is *"competitive with HRL"* on **3 of 4** tasks, and
non-hierarchical *Explore & Exploit* / *Switching Ensemble* baselines match HRL. Conclusion: *"most of
the observed benefits of hierarchy can be attributed to improved exploration"*, **not** semantic
decomposition. ⇒ Supports our refusal of hierarchical selection — **and warns that our hierarchy may
be buying less than we assume even as a prior**, which is exactly what §5.2's ≈ 0.0092 m says.

---

## 8. THE PRE-REGISTERED FALSIFIER — verdict against §0.3

| # | falsifier | fired? | evidence |
|---|---|---|---|
| **F1** classifier-quality threshold | ❌ **NO** | no published result quantifies an accuracy/calibration threshold above which hard mode gating wins. C3 meets the *spirit* (hand-verified classifiers) at PROVISIONAL grade only |
| **F2** coverage confound | ⚠️ **PARTIAL — and it was OURS, not the literature's. MEASURED in §9** | no published demonstration either way, so I measured it. **E-H1 result: coverage explains 0 % of the harm at q ≥ 32 and ~30 % at q ≤ 16.** At q = 64 a *random* 64-subset contains a candidate **0.3750 m better than flat**, separated — yet committing to the graft's 64 is 0.2058 m worse. F2 does **not** carry the result, but it does force a quantitative restatement of the q = 8 number |
| **F3** horizon / replanning frequency | ⚠️ **PARTIAL** | C2 DEMONSTRATES that zero commitment is catastrophic for *option duration*. But that is temporal extent, not candidate restriction, and its interior optimum matches ours |
| **F4** closed-loop inversion | ⚠️ **PARTIAL, and the strongest caution** | C1 DEMONSTRATES a closed-loop gain (16.71 → 18.11 % SR) from committing to the previous plan, on a metric open-loop ADE cannot see. Our numbers are **all** open-loop 2 s ADE, and this program has measured that open-loop does not predict closed-loop (0.45 m open → 1.69 m closed) |
| **F5** metric inversion | ❌ **NO** | no evidence found that hard commitment loses on displacement and wins on collision / rule-compliance / worst-case |

**§0.3 required two falsifiers at PUBLISHED-DEMONSTRATED grade *for our variable*. Zero clear that
bar; two clear it for adjacent variables; and F2 — the one I could test myself — was tested and does
not carry the result (§9).** So I do **not** write *"hard commitment is right after all."* I write the two things F3 and F4 do license, and they are scope limits, not rescues:

> **(a) Our verdict is scoped to OPEN-LOOP, PER-WINDOW RE-SELECTION** — the maximum-replanning,
> minimum-horizon corner, where interruption is cheapest and commitment has least to offer *by
> construction*. A closed-loop re-measurement could invert the **temporal** half of it. **D3 exists
> because of this line.**
>
> **(b) The finding is about HIERARCHICAL CLASS commitment, not about commitment in general.**
> Temporal commitment to one's own previous plan is a different lever, is supported by published
> closed-loop evidence, and **we have never tested it.**

**And the honest summary of the whole literature read**, which is not the summary the brief's framing
anticipated:

> Soft-over-hard is **well supported** (interruption theorem; CFG; aWTA; Soft MoE; every SOTA driving
> planner ranking flat over a mode-decorated set). **Monotone harm in commitment tightness is NOT the
> field's picture and is not ours either** — the field's picture, in three independent domains, is an
> **interior optimum in prior strength**, and re-reading our own table shows we measured one too
> (0.8781 → **0.8563** → 1.06 → 6.68). Our contribution is not the sign; it is that we measured the
> curve **with the prior's information content held fixed**, which nobody has published — and we
> measured it with **`q`, a knob that confounds commitment with coverage**, which §6 fixes.

---

## 9. E-H1 RESULT — the coverage control, run in this stream (CPU, staged artifacts, no pod)

Harness: `eh1_coverage_control.py` · artifact: `eh1_coverage_control.json` · host `SAYED-PC`,
Python 3.13.5, torch 2.11.0, numpy — **CPU only, no GPU, no pod contacted**.
Estimator: `taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`, B = 2000, unit = episode
cluster, 40 clusters / 881 windows.

### 9.1 S1 — committed numbers reproduced before anything was adjudicated

| check | recomputed | committed | |
|---|---:|---:|---|
| `F_flat` from `ref_sel_idx` | **0.8563** | 0.8563 | ✅ |
| `F_flat` from the hier dump | **0.8563** | 0.8563 | ✅ |
| `oracle_in_fan` | **0.2505** | 0.2505 | ✅ |
| paired `H_graft(64) − flat` (reproduced end-to-end) | **+0.2059 [+0.1316, +0.2897]** | +0.2059 [+0.1290, +0.2975] | ✅ |

*(Also recomputed: the **exact expectation** of a uniform random pick is **15.0111**, against E-V5-1's
**15.8738**, which was a *single* random draw per window. Consistent, not identical, and the
difference is draw noise on a distribution with a 108.7 m per-window span — not a discrepancy.)*

### 9.2 ⛔ S2 — I RETRACT MY OWN PRE-REGISTERED ARM

§6.2 pre-registered `R_rand(q)` as *"pick the member of the random subset that ranks best under the
**deployed** score"*, implemented via `v5_v4_windows_reduced.pt::base_rank`. **A self-test written
after the first run refutes the premise.** `base_rank` is **not** a score ranking. It is

```
base_rank[w] = [ the as-trained pick ] ++ [ anchor indices 0..255 with the pick removed ]
```

**verified on 881/881 rows**, and its construction site says so
(`…/2026-07-26-v5-imagination-selection/code/v5_cost_curve.py:109-122`, comment: *"approximate the
base RANKING … using the recorded per-window pick plus fan-order for the rest (documented, not
hidden)"*).

So `R_rand(q)` actually measured *"take the flat pick if it survived the draw, else take the lowest
**anchor index** in the draw"* — an artefact of anchor numbering. **Its first-run numbers
(`R_rand(8) = 16.35` etc.) are retracted and are not quoted anywhere in this file.** The reason I
caught it is that the run produced a number that could not be true — a best-of-8 under an informative
score coming out *worse* than a single random pick — and I measured the tensor instead of explaining
the number.

> **For `RETRACTION_LOG.md` — root-cause class C-NEW: *a tensor's semantics taken from its NAME
> rather than from its construction site.*** `base_rank` is not a rank. Generalises to every reduced
> dump in the program.

**`O_rand(q)` is unaffected** — it never touches `base_rank` — and it is the arm that answers F2.

### 9.3 THE RESULT — coverage explains ~30 % of the harm at tight `q` and NONE of it at loose `q`

`O_rand(q)` = the **oracle inside a uniformly random `q`-subset**, 200 seeds, per-window mean. It is
the *coverage floor*: the best any selector could achieve if restricted to `q` candidates of random
composition.

| q | `O_rand(q)` coverage floor | paired vs `F_flat` 0.8563 | `H_graft(q)` | total harm | **harm NOT explained by coverage** | share |
|---:|---:|---|---:|---:|---:|---:|
| 8 | 2.6018 | **+1.7455 [+1.3518, +2.1658]** ✅sep | 6.6752 | +5.8189 | **+4.0734 [+2.4800, +5.7277]** ✅sep | **70 %** |
| 16 | 1.3642 | **+0.5079 [+0.2761, +0.7457]** ✅sep | 2.6510 | +1.7947 | **+1.2869 [+0.6632, +1.9815]** ✅sep | **72 %** |
| 32 | 0.7645 | −0.0918 [−0.2499, +0.0666] *not sep* | 1.2000 | +0.3437 | **+0.4356 [+0.2260, +0.6565]** ✅sep | **100 %** |
| 64 | 0.4813 | **−0.3750 [−0.5126, −0.2481]** ✅sep | 1.0621 | +0.2058 | **+0.5809 [+0.4128, +0.7563]** ✅sep | **100 %** |
| 128 | 0.3365 | −0.5198 [−0.6493, −0.4048] ✅sep | — | — | — | — |
| 256 | 0.2505 | −0.6057 [−0.7339, −0.4893] ✅sep | — | — | — | — |

> ⭐ **VERDICT: F2 fires PARTIALLY, and the commitment reading survives.**
>
> - **At q = 64 a uniformly random 64-subset contains a candidate that is 0.3750 m BETTER than what
>   the flat selector picks, separated.** Yet committing to the graft's 64 and searching inside is
>   0.2058 m **worse** than flat. **Coverage explains none of the harm at q = 64, and none at q = 32.**
> - At q = 16 and q = 8 coverage *is* binding and costs **+0.51 m** and **+1.75 m** respectively,
>   separated — but that is only **28–30 %** of the measured harm. **~70 % remains unexplained by
>   coverage at every q.**
>
> ⇒ **"Hierarchical commitment costs" is NOT a coverage curve wearing a commitment costume.** The
> pre-registered escape hatch was tested and it does not carry the result. The *quantitative* claim
> does need restating: the +5.82 m at q = 8 is **not** all commitment — about 1.75 m of it is the
> candidate set simply being too small.

### 9.4 What E-H1 could NOT settle, stated plainly

The two arms that would close it completely need tensors **no staged artifact contains**:

| arm | needs | status |
|---|---|---|
| `O_graft(q)` — oracle inside the **graft's** admissible set | `prior [W, 256]` | **not computable here.** Would say whether the graft's subset is better- or worse-*covered* than a random one |
| `H_rand(q)` — best under the **real deployed score** inside a random subset | `sel_score [W, 256]` | **not computable here.** Would isolate the graft's subset *choice* from the restriction |

Both are one `torch.save` key away (§5.6). **This is the concrete reason E-H2 must dump `prior` and
`sel_score`, and it is now an evidenced requirement rather than a preference.**

### 9.5 ⚠️ ESCALATION — a staged artifact is misnamed, and one V5 label is wrong because of it

`base_rank` in `raw/v5_v4_windows_reduced.pt` is **not** a base-score ranking (§9.2). Consequences,
scoped precisely:

- **V5 §5.2's label is wrong.** Its code comment claims *"keep the top-n candidates by the AS-TRAINED
  base ranking"*; what the columns actually vary is *"the as-trained pick plus anchors 0..n−2 in
  index order."* The `n = 1` column (0.8563) is exact; **`n = 2 … 256` are not "the top-n by score."**
- **V5 §5.2/§5.4's CONCLUSIONS survive**, and I say so rather than over-claiming the correction: the
  finding is that *letting the imagination rule consider more candidates makes it worse*, which holds
  for **any** nested family of candidate sets — index order is still a nested family. *"Breadth costs
  −10.66 m"* and *"at every budget, spend none of the imagination budget"* stand.
- **What must change is the LABEL and the dump's key name**, before some later stream does what I
  did. Suggested: rename to `nested_order` with a docstring, or dump the real `sel_score` argsort.

---

## 10. CITATION TABLE

| # | work | link | used for | tier | DEMONSTRATED vs ASSERTED |
|---|---|---|---|---|---|
| 1 | Sutton, Precup & Singh 1999, *Between MDPs and semi-MDPs*, AIJ 112:181–211 | https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf · https://www.sciencedirect.com/science/article/pii/S0004370299000521 | the **Interruption Theorem** — interruption cannot hurt | CONFIRMED | **DEMONSTRATED** (proved). ⚠️ for true `Q^μ`, not a learned one |
| 2 | Sutton, Precup & Singh, NIPS 1999, *Improved Switching among Temporally Abstract Actions* | https://wsai.iitm.ac.in/~ravi/papers/sutton_spr_NIPS99.pdf | the empirical companion to (1) | PROVISIONAL | ⚠️ **PDF unreadable via fetch — not read in this stream.** Listed for the reader, not relied on |
| 3 | Harb, Bacon, Klissarov & Precup, AAAI 2018, *When Waiting is not an Option* | https://arxiv.org/abs/1709.04571 | deliberation cost `η`; **non-monotone, interior optimum**; η=0 ⇒ 1-step collapse | CONFIRMED | **DEMONSTRATED** (η swept 0→0.03; ALE scores) |
| 4 | Bacon, Harb & Precup, AAAI 2017, *The Option-Critic Architecture* | https://arxiv.org/abs/1609.05140 | learnable termination | PROVISIONAL | secondary-source level only; **full text not read** |
| 5 | Nachum et al., ICLR 2020, *Why Does Hierarchy (Sometimes) Work So Well in RL?* | https://arxiv.org/abs/1909.10618 | hierarchy's benefit is mostly **exploration**; shadow-agent control | CONFIRMED | **DEMONSTRATED** (c_train/c_expl decoupling; shadow agent competitive on 3 of 4 tasks) |
| 6 | Zhang et al., NeurIPS 2020, *Generating Adjacency-Constrained Subgoals in HRL* | https://arxiv.org/abs/2006.11485 | unreachable subgoals mislead; constrain the high level's action space | PROVISIONAL | search/abstract level only |
| 7 | Dabney, Ostrovski & Barreto 2020, *Temporally-Extended ε-Greedy Exploration* | https://arxiv.org/abs/2006.01782 | persistence has value (contradicting evidence C5) | PROVISIONAL | abstract level only |
| 8 | Ho & Salimans 2022, *Classifier-Free Diffusion Guidance* | https://arxiv.org/abs/2207.12598 | **guidance strength `w` has an interior optimum**; U-shaped FID | CONFIRMED | **DEMONSTRATED** (Tables 1–2, numbers quoted in §1.2) |
| 9 | Puigcerver, Riquelme, Mustafa & Houlsby, ICLR 2024, *From Sparse to Soft MoE* | https://arxiv.org/abs/2308.00951 | soft routing beats hard token assignment | CONFIRMED (own domain) | **DEMONSTRATED**; transfer to planners is analogy |
| 10 | Shi, Jiang, Dai & Schiele, NeurIPS 2022, *MTR* | https://arxiv.org/abs/2209.13508 | intention queries = per-mode generation, ranked at inference; hard assignment is a **training** rule | CONFIRMED (abstract) / PROVISIONAL (inference detail) | abstract **ASSERTS** stabilisation; the leaderboard result is DEMONSTRATED |
| 11 | *Annealed Winner-Takes-All for Motion Forecasting* | https://arxiv.org/abs/2409.11172 · https://arxiv.org/html/2409.11172v2 | **soft→hard annealing beats hard WTA**: −9.41 % minADE, −36.67 % MissRate on MTR/Argoverse 2 | CONFIRMED | **DEMONSTRATED** (numbers quoted) |
| 12 | Cheng, Chen & Chen 2024, *PLUTO* | https://arxiv.org/abs/2404.14327 · https://arxiv.org/html/2404.14327 | SOTA imitation planner decodes the **full lateral×longitudinal product** and ranks **flat** | CONFIRMED | **DEMONSTRATED** (mechanism + equations quoted) |
| 13 | Dauner et al. 2023, *PDM* (nuPlan 2023 winner) | https://arxiv.org/abs/2306.07962 | propose → **simulate & score** → select flat | PROVISIONAL | search level only |
| 14 | Jaeger, Chitta & Geiger, ICCV 2023, *Hidden Biases of End-to-End Driving Models* | https://arxiv.org/abs/2306.07957 · https://openaccess.thecvf.com/content/ICCV2023/html/Jaeger_Hidden_Biases_of_End-to-End_Driving_Models_ICCV_2023_paper.html | **target-point shortcut**; catastrophic failure on distant TPs; 32→39 DS, 56→84 % RC | CONFIRMED | **DEMONSTRATED** (ablations quoted) |
| 15 | Codevilla, Santana, López & Gaidon, ICCV 2019, *Exploring the Limitations of Behavior Cloning* | https://arxiv.org/abs/1904.08980 | command-conditioned branching; causal confusion; training instability | PROVISIONAL | abstract level only |
| 16 | Zheng et al., ICLR 2025 Oral, *Diffusion-Based Planning for Autonomous Driving with Flexible Guidance* | https://arxiv.org/abs/2501.15564 · https://arxiv.org/html/2501.15564v2 | inference-time classifier guidance for planning; route via MLP-Mixer→adaLN | CONFIRMED (mechanism) | ⚠️ **NO guidance-strength sweep** — case studies only. The gap in §4 |
| 17 | Xing et al., CVPR 2025, *GoalFlow* | https://arxiv.org/abs/2503.05689 · https://arxiv.org/html/2503.05689v3 | **the none/predicted/oracle goal ladder 85.6 / 90.3 / 92.1 PDMS**; the **shadow trajectory** fallback | CONFIRMED | **DEMONSTRATED** (ablation numbers quoted) |
| 18 | Song et al., CVPR 2025, *Don't Shake the Wheel* (MomAD) | https://arxiv.org/abs/2503.03125 · https://arxiv.org/html/2503.03125v2 | **contradicting evidence C1** — closed-loop gain from committing to the previous plan | CONFIRMED | **DEMONSTRATED** (Bench2Drive 16.71→18.11 % SR; TPC 0.81→0.65) |
| 19 | Li et al. 2024, *Hydra-MDP* | https://arxiv.org/abs/2406.06978 | fixed trajectory vocabulary + **simulator-distilled scores**, flat selection | PROVISIONAL | challenge report / secondary source |
| 20 | Zhao et al., CoRL 2020, *TNT* · Gu et al. 2021, *DenseTNT* | https://arxiv.org/abs/2008.08294 · https://arxiv.org/abs/2108.09640 | goal-then-trajectory staging with a **large** goal set + final re-rank | PROVISIONAL | abstracts + secondary sources |
| 21 | *Closing the Navigation Compliance Gap in End-to-end Autonomous Driving* | https://arxiv.org/abs/2512.10660 | planners ignore/misuse route commands; remedy is a **soft** loss weighting | PROVISIONAL | ⚠️ qualitative only — **numbers UNVERIFIED**, PDF not parseable |
| 22 | DARPA Urban Challenge FSM planners (Boss, Junior, AnnieWAY, Odin) | https://www.romela.org/wp-content/uploads/2015/05/Odin-Team-VictorTango%E2%80%99s-Entry-in-the-DARPA-Urban-Challenge.pdf | **contradicting evidence C3** — hard hierarchical mode switching that won | PROVISIONAL | search level only |

**Our own primary artifacts cited (MEASURED):**

| artifact | used for |
|---|---|
| `…/Implementation/incoming/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md` + `raw/v5_hier.json`, `raw/v5_hier_windows.pt`, `raw/v5_v4_windows_reduced.pt` | the whole §0.1 table; §3.3's goal ladder; §9's inputs |
| `…/Implementation/incoming/2026-07-26-v4-restart-lever/V4_RESTART_LEVER.md` + `raw/v4_selgate_ablation.json`, `raw/v4_sel_gate.json` | §5.2's decomposition; `sel_gate` 0.1101→0.1580; `v_goal ≡ v0` |
| `…/Benchmarks & Eval/Implementation/incoming/2026-07-26-bar-a-selector/BAR_A.md` | the 0.4907 feature-only ceiling |
| `stack/tanitad/models/flagship_v4.py:161-195`, `stack/tanitad/models/flagship_v15.py:451-466` | §5.1's read of the deployed selector |
| `Project Steering/MODEL_REGISTRY.md §1.2a` | the n=40 → n=600 power factor (×2.8–3.9) |

---

## 11. DELIVERABLE MANIFEST

**STAGED, NEVER COMMITTED, NEVER PUSHED.** All paths relative to the repo root
`G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/`.

| artifact | location | exists elsewhere? | note |
|---|---|---|---|
| `HIERARCHY_PRIOR_RESEARCH.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-07-27-hierarchy-prior-vs-constraint/` | **repo only** | §0 pre-registration staged **before** any paper was read; §6.2 staged **before** E-H1 ran |
| `eh1_coverage_control.py` | same directory | **repo only** | CPU-only harness; reruns in ~90 s with `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` |
| `eh1_coverage_control.json` | same directory | **repo only** | E-H1 raw output incl. both self-tests and the S2 retraction record |

⚠️ **Everything this stream produced exists in exactly ONE place — the repo working tree — and is
staged.** Nothing lives on a pod: **no pod was contacted.** No GPU was used. No credential was read.

**Reproducing E-H1 from scratch:** the only inputs are two already-staged tensors
(`…/2026-07-26-v5-imagination-selection/raw/v5_v4_windows_reduced.pt`, `raw/v5_hier_windows.pt`) and
`taniteval/taniteval/ci.py`. No GPU, no pod, no network.

### 11.1 ESCALATIONS — these must not sit in a file

1. ⛔ **`base_rank` is misnamed in a staged artifact and one V5 label is wrong because of it** (§9.5).
   It is `[pick] ++ [anchor index order]`, verified 881/881. V5 §5.2's *"top-n by the as-trained base
   ranking"* is not what the columns are. **The E-V5-3 conclusions survive; the label does not.**
   Owner needed: rename the key or dump the real `sel_score` argsort. **I nearly published a false
   mechanism off this name, and the next stream will too.**
2. ⛔ **The headline "+0.0218 m useful as a soft prior" is not the hierarchy alone** (§5.2). Splitting
   two independently staged artifacts on the oracle surface gives **graft ≈ 0.0092 m** and
   **constant-velocity term ≈ 0.0100 m**. `MODEL_REGISTRY.md` and any brief quoting +0.0218 as *"the
   hierarchical prior"* needs the qualifier. **DERIVED/PROVISIONAL** — E-H0b makes it decision-grade
   in one forward pass.
3. ⛔ **F2 is settled enough to act on, and it changes the wording, not the decision** (§9.3):
   coverage explains **0 %** of the harm at q ≥ 32 and **~30 %** at q ≤ 16. *"Never truncate the
   candidate set"* stands; *"the whole 5.82 m at q = 8 is commitment"* does not.
4. ⭐ **The cheap experiment to run next is E-H2, and it is ONE forward pass** (§6.3) — the (λ, τ)
   surface is 42 CPU argmaxes over cached tensors. It is the measurement the published literature
   does **not** have (§4), and it is the only thing that can tell us whether commitment *sharpness*
   or set *truncation* is the cost. It requires the ~10-line dump delta in §5.6.
5. ⚠️ **Our verdict is open-loop-only** (§8). MomAD demonstrates a closed-loop gain from temporal
   commitment; D3 (§5.5) is the design element that anticipates it and currently has **no owner**.
