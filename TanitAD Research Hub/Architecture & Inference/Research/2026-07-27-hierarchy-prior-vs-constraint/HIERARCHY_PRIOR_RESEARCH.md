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
