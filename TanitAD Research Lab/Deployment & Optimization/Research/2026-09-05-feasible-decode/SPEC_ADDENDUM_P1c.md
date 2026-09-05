# SPEC ADDENDUM — P1c, pre-registered BEFORE it ran: does the reward defect SURVIVE the fixed decode?

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-feasible-decode/SPEC_ADDENDUM_P1c.md`
2026-09-05 · Architecture & Inference FlyWheel · ZERO GPU.

⛔ **P1b also FAILED as committed** (`raw/progress_rank_fix_240w.json`,
`raw/progress_rank_fix_400w.json`, both draws agreeing): `progress_v3` reads
ρ(peak_g) **+0.5042 [+0.3920, +0.6064]** on 240 w and **+0.5135 [+0.3755, +0.6353]** on
400 w — **worse than the stock term's +0.2900 / +0.3130**, and C2's ratio is 0.827 / 0.803
against the committed 0.95. Reported as written.

⭐ **And the failure has a clean mechanism, which is what makes the next step obvious
rather than a guess.** `along` is the NET `+x` displacement, so a candidate that swings
sideways is *penalised* by its own lateral excursion. The projection straightens exactly
those candidates ⇒ their `+x` **rises**. So crediting a candidate with its projected
distance *increases* the correlation with the original path's `peak_g`. **The projection
is the right fix for the DECODE and the wrong fix for this REWARD TERM, and those two
sentences are not in tension.**

---

## 1. The question P1c asks, and why it is a different question

The register's defect `D-RL-REWARD-RANK-1` is *"the reward ranks envelope-VIOLATING
candidates above feasible ones within a window's own fan"*. That question presupposes a
population: **windows whose fan contains both.**

MEASURED in P2 (`raw/mu_frontier.json`): after the feasibility-aware decode, the fan's
`envelope`, `kamm_over` and `infeasible` rates are **0.0000 exactly** over 400 windows ×
128 candidates. ⇒ **there are no violating candidates left to rank.** Spearman against a
constant flag is UNDEFINED, not zero — and this SPEC requires it to be REPORTED as
undefined, never averaged in as a null (the `spearman` contract in the tool).

⇒ The residual, weaker question, and the ONLY one that still has a population:

> **Within a fan that is feasible by construction, does the reward still prefer the
> HARDER-driving candidate?** — i.e. ρ(term, `peak_g`) where **both** the term and
> `peak_g` are computed on the **projected** fan.

⚠️ This is a **different question with a different stake** and must never be reported as
if it were the original one: preferring a 0.6 g candidate over a 0.3 g one is a comfort
and margin question; preferring a 4 g one over a 0.5 g one was a safety defect.

## 2. Committed criteria

| id | criterion | reading |
|---|---|---|
| **P1c-C0** (the structural claim) | on the projected fan, per-window Spearman against `envelope` and against `kamm_over` is **UNDEFINED on 100 % of windows**, and the tool reports `n_undefined = n_windows` rather than 0.0 | PASS / FAIL |
| **P1c-C1** | on the projected fan, ρ(`progress` stock, `peak_g`) — reported with its CI. **< +0.05 ⇒ the defect does not survive the decode fix.** ≥ +0.05 ⇒ it survives in the weaker form and a reward change is still owed | PASS / FAIL |
| **P1c-C2** | the same for the composed DEFAULT reward, and for `progress_v2` | reported |
| **P1c-C3** | ρ against `ttc_below` on the projected fan — ⛔ the projection does **not** touch the lead-vehicle geometry, so this one is **predicted in advance to be essentially unchanged**, and if it is not, something other than feasibility moved and the reading is suspect | reported, with the prediction committed |

**Controls (unchanged in form):** C-SELF must read +1.0000; C-CONST must read UNDEFINED on
every window; C-RAND must read the probe's bias floor; and **C-OBJECT** must assert that
the ranked tensor is the *projected* fan by re-deriving `envelope`/`kamm_over` = 0.0000
through `fan_safety.score_paths` — the consumer — rather than trusting the projection's own
bookkeeping.

## 3. What P1c cannot settle, stated in advance

It is a **T0 readout on a fan**, not a training result. It says what a reward *would* rank
if a post-train stage were run on a feasibility-aware decode. It does **not** say that such
a stage would improve driving, and no arm here licenses that claim.
