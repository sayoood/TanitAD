# E1f RESULT — Outcome C, and it REFUTES the inference that motivated the arm

**MEASURED 2026-07-28**, pod3. Train `rc=0` 11:17:22Z; eval complete.
Artifact `pod3:/workspace/e1f/e1f_frontier.json`.
**Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), 44 held-out episodes /
43 clusters at K=185. `overlapping_holdout_se` used nowhere.
Pre-registration `PRE_REGISTRATION_E1F.md` (`e06281c`), committed **before** the run existed.

## 0. Controls

Base re-rolled to `ADE 0.4747 / acc 0.6815 / l1 0.1775 / dep 0.5877 / junc_dep 0.8414 /
peakXTE 38.944 / OODpeak 1.266` — **identical on every field for the FOURTH consecutive run**
(E1c, E1e-A, E1e-B, E1f). Train-time leak-guard: buffer 102 episodes × held-out 44 → **intersection 0**.
Step-1000 open-loop ADE from the frontier (0.5451) equals the in-training gate's value **exactly** —
two independent code paths agreeing.

## 1. Verdict: BOUND — `primary_ok 0/4`, `guardrails_ok 0/4`, 0 success points

| step | `dep_overall` Δ | `dep_junction` Δ | open-loop ADE@2s Δ [lo, hi] | P1 | P2 | Ga |
|---|---|---|---|---|---|---|
| 1000 | +0.0025 | −0.1270 | +0.0705 [+0.042, +0.103] | ✗ | ✗ | ✗ |
| 2000 | +0.0576 | −0.1459 | +0.0466 [+0.021, +0.077] | ✗ | ✅ | ✗ |
| 3000 | −0.0302 | −0.1568 | +0.0564 [+0.020, +0.104] | ✗ | ✅ | ✗ |
| 4000 | +0.0004 | **−0.2108** | +0.0555 [+0.023, +0.099] | ✗ | ✅ | ✗ |

**Pre-registered OUTCOME C, exactly as named in advance:** P2 (junction) separates-better at 3 of 4
points and strengthens monotonically; **P1 (overall corridor) fails at all four, sitting at zero**
(+0.0025, +0.0576, −0.0302, +0.0004). `primary_ok` is 0/4 because the primary requires **both**.

## 2. ⭐ THE RESULT REFUTES THE INFERENCE THAT MOTIVATED THE ARM

E1d measured that under **α-interpolation**, junction recovery is cheap and monotone while
overall-corridor recovery is expensive and barrier-crossing. The natural next thought — and the one
that launched E1f — was that the expensive half was **dragging down** the cheap one, so removing it
should isolate the good part.

**MEASURED, it does not:**

| arm | buffer | best `dep_junction` | at open-loop cost |
|---|---|---|---|
| **E1c** | full (3,537 rec) | **−0.4982** (step 3500) | +0.2026 |
| **E1f** | junction-only (733 rec) | **−0.2108** (step 4000) | +0.0555 |

⇒ **Training on junctions alone makes junction recovery WORSE — by more than half — not better.**
If the overall-corridor half were interfering, removal should have left junction recovery **≥** its
full-buffer value. It halved it.

**What E1f actually buys:** 42 % of E1c's junction gain at 27 % of its open-loop cost, and **no
overall-corridor gain at all**. That is **more efficient per unit open-loop cost, and strictly smaller
in absolute terms** — a scaled-down version of the full-buffer arm, not a targeted improvement.

⇒ **The cheap/expensive asymmetry E1d measured is a property of MOVING ALONG A PATH between two
models trained on everything. It is NOT a prescription for what to train on.** Logged as **C55**.

## 3. Four levers, four BOUNDs

| lever | experiment | structural reason it stops |
|---|---|---|
| training time | E1c | open-loop plateaus from step 2250 (8 points inside ±0.02) |
| weight space | E1d | separated-WORSE at 5 consecutive interior α; endpoints not linearly mode-connected (**C52**) |
| loss weighting | E1e-A/B | λ sets the asymptote; Ga's lower bound flattens at **+0.023** |
| **the target** | **E1f** | **junction-only gives less of everything; restriction does not isolate the cheap half** |

**There is still no D-A deliverable.** Nothing in four experiments satisfies the pre-registered
success criterion.

## 4. What is NOT concluded

- ⛔ **"Junctions don't matter" is NOT the finding.** P2 separated-better at 3/4 points, monotonically
  strengthening to −0.2108 at a mere +0.0555 open-loop cost. Junction recovery is real and cheap to
  obtain — it is simply **not improved by restricting supervision to it**.
- ⚠️ **Diversity is a live confound and was flagged before the run:** 102 episodes vs 362. A larger
  junction corpus might behave differently; this arm cannot distinguish "junction-only is the wrong
  target" from "733 records across 102 episodes is too little". **Stated, not resolved.**
- ⚠️ `lam_replay = 3.0` was inherited from E1e-A and **not re-tuned** for a 4.8× smaller buffer. A
  buffer change may shift the optimal λ, and this arm cannot see that.
- ⛔ **The pre-registration's Outcome C follow-up — a MIXED buffer with junction over-weight — is NOT
  launched here.** It is a **new pre-registration**, and §2's refutation materially weakens its
  motivation, so it should be re-argued rather than inherited.

## 5. The question that is now the PI's

Four levers — training time, weight space, loss weighting, and the target — have each failed to reach
**Ga** (open-loop ADE@2s not separated-higher), while the closed-loop primary was reachable in three
of them. Across every arm, **every single open-loop CI excluded zero**; the best lower bound ever seen
is **+0.020**.

⇒ **The open question is no longer "which lever" but whether Ga — a STRICT non-regression on
open-loop ADE — is the right guardrail for a closed-loop fine-tune of a frozen-encoder arm at all.**
That is a judgement about what the program is willing to trade, not another experiment.
**No arm should be run against Ga until that is settled.**
