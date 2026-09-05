# PREREG_D-VOCAB-L3_FLOOR_GAP — ERRATUM 2: **WITHDRAWN AT THE PREMISE**

**Issued 2026-09-05.** ⛔ **A separately staged document, not an edit.** The parent and ERRATUM-1
stay exactly as written; this records what happened to them.

---

## The outcome is WITHDRAWN, which is NOT the parent's REFUTED branch

⚠️ **The distinction is the point.** The parent committed three outcomes — SUPPORTED / PARTIAL /
REFUTED — and all three presuppose that the panel RAN. It did not. Calling this REFUTED would
imply a test produced a negative; **the premise dissolved before any test existed.** A prereg whose
motivating fact turns out not to be the fact is **withdrawn**, and saying so is the difference
between a prediction that failed and a prediction that was never about anything.

## Why: the motivating fact is not turning-specific, so a curvature vocabulary cannot be its cause

The parent rested on *"both A/B arms sit **3.6–5.9× below** the trivial `ha`/`ha0_ext` floors **on
turning windows**"*, and argued that a two-level sustained-curvature vocabulary must lose to a
continuous ego-extrapolation floor **there**.

⛔ **MEASURED: the same panel's STRAIGHT windows read 3.77–5.13×**, against turning 3.15–5.74×.
**A deficit equally present on straight road cannot be caused by a sustained-curvature
vocabulary** — straight road needs no sustained curvature to express. The number was real; my
reading of it as *turning-specific* was not, and I never checked the complement.

⭐ **The class, and it is one this programme keeps paying for:** *a statistic quoted on the subset
that motivated the hypothesis, without its complement.* Same family as *"20.5 % turn recall"*
(a claim about a label set), the oracle ceiling quoted as a payoff (#29), and the extrapolation
validated only where its assumption holds (#28). ⇒ **Before attributing a subset effect to a
subset-specific mechanism, read the complement.** One line of arithmetic would have prevented this
document.

⚠️ **Declared, not attributed:** those two panels also differ in `W_KAPPA` (32.149 vs 0.0), so the
deficit's own cause is not settled here either.

## And the panel was independently unrunnable — ERRATUM-1 §E2 did its job

On the representative panel (282 windows), at the MEASURED crossover |gt_κ| > 4e-2: only **10**
turning windows of 266 valid, and the deficit `cl − ha0_ext` is **−0.0628 m** — ⭐ **the shipped
arm BEATS the floor there.** The 50 % target is therefore **−0.0314 m** against an inference-seed
floor of **0.1907 m**: **6.1× inside the planner's own noise, with the wrong sign, at n = 10.**

⇒ **BLOCKED before any GPU**, exactly as ERRATUM-1 §E2 required. Without that blocking step this
panel would have run, produced a confident-looking null, and been quoted.

## What replaced it — the measurement that matters

The de-confounded oracle exists and runs: `plan(goal_keeps_seed=True)` lets the head build its
canonical seed **bit-identically to the shipped arm** and replaces **only** `goal_t`, verify-gated
per window. PARTIAL (3/8 episodes, n = 10 valid; **T0, DIRECTIONAL ONLY**):

| arm | tier | ALL | TURNING (n=4) | STRAIGHT (n=6) |
|---|---|---|---|---|
| `cl` shipped | T1 | 1.5158 | **0.7868** | 2.0018 |
| `cl_oraclegoal` (confounded) | T0 | 2.9696 | 1.9486 | 3.6503 |
| **`cl_oracleseed` (de-confounded)** | T0 | 2.3902 | **1.9486** | 2.6845 |
| `ha0_ext` (INTEGRATOR, M11) | T1 | 1.2483 | 1.8567 | 0.8427 |

⇒ **A perfect goal, delivered honestly, LOSES to `ha0_ext` on turns (ratio 1.049) while the shipped
arm WINS (0.424).** Paired, the perfect goal is worse everywhere: +0.874 / +1.162 / +0.683.

⭐ **And ERRATUM-1 §E3's saturation readout survives de-confounding**, which is what makes it
attributable: `frac at kappa_max` reads `cl` **0.100**, confounded **0.500**, de-confounded
**0.400** — and **on turning windows the seed moves it not at all (0.250 both)**. ⇒ the saturation
is the **GOAL FIELD**, not the missing seed. Adding §E3 before the panel ran is the only reason
this can be said at all.

## Consequences

1. ⛔ **No vocabulary claim may cite this pre-registration.** `M15`/`M19`'s approval rests on its
   own oracle-and-realised evidence, unchanged; it never rested on this document.
2. ⭐ **The goal side is now the LEAST likely place to fix refav1.** A perfect goal makes it
   saturate harder, not steer better. `M20`'s reordering — **`W_KAPPA` first** — is reinforced, and
   the `run_ab*.sh` scripts pass `W_KAPPA = 0`, so **every arm banked through them sits on a planner
   that curves at the clip bound on straight road.**
3. **A replacement pre-registration is owed**, and its object is the **cost geometry**, not the
   goal: a turn-enriched panel, both inference seeds, the saturation readout, and a stated
   power target that clears the 0.1907 m floor **before** it is written.
