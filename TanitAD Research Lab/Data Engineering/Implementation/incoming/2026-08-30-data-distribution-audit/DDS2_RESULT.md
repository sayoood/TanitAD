# DDS v2 — scored against the design's own cells, and its operating limit

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Status** controls 8/8 PASS ·
⛔ NOT YET validated against trained outcomes · supersedes DDS v1

## What changed from v1

v1 scored coverage over axes **I chose**. v2 scores it over the cell grid the
**PI's own selection design specified** (`2026-08-06-alpamayo-augmentation/
DESIGN.md:49` — equal-weight cells over road-class × day/night × country, 150
cells). The score now measures deviation from a design someone actually chose,
which is defensible in a way an invented axis set is not.

Two **competence** terms sit beside coverage, because coverage alone is satisfied
by a broad-but-shallow subset. Both enter a geometric mean, so **a zero on either
sinks the score**:

* `defect_regime` — mass in fast_arterial + highway (88.7 % of our oracle gap is
  longitudinal, and that is where longitudinal control is exercised).
* `stop_launch` — share of clips with a full stop→launch cycle. ⭐ **A new axis
  the corpus itself is thin on** (11.8 % on the 20 s window), so the score can
  now see a competence gap that speed bands cannot.

⚠️ All ego-derived inputs are computed on the **20 s camera clip**, not the 139 s
egomotion span (MM-C10).

## Controls — 8/8 PASS at equal size (n = 302)

| arm | DDS2 | coverage | diversity | defect | stop-launch |
|---|---|---|---|---|---|
| designed (balanced) | **0.4792** | 0.093 | 0.580 | 39.7 % | 7.6 % |
| random | 0.4681 | 0.140 | 0.530 | 22.5 % | 8.9 % |
| bad: single country | 0.2213 | 0.040 | 0.036 | 30.5 % | 16.9 % |
| bad: r0 gate | 0.0192 | 0.200 | 0.410 | **0.0 %** | 19.5 % |
| bad: all-slow | 0.0187 | 0.187 | 0.424 | **0.0 %** | 18.2 % |
| ⭐ bad: no stop-launch | **0.0173** | 0.140 | 0.488 | 34.4 % | **0.0 %** |

Every bad arm ranks below random; the new stop-launch arm ranks **lowest of
all**, so the added axis can genuinely fail a subset rather than decorating one.

## ⛔ THE OPERATING LIMIT — and it would have been easy not to look for

Reading "8/8 controls pass" as success would have missed the two things in that
table that matter: **the designed arm beats random by only +2.4 %**, and its
**coverage (0.093) is WORSE than random's (0.140)**. A curation tool whose whole
job is ranking good subsets cannot separate designed from random by 2 %.

Tested rather than assumed — the same score, the same arms, varying only n:

| arm size | designed vs random | random coverage |
|---|---|---|
| **302** | **+2.4 %** | 0.140 |
| **600** | **+14.9 %** | 0.313 |
| 1,200 | +12.1 % | 0.573 |
| 2,000 | +11.8 % | 0.727 |

⇒ **The weak separation is a SIZE ARTIFACT, not a score defect.** The design grid
has 150 cells and the coverage term needs ≥ 5 clips per cell, so **750 clips is
the arithmetic minimum to reach coverage 1.0**. Below ~600 the coverage term is
structurally starved and the score is dominated by the competence terms alone.

⭐ **STATED OPERATING LIMIT: DDS2 is admissible for subsets of n ≳ 600. Below
that its coverage term is not meaningful and the score must not be used to rank
candidates.** Anyone applying it to a 300-clip subset would get noise that looks
like a number.

*(The n = 302 control run was forced to that size by the single-country arm — the
largest country in the corpus holds only 302 clips. Equal-size discipline is
right; it just put the control run below the score's own usable range, which is
how the limit surfaced.)*

## What is still NOT established

⛔ **DDS2 has passing controls and NO validation against trained outcomes.** It is
a designed instrument, not a predictor, and no selection decision should rest on
it yet. The validation plan is unchanged and now has a fourth arm worth running:

1. ≥ 3 subsets, **equal size (n ≥ 600) and equal GPU-time**.
2. Rank-correlate DDS2 against a **CONTROL** metric — T1 closed-loop longitudinal
   error, never reconstruction loss (the DINO-WM lesson: 92× data moved fidelity
   +4 % and control 11.5×).
3. Free negative control: the r0-gated arm must train worse on longitudinal.
4. ⭐ **The stop-launch-enriched arm** — which also serves as the discriminating
   experiment for whether thin stop-launch coverage explains "0/881 accelerate".
   One extra arm, two questions.

## Deliverable manifest

| artifact | where |
|---|---|
| `code/dds2.py` — the score | repo, this package |
| `code/dds2_controls.py` — control harness, 8 assertions | repo |
| `raw/dds2_controls.json` — arm results | repo |
