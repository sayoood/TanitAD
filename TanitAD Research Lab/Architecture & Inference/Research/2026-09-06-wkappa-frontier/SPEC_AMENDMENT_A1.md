# SPEC AMENDMENT A1 — a PRE-REGISTRATION DEFECT, recorded rather than hidden

`…/2026-09-06-wkappa-frontier/SPEC_AMENDMENT_A1.md` · 2026-09-06 · written **before** any arm of
this package produced a result (batch 1 relaunched at the timestamp in `raw/AS_LAUNCHED.txt`;
first plan ~53 s, full arm ~37 min).

⛔ **NO CRITERION IN `SPEC.md` §3 IS CHANGED BY THIS AMENDMENT.** The PASS / FAIL / PARTIAL
clauses, the floors, the estimator, the tier stamps and the `I22` reporting rule all stand exactly
as written. What changes is **which rungs the pre-registered protocol is executed on**.

## 1. The defect

`SPEC.md` §2.1 correctly derived that **no NEW scalar `W_KAPPA` rung between 7 and 15.11 is
resolvable** on curvature at the 0.00200 inference-seed floor, and refused to run three arms on
that basis. That derivation is unchanged and remains right.

⛔ **But it asked the wrong follow-up question.** Having refused to add scalar rungs, it never
asked whether an **ALREADY-BANKED** scalar rung already satisfies the PASS clauses. It planned a
sweep for a frontier point that may already be on disk.

## 2. What the free measurement found

`raw/feas_census_all.txt` — **ZERO GPU**, nine dumps, **ONE generation** (the
`D-REFAV1-CG-ZEROVIOL-SCOPE` fix: the vacuity comparison and the rate a claim rests on may not
come from different files), instrument `assert_feasible`, `FEAS_VMIN=2`, `n = 27`:

**`wk1`, `wk3` and `wk7` each read `kamm_over` 0.0000 with `max|kappa|` 0.0800 = 4.0× the `M58`
vacuity gate — and their `turn_left` recall is 0.3636 / 0.3636 / 0.2727, all above the exact
0.00000 recall floor.**

Their feasibility had **never been measured**: the banked census
(`BEST_FEASIBILITY_SEEDS.md` §3) covers `cos_wk`, `wk15`, `wk151`, `combined`, `best`, `bestlad`
— **the three rungs are absent from it**, which is why the "six configurations" census did not
see them.

⇒ **`wk7` satisfies all three PASS clauses at seed 0**: curvature MAE **0.033522**, i.e.
**0.006561 below the `ha0` straight floor = 3.3× the 0.00200 curvature floor**; `turn_left`
recall **0.2727 = 3 of 11**; `kamm_over` **0.0000** non-vacuously.
⇒ `wk1` / `wk3` turn MORE (4 of 11) and are also zero and non-vacuous, but their curvature sits
**0.000677 / 0.001346** below the floor — **inside the 0.00200 resolution**, so they are **NOT**
separated below the straight line and do not clear PASS clause 1.

## 3. The amendment

**The pre-registered replication protocol (`SPEC.md` §2.3, ≥ 3 inference seeds) is executed on
`W_KAPPA = 7` — the rung that qualified — in addition to the by-goal rungs.**
`wk7` seed 0 is banked (`…/p4out/rec_wk7.json`); this package runs **seeds 1 and 2**.

Batch 1 is therefore `wk7_s1`, `wk7_s2`, `t5_s0` (`w_turn` 5.03748), `t10_s0` (`w_turn`
10.07497). The `gkappa` / `wk15` seed replicates move to batch 2.

⚠️ **Why this is not a moved goalpost.** The qualifying evidence for clause 1 (curvature) and
clause 2 (recall) was **already public in `A2_WKAPPA_SWEEP_RUNGS.md`, read before `SPEC.md` was
written**; only clause 3 (feasibility) was new, and it was measured by an instrument and a gate
both fixed in advance. Nothing was relaxed to let `wk7` through — `wk1` and `wk3` are reported
here as **failing** clause 1 on the very floor this SPEC adopted, which is what a criterion that
still bites looks like.

## 4. ⭐ What the same free census also settled, and it is the `I22` table

| `W_KAPPA` | `kamm_over` (n=27) | `max|kappa|` | vacuity 0.02 | `peak_g` max | **`turn_left` recall** | curv MAE |
|---|---|---|---|---|---|---|
| 0 (`ccos_argmax`) | ⛔ **0.2963** | 0.2000 | pass | **3.262** | 0.3636 | 0.055369 |
| **1** | **0.0000** | 0.0800 | ✅ 4.0× | 0.371 | **0.3636** | 0.039406 |
| **3** | **0.0000** | 0.0800 | ✅ 4.0× | 0.332 | **0.3636** | 0.038737 |
| ⭐ **7** | **0.0000** | 0.0800 | ✅ 4.0× | 0.332 | ⭐ **0.2727** | **0.033522** |
| 15.11 (`wk15`) | 0.0000 | 0.0800 | ✅ 4.0× | 0.332 | ⛔ **0.0000** | 0.030982 |
| 151 (`wk151`) | 0.0000 | 0.0166 | ⛔ VACUOUS | 0.158 | ⛔ 0.0000 | 0.038019 |
| by-goal `w_turn=0` (`gkappa`) | ⛔ **0.0370** | 0.2000 | pass | **2.176** | 0.3636 | 0.040167 |
| the human `g` (CONTROL) | **0.0000** | 0.1701 | — | 0.373 | — | — |

⭐ **The friction-circle zero is bought at `W_KAPPA = 1`, not at 15.** Everything the scalar
sweep spends between 1 and 15 buys curvature and **costs turn recall**; it buys **no additional
safety**, because the safety statistic is already an exact structural zero at `W_KAPPA = 1`.
⇒ the census that produced *"`best` is the only replicated, non-vacuous zero"* was scoped to a
set of arms that **excluded the three cheapest ones**.

⛔ **And the goal-conditioned arm is the counter-example that proves the term is doing safety
work**: `gkappa` sets `w_turn = 0` and the zero **breaks** (0.0370, `peak_g` max **2.176**,
`max|kappa|` at the 0.2000 cap). ⇒ **on turn-goal windows the curvature weight is simultaneously
the road-tracking lever, the turn-suppression lever AND the friction-circle lever.** That is why
setting it to zero cannot be the frontier, and it is measured, not argued.
