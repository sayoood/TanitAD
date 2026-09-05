# ESCALATION — the fix, PRICED before any GPU is spent

**Row:** `D-REFAV1-VOCAB-DESIGN` · **Class:** MEASURED, **zero GPU**
**To:** the PI / Master Mind · **Date:** 2026-09-05
**Instrument:** `tools/vocab_design.py` · **Raw:** `raw/vocab_design.json`
**Panel:** the dense bank, **4520 windows / 141 episodes** (266 near-stationary windows
excluded, see §0), 644 real turns (`|κ| > 1e-2`, R ≤ 100 m), ckpt 21,109.

---

## The ask, in one line

⭐ **Give the lateral goal vocabulary THREE sustained curvature magnitudes instead of one.**
MEASURED payoff: **100 % of real turns become expressible instead of 38.7 %**, and the median
curvature error on a turn falls **3.7×** — at the same RMSE as today, for a change to five
constants and one function.

## §0 ⛔ A data defect that had to be fixed first, and it bit this instrument

`gt_kappa = yaw_rate.mean / speed.mean.clamp_min(0.5)`. Below ~1 m/s that divides by the
**floor**, and the result is not a curvature. MEASURED on the dense panel: **13 windows
(0.27 %)** read `|κ| > 0.2`, **92.3 % of them at v0 < 1 m/s** (median v0 **0.366 m/s** against
**10.26** elsewhere), with a maximum of **3.565 1/m = R 0.28 m** — narrower than the car.

⇒ Those 13 windows carried **the entire RMS**: `RMS|κ|` **0.15827** raw against **0.02192**
clipped to `GOAL_KAPPA_MAX`. My first run of this instrument reported the raw figure and its
table was meaningless. ⇒ **the tool now excludes `v0 < 1 m/s` and clips to `GOAL_KAPPA_MAX`,
and prints both counts.**

⚠️ **Blast radius, checked rather than assumed:** the other results in this package are
**rank-based or already clipped** and are therefore unaffected — `vocab_fit` reports
fractions, `threshold_sweep` reports AUC and thresholds, `kappa_head` already clipped its
target to ±0.2, `soft_kappa` to ±0.08. Only this instrument's RMSE columns were swamped.
⭐ **Generalisable:** any `yaw_rate / speed` curvature metric anywhere in the programme needs a
speed floor on the WINDOW, not just inside the ratio.

## §1 What the shipped vocabulary actually buys

Scoring: each design is given an **ORACLE** that always picks its best available token; the
residual is that token's sustained curvature (scaled by the `TURN` profile's **0.667** duty
cycle over the 6 s horizon) against the road's constant curvature.

| design | RMSE all | **medAE on turns** | **turns expressible** | straights given κ |
|---|---|---|---|---|
| **L=0 — `LANE_KEEP` only (FLOOR)** | 0.01785 | 0.01942 | **0.0000** | 0.0000 |
| L=1 κ=0.01 (R 100 m) | 0.01600 | 0.01276 | 1.0000 | 0.1638 |
| L=1 κ=0.015 | 0.01526 | 0.00942 | 1.0000 | 0.0900 |
| ⭐ L=1 κ=**0.02** (R 50 m) | 0.01460 | **0.00609** | **1.0000** | 0.0501 |
| L=1 κ=0.03 (R 33 m) | 0.01350 | 0.00897 | 1.0000 | **0.0000** |
| L=1 κ=0.04 | 0.01257 | 0.01097 | 0.7453 | 0.0000 |
| ⛔ **L=1 κ=0.08 — SHIPPED** | **0.01015** | **0.01551** | **0.3866** | 0.0000 |
| L=1 κ=0.12 | 0.00977 | 0.01717 | 0.2578 | 0.0000 |
| L=2 at corpus quantiles | 0.01198 | 0.00702 | 1.0000 | 0.1104 |
| ⭐⭐ **L=3 at corpus quantiles** | **0.01032** | **0.00421** | **1.0000** | 0.1365 |
| L=4 at corpus quantiles | 0.00951 | 0.00292 | 1.0000 | 0.1458 |
| L=6 at corpus quantiles | 0.00850 | 0.00215 | 1.0000 | 0.1507 |
| **CONTINUOUS κ (CEILING)** | **0.00000** | **0.00000** | 1.0000 | — |

⛔ **Controls:** the CONTINUOUS ceiling reads **exactly 0.0** (else the scorer is broken); the
`L=0` floor reads **exactly** the road's own RMS curvature to 1e-12 (else the scorer is not
measuring what it claims); the SHIPPED row is present in the table by assertion, not by eye.

### The reading, and the trap inside it

⚠️ **RMSE and medAE RANK THE DESIGNS DIFFERENTLY, and quoting either alone gets it wrong.**
`κ=0.08` has the **best** single-magnitude RMSE (0.01015) and nearly the **worst** medAE on
turns (0.01551). RMSE is dominated by the rare sharp turn, which 0.08 serves; the median turn
is gentle, and 0.08 serves it barely better than doing nothing (floor 0.01942).

⇒ **A single magnitude cannot serve both, and that is exactly the argument for more levels.**
**L=3 dominates:** it matches the shipped vocabulary's RMSE (0.01032 vs 0.01015) **and** cuts
the median turn's curvature error **3.7×** (0.00421 vs 0.01551) **and** takes expressible turns
from **38.7 % to 100 %**.

⭐ **The shipped vocabulary recovers only ~20 % of the available curvature error on a typical
turn** (floor 0.01942 → shipped 0.01551, against a ceiling of 0). That is the size of the hole.

## §2 The proposal

1. **`GOAL_KAPPA_TURN` becomes a level SET** — 3 magnitudes sited at the corpus's own
   `|κ|` quantiles among real turns, so the sizing is DERIVED from the road rather than chosen.
   The head's 8 lateral slots already include **3 tokens that are never labelled and never
   emitted** (`LANE_CHANGE_L/R`, `ABORT_LC` — MEASURED by the predecessor), so the extra levels
   need **no new parameters and no wider head**.
2. ⛔ **Flag-gated, exactly as `lat_logit_bias` was.** A zero-flag arm must be **bit-identical**
   to today, pinned by a test with a same-breath control that must differ — the pattern
   `stack/tests/test_refa_v1_lat_bias.py` already establishes. Without that, this change breaks
   bit-parity with every banked refav1 number.
3. **Then re-measure the head against the NEW crossover.** The crossover moves with the level
   set (`κ_turn/2`: 0.040 today, measured 0.04101), and the head's decode statistics must be
   re-read against it — `D-REFAV1-TURN-THRESHOLD` shows how strongly they depend on it.

## §3 ⚠️ What this does NOT claim

* It is a **VOCABULARY-ADEQUACY BOUND under an ORACLE chooser**. It bounds the system from
  above; it does **not** promise the head will select the right level. Head selection is
  measured separately: AUC **0.8806** and recall **0.744 @ 0.120** at the crossover
  (`raw/threshold_sweep_dense.json`), which is why this is worth doing — but the two must be
  multiplied, not confused.
* It does **not** predict an ADE or a four-family improvement. A finer goal changes the field
  the planner tracks; whether iCEM then tracks it better is a **plan-level** question that the
  shortcut explicitly does not answer (`GATE_ON_REAL_MODEL.md` §3: the lookup shortcut does not
  hold on turn windows). It must be re-planned and measured.
* It does **not** touch **gate 2**. Under the shipped `cos` metric a correctly decoded turn is
  refused on **38/38** windows (`D-REFAV1-DRIVE-GATE2`, INHERITED). ⇒ **a finer vocabulary
  without `ccos` — or without the `ccos` hold-branch fix — changes nothing that reaches the
  wheels.** The two must land together.
* One checkpoint, one corpus. The level siting is fitted to **this** corpus's curvature
  distribution and must be re-derived, not copied, for another.

---

## §4 ⛔⛔ SELF-CORRECTION: the REALISED payoff is 2.3 %, not 3.7× — the oracle bound is not the expected gain

**Instrument:** `tools/realised_kappa.py` · **Raw:** `raw/realised_kappa.json`
**Same panel:** 4520 windows / 141 episodes, 644 real turns (`|κ| > 1e-2`).

§1's table is an **ORACLE** bound and is labelled as one — but a reader will take
*"100 % expressible, median error −3.7×"* as the payoff, and **it is not.** The oracle never
picks the wrong token. The shipped head does. This composes them: for each candidate `κ_turn`,
apply the **SHIPPED head's ACTUAL argmax decode**, push it through the real
`canonical_controls` profile, and score the goal curvature the planner would actually be given.

| design | crossover | **medAE on turns** | RMSE all | **turns goaled correctly** | straights given κ |
|---|---|---|---|---|---|
| **ZERO — the goal never turns (FLOOR)** | — | **0.01942** | 0.01785 | 0.0000 | 0.0000 |
| realised, κ_turn = 0.02 | 0.0100 | 0.01831 | 0.01709 | **0.2811** | 0.0965 |
| ⭐ realised, κ_turn = **0.04** (best) | 0.0200 | **0.01734** | 0.01767 | **0.2811** | 0.0965 |
| ⛔ realised, κ_turn = **0.08 (SHIPPED)** | 0.0400 | **0.01775** | 0.02219 | **0.2811** | 0.0965 |
| realised, κ_turn = 0.12 | 0.0600 | 0.01942 | 0.02931 | 0.2811 | 0.0965 |

⛔ **Controls, all passed:** the ZERO floor is in the table (an oracle can never show that a
vocabulary is worse than never turning — only a realised decode can); the SHIPPED row is
asserted present; and a **RANDOM-DECODE control** with the same marginal rate is worse than the
real head at both magnitudes (**0.02035 ± 0.00047 vs 0.01831** at 0.02; **0.02199 ± 0.00058 vs
0.01775** at 0.08), so the head IS using information.

### What this changes

1. ⛔ **Tuning `κ_turn` ALONE buys 2.3 %** (0.01775 → 0.01734), not 3.7×. Against the
   never-turn floor the entire shipped lateral goal is worth **8.6 %**, and the best reachable
   constant **10.7 %**.
2. ⭐ **The reason is visible in one column: `turns goaled correctly` is 0.2811 for EVERY
   `κ_turn`.** `κ_turn` does not touch the head's logits, so the decode is unchanged; only
   **28.1 %** of real turns receive a correctly-signed sustained goal whatever magnitude is
   commanded. **The binding term is the head's RECALL at the crossover, not the magnitude.**
3. ⚠️ **And the two levers fight each other.** Lowering `κ_turn` lowers the crossover, which
   makes more road expressible but asks the head to decide where it is weaker: argmax recall is
   **0.744 @ false 0.120** at crossover 0.040, and **0.388 @ 0.107** at crossover 0.010. That is
   why the realised optimum (0.04) sits well above the oracle optimum (0.02) — and why neither
   table alone can choose the constant.

⇒ **The escalation stands but its framing is corrected: a finer sustained curvature is
NECESSARY (it removes a ceiling that no head can beat) and NOT SUFFICIENT (the head only goals
28.1 % of real turns correctly).** Both must move. The ordering is still vocabulary-first,
because until the ceiling is removed a better head has nothing to command — but nobody should
expect the constant alone to make refav1 drive.

⚠️ **Recorded as a self-correction rather than quietly folded in.** §1 was banked (commit
`e0b1276`) before this table existed, and its oracle numbers are correct as an upper bound; what
was missing was the composition with the head's own selection. *Same class as the retraction
this package already logged — a true number whose implication is wrong because a second factor
was never multiplied in.*
