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
