# "Must we retrain?" — the measured answer, and it is not the one the recall number suggests

**Row:** `D-REFAV1-DRIVE-CEILING` · **Class:** MEASURED, zero GPU
**Instrument:** `tools/decision_ceiling.py` · **Raw:** `raw/decision_ceiling.json`
**Panel:** 282 windows / 141 episodes, step 21,109 (132 GT-turn, 150 GT-straight)

---

## ⛔ SUPERSEDED IN ITS RECOMMENDATION — read this first (2026-09-05, same day)

This file's **measurements stand**. Its **recommendation does not**, and it was wrong for a
reason worth stating rather than quietly editing out.

**1. The recall number was measuring the wrong target.** MEASURED by the Master Mind (M15/M16,
commit `4013231`) on a dense panel: restricted to turns the goal vocabulary can actually
*express* (**R ≤ 25 m**, the measured crossover **0.04101**, agreeing with the analytic 0.040),
the shipped head already decodes a curvature-carrying token on **74.4 %** of windows at
**12.0 %** false-turn, **AUC 0.8806**. ⇒ **the "20.5 % turn recall" that motivated this file is
a statistic about R ≈ 1000 m curves that `GOAL_KAPPA_TURN = 0.08` (R 12.5 m) cannot express
anyway.** The head was not failing to see them; the *action space* had no token for them.
⚠️ This also explains why my own lever HURT (`AB_RESULT.md`): it forced κ = 0.08 onto windows
whose true curvature was ~80× gentler. The extra turns were not merely wrong in direction —
they were wrong in *magnitude*, by construction.

**2. The pathway, not the head, is the binding constraint.** `ORACLE_GOAL.md`: a goal taken
from the **true future** produces **ADE 7.4708**, saturating `kappa_max` and `a_max` on ~100 %
of windows.

⇒ **Do NOT launch "Step 2: fine-tune the goal head" on the strength of this file.** The
margin-gap criterion it proposes (0.297 logits) is measured against a *mis-specified target*
and would optimise a head to emit tokens the action space cannot use. The live rungs are the
**vocabulary** (M15, approved) and the **goal→plan pathway** (`ORACLE_GOAL.md`).

⭐ The general lesson, and it is the same family as the estimator rules in CLAUDE.md: **a
recall number is a claim about a LABEL SET, and this one's label set contained events the
model was never able to express.** Before quoting a recall, ask what the model would have had
to emit to be scored correct — and whether it *can*.

---

## The answer, in three lines

| what | verdict |
|---|---|
| **the world model** | ⛔ **do not retrain.** Already MEASURED sound (TURN_L +0.502, TURN_R −0.387, separated). |
| **the trunk / encoder** | ⛔ **do not retrain.** The goal head's turn-vs-hold score reads **AUC 0.712** against a shuffled-label control of **0.506 ± 0.034** — the information is present. |
| **the goal head** | ⚠️ **probably yes — and it is hours, not GPU-weeks.** Not to add capability, but to **sharpen a separation that is 0.30 logits wide.** |

## Why a decision rule alone is unlikely to be enough

A decision rule can only re-threshold a score the head already computed. So the honest bound
on Step 1 is the **ROC of that score** (`max(turn-ish logits) − LANE_KEEP logit`), and a
monotone rule — commit threshold, single-slot bias — is exactly a point on it.

**The frontier (the ceiling of the entire Step-1 family):**

| false-turn budget on straights | max reachable turn recall |
|---|---|
| 0.05 | 0.174 |
| **0.073 (argmax's own rate)** | **0.220** — argmax already achieves **0.205** |
| 0.10 | 0.265 |
| 0.15 | 0.348 |
| 0.20 | 0.470 |
| 0.30 | 0.644 |

⭐ **Argmax is NOT badly mis-thresholded.** At its own false-turn rate the best possible
threshold buys **+0.015** recall. Every real gain comes from *moving along* the frontier —
i.e. from **buying recall with false turns**, at roughly **0.49 extra false-turn rate per
unit of recall**.

## ⛔ The near-miss framing is true and misleading — the control is what shows it

Of the **105 missed turns**, **100 % lose by ≤ 2 logits** and **none by more than 4**
(p50 **1.252**). Read alone that says *"a calibration failure — one scalar fixes it."*

**It does not**, and the same-breath control is why:

| | n | LANE_KEEP wins by (p10 / p50 / p90) |
|---|---|---|
| **missed turns** (head wrong) | 105 | 1.063 / **1.252** / 1.652 |
| **correctly held** (head right) | 139 | 1.117 / **1.549** / 1.760 |

**Median gap: 0.297 logits**, with distributions that overlap almost completely (missed p90
1.652 > held p10 1.117). The score is **nearly flat**: LANE_KEEP wins by ~1.1–1.8 logits
whether the road turns or not. ⇒ **a uniform threshold shift flips holds and turns nearly
together**, which is precisely the steep price the frontier shows.

⚠️ Without the correctly-held control, "every miss is within 2 logits" would have been
banked as evidence that Step 1 suffices. It is the same family as the CLAUDE.md probe traps:
a true number whose implication is wrong because its complement was never measured.

## Composing with what is already banked

`D-REFAV1-CCOS-ARMS` / `D-REFAV1-SURFACE-PAIRED` MEASURED that when the planner *does*
execute more turns (`ccos`), **every family degrades** (ADE 0.5474 → 0.7116). Put beside the
frontier:

> Step 1 buys recall **only** by buying false turns, and false turns are exactly what the
> paired measurement showed to be net-harmful. ⇒ **Step 1 is real but bounded, and unlikely
> on its own to make refav1 drive.**

⇒ **Step 2 is indicated and is now precisely specified:** fine-tune **only the goal head**
(trunk and WM frozen, freeze asserted by parameter count) with a class-balanced / focal loss
against the v7.2 tactical labels. The objective is **not** more turns — it is a **wider
margin gap**, and this instrument measures exactly that (`median_margin_gap_logits`,
today **0.297**) as its success criterion, alongside the four families at T1.

## What this does NOT say

* It does **not** say Step 1 is worthless — recall 0.205 → 0.303 at a false-turn rate of
  0.107 is a real move, and its **plan-level** effect was under measurement (the paired A/B)
  when this was written. The frontier bounds it; it does not evaluate it.
* It does **not** license retraining the WM or the trunk. Both are measured adequate.
* ⚠️ One checkpoint, one panel, `n_gt_turn = 132`. `H-ESTIM-SEED-1` applies to any
  arm-versus-arm claim built on top of this.

**Controls:** shuffled-label AUC **0.5055 ± 0.0336** over 200 shuffles (expect 0.5) —
without it a high AUC could be a scoring artefact; the argmax operating point is located on
the ROC and reproduces the banked **0.2045 / 0.0733** exactly; `n` printed for every cell.
