# "Must we retrain?" — the measured answer, and it is not the one the recall number suggests

**Row:** `D-REFAV1-DRIVE-CEILING` · **Class:** MEASURED, zero GPU
**Instrument:** `tools/decision_ceiling.py` · **Raw:** `raw/decision_ceiling.json`
**Panel:** 282 windows / 141 episodes, step 21,109 (132 GT-turn, 150 GT-straight)

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
