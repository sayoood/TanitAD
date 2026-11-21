# P6: how confident are we that it beats 88.25 and reaches 94.55?

> 2026-10-11. Design: `SPEC_P6_PAPER_FAITHFUL.md`. Pre-registration: `eval/PREREG_P6.md` (DRAFT).

## Verdict

- **Beats SFT-4 (88.25): MODERATE.** The levers point the right way, but none is measured end to end on a full run.
- **Reaches 94.55: LOW.** That needs every lever at the top of its range, and the largest sub-score gap (EP) has no
  measured lever.
- This is a judgment over ESTIMATED ranges, not a computed probability.

## The time question, settled with a MEASURED number

**One A40 runs the paper's 25 scene-epochs at our data scale in 7.04 days.** The final REFe run did exactly that: 2,579,126
sample-passes in 168.97 h (`E:/…/refe-final-2026-10-01/metrics.jsonl`, `summary.json`).

**The 36–39 days of `TRAINING_TIME.md` row C is the FP32 figure.**
- Measured on the A40, FP32 runs at 1.282 s/sample (`train.py:1070`), against 0.2227 s/sample in bf16 + compile.
- That is a 5.76× difference.
- There is no caching difference: the LoRA trunk is re-run on every pass.

**P6 fits about 7 days only with the zero-extra-forward labeller (L1): 6.8–7.1 days, ESTIMATED.**
- If memory forces micro-batch 2: up to 8.2 days.
- With a prefetch forward: 9.2–13.5 days. A 7-day cap then allows 13–19 epochs, at an ESTIMATED −0.1 to −1.2 PDMS.

## Starting point (MEASURED; do not double count)

**SFT-4 u400 reads 88.25 on the full navtest** (`eval/RESULT_SFT4.md`). It already contains three of the levers:
- route fix ON;
- A7 heading repair ON;
- executed-plan PDM labels (a scorer-only fine-tune).

**The gap to 94.55 is 6.30.**

| sub-score | SFT-4 | paper (Table A13) | gap |
|---|---|---|---|
| EP | 81.20 | 92.72 | **11.5** |
| DAC | 96.58 | 99.25 | 2.7 |
| NC | 98.94 | 98.80 | none |
| TTC | 95.63 | 95.74 | none |

**There is headroom in selection.**
- Best of 64 reads **97.07** (MEASURED, REVIEW_7 §2.4).
- The teacher the trajectory head imitates reads 89.81 on navtest (INHERITED, `eval/RESULT_P4_TEACHER_NAVTEST.md`).
- So the scorer has to carry the gap. P6's levers are all scorer- and proposal-side.

## Levers P6 adds beyond SFT-4

Ranges are ESTIMATED unless marked otherwise, and the levers overlap, so the ranges are not strictly additive.

| lever | expected Δ PDMS | evidence class / basis |
|---|---|---|
| un-detached scorer context (P2's lever, inside P6) | +0.5 to +2.0 | ESTIMATED. A re-ranker with NAVSIM's labels but no images recovers only +1.12 [+0.50, +1.76] (REVIEW_7 §0.5, MEASURED), so the missing information is visual. P2 measures this lever first. |
| own proposals, lag 0, 100 % coverage, 25 epochs (vs 0–37 % coverage, lag ≤ 3,451) | +0.5 to +2.0 | ESTIMATED. Anchor: V3r +2.00 [+0.53, +3.57] (REVIEW_7 §3.3, INHERITED). Overlaps SFT-4's realised +3.79. |
| full bank from step 0, no growth | +0.3 to +1.0 | ESTIMATED (REVIEW_7 §3.6) |
| unwrapped yaw (consistent proposal headings) | 0 to +0.5 | ESTIMATED. A7 repair, already inside 88.25, captures part of it. |
| twin-route-consistent labels | 0 to +0.3 | ESTIMATED. The paper's whole augmentation effect is +0.80 (PUBLISHED). |
| command instead of goal points | **−1.0 to +1.0** | ESTIMATED, sign unknown. It may lose the route fix's gain inside 88.25, and it gains on the 3.9 % of tokens whose goal points score 55.45 (REVIEW_7 l. 325). |
| bf16 instead of FP32 | ≈ 0 | INHERITED (REVIEW_7 §1.3) |
| **sum of ranges** | **+0.3 to +6.8** | ⇒ about **88.6–95.1**, with the midpoints summing to about **91.8** |

## What would move the verdict

**Before launch:**
- P2's full-navtest Δ. P2 ≥ 90.5 would make the top half of the range plausible.
- M-LAB's labeller cost, which decides whether L1 is buildable on a 24-vCPU pod.
- G-LIVE-MEM, which decides whether micro-batch 4 fits.

**Unknown, and the reason for "LOW":** no measured lever targets **EP**, the sub-score with the largest gap (11.5 EP
points).
