# VERDICT: PREREAD_V8F_STOPTAU = TAU-TRADES (BAL-G) and TAU-NULL (BAL-L): hardening is closed as a lever (Master Mind, 2026-11-11 ~19:40)

Card frozen 19:04:31 (sha256 `e940a0a2`, re-hashed equal after the run). Read by a Sonnet agent (`stoptau_read.py` on STOPHARD's machinery; `TAU_STOPTAU.json`, `RESULT_STOPTAU.json`). Box 2 was used for read-only pulls only. Re-checked by the Master Mind: `test_stoptau.py` 24 passed, `controls_ok` true, outcome and cells read from the JSON. Tier: **T1-kin first tick, never driving performance**. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED.

## Threshold, from training windows only
141 WMACT train clips (0 clip ids shared with eval-141), 2,396 first-tick windows on the eval window rule, 114 GT stop-point windows, labels from the STOPBAL argv's `--join-train` file. tau by the card's rule, no fallback: NULL arms 0.50 (goal and lon); **BAL-G goal 0.80** at both seeds; **BAL-L lon 0.75** at both seeds. At tau the hard stop fires far less often (BAL-G s0: 11 stop-point and 11 other windows, against 26 and 62 at 0.5).

Controls: replay at 0.5 / argmax equals STOPHARD's transform bit for bit (all replay cells equal `RESULT_STOPHARD.json`); training and eval clips disjoint; planted vectors give the planted tau; 51 eval windows through the new batch code match STOPDEC's probabilities to 1e-7.

## Cells: tactical ADE (knots 11-30), TAU minus OWN (m)
| arm | tau | S: 117 stop-point windows | A: 2,351 windows |
|---|---|---|---|
| BAL-G s0 | 0.80 | -0.0088 [-0.0205, -0.0014], separated, below the 0.022 floor: not a gain | -0.0005 [-0.0011, -0.0001], better side |
| BAL-G s1 | 0.80 | +0.0040 [-0.0001, 0.0090], not separated | +0.0006 [0.0000, 0.0013], above the 0.00056 floor: cost |
| BAL-L s0 | 0.75 | -0.0040 [-0.0098, 0.0008], not separated | -0.0000, not separated |
| BAL-L s1 | 0.75 | -0.0030 [-0.0075, 0.0010], not separated | +0.0007 [-0.0000, 0.0015], not separated |

## Outcome, as committed
* **BAL-G: TAU-TRADES** (A a cost at s1). The TAU-NULL row also holds (S not a gain at either seed); the card gives both the same next step.
* **BAL-L: TAU-NULL.**
* Committed next step for both: **hardening is closed as a lever, with its measured rows** (STOPHARD: +0.04-0.09 m at 0.5 / argmax, with an all-window cost; STOPTAU: no gain once the false stops are held to the NULL rate). **The second dose proceeds unhardened, and the perception card goes to the PI.**

## What the chain of reads established (STOPBAL -> STOPORC -> STOPHARD -> STOPTAU)
1. O plans a stop when told to: a correct stop decision is worth about 1.5 m tactical ADE on the stop-point windows (STOPORC).
2. Loss weighting moves T's operating point along an unchanged ROC (STOPBAL, AUROC within its floor at every read).
3. Making T's own decision firmer recovers at most 3-6 % of the 1.5 m, and nothing once its false stops are held to the NULL rate (STOPHARD, STOPTAU).
4. So the 1.5 m sits in the stop-point windows T does not flag. The lever with the largest measured headroom is T's **discrimination** of stop points, the perception card. The second dose stays the committed training arm, with its pre-registered prediction BAL-FP-COSTS.
