# SHIP-R8 AMENDMENT 4c -- H2cn-L: the clean-split progress head that also SEES the lon head's own decision (Master Mind, 2026-10-26 ~05:14 CET)

Sealed BEFORE any H2cn-L fit, stage, position, cell or clause exists. Amendments 1-4, 4b and 5 stand; nothing registered is edited.

## 0. Why (disclosed; descriptive, no bar -- `refcv9_l2c/result/H2CN_FDMD_DIAG.txt`)
On the battery draw, S6 compliant, paired episode-cluster bootstrap over episodes: the shipping `H2CN_DEPLOY8` cuts false decels vs
`RETIME_DEPLOY8` **0.137 -> 0.060, -0.0765 [-0.1164, -0.0377]** (s1 -0.0770, both separated) but misses MORE decels, **0.47 -> 0.53,
+0.0642 [-0.0118, +0.1473]** (s1 +0.0610; not separated). The self-sharpened lon head does the opposite (`H-R9-SSGUARD-1`, REPLICATED: MD
-0.08 separated, FD not separated). A progress REGRESSION fitted with an L1 loss predicts typical progress and under-shoots the rare decels
the classification head can see. The cheapest lever that lets the regression use that decision: give it the head's output as an input.

## 1. The arm `H2CNL_DEPLOY8`
* Identical to Amendment 4's H2cn (the `l1c_h2cn_confirm.py` / `h2cn_freeze.py` recipe: S2 and S6 heads, same inputs, same CV grid and
  selection rule for weight decay and epochs, same seeds, same FRESH 300-clip split and compliant rows, same disjointness checks), with ONE
  change: the input vector gains the **softmax of the 6 LON6 logits** (`l2_common.lon6` of `nav_true.tacv6_lon_logits`) of the SAME forward
  (fresh: `F:/l1c/tree/step48852ema_fresh/dump_s0/refcv6_extras.npz`; eval: the battery tree's `dump_s{s}/refcv6_extras.npz`, joined by
  (clip index, ws)). The lon head is the tactical decision head, not the situation classifier; the stage is a post-planner progress
  re-timing, not a goal input (BINDING goal/situation rule: stated, not violated).
* Frozen to `H2CNL_STAGE_FROZEN.pt` and its sha256 written to `REGISTERED_H2CNL_STAGE.sha256` by the Master Mind BEFORE any eval-window output.
* Positions exactly as `H2CN_DEPLOY8` (`h2cn_progress`, running max, `eval_polyline(polyline_geometry(plan8), s8)`).

## 2. Controls (a failed control voids the bars)
* **K-H2CN-REPRO:** the same fitting code with the 6 new input columns DROPPED reproduces the registered H2cn stage's CV record and
  predictions bit-for-bit (proves the logits are the only change).
* **K-JOIN:** every fresh FIT row and every eval row joins exactly one extras row; the joined `ws` / clip index match.
* Amendment 4's K-REG, K-DISJ, K-FIT, K-RELOAD, K-DETERM, K-RETIME and RETIME-TAC identities, re-run for the new stage.
* A mutation arm: the logit columns permuted across FIT episodes before the fit -> K-H2CN-REPRO still holds and the MD bar must NOT pass
  (a regression that "uses" deranged logits is not using the decision).

## 3. Bars (on `H2CNL_DEPLOY8`, both inference seeds, S6 compliant, paired vs `H2CN_DEPLOY8` on the SAME draw)
* **L1** MD separated BELOW `H2CN_DEPLOY8`.  **L2** FD NOT separated above `H2CN_DEPLOY8`.
* **L3** SHIP-R8 clauses (1)-(7)+(1b) as Amendment 4, arm swapped -> OVERALL PASS.
* Beside: the compliance row; MD / FD vs RETIME_DEPLOY8.

## 4. Committed next steps
* **PASS (L1-L3)** -> the PI chooses the shipped stage among H2CN / H2CN-L (and their capped forms if Amendment 4b passes); recommendation
  follows the four-family rows. refcv9's design input: the progress head consumes the lon decision.
* **FAIL L1** -> the decision is not usable by this head class: next = a learned per-window gate between H2cn progress and the
  self-sharpened progress, fitted on the fresh split (needs a SELFSHARP roll on the fresh clips: one GPU roll), pre-registered first.
* **FAIL L2 or L3** -> name the clause; H2CN_DEPLOY8 stands.
