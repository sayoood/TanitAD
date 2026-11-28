# PREREG_V8F_PROOF_ADD24_N1: note to card ADD24 (LEADSUP), the slot part of control k3 and an attributable slot quantity

Written 2026-11-15T04:15Z. The card file is unchanged. What the lead had seen when this note was written, stated in full: the preliminary run of the lead reader on the three seed-0 arms (`spatial/leadsup/read/out_prelim_s0/`, a run that names nothing): its controls block, and the seed-0 lines of LS_s0 (closing-speed R2 +0.155 [-0.043, +0.283]; head gap R2 +0.724; rule R3 over the bias-corrected copy +0.065 [-0.052, +0.180]; presence AUROC 0.914). No seed-1 arm existed (they were training), and the quantity defined in section 2 below had not been computed for any arm.

## 1. Finding: the slot part of control k3 fails by construction (reported as written)

Control k3 of the card: with the states permuted across windows, the closing-speed R2 and the rule R3 skill are at most +0.02. Seed-0 run, states permuted among the 429 scored windows:

| arm | closing-speed R2, permuted | R3 skill over the copy, permuted | k3 |
|---|---|---|---|
| LS_s0 | -0.230 | +0.019 | PASS |
| LSH_s0 | -0.320 | +0.095 | FAIL |
| LSD_s0 | -0.336 | +0.055 | FAIL |

By the card as written, no outcome is named for an arm whose control fails. The closing-speed part passes in every arm. The slot part fails because rule R3 (card ADD20, D8: pred = gap + a * T * c + b * X + d) carries the ego term X, the logged ego motion over the next 2.0 s, which is the action input of a rollout and does not come from the states. With the states permuted the copy loses its information and the ego term keeps it, so R3 beats the copy without any state. The control therefore cannot pass for a rule with an action input, and the skill of R3 over the copy mixes the head and the ego term.

## 2. The attributable slot quantity (fixed here, before it is computed for any arm)

* R3c, the rule of the card: pred = g_h + a * (T * c_h) + b * X + d, with g_h and c_h the gap and closing speed of the head at S0, coefficients cross-fitted by episode folds as in card ADD20.
* R3n, the same rule without the closing speed of the head: pred = g_h + b * X + d, cross-fitted in the same folds.
* Slot increment: DELTA = 1 - SSE(R3c) / SSE(R3n) on the 429 scored windows, episode-cluster bootstrap B 2,000, rng 0. It is the share of the error of imagining the gap 2.0 s ahead from the present gap and the ego action that the closing speed read from the state removes.
* Control k3-slot: with the closing-speed output of the head permuted among the scored windows (gap and ego term in place, seed 0, and four more permutations printed), DELTA is at most +0.02.
* Control k3-closing: the closing-speed part of k3 as in the card.
* The k3 slot part of the card (states permuted, R3 over the copy) is computed and printed for every arm as a card-form result with PASS or FAIL; under this note it does not gate.

## 3. Names under this note (marked N1 wherever they are quoted)

* `SUP-SLOT-IMAGINES-N1` if DELTA is above 0 with an interval that excludes 0 at both seeds of the arm type; else `SUP-SLOT-FLAT-N1`.
* `SUP-SLOT-BEATS-ROLLOUT`, the closing-speed names, the gradient question and the cost names: as in the card; the closing-speed names and the gradient question are gated by k1, k2, k3-closing and k4.
* Reporting rule: every verdict and register row first states the card-form result (which arms fail the k3 slot part and that the card names no outcome for them), then the N1 names, each marked N1. R3 over the bias-corrected copy and over the raw copy stay printed and name nothing.

## 4. Unchanged

Arms, labels, reads M1, M2, M4, the cost read, the committed next steps (read with the N1 slot names in place of the slot names of the card), cost and limits.
