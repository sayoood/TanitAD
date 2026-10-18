# EP-1 step 1: why P2's pick loses ego progress (descriptive, FIT table only)

Agent for the Master Mind, 2026-10-18. CPU only, nothing fitted, **navtest not read in this step**.

**Tier / data / estimator / class.**
- Fit table: P2's OWN navtrain held-out 64-proposal table, `D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz`, sha256 `8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b` (verified), 1,705 tokens / 24 logs. It carries all 64 proposals' true sub-scores, the scorer's six logits per proposal, and the 8-pose plans. Package `raw/2026-10-18-p2-heldout-table/` (scorer = P2 `model_final`, fan = P2's; disjoint from navtest by logs and tokens and from P2's training bank: that package's `disjointness_proof.json`, PASS; inherited, not re-run here).
- Tier: SELECTION tier on navtrain held-out (NAVSIM v1.1 PDMS of the selected proposal, looked up). Never an endpoint.
- Estimator: paired log-cluster bootstrap (`zero_tail.Boot`, vendored byte-identical, sha256 `149c3e97...`), 10,000 resamples, seed 20260927. It answers "another draw of LOGS" only (24 logs here, so wide). One training seed, deterministic inference (H-ESTIM-SEED-1).
- Class: MEASURED (ours), `raw/diag.json`, `raw/diag2.json`; code `code/ep_diag.py`, `code/ep_diag2.py`.
- Gates: PDMS = NC*DAC*(5EP+5TTC+2C)/12 reproduces the table, max abs diff **0.0**, DDC-in-product control **1.0** (has teeth); the planner rule reproduces `pick` on **1,705 / 1,705**; PDMS at pick **90.64 [88.28, 92.71]**, oracle **98.74 [98.06, 99.28]**, oracle - pick **8.10 [6.26, 10.19]** (x100).
- Definitions: *safe* = NC = DAC = TTC = 1; *geometry-safe* (gs) = NC = DAC = 1 (EP is exactly 0 on 98.4 % of non-gs proposals; the rest are NC = 0.5); plan length L = distance of pose 8 (4 s) from the origin (m).

## (a) The fan is NOT the limit: the pick gives up progress that a safe plan in the same fan holds

1,698 of 1,705 tokens hold at least one safe proposal (mean 48.5 of 64). On those tokens:

| | value [95 %] |
|---|---|
| EP of the best SAFE proposal | **0.978 [0.966, 0.988]**; = 1.000 on 74.5 % of tokens, >= 0.95 on 82.7 % |
| EP at P2's pick | **0.851 [0.823, 0.881]** |
| gap | **0.127 [0.102, 0.150]** |
| gap when the pick is safe (1,604 tokens, 76.9 % of the gap) | 0.103 [0.083, 0.119] |
| gap when the pick is unsafe (94 tokens, 23.1 % of the gap) | 0.531 [0.281, 0.727] |
| PDMS x100 if the best-safe-EP proposal were picked | 98.70 vs pick 90.64: **+8.07 [6.23, 10.12]**, i.e. 99.6 % of the oracle gain (+8.10) |

The pick sits at length percentile **0.41** of its token's fan (median 0.33; **34.7 %** of picks in the shortest quintile against 20 % by chance), the best-safe-EP proposal at 0.76; the pick is **2.9 m shorter** (18.0 vs 20.9 m at 4 s).

## (b) EP-head calibration: the head is an UNCONDITIONAL expectation, and the rule counts the violation risk twice

- On ALL 64 proposals the EP head is calibrated by within-token length quintile (pEP / true EP: 0.678/0.676, 0.754/0.753, 0.759/0.765, 0.734/0.746, 0.671/0.675), and `E[NC*DAC*EP]` realised **0.7216** vs mean pEP **0.7186**. On violators it predicts 0.428 against a true 0.010; on gs proposals 0.782 against 0.877. So pEP is "expected progress with violators scored 0", i.e. it already contains the violation probability.
- The planner then multiplies by `pNC*pDAC` again (`planner.py:558-561`). On gs proposals the head **under-predicts progress, and by more the longer the plan**: bias pEP - EP = **-0.043 [-0.086, -0.009]** (shortest within-token quintile), -0.094 (middle), **-0.168 [-0.195, -0.140]** (longest quintile). Predicted `pNC*pDAC` falls from 0.936 to 0.837 across the same quintiles (realised 1.000 on gs by definition; calibrated on all proposals, 0.903->0.699 vs 0.905->0.700), so the rule's effective EP term `pNC*pDAC*pEP` on gs proposals reads 0.668 (shortest quintile), 0.750, 0.758, 0.733, 0.699 (longest) while the true EP given gs reads 0.747, 0.862, 0.907, 0.938, **0.964**: the shortfall of the term against the truth grows from 0.08 to **0.27** with plan length. That is a length penalty.
- Dividing the head's violation probability back out repairs it: `pEP/(pNC*pDAC)` vs true EP|gs by quintile 0.751/0.747, 0.863/0.862, 0.898/0.907, 0.918/0.938, 0.939/0.964 (bias +0.005 ... -0.025).
- Ranking power on gs proposals, within-token Spearman with true EP (n = 1,477): plan length **0.904**, `pEP/(pNC*pDAC)` **0.791**, raw `pEP` **0.363**. Within a token, true EP among safe plans is almost a function of plan length; the raw head hardly ranks it.
- TTC head: also calibrated on all proposals (p / true by quintile 0.955/0.960 ... 0.693/0.695), under-predicts on gs (0.797 vs 0.850 at the longest quintile). C head: constant 1.0.

## (c) Risk aversion of the rule

On 1,556 of 1,705 tokens a truly safe plan at least 0.25 m longer than the pick exists; on **1,039 (61 %)** the best such plan is truly better. Swapping to it would give **+7.12 [5.17, 9.24]** PDMS x100 (EP +0.123, +4.6 m).

Margin `ln S(pick) - ln S(alt)` over those 1,039 tokens, exact decomposition (check max abs error 8.9e-16; the W term allocated by the logarithmic mean): mean 0.423, median 0.212; 18.9 % below 0.05, 49.1 % below 0.2.

| term | mean contribution to the margin | tokens where it favours the PICK | tokens where it is the largest term |
|---|---|---|---|
| pNC | +0.137 | 94 % | 204 |
| pDAC | +0.120 | 92 % | 464 |
| TTC term | +0.117 | 95 % | 363 |
| **EP term** | **+0.049** | **68 %** | 8 |
| C term | +0.000 | 76 % | 0 |

- In **68 %** the EP head itself ranks the pick above a truly better, faster plan (alt true EP 0.97, pick 0.77; predicted 0.73 vs 0.81): the double count at work. In the other **32 % (330 tokens)** the alternative has the higher or equal pEP and loses only on the safety terms (pNC +0.050, pDAC +0.031, TTC +0.063 against EP -0.016).
- Winner's-curse warning: the alternative is picked with hindsight, so its predicted safety (pNC*pDAC 0.776 against the pick's 0.954) is a selection effect, not a calibration claim. The calibration claims are the all-proposal tables in (b).
- The rule's margins are tiny: top1 - top2 log-score median **0.0015**, top1 - top8 **0.012**; a band of 0.02 holds 14.6 proposals on average, 0.05 holds 24.4 (`raw/diag2.json`).

## (d) The EP normaliser

NAVSIM EP is progress relative to PDM-Closed's progress, so one metre of shortfall costs `1 / reference`. The reference is not in the table; it is INFERRED per token as the median of `path length / EP` over gs proposals with 0.02 < EP < 0.98 (1,418 tokens; median IQR/median **0.012**, so the inference is internally consistent; ESTIMATED, Euclidean path rather than centreline progress). Reference quantiles 5/25/50/75/95: 6.1 / 10.7 / 18.0 / 33.6 / 45.2 m.

| class | tokens | share of EP gap | mean EP gap | gap in metres | EP cost of 1 m |
|---|---|---|---|---|---|
| EP saturated, no gradient | 16.8 % | 6.0 % | 0.046 | n/a | n/a |
| reference < 10 m | 17.7 % | 18.3 % | 0.131 | 0.95 | **0.134** |
| 10-20 m | 29.3 % | 24.1 % | 0.104 | 1.5 | 0.069 |
| 20-35 m | 17.5 % | 26.2 % | 0.189 | 5.0 | 0.036 |
| >= 35 m | 18.7 % | 25.4 % | 0.172 | 7.2 | 0.024 |

The normaliser dominates in **slow scenes**: below 10 m reference a 0.95 m shortfall already costs 0.13 EP. In fast scenes the shortfall is 5-7 m (about 1.3-1.8 m/s over 4 s) at a lower price per metre. Average shortfall where inferred: 3.4 m.

## Answer (one paragraph)

The pick is a **safe but slow** plan, not a missing fan: three tokens in four hold a full-progress safe plan, and picking the best safe-EP proposal would close 99.6 % of the oracle gap. Two things hold the pick short. (1) The EP head predicts progress with violators already scored 0, and the rule multiplies by `pNC*pDAC` on top, so the violation risk is charged twice and more heavily the longer the plan (an effective EP term of 0.70 against a true 0.96 for the longest plans). (2) The rule's margins are so small (0.0015 median top-two) that the remaining safety terms (pDAC, TTC, pNC) decide most of the 1,039 recoverable tokens, which a score-band tie-break or a predicted-safety floor would hand to the head-free progress proxy (plan length, within-token rank 0.90 with true EP). The normaliser makes this most expensive in slow scenes. These are the three zero-training candidates in `PREREG_EP1.md`.

## Limits

One fit table (24 logs, so the intervals are wide); P2 one training seed; the fit-set oracle (98.74) and pick (90.64) differ from navtest's (98.23 / 88.82), so level transfer is an assumption. The reference inference in (d) is ESTIMATED. Nothing in this step evaluated a candidate rule.
