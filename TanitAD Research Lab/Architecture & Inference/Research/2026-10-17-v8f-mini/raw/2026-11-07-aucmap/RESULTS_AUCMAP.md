# RESULTS_AUCMAP: how much go/stay discriminability does the dominance corner need? (T2, quantitative form)

Agent: AUCMAP (Sonnet). Written 2026-11-07 03:17:36 +0100. Frozen card `PREREG_AUCMAP.md` (sha256 `1daf6fbceff28a46a1d03c4576a2037099d060d406ac3638b27221f4d6ee1227`, verified before the run) with `AMEND_1_AUCMAP.md` (C6 only). Verdict follows card sections 8-10 as written. Every number is [M] (measured, ours) from `raw/AUCMAP_RESULT.json` / `raw/AGG.json` / `raw/CONTROLS.json` in this folder; tier T1-kin (logged cameras, `reindexed` longitudinal-only re-perception; NOT T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (answers 'another draw of EPISODES' only; blind to training variance, two seeds one training run each, `H-ESTIM-SEED-1`; the rollouts are not iCEM-sampled, so no inference variance). The noise draws of the synthetic posterior are a third variance, carried as the reach rate (K=20 draws per level).

**CEILING-TYPE.** Synthetic posteriors are built from the window's true label (ego and future, offline) and the oracle-level levers (`LFO*`, `LFOB*`) force the logged speed. Neither is admissible at inference. These arms decide NO driving claim and are never a driving result. The only admissible arm is the real-posterior anchor gated by the heads-level lever. STRATEGIC family: PAUSED (PI ITEM 25, 2026-09-19), reported beside, never a pass.

## 1. Verdict
* **Outcome (iii): the corner is not reachable by any gate on this lever family.** CORNER-FULL (M1-FULL and GUARD, both seeds) has reach rate **0.00 at all 8 AUC levels, 20 of 20 draws not reaching at each level, for all 5 admissible levers in both scopes**, including nominal AUC 1.0 (perfect separation of the GO label, q free on a 21-point grid). The CEILING-LEVEL family (`LFO075`, `LFO100`, `LFOB05`, `LFOB10`) also reads 0.00 at AUC 1.0 in both scopes (read beside).
* **A\*_any (CORNER-FULL, ANCHOR scope, label GO, 5 admissible levers) is undefined (> 1.0).** Interval undefined (no level reaches 0.05 reach rate). No lever is FRAGILE (none reaches at any q). A\* is not a go/stay AUC target for L (outcome (i) needs A\*_any <= 1.0); the LoRA arm's P4 read has no AUC to compare with under this lever family.
* Outcome tests as written: (i) needs headline reach at 0.785 < 0.5 AND no anchor corner AND A\*_any <= 1.0: the first two hold (reach 0.00; anchor reaches no corner), the third fails, so (i) is not the outcome. (ii) needs reach at 0.785 >= 0.5 or the anchor reaching CORNER-FULL: neither holds. (iv) needs CORNER-FULL in ALL scope but not ANCHOR scope: it is reached in neither. (iii) holds: no admissible lever at AUC 1.0, nor the CEILING-LEVEL family at AUC 1.0.
* **Real anchor** (P1P `post_primary`, v0 < 8, 861 windows, realized AUC 0.749 on the card label): no admissible lever reaches CORNER-FULL or CORNER-P at any of the 21 q points, both seeds. NEAR test (section 5): **NEAR for 0 of 5 levers**.
* **Binding predicate (beside, section 3):** at the oracle gate the lever operator itself fails the guard: `cross-track MAE` (LATERAL, ACTIVE) is separated-worse at both seeds at the smallest q that closes enough of M1, and M1-FULL (X - P10 not separated-worse) is met only by the ungated lever (ALL scope q = 1.0, LF100), where 6 (seed 0) and 5 (seed 1) ACTIVE rows are separated-worse and ADE6 is +1.256 / +1.089 m. Reading per card outcome (iii) [E]: the guard rows are driven by the lever operator and not by go/stay discrimination (consistent with the A5 joint reading).

## 2. A* per (lever, scope, corner), label GO  (CEILING-TYPE: synthetic posterior; oracle-level rows are double-ceiling)
| lever | level type | scope | A* FULL | A* P (beside) | FULL reach @0.785 | FULL reach @1.0 | max share_closed seen (AUC 1.0, mean of seeds, any q) | draws with M1-P at some q (AUC 1.0) | draws with M1-FULL at some q | draws with CORNER-P |
|---|---|---|---|---|---|---|---|---|---|---|
| LF050 | ADMISSIBLE | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.051 | 0.00 | 0.00 | 0.00 |
| LF050 | ADMISSIBLE | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.059 | 0.00 | 0.00 | 0.00 |
| LF075 | ADMISSIBLE | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.348 | 0.00 | 0.00 | 0.00 |
| LF075 | ADMISSIBLE | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.394 | 0.00 | 0.00 | 0.00 |
| LF100 | ADMISSIBLE | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.760 | 1.00 | 0.00 | 0.00 |
| LF100 | ADMISSIBLE | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.841 | 1.00 | 1.00 | 0.00 |
| LFB05 | ADMISSIBLE | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.313 | 0.00 | 0.00 | 0.00 |
| LFB05 | ADMISSIBLE | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.355 | 0.00 | 0.00 | 0.00 |
| LFB10 | ADMISSIBLE | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.469 | 0.00 | 0.00 | 0.00 |
| LFB10 | ADMISSIBLE | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.531 | 1.00 | 0.00 | 0.00 |
| LFO075 | CEILING-LEVEL | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.098 | 0.00 | 0.00 | 0.00 |
| LFO075 | CEILING-LEVEL | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.129 | 0.00 | 0.00 | 0.00 |
| LFO100 | CEILING-LEVEL | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.647 | 1.00 | 0.00 | 0.00 |
| LFO100 | CEILING-LEVEL | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.738 | 1.00 | 0.00 | 0.00 |
| LFOB05 | CEILING-LEVEL | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.306 | 0.00 | 0.00 | 0.00 |
| LFOB05 | CEILING-LEVEL | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.356 | 0.00 | 0.00 | 0.00 |
| LFOB10 | CEILING-LEVEL | ANCHOR | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.449 | 0.00 | 0.00 | 0.00 |
| LFOB10 | CEILING-LEVEL | ALL | none (> 1.0) | none (> 1.0) | 0.00 | 0.00 | 0.516 | 1.00 | 0.00 | 0.00 |

Reach rate by level for every (lever, scope), label GO, FULL and P: all 0.00 at 0.50, 0.68, 0.785, 0.85, 0.90, 0.95, 0.99, 1.0 (`raw/AGG.json` `a_star.*.rate_FULL`, `rate_P`). Realized AUC of the synthetic draws, LF100 ANCHOR: 0.502, 0.685, 0.784, 0.847, 0.905, 0.951, 0.989, 1.000 (nominal 0.50 ... 1.0; C4 within 0.02 holds).
Multiplicity, as disclosed in the card: 21 q points (adjacent points correlated), 5 admissible levers for A\*_any, 2 corners, 2 scopes, 2 labels. Multiplicity inflates false positives; the result here is a zero everywhere, so it is not inflated by it.

## 3. What binds (beside; decides nothing; CEILING-TYPE: AUC 1.0 oracle gate, draw 0)
Per q of the 21-point grid with point share_closed >= 0.4. Columns: share_closed (mean of seeds), ADE6 delta(X - R) m (seed 0 | seed 1), M1-P, M1-FULL, GUARD, ACTIVE rows separated-worse (seed 0 | seed 1 counts) and their names (union).

**LF100, ANCHOR scope** (ADMISSIBLE heads level):

| q | share_closed | ADE6 d(X-R) s0 | ADE6 d(X-R) s1 | M1-P | M1-FULL | GUARD | #worse s0 | #worse s1 | rows |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.424 | +0.002 | -0.004 | False | False | False | 1 | 1 | cross-track MAE |
| 0.45 | 0.487 | -0.002 | -0.011 | False | False | False | 1 | 1 | cross-track MAE |
| 0.50 | 0.518 | +0.013 | +0.002 | True | False | False | 1 | 1 | cross-track MAE |
| 0.55 | 0.547 | +0.055 | +0.038 | True | False | False | 2 | 2 | DK-VIOL, cross-track MAE |
| 0.60 | 0.575 | +0.121 | +0.105 | True | False | False | 2 | 2 | DK-VIOL, cross-track MAE |
| 0.65 | 0.603 | +0.166 | +0.147 | True | False | False | 4 | 3 | DK-HW, DK-VIOL, along-track MAE, cross-track MAE |
| 0.70 | 0.625 | +0.221 | +0.193 | True | False | False | 4 | 3 | DK-HW, DK-VIOL, along-track MAE, cross-track MAE |
| 0.75 | 0.644 | +0.278 | +0.238 | True | False | False | 5 | 3 | DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.80 | 0.667 | +0.309 | +0.257 | True | False | False | 6 | 3 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.85 | 0.688 | +0.372 | +0.307 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.90 | 0.716 | +0.411 | +0.337 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.95 | 0.738 | +0.465 | +0.383 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 1.00 | 0.760 | +0.520 | +0.436 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |

**LF100, ALL scope** (ADMISSIBLE heads level):

| q | share_closed | ADE6 d(X-R) s0 | ADE6 d(X-R) s1 | M1-P | M1-FULL | GUARD | #worse s0 | #worse s1 | rows |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.422 | +0.017 | +0.018 | False | False | False | 1 | 1 | cross-track MAE |
| 0.30 | 0.510 | +0.007 | +0.005 | True | False | False | 1 | 1 | cross-track MAE |
| 0.35 | 0.538 | +0.111 | +0.069 | True | False | False | 2 | 2 | DK-HW, DK-VIOL, cross-track MAE |
| 0.40 | 0.567 | +0.207 | +0.149 | True | False | False | 4 | 3 | DK-HW, DK-VIOL, along-track MAE, cross-track MAE |
| 0.45 | 0.589 | +0.311 | +0.254 | True | False | False | 5 | 4 | DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.50 | 0.618 | +0.399 | +0.332 | True | False | False | 5 | 4 | DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.55 | 0.648 | +0.489 | +0.410 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.60 | 0.661 | +0.575 | +0.490 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.65 | 0.681 | +0.656 | +0.562 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.70 | 0.698 | +0.727 | +0.618 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.75 | 0.722 | +0.800 | +0.686 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.80 | 0.746 | +0.903 | +0.767 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.85 | 0.778 | +0.996 | +0.846 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.90 | 0.806 | +1.078 | +0.931 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 0.95 | 0.817 | +1.165 | +1.006 | True | False | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |
| 1.00 | 0.841 | +1.256 | +1.089 | True | True | False | 6 | 5 | DK-HW, DK-TG, DK-VIOL, along-track MAE, cross-track MAE, speed MAE |

**LFO100, ANCHOR scope** (CEILING-LEVEL logged level):

| q | share_closed | ADE6 d(X-R) s0 | ADE6 d(X-R) s1 | M1-P | M1-FULL | GUARD | #worse s0 | #worse s1 | rows |
|---|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.423 | -0.103 | -0.104 | False | False | True | 0 | 0 | - |
| 0.60 | 0.448 | -0.091 | -0.089 | False | False | False | 0 | 1 | cross-track MAE |
| 0.65 | 0.479 | -0.091 | -0.091 | False | False | False | 0 | 1 | cross-track MAE |
| 0.70 | 0.516 | -0.083 | -0.084 | True | False | False | 2 | 1 | DK-VIOL, cross-track MAE |
| 0.75 | 0.540 | -0.074 | -0.076 | True | False | False | 2 | 1 | DK-VIOL, cross-track MAE |
| 0.80 | 0.575 | -0.072 | -0.081 | True | False | False | 2 | 1 | DK-VIOL, cross-track MAE |
| 0.85 | 0.586 | -0.053 | -0.075 | True | False | False | 2 | 2 | DK-VIOL, cross-track MAE |
| 0.90 | 0.612 | -0.046 | -0.073 | True | False | False | 2 | 2 | DK-VIOL, cross-track MAE |
| 0.95 | 0.625 | -0.040 | -0.067 | True | False | False | 2 | 2 | DK-VIOL, cross-track MAE |
| 1.00 | 0.647 | -0.038 | -0.064 | True | False | False | 2 | 2 | DK-VIOL, cross-track MAE |

Control C7 value [M]: the oracle gate (AUC 1.0, q = GO rate 0.304, ALL scope) on LF100 reads share_closed 0.517 (mean of seeds) with ADE6 d(X - R) +0.011 / +0.006 m, against 0.841 for the ungated LF100.

## 4. The real anchor (P1P `post_primary`, v0 < 8, 861 windows), per admissible lever (heads-level lever, ADMISSIBLE; label GO of the card)
Realized AUC of the posterior [M]: card label v0 < 6 (n 608) **0.709**; card label v0 < 8 (n 861) **0.749**; against P1P `go` (population-limited label, v0 < 8) **0.785**. See section 8 for the label scope.

| lever | max share_closed over q (mean of seeds) | at q | share_closed at q=1.0 (s0 / s1) | ADE6 d(X-R) at q=1.0 m (s0 / s1) | M1-P at some q | smallest q with M1-P | M1-FULL at some q | CORNER-P at some q | CORNER-FULL at some q |
|---|---|---|---|---|---|---|---|---|---|
| LF050 | 0.051 | 0.90 | 0.054 / 0.045 | +0.108 / +0.065 | False | - | False | False | False |
| LF075 | 0.348 | 1.00 | 0.357 / 0.338 | +0.285 / +0.169 | False | - | False | False | False |
| LF100 | 0.760 | 1.00 | 0.821 / 0.698 | +0.520 / +0.436 | True | 0.75 | False | False | False |
| LFB05 | 0.313 | 1.00 | 0.335 / 0.291 | +0.023 / +0.006 | False | - | False | False | False |
| LFB10 | 0.469 | 1.00 | 0.510 / 0.428 | +0.121 / +0.103 | False | - | False | False | False |

Only LF100 reaches M1-P (smallest q 0.75: share_closed 0.630 / 0.535, ADE6 d(X-R) +0.230 / +0.190 m, GUARD False, M1-FULL False). The anchor's corner status is read with the same predicates; one posterior, so no reach rate.

## 5. NEAR test (card section 9): is the real-anchor curve inside the synthetic band?
Synthetic level closest to the anchor's realized AUC on the card label (0.749 on v0 < 8) is 0.785 (|0.749 - 0.785| = 0.036; the 0.68 level is 0.069 away), which is also the card's second test level, so one test level is read. Fraction of the 21 q points with BOTH coordinates (share_closed, ADE6 d(X-R)) inside the pointwise [2.5, 97.5] percentile band of the 20 draws, per seed; NEAR needs >= 0.75 at both seeds.

| lever | frac inside s0 | frac inside s1 | NEAR | anchor share_closed at q=0.5 (mean seeds) | synthetic 0.785 median at q=0.5 |
|---|---|---|---|---|---|
| LF050 | 0.571 | 0.714 | False | 0.018 | 0.030 |
| LF075 | 0.238 | 0.238 | False | 0.123 | 0.193 |
| LF100 | 0.095 | 0.143 | False | 0.373 | 0.450 |
| LFB05 | 0.286 | 0.333 | False | 0.141 | 0.184 |
| LFB10 | 0.286 | 0.238 | False | 0.226 | 0.277 |

The real posterior (realized AUC 0.749) closes less M1 mass per q than a synthetic 0.785 posterior on every lever with a visible curve, so the synthetic 0.785 curve is an optimistic stand-in for the anchor. This does not change the outcome: even the perfect-separation synthetic posterior reaches no corner.

## 6. Sensitivity arms (beside)
* **Label KEEP** (`v6 >= v0 - 1.0`, ANCHOR scope, 5 admissible levers): reach rate 0.00 at all 8 levels for every lever (FULL and P). A\* none (> 1.0).
* **Shuffled-label arm** (LF100, ANCHOR, labels permuted, then the AUC-1.0 construction): realized AUC against the TRUE label 0.500 (range 0.469 to 0.541); its mean share_closed curve over q sits on the nominal-0.50 curve (max absolute gap 0.014; at q = 0.5: 0.389 against 0.388); reach rate FULL 0.00 at all levels. No-skill false-positive rate: 0 of 20 draws.
* **Corner-P** (M1-P and GUARD, beside): reach rate 0.00 for all 18 (lever, scope) rows, label GO, all levels (table in section 2, last column).
* Not gated, named in the card: `lonfloor` / `kbox2lf` (no per-window arrays in this layout) and the `LFN*` noise arms.

## 7. Four families at the best q (beside; STRATEGIC PAUSED, PI ITEM 25)
No (draw, level) reaches the corner, so the card's 'reached' q does not exist; the best q is fixed here, after the data and deciding nothing: the smallest q on the AUC-1.0 oracle-gate curve (ANCHOR scope, draw 0) whose point share_closed (mean of seeds) is >= 0.5, or q = 1.0 if none. All 9 levers: `raw/FOUR_FAMILIES.md` (every row, both seeds, delta [lo, hi], direction-aware flag). Summary [M, CEILING-TYPE]:

| lever | type | q | share_closed s0 / s1 | ADE6 d(X-R) m s0 / s1 | M1-P | M1-FULL | GUARD | ACTIVE rows separated-worse s0 | s1 |
|---|---|---|---|---|---|---|---|---|---|
| LF050 | ADMISSIBLE | 1.00 | 0.054 / 0.045 | +0.108 / +0.065 | False | False | False | cross-track MAE | DK-VIOL, cross-track MAE |
| LF075 | ADMISSIBLE | 1.00 | 0.357 / 0.338 | +0.285 / +0.169 | False | False | False | DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE | DK-VIOL, along-track MAE, cross-track MAE |
| LF100 | ADMISSIBLE | 0.50 | 0.580 / 0.455 | +0.013 / +0.002 | True | False | False | cross-track MAE | cross-track MAE |
| LFB05 | ADMISSIBLE | 1.00 | 0.335 / 0.291 | +0.023 / +0.006 | False | False | False | - | DK-VIOL |
| LFB10 | ADMISSIBLE | 1.00 | 0.510 / 0.428 | +0.121 / +0.103 | False | False | False | DK-HW, DK-VIOL, along-track MAE | DK-VIOL, cross-track MAE |
| LFO075 | CEILING-LEVEL | 1.00 | 0.099 / 0.097 | -0.014 / -0.013 | False | False | False | cross-track MAE | accel MAE, cross-track MAE |
| LFO100 | CEILING-LEVEL | 0.70 | 0.522 / 0.510 | -0.083 / -0.084 | True | False | False | DK-VIOL, cross-track MAE | cross-track MAE |
| LFOB05 | CEILING-LEVEL | 1.00 | 0.334 / 0.278 | -0.064 / -0.063 | False | False | False | - | cross-track MAE |
| LFOB10 | CEILING-LEVEL | 1.00 | 0.480 / 0.418 | -0.084 / -0.087 | False | False | False | DK-VIOL, cross-track MAE | accel MAE, cross-track MAE |

LF100 at q = 0.5 (oracle gate, CEILING-TYPE), the arm that comes closest on M1-P and ADE6, all four families [M, T1-kin], X - R, delta [lo, hi], seed 0 | seed 1:

| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.0135 [-0.0369, +0.0627] | +0.0024 [-0.0436, +0.0460] | False/False |
| LONGITUDINAL | speed MAE | -0.0407 [-0.0693, -0.0128] | -0.0359 [-0.0627, -0.0116] | False/False |
| LONGITUDINAL | along-track MAE | -0.0124 [-0.0695, +0.0416] | -0.0273 [-0.0845, +0.0249] | False/False |
| LONGITUDINAL | accel MAE | -0.0675 [-0.1284, -0.0110] | -0.0354 [-0.0877, +0.0155] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0034 [+0.0000, +0.0085] | +0.0034 [+0.0000, +0.0084] | False/False |
| LONGITUDINAL | DK-HW | +0.0581 [-0.0227, +0.1585] | -0.0156 [-0.1384, +0.0969] | False/False |
| LONGITUDINAL | DK-TG | +0.0156 [-0.0499, +0.0914] | -0.0257 [-0.1289, +0.0485] | False/False |
| LATERAL | heading MAE | -0.7316 [-1.1747, -0.3620] | -0.7761 [-1.3047, -0.3700] | False/False |
| LATERAL | yaw-rate MAE | -4.4601 [-7.0550, -2.4325] | -4.5869 [-7.6908, -2.3806] | False/False |
| LATERAL | curvature MAE | -0.0385 [-0.0582, -0.0217] | -0.0428 [-0.0686, -0.0220] | False/False |
| LATERAL | cross-track MAE | +0.0294 [+0.0076, +0.0547] | +0.0307 [+0.0078, +0.0577] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.3661 [-0.7143, -0.1097] | -0.2297 [-0.5033, -0.0164] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0326 [-0.0582, -0.0144] | -0.0319 [-0.0574, -0.0106] | False/False |
| TACTICAL | TAC-LAT | +0.0188 [+0.0088, +0.0304] | +0.0208 [+0.0117, +0.0312] | False/False |
| TACTICAL | TAC-LON | +0.0509 [+0.0342, +0.0691] | +0.0454 [+0.0304, +0.0626] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.1629 [-0.3043, -0.0303] | -0.3009 [-0.4915, -0.1398] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.3084 [-0.5770, -0.0677] | -0.4787 [-0.8103, -0.2056] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.5350 [-0.9796, -0.1492] | -0.7478 [-1.3092, -0.2938] | False/False |
| crawl | M1 crawl share | -0.0309 [-0.0444, -0.0190] | -0.0271 [-0.0412, -0.0156] | False/False |

Real anchor LF100 at its smallest M1-P q = 0.75 (ADMISSIBLE arm; X - R; STRATEGIC PAUSED rows beside):

| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse s0/s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 | +0.2302 [+0.1065, +0.3649] | +0.1900 [+0.0807, +0.3083] | True/True |
| LONGITUDINAL | speed MAE | +0.0471 [-0.0071, +0.1065] | +0.0307 [-0.0132, +0.0782] | False/False |
| LONGITUDINAL | along-track MAE | +0.1723 [+0.0553, +0.3033] | +0.1373 [+0.0296, +0.2623] | True/True |
| LONGITUDINAL | accel MAE | -0.0820 [-0.1466, -0.0186] | -0.0444 [-0.0934, -0.0030] | False/False |
| LONGITUDINAL | DK-VIOL | +0.0218 [+0.0033, +0.0467] | +0.0168 [+0.0048, +0.0329] | True/True |
| LONGITUDINAL | DK-HW | +0.2430 [+0.0966, +0.4332] | +0.0865 [-0.0240, +0.2175] | True/False |
| LONGITUDINAL | DK-TG | +0.1051 [+0.0111, +0.2452] | +0.0036 [-0.0937, +0.1006] | True/False |
| LATERAL | heading MAE | -0.8189 [-1.3839, -0.3374] | -0.8576 [-1.6123, -0.3280] | False/False |
| LATERAL | yaw-rate MAE | -5.6663 [-9.2826, -3.0484] | -5.8574 [-9.9210, -2.9290] | False/False |
| LATERAL | curvature MAE | -0.0552 [-0.0857, -0.0305] | -0.0617 [-0.1017, -0.0305] | False/False |
| LATERAL | cross-track MAE | +0.0917 [+0.0448, +0.1503] | +0.0800 [+0.0385, +0.1261] | True/True |
| LATERAL | L1 lateral-accel MAE | -0.3666 [-0.7675, -0.0806] | -0.2706 [-0.5356, -0.0925] | False/False |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0323 [-0.0585, -0.0140] | -0.0370 [-0.0632, -0.0126] | False/False |
| TACTICAL | TAC-LAT | +0.0263 [+0.0146, +0.0396] | +0.0308 [+0.0175, +0.0462] | False/False |
| TACTICAL | TAC-LON | +0.0271 [+0.0067, +0.0484] | +0.0279 [+0.0104, +0.0467] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@30 | -0.2368 [-0.3974, -0.0881] | -0.4003 [-0.6464, -0.1864] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@50 | -0.4432 [-0.7390, -0.1808] | -0.6328 [-1.0811, -0.2643] | False/False |
| STRATEGIC PAUSED (beside) | S-POS@80 | -0.7058 [-1.2284, -0.2598] | -0.9705 [-1.7136, -0.3591] | False/False |
| crawl | M1 crawl share | -0.0336 [-0.0478, -0.0210] | -0.0319 [-0.0473, -0.0188] | False/False |

The S-POS `b` arrays (P10 side) differ between arms, so no X - P10 contrast on S-POS is read; the S-POS rows above are X - R and sit beside, never a pass.

## 8. Note: 0.785 versus 0.749 (label scope of the banked probe AUC)
The banked 0.785 (P1P / THEORY_CRAWL) is the AUC of `post_primary` against P1P `go` over the 861 windows with v0 < 8. P1P `go` is limited to the probe population (v0 < 6) and reads False for every window at 6 <= v0 < 8 (AMEND_1; equal to the card label on all 608 windows v0 < 6, differs on 81 of 253 at 6 <= v0 < 8). Against the card label the same posterior reads **0.749** on the 861 windows and **0.709** on the 608 windows with v0 < 6. So 0.785 is the AUC on a label that scores the 6 to 8 m/s windows as non-go; on the population-free card label the real posterior sits at 0.71 to 0.75. The synthetic 0.785 level is a nominal construction on the card label and is unaffected. Nothing in the verdict depends on which of the three numbers is quoted: no level up to 1.0 reaches the corner.

## 9. Controls and caveats
* Controls (`raw/CONTROLS.json`, `test_aucmap.py`): selection PASS, auc PASS, label_matches_p1p PASS, q0_is_base PASS, q1_is_lever PASS, random_gate_matches_reader PASS, oracle_gate PASS, noskill_line PASS, corner_known_values PASS. Deliberate-regression mutants RED: 7 of 7 (selection inverted, scope mask dropped, label margin changed, shuffle removed, P10 contrast ignoring the base excess, GUARD ignoring ACTIVE rows, ADE6 guard disabled). C9 (added after the sweep, before this write-up; it asserts the corner predicate CAN read True: a lever equal to the base with the M1 row at P10 reads CORNER-FULL True and share_closed 1.0; an ACTIVE row pushed worse, or ADE6 +0.2 m, reads CORNER-FULL False) shows the zero reach rate is not a predicate that can only read False.
* `run_synthetic` was extended before the sweep to also store the per-seed share_closed (`share_s`) the NEAR test needs; nothing else in `aucmap.py` changed after the controls that fixed the sweep code, and the controls were re-run in full afterwards (all PASS, 7 of 7 mutants RED).
* Scope of what this measures: banked levers (LF050, LF075, LF100, LFB05, LFB10 heads level) gated per window by a posterior of controlled AUC, applied to rollouts that already exist. It does not build a rollout arm with the gate inside the replan loop, does not test `lonfloor` / `kbox2lf`, and does not test any other lever. 'No gate on THIS lever family reaches the corner' is the claim; it says nothing about a different lever.
* The corner is evaluated on 141 episodes, two seeds, one training run each; a separated difference is necessary and not sufficient for a lever effect (`H-ESTIM-SEED-1`). A zero reach rate is not the false-positive risk that `H-ESTIM-SEED-1` describes; it leaves open that another training run of a lever would read differently.
* Pre-freeze inspection of ungated M1 mass (card section 2): v0 < 8 carries about 91 % of the base-over-P10 excess, which is why the ANCHOR scope is not a small fraction of the problem.

## 10. What this leaves (Rule Zero: the next lever, not only the verdict)
* **Outcome (iii) names the next arm: change the lever / the decode path (F), not perception.** The data say what that arm must achieve at the lever, with a perfect gate: close M1 by share_closed >= 0.5 (and reach X - P10 not separated-worse) while leaving `cross-track MAE` (LATERAL), `DK-VIOL`, `along-track MAE` and `speed MAE` (LONGITUDINAL) and ADE6 inside the guard. LF100 closes M1 at the oracle gate (q = 0.304 of ALL windows) with ADE6 +0.011 / +0.006 m, but pays `cross-track MAE` separated-worse at both seeds, and only the ungated lever reaches M1-FULL.
* The bounded lifts (LFB05, LFB10) keep the guard rows nearer base at q = 1.0 (LFB05: 0 | 1 ACTIVE rows worse, ADE6 +0.023 / +0.006 m) but close only 0.31 and 0.47 of M1 at q = 1.0 in ANCHOR scope. The programme's open point is a lever with LF100's M1 closure and LFB05's guard footprint; a gate does not create that.
* **Blocker, named:** the next arm (a lever or decode-path change, F) needs its own pre-registered card and a rollout arm; it is a Master Mind / PI scheduling decision, not something this card can launch. The AUC target for L (outcome (i)) does not exist under this lever family, so the LoRA arm's P4 read should not cite an A\*.

## 11. Deliverable manifest (all in `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/aucmap/`; single location unless noted; nothing staged, committed or pushed)
| artifact | path (repo-external, dev box E:) |
|---|---|
| frozen card + sha | `PREREG_AUCMAP.md`, `PREREG_AUCMAP.sha256` |
| amendment (C6) + sha | `AMEND_1_AUCMAP.md`, `AMEND_1_AUCMAP.sha256` |
| instrument | `aucmap.py` |
| controls + mutants | `test_aucmap.py`, `raw/CONTROLS.json` |
| sweep driver | `run_all.py`, `raw/AUCMAP_run.log`, `raw/AUCMAP_stdout.log`, markers `raw/AUCMAP.done`, `raw/AUCMAP.running.ended` |
| per-task results (29) | `raw/tasks/*.json` |
| merged results | `raw/AUCMAP_RESULT.json` |
| aggregation | `aggregate.py`, `write_results.py`, `raw/AGG.json`, `raw/FOUR_FAMILIES.md` |
| this report | `RESULTS_AUCMAP.md` |
| handoff | `HANDOFF_AUCMAP.md` |

All of this lives only in this folder on the dev box E: (not in the repo, not staged). Integration for the Master Mind: land the folder (excluding `raw/tasks/` if size matters) and add the outcome-(iii) row to `GOALS_AND_CLAIMS.md`.
