# H-P6-SELECT-1 result at nomination: outcome S2, NO SELECTOR GAIN (2026-11-09)

Card: `PREREG_P6_SELECTORS.md` (sha256 d4778ca4...), Amendment 1 (sha256 d8efa150...). All numbers MEASURED on navtrain FIT logs: `FIT_REPORT.md`, `nominees.json`, `split.json` in this folder. Tier: FIT-B is a selection set, not a driving tier. Estimator: paired log-cluster bootstrap on FIT-B, 10,000 draws, seed 20260927, which answers "another draw of logs" only.

**Verdict as written.** No family has a variant with CI-lo > 0 against P6's rule on FIT-B, so none nominates. No selector was read on E2 or E3; K stays 0. Card section 5 commits this to S2: selection on P6's existing outputs is not a lever on this evidence, and the trained arms are.

| variant | FIT-B PDMS x100 | delta against P6's rule [95 % CI] |
|---|---|---|
| R1 rule shape | 93.786 | +0.079 [-0.009, +0.133] |
| R2 rule shape | 93.804 | +0.097 [-0.102, +0.305] |
| C_platt | 93.707 | -0.000 [-0.019, +0.019] |
| C_isotonic | 93.673 | -0.034 [-0.198, +0.101] |
| C_platt_binned | 93.675 | -0.032 [-0.138, +0.093] |
| S_gbdt_pdms | 93.846 | +0.139 [-0.133, +0.414] |
| S_gbdt_heads | 93.863 | +0.156 [-0.230, +0.527] |
| S_mlp_heads | 92.281 | -1.427 [-1.931, -1.003] |
| P_pg | 93.694 | -0.013 [-0.334, +0.292] |

**Size of the gap on the same set** (2,514 FIT-B sets, unweighted; `F:/tanitad_scratch/p6sel/gap.py`, which reproduces the report's 93.707 independently): P6's pick 93.707, best of the 64 proposals 97.515, mean of a random proposal 67.886. The largest point gain, +0.156, is 4 % of the 3.808 gap, and its interval contains zero.

Sub-scores x100, at P6's pick and at the best proposal: EP 88.99 and 94.34, TTC 97.02 and 99.84, NC 99.46 and 99.96, DAC 99.32 and 99.88, comfort 100.00 at both.

## What this does and does not say

- ⚠️ **P6's logits are in-sample on these logs** (card section 1). P6 was trained on navtrain, so the selectors were fitted where the heads are at their best. This result says a selector adds nothing measurable there. It does not say what one adds on logits from logs the scorer has not seen.
- The earlier family probe (E1, 1,058 navtest tokens, cross-fitted, descriptive, unadjusted, carried nowhere) read rule shape at +0.79 [+0.08, +1.65]. It is not comparable with this table: other tokens, other weighting, and navtest logits, which are out of sample. The difference is why the next card exists; it is not evidence of anything by itself.
- No navtest number was produced, so no driving claim is made and the four metric families have nothing to report. STRATEGIC is PAUSED (PI 2026-09-19), not a pass.

## Data notes owed by the card

- `fit_pack.npz` held P6's raw 5 Hz proposals [N,64,20,x]. The runner's shape check refused them before any fit. `pack_to_8.py` writes the 8-pose executed plans with the labeller's own `executed_plans8(repair=True)`; out sha256 9a6f7f1b..., selftest passes.
- Geometry and yaw come from the set files, rounded there to 0.01 m and 0.001 rad.
- Ego speed is the proposal proxy of Amendment 1.
- Labels use the cached route; on 3 of 120 checked sets EP differs from the database route (max EP 0.167, PDMS 0.070; best proposal differs on 1 of 120).

## Next lever (RULE ZERO)

1. The Batch 1 arms, C0, C0r and S1E, are launching now. S1E targets EP directly, the largest sub-score gap on this set.
2. `PREREG_P6_SELECTORS_ARM.md`, frozen with this result: the same nine variants on a trained arm's logits on these logs, which its fine-tune never trained on.
