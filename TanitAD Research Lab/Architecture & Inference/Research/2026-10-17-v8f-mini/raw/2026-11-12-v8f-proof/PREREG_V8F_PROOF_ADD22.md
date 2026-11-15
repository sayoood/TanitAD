# PREREG_V8F_PROOF_ADD22: SPATIALREP. Does a world-model state that keeps the layout improve the T1-kin accuracy of v8F-mini when seeds are the unit? Four more seeds of the grid-state arm, six in all.

Frozen before any arm of this card is trained. Addendum to `PREREG_V8F_PROOF.md` and to `PREREG_V8F_PROOF_ADD12.md`. Tier T1-kin (kinematic rollout on the model's own actions, logged frames): a diagnostic, never driving performance. Evidence class of every result: MEASURED once run. Four families; STRATEGIC: PAUSED (PI item 25).

## 1. Why this card

* Known (MEASURED, row H-V8F-PROOF-42, `spatial/q3/out_final/VERDICT_Q3.md`): with two seeds the grid-state arm G reads below the baseline HIER on 6 s ADE, all windows: -0.1198 m [-0.1982, -0.0349], seed differences -0.101 and -0.138; common-case windows -0.1384 m; cross-track -0.1071 m; tactical-band ADE -0.1750 m. Every one of these lies inside the six-seed spread of the baseline, so the card of that read names `NO-COST` and nothing more.
* Two seeds cannot separate an effect of this size from the run-to-run spread (0.2845 m on that cell). HIERREP showed that six paired seeds can (card ADD11). This card asks the same question of the state: is the gain real when seeds are the unit?

## 2. Arms (fixed)

* G_s2, G_s3, G_s4, G_s5: the argv of G_s0 and G_s1 (`--wm-state grid --wm-cell-dim 16`, the pinned tree of card ADD12, the HIER argv of HIERREP, 19,164 steps, the same cache, labels and exclusion), seeds 2 to 5. The gate of the pod queue (selfcheck on the exact argv, token, lever expectations, deliberate-regression expectation) must pass for each arm; an arm without its token is not trained and not read.
* Baseline: HIER_s0 to HIER_s5 of HIERREP. Pairing: by seed (G_sk against HIER_sk), as HIER against FLAT in HIERREP.
* T1-kin dump per arm with the pinned eval tool and the eval-141 grid (2,399 windows, 141 episodes; 1,305 selection-critical, 1,094 common-case).

## 3. Read (fixed)

* Tool: `spatial/q3/q3_read.py read --pairs G:HIER --seeds 0,1,2,3,4,5 --spread-arm HIER --spread-seeds 0,1,2,3,4,5`, its five controls first (same windows, window counts, self is zero, swap is minus, planted regression). A failed control: no name.
* Rule (that of card ADD11): a cell is an effect only if all six seed differences have the same sign and the two-level interval (seeds, then episodes, B 2,000, rng 0) excludes 0. The six-seed spread of the baseline is printed beside every cell and decides nothing here, because the pairs are six.
* Primary cell: 6 s ADE on all windows. Secondary: 6 s ADE on selection-critical and on common-case windows, and every cell of the LONGITUDINAL, LATERAL and TACTICAL families.
* Beside, T0, deciding nothing: the lead-gap R2 of the trained state on held-out clips for the six G seeds against the six HIER seeds (the trained-state read of card ADD12), when the states are pulled.

## 4. Outcomes (both committed)

* `GRID-IMPROVES`: the primary cell is below 0 under the rule. Named with every secondary cell that is an effect, in either direction.
* `GRID-NO-COST`: the primary cell is not an effect and no secondary cell is worse under the rule.
* `GRID-COSTS`: a cell is worse under the rule (named).

## 5. Committed next steps

* `GRID-IMPROVES`: the grid state becomes the default state of the v8F-mini recipe for every later arm of this programme, and the own-state fine-tune of HIERREP is run on it (two seeds first), own card.
* `GRID-NO-COST`: the grid state stays the state of the world-model arms (it reads the lead and costs nothing); the driving gain is not claimed.
* `GRID-COSTS`: the cell is reported as written and the state stays an auxiliary of the world model only.

## 6. Cost and limits

* Four arms of about one L40-hour each when alone (MEASURED on the same pod: F1_8_s1 trained 19,164 steps in 59 min); two at a time beside the last DATAEFF arm, never more than three training jobs, the container memory read before each start.
* One corpus (4,512 training clips), one evaluation set, a kinematic rollout on logged frames; the planner is deterministic, so inference variance is not sampled. Headway, time gap and the stop-goal decision are not measured by this read.
