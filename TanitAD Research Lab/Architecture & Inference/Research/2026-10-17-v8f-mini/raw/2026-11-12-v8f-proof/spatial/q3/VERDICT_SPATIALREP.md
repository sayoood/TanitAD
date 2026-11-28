# VERDICT SPATIALREP: the grid state against the baseline with six paired seeds (card PREREG_V8F_PROOF_ADD22)

Written 2026-11-15T02:35Z. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours; `out_rep6/VERDICT_Q3.md`, `out_rep6/RESULT_Q3.json`). Reader: `q3_read.py read --pairs G:HIER --seeds 0,1,2,3,4,5 --spread-arm HIER --spread-seeds 0,1,2,3,4,5 --same-sign`, two-level bootstrap (seeds, then episodes), B 2,000, rng 0. Arms G_s2 to G_s5 trained on the pod in two gated chains (card ADD22, 00:06Z to 02:27Z), 19,164 steps each, argv of G_s0; 141 episode files per arm pulled with equal sha256.

## Controls (first)

All five reader controls PASS: same windows in the 12 dumps; window counts 2,399 / 141 / 1,305 / 1,094 as in the card; self is zero (0.0); swap is minus (0.0); planted regression of 0.899 m/s read back as 0.899 and named COST-LONGITUDINAL.

## Outcome

Rule of the card: `GRID-IMPROVES` needs all six seed differences G minus HIER of one sign and a two-level interval that excludes 0 on the primary cell (ADE over 6 s, all windows).

| cell | per seed (0 to 5) | D | two-level interval | six-seed spread of HIER (max) |
|---|---|---|---|---|
| ADE 6 s, all windows (primary) | -0.1011 / -0.1384 / +0.0421 / -0.1174 / +0.0298 / -0.2281 | -0.0855 m | [-0.1705, +0.0044] | 0.2845 |
| ADE 6 s, common-case | -0.1943 / -0.0826 / -0.1700 / -0.2003 / +0.0474 / -0.3134 | -0.1522 m | [-0.2623, -0.0312] | 0.3850 |
| ADE 6 s, selection-critical | -0.0230 / -0.1852 / +0.2200 / -0.0479 / +0.0151 / -0.1566 | -0.0296 m | [-0.1419, +0.0937] | 0.4093 |

Four of six seed differences are negative on the primary cell and the interval includes 0: **`GRID-NO-COST`**. The grid state is not an improvement of the rollout under the rule, and no cell of the four families is worse.

Beside the name: with the first two seeds alone the primary cell read -0.1198 m [-0.1982, -0.0349] (both seeds negative). Four more seeds moved the mean to -0.0855 m and brought two positive seeds, so the two-seed read was inside the run-to-run variation of training, which the six-seed spread (0.2845 m) had already said. The common-case windows are lower for the grid state in five of six seeds with an interval that excludes 0, inside the spread; it names nothing.

## Four families (all windows, six seeds, G minus HIER)

* LONGITUDINAL: speed error -0.0643 m/s [-0.1295, -0.0012], five of six seeds negative, inside the spread (0.2997); along-track error -0.0410 m [-0.1138, +0.0336], OPEN. Headway and time gap: not measured (no lead block in the dumps).
* LATERAL: cross-track error -0.0757 m [-0.1555, -0.0031], five of six negative, inside the spread (0.2943); heading error -0.53 deg [-1.40, +0.55], curvature error +0.0117 1/m [-0.0056, +0.0311], yaw-rate error -0.19 deg/s [-2.27, +2.26]: OPEN.
* TACTICAL: longitudinal mode error rate -0.0224 [-0.0504, +0.0059], lateral -0.0424 [-0.0901, +0.0133], joint -0.0327 [-0.0634, +0.0021], tactical-band ADE -0.1231 m [-0.2492, +0.0097]: OPEN. Stop-goal decision: not measured.
* STRATEGIC: PAUSED (PI item 25).

## What follows (committed in the card)

`GRID-NO-COST`: the grid state stays the state of the world-model path, where it holds the position of the lead vehicle (one-frame probe R2 0.552 / 0.541 against the pooled state) at no cost in the rollout. No fine-tune of the driving path on the grid state is run from this read. The grid arm is the base of the lead-supervision arms of card ADD24, which are training on the pod, and its six seeds are the baseline spread of that card.

## Limits

One corpus, one evaluation set of 141 episodes; six training runs per arm; the planner is deterministic, so inference variance is not sampled. The interval answers the question about seeds and episodes of this rig only.
