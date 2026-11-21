# RESULT_RBDIST_ZT -- does the route text carry the roundabout exit side?

Evidence class: MEASURED (ours; pinned `navtok_table.npz` sha256 2836369e..., frozen navtok grammar, zero training, CPU). Tier: T0 nav-only decodability, NOT driving performance; route supplied along the driven path. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` on per-window correctness, B 2,000, seed 0, 16 episodes; it answers "another draw of EPISODES" only. Card: `PREREG_RBDIST_ZT.md` (sha256 f00a7725..., written before any eval number). Artifact: `F:/TanitAD-scratch/strat1/rbdist/out/RESULT_RBDIST_ZT.json`. Controls held, problems: none.

## Where the features are
Exit number: `navtok.py:32` (`At roundabout, take exit (?P<exit>\d+)`), parsed by `parse_full` into `exit_n` (`navtok.py:76-92`); countdown distance: the ` in <d>m` suffix (`navtok.py:35`), table column `{split}_dist_m` (`navtok/build_table.py:42-71`). Both exist in the route text; no roundabout text has a side word (NAVRB, `RESULT_NAVRB.md` section 1).

## Reading: TEXT-SILENT
Rule (committed): TEXT-CARRIES-SIDE iff accuracy(ED) - accuracy(S) >= 0.0383 and the paired CI is separated. Observed gain **-0.0792, CI [-0.2422, +0.0532], not separated** (202 scored RB windows, 16 episodes; 235 RB windows, 33 carry no v9 label). The point estimate is below zero, so this is not even SUGGESTIVE-UNSEPARATED.

| feature set | accuracy (5-way, RB) | gain over S, paired | macro-F1 | predicted classes (LK / L / R) |
|---|---|---|---|---|
| S = state only (constant LANE_KEEP) | 0.787 [0.636, 0.920] | - | 0.176 | 202 / 0 / 0 |
| ED = state + exit + distance (primary) | 0.708 [0.548, 0.854] | -0.0792 [-0.2422, +0.0532] | 0.317 | 152 / 26 / 24 |
| E = state + exit number | 0.787 [0.636, 0.920] | 0.0000 [0, 0] | 0.176 | 202 / 0 / 0 |
| D = state + distance | 0.748 [0.570, 0.903] | -0.0396 [-0.1121, +0.0357] | 0.261 | 174 / 28 / 0 |

TURN windows (43 windows, 6 episodes, descriptive): ED names a turn on 39.5 % of them and the right side on 39.5 %; S names none. Eval label by `exit_cur` (LK / L / R): exit 1: 64 / 0 / 20, exit 2: 69 / 8 / 4, exit 3: 4 / 0 / 7, none: 22 / 0 / 4. Train label by `exit_cur` (24,722 RB windows, 426 episodes): exit 1: 4,801 / 514 / 1,453, exit 2: 5,775 / 2,169 / 861, exit 3: 1,711 / 708 / 302. The train side for exit 3 is mostly L and the eval side (7 of 7, one episode) is R, and exit 2 leans L on train: the side is not a stable function of the exit number, so a classifier fitted on train does not transfer. The exit number alone (E) changes no eval prediction.

## Controls (all held)
* K-CONST: S predicts the single class LANE_KEEP; accuracy 0.7871 equals the eval share of that class exactly.
* K-REACH: the label as input reads accuracy 1.0.
* R-SHUF-EVAL (eval feature rows permuted, 10 seeds): mean gain over S -0.1673, 0 of 10 satisfy the rule (bar: mean < 0.0383).
* R-SHUF-TRAIN (train feature rows permuted, 5 seeds): mean gain over S -0.0010, 0 of 5 satisfy the rule.

## Next lever (committed in the card)
TEXT-SILENT: the roundabout side must come from vision (exit direction and lane layout in the scene). The exit-number and countdown-distance channels are dropped for this gap; no RB-DIST training arm is built. Cheapest next experiment on that branch, zero-training: a frozen-encoder probe of the exit side on the 43 RB-TURN windows (own card).

## Limits
16 episodes (wide intervals; a gain of +0.0383 could not be excluded: upper bound +0.0532). The exit number is a weak, inconsistent side signal on this corpus (above). T0 decodability from the text says nothing about a driving model. The text's countdown is not the dead-reckoned channel 3 the trained arms were fed. One classifier family (HGB, unweighted, fixed hyper-parameters).
