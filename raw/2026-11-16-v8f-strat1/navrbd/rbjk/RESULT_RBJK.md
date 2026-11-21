# RBJK (RB-JOINT-K) result

Reading per card (PREREG_V8F_NAVRBD.md sha256 881d7f3d, sec 7): **RBJK-CARRIES**. Control SHUF_ED - S < 0.0383: **PASS**.
Tier T0 decodability, never driving performance. Evidence class MEASURED (ours; `RESULT_RBJK.json`, `rbjk.py`, CPU, zero model training).
Estimator: taniteval paired episode-cluster bootstrap, B 2000, seed 0, episodes = train clips; answers "another draw of EPISODES" only.

## Numbers (target = 3-class kinematic lat label of the GT 6 s track)
| row | value |
|---|---|
| ED - S | **+0.2331 [+0.1906, +0.2748]**, separated, margin 0.0383 |
| SHUF_ED - S | -0.0642 [-0.0896, -0.0409] (below 0.0383: PASS) |
| accuracy S / ED / SHUF_ED | 0.4075 / 0.6406 / 0.3433 |
| majority rate (lane_keep) | 0.4075; S predicts the majority class in every fold (pred counts 9236/0/0), so S accuracy = majority rate |
| class shares lane_keep / turn_left / turn_right | 0.4075 / 0.2943 / 0.2982 |
| n windows / episodes | 9,236 / 395 |

## Population
Same 12,201 strided windows / 426 episodes, folds (`fold_of`), HGB (RD.HGB), feature sets RD.FEATS["S"] / ["ED"] as RB-JOINT. Windows whose GT 6 s future is incomplete (`gt_ok` False, the rule the pinned T1 tool applies) have no label: 2,965 of the 12,201 strided windows (and 5,925 of 24,207 fold-train windows) are excluded from fitting and scoring, leaving 9,236 / 395. This is the same completeness rule as the card's feasible grid (sec 0a item 3). Fold-train sets use all gt_ok windows of the training clips (RB-JOINT fitted on all fold-train windows, scored the strided held-out ones; unchanged).

## Proof of the label (before use)
GT builder `D.gt_future(cache pose_grid)[t]` (= ClipStore `gt[t0]` = dump `g`) plus `obd_geom.build().labels` (= `read_otband.labels`): on all **2,399 / 2,399** windows (141 episodes) of the STRAT1 C0_s0 eval dump, labels agree, and the g arrays are bit-identical (max abs diff 0.0, float32 equal). Class counts in that dump 1692 / 245 / 462. `PROOF_RBJK.json`. The train-cache `pose_grid` is the same source as the eval cache (same cache builder); the train native-track even-row check was not separately re-run.

## RB-JOINT reference (5-way v9 lat_cls, same windows without the gt_ok drop)
ED - S +0.0581 [+0.0287, +0.0857] (S 0.6575 = majority share, ED 0.7156), SHUF_ED - S -0.0028, n 12,201 / 426.

## Reading notes
* The effect on the kinematic label is much larger than on lat_cls because S has no information beyond the majority and the kinematic label is nearly balanced (lane_keep 0.41); ED reaches 0.64 against 0.41. The 0.2331 is a gain over a majority-class state-only tree, not over a vision arm.
* SHUF_ED is below S because a permuted-label tree predicts the class mix rather than the majority; this is the control reading as expected and is separated on the negative side only.
* Estimator limit: episode draws only; no training-run or inference variance enters (tree is deterministic, random_state 0).

## Files
`F:/TanitAD-scratch/strat1/navrbd/rbjk/`: `rbjk.py`, `RESULT_RBJK.json`, `RESULT_RBJK.md`, `PROOF_RBJK.json`, `RBJK.done`.
