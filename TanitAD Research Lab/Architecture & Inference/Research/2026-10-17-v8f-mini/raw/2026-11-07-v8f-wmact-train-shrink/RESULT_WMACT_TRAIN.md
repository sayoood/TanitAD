# RESULT_WMACT_TRAIN: the same statistics on TRAIN windows, beside eval-141 (card `PREREG_V8F_WMACT.md`, frozen prefix 5,478 bytes `d21b14cc`)

MEASURED 2026-11-07 on the dev-box CPU. Tier T0: teacher-forced, logged actions, true future frames. A WM diagnostic, never a driving claim.

## Controls (TRAIN read)
| control | M2b s0 | M2b s1 |
|---|---|---|
| 1. identity donor: sens max abs, ACTION-USE max abs | 0.0 , 0.0 | 0.0 , 0.0 |
| 2. zeroed level-1 action embedding: ACTION-USE at k 1 / 5 / 10, sens at L1 | 0.0 / 0.0 / 0.0, 0.0 | 0.0 / 0.0 / 0.0, 0.0 |
| 3. rollout from another window's start: SKILL at L1 k1 (must be below 0, separated); least-negative row of all 7 | -42.7332 [-46.6085, -39.3924]; L3/k3 -2.7124 | -42.8624 [-46.9519, -39.2969]; L3/k3 -2.2131 |
| 4. WM loss via `HJEPAStack.loss` against `WMAux.forward`, 64 TRAIN windows, max abs diff | 0.0 | 0.0 |

Control 3 is below 0 and separated at every level and horizon read, both seeds.

## Sample
* eval-141: 1,976 windows of 2,399 (all 16 frames real), 141 episodes (clips).
* TRAIN: 1,833 windows of 2,397 (all 16 frames real), 141 episodes (clips); 17 starts per clip, 13 of which hold 16 real frames. Clips sha256 `e9e0e2c49c317d55d524bbb8bdf59d6e2b849997c54c9549067ddc9d4e6d3629`.
* Estimator: episode-cluster bootstrap B 2,000, seed 0, `taniteval/ci.py` draws; * = the 95 % interval excludes 0. Another draw of episodes (not another training run, not another inference).

## Requested rows
| row | eval s0 | TRAIN s0 | eval s1 | TRAIN s1 |
|---|---|---|---|---|
| L1 k1 (0.2 s) SKILL | +0.0010 [-0.0114, +0.0118] | -0.0010 [-0.0141, +0.0122] | -0.0010 [-0.0141, +0.0103] | +0.0063 [-0.0078, +0.0196] |
| L1 k5 (1.0 s) SKILL | -0.0689 [-0.1145, -0.0284]* | +0.0009 [-0.0456, +0.0430] | -0.0577 [-0.1024, -0.0183]* | +0.0084 [-0.0372, +0.0489] |
| L1 k10 (2.0 s) SKILL | -0.1500 [-0.2324, -0.0777]* | -0.0564 [-0.1411, +0.0157] | -0.1724 [-0.2609, -0.0938]* | -0.0616 [-0.1458, +0.0102] |
| L1 k1 ACTION-USE | +0.0054 [+0.0025, +0.0085]* | +0.0159 [+0.0105, +0.0225]* | +0.0049 [+0.0015, +0.0084]* | +0.0155 [+0.0103, +0.0211]* |
| L1 k5 ACTION-USE | +0.0478 [+0.0327, +0.0634]* | +0.0938 [+0.0727, +0.1175]* | +0.0509 [+0.0350, +0.0674]* | +0.0919 [+0.0666, +0.1192]* |
| L1 k10 ACTION-USE | +0.0897 [+0.0620, +0.1166]* | +0.1464 [+0.1155, +0.1796]* | +0.0900 [+0.0624, +0.1192]* | +0.1382 [+0.1001, +0.1781]* |
| L2 k1 (0.4 s) SKILL | +0.0295 [+0.0095, +0.0475]* | +0.0471 [+0.0213, +0.0732]* | +0.0264 [+0.0061, +0.0436]* | +0.0578 [+0.0318, +0.0835]* |
| L3 k1 (0.8 s) SKILL | +0.1316 [+0.0953, +0.1694]* | +0.1801 [+0.1253, +0.2336]* | +0.1351 [+0.0925, +0.1790]* | +0.1897 [+0.1370, +0.2400]* |

## Also computed (same statistics, deciding nothing here)
| row | eval s0 | TRAIN s0 | eval s1 | TRAIN s1 |
|---|---|---|---|---|
| L1 k1 ACTION-USE-MEAN | +0.0029 [+0.0005, +0.0052]* | +0.0055 [+0.0026, +0.0089]* | +0.0030 [+0.0002, +0.0061]* | +0.0055 [+0.0026, +0.0088]* |
| L2 k1 ACTION-USE | +0.0424 [+0.0286, +0.0576]* | +0.0853 [+0.0629, +0.1092]* | +0.0462 [+0.0284, +0.0660]* | +0.0853 [+0.0617, +0.1113]* |
| L3 k1 ACTION-USE | +0.1850 [+0.1364, +0.2343]* | +0.2800 [+0.2195, +0.3389]* | +0.2000 [+0.1389, +0.2643]* | +0.2737 [+0.2127, +0.3354]* |
| L2 k3 SKILL | +0.1084 [+0.0588, +0.1576]* | +0.1818 [+0.1206, +0.2383]* | +0.1185 [+0.0622, +0.1740]* | +0.2020 [+0.1449, +0.2537]* |
| L3 k3 SKILL | +0.3283 [+0.2624, +0.3873]* | +0.4217 [+0.3451, +0.4853]* | +0.3175 [+0.2459, +0.3823]* | +0.4308 [+0.3573, +0.4927]* |

n per row: eval 1,976 windows / 141 episodes; TRAIN 1,833 windows / 141 episodes. Every row above uses these n (all rows share the windows).

## Outcome by the brief's rule
* TRAIN L1 k1 SKILL: M2b_s0 -0.0010 [-0.0141, +0.0122]; M2b_s1 +0.0063 [-0.0078, +0.0196]
* **L1 DOES NOT FIT** (TRAIN L1 k1 SKILL is not above 0 and separated at both seeds).

Beside, `sens_rom/all` at L1 k1 (relative action sensitivity): eval M2b_s0 0.189, M2b_s1 0.200; TRAIN M2b_s0 0.226, M2b_s1 0.237.

Wall time: stream and latents 271 s; statistics and controls 34 s.

## Reading (point estimates; eval and TRAIN are different episodes, so no eval-minus-TRAIN difference was tested)
* **L1 does not beat "nothing changes" on its own training clips.** TRAIN L1 k1 SKILL is -0.0010 / +0.0063, intervals straddle 0 at both seeds; eval was +0.0010 / -0.0010. The same holds at k5 (+0.0009 / +0.0084). At k10 the TRAIN points are -0.0564 / -0.0616 with upper ends +0.0157 / +0.0102, not separated; on eval k5 and k10 were separated below 0.
* **The action dependence is fitted harder to TRAIN than to eval while the net skill stays near 0.** L1 k1 ACTION-USE is +0.0159 / +0.0155 on TRAIN against +0.0054 / +0.0049 on eval (about 3x); ACTION-USE-MEAN is +0.0055 / +0.0055 against +0.0029 / +0.0030. `sens_rom/all` at L1 k1 is 0.226 / 0.237 against 0.189 / 0.200.
* **The upper levels show the fit-versus-generalisation gap that L1 lacks.** L2 k1 SKILL is +0.0471 / +0.0578 on TRAIN against +0.0295 / +0.0264 on eval; L3 k1 SKILL is +0.1801 / +0.1897 against +0.1316 / +0.1351; ACTION-USE at k1 is 1.9x to 2.0x larger on TRAIN at L2 and 1.4x to 1.5x at L3. Both seeds show it.
* **Control 2 shows what the L1 action embedding is worth.** With the embedding zeroed, TRAIN L1 k1 SKILL is -0.0177 [-0.0318, -0.0042] / -0.0154 [-0.0307, -0.0009] (separated below 0); the real model sits 0.017 to 0.022 higher, so the action input lifts a predictor that is otherwise worse than copy to about copy.
* **Limits.** 141 of the 4,512 TRAIN clips (3.1 %, the smallest sha256), 17 starts per clip, 13 of which hold 16 real frames: 1,833 windows. The rollout uses context 1 (the planning convention), while training is teacher-forced with context 3; the eval read uses the same convention, so the comparison is like for like, but this is a statement about this read-out, not about the training loss itself. The interval answers another draw of episodes, not another training run; the two seeds fall in the same band.
* **Implementation choices, logged here because the card does not fix them.** (1) Window extraction is `ClipWins.item` in `collect_latents_train.py`: the feats / act / frame_ok expressions of `V8FMiniWindows.item` (`v8f_mini_data.py:573-597`), equal array for array to the real `item` on 60 eval windows (`test_train_item_shim.py`, 12 of them with a padded frame); the real `item` needs a label join that the TRAIN store here does not have. (2) The valid starts are the trainer's own `D.train_windows` (`v8f_mini_data.py:607-619`, called at `v8f_mini_train.py:2411`) on a two-key join, then `np.unique(np.round(np.linspace(first, last, 17)))`. (3) Control 4 is computed in the stream on the first 64 TRAIN windows, with the expressions of `run_wmact.py`. (4) Every clip file was md5-checked against box 2's manifest (141 of 141 equal, no retry). (5) The exclusion is `sorted(clips, key=sha12)[540:600]` (`v8f_mini_train.py:2383-2388`); the resulting train and excluded digests equal M2b's own `summary.json` values (`train_sha256` `37f4f9ee...`, `excluded_sha256` `f6a1dee0...`), and the 141 picks are positions 0 to 140 of the sha12 order, none excluded.
