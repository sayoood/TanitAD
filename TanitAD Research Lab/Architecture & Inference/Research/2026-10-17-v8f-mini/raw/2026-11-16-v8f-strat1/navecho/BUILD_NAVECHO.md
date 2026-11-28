# BUILD_NAVECHO -- D-STRAT1-TACLAT-F1 (nav echo or skill cost)

Built 2026-11-17. CPU only, `CUDA_VISIBLE_DEVICES=-1`, zero training, nothing re-dumped, no landed file edited. Nothing committed or pushed. Evidence class MEASURED, tier T1-kin tick 0.

## What was built (all beside the landed fam/ tools)
* `navecho.py` -- reuses `fam/fam_extra.py` (`confusion`, `conf_stats`, `conf_block`, `conf_pair`, `v9_rows`, `load_dump`) and `strat1_lib`; the bootstrap is `taniteval.ci` (paired episode-cluster, B 2,000, seed 0). New: `nav_key`, `nav_key_dist`, `informative`, `fit_map` / `apply_map` / `episode_halves` / `crossfit` (fit on one half of the EPISODES, score the other, swap), `shuffled_nav`, `split_masks`, `net_correct`, `reading` (the committed ECHO / SKILL-COST / MIXED rules), `run`.
* `test_navecho.py` -- 26 tests green (`python -m pytest test_navecho.py -q`, 0.3 s): closed-form values for every function plus 15 source mutations (no-row key, informative set, distance bin edge side and invalid bin, majority vs minority, ignore mask, unseen key, half overlap, fit on the scored half, mean of halves, net-correct sign, mask polarity, split AND vs OR, echo threshold, both seeds vs any, skill needs both, echo needs not-separated) each turning a check RED.
* `render_md.py` -- JSON to `out/RESULT_NAVECHO.md`.
* Run: `python navecho.py` (134 s at B 2,000).

## Layout facts used (source, commit 0ddfb65)
nav_in channels: 0-2 one-hot (FOLLOW, TURN_L, TURN_R), all-zero = no nav row, 3 = distance/100 clipped [0,3], 4 = valid (`v8f_mini_data.py:219-222`, `nav_text_arrays :390-394`, read as such by `strat1_strategic.py:204-210`). C0 arms carry the v9 nav, N1 / S1 / S1d the text nav; control K-FEED checked each family's feed is identical across seeds and arms.

## Choices made by the builder (declared before reading the result, not changed after)
* The committed "uninformative" set is UNINF_BOTH (neither feed names a turn): the only set where no arm holds route information, and a common window set for the paired gap. UNINF_V9 / INF_V9 (C0's own feed) are reported beside it.
* The nav-only predictor's key is the token (4 values). A token x distance-bin version is a sensitivity only; it scores lower (0.477 v9, 0.533 text), because the bins overfit on 10-turn splits.
* One fixed episode split (seed 0) carries the figure; 20 other random splits are reported as a spread.
* K-SHUFFLE band (|shuffled - majority| <= 0.08) was fixed before the run.

## Controls on the run (all pass, problems list empty)
K-LANDED (every arm full-set head macro-F1 equals `fam/out/RESULT_FAM_EXTRA.json` to 1e-12, 17 arms), K-UNION (split plus complement confusion equals the full confusion for every arm), K1 (head against itself reads exactly 0.0 [0.0, 0.0]), K-SHUFFLE (0.3170 against majority 0.3170), K-FEED.

## Honest limit
UNINF_BOTH has 10 labelled turn windows (1 L, 9 R) in 131 episodes; the head split there is the majority-class score, so the committed rules return MIXED and the uninformative read cannot decide. See `out/RESULT_NAVECHO.md`.
