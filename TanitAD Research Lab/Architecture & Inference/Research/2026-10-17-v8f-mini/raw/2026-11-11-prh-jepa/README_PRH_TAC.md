# README_PRH_TAC

Card `PREREAD_PRH_TAC.md`, sha256 647c8ed5a10aa575ef01e081ce37b94a6ae21c983c5b9aa16b9fd785f2bd6fbd (checked before any code ran). Code: `prh_tac.py` (reuses `prh_seed.py`: extraction, `fit_cca`, `ci95`, `is_sep_*`; `prh_seed_v2.py`: `_ctx` split + bootstrap counts, `metric("CC32")`, `permuted`). CPU, 2 threads, `tanitad` asserted under the read tree.

## Outcome (MEASURED, ours; `RESULT_PRH_TAC.json` / `.md`, sha256 of the json b4fe1dcb32de67a8f8c79a11911bfad5998f102b371085ad8020ba39af09ae3a)
**NO-TAC-LEAN**, controls all pass. Q1 is separated positive at seed 1 only.

| seed | S(TR,TAC) | S(RI,TAC) | diff | CI95 | separated, positive |
|---|---|---|---|---|---|
| s0 | 0.5133 | 0.5054 | +0.0079 | [-0.0208, +0.0368] | no |
| s1 | 0.5338 | 0.5014 | +0.0324 | [+0.0107, +0.0518] | yes |

d_TAC = 25 (34 target columns, 3 constant dropped = 31, rank-reduced to 25; CC over 25 dims). Frames: 8,781 TAC frames of 14,237 (fit 4,393 / score 4,388); score clips with frames 67 of 70. Wall: extraction + analysis ~22 s the first time, 5.9 s with cached extraction.

Beside rows (no bar): S(TR0,TAC) - S(TR0,SLOW) = -0.432 [-0.462, -0.405]; same for s1 = -0.410 [-0.439, -0.385] (S(TR,SLOW) = 0.945 / 0.943 on these frames); S(PCA,TAC) = 0.614 [0.587, 0.635]. R2 recovery row: see below, not interpretable.

Controls: C-SELF S = 1.0 (pass). C-SHUF S(TR0,TAC_shuf) -0.0035, S(TR1,TAC_shuf) -0.0081, both below the v2 FLOOR 0.9777 (pass). C-LABEL-SHUF (seed 11): Q1 s0 +0.0185 [-0.0086, +0.0446], s1 +0.0117 [-0.0175, +0.0415], both non-separated (pass; also non-separated two-sided). Four extra shuffle seeds, points only: s0 +0.0123/+0.0115/-0.0068/-0.0145, s1 -0.0370/+0.0041/+0.0088/+0.0093. Over all 10 shuffled Q1 points the mean is +0.002 and the sd 0.017, so s1's real +0.0324 is ~1.8 sd above the label-shuffled spread and s0's +0.0079 is inside it. The bootstrap resamples episodes with the TAC fit held fixed, so it does not see this fit-to-fit spread (the same blindness `H-ESTIM-SEED-1` records for training runs); the card's outcome rule was applied as written and this is not used to change it.

## Label source and frame rule (file:line)
* Source: `join_eval141.npz` (the file `--eval-join` loads at `stack/scripts/v8f_t1_eval.py:515`, `D.load_join`), built by `tanitad/data/v8f_mini_data.py:build_label_join` (lines 165-240). The trainer's tactical code targets are packed at `v8f_mini_data.py:_lab_pack` (607-615) and defined at `tanitad/models/v8f_mini.py:_code_gt` (lines 891-902: goal bits `y*w + 0.5*(1-w)`, lat/lon one-hots from the allowed sets).
* Frame-to-window rule: none needed. Labels are per frame: join row `off[c] + j` is cache step `j` of clip `c` (`v8f_mini_data.py:162-163`), the same step index the v2 extraction stores. Asserted in the run: clip order identical and `n[c]` equals each clip's extracted frame count. The 2,399-window eval grid (`eval_grid_windows.json`) is not used.
* Frames: v2 `ok` AND `has` AND single-bit `lat_allowed` and `lon_allowed` (13,896 have a label; 8,781 are single-class in both). Lat classes present 0,1,2; lon 0-6. All representations (TR, RI, TAC, SLOW, PCA) are scored on exactly these frames.
* Dropped columns (of 34): goal bit 7 (never labelled) and lat classes 3, 4 (absent).

## Choices left open (all also in the JSON `choices`)
* Goal bits unlabelled for a frame (`goal_w` = 0) take 0.5, the trainer's own target rule.
* TAC ridge: lambda 1e-3 x mean eigenvalue of nm's covariance on the fit-half TAC frames, centred by fit-half means.
* **Rank reduction (added after a first run).** The first run, with the 31 non-constant outputs as the representation (d = 31), FAILED C-SELF (S(TAC,TAC) = 0.8341) and was therefore unusable under the card. Cause: the one-hot blocks sum to 1 and further goal bits are dependent, so the ridge outputs carry exact null directions whose canonical correlations are noise. TAC is therefore rotated onto the eigen-directions of its fit-half covariance above 1e-8 x the largest (d = 25); this is an orthogonal change of basis inside the span, so it does not alter which subspace TAC is. The choice was made on the C-SELF failure alone. The unusable first run is recorded for completeness: Q1 s0 +0.0016 [-0.0277, +0.0328] (no), s1 +0.0241 [+0.0010, +0.0464] (yes); same sign pattern as the final read. Its scratch file was deleted.
* C-LABEL-SHUF construction: the fit clips' target blocks are re-dealt over a random permutation of the clips (seed 11), each block tiled or truncated to the receiving clip's frame count. Pass = Q1 not separated-positive, decided before reading it.
* SLOW and PCA are built as in `prh_seed_v2.step_b` on all ok fit-half frames (that block is inline in `step_b`, so `build_slow_pca` re-states those lines; it calls the same `P.fit_cca`) and then evaluated on the TAC frames.

## UNVERIFIED / caveats
* **The R2 recovery row is not interpretable.** With nm's 1024 dimensions, a ridge at 1e-3 x mean eigenvalue and a score half of 67 clips, R2 is negative for every source (nm -1.12, TR0 -0.54, TR1 -0.59, RI0 -0.71, RI1 -0.75) with very wide CIs (nm [-24.3, -0.97]); the "fraction" ratios of negative numbers (TR 0.48/0.53 vs RI 0.63/0.67) carry no meaning. Fractions and CIs are in the JSON only as raw output. The readout was run as the card describes; its failure is a property of this ridge setting and the 34 targets (per-column R2, some near-constant columns), not a finding.
* CC32 here is the mean of 25 correlations, not 32; values are not comparable in level to v2's 0.9+ (a 25-d TAC against 256-d adapters). Q1 compares TR with RI on the same TAC, so the level is common to both.
* Only 2 trained seeds and 2 random-init seeds (torch 100 / 101), each paired by index; no replicate of the training run.
* Separation at s1 (+0.032) is small against the label-shuffle spread (sd 0.017, 10 points) and the two seeds disagree.

## Tests
`test_prh_tac.py` (7 tests, planted data): a representation containing the TAC subspace gives Q1 positive and separated; a label-shuffled TAC gives a diff below 25 % of the real one and S below 0.7 x real; shuffle keeps score rows and changes fit rows; C-SELF = 1 after rank reduction of exactly dependent targets; the outcome table (TAC-LEAN, NO-TAC-LEAN, UNUSABLE); the Q1 separation rule; target building (single-class mask, 0.5 imputation). With the v2 tests: 20 passed. A planted check showed that a shuffled-label TAC can read "separated" at a small size on isotropic data (3 of 4 seeds), which is why the planted test asserts size, not the CI flag.
Mutation check (`tests/mutation_check_tac.py`, record `tests/MUTATION_PRH_TAC.json`): flipping the Q1 separation rule, replacing `all` by `any` in the outcome rule, and making the label shuffle the identity each turn the suite RED (clean run green); `all_detected` true.

## Deleted temp data (own; re-creatable)
* `F:/TanitAD-scratch/prh_seed/work/extract.npz` 116,894,913 B
* `F:/TanitAD-scratch/prh_seed/work/tac_out` 27,796 B (scratch copies of the result files and the first-run files; the final files live here)
* `F:/TanitAD-scratch/prh_seed/work/diag_shuf.py` 423 B
Nothing staged, committed or pushed. No box-2 access.
