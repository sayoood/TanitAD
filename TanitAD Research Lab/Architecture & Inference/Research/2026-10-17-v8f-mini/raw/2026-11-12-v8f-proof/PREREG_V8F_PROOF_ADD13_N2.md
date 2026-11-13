# Note N2 to PREREG_V8F_PROOF_ADD13 (DATAEFF): how the chosen clip set is passed, gate g1, list facts (2026-11-13)

Written before any arm of the card exists. No arm, seed, step count, read, bar or outcome name of the card moves. Source of the facts: `dataeff/DATAEFF_DEFS.md`, `dataeff/G1.json`, `dataeff/CONTROLS.json`, `dataeff/HANDOFF.md` (label statistics and source reading only; no model output).

## N2a. S-1/4 is passed as a restricted cache manifest with the FULL training join (lead decision)

The card (section 3) says the chosen set is passed as a prebuilt join restricted to those clips. MEASURED from source at commit 3e264bbf:
* the trainer takes its clip set from the train manifest of the cache root, not from the join (`v8f_mini_train.py:2560-2561`);
* `train_windows` indexes the join for every manifest clip (`v8f_mini_data.py:611-619`), so a restricted join alone stops with a KeyError;
* the normalisation of the constraint head is computed from the join (`v8f_mini_train.py:2682`, `cons_stats(J)`), so a restricted join changes that normalisation (largest shift of a standard deviation 0.33, measured on the S-1/4 join): a second lever beside the clip set.

Decision: S-1/4 runs on a cache root whose train manifest holds the 1,128 chosen clips (`dataeff/s_cache_setup.py`, links to the same clip files), with the SAME full training join as every other arm and no `--exclude`. The constraint statistics are then identical in all arms of the card and only the clip set moves. The file `join_S_quarter.npz` is not used.
Control d5 reads accordingly: the restriction applied with the R-1/4 list gives the R-1/4 window count (76,631) and 1,128 clips; the selfcheck of S-1/4 must report 1,128 clips, 0 excluded and 76,663 windows.

## N2b. Gate g1 (label statistics, read before any arm as the card allows)

Mean clip score of S-1/4 0.8665, of R-1/4 0.5986, ratio 1.4475 against the bar 1.3: PASS. S-1/4 is trained.
Score distribution over the 4,512 eligible clips: median 0.72, 13.9 % of the clips at 0, 12.5 % at 1. The 846 chosen clips have mean score 0.989; 146 clips tie at the cut score 0.941 and hash order decides among them; the floor of 282 clips has mean 0.50.

## N2c. Facts fixed by the tool

* `--exclude A:B` is a slice of the ascending hash order: start inclusive, end exclusive (`v8f_mini_train.py:2576`). Ranges: R-1/2 `540:2856`, R-1/4 `540:3984`, R-1/8 `540:4548`. The lists are nested; F-1/4 uses the R-1/4 list (equal sha256).
* The full arm has 4,512 clips and 306,566 windows; 4,791 steps per pass at batch 64, times 4, is 19,164. `--steps 19164` is passed explicitly beside `--steps-mult 4` and wins.
* The window rule on train windows uses the logged future of the clip (the join carries no trajectory); applied to the eval windows it gives 1,305 selection-critical and 1,094 bulk, the HIERWHERE counts, exactly (control d6).
* No eval clip is in any list (control d2: 0 of 141).
* The lists were built from the box-2 copy of the train manifest. Before the arms, the tool is rerun on the pod's own manifest; the clip id sets must be equal, else the lists are rebuilt there and g1 is read again.
