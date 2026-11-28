# ESTCLOSE_READ_DEFS (card PREREG_V8F_PROOF_ADD79, ESTCLOSE10K): the save tool and the reader

Tier of what the reader gives: T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Evidence class: MEASURED. Nothing here is a statement on a vehicle's behaviour on the road.

## 1. Files

| file | what |
|---|---|
| `closeread_save.py` | CPU tool: fits and saves the two estimators (card section 2) and the est-score tables; control e0 |
| `test_closeread_save.py` | pytest of the save tool (12 tests, synthetic dumps) |
| `read_estclose10k.py` | the reader of the card (pairs, names, verdicts, controls g s a1 r1 x1 m1 i1 e2 e3 h1) |
| `podread_estclose10k.sh` | pod runner of the reader (paths as arguments) |
| `run_mutations_ecr.py` | mutation runner for both tools (`--suite save` and `--suite read`) |
| `MUTATIONS_SAVE.json`, `MUTATIONS_READ.json` | results of the runner |

`closeread/closest.py` and `closeread/closeread_fit.py` are imported unchanged. The reader imports the engine modules of card ADD76 (`read_handsrc_decomp`, `read_closrc10k` and under them `read_longroll10k` with its notes N3 and N4) unchanged.

## 2. The save tool

`python closeread_save.py --dump-dir D --out-dir NEWDIR --add78-result JSON [--seeds 0,1] [--n-epochs 120]`; `--selftest`.

* Per reader seed k, from `reader_s<k>_fit.npz`, `reader_s<k>_score.npz` (and `reader_s<k>_lists.json` when present: its `train_ids` go to the leak guard of `run_cell`): N10 = arm N at H = 10 (mean of three nets, seeds 0, 1, 2, 120 epochs), L25 = arm L at H = 25. Same `build_rows`, same `split_by_clip(clips, 0.25, 0)`, ridge weight by `select_ridge` then `ridge_fit` on all fit rows, nets normalised by mean and sd of the train part of the split.
* The net training is a copy of `train_mlp` that returns the weights (`train_state`); the imported file is not edited. The estimate of the fit tool (`run_cell`) is computed in the same process and compared.
* Control e0, computed in full before anything is written:
  * e0a: repeated training equals `run_cell` on the score rows, at most 1e-7 m/s;
  * e0b: the numpy forward (`closest.ClosingEstimator`, loaded from the exact bytes that are written) equals the fit tool row by row: L25 at most 1e-6, N10 at most 1e-4 m/s;
  * e0c: the per-seed Pearson r of the saved estimator on the score rows is at most 0.005 from `cells.N_H10.r_per_seed[k]` / `cells.L_H25.r_per_seed[k]` of the add78 result (index k = position of the seed, as in the fit tool where seeds are 0, 1);
  * e0d: the table rows contain the rows of `build_rows` (same X, same frames) and the table equals the estimate on them.
  The tool prints PASS or FAIL and the largest differences; it prints no r value. A failed check: rc 1, no out dir.
* Writes: `closest_N10_s<k>.npz`, `closest_L25_s<k>.npz`, `planted_zero_H10.npz`, `planted_const2_H10.npz`, `planted_readerclos_H10.npz`, `est_score_<N10|L25>_s<k>.npz` (`clip_index`, `frame`, `est` for every row of the score dump; NaN where the H frames ending at the row do not all carry a valid finite read in one clip), `SAVE_RECORD.json` (sha256 of inputs and outputs, e0 flags and differences, tolerances). An existing out dir is refused (rc 2).
* The estimate for a look at frame j uses the reads of frames j-H+1 .. j, so the table row (clip_index, j) is what a rollout look at frame j has to reproduce.

## 3. The reader

Settings AGE (grid_p30, pre-roll 30) and FRESH (pinned grid, pre-roll 0), each read completely before the next. Run names (engine form of the run table: `RRe`, registered in memory like RRc and RRh of ADD76): AGE `RQ50eN10KF5P30Hb20`, `RQ50eL25KF5P30Hb20`; FRESH `RQ50eN10KF5Hb20`, `RQ50eL25KF5Hb20`; the banked forms have the names of the ADD76 read.

Pairs (A minus B) per estimator E: `EST-CLOSING-E` (RRe, RR), `EST-VS-LABEL-CLOSING-E` (RRe, RRc), `EST-VS-LABEL-FED-E` (RRe, RLc), `EST-VS-NO-RULE-E` (RRe, no rule). Internal, not named: RR-VS-LABEL-FED, RR-VS-NO-RULE, LABEL-CLOSING (RRc, RR). Names `ESTCLOSE10K-<PAIR>-<SETTING>-<C|U|family>-<word>`; U, C, F (falling behind) per pair are printed where the headway instrument is qualified.

Verdicts per E: `ESTCLOSE10K-E-SUPPORTED` when AGE C is LOWER, AGE U not HIGHER, no AGE family WORSE, FRESH C not HIGHER, FRESH U not HIGHER, no FRESH family WORSE (all on pair EST-CLOSING-E); else `-NOT-SUPPORTED`. A single SUPPORTED of the two also gets `ESTCLOSE10K-E-SUPPORTED-ONE-OF-TWO`. A condition that needs a word of an unqualified setting: no name unless another condition already fails (as in ADD76).

Share of the contact gap (printed at AGE, names nothing): gap = C(RR) - C(RRc) = minus the pooled C of LABEL-CLOSING; removed = minus the pooled C of EST-CLOSING-E; share = removed / gap; no share when the gap is not positive.

Controls, all printed first; any FAIL: no name and no value is printed or written.
g, s (engine grid read, strategic read), a1 (levels U, C, F per seed of the five banked forms equal `settings/<setting>/levels` of `READ_MIDLOOK10K.json` to 1e-9; needs `controls_all_pass` true there and levels not the "not reported" text), r1 (run text of the RRe runs: form RRe, replan 10, rule every 5, readout causal_g, kappa 0.2, pre-roll, wrapper hash given by `--wrapper-sha`, rc 0; dump file count = clips of the grid), x1 (RRe feeds a closing that is not 0 on a present row in both logs, exactly 0.0 on present rows without a full history, form RRe in the rule log; RR feeds 0 on every row), m1 (later-look row counts of RRe equal RR), i1 (level identities to 1e-12), e2 (FRESH: `c_est` of both logs equals the table at (ci, j) on `est_ok` rows, at most 1e-3 m/s, rows with k == 0 among them, `c_est` NaN where `est_ok` is False), e3 (below), h1 (planted name rule: LOWER, HIGHER, opposite seeds, no effect).

e3: per setting and estimator, the share of `est_ok` over the logged looks (rule log and later-look log together, both seeds), and the counts of the run-text line. The control passes when every number is present. At AGE the estimator is admitted (gate) only if the scored-window share is at least 0.90 BOTH by the logs and by the run-text line; otherwise the AGE names of that estimator and its verdict are not written, the reason is printed, and the other estimator is still read. The pre-roll share is reported, not judged.

## 4. Assumptions about the pod layout and the wrapper (not verified: no pod access)

1. Folder layout as in the ADD76 read: `PULL/I10<run>_s<k>/ticks/{run_<key>.txt, rule_log_<key>.npz, rule_log_mid_<key>.npz}` and the dump folder the engine finds with `R.dump_dir`. The runner takes `--pull` (a link folder as in `podread_midlook10k.sh`), `--code`, `--grid-dir` (holds `grid_p30.json`), `--fresh-grid`, `--midlook-json`, `--est-dir` (the save tool's out dir, holding `est_score_*`), `--w`, `--tree`, `--out`, `--wrapper-sha` (16 hex digits), optional `--read-dir` (default the current folder: the reader and the engine modules) and `--py`.
2. The queue marker `ESTCLOSE10K_DONE.txt` (and no `ESTCLOSE10K_STOP.txt`) in `--w` is my assumption, by analogy to MIDLOOK10K_DONE.txt. `HEADWAY_JOIN` and `HEADWAY_BANK` default to `$CODE/../data/...` unless already set in the environment (ADD76 used `$P/data/...`): set them when the layout differs.
3. Interface of the wrapper generation, as given: rule logs hold `ci`, `j`, `k`, `c_est` (float64, NaN where no full history), `est_ok` (bool), row-aligned; run text holds `est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>`. The reader does not check the `file` hash against the estimator file; it only requires it to be 64 hex digits.
4. "Rows of the first replan tick" of e2: taken as the rows with `k == 0`. If the wrapper numbers its ticks differently, e2 needs that definition changed (a mutation test pins the current one).
5. The presence of the `present` and `c_src` arrays in the logs (as in the ADD76 logs) is assumed for x1; for RRe the fallback check uses `present & ~est_ok`.
6. The save tool takes `lists.json` with a `train_ids` list when present; the dump layout is that of `closeread/CLOSEREAD_DEFS.md`.
7. Never run on real data here: the engine part of the reader (g, s, grid read, window arrays) is not exercised by the selftest; only the controls, names, verdicts and the printing are, on synthetic and planted input.
