# Definitions of the reader of card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0)

Tier T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance). The four families (longitudinal, lateral, tactical, strategic) are not read: n = 0 each, a T0 read of a perception head produces no trajectory.
Files: `read_boxsize.py` (numpy only, imports no other kit), `podread_boxsize.sh`, `pull_boxsize.sh`, `read_args.txt`, `test_boxsize_read.py`.

## Size encoding (the same two lines as `decode` of the fit kit)
`u = ln(angle in degrees) - 1.6`; `az = 16 u_w`, `el = -2 + 8 u_h`.
`decode(a, b)`: width = `exp(1.6 + a / 16)`, height = `exp(1.6 + (b + 2) / 8)`, degrees. The limits `u` in -2..+2 are 0.67..36.6 degrees. No clipping is applied on decode.

## Map G (card section 2)
Train frames: frames of the clips of the train list (first token per line) with `has_lead`, finite `gap_re`, finite positive `ang_w` and `ang_h`; no cut on the gap. Features `(ln h, ln w)`, each scaled by its mean and standard deviation over the train frames
(a zero standard deviation is replaced by 1). A query: the 25 train frames nearest by Euclidean distance in the scaled features, ties by the lower train index (argpartition gives the 25th distance, the candidates are re-sorted stably in index order); the read is the median of their `gap_re`.
A query that is not finite gives a nan gap. The gap of fit k on a frame is G(decoded width, decoded height).

## Rows
Evaluation join: `--eval-join` (npz: `clip`, `n`, `off`). Banked dumps of arm D: `D_0`, `D_1`, `D_2`, laid on the join rows by (clip name, clip index, frame). t3 rows = rows with `lab_present` and a finite `lab_gap` in D_0 and a finite gap in D_0 to D_2
(the rows of control t3 of card ADD122; not clamped to 60 m). The label gap of every error is `lab_gap` of D_0. Prediction files `<O>/pred/LOC_s0.npz`, `LOC_s1`, `LOC_s2`, `LOC_s0_perm` give S_0, S_1, S_2 and SP_0; rows without an output are nan.
Label sizes of the t3 rows come from `<box-dir>/<clip>.npz` (`has_lead`, `ang_w`, `ang_h`; finite and positive).

## Controls (stage `controls`; all seven must pass; exit code 0, else 2)
* s0: sha256 of `train_list` and `eval_list` equal the entries of the banked fit record (`--list-record`; keys `*_sha256`); every locator `FIT_RECORD.json` of the four fits carries the same two hashes; every record (`FIT_RECORD.json` and `BOXSIZE_FIT_RECORD.json`) names a token root in its argv (`--root`) and it is one root.
* s1: `<O>/IDENTITY_K5.json` has `PASS` true, `abs_diff` at most 1e-4 and equal sample counts; `<O>/BOXSIZE_IDENTITY_RECORD.json` exists (its `locator_fit_sha256`, when present, equals that of the fits).
* s2: each `BOXSIZE_FIT_RECORD.json` has `module_values_changed` equal to exactly the one item `load_box`, `module_values_restored` true, the same `locator_fit_sha256` in all four, seed and `permute_targets` as the arm; encode then decode of the label sizes of the t3 rows inside the limits returns them within 1e-9.
* s3: E of D_0, D_1, D_2 on the t3 rows within 0.001 m of 3.960, 3.913, 4.112, and 3,841 rows.
* s4: E of G on the label sizes of the t3 rows at most 2.2 m; E of G on label sizes permuted across those rows (pairs permuted together, `RandomState(124)`) at least 0.95 of E of the constant (median label gap of the t3 rows); every train clip has a box file.
* s5: E of the gap through SP_0 at least 0.95 of E of the same constant.
* s6: at least two test outputs (`--test-out NAME=PATH`), each ending with a `<n> passed` line without failed or error, each file time before `--test-before NAME=EPOCH`; every t3 row has a finite size output of S_0, S_1, S_2 and SP_0.
In this stage no gap error of arm S and no size error of a fit is computed. The controls file holds pass flags, failure texts, the inputs' sha256 and the control values of s3, s4, s5.

## Values and names (stage `names`)
Refuses (rc 3) unless the controls file is all-pass with s0 to s6 present and every input hashes as at the controls; refuses (rc 2) when the output or the log exists.
* E per fit; ratio R = mean E(S_0..S_2) / mean E(D_0..D_2); interval: cluster bootstrap over the evaluation clips of the ratio of the per-clip sums, 2,000 draws, `RandomState(124).randint(0, n_clips, size=(2000, n_clips))`, two-sided 95 % percentile, fits held fixed.
* Signed error (read minus label) and share more than 5 m short, mean of the three fits, by label-gap bin 0-10, 10-20, 20-40, 40-60 m (bins closed below, the last also closed above).
* Names: ratio interval upper end below 1 -> `BOXSIZE-T0-CUTS-GAP-ERROR`; lower end above 1 -> `BOXSIZE-T0-WORSE`; else `BOXSIZE-T0-NOT-SEPARATED`. Far leads (label gap 40-60 m): `BOXSIZE-T0-FAR-LEADS-READ` when |mean signed error of S| is at most half of |D's| and the paired cluster bootstrap interval of the difference (S minus D, same draws) excludes 0, else `BOXSIZE-T0-FAR-LEADS-STAY-SHORT`.
* Bar for going on: R at most 0.85 with the interval upper end below 1. Both names and the bar are derived twice, the second time apart (per-clip sums, plain loops, hand-made percentiles); disagreement stops with rc 4.
* Printed beside, judged by nothing: size read error (mean absolute error of ln width and ln height, in per cent, mean of three fits) by gap bin; E of G on the label sizes by bin and overall; share of label sizes outside the limits (width, height, either).
* The interval answers what another draw of evaluation clips would say, not what another fit would say; a separated interval is necessary, not sufficient.

## Definitions the card leaves open (most literal reading taken)
1. "The same token root" is read from `--root` in the argv of each record; the banked record of arm D is not asked for a root.
2. Evaluation rows without a finite positive label size are left out of the G-on-label-sizes figure (n printed); they do not fail s4.
3. The test-output rule needs two outputs (one before the fits, one before the read).
