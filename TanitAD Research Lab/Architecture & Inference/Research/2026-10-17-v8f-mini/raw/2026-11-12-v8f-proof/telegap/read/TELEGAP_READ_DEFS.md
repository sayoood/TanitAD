# TELEGAP read kit: what is coded (card PREREG_V8F_PROOF_ADD122 and its note N1, tier T0)

Dev-box folder `F:/TanitAD-scratch/v8proof/telegap/read/`; pod copy `/workspace/v8proof/telegap/kit/read/`. Written and tested on synthetic dumps only (43 tests); nothing has run on the pod and no result file was opened. Evidence class of every statement about real data below: ASSUMPTION unless it says otherwise.

## Clauses that could fail by construction on real data

1. **Dump format of the new dumps.** The reader lays every dump with `read_lookgap.lay` / `read_lookat.on_rows`, so a dump needs the keys `clips`, `clip_index`, `frame`, `gap`, `lab_present`, `lab_gap` (as the banked reader dumps of cards ADD99 / ADD108) and, for M_0..M_2 and F_0..F_2, a bool array `tele_dropped` of the same length. A new dump without `tele_dropped` makes t3, t4 and t5 FAIL ("could not be computed"), not just the names. D_*, D2_0 and FP_0 need no such key. A `const_gap` scalar key in the FP_0 dump is used as the constant of t5 when present.
2. **Where the three banked D dumps lie (not verified).** Defaults are taken from `read_lookgap.dump_paths`: D_0 = `/workspace/v8proof/cropread/run1/dump/D_H5/reader_s0_score.npz`, D_1 = the same folder `reader_s1_score.npz`, D_2 = `/workspace/v8proof/visseed10k/fitroot/dump/D_H5/reader_s2_score.npz` (that reader's D0, D1, D2; its D3, D4 are not used). If they lie elsewhere, `--dump D_0=...` overrides.
3. **t3 row count.** t3 uses the rows with `lab_present` of D_0, a finite `lab_gap`, and a finite gap in D_0, D_1, D_2 (the ADD108 base rows also required D3 and D4 finite). The card says 3,841 frames; if the three fits alone give another count, t3 FAILS on the count although the three errors may reproduce. Not clamped to 60 m (as the banked base), so t3 is the banked definition.
4. **"Lead within 60 m"** in the common frame set is `0 <= lab_gap <= 60` on top of the t3 definition (the number of lead rows beyond 60 m is printed). If banked label gaps exceed 60 m, the common set is smaller than the t3 rows by more than the dropped frames.
5. **t0 input format is a guess.** The banked list record must be a JSON whose entries (top level, or under `files`) are 64-hex sha256 values named `train_list_sha256`, `eval_list_sha256` (a trailing `_sha256` or `_sha` is stripped; further entries such as `frames_list_sha256` are compared when a matching `--list-file frames_list=PATH` is passed). `train_list` and `eval_list` must be passed. A record with other key names FAILS t0 with "no entry in the list record".
6. **t1 inputs.** Stage rows come from the stage record(s) (`{"clips": [ {clip, status, width, height, n_frames, crc_ok} ]}`, status `ok` or `skipped_complete`; later records override earlier ones only when they are ok), the encode `MANIFEST.json` (`clips.<id>.n_frames`), the match files `<id>.npz` key `mask`, and the pilot clip list the encode used (`--all-clips`, 641 clips; `--expect-clips`). A clip is bad when it has no ok stage row, no manifest entry, no match file, fewer than 0.95 of its frames matched, or a manifest frame count that differs from the match mask length. Pilot frames = the mask length. 5 per cent of 641 is 32 clips (bound is `bad <= 0.05 * n`).
7. **t2** requires `threshold` to equal 0.9 (a tele kit run with another threshold fails) and `n_frames > 0`. The kit's note about texture and the clock offset (TELE_ENC_DEFS section 0, clauses 1 to 3) applies: t2 can fail with correct calibration.
8. **t4** compares E(D2_0) on the t3 rows with `[3.913 - 0.15, 4.112 + 0.15] = [3.763, 4.262]` (min and max of 3.960, 3.913, 4.112, 4.079, 4.056). D2_0 must be finite on all t3 rows. The fit record must have `seed` 0, `n_win` 2, an integer `n_tok`, a non-empty `crop_fit_sha256`, and `changed_module_values` (list of names or `{"name": ...}`) inside the allowed set N_WIN, N_TOK, token store, sample builder (compared lower-case, non-alphanumerics to `_`, a leading `the_` removed). Another spelling of a declared item (for example `n_windows`) FAILS. The record's `arm` is printed in the file but not tested.
9. **t5 constant.** Order: `--const`; else the dump key `const_gap` of FP_0; else the median label gap of the t3 rows (the evaluation frames), NOT the mean label gap of the fit clips. If the lead means the fit-clip mean, pass `--const`. FP_0 must be finite on all t3 rows. Pass bar: E(FP_0) >= 0.95 * E(const) on the t3 rows.
10. **t6** needs at least 2 `--test-out NAME=PATH` with `--test-before NAME=EPOCH` each; the last non-empty line must contain `<n> passed` with no `failed` or `error`, and the file's modification time must be less than the epoch (a file touched or copied after the start epoch fails). Token files: `--n-tok-files N` must be at least the number of clips with an ok stage row (taken from the stage records), so a driver that counts only `tok_T/*.npy` passes the count of those files. The test output of this reader kit itself is not required unless the lead passes it.
11. **Slope read** assumes the join rows of a clip are consecutive pilot frames (as the banked join gives), so a 25-row window is 25 pilot frames; windows are greedy, non-overlapping, and need every row in the common set. With very few such windows the correlation is `None` (fewer than 3) or noise; the number of windows is printed.
12. **Far-lead rule** with no label gap of 40 to 60 m in the common set gives no interval, hence `FAR-LEADS-STAY-SHORT` (the card's "else"), with n printed as 0.
13. **A one-sided bar.** `bar met: yes` needs the ratio of F at most 0.85 and the upper end below 1; a name `F-CUTS-GAP-ERROR` with a ratio of 0.9 is therefore printed with `bar met: no`.
14. **Label fields.** `n_label_fields_differ_on_common` (dumps whose label fields differ from D_0 by more than 1e-5) is printed as a fact; it is not a control (the card has none for it).
15. The common frame set removes a row if the gap of ANY of the nine dumps is non-finite or `tele_dropped` is set in ANY of the six new dumps; D2_0 and FP_0 do not enter it.

## Definitions

* Arms: D = `D_0 D_1 D_2` (banked fits of the fixed wide windows), M = `M_0 M_1 M_2`, F = `F_0 F_1 F_2`. Extra dumps: `D2_0` (t4), `FP_0` (t5).
* Rows: t3 rows (see clause 3); common frame set = lead within 60 m, a finite gap in all nine dumps, no `tele_dropped` in M_* and F_*. t3 is computed on D alone BEFORE this restriction; every other value after it.
* E(fit) = mean absolute difference of gap and `lab_gap` (D_0's label) on the common set. Ratio X / D = mean E(X_0..X_2) / mean E(D_0..D_2) (equals the ratio of summed per-clip errors). Interval: per-clip sums of absolute error, numerator = mean of the three X sums, denominator = mean of the three D sums; `idx = RandomState(122).randint(0, n_clips, size=(2000, n_clips))`; ratio of the sums per draw; two-sided 95 % percentile (2.5, 97.5); fewer than 1,000 finite draws gives no interval and no name. The same draws serve F, M and the far-lead difference (clips are the units; fits held fixed, so this is the "another draw of evaluation clips" question, not training variance).
* By gap bin (0-10, 10-20, 20-40, 40-60 m; the last bin includes 60): n, E, mean signed error (read minus label), share read more than 5 m short, each as the mean over the three fits of the arm. Printed for D, M, F.
* Far leads: label gap 40 to 60 m. Signed mean of an arm = mean over its three fits of the mean signed error on those rows. Paired difference = (sum over clips of the arm-F three-fit mean of the per-clip signed sums - the same for D) / (sum of per-clip counts), cluster bootstrap with the same draws; "separated" = the interval excludes 0.
* Printed with no name: share of lead frames (common set, with a label box) whose label box centre lies inside T8 (|az| <= 4, |el + 2| <= 4) and inside T4 (|az| <= 2, |el + 1| <= 2) in the WIDE camera frame (note N1: T8 and T4 are centred on (0, -2) and (0, -1) of that frame, so `--match-dir` direction files are not needed for it); error of arm F (mean of three fits per row) on frames inside and outside T8; correlation, over windows of 25 consecutive frames, of the label closing speed (minus the slope of the label gap) with the read closing speed (minus the slope of the read gap, the mean of the arm's three fits) for D, M and F. The box files are those that `read_lookgap.py` reads (`<clip>.npz` with `has_lead`, `ctr_az`, `ctr_el` per frame; `--box-dir`).

## Names (section 4 exactly) and the bar line

* `TELEGAP-T0-<X>-CUTS-GAP-ERROR` (upper end of the X / D ratio interval below 1), `-WORSE` (lower end above 1), `-NOT-SEPARATED` otherwise, X in F, M. No interval: no name for X, rc 5.
* `TELEGAP-T0-FAR-LEADS-READ`: |signed mean F| at most half of |signed mean D| and the paired difference separated; else `TELEGAP-T0-FAR-LEADS-STAY-SHORT`.
* `bar met: yes|no`: F ratio at most 0.85 and interval upper end below 1.
* Two independent routines: the first reads the payload's values (numpy), the second works from per-clip sums only with plain loops and hand-made percentiles. Names, bar and all three intervals must agree (1e-9) or the stage stops with rc 4.

## Controls (printed only as `t<i> PASS|FAIL`; every value goes to `TELEGAP_CONTROLS.json`)

t0 lists (sha256 of the lists the encode used against the banked list record); t1 staging (clause 6); t2 the tele kit's `ctl_t2.py` JSON (pass, share >= 0.9, n_frames, threshold); t3 D reproduces 3.960, 3.913, 4.112 within 1e-3 on 3,841 rows; t4 the variant (clause 8); t5 permuted labels (clause 9); t6 tests and counts (clause 10). With all seven passing, the values are under `payload`; otherwise under `withheld`, and the names stage refuses (rc 3) with no arm value printed. Each control has a planted fault in the tests that flips exactly that control.

## Files and commands

`read_telegap.py` (stages controls, names, `--selftest`), `podread_telegap.sh`, `pull_telegap.sh`, `test_telegap_read.py`, `helper_pytest.out`. The reader imports `read_lookgap.py` (`--lookgap-reader`, default `/workspace/v8proof/lookgap/kit_read/read_lookgap.py`, else the sibling folder on the dev box) and through it `read_lookat.py` (`--lookat-reader`) UNCHANGED; the sha256 of read_telegap.py, read_lookgap.py and read_lookat.py are recorded in the controls file under `inputs`.

The driver needs `$ROOT/kit/read/read_args.txt` (one argument token per line; lines are not shell-expanded), where ROOT = `/workspace/v8proof/telegap`. The lead writes it with exactly:

```
--dump
D_0=/workspace/v8proof/cropread/run1/dump/D_H5/reader_s0_score.npz     (default; may be omitted for D_0 D_1 D_2)
--dump
M_0=/workspace/v8proof/telegap/fit_out/<...>/reader_s0_score.npz       (the lead's exact paths: M_0 M_1 M_2 F_0 F_1 F_2 D2_0 FP_0)
--list-record
<banked list record JSON>
--list-file
train_list=<the train list the encode used>
--list-file
eval_list=<the eval list the encode used>
--all-clips
<the 641-clip list of the encode>
--stage-record
<stage_*.json of the tele stage, one --stage-record per file, oldest first>
--enc-manifest
/workspace/v8proof/telegap/enc_out/tok_T/MANIFEST.json
--match-dir
/workspace/v8proof/telegap/enc_out/match
--t2-json
/workspace/v8proof/telegap/enc_out/ctl/T2.json
--d2-fit-record
<FIT_RECORD.json of D2_0>
--test-out
enc=<lead_pytest.out of the encode kit>
--test-before
enc=<epoch of the encode start>
--test-out
fit=<lead_pytest.out of the fit kit>
--test-before
fit=<epoch of the first fit start>
--n-tok-files
<count of token files under tok_T>
```
Optional: `--const`, `--box-dir` (default `/workspace/v8proof/cropread/enc_out/box`), `--eval-join` (default `/workspace/v8proof/hierrep/pins/join_eval141.npz`).

Sequence: `bash podread_telegap.sh controls` (needs `TELEGAP_FIT_DONE.txt`, refuses with `TELEGAP_FIT_STOP.txt` or `TELEGAP_ENC_STOP.txt` present, runs the selftest first, writes `read/TELEGAP_CONTROLS.json`), then, only when all seven print PASS, `bash podread_telegap.sh names` (writes `read/TELEGAP_RESULT.json` and `read/read_names.log`; the sections are in that log, the driver prints only the reader's last line). Then `bash pull_telegap.sh`.

## Not done

No run on the pod and none on real dumps; the formats in clauses 1, 5 and 6 are taken from the tele kit's own writers (`tele_stage.py`, `tele_encode.py`, `ctl_t2.py`) and from `read_lookgap.py`, not from the fit variant's writer, which did not exist when this kit was written.
