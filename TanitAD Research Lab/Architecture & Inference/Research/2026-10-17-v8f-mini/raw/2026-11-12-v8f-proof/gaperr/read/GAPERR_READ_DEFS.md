# GAPERR_READ_DEFS: what the reader kit of card ADD111 (GAPERR10K) codes

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s). Arms NR, BR (attribution arms, label values in the input, never candidates), SM, SL (base gap file smoothed, no label), two model seeds each (eight rollouts, run folders `I10RQ50KF5P30Hb20b<ARM>_s<S>`); references GEc, RLc, RR, LA (banked rollouts; LA is run `RQ50KF5P30Hb20aLA` of card ADD110); control run V0 of card ADD102. T0 numbers of the gap-file tool are printed as recorded (recorded frames, not driving performance).

## What could fail by construction (read first)

Clauses or driver steps that can fail on real inputs without a fault of the subject:

1. b0, record keys: the record must have EXACTLY the top keys `pass_b0, clauses, files, base, la_file, eval_join_sha256, t0, written_utc`. A gap-file tool that writes one more or one fewer key fails b0 although the files may be right. Same for `files` (exactly NR, BR, SM, SL, each with `n_rows, n_diff_rows, n_label_value_rows, path, sha256`, none null) and `t0` (exactly near, middle, far, all, each with `n_error_rows, rms_e, rms_s, rms_f, rms_sm, rms_sl`, none null).
2. b0, base file path: the base file is opened at `base.path` of the record (a pod path of an earlier card), unless `--base-file` is given; the driver does not pass it (as in the r1 driver of card ADD110). A record path that is not readable on the pod when the read runs fails b0.
3. b0, file LA: opened at `--la-file`; the driver passes `$V/fallattr/gapfiles/VGAPA_LA.npz` (the gapfiles of card ADD110 on the pod). Its sha256 must start with the 16 or more hex digits of `la_file.sha256` of the record. A record that names another LA file (or a file that was rewritten) fails b0.
4. b0, row counts: `files.<ARM>.n_rows` is compared with the number of rows of the file (all rows, not only rows with an output).
5. b0, gap file names: the file of an arm is taken by the NAME PATTERN `VGAPA_<ARM>.npz` in `--gap-dir`, as the previous reader did; `files.<ARM>.path` of the record (a pod path) is not opened. The sha256 of that file is compared with `files.<ARM>.sha256` (16 or more hex digits).
6. b0, not tested: `n_diff_rows`, `n_label_value_rows` and every number of `t0` are checked for presence only and printed as recorded; they are not recomputed here.
7. b0, raw field: for SM and SL a row with no base output must keep the raw gap field of the base file (compared where both files name the row, nan equal to nan). A tool that writes another filler value (for example 0) there fails b0 although the card says "keeps its base gap field" and no output flag is set.
8. b0, the run rule: the step test `abs(difference) <= 3.0` is done in float64 on the base gap as stored in the base file; a tool that computes the difference in float32 can cut a run on a borderline step that this reader keeps (a difference within 1e-6 of 3.0 m). The median of an even run is the mean of its two middle values (as `numpy.median`). Rows of one clip are taken as consecutive frames (row = offset + frame), as the join defines them.
9. b0, error rows: a row of the base file counts as having an output when the flag is set AND the gap is finite (as in the earlier readers); a row the base file does not name has none.
10. b2: the clause "closing input non-zero on a positive share of the steps with a lead" fails for any rollout whose rule never closes; the card states it as intended. A run text without a `wrapper sha256` line fails b1.
11. b3: needs the banked rollouts of GEc, RLc, RR and LA in the pull folder; the driver links GEc, LA and V0 from `$W/t1` and requires RR and RLc to be there already (`MISSING-BANKED-RUN` stop otherwise).
12. b4: the newer of the two `lead_pytest.out` files must be older than the first token of the first line of `GAPERR10K_START.txt` (`<epoch seconds> <ISO time> pid <pid>`). The push script copies `lead_pytest.out`, so its pod modification time is the push time: a queue started before the push fails b4 by order of events. The driver line that parses the start marker stops (`START-MARKER-UNREADABLE`) on a first token that is not an integer; the tests run that very line on the line `1795335883 2026-11-22T08:24:43Z pid 774424`.
13. Driver: the free-memory floor (20 GB of the cgroup limit) and the kit files (`lead_pytest.out` of both kits) stop the stage before the reader starts.

## Stages

Controls stage: runs b0 to b5, the grid read of the arms against GEc, RLc, RR and LA, and the strategic grid (the engine's own controls inside it; strategic pairs for the four arms against GEc and RLc, the strategic grid runs the four arms and GEc, RLc, RR). Output: `GAPERR10K_CONTROLS.json` with `controls` (b0 to b5, each with `pass` and `failures`), `all_pass`, `inputs` (sha256 of the model reader, this reader, the join, the gap-file record, the gap files, the wrapper file prefix) and the values under `payload` when every control passes, under `withheld` otherwise. The payload has `gapfile_counts_from_record` (per arm `n_rows`, `n_diff_rows`, `n_label_value_rows`) and `t0_from_record` (the `t0` block as it is). Printed: `b<i> PASS|FAIL` and `ZZ READ-GAPERR10K controls all_pass <bool>`; the engine's output goes to `<out>.engine.log`.

Names stage: refuses unless `all_pass` is true and b0 to b5 are all present with `pass` true. Names are made by two separate routines (`names_of` and `derive_all`) that must agree, else the stage stops. Writes `GAPERR10K_RESULT.json` (bound to the controls file by sha256) and a names log, printing the counts and the t0 block as well.

## What each control reads

- b0: second routines of this reader (`compose_attr`, `slow_fast`, `smooth_arms`, written apart from the gap-file tool) rebuild every file from the base file and the join: NR and BR (label gap plus s, label gap plus f on error rows; label rows with no base output take the label gap and an output; every other row the base gap and flag), SM and SL (trailing run of the base gap: the row and the frames before it in the clip while each has a base output and differs from the next by at most 3.0 m, at most 5 frames; SM median, SL end value of the least-squares line from 3 frames on, else the base gap). Each file equals its definition on every join row within 1e-9 m with equal output flags; NR plus BR minus the label gap equals the base gap on every error row; NR and BR equal the file LA on every row that is not an error row; SM and SL have the base output flags, keep the base raw field where there is no output, and come out byte for byte the same when the label fields of the join are replaced (the routine reads only clip lengths and offsets); a constant course comes back unchanged from both, a straight course from SL within 1e-9 and from SM two frames late once five frames are in the run (checked on synthetic courses with the routine). Besides: record keys exact, `pass_b0` true, clauses pass, files and t0 entries filled, file hashes and row counts as the record names, base file and file LA are the files whose hashes the record names (`base.named_in_record_add106` true), join named `join_eval141.npz` with the hash the record names.
- b1: wrapper file 28bc31f21ecb4ccb, the check record of card ADD102 ends with `ZZ CHECK-WRAPPER-V PASS` and names the hash, every one of the eight run texts names it.
- b2: as g5 of the model reader (`w2_one`) on the eight rollouts with their files `VGAPA_<ARM>.npz`: received gap equals the file's gap within 1e-3 m on steps with a lead and an output; steps with a lead and no output equal V0 within 1e-3 m; meta names the gap file hash and the closing switch 1; the closing input is non-zero on a positive share of steps with a lead. Per rollout the share of rule steps with a lead and the share of them with a non-zero closing input are kept for the print (the share with a label gap is dropped).
- b3: pooled levels of GEc (0.0062, 0.5154, 0.0839), RLc (0.0000, 0.4908, 0.0714), RR (0.0954, 0.4954, 0.1339), LA (0.0000, 0.4877, 0.0839) as contact, falling behind, unsafe within 1e-4.
- b4: the last non-empty line of each `lead_pytest.out` holds a count of passed above 0 and neither `failed` nor `error`; the newer modification time is below the first start time (seconds since the epoch, or an ISO time).
- b5: as g8 of the model reader (engine controls of the grid read and of the strategic grid).

## Names (card section 3)

Per arm X in NR, BR, SM, SL and measure M in FALLING-BEHIND, CONTACT, UNSAFE against GEc: `GAPERR10K-<X>-LOWERS-<M>` (upper end of the interval below 0), `-RAISES-<M>` (lower end above 0), else `GAPERR10K-<X>-<M>-NOT-SEPARATED`. The parts from falling behind of NR and BR against GEc: `-FAST-PART-CARRIES` (NR lowers, BR does not), `-SLOW-PART-CARRIES` (BR lowers, NR does not), `-BOTH-PARTS-CARRY`, else `-PARTS-NOT-SEPARATED`. Per smoothing arm X of SM, SL: `GAPERR10K-<X>-PAYS-SAFETY` when X raises contact or raises unsafe against GEc, else `-RECOVERS` when X lowers falling behind, else `-NO-GAIN`.

Printed with no name (in this order): the tier line; the names; the levels of every arm and reference per model seed and pooled; every arm against GEc, RLc, RR and LA on the three measures; the four families per arm against GEc and RLc (LONGITUDINAL, LATERAL, TACTICAL from the engine's pair read, STRATEGIC from the strategic read at T1-kin); per gap file the counts from the record; the t0 block per range from the record; per rollout the share of rule steps with a lead and of them with a non-zero closing input.

## Arguments of the reader

`--stage controls|names`, `--selftest`, `--out` (required with a stage), `--log` (names log, default `<out>_names.log`), `--n-boot` (default 2000; the controls read uses 300 for its own engine controls), `--pull` (folder of the run folders), `--code`, `--grid-dir` (folder with `grid_p30.json`), `--w`, `--tree`, `--wrapper-file`, `--wrapper-check`, `--wrapper-hashes`, `--controls` (names stage), `--eval-join` (default `$WS/hierrep/pins/join_eval141.npz`), `--gap-record` (default `<gap-dir>/GAPERR_GAPFILES_RECORD.json`), `--gap-dir` (default `<fa>/gapfiles`), `--tests-gap`, `--tests-read` (defaults `<fa>/kit_gap/lead_pytest.out`, `<fa>/kit_read/lead_pytest.out`), `--tests-mtime-gap`, `--tests-mtime-read`, `--queue-start` (seconds since the epoch or an ISO time), `--model-reader` (default the pod path `/workspace/v8proof/cropscale/kit_read/read_cropscale10k.py`, else the dev-box copy, else env `GAPERR_MODEL_READER`), `--base-file` (default: `base.path` of the record), `--la-file` (default: `la_file.path` of the record), `--fa` (default `/workspace/v8proof/gaperr`). The reader refuses to write into an existing `--out`.

## The exact reader calls of the two driver stages (`podread_gaperr10k.sh`)

Selftest first (controls stage): `"$PY" "$K/read_gaperr10k.py" --selftest --model-reader "$MODEL"`.

Controls stage:

```
"$PY" "$K/read_gaperr10k.py" --stage controls --model-reader "$MODEL" --pull "$PL" --code "$CODE" --grid-dir "$W/longroll" --w "$W" --tree "$W/tree" --wrapper-file "$WRAP" \
  --fa "$FA" --gap-dir "$GF" --gap-record "$GF/GAPERR_GAPFILES_RECORD.json" --la-file "$LAF" --eval-join "$JOIN" --wrapper-check "$CHK/CHECK_WRAPPER_V_DONE.txt" --wrapper-hashes "$CHK/CHECK_WRAPPER_V.hashes" \
  --tests-gap "$KG/lead_pytest.out" --tests-read "$K/lead_pytest.out" --tests-mtime-gap "$MG" --tests-mtime-read "$MR" --queue-start "$QS" --out "$OUT/GAPERR10K_CONTROLS.json"
```

Names stage:

```
"$PY" "$K/read_gaperr10k.py" --stage names --model-reader "$MODEL" --controls "$OUT/GAPERR10K_CONTROLS.json" --out "$OUT/GAPERR10K_RESULT.json" --log "$OUT/read_names.log"
```

Pod root `/workspace/v8proof/gaperr`: kits `kit_gap`, `kit_read`, gap files `gapfiles`, outputs `read/`; markers `GAPERR10K_START.txt`, `GAPERR10K_DONE.txt`, `GAPERR10K_STOP.txt`. Last line of the driver: `ZZ PODREAD-GAPERR10K <stage> DONE rc <n>` (or `ZZ PODREAD-GAPERR10K STOP <why>`).

## Diff of the driver against the r1 driver of card ADD110 (all made by `make_read_gaperr10k.py`, asserted)

1. Header comment rewritten (card ADD111, b4 start parse, file LA).
2. `FA=$V/fallattr` becomes `FA=$V/gaperr`; new line `FA0=$V/fallattr; LAF=$FA0/gapfiles/VGAPA_LA.npz`.
3. Required gap files `VGAPA_LA/LN/LM/LF.npz` become `VGAPA_NR/BR/SM/SL.npz` plus `$LAF`.
4. Run folders linked and checked: `aLA aLN aLM aLF gGEc vV0` becomes `bNR bBR bSM bSL aLA gGEc vV0`.
5. `READER_HASHES.txt` also hashes `$LAF`; the controls call gets `--la-file "$LAF"`.
6. Control names: grep pattern `^a[0-5]` becomes `^b[0-5]`.
7. Names: every `FALLATTR10K` becomes `GAPERR10K`, `read_fallattr10k.py` becomes `read_gaperr10k.py`, `FALLATTR_GAPFILES_RECORD.json` becomes `GAPERR_GAPFILES_RECORD.json`, `PODREAD-FALLATTR10K` becomes `PODREAD-GAPERR10K`.
8. Unchanged on purpose: the start-marker line (`QS=$(head -n 1 ... | awk '{print $1}')` with the integer check), the memory floor, the stat lines, the export lines, the model reader path.

## Substitutions of the reader (`make_read_gaperr10k.py`, each asserted by count or by marker)

Module docstring; card and prefix constants; arms, run names, references (adds LA) and expected levels (adds LA); constants (run rule, record keys, t0 keys; controls b0 to b5); tier text (label enters NR and BR only); removal of `in_range`; the block from `carried_rows` to the section header replaced by the b0 routines; `ctl_a1` to `ctl_b1`, `a2_from_files` and `ctl_a2` to b2 (label-gap share dropped), `ctl_a3` to `ctl_b3`, `ctl_a4`, `ctl_a5`; `compute_controls`; the record counts and `t0_from_record` in the payload; the names block (`names_of`, `derive_all`); pairs against LA in the grid read and in the print; the strategic grid runs without LA; the print of the t0 block; the world, payload builder and selftest; the new argument `--la-file`; renames of file and variable names. Everything not listed (engine calls, `measure` apart from the two lines above, `assemble_controls`, `flat_names`, `file_on_join`, `label_rows`, `sha_file`, `to_epoch`, `last_line`, `run_shas_of`, the stages in `main`) is the text of `read_fallattr10k.py`. The tests check that the reader and the three scripts are exactly what the generator writes from the unchanged files of card ADD110.

## Files

`read_gaperr10k.py` (imports the model reader `read_cropscale10k.py` of card ADD106 from `--model-reader`; calls `rows_of`, `load_logs`, `lead_multi`, `w2_one`, `gapfile_ref`, `meta_of`, `run_file_sha`, `ctl_g8`), `make_read_gaperr10k.py`, `podread_gaperr10k.sh`, `pull_gaperr.sh` (pulls into `gaperr/pod_add111`, refuses if it exists), `push_gaperr_read.sh`, `test_gaperr_read.py`, this file. The lead writes `lead_pytest.out`.
