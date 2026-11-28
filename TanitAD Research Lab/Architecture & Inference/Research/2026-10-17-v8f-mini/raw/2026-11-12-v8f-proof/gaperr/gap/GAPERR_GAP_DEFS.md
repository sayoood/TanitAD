# GAPERR_GAP_DEFS -- gap-file and rollout-queue kit of card ADD111 (GAPERR10K)

## What could fail by construction (read first)

None of these makes a result wrong; each stops the chain, and each is a fact about an input this kit could not open (no real file was read when the kit was written).

1. **The key names of the real evaluation join.** `make_gaperr_gapfiles.py` reads `clip`, `off`, `n`, `lead_gap`, `lead_valid` from `join_eval141.npz`. These names come from the kit of card ADD110, not from the file. If one is missing the tool stops with a key error before it writes a gap file.
2. **The base file must have the rows of the join, in join order.** Clause `layout` is red when the base file has not exactly one row per join frame with the join's clips (the tool also walks rows by position when it forms runs, so a base file in another order cannot pass).
3. **Clause `base_is_recorded` needs the layout of the ADD106 record.** It looks for the sha256 of `--base` under `files.<name>.file_sha256` in `CROPSCALE_GAPFILES_RECORD.json`. Another layout, or a base file that is not byte for byte the recorded one, turns it red.
4. **Clause `la_equivalence` needs the LA file of card ADD110 on the pod** at `/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz`, in the layout of the base file, built from the same base file. A missing file stops the queue (`NO-BASE-FILE-RECORD-OR-LA-FILE`); an LA file built from another base, or by a rule that differs from the definition of card ADD110 on any row that is not an error row, turns the clause red. The kit cannot know the LA file's origin; it compares rows.
5. **Error rows are taken with a finite base gap.** An error row is a label row where the base file has an output and a finite gap. If the real base file had an output flag with a NaN gap on a label row, that row is not an error row here (it carries the label gap and an output, as in LA). The card text does not mention such a row; a base file of card ADD106 is expected to have none.
6. **The gate (step 0) needs both test outputs.** `$GE/kit_gap/lead_pytest.out` is pushed by `push_gaperr_gap.sh`; `$GE/kit_read/lead_pytest.out` comes from the push of the reading kit, which this kit does not write. Without it the queue exits 4.
7. **The banked runs.** Step 2 stops unless the pod has `I10R0Hb20`, `RR`, `RLc` (dump folders), `vV0`, `gGEc` and `aLA` (the last three with `T1_DONE`), both model seeds, the wrapper check record of card ADD102, the overlay tree, the grid, the qmaps and the checkpoints `UE_s0`, `UE_s1`. They are checked, not assumed. The `aLA` check is the one precondition added to the queue of card ADD110 (the read compares against LA).
8. **A partly written run folder blocks `RESUME=1`.** The queue deletes and moves nothing; a rollout that failed leaves its folder, and the next start stops with `OUTPUT-EXISTS` until the folder is dealt with by hand. A time stop leaves no folder and resumes cleanly (with a larger `GE_STOP_S`).
9. **The time rule counts the gap-file step too.** The step is CPU work (about 14,000 rows, a few loops per row; expected well under the stop rule), and it is the first job start.
10. **Float limits.** Clause tolerances are 1e-9 m. The least-squares end value is computed by `numpy.polyfit` in the tool and by explicit sums in the second routine; on gaps of 0 to 60 m they agree to about 1e-13. `SM` of a constant course is exact; `SL` of a constant course is within 1e-9, not byte for byte.

## Gap-file name pattern and record keys

* Files: `VGAPA_<ARM>.npz` with `<ARM>` in `NR`, `BR`, `SM`, `SL` (same pattern as card ADD110), in `/workspace/v8proof/gaperr/gapfiles/`.
* Record `GAPERR_GAPFILES_RECORD.json`, top keys exactly: `pass_b0`, `clauses`, `files`, `base`, `la_file`, `eval_join_sha256`, `t0`, `written_utc`.
  * `clauses`: clause name to `{"pass": bool, ...facts}`; names `second_routine`, `error_identity`, `la_equivalence`, `smoothing_properties`, `no_label_in_smoothing`, `base_is_recorded`, `layout`.
  * `files`: keys `NR`, `BR`, `SM`, `SL`, each `n_rows`, `n_diff_rows` (rows that differ from the base file in gap or output flag), `n_label_value_rows` (rows that carry a label value; 0 for SM and SL), `path`, `sha256`.
  * `base`: `path`, `sha256`, `named_in_record_add106`. `la_file`: `path`, `sha256`.
  * `t0`: keys `near`, `middle`, `far`, `all`, each `n_error_rows`, `rms_e`, `rms_s`, `rms_f`, `rms_sm`, `rms_sl` (null when a range has no error row). Ranges are by the label gap: below 20 m, 20 m to below 40 m, 40 m and above. `rms_sm` and `rms_sl` are the root mean square of SM and SL minus the label on the error rows. Tier T0 on the evaluation frames: recorded frames, not driving performance.

## The gap files (`make_gaperr_gapfiles.py`)

* Label rows: `lead_valid` is 1 and `lead_gap` is finite from 0 to 60 m, both ends included. Error rows: label rows where the base has an output. `e` = base gap minus label gap. An error run is a largest set of error rows of one clip on consecutive frames. `s` = mean of `e` over the rows of the run within 5 frames before and 5 frames after the row, the row included (fewer near the ends of the run); `f` = `e` minus `s`.
* NR: label gap plus `s` on every error row. BR: label gap plus `f` on every error row. In both, a label row with no base output has the label gap and an output (as in LA); every other row has gap and output flag of the base file.
* Trailing run of a row with a base output: the row and the frames before it in the same clip, going back while each frame has a base output, is the frame before the next, and differs from the next by at most 3.0 m in the base gap; at most 5 frames.
* SM: median of the base gap over the trailing run. SL: end value of the least-squares line over the trailing run when it has 3 or more frames, the base gap itself when it has fewer. Output flags are those of the base file; a row with no base output keeps its base gap field. `smooth_gap` takes `clip_index`, `frame`, `gap`, `has_out`, `kind` (and a test-only `fault`): no label array.
* Array fields are those of the base file (`clip_index`, `frame`, `gap`, `has_out`, `clips`, `seed`, with `seed` -1) and no other.

## Control b0 (clauses)

* `second_routine`: each saved file reloaded from disk equals a separately written routine (plain loops over the raw join and the raw base, with its own median and its own least-squares sums) on every row: largest absolute gap difference at most 1e-9 m, equal `has_out`, equal NaN pattern, same rows.
* `error_identity`: on every error row, label plus `s` plus `f` equals the base gap within 1e-9 m (read from the saved NR and BR files).
* `la_equivalence`: NR and BR equal the LA file on every row that is not an error row, in gap and output flag.
* `smoothing_properties`: SM and SL have the output flags of the base file (and keep the base gap where there is no output); a constant course comes back unchanged; SL gives a straight course back within 1e-9 m; SM gives it back two frames late once five frames are in the run; a jump above 3.0 m and a frame without output end the run.
* `no_label_in_smoothing`: SM and SL are rebuilt with the label fields of the join replaced by other values (three replacements); the saved arrays are byte for byte equal.
* `base_is_recorded`, `layout`: as in card ADD110 (the base file is the one the record of card ADD106 names; fields, dtypes, rows and clips are those of the base file, `seed` -1). The record also holds the sha256 of the evaluation join and of the LA file.
* The tool exits 0 only when `pass_b0` is true and prints one `clause <name> <True|False>` line per clause and `ZZ GAPERR-GAPFILES pass_b0 <True|False>`; no gap value and no rms value. `--selftest` builds a synthetic world: the clean build passes every clause and one planted fault per clause turns that clause red (last line `ZZ GAPERR-GAPFILES SELFTEST OK`).

## The queue (`pod_gaperr10k_queue.sh`, written by `make_gaperr_queue.py` from the queue of card ADD110)

* Step 0: exit 4 unless both `lead_pytest.out` files exist. Step 1: when `$GE/gapfiles` is absent, the tool runs on the CPU (`CUDA_VISIBLE_DEVICES=-1`); the queue stops unless the record has `pass_b0` true, the hash of the evaluation join, of the base file, of the LA file and of the four files. Step 2: eight rollouts, arms NR, BR, SM, SL, model seeds 0 and 1, run folder `/workspace/v8proof/integ10k/t1/I10RQ50KF5P30Hb20b<ARM>_s<S>`.
* The tool call inside the queue: `make_gaperr_gapfiles.py --join $JOINE --base $BASEF --base-record $BASER --la-file $LAF --out-dir $GF`.
* The rollout function is the one of card ADD110, unchanged. One rollout at a time, each started only with fewer than 3 GPU compute processes and more than 28 GB free.
* Time rule: seconds since the first token of the first line of `$GE/GAPERR10K_START.txt` (format `<epoch seconds> <ISO time> pid <pid>`) at most `GE_STOP_S` (default 10800), else a line in `$GE/GAPERR10K_STOP.txt`. A failed rollout writes the stop marker. `RESUME=1` continues after a stop marker and appends a line to the start file; the first line keeps the clock. Done marker `$GE/GAPERR10K_DONE.txt` (one line starting `ZZ GAPERR10K DONE`), log `$GE/gaperr_queue.log`.
* `DRY=1` prints every command and creates nothing; it needs no pod path.
