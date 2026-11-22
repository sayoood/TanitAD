# FALLATTR_GAP_DEFS -- gap-file and rollout-queue kit of card ADD110 (FALLATTR10K)

## What could fail by construction (read first)

None of these makes a result wrong; each stops the chain, and each is a fact about an input this kit could not open.

1. **The key names of the real evaluation join.** `make_attr_gapfiles.py` reads `clip`, `off`, `n`, `lead_gap`, `lead_valid` from `join_eval141.npz`. These names were read from `make_scale_gapfiles.py` (card ADD106), not from the file. If one is missing the tool stops with a key error before it writes a gap file.
2. **The base file must have the rows of the join, in join order.** Clause `layout` is red when the base file has not exactly one row per join frame with the join's clips. The base file of card ADD106 is built that way (one row per join frame), but this kit has not opened it.
3. **Clause `base_is_recorded` needs the layout of the ADD106 record.** It looks for the sha256 of `--base` under `files.<name>.file_sha256` in `CROPSCALE_GAPFILES_RECORD.json` (the layout written by `make_scale_gapfiles.py`, line 664). A record with another layout, or a base file that is not byte for byte the recorded one, turns it red.
4. **The gate (step 0) needs both test outputs.** `$FA/kit_gap/lead_pytest.out` is pushed by `push_fallattr_gap.sh`; `$FA/kit_read/lead_pytest.out` comes from the push of the reading kit, which this kit does not write. Without it the queue exits 4.
5. **The banked runs.** Step 2 stops unless the pod has `I10R0Hb20`, `RR`, `RLc` (dump folders), `vV0` and `gGEc` (both with `T1_DONE`), both model seeds, the wrapper check record of card ADD102, the overlay tree, the grid, the qmaps and the checkpoints `UE_s0`, `UE_s1`. They are checked, not assumed.
6. **A partly written run folder blocks `RESUME=1`.** The queue deletes and moves nothing; a rollout that failed leaves its folder, and the next start stops with `OUTPUT-EXISTS` until the folder is dealt with by hand. A time stop leaves no folder and resumes cleanly (with a larger `FA_STOP_S`).
7. **The time rule counts the gap-file step too.** The step is CPU work of seconds; it is the first job start.

## The gap files (`make_attr_gapfiles.py`)

* Label rows of the join: `lead_valid` is 1, `lead_gap` is finite and lies from 0 to 60 m, both ends included (60.0 is a label row, 60.01 is not).
* LA has the label gap on every label row. LN has it where the label gap is below 20 m, LM from 20 m to below 40 m (20.0 is in LM), LF from 40 m up (40.0 is in LF). On every other row a file has gap and output flag of the base file laid on the join rows by clip name and frame.
* On a row of its range a file has an output whether or not the base has one. Nothing is smoothed, clipped or blended. The array fields are those of the base file (`clip_index`, `frame`, `gap`, `has_out`, `clips`, `seed`, with `seed` -1) and no other.
* `n_source_changes` is the number of consecutive frame pairs inside one clip where exactly one of the two frames carries the label gap.

## Control a0 (record `FALLATTR_GAPFILES_RECORD.json`; keys `pass_a0`, `clauses`, `files`, `base`, `eval_join_sha256`, `written_utc`)

* `second_routine`: each saved file, reloaded from disk, equals a separately written routine (plain loops over the raw join and the raw base) on every row: largest absolute gap difference at most 1e-9 m, equal `has_out`, equal NaN pattern, same rows.
* `ranges_partition`: the label rows of LN, LM and LF are pairwise disjoint, their union is exactly the label rows of LA, LA is the label rows of the join, and each range holds only gaps inside it.
* `outside_is_base`: off its range every file equals the base laid on the join rows, in gap and `has_out`.
* `base_is_recorded`: the sha256 of `--base` is listed in the gap-file record of card ADD106 given by `--base-record`.
* `layout`: fields and dtypes are those of the base file and of the card (`seed` -1), the file has the rows and clips of the join, and the base file has them too.
* The tool exits 0 only when `pass_a0` is true and prints `ZZ ATTR-GAPFILES pass_a0 <True|False>` and the clause lines, no gap value. `--selftest` builds a synthetic world; the clean build passes every clause and one planted fault per clause turns it red (last line `ZZ ATTR-GAPFILES SELFTEST OK`).

## The queue (`pod_fallattr10k_queue.sh`, written by `make_fallattr_queue.py` from the queue of card ADD106)

* Step 0: exit 4 unless both `lead_pytest.out` files exist. Step 1: when `$FA/gapfiles` is absent, the tool runs on the CPU (`CUDA_VISIBLE_DEVICES=-1`); the queue stops unless the record has `pass_a0` true, the hash of the evaluation join, the hash of the base file and the hashes of the four files. Step 2: eight rollouts, arms LA, LN, LM, LF, model seeds 0 and 1, run folder `/workspace/v8proof/integ10k/t1/I10RQ50KF5P30Hb20a<ARM>_s<S>`.
* The rollout is the one of card ADD106 (same wrapper file 28bc31f21ecb4ccb, same arguments, `--vis-closing`), with two added lines: a run with its `T1_DONE` is not run again, and the time rule is checked before each start.
* One rollout at a time, each started only with fewer than 3 GPU compute processes and more than 28 GB free; so at most 3 run at once on the pod with other jobs.
* Time rule: seconds since the first line of `$FA/FALLATTR10K_START.txt` at most `FA_STOP_S` (default 10800), else a line in `$FA/FALLATTR10K_STOP.txt`. A failed rollout writes the stop marker. `RESUME=1` continues after a stop marker and appends a line to the start file; the first line keeps the clock. Done marker `$FA/FALLATTR10K_DONE.txt` (one line starting `ZZ FALLATTR10K DONE`), log `$FA/fallattr_queue.log`.
* `DRY=1` prints every command and creates nothing; it needs no pod path.
