# GAPSPREAD10K read definitions (card PREREG_V8F_PROOF_ADD134, kit `gapspread10k/read`)

`make_gapspread_reader.py` writes `read_gapspread10k.py` and `podread_gapspread10k.sh` from `repl/read/read_repl10k.py` (LF text, sha256 prefix d5cdf986de3bcd5a) and
`repl/read/podread_repl10k.sh` (62228aa60cea7c89) by asserted substitutions (an anchor not found exactly once, a changed source and an existing output are refused; `--print reader|driver` writes to
stdout, `--out-dir` names the folder). Tests: `test_gapspread_read.py` (child processes run in this folder, or in the folder named by `GAPSPREAD_TEST_CWD`). Pull: `pull_gapspread10k.sh`
(pod root `/workspace/v8proof/gapspread10k`, local folder `gapspread10k/pod_add134`, transfer files `gapspread10k/tmp_pull_add134.txt|.tar`; text only, never `.npz`).
Kept from the source: the engine calls, the two-level bootstrap (2,000 draws, `--n-boot`), the wrapper control, the parser of a run line, the loader of rule logs, the seed context of the engine.
The generated reader refuses `--selftest`. Tier on every number: T1-kin/hb0.2 (kinematic rollout on the model's own actions); every result is stated as "not driving performance".

## Read
One setting K0 (readout K, no pre-roll, rule every 5 steps, pinned grid). Arms SP90 (new), A90, GEc, RLc, RR, all at model seeds 0, 1, 2, 3 (SP90 runs `I10RQ50KF5Hb20gSP90_s<m>`; the banked runs are
those of cards ADD117, ADD121 and earlier; the spread run `I10R0Hb20` of the four seeds is the engine's reference of the windows, and for model seeds 2 and 3 the strategic grid takes it as R0 (`R0Alias`)).
ONE grid read over the four seeds (the engine's module global `SEEDS` set for the call): levels per model seed and pooled (C, U, F), pairs first minus second, paired, two-level bootstrap over model
seeds and episodes: SP90-VS-A90 (`VS-FLAT`), SP90-VS-GEc (`VS-CANDIDATE`), SP90-VS-RLc and A90-VS-GEc (printed, no name). Families (longitudinal, lateral, tactical from the engine, strategic
from the strategic grid at T1-kin, position error at 30, 50, 80 m) for SP90-VS-A90 and SP90-VS-GEc.

## Controls (card section 5; printed `ctl p0` to `ctl p7`; no level, no difference, no name in the output of the stage)
- **p0 gap file.** `--gap-record` (pod `$GE/gapfiles/GAPSPREAD10K_GAPFILE_RECORD.json`): `card` == PREREG_V8F_PROOF_ADD134, `pass_p0` the JSON value true, `files.VGAPSP90.sha256` 64 hex and equal to the
  sha256 of the file given by `--gap-file VGAPSP90=<path>` (pod `$GE/gapfiles/VGAPSP90.npz`). The clauses of the card text (arrays, factor, k, label rows, tools, hashes, counts) are read from `pass_p0`.
- **p1 wrapper.** The source's wrapper control: file hash 28bc31f21ecb4ccb, banked check files in place, pod check ends `ZZ CHECK-WRAPPER-V PASS`, hashes name the wrapper, each of the four new run texts
  names a wrapper hash that begins with it.
- **p2 arguments.** First line of each new run against that of the banked A90 run of the same model seed (`d2_same_failures`): form, replan, every, readout, kappa (the heading bound 0.2 1/m), model seed and
  reader seed present in both; every field both carry equal (also pre-roll, exec, nav, grid, checkpoint hash by prefix of 16); a field the banked line carries and the new line lacks is a failure; the
  new line's model seed is m; where the banked line has no pre-roll, grid or nav the new line reads 0, `eval_grid_windows.json`, deadreckon.
- **p3 identity.** Pooled levels (C, U, F) over four seeds of GEc, A90, RLc, RR within 1e-4 of the card's table, and the pair A90-VS-GEc: pooled difference and both interval ends of U and F within 1e-4.
  A failure text names the cell and the measure and holds no value.
- **p4 tests.** The two lead test outputs (`$GE/lead_tests/lead_pytest_queue.out`, `lead_pytest_read.out`) read `N passed` (N > 0) and no failed or error count; the first not later than the first line of
  `GAPSPREAD10K_START.txt`, the second not later than the start of the stage (times from `stat -c %Y` and `date +%s` in the driver); neither path has a component `kit_read` or `kit_queue`.
- **p5 engine.** The engine's grid controls and both stages of its strategic grid pass (`ctl_g8` of the model reader).
- **p6 counts.** Each of the four runs has `T1_DONE` and 141 `ep*.npz` dump files; the queue done marker's last line reads `ZZ GAPSPREAD10K DONE <time> all steps done`.
- **p7 rule logs.** For each new run the model reader's `w2_one` against `VGAPSP90.npz` (gap at steps with an output within 1e-3 m; at steps without one the gap of the RR run of the same model seed;
  closing input non-zero on a positive share of lead rows; meta `vis_closing` 1) and the rule log's `vis_gap_file_sha256` names the file's hash (prefix of at least 16).

## Names (card section 4; interval below 0 = upper end < 0, above 0 = lower end > 0)
Per pair P in VS-FLAT, VS-CANDIDATE and measure M in UNSAFE, CONTACT, FALLING-BEHIND: `GAPSPREAD10K-K0-<P>-LOWERS-<M>`, `-RAISES-<M>`, `-<M>-NOT-SEPARATED`. One overall name from SP90-VS-A90,
the first clause that holds: 1 `-PLACED-GIVES-UP-SAFETY` (unsafe or contact interval above 0); 2 `-PLACED-SAME-SAFETY-AT-A-LOWER-PRICE` (upper end of the unsafe interval <= +0.0058, contact interval not above
0, falling-behind interval below 0); 3 `-PLACED-SAFER-AT-NO-HIGHER-PRICE` (unsafe below 0, falling behind not above 0); 4 `-PLACED-NOT-SEPARATED-FROM-FLAT`. Two routines in the reader derive the names and
must agree (else refused); the test adds a third. SP90-VS-RLc and A90-VS-GEc carry no name.

## Stages and files
`bash podread_gapspread10k.sh controls|names`; outputs in `$GE/read_out`, never overwritten: `GAPSPREAD10K_CONTROLS.json` (values under `payload` when all pass, under `withheld` otherwise; hashes of every
input under `inputs.files`), `GAPSPREAD10K_RESULT.json`, `GAPSPREAD10K_VERDICT.md`, `read_names.log`, `read_names.out`, `read_controls.log`, `READER_HASHES.txt`, `selftest_pod.log`; the engine's own output
`GAPSPREAD10K_CONTROLS.json.engine.log`. The controls stage ends `ZZ READ-GAPSPREAD10K controls all_pass True|False`; the names stage refuses (rc 3) unless p0 to p7 are present and pass and every input
file (readers, wrapper and its check files, gap record and gap file, grid pin, the two test outputs, the done marker, the four run texts) hashes as the controls stage hashed it; it ends
`ZZ READ-GAPSPREAD10K names done`.

## Open definitions
`kappa` in the run line is read as the heading bound. The strategic read of the card uses R0 = the spread run for model seeds 2 and 3 (as card ADD121). The pair estimates of p3 come from the same read
as the new pairs (the bootstrap is the engine's; whether it reproduces to 1e-4 with more pairs in the call is a property of the engine, shown by p3 itself).
