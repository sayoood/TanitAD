# FALLATTR_READ_DEFS: what the reader kit of card ADD110 (FALLATTR10K) codes

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s). Arms LA, LN, LM, LF, two model seeds each (eight rollouts); references GEc, RLc, RR (banked rollouts); control run V0 of card ADD102. The label gap is a tool of attribution only; no arm is a candidate.

## Control clauses that could fail by construction (read these first)

1. a0, disjointness: a row "carries the label gap" when the gap of the file equals the join's label gap within 1e-9. A range file whose base value on a label row of another range happens to equal the label gap within 1e-9 would count that row as carried and break the disjointness. With a float gap from the three large fits this needs an exact coincidence; the card's composition rule itself cannot produce it.
2. a0, base file: the reader opens the base file at the path the record names (`base.path`, a pod path), unless `--base-file` is given. If the record names a path that is not readable on the pod when the reader runs, a0 fails although the arms may be right.
3. a0, row count: the record's `n_rows` per file is compared with the number of rows in the file. A record that counts only the rows with an output would fail here.
4. a0, composition clause: besides the clauses the task names, the reader also checks, on every join row, that each file equals the label gap where its range holds and the base file's gap and output flag elsewhere (largest difference at most 1e-9), which is the card's own wording of a0. It needs the base file to be the whole file of card ADD106 on the join rows.
5. a2, label share: the share of rule steps whose received gap equals the label gap is printed, not tested. The closing-input clause (a positive share of non-zero closing input on steps with a lead) fails for any rollout whose rule never closes; the card states this as intended.
6. a4: the newer of the two `lead_pytest.out` files must be older than the first line of `FALLATTR10K_START.txt`. The push script copies `lead_pytest.out` to the pod, so its pod modification time is the push time; if the queue is started before the push, a4 fails by order of events, not by the tests.
7. a1: all eight run texts must name the wrapper hash; a run text that has no `wrapper sha256` line fails the clause.

## Stages

Controls stage: runs a0 to a5, the grid read of the arms against GEc, RLc and RR, and the strategic grid (the engine's own controls inside it). Output: `FALLATTR10K_CONTROLS.json` with `controls` (a0 to a5, each with `pass` and `failures`), `all_pass`, `inputs` (sha256 of the model reader, this reader, the join, the gap-file record, the gap files, the wrapper file prefix) and the values under `payload` when every control passes, under `withheld` otherwise. Printed: `a<i> PASS|FAIL` and `ZZ READ-FALLATTR10K controls all_pass <bool>`; the engine's output goes to `<out>.engine.log`.

Names stage: refuses unless `all_pass` is true and a0 to a5 are all present with `pass` true. Names are made by two separate routines (`names_of` and `derive_all`, signs and tables) that must agree, else the stage stops. Writes `FALLATTR10K_RESULT.json` (bound to the controls file by sha256) and a names log.

## What each control reads

- a0: the record has `pass_a0` true and every clause `pass`; each gap file has the sha256 and row count the record names; each composed file equals the card's composition on every join row; the label-gap rows of LN, LM, LF are disjoint and together those of LA (label rows: `lead_valid` 1, finite gap, 0 to 60 m; ranges: LA all, LN below 20 m, LM 20 m to below 40 m, LF 40 m and above); the base file is the record's base (`named_in_record_add106` true, sha256 equal); the join is named `join_eval141.npz` and has the hash the record names.
- a1: as g4 of the model reader: wrapper file 28bc31f21ecb4ccb, the check record of card ADD102 ends with `ZZ CHECK-WRAPPER-V PASS` and names the hash, every one of the eight run texts names it.
- a2: as g5 of the model reader (`w2_one`) on the eight rollouts with their files `VGAPA_<ARM>.npz`: received gap equals the file's gap within 1e-3 m on steps with a lead and an output; steps with a lead and no output equal V0 within 1e-3 m; meta names the gap file hash and the closing switch 1; the closing input is non-zero on a positive share of steps with a lead.
- a3: pooled levels of GEc (0.0062 contact, 0.5154 falling behind, 0.0839 unsafe), RLc (0.0000, 0.4908, 0.0714), RR (0.0954, 0.4954, 0.1339) within 1e-4.
- a4: the last non-empty line of each `lead_pytest.out` holds a count of passed above 0 and neither `failed` nor `error`; the newer modification time is below the first start time (seconds since the epoch, or an ISO time).
- a5: as g8 of the model reader (engine controls of the grid read and of the strategic grid).

## Names (card section 3)

Per arm X in LA, LN, LM, LF and measure M in FALLING-BEHIND, CONTACT, UNSAFE against GEc: `FALLATTR10K-<X>-LOWERS-<M>` (upper end of the interval below 0), `-RAISES-<M>` (lower end above 0), else `FALLATTR10K-<X>-<M>-NOT-SEPARATED`. The whole from falling behind of LA: `-GAP-CARRIES-THE-PRICE` (LA lowers against GEc and the lower end of LA minus RLc is at most 0), `-GAP-CARRIES-PART` (LA lowers and that lower end is above 0), `-PRICE-NOT-IN-THE-GAP` (LA does not lower and that lower end is above 0), else `-NOT-DECIDED`. The range: `-PRICE-IN-NEAR`, `-IN-MIDDLE`, `-IN-FAR` for each of LN, LM, LF that lowers falling behind against GEc, else `-NO-RANGE-SEPARATED`.

Printed with no name (in this order): the tier line; the names; the levels of every arm and reference per model seed and pooled; every arm against GEc, RLc and RR on the three measures; the four families per arm against GEc and RLc (LONGITUDINAL, LATERAL, TACTICAL from the engine's pair read, STRATEGIC from the strategic read at T1-kin); per gap file `n_label_rows` and `n_source_changes` from the record; per rollout the share of rule steps with a lead, of them with a non-zero closing input, and the share of steps with a lead whose received gap equals the join's label gap within 1e-3 m.

## Files

`read_fallattr10k.py` (imports the model reader `read_cropscale10k.py` of card ADD106 from `--model-reader`, default the pod path, else the dev-box copy; calls `rows_of`, `load_logs`, `lead_multi`, `w2_one`, `gapfile_ref`, `meta_of`, `run_file_sha`, `ctl_g8`), `podread_fallattr10k.sh`, `pull_fallattr.sh`, `push_fallattr_read.sh`, `test_fallattr_read.py`, this file. The lead writes `lead_pytest.out`.
