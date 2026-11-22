# PRESATTR_READ_DEFS: what the reader kit of card ADD114 (PRESATTR10K) codes

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions). Settings K0 (readout K, no pre-roll, pinned grid, engine grid key `full`) and K30 (readout K, pre-roll of 30 steps (6 s), `grid_p30.json`, key `30`).
Arms per setting (run names in the table `SPEC` of `read_presattr10k.py`, prefix I10): RLc (label presence, label gap), LA (camera presence, label gap), PL (label presence, camera gap through the planted join JLE),
GEc (camera presence, camera gap), RR (reference). New runs: LA at K0, PL at K0, PL at K30, both model seeds (six rollouts); every other cell is banked and read again. The reader imports the model reader
`read_cropscale10k.py` (engine calls `read_grid`, `strategic_grid`, `ucf`, `reading`, `family_of`, and `rows_of`, `load_logs`, `lead_multi`, `w2_one`, `gapfile_ref`, `meta_of`, `run_file_sha`, `ctl_g8`);
no existing reader or engine file is edited. Built from the reader of card ADD113 (`read_candset10k.py`).

## What could fail by construction (read first)

1. e7, PL runs: the check assumes that in the rule log of an RLc run with `--rule-join`, `present` means "the rule had a lead at this step" and `g_src` is the gap the rule received from the join, and that
   the logged frame `j` is the frame index of the join row. If the RLc log fills `present` from the label marking alone (not applying the 0 to 60 m test), or logs `g_src` in another unit or after a lag, e7 fails
   for every PL run and the failure text gives the number of disagreeing steps by kind. If the log has pre-roll steps (`j` below 0 or past `n[ci]`), they count as steps that map to no row.
2. e7, steps with the label valid and the planted gap above 60 m: the rule has no lead there (the label form is not run on them with a lead). The check expects `present` false on them. Their count is
   `n_replaced_rows_above_gap_max` of the record (printed with no name). If the rule log still shows a lead on them, e7 fails (this is the intended reading, not a defect of the check).
3. e7, LA at K0: for lead steps where `VGAPA_LA.npz` has no output, the received gap is compared with the rule log of the reference run RR of K0 at the same model seed (as the reader of card ADD113 does). A
   reference rule log that lacks a lead step of the LA run (the reader's presence at the step differs between the two runs), or lacks `rule_log_mid`, fails e7 by construction.
4. e2, K0 references: the banked lines of RR and RLc at K0 carry no pre-roll, exec, nav or grid field. Note N1 of card ADD113 applies: the new line has to read `preroll 0`, `grid eval_grid_windows.json`,
   `nav deadreckon`. The queue's lines of the three new names do carry them. A field the reference has and the new line lacks is not a failure of its own.
5. e4, order: the reader's test output must be not later than the start of the controls stage (the driver takes `date +%s` at its start), the queue test output not later than the first start epoch.
6. e5, strict: both strategic grids are read; if the engine cannot read one the card fails and nothing is opened.
7. e6, 141 dump files for each new run (`--expect-dumps`, default 141); the grid clip count is not compared.

## Definitions the card leaves open

- Pooled levels: mean over the two model seeds of the per-seed level. Pair difference: first run minus second; interval: the engine's two-level bootstrap over model seeds and episodes, 2,000 draws (`--n-boot`);
  the controls read uses 300 draws. Pairs: `LA-VS-RLc` (PRESENCE-AT-LABEL-GAP), `GEc-VS-PL` (PRESENCE-AT-CAMERA-GAP), `PL-VS-RLc` (GAP-AT-LABEL-PRESENCE), `GEc-VS-LA` (GAP-AT-CAMERA-PRESENCE), and `GEc-VS-RLc`
  (printed, no name). Strategic grid at T1-kin for each of the five pairs.
- Names: per setting S, contrast and measure M in UNSAFE (`U`), CONTACT (`C`), FALLING-BEHIND (`F`): interval above 0 gives `PRESATTR10K-<S>-<contrast>-RAISES-<M>`, below 0 `-LOWERS-<M>`, else
  `-<M>-NOT-SEPARATED`. Over-all name per setting on U: at least one presence contrast raises and no gap contrast does gives PRESENCE-CARRIES-UNSAFE; gap and no presence gives GAP-CARRIES-UNSAFE; both kinds give
  BOTH-CARRY-UNSAFE; none gives UNSAFE-NOT-ATTRIBUTED. Two routines (`names_of`, `derive_all`) must agree, else the stage stops. 26 names.
- OPEN DEFINITION the lead has to confirm on real data: `ci` of the rule log indexes the `clip` order of the join (the same order as `clip_index` of the gap files), and the join row of a logged step is
  `off[ci] + j` (rows of a clip from `off[ci]` for `n[ci]` frames). The failure text of e7 names how many steps disagree.
- e0: `JLE_RECORD.json` has `pass_e0` true and every clause true; the sha256 of `JLE.npz`, of `VGAPLE.npz` and of `join_eval141.npz` on disk equal `join.sha256`, `base.sha256`, `eval_join_sha256` of the record
  (full 64 digits). The record's own clauses carry the plant (only `lead_gap` differs, only on the named rows, within 1e-4 m, a second routine agrees); the reader does not open the join arrays for e0.
- e2: first line of each run log, `key value` pairs parsed tolerantly; reference run RR of K0 for LA, RLc of the setting for PL; rules of note N1 of card ADD113 (see `d2_line_failures`); the rule log shows a
  share of rule steps with a lead above 0. The q-map file of each new run is checked by the lead from the dry run and is not coded.
- e3: banked levels within 1e-4 (C/U/F). K30: GEc 0.0062/0.0839/0.5154, RR 0.0954/0.1339/0.4954, RLc 0.0000/0.0714/0.4908, LA 0.0000/0.0839/0.4877. K0: GEc 0.0000/0.0498/0.1651, RR 0.0296/0.0738/0.1682,
  RLc 0.0000/0.0365/0.1550. Read by the same function (`measure_setting`) as the new cells.
- e7, LA: the sha256 of the gap file on disk begins with (or equals) `la_file.sha256` of `GAPERR_GAPFILES_RECORD.json`; the rule log meta names that hash and closing switch 1; the checks of `w2_one` (1e-3 m,
  closing input non-zero on a positive share). PL: see `pl_check`: on each logged step with `present` the gap equals `lead_gap` of the join row within 1e-3 m, and `present` equals "lead_valid == 1, gap finite,
  0 <= gap <= 60" step by step over both logs. The rule logs are loaded with `load_logs_setting` (two logs at every-5 settings).
- Controls stage prints only `e<i> PASS|FAIL` and `ZZ READ-PRESATTR10K controls all_pass <bool>`; names stage refuses unless the controls file has the card name, `all_pass` true and e0 to e7 all pass.

## Files

`read_presattr10k.py`, `podread_presattr10k.sh` (kit `/workspace/v8proof/presattr/kit_read`, outputs `/workspace/v8proof/presattr/read/`), `pull_presattr.sh` (into `presattr/pod_add114`, refuses if it exists),
`push_presattr_read.sh`, `test_presattr_read.py`, this file. The lead writes `presattr/lead_tests/lead_pytest_queue.out` and `lead_pytest_read.out` on the pod. Markers: `PRESATTR10K_START.txt`
(first line `<epoch> <ISO time>Z pid <pid>`), `PRESATTR10K_DONE.txt` (line `ZZ PRESATTR10K DONE <ISO time> all steps done`).
