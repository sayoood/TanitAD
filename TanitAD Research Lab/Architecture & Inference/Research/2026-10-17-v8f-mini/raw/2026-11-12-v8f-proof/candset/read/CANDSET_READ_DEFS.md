# CANDSET_READ_DEFS: what the reader kit of card ADD113 (CANDSET10K) and its note N1 codes

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions). Settings CONV (no pre-roll, pinned readout, pinned grid), K0 (no pre-roll, readout K, pinned grid), K60 (pre-roll of 60 steps (12 s), readout K, `grid_p60.json`),
K30 (pre-roll of 30 steps (6 s), banked, read again as identity). Arms GEc (new at CONV, K0, K60; banked at K30), RR, RLc (banked at all four). Run names are in the table `SPEC` of `read_candset10k.py`; run folders
`I10<name>_s<k>` in the pull folder. The reader is a tool that imports the model reader `read_cropscale10k.py` of card ADD106 (engine calls `read_grid`, `strategic_grid`, `ucf`, `reading`, `family_of`, and
`rows_of`, `load_logs`, `lead_multi`, `w2_one`, `gapfile_ref`, `meta_of`, `run_file_sha`, `ctl_g8`); no existing reader or engine file is edited.

## What could fail by construction (read first)

1. d2, presence clause: the card text asked for pre-roll, grid and nav in both lines; the reference lines of CONV and K0 carry none of them. Note N1 replaces it (coded as written, see the definition below).
2. d0, the reader's own logged gap: card ADD102's control run V0 exists at the pre-roll of 30 steps only. For the lead steps where VGAPLE has no output the received gap is compared with the rule log of the reference run RR
   of the same setting and model seed (the reader-fed form logs the reader's own gap at every lead step). A reference rule log that lacks a lead step of the new run, or lacks `rule_log_mid`, fails d0 by construction.
3. d0, meta: the rule log meta must name the gap file hash (16 or more hex digits, a prefix of the file's sha256 either way) and closing switch 1; a log written without that meta fails.
4. d4, order: the reader's test output must be not later than the start of the controls stage (the driver takes `date +%s` at its start), the queue test output not later than the first start epoch.
5. d5, strict: the strategic grid is read at all four settings; if the engine cannot read one (an exception, or a failed control of the strategic stage) d5 fails and nothing is opened; the family is then reported as not read
   only in the failure text of d5 (the card has no map of arms to controls). The engine has read the strategic grid at P = 0 on the pinned grid (card ADD79), P = 30 and P = 60 (cards ADD70, ADD71).
6. Grid read at K60: 284 windows; the engine's headway instrument has its own bar (200 windows with a lead and 20 episodes). The values are read whether or not the bar is met; `instrument_bar_met` and the counts are
   printed per setting. The engine's own controls (d5) are those of the model reader, as read at P = 60 before.

## Definitions the card leaves open

- Pooled levels: mean over the two model seeds of the per-seed level (as the reader of card ADD111). Pair difference: first run minus second (GEc minus RR, GEc minus RLc, RR minus RLc); interval: the engine's two-level
  bootstrap over model seeds and episodes, 2,000 draws (`--n-boot`); the controls read uses 300 draws.
- Pooled contact rate of RR (NO-CONTACT-TO-CUT): the pooled level `C` of RR, below 0.01.
- Names: contact name from the interval of C(GEc) - C(RR) (upper end below 0: CUT-HOLDS; lower end above 0: RAISED; else NOT-SEPARATED); unsafe name from U(GEc) - U(RR); price from the lower end of F(GEc) - F(RLc)
  above 0; label-form contact from the lower end of C(GEc) - C(RLc) above 0. Overall name: CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT when at least one setting is not NO-CONTACT-TO-CUT and every such setting reads
  CONTACT-CUT-HOLDS; CUT-RAISED-AT-A-SETTING when any reads CONTACT-RAISED; else CUT-NOT-AT-EVERY-SETTING. Two routines (`names_of`, `derive_all`) must agree, else the stage stops.
- d0: gap file on disk: its sha256 begins with `base.sha256` of the record of card ADD111 (16 or more hex digits); the rule-log checks are those of `w2_one` (1e-3 m, closing input non-zero on a positive share).
- d1: the wrapper file sha256 begins 28bc31f21ecb4ccb; both check files exist; the last line of the pod check is `ZZ CHECK-WRAPPER-V PASS`; the hashes name the wrapper; the run text of each of the six new runs names a wrapper
  hash that begins with it.
- d2 (note N1): first line of each run log, `key value` pairs parsed tolerantly; every field both lines carry is equal (form, replan steps, rule every, readout, kappa, preroll, exec, nav, grid, model seed, reader seed,
  checkpoint hash; numbers compared as numbers); form, replan steps, rule every, readout, kappa, model seed and reader seed present in both; a reference line without preroll needs `preroll 0` in the new line, without grid
  `grid eval_grid_windows.json`, without nav `nav deadreckon`; the wrapper hash is not compared; the rule log shows a share of rule steps with a lead above 0. The q-map file of each new run is checked by the lead from
  the dry run and is not coded. A field the reference carries and the new line lacks is not a failure of its own (as N1 words it).
- d3: K30 levels of GEc, RR, RLc read by the same function (`measure_setting`) as the new settings, within 1e-4 of the banked values.
- d4: a pytest line `N passed` (N above 0) and no `<n> failed` or `<n> error(s)` count in each output file; modification times as in risk 4; times are epoch seconds or ISO.
- d6: `T1_DONE` in each new run folder; the number of `ep*.npz` in its dump folder equals the number of clips (`order`) of its grid file (pinned grid, `grid_p60.json`); when `--done-marker` is given its last line must
  read `ZZ CANDSET10K DONE <ISO time> all steps done`.
- Controls stage prints only `d<i> PASS|FAIL` and `ZZ READ-CANDSET10K controls all_pass <bool>`; names stage refuses unless the controls file has the card name, `all_pass` true and d0 to d6 all pass; the printed tier line gives
  per setting its pre-roll as "no pre-roll" or "pre-roll of 60 steps (12 s)".

## Files

`read_candset10k.py`, `podread_candset10k.sh` (kit `/workspace/v8proof/candset/kit_read`, outputs `/workspace/v8proof/candset/read/`), `pull_candset.sh` (into `candset/pod_add113`, refuses if it exists),
`push_candset_read.sh`, `test_candset_read.py`, this file. The lead writes `candset/lead_tests/lead_pytest_queue.out` and `lead_pytest_read.out` on the pod.

- Lead, before any test run or read: the rule logs of a run are read by `load_logs_setting`. With `--rule-every 5` (K0, K60, K30) both logs are needed, by the model reader's own `load_logs`; with no `--rule-every` (CONV, pinned readout) a run writes `rule_log` only (seen on the pod for the banked reference `I10RQ50Hb20_s0` and for the new run `I10RQ50Hb20gGEc_s0`: file names only), so the first log is needed and a mid log is read only if it lies on disk. Without this, d0 would fail at CONV by construction.
