# TABCLOSE_DEFS -- definitions of the queue and reader kit of card PREREG_V8F_PROOF_ADD84 (TABCLOSE10K)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions). Everything here was built and tested on synthetic data only; nothing ran on the pod or on real data. The table tool (`make_tab_tables.py` and its files) is another helper's; the two kits meet in the file contract below.

## Files and their source

| file | role | derived from |
|---|---|---|
| `make_tab_queue.py` -> `pod_tabclose10k_queue.sh` | 18 rollouts of form RRe with `--closing-table`, part switch, admission | `shufclose/pod_shufclose10k_queue.sh` (5cac167387fe4654), 7 counted edits; the admission block and the rollout list are new |
| `make_read_tabclose.py` + `read_tabclose_frag.txt` + `tabclose_selftest_frag.txt` -> `read_tabclose10k.py` | reader | `shufclose/read_shufclose10k.py` (8d5391d73cde4ca8): blocks `log_input`, `join_inputs`, `ctl_x1s`, `ctl_m1s`, `read_e_json`, `register_runs`, `log_arrays`, `sha_file` copied, `ctl_r1s`, `ctl_u2`, `ctl_h1`, `run_setting` copied with 13 counted edits; everything else is new. Engine imported as `EC` (`estclose_read/read_estclose10k_n2.py`), not edited |
| `make_pod_read_tabclose.py` -> `podread_tabclose10k.sh`, `chain_read_tabclose10k.sh` | pod wrappers | `shufclose/podread_shufclose10k.sh` (338f2c2fd183acca), `shufclose/chain_read_shufclose10k.sh` (39e98b463d712faf), counted edits |
| `test_tabclose.py`, `run_mutations_tb.py` -> `MUTATIONS_TB.json` | tests and mutations | `shufclose/run_mutations_tools.py` pattern |

The wrapper (`v8f_t1_eval_reader_rule_rzgfhqet.py`, sha256 prefix 02a58e655e5c1979) is the one of card ADD83 and is not edited; the queue asserts its hash from the file in `shufclose/`.

## File contract with the table tool (unchanged)

Folder `/workspace/v8proof/closeread/est_tab`; files `tab_<ARM>_s<k>.npz`, ARM in TK, STK, L100, W00, W50, W70, W90, S70, k in 0, 1; arrays `clip_index` (int64), `frame` (int64), `est` (float64, NaN = no value); record `TAB_TABLES_RECORD.json` with top-level `"pass"`, `"tables"` (file name -> sha256 of the whole file) and control dicts `a0`, `a1`, `a4_u1`, `a4_u4`, each with a boolean `"PASS"`.

## Definitions

* **Run names** `I10RQ50e<ARM>KF5P30Hb20` (AGE: pre-roll of 30 steps, 6 s, grid `grid_p30.json`) and `I10RQ50eTKKF5Hb20` (FRESH, pinned grid). Order of the queue: TK, STK, L100, W70, W50, W90, W00, S70 (AGE, each seed 0 then 1), then TK FRESH. Part A = TK AGE, STK AGE, TK FRESH (6 rollouts); part B = the other 12.
* **Queue switches**: `TB_PART=A|B|ALL` (default ALL; anything else is refused with rc 2); `I10_TABLES` (table folder, default `/workspace/v8proof/closeread/est_tab`); `SC_TAG` (tag of the pod check of card ADD83, default `sk1`); `DRY=1`. Markers in `$W`: `START_TABCLOSE10K[_A|_B].txt`, `TABCLOSE10K_DONE[_A|_B].txt`, `TABCLOSE10K_STOP.txt`; step markers `markers/ro_<arm>.done` (shared by the parts, so a later ALL skips what a part finished); log `queue_tabclose.log`; small outputs `small_tb`.
* **Admission** (not in DRY): `$W/shufclose_test/CHECK_SHUF_DONE_<tag>.txt` has a line starting `ZZ CHECK-SHUFCLOSE PASS`; the wrapper sha256 prefix equals 02a58e655e5c1979 and `CHECK_SHUFCLOSE_<tag>.txt.hashes` holds it; `TAB_TABLES_RECORD.json` admitted by the python block `admit()` of the queue: record controls `a4_u1`, `a4_u4` say PASS for a read of part A, `a0`, `a1` for part B (both for ALL); every table file of the part exists, is in the record and has the record's sha256; the banked runs (RR, RRc at both settings, no rule R0K at AGE and FRESH, R0, the four runs of card ADD79) exist.
* **Reader** (`--part A|B|ALL`, default ALL). Settings AGE (all arms of the part) and FRESH (TK; only when part A is read). Each setting is read completely before the next.
  * Pairs, A minus B. Part A, both settings: `TOK-VS-RR`, `TOK-VS-EST-L25`, `TOK-VS-EST-N10`, `TOK-VS-LABEL-CLOSING`, `TOK-VS-NO-RULE` (TK against RR, RRe(L25), RRe(N10), RRc, R0K); AGE only: `TOK-VS-SHUF` (TK, STK), `SHUF-VS-RR` (STK, RR). Part B, AGE: for X in L100, W00, W50, W70, W90, S70: `X-VS-RR`, `X-VS-LABEL-CLOSING` (B = banked RRc), `X-VS-NO-RULE` (R0K); and `W50-VS-TOK`, `W50-VS-EST-L25`, `S70-VS-W70`. `W00-VS-RR` is the `X-VS-RR` pair of W00 (one pair, not two). `W50-VS-TOK` needs the TK runs and is therefore read only with part ALL.
  * Names `TABCLOSE10K-<PAIR>-<SETTING>-<C|U|F|family>-<word>`; four families always; C, U, F only when the headway instrument is qualified on the grid.
  * Verdicts (three-valued: a failing condition decides without the others; an unavailable word that is needed means the name is not written and `why` says so). `VERDICT_TK`: `TABCLOSE10K-TK-SUPPORTED` when AGE TOK-VS-RR reads C LOWER, U not HIGHER, no family WORSE and FRESH reads C not HIGHER, U not HIGHER, no family WORSE, else `TK-NOT-SUPPORTED`; beside it `VERDICT_TK_BESIDE_F_AGE` and `_FRESH` (the F names of TOK-VS-RR). `VERDICT_TK_BEYOND_L25`: `TK-BEYOND-L25` when AGE TOK-VS-EST-L25 reads C LOWER, else `TK-NOT-BEYOND-L25`. `VERDICT_TK_GAIN`: `TK-GAIN-FROM-THE-SCENE` when AGE TOK-VS-SHUF reads C LOWER and U not HIGHER, else `TK-GAIN-NOT-SHOWN-FROM-THE-SCENE`, with `VERDICT_TK_GAIN_BESIDE_SHUF_VS_RR` (the C name of SHUF-VS-RR at AGE). Per arm X: `ARM_<X>_REMOVES` = `X-REMOVES-CONTACT` (X-VS-RR C LOWER and U not HIGHER) or `X-NOT-SHOWN`; `ARM_<X>_LABEL_LEVEL` = `X-AT-LABEL-LEVEL` (X-VS-LABEL-CLOSING C not HIGHER) or `X-ABOVE-LABEL-LEVEL`. `VERDICT_REQUIREMENT`: `REQUIREMENT-R<nn>` with nn the lowest of 50, 70, 90 whose W arm has REMOVES-CONTACT and a pooled share of at least 0.5 (exactly 0.5 counts; a share of None, i.e. a zero gap, does not reach half); `NO-TESTED-LEVEL-REACHES-HALF` when none; not written when the REMOVES name of a lower arm is unavailable and no lower arm qualified. `VERDICT_SPREAD_ALONE`: `SPREAD-ALONE-REMOVES-CONTACT` when W00-VS-RR reads C LOWER, else `SPREAD-ALONE-NOT-SHOWN`. `S70_BESIDE_W70` carries the S70 and W70 names side by side; `VERDICT_REQUIREMENT_MARK` = `TABCLOSE10K-REQUIREMENT-DEPENDS-ON-THE-TIME-COURSE-OF-THE-ERROR` when the REMOVES name or the LABEL-LEVEL name of S70 differs from that of W70.
  * **Table** (reported without names): per form in the order R0K, RR, W00, N10, L25, TK, STK, W50, W70, S70, W90, L100, RRc, RLc: C, U, F per seed and pooled (mean over the two seeds), and the share of the RR-to-RRc contact gap, `(C_RR - C_X) / (C_RR - C_RRc)`, per seed and pooled (pooled from the pooled levels); None when the gap is exactly 0.
  * **Controls.** Inherited (a failure leaves no name and no value, as the parent): `g`, `a1` (levels of the banked runs vs `--midlook-json`), `k3` (levels of the ADD79 runs vs `--estclose-json`, required), `s`, `r1` (run text as before; the est line's file hash equals the table's sha256 read here), `x1`, `m1`, `i1`, `h1`. Card controls: `a0_tables`, `a1_tables` (record control PASS and every part-B table hash equals the record's), `a4_u1`, `a4_u4` (same for the part-A tables), `a2_<setting>` (the estimate column of both rule logs equals the table at the clip and frame to 1e-9, with `est_ok` equal to "the table holds a finite value", first look of the mid-look log = its smallest tick), `a3_AGE` (C, U, F of the L100 runs equal those of the banked RRc runs per seed within 1/325), `a4_u3_AGE` (over looks with a lead read present: sd of the STK input within 0.80 to 1.25 times that of the TK input, |mean| of the STK input <= 0.15 m/s, share of looks marked full >= 0.90; this is the coverage gate). `controls_all_pass` is true only when every control passed.
  * **Withholding by a failed card control.** `a0_tables`, `a1_tables` or `a3`: no part B pair name and no part B verdict (arms, REQUIREMENT, SPREAD-ALONE, S70 mark). `a4_u1`, `a4_u4` or `a4_u3`: no name of TOK-VS-SHUF and SHUF-VS-RR and no GAIN verdict. `a2` failed for a run: no name of any pair at that setting that uses the run, and verdicts that need such a pair are not written. The table is still printed and carries `controls_failed` beside it in the JSON.
  * A part that was not run is listed in `parts_not_run`; its pairs and verdicts are absent and its controls are not run, which is not a failed control.
* **Output** `READ_TABCLOSE10K.json` (+ `.txt` of the printed lines): `card`, `tier`, `controls`, `controls_all_pass`, `controls_failed`, `names`, `why` (= `why_not_written`), `settings/<AGE|FRESH>/{levels, table, pairs, words, ...}`, `pairs` (words per pair), `table` (AGE), `part`, `parts_not_run`. The control lines are printed first, then a `== names` separator line, then the names.

## Differences from the card text

1. The queue requires the record's top-level `pass` only for `TB_PART=ALL`; with a part switch it requires the controls that guard the part (A: `a4_u1`, `a4_u4`; B: `a0`, `a1`) so that part A can start while a control of part B is open. The hashes of the tables of the part are always compared.
2. The `W00-VS-RR` pair is the `X-VS-RR` pair of W00 (identical comparison, one set of names). `W50-VS-TOK` is read only when the TK runs are read in the same call (part ALL).
3. The banked form RRh of the parent reader is not read (not named by the card); its runs are not required.
4. The share is None when the gap is exactly 0 (as the card), and a negative gap gives the arithmetic value; a share of None counts as not reaching half.
5. a2 additionally requires `est_ok` to equal "the table holds a finite value at (ci, j)" on every row (as the parent's u2); the lookup frame is the `j` column of the log (assumed to be the frame of the look, tick look and later look alike).
6. `a0_tables`, `a1_tables`, `a4_u1`, `a4_u4` are read from the record; the reader does not recompute them. The table hashes of part A are not compared with the record by a control that guards the TOK pairs (only the run line's hash against the table read here, in r1, guards those).
7. The coverage gate of the reader is the share of full looks of `a4_u3` (the table arms have no other coverage measure).
8. The pod admission of the queue is checked as text and through its python block (`admit()`) in the tests; the non-DRY branch was not run end to end (no pod, no banked runs).

## Unverified pod assumptions

1. `v8f_t1_eval_reader_rule_rzgfhqet.py` is in `$W/build` with its parents; `CHECK_SHUF_DONE_<tag>.txt` and `CHECK_SHUFCLOSE_<tag>.txt.hashes` exist as card ADD83's pod check writes them.
2. `TAB_TABLES_RECORD.json` follows the contract above (`tables` maps file names without a directory to the sha256 of the whole file; control dicts hold a boolean `PASS`).
3. The rule logs (`rule_log_<key>.npz`, `rule_log_mid_<key>.npz`) hold `ci`, `j`, `k`, `present`, `c_src`, `c_est`, `est_ok` as in the runs of card ADD83, and `j` is the frame the wrapper looked up.
4. The banked runs of rows 137 and card ADD79 are in `$W/t1` and the pull folder holds links to them (as the parents).
5. Every engine module of the parent reads (`read_estclose10k_n2.py`, `read_handsrc_decomp.py`, `read_closrc10k.py`, `read_agedose_decomp.py`, `read_agerule_decomp.py`, `read_longroll10k*.py`, `strat_roll_read.py`) is in the read dir; `read_tabclose10k.py` imports `read_estclose10k_n2` from `TB_ESTREAD_DIR`, `../estclose_read` or its own folder.
6. The strategic read of the engine handles 14 forms per setting (the parent read 11 and 13).
7. The headway instrument qualification and the engine's `read_grid` accept the new run names registered in memory by `register_runs`.

## Exact pod command lines

```
# 0. tables: made by the table tool into /workspace/v8proof/closeread/est_tab (record TAB_TABLES_RECORD.json); pod check of card ADD83 passed
# 1. copy to /workspace/v8proof/integ10k/build nothing new: the wrapper of card ADD83 is used; copy pod_tabclose10k_queue.sh to the pod
# 2. queue (markers START_TABCLOSE10K[_part].txt, TABCLOSE10K_DONE[_part].txt / TABCLOSE10K_STOP.txt in $W; DRY=1 prints the commands)
bash pod_tabclose10k_queue.sh                      # TB_PART=ALL
TB_PART=A bash pod_tabclose10k_queue.sh            # six rollouts: TK AGE, STK AGE, TK FRESH
TB_PART=B bash pod_tabclose10k_queue.sh            # twelve rollouts of part B
# 3. reader chain (waits for the queue marker and GO_READ_TABCLOSE10K.txt in $W; then podread_tabclose10k.sh)
bash chain_read_tabclose10k.sh                     # TB_PART=A|B for a part
#    direct form: bash podread_tabclose10k.sh --pull PL --code CODE --grid-dir $W/longroll --fresh-grid FG --midlook-json MJ --estclose-json EJ \
#                   --tab-dir /workspace/v8proof/closeread/est_tab --w $W --tree TREE --out OUT --wrapper-sha 02a58e655e5c1979 --read-dir RD [--part A|B|ALL]
# local (dev box, from F:/TanitAD-scratch/v8proof):
PYCAP_MB=3000 bash ops/pycap.sh tabclose/read_tabclose10k.py --selftest
PYCAP_MB=3000 bash ops/pycap.sh -m pytest tabclose/test_tabclose.py -q -p no:cacheprovider --basetemp tabclose/tmp_pt/<new name>
PYCAP_MB=3000 bash ops/pycap.sh tabclose/run_mutations_tb.py
```
