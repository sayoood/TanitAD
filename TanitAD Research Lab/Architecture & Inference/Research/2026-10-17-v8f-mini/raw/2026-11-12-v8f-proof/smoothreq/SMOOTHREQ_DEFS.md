# SMOOTHREQ_DEFS -- definitions of the kit of card PREREG_V8F_PROOF_ADD85 (SMOOTHREQ10K)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions). Everything here was built and tested on synthetic data only; nothing ran on the pod or on real data.

## Files and their source

| file | role | derived from |
|---|---|---|
| `make_smooth_tables.py` | pod table tool (CPU), tables of S50, S90, S7B, W7B, Z00, STKM, TKD, controls b0, b1, b2, record `SMOOTH_TABLES_RECORD.json`, `--selftest` | imports `make_tab_tables.py` (ADD84: `ar_along_clips`, `plant_noise`, `_lag1`, `load_tab`, `synth`, `run`) and through it `make_surrogate.py` (ADD83: `u4_rows`, `check_u4`, `load_fit`) |
| `make_smooth_queue.py` -> `pod_smoothreq10k_queue.sh` | 14 rollouts of form RRe at age, seed switch `SR_SEED` | `tabclose/pod_tabclose10k_queue.sh` (cd0a0c2903436bae), 7 counted edits; admission block and rollout list new |
| `make_read_smooth.py` + `read_smooth_frag.txt` + `smooth_selftest_frag.txt` -> `read_smoothreq10k.py` | reader | `tabclose/read_tabclose10k.py` (5fa729f4d5880254): 13 blocks copied (`ctl_rec`, `log_input`, `join_inputs`, `ctl_r1s`, `ctl_x1s`, `ctl_m1s`, `ctl_u2`, `ctl_h1`, `assemble`, `read_e_json`, `register_runs`, `log_arrays`, `sha_file`) and `run_setting` with 6 counted edits; the rest is new. Engine `read_estclose10k_n2` (EC) is imported, not edited |
| `make_pod_read_smooth.py` -> `podread_smoothreq10k.sh`, `chain_read_smoothreq10k.sh` | pod wrappers | `tabclose/podread_tabclose10k.sh`, `tabclose/chain_read_tabclose10k.sh`, counted edits (24) |
| `test_smoothreq.py`, `run_mutations_sr.py` -> `MUTATIONS_SR.json` | tests, mutations | `tabclose/test_tabclose.py`, `run_mutations_tb.py` pattern |

The wrapper `v8f_t1_eval_reader_rule_rzgfhqet.py` (sha256 prefix 02a58e655e5c1979) is not edited; the queue asserts its hash from `shufclose/`. Engine reader files are not edited.

## Definitions

* **Tables** (rows of `tab_L100_s<k>.npz`, arrays `clip_index`, `frame`, `est`, NaN = no value): S50, S90, S7B, W7B on the frames where L100 has a value: `m + r^2 (L - m) + s sqrt(r^2 (1 - r^2)) e`, r = 0.50, 0.90, 0.70, 0.70; e planted as in note N1 of card ADD84 (mean 0, exactly uncorrelated with L, sd 1; smooth arms by the autoregression 0.9 along each clip, restarted per clip). Z00 on every row: `z e`, e the autoregression over all rows made exactly mean 0 and unit sd (not made uncorrelated with L), z = sd (ddof 0) of the finite TK values. Seeds `default_rng(85000 + 10 a + k)`, a = 0..4 for S50, S90, S7B, W7B, Z00, one standard normal per row.
* **Look mean of a run**: mean of `c_est` over the rows with `present` AND `est_ok`, over both rule logs (`rule_log_<run>_s<k>.npz`, `rule_log_mid_...`) together. `c_k` = look mean of `I10RQ50eTKKF5P30Hb20_s<k>`, `d_k` = look mean of `I10RQ50eSTKKF5P30Hb20_s<k>` minus `c_k`. STKM = STK - d_k, TKD = TK - c_k (NaN pattern unchanged).
* **Controls of the table tool**: b0 (as a1 of card ADD84 for the four planted arms; second draws: r between the planted noise of S7B and the noise recovered from the banked S70, of W7B and W70, inside [-0.10, +0.10]), b1 (Z00), b2 (STKM, TKD, constants recomputed by a second routine to 1e-12, r of STKM against the label for H 25 by the u4 code path in [-0.15, +0.15]). Tables are written on a FAIL. `error_autocorr` (lag-1 of table minus L, TK, N10, L25, banked S70, W70, per seed with pair counts) is reported with no control.
* **Queue**: markers `START_SMOOTHREQ10K[_s<k>].txt`, `SMOOTHREQ10K_DONE[_s<k>].txt`, `SMOOTHREQ10K_STOP[_s<k>].txt`, log `queue_smoothreq[_s<k>].log`, small folder `small_sr`. `SR_SEED=0|1|ALL` (default ALL; anything else rc 2). Run order STKM, TKD, Z00, S90, S50, S7B, W7B (seed 0 then 1 each). Admission: pod check of card ADD83, wrapper hash, record `pass` true with b0, b1, b2 PASS, all 14 table hashes equal the record, banked runs exist.
* **Reader**: pairs (19): `X-VS-RR` for S50, S90, S7B, W7B, Z00, STKM (written SHUFM), TKD, banked S70, W70, W50, W90; `S90-VS-W90`, `S50-VS-W50`, `S7B-VS-W7B`, `S7B-VS-S70`, `W7B-VS-W70`; `TOK-VS-SHUFM` (banked TK minus STKM), `TKD-VS-TOK`, `TKD-VS-SHUFM`. Names `SMOOTHREQ10K-<PAIR>-AGE-<C|U|F|family>-<word>`; per arm `...-<ARM>-REMOVES-CONTACT|NOT-SHOWN` (11 arms); verdict keys `VERDICT_SMOOTH_REQUIREMENT` (a), `VERDICT_TIME_COURSE` (b), `VERDICT_SMOOTH_SPREAD_ALONE` (c), `VERDICT_TK_GAIN` (+ `..._BESIDE_SHUFM_VS_RR`) (d), `VERDICT_TKD_BIAS` (e). Controls: inherited g, s, r1, x1, m1, a1, k3, i1, h1 and k4 (levels of the banked runs equal the ADD84 result file; admitted when every control but `a4_u3_AGE` passed, the controls not passed are printed); card controls b0_tables, b1_tables, b2_tables (record control PASS and the sha256 of all 14 tables equal the record), b3 (rule log vs table to 1e-9, parent a2 on the 14 runs), b4.
* **Withholding** (a pair is withheld when it uses a guarded form): b0: forms S50, S90, S7B, W7B; b1: Z00; b2 or b4: TOK-VS-SHUFM, SHUFM-VS-RR, TKD-VS-SHUFM; b2 also every TKD pair; b3 for a run: every pair that uses it. Verdicts follow from the three-valued words of the unwithheld pairs. Values of withheld pairs sit under `withheld_pairs`; an arm of the card whose every pair is withheld has its table row under `withheld_table_rows` and its levels leave `settings/AGE/levels`.

## Differences from the card text

1. Controls b0, b1, b2 of the reader compare the sha256 of ALL 14 tables with the record, not only those of the arm the control guards.
2. (d): the SHUFM-VS-RR name is written beside the verdict only for the `-NOT-SHOWN` reading, as the card says.
3. (a) at 70 with one share None: the level is not reached (a share of None counts as not reaching half).
4. The banked TK appears in the pair names as `TOK` and the banked rows keep their `RRe_` form labels; the arm STKM is written SHUFM in names.
5. The queue checks the banked run `I10RQ50cKF5P30Hb20` (RRc) only as a folder `$W/t1/<run>_s<k>`; the others by their dump folder, as the parent.
6. Z00 is the only table valued on every row; b1 judges r against the label on the L100 frames only.
7. `error_autocorr` of N10 and L25 is looked up by (clip_index, frame) in `est_score_<E>_s<k>.npz`.

## Unverified pod assumptions

1. `est_score_N10_s<k>.npz`, `est_score_L25_s<k>.npz` sit in `/workspace/v8proof/closeread/est`; run folders `<run-dir>/<run>_s<k>/ticks/rule_log_*.npz` hold `ci`, `j`, `present`, `est_ok`, `c_est` for the banked TK and STK runs.
2. The record of card ADD84 is `/workspace/v8proof/closeread/est_tab/TAB_TABLES_RECORD.json` with `a0`, `a1` dicts holding a boolean `PASS` and `tables` (file name to sha256); the tables used are `tab_{L100,TK,STK,S70,W70}_s<k>.npz`.
3. The result file of the read of card ADD84 is `/workspace/v8proof/podread/out/read_tabclose10k/READ_TABCLOSE10K.json` (the chain takes it from there; `SR_TBJSON` overrides) with `card`, `controls` and `settings/AGE/levels/<run>/{U,C,F}` keyed by run names without the `I10` prefix, as the parent reader writes them.
4. `I10RQ50cKF5P30Hb20_s<k>` is a folder in `$W/t1`; the 18 runs of card ADD84 have their dump folders.
5. Two queue processes side by side take two slots of the same GPU slot rule (`MAXGPU=3`); each waits for its own slot.
6. The reader's run_setting wiring (b3, b4, k4, stats of the banked TK logs) was verified by its selftest and tests on the functions, not by a read of real rollouts (no pod, no banked runs here).
7. The strategic read of the engine handles 22 runs at one setting.

## Exact pod command lines

```
# 1. tables (CPU; the out folder must not exist; the tool finds make_tab_tables.py and make_surrogate.py in its own folder)
/workspace/venv/bin/python make_smooth_tables.py --tab-dir /workspace/v8proof/closeread/est_tab --dump-dir /workspace/v8proof/closeread/dump/both \
  --est-dir /workspace/v8proof/closeread/est --fit-dir /workspace/v8proof/closeread/tools --run-dir /workspace/v8proof/integ10k/t1 --seeds 0,1 \
  --out-dir /workspace/v8proof/closeread/est_smooth --tool-dir "$PWD"
# 2. queue dry run, then the two seeds side by side as two processes (markers and logs carry _s0 / _s1), or one process for both
DRY=1 bash pod_smoothreq10k_queue.sh
SR_SEED=0 nohup bash pod_smoothreq10k_queue.sh > /workspace/v8proof/integ10k/queue_smoothreq_s0.out 2>&1 &
SR_SEED=1 nohup bash pod_smoothreq10k_queue.sh > /workspace/v8proof/integ10k/queue_smoothreq_s1.out 2>&1 &
bash pod_smoothreq10k_queue.sh                      # SR_SEED=ALL
# 3. chain (waits for SMOOTHREQ10K_DONE.txt or both _s0/_s1 markers AND GO_READ_SMOOTHREQ10K.txt in $W; any SMOOTHREQ10K_STOP*.txt ends it)
bash chain_read_smoothreq10k.sh
#    direct form: bash podread_smoothreq10k.sh --pull PL --code CODE --grid-dir $W/longroll --fresh-grid FG --midlook-json MJ --estclose-json EJ --tabclose-json TBJ \
#                   --tab-dir /workspace/v8proof/closeread/est_smooth --w $W --tree TREE --out OUT --wrapper-sha 02a58e655e5c1979 --read-dir RD
# local (dev box, from F:/TanitAD-scratch/v8proof):
PYCAP_MB=3000 bash ops/pycap.sh smoothreq/make_smooth_tables.py --selftest
PYCAP_MB=3000 bash ops/pycap.sh smoothreq/read_smoothreq10k.py --selftest
PYCAP_MB=3000 bash ops/pycap.sh -m pytest smoothreq/test_smoothreq.py -q -p no:cacheprovider --basetemp smoothreq/tmp_pt/<new name>
PYCAP_MB=3000 bash ops/pycap.sh smoothreq/run_mutations_sr.py
```
