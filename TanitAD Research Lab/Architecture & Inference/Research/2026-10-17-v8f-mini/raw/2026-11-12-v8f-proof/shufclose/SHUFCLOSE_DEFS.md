# SHUFCLOSE_DEFS -- definitions of the tool kit of card PREREG_V8F_PROOF_ADD83 (SHUFCLOSE10K)

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions). Everything here was built and tested on synthetic data only; nothing ran on the pod or on real data.

## Files and their source

| file | role | derived from |
|---|---|---|
| `make_surrogate.py` | pod tool: map pi, surrogate / identity / zero / const2 tables, controls u1 and u4, `SURROGATE_RECORD.json` | new; uses `closeread/closeread_fit.py` (`load_dump`, `build_rows`) |
| `make_shuf_wrapper.py` -> `v8f_t1_eval_reader_rule_rzgfhqet.py` | wrapper with `--closing-table` | `estclose/v8f_t1_eval_reader_rule_rzgfhqe.py` (sha256 a747abeb51ef041f asserted), 6 counted edits Y1-Y6 |
| `test_shuf_wrapper.py`, `run_mutations_sc.py` -> `MUTATIONS_SC.json` | tests and mutations of the wrapper | `estclose/test_estclose_wrapper.py`, `run_mutations_ec.py` |
| `check_shufclose.py`, `pod_check_shufclose.sh` | pod control u0 | `estclose/check_estclose.py`, `pod_check_estclose.sh` |
| `make_shuf_queue.py` -> `pod_shufclose10k_queue.sh` | eight rollouts | `estclose/pod_estclose10k_queue.sh` (b68b73b66c9ead6f), 10 counted edits |
| `read_shufclose10k.py`, `podread_shufclose10k.sh`, `chain_read_shufclose10k.sh` | reader and its pod wrappers | `estclose_read/read_estclose10k_n2.py` (imported as `EC`, not edited), `constclose/read_constclose10k_n1.py`, `podread_estclose10k_n2.sh`, `constclose/chain_read_constclose10k.sh` |
| `run_mutations_tools.py` -> `MUTATIONS_TOOLS.json` | mutations of `make_surrogate.py`, `check_shufclose.py`, `read_shufclose10k.py` through their `--selftest` | `estclose/run_mutations_ec.py` |

## Definitions

* **Map pi** (card section 2): clips = the distinct `clip_index` values of the score dump of reader seed k. Share of a clip = frames with `p >= 0.5` over frames of the clip in that dump (0.5 counts). Order by (share, clip index). Neighbours exchanged: first with second, third with fourth, ... With an odd count the last three of the order (a, b, c) form the cycle a -> b -> c -> a (the card does not fix the direction; this is the choice). No fixed point, bijection. The same pi serves both estimators of the seed.
* **Surrogate table** `surr_<E>_s<k>.npz`: arrays `clip_index`, `frame`, `est` (row for row the rows of `est_score_<E>_s<k>.npz`), `map`, `map_clips` (`map[i]` = pi of `map_clips[i]`). `est` at row (c, j) = estimate of clip pi(c) at frame j; NaN where clip pi(c) has no row at frame j or its estimate is NaN. `ident_<E>_s<k>.npz` = the estimate table itself. `zero_table.npz` / `const2_table.npz` = the rows of the first estimate table with 0.0 / 2.0.
* **`--closing-table NPZ`** (wrapper): arrays `clip_index`, `frame`, `est`. Admitted only with form RRe, refused with `--closing-est` (message: "REFUSED: --closing-est and --closing-table together ..."), refused with another form ("REFUSED: --closing-table belongs to form RRe only"); form RRe with neither flag is refused. At a look the value is the table at (`s.ci`, frame of the look: j for the tick look, j + q for a later look), the same key `_est_at` uses. No row, NaN or an infinite value: `est_ok` False, `c_est` NaN, closing input exactly 0.0. The file hash in the run line `est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>` is the table's. The table path reads no reader features. A table row order does not matter; duplicate keys, negative clip index or frame, a frame >= 2**20 and missing arrays are refused.
* **u0** (pod check, tag default `sk1`): U0A table `ident_L25_s0` through the new form against banked `I10RQ50eL25KF5Hb20_s0` on the rows of the first three clips (fresh, readout K, rule every 5, q50 map, seed 0): `c_src` within 1e-3 on every common row of the replan and the later-look log (every subset row has to be a banked row), `est_ok` equal exactly, first plans (`p1_first`) within 1e-3, the check is not empty (full rows, present full rows, non-zero closing exist); the other columns are reported, not judged. U0B `zero_table` equals a form RR run of the same check with largest difference 0 on dump, tick, rule-log, side, later-look, later-look side and first-plan files; `c_src` 0 everywhere; `c_est` / `est_ok` present row for row. U0C `const2_table` shows 2.0 on every present look with `est_ok` (0.0 on present looks without; `v_L = max(0, v_l + c_src)`). U0L run lines. U0F three refusals (RR with table; RRe with both flags; RRe with neither).
* **u1** (make_surrogate): per table, read back from disk: clips = dump clips; map bijection; no fixed point; equals the map recomputed by an independent routine (exact fractions, Python sort); rows equal the estimate rows; every value equals the estimate table at (pi(c), j) exactly (NaN where none). **u4**: rows of `closeread_fit.build_rows(dump, H)` (H 10 for N10, 25 for L25), surrogate value at (clip_index of the row, frame) against `lab_closing`, pooled over the seeds, rows with a finite surrogate value; Pearson r inside [-0.15, +0.15]; the r of the estimate is printed beside it (not judged).
* **u2** (reader): in all eight runs and both logs: `c_est` equals the surrogate table at (ci, j) within 1e-9 on `est_ok` rows; `est_ok` is True exactly where the table holds a finite value at (ci, j); `c_est` NaN where not `est_ok`; first look rows are among the compared (rule log k == 0, later-look log: its smallest tick).
* **u3** (reader): looks with `present` True, both logs, per estimator pooled over seeds; input = `c_src`. Gate (AGE, judged): sd(surrogate input) / sd(estimate input of the ADD79 runs) in [0.80, 1.25] (population sd), |mean of surrogate input| <= 0.15, share of those looks with `est_ok` >= 0.90. The share over all logged looks is reported beside. FRESH: same numbers, `judged` False. A gate that fails: no AGE name and no verdict of that estimator (the other is still read); the control itself passes when every number is there.
* **Reader controls**: `sur` (record says pass; u1 and u4 of every table and estimator passed; the sha256 of every surrogate table read equals the record's; u1 and u4 carried into the result), u2, u3, `k3` (levels of the ADD79 runs equal `--estclose-json`; the file has to carry card PREREG_V8F_PROOF_ADD79 and `controls_all_pass` true; required, a missing one fails k3 and no name is written), g, s, a1 (banked levels vs `--midlook-json`), r1 (run text as before plus: the est line's file hash equals the surrogate table's), x1, m1, i1 (triples include the ADD79 runs), h1.
* **Pairs** (A minus B): EST-VS-SHUF-E = RRe(E) - RRe(surrogate of E); SHUF-VS-RR-E = RRe(surrogate) - RR; SHUF-VS-NO-RULE-E = RRe(surrogate) - no rule (R0K). **Names** `SHUFCLOSE10K-<PAIR>-<SETTING>-<C|U|F|family>-<word>` (C, U, F need a qualified headway instrument; four families always). **Verdicts** per E on AGE: `SHUFCLOSE10K-<E>-GAIN-FROM-THE-SCENE` when EST-VS-SHUF-E reads C LOWER and U not HIGHER (LOWER or NOT-SEPARATED); `...-GAIN-NOT-SHOWN-FROM-THE-SCENE` when either fails (a failing word decides without the other); not written when a needed word is missing. `VERDICT_<E>_BESIDE_SHUF_VS_RR` = the name of SHUF-VS-RR-E (C, AGE); `VERDICT_N10_MARK` = `SHUFCLOSE10K-N10-VERDICT-RESTS-ON-A-GAIN-THAT-ROW-141-DID-NOT-HAVE-SEPARATED`; `VERDICT_L25_MARK` = `SHUFCLOSE10K-L25-GAIN-OF-ROW-141-WAS-SEPARATED`.

## Unverified pod assumptions

1. `est_score_<E>_s<k>.npz` and `reader_s<k>_score.npz` have the arrays the lead stated; the est table's clip set equals the dump's clip set (refused otherwise); the dump satisfies `closeread_fit.load_dump` (sorted by clip, consecutive frames).
2. `clip_index` of the est table is the same index as `s.ci` of the rollout states (the card says so; the u0 check against the banked run tests it on three clips).
3. `closeread_fit.py` is found in `--fit-dir`, `CLOSEREAD_DIR`, the folder of the tool or `../closeread`; on the pod copy it beside `make_surrogate.py` or give `--fit-dir`.
4. The reader dump `cid` indexes `clips`, and `clip_index` is constant per `cid`.
5. The queue needs, beside the banked runs of row 137, the four ADD79 runs `I10RQ50e{N10,L25}KF5{P30,}Hb20_s{0,1}` with dumps; the pod check needs `I10RQ50eL25KF5Hb20_s0/T1_DONE`; build folder `$W/build` must hold the wrapper `v8f_t1_eval_reader_rule_rzgfhqet.py`, its parent `..rzgfhqe.py`, `check_shufclose.py`, `closest.py`.
6. `SURROGATE_RECORD.json` is written with `json.dump(indent=1, sort_keys=True)`; the queue and the pod read test the line ` "pass": true` (one space of indentation).
7. The rule-log arrays `ci`, `j`, `k`, `present`, `c_src`, `c_est`, `est_ok` exist in both logs of the new runs (as for the runs of card ADD79); `read_estclose10k_n2.py` and the engine modules are in the read dir.
8. The check and the queue assume the tool tree, grids, qmaps, checkpoints at the paths of the estclose scripts (`/workspace/v8proof/...`).

## Exact pod command lines

```
# 1. surrogate tables (new folder; refuses an existing one)
/workspace/venv/bin/python make_surrogate.py --est-dir /workspace/v8proof/closeread/est --dump-dir /workspace/v8proof/closeread/dump/both \
    --out-dir /workspace/v8proof/closeread/est_shuf --seeds 0,1 --fit-dir /workspace/v8proof/closeread
#    (est-dir = the folder that holds est_score_<E>_s<k>.npz; last line ZZ SURROGATE PASS)
# 2. copy to /workspace/v8proof/integ10k/build: v8f_t1_eval_reader_rule_rzgfhqet.py, check_shufclose.py (rzgfhqe.py, rzgfhqc.py, closest.py are there)
# 3. wrapper check (marker $W/shufclose_test/CHECK_SHUF_DONE_sk1.txt; last line ZZ CHECK-SHUFCLOSE PASS)
bash pod_check_shufclose.sh                 # TAG=sk1 default; ANALYZE=1 TAG=sk2 RUNTAG=sk1 re-reads a finished folder
# 4. queue (markers START_SHUFCLOSE10K.txt, SHUFCLOSE10K_DONE.txt / _STOP.txt in $W; DRY=1 prints the commands)
bash pod_shufclose10k_queue.sh
# 5. reader chain (waits for the queue and for GO_READ_SHUFCLOSE10K.txt in $W; then podread_shufclose10k.sh)
bash chain_read_shufclose10k.sh
#    direct form: bash podread_shufclose10k.sh --pull PL --code CODE --grid-dir $W/longroll --fresh-grid FG --midlook-json MJ --estclose-json EJ \
#                   --shuf-dir /workspace/v8proof/closeread/est_shuf --w $W --tree TREE --out OUT --wrapper-sha 02a58e655e5c1979 --read-dir RD
# local (dev box): PYCAP_MB=3000 bash ops/pycap.sh <tool>.py --selftest ; PYCAP_MB=3000 bash ops/pycap.sh -m pytest test_shuf_wrapper.py -q --basetemp=tmp_pt/<new name> -p no:cacheprovider
```

## Differences from the card

* The direction of the three-clip cycle is not given by the card; chosen a -> b -> c -> a over the last three of the sorted order.
* u3 judges the three numbers on looks with a lead read present (the population of the sd and the mean); the share of full looks over all logged looks is reported beside it.
* u2 additionally requires `est_ok` to equal "the table holds a finite value there" on every row (stronger than the card text).
* The reader writes the SHUF-VS-RR name and a mark beside every written verdict as keys of `names` (`VERDICT_<E>_BESIDE_SHUF_VS_RR`, `VERDICT_<E>_MARK`); the L25 mark is added beside the N10 mark that the card asks for.
