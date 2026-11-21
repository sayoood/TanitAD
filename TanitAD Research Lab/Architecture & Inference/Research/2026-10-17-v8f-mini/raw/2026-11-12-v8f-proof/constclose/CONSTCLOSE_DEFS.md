# CONSTCLOSE_DEFS: definitions, sources and pod commands of the tool kit of card ADD80 (CONSTCLOSE10K)

Card: `PREREG_V8F_PROOF_ADD80.md` (called "the card"; section numbers are its sections). Parent card: `PREREG_V8F_PROOF_ADD79.md`. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions).
Parent files (all unchanged): `estclose/pod_estclose10k_queue.sh` (b68b73b66c9ead6f), `estclose/pod_check_estclose.sh`, `estclose/check_estclose.py`, `estclose_read/read_estclose10k.py` (826bacc8d0e36be1),
`estclose_read/podread_estclose10k.sh`, `closeread/closest.py`. Only `constclose/` holds new files.

## 1. Files

| file | job | derived from |
|---|---|---|
| `make_const_files.py` | writes `const_C10_H1.npz`, `const_C20_H1.npz`, `const_C40_H1.npz`; control k0 | `closest.save_planted(path, "const", 1, value)` (closest.py:102), card section 2 |
| `make_constclose_queue.py` -> `pod_constclose10k_queue.sh` | the queue of twelve rollouts | `estclose/pod_estclose10k_queue.sh` by counted text edits (parent generator `estclose/make_estclose_queue.py`) |
| `check_constclose.py`, `pod_check_constclose.sh` | pod check, control k1 | `estclose/check_estclose.py` (run_cmd, lines 124-143; run line, L1 lines 381-396), `estclose/pod_check_estclose.sh` |
| `read_constclose10k.py`, `podread_constclose10k.sh` | the read of card ADD80 | imports `read_estclose10k.py` unchanged (helpers `log_fed`, `ctl_a1`, `ctl_i1`, `run_text_ok`, `parse_est_looks`, `qualified`, `render`), engine modules under it |
| `test_constclose.py`, `test_constclose_queue.py`, `run_mutations_cc.py` | tests and mutations | `estclose_read/run_mutations_ecr.py` |

## 2. Definitions

* **Constant file** (card section 2): `closest.save_planted(path, "const", H=1, value)`: arm L, H 1, mu 0, sd 1, w 0, b = value; estimate = value for every input. `need_frames()` = H + 5 - 1 = 5 (closest.py:59). Values C10 = 1.0, C20 = 2.0, C40 = 4.0 m/s.
  The same file is used for model seed 0 and 1.
* **Form RRe with a constant** (card section 2): the wrapper `v8f_t1_eval_reader_rule_rzgfhqe.py` (a747abeb51ef041f) with `--closing-est <file>`; closing = c at a look with a full five-frame window (`est_ok` True), 0.0 without (wrapper lines 9-10, 349-371).
* **k0** (card section 4): each file loaded by `ClosingEstimator.load` returns exactly c for random inputs of shape [1, 3] and [n, 1, 3], H 1, need 5. `make_const_files.py` (tool, before the queue) and `read_constclose10k.py` (`ctl_k0`, in the read, plus equality of the sha256 of each file with the sha256 in the run line of each run of its value).
* **k1** (card section 4, `check_constclose.py`): one rollout with C20 on three clips, fresh. K1a closing column `c_src` = 2.0 on rows `present` and `est_ok`, 0.0 on `present` rows without `est_ok`; K1b `c_est` = 2.0 on every `est_ok` row of the replan log and the mid-look log, NaN elsewhere;
  K1c no row with `j` >= 4 has `est_ok` False; K1d `v_L` = max(0, `v_l` + `c_src`) on present rows; L1 the run line `est looks scored full n short n preroll full n short n file <sha256>` (parent line format, check_estclose.py:287) has counts equal to the
  counts recomputed from the two logs (pre-roll 0 0) and the sha256 of the constant file; F0 the file itself (k0); W0 the wrapper hash. Last line `ZZ CHECK-CONSTCLOSE PASS|FAIL ...`.
* **k2** (card section 4, `ctl_k2`): in each of the twelve runs and in both logs (replan, mid-look) `c_est` = c within 1e-6 on `est_ok` rows, finite, at least one row, at least one row of the first replan tick (k == 0, as parent control e2),
  NaN where `est_ok` is False. Coverage (`ctl_cov`): per setting pooled over values and seeds, share of `est_ok` looks over the logged looks and over the scored window of the run text, both >= 0.99; else the names of that setting are not written
  (values are), and no verdict is written (a verdict needs both settings; the E verdicts need AGE). Per-value shares and the pre-roll share are reported.
* **k3** (card section 4): only when `--estclose-json` points to a file with `card` = PREREG_V8F_PROOF_ADD79 and `controls_all_pass` true: the levels (U, C, F per seed) of the runs RRe(N10), RRe(L25) recomputed here equal `settings/<setting>/levels` of that file to 1e-9
  (the function of parent control a1, `read_estclose10k.ctl_a1`). A file that is absent, of another card or not all-pass: the pairs with E and the E verdicts are not written, the reason is printed, this is not a failed control.
* **g, s, a1, r1, x1, m1, i1, h1**: as the parent read (read_estclose10k.py lines 140-345), adapted to the twelve runs: r1 and x1 and m1 run over C10/C20/C40; x1 checks the closing column equals c on present rows with `est_ok`, 0.0 without, form RRe in the rule log, and that RR feeds 0;
  m1 later-look row counts equal the banked RR run of the seed; i1 triples (a minus RLc) = (a minus RR) + (RR minus RLc) per run and RRc; a1 levels of the five banked forms equal `READ_MIDLOOK10K.json`; h1 planted names through the engine, name builder of this reader (F rule shown).
* **Pairs** (card section 3), A minus B: `CONST-VS-RR-<c>` (RRe(c), RR), `CONST-VS-LABEL-CLOSING-<c>` (RRe(c), RRc), `CONST-VS-NO-RULE-<c>` (RRe(c), R0K), and `CONST-VS-EST-<E>-<c>` (RRe(c), RRe(E)) for E in N10, L25; c in C10, C20, C40.
* **Names**: `CONSTCLOSE10K-<PAIR>-<SETTING>-<C|U|F>-<LOWER|HIGHER|NOT-SEPARATED>` (interval rule of the engine, `R.u_name`, applied to C, U and F), and `...-<FAMILY>-<BETTER|WORSE|MIXED|NOT-SEPARATED>` for LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC (STRATEGIC by the instrument of card ADD66 at T1-kin, as parent).
* **Verdict `CONSTCLOSE10K-C<cc>-SUPPORTED` / `-NOT-SUPPORTED`** (card section 3): pair CONST-VS-RR-c at AGE: C LOWER, U not HIGHER, no family WORSE; and at FRESH: C not HIGHER, U not HIGHER, no family WORSE. A condition that already fails decides NOT-SUPPORTED even when another word is unavailable;
  otherwise an unavailable word writes no verdict. The F names of the pair at AGE and FRESH are written beside every verdict written (`VERDICT_<cc>_F_AGE`, `VERDICT_<cc>_F_FRESH`). Exactly one SUPPORTED of three: `CONSTCLOSE10K-C<cc>-SUPPORTED-ONE-OF-THREE`.
* **Verdict `CONSTCLOSE10K-<E>-BEYOND-A-CONSTANT` / `-NOT-BEYOND-A-CONSTANT`** (card section 3): value c matches E when CONST-VS-EST-E-c at AGE reads C not HIGHER and F not HIGHER (NOT-SEPARATED counts as not HIGHER). NOT-BEYOND when at least one value matches; BEYOND when none does.
  A value with an unavailable word that has no match among the others writes no verdict.
* **A failed control**: `assemble` leaves names, settings and values out; `render` (parent) prints no name and no value. Output `READ_CONSTCLOSE10K.json` and text; last line `ZZ READ-CONSTCLOSE10K rc 0 controls_all_pass <bool>`.

## 3. Differences from the card, decisions for the lead

1. **Coverage bar 0.99 is applied to the pool of the setting** (all six runs of a setting, both logs together and the run texts of the scored window), as the parent applies its bar per estimator and setting. The card says "per setting". A single value or seed below 0.99 inside a passing pool is reported (per_value) and not judged.
2. **The coverage gate also needs the share by the logs**, not only by the run text (parent practice, control e3).
3. **Control k1 "the count of looks without a full history is zero from the fifth frame of a clip on"** is implemented on the column `j` (clip frame of the look). In the mid-look log the wrapper estimates at frame j + q (wrapper line 787), so a row with j < 4 may be `est_ok` True; that is allowed.
4. **The pod check needs no banked run** (the parent check compared with banked RR/RRh; the card asks only the constant); it uses the real file `const_C20_H1.npz`, and the queue binds to it by the sha256 prefix in the check record.
5. **The queue also requires the banked RRc and RRh runs** (the parent list lacks them; the reader needs them for CONST-VS-LABEL-CLOSING and for control a1).
6. **The reader reads levels of RRh and RLc** (control a1 and i1 as parent) although no pair of the card uses them.
7. **The selftest of the reader** plants outcomes on the pure functions (verdict branches, E branches, gates, controls, names, not-written branches) and uses the stand-in engine for h1; the engine path (`run_setting`) is NOT exercised locally because it needs the pod data.

## 4. Pod assumptions I could not verify (no pod, no real data)

* The wrapper accepts an estimator file with H = 1 (need 5): its code takes `est.H` and `est.need_frames(rd.hist)` generically (lines 346-366); never run here with H 1.
* `closest.py` is in the build folder `$B` (parent check sets `CLOSEST_DIR=$B`); `make_const_files.py` and `check_constclose.py` have to be copied into `$B` (and the check by name `check_constclose.py`, as `pod_check_constclose.sh` expects).
* The log arrays `present`, `c_src`, `c_est`, `est_ok`, `j`, `k`, `v_l`, `v_L` exist with these names in `rule_log_<arm>.npz` and `rule_log_mid_<arm>.npz` (taken from parent check C1 and read code); `present` = lead read present.
* The logs hold the looks of the scored window (the pre-roll has its own counts in the run line; wrapper line 1293).
* The run text line is exactly `est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>` and the sha256 is of the file passed with `--closing-est` (parent check L1).
* `READ_ESTCLOSE10K.json` has `card` = PREREG_V8F_PROOF_ADD79 (the parent reader writes `CARD`), `controls_all_pass` and `settings/<AGE|FRESH>/levels/<run name>`.
* Banked run folders `I10RQ50cKF5P30Hb20`, `I10RQ50hKF5P30Hb20`, `I10RQ50cKF5Hb20`, `I10RQ50hKF5Hb20` exist (names from the parent reader table).
* `podread_constclose10k.sh` expects all reader modules in `--read-dir` (the parent list plus `read_constclose10k.py`, `read_estclose10k.py`, `closest.py`); the selftest on the pod writes temporary folders `tmp_cr_*` and `tmp_cf_*` beside the tools.
* The queue's `DRY=1` was run locally under Git bash and printed the twelve commands (it touches no pod path); the real start checks (`$W`, checkpoints, grids) were not run.

## 5. Commands for the pod

```
# 0. copy the tools into the build folder $B (= /workspace/v8proof/integ10k/build; closest.py and the wrapper are already there)
cp make_const_files.py check_constclose.py $B/ ; cp pod_check_constclose.sh pod_constclose10k_queue.sh /workspace/v8proof/integ10k/
# 1. control k0, the three files (refuses when the folder exists)
CLOSEST_DIR=$B /workspace/venv/bin/python $B/make_const_files.py --out-dir /workspace/v8proof/closeread/est_const        # last line: ZZ CONSTFILES PASS
# 2. control k1, the pod check (three clips; markers in $W/constclose_test)
bash /workspace/v8proof/integ10k/pod_check_constclose.sh                                                                 # TAG=ck1; last line of CHECK_CONST_DONE_ck1.txt: ZZ CHECK-CONSTCLOSE PASS
# 3. the queue (twelve rollouts); DRY=1 first
DRY=1 bash /workspace/v8proof/integ10k/pod_constclose10k_queue.sh | grep -c '^CMD'                                        # 12
nohup bash /workspace/v8proof/integ10k/pod_constclose10k_queue.sh > /workspace/v8proof/integ10k/queue_constclose.out 2>&1 &
# 4. the read (CPU only), after CONSTCLOSE10K_DONE.txt; with the read of card ADD79 when it exists
bash podread_constclose10k.sh --pull PULL --code CODE --grid-dir DIR --fresh-grid FILE --midlook-json READ_MIDLOOK10K.json --const-dir /workspace/v8proof/closeread/est_const \
     --w /workspace/v8proof/integ10k --tree TREE --out OUTDIR --wrapper-sha a747abeb51ef041f [--estclose-json READ_ESTCLOSE10K.json] --read-dir READDIR
```

Local tests (dev box, one capped job at a time): `PYCAP_MB=3000 bash ops/pycap.sh constclose/<tool>.py --selftest`; `PYCAP_MB=3000 bash ops/pycap.sh -m pytest -q -p no:cacheprovider constclose/test_constclose.py --basetemp constclose/tmp_pt/<x>`;
`PYCAP_MB=3000 bash ops/pycap.sh constclose/run_mutations_cc.py --out constclose/<file>.json --tag <t>` (60 mutations; MUTATIONS_CC2.json = 60 of 60 red, controls green before and after; MUTATIONS_CC.json = the first run, 59 of 60, kept: the check of k0 did not plant a value off by 1e-9, added after).
