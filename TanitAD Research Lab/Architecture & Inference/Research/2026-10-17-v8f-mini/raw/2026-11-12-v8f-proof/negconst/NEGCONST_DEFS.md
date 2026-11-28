# NEGCONST_DEFS -- tool kit of card PREREG_V8F_PROOF_ADD82 (NEGCONST10K)

Built on the dev box from the reviewed kit of card ADD80 (folder `constclose/`) by generator scripts with counted, asserted text edits; synthetic data only; nothing was run on real data and nothing touches the pod.
Tier T1-kin/hb0.2, evidence class of every value the kit will give: MEASURED. Never a statement on driving performance.

## 1. Files and where each comes from

| file | derived from (sha256 prefix) | generator | what changed |
|---|---|---|---|
| `make_negconst_files.py` | `constclose/make_const_files.py` | `make_negconst_gen.py` | values -1.0, -2.0, -4.0 (arms M10, M20, M40), names `const_M<cc>_H1.npz`, default out dir `est_negconst`, last line `ZZ NEGCONSTFILES PASS/FAIL`, planted flipped sign in the selftest |
| `check_negconst.py` | `constclose/check_constclose.py` (25185711c68ad55b) | `make_check_negconst.py` | value -2.0, run name `k1_m20`, new check K1e (the sign check), five planted failures |
| `pod_check_negconst.sh` | `constclose/pod_check_constclose.sh` (d7e6a17fc819efe1) | `make_pod_check_negconst.py` | tag `nk1`, markers `CHECK_NEG_DONE_/CHECK_NEG_STOP_<tag>.txt`, record `CHECK_NEGCONST_<tag>.txt` (+ .hashes, .rc), folder `negconst_test` |
| `pod_negconst10k_queue.sh` | `constclose/pod_constclose10k_queue.sh` (493d6361af2ec932) | `make_negconst_queue.py` | 12 rollouts M10, M20, M40 (AGE then FRESH), markers `*_NEGCONST10K*`, log `queue_negconst.log`, binding to the check (tag nk1, wrapper hash, hash of `const_M20_H1.npz`), banked runs of cards ADD79 and ADD80 required |
| `read_negconst10k.py` | `constclose/read_constclose10k_n1.py` (c777b3898a5b6238) | `make_read_negconst.py` | see section 4 |
| `podread_negconst10k.sh`, `chain_read_negconst10k.sh` | `constclose/podread_constclose10k_n1.sh` (07a4eb718a1e9415), `constclose/chain_read_constclose10k.sh` | `make_podread_negconst.py` | both result files required and admitted; runs of ADD79 and ADD80 required |
| `test_negconst.py`, `run_mutations_nc.py` -> `MUTATIONS_NC.json` | `constclose/test_constclose*.py`, `run_mutations_cc.py` | written here | tests and mutations of the new logic |

Every generator asserts the sha256 prefix of its parent (the reader's parent: c777b3898a5b6238) and refuses to overwrite its output (`NC_OVERWRITE_OWN=1` rewrites it).

## 2. Definitions

* **Closing input c** of the rule: the lead is imagined at `v_L = max(0, v_l + c)` (`distrule/dist_rule.py`, line 13). c below zero: the gap closes, the rule brakes more. The cautious sign is negative.
* **Arms**: M10 = -1.0, M20 = -2.0, M40 = -4.0 m/s, history one frame, estimator files written with `closest.save_planted(path, "const", 1, value)`. The look needs 5 frames (H + reader history 5 - 1).
* **Run names**: AGE `I10RQ50e<M10|M20|M40>KF5P30Hb20` (pre-roll 30 steps, 6 s, grid_p30), FRESH `I10RQ50e<..>KF5Hb20` (pinned grid), seeds 0 and 1; wrapper `v8f_t1_eval_reader_rule_rzgfhqe.py` (a747abeb51ef041f).
* **Control k0** (files): each file returns exactly its value for any input of shape [1, 3] and [n, 1, 3], H 1, needs 5 frames (in `make_negconst_files.py`, and in the reader against the file hash of the run line).
* **Control k1** (pod check, file of -2.0 m/s, 3 clips, `check_negconst.py`): K1a closing column -2.0 at every present look with est_ok and 0.0 on present looks without a full history; K1b estimate column -2.0 on every est_ok row (NaN elsewhere); K1c no short look at frame index >= 4; K1d `v_L = max(0, v_l + c_src)`; **K1e, the sign check of the card**: on the present rows with est_ok, `v_L = max(0, v_l - 2.0)` and v_L below v_l wherever v_l > 0 (the 2.0 is taken from the card, not from the c_src column; a log with no row with v_l > 0 fails); L1 run-line counts and file hash; F0 file; W0 wrapper hash. Both logs (replan and second look) each. Note: given equality with `max(0, v_l - 2.0)`, "below v_l" follows on rows with v_l > 0; it is kept as the card's wording.
* **Control k2** (reader, in the runs): estimate column equals c to 1e-6 on est_ok rows of all twelve runs, both logs, NaN elsewhere, a compared first-tick row (the first look of the mid-look log is its smallest tick: the repaired first-tick clause of the parent); coverage of full histories at least 0.99 per setting (logs and run text), else that setting is not named.
* **Control k3**: the levels (U, C, F per seed, to 1e-9) of the runs of card ADD79 (N10, L25) equal `--estclose-json` and the levels of the runs of card ADD80 (C10, C20, C40) equal `--constclose-json`; each file must be the result of its card (`card` field `PREREG_V8F_PROOF_ADD79` / `PREREG_V8F_PROOF_ADD80`) with `controls_all_pass` true. A missing or not admitted file is its own failed control (`k3_estclose_json_admitted`, `k3_constclose_json_admitted`), not a silent skip.
* **Control k4** (AGE only; the sign read from the data): the contact level C of the windows with a lead, pooled as the mean over the two seeds, is not higher at M10 than at C10 and not higher at M40 than at C40 ("not higher" = `<=`). A failure fails the control, so no name is written.
* **Other controls** as the parent: g, s (STRATEGIC by the instrument of card ADD66 at T1-kin), a1 (banked levels equal row 137 / card ADD76 file), r1, x1, m1, i1, h1.
* **Pairs** (A minus B) per value c and setting: CONST-VS-RR-c (RRe(c), RR), CONST-VS-LABEL-CLOSING-c (RRe(c), RRc), CONST-VS-NO-RULE-c (RRe(c), R0K), CONST-VS-EST-E-c (RRe(c), RRe(E)), E in N10, L25: 15 pairs per setting.
* **Names**: `NEGCONST10K-<PAIR>-<SETTING>-<C|U|F|LONGITUDINAL|LATERAL|TACTICAL|STRATEGIC>-<LOWER|HIGHER|NOT-SEPARATED|...>`.
* **Verdict of a value** `NEGCONST10K-<c>-SUPPORTED` / `-NOT-SUPPORTED`: AGE pair CONST-VS-RR-c reads C LOWER, U not HIGHER, no family WORSE; FRESH reads C not HIGHER, U not HIGHER, no family WORSE. The F name of the pair (AGE and FRESH) is written beside every verdict. A single SUPPORTED of the three is marked `NEGCONST10K-<c>-SUPPORTED-ONE-OF-THREE`.
* **Verdict of an estimator E** (match rule): value c matches E when AGE pair CONST-VS-EST-E-c reads C not HIGHER and F not HIGHER (NOT-SEPARATED counts as a match). `NEGCONST10K-<E>-NOT-BEYOND-A-CAUTIOUS-CONSTANT` when at least one of the three values matches, `NEGCONST10K-<E>-BEYOND-A-CAUTIOUS-CONSTANT` when none does.
* **Dose table** over the constants -4, -2, -1, 0 (form RR), +1, +2, +4 m/s: C, U, F levels per seed and pooled (mean over the seeds), printed after the controls and carried in `settings/<st>/dose` of the result file; reported, names nothing.

## 3. Differences from the card (as implemented)

1. The control k3 is split into three entries per run of the reader: the level comparison per setting for each file (`k3_estclose_<st>`, `k3_constclose_<st>`) and the admission of each file (`k3_*_json_admitted`). Same content as the card.
2. The reader needs no ADD80 constant files (it reads the runs of ADD80 and their levels from `--constclose-json`); `--const-dir` is the folder of the new files `est_negconst` only. The chain therefore has no second const-dir argument.
3. k4 reads the contact level `lev` computed in the reader from the cut dumps (the level that i1 compares with the grid read to 1e-12), not the grid read's field.
4. The wiring of `run_setting` and `run` (what the card needs the reader to reach) could not be run without the engine on real data: the selftest covers the pure functions, and `test_negconst.py::test_reader_wiring` pins the calls as static text. The mutations of the wiring are in suite "wiring".
5. The queue requires the 10 banked run groups of cards ADD79 and ADD80 (21 banked run names in all with the earlier ones) by directory presence of the dump, as the parent does for its banked runs.

## 4. Edits of the reader against its parent (counted by the generator)

Values M10/M20/M40 with the negative CVAL; `OLDV`/`OLDVAL` for the runs of card ADD80; run table and form table carry the runs of ADD80 always and of ADD79 when e_on (the reader runs with e_on true); names with prefix NEGCONST10K and the verdict names with CAUTIOUS; `read_c_json`, `ctl_k3`, `ctl_k4`, `dose_table`, `render_dose`; `run_setting` gets the argument `cj`; `run` reads both files and adds the admission controls; `main` makes both `--estclose-json` and `--constclose-json` required.

## 5. Unverified pod assumptions

* `closest.py` on the pod (`$B/closest.py`, `CLOSEST_DIR`) accepts a negative value in `save_planted` exactly as the local copy does; k0 on the pod decides.
* The wrapper `v8f_t1_eval_reader_rule_rzgfhqe.py` passes a negative estimator value to the rule without clamping it; k1 (K1a, K1b, K1e) decides before the queue.
* The result file of card ADD79 (`podread/out/read_estclose10k_n2/READ_ESTCLOSE10K.json`) carries `card`, `controls_all_pass` and `settings/<AGE|FRESH>/levels/<run>/{U,C,F}`; the result file of card ADD80 (`podread/out/read_constclose10k/READ_CONSTCLOSE10K.json`) the same. Both runs of the reader that wrote them must have had every control pass.
* The run folders `I10RQ50eN10..`, `I10RQ50eL25..` (ADD79) and `I10RQ50eC10/C20/C40..` (ADD80) are linked in the pull folder `$P/root/integ10k/pull` before the chain starts (the chain links only the new `I10RQ50eM*` runs).
* The kit files are placed where the parents sit: `check_negconst.py` and `closest.py` in `$B` (`/workspace/v8proof/integ10k/build`), `read_negconst10k.py` in `$RD` (`/workspace/v8proof/podread/root/integ10k`) beside `read_estclose10k.py`, `podread_negconst10k.sh` and `chain_read_negconst10k.sh` in `$P` (`/workspace/v8proof/podread`).
* Memory and GPU slot behaviour as the parent queue (28 GB free, at most three compute processes).

## 6. Exact pod command lines

```
# 1. the files (control k0); closest.py from the build folder
CLOSEST_DIR=/workspace/v8proof/integ10k/build /workspace/venv/bin/python make_negconst_files.py          # prints "ZZ NEGCONSTFILES PASS"; out dir /workspace/v8proof/closeread/est_negconst (must not exist)
# 2. the pod check (control k1), tag nk1; check_negconst.py already in /workspace/v8proof/integ10k/build
bash pod_check_negconst.sh                                                                                 # last line "ZZ POD-CHECK-NEGCONST DONE ZZ CHECK-NEGCONST PASS ..."; record $W/negconst_test/CHECK_NEGCONST_nk1.txt(.hashes)
# 3. the queue: dry run, then the real one
DRY=1 I10_W=/workspace/v8proof/integ10k bash pod_negconst10k_queue.sh                                      # prints 12 CMD lines and "DRY DONE"
bash pod_negconst10k_queue.sh                                                                              # markers START_NEGCONST10K.txt, NEGCONST10K_DONE.txt or NEGCONST10K_STOP.txt in $W
# 4. the read: chain waits for NEGCONST10K_DONE.txt and the go file
echo "go" > /workspace/v8proof/integ10k/GO_READ_NEGCONST10K.txt                                            # written by the lead
bash chain_read_negconst10k.sh                                                                             # output /workspace/v8proof/podread/out/read_negconst10k/READ_NEGCONST10K.json
```
Local tests: `PYCAP_MB=3000 bash ops/pycap.sh -m pytest -q -p no:cacheprovider negconst/test_negconst.py --basetemp negconst/tmp_pt/x`; mutations: `PYCAP_MB=3000 bash ops/pycap.sh negconst/run_mutations_nc.py --out negconst/MUTATIONS_NC.json`.
