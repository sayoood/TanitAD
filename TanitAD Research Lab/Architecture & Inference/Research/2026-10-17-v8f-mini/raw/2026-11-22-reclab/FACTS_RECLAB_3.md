# FACTS_RECLAB_3: `speed_v1` rebuilt on ONE speed source (base 97fe0a56, tree97). Nothing staged, committed or pushed. Evidence MEASURED (logs named).

## 1. What changed vs build 2 (`stack/tanitad/data/v8f_recovery.py`, integration section only; the verbatim V1 body and its sha pin are untouched)
Build 2 mixed the store's speed column (`v_h`) with pose chords; the label's step speed was v_l + (chord - column), so while v_l ran at 0.99 A_MAX the noise broke the 1 % margin (28.5 % refusals). Now:
* `speed_label(grid, r, v_own)` (:287; `v_log` argument removed): P = human polyline from row r' in its ego frame, s = cumulative arc; **u_h[k] = (s[k]-s[k-1])/0.2** (:~305); `label_steps` (:266) gives u_l: u_l[0] = v_own, then move toward u_h at <= **A_LABEL = 3.0** m/s2 (:263) and follow u_h unconditionally from the first k with |u_h[k]-u_l[k-1]| <= A*dt; L[k] = cumsum(u_l[1..k])*0.2; label point k = `_at_arc(P, s, L)`. Refused (human label kept): L past the path end (`path_exhausted`), L decreasing (`arcs_decreasing`), V1's `violations(label, v_own, 30)` (`refused_infeasible`, terms kept). Identity: u_l[1:] == u_h[1:] exactly -> `human_label` itself (`np.array_equal`).
* The logged speed column is not read anywhere (test `test_one_speed_source_...`). `label_speed` is no longer reused.
* Census (`speed_census` :378): per tick ok / relabelled / identity / refused by status, **`refused_by_term` (accel / curv / fric, not exclusive)**, p50/p95 of **|v_own - u_h[1]|** (key `abs_v_own_minus_u_h1`) and **|L[30]-s[30]|** (key `abs_delta30`), relabelled share.

## 2. Hops (unchanged from FACTS_RECLAB_2 except the labeller lines)
argparse `--dagger-label` `v8f_mini_train.py:243` -> `load_dagger_bank` `:606` `DR.label_bank` (`v8f_recovery.py:333`, per ok record `speed_label` :359) -> `bank.attach_recovery` `:607` / `v8f_dagger.py:285` -> gather `:339` (`rl_has`, `rl_label`) -> `swap_rows` `:611-618` (`gt[rows[hm]] = label` :617 the one install line; `s_h` by `arc_from_label` `v8f_recovery.py:255`, :618) -> `DaggerSwap.apply` counter `:761` -> loss `v8f_mini.py:1182-1190` (ade/fde/acc), `:632`, `:1215` (s_h).

## 3. Checks
* PASSED: dagger-first order, whole files (`test_log_reclab3_run1.txt`): 233 passed, 2 skipped, 2 failed = the compose test (**PRE-EXISTING**, fails identically on unmodified 97fe0a56, `test_log_compose_control_97fe0a56.txt`) and one parametrised formula case of my own (`dv = 0.7`: only one rate step, |L[30]-s[30]| = 0.02 < the 0.05 the test demanded; test case changed to 1.4). Re-run of the three recovery files whole: `test_log_reclab3_rec.txt` **47 passed**. The full dagger-first run was NOT repeated after that one-parameter test edit.
* Tests added/changed: formula vs independent loop (A = 3.0); rate limit; one-speed-source; identity within 0.6 m/s; path-end and kink refusals; new `test_a_rate_equal_to_a_max_without_margin_...` (A = 4.0 refuses on the planted curved path, A = 3.0 does not); gold under `human`, reachability (relabelled rows > 0 under speed_v1, exactly 0 under human; eligible index equal) as before.
* Mutants (`mutants/MUTANTS_RESULT.json`, `run4.log`): **12 of 12 RED** (the 11 retargeted + M12 `A_LABEL = 4.0`).
* Pod suite (`test_log_rl_checker_old_suite3.txt`, RL checker under test): old suite + RL file **125 passed**; the arm-set test in `pod/test/test_dscale10k_check_rhk.py` now expects RL_s0/RL_s1 when the RL checker is loaded. The RHK checker file is not in this tree, so the test was not run against it.
* Real shard (bank/s0 shard 0; the three inputs sha256-verified against the recorded digests, then deleted: `DELETIONS.log`): `REAL_SHARD_CENSUS_2.json`, `REFUSAL_DIAG_2.json`, `real_shard_census_2.log`.

## 4. Real census (372,791 ok of 492,478 records; label_bank 20.0 s)
| | tick 1 (208,340) | tick 2 (164,451) | all ok |
|---|---|---|---|
| relabelled | 80,810 (38.8 %) | 121,266 (73.7 %) | 202,076 (**54.2 %**) |
| identity (human label) | 123,459 | 37,058 | 160,517 (43.1 %) |
| refused | 4,071 (2.0 %) | 6,127 (3.7 %) | 10,198 (**2.7 %**, was 28.5 %) |
Refusals: path_exhausted 1,940 / 2,779; refused_infeasible 2,131 / 3,348. By checker term (not exclusive) tick 1: accel 644, curv 1,556, fric 277; tick 2: accel 606, curv 2,738, fric 341. In the 40,000-record sample: 22,245 non-identity candidates, 571 refused (accel 130, accel-only 105, curv 436, fric 74); candidates' max |accel| p50/p95/p99 = 3.000 / 3.006 / 3.526 (so the 0.99 A_MAX limit is no longer touched at the median).
|v_own - u_h[1]| p50 0.773 / p95 6.221 m/s. |L[30]-s[30]| p50 0.035 / p95 6.326 m. Reachability: 202,076 labels differ from the human label; human kind attaches 0; eligible index equal (372,791 both).
Reading: of the 5,479 checker refusals 4,294 involve curvature (the chord-against-arc term) and 1,250 acceleration; 4,719 more are path-end; acceleration is no longer the main cause. NOT proven on real data: a real swapped batch / loss.

## 5. Not done
Pod commit, pins, G-SUITE, `--selfcheck-only` on the real RL argv (GPU); DRY_MINT_PLAN / read extension not touched (the flag name and argv are unchanged). The identity share (43 %) means those swapped samples carry the human label under speed_v1: the lever acts on 54 % of ok records, 74 % at tick 2.
