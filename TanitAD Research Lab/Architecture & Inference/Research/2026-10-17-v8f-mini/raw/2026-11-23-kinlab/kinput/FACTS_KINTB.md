# FACTS_KINTB: `--dagger-kin-tb SECONDS` (BRIEF_BUILD_KINTB, 2026-11-23)

Evidence class: MEASURED here (dev box, CPU, `CUDA_VISIBLE_DEVICES=-1`). Tree `F:/TanitAD-scratch/kinlab/tree3` = `git archive 1a8b11a1 stack taniteval` + 7 files (`tree3/CHANGED_FILES_KINTB.txt`, sha256 and base blobs). Nothing staged, committed or pushed; nothing on C:; no GPU; nothing run on the pod (one read-only pull of shard 0 of RH bank s0 and the tracks union, sha256 aefbb46c / 90229524 verified, deleted afterwards, see `DELETIONS.log`). No RK number is used or claimed.

## What was built
| piece | where |
|---|---|
| `T_B = 1.0` stays the module default; new `KIN_TB_MIN = 0.2`, `KIN_TB_MAX = 3.0`, `check_kin_tb(tb)` (real number, not bool, finite, in [0.2, 3.0], else ValueError); `kin_label(..., tb=None)` and `kin_label_bank(..., tb=None)`: `None` = the module constant read at the CALL (an existing test monkeypatches `RV.T_B`); blend weight `w = min(1, t / tb)` and the merge test `t >= tb` use it. No other constant changed | `v8f_recovery.py` |
| flag `--dagger-kin-tb` (float, default 1.0); `kin_tb_refusal(a)` (range, and non-default only with `--dagger-label kin_v1`) called at the load (-> `ZZDVB-REFUSEDZZ`) and again in `bad_dagger`; passed to `DR.kin_label_bank(tb=)` and to `label_cache_identity(kin_tb=)`; `bank.kin_tb` set for kin_v1; FLAG_CHECKS `"dagger_kin_tb": "dagger:dagger_kin_tb"` (wired exactly as `dagger_label_cache`); `dfl` gains the key (flag at p 0 / with no swap is refused); declared-vs-built message `lever dagger_kin_tb: declared X but built Y` | `v8f_mini_train.py` |
| `DaggerBank.kin_tb` (default 1.0); `built_levers()["dagger_kin_tb"]` (1.0 with no bank); the cache header of a **kin_v1** cache gains `kin_tb` (a speed_v1 header does not: its labels do not depend on it); `read_label_cache` compares it and refuses `[kin_tb]` naming both values; a header **without** the field reads as 1.0 | `v8f_dagger.py` |
| `LABEL_CACHE_FIELDS` is unchanged (so `info...label_cache.header` keeps its key set); `kin_tb` is compared separately | `v8f_dagger.py` |
| `test_v8f_kincache.py` adjusted in two lines: the pin "labeller bytes == 0ca6f49d" is dropped (the labeller was edited on purpose), and the no-flag levers comparison pops the new key | tests |

## Results (logs in `kinlab/`)
* `test_v8f_kintb.py` (new, 34 tests), whole file, final sha256 b448edd8: **34 passed** (`kt_kintb_final.log`).
* Neighbours on tree3, whole file each: `test_v8f_kinlab` 35, `test_v8f_kincache` 24, `test_v8f_dagger` 57, `test_v8f_dagger_train` 95 (2 skipped), `test_v8f_recovery` 22, `test_v8f_recovery_wiring` 4, `test_v8f_rolladd` 25: **all passed** (`kt_nb_*.log`). (First run of `test_v8f_kinlab` had 1 failure, `test_t_b_is_the_blend_horizon...`: it patches `RV.T_B`, and a `tb=T_B` default had bound the value at definition. Fixed by `tb=None` resolved at the call.)
* All 21 launch-gate files (`test_launch_gate*.py`, `test_profile_testsuite.py`), whole file each: **all passed**, the counts of the KINCACHE run (150 / 46 / 24 / 99 / 43 / 58 / 12 / 49 / 41 / 12 / 68 / 3 / 26 / 59 / 17 / 11 / 4 / 2 / 10 / 56 / 9) (`kt_lg_*.log`).
* Not run: the whole suite (`pytest stack/`), G-SUITE at the new commit, anything on the pod or GPU.

## Default = 1a8b11a1, bit for bit (tests + real records)
* Fixture, natural and planted banks: `kin_label_bank` with the flag absent, `tb=1.0`, `tb=1` and `tb=RV.T_B` equals the golden 1a8b11a1 labeller (`tests/golden_v8f_recovery_1a8b11a1.py`, sha256 795d380c) in slot, label, is_kin, status_counts (order), census; per record status, merge time, max_dev and label bytes equal.
* **Real records**: 60,000 ok (+500 non-ok) of shard 0, 583 clips: gold = tb 1.0 = flag absent in slot, label, is_kin, status_counts, census, all bit for bit (`KINTB_CENSUS.json`, `gold_1a8b11a1_equals_...`).
* The swap (human, speed_v1, kin_v1) equals the golden 1a8b11a1 `v8f_dagger.py` (sha256 3a409ced) at two steps; `built_levers()` equals it except the new key `dagger_kin_tb` = 1.0.

## Cache bytes (answer to design item 3)
**The cache FILE bytes are NOT identical to a 1a8b11a1-built file, for a reason independent of `kin_tb`:** the header holds `labeller_sha256` and `dagger_sha256`, the sha256 of `v8f_recovery.py` and `v8f_dagger.py`, and this change edits both files. Tested: a default cache written by the new code has **byte-identical ARRAYS** (slot, label, is_kin, status_counts_json, census_json, and the same `arrays_sha256`) to one written by 1a8b11a1's code on the same bank, and its header differs in exactly `labeller_sha256`, `dagger_sha256` and (kin_v1 only) the new `kin_tb: 1.0`.
**Consequence the MM must decide on:** the two RK caches built at 1a8b11a1 are valid for the 1a8b11a1 code and are **refused by this tree, naming `[labeller_sha256, dagger_sha256]`**, at the default and at 2.0 alike. That is KINCACHE's own binding ("any later edit of `v8f_dagger.py` or `v8f_recovery.py` invalidates a cache; rebuild"), not a `kin_tb` effect. What the new code does with an old header's missing `kin_tb` is tested: read as 1.0, so with the code hashes held equal (the test names the golden files in the identity) the old cache loads at 1.0 and is refused `[kin_tb]` at 2.0 (`test_a_cache_written_by_1a8b11a1s_code_reads_at_the_default...`). I did not add a list of "accepted predecessor code hashes" to let the old caches load on the new code: that would weaken the code binding and is the MM's call. RK itself runs on the 1a8b11a1 code and is unaffected; a T_B 2.0 arm needs a new cache anyway (about 14 min CPU per bank, KINCACHE ESTIMATE).

## Fingerprints
No test-pinned launch-gate fingerprint moved: the 21 launch-gate files (incl. the profile fingerprint tests) pass whole-file on tree3. What does move: the bytes (hence the gate token's `tree_sha256`, any trainer pin the MM keeps, the G-SUITE) of `v8f_mini_train.py`, `v8f_dagger.py`, `v8f_recovery.py`. Not checked: the pod's own pin lists.

## 2.0 moves only kin labels
Test, planted + natural fixture banks: records that are `identity_rule` or whose reference speed_v1 label was refused keep status and label exactly; every emitted record at 2.0 has `t_merge >= 2.0`; a record emitted at neither horizon keeps its speed_v1 label; at least one emitted record's label differs between 1.0 and 2.0. **Interpretation:** "records that are not kin_emitted at 1.0 keep their label" cannot hold for a record REFUSED at 1.0 that merges at 2.0 (it becomes a kin label; 1,160 of 60,000 real records do); it holds for identity and speed_ statuses, which do not depend on T_B (1,971 real records, all unchanged).

## Census, real records (shard 0 of RH bank s0, first 60,000 ok of 60,500 records; `KINTB_CENSUS.json`, `census_kintb.py`, `census_kintb.log`)
| | T_B 1.0 | T_B 2.0 |
|---|---|---|
| ok records | 60,000 | 60,000 |
| relabelled (kin_emitted) | 38,042 | 33,638 |
| **relabelled share of ok** (KINLAB bar 0.5) | **0.6340** | **0.5606** (bar cleared, by 0.061) |
| refused_not_merged | 13,003 | 17,105 |
| refused_path_exhausted | 4,221 | 4,391 |
| refused_infeasible (by term, not exclusive: accel / curv / fric) | 2,763 (484 / 2,281 / 903) | 2,895 (886 / 2,030 / 676) |
| speed_path_exhausted / speed_refused_infeasible (reference refused, T_B independent) | 862 / 1,109 | 862 / 1,109 |
| identity_rule | 0 | 0 |
| merge time s: min / p50 / p95 / p99 / max | 1.0 / 2.2 / 5.4 / 6.0 / 6.0 | 2.0 / 3.2 / 5.6 / 6.0 / 6.0 |
| merge-time mass at the earliest allowed point | 5,475 at 1.0 s (14.4 %) | 6,083 at 2.0 s (18.1 %) |
| max abs a over emitted labels, m/s2: p50 / p95 / max (A_LIM 3.96) | 2.30 / 3.76 / 3.9600 | 2.07 / 3.90 / 3.9600 |
| max abs kappa, 1/m: p50 / p95 / max (K_LIM 0.2) | 0.0081 / 0.125 / 0.1980 | 0.0088 / 0.118 / 0.1980 |
Transitions 1.0 -> 2.0 on the same 60,000: emitted at both 32,478 (all 32,478 labels differ); emitted at 1.0 only 5,564; at 2.0 only 1,160; neither 20,798. The census at 2.0 therefore **shrinks** the kin share by 7.3 points while it moves every kin label (the merge-time histograms are in the JSON). Bank-level `kin_label_bank` census and the per-record loop agree (asserted).

## Mutants (`mutants_kintb/MUTANTS_RESULT.json`, runner `run_mutants_kintb.py`)
Control M0 GREEN. **14 of 14 mutants RED:** flag parsed but not passed to the labeller (K1); header check ignoring `kin_tb` (K2); merge test using the constant (K3); built-levers key missing (K4); blend weight using the constant (K5); cache identity always writing 1.0 (K6); upper range unchecked (K7); lower range unchecked (K12); non-kin label not refused (K8); trainer not setting `bank.kin_tb` (K9); trainer not passing `kin_tb` to the identity (K10); a header without `kin_tb` read as the asked value (K11); gate not comparing declared and built (K13: first run STILL GREEN, an equivalent mutant because no test built a swap at one value and declared another; fixed by adding the cross-declared `dvb(..., dag=...)` assertions to the reachability test, then RED); header `kin_tb` never written (K14). M0 and K13 were run on the final test file; K1 to K12 and K14 ran on the file one assertion block shorter (the K13 addition) and were not re-run.

## Not run / limits
Full-bank labelling at 2.0 (the census is 60,000 records of one shard); the whole suite; G-SUITE; the pod. `v8f_t1_eval.py`, `launch_gate.py`, `stack/tanitad/models/__init__.py`, `E:/Projects/TanitAD/stack` untouched.
