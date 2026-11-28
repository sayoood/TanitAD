# PREREG_V8F_ROLLHIST, Amendment 3 (FROZEN 2026-11-21 02:08Z by the MM, before any RH or B0 bank, run or number)

Card `PREREG_V8F_ROLLHIST.md` (781eea87), Amendment 1 (87928487) and Amendment 2 (15004dc9) all stand. This amendment only changes the commit of record, by one test file, and changes nothing else.

## What happened (MEASURED)
* The chain on 43319227 began at 01:21:55Z. It stopped at G-SUITE at 01:57Z with `ZZ ROLLHIST FAIL ZZ step gsuite`. No bank, run or GPU step happened: `/root/rollhist/bank` and `runs/` are empty.
* `gsuite_result_rh.json` had **0 regressions** and **0 missing**, against base cdbff971 (14,851 tests) and new (14,910 tests). It had **1 new-test failure**: `tests.test_v8f_t1_readout::test_promoted_functions_equal_the_study_functions_bit_for_bit_golden_digest`.
* That test pins a single golden digest of the study functions' outputs, `1c42d85f...`, which B1 computed on the dev box (Windows, numpy 2.5.1). On the pod (Linux, numpy 2.5.4), the promoted functions gave `6b93f03a...`.

## The discriminating check (MEASURED, pod, 02:00Z)
* `F:/TanitAD-scratch/rollhist/build/b1/study_vs_promoted.py` (sha256 58ed71929514e97e...) execs the STUDY file's own functions from `integ10k/v8f_t1_eval_reader_rule_rzg.py`. That file is sha256 93879db8c0233867..., identical to its blob at 7b26a8ac.
* The check runs them beside the promoted functions of `/root/rollhist/tree_cdb` (43319227).
* Result: **arrays compared 2,880, bit-identical True**. The study digest on the pod is `6b93f03a...`, equal to what the promoted functions gave. The promoted code matches the study code on the pod, and the golden digest is platform-specific (float64 cos/sin and SIMD last bits).

## The change
* **Commit 97fe0a56767320dbc5f4b823d67fb07a3209a745**, branch `agent/v8f-rollhist-cdb-20261120`, parent 43319227, pushed, with `ls-remote` equal.
* It changes exactly one file, `stack/tests/test_v8f_t1_readout.py` (new sha256 f5c69eae97e9718e...). `GOLDEN_SHA256` becomes a set of the study digests on the two platforms, and the assertion becomes `in`.
* The test file passes 14 of 14 on the pod and 14 of 14 on the dev box.
* Blob list: 3,407 entries, which differ from 43319227's in that one line only.
* **The code of record is now 97fe0a56.** The trainer, the model, the bank generator, the readout functions and `v8f_t1_eval_hb.py` are byte-identical to 43319227.
* The read package's tree (`read/tree`, 43319227) differs from 97fe0a56 only in that test file, which the read does not run. It stays as it is.

## Pod repoint (all before the relaunch)
* `tree_cdb` gets the new test file, sha-checked.
* `all_blobs_97fe0a56.txt` replaces `all_blobs_43319227.txt` in `rh_chain.sh` and `rh_gsuite.sh`.
* `COMMIT` changes in `rh_env.sh` and `rh_gate_record.py`.
* Copies of the old files are kept as `*.b43319`.
* The done markers move to `done_43319227`, so every step re-runs.
* The failed chain log moves to `rh_chain_43319227.log`, and the G-SUITE result moves to `gsuite_result_rh_43319227.json`.
* G-SUITE re-runs in full; its PASS rule is unchanged.
