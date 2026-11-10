# PREREG_V8F_STOPBAL, AMENDMENT 1: the code of record gains one test-only commit (Master Mind, 2026-11-10 ~19:00, before any STOPBAL run)

Status: written before any STOPBAL run and before any STOPBAL data. The frozen card (`PREREG_V8F_STOPBAL.md`, sha256 in `PREREG_V8F_STOPBAL.sha256`) is not edited; this file amends one line of it, the code of record. Arms, argvs, read, rules, PRIMARY, gates, outcomes and controls are unchanged.

## Trigger (MEASURED, control 5)
* The wider suite on an export of `2bca1c52` against the pristine parent `5514a684` (`suite/SUITE_DIFF.txt`, 2026-11-10 18:5x) named **one new failure**: `stack/tests/test_v8f_ego_dagger_compose.py::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit`. The launcher refused, as the card requires (`suite/STOPBAL_NOT_LAUNCHED.txt`, exit 4).
* Cause (`suite/pytest_patched2.log`, line 117): the test's last check in its `dag` loop, *"levers only GAIN"* a fixed set of key names. The STOPBAL commit adds two keys to `built_levers` (`goal_pos_weight_stop`, `lon_class_weight_stop`, `v8f_mini.py` built_levers), which that set did not list. It is a golden **key-set** test, not a behaviour change.
* Every check before it in the same loop passed for the `dag` family: every logged loss and key, the final weights and optimiser state (sha256), the counters, the summary.json and checkpoint key sets. That is direct evidence that, with both flags absent, the code of record trains DAG-OACC-style bit for bit like the tip it was built on. The `orc` and `m2b` families were not reached because the loop stopped at `dag`; the fixed test checks all three.

## Change
* New commit on `agent/v8f-stopbal-20261110`, parent `2bca1c52`, changing **only** `stack/tests/test_v8f_ego_dagger_compose.py` (tree diff: one path, MEASURED with `git diff-tree`): the two STOPBAL keys join the allowed gained set, and a new assertion requires both to read 1.0 when the flags are absent, for each of the three families.
* `stack/scripts/v8f_mini_train.py` (blob `c96ef807`, sha256 `1ae4ef3b…`, the value `make_argvs_stopbal.py` pins) and `stack/tanitad/models/v8f_mini.py` (blob `1cb245e4`) are byte-identical in both commits (MEASURED with `git ls-tree`). The box-2 ship list (`recover_ft_launch/export_list.txt`, 66 paths) contains no test file, so the shipped tree is the same bytes at either commit.
* **Code of record: the new commit** (sha in `AMENDMENT_1_CODE.sha256`), and `STOPBAL_COMMIT` is set to it at launch. `2bca1c52` stays the commit that introduced the lever.

## Control 5, re-read
* The parent side of the first suite comparison **ran out of disk** (F: at 0 bytes free): 633 errors (`ios_base badbit`, `unexpected pos`, `No space left on device`, basetemp creation refused), so its "pre-existing" list is not a valid comparator: 82 of its 166 shared ids were errors on the parent side. The parent side is rerun on freed disk before launch, and the patched side's fixed test is rerun at the new commit. The launcher starts the chain only if that rerun names no new failure.
