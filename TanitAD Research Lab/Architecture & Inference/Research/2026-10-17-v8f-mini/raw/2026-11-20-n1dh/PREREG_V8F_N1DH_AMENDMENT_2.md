# PREREG_V8F_N1DH, Amendment 2 (FROZEN 2026-11-21 08:37Z by the MM, before any N1DH training step, dump or number)

The card `PREREG_V8F_N1DH.md` (sha256 f1df05bc) and Amendment 1 (25d6b42a) stand. This amendment changes only the commit of record, by one test file.

## What happened (MEASURED, pod)
* The N1DH chain on f2e91e33 started at 07:50:56Z, right after ROLLHIST DONE. It held the GPU lock and passed tree and inputs.
* It stopped at G-SUITE (`ZZ N1DH FAIL ZZ step gsuite`, about 08:26Z). No training step ran.
* `gsuite_result_n1dh.json` (base bcf3ba3f, new f2e91e33) read **1 regression**, **0 new-test failures** and **0 missing**:
  `tests.test_v8f_n1dc::test_every_stack_script_that_builds_the_v8f_config_from_a_checkpoint_is_in_the_loader_list`.
* The cause is the same census as Amendment 1. `stack/scripts/n1dh_klim.py` builds `V8FMiniConfig(**ck["cfg"])`. Amendment 1 added it to the `LOADERS` list of `test_v8f_n1dh.py` only.
  * The older `test_v8f_n1dc.py` carries the same census with its own `LOADERS`, and it now finds a loader it does not list.
  * That test passes at base and fails at new: a regression, caught as designed.
* Before the Amendment 1 launch, the MM ran the new and modified test files on the pod. This file was neither, so it was not run.

## The change
* **Commit 1f217bf015ebd6d69b888e424a2675970ff72075** on branch `agent/v8f-n1dh-20261120`, parent f2e91e33. It is pushed, and `ls-remote` is equal.
* It changes exactly one file: `stack/tests/test_v8f_n1dc.py` gains `"n1dh_klim.py"` in `LOADERS`. The new sha256 is 646e0608..., git blob 02dc9af0.
* On the pod tree copy, both census files were run whole (`test_v8f_n1dc.py` + `test_v8f_n1dh.py`):
  * as committed: **1 failed**, 107 passed (it reproduces the regression);
  * fixed: **109 passed**, 1 skipped.
* **The code of record is now 1f217bf0.** Model, trainer, checker and evaluators are byte-identical to f2e91e33 and cc8c98a1.

## Pod repoint (`F:/TanitAD-scratch/n1dc/build/n1dh_b4fix/repoint_1f21.sh`, ZZ REPOINT 1f217bf0 OK ZZ)
* The tree gets the new test file.
* `all_blobs.txt` changes that one line, and blobcheck reads 3,439 of 3,439.
* `n1dh_tree.sha256` gains the file, so the delta list is now 15.
* `COMMIT` changes in `n1dh_arm.sh` and `n1dh_gsuite.sh`.
* In `n1dh_gsuite.sh`, `NFILES` goes from 14 to 15, because the tree now differs from bcf3ba3f in 15 entries (checked: 15).
* The chain banner changes.
* Copies are kept as `*.bf2e9`. The failed result and chain log are renamed with the `_f2e91e33` suffix. Nothing is deleted.
* Nothing in the arms, steps, dumps, reads or bars changes.

## Lesson
A NEW script can turn an EXISTING census test red. Before a pod launch, run every test file that greps `stack/scripts`, not only the new and modified test files. G-SUITE catches it, but only after 35 minutes.
