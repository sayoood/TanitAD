# HANDOFF -- refcv9 in-model R9-PF (Architecture FlyWheel, 2026-10-26)

State: design memo, implementation, unit tests (28 pass), flags-off bit-identity vs pristine (MEASURED, True + controls),
10-step trainer-step smoke (ALL_PASS) and DRAFT prereg are DONE. Only the FULL-SUITE count was in flight.

* Full suite run 3 (final tree): log `F:/refcv9_inmodel/pytest_full_modified3.log`; introduced-vs-pristine IDs in `fail_ids_INTRODUCED.txt` (pristine baseline: `fail_ids_pristine.txt`, 289 IDs, all environment: files outside `stack/taniteval/tools`).
  were absent): log `F:/refcv9_inmodel/pytest_full_modified2.log`, last line `PYTEST_RC=<n>`.
* If any test FAILS there: re-run that test on the PRISTINE copy (`F:/refcv9_inmodel/pristine/stack`, extract `taniteval`
  and `tools` beside it first with `git archive refs/heads/agent/arch-inf-20260803 taniteval tools`) to classify
  pre-existing vs introduced. Env: `CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 PYTHONPATH=<copy>/stack`,
  python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8`, `--basetemp` on F:.
* Then re-run the flags-off digest on the final tree (`tools/bitid_digest.py`, see its docstring) and compare with
  `bitid_pristine.json`; write the sha256 manifest of every file in `F:/refcv9_inmodel/` named in the hand-back.
* Patch to land: `F:/refcv9_inmodel/REFCV9_INMODEL.patch` (7 files; 3 replaced lines, the rest additive).

* DONE 2026-10-26: run 3 = 1 introduced failure (G-LIVE rule for w_r9_*), FIXED in scripts/launch_gate.py; affected files re-run 336 passed / 2 failed, both pre-existing on pristine.
