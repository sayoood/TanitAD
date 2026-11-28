# HANDOFF 2026-10-18 -- the full-suite baseline-vs-mine comparison for the yaw-rate hold (D-PI-YAWRATE-1)

Result note: `RESULT_YAWRATE.md`. Hold: `C:/Users/Admin/qland/hold_yawrate/` (13 mods-rw, 2 mods, 19 stage, all lander checks OK). Nothing committed or pushed.

## Done (MEASURED, CPU, `CUDA_VISIBLE_DEVICES=-1`, tree = tip `3766dcd` + the hold, `C:/Users/Admin/yawfix/tree`)
* `test_yaw_rate_ruling` + `test_ra1_hjepa` + `test_v8f_mini` + `test_hjepa`: **145 passed** (50 s).
* The other reverse dependents (`test_ddv2_refc_chain`, `test_launch_gate`, `test_launch_gate_v8f`, `test_p10_pose_prior`, `test_p10_prior`, `test_refcv8_conditioning`, `test_residual_prior`): 324 passed, 1 skipped, **3 failed**, all in `test_launch_gate.py`: `test_G_SUITE_the_pinned_env_list_is_content_bound_by_the_profile`, `test_G_MAP_OVERFIT_protocol_literals_are_the_A18_spec_itself`, `test_A19_the_map_spec_is_the_A18_spec_minus_its_must_fail_rows`. They compare a sha256 of a spec file with a pin; the hold touches none of those files or `launch_gate.py`. For the env-failures JSON the extracted file equals the tip blob once CR is stripped (extracted 46,635 bytes with CRLF, blob 45,648): a line-ending artifact of the extraction, expected on a baseline too. NOT yet confirmed on a baseline run -- that is step 2.

## NOT done: the full suite (10,602 tests; collection alone takes 2 m 43 s)
## TRAP found: a colon-separated `PYTHONPATH` on Windows python is ignored
`run_suite.sh` (colon form) gave 127 collection errors on BOTH trees, because `tanitad` then imported from the editable install under `G:\Meine Ablage\...` (the old mount), not from the tree. Use `;` and prove it: `python -c "import tanitad; print(tanitad.__file__)"` must print the tree.

## Next steps (run detached; poll with a script, do not wait alive)
1. mine:
   `cd C:/Users/Admin/yawfix/tree`; `export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="C:/Users/Admin/yawfix/tree/stack;C:/Users/Admin/yawfix/tree/taniteval"`;
   `python -m pytest -q --tb=no -p no:cacheprovider -rfE stack/tests --continue-on-collection-errors > C:/Users/Admin/yawfix/out/suite_mine2.log 2>&1`
2. baseline (C: has 5.5 GB free, the tree is 7.9 GB, so do NOT extract a second tree): in the same tree, copy `git show 3766dcd:<path>` over each of the 13 mods-rw + 2 mods paths in the hold, move the 19 stage files aside (`stack/tests/test_yaw_rate_ruling.py` first), and rerun step 1 into `suite_base2.log`. `git --git-dir=C:/Users/Admin/tanitad-push/.git`. Restore the hold's files afterwards (copy from `hold_yawrate/mods-rw`, `mods`, `stage`).
3. compare: `grep -E '^(FAILED|ERROR)' suite_base2.log | sort > b.txt`, same for mine, `diff b.txt m.txt`. Mine must add no id; the only expected differences are the new tests (`test_yaw_rate_ruling.py`, the `["yaw","steer"]` parametrisation of the `test_v8f_mini.py` e2e).
4. Then delete `C:/Users/Admin/yawfix/tree` (7.9 GB, my own scratch) to free C:.

## Single-location artifact
`native_tracks.pt` (16,591,688 B, sha256 `1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7`) exists only at `C:/Users/Admin/tanitad-data/yawfix/native_tracks.pt` (copy in `C:/Users/Admin/yawfix/out/`). It is not in git; rebuild with `stack/scripts/build_native_tracks.py`.
