# MM fixes to the Option B read chain (2026-11-18, after FTRSP_s0 banked; no outcome read)

FTRSP_s0 retrained on box 2, passed the post check and banked at 07:05Z. The read chain then stopped at its evidence step. Three defects were found and fixed. Each fix has a test that goes RED on the old code. None changes an outcome bar or reads an outcome metric.

## 1. A second copy of the C5 count (the cause of the stop)

* `readchain_tools.py evidence` had its own `--expect-adam` default of **141**.
* The post-check ruling (`build_box2/MM_RULINGS_POSTCHECK.md` on the Option B branch) set C5 to **129 of 129** (133 `wm.*` parameters − 4 `wm.proj`), but it was applied only to the box-2 checker.
* The stop, MEASURED (kept as `c5fix/FTRSP_s0.evidence.c5_141.log` and `c5fix/ZZ_STOP.c5_expect141_20261118.txt`): "C5: 129 of 129 optimiser tensors have Adam state, wanted 141"; C2 and C3 held.
* **Fix:** the default is `None`, and the count resolves to the checker's own constants, `C.N_WM_PARAMS − C.N_PROJ_TENSORS`, so the two sites cannot drift apart again.
* **Test** `test_c5_default_is_the_checkers_129_and_the_old_141_is_refused_on_the_real_bank`:
  * the real FTRSP_s0 bank passes on the default (129 with state);
  * `--expect-adam 141` refuses it with rc 4 and names 141;
  * on the pre-fix tool the test is RED (`c5fix/test_tools_c5mut.log`).
  * That mutation swapped `readchain_tools.py` in place for about 3 s while the chain's planner ran. The planner does not call this tool; its next use is the delta step after the planner. The file was restored and compared byte-equal (`cmp`). The route mutation below was run in a copy folder instead.
* After the fix, FTRSP_s0's evidence holds on C2, C3 and C5.

## 2. `mk_fakebank.py` copied `argv.txt` unconditionally

* That copy was an MM change of the same night (d8feaabc). The unit-test fixture `fixture/FT` has no `argv.txt`, so 3 of 5 `test_tools.py` tests failed. They had not been rerun after the change; the dry run passed because its source run carries `argv.txt`.
* **Fix:** the copy happens only if the source has the file.

## 3. The route never called its tree-manifest check

* `optb_route.py`'s docstring said `tree_manifest.sha256` "re-checks it on every start". But `check_tree_manifest()` was called only by `test_route.py`. The same class as the binding rule on mechanisms that are built and tested but unreachable from their caller.
* The read tree had meanwhile taken 86da79cd's `hjepa.py` (the CUDA fix of `built_rsp`). MEASURED: all 1,865 `.py` files under the tree's `stack/` and `taniteval/` are blob-identical to 86da79cd.
* **Fix:**
  * the manifest is regenerated from the tree; only `stack/tanitad/models/hjepa.py` changed, and the 4c396aff manifest is kept beside as `tree_manifest.4c396aff.sha256`;
  * `main()` calls the check before the probe remap is installed.
* **Test** `test_main_refuses_a_tree_that_is_not_the_manifest_and_runs_on_the_real_one`:
  * a fake tree is refused with the manifest message;
  * the real tree runs.
  * On the pre-fix route the test is RED (`c5fix/test_route_manifest_mut.log`, run in a copy folder so the live chain's file was never touched).
* The FTRSP_s0 planner pass started before this edit. Its process loaded the pre-fix route. The tree it read is the one verified above.

Tests after all three fixes: `test_route.py` + `test_tools.py` 16 passed (`c5fix/test_route_tools_manifest.log`).
