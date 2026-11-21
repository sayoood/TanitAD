# BUILD_ISOFIX -- T-ISO (c) recursion fix (2026-11-17)

## Defect (MEASURED, run2/logs/iso_c_s0.log)
`build/wh_iso.py::_record_run` (wrapped, line 37-41 before the fix): on the first real batch of a pooled-head run, `wrapped` called `M.ctx_pool_isolation_problems(self, batch)` while `M.V8FMini.forward_infer`
was still `wrapped`. That probe calls `model.forward_infer` twice (`tree/stack/tanitad/models/v8f_mini.py:1817,1822`), re-entering `wrapped` with `probe` still empty -> the probe started again without end
-> `RecursionError` inside `model.eval()` (v8f_mini.py:1812). Even without the recursion, the probe's two calls (second one with a perturbed head) would have been appended to `rec` and shifted the batch alignment of
the identity comparison. T-VIS (`assert_ctx_pool_vision_only`, v8f_mini.py:1794) was checked: it runs `model._ctx_pool` through `_vision_only_run` and never calls `forward_infer`; it sits inside the same unwrapped
window anyway. Semantics of T-ISO (b)/(c)/(d) are unchanged.

## Fix (build/wh_iso.py, 8 lines added, 4 moved)
```
-            probe["T-ISO-b"] = M.ctx_pool_isolation_problems(self, batch)
-            try: probe["T-VIS"] = ...
+            M.V8FMini.forward_infer = orig        # probe runs on the UNWRAPPED method: no re-entry, nothing recorded
+            try:
+                probe["T-ISO-b"] = M.ctx_pool_isolation_problems(self, batch)
+                try: probe["T-VIS"] = ...  (unchanged)
+            finally:
+                M.V8FMini.forward_infer = wrapped
```
Pre-fix copy: `tmp/wh_iso.before_isofix.py`.

## Tests (build/test_wh_study.py, +2; pre-fix copy tmp/test_wh_study.before_isofix.py)
* `test_record_run_probe_runs_on_the_unwrapped_method_without_recursion_or_extra_records`: pooled-head model, fake T0 evaluator calling `forward_infer` on 3 batches, probe stand-ins that call `forward_infer`
  twice like the real T-ISO (b). Asserts one probe call each, `rec` holds exactly the 3 real batches in order (values 0,1,2), 5 total `forward_infer` calls, wrapper removed afterwards. PASS.
* `test_mutation_the_probe_called_on_the_wrapped_method_reads_red`: the unwrap line is replaced by `pass` in a copy of `_record_run`; it raises `RecursionError` (RED as required). PASS (the test expects the error).
Study set run serially: **33 passed in 155.80 s** (was 31; 2 new). Run: CUDA_VISIBLE_DEVICES=-1, TMP/basetemp on F:, one job at a time, commit-gated.

## Build hash
* old `build_sha256` baad5d3ca655c67ae42aeb0c15f2aea222d057504cd22aea9af662d04e1f217a (H_s0, H_s1 tokens, untouched, minted 2026-11-16 23:07:26 / 23:07:42Z; both fits ran under it)
* new `build_sha256` 63df633e22f0bee093a7c4133367a300bb7dc89a86492334bf63fdbfe1762167 (`wh_check.build_sha()` over BUILD_FILES; only wh_iso.py changed)

## Tokens re-minted with `wh_check.py mint ARM --runs run2 --device cuda --family text|v9` (CPU only, as the original gate; every minted rc 0)
RA_s0, RA_s1, R1_s0, R1_s1, H_s0_rep, K4p_s0, K4p_s1, FEAT_s0, FEAT_s1 (family text); C0A_s0, C0A_s1 (family v9). The gate evidence that does not read the build files was reused as it stands:
`run2/gate/gsuite_result.json`, `geval_text.json`, `geval_v9.json` (they bind the tree manifest ac1bcd9e, which did not change). G-HYG, G-CLOCK, G-CKPT, G-DVB, G-LIVE re-ran inside mint.
All 11 new tokens carry the new build hash; no GPU step was needed. H_s0 and H_s1 tokens were not touched. Old tokens+checks of the 11 arms: `run2/gate/before_isofix/<ARM>/`. Mint logs: `run2/gate/mint_isofix_<ARM>.log`.

## Relaunch (prepared, NOT run; MM only)
```
rm "F:/TanitAD-scratch/strat1/whead/run2/state/FAILED"
powershell -NoProfile -File F:\TanitAD-scratch\strat1\whead\launch_wh_wmi.ps1 -Job job_wh2.cmd
```
`launch_wh.sh` first re-runs the 33 study tests and checks all 13 tokens exist, then `wh_chain.py all --runs run2`. Skipped by the done markers: fit_H_s0, t0_H_s0, kid_H_s0, fit_H_s1, t0_H_s1, kid_H_s1, iso_a_s0
(H_s0 and H_s1 are logged "arm done"). It resumes at `iso_c_s0` (T-ISO (c) seed 0, re-run from scratch, outputs in run2/iso overwritten), then iso_d_s0, iso_a/c/d for seed 1, then RA_s0 onward. The fit steps take the GPU lock
(devbox_gpu.lock) as before. Wait for `ZZ WH CHAIN DONE ZZ` / `ZZ WH CHAIN FAIL ZZ` in run2/wh_launch.log.

## Deletions
F:/TanitAD-scratch/strat1/whead/tmp/bt_isofix2 (pytest basetemp, 7.6 MB) deleted; bt_isofix1 was not created (deselected-run folder absent). Kept: the two `*.before_isofix.py` copies and `run2/gate/before_isofix/` (delete once the MM has landed the fix and the chain has verified).

## Manifest
| artifact | where |
|---|---|
| fixed wh_iso.py | F:/TanitAD-scratch/strat1/whead/build/wh_iso.py (only here; not staged, repo staging is the MM's) |
| tests | F:/TanitAD-scratch/strat1/whead/build/test_wh_study.py |
| new tokens | F:/TanitAD-scratch/strat1/whead/run2/gate/<ARM>/wh_gate_token.json |
| this note | F:/TanitAD-scratch/strat1/whead/BUILD_ISOFIX.md |
