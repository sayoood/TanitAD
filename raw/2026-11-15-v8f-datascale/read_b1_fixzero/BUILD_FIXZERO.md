# BUILD_FIXZERO -- the DSCALE10K block-1 `zero` route passes (instrument fix, no evaluator edited, no bar moved)

## Cause
`ds_run.py:20` mapped the `zero` variant to `--nav-withheld`, but ran it on the read tree (`F:/TanitAD-scratch/otband_read/tree`, cdbff971, `v8f_mini_eval.py` sha256 50c70ae8). That evaluator has no such flag (argparse rc 2, `unrecognized arguments`). Only the STRAT1 tree (`F:/TanitAD-scratch/strat1/read/tree`, git e81bde01, `v8f_mini_eval.py` 008f48b7) has it. `test_ds_read.py:249` asserted the argv string only.

A second fault sat behind the first: `ds_read.py route_arrays` requires `win_ci` / `win_t` in every route npz. Only the 008f48b7 evaluator writes them, so the 50c70ae8 `real` / `donor` files fail with `KeyError: win_ci` (first re-run, log line 123-134).

## Diff of the two trees (50c70ae8 vs 008f48b7; stack + taniteval)
- `taniteval/` : the three pinned files (ci.py, v8f_consistency.py, four_families.py) are byte-identical (sha256 equal).
- `v8f_mini_eval.py` (30 diff lines): adds `--nav-text-npz`, `--nav-text-sha256`, `--nav-withheld` (zeroes the five nav channels; refused with `--nav-shuffle`), refusals tied to `cfg.nav_source`, `ctx_logits` collection when `model.ctx_head` exists, and `win_ci`, `win_t`, `nav_variant`, `nav_source` written to the npz, `nav_variant` and `nav_source` to the JSON.
- `v8f_mini.py` (128 lines): additive STRAT1 context head (`ctx_heads`, `w_ctx`, `ctx_detach`, `nav_source` config fields, all default off; head built only when on).
- `v8f_mini_data.py` (82 lines): additive `ctx_label_arrays`, `nav_text_arrays`.
- Also differing, not imported by the route passes: `v8f_t1_eval.py`, `v8f_mini_eval_kindx.py`, `v8f_mini_train.py`, `v8f_t1_speedsub.py`.
All the imported differences are opt-in; K-EVALX below tests that they do not move the default path.

## Fix (instrument only)
- `ds_lib.py`: `TREE_ZERO`, `VARIANT_TREE` (real, donor -> read tree; zero, realx, donorx -> STRAT1 tree), `tree_of()`, `tree_problem(tree, key)`.
- `READ_PINS.json`: new `tree_zero_sha256` (7 files; mini_eval 008f48b7, model 776643b5, data fc348697 were checked equal to git blobs of e81bde01), `tree_zero_git`, `variant_tree`. The original `tree_sha256` is unchanged.
- `ds_run.py`: per-tree `env()`, `import_check(tree)`, `_t0_common(tree)`; `route_argv` picks the tree through `L.tree_of`. New variants `realx` and `donorx` (the `real` / `donor` argv again, on the STRAT1 tree). `preflight(route=True)` pins and import-checks the zero tree and dry-parses every variant's argv with its tree's own argparse (`argv_accepted`, `route_argv_problems`).
- Gate **K-EVALX** (`kevalx_compare`, `cmd_kevalx`, run at the end of `ds_run.py route`, writes `out/t0route/KEVALX.json`): for each stage-B checkpoint, `real` (50c70ae8) vs `realx` (008f48b7) and `donor` vs `donorx`: every array of the old npz equal with `np.array_equal` (atol 0, same dtype and shape, NaN equal) and every number of the result JSON equal. `ds_read.py` (`kevalx_problem`) writes NOT-YET for STRATEGIC unless KEVALX.json is a PASS over all 6 items.
- `ds_read.py route_arrays`: reads real from `realx`, donor from `donorx`, zero from `zero`, all 008f48b7 outputs, so the window check holds; K-EVALX is what ties them to the banked 50c70ae8 passes. The 50c70ae8 `real` / `donor` files are kept and are the K-EVALX reference.

## K-EVALX result (MEASURED, `out/t0route/KEVALX.json`)
`K-EVALX PASS {"B_s0.B real": true, "B_s0.B donor": true, "U_s0.B real": true, "U_s0.B donor": true, "B_rep.B real": true, "B_rep.B donor": true}`. Zero diffs on every array and JSON number; the only extra keys in the 008f48b7 output are the new ones (`win_ci`, `win_t`, `nav_variant`, `nav_source`). The zero passes are admissible.

## Tests and mutations
`test_ds_read.py`: 37 passed, 2 skipped (DS_SLOW). New tests:
- `test_variant_tree_map_is_explicit_and_pinned`.
- `test_every_route_variant_argv_is_accepted_by_its_evaluator`: dry parse through each tree's argparse. Mutation 1 (zero routed back to the read tree) turns it RED with `--nav-withheld` named; mutation 2 (zero mapped to `--nav-bogus-flag`) turns it RED.
- `test_kevalx_compare_exact_and_detects_one_ulp`.
- `test_reader_refuses_zero_passes_without_a_kevalx_pass`.

## Re-run outcome
Re-ran the launcher (it skipped every finished item): 3 zero + 3 realx + 3 donorx passes (rc 0, 33-107 s each), K-EVALX PASS, then the reader with `--kre-t1 --kre-t0`:
- K-RE_T1 pass (216 rows, 0 mismatches); K-RE_T0 pass (5 rows, 0 mismatches); K1 self-pair [0, 0] pass; reader problems 0.
- The STRATEGIC route read ran (all three stage-B arms).
- New log line: `ZZ DSCALE10K B1 READ DONE ZZ`.
- **The result file's outcome is `NOT-YET (the T0 mechanism evals are missing: the R_s0 precondition is unread)`.** This is a different, pre-existing instrument refusal, NOT the zero fault: `mechanism_stage_A_T0.reason` = `T0 INSTRUMENT REFUSED: DSUs0A_s0: the GT constraint block differs from M2b_s0's -- not the same labels on the same windows` (from `ds_code/code/v8f_wc_read.py:168` `present_error`, called by `ds_t0.mech_pair`). The mechanism evals themselves exist (16 files, rc 0). Not fixed here (out of scope). The next agent should check why the U_s0 stage-A oracle eval's GT constraint block differs from B_s0's (the `cons_pred` arrays do differ; `has` / `g_cons` are carried under other keys in these npz).

## Deliverable manifest
| artifact | where |
|---|---|
| ds_lib.py, ds_run.py, ds_read.py, test_ds_read.py, READ_PINS.json (edited; `*.before_fixzero` kept beside them) | `F:/TanitAD-scratch/datascale/read/` (one place only, scratch; not in the repo, not staged) |
| out/t0route/eval_DS-*-B_zero / _realx / _donorx .json .npz, KEVALX.json, out/RESULT_DSCALE10K_B1.json | `F:/TanitAD-scratch/datascale/read/out/` |
| patch_*.py (the edit scripts), waiter_out*.txt, this file | `F:/TanitAD-scratch/datascale/read/fixzero/` |

## Deletions (logged)
- `F:/TanitAD-scratch/datascale/read/fixzero/tmp` (own TMP and pytest basetemps, 76 KB): deleted.
- `F:/TanitAD-scratch/datascale/read/bt` (launcher's own pytest basetemp): deleted by the launcher.
