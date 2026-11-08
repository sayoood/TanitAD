# HANDOFF: band labels (stream A)

## State
Priorities 1 and 3 are done and tested. Priority 2: `v10_labels_eval139` is done and compared; **`v10_labels_train` is building** (detached, started 21:13; v9 took about 17 min on an idle box, this box is loaded).

## Check the running job
* Log `F:/strat_out/band_labels/v10/build_v10_train.log` (prints every 100 of 4,369 clips); done marker `F:/strat_out/band_labels/v10/v10.done`.
* `post_v10_train.cmd` (launched, polls every 30 s) waits for the marker, then writes `v10/compare_v9_v10_train.json`, `compare_v9_v10_train.log`, `tests_after_train.log` (both test files) and `v10/post.done`.
* If the build died (no `v10.done`, a Traceback in the log): rerun the third command of `run_v10_labels.cmd` (the train build) detached via WMI `cmd /c ... < NUL`.

## Exact next steps for whoever finishes
1. When `post.done` exists: read `tests_after_train.log` (expect 25 passed, 0 skipped) and run `python make_table.py F:/strat_out/band_labels/v10/compare_v9_v10_train.json > F:/strat_out/band_labels/v10/compare_v9_v10_train.md`.
2. Add the train row of the change table to `RESULT_BAND_LABELS.md` section 2 (eval139 numbers are already there).
3. The Master Mind lands from this folder; outputs on F: have no second copy, bank them (md5 in the manifests).

## Deliverable manifest
| artifact | where | copies |
|---|---|---|
| code, tests, SCHEMA.md, RESULT_BAND_LABELS.md, cmd launchers | `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/band_labels/` | one place (not staged, not committed) |
| `strategic_geom_{train,eval}.npz` + manifests (md5 inside) | `F:/strat_out/band_labels/` | one place |
| `strategic_tok_{train,eval}.npz` + manifests, `strategic_tok_census.json` | `F:/strat_out/band_labels/` | one place |
| `v10/v10_labels_eval139.npz` + manifest, `compare_v9_v10_eval139.{json,md}` | `F:/strat_out/band_labels/v10/` | one place |
| `v10/v10_labels_train.npz` + manifest (pending) | `F:/strat_out/band_labels/v10/` | one place |
| control rebuild of v9 eval139 (md5 equals the banked file) | `F:/TanitAD-scratch/hwm_bands/a/ctl/x/` | re-creatable; keep until the Master Mind has read RESULT |
| build and progress logs | `F:/strat_out/band_labels/*.log`, `F:/TanitAD-scratch/hwm_bands/a/prog_*.txt` | scratch |

## Deletions (own temp data)
| path | size |
|---|---|
| `F:/TanitAD-scratch/hwm_bands/a/x/strategic_geom_eval.npz` | 203,471 B |
| `F:/TanitAD-scratch/hwm_bands/a/x/strategic_geom_eval.manifest.json` | 1,723 B |
| `F:/TanitAD-scratch/hwm_bands/a/x/strategic_tok_eval.npz` | 4,553 B |
| `F:/TanitAD-scratch/hwm_bands/a/x/strategic_tok_eval.manifest.json` | 815 B |

Remaining scratch under `F:/TanitAD-scratch/hwm_bands/a/` is about 8 MB; delete `ctl` and `prog_*.txt` once nothing needs them.
