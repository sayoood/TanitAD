# HANDOFF: SPATIAL lever patch (PREREG_V8F_PROOF_ADD12 stage 1), 2026-11-13

State: DONE, CPU only, nothing trained, nothing staged or committed (mirror read only). Folder: `F:/TanitAD-scratch/v8proof/spatial/stage1/`.
Base: mirror commit `3e264bbf`. Patch: `SPATIAL.patch` (4 files, +286 / -9 lines incl. a 168-line test). Full changed files (LF): `patched/tanitad/models/{v8f_mini,v8f_beta}.py`, `patched/scripts/v8f_mini_train.py`, `patched/tests/test_v8f_spatial_state.py`; bank them under `stack/` of the repo.
Apply: `git -c core.autocrlf=false apply SPATIAL.patch` (plain `git apply` on this box writes CRLF). Re-applied to a fresh export: byte-identical to `patched/` (cmp, 4 files).

## What changed
* `v8f_mini.py`: config fields `wm_state` (mean|grid), `wm_cell_dim` (16), validation in `__post_init__`, derived `wm_l1_dim` / `wm_l1_pred_width`; `WMAux` builds level 1 at `n_tok x cell_dim` (640) with predictor 256 for grid or `--wm-dim 640` mean, levels 2 and 3 fixed at 128; `z1` grid path = LayerNorm+Linear(d, cell_dim) on every cell, cell-major flatten; `probe_state` (behavioural read: output width and whether a cell swap moves it); `built_levers` adds `wm_state`, `wm_z1_width`, `wm_l1_dim`, `wm_l1_pred_width`, `wm_upper_dims`, `wm_cell_dim`, `wm_dim` (only when a WM is built); `SPATIAL_CFG_FIELDS` are written into a checkpoint cfg only for grid, so a mean checkpoint keeps the tip's cfg keys.
* `v8f_mini_train.py`: flags `--wm-state --wm-cell-dim --wm-dim`, `FLAG_CHECKS`, `make_config`, `bad_wm_state` inside `declared_vs_built` (reads the BUILT model), `INIT_CKPT_MATCH` (absent key = mean default).
* `v8f_beta.py`: `refuse_grid_state` in `collect_beta_inputs`, `goal_latents`, `plan_beta`.
* No change to `hjepa.py`: the stack already takes a level-1 width different from the upper levels (LevelEncoder in/out dims, IDM, SIGReg are per level width). No change to `v8f_t1_eval.py`.

## Decisions to know
* Levels 2 and 3 are fixed at 128 for every wm_dim. Before, `wm_dim` set all three levels, but no flag existed, so every argv had 128; at 128 nothing moves (c1).
* `wm_dim` / `wm_cell_dim` given with wm off are refused by the gate (declared, builds nothing).
* Mean default adds the new keys to `selfcheck.json` levers of WM arms (bookkeeping only; state dict and checkpoint cfg unchanged).

## Not run / gaps
* Selfcheck of the arm shape used `--synthetic --exclude 1:3` (6-clip world; `540:600` drops 0 and the gate refuses it). Real-data gate on the pod is the lead's step.
* `--help` of the trainer crashes on a `%` in an older help string in both trees (not touched).
* 14 existing tests skip in the partial export (queue, verdict, renderer files and v9 eval release absent); not a failure of either tree.
* c4 (participation ratio) is a read on trained states, not built here. Beta planner not ported (refuses, as asked).
* This agent's footprint peaked near 950 MB (two trees, synthetic worlds and checkpoints of the checks), above the 300 MB ask; all deleted.

## Deletions (literal paths, by this agent, all re-creatable: export of `3e264bbf` + SPATIAL.patch, or re-run of `checks/`)
| path | size |
|---|---|
| `F:/TanitAD-scratch/v8proof/spatial/stage1/tree_base` | 39,225 KB |
| `F:/TanitAD-scratch/v8proof/spatial/stage1/tree_patch` | 39,241 KB |
| `F:/TanitAD-scratch/v8proof/spatial/stage1/tmp` (synthetic worlds, c1 runs, fresh export) | 890,616 KB |

Re-running checks needs the two trees back: export as in the brief, plus `taniteval/taniteval` and `stack/scripts/v8f_mini_eval.py` for the old tests; scripts in `checks/` hard-code `.../stage1/tree_base|tree_patch`.
