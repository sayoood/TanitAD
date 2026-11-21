# CELLPRED_DEFS: what was built for PREREG_V8F_PROOF_ADD16 (2026-11-14)

Card: `F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD16.md` (frozen, not edited). Base: tree of stage 1 of SPATIAL (`latvis_gw/tree` plus the stage-1 trainer and test).
CPU only, nothing trained. Trees: `tree_base` (pins verified: v8f_mini.py 687a4103..., v8f_beta.py d30fbdc3..., v8f_mini_train.py 3ea2a796..., v8f_t1_eval.py 61e140ed... taken from `wm/tree`, byte-identical to the pod pin) and `tree_patch`.
The trainer, the test and `v8f_t1_eval.py` were absent from `latvis_gw/tree`; the trainer and test came from `stage1/patched/`, the T1 eval from `wm/tree/stack/scripts/` (sha256 equal to the pin). Nothing was taken from E:.

## What was built, with the hooks (paths relative to `tree_patch/stack/`)

| piece | where |
|---|---|
| `_SpatialBlock` (pre-LN block, not causal, attention over the cells of ONE step plus one action token) | `tanitad/models/hjepa.py:420` |
| `CellCausalPredictor` (same interface as `CausalPredictor`) | `tanitad/models/hjepa.py:448` |
| `LevelConfig.pred_kind / pred_cells / pred_cell_dim` (default `flat`, 0, 0 = today) | `tanitad/models/hjepa.py:540-542` |
| `HJEPAStack` builds the cell predictor for a level with `pred_kind == "cell"`, refuses other kinds | `tanitad/models/hjepa.py:619-630` |
| `WM_L1_PRED_MODES`, `WM_CELL_PRED_WIDTH` (128), `L1PRED_CFG_FIELDS` | `tanitad/models/v8f_mini.py:161-163` |
| config field `wm_l1_pred` and its width rule | `v8f_mini.py:214`, `:261-264` |
| validation (cell needs grid state and wm on) | `v8f_mini.py:285-290` |
| `WMAux` builds level 1 with the cell predictor (width 128, depth 2, 4 heads, cells = n_tok, cell width = wm_cell_dim) | `v8f_mini.py:719-720` |
| `WMAux.probe_pred` (reads the BUILT predictor) | `v8f_mini.py:758` |
| `built_levers`: `wm_l1_pred`, `wm_l1_pred_params`, `wm_l1_pred_params_flat_ref`, `wm_l1_pred_params_le_flat`, `wm_l1_pred_cross_cell` (`wm_l1_pred_width` already existed and reads `in_proj.weight.shape[0]`, now 128 for the cell predictor) | `v8f_mini.py:1266-1273` |
| flag `--wm-l1-pred {flat,cell}` | `scripts/v8f_mini_train.py:139` |
| `FLAG_CHECKS` entry (`lever:wm_l1_pred`) | `v8f_mini_train.py:72` |
| `make_config` passes it | `v8f_mini_train.py:345` |
| `INIT_CKPT_MATCH` entry and absent key = flat | `v8f_mini_train.py:686`, `:706` |
| `bad_wm_state`: declared vs built both directions, width 128, c2 and c4 refusals, wm-off refusal | `v8f_mini_train.py:1027-1042` |
| checkpoint cfg records `wm_l1_pred` only when `cell` | `v8f_mini_train.py:2911` |
| tests | `tests/test_v8f_cell_predictor.py` |

## The cell predictor

State `[B, L, 640]` is read cell-major (cell `c` owns numbers `16c..16c+15`, the layout of `WMAux.z1`). One shared `Linear(16, 128)` embeds a cell; a learned cell position (40 x 128) and time position (3 x 128) are added. Two layers; each layer = `_SpatialBlock` over the 40 cells plus one action token (`Linear(32, 128)` of the embedded action of that step), then `_AdaLNBlock` (causal over time, conditioned on the 32-number action embedding as in the flat predictor) on each cell's own sequence. Output: `LayerNorm` then a shared `128 -> 128 -> 16` map per cell, reshaped to 640. No skip. Levels 2 and 3, the encoders and every loss are untouched; `HJEPAStack._pred_loss` and `rollout` run unchanged (tested).

## Parameter counts (c2, MEASURED, selfcheck of the synthetic arm shapes, `checks/selfchecks/`)

* flat level-1 predictor of the grid arm G (width 256, depth 2, 4 heads): **2,074,496**
* cell level-1 predictor of arm P (width 128, depth 2, 4 heads): **873,488** (42.1 % of G; the card's bound "at or below" holds with room)
* both are in the selfcheck of P: `wm_l1_pred_params` and `wm_l1_pred_params_flat_ref`; `wm_l1_pred_params_le_flat` must be true or the gate refuses.

## Reading choices (each: it moves no threshold, bar, outcome name or committed next step of the card)

1. **Spatial block carries no AdaLN modulation.** The card puts the action in the spatial attention as a token and conditions only the temporal block "in the way the flat predictor is". The spatial block is a plain pre-LN block. No bar involved.
2. **The action token is re-appended at every layer from one shared embedding; its updated value is dropped.** The card says "one token that carries the embedded action of that step". Moves nothing in section 5.
3. **No separate positional term for the action token.** Its own linear layer (with bias) distinguishes it from the cells. Moves nothing.
4. **Hidden width of the MLPs is 4 x 128 = 512** (the flat predictor's rule, `ff = 4 x width`); the card gives no number. The count stays under the c2 bound; the bound is unchanged.
5. **Time-position table has `context` = 3 rows** like the flat predictor, and the rollout at planning time uses 1 to 3 steps as for the flat one. Moves nothing.
6. **`wm_l1_pred` kind is read from the BUILT module:** class `CellCausalPredictor` with a per-cell embedding of `latent_dim / n_cells` inputs reads `cell`; the exact class `CausalPredictor` with a dense embedding of `latent_dim` inputs reads `flat`; anything else reads `other:<class>` and is refused. The behavioural probe (`wm_l1_pred_cross_cell`: one cell of the input moves the other cells of the output) is reported and the gate requires it for `cell`; a dense flat map also moves all cells, so the probe alone does not separate the kinds, the class and the embedding shape do. This is how c4 is read ("the probe reads the BUILT module, not the flag"). No bar involved.
7. **`wm_l1_pred_params_flat_ref`** is the parameter count of the `CausalPredictor` (width 128 for a 128-wide state, else 256, depth 2, 4 heads) that this state width gets in G; it makes c2 a gate refusal, not only a test. For `flat` arms it equals the arm's own count.
8. **The three new selfcheck keys** are `wm_l1_pred`, `wm_l1_pred_width` (existed since stage 1) and `wm_l1_pred_params`; four more bookkeeping keys are added (`..._flat_ref`, `..._le_flat`, `..._cross_cell`). Flat and mean arms also carry them (bookkeeping only; state dict, checkpoint cfg and loss unchanged, c1).
9. **c5 gradient reading:** the WM term alone is back-propagated (`w_wm x terms["wm"]`, as in the stage-1 test) through `V8FMini.loss` on a batch of the synthetic world; groups are the module's blocks (`in_proj`, `cell_pos`, `time_pos`, `act_tok`, each spatial and temporal block, `final_ln`, `out`). A per-tensor check is not used because the key bias of attention has an exact analytic gradient of 0.
10. **c1 argv shapes:** `--arm M2b --grid pool4x10 --kin-mode yaw --wm-state grid --wm-cell-dim 16` and `--arm M2b` (mean default), synthetic world, batch 8, d 32, seed 0, 200 steps, CPU, 2 threads. The pod-size model (d 256) is not run on the CPU.
11. **Relative cost** is measured on that tiny synthetic model (batch 8) and is a CPU ratio of a whole training step, not a pod number.

## Files that the chain pins and that are unchanged

`v8f_beta.py`: the planner refuses every grid state already (`refuse_grid_state`, lines 190, 223, 246), and a cell predictor exists only with a grid state, so it is refused there as well; no guard added.
`v8f_t1_eval.py`: `cmd_run` builds `V8FMiniConfig(**ck["cfg"])`, then `V8FMini`, then strict `load_state_dict`; the new cfg key is a dataclass field, so a cell checkpoint builds the cell predictor and an earlier checkpoint (key absent) builds flat. No change (tested: `test_checkpoint_cfg_keys_and_the_t1_builder_round_trip`).
