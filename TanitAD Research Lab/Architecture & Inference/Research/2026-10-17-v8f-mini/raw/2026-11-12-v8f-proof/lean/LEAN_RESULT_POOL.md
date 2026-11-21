# LEAN_RESULT_POOL: group P (what a pooled state keeps and loses) checked by the Lean 4 kernel, in general form
Evidence class: PROVED (Lean 4 kernel with Mathlib; `lake build` rc 0 on the pod, 2026-11-13 18:48:26Z; project rebuilt with `Hier.lean` 19:15:02Z, rc 0). A theorem about a model class, not a measurement and not a driving-performance number. STRATEGIC family: PAUSED (PI item 25); no metric family is measured here.
## What was checked
* Source `lean/src/Pool.lean` (sha256 2852afef70f945ed), 35 theorems, no `sorry`; axiom audit `lean/src/Audit.lean` (sha256 fc68f835d3ac9cff): 31 of 31 audited theorems depend on `propext`, `Classical.choice`, `Quot.sound` only (the standard axioms), read from the build log of the project `lean/src/lake_build.log` (sha256 e05125a62a652469).
* Toolchain `leanprover/lean4:v4.36.0-rc2` (`lean/src/lean-toolchain`), Mathlib rev `9d581dfff2e41de989072ea63abf3daf69311520` (`lean/src/lake-manifest.json`, sha256 fdfaeb46c02498cc). Build of the project (this file and `Hier.lean`): 18,190 jobs, 0 errors; 25 warnings on this file, all of them style notes (line length, `show` tactic, module system, docstring delimiter).
* The first check of the file (before the style clean-up) also ended with 0 errors; no proof was changed after a failed check.
## What the Lean version adds to the solver version (group P of `fv/FV_SPEC_V5_POOL.md`: 4 cells, one real feature per cell)
| solver statement | Lean theorem(s) | generality in Lean |
|---|---|---|
| VC-P1 pair bound | `pair_bound`, `pair_bound_vec (vector targets)` | real and inner-product-space targets |
| VC-P2 bound attained | `pair_bound_attained` |  |
| VC-P3 moving the object leaves the mean state equal | `mean_move`, `pool_move_inside` | any number of cells, vector features, any pooled set and weight |
| VC-P4, VC-P5 position unreadable from the mean state | `location_zero_skill_mean`, `location_skill_le_zero_mean`, `zero_skill_of_equal_states`, `variance_is_floor` | n cells, EVERY decoder (any function of the state), EVERY real target of the position |
| VC-P6 grid state reads the position, for every background | `grid_scene_diff`, `grid_separates` | n cells, vector features |
| VC-P7 the mean state keeps the count | `mean_sceneList`, `mean_depends_on_count_only`, `mean_count_diff`, `mean_separates_counts` | any list of object positions, repeats allowed |
| VC-P8 motion invisible to the mean state, visible to the grid state | `mean_blind_to_motion`, `grid_sees_motion_enter`, `grid_sees_motion_leave` | n cells, vector features |
| VC-P9 pooling at an intermediate scale | `blockState_move_inside`, `blockState_move_across`, `location_zero_skill_blocks` | any family of pooled sets (blocks need not be equal or disjoint), any weights, every decoder of the whole block state |
| M-P1, M-P4, M-P5 (stronger bounds are false) | `stronger_pair_bound_false`, `two_position_floor_tight`, `four_position_floor_tight` | counterexamples checked by the kernel |
| M-P6 (no read without a live adapter direction) | `grid_blind_without_live_adapter` |  |
| M-P7 (the mean state does not separate counts of unequal strength) | `mean_confuses_two_with_double` |  |
| M-P8, M-P9 | `mean_blind_to_motion`, `pool_move_inside` | the mutated statements are the negations of these |
| control c3 of SPATIAL stage 1 (cell permutation) | `sum_perm_invariant`, `readout_perm_invariant` | ANY per-cell map (LayerNorm followed by a linear layer is one) plus an additive per-cell term |

The step the solver version left as a paper step ("to n cells and vector features by linearity") is now machine-checked. New beyond that step: every decoder and every target (the solver had one decoded value and an indicator target); pooling over an arbitrary set of cells with an arbitrary weight, which makes the mean state, a block state and the grid state three cases of one statement; object counts as lists; the permutation statement for an arbitrary per-cell map.
## Setting and its relation to v8F-mini
* Cells `i`, input feature `x i` in a real vector space `E`; per-cell adapter `cell A b x i = A (x i) + b i` with `A` linear and `b i` a constant per cell (bias and position term); a scene is a background `u` plus an object of strength `s` in one cell. Pooled state: `pool w K A b x = w • Σ_{i in K} cell A b x i`; mean state: all cells, weight 1/n; grid state: one value per cell.
* v8F-mini's adapter is LayerNorm then Linear per token (not linear in the input). Two readings keep the theorems exact for it: (1) take `E` as the space of adapted features and `A` as the identity (an object changes the adapted feature of its cell by `s`); (2) `sum_perm_invariant` and `readout_perm_invariant` need no linearity at all: for ANY per-cell map and an additive per-cell term, reordering the cell contents leaves the pooled sum, and every function of it (the projection `WMAux.proj` and every decoder after it), equal.
* Not covered: attention pooling or any state that is not a function of a per-cell sum; learned dynamics; anything about training. The theorems say what such a state CAN hold, not what a trained model does hold (that is measured: LATVIS, FULLRES f5, SPATIAL stage 0b to 0d, the trained-state reads of note N8).
## Main statements in words
1. `location_zero_skill_mean`: for every number of cells, background, object strength, linear adapter, position term, decoder and target of the object position, the squared error of the decoder summed over the positions is at least the summed squared deviation of the target from its mean. Skill 1 - risk / variance is at most 0 (`location_skill_le_zero_mean`).
2. `location_zero_skill_blocks`: the same for every state made of pooled blocks, over any set of positions that no block separates.
3. `pool_move_across`, `grid_scene_diff`: a block that separates two positions reads the move by `w • A s`, independent of the background; the grid state reads every move.
4. `mean_sceneList`: with m objects of strength s anywhere, mean state = background mean + m • (1/n) • A s: it depends on the count only.
5. `mean_blind_to_motion`: an object moving between cells on a constant background changes nothing in the mean state, so a copy predicts the next mean state exactly; a learned predictor of the mean state has nothing to imagine about that object.
