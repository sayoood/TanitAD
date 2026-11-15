# PREREG_V8F_PROOF_ADD38 : LATVIS-2, the layered sheet of the imagination and a token-level read of what the rollout transports

Card of the v8F proof programme, MOTION line. Asked for by the PI on 2026-11-15 ("Yes, build it"). Frozen before the tool exists and before any token-level number or new picture. Tier of every read: T0 (world-model and perception diagnostics on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this card. STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED, rows H-V8F-PROOF-16, 28, 39, 68)

* A pixel decoder recovers the scene gist from the level-1 state (held-out skill 0.36 for the mean state, 0.56 to 0.61 for the grid state, 0.78 for the token grid the model sees), and the decoded imagined state at 2.0 s is no closer to the true future frame than the decoded present state at both seeds of every learned rollout. The headroom in pixels is small (0.0004 to 0.0043 of mean squared error) and carried by static background.
* The supervised lead head reads the lead gap 2.0 s ahead better from imagined states than from present states (6.14 m against 6.83 m). So content is transported that the pixel decode does not show.
* One read-out at one level of abstraction cannot say what the imagination transports. This card fixes a ladder of read-outs from the same states, the windows they are shown on, and one new measurement at the level between pixels and objects.

## 2. Windows of the sheets (fixed by rule from labels and logged signals; no model output is used)

* The 12 windows of `latvis/PANEL_WINDOWS.json` (6 with the largest speed change over 2.0 s, 6 with the smallest lead gap; chosen before any decoder on 2026-11-13).
* 6 lead windows: among the windows with a label gap at t between 5 and 50 m, an ego speed at t of at least 5 m/s and a label gap at t + 2.0 s, at most one per episode, the 3 with the most negative and the 3 with the most positive change of the label gap over 2.0 s; ties by the lower window index.

## 3. Levels of a sheet (each for the present state, the imagined state at 2.0 s and the true state at 2.0 s)

* L0, frames: frame t and frame t + 2.0 s (camera content, private).
* L1, pixels: the decoded tiles of the earlier pixel reads where they exist (private).
* L2, tokens: the 4x10 token grid the model sees, shown as colours (three principal components fitted on training-fold frames); from the states it is read by a linear map fitted out of fold; beside it the change (t + 2.0 s minus t) of the truth and of the imagination.
* L3, cells: for each of the 40 cells of the grid state, the squared distance of the present state to the true future state minus that of the imagined state (above 0: the imagination moved this cell toward the future).
* L4, object: a top-down sketch with the label gap at t and at t + 2.0 s and the lead head's read from the present, the imagined (1.0 s and 2.0 s) and the true future states (arms with a lead head).
* L5, what-if: the imagined gap at 1.0 s and 2.0 s under offsets of the ego acceleration of -2, -1, 0 and +1 m/s2, with the line the kinematics give.
* A level that does not exist for a window (no lead, no decoded tile, no frame on the machine) is drawn as an empty tile with the reason. Arms: G_s0 and RR_s0 for L2 and L3; LS_s0 for L4 and L5, and the arms of card ADD37 beside it when they exist.

## 4. The token-level read (new numbers)

Arms G and RR, seeds 0 and 1; the 1,948 windows of the r-b read; rollout of the stack from the state of frame t, 10 steps, logged actions (the rollout of card ADD36). Targets: the pooled token grid of a frame, each cell reduced to 64 principal components fitted on the training-fold frames. A ridge map from the level-1 state to the 40 x 64 targets is fitted on the frames of four folds of episodes and applied to the fifth.
* Per window and cell, the squared error against the true token grid of frame t + 2.0 s of the map applied to (i) the imagined state, (ii) the present state, (iii) the true future state.
* Statistic D = error (ii) minus error (i) (above 0: the imagination is closer to the future than the copy); headroom = error (ii) minus error (iii). On all cells (reported) and on the moving cells (named): per window the 10 of 40 cells whose true tokens changed most between t and t + 2.0 s. Paired episode-cluster bootstrap per seed, two-level (seeds, then episodes) for the pooled value, B 2,000, seed 0. Beside: the share of the headroom reached; D per row of the grid (sky, far, near, hood).
* Names (each needs both seeds of the arm): `TOKENS-IMAGINED` if D on the moving cells is above 0 at both seeds and the two-level interval lies above 0; else `TOKENS-FLAT`. `RR-IMAGINES-TOKENS-BETTER` if D of RR minus D of G, paired on the same windows, is above 0 at both seeds with the two-level interval above 0; else nothing is named for the pair.

## 5. Controls (all must pass for an arm; else none of its names is given)

* k1: with the states permuted across frames the held-out skill of the token map is at most 0.02.
* k2: the skill on the training folds is above the held-out skill, and the held-out skill of the true present state on the present tokens is above 0.
* k3: with the present state given as the imagined state D is exactly 0.
* k4 (known positive): with the true future state given as the imagined state D equals the headroom, and its two-level interval lies above 0.
* k5: the rollout of the tool equals the banked rollout of the r-b read of the same checkpoint (`pred_2s` of the banked file) to 1e-4.
* k6: the windows, folds and episodes equal those of the r-b read (same ids, same count).

## 6. Sheets and what may leave the machine

Sheets with camera frames or decoded pixels are private: they stay on the dev box and are shown to the PI only. A second form of each sheet holds the levels L2 to L5 only (colours of a 4x10 grid of derived values, numbers of the head, no camera picture) and may be banked. No picture is evidence for a claim; the numbers of section 4 and of card ADD36 are.

## 7. What follows (committed before the data)

* `TOKENS-IMAGINED` for RR: the sheets of the action-consistency arms (card ADD37) are drawn with the same levels, and the token level is the read for any later claim that a rollout transports scene layout.
* `TOKENS-FLAT` for both arms: what the rollout transports is readable at the object level only (the supervised lead head), not as scene layout at 4x10 tokens; the lever is a finer token grid, which needs the camera frames encoded again and is a decision of the PI (ESTIMATED 2 to 4 GPU-h and 30 to 150 GB of pod volume).

## 8. Limits

Linear maps show what is linearly readable. The token grid is itself a pooled read of the picture (4x10 cells). Two seeds per arm. One corpus. 139 episodes. The windows of the sheets are 18 of 1,948 and are illustrations.
