# HANDOFF: TARGET levers EGOV and FUTTOK (level-1 predictor), 2026-11-14

State: DONE on the CPU. Nothing trained, no git, no pod. Folder `F:/TanitAD-scratch/v8proof/spatial/target/`. Base = `spatial/motion/tree_patch` (MOTION result).
Apply: `cd <tree root>; patch -p1 --binary < TARGET.patch` on a tree holding the MOTION files; or bank `patched/` under `stack/`. Pins: `PINS.txt` (5 changed files, 4 unchanged pinned). Definitions and every decision: `TARGET_DEFS.md`.
Changed: `tanitad/models/hjepa.py`, `tanitad/models/v8f_mini.py`, `scripts/v8f_mini_train.py`, `tanitad/data/v8f_mini_data.py` (new batch key `wm_speed`), new `tests/test_v8f_target_levers.py`.

Flags (defaults = the base tree, bit for bit):
* PE : `--wm-l1-act speed`            selfcheck: `levers.wm_l1_act=speed levers.wm_l1_futtok_k=0 levers.w_wm_futtok=0`
* PF : `--wm-l1-futtok-k 10 --w-wm-futtok 1.0`   selfcheck: `levers.wm_l1_act=acc_yaw levers.wm_l1_futtok_k=10 levers.w_wm_futtok=1`
* PEF: both                           selfcheck: `levers.wm_l1_act=speed levers.wm_l1_futtok_k=10 levers.w_wm_futtok=1`
* all arms: P's argv (`--wm-state grid --wm-cell-dim 16 --wm-l1-pred cell`) plus `--wm-l1-diag on`.
* ⚠ `gate_extra.py` compares the string form with whole floats as ints: write `levers.w_wm_futtok=1`, NOT `=1.0` (measured: REFUSED `built '1' != expected '1.0'`, `checks/t6_selfcheck_firstrun_1.0.log`).

Train-log keys: FUTTOK on: `wm_futtok` (required by G-LIVE); with diag on also `wm_futtok_copy`, `wm_pred_l1`, `wm_copy_l1`. EGOV adds no key. Checkpoint cfg gains `wm_l1_act / wm_l1_futtok_k / w_wm_futtok` only when not default.
Read tools: `from tanitad.models.v8f_mini import wm_frame_speed`; `act3 = wm.scale_actions(act, wm_frame_speed(batch))` (width 3 on a speed model; `scale_actions` raises a clear error without speed). `wm_speed` = logged ego speed [m/s] of each cached frame (cache `v`).

Results (all read from files in `checks/`, CPU):
* t1 identity, 200 steps, G / P / PR argv: keys, init tensors, all logged fields, final weights, ckpt cfg keys: max abs difference 0.0, PASS (`t1_identity.json`).
* t2 EGOV (real sizes, flat and cell): speed-column gradient 8.6e-4 / 3.3e-3 (non-zero); controls EXACTLY 0.0 (zero speed; levels 2, 3); permuted or shuffled speed moves the level-1 loss (~1e-5 .. 1e-4, small); lever off ignores speed. PASS (`t2345_checks.json`).
* t3 FUTTOK alone: tok_head, every level-1 predictor group, action embedding, state projection non-zero; levels 2 and 3 exactly 0.0; control (total) reaches them. PASS.
* t4 rolled states in the term vs `HJEPAStack.rollout`: max abs 5.1e-7 flat, 3.9e-7 cell (bar 1e-6; not 0.0 because the loss embeds 15 actions at once and `rollout` embeds the K-step slice: `t4_note.json`). PASS.
* t5 float64: total(W) - total(0) = W x term to 2e-14; line deleted -> red ("= 0.0 but W x futtok/L1 = 0.7197"), flat and cell. PASS.
* t6 selfcheck PE / PF / PEF / default: declared-vs-built PASS, G-LIVE pass, right expectations rc 0, wrong ones rc 1. PASS.
* t7 cost, tiny model CPU, s/step best (mean ratio over 3 reps): P 0.67; PE 0.70 (x0.88, noise); PF 3.48 (x4.5; best x5.2); PEF 3.70 (x4.7; best x5.5). Same order as PR of MOTION (x5.2); the rollout dominates, the token head adds ~0.
* t8 diag on vs off, 40 steps: every weight difference 0.0 for PE, PF, PEF. PASS.
* Tests: 32 new + the 55 of MOTION/CELLPRED/SPATIAL = 87 passed, 0 failed (`tests.log`).

Known limits: no real data, no pod-size run, `v8f_t1_eval.py` / read tools untouched (e1, e2 of the card belong there). PE re-draws the WM init (action embedding 32x3), the main model init is P's. IDM (wm b) also predicts the speed channel under EGOV. FUTTOK needs cached `feats` (not `--encoder lora`). Scratch on C: (`C:/Users/Admin/v8proof_c/target/`) ~1.5 GB, deletes are refused here; safe to remove once banked.
