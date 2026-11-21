# HANDOFF: CELLPRED model lever (PREREG_V8F_PROOF_ADD16), 2026-11-14

State: DONE on the CPU, nothing trained, no git, no pod. Folder `F:/TanitAD-scratch/v8proof/spatial/cellpred/`.
Apply: `CELLPRED.patch` on a tree with the stage-1 files (hjepa.py sha256 368a01f2..., v8f_mini 687a4103..., trainer 3ea2a796...): `patch -p1 --binary < CELLPRED.patch` (re-applied to base copies: byte-identical to `patched/`, cmp, 4 files). Bank `patched/` under `stack/`.
Pins: `PINS.txt`. Pod chain file: `ARMS_CELLPRED.txt` (5 PIN lines for `/workspace/v8proof/tree_cellpred`, arms P_s0, P_s1). `gate_extra.py` needs no change (string compare; the new keys pass, the deliberate wm_l1_pred regression is REFUSED, rc 1).

Results (details in `CELLPRED_DEFS.md`, `tests.log`, `checks/`):
* c1 identity: max abs difference 0.0 for the grid-G argv and the mean default (200 steps; keys, init, every logged field, final weights, ckpt cfg keys).
* c2: level-1 predictor 873,488 (cell) vs 2,074,496 (flat G).
* c3, c4, c5, c6, c7 pass as tests and checks. The new file has 11 tests, all pass.
* 6 tests of the older `test_ra1_hjepa.py` fail identically on tree_base (missing queue and ridge-probe scripts in the partial export), not related.
* CPU cost, tiny synthetic model, batch 8: cell 0.69-0.76 s/step vs flat 0.20 s/step, ratio 3.4 (best of 2) / 3.5 (mean). Expect the arm to run slower than the card's 66 min estimate on the pod; check the chain's time limits.

Not run: real-data selfcheck and pod-size model (lead's gate step on the pod); `v8f_beta.py` and `v8f_t1_eval.py` unchanged.

Scratch (nothing deleted, per brief): `tmp/` about 1.4 GB (pytest basetemps, c1 and timing runs; re-creatable by `checks/`), `tree_base/`, `tree_patch/` 39 MB each. Extra copies in the trees for old tests (not in the patch): `test_hjepa.py`, `test_predictor_failfast.py`, `test_ra1_hjepa.py`, `test_v8f_beta.py`, `v8f_mini_eval.py`, `ra1_hjepa_eval.py`, `ra1_hjepa_train.py` (from `wm/tree`). Footprint 1.7 GB, above the 300 MB ask.
