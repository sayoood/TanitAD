# MM note: `eval_grid_windows.json` in the judged pod trees is a commit blob and a required trainer input. It stays (2026-11-19)

**Question (coordinator review, item 2):**
* The N1DW build record (`../../MM_BUILD_N1DW.md`) traced the 5 skip-to-fail tests in the pod G-SUITE to one file present in both pod trees: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json`.
* The item asked to remove it before the next mint, after checking its consumers.

## What was measured (MEASURED, 2026-11-19 ~18:55Z)
- **It is in the commit.** At `134f5da1`, the path resolves to blob `c489b553fe9355cf50e6007bdf395bd4a07a6e9a`, 21,050 bytes. The pod copy in `/root/strat1/n1dw/tree` hashes to the same blob id.
  - It is absent from the 3,429-file partial export that built the pod tree, and was added beside it on 11-18 at 21:02Z.
- **It is read at launch.** In `stack/scripts/strat1_check.py` at 134f5da1:
  - line 485: `EVAL_GRID_REL` names this path;
  - lines 634 to 637: `binding()` hashes it into the token's data binding. The comment there says it is the trainer's `DAGGER_EVAL_GRID`, and that K-B3 refuses a launch without it.
- **Other readers in the pod trees:** `stack/tanitad/data/v8f_mini_data.py`, `build_navtext_nav_rb.py`, `ra1_hjepa_eval.py`, `chain_v8f_kscore.sh`, and `v8f_dagger_bank.py` (navrbd tree).

## Decision
- **The file stays.** Removing it would make the next mint bind a different data set, or make the trainer refuse to launch.
- **The 5 skip-to-fail tests** (`test_v8f_wc` x3, `test_refcv7_vis1` x2) are a property of a partial export that holds this one file. With the file present, the tests treat the tree as a full checkout and fail on files the export lacks.
  - They fail identically on the base tree with the same file (rerun 11-19 ~16:05Z: 5 failed, 24 passed, 3 skipped).
  - So the base-against-new G-SUITE comparison stays symmetric.
- **For the next mint:** run the base junit with this file present, as the 11-19 rerun did. Then the comparison needs no exception.
- **A durable fix,** if wanted, belongs in the export step: export this file with the tree. Neither the tree nor the tests change here.
