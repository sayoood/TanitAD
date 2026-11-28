# PREREG_V8F_N1DREP Amendment 1: the navrbd tree's file count admits the eval-141 grid

Card `PREREG_V8F_N1DREP.md` (sha256 0c641197) stands. This amendment moves no arm, argv, cell, floor, estimator, tier or outcome. It is written before any N1DREP number exists: the replicate has not trained.

## What happened (MEASURED, pod, 2026-11-22)
* The chain took `/root/gpu_queue.lock` at 07:20:15Z and stopped at its tree check with `ZZ N1DREP FAIL ZZ navrbd tree file count`, before render, mint or training. The lock was released when the chain exited.
* `navrbd/all_blobs.txt` has 3,425 rows; the tree has 3,426 files outside `__pycache__`. Blob check 3,425 of 3,425 OK; `navrbd_tree.sha256` OK. The one file not in the list is `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json`, git blob c489b553fe9355cf50e6007bdf395bd4a07a6e9a, placed 2026-11-18 21:02:08Z. N1D_s0's gate token was written at 21:05:37Z, so N1D_s0 minted and trained with this same file present.
* The card's section 2 says the tree checks "as N1D's chain did (... file count)". N1D's chain (`navrbd_chain.sh`) ran no file count; the count was added by this card's chain.

## The change
* The count check admits exactly that one file: tree files = `all_blobs.txt` rows + 1, `all_blobs.txt` does not list the grid, the tree holds exactly one file under `TanitAD Research Lab/`, and that file's git blob is c489b553. Any other extra file still stops the chain.
* K-REP-CODE is unchanged and still compares the new gate token's `tree_manifest_sha256` with N1D_s0's.
