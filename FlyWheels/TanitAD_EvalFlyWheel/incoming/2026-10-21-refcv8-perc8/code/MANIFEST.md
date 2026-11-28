Everything below is **unstaged and uncommitted** (the orchestrator lands it). Dev-box root `C:/Users/Admin/refcv8_pf/perc8/` (not on G:/D:); Thor root `/home/nvidia/r8ns/perc8/`.

| artifact | where |
|---|---|
| `PERC8_EVAL.json` (merged: provenance, controls, box read, map read, refit, re-score, WiSE control) | dev box `C:/Users/Admin/refcv8_pf/perc8/PERC8_EVAL.json` |
| `RESULT.md` (this file) | dev box `C:/Users/Admin/refcv8_pf/perc8/RESULT.md` |
| code (all scripts, the verbatim originals in `code/orig/`, the two Thor remap JSONs, the chain / waiter / smoke launchers, `READING.md`, `MANIFEST.md`) | dev box `.../perc8/code/` and Thor `.../perc8/code/` (the Thor copy is the one that ran) |
| pass A, EMA-48,852, S2 (4,754 windows): `perc.json`, `perc_det.pkl` (per-window box/agent packs), `perc_map_table.npz` (per-window map statistics), log | dev box `.../raw/A_nosync/`; Thor `.../out/A_nosync/` |
| TRAIN-DIAG pass (1,112 windows / 139 train episodes): `train_diag.json`, `train_diag.packs.pkl`, refit outputs `refcv8_det_presence_gates_train.json`, `refcv8_det_nms_train.json`, logs incl. the FAILED first attempt | dev box `.../raw/T_nosync/`; Thor `.../out/T_nosync/` (+ `train_diag.acc.pt`, 6 MB, Thor only) |
| WiSE DEC control pass (297 windows) + comparison `wise_control.json` | dev box `.../raw/W_dec_sub/`, `.../raw/wise_control.json`; Thor `.../out/W_dec_sub/` |
| box read: baseline (all AP/AUROC cells, controls C1-C3) `box_read_declared.json`; variants (refit gate / NMS) `box_read_final.json`; per-window rows `rows/*.npz` (regenerable with `code/perc8_rows.py`) | dev box `.../raw/` |
| map read `map_read.json`; GT control `gt_control.json`; refcv7 re-score control `rescore_refcv7_control.json`; tables `tables/*.md`; self-tests `selftest/` | dev box `.../raw/` |
| inputs: `stage/s2_windows.json` (md5 a3a78dba889e70aa69a05271559794ad), `s2_every16.json`, `s2_every4.json`, refcv7's shipped gate / NMS JSONs | dev box `.../stage/`; Thor `.../inputs/` |
| md5 evidence for the eval cache (dev box 139, other agent's Thor 139, my Thor sample 4) | dev box `.../raw/md5_*.txt` |

Not run: the corrected-clock (pose-synced) read -- the launcher supports it (`status/RUN_A_SYNC`, `status/RUN_SYNC_SUB`); blocker is wall clock only.
Modified originals: `code/r8_repoint.py` (one line: the remap file name now comes from env `R8_REMAP_FILE`, default `refcv8_eval_remap_thor.json`; verbatim original in `code/orig/`), `code/run_diag.py` (the map-loss wrapper forwards extra keywords: the launch tree's `map_hires_loss_row` gained `class_thresholds` after the wrapper was written; the first TRAIN-DIAG attempt died on it). Nothing under the repo tree was edited.

Script provenance: `raw/box_read_declared.json` = `code/perc8_box_read.py` (full AP/AUROC grid + controls C1-C3, 1,000 draws); `raw/box_read_final.json` = `code/perc8_box_read2.py --light` (a copy of the former with a variants-only mode: AP/mAP@2 m raw + NMS and the census at the refit gate / NMS; 1,000 draws, same seed 0 draw stream, controls skipped because the baseline run already holds them).
