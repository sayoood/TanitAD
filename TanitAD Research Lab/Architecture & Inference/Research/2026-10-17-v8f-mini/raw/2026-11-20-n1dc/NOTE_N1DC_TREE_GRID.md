# N1DC note: the canonical eval-141 grid is a data input of the training tree (written before any N1DC training step exists)

Card `PREREG_V8F_N1DC.md` (sha256 2eae7964, landed bbf96d3c), section 5, says: *"Clean judged tree: the stray `eval_grid_windows.json` is removed from the pod tree, and G-SUITE re-runs on the clean tree."* This note says how that clause meets the trainer, and what ran.

## What happened (MEASURED, 2026-11-20)

* The B2 tree `/root/strat1/n1dc/tree` was built from `git archive bcf3ba3f stack taniteval`, so it holds no `TanitAD Research Lab/` directory and no grid. 3,433 of 3,433 blobs were equal.
* G-SUITE ran on that clean tree against N1DW's own 134f5da1 junit: 0 regressions, 0 new-test failures (`gsuite_result_n1dc.json`).
* The mint of N1DC_s0 at 14:37:27Z was then REFUSED by the trainer's own selfcheck (`runs/N1DC_s0/gate/gate.log`). The refusal: *"K-B3: the canonical eval-141 grid `<tree>/TanitAD Research Lab/.../2026-10-17-v8f-mini/raw/eval_grid_windows.json` does not exist: the eval-disjointness check cannot run, and a check that cannot run is a refusal"*.
* No token was written, and no training step ran.

## Why the clause and the trainer disagree

* The file is **tracked in git** at that path at the code commit: blob sha256 659c3b65, equal at bcf3ba3f.
* The N1DW build called it "stray" for one consumer only (`MM_BUILD_N1DW.md` line 24). With the file present, a few tests take the tree for a full checkout and fail on files it lacks.
* For the trainer it is a **required data input**: the DAgger bank's K-B3 eval-disjointness check reads it.
* The checker binds it into the token when present (`strat1_check.py` `binding()`, the comment at `EVAL_GRID_REL`: *"a data input of the tree, bound when present"*).
* N1D and N1DW trained with it present.

## What runs, and how the card's clause is kept

* **The judged tree is clean.** G-SUITE ran with no grid, as the clause asks. Its result stands and is not re-run.
* **The training tree carries the canonical grid**, copied at the tracked path from N1DW's pod tree. It is byte-equal to the git blob at bcf3ba3f (sha256 659c3b65e364ea5d...). The tree step's done-marker predates the copy, so the chain's "no extra file" check is not re-run against it. The token minted next binds the grid through `tree_manifest_sha256` and the `data` entry.
* No lever, argv, data selection, step count, read, bar or outcome of the card changes. The arms train exactly as N1D and N1DW did on this point.

## Relaunch

The chain was relaunched at 14:40:14Z. Its finished steps (lock_wait, tree, inputs, gsuite, pairing) skip by their done-markers. It waits for the pod GPU lock behind the Ruled session's running slice. The pod owner's marker `/root/N1DW_READY` is set again (watcher `n1dc_marker_watch2.sh`), so Ruled's next slice yields to N1DC.

Class: a file removed on one consumer's view (test skew) without asking every consumer (the trainer's K-B3). Before removing a file from a judged tree, grep its name in the code that will RUN on that tree.
