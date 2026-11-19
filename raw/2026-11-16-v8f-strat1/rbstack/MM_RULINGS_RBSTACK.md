# MM rulings on RBSTACK (made 2026-11-19 ~00:40Z, before any RBSTACK number exists)

At the time of this ruling `out/` holds no `stage_*.json`; the run log's last lines are the first probe fit (kin3 seed 0, outer fold 0, epoch 2 of 6).

## R1. The deciding contrast is EDV - ED_STR (EDV - ED is printed beside)
* Build deviation 2 (HANDOFF_RBSTACK.md): probe features exist only for RB-JOINT's strided windows, so EDV and EDV_SHUF are fit on strided labelled fold-train windows, while the card's ED reference (the RBJK tree) is fit on the dense fold-train windows.
* EDV - ED would then compare trees fit on different row sets (about half the rows for EDV), a bias against EDV that has nothing to do with vision.
* The card defines EDV as "the ED tree with the probe's 3 class probabilities appended". ED_STR is that same ED tree fit on exactly EDV's rows, so EDV - ED_STR isolates the appended probabilities, which is the question in section 3.
* Therefore: the section-3 reading (STACK-CARRIES / WEAK / SILENT, both probe seeds, lower reading if they disagree) is read on **EDV - ED_STR**. K-SHUF is read on **EDV_SHUF - ED_STR** (< 0.0383 at both seeds). EDV - ED and EDV_SHUF - ED are reported beside, with no bar.
* K-REPRO stays on the dense ED and S (they must reproduce RESULT_RBJK exactly).
* rbstack.py's `reading_of` uses EDV - ED. The MM applies R1 from the stage JSONs' `paired` blocks (both contrasts are computed by the same `RD.paired` on the same strided scored windows); the code is not changed while the job runs.

## R2. Deviations 1, 3, 4 accepted
* 1: X8 re-extracted under the same window rule and meta (the 5-way beside run checks the replay against RB-JOINT's RAW8 fold files).
* 3: inner fit (j, k) and (k, j) share one fit set, so 10 inner fits per seed serve all 20; K-NEST reads the folds each probe was given.
* 4: stage B NOT-YET, with the reason in RESULT_RBSTACK.md (no held-out features on the dev box; the card forbids a pod pull for this check).
