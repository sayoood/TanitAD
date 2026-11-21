# MM note: NAVRB2 committed, pod G-SUITE running, arms queued behind DSCALE block 3 (2026-11-17 ~22:30Z)

PI 2026-11-17: "add the channel for the roadabout". Card `../frozen/PREREG_V8F_NAVRB2.md`, sha256 e238ba8d, frozen before any mint.

## Dev-box regression (MEASURED, `regress/regress_summary.json`)

* Base 4dedac45 against the patched tree, on the 82 test files the patch can reach. A regression is a test that fails on new and passes on base.
* **1 regression:** `test_v8f_strat1::test_mutation_putting_the_head_in_decay_too_turns_the_optimiser_check_red`.
  * Cause: the frozen STRAT1 mutation edits the trainer text `and id(p) not in ctx_ids]\n    nodecay`. The patch had appended `and id(p) not in navrb_ids` after the ctx exclusion, so the anchor count was 0.
  * Fix (MM): the decay line lists the navrb exclusion first and the ctx exclusion last. The parameter sets are the same.
  * Re-run on the fixed tree: `test_v8f_strat1.py` + `test_v8f_navrb2.py`, **64 passed**.
* The other failing files fail identically on base (dev-box environment). The pod G-SUITE is the judge.

## Commit (plumbing, revert guard)

* `f95de67a4c0776adec4e79de33dd4eabbe15a0e9` on `agent/v8f-navrb2-20261117`, parent 4dedac45.
* 14 paths: 5 modified, 9 added, nothing else. `ls-remote` equals the local commit.
* `navrb2.patch` and `pod/navrb2_tree.sha256` were regenerated after the fix; only `v8f_mini_train.py` moved.

## Pod (LoRA pod)

* Tree `/root/strat1/navrb2/tree`: the 4dedac45 tree new8 plus the 14 files from a `core.autocrlf=false` archive of f95de67a. `sha256sum -c` passes 14 of 14.
* Inputs: `navtext_nav_rb.npz` and its report, `sha256sum -c` OK. `COMMIT` is set in `navrb2_gsuite.sh` and `navrb2_chain.sh` (`pod/`).
* G-SUITE started 22:21Z: CPU only, 765 files in 8 shards, two waves of 4, under the script's memory guard. It runs beside DSCALE block 3's training (128 CPUs, 71 GB container cap, 5 GB in use at the start).
* `queue_navrb2.sh` waits for block 3's end line, then takes `/root/gpu_queue.lock`. The chain refuses to start unless the G-SUITE ended DONE and the tree and input hashes check.

## Deletions (logged in `DELETIONS_MM_2026-11-17.tsv`)

* 52 test-temp folders under `F:/TanitAD-scratch/strat1/navrb2/tmp/`, 9.67 GB (pytest basetemps, the base-tree copy, the patch workspace).
* `C:/Users/Admin/actsens_launch/navrb2_kit` (about 1 MB, re-creatable from f95de67a).
