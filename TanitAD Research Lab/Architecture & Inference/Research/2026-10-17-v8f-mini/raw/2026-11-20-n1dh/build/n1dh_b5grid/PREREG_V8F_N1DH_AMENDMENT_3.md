# PREREG_V8F_N1DH, Amendment 3 (FROZEN 2026-11-21 09:28Z by the MM, before any N1DH training step, dump or number)

The card `PREREG_V8F_N1DH.md` (sha256 f1df05bc), Amendment 1 (25d6b42a) and Amendment 2 (0f78a055) stand. This amendment changes only the pod tree, by one data file of the commit of record. The code of record stays **1f217bf0**.

## What happened (MEASURED, pod)
* The relaunched chain on 1f217bf0 passed tree, inputs, G-SUITE and pairing. G-SUITE read 0 regressions, 0 new-test failures and 0 missing over 15,476 tests (`gsuite_result_n1dh.json`).
* At 09:13:14Z the N1DH_s0 mint refused before any training step:
  * chain log: `ZZ ARM N1DH_s0 FAIL ZZ mint refused: REFUSED: the trainer's --selfcheck-only exited rc=1 and did not write selfcheck.json`;
  * `runs/N1DH_s0/gate/gate.log` (kept as `gate_kb3.log`): `ZZDVB-REFUSEDZZ ... K-B3: the canonical eval-141 grid .../2026-10-17-v8f-mini/raw/eval_grid_windows.json does not exist: the eval-disjointness check cannot run, and a check that cannot run is a refusal`.
* The cause: N1DH trains on a dagger bank (`--dagger-bank /root/strat1/inputs/dagger_bank/s0`, `--dagger-p 0.1`). For a dagger-bank arm the trainer's K-B3 check reads the repo-relative eval-141 grid. The N1DH pod tree left that file out, and its `chk_tree` refused any `TanitAD Research Lab` path.
  * The MM's earlier static check said N1DH needs no grid. That was wrong: it did not follow the `--dagger-bank` path.
  * This is the same defect as ROLLHIST Amendment 4, in a second tree.

## The change
* The grid is copied from `/root/strat1/n1dc/tree` into `/root/strat1/n1dh/tree`. It is **git blob c489b553** of 1f217bf0 (checked on the copy), 21,050 bytes, sha256 659c3b65....
* `all_blobs.txt` gains that one line (3,440 lines). Blobcheck reads 3,440 of 3,440, bad 0. The tree holds 3,440 files.
* `n1dh_chain.sh`, `chk_tree`: the check "no `TanitAD Research Lab` path" becomes "exactly one file under `TanitAD Research Lab`" (the grid). The file-count comment and the chain banner change. That is 3 lines in all.
* `done/tree.done` is moved aside to `tree.done_bgrid`, so the tree step re-runs under the new check on relaunch.
* Copies are kept as `*.bgrid`. The failed chain log becomes `n1dh_chain_1f217bf0_kb3.log`, and the failed mint log becomes `runs/mint_N1DH_s0_kb3.log`. Nothing is deleted.
* G-SUITE is not re-run. The change adds a file that the base tree (`/root/strat1/n1dc/tree`) already holds, and it touches no file under `stack/` or `taniteval/`, so the 15-entry delta vs bcf3ba3f is unchanged.
* Nothing in the arms, argv, steps, dumps, reads or bars changes.
* Script: `F:/TanitAD-scratch/n1dc/build/n1dh_b5grid/fix_grid_n1dh.sh` (`ZZ GRID FIX OK ZZ`).

## Dry mint before the relaunch
* `F:/TanitAD-scratch/n1dc/build/n1dh_b5grid/dry_mint_n1dh.sh` ran the arm's own mint command for N1DH_s0 by hand, under the GPU queue lock, 09:22:48Z to 09:28:04Z: **rc 0, `TOKEN-WRITTEN N1DH_s0`**.
* `runs/N1DH_s0/gate/checks.json`, read in full: G-SUITE, frozen_argvs_sha256, G-HYG, G-CLOCK, G-DVB, G-LIVE, G-EVAL and G-CKPT all **pass** (G-CKPT round trip bit-equal). The token binds `eval_grid_windows.json` and commit 1f217bf0.
* The token stays in place, so the chain's arm skips its mint for s0 (it mints only when no token exists). N1DH_s1 mints in the chain on the same tree.

## Lesson
A tree check that refuses a path can hide a file the trainer needs. Build a pod tree from the trainer's actual read list, following every flag the arm sets (here `--dagger-bank`), and mint the arm by hand before the launch.
