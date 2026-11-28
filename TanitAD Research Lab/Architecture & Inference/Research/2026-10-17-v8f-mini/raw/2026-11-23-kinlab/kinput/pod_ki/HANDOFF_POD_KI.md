# HANDOFF_POD_KI (MM, 2026-11-23): the KI kit is shipped and every pre-lock check passed; the launch waits on the PI's go

The build agent stopped twice on an API 529 (server overload) after the kit, its tests and the G-SUITE launch. The MM ran the dry mint and the dry walk, both with the agent's own scripts, which the MM read before running, and wrote this file and `PREP_KI.md`.

## State
* **Card:** `PREREG_V8F_KINPUT.md`, sha256 05e786a8, frozen 2026-11-23T08:57:56Z.
* **Code:** 4097a89e on `agent/v8f-rollhist-cdb-20261120`.
* **Kit:** `/root/rollhist/kit_ki/`, local copy in `F:/TanitAD-scratch/kinlab/pod_ki/` (`SHA256_KI.txt`).
* **Tests (dev box, agent):**
  * all PASS: `test_ki_argv`, `test_ki_checker` (28 passed), `test_ki_gate_record`, `test_ki_marker_watch`, `test_chain_dry_ki`;
  * all RED as required: the mutants of the checker, the gate record, the marker watch and the chain.
* **Tree:** `/root/rollhist/tree_ki`, 3,425 blobs verified at 4097a89e.
* **G-SUITE** (`/root/rollhist/gsuite_result_ki.json`):
  * Window 09:21:36Z to about 10:00Z.
  * **0 regressions, 0 new-test failures, 0 missing in new.** Base cdbff971: 14,851 tests. New: 15,088.
  * Beside, read by the MM from the JUnit files: 5 tests went from skipped to failure: `test_v8f_wc` ×3 and `test_refcv7_vis1` ×2.
  * Their cause is the tree shape. `tree_ki`, like `tree_rk`, holds a `TanitAD Research Lab` folder with 1 file, so the tests' stack-only / NO_TREE skip guards see a Research Lab tree and then miss its files. It is not the KINPUT code.
* **Dry mint** (CPU, 10:04-10:10Z, `dry/DRY_VALUES.json`), card section 4, per seed:
  * selected rows of the probe batch: **s0 46 of 64**, **s1 51 of 64**;
  * share of selected rows whose own `base` differs from the logged one: **1.0 (46/46)** and **1.0 (51/51)**;
  * under `logged`, `rows_changed.base` is 0, while p10f, kin3, p10_t and pose change on every selected row, as in RH;
  * `declared_vs_built` PASS; `control_p0_grad_diff` 0.0;
  * gate record mint and verify rc 0 at both seeds (scratch root); the study-side dry mint wrote its token (scratch root).
* **Dry walk:** `DRY=1 rh_ki_chain.sh` printed `ZZ ROLLHIST-KI DRY PASS ZZ`. Launch argv: 64 tokens, sha256 s0 `baf33625…`, s1 `2ad1ccae…`. It wrote no chain log and set no marker.

## Launch (MM only, after the PI's go with its cost)
* `pssh 'bash /root/rollhist/kit_ki/launch_ki.sh'`, wrapped in `timeout 30`; the ssh may hang on the setsid, so check the log.
* The watcher sets `/root/N1DW_READY` if it is absent, with its own token (`/root/rollhist/ki_marker_mine`), and clears it when the chain logs "GPU queue lock held to the end of the chain".
* Chain log `/root/rollhist/rh_ki_chain.log`; last line `ZZ ROLLHIST-KI DONE ZZ` or `ZZ ROLLHIST-KI FAIL ZZ <why>`.
* Message TanitADRuled before the launch.

## Cost
* **ESTIMATED** 0.5-0.6 pod-h of lock for the two arms, from the MEASURED RA 0.52 and RK 0.55 on the same banks and steps. The human label needs no label cache, and the selfchecks are about 50 s each on CPU.
* The card's return line is 0.9 pod-h.

## After the chain
The dev-box read is `F:/TanitAD-scratch/rollhist/read/rh_read_ki_retry.cmd` via WMI, read extension checked by the MM. Then the MM verdict per card section 5.
