# FACTS_POD_RK: the RK pod kit (card PREREG_V8F_KINLAB, commit 0ca6f49d), build, tests, dry mint

Evidence class: MEASURED here (dev box CPU, or the LoRA pod CPU for the dry mint) unless marked INHERITED. Folder `F:/TanitAD-scratch/kinlab/pod/`. Nothing staged, committed or pushed; nothing on C:; no GPU, lock or marker touched.

## What was built (13 shipped files, `SHA256_RK.txt`; the RA kit's pattern)
| file | change against the RA kit |
|---|---|
| `rk_env.sh` | COMMIT 0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a; TREE `$R/tree_rk`; `BLOBS` = `all_blobs_rk.txt` beside the script; `GSUITE_RK` |
| `all_blobs_rk.txt` | `git ls-tree -r 0ca6f49d stack taniteval` on the C: mirror (read only) + the eval-grid row: 3,418 rows, sha256 fb1a1ff1…. Against `all_blobs_ra.txt`: exactly the six blobs of `CHANGED_FILES.txt` (3 modified, 3 added) |
| `rk_delta.tar` | `git archive` of those six paths (sha256 bba9d744…; each member's git blob id equals the list's) |
| `rk_tree_prep.sh` | tree_ra (blob-verified first) + delta = tree_rk, blob by blob against `all_blobs_rk.txt` and by file count; `.part` then rename; `RH_ROOT` moves R (the dry mint) |
| `rk_gsuite.sh` | copy of `ra_gsuite.sh` (names, commit); wall MEASURED on the same shards: RA 36 min 32 s (waves 9:46, 15:01, 11:45), RL 37 min 49 s |
| `rk_argv.py` | RL_s{k}'s argv with the `--dagger-label` value `speed_v1` -> `kin_v1` (same position, same token count) and `--out`; `render` and `verify`; refuses any third difference |
| `rk_gate_record.py` | RH branch + kin checks + the census bar (below) |
| `chk_arm_rk.sh`, `dscale10k_check_rk.py` | the study-side checker with an `RK` kind (`patch_rk.py` on the RA checker ba5dab8e): RH argv + exactly `--dagger-label kin_v1`, bank under `bank/`, dose 0.1 |
| `rk_arm.sh` | `rk_argv.py verify`, pair check (argv minus the two label tokens vs B0), selfcheck, gate record, token, train, **lever-acts** (train log `dagger_label_kin` summed > 0, card section 3), summary, post, zerograd, K-CLIP |
| `rh_rk_chain.sh` | see below |
| `rh_rk_marker_watch.sh`, `launch_rk.sh` | see below |

## Chain (`rh_rk_chain.sh`)
1. waits (poll 5 s, cap 12 h) for `ZZ ADD123 ` in `/root/dscale_lead/mm_add123_chain.log` (the ADD123 script's last line is DONE, STOP or FAIL);
2. `tree_prep`, then `gsuite` (CPU, about 37 min), **under the marker and before the lock**;
3. `flock` fd 9 on `/root/gpu_queue.lock`, held to the end; an EXIT trap runs `flock -u 9`, so a refusal hands it back at once;
4. tree check (writes `done_rk/tree.done`, the gate's marker), inputs (RH, RL, B0 finished with checkpoints; banks), disk, render, then RK_s0, RK_s1 (a partial arm or `tree_rk.part` is moved aside, never deleted), then the RA chain's end (`free disk`, `ZZ ROLLHIST-RK DONE ZZ`). Every step leaves `done_rk/<step>.done`; STOP file read between steps.

**Marker.** The ADD123 script sets `/root/N1DW_READY` (empty file, own token file `add123_marker_mine`) and removes it in `unmark`, which runs BEFORE its tar step and its last line. So the watcher takes over on the earlier of its two end lines, `marker removed (mine)` or `ZZ ADD123 `, in the same check, with a noclobber create (only if absent, own token as content), polling every 2 s; it clears only a marker whose content is its own token, when the chain logs `GPU queue lock held to the end of the chain`, or on a chain end / no chain process. **This is the one place the kit goes beyond the brief's wording** (waiting for `ZZ ADD123 ` alone would leave the marker absent during the tar); drop the first pattern in `rh_rk_marker_watch.sh` if the MM wants the literal form. Tested (T1-T5, scratch paths via `RKW_*`, fake chain process): `test_log_rk_marker_watch.txt`.

## Gate record (`rk_gate_record.py`) mints only if
argv `--dagger-label kin_v1`, `--dagger-bank` = the RH bank of the seed with `--dagger-bank-sha256` = sha256(bank.json), no `--dagger-mode`; built levers `dagger_label` exactly `kin_v1`; `dagger_probe.n_label_kin` > 0 and `rl_label_rows_changed` > 0; `grad_diff_all` > 0, `control_p0_grad_diff` exactly 0.0; **census bar**: `kin_status_counts["kin_emitted"] / sum(counts)` >= 0.5 (the labeller visits ok records only; share and counts are written into the record); G-SUITE new commit = 0ca6f49d with 0 regressions / new-test failures / missing; `done_rk/tree.done`; pilot, pair log, full-bank check; `verify` re-hashes every bound file and re-derives the share from the bound selfcheck.
Tests (`test_log_rk_gate_record.txt`): well-formed mints and verifies (seed 0 and 1), the boundary share 0.50 mints, **18 mutations refuse each for its stated reason** (no kin_v1, speed_v1, levers say speed_v1, n_label_kin 0 and missing, share 0.49, counts missing, G-SUITE at df250a00, G-SUITE regression, tag RA_s0, tree marker missing, bank sha mismatch, dagger-mode add, grad_diff 0, control_p0 != 0, rl rows 0, bank_k, pair log), the commit placeholder, and `verify` refuses after: a bound file changed (gsuite, argv, bank.json, tree marker), the marker removed, the record's census or label edited, the selfcheck counts edited. Six code mutants of the gate record (census bar, label check, G-SUITE commit, bank sha, verify's census re-derivation, n_label_kin) each turn the test RED (`mutants_rk_gate_record_result.txt`).

## Argv (`test_log_rk_argv.txt`)
Well-formed RK differs from RL in exactly 2 token positions (the `--out` value and the label value); 12 deliberate RK regressions (label left speed_v1 / human, a third flag, a third value, another bank, `--out` left RL's, dose 0.3, `--dagger-mode add`, bank sha, label token moved, init, RA's `--out`) and 7 source regressions (RL label human / not last, dose, bank sha, dagger-mode, init, readout) all refuse. Checker kind: 19 tests pass (`test_log_rk_checker.txt`: a second changed flag, another bank / sha, speed_v1 / human / kin_v2 / KIN_V1 under RK, label missing, dose != 0.1, a mode flag, the label under the RH name); 5 one-pin checker mutants each RED (`mutants_rk_checker_result.txt`).

## Read extension (dev box, `F:/TanitAD-scratch/rollhist/read`; `read_ext/`)
`RH_AKIND=RK`: `rh_common.py` (assert, beside pairs, `ATTR_RK_minus_RL_R0C10XHb20`), `rh_read.py` (`NEXT_RK`, `rk_answer`), `rh_t0.py` (doc), `rh_roll.py` (choice); `.bkRK` copies of all five, `mech_rk.py`, `rh_read_rk_chain.sh/.cmd`, `rh_read_rk_retry.sh/.cmd` (a `mech3` step after `gt0`; gate0 also needs RL's dumps, T0 evals and checkpoints). `dry_check.py` under the default (57 ok), RA (68 ok) and RK (70 ok): 0 FAIL, and the default / RHp3 / RHK / RL / RA pair tables equal their pre-RK copies. Reader selftest under RK: all hold. **`DRY_CHECK.json` and `DRY_CHECK_RA.json` were rewritten by the new runs** (each now also holds the RK-unchanged checks).
**MECH3 control** (`test_log_mech3_control.txt`): RL planted as RK, RH as RL, `mech_rk.py` reproduces `MECH_RECLAB.json` exactly (s0 -0.0024 [-0.0273, +0.0247], s1 -0.0156 [-0.0376, +0.0058], plus the beside rows).

## Dry mint on the pod (`DRY_MINT_RK.md`)
Real tree delta (3418 of 3418 blobs), real banks, real init, CPU selfcheck at both seeds (28 min each, 5.2 GB RSS): census **0.6398 (s0) and 0.6397 (s1)**, bar 0.5 met; gate record MINTED and VERIFIED; study-side dry mint 21 of 21 checks at both seeds. **Cost finding:** the kin labeller wall is 1,631 / 1,643 s per load on the loaded pod (RL's speed_v1: 192 / 200 s) and runs twice per arm under the lock: projected chain 1.5-2.3 pod-h against the card's 0.6-0.75 (ESTIMATED; above the 0.9 line it goes back to the PI).

## Not run / limits
* G-SUITE at 0ca6f49d (the chain runs it; ~37 min). The dry mint used a planted G-SUITE record and tree marker in its scratch root only (deleted with it).
* No GPU selfcheck; the dry selfcheck is CPU (`--device cpu` is the only argv difference).
* The real RK argv files (render writes under `/root/rollhist/runs/RK_s*`) were not written; the dry render is in the scratch root.
