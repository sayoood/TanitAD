# HANDOFF: STRAT1 launch package (Arch+Inf agent, 2026-11-16), written at ~130k context

Commits: base cdbff971fa6dad4d20b937fe8abda6cb9f622adb, new b9250158fc6c41aefac50a3e7459d46a2e5c2f83 (branch agent/v8f-strat1-20261116). Nothing committed or pushed by me.
Pod access: `source C:/Users/Admin/lorapod/podenv.sh` (pssh/pscp; filter `post-quantum|store now|upgraded`). Never write the pod address anywhere.

## Pod layout (/root/strat1)
- `base/`, `new/` = `git archive` exports (stack + taniteval) of the two commits (tar sha256 of the base gz a8f4991b..., delta tar e11f6d2c...). `tree/` = `new/` + the updated `stack/scripts/strat1_check.py` (sha a3b75f8d...) + `stack/tests/test_strat1_token.py` (sha a2d5db98...). The MM must land those two files; the tokens bind the tree manifest of `tree/`.
- `inputs/` = M2b_s0/s1 ckpt, dagger_bank/s0/s1, data/v9/*, join_train.npz, p10/, native_tracks.pt, strat1/{navtext_nav.npz, navtext_nav_report.json, strat_labels_table.npz, join_eval141.npz, strat_platform_by_clip.npz}. Verified by `sha256sum -c /root/strat1/inputs.sha256` (list taken from the box-2 source hashes; the strat1/ files from the dev box). The pilot script runs that check first and refuses on any mismatch.
- junit: `junit/{base,new}/shard{0..7}.xml` + `.log`; shard 2 of each tree was re-run with `--continue-on-collection-errors` (test_secret_scan.py needs `secret_scan` from tools/, absent from the export: a collection error on BOTH trees).
- No pytest-xdist on the pod (a pip install was refused by the permission layer): the suite runs as 8 file shards per tree, both trees at once, OMP 4 threads, CUDA hidden.

## Running detached (all log to /root/strat1/logs/)
1. `gsuite_all.sh` -> `logs/gsuite_all.done` when both trees finish (started 11:11 pod clock).
2. `strat1_pilot.sh` (log `logs/pilot.log`, ends `ZZ STRAT1 PILOT DONE ZZ` or `ZZ STRAT1 PILOT FAIL ZZ <why>`): waits for inputs + suite, runs `strat1_check.py suite` (writes `/root/strat1/gsuite_result.json`, exit 1 on any regression), then mint + run through the runner for pilot200_{C0,N1,S1}_s0 one at a time, then `pilot_summary.json` (s/step from cumulative `sec`, sampled peak GPU MiB, projected chain hours ESTIMATED).
3. `dry_mint.sh` (`logs/dry.log`, `dry/mint.log`): debug mint of C0 pilot in `/root/strat1/dry` against a synthetic suite result (commit `dry`): NOT a launch token. Read `dry/mint.log` and `dry/runs/pilot200_C0_s0/gate/` first: any checker bug shows there before the real pilot reaches it. If it shows a bug, fix `strat1_check.py` in the package AND `tree/` (and `build/tree`), re-upload, and restart the pilot (kill only by PID: `ps -eo pid,args | grep strat1_pilot` and exclude the grep itself).

## State at hand-off (pod clock ~11:50)
- Inputs: all 21 sha256 entries verified OK (`logs/inputs_verify.log`). Frozen argvs shipped to `inputs/frozen/argvs_obd.json` (sha 346a48ed...); the eval grid `eval_grid_windows.json` (sha 659c3b65..., git blob equal at both commits; NOT compared with box 2's copy: UNVERIFIED) sits in `tree/TanitAD Research Lab/.../raw/`, because the trainer's K-B3 check refuses a launch without it. strat1_check.py (sha 338a0efe...) binds it.
- Dry mint found two real defects, both fixed (frozen-argv path, eval grid). The third try was running at hand-off (`dry/mint.log`): the first short trainer process had run 8 min with no output (start-up under the suite's CPU load). The pilot's GPU-idle check (< 3500 MiB) can fail if the dry run still holds the GPU when the suite ends: then `strat1_pilot.sh` ends FAIL "GPU busy": just relaunch it.
- G-SUITE at ~40 % per shard; failures on BOTH trees already visible (shards 0, 2, 7: pre-existing). Pilot/chain not started.
- Token tests (`test_strat1_token.py`, 18) PASSED locally on the checker before the last two edits (frozen path, eval grid); re-run them once (serial, 10 min, F: busy).

1. Read `logs/pilot.log`, `dry/mint.log`. Regressions: `gsuite_result.json` lists `regressions`, `new_test_failures`, `fail_on_both`. Re-run each regression serially to separate a real regression from shard collisions (shards of one tree share /root/strat1/tmp/<tree>).
2. Report pilot s/step, peak GPU, chain projection (>10 h projected = STOP and report, BUILD_SPEC section 8).
3. The 7-arm chain `strat1_chain.sh` (arms C0:s0 N1:s0 S1:s0 C0:s1 N1:s1 S1:s1 C0:rep, one at a time, ARM_DONE markers, ends `ZZ STRAT1 CHAIN DONE ZZ` / `ZZ STRAT1 CHAIN FAIL ZZ <why>`, STOP file `/root/strat1/chains/STOP`) is staged and NOT launched. The MM launches it after freezing the card.
4. Not built: the 8-window forward_infer parity part of G-EVAL (geval checks the strict load, built levers == selfcheck levers, and that a cfg dropping ctx_heads is refused).

## Local files (F:/TanitAD-scratch/strat1/)
package/stack/scripts/strat1_check.py + package/stack/tests/test_strat1_token.py (identical copies in build/tree/stack/...); launch/{gsuite.sh, strat1_pilot.sh, strat1_chain.sh, pilot_summary.py, dry_mint.sh, inputs.sha256}; launch/tmp/ is scratch (delete at the end; log in launch/deletions.log).
