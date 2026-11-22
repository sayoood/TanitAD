# Brief: build the REPS2 kit, DSCALE10K stage A for UE_s2 and UE_s3, by the reps1 pattern. Build and dry-check only: run nothing on the GPU, launch nothing

You are a TanitAD build agent working for the Master Mind (MM). First read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands and launches.
* No GPU.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* No secrets, and no IPs, ports or pod ids in any file. Never name a file `*token.json` (use `*gatetok.json`).
* This is a zero-spend build. Running the arms needs the PI's go, which the MM holds.

## What the arms are
* **UE_s2.A** and **UE_s3.A** are UE_s0.A's stage-A argv on the LoRA pod (`/root/dscale10k/runs/UE_s0/A/argv.json`: 35 tokens, arm M2b, from scratch, `--steps 61568`, union cache and merged labels), with exactly two changes: `--seed 2` (resp. 3) and `--out /root/dscale10k/runs/UE_s2/A` (resp. UE_s3).
* Same code, data manifests, checker and gate as the reps1 chain that ran UE_s1.A.
* Purpose: two more model seeds of the 10k line, which the PI approved for the v8proof fork's replicate card ADD121, and which are also the third and fourth seeds for this line's own cards.

## Read first (they bind)
* The reps1 kit:
  * local `F:/TanitAD-scratch/dscale10k/reps1/`;
  * landed under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-15-v8f-datascale/reps1/`;
  * especially `build/HANDOFF_REPS1.md`, `gen_argv_reps1.py`, `patch_checker_reps1.py`, `pod_copy/dscale10k_check_reps1.py`, `chk_arm_reps1.sh`, `blockreps1*.sh`, `queue_reps1.sh`, `dryrun_reps1.sh`, `verify_token_reps1.py`, `verify_UE_s1.A.json`.
* The pod's `/root/dscale10k/` (BUILD_REPORT_DSCALE10K.md, MM_RULINGS_DSCALE10K.md, `study/`, `chainlib.py`, `chk_arm.sh`) and `/root/dscale10k/runs/UE_s1/A/` (argv, selfcheck, summary, post, A.run.log).
* The GPU-sharing protocol in `/root/rollhist/rh_rl_chain.sh` and `rh_rl_marker_watch.sh`: one GPU job at a time under `flock /root/gpu_queue.lock`; the `/root/N1DW_READY` marker makes the other session (TanitADRuled) yield at its slice boundaries.

## Priority order (bank each part as it is done)
1. **`FACTS_REPS2.md`**:
   * which code commit and pod tree UE_s1.A ran at, and the `file:line` where its gate token binds the commit, the argv sha256 and the data manifests;
   * whether the checker admits seeds 2 and 3 today, and if not, the exact line that refuses them;
   * how the reps1 queue held the GPU lock and the marker;
   * UE_s1.A's measured walls (selfcheck, train, post).
2. **The kit**, in `F:/TanitAD-scratch/dscale10k/reps2/`:
   * the argv generator: UE_s0.A's argv with exactly `--seed` and `--out` changed. An argv diff against UE_s0.A that shows any third change must refuse.
   * the checker change: admit UE seeds 2 and 3 for stage A, and nothing else. Mutants must be refused: a second changed flag, another data manifest, the wrong seed under a name, and a UM or other arm name. Write tests, and run whole files on the dev box.
   * the per-arm chain: lock, then the pair or argv check, the selfcheck, the gate token mint and verify, training, the summary and post checks, `zerograd_gate.py` if reps1 ran it, and an ARM_DONE marker.
     * It must be RESUMABLE: an arm with ARM_DONE is skipped, and a killed run restarts from `ckpt_last.pt` (the trainer resumes from `--out`). Check this in the trainer source at that commit and quote the line.
     * It keeps the marker protocol unchanged, with a marker watcher like `rh_rl_marker_watch.sh`.
     * A refusal must hand the GPU back at once (release the lock, clear the marker).
   * the delivery script (do not run it): `ckpt_last.pt` of each finished arm to the v8proof fork pod at `/workspace/v8proof/integ10k/ckpt/UE_s{2,3}/A/`, by the route used on 2026-11-19 for UE_s0 and UE_s1. Find and quote that route from `E:/Projects/TanitAD-artifacts/dagland/LOOP_STATE_2026-11-02.md` or the landed records. Carry the sha256 checked at both ends, and an arm record JSON (commit, argv sha256, gate token path and sha256, summary seconds).
3. **Dry run on the pod**, CPU only, no lock, no marker, in `/root/dscale10k/dry_reps2/`:
   * render both argv;
   * the checker's mint in dry mode, as `dryrun_reps1.sh` did (read the queue for a DRY flag first; memory: a queue without one starts for real);
   * read `checks.json` in full;
   * delete `dry_reps2/` by literal path afterwards and log it.
   * You may write to the pod ONLY there and ONLY for this dry run. Touch no process, lock or marker, and nothing under `/root/ruled`, `/root/rollhist` or another session's folders.
4. **`DRY_RUN_REPS2.md`** (what ran, the results, what is not yet exercised) and **`HANDOFF_REPS2.md`** (at most 800 tokens, with the exact launch command for the MM).

## Pod (`source C:/Users/Admin/lorapod/podenv.sh`; `pssh "..."`, `pscp local "$POD_TARGET:/path"` and `pscp "$POD_TARGET:/path" local`; wrap calls as `timeout 60 bash -c 'source ...; pssh "..."' < /dev/null`)
* The pod GPU is in use: by the MM's RA chain now, then by TanitADRuled. Never use it.
* Container memory: read `memory.max` and `total_rss` in the cgroup (never `free`); run nothing heavier than the checker and argv tools.

## Dev box
* CPU only, one python process at a time, each gated on commit >= 5.5 GB: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`.
* Set `CUDA_VISIBLE_DEVICES=-1`.
* ⛔ C: has about 2 GB free: write nothing to C:. Set TMP, TEMP and pytest's `--basetemp` under `F:/TanitAD-scratch/dscale10k/reps2/tmp`.
* F: must keep at least 15 GB free. Your folder stays under 500 MB.
* Never kill a process you did not start. A blanket `taskkill` or `pkill` is forbidden.
* Never wait more than about 4 minutes in one call. If your context passes about 150k, write HANDOFF_REPS2.md and end.

## Deliverables
In `F:/TanitAD-scratch/dscale10k/reps2/`:
* FACTS_REPS2.md;
* the kit, with tests and mutants;
* DRY_RUN_REPS2.md and HANDOFF_REPS2.md;
* MANIFEST_SHA256.txt and DELETIONS.log.

Hand back in at most 250 words, with a deliverable manifest, and say plainly which checks PASSED and which did not run.
