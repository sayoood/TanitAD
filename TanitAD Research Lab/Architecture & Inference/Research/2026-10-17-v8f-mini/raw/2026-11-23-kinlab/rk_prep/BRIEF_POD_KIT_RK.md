# Brief: build the RK pod kit (card PREREG_V8F_KINLAB, arms RK_s0 / RK_s1) by the RA kit's pattern. Build, test and dry-check only: no GPU, no lock, no marker, launch nothing

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` (mirror tip, as in earlier briefs);
* the card `F:/TanitAD-scratch/kinlab/PREREG_V8F_KINLAB.md` (FROZEN, sha256 7db5d7aa, do not edit);
* `F:/TanitAD-scratch/kinlab/FACTS_KINLAB_3.md`.

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands and launches.
* No GPU, lock or marker.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* No secrets, IPs or ports in any file. Never name a file `*token.json` (use `*gatetok.json`).
* Pod access: `source C:/Users/Admin/lorapod/podenv.sh`; `pssh "..."`; `pscp local "$POD_TARGET:/path"`. Wrap each call as `timeout 60 bash -c '...' < /dev/null`.
* The LoRA pod is TRAINING (UE_s2 / UE_s3 under the MM's lock until about 05:00Z, then the ADD123 probe until about 06:00Z). On the pod:
  * write ONLY under `/root/rollhist/kit_rk/` and `/root/rollhist/dry_rk/`;
  * touch no process, lock or marker, and nothing under `/root/ruled`, `/root/dscale10k` or `/root/dscale_lead`;
  * a CPU dry step runs only at `nice -n 19` with `OMP_NUM_THREADS=2`, and only if the container's total_rss (cgroup, never `free`) is below 40 GB; check it right before.

## The model to copy
The RA kit is `F:/TanitAD-scratch/rolladd/pod/`: ra_env.sh, ra_tree_prep.sh, ra_gsuite.sh, ra_gate_record.py with test_ra_gate_record.py, rh_ra_chain.sh, ra_arm.sh, chk_arm_ra.sh, ra_argv.py, rh_ra_marker_watch.sh, launch_ra.sh, dscale10k_check_ra.py, all_blobs_ra.txt, ra_delta.tar and SHA256_RA.txt. Its landed records are under `raw/2026-11-22-rolladd/pod/`, and the pod logs of the RA run are in `/root/rollhist/` (read only).

Build the RK twin in `F:/TanitAD-scratch/kinlab/pod/`. Change exactly these:
* **Commit:** **0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a**.
  * The tree delta goes against the pod's current RA tree (df250a00): the six files of `F:/TanitAD-scratch/kinlab/CHANGED_FILES.txt`.
  * `all_blobs_rk.txt` comes from `git ls-tree -r 0ca6f49d` on the C: mirror (stack + taniteval, as all_blobs_ra.txt).
  * The tree check refuses any blob that differs.
* **Argv:** RL_s0 / RL_s1's argv (`/root/rollhist/runs/RL_s{0,1}/argv.json`, read only) with exactly `--dagger-label speed_v1` → `kin_v1` and `--out` → `runs/RK_s{0,1}/train`, nothing else.
  * The argv tool refuses any third difference. Prove it with deliberate regressions, as the RA kit did.
* **Gate record (`rk_gate_record.py`)** mints only when all of these hold:
  * the argv names `--dagger-label kin_v1` and the RH bank of that seed, with its sha256 equal to `bank.json`'s;
  * the selfcheck's `levers.dagger.dagger_label` is `kin_v1`;
  * `dagger_probe.n_label_kin` is greater than 0;
  * **the census bar of card section 4 holds**: from `dagger_probe.kin_status_counts` (the full bank's counts), the relabelled share of ok records is ≥ 0.5. Record the share in the gate record.
  * G-SUITE ran at 0ca6f49d with 0 regressions, 0 new failures and 0 missing;
  * the RK tree marker exists.

  It also covers verify-after-mint tamper refusal. Tests: a well-formed record mints and verifies, and each mutation refuses. The mutations: no kin_v1, speed_v1 instead, n_label_kin 0, share 0.49, G-SUITE at df250a00, tag RA_s0, the tree marker missing, the bank sha mismatch, a bound file changed after the mint, and the commit placeholder unfilled.
* **Chain (`rh_rk_chain.sh`):**
  * it waits (cap 12 h) for the ADD123 chain's last line `ZZ ADD123 ` in `/root/dscale_lead/mm_add123_chain.log`;
  * then: tree prep → G-SUITE at 0ca6f49d (inside the chain, under the marker, before the lock; measure its wall in the dry plan from the RA prep logs) → render → per arm (pair check, selfcheck, gate mint and verify, train, summary and post checks, zerograd, ARM_DONE) → the RA chain's end steps.
  * GPU lock: `flock` on `/root/gpu_queue.lock` held from the first GPU step to the end, as `rh_ra_chain.sh`.
  * Marker: a watcher like `rh_ra_marker_watch.sh` sets `/root/N1DW_READY` ONLY if it is absent and clears only its own token.
    * The ADD123 chain holds the marker until its own end. The RK watcher must take over without a gap, by waiting for `ZZ ADD123 ` and setting the marker in the same check.
  * Resumable: done-markers per step, ARM_DONE skips.
  * A refusal hands the lock back at once.
* **Read extension (dev box, `F:/TanitAD-scratch/rollhist/read`):** the RA read added `RH_AKIND=RA`. Add `RH_AKIND=RK` the same way, with these pairs:
  * the treated arm RK;
  * P1 RK − B0;
  * ATTR RK − RL;
  * RK − RH beside;
  * MECH3 = RK − RL at T0 (`mech_rk.py`, modelled on `mech_ra.py`, with correct arm names in the printout; `mech_ra.py` printed "RL" for RA).

  Also make `rh_read_rk_chain.sh` / `.cmd`, modelled on `rh_read_ra_chain.*` together with `rh_read_ra_retry.*`.
  * Keep `.bkRK` copies of every edited read file.
  * Prove the read extension with its dry check under the default, under RA and under RK (the RA read's `dry_check.py`).
  * Prove the MECH3 tool against RL's planted T0 evals: plant RL as "RK" and RH as "RL", and it must reproduce `MECH_RECLAB.json`'s RL − RH numbers exactly.
* **Dry mint on the pod**, CPU, inside `/root/rollhist/dry_rk/` only: render both argv, then run the selfcheck-only mint on the real tree delta and the real banks.
  * This also gives the pod census of card section 4 on BOTH full banks. Report it.
  * Read `checks.json` in full.
  * Delete `dry_rk/` afterwards by literal path, and log it.
  * If the selfcheck needs more than about 12 GB of RAM or 30 min, stop and report rather than run it.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time behind `wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so).
* Nothing on C:. TMP, TEMP and `--basetemp` go under `F:/TanitAD-scratch/kinlab/tmp`.
* Stop and hand back if F: drops below 6 GB free.
* Never kill a process you did not start.
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_POD_RK.md`.

## Deliverables
In `F:/TanitAD-scratch/kinlab/pod/`:
* the kit;
* `SHA256_RK.txt`;
* the test logs (gate record, argv, read dry checks, MECH3 control);
* `DRY_MINT_RK.md`, with the census per bank and tick;
* `FACTS_POD_RK.md`;
* `HANDOFF_POD_RK.md` (at most 600 tokens, with the exact launch commands);
* `DELETIONS.log`.

Hand back in at most 250 words, saying plainly what passed and what did not run.
