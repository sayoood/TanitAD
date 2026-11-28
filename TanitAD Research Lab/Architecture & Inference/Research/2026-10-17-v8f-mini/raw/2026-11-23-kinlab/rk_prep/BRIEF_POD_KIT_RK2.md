# Brief: RK pod kit, revision 2 (commit 1a8b11a1 + the label cache), then three CPU prep jobs on the pod BEFORE the lock. No GPU, lock or marker; do not launch the chain

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip;
* `F:/TanitAD-scratch/kinlab/PREREG_V8F_KINLAB_AMENDMENT_1.md` (FROZEN, sha256 948670d9);
* `F:/TanitAD-scratch/kinlab/pod/HANDOFF_POD_RK.md` and `F:/TanitAD-scratch/kinlab/pod/FACTS_POD_RK.md` (revision 1 of the kit, built by an earlier agent);
* `F:/TanitAD-scratch/kinlab/FACTS_KINCACHE.md` and `F:/TanitAD-scratch/kinlab/HANDOFF_KINCACHE.md`.

## Rules
* Stage, commit and push nothing.
* No GPU, `/root/gpu_queue.lock` or `/root/N1DW_READY`. Do not launch `rh_rk_chain.sh` or its marker watcher.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* No secrets, IPs or ports in any file. Never name a file `*token.json`.
* Pod access: `source C:/Users/Admin/lorapod/podenv.sh`, then `pssh` / `pscp`. Wrap each call as `timeout N bash -c '...' < /dev/null`.
  * Long pod jobs go under `nohup setsid ... &`, and you poll their log. An ssh launch may hang: that is the known setsid trap, so check that the job is alive rather than relaunching it.
* The pod is TRAINING: the MM's UE_s3 until about 04:50Z, then the MM's ADD123 probe until about 05:45Z.
  * On the pod, write ONLY under `/root/rollhist/kit_rk/`, `/root/rollhist/tree_rk/` (or whatever tree path revision 1 uses for RK), `/root/rollhist/kin_cache/`, and the RK G-SUITE result path revision 1 uses.
  * Touch no other session's folders or processes.
  * Every pod CPU job runs at `nice -n 19` with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` and `CUDA_VISIBLE_DEVICES=""`. Before each start, the container's total_rss (cgroup) must be below 40 GB; otherwise wait (cap 30 min), then report.
  * Run one pod job at a time.

## What to do, in this order (bank each step)
1. **Kit revision 2**, in `F:/TanitAD-scratch/kinlab/pod/`. Keep revision 1's files as `*.rev1`.
   * **Commit:** 1a8b11a1abd50c09636da8b7425e97573cdf1d04 everywhere. That covers `rk_env.sh`, the tree-check blob list (`git ls-tree -r 1a8b11a1` on the C: mirror) and the delta tar (the files that differ from the pod's RA tree at df250a00: revision 1's six plus `test_v8f_kincache.py` and `golden_v8f_dagger_0ca6f49d.py`, with the two modified files at their 1a8b11a1 blobs).
   * **Argv tool:** RK = RL with exactly the three changes of Amendment 1 section 2; refuse a fourth. Extend the regression list, adding one regression for a cache path naming the other seed's bank.
   * **Gate record:** additionally require
     * argv's `--dagger-label-cache` equal to `/root/rollhist/kin_cache/s<K>_kin_v1.npz`;
     * the selfcheck's built levers `dagger.dagger_label_cache_sha256` equal to that file's sha256 now;
     * the commit 1a8b11a1.

     Add the matching mutants: cache absent; cache sha mismatch; the other seed's cache. Keep all revision-1 mutants.
   * **Chain:** the tree and G-SUITE steps read the prep results of step 2 below. Done-markers skip them under the lock if the commit and the tree check still match; otherwise the chain refuses BEFORE the lock. Everything else stays as revision 1, including the marker take-over on ADD123's end.
   * Tests on the dev box, whole files; mutants all RED.
2. **Ship revision 2** to `/root/rollhist/kit_rk/`, as new files beside revision 1 (never overwrite a differing file without a `.rev1` copy). Then, on the pod, in this order:
   * (a) **Tree prep** at 1a8b11a1, with the full blob check.
   * (b) **The two cache precomputes:** the trainer's `--selfcheck-only` on CPU with each RK seed's argv (`--device cpu` the only difference; revision 1's dry mint did the same), writing `/root/rollhist/kin_cache/s0_kin_v1.npz` and `s1_kin_v1.npz`.
     * Record each file's sha256, size and wall time, and the full-bank census from `kin_status_counts`. The bar is 0.5.
     * Keep each precompute's selfcheck.json beside the cache (`kin_cache/s<K>_selfcheck_cpu.json`) for the gate record's census check.
   * (c) **G-SUITE at 1a8b11a1** (revision 1's `ra_gsuite`-style script; about 37 min): 0 regressions, 0 new failures, 0 missing.
3. **Dry check of the chain** with `DRY=1` (if revision 1 has a dry mode; otherwise a static walk of the steps it would run): the tree and G-SUITE steps read as done, the argv renders with the three changes, and the gate record would find the caches.
4. Write `HANDOFF_POD_RK2.md` (at most 600 tokens): the exact launch commands for the MM, the cache sha256s, the census per bank, the G-SUITE result, and the lock time projected from MEASURED walls.

## Dev box
* CPU only, one python process at a time behind `wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so).
* Nothing on C:. TMP and TEMP go under `F:/TanitAD-scratch/kinlab/tmp`.
* Stop if F: drops below 6 GB free.
* Never kill a process you did not start.
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_POD_RK2.md` with what is left.

## Deliverables
* Revision 2 in `F:/TanitAD-scratch/kinlab/pod/`, with `SHA256_RK2.txt`;
* `PREP_RK2.md`: the tree check, both caches (sha256, size, wall, census), G-SUITE;
* the test logs;
* `HANDOFF_POD_RK2.md`;
* additions to `DELETIONS.log`.

Hand back in at most 250 words, saying plainly what passed and what did not run.
