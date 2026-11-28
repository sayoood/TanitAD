# Brief: KINCACHE: a label cache for `--dagger-label kin_v1`, so that the labeller runs once per bank OUTSIDE the GPU lock. Code only, zero training, dev box

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip;
* `F:/TanitAD-scratch/kinlab/FACTS_KINLAB_3.md`;
* `F:/TanitAD-scratch/kinlab/pod/FACTS_POD_RK.md`;
* `F:/TanitAD-scratch/kinlab/pod/DRY_MINT_RK.md`.

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands.
* No GPU, and nothing run on the pod.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* Work in a fresh export of **0ca6f49d** at `F:/TanitAD-scratch/kinlab/tree2`, made with `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 0ca6f49d stack taniteval | tar x -C ...`. Do not touch `kinlab/tree`.

## Why (MEASURED, the RK dry mint)
`kin_label_bank` takes about 1,640 s per load on the loaded pod, against about 196 s for speed_v1. It runs at every trainer load: the selfcheck and the training, twice per arm, under the GPU lock. That is about 1.8 h of idle GPU over two arms. The labels are deterministic per bank, so compute them once, outside the lock.

## The design (MM, binding)
1. **New trainer flag `--dagger-label-cache PATH`** (default none: compute as today, bit for bit at 0ca6f49d). It is valid only with `--dagger-label speed_v1 | kin_v1`; otherwise it is refused, as is any use at `--dagger-p 0`.
2. **If PATH exists**, load the labeller's outputs from it instead of calling the labeller: `slot`, `label`, `status_counts`, `is_kin` (kin_v1), `n_labelled`, and every other field `attach_recovery` and `rl_ctl` use.
   * Before using it, check its header. Each field must equal the value computed now, or the load is refused with a named reason:
     * `label_kind`;
     * `bank_json_sha256` (sha256 of the bank's bank.json file);
     * `labeller_sha256` (sha256 of the `v8f_recovery.py` file bytes as loaded, from `DR.__file__`);
     * `dagger_sha256` (`v8f_dagger.py` bytes);
     * `n_records` and the eligible index's sha256;
     * `clip_store_sha256` (a sha256 over the per-clip logged grid and speed arrays the labeller reads, in bank clip order).
3. **If PATH does not exist**, compute as today, then write PATH atomically (`.part`, then rename) with the header and arrays (npz, no pickle).
4. **The built levers / G-DVB** gain `dagger.dagger_label_cache` (the path, or none) and `dagger.dagger_label_cache_sha256` (the file's sha256 as loaded, or none). Add the FLAG_CHECKS entry exactly as `speed_v1` / `kin_v1` were wired.
5. **Precompute entry**: no new script. The trainer's existing `--selfcheck-only` with the flag and an absent PATH computes and writes the cache. Say how long it runs on CPU in your facts; the pod measured 28 min per bank.

## Tests (new file `stack/tests/test_v8f_kincache.py`, whole-file runs)
* **Round trip, bit for bit:** labels from a written-then-loaded cache equal a fresh compute on planted and fixture banks, and on 4,000 real records if you can pull shard 0 of RH bank s0 read-only (as phases 2-3; delete afterwards). This holds for both speed_v1 and kin_v1, comparing slot, label, status_counts and is_kin.
* **Refusals**, each for its named reason: bank.json changed; labeller bytes changed; dagger bytes changed; label kind differs; eligible index differs; clip store differs; a truncated `.part` file left behind is never read.
* **No flag:** identical to 0ca6f49d (gold).
* **Reachability:** with the flag and a cache, the batch's `gt` equals the no-cache batch's `gt` exactly, and the plan loss is equal to 0.0 difference.
* **Mutants** (expect RED): the header check skipped; one header field ignored (one mutant per field); load used without the flag; the write not atomic; the cache sha not recorded in the built levers.
* Run the existing `test_v8f_kinlab.py`, the dagger, recovery and rolladd files, and the launch-gate test files, all whole file. Report whether any fingerprint moves.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time behind `wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so).
* Nothing on C:. TMP, TEMP and `--basetemp` go under `F:/TanitAD-scratch/kinlab/tmp`.
* Stop if F: drops below 6 GB free.
* Never kill a process you did not start.
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_KINCACHE.md`.

## Deliverables
In `F:/TanitAD-scratch/kinlab/`:
* `tree2/` with `CHANGED_FILES_KINCACHE.txt` (path, sha256, base blob at 0ca6f49d);
* `FACTS_KINCACHE.md`;
* test logs;
* `mutants_kincache/MUTANTS_RESULT.json`;
* `HANDOFF_KINCACHE.md`, plus additions to `DELETIONS.log`.

Hand back in at most 200 words.
