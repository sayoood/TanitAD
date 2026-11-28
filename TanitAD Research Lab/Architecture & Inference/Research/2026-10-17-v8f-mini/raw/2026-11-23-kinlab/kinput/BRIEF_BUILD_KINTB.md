# Brief: KINTB: make kin_v1's blend horizon T_B a trainer flag (`--dagger-kin-tb SECONDS`, default 1.0 = commit 1a8b11a1 bit for bit), so the KINLAB card's next lever (T_B 2.0 s) is ready. Code only, zero training, dev box

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip (`git -C C:/Users/Admin/tanitad-push show origin/agent/arch-inf-20260803:"Project Steering/AGENT_OPERATING_STANDARD.md"`);
* `F:/TanitAD-scratch/kinlab/PREREG_V8F_KINLAB.md` (FROZEN 7db5d7aa): its outcome table names "T_B 2.0 s" as the next lever under HARMS / PARTIAL with MECH3-YES;
* `F:/TanitAD-scratch/kinlab/FACTS_KINLAB_3.md` and `F:/TanitAD-scratch/kinlab/FACTS_KINCACHE.md`.

## Why
RK_s0/RK_s1 (kin_v1, T_B = 1.0 s) are training now; their read decides the next lever. If it is T_B 2.0 s, the arm needs this flag. Build it now so the arm is ready when the verdict lands. No RK number exists yet, and none is needed for this build.

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands.
* No GPU, and nothing run on the pod.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* Work in a fresh export of **1a8b11a1** at `F:/TanitAD-scratch/kinlab/tree3`, made with `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 1a8b11a1 stack taniteval | tar x -C ...`. Do not touch `kinlab/tree` or `kinlab/tree2`.
* No secrets, IPs or ports in any file. Never name a file `*token.json`.

## The design (MM, binding)
1. `T_B = 1.0` is a module constant in `stack/tanitad/data/v8f_recovery.py` (line ~421), used by the blend weight (~530) and the merge test (~561). Make it a parameter of the kin_v1 labeller path, with the default 1.0. No other constant changes.
2. **New trainer flag `--dagger-kin-tb SECONDS`**, default 1.0. With the default, everything is 1a8b11a1 bit for bit (labels, cache files, built levers apart from the new key). It is valid only with `--dagger-label kin_v1`; a non-default value with any other label kind is refused. It must be finite, at least 0.2 and at most 3.0 (the label spans 6 s); otherwise refused.
3. **The label cache header gains `kin_tb`.** A load with a different value is refused, with a named reason. A cache written at 1a8b11a1 (no `kin_tb` field) is read as 1.0, so the two RK caches on the pod stay valid at the default and are refused at 2.0. Say in your facts whether this keeps the cache file bytes identical at the default; if adding the field changes them, then the default path must still LOAD the old caches.
4. **Built levers / G-DVB** gain `dagger.dagger_kin_tb`. Add the FLAG_CHECKS entry exactly as `--dagger-label-cache` was wired in 1a8b11a1. Memory note: a pin edit can move the launch-gate profile fingerprints, so run all the launch-gate test files and report whether any fingerprint moves.
5. **Census at T_B 2.0** on real records, if you can pull shard 0 of RH bank s0 read only, as FACTS_KINCACHE did (delete it afterwards and log the deletion). Use the kin_v1 census function and report, at T_B 1.0 and 2.0 on the same records: the relabelled share of ok (the KINLAB bar is 0.5), the refusal reasons, the merge-time distribution, and the max |a| and |kappa|.

## Tests (new file `stack/tests/test_v8f_kintb.py`, whole-file runs)
* **Default, gold:** labels and the cache at `--dagger-kin-tb 1.0`, or with the flag absent, equal 1a8b11a1 bit for bit on the fixture, planted and real records.
* **2.0 moves only kin labels:** records that are not kin_emitted at 1.0 keep their label; the merge time is never below 2.0 s on an emitted record.
* **Refusals:** a non-default value with speed_v1 or human; out of range; a cache whose `kin_tb` differs; an old cache at 2.0.
* **Reachability:** with 2.0 through the trainer argv, the batch's `gt` differs from the 1.0 batch on at least one kin record, and the built levers read 2.0.
* **Mutants** (expect RED): the flag parsed but not passed to the labeller; the header check ignoring `kin_tb`; the merge test still using the constant 1.0; the built-levers key missing.
* Run `test_v8f_kinlab.py`, `test_v8f_kincache.py`, the dagger, recovery and rolladd files, and the launch-gate test files, all whole file.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time behind `wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so).
* **The MM's RK read chain runs on this dev box from about 07:30Z** (CPU rollouts). Never touch its files or processes (`F:/TanitAD-scratch/rollhist/read/`). If the commit gate makes you wait, wait; never lower the gate to get past it.
* Nothing on C:. TMP, TEMP and `--basetemp` go under `F:/TanitAD-scratch/kinlab/tmp`.
* Stop if F: drops below 6 GB free.
* Never kill a process you did not start.
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_KINTB.md` with what is left.

## Deliverables
In `F:/TanitAD-scratch/kinlab/`:
* `tree3/` with `CHANGED_FILES_KINTB.txt` (path, sha256, base blob at 1a8b11a1);
* `FACTS_KINTB.md` (the census table at 1.0 and 2.0, the fingerprint result, the cache-bytes answer);
* test logs and `mutants_kintb/MUTANTS_RESULT.json`;
* `HANDOFF_KINTB.md`, plus additions to `DELETIONS.log`.

Priority order if you run short: (1) the flag and its tests at the default, gold; (2) the cache header; (3) G-DVB; (4) mutants; (5) the census.

Hand back in at most 200 words, saying plainly what passed and what did not run.
