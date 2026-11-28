# BUILD: DSCALE10K banker and block-2 chain (build agent, 2026-11-17)

Nothing was committed or launched. The GPU, `/root/gpu_queue.lock`, the CKPT120 chain, `lp3_sampler.py`, `bank_ckpts.py` and other studies' folders were not touched. Pod cgroup at start: limit 70,999,998,464 B, total_rss 3.14 GB.

## 1. The banker (done, and run on the 7 pending checkpoints)

* Files: `F:/TanitAD-scratch/datascale/gate/banker/dscale_banker.py` (pod copy `/root/dscale10k/dscale_banker.py`), test `banker/test_dscale_banker.py` (11 tests, a fake hub, run on this machine: OK).
* Repo: **PRIVATE dataset repo `Sayood/tanitad-v8f-dscale10k-runs`** (`private` read back True, 81 files). Layout `<arm>/<stage>/ckpt_last.pt` plus argv, selfcheck, summary, post, train_log, the gate token and checks, the dry record and evidence (stage A), `argv_B.json` and `bank/bank.json` + `PROVENANCE.json` (stage B).
* Rules (`dscale_banker.py`): `check_record` (:165) refuses unless the record path is exactly `runs/<arm>/<stage>/ckpt_last.pt` (regular file), the file sha256 equals the bank_ready record and the chain's `ARM_<stage>_DONE`; `bank_one` (:189) uploads in ONE commit, reads every file back FROM THAT COMMIT (size + sha256), requires the hub LFS id of the checkpoint to be present and equal to the sha256 (:208), and only then writes `bank_ok/<tag>.sha256` atomically (`<sha256> <repo> <revision> <path> <size> <utc>`; chainlib reads the first token). A marker for other bytes is reported as CONFLICT and never overwritten. It deletes nothing. flock on `banker.lock`; ledger `/root/dscale10k/banker_ledger.jsonl`. Token read in place from `/root/.hf_token`, `HF_HUB_DISABLE_XET=1`.
* Tests cover: bank + idempotent + nothing deleted; a corrupt upload (every attempt) writes no marker; one corrupt attempt then a verified retry; LFS id missing or wrong; upload failure; record sha mismatch (no upload); wrong path; DONE-marker disagreement; conflicting marker; public repo refused; sidecars ride along.

Banked on the pod at 14:16-14:17Z (0 failed attempts, all 7 verified from the commit; sha256 = the bank_ready record):

| tag | revision | sha256 |
|---|---|---|
| B_rep.A | 2dfbd72e176958192e8bf4b761c8245a1c6bb18b | d8b0da4d19b8da481793344404713bf4d55398bff72858e6174b389323891fb4 |
| B_rep.B | 784b1e5d26ff1c7a930649b6a2b3c89674acfda9 | 2e88ebfa9d359561042d3c74092a9a3de6cc03894ae9579b318cfe63e8d689a0 |
| B_s0.A | 039afc165ce3cc6d2e81bc246f97a9050228a58b | 96bb1d9bfffbcf2c8491abec244e097baaa260f870b2a0de1b2d9b5cc25c4e95 |
| B_s0.B | 5324394bd0fb6786fb8ca2c5db92be44eb233841 | 4e97997df91b7c574b5b6c4cd5b36b7ccd408f1e025a44b6eac979925268febe |
| R_s0.A | 58c5c158bc4e3896c7658e0467ea2e04afd4b2bb | 48263dde91149878ff93c626e7a1d4d9b32d40bd661ee342b7b0583bc76db9c1 |
| U_s0.A | 4d3ea88322675b9babb3ce93d574bca5c14e4018 | 3ce11963a21dc212518dca506cc22d33ea98ea1fe95ec02665329d75fe01a298 |
| U_s0.B | 1d465ff14c47566e0d66c1709542f377a985866b | 9359124400f7072717af83b0271f9b834f676506806cde5dbabc190c67f8b540 |

The 7 `bank_ok/*.sha256` markers exist now. Nothing was deleted by this build; the next chain start (block 2 pre-flight `sweep_all`) deletes the block-1 bank shards and stage-B selfcheck scratch that these markers cover, each logged in `deletions_block1.log`.

## 2. Block-2 chain (written, dry-run, NOT launched)

* `/root/dscale10k/block2.sh` (local `block2/block2.sh`), built from `block1.sh` (same gate_wait, runner, summary and post gates, in-chain stage-B mint, 60 GB guard). Arms B_s1 then U_s1, each stage A, 4-shard bank, stage B (card 2.3 rows 5 and 6). Differences: pre-flight banks anything pending, sweeps what is banked, and runs `block2_prep.sh` when a stage-A token is missing; after every stage the banker runs (a failure prints `BANK-DEFERRED` and the chain goes on, deletions stay deferred); the sweep covers all five arms of blocks 1 and 2. Ends `ZZ DSCALE10K BLOCK2 DONE ZZ` or `... FAIL ZZ <why>`.
* Rendered argvs (`/root/dscale10k/argv/{B_s1,U_s1}.{A,B.tmpl}.json`, written on the pod by `gen_argv_s1.py`, log `gate/gen_argv_s1.log`): B_s0 / U_s0 files with ONLY `--seed 1`, `--out` (`runs/<arm>/<stage>`) and, for stage B, `--dagger-seed 1` changed. B_s1.A = B_s0.A with `--seed 1 ... --steps-mult 4`; U_s1.A = U_s0.A with `--seed 1 ... --fit-join --steps 19164`; stage B templates keep `--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --dagger-p 0.1 --w-acc 49.6969 --detach-upper` and the placeholders `@INIT@ @BANK@ @BANKSHA@`. The checker's own `diff` reads PASS on all four against B_s0 (stage A: `B_s0.A`; stage B: `B_s0.B.tmpl`), and a mutated seed 3 reads REFUSED.
* Token path: the checker already lists B_s1 / U_s1 (`ARMS`, seed 1, dagger-seed 1). **Stage-A tokens do not exist**; `block2_prep.sh` mints them inside the chain (CPU dry records `dscale10k_check.py dry`, `init_s1.py` G-HYG / C-INIT, `reach_probe2.py` G-LIVE, `assemble_evidence_s1.py`, `chk_arm.sh mint ARM A`, `verify_token_b2.py`). **Stage-B tokens** are minted in-chain after each bank exactly as block 1 did. The only GPU jobs of the prep are the two reach probes (about 15 s each in block 1), and they run inside the chain, under the lock, with the idle-GPU wait. No hand-written token.
* G-SUITE / G-EVAL / G-CKPT / G-CLOCK / C-CACHE evidence is the block-1 item record. `init_s1.py` REFUSES if the pod tree manifest is not the one the block-1 suite item ran on, so a changed tree forces a re-run instead of reusing stale evidence.
* Changed pod files: `chk_arm.sh` (U_s1 added next to U_s0 in the two union cases; backup `chk_arm.sh.before_block2`, delete after block 2 is verified), new `block2.sh`, `block2_prep.sh`, `block2_files/*`, `gate/dryrun_block2.sh`, `dscale_banker.py`, argv files.
* Checks: `bash -n` on all shell files; dry run `gate/dryrun_block2.sh` on the pod (fake leaves, REAL chainlib and banker with a directory as hub): 27 of 27 scenarios PASS (order; banker writes 4 markers; shards swept and logged; rerun skips; prep failure; tokens present; a conflicting marker defers one arm's deletion and the chain finishes; disk guard; STOP; failed arm; refused mint). The dry run deleted its own fake root.

**Launch (MM), after the CKPT120 chain has released the lock:**
`nohup setsid flock /root/gpu_queue.lock bash /root/dscale10k/block2.sh < /dev/null > /root/dscale10k/block2.log 2>&1 &`

## 3. Open items

1. The in-chain prep is first run on real data by the chain itself: the U_s1 dry build writes `join_union.npz` (as `--fit-join` does in every U run); the token binds that file after the dry, and no other job reads it meanwhile. If the prep refuses, the chain ends `FAIL` before any training with the checker's reason in `gate/block2_prep.out`.
2. Estimated duration (E, from block 1 timings): about 4.5 GPU-h plus the U bank.
3. R_s0 is banked and its release is swept by the first `sweep_all` (marker present).
4. Land with the study folder: `dscale_banker.py`, `block2*`, `chk_arm.sh`, the tests.

## Deliverable manifest

| artifact | where |
|---|---|
| banker + test | `F:/TanitAD-scratch/datascale/gate/banker/{dscale_banker.py,test_dscale_banker.py}`; pod `/root/dscale10k/dscale_banker.py` |
| block-2 chain and helpers | `F:/TanitAD-scratch/datascale/gate/block2/` (`block2.sh`, `block2_prep.sh`, `chk_arm.sh`, `gen_argv_s1.py`, `init_s1.py`, `assemble_evidence_s1.py`, `verify_token.py`, `fake_leaf2.sh`, `dryrun_block2.sh`); pod `/root/dscale10k/{block2.sh,block2_prep.sh,chk_arm.sh,block2_files/,gate/dryrun_block2.sh}` |
| banked checkpoints | HF private dataset `Sayood/tanitad-v8f-dscale10k-runs` (revisions above); pod `/root/dscale10k/bank_ok/`; ledger `/root/dscale10k/banker_ledger.jsonl` |
| this file | `F:/TanitAD-scratch/datascale/gate/BUILD_BANKER_BLOCK2.md` |

Not in a repo yet (F: scratch and pod only): all of the above until the MM lands them.

## Deletions (own temporary data)

* `F:/TanitAD-scratch/datascale/gate/block2/_render/` (argv render scratch, 36 KB, deleted).
* Pod `/root/dscale10k/fake_block2` (dry-run fake root, deleted by the script, absent now), pod `tmp/mut_B_s1.A.json` (mutation test file, deleted).
* Pending, not mine to delete before verification: `/root/dscale10k/chk_arm.sh.before_block2`.
