# BUILD: DSCALE10K block 3 (UE_s0, BL_s0), build agent, 2026-11-17

Built under `MM_RULING_BLOCK3_STEPS.md`. Nothing launched, committed or minted. On the pod I wrote only under `/root/dscale10k/block3/` (scripts, `study/`, `argv_render/`, `REGRESSION_EVIDENCE.json`); the dry run's fake roots were deleted by the script. The lock, the running jobs, `lp3_sampler.py` and `bank_ckpts.py` were not touched. Local copy of everything: `F:/TanitAD-scratch/datascale/gate/block3/` (scripts in `block3_files/`, same bytes as the pod).

## 1. The checker fix (MM ruling), and is the checker a G-SUITE file?

* Fixed copy: pod `/root/dscale10k/block3/study/dscale10k_check.py`, sha256 `89fd80a6fc95acbf82826578d137150218a6b4139c91c24c85fe1dc7c8244f70`. The copy in `/root/dscale10k/study/` was left byte-for-byte as it was (sha256 `7467e5d7c05ed00971e229dd9c293b7be9a28fa76b998b2aa51e2ea3aed8accc`), because the block 1 and 2 tokens bind it. Diff: `CHECKER_DIFF.patch`; tests: `TEST_DIFF.patch`, `study/test_dscale10k.py`.
* The change: `steps_problems(..., u_rec=None)`. UE_s0 = 4 x S_U of its own union record. BL_s0 = 4 x S_U of the UNION record `u_rec` (U_s0's dry record), REFUSED if `u_rec` is missing or its `n_clips` is not the union's 14,512. `ue_steps` (the evidence value) must equal that target. `mint` takes a new `--ue-dry` (U_s0's dry record; its sha256 goes into the C-STEPS check). No bar, arm, dose or other check changed.
* **Is it a G-SUITE file? No.** G-SUITE is the identity item in `gate/suite.json`: `tree_manifest_sha256` over `TREE_DIRS` of `/root/dscale10k/tree` equals `/root/strat1/base` (`mode: identity`). `dscale10k_check.py` is not in that tree (`find tree -name 'dscale10k_check*'` is empty) and `tree_manifest_sha` does not cover `study/`. `suite.json` only carries the study-test tail (47 passed, 1 skipped) as information. So no G-SUITE re-run is needed; `init_b3.py` still REFUSES if the tree manifest changes. The study tests were re-run with the fixed checker: **47 passed, 1 skipped**, the same count as before (the new steps asserts are inside it). Run against the old checker, the new steps test FAILS (TypeError), so it is not vacuous.
* Regression evidence on the real dry records (`REGRESSION_EVIDENCE.json`, `block3_files/regress_steps.py`, pass true). PASS: UE_s0 argv diff and C-STEPS, BL_s0 argv diff and C-STEPS. REFUSED: BL at 19,164; BL at 61,569; UE at 61,569; BL with the target taken from the B record (no `u_rec`); the B record passed as `u_rec`; evidence `ue_steps` 19,164; BL `--steps` 61,569 against the pinned `ue_steps`; UE without `--steps`.
* Binding: tokens bind `checker_sha256` of the file that runs. `chk_arm3.sh` runs `block3/study/dscale10k_check.py`, so block-3 tokens bind `89fd80a6...`. The chain mints inside itself, so launching block3.sh is the first mint and the fix is in place before it.

## 2. Arm argvs (rendered by `gen_argv_b3.py` on the pod into `block3/argv_render/`; the prep regenerates the same into `argv/` at run time)

S_U MEASURED 15,392 (`dry/U_s0.A.dry.json`), so 4 x S_U = **61,568**. Both seed 0, stage A only.

| arm | argv | file sha256 |
|---|---|---|
| UE_s0 | U_s0.A with `--steps 61568` (replaces 19164), `--steps-mult 4` and `--fit-join` kept, `--out runs/UE_s0/A` | `1f358372493b011bfc98b77773f0dc3e11b45e3a5cc9462aa47156ac3220228f` |
| BL_s0 | B_s0.A + `--steps 61568`, `--exclude 540:600` kept, `--out runs/BL_s0/A` | `11b42563e761a4e3077a3feaf378de17fda271f565ca7b65830ce7ef6d8cc694` |

The checker's own argv hashes are in `REGRESSION_EVIDENCE.json` (`argv_sha256_checker`). Mint and gate tokens per arm are minted by the chain's prep (`block3_prep.sh`, steps 1 to 6, ending `chk_arm3.sh mint ARM A` and `verify_token3.py ARM A`). They could not be minted here: the prep needs the two GPU reach probes and writes under `/root/dscale10k/gate`, outside block3/. `init_b3.py`, `assemble_evidence_b3.py` and the prep are therefore first run on real data by the chain itself, as block 2's prep was. They are `py_compile` / `bash -n` clean, and the dry run covers the prep's calling path with a fake leaf.

## 3. Chain, diff against block2.sh (`DIFF_block3.sh_vs_block2.sh.txt`)

Only these changed: header; `STEPS_A=61568` and `B3`; the sentinels `ZZ DSCALE10K BLOCK3 DONE|FAIL ZZ`; the runner and prep paths (`$B3/chk_arm3.sh`, `$B3/block3_prep.sh`); the pre-flight file list (plus an assertion that 4 x S_U of `dry/U_s0.A.dry.json` equals 61,568); the two token loops and the arm list (`UE_s0 BL_s0`); the arms `do_arm UE_s0 no`, `do_arm BL_s0 no` (no DAgger bank, selfcheck or stage B). Kept: gate_wait (STOP, >= 60 GB, idle GPU), chk_arm, the stage-A summary and post-training zero-grad gate (252 registered tensors), the banker after each stage (tags `UE_s0.A`, `BL_s0.A`), the sweep, the bank at start. `chk_arm3.sh` diff: `DIFF_chk_arm3_vs_block2.txt` (checker path, PYTHONPATH with block3/study first, UE_s0 in the union cases, `--ue-dry` for UE and BL). `block3_prep.sh` diff: `DIFF_block3_prep_vs_block2.txt`.

`queue_block3.sh` waits for `^ZZ STRAT1 C0REP (DONE|FAIL) ZZ` in `/root/strat1/logs/c0rep.log` (the log held only its queue line when I looked), then `exec flock /root/gpu_queue.lock bash /root/dscale10k/block3/block3.sh`. It ends `ZZ DSCALE10K BLOCK3 FAIL ZZ queue` after 48 h.

## 4. Dry run (CPU, no GPU): `bash /root/dscale10k/block3/dryrun_block3.sh` = **29 of 29 PASS**, then `ZZ DRYRUN3 DONE ZZ`

Fake leaves, the REAL `chainlib.py` and `dscale_banker.py` (a directory as the hub). Covered: order prep, UE_s0 A, BL_s0 A and nothing else; two markers written; both checkpoints at step 61,568; ARM_DONE; rerun skips; prep failure (nothing trained); tokens present (no prep); disk guard; STOP; UE fails (BL never runs); BL fails (UE already banked); the queue wrapper (waits, times out, a mid-line quote does not release it, DONE and FAIL lines release it, it waits for a held lock then runs the chain under it).

## 5. Projection (ESTIMATED; rates MEASURED in `runs/<arm>/A/summary.json`)

0.181 to 0.197 s/step in blocks 1 and 2 (union arms 0.1882, 0.1883; B arms 0.1881, 0.1812, 0.1972). UE_s0 61,568 x 0.1883 = 3.22 h; BL_s0 3.22 h (3.37 h at 0.1972). **About 6.5 h**, against the card's 7.2 h (worst case 12.6 h; 1.5 x = 18.9 h): inside the stop rule. Free disk was 122 GB (guard 60 GB).

## 6. MM launch (its own ssh command, not inside an `&&` list)

`nohup setsid bash /root/dscale10k/block3/queue_block3.sh < /dev/null > /root/dscale10k/block3.log 2>&1 &`

It can start now: it waits for C0REP, then queues on the lock. Sentinel: the last `ZZ DSCALE10K BLOCK3 (DONE|FAIL) ZZ` line of `/root/dscale10k/block3.log`. Open item: the prep runs for the first time on real data inside the chain; if it refuses, the chain ends FAIL before training with the checker's reason in `gate/block3_prep.out`. The U_s0 dry record must stay in place (it is).

## Deliverable manifest

| artifact | where |
|---|---|
| fixed checker, its tests | pod `/root/dscale10k/block3/study/`; local `.../block3/study/`; diffs `CHECKER_DIFF.patch`, `TEST_DIFF.patch` |
| chain and helpers (`block3.sh`, `block3_prep.sh`, `chk_arm3.sh`, `queue_block3.sh`, `gen_argv_b3.py`, `init_b3.py`, `assemble_evidence_b3.py`, `verify_token3.py`, `fake_leaf3.sh`, `dryrun_block3.sh`, `regress_steps.py`) | pod `/root/dscale10k/block3/`; local `.../block3/block3_files/` |
| rendered argvs | pod `/root/dscale10k/block3/argv_render/{UE_s0,BL_s0}.A.json` |
| regression evidence | pod `/root/dscale10k/block3/REGRESSION_EVIDENCE.json`, local copy; `DIFF_*.txt` |
| this file | `F:/TanitAD-scratch/datascale/gate/block3/BUILD_BLOCK3.md` |

All of it is on the pod plus F: scratch only; nothing is in a repo yet. Land with the study folder: the `block3/` files, the fixed checker and its test.

## Deletions (own temporary data)

Local: `block3/study_orig/` (pulled originals, about 100 KB; the diff patches hold the change), the two `*.draft.json` files, `__pycache__`. Pod: the dry run's `fake_root3` and `fake_q` (by the script), pytest basetemps under `block3/tmp/` (literal paths), the empty `block3/tmp`. No other pod files were deleted.
