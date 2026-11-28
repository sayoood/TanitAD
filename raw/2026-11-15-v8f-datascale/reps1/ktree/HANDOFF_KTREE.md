# HANDOFF_KTREE (builder agent, 2026-11-19 10:47Z): K-TREE control of UM-A3 R1, QUEUED on the LoRA pod

## State
* `/root/dscale10k/ktree/ktree_chain.sh` is waiting on `/root/gpu_queue.lock` (blocking `flock 9`), log `/root/dscale10k/ktree.log` (line 1: `KTREE queued`). Nothing has trained.
* The REPS1 chain holds that lock for its whole run: `queue_reps1.sh` does `exec flock "$LOCK" bash blockreps1.sh` (`queue_reps1.sh` last line), so K-TREE starts only after UM_s1, its post check, the banker and the DONE/FAIL line. UM_s1 stage A started 09:03Z (about 3.2 h of trainer time, so about 12:15Z).
* After the lock: STOP-file check, disk >= 20 GB, GPU < 3500 MiB (up to 30 min), then A, B, C, then `ktree_result.py`. Ends `ZZ KTREE DONE ZZ` or `ZZ KTREE FAIL ZZ <reason>`. Stop it: `touch /root/dscale10k/chains/STOP`.
* Idempotent: a finished run (summary.json) is skipped, a partial run folder is moved aside to `<out>.partial.<time>`, a second start waits on the same lock and exits on KTREE_DONE.

## Runs (all seed 1, UE_s1's argv, `--steps 20 --log-every 1 --ckpt-every 20`)
* A = tree (cdbff971) -> `/root/dscale10k/ktree/A_tree`; B = tree_um (648d4ca3), no `--tac-src-*` -> `B_treeum`; C = A again -> `C_tree_rep` (beside, no bar).
* Argv diff (`ARGV_DIFF.txt`): A vs B differ in the trainer path and `--out` only; A vs C in `--out` only. UE_s1's argv has no `--tac-src` token (asserted). CPU argparse check of all three argv on both trees passed; on tree_um `tac_src_w` = 1.0 and clips `''`, so `make_tac_src` returns None (no stage built, no `tac_w` key).
* Env per run mirrors the runner (`dscale10k_check.py:1037-1039`): PYTHONPATH = tree/stack:tree, CUDA_VISIBLE_DEVICES=0, HF_HUB_DISABLE_XET=1, plus chk_arm_reps1.sh's thread and temp vars.

## How bit identity is read
* PRIMARY: `train_log.jsonl` step records 1..20 (`--log-every 1`; json.dumps of `float(T[k])`, shortest repr, full precision) for every term except wall-clock `sec`. Compared as strings.
* BESIDE: sha256 of every tensor of the final model state dict (`ckpt_last.pt["model"]`), plus a digest of those. Bit-equal parameters after 20 equal steps imply equal losses.
* `KTREE_RESULT.json` holds per-step losses (repr, plus hex of `total`), `A_vs_B_bit_identical`, `A_vs_C_bit_identical`, first differing step and term, state-dict digests, device, torch, both commits, `K_TREE_PASS`.
* Read: PASS = A_vs_B True. If False and A_vs_C True: the difference is the commit, print UM - UE as "lever plus commit". If both False: this device is not bit-reproducible at 20 steps and A-vs-B cannot attribute the commit; say so.
* Nondeterminism: no `use_deterministic_algorithms` or `cudnn.deterministic` anywhere in the trainer (grep, 0 hits); amp is off (fp32); per-step RNG is seeded (`torch.manual_seed(seed*1000003+step)`); one prefetch thread; DAgger off (p 0). Float atomics in CUDA backward kernels remain possible, which is why C exists.

## Launch-gate rule
The trainer has no token check of its own; the token is enforced by the runner (`dscale10k_check.py run_arm`, lines 1024-1039), bound to an arm's exact argv. A 20-step control with a new argv cannot carry one. It is started the way the study's smokes are: the trainer called directly (`/root/dscale10k/smoke.sh` lines 1-13, `block0 pilot_run.py`), with the trainer's in-process G-DVB / G-LIVE gate still running. No refusal is expected; if the trainer refuses, the chain FAILs with its text and nothing is hand-written.

## Waiter
`C:/Users/Admin/actsens_launch/wait_ktree.sh` (not started): 300 s poll, ends on the KTREE sentinel or the driver gone twice, cap 6 h. The MM starts it.

## Next step (fresh agent / MM, after the sentinel)
Pull `/root/dscale10k/ktree/KTREE_RESULT.json` and `ARGV_DIFF.txt`, read K_TREE_PASS and A_vs_C, put the result beside the UM-A3 rows. Delete the three run folders (about 3 x 0.3 GB, weights kept for nothing) after the result is pulled and its sha256 checked, and log it.
