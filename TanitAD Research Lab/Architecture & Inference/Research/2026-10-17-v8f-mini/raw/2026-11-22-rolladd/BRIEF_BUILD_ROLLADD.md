# Brief: build `--dagger-mode add` (own rolled states APPENDED to the batch, logged samples kept whole) into the v8f DAgger swap, with its proofs, the RA arm's pod chain and the read extension. Build and dry-check only: run nothing on the pod

You are a TanitAD build agent working for the Master Mind (MM). First read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands and launches.
* No GPU.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* No secrets, and no IPs, ports or pod ids in any file.
* This is a zero-spend build; implementation around a pending PI go is the MM's. Running the arms needs a PI go with the cost.

## Why (read these; they bind)
* Card draft: `F:/TanitAD-scratch/rolladd/PREREG_V8F_ROLLADD_DRAFT.md` (MM design; the arm, the reads, the outcomes).
* Previous verdict: `F:/TanitAD-scratch/reclab/readrl/MM_VERDICT_RECLAB.md`.
* The build this one copies, file for file: `F:/TanitAD-scratch/reclab/` (`FACTS_RECLAB*.md`, `CHANGED_FILES.txt`, `mutants/`, `pod/cf27/README_POD_CF27.md` and its scripts, `DRY_MINT_PLAN.md`).
* Code of record: **cf27dbb7** on `agent/v8f-rollhist-cdb-20261120`.

## The mechanism (MM design; flag it in FACTS if the source contradicts any line)
* **Flag** `--dagger-mode {swap,add}` on `stack/scripts/v8f_mini_train.py`.
  * Default `swap` must reproduce cf27dbb7 **bit for bit**: a gold test on a fixed draw set, as the RECLAB golden `stack/tests/golden_v8f_dagger_97fe0a56.py`.
  * FLAG_CHECKS entry `"dagger_mode": "dagger:dagger_mode"`, read back from the built `DaggerSwap` (`built_levers()`).
  * `add` is refused at `--dagger-p 0` (a lever declared that does nothing).
* **`add` in `DaggerSwap.apply`** (`stack/tanitad/data/v8f_dagger.py` :733):
  * The selection is the same: the same generator, `u`, `sel`, `rows` and `pick`, and the same `swap_rows` call.
  * The output batch is the logged batch, unchanged bit for bit, followed by the N rows `swap_rows` produced for `rows`.
  * Concatenate every per-row entry along dim 0, including nested dicts (`lab_s` and anything else). Non-per-row entries stay as they are.
  * `self.last` gains `n_appended` (equal to `n_selected` under `add`, exactly 0 under `swap`) and the new row count.
* **Enumerate in FACTS before you code** (`file:line` for each):
  1. Every batch key and whether it is per-row.
  2. Every consumer between `dag.apply` and `loss.backward()` that assumes a row count of B: the post-stage label attachments (launch, level, lead), the composition stages (`EgoDaggerCompose`, `rcv`, `t2b`, `ego`), the counters, and `to_dev`.
  3. How every loss term is normalised: a mean over rows, or a fixed constant.
  4. The global-RNG draws inside `model.loss` (code mix, `p_gt`) and how they scale with the row count.
  * Every stage that cannot run correctly beside `add` must be REFUSED at gate time (G-DVB message), not approximated.
  * RH's argv (`F:/TanitAD-scratch/reclab/pod/src/RH_s0_argv.json`) uses `--w-acc 49.6969 --detach-upper --fit-join` and `--ego-perturb-p 0`, with no launch, level, lead, compose, recover or t2-bank. These must work under `add`; refuse the rest.
* ⛔ If a loss term is normalised by a fixed B, or a label tensor computed before the stage would misalign with the appended rows, STOP and report. Do not approximate.

## Priority order (bank each part as it is done)
1. **`F:/TanitAD-scratch/rolladd/FACTS_ROLLADD.md`**: the enumeration above. Bank it first; the MM freezes the card from it.
2. **The mechanism, in your tree, with tests:**
   * the gold test (`swap` = cf27dbb7 bit for bit);
   * `add` on a real batch: the first B rows `torch.equal` the logged batch, and the appended rows equal the `swap` output's `rows` bit for bit;
   * the refusals;
   * reachability, `file:line` per hop: argparse -> namespace -> `DaggerSwap` -> appended rows -> loss -> backward;
   * a NON-ZERO gradient difference between `add` and p = 0 on the model's parameters, and a control that reads EXACTLY 0.0;
   * mutants that must turn RED: no append; logged rows appended instead of the swapped ones; replace instead of append under `add`; `n_appended` not counted;
   * the whole dagger, v8f and recovery test files pass (run whole files, not `-k`).
   * Then the census: on one real RH bank shard (`/root/rollhist/bank/s0`; pull at most 300 MB) through `--selfcheck-only` or the swap on real batches. Report the appended rows per step (p50 and p95), the step's row count, and the CPU wall per step against `swap`.
3. **The RA pod chain**, staged in `F:/TanitAD-scratch/rolladd/pod/`, built the way `F:/TanitAD-scratch/reclab/pod/cf27/` was built:
   * tree prep for the new commit on top of `tree_cf27`, with a blob-by-blob check and the eval-grid row in the blob list (README_POD_CF27 explains why);
   * a G-SUITE script for the new commit;
   * a gate record of kind RA (the `rl_gate_record.py` pattern), refusing any other tag;
   * the checker: RA = the RH argv plus exactly `--dagger-mode add` and a new `--out`, with the bank sha pinned to RH's; mutants (a second changed flag, another bank, `swap` under the RA name) refused;
   * the chain under `/root/gpu_queue.lock`, keeping the `/root/N1DW_READY` marker protocol of `rh_rl_chain.sh` and `rh_rl_marker_watch.sh` unchanged (another session trains on that GPU between its slices).
4. **Read extension** in `F:/TanitAD-scratch/rollhist/read/` (keep `.bk` copies of any file you edit):
   * the RL extension's pattern with `RH_AKIND=RA`;
   * `dry_check.py` PASS under RA and the default;
   * MECH2 = RA - RH at T0, generalising `F:/TanitAD-scratch/reclab/readrl/mech_rl.py`.
5. **`DRY_MINT_PLAN.md`** and **`HANDOFF_ROLLADD.md`** (at most 800 tokens).

## Tree
* Export: `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive cf27dbb79f2491a6db1197860bc50ca7c8cdeff4 | tar -x -C F:/TanitAD-scratch/rolladd/tree`.
* Run tests with `PYTHONPATH` set to the tree's `stack` (Windows separator `;`), and assert that `tanitad.__file__` is inside your tree.
* Set `CUDA_VISIBLE_DEVICES=-1` (an empty value shows the GPU on Windows), assert that torch sees no CUDA, and pass `--basetemp` under `F:/TanitAD-scratch/rolladd/tmp`.

## Pod (`source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp "$POD_TARGET:/path" local`)
* Read and pull only, small files only.
* Never write there, run no python or pytest there, and never touch a process, lock or marker.
* Another session trains on the pod's GPU.

## Dev box
* CPU only, one python process at a time, each gated on commit >= 5.5 GB: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`.
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, with TMP and TEMP under your folder.
* ⛔ F: is shared and tight. Your folder stays under 1.5 GB, and F: must keep at least 15 GB free.
* ⛔ C: has about 7 GB free: write nothing there.
* Delete your own temp at the end by literal path, and log each deletion in `F:/TanitAD-scratch/rolladd/DELETIONS.log`.
* Never wait more than about 4 minutes in one call. If your context passes about 150k, write `HANDOFF_ROLLADD.md` and end.

## Deliverables
In `F:/TanitAD-scratch/rolladd/`:
* FACTS_ROLLADD.md;
* the changed tree files, listed with sha256 in `CHANGED_FILES.txt`;
* the test logs, `mutants/` and `pod/`;
* DRY_MINT_PLAN.md, HANDOFF_ROLLADD.md and MANIFEST_SHA256.txt.

Hand back in at most 250 words, with a deliverable manifest, and say plainly which checks PASSED and which did not run.
