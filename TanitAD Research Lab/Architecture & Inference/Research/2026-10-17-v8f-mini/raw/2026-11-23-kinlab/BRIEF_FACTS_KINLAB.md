# Brief: KINLAB phase 1, facts and a kinematic-gap census for the ROLLHIST line's next lever (the kinematic label). Zero training, no code change in the repo

You are a TanitAD build agent working for the Master Mind (MM). First read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

## Rules
* Stage nothing, commit nothing, push nothing.
* No GPU.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* No secrets, no IPs or ports in any file.
* The LoRA pod is training (UE_s2 / UE_s3 hold its GPU until about 06:00Z). Only READ-ONLY pulls from it are allowed (`pcat.sh` / `pscp`), at most about 1.5 GB in total. Never run anything on the pod.

## Why
On the ROLLHIST line, stage B trains on the model's own rolled history (DAgger). The own states are swapped into the batch (RH), or appended (RA).

* **What it costs:** the T0 forecast on logged history (G-T0 fails at s0 on RH, RL and RA).
* **What it buys:** large own-action gains at T1.
* **What has been ruled out:**
  * the bank readout;
  * the speed label (RECLAB, `--dagger-label speed_v1`: MECH-NO);
  * replacement (ROLLADD, `--dagger-mode add`: MECH2-NO).

The ROLLADD card commits the next lever, the **kinematic label**. In the own-state sample, the yaw-rate and acceleration targets would blend from the own history into the human's, under the speed label's feasibility envelope.

Before the MM designs it, we need two things. First, exactly what the label and the kinematic inputs are. Second, how large the own-versus-human kinematic gap at NOW actually is. If the gap is negligible, the lever cannot work.

## Read first
* Code at commit **df250a00** (branch `agent/v8f-rollhist-cdb-20261120`; export a tree with `git -C C:/Users/Admin/tanitad-push archive df250a00 stack taniteval`, no worktree):
  * `stack/tanitad/data/v8f_dagger.py`: `swap_rows`, `reanchor_rows`, `append_rows`, `DaggerBank`, `attach_recovery`, and the speed_v1 path around :615-640;
  * `stack/tanitad/data/v8f_recovery.py` (`label_bank`, the speed_v1 label);
  * `stack/scripts/v8f_mini_train.py`: `--kin-mode yaw`, the build of `kin3` and `p10f`, and the plan-label tensors `gt` / `s_h`;
  * `feasible_decode.py` (A_MAX and any yaw / jerk limits).
* The RECLAB card `F:/TanitAD-scratch/reclab/PREREG_V8F_RECLAB.md` (section 1, risk R2; section 2, semantics) and its census script `F:/TanitAD-scratch/reclab/real_shard_census.py` (how a bank shard was pulled and read on the dev box).
* The ROLLADD card `F:/TanitAD-scratch/rolladd/PREREG_V8F_ROLLADD.md` (the next-lever row) and the verdict `F:/TanitAD-scratch/rolladd/readra/MM_VERDICT_ROLLADD.md`.

## Priority order (bank each part as it is done)
1. **`FACTS_KINLAB.md`**, every item with a `file:line`:
   * What exactly are the inputs at NOW for a swapped / appended sample: the `kin3` channels under `--kin-mode yaw` (which channel is the yaw rate, and in what units and frame), the `p10f` channels, the pose history.
   * What `reanchor_rows` changes and what it leaves as the own value (speed? yaw rate? acceleration?).
   * What the plan label is (`gt` [n, 30, 2] positions at 0.2 s? `s_h`?) and every other label of a swapped row: tactical, lead, speed or acceleration heads. Which of those are human values and which are own values after the swap.
   * Which losses read them.
   * The limits available for a feasibility envelope (`feasible_decode.py`: A_MAX, any yaw-rate, curvature or jerk limits).
2. **The census** (`census_kinlab.py`, CPU, one python process, dev box), on the RH bank s0. Pull read-only one or two shards as `real_shard_census.py` did, into `F:/TanitAD-scratch/kinlab/tmp/`, and delete them afterwards.
   * Over the ok records that a swap would use, report the own-versus-human gap at NOW and over the first 1 s and 2 s:
     * yaw rate (own kin3 value against the human's at r', from the human pose polyline's heading change);
     * acceleration;
     * speed, for a control: it must reproduce the RECLAB census's |v_own - u_h[1]| figures within rounding.
   * Give p50 / p90 / p95 / p99 of the absolute gaps, by tick and pooled, with units.
   * Report the share of records whose yaw-rate gap exceeds 1, 2 and 5 deg/s, and whose acceleration gap exceeds 0.5, 1 and 2 m/s².
   * Beside it, report the same quantities between two consecutive LOGGED windows (human vs human, the natural variation) as the scale.
   * Output: `KINLAB_CENSUS.json` plus a short `KINLAB_CENSUS.md` (what ran, n, the table, what is not covered).
3. **`HANDOFF_KINLAB.md`** (at most 600 tokens): what the label is, the measured gaps, and which design choices the facts force or rule out. For example: is a yaw-rate blend even expressible in a positions-only label; does re-anchoring already erase the yaw-rate gap. Do not design the lever. The MM designs and freezes it.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time.
* Gate each python run with `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`. If the gate does not open within 60 min, use 4.0 for a census that stays under 1.5 GB of RAM, and say so.
* ⛔ C: has about 2 GB free: write nothing to C:. Set TMP and TEMP to `F:/TanitAD-scratch/kinlab/tmp`.
* F: must keep at least 15 GB free. Your folder stays under 2 GB and is under 50 MB after cleanup.
* Never kill a process you did not start, and never use a blanket `taskkill` or `pkill`.
* Never wait more than about 4 minutes in one call. If your context passes about 150k, write HANDOFF_KINLAB.md and end.

## Deliverables
In `F:/TanitAD-scratch/kinlab/`:
* FACTS_KINLAB.md;
* census_kinlab.py, KINLAB_CENSUS.json and KINLAB_CENSUS.md;
* HANDOFF_KINLAB.md;
* MANIFEST_SHA256.txt and DELETIONS.log.

Hand back in at most 250 words with a deliverable manifest, and say plainly what ran and what did not.
