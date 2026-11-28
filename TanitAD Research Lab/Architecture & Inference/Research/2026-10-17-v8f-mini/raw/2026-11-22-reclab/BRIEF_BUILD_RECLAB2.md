# Brief 2: RECLAB, the label becomes a SPEED-consistent label (not a pose recovery); finish, prove on a real shard, checker, read extension. Build and dry-check only: run nothing on the pod

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first (`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`). Then read, in `F:/TanitAD-scratch/reclab/`: `BRIEF_BUILD_RECLAB.md` (brief 1; its rules bind), `FACTS_RECLAB.md`, `HANDOFF_RECLAB.md`, `DRY_MINT_PLAN.md`. You continue the work in `tree97/` (base 97fe0a56, the remote tip of `agent/v8f-rollhist-cdb-20261120`).

Rules (as brief 1): stage, commit and push nothing; no GPU; no protected files; no secrets, IPs, ports or pod ids in any file; pod read-only; one python process at a time on the dev box, gated on commit >= 5.5 GB; never wait more than about 4 minutes in one call; hand off at about 150k context.

## The MM's design change, and why (MEASURED from source, `tree97/stack/tanitad/data/v8f_dagger.py`)
`swap_rows` re-anchors the own poses by the rigid motion that maps the own NOW onto the LOGGED NOW at r' (`reanchor_rows` :161, called in `swap_rows`, comment "the own poses in the LOGGED frame at r'"). The scene is logged and nav channels 0-2 and 4 stay logged. So in a swapped sample **the own pose offset (e_lat, dpsi) is erased from every input**: NOW sits on the human's path at r', with the human's heading. A label that recovers a pose offset (brief 1's `recovery_v1`) would then tell the model to rejoin a path that every input shows it is already on: a label no input can predict. What the re-anchoring does NOT erase is the own KINEMATICS: the own history's speed (`p10f`, `kin3`) against the human label, which starts at the human's speed. That speed discontinuity is the conflict this arm tests.

## The mechanism to build (replaces `recovery_v1`)
* Flag values `--dagger-label {human,speed_v1}`; remove `recovery_v1` from the trainer and its off-route eligibility path (keep the library functions the new value reuses; keep tests that still apply). Default `human`, bit-identical to 97fe0a56 (the gold test stays).
* `speed_v1`, for matcher-`ok` records only (off-route, end_edge, nonfinite: dropped exactly as today; the eligible index must be EQUAL under both values):
  * v_h[k], k = 0..30: the logged speed of the clip at rows r'+k (the store's speed column, as V1 reads it); v_own = the own speed (V1's source, `p10f[:, -1, 0]`).
  * v_l = V1's `label_speed(v_own, v_h)` (starts at own speed, rate-limited at 0.99 A_MAX, follows the human once equal).
  * the label's arc offset against the human: delta[0] = 0, delta[k] = delta[k-1] + 0.5 * ((v_l[k-1] - v_h[k-1]) + (v_l[k] - v_h[k])) * 0.2.
  * the human polyline P = the logged rows r', r'+1, ... (as many as the clip has) in the human's ego frame at r' (so P[1..30] is exactly the human label); s = its cumulative arc. Label point k = the polyline point at arc s[k] + delta[k] (`_at_arc`). Refuse (keep today's human label) if that arc runs past the polyline's end, if the arcs are not non-decreasing, or if V1's feasibility checker (`violations` with the own speed as predecessor) finds any violation.
  * identity: when delta is all exactly 0, the label is the human label `np.array_equal` (return it directly).
  * `s_h` is re-derived from the label by the existing re-derivation; every other label unchanged (FACTS section 1).
* `--selfcheck-only` (or its info record) reports a census on the real bank: per tick, counts of ok-relabelled, ok-identity, ok-refused (by reason); p50 / p95 of |v_own - v_h[0]| and of |delta[30]|; the relabelled share of ok records.

## Priority order (bank each part as it is done)
1. The mechanism above in `tree97`, with tests: identity; gold under `human`; reachability (relabelled rows > 0 under `speed_v1`, exactly 0 under `human`; eligible index equal under both); mutants that must go RED: (a) delete the install line, (b) use the human speed in place of the own speed, (c) skip the refusal, (d) jump the speed (`clip=False`). Then fix and re-run the three tests brief 1 left failing (`test_the_single_applying_lines_exist_exactly_once`; `test_kb5_...` in the dagger-first order; the compose test against a control run on the unmodified 97fe0a56 files, to show whether it was failing before). Run whole files, never `-k`.
2. **Real-shard proof**: pull ONE shard of `/root/rollhist/bank/s0` (at most 600 MB) and the clip grids it needs into `F:/TanitAD-scratch/reclab/tmp/`, run the census and the reachability counts on it, write `REAL_SHARD_CENSUS.json`, then delete the pulled files by literal path and log them in `DELETIONS.log`.
3. Checker: apply `pod/patch_rl.py` (kind RL now names `--dagger-label speed_v1`), test it, and run its five mutants (a second changed flag; another bank or sha256; `human` under the RL name; the flag missing; dose 0.3): each must give a non-empty C-ARGV problem list. Update `rl_argv.py`, `rl_arm.sh`, `DRY_MINT_PLAN.md` to `speed_v1`.
4. Read extension `RH_AKIND=RL` in `F:/TanitAD-scratch/rollhist/read/` (pattern: RHK in `rh_common.py`, `dry_check.py`, `rh_read.py`; re-export the read tree from the mirror if it is missing, as its handoffs describe). `dry_check.py` must PASS under RL and at the default. Keep `.bk` copies of any file you edit.
5. `FACTS_RECLAB_2.md` (what changed and the file:line of every hop: argparse -> namespace -> swap -> `gt` / `s_h` -> loss), `CHANGED_FILES.txt` with every changed file's sha256, `HANDOFF_RECLAB2.md` (at most 800 tokens).

## Disk
F: is shared: your folder stays under 1.5 GB apart from the one pulled shard, which you delete as soon as step 2 is done; F: free stays >= 15 GB. Delete your own temp at the end (tmp/pt*, mutants/tree, research/) by literal path and log it.

Hand back in at most 250 words with a deliverable manifest, saying plainly which checks PASSED and which did not run.
