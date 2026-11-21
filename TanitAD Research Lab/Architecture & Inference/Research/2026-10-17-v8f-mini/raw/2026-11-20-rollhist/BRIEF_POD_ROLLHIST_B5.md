# Brief: ROLLHIST B5: repoint the pod chain to the ported commit 43319227 (light pod work only; do NOT launch)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing.
* No secrets. Never write an IP, port or pod id into any file.

## Context
* The code of record is now **433192275ba50f11c67bfc5df2e110d9cf465df2** (branch `agent/v8f-rollhist-cdb-20261120`, parent cdbff971, pushed). See card Amendment 1: `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_AMENDMENT_1.md`.
* The port records are in `F:/TanitAD-scratch/rollhist/build/b4port/`. Read:
  * `HANDOFF_B4.md`;
  * `GATE_KIT_CHANGES.md` (the kit's commit placeholder);
  * `POD_SCRIPT_LINES.txt` (every pod-script line naming 2bda9e1a, `/root/rollhist/tree`, or the G-SUITE base);
  * the patched kit in that folder.
* The pod chain `/root/rollhist/` (launch lines in `F:/TanitAD-scratch/rollhist/build/b2/HANDOFF_B2.md`) ran once on 2bda9e1a. It stopped at G-SUITE: `ZZ ROLLHIST FAIL ZZ step gsuite`, 21:14:28Z. Steps inputs, disk0 and render had written done markers.

## Pod rules (another session's trainer holds the GPU until about 01:25Z)
* Allowed: tar push and extract, `sha256sum`, `cat`, `ls`, writing scripts, `bash -n`, and a sub-second python check.
* ⛔ No pytest, no model load, no GPU.
* ⛔ Never touch `/root/gpu_queue.lock`, `/root/N1DW_READY`, `/root/ruled`, `/root/strat1`, `/root/dscale10k` (read only), or any process.
* Run `source C:/Users/Admin/lorapod/podenv.sh` for `pssh` / `pscp` with `"$POD_TARGET:/path"`. No `pgrep -f`.

## Do
1. **Tree.** `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 43319227 stack taniteval` -> `/root/rollhist/tree_cdb`. Verify it blob by blob against 43319227 and report the count.
2. **Gate kit.**
   * Fill the commit placeholder with the full 43319227 sha.
   * Install the patched kit (`dscale10k_check_rh.py`, `chk_arm_rh.sh`, `rh_evidence.py`, and `rh_arm.sh` if B4 changed it) into `/root/rollhist/` and `/root/rollhist/study/`, next to the current ones. Keep the current ones as `*.b2bda`.
   * sha256 every installed file against your local copy.
3. **Repoint.** Change every line listed in `POD_SCRIPT_LINES.txt` to the new commit and `tree_cdb`, keeping `.b2bda` copies, then run `bash -n`.
   * The G-SUITE base stays `/root/strat1/junit/base`: it is the cdbff971 run, which is right now.
   * Check that the pilot's check (a), "defaults pilot of the new tree vs the unpatched DSCALE10K tree", points at a cdbff971 tree. Read which tree `/root/dscale10k` used and report it.
4. **Done markers:** move `/root/rollhist/done` (or wherever the markers are) aside to `done_2bda9e1a`, so every step re-runs on the new tree. Do the same with `RH_READOUT.chosen` only if a check binds it to the commit; otherwise leave it. Record both in `/root/rollhist/deletions.log` as moves, not deletions.
5. **Launch lines.** Write the exact launch lines for the MM: marker watcher + chain, `RH_READOUT=causal_g1x`, the same form as HANDOFF_B2.

## Deliverables in `F:/TanitAD-scratch/rollhist/build/b5pod/`
* the repointed script copies, with a diff against the b2 versions;
* `REPORT_B5.json`: tree count, the shas of every installed file, the dscale10k tree finding, the markers moved;
* `HANDOFF_B5.md` (at most 600 tokens) with the launch lines.

Hand back in at most 200 words. Never wait more than about 4 minutes in one call.
