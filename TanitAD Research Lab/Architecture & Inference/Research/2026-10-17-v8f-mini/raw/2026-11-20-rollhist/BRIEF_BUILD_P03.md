# Brief: build the ROLLHIST next arm "share p = 0.3 on the SAME bank" (pod chain + gate + read extension). Build and dry-check only: run nothing on the pod

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing; the MM lands and launches.
* No GPU, and no edits to protected files (launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, the sha-pinned v8f_t1_eval.py).
* No secrets, and no IPs, ports or pod ids in any file.

## Why
* The two-seed read of record is **ROLLHIST-PARTIAL** (`F:/TanitAD-scratch/rollhist/read/out/read/RESULT_ROLLHIST.json`): P1 BETTER on LAT, TAC and STRAT, LON MIXED, none WORSE; P2, P3 and P4 fail; G-T0 fails at s0 and holds at s1.
* The card's PARTIAL row (`F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST.md`, sha256 781eea87, section 3.4, line ~121) names the next arm: **share p = 0.3 on the SAME bank**. That is stage B only, K-B2 holds, about 0.27 pod-h per seed.
* The RH arms used `--dagger-p 0.1`. Read the card and Amendments 1 to 6 (`F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_AMENDMENT_*.md`). They bind.

## Priority order (bank each part as done)
1. **The arm spec + draft Amendment 7** (`DRAFT_AMENDMENT_7.md`; the MM freezes it):
   * Arms RHp3_s0 and RHp3_s1 = the RH_s0 / RH_s1 argv (`/root/rollhist/runs/RH_s*/argv.json`, read them on the pod) with exactly `--dagger-p 0.3` and `--out` changed.
   * The same banks (`/root/rollhist/bank/s0`, `s1`, sha256 in each `bank.json`), the same init, steps, seed and code of record 97fe0a56.
   * Comparator: the existing B0_s0 and B0_s1 (no new B0).
   * The read: card section 3 on RHp3 - B0 with the same cells, floors, estimator and tier. RHp3 - RH goes beside.
   * Outcome names: the card's table with its precedence, and a one-line answer to "did p = 0.3 repair P2/P3?".
   * Cost, ESTIMATED from the measured RH arm times in `/root/rollhist/rh_chain.log` and `runs/RH_s*/train.log`.
2. **Pod chain**, staged in `F:/TanitAD-scratch/rollhist/build/b8p03/` (the MM copies it to the pod):
   * `rh_p03_chain.sh`, modelled on `/root/rollhist/rh_chain.sh`. It holds the GPU queue lock to the end, follows the marker protocol (`rh_marker_watch.sh`), runs no bank step, and runs the arms through the existing `rh_arm.sh` / `chk_arm_rh.sh` if they take any tag; otherwise thin copies.
   * The study checker `/root/rollhist/study/dscale10k_check_rh.py` (sha256 69ff1a0f, Amendments 5 and 6) must mint for RHp3. Find, from source, every check that pins `--dagger-p 0.1` or the RH flag set (C-ARGV, C-RH-PAIR, the "fifth changed flag" rule, `rollhist_pair_check.py`). Change only what p = 0.3 needs.
   * Test the checker change with `F:/TanitAD-scratch/rollhist/build/b7grid/gatefix/test_dscale10k_check_rh.py` (40 passed; its fixture writes the merge's real shape). Add a test that RHp3 mints, and a mutation arm that p = 0.5 or a second changed flag is refused.
   * Write a `DRY_MINT_PLAN.md`: the exact commands the MM runs on the pod BEFORE the launch (`chk_arm_rh.sh mint RHp3_s0`, read `gate/checks.json` in full), because three pre-train refusals tonight came from paths only a pod mint exercises.
3. **Read extension** in `F:/TanitAD-scratch/rollhist/read/`:
   * Add arm RHp3 (rh_common.py kinds/forms/pairs; pull, rollouts and T0 for RHp3 only; B0 and RH dumps are reused).
   * Add a chain `rh_read_p03_chain.sh` and `.cmd` that reads RHp3 - B0 through the frozen readers and `rh_read.py`.
   * `dry_check.py` must PASS. Do not run any rollout.
   * Do NOT edit any file the finished read used without keeping a `.bp03` copy. The MM must still be able to re-run the PARTIAL read.

## Pod (`source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp "$POD_TARGET:/path" local`)
* The pod is RUNNING the N1DH chain (GPU lock held). You may read files and pscp them down, nothing else.
* No python and no pytest on the pod. Never write anything there, and never touch a process, lock or marker.

## Dev box
* CPU only, one process at a time, each gated on commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`).
* Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`. TMP/TEMP under your folder. Delete your own temp at the end, by literal path, and log it.
* F: has about 6.5 GB free: stop if it falls below 6 GB. Never wait more than about 4 minutes in one call.

## Deliverables
* `F:/TanitAD-scratch/rollhist/build/b8p03/`: DRAFT_AMENDMENT_7.md, the chain and any checker change with its tests and logs, DRY_MINT_PLAN.md, HANDOFF_P03.md (at most 800 tokens), MANIFEST_SHA256.txt;
* the read extension in `read/` with its dry check.

Hand back in at most 250 words, with a deliverable manifest.
