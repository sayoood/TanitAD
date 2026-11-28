# Brief: the stage-2 package as an OVERLAY on box 2's extracted stage-1 package (AGENT G; CPU only; do NOT launch)

**Rules.** `BRIEF_T2V8F_STAGE1_E.md` and `BRIEF_T2V8F_AUDIT_STAGE2.md` bind:
* the preamble, CPU only, no ssh, never commit, stage or delete, LF, banned words;
* ≤ 150k context, with `HANDOFF_G.md` at about 130k;
* the frozen card is never edited.

You own `t2v8f/`. Another agent works in `motion_c/exec_standstill/` only; do not touch it.

**Your plan is `HANDOFF_F.md`** (its LEFT list) and `audit/stage2_plan.json`: 42 launch scenes, 81 starts, 24.09 GB.

## The Master Mind's ruling on the disk blocker
Agent F estimated 26.6 GiB, against 25.58 GiB usable on box 2's D: (35.58 GB free, keeping ≥ 10 GiB). That estimate double-counts a
tar of the scenes plus its extraction. Do not tar the scenes.
* **Reuse the stage-1 package already on box 2** at `D:\t2v8f\pkg`: 471 files, sha-verified by stage 1's preflight. Ship only a SMALL
  overlay tar: the new driver/code, `MANIFEST2.json` and `jobs/stage2.cmd`.
* **Copy the scene folders directly** with `sftp put -r` (`reput` resumes), into a new `D:\t2v8f\scenes2\<id>\`.
  * Ship only the files the renderer and scorer need per scene, by `DOWNLOAD_MANIFEST.json`.
  * The preflight verifies every file by sha256 against the manifest, and refuses with a named reason.
* **State the box-2 footprint** on 1 MiB clusters: scenes, overlay, pull and the gsplat cache. Show that ≥ 10 GiB stays free.
  * If it still does not fit, name the exact shortfall. Do NOT plan any deletion: that is the PI's decision.

## Items
1. **The driver** `code/t2v8f_stage2.py`. Reuse the stage-1 driver's preflight, markers and plumbing pattern.
   * Rollouts per card section 3: C0_s0/_s1, P10, ZERO and BLIND under T2 and RR on the 42 launch scenes, with DAGOACC_s0/_s1 beside.
     The starts come from `stage2_plan.json` (6b clamp included).
   * **Order:** run all T2 and RR cells of the deciding arms first, then the beside arms, so a crash still leaves the deciding cells.
   * Finished cells (`DONE`) are skipped on relaunch.
   * Add an `--executor {base,hold}` switch, default `base`. `hold` holds the heading when an executed step is ≤ 0.1 m; that is the
     other agent's variant, implemented here independently in `t2v8f_exec.py`.
   * Test the switch with G-ID: at ε = 1e-3, hold equals base bit for bit, and a RED arm. The frozen card runs `base`. `hold` only ever
     runs under a later dated amendment.
2. **The builder** `code/make_stage2.py` (overlay tar), the stub rehearsal with a planted missing scene file, and tests with mutations.
   Every existing test stays green.
3. **`STAGE2_LAUNCH.md`.**
   * Commands: busy marker, scene copy with resume, overlay put, certutil, extract, `launch.ps1`, a background waiter, pull, dev-box
     score.
   * The scorer's output reports lateral and heading rows per start class (launch / moving).
   * ESTIMATED wall time from stage 1's MEASURED 12.3 s per cell.
   * Note that box 2 runs one job at a time, and the PI approved the T2 box-2 jobs ("Approve T2 path").

## Hand-back (≤ 300 words)
Tests (n passed) and mutations, the rehearsal, the overlay tar sha256 and size, the scene copy plan with the box-2 footprint, the
exact launch steps, open items, blockers, and a deliverable manifest.
