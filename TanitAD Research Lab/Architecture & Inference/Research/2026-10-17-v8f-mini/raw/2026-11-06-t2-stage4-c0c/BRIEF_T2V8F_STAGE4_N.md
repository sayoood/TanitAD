# Brief: the stage-4 package for the composite arm C0C (AGENT N, continuing AGENT M; dev-box CPU; no download; do NOT launch)

**Read first:**
* `t2v8f/HANDOFF_M.md`: the state, and the exact next steps for items 3 to 6.
* `t2v8f/BRIEF_T2V8F_C0C_L.md` items 3 to 6: the task. Its rules bind you unchanged.
* `t2v8f/HANDOFF_K.md` and `STAGE3_LAUNCH.md`: stage 3 is the template.
* ⛔ Never edit the frozen cards in `motion_c/t2_prereg/`.

## Done (MEASURED, do not redo)
* **The arm:** `code/t2v8f_c0c.py`.
* **Gate C0C-ID: PASS,** in `C0C_ID_RESULT.json` (`c8df31eb...`).
* **The read script:** `code/read_t2_dominance.py`, pinned as sha256 `6345d4c7...`.
  * ⛔ Do not edit it. If stage 4 needs a change to it, STOP and say so in the hand-back.

## Items
1. **The stage-4 package**, exactly to the contract the pinned read script reads (`HANDOFF_M.md`, "Open for stage 4"):
   * `stage4/t2v8f_stage4_overlay.tar`, with member `pkg/MANIFEST4.json`;
   * `STAGE4_RESULT.json` recording `manifest4_sha256` and the package file counts;
   * the pull layout `<root4>/<scene8>/objects/<T2|RR>/C0C_s{0,1}/{DONE,records.jsonl,scene_<idx>.npz}`.

   The cells:
   * `C0C_s0` and `C0C_s1` under T2 (deciding) on all 121 starts, then under RR (beside). The starts are stage 2's 81 and stage 3's 40.
   * `base` executor, `objects` agents.

   The package:
   * The RI tool ships as a pinned byte copy (76,704 B).
   * The scenes are already on box 2: `D:\t2v8f\scenes2` and `C:\t2v8f\scenes3`. The preflight sha-checks them against MANIFEST2 and
     MANIFEST3, and refuses if `scenes3` is absent.
   * A new driver file. ⛔ The stage-2 driver (`33b65b19...`) and the stage-3 driver (`0688bb47...`) stay byte-identical.
   * The pull goes to `D:\t2v8f\pull4`. State the footprint per drive.
2. **Rehearsal and mutations.**
   * Run the stub rehearsal, plus the REAL `read_ref_frame` / `falsify_actors` path.
   * Mutation arms that must go RED: composite off, wrong prior, a scene byte changed, scenes3 absent, a P10 key mismatch against
     stages 2 and 3, and the RI copy's hash changed.
   * Then run the pinned read script on the rehearsal's SYNTHETIC stage-2, stage-3 and stage-4 pulls. It must accept the layout; this is the
     end-to-end contract test.
3. **`STAGE4_LAUNCH.md`.**
   * Overlay put + certutil, extract, verify the hashes, launch, canary.
   * ⛔ The waiter uses EXACT marker names: `Test-Path <pull>/STAGE4.done -PathType Leaf`, then `grep -x`.
   * Pull to `stage4/pull_box2`, then the read command with every argument spelled out.
   * ESTIMATED wall time at the MEASURED ~18 s per cell.
4. **Close-out.**
   * Run the full suite detached through a finisher. The suite log goes to `logs/pytest_full_M.log` (the name the read's G2(b) cites).
     `logs/N_FINISHED.txt` ends with `ZZ N-CHAIN-DONE ZZ`.
   * Then write `PINS_N.txt`.

**Rules:**
* ⛔ No ssh, no box 2, no commit, stage or delete, no download, CPU only (`CUDA_VISIBLE_DEVICES=-1`), LF line endings, banned words.
* ⛔ Do not open any real stage-2, 3 or 4 pull: `stage2/pull_box2_run2*`, `stage3/pull_box2`.
* Hand off at about 130k context with `HANDOFF_N.md`.
* The hand-back is ≤ 300 words, with a deliverable manifest.
