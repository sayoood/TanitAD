# Brief: the stage-1 box-2 package for the v8F T2 harness (AGENT D; build and rehearse on CPU; do NOT launch)

**Rules.** Everything in `BRIEF_T2V8F_E3E6.md` (same folder) binds: the preamble, CPU only, no ssh to box 2, never commit, never
delete, LF line endings, banned words, ≤ 150k context with a handoff at about 130k. No downloads and no new packages, on the dev box
or for box 2.

**Read first:**
* `HANDOFF_C.md`: the ordered job list and the tar contents. This is your plan.
* `HANDOFF_B.md`: the record format of `rows_for_records`, in the `t2v8f_lead.py` docstring.
* `HANDOFF_A.md`.
* Card `motion_c/t2_prereg/PREREG_T2_CRAWL.md`: sections 3, 4 and 6b.
* Memory `box2-second-computer.md` and `devbox-drive-letter-and-detached-jobs.md`.

The Master Mind re-ran the test set. All files in `t2v8f/` are now yours to edit as needed: no other agent is editing them. Every
existing test must stay green.

## Items
1. **Fix at source.** `t2v8f_exec.ground_correct` multiplies a 4×4 by a 3×3, so condition T2 raises on its first step. Correct it in
   `t2v8f_exec.py`, keep it equal to the harness `GroundFollower.correct`, and add a test that runs condition T2 (E3's tests ran only
   RR and logged), with a mutation.
2. **Wire scoring.** `t2v8f_run.py score` calls `rows_for_records(recs, scenes)` from `t2v8f_lead.py`. Run the CPU stub rehearsal end
   to end through the 19 rows and the X1 / X2 bootstrap.
3. **MM rulings.**
   * The rollout schedule is `periodic`, the T1-kin tool's default.
   * The plumbing start for scene `00040136` is f0 = 60.
   * The plumbing rollouts of stage 1 are stamped plumbing only, and never feed H1 or H2.
4. **The package (`stage1/`).**
   * **Contents:** code, the C0_s0/_s1 and DAGOACC_s0/_s1 checkpoints with their sha256, the BLIND tokens, and the 2 scenes'
     `camera_front_wide_120fov.mp4` and `extracted/sequence_tracks.json`.
   * **Where they come from:** take them from the on-disk copies at
     `E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/<id>/` or from `F:/TanitAD-scratch/t2_scenes/`. Hash each one
     against the pinned `lfs_sha256` in `t2_scenes/raw/RELEASE_LISTING.json` where it is published, and record the result.
   * **Box-2 target:** the T2-A kit at `D:\t2a\box2_kit` and DINOv3-L at `D:\tanitad\hf`. The first job step is a preflight that checks
     every path, the CUDA device, the free disk, and every hash. It refuses with a named reason if anything is missing.
   * **No `cv2` on box 2.** Implement the actor falsifier without OpenCV (numpy / PIL / imageio), or report exactly which harness path
     needs `cv2`.
   * **Jobs, in order, one process (safe mode):**
     1. preflight;
     2. T5 determinism, one C0 start rendered twice;
     3. T1 render gap: frames rendered at the logged pose against the real mp4 frames, pooled-token cosine and rel-L2, beside;
     4. the actor falsifier: `objects` draws the lead in `7c72937c`, and a planted no-lead read is absent;
     5. a plumbing run of every arm under T2 and RR on both scenes.
   * **Outputs:** to a pull folder, with `STAGE1.done` / `STAGE1.fail` markers and a `STAGE1_RESULT.json`.
   * **Launch:** `launch.ps1`-style, detached.
   * **Size:** state the package byte size, and check that box 2's D: keeps ≥ 10 GB free (45.4 GB free at the last probe).
5. **Rehearsal.** Rehearse the package's whole control flow on CPU with the stub renderer and box-2 paths substituted, including a
   planted missing file that must make the preflight refuse.
6. **`STAGE1_LAUNCH.md`:** the exact copy and launch commands for the Master Mind, the pull commands, and what each marker means.

## Hand-back (≤ 300 words)
Fix and tests (n passed), mutation count, the rehearsal result, the package contents, size and hashes, the exact launch steps, open
items, blockers, and a deliverable manifest.
