# Brief: the stage-2 reference-frame fix (AGENT J; dev-box CPU; one approved download; do NOT launch)

## What happened (MEASURED, `stage2/pull_box2/pull2/STAGE2_RESULT.json`, `rollouts_deciding.log`)
* Stage 2 ran on box 2 from 14:41 to 16:05.
  * Preflight and gate PASS.
  * Rollouts INCOMPLETE: 10 of 420 deciding cells and 4 of 168 beside cells DONE, all on `7c72937c`, the kit scene from stage 1.
* **Every cell on the 41 new scenes failed with the same error:**
  `av.error.FileNotFoundError: D:\t2v8f\scenes2\<id>\camera_front_wide_120fov.mp4`.
  * The harness's actor falsifier reads reference camera frames 0, 60 and 120. `t2v8f_nocv2.read_ref_frame` looks for
    `ref_frames.npz` beside the scene, then falls through to PyAV.
  * Stage 2 shipped neither the npz nor the mp4.
* **Why the rehearsal missed it.** The stub installs a fake `actor_map`, so `read_ref_frame` never ran. The stage-2 preflight did not
  check reference frames.
* **The class.** "Built, tested, and unreachable from the real caller", in the stub-bypass form.

## Rules
`BRIEF_T2V8F_STAGE2_I.md` binds:
* the preamble, never commit, stage or delete, LF, banned words;
* ≤ 150k context, with `HANDOFF_J.md` at about 130k;
* the frozen card is never edited;
* no ssh and no box 2.

**One exception: the download.** Download the 41 scenes' `camera_front_wide_120fov.mp4` (about 1.6 GB) from
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` at revision `5fe7a149`.
* **It is approved.** The PI approved the NuRec scene download up to 80 GB; about 36 GB is used. The frozen card's 6b already covers
  these mp4 files as references.
* **How.** Set `HF_HUB_DISABLE_XET=1`. Read the token in place from `E:/Projects/TanitAD/Keys.txt`
  (`grep -oE 'hf_[A-Za-z0-9]+'`, held in a variable). Never print it, write it, or put it in argv.
* **Where.** Into `F:/TanitAD-scratch/t2_scenes/<id>/`, or the C: twin where that scene lives. Each download goes into its scene's
  own folder, which holds only that release's data.
* **Verify.** Check every file against its `lfs_sha256` in `t2_scenes/raw/RELEASE_LISTING.json`, and log the results.

## Items, in order
1. **Fail closed first.**
   * The stage-2 preflight must REFUSE any scene folder without a `ref_frames.npz` whose sha256 matches `MANIFEST2`, and name it.
   * Write a CPU test with the REAL `t2v8f_nocv2.read_ref_frame`, not the stub: a scenes2-style folder holding only `ref_frames.npz`
     returns frames 0, 60 and 120 and never opens the mp4.
   * RED arm: remove the npz, and the preflight refuses with a named reason.
   * Every existing test stays green.
2. **The download.** As above, with resume.
3. **The reference frames.** Extract frames 0, 60 and 120, plus any gap ticks the stage-2 driver reads, into `ref_frames.npz` per
   scene. Use `t2v8f_nocv2.extract_frames`, as stage 1 did.
   * Check the result against stage 1's `7c72937c` npz, which must reproduce bit for bit when re-extracted from its mp4.
4. **Overlay 2.** A small tar: the 41 npz files into `D:\t2v8f\scenes2\<id>\`, the updated driver/preflight, and the updated
   `MANIFEST2`.
   * The mp4 files are NOT shipped.
   * State the box-2 footprint.
5. **`STAGE2_RELAUNCH.md`.**
   * Put the overlay 2, certutil it, extract it, and relaunch the same `stage2.cmd`. DONE cells are skipped.
   * Waiter, pull, and dev-box score.
   * The ESTIMATED wall time: stage 2's first run spent 4,203 s, mostly hashing and failing. Use stage 1's MEASURED 12.3 s per cell.
6. **Close-out.** Run the full suite, then write `PINS_J.txt`.

## Hand-back (≤ 300 words)
The fail-closed test and its RED arm, the download log (41 hashes against the listing), the npz bit-for-bit check, the overlay-2 sha256
and size, the relaunch steps, open items, blockers, and a deliverable manifest.
