# Brief: stage 3, the 20 guard scenes of PREREG_T2_CRAWL (AGENT K; dev-box CPU; one approved download; do NOT launch)

## Why (MEASURED / card text)
* The frozen card `motion_c/t2_prereg/PREREG_T2_CRAWL.md` (sha256 in `FROZEN_SHA.txt`, never edited) defines 62 scenes: 42 launch + 20 guard
  (`t2_scenes/SCREEN_RULE.md`). Section 3 reads M1 over the launch scenes and the four families over all 62.
* Stage 2 (`t2v8f/STAGE2_LAUNCH.md`, `STAGE2_RELAUNCH.md`) covers the 42 launch scenes only and is running on box 2 now. Nothing builds the
  guard pass: `grep -i guard` over `make_stage2.py`, `t2v8f_stage2.py` and `STAGE2_LAUNCH.md` returns zero.
* The 20 guard scenes are on the dev box (`t2_scenes/raw/DOWNLOAD_PLAN.json`, slots `guard_1..guard_20`, 10.79 GiB, 0 failures in
  `DOWNLOAD_STATUS.json`). Their `camera_front_wide_120fov.mp4` references are NOT (stage 2 needed the same download: `BRIEF_T2V8F_STAGE2_REF_J.md`).

## Rules
`BRIEF_T2V8F_STAGE2_I.md` binds:
* the preamble (`Project Steering/AGENT_OPERATING_STANDARD.md`), never commit, stage or delete, LF, banned words;
* <= 150k context, with `HANDOFF_K.md` written at about 130k; bank incrementally;
* the frozen card is never edited;
* no ssh and no box 2. CPU only (`CUDA_VISIBLE_DEVICES=-1`). No GPU, no training.
* Do not edit the stage-2 driver's shipped behaviour for the launch scenes: stage 2 is running on box 2 with driver sha256
  `33b65b19...`. Stage 3 is a NEW driver mode or a new file; every existing test stays green.

**One exception: the download.** The 20 guard scenes' `camera_front_wide_120fov.mp4` (about 0.8 GB) from
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` at revision `5fe7a149`, exactly as `t2v8f/code/download_refs.py` did it for stage 2:
`HF_HUB_DISABLE_XET=1`; token read in place from `E:/Projects/TanitAD/Keys.txt` (`grep -oE 'hf_[A-Za-z0-9]+'`, held in a variable,
never printed, written or put in argv); each file into its own scene's folder (or `C:/Users/Admin/TanitAD-scratch/t2_refs/<id>/` when that
drive is below its floor); every file checked against `lfs_sha256` in `t2_scenes/raw/RELEASE_LISTING.json`. It is approved: the PI approved
the NuRec download up to 80 GB, about 37.6 GB is used, and the card's 6b covers these references.

## Items, in priority order (bank each before the next)
1. **Starts.** The card's guard rule, verbatim from section 3: two starts per guard scene by `numpy.random.default_rng(1)`, uniform over the
   renderable span, leaving 6 s; then the logged 16-row history. Read section 6b for any later fixing of the draw (order of scenes, units,
   inclusive bounds) and implement exactly that; where the card leaves a choice open, write it in `DEFINITIONS_K.md` BEFORE the draw and
   name it in the hand-back. Start class (launch / moving) by the stage-2 rule (`make_stage2.start_class`). Test: the draw is reproducible
   bit for bit, and a RED arm (`default_rng(2)`) changes it.
2. **References.** The download above, then `ref_frames.npz` per guard scene with `t2v8f/code/extract_ref_frames.py` (the stage-2 frames,
   same keys). Re-check the stage-1 `7c72937c` bit-for-bit control as J did.
3. **The stage-3 package.** Same cell matrix and order as stage 2 (deciding arms C0_s0, C0_s1, P10, ZERO, BLIND under T2 and RR first, then
   DAGOACC beside), `base` executor, `objects` agents. A `MANIFEST3` with every scene file's size + sha256 and every npz. The preflight
   REFUSES a scene without its npz, as stage 2's does now (reuse `check_ref_frames`). **Scenes live on box 2 `C:\t2v8f\scenes3\<id>\`**
   (box 2 D: has ~13.6 GiB free against a 10 GiB floor; box 2 is NTFS, 4 KiB clusters, C: ~29.5 GiB free); the pull stays on D:. Prove the
   driver reads scenes from C: and checks the floor on the pull drive (a test with two roots).
4. **Rehearsal.** The full stub rehearsal on a guard-scene fixture, plus the REAL `read_ref_frame` / `falsify_actors` path on one guard npz
   (the stage-2 lesson: a stub that never calls the real reader proved nothing). Mutation arms RED: missing npz, changed scene byte,
   wrong-drive floor, start draw.
5. **`STAGE3_LAUNCH.md`.** The scene puts (sftp batch to `/C:/t2v8f/scenes3/...`, ~10.8 GiB at the MEASURED ~24 MB/s), overlay put +
   certutil, the launch line, canary, waiter, pull into its own folder, and the score: the families over all 62 scenes pool stage-2 and
   stage-3 rollouts, per start class, never pooled across classes. ESTIMATED wall time from stage 1's MEASURED 12.3 s per cell.
6. **Close-out.** The full suite (`cd t2v8f && CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests -q -p no:cacheprovider`, detached, ~12 min),
   then `PINS_K.txt` (sha256 of every file written).

## Hand-back (<= 300 words, via the hand-back message)
Starts table (scene, f0, class), the download log (20 hashes against the listing), the npz checks, the package sha256 and size, the
box-2 footprint per drive, the RED arms, open items, blockers, and a deliverable manifest (every artifact and where it lives).
