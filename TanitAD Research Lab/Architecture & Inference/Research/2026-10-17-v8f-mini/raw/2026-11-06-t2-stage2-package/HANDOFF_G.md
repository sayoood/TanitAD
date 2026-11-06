# HANDOFF G (stage-2 overlay: code WRITTEN and compiles; NOT tested, NOT built, NOT launched) -- 2026-11-06

Evidence: MEASURED = ours, CPU. Nothing committed, staged, deleted or launched. Context ran out before tests; a fresh agent finishes from here.

## Design (decided; do not re-derive)
* Overlay = NEW files only, extracted over `D:\t2v8f\pkg` (stage 1's 471-file MANIFEST stays valid). Never edit any of the 7 stage-1 `code/` files (the stage-1 test refuses a stale tar).
* WRITTEN, compiles, LF, untested: `code/t2v8f_exec_hold.py` (hold executor: `hermite_frames_eps`, rule `norm > eps`, `install/uninstall(X)`; independent of the other agent), `code/t2v8f_stage2_cell.py`
  (`--executor base|hold`, `--hold-eps`; base patches nothing), `code/t2v8f_stage2.py` (driver: `all`, `plan`, `selfgate`; reuses `t2v8f_stage1` Ctx/preflight/marker/RENDER_FLAGS; own cell loop, 588 cells =
  420 deciding then 168 beside, scene-major, DONE skipped, executor in DONE text, per-pass disk check refuses to START a pass ending < 10 GiB, `--scene-policy refuse|skip`), `code/make_stage2.py`
  (`build`, `resume`; sftp batch of one explicit `put` per file; start classes launch/moving/standstill_other; footprint), `stage2_src/jobs/stage2.cmd` (pull2, scenes2).
* 7c72937c is already in box 2's kit: `where="kit"`, verified by sha256 (dev kit volume/rig and stage-1 tracks all equal the plan: MEASURED). 41 scenes are copied (23.5 GB) to `D:/t2v8f/scenes2`.
* MEASURED: stage-1 cell = 6 MiB on 1 MiB clusters (dir + 5 files); transfer 549.65 MB in 23 s (~24 MB/s); dev F: reads 110 MB/s (24 GB hash ~4 min); all 126 plan files exist on F: with matching sizes.

## Disk (ESTIMATED, builder prints exact): free now ~35.23 GiB (35.58 at stage-1 preflight minus install 160 + pull 201 MiB). Scenes 22.03 GiB, overlay ~0.01, deciding pass 2.65, beside 0.99.
After deciding ~10.5 GiB free (fits, margin ~0.5 over the 10 floor). After beside ~9.55: SHORT ~0.45 GiB. No deletion planned: PI candidates (all banked on dev box): stage-1 tar 525 MiB, stage-1 pull 201 MiB, kit scene 00040136.

## LEFT (in order)
1. `tests/test_t2v8f_stage2.py`: (a) hold gates on planted plans: G-ID eps=1e-3 == base bit for bit (+RED eps=0.1 differs on standstill-noise plan), G-MOV, G-PHYS (+RED hold disabled); (b) cell argv ==
   stage-1 `entry_args` minus `--t2-plumbing`; (c) cell order: 588, pass switch at 420, deciding first; (d) stub rehearsal: build a FIXTURE overlay with `make_stage2.build(..., fixture_tag="_fx")` from a 3-scene fixture plan
   (real small `rig_trajectories.json` and `sequence_tracks.json` copied from F:, 16-byte volume.nurec placeholders, one scene `where=kit` into a shadow kit like `tests/test_t2v8f_stage1.py`), extract the REAL stage-1 tar
   (reuse its helpers) plus the fixture overlay into `C:/Users/Admin/t2a_scratch/stage2_rehearsal`, run `t2v8f_stage2.py all --stub --inproc`; positive (COMPLETE, 42 cells DONE, base == direct `t2v8f_run.py` records),
   planted missing scene (refuse -> STAGE2.fail named MISSING; skip -> others run, INCOMPLETE), changed byte (HASH_MISMATCH), DONE skip on relaunch, executor mismatch refuses, gate RED (rig with wrong n_ticks).
   Stub: 600 frames = 200 ticks, so f0+60<199 holds for every plan start (max f0 109). Then `code/run_mutations_stage2.py` (mutation arms RED). Existing tests stay green (stage-1 37 passed in 385 s; audit 3).
2. `code/t2v8f_stage2_score.py` (dev box): run `t2v8f_run.py score` over `pull2/rollouts` with `--tracks idx=...` from `stage2/SCENE_SOURCES.json`, plus lateral/heading rows per start class from MANIFEST2 `start_class`.
3. Build the real overlay: `python -I code/make_stage2.py build` (verifies 24 GB by sha256, ~4 min: run detached; the stage dir `C:/Users/Admin/t2a_scratch/stage2_pkg_build` must be new; out `stage2/`). Re-run after ANY code edit.
4. `STAGE2_LAUNCH.md` (busy marker, mkdir, `sftp -b stage2/scenes_put.sftp`, resume via `make_stage2.py resume --remote-sizes` from a `Get-ChildItem -Recurse -File` listing, overlay put + certutil + `tar -xf` over `D:\t2v8f`,
   `launch.ps1 -Job D:\t2v8f\pkg\jobs\stage2.cmd`, background waiter on `pull2\STAGE2.*`, pull, score). ESTIMATED wall: copy ~16 min (24 MB/s), preflight ~20 min (24 GB hash at the stage-1 rate), gate ~5, rollouts
   588 x 12.3 s = 2.0 h up to 3.9 h (81/42 starts), cold scene reads +15 min per pass; total ~4 h central. Box 2 runs one job at a time; PI approved "Approve T2 path". Executor `hold` never runs without a dated amendment.
5. `PINS_G.txt` (sha256 of every file written), hand back.
