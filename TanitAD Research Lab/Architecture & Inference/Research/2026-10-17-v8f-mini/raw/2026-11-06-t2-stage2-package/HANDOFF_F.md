# HANDOFF F (audit DONE, verdict (a); stage-2 package NOT yet built) -- 2026-11-06

Evidence class: MEASURED (ours, CPU) unless marked. Nothing launched, staged, committed or deleted. `t2v8f/code` untouched, so stage-1 tar `1785337f...3421c` is still current.

## DONE (item 1)
* `AUDIT_NUREC_SCORE.md`: verdict **(a) real difficulty**. The "36 deg" = mean of 10.25 deg (00040136 f0 60, v0 17.9 m/s, NOT standstill) and 61.43 deg (7c72937c f0 22, standstill launch). Logged-plan self-score
  ~1e-14 on both; T2 `g` == T1 reader path bit for bit; mutations leave 0; banked T1 P10 launch-class median 61.2 deg / 283.0 deg/s = the stage-1 launch start.
* Pins: `tests/test_t2v8f_audit.py` (3 passed, 3.3 s), `audit/run_mutations_audit.py` (9 of 9 as expected), `audit/selfscore_run.py` + `selfscore_score.py` (two processes; the scorer binds the reader tree).
* Stage-2 plan data: `audit/stage2_plan.py` -> `audit/stage2_plan.json`: **42 launch scenes, 81 starts** (39 scenes x 2, 3 scenes x 1; card 3 + 6b: f0 = max(round(10*(t_c - d)), 18), d in {4.0, 2.0}, collapse
  equal; all fit f0 + 60 < n_ticks), per-scene file lists (sha256, bytes, F: location). Scenes = 24.09 GB (22.43 GiB; 22.51 GiB allocated on 1 MiB clusters): `volume.nurec` 23.93 GB,
  `rig_trajectories.json` 0.09, `sequence_tracks.json` 0.06 (no mp4 needed). 7c72937c (starts 22, 42) is a launch scene; stage 1 read it (and 00040136, on-hand but NOT a launch scene) from the sample_set under `E:/thor_archive/.../26.04_release/`; whether a copy of 7c72937c
already sits on box 2 is UNVERIFIED (the disk arithmetic below assumes 41 new scenes; with 42 add ~0.6 GiB).

## DISK (ESTIMATED, box 2 D: 35.58 GiB free after stage 1, rule keeps >= 10 GiB => 25.58 GiB usable)
Incremental one-batch need: scenes ~21.9 (41 new) + stage-2 package ~0.6 + outputs 588 cells x ~7.2 MiB (stage 1: 201 MiB / 28 cells) ~4.1 = **~26.6 GiB > 25.58: does NOT fit in one batch.**
Plan **two batches** (21 + 21 scenes, ~13.3 GiB each incl. outputs). Both coexisting = ~26.6 GiB, still over by ~1 GiB. **BLOCKER (PI decision, a deletion):** between batches free the batch-1 scene
copies (~11 GiB) after pulling their outputs, or free stage-1 leftovers (tar + pull ~0.7 GiB, already banked here), or relax the 10 GiB margin to ~8.9. Do not delete; name it in STAGE2_LAUNCH.md.

## LEFT (exact next steps, in order; keep each under ~150k context)
1. Read `STAGE1_LAUNCH.md` + `code/make_stage1.py` + `code/t2v8f_stage1.py` (the builder/driver pattern; manifest, preflight, `matrix_main` per scene).
2. `code/make_stage2.py` + `code/t2v8f_stage2.py` (NEW files; do not edit stage-1 ones): preflight as stage 1, then per batch: for each scene `t2v8f_run.py matrix --scene-idx i --scene-id S --arms C0_s0,C0_s1,P10,ZERO,BLIND,DAGOACC_s0,DAGOACC_s1
   --t2-conds T2,RR --agents objects --starts <the scene's starts> --t2-plumbing OFF`. Ship `audit/stage2_plan.json` as the manifest (starts + scene sha256); the scenes themselves travel separately (hash-verified copy from
   `F:/TanitAD-scratch/t2_scenes/` or `C:/Users/Admin/TanitAD-scratch/t2_scenes/`). BLIND mean = the stage-1 file (sha `cd6a7ab5...`). Add the CPU gate: logged-plan self-score on every scene's rig trajectory before rendering.
3. Rehearse on CPU: `T2A_STUB=1`, real tar + shadow kit, a planted missing scene (must fail loud, the others continue), mutation arm RED. Run the whole suite (stage 1 was 37 passed, 385 s) + `tests/test_t2v8f_audit.py`.
4. `STAGE2_LAUNCH.md`: commands, sizes, **ESTIMATED wall time**: stage 1 MEASURED 345 s / 28 cells = 12.3 s per cell at 1 start per scene; 588 cells x 12.3 s = ~2.0 h; x 1.93 starts/scene (81/42) => up to ~3.9 h (BLIND renders nothing, so lower). Pull + `t2v8f_run.py score` steps; report lateral rows per start class (launch / moving), never pooled (AUDIT section 5).
5. Hand-back: tar path, size, sha256, box-2 footprint, batch plan, blocker.

## Traps met
Stage-1 `PLUMBING_SCORE.json` groups are 2-start means: read per start. `D:` on box 2 is exFAT with 1 MiB clusters (every small output file costs 1 MiB). New files go in `audit/` or `tests/`, not `code/`, or the stage-1 tar test goes stale.
