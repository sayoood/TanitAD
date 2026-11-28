# HANDOFF J (stage-2 reference frames: fix BUILT and TESTED; full suite + mutation run were still RUNNING at hand-off) -- 2026-11-06

MEASURED = ours, CPU. Nothing committed, staged, deleted, launched or sent to box 2. The frozen card is untouched. Token never printed or written.

## DONE
* Item 1 (fail closed). `code/t2v8f_stage2.py`: `check_ref_frames` (npz present, size + sha256 = MANIFEST2, f0/f60/f120 in the zip; ALWAYS a refusal, any `--scene-policy`), runs BEFORE the 24 GB hash; `pass_need_mib` (relaunch pays only for cells with no DONE). New `tests/test_t2v8f_stage2_refs.py` (12): REAL `read_ref_frame` on an npz-only folder, the kit's REAL `actor_map.falsify_actors`, full stub-rehearsal refusals (missing, kit, changed byte, no MANIFEST2 entry, `skip` policy), the shipped tar. Existing `tests/test_t2v8f_stage2.py` fixture now carries npz + builds overlay 2 (its 21 tests unchanged).
* Item 2: 41/41 mp4 downloaded (1,589,242,408 B, rev 5fe7a149, `HF_HUB_DISABLE_XET=1`) and verified against `lfs_sha256`; `stage2/ref_frames/DOWNLOAD_RESULT.json`. 4 sit in their C: scene folders; 37 are STAGED in `C:/Users/Admin/TanitAD-scratch/t2_refs/<id>/` (F: had 7.97 GB free, under the 10 GB floor of the T2-scenes brief; run 1 refused: `*.run1_floor_refused.*`).
* Item 3: 41 npz (434,028,861 B), `extract_ref_frames.py`. Bit check: 12 of 12 arrays of stage 1's 7c72937c npz (frames 0, 54, ..., 417) reproduce bit for bit from its mp4 (`BITCHECK_7c72937c.json`).
* Item 4: `stage2/overlay2/t2v8f_stage2_overlay2.tar`, 434,155,520 B, sha256 afa9dba7ff6c11b8f27c325fa829bf8ddd167431997536c13eb5f4aa5ecf2a50 (43 members, no mp4).
* Item 5: `STAGE2_RELAUNCH.md` (commands, canary, waiter, pull, score, ETA ~2.7 h central).
* Mutation arms RED so far: real_read, real_caller, need_scales, artifact, positive_refs, refused_missing (not_applied AND driver_unchecked).

## RUNNING (detached; logs in t2v8f/logs)
* `logs/pytest_stage2_refs_full.log` (full suite, `pytest tests -q`, expect all passed) and `logs/mutations_refs.log` (`mutations RED: 11 of 11`).
* `audit/finish_j.py` (WMI, cap 90 min) waits for both, then writes `PINS_J.txt` and `logs/J_FINISHED.txt` (mutation + suite verdict lines).

## UPDATE (after the first hand-back)
* Mutations: **11 of 11 RED** (`logs/mutations_refs.log`).
* Full suite run 1 (`logs/pytest_stage2_refs_full.log`, `J_FINISHED.txt`): **1 failed, 72 passed**. The failure was MY new test `real_caller`: in a full-suite run another module has already cached the STUB `gsplat_renderer` in `sys.modules`, so `import gsplat_renderer` gave a module without `grad_ncc` (it passed alone). FIXED in `tests/test_t2v8f_stage2_refs.py` (the cached stub entries are set aside, the kit's real files imported fresh from `KIT_HARNESS` and asserted by path, the stub restored); reproduced and verified with the stub cached (`repro_stub_first.py` in the scratchpad), the RED arm is RED for the right reason (the box-2 `FileNotFoundError`). No shipped file changed: the overlay-2 tar, the driver and the sha256 values stand.
* Full suite RE-RUN detached: `logs/pytest_stage2_refs_full2.log`; `audit/finish_j2.py` writes `logs/J_FINISHED2.txt` and rewrites `PINS_J.txt`. Expect `73 passed`.

## LEFT (in order)
1. Read `logs/J_FINISHED2.txt` (the run-1 line in `J_FINISHED.txt` is superseded). Any failed test or GREEN mutation arm: fix the arm, not the claim; if the driver changes, REBUILD overlay 2 (move `stage2/overlay2/` aside first; `make_overlay2.py build --refs stage2/ref_frames/REF_FRAMES_SOURCES.json --stage1-manifest stage1/t2v8f_stage1.tar`; then patch the sha256 and size in `STAGE2_RELAUNCH.md`).
2. Hand to the Master Mind: **the relaunch will be REFUSED by the 10 GiB floor** unless box 2 gets 0.8 to 1.6 GiB back or the floor is lowered (ESTIMATED, `STAGE2_RELAUNCH.md` section 0): a PI decision. The refusal now takes seconds.
