# HANDOFF K (stage 3, the 20 guard scenes) -- 2026-11-06; the close-out chain was still RUNNING at hand-off

MEASURED = ours, CPU. Nothing committed, staged, deleted, launched or sent to box 2. Frozen card untouched. Stage-2 driver `code/t2v8f_stage2.py` NOT edited (sha256 33b65b19...f4898; `audit/make_pins_stage3.py` prints it).

## DONE (all under t2v8f/)
* Item 1 starts: `DEFINITIONS_K.md` (written before the draw), `code/make_starts3.py` -> `stage3/GUARD_STARTS.json`: 40 starts, 20 scenes, all MOVING, starts_sha256 34c35cfe...f512 (kit == hand derivation == fresh process).
* Item 2 refs: `code/download_refs3.py` (20/20 mp4, 850,400,161 B = lfs_sha256), `code/extract_ref_frames3.py` (20 npz, 192,554,391 B, in the guard scene folders on C:; stage-1 7c72937c bit check 12/12 arrays equal).
* Item 3 package (REAL, built, source files sha256-verified): `stage3/t2v8f_stage3_overlay.tar` 51,200 B sha256 bd96fca1...b840 (3 NEW files; driver `code/t2v8f_stage3.py` 0688bb47...733c), `stage3/BUILD_RESULT3.json`, `SCENE_SOURCES3.json`, `scenes_put3.sftp` (80 puts to `/C:/t2v8f/scenes3/...`).
* Items 4 and 5: `tests/test_t2v8f_stage3.py` (18 tests, all passed: 16 in `logs/pytest_stage3.log` run 1 + the 2 fixed ones in `logs/pytest_stage3_b.log`), `code/run_mutations_stage3.py` (17 arms), `STAGE3_LAUNCH.md` (waiter with exact marker names and the `.prev` control, per MM message), `code/t2v8f_stage3_score.py`.

## RUNNING (detached chain `audit/finish_k.sh`, started ~17:05)
mutations (`logs/mutations_stage3.log`, expect `mutations RED: 17 of 17`), then the FULL suite (`pytest tests -q`, `logs/pytest_stage3_full.log`, 73 earlier + 18 new = 91 expected, ~15-20 min), then `PINS_K.txt`. Verdict lines land in `logs/K_FINISHED.txt` (ends with `ZZ K-CHAIN-DONE ZZ`).

## LEFT (in order)
1. Read `logs/K_FINISHED.txt`. A GREEN mutation arm or a failed test: fix the ARM or the test, never the claim. If `t2v8f_stage3.py` or `stage3.cmd` changes, REBUILD: move `stage3/t2v8f_stage3_overlay.tar*`, `BUILD_RESULT3.json`, `SCENE_SOURCES3.json`, `scenes_put3.sftp` and `C:/Users/Admin/t2a_scratch/stage3_pkg_build` aside, rerun `python -I code/make_stage3.py build` (15 s), patch the sha256 values in `STAGE3_LAUNCH.md` section 0.
2. Re-run `python -I audit/make_pins_stage3.py` if any file changed after the chain.
3. The Master Mind's decisions: (a) D: on box 2 must hold >= 11.75 GiB free for both passes, >= 11.28 GiB for the deciding pass (the driver refuses in seconds); (b) stage 2 must be STAGE2.done before stage 3 (the driver also refuses while `D:/t2v8f/pull2/STAGE2.running` exists).
