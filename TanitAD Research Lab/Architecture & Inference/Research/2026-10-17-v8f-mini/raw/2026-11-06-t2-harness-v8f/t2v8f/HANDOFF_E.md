# HANDOFF E (stage-1 package for the v8F T2 harness: BUILT and REHEARSED on CPU, NOT launched) -- 2026-11-06

Evidence class: MEASURED (ours) for the package, hashes and the stub rehearsal; ESTIMATED for box-2 wall time; UNVERIFIED for anything that needs box 2 or a real render. CPU only, no ssh, nothing committed or staged, nothing deleted (old tar, build dir moved aside).

## DONE
0. `code/t2v8f_nocv2.py` self-check ran (`{'self': 1.0, 'shifted': 0.0166}`); `code/t2v8f_stage1.py` compiled; its first real run found no code error. The MM suite (`logs/pytest_mm_post_d.log`) was 30 passed, nothing red.
1. **Package.** `code/make_stage1.py` -> `stage1/t2v8f_stage1.tar` (549,652,480 B, sha256 `1785337f...3421c`, `.sha256` beside, `stage1/BUILD_RESULT.json`). 472 files, 42 dirs; allocated on 1 MiB clusters: tar 525 + extracted 1,019 + kit install 160 + pull about 201 = about 1.9 GiB (45.4 GB free -> about 43.5 GB left). BLIND tensor sha `cd6a7ab5...` asserted (zeros stand-in refused); both mp4 equal the pinned `lfs_sha256`; 4 distinct checkpoints.
2. **MM ruling applied** in `stage1.actor`: deciding planted read = condition `empty` (second renderer, no actors) at the same pose and tick; the 30 s clock read is `beside_clock_past_end_*` and never decides; thresholds unchanged. Lead ticks 31 / 44 / 58 (9.2 / 11.0 / 12.6 m ahead), chosen from the scene's own tracks (DK join state LEAD) and shipped in the manifest.
3. **Rehearsal** `tests/test_t2v8f_stage1.py` (7 tests) on the REAL tar + a shadow kit: positive run ALL_GATES_PASS in 50 s; planted missing ckpt, changed byte, shifted ROI, `empty` that draws the lead, score e2e: all behave. `code/run_mutations_stage1.py`: **6 of 6 RED** (`logs/mutations_stage1.log`).
4. `STAGE1_LAUNCH.md` written. 5. Whole suite: **37 passed in 385 s** (`logs/pytest_full_E.log`). `PINS_E.txt` written.

## LEFT (for the Master Mind / PI)
* PI go for box-2 execution, then `STAGE1_LAUNCH.md` section 2, step by step.
* Box 2's `D:\tanitad\venv` imports `gsplat==1.5.3`, `msgpack`, `ninja` are UNVERIFIED (documented there: torch and transformers only). The preflight names each; installing them is new packages on the PI's machine.
* If any file in `t2v8f/code` changes, rebuild: move `stage1/t2v8f_stage1.tar*` and `BUILD_RESULT.json` aside, move `C:/Users/Admin/t2a_scratch/stage1_pkg_build` aside, rerun `make_stage1.py`; the test refuses a stale tar.
