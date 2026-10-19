# HANDOFF -- force_s2 (forced tactical-lon re-roll on the S2 windows where BAR-R8-E1 is lost), written 2026-10-18 ~21:45 +0200

NOT committed, NOT staged (qland hold). PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery`.
Single copy of everything below is in PKG (new files only; no existing file was edited) and the scratch dir `E:/refcv8_eval_kit/force_s2_scratch/` (created by `select`).

## State
* Code: `PKG/code/force_s2_reroll_r8.py` (stages `select | roll | status | analyze`; imports `force_lon_reroll_r8` for the model build, `set_r8_force`, per-window seeding and the chunked driver). Pre-registered `decide()` is in it and in `raw/force_s2/PREREG.md`.
* Tests (CPU, no data): `code/test_force_s2_r8.py` 48 passed, `code/test_force_s2_launcher_r8.py` 23 passed, and the existing `test_force_lon_r8.py` 25 still pass beside them (`raw/force_s2/pytest_*.log`). Mutation checks: `raw/force_s2/MUTATION_CHECK_*.txt`.
* **Launcher ARMED**: `code/force_s2_launcher_r8.sh`, pid in `E:/refcv8_eval_kit/force_s2_scratch/launcher.lock/pid`, log `E:/refcv8_eval_kit/chain/force_s2.log` (markers `ZZFS2...ZZ`). It WAITS (gate below), then runs `select` -> `roll --driver 4` -> `analyze` by itself. Nothing has run on real data yet.
* **No real-data smoke was done**: free commit was 2.2-5.3 GB (< 12 GB) the whole turn, so by the brief the `select` loader (`load_s2_arrays`) is exercised only against FAKE e1_decomp / clamp_posthoc modules whose keys I read from source (6 tests). First real contact is the launcher's `select` stage: if it fails, `select.log` names the control (the 434 / 424 set-size refusal is deliberate).

## The gate (all of, in order)
G1 `ZZQQUEUEENDZZ` in `chain/queue_30000_r8.log`; G2 a terminal fp64 marker AFTER THE LAST `ZZF64WAITZZ` in `wrapper_fp64/raw/run.log` (the file already holds a STALE `ZZF64COMBINEFAILZZ` from the aborted 19:56 attempt -- a plain grep for the four markers would have opened the gate at once; pinned by a test); G3 the earlier force_lon roll finished or silent > 15 min (extra to the brief, see below); G4 free commit >= 12 GB twice 3 min apart.

## How to check (one command, do not loop with turns)
`tail -5 /e/refcv8_eval_kit/chain/force_s2.log; ps -ef | grep -c '[f]orce_s2_launcher'` (launcher alive = 1 once the mutation checks have ended). Roll progress: `grep -c ZZFLPROGZZ /e/refcv8_eval_kit/force_s2_scratch/roll_full.log`; `python force_s2_reroll_r8.py status` prints `ZZFS2STATUSZZ rolled=a/b` (b = the plan size: 440 at the 220-per-set cap).
Expected wall once the gate opens: ESTIMATED ~3 h (force_lon: N ~10 s + ~5 s per forced forward, 8 threads; 440 windows x (N + F1 + F_slow) + 200 control windows x (Fown + N2)). Each of the 4 chunks reloads the model (~7.6 GB working set).

## If it stops
* Launcher died / was never going to open: `rmdir` is NOT needed -- a dead holder's lock is reclaimed. Re-arm: `nohup sh PKG/code/force_s2_launcher_r8.sh >> /e/refcv8_eval_kit/chain/force_s2.out 2>&1 < /dev/null &`.
* Roll incomplete (`ZZFS2ROLLFAILZZ`, analyze deliberately NOT run): re-run `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 R8_NO_POSE_SYNC=1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;PKG/code' python -X utf8 -u force_s2_reroll_r8.py roll --tag full --driver 4 --save-every 2` (resumable on the same artifact), then `analyze --tag full`.
* Result: `PKG/raw/force_s2/FORCE_S2.json` + `.md` (verdict = `PRIMARY.verdict`, mechanical from the committed table; also read `content_check`, `coverage.PARTIAL`, KA, KB, the like-for-like closure, `n_needed`). Land `code/force_s2_*`, `code/test_force_s2_*`, `code/mutation_check_force_s2_r8.py` and `raw/force_s2/*`; `E:/.../force_s2_scratch/` (raw clip ids, the roll JSON) need NOT be landed.

## Findings the Master Mind should know now
1. **The earlier force_lon roll is DEAD.** Its `roll_full.log` was last written 19:37:34 (ZZFLPROGZZ 20/100), `roll_full.json` has 20 windows, and no `force_lon_reroll` process exists at 21:22. Its HANDOFF ETA (22:45-23:15) is stale. Not touched; re-run per its HANDOFF if wanted. (Cause unknown; no traceback in the log.)
2. G3 was added for that roll's sake and costs nothing now: a stale log counts as "not alive".
3. F_slow reuses force_lon's code without editing it by rebinding the module global `CRUISE6` (read at call time in its `specs["F2"]`) per single-set chunk; the arm named `F2` in the roll JSON therefore means F_slow. Guarded statically (`assert_fl_internals`, also run in `select`) and by content (analysis refuses unless every `forced6` equals the plan). If force_lon_reroll_r8.py is ever edited, re-run the tests.
4. closure_F1 and closure_F_slow are on different populations (F1 needs a labelled `lon_v7`; ~43 % of banked slow_both windows carry one). The like-for-like closure is reported beside the primary; the committed table is applied as written.

## Unverified
Everything on real data (loader, S2-window selection by `(clip, ws)` through the roll, CPU-vs-GPU KA on the S2 slots); the ~3 h estimate; that `--grid 2s` rolls S2 windows without a 6-s future (the original battery did, 4,754 windows, 0 skipped).

## Frozen hashes (sha256, first 16) at 2026-10-18T21:38:42+0200; launcher armed 2026-10-18T21:31:11+0200, pid 1063192
22d72c4ac82d1323 *force_s2_reroll_r8.py
894100c5e41e1c68 *test_force_s2_r8.py
2f5c401c6e676140 *force_s2_launcher_r8.sh
a85a3dd4bc389df5 *test_force_s2_launcher_r8.py
907e8abcd1ca55ab *mutation_check_force_s2_r8.py

Mutation checks (raw/force_s2/MUTATION_CHECK_*.txt): module 23 RED / 0 survived / 0 not applied; launcher 11 RED / 0 survived / 0 not applied. (The first module run read 4 'survivors' because force_lon_reroll_r8 prepends the real code dir to sys.path and the test imported the ORIGINAL module; the harness now preloads the mutated copy and asserts its path.)
