# HANDOFF I (stage-2 overlay: code TESTED; real build + mutation run were still RUNNING at hand-off) -- 2026-11-06

MEASURED = ours, CPU. Nothing committed, staged, deleted or launched. Shipped code files were NOT edited after the fixture/real build started.

## DONE
* `tests/test_t2v8f_stage2.py` (21 tests) + `code/run_mutations_stage2.py` (21 arms). Fast arms 11/11 RED (G-ID, G-MOV, G-PHYS, tie, cell switch, order, argv, footprint, resume, scorer classes, class rule).
  Rehearsal (REAL stage-1 tar + fixture overlay, 3 scenes, 42 cells, stub, `--inproc`): positive, planted missing (refuse/skip), changed byte, DONE skipped, executor mismatch, gate RED, builder refusal: 8 passed in the run; base == direct and hold reachable: 2 passed after the fix below.
* Finding: the driver's in-process (`--inproc`) cell differs from a fresh-process cell by ~2e-3 m in plans (stub policy float drift; the real driver runs one process per cell). The bit-equality checks therefore compare FRESH processes; the in-process cell is bounded < 0.05 m.
* Scorer `code/t2v8f_stage2_score.py`: launch = backward-difference v0 at f0 < 0.5 m/s (the policy's own spelling; equals each record's `v0` exactly). Stage-1 pull: 00040136 moving (17.88 m/s), 7c72937c launch (0.0008); P10(T2) heading 10.25 / 61.43 deg, yaw rate 5.06 / 283.03 deg/s per class (pooled 35.84). `make_stage2.start_class` now uses the same rule (classes launch/moving only).
* `STAGE2_LAUNCH.md` written with placeholders `@@SIZE@@`, `@@SHA@@`, `@@FOOTPRINT@@`.

## RUNNING (started detached; logs in t2v8f/logs)
* `make_stage2.py build` (24 GB sha256, ~5-10 min) -> `logs/make_stage2.log` ends with `rc=0`; outputs in `stage2/`. Stage dir `C:/Users/Admin/t2a_scratch/stage2_pkg_build`. Re-run after ANY edit of the 3 shipped code files or stage2.cmd (move `stage2/` and the stage dir aside first).
* `run_mutations_stage2.py` -> `logs/mutations_stage2.log`, expect `mutations RED: 21 of 21`.

## LEFT (in order)
1. Read `stage2/BUILD_RESULT.json` (tar_sha256, tar_bytes, footprint); patch the 3 placeholders in `STAGE2_LAUNCH.md` (footprint = scenes_mib, overlay_mib, pass_mib, free_gib_*; shortfall_gib_all is the beside-pass shortfall).
2. Check the mutation log; any GREEN arm = fix the arm, not the claim.
3. Full suite, detached (~12 min): `cd t2v8f && CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests -q -p no:cacheprovider > logs/pytest_stage2_full.log 2>&1` (stage-1 37 passed before; + audit 3 + stage-2 21).
4. `python -I audit/make_pins_stage2.py` -> `PINS_G.txt`. Hand back to the Master Mind (<= 300 words).

## UPDATE (end of run): ALL DONE
Build finished (212 s; overlay sha256 8c94def6...ffe41, 81,920 B, 5 files; STAGE2_LAUNCH.md filled). Mutations 21 of 21 RED (one arm fixed and rerun). Full suite `pytest tests -q`: 61 passed in 479 s (CUDA hidden), logs/pytest_stage2_full.log. PINS_G.txt written. Nothing left except the Master Mind's decisions.
