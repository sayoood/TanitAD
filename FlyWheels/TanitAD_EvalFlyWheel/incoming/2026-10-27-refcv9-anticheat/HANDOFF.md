# HANDOFF -- PREREG_ANTICHEAT_REFCV8 (EvalFlyWheel, 2026-10-27 ~20:55 Berlin)

**State.**
* Prereg sealed 20:27:00+0100 (sha256 `16a6ed68c5d815afa74335455ae1b67f7a620e917e5bbca831413c4a725f2ab2`), before any score.
* **W5 (ego-only floor): DONE, registered, inference seeds 0 AND 1: PASS** (`RESULT.md`, `RESULT.json`). All controls PASS.
* **W4 (camera-blind): the GPU rolls run DETACHED** (below). Until `final_id` exists the only W4 evidence is `PREVIEW_W4.md` (UNREGISTERED, first 1,500 of 4,754 windows).
* Instruments built and unit-tested (`tests/`: 15 passed; mutation arms RED). Nothing committed or staged; nothing on G:; Thor untouched; `launch_gate.py` untouched.

## What is running (all detached)

| process | what | output | ETA |
|---|---|---|---|
| `with_gpu_lock.py` job `refcv9-anticheat-w4` -> `code/run_rolls.sh` | stage 1 blind roll s0 (started 20:32:05, ~0.56 s/window) -> stage 2 identity roll s0 (control (a), all 4,754) -> stage 3 blind roll s1 (beside) | `C:/Users/Admin/refcv8_pf/refcv9_anticheat/bank/ztac_{blind_s0,identity_s0,blind_s1}.npz`; markers in `logs/rolls.log` (`ZZRLDONEZZ <tag> n=4754`, `ZZRLENDZZ`) | blind_s0 ~21:17, identity_s0 ~22:02, blind_s1 ~22:40 |
| `code/watch_and_score.sh` (CPU) | scores each pass when its bank is FINAL; writes ONLY under `.../refcv9_anticheat/result/` (never into the hold) | `result/final_prov/render/RESULT.md` (blind_s0 final; control (a)(b)(c) pending), `result/final_id/render/RESULT.md` (**the registered seed-0 W4 read**, controls included), `result/final_s01/render/RESULT.md` (seed 1 beside); markers `ZZWATCH...ZZ` in `logs/watch.log` | ~5 min after each bank |

The GPU lock is `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`, held by the `with_gpu_lock.py` wrapper and released when the chain ends. A foreign lock is
never broken. **Do not `cat logs/roll_*.log`**: the battery's own progress lines print a clip-id prefix; read `ZZ...ZZ` markers only (`grep -a -o "ZZ[A-Z0-9]*ZZ.*"`).

## Exact resume commands (CPU only)

If the watcher is not running (check `logs/watch.log`), or to redo a pass by hand:
```
bash "C:/Users/Admin/qland/hold_anticheat_r8/code/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat/code/resume_w4.sh" identity
```
(checks the banks are FINAL with 4,754 windows; runs `anticheat_score.py`; renders `RESULT.md` / `RESULT.json` INTO the package, replacing the W5-only ones.)
If the roll chain died: `tail logs/rolls.log`, then re-run the chain under the lock (stages are skipped when their bank is final):
```
cd C:/Users/Admin/refcv8_pf/refcv9_anticheat; source <package>/code/env_anticheat.sh
$PY -X utf8 "$LOCKDIR/with_gpu_lock.py" --job refcv9-anticheat-w4 --log $W/logs/rolls_lock.log --rec $W/logs/rolls_lockrec.json --max-wait-s 43200 -- "C:/Program Files/Git/usr/bin/bash.exe" <package>/code/run_rolls.sh
```
(the full Git-Bash path matters: a bare `bash` resolves to WSL on this box and the wrapper dies instantly).

## Then
1. Open `result/final_id/render/RESULT.md`; read the **Controls (verbatim)** block FIRST. **W4 is VOID unless** (a) identity == BASE bit-equal on all 4,754 windows, (b) changed fraction 100 %, (c) ego-input digests equal, and the derangement lines read `fixed_points 0`.
2. Copy `result/final_id/render/RESULT.md` + `RESULT.json` (or `final_s01` if seed 1 finished) over the package's `RESULT.md` / `RESULT.json`; drop `PREVIEW_W4.*` (superseded) or keep them labelled.
3. `python code/make_manifest.py --hold C:/Users/Admin/qland/hold_anticheat_r8` -> `MANIFEST.sha256`, the clip-id scan (must print `ZZMANIFESTOKZZ`), `expect.tsv`.
4. Land the hold (MM). The 21 paths in `expect.tsv` are all NEW (no existing repo path is edited).

## Next, if W4 passes (planned in the prereg, section 5)
refcv9-W must keep both: run the same two instruments on REFCV9-W (the camera-blind roll needs REFCV9-W's checkpoint; the floor is model-free and is
already fitted), with REFCV8's deltas as the reference. A nav-augmented floor (beside; not built) would separate nav from pixels in the lateral rows.
