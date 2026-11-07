# HANDOFF_WMACT (stopped at the context cap; PREREG_V8F_WMACT not yet read out)

Card: `../PREREG_V8F_WMACT.md`. Frozen prefix 5,478 bytes, sha256 `d21b14ccc3ea41692f2b615720b2814e81119019e78fc1ad85ba5b58474d9bb8` (verified, card not edited). Tier T0, CPU only.
No statistic, control value or outcome exists yet. Nothing here is a result.

## State
* `latents.pt` is being written by a background job (PID 30620 / 62896, `collect_latents.py`, started 09:02:33). Check: `ls wmact/latents.pt`; the log `collect_latents.log` prints one line when done. If the PIDs are gone and `latents.pt` is absent, rerun it (about 5 to 10 min; a smoke on 128 windows took 37 s, dominated by clip loads).
* `weights_check.json` is DONE (MEASURED, `wmact/weights_check.py`):
  * M2b HACC `wm.*` vs the source `sfresh_inputs/inputs/M2b_s{0,1}/ckpt_last.pt`: 145 of 145 tensors bit-equal at both seeds; adapter and `pos` 5 of 5 bit-equal; the only extra key in HACC is `hacc_w`. The HACC file's declared source sha equals the source file's sha.
  * C0 `wm.*` vs M2b: only 12 of 145 tensors bit-equal (differ in proj, act_embed, encoders ...); adapter and `pos` 0 of 5. So C0's WM DIFFERS and C0 must be read beside (item 5), by the same scripts with `C0_s{0,1}` paths in `wmact_lib.CKPT_C0` (`ckpt_last.pt` has the same `model` / `cfg` layout; unverified).
* Code written, in this directory: `wmact_lib.py` (loading, windows, rollouts, stats, bootstrap, zero-embedding copy), `collect_latents.py`, `weights_check.py`, `run_wmact.py` (UNTESTED as written: first run will likely need small fixes).

## Next steps, in order
1. Wait for `latents.pt`. Run `run_wmact.py` (command below). Fix errors. It writes `RESULT_WMACT.json` with controls 1, 2, 4, rows for both seeds, `sens`.
2. Check the controls BEFORE reading any row: control 1 (identity) max abs exactly 0.0; control 2 (zeroed level-1 `act_embed`) ACTION-USE and L1 sens exactly 0.0; control 3 (`CTRL3_SKILL_startdonor`, key SKILL with z_start of another window) must be below 0 and separated at BOTH seeds in every row; control 4 max abs diff below 1e-6. If one fails: report INVALID, no statistics.
3. Write `RESULT_WMACT.md` (tables per seed: level, k, n, SKILL, ACTION-USE, ACTION-USE-MEAN with intervals; level-1 per-channel rows; `sens` / `sens_rom` beside) and apply the card's PREDICTS / ACTION-REACTIVE / ACTION-INERT / MIXED rules per level, verbatim (ACTION-INERT also needs `sens_rom/all/k1` of the level, key `L{l}/sens_rom/all/k1`, threshold 0.01).
4. `test_wmact_controls.py` (pytest, CPU): controls 1, 2, 4 as tests on 64 windows, control 3 on 256 windows; plus `check_draws_equal_ci`. Save the pytest log.
5. C0 s0 / s1 rows beside (WM differs). Reproduce the whole thing for them: add C0 to `collect_latents.py` models dict (needs `load_model` on `ckpt_last.pt`, hmax 16 expected; assert).

## Commands (Git bash, from this directory)
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
export WMACT_CODE_ROOT="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tip_1246c8dd"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY collect_latents.py > collect_latents.log 2>&1      # only if latents.pt is absent
$PY weights_check.py                                    # done
$PY run_wmact.py
```
Re-create the code export if the scratchpad is gone (read-only on the mirror): `cd C:/Users/Admin/tanitad-push && git -c safe.directory='*' -c core.autocrlf=false archive origin/agent/arch-inf-20260803 stack/tanitad taniteval/taniteval | tar -x -C <dir>`; point `WMACT_CODE_ROOT` at `<dir>`. The tip moved during the session (1246c8dd at first read, e0cc3261 later); the files used were exported once and are unchanged since.

## Deviations and implementation choices (for the Master Mind to log; the card is untouched)
* D1. `E:/Projects/TanitAD/stack` is an old checkout (HEAD 37645fcc, 2026-09-19): it has no `v8f_beta.py`, no `WMAux.z1` / `scale_actions`. Code is a read-only `git archive` export of the mirror tip `origin/agent/arch-inf-20260803` into the scratchpad; `tanitad.__file__` is asserted to lie under it (printed in `RESULT_WMACT.json` as `tanitad_file`).
* D2. The card does not fix the start frame. Choice: z_start = frame 0 of the 16-frame training window (`hmax` 16, the format the WM was trained on); true futures are the later frames of the same window; windows with any padded frame (`frame_ok` false) are dropped (about 82 % keep, n per row is printed). Level-3 k = 3 needs frame 12, held by every kept window. A NOW-start variant (future frames from the clip) was not run.
* D3. The TRAIN-mean action needs the train cache, which is on box 2 (not touched). Used: the mean of the raw (a, yaw rate) over real cached eval-141 frames, passed through `scale_actions`.
* D4. Rollout is `HJEPAStack.rollout` with context 1 (planning form); training is teacher-forced with context 3. Levels 2 and 3 use macro-actions from `encode` of the same raw action sequence (donor / mean sequences re-encoded).
* D5. Per-channel reads: ACTION-USE with only that channel swapped to the donor (other true) and ACTION-USE-MEAN with only that channel set to the mean.
* D6. Bootstrap weights reuse the rng calls of `taniteval.ci._draws` (seed 0, B 2000), shared across rows; `check_draws_equal_ci` asserts the same selections as `ci._draws`. Statistic is a ratio of per-episode sums.

## Manifest (all `repo`-external, E: artifacts; nothing staged, committed or pushed)
| file | where |
|---|---|
| wmact_lib.py, collect_latents.py, weights_check.py, run_wmact.py, HANDOFF_WMACT.md, weights_check.json | `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wmact/` (one place only) |
| latents.pt, collect_latents.log | same directory (latents.pt pending) |
| code export (tip) | scratchpad `tip_1246c8dd` (one place; re-creatable from the mirror) |
