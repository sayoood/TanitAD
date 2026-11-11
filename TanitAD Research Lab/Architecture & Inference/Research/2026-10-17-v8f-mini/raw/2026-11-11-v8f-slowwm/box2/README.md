# SLOWWM box-2 launch package (built 2026-11-11, NOT launched)

Runs the five SLOW / PCA-B arms of `theory/slowwm/PREREG_V8F_SLOWWM.md` (sha256 `7c2775e3...94ca`) on box 2. Code of record: commit `f7e790ab57e10a9004b04b206195a0ae5a5295b8`. Pattern: `theory/wmpool/box2/` (the template is unchanged), with the basis lever in place of the attention pool.
Nothing here touched box 2; nothing is staged, committed or pushed.

## Launch (dev box, Git bash; the PI or Master Mind starts it)

```
export SLOWWM_COMMIT=f7e790ab57e10a9004b04b206195a0ae5a5295b8 SLOWWM_SSH="$(command -v ssh)" SLOWWM_SFTP="$(command -v sftp)"
bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/box2/slowwm_chain.sh
```
Run it detached (WMI `cmd /c ... < NUL`), as the other chains. Optional: `SLOWWM_AMENDMENT=<file>` (72 h rule), `SLOWWM_RESUME=1` after a STOP, `SLOWWM_PRED_STATE=<dir>` (default `C:/Users/Admin/obd_seed3_state`).

## Order
1. **wait**: for the OBD seed-3 chain. Ended = `DONE.txt` (first bytes `ZZ OBD3-DONE`), or a non-empty `STOP.txt` whose head (text before ` :: `) does not end in `rc=12`; a busy stop keeps waiting. The shared marker `C:/Users/Admin/box2_ship/BOX2_BUSY.txt` must be absent, then the chain claims it (released on every exit). Timeout 1152 polls x 300 s, then rc 12.
2. **ship**: the commit's trainer closure from the mirror `C:/Users/Admin/tanitad-push` (read only, `core.autocrlf=false`, blob hashes verified) into `D:\tanitad\slowwm\repo`; the checker and frozen argvs to `slowwm\check`; `slow_basis.py` (from `theory/slowwm/`) and `clips8.json` (from `theory/ladder/`) to `slowwm\basiscode`, sha256 pinned in the checker and re-checked on box 2. Needles: `--wm-basis`, `wm_basis_readback`, `zbasis`, `norm_mean_tokens`.
3. **basis**: `slow_basis.py fit` (train, 40,358 windows, `--exclude 540:600`, spool), `cache` slow and pca x train and eval into the ROOT `D:\tanitad\zbasis` (cache appends `<kind>\<split>`), `check3` on the 8 clips, counts 4572 / 141, then `BASIS.json`, `FIT_RECORD.json`, `SPECTRA.json`, `CHECK3.json` pulled (sha256 against box 2's certutil) and re-judged on the dev box (`verify-basis`). Refuses below 6 GB free on D:. Basis caches are never deleted.
4. **pilot**: 200 steps of SLOW_s0 (its argv plus `--steps 200`); the same job gates all five full argvs; the projected total must be <= 72 h or the chain stops for an amendment.
5. **runs**: SLOW_s0, PCA-B_s0, SLOW_s1, PCA-B_s1, SLOW_rep. Each: argv pin against `frozen/argvs_slowwm.json` (sha256 `192168f1...`), gate (selfcheck, BUILT-lever assertions incl. `levers.wm_basis_readback`, token), training under the token, post check (non-WM tensors bit-equal M2b's), bank to `C:/Users/Admin/t2v8f_pulls/slowwm_<tag>/`, cleanup of the run's own gate dirs after `BANK_VERIFIED`.

State `C:/Users/Admin/slowwm_state`. Sentinels live in their own files only: `DONE.txt` (`ZZ SLOWWM-DONE`), `STOP.txt` (`ZZ SLOWWM-STOP`), `ARM_REFUSED_<tag>.txt`. A gate / pin / token refusal of one arm continues with the next and ends with STOP, never DONE. Every other failure stops the chain.
Exit codes: 3 permit refused, 4 gate, 5 job or transfer, 6 disk, 7 pull sha mismatch, 8 commit or package or transport, 9 ship refused, 10 pilot unreadable, 11 box timeout, 12 busy or not ended, 13 blocked tag.

## Tests (recorded in `tests/`)
* `tests/UNIT_RESULT.log`: `python -I tests/test_slowwm_check.py` -> **UNIT ok=69 bad=0** (frozen argv pin, audit, builder guard, 16 mutations of the basis read-back plus missing / wrong-kind / sha cases, check3 and fit records, 72 h permit, verify-basis).
* `tests/DRYRUN_RESULT.log`: `bash tests/test_slowwm_dryrun.sh` -> **DRYRUN ok=23 failed=0** (wait: RUNNING, DONE, wrong-prefix DONE, busy STOP, non-busy STOP, busy marker, timeout rc 12; `SLOWWM_DRY` refusals; commit refusals; standing STOP; stage failure stops with STOP and releases the marker; poison ssh/sftp/scp never ran).

## UNVERIFIED (read before launching)
* **No full-flow fake-box dry run.** WMPOOL's harness (fake box, trainer stand-in, planted selfchecks) was not rebuilt for the basis lever. Not exercised end to end: check3-fail stop (rc 4 path of the basis stage), one refused arm ending in STOP, pull sha mismatch (rc 7), the stage / pilot / cleanup jobs on cmd.exe. The shell paths for those are line-for-line WMPOOL's with renames; the checker side (check3, fit, verify-basis, permit, gate read-back) is unit tested.
* `wm_groups_grad` key `proj` is checked only if present; the WM parameter count is read, not pinned (first real selfcheck decides).
* The OBD3 STOP head is matched by `rc=12` only; any non-empty STOP.txt counts as a stop of that chain (its prefix is not pinned).
* `slow_basis.py` CLI is taken from `theory/slowwm/HANDOFF_SLOWWM.md` and the checker; first real run is the test.
* A `__pycache__` folder from the test runs sits in the package (a shell delete in `dagland/` is refused by the hook); it is not in the manifest.

## Cleanup log
Scratch `F:/TanitAD-scratch/slowwm_box2_build/` (the argv copy, one unit-test tmp dir, one dry-run tree, ~78 KB) was deleted by literal path on 2026-11-11 after the tests were recorded; both sources (this package) exist.

`PACKAGE_MANIFEST.sha256`: sha256 of every package file; check with `sha256sum -c PACKAGE_MANIFEST.sha256` from this folder.
