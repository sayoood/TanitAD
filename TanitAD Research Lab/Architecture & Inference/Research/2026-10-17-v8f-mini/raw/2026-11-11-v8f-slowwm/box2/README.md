# SLOWWM box-2 launch package (built 2026-11-11, amended 2026-11-11 for `AMEND_SLOWWM_2.md`; NOT launched)

Runs the five SLOW / PCA-B arms of `theory/slowwm/PREREG_V8F_SLOWWM.md` (sha256 `7c2775e3...94ca`) on box 2. Code of record: commit `f7e790ab57e10a9004b04b206195a0ae5a5295b8`. Pattern: `theory/wmpool/box2/` (the template is unchanged), with the basis lever in place of the attention pool.
Nothing here touched box 2; nothing is staged, committed or pushed.

## Launch (dev box, Git bash; the PI or Master Mind starts it)

```
export SLOWWM_COMMIT=f7e790ab57e10a9004b04b206195a0ae5a5295b8 SLOWWM_SSH="$(command -v ssh)" SLOWWM_SFTP="$(command -v sftp)"
bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/box2/slowwm_chain.sh
```
Run it detached (WMI `cmd /c ... < NUL`), as the other chains. Optional: `SLOWWM_AMENDMENT=<file>` (72 h rule), `SLOWWM_RESUME=1` after a STOP, `SLOWWM_PRED_STATE=<dir>` (default `C:/Users/Admin/obd_seed3_state`), `SLOWWM_PRED_DONE_PREFIX=<text>` (first bytes of the predecessor's `DONE.txt`, default `ZZ OBD3-DONE`).

### Relaunch after the basis STOP of 2026-11-11 20:44, behind the STOPBAL second dose
The chain stopped at `basis` (`check3 ran on fewer than 8 clips`); box 2 now runs the STOPBAL second dose, so the relaunch waits for it (`STOP.txt` of dose 2 with a head ending ` rc=12` = busy, keeps waiting; `DONE.txt` starting `ZZ STOPBAL2-DONE` releases; any other non-empty `STOP.txt` also counts as ended; the shared busy marker must be free):
```
export SLOWWM_COMMIT=f7e790ab57e10a9004b04b206195a0ae5a5295b8 SLOWWM_SSH="$(command -v ssh)" SLOWWM_SFTP="$(command -v sftp)"
SLOWWM_RESUME=1 SLOWWM_PRED_STATE=C:/Users/Admin/stopbal_dose2_state SLOWWM_PRED_DONE_PREFIX="ZZ STOPBAL2-DONE" bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/box2/slowwm_chain.sh
```
* **Resume vs restart.** `SLOWWM_RESUME=1` renames this chain's own `slowwm_state/STOP.txt` to `STOP.<timestamp>.txt` (kept, never deleted) and goes on from the stage markers in `slowwm_state/done/`. `ship.done` stays but now records the sha256 of the checker it shipped (`checker=<sha>`): a marker written by an older chain, or for another checker, is ignored, so the amended `slowwm_check.py` is shipped to box 2 again before the basis stage re-hashes it there (the stage refuses otherwise, rc 16). The ship stage overlays only SLOWWM's own tree (`D:/tanitad/slowwm/...`); it never touches `D:/tanitad/repo` or another chain's files. The `basis` stage has no marker (it stopped) and runs again in full: fit, the four caches and the two per-split check3 runs write only SLOWWM's own folders (`D:/tanitad/zbasis`, `D:/tanitad/slowwm/basis`), replacing files by name, deleting nothing. Nothing of another chain is removed, moved or edited by the relaunch.
* Order at start: commit and transport checks, the stale `STOP.txt` rename, then the wait (no busy marker is claimed and nothing is shipped until the predecessor has ended), then ship (if needed) and basis.

## Order
1. **wait**: for the predecessor chain (default the OBD seed-3 chain; see the relaunch block for STOPBAL dose 2). Ended = `DONE.txt` (first bytes = `SLOWWM_PRED_DONE_PREFIX`, default `ZZ OBD3-DONE`), or a non-empty `STOP.txt` whose head (text before ` :: `) does not end in `rc=12`; a busy stop keeps waiting. The shared marker `C:/Users/Admin/box2_ship/BOX2_BUSY.txt` must be absent, then the chain claims it (released on every exit). Timeout 1152 polls x 300 s, then rc 12.
2. **ship**: the commit's trainer closure from the mirror `C:/Users/Admin/tanitad-push` (read only, `core.autocrlf=false`, blob hashes verified) into `D:/tanitad\slowwm\repo`; the checker and frozen argvs to `slowwm\check`; `slow_basis.py` (from `theory/slowwm/`) and `clips8.json` (from `theory/ladder/`) to `slowwm\basiscode`, sha256 pinned in the checker and re-checked on box 2. Needles: `--wm-basis`, `wm_basis_readback`, `zbasis`, `norm_mean_tokens`.
3. **basis**: `slow_basis.py fit` (train, 40,358 windows, `--exclude 540:600`, spool), `cache` slow and pca x train and eval into the ROOT `D:/tanitad/zbasis` (cache appends `<kind>\<split>`), `check3` once per split of `clips8.json` (eval 6 clips against the eval cache, train 2 clips against the train cache; `CHECK3.eval.json`, `CHECK3.train.json`), combined with the cov identity into ONE `CHECK3.json` (schema `slowwm-check3-v2`; AMENDMENT 2), counts 4572 / 141, then `BASIS.json`, `FIT_RECORD.json`, `SPECTRA.json`, `CHECK3.json` pulled (sha256 against box 2's certutil) and re-judged on the dev box (`verify-basis`). Refuses below 6 GB free on D:. Basis caches are never deleted.
4. **pilot**: 200 steps of SLOW_s0 (its argv plus `--steps 200`); the same job gates all five full argvs; the projected total must be <= 72 h or the chain stops for an amendment.
5. **runs**: SLOW_s0, PCA-B_s0, SLOW_s1, PCA-B_s1, SLOW_rep. Each: argv pin against `frozen/argvs_slowwm.json` (sha256 `192168f1...`), gate (selfcheck, BUILT-lever assertions incl. `levers.wm_basis_readback`, token), training under the token, post check (non-WM tensors bit-equal M2b's), bank to `C:/Users/Admin/t2v8f_pulls/slowwm_<tag>/`, cleanup of the run's own gate dirs after `BANK_VERIFIED`.

### Check 3 under AMENDMENT 2 (`theory/slowwm/AMEND_SLOWWM_2.md`, sha256 `62bb5109...b3c5`)
* **A. 8 distinct clips.** `check3_problems` refuses unless both split records are present, each carries clips, the union is 8 DISTINCT clips (no overlap), the identity (1e-4) passes on both splits for both kinds, and the wrong-kind basis fails the identity on both splits for both kinds.
* **B. cov.** SLOW: V^T (C + ridge I) V = I within 1e-3 (computed on box 2 from the shipped `basis_slow.npz` (`proj`, `meta_json.ridge`) and `fit_stats.npz` (`C`); also via cov(u) = I - ridge V^T V, the larger of the two deviations is judged). PCA-B: cov(u) = I within 1e-3. The record carries SLOW's per-direction deviation of cov(u) from I as a beside number (`cov_identity.slow_cov_u_dev_from_I_per_direction`); it is reported, not judged. The number decides, not the flag beside it. A record whose overall `pass` is true while either cov check, or any other check, fails is refused (`overall pass is true while a check above fails`).
* `slow_basis.py` is unchanged (pinned sha256); its own per-split `pass` / `slow_pass` fields are kept inside `splits.<split>` and in `cov_u_fit_frames_raw` as read, not judged on `slow_pass`.

### Basis disk rule (2026-11-11 ~21:40, before the basis stage ran again; not a card change)
* The 6 GB of the basis stage counted bytes the stage would allocate, and the dev-box probe read whole GiB (`Floor(Free/1GB)`). With the first run's files on box 2 the re-run would have refused: MEASURED 21:28, D: free 6,431,051,776 bytes (5 by that probe), and 2,977,042,027 bytes in 14,152 files already under `D:\tanitad\zbasis` (spool, slow, pca, fit).
* A re-run reads the spool files that exist and replaces every cache and fit file in place (a `.tmp`, then `os.replace`), so those bytes are not new allocation. The need is now **6 GB less the bytes held under `D:\tanitad\zbasis`, at least 1 GB**, in bytes on both sides: `basis_need_bytes` in `slowwm_check.py` (box 2, `cmd_basis`) and `need_basis_disk` in `stages/slowwm_common.sh` (dev box, `slowwm_check.py disk-basis`). A fresh build still needs 6 GB. Unreadable probes refuse (rc 19).
* Tests: 15 unit cases, including red arms and a `cmd_basis` wiring case that fails under the old rule (MEASURED by mutation on a scratch copy: 110/1). The ship and run stages keep `need_free_gb`.

State `C:/Users/Admin/slowwm_state`. Sentinels live in their own files only: `DONE.txt` (`ZZ SLOWWM-DONE`), `STOP.txt` (`ZZ SLOWWM-STOP`), `ARM_REFUSED_<tag>.txt`. A gate / pin / token refusal of one arm continues with the next and ends with STOP, never DONE. Every other failure stops the chain.
Exit codes: 3 permit refused, 4 gate, 5 job or transfer, 6 disk, 7 pull sha mismatch, 8 commit or package or transport, 9 ship refused, 10 pilot unreadable, 11 box timeout, 12 busy or not ended, 13 blocked tag.

## Tests (recorded in `tests/`)
* `tests/UNIT_RESULT.log`: `python -I tests/test_slowwm_check.py` -> **UNIT ok=96 bad=0** (frozen argv pin, audit, builder guard, 16 mutations of the basis read-back plus missing / wrong-kind / sha cases, the 8-clip combine and the cov identity on synthetic bases with red arms (6 / 7 / duplicate clips, wrong kind passing, identity failing, SLOW cov off by ~2e-3, PCA-B off, overall pass true with a cov failure, no cov block, SMOKE), fit records, 72 h permit, verify-basis incl. a failing cov record).
* `tests/DRYRUN_RESULT.log`: `bash tests/test_slowwm_dryrun.sh` -> **DRYRUN ok=33 failed=0** (wait: RUNNING, DONE, wrong-prefix DONE, busy STOP, non-busy STOP, busy marker, timeout rc 12; the STOPBAL dose-2 relaunch: running waits, `ZZ STOPBAL2-DONE` releases, an OBD3 DONE does not, busy STOP waits, busy marker waits, non-busy STOP ends; resume: old ship marker re-ships, a marker for another checker re-ships, a marker for this checker is skipped, the old STOP is renamed and kept, no RESUME refuses; `SLOWWM_DRY` refusals; commit refusals; standing STOP; stage failure stops with STOP and releases the marker; poison ssh/sftp/scp never ran).

## UNVERIFIED (read before launching)
* **No full-flow fake-box dry run.** WMPOOL's harness (fake box, trainer stand-in, planted selfchecks) was not rebuilt for the basis lever. Not exercised end to end: check3-fail stop (rc 4 path of the basis stage), one refused arm ending in STOP, pull sha mismatch (rc 7), the stage / pilot / cleanup jobs on cmd.exe. The shell paths for those are line-for-line WMPOOL's with renames; the checker side (check3, fit, verify-basis, permit, gate read-back) is unit tested.
* `wm_groups_grad` key `proj` is checked only if present; the WM parameter count is read, not pinned (first real selfcheck decides).
* `cmd_basis` (the two check3 runs, the combine, `CHECK3.json`) is not run end to end here (no box); `build_check3`, `cov_identity` and `check3_problems` are unit tested, and the dev-box `verify-basis` re-judges the pulled record. The first real basis run is the test of the per-split CLI calls (`--split train` against the train cache).
* The predecessor's STOP head is matched by `rc=12` only; any non-empty STOP.txt counts as a stop of that chain (its prefix is not pinned).
* `slow_basis.py` CLI is taken from `theory/slowwm/HANDOFF_SLOWWM.md` and the checker; first real run is the test.
* A `__pycache__` folder from the test runs sits in the package (a shell delete in `dagland/` is refused by the hook); it is not in the manifest.

## Cleanup log
Scratch `F:/TanitAD-scratch/slowwm_amend2_build/` and `F:/TanitAD-scratch/slowwm_box2_build/` (amendment-2 test runs; the unit-test tmp dirs and one dry-run tree) were deleted by literal path on 2026-11-11 after the tests were recorded.
Earlier: scratch `F:/TanitAD-scratch/slowwm_box2_build/` (the argv copy, one unit-test tmp dir, one dry-run tree, ~78 KB) was deleted by literal path on 2026-11-11 after the tests were recorded; both sources (this package) exist.

`PACKAGE_MANIFEST.sha256`: sha256 of every package file; check with `sha256sum -c PACKAGE_MANIFEST.sha256` from this folder.
