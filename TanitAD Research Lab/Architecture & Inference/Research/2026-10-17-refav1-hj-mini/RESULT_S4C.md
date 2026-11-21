# RESULT S4C — train-600 completion + matched eval-141 native rebuild (2026-10-17)

**Headline:** the train-600 native 256x640 build is COMPLETE and verified (600/600, `ZZCHECK-OKZZ`, 22.19 GB, deterministic). The eval-141 rebuild through the SAME builder is staged with inputs proven equal to the HF corpus bytes and is RUNNING detached; its completion check runs by itself (section 3 PENDING, see `HANDOFF_S4C.md`).

## 1. Train-600 completion (MEASURED; artifacts `raw/s4c/`)
| check | result |
|---|---|
| build.log | `[build] DONE ok=600 fail=0 20.67GB in 82.3min (437eps/h)` (16:31 -> 17:54), `raw/s4c/build_train600.log` |
| `check_train600_native256.py --workers 3` | **`ZZCHECK-OKZZ`** in 47 s. C1 set == picker's 600 (0 missing/extra); C2 eval-141 overlap 0; C3 0 bad of 600 (banked content verify + same keys/dtypes/geometry as a banked eval file); C4 no `_failures.json`, DONE `ok=600 fail=0`. `raw/s4c/check_report_train600.json` |
| frames per clip | {191,199,200,201,202,207,209} (all consistent with poses/actions/jpeg_len) |
| size | 22,192,351,528 B = **22.19 GB** (20.67 GiB), `_manifest_sha256.jsonl` written (`raw/s4c/manifest_sha256_train600.jsonl`) |
| dry-run preflight (`--source local --src-dir ...refav1-train600-256 --src-geom 256 --split train --n 600`) | `train==labels=True train&eval=False`; `[parity] n_in 600, n_disqualifying 0, n_kept 600`; `[source] kind=local256 ... resample416=False`; `[plan] 600 in pick, 0 done, 600 to build` (`raw/s4c/dry_run_train_post.out`) |
| determinism | sha256 of clips `8f387977` and `bf5ec76e` equal the copies built earlier in a different process (byte-identical) |

## 2. Eval-141 inputs (MEASURED, `raw/s4c/stage_report_eval141.json`, script `code/stage_eval141_native256.py`)
* clip set = the 141 clip ids of the banked native eval dir (`refav1-eval141/eps`); `clip_sets()` asserts all pass (eps == eval cache, subset of the 4,713 listing, disjoint from train pool and train labels, n=141). Pick overlap with train pool 0.
* mp4: the 141 local PhysicalAI files match HF `camera/camera_sha256.json` (0 missing, 0 mismatch; 5 s) -> no download, hardlinks only.
* timestamps: 141 from HF `timestamps/timestamps.tar`, local parquets byte-identical (0 differ). egomotion: 141 from HF `egomotion/egomotion_alpamayo.tar` repacked into a STORED zip, local parquets byte-identical (0 differ). intrinsics: local CSV vs HF `calibration/camera_intrinsics.parquet`: 0 mismatch, max |diff| 1.14e-13.
* builder parity gate precheck (role=train, mode=exclude, as the builder runs it): kept 141, dropped 0 (11 are inside physicalai-train-e438721ae894, 0 of the 40 deployed-val).
* The two HF tars (~2.0 GB) were scratch and are deleted. Token read in place, never printed.

## 3. Eval-141 rebuild — PENDING (filled by the next agent from `raw/s4c/post_summary.json`)
Launched 18:02 with `code/launch_build_eval141.ps1 -Workers 6` (CreateFlags `0x08000200`), builder banked and unchanged, flags identical to train-600 except `--clips clips141.txt`, `--root`/`--out` to the eval dirs. Output `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/`.
* build.log final line: ___
* `check_eval141_rebuild.py` -> `ZZEVALCHECK-OKZZ`: ___ (C1 141 == banked set; C2 verify/keys/meta/n; C3 poses+actions bit-identical to banked `refav1-eval141/eps`; C4 `ok=141 fail=0`)
* pixel difference vs banked native eval (9 evenly spaced PNG frames per clip, 141 clips): pooled MAE ___ /255, per-clip min/median/max ___, max abs level ___, fraction of pixels differing ___; n jpeg_buf bit-equal ___ (expected: 0, the same PyAV 18 / swscale 9 residue as train, S4B reported MAE 0.76/255 on 2 clips)
* eval dry-run (`--split eval --source local --src-dir ...refav1-eval141-256rb --src-geom 256`): ___ (syntax pre-tested on the partial dir: `ZZSPLIT-OKZZ split=eval pick=141`, `[source] kind=local256`, `[plan] 141 in pick`; the `[parity]` line reports DROPPED 11 of 141 as informational only, the pick stays 141)
* size / timing: ___ GB (banked native eval 5.3 GB), ___ min

## 4. Notes
* Launch quirks: PowerShell scripts need `-ExecutionPolicy Bypass` when started from bash; WMI `CreateFlags 0x08000200` is still mandatory for a torch child.
* Resources: C: free fell from 19 GB (17:55) to 11.9 GB (18:05) because of other sessions; RAM free 0.9-2.2 GB with six 1.3 GB workers. Eval needs ~5.3 GB.
* The 11 eval clips inside `physicalai-train-e438721ae894` are a property of the prereg'd eval-141, not of this rebuild.

## 5. Deliverable manifest
| artifact | where | copies |
|---|---|---|
| 600 native v2ep + `_manifest_sha256.jsonl` | `C:/Users/Admin/tanitad-data/refav1-train600-256/` (dev box C:) | **ONE place only** (rebuildable in 82 min from staged inputs; orchestrator ships to box2) |
| eval-141 matched v2ep (in progress) | `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/` | **ONE place only** (rebuildable ~18 min) |
| staged inputs | `C:/Users/Admin/tanitad-data/refav1-train600-256_stage/`, `refav1-eval141-256rb_stage/` | hardlinks + small files, rebuildable |
| code | `code/stage_eval141_native256.py`, `launch_build_eval141.ps1`, `check_eval141_rebuild.py`, `post_build_eval141.py`, `launch_post_eval141.ps1` | repo working tree, **not staged** (orchestrator lands) |
| docs / evidence | `RESULT_S4C.md`, `HANDOFF_S4C.md`, `RESULT_S4B.md` (section 6 filled), `raw/s4c/*` | repo working tree, not staged |

## Update 2026-10-17 (orchestrator, before landing): the eval-141 rebuild COMPLETED and PASSED
The headline's "RUNNING detached" is superseded. MEASURED in `raw/s4c/post_summary.json` and `raw/s4c/check_report_eval141.json`:
* Build: `[build] DONE ok=141 fail=0 4.95GB in 13.2min`.
* Check: **`ALL_OK: true`**. C1: the set equals the banked eval-141 (0 missing, 0 extra). C2: 0 bad. C3: 0 not bit-equal on poses, actions, keys and geometry. C4: no `_failures.json`.
* Pixels vs the banked eval v2ep, over 141 clips × 9 frames = 1,269 frames:
  * pooled MAE **0.730 / 255** (per clip 0.312–0.906, median 0.750);
  * max |diff| **3** levels;
  * 69.6 % of pixels differ.
* This matches S4B's train-side finding: the same PyAV/swscale YUV→RGB step, so the result is **not pixel-exact** to the banked cache.
* ⇒ Eval-141 and train-600 now come from **one** builder and **one** decode path, so train and eval are matched. ⛔ They must not be mixed with the banked eval v2ep within one arm.
