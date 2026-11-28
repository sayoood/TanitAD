# HANDOFF S4B — native 256x640 v2ep rebuild of the 600 stage-A TRAIN clips (2026-10-17 ~16:40)

**State: inputs staged and proven, builder validated, the 600-clip build is RUNNING detached (CPU only, 6 workers). Nothing else is pending.**
Started 16:31 local; measured 10 clips per 127 s (~12.7 s/clip with 6 workers, CPU shared); ETA ~18:30-18:45 (about 21 GB final). No agent needs to wait: check, then run the completion check.

## Paths
* Output (what `--src-dir` wants): `C:/Users/Admin/tanitad-data/refav1-train600-256/<clip>.v2ep.pt` (+ `_build_manifest.json`; on completion also `_manifest_sha256.jsonl`).
* Stage (inputs, hardlinks, small): `C:/Users/Admin/tanitad-data/refav1-train600-256_stage/` — `clips600.txt`, `build.log`, `stage_report.json`, `validation_native256.json`.
* Code (package `code/`): `stage_train600_native256.py` (inputs), `validate_native256.py` (priority 1), `launch_build_train600.ps1` (detached launcher), `check_train600_native256.py` (completion check). Builder = banked `stack/scripts/pod_build_b1_epcache.py`, UNCHANGED.
* Evidence copy: `raw/s4b/`.

## Is it still running?
```
Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'pod_build_b1_epcache' } | select ProcessId     # parent + 6 workers while running
(Get-ChildItem C:\Users\Admin\tanitad-data\refav1-train600-256 -Filter *.v2ep.pt).Count                   # 600 when done
Get-Content C:\Users\Admin\tanitad-data\refav1-train600-256_stage\build.log -Tail 3                      # ends '[build] DONE ok=600 fail=0'
```
## If the process is gone before 600 (crash, reboot, session kill)
Just relaunch — resume is by CONTENT (every file on disk is re-verified, bad ones deleted, done ones skipped):
`powershell -NoProfile -File "<package>\code\launch_build_train600.ps1" -Workers 6`
* ⛔ The launcher MUST keep `CreateFlags 0x08000200` (CREATE_NO_WINDOW|NEW_PROCESS_GROUP). A plain `Win32_Process.Create` child that imports torch dies instantly with `forrtl: error (200): window-CLOSE event` (measured twice; DETACHED_PROCESS alone also failed).
* RAM is the limit, not CPU: workers sit at ~1.3 GB each and other sessions hold ~5 GB; free was 3.8 GB. Keep 6 or fewer. Disk: needs ~22 GB, builder has no disk guard — keep `C:` free >= 8 GB.
* `_failures.json` in the output dir lists failed clips (re-run resumes them).

## Exact next step when the count reaches 600
1. `cd "<package>\code"; $env:CUDA_VISIBLE_DEVICES='-1'; C:\Users\Admin\venvs\tanitad\Scripts\python.exe check_train600_native256.py --workers 4` -> must print `ZZCHECK-OKZZ` (C1 set == picker's 600, C2 eval-141 overlap 0, C3 content-verify + same keys/dtypes/geometry as an eval native file, C4 build.log `ok=600 fail=0`). It also writes `_manifest_sha256.jsonl` (use it to verify the copy on the encode machine).
2. Dry-run preflight against the finished dir (must show `train==labels=True`, `n_disqualifying 0`, `[source] kind=local256`):
   `python code/pull_encode_pooled.py --source local --src-dir C:/Users/Admin/tanitad-data/refav1-train600-256 --src-geom 256 --split train --n 600 --grids 4x10,8x20 --dry-run --out C:/Users/Admin/tanitad-data/refav1-train600-256_stage/_dry`  (PYTHONPATH=E:/Projects/TanitAD/stack;E:/Projects/TanitAD/taniteval, `CUDA_VISIBLE_DEVICES=-1`). Passed already on the empty dir.
3. Determinism spot check: sha256 of `8f387977-4086-466d-bba4-b04f8cdc0743.v2ep.pt` and `bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a.v2ep.pt` in the output must equal those in `<stage>/_valtrain/` (same clip built twice, different process).
4. Fill the PENDING block of `RESULT_S4B.md` (timings from `build.log`, `check_report.json`, `Get-ChildItem | measure Length`), then delete `<stage>/_valroot`, `<stage>/_valtrain`, `<stage>/_dry`. Keep the hardlinked `<stage>/r0` (no extra space) only if a rerun is likely; otherwise delete the stage after copying `clips600.txt`.
5. Orchestrator decision still open (see RESULT_S4B.md section 2): the rebuilt pixels differ from the banked eval v2ep by MAE 0.76/255 (swscale/PyAV version), so train and eval are NOT yet byte-matched; option = rebuild eval-141 through the same builder (+5.3 GB, ~15 min at 6 workers) for a matched A/B.

## Not done / not verified
* No GPU encode (by design). Nothing staged in git; the orchestrator lands the package files.
* Banked native copies of TRAIN clips: none found locally (only the 141 eval eps), so the bit-for-bit comparison was done on 2 EVAL clips.

## Orchestrator decision (2026-10-17 ~16:50): MATCHED eval through the same builder
* The rebuilt train pixels differ from the banked native eval v2ep: mean |Δ| 0.76/255 from the PyAV/swscale YUV→RGB step, giving a pooled-feature residue of about 2.5 % rel-L2.
* **That is a systematic train/eval shift, and removing it costs about 15 min of CPU and +5.3 GB. Do it.**
* After the train-600 build reaches `DONE ok=600`:
  1. Run the completion check (`ZZCHECK-OKZZ`) and the dry-run.
  2. Rebuild the **eval-141** clips with the same builder, same flags, `--workers 6`, into `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/`.
  3. Run the dry-run with `--split eval --source local --src-dir <that dir> --src-geom 256` and assert 141 resolved.
* The 25 GB cap is raised to **32 GB** for this package; C: had 50 GB free at 16:50.
* **M1 trains AND evaluates on this builder's output.** The banked native eval v2ep stay the reference for refav1-cache comparisons only.
