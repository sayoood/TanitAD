# HANDOFF S4C — eval-141 matched native rebuild RUNNING detached (2026-10-17 ~18:06)

**State:** train-600 DONE and verified (`ZZCHECK-OKZZ`, see RESULT_S4B section 6 / RESULT_S4C section 1). Eval-141 inputs staged + proven; the rebuild runs detached since 18:02 (CPU, 6 workers, ~8 clips/min measured, **ETA ~18:20**). A detached **post-build runner** (`code/post_build_eval141.py`, launched 18:04) polls the build log and, on `[build] DONE`, runs the completion check and the eval dry-run by itself, then banks the evidence. Nothing needs a model turn until it finishes.

## Paths
* Out: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/<clip>.v2ep.pt` (+ `_build_manifest.json`; `_manifest_sha256.jsonl` appears after the check). Stage: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb_stage/` (`build.log`, `post_build.log`, `check_report.json`, `dry_run_eval.out`, `stage_report.json`, `clips141.txt`).
* Code (`code/`): `stage_eval141_native256.py`, `launch_build_eval141.ps1`, `check_eval141_rebuild.py`, `post_build_eval141.py`, `launch_post_eval141.ps1`. Builder = banked `stack/scripts/pod_build_b1_epcache.py`, UNCHANGED.

## Is it done?
```
Get-Content C:\Users\Admin\tanitad-data\refav1-eval141-256rb_stage\post_build.log -Tail 6      # ends 'ZZPOST-DONEZZ' when everything finished
Get-Content C:\Users\Admin\tanitad-data\refav1-eval141-256rb_stage\build.log -Tail 2          # '[build] DONE ok=141 fail=0'
```
Then read `raw/s4c/post_summary.json` (check numbers + dry-run lines; copied there by the runner) and `raw/s4c/check_report_eval141.json` (`per_clip` rows).
Pass criteria: `ZZEVALCHECK-OKZZ` (C1 set == banked 141, C2 verify+keys+meta+n, C3 poses/actions bit-identical, C4 `ok=141 fail=0`) and the dry-run shows `[plan] 141 in pick`, `[source] kind=local256`.
(The dry-run's `[parity]` line says "DROPPED 11 of 141 ... inside physicalai-train-e438721ae894": that is INFORMATIONAL for `--split eval` — the eval-141 fixed by prereg has 11 clips inside the parity train corpus; the pick stays 141. Not a failure.)

## Exact next step (fresh agent, Haiku/Sonnet, ~10 min)
1. Confirm `ZZPOST-DONEZZ` in `post_build.log`. If the runner or build died: relaunch `powershell -NoProfile -ExecutionPolicy Bypass -File code\launch_build_eval141.ps1 -Workers 6` (resume is by content) then `... code\launch_post_eval141.ps1`. ⛔ keep CreateFlags `0x08000200`. ⚠ scripts need `-ExecutionPolicy Bypass` from bash.
2. Fill the PENDING block of `RESULT_S4C.md` section 3 from `raw/s4c/post_summary.json` (pooled pixel MAE vs banked, per-clip min/median/max, size, timings from build.log).
3. Delete `<stage>/_dry` and, if no rerun is likely, `<stage>/r0` (hardlinks to PhysicalAI mp4s, no extra space) after keeping `clips141.txt`.
4. Hand to the orchestrator: both dirs are final (train `refav1-train600-256` 22.19 GB, eval `refav1-eval141-256rb` ~5.3 GB); `_manifest_sha256.jsonl` in each verifies the copy on box2.

## Constraints still in force
CPU only (`CUDA_VISIBLE_DEVICES=-1`), no Thor, do not touch `tanitad-box2`. C: free was 11.9 GB at 18:05 and falling (other sessions); the eval build needs ~5.3 GB; RAM free was 0.9-2.2 GB with 6 workers (they hold ~1.3 GB each) — do not add load.
