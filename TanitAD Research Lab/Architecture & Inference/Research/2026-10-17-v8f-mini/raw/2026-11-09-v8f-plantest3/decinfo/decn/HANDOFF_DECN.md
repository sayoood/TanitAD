# HANDOFF_DECN (agent, dev box, 2026-11-09 07:50)
Card `../../PREREG_V8F_DECN.md` (frozen prefix sha256 cfa5bb1f..., not edited). Tier T0, CPU only. `IMPL_CHOICES_DECN.md` + `.sha256` written before any held-out row; `clips_extra.json` (859 clips, salt decn-v1, file sha256 f6813380a87c...) before any pull.

## State
* Controls as pytest: `test_decn_controls.py`, 18 passed (`TEST_DECN.log`, `TEST_DECN.done` rc=0). Control 1 on the 982 TUNE windows: R2 1.0, mean abs diff 0.0. Control 2 reproduction exact (M2b s0 speed 0.27074 / yaw -0.12238; s1 0.31447 / 0.26797).
* Real run DETACHED (started 07:47) via `launch_wmi.ps1` -> `job_run.cmd`: stage a (`run_decn.py a`: TUNE inputs, control 1, stream 1,000 clips read-only over sftp, one at a time, md5 each; per-clip shards in scratch, resumable), then stage b (`run_decn.py b`: ridge + MLP, one and four frames, N53 / N194 / N1053, R0 + R2 for M2b s0, s1, then R1 for 5 ladder tags; `runs/<rung>_<tag>_<N>.json`, written incrementally). Log `run_decn.log`. Marker `DECN.done` (`rc=<n> stage_a_rc=<n> finished=<time>`) written last. Stage a exit codes: 0 done, 10 network stop (stage b still runs on what exists, `STREAM_STOPPED.txt`), 2 control 1 failed, 3 md5 mismatch (everything stops).
* Rerun after a crash: `powershell -NoProfile -ExecutionPolicy Bypass -File launch_wmi.ps1` (shards and unit files present are skipped).

## Report (after DECN.done)
`cmd /c "call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd && C:\Users\Admin\venvs\tanitad\Scripts\python.exe report_decn.py"` in this folder -> `RESULT_DECN.md/.json`. Read "Validity first", then the outcome lines. The report can also be run on a partial `runs/` (outcome lines then read UNDETERMINED).

## Scratch to clean at the end (Master Mind, after the report exists)
`F:/TanitAD-scratch/decn_scratch/run1/` (shards `lat/`, `tune_inputs.npz`, `stream/`, `pt/`; about 1.5 GB plus F: allocation). Do not delete `F:/TanitAD-scratch/plantest_tree`. Nothing was deleted by this agent.
