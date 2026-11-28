# W-A7 HANDOFF (2026-11-14). CPU only, 1 thread, no git writes, nothing shipped.

Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7/` (W below). Frozen definition: `../A2/PREREG_P5C_A2.md` section 6. Inputs: `INPUTS.md`.

## State
- Built: `w_a7_read.py` (driver: prep -> pinned dual labeller subprocess -> tracked scoring of 7 arms per row -> bootstrap/controls -> `run/w_a7_result.json` + `run/W_A7_RESULT.md`; resumable; last log line `ZZW_A7 DONE <row>`), `test_w_a7.py` (15 passed, 1 real-scorer test not yet run; includes mutations: A7 skipped, bootstrap seed changed, each of controls i-iv failing -> INVALID), `run_w_a7.sh`, launchers.
- Per row ONE scorer call scores [PDM-Closed, R_noA7, R_A7, T_noA7, T_A7, RED (R_noA7 with index-19 heading +0.5 rad), IDN_R_pre (R with h19:=h18), IDN_R_A7 (A7 of it)]; every 10th row is rebuilt and rescored for control (iv).
- Units are PDMS x 100 (pp): the card's B = 0.7031 is in pp, so the REBUILD bar "point <= -0.10" is -0.10 pp (-0.001 on 0..1).
- Smoke (`run_smoke`, 20 rows, deltas withheld) and the full run were started DETACHED. Both sit behind the 8 GB free-virtual-memory gate (other sessions held it at 3.5-7 GB all afternoon; log lines `VM-WAIT`). The smoke is a pure code check: the chain `launch_chain.cmd` (pid 47136 at launch) starts the full run ONLY if the smoke log reaches `ZZW_A7 DONE SMOKE` with no Traceback and no failed control other than (ii).

## Commands
- Chain: `$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_launchkit\w_a7\launch_chain.cmd < NUL'; ProcessStartupInformation=$su}`
- Full only (no smoke gate): same with `launch_full.cmd`. Resume after any kill: just relaunch it (labels skip finished logs, scores skip finished rows).
- Logs: `run_smoke/w_a7.log`, `run/w_a7.log` (heartbeat first line), `run*/labeller.log`.

## Reading the result
`grep "ZZW_A7 DONE" run/w_a7.log` gives W-A7-PASS / W-A7-REBUILD / INVALID. Detail in `run/W_A7_RESULT.md` (Delta [95% CI] per label, four controls, n rows/logs, blobs). If a `ZZW_A7 STOP` line appears, it names the cause (pinned-blob mismatch, labeller rc, VM wait over 6 h).
Expected full wall time (ESTIMATED): labeller 0.6 s/row ~ 10 min, scoring ~2-3 s/row ~ 40-60 min, plus the VM-gate wait.

## Deletions
Not done: the shell denied my `rm` of my own scratch (`F:/tanitad_scratch/w_a7_tmp/`: `tree/` 12 MB git export, `pt/` pytest basetemp, `list.txt`, `mk.py`, `mk2.py`). The chain uses that folder as TMP, so delete it by literal path after `ZZW_A7 DONE`. Nothing else was deleted.
