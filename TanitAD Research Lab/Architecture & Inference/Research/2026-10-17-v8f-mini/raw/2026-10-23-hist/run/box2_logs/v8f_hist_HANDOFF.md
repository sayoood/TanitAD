# PREREG_V8F_HIST handoff (written 2026-10-23 ~06:00) -- NOT ARMED: the box-2 run awaits the PI's go

* PREREG: `Project Steering/PREREG_V8F_HIST.md` (+ Amendments 1 and 2). Instruments landed at 6b860b95 (agent/arch-inf-20260803).
* Arm H = M2b + `--hist 10 --window-start 3` (B's 306,566 windows / 19,164 steps / held ages). B = banked M2b, F 0.1207, 2F 0.2415.
* ARM IT (only after the PI's go, quoted verbatim in the report):
  1. `git -C /c/Users/Admin/tanitad-push show <tip>:"<PK>/code/box2_ops/chain_v8f_hist.sh" > /c/Users/Admin/box2_ship/chain_v8f_hist.sh`; cmp it to the tip.
  2. Wrapper `box2_ship/run_chain_v8f_hist.cmd` (LF, same form as run_chain_v8f_grid.cmd), writing `logs/chain_v8f_hist.out`.
     The default predecessor (`logs/chain_v8f_grid.out`) already ends in ZZV8FMINI-DONE.
  3. `--dry-run` first: every train line must carry `--hist 10 --window-start 3` and nothing else beyond B's argv.
  4. No BOX2_BUSY.txt may exist; launch via WMI (Win32_ProcessStartup ShowWindow 0, CreateFlags 0x08000200; `cmd /c <wrapper> < NUL`).
  5. Waker: a copy of wake_v8fgrid.sh pointed at chain_v8f_hist.out and logs/v8f_hist/hist_read_s0.json (check the reader's --out name in the chain).
* ESTIMATED ~4.5-5 h per seed; the chain stops at seed 0 unless the committed read says CONTINUE.
* On DONE: report the verdict as committed (four families, T1-class tier, paired episode-cluster bootstrap); bank RESULT_V8F_HIST.md + raw/2026-10-23-hist/run/ (template: grid_a2/mk_hold19.py); register H-V8F-HIST-1; copy ckpts to D:/Projects/TanitAD-artifacts/v8f_hist/.

## CLOSED 2026-10-25
* Armed after the PI's go (2026-10-23, "Go. Please remember that our goal is to accumulate achievements and not fails ..."); ran 07:04-11:10 at tip 6b860b95, STOP at seed 0, FAIL-NO-MECHANISM as committed (`hist_read_s0.json`).
* Banked as `RESULT_V8F_HIST.md` + `raw/2026-10-23-hist/run/`; the checkpoint went to the artifacts drive, now mounted as E: (`E:/Projects/TanitAD-artifacts/v8f_hist/M2b_s0_h10/`), not D:.
