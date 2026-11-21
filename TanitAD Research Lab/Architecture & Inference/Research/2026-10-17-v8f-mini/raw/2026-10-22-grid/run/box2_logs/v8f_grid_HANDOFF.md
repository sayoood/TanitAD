# PREREG_V8F_GRID handoff (written 2026-10-22 21:56)

* Instruments landed at 4abdf834 (agent/arch-inf-20260803), with Amendment 1 written before any data.
* Chain: /c/Users/Admin/box2_ship/chain_v8f_grid.sh (cmp == tip 4abdf834), wrapper run_chain_v8f_grid.cmd, launched 21:55 via WMI (cmd pid 43240).
* Log: logs/chain_v8f_grid.out. It WAITS for `^ZZV8FMINI-DONE` in logs/chain_v8f_pgt.out (polls 300 s); stops on PGT's ZZV8FMINI-STOP.
* Base B is decided by `v8f_grid_read.py --decide-base` from logs/v8f_pgt/ (M2b-pgt iff PGT's FINAL read = SUCCESS, else M2b).
* Outputs: logs/v8f_grid/{base.json, check_inputs.json, cache_check.json, grid_read_s0.json, grid_read.json}, runs/, eval/; guard log logs/v8fgrid_guard.out.
* Stage 1 = G s0 -> read s0 (committed STOP or CONTINUE); stage 2 = G s1 -> final read. ~7-9 h per seed ESTIMATED; guard cap 10 h, wait 11 h.
* Dry run of every stage's exact line: scratchpad dkland/grid_dryrun.out (10 stages; the only lever is --grid pool8x20).
* NEXT on DONE: report the verdict as committed (four families, tier T0 + T1-kin read, paired episode-cluster bootstrap B=2000);
  bank RESULT_V8F_GRID.md + raw/2026-10-22-grid/run/; register H-V8F-GRID-1; move ckpts to D:/Projects/TanitAD-artifacts/v8f_grid/.

## CLOSED 2026-10-23 03:45

STOP at seed 0, FAIL-NO-MECHANISM as committed; seed 1 not trained. The first read refused on a lever-record schema difference; Amendment 2 (cf5956f2) fixed it before any number was read, and run 2 read at 03:29. Banked with RESULT_V8F_GRID.md. Checkpoint: D:/Projects/TanitAD-artifacts/v8f_grid/M2b_s0_8x20/. Box 2 is idle. Committed next: more history frames (plan 13.6), own PREREG; its box-2 run needs a PI go.
