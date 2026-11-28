# NAVTEST feature dump for L2 SAFE-HEAD — HANDOFF (2026-11-27 ~11:40Z, data agent)

State: chain LAUNCHED detached (WMI, 0x08000200). cmd PID 74664 -> python launcher 69544 -> nt_chain.py PID 36724. Give-up 36 h.
Log: F:/safehead_cache/navtest/NAVTEST_CHAIN.log (+ dump.log, ctrl.log, chain_stdout.log). Markers: F:/safehead_cache/markers/navtest.DONE | navtest.FAILED.

Stages (resumable: re-run `cmd.exe /c F:\safehead_cache\code\run_navtest_chain.cmd` via WMI; finished log files are skipped):
1. wait for markers/chain.DONE (navtrain chain) or a chain.FAILED newer than start, or 30 min with no sh_run.py process;
2. nt_prep.py -> navtest/plan/{job_navtest.json,commands_navtest.json,anchor_pick.json} (136 logs, 12,146 tokens, built from the anchor table);
3. SMOKE = first 2 logs through the same tool, then `nt_ctrl.py smoke` -> navtest/ctrl_smoke.json (FAIL stops the chain; TRK is still relaunched);
4. FULL dump: sh_dump_nt.py (= sh_dump.py + memgate before EACH log [commit >= 8 GB, RAM >= SH_NEED_MB 2000] + `log`/`split` scalars + file name `<log>.npz`), GPU lock per log;
5. `nt_ctrl.py full` -> navtest/ctrl_full.json + navtest/index.csv (token,log,split=navtest,file,sha256, ready for `safehead_data.py seal`);
6. navtest.DONE/FAILED written, THEN TRK-SCORE relaunched (`cmd.exe /c E:\Projects\TanitAD-artifacts\trkscore\run_trk_chain.cmd`, ReturnValue + PID in NAVTEST_CHAIN.log) — also on any failure.

Controls (each can FAIL; results in ctrl_*.json): CTRL-PICK-FULL (0 mismatches vs anchor `pick`, 0 missing, all 12,146 in full mode), CTRL-LOGIT (<= 1e-5), CTRL-SELECT (agg32 argmax of logits == pick), CTRL-SHAPE, COVERAGE.
Pre-chain check (dry): nt_ctrl on the 10 existing ctrl-dump tokens of log 1: 0 pick mismatches vs anchor, CTRL-LOGIT 5.7e-6 (the 69 other tokens read as missing, as designed). Test files deleted (navtest/<log>.npz, ctrl_smoke.json).
No PDMS, no aggregate computed.

ETA (ESTIMATED): navtrain chain ~1.5-2 h from 11:30Z; then 12,146 x ~1.28 s = ~4.3 h + per-log model moves + gates => navtest.DONE ~ 18:00-19:00Z.
Output ~1.1 GB (about 89 KB/token) in F:/safehead_cache/navtest/; F: had 15 GB free at start. Extra arrays kept: props8, ego, cmd, meta.

If FAILED: read NAVTEST_CHAIN.log / dump.log tail; a dump FAIL names the log. Fix and relaunch (skips finished logs). Do not delete navtest/*.npz of finished logs.
Code: run copy F:/safehead_cache/code/{nt_lib,nt_prep,nt_ctrl,nt_chain,sh_dump_nt}.py + run_navtest_chain.cmd; staged copy E:/Projects/TanitAD-artifacts/safehead/navtest_stage/ (MM lands them; nothing git-added).
Not touched: navtrain chain files/processes, trkscore/ensscore/scorerft_read dirs (except the one TRK relaunch at the end). Deletions by me: 2 test files above (~0.9 MB).
