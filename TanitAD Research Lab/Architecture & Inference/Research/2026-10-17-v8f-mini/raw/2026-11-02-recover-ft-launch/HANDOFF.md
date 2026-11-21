# HANDOFF: RECOVER-FT six-run chain on box 2 (launched 2026-11-02 21:42:35, box-2 clock)

**State:** the chain is RUNNING on box 2 (self-contained: it needs no dev-box process). Card `PREREG_RECOVER_FT_FROZEN.md` (sha 57caba3b...) + `AMENDMENT_1.md`. Code = landed tip `ad0bcb72718aa3c9e63ac4f589e65b94e8bf4ace` (box-2 trainer sha256 `1576a578...`, `v8f_recover.py` sha256 `eff087da...`, verified by Get-FileHash).
**Pattern copied:** `C:/Users/Admin/box2_ship/chain_v8f_launchintent.sh` (most recent v8F chain; DAG-OACC r2 lineage): `git archive` export of the trainer's import closure (its EXPORT list + `v8f_recover.py`), tar sha-verified, trainer's own `--selfcheck-only` as the launch gate, pinned argv, launch.ps1 + `%RUNSAFE%` + env.cmd. Differences: it runs ON box 2 as one .cmd chain (this agent ends), failed runs are marked and skipped (no stall), the aside copies only the 2 files the tar changed (`D:\tanitad\repo_aside_20261102\`).

## Paths (box 2) and logs
* Chain: `D:\tanitad\recft\recft_chain.cmd`; log `D:\tanitad\logs\recft_chain.log`. Guard `D:\tanitad\logs\recft_guard.log` (USB HDD >= 4/20 samples >= 61 C or one >= 63 C => `GUARD_STOP.txt`, kills our trainer, chain HALTS), telemetry `D:\tanitad\logs\telemetry_full.csv`.
* Markers + pins + per-run logs: `D:\tanitad\runs\v8f_mini_full\recft\` : `RUN_<tag>.done` | `RUN_<tag>.FAILED`, `gate_token_<tag>.json`, `logs\<tag>.{gate,train}.log`, `CHAIN_RECOVER.done` (chain reached its end) or `CHAIN_RECOVER.STOPPED` (safety stop). tags: `R_0 C0_0 R_1 C0_1 C0rep_0 C0rep_1`.
* Runs: `D:\tanitad\runs\v8f_mini_full\M2b_s{0,1}_recft_{R,C0,C0rep}\` (`ckpt_last.pt summary.json argv.json selfcheck.json train_log.jsonl`); gates in `...\gate_recft\`.
* A refusal / failed gate / failed run writes `RUN_<tag>.FAILED` and the chain goes on. A stopped trainer resumes from `ckpt_last.pt`: relaunch the same chain command (done runs are skipped; a cached gate token is reused only if commit, argv, selfcheck and shipped bytes are identical).

## Timing (MEASURED; the card's "<= 2 % overhead" is wrong for the table build)
**Each R gate and each R training start reads ALL 4,512 TRAIN clips once from the USB HDD (20.9 MB each, ~95 GB at ~37 MB/s): ~42 min per table build** (R_0 gate wall 45 m 20 s). R_0's gate was run before the chain and reused (token bound). Expected: R_0 first training step ~22:27, R_0 done ~23:05, C0_0 ~23:47, R_1 ~01:55, C0_1 ~02:35, C0rep_0 ~03:15, **CHAIN_RECOVER.done ~04:00 on 11-03 (+-20 min)**. The first run's first logged step was therefore NOT visible at hand-off (still building).

## Poll (short ssh; or run `wake_recft.sh` as a background task)
`ssh tanitad-box2 "powershell -NoProfile -Command Get-Content D:\tanitad\logs\recft_chain.log -Tail 6"`, then `... Get-ChildItem D:\tanitad\runs\v8f_mini_full\recft -Name` (markers), then `... Get-Content D:\tanitad\runs\v8f_mini_full\<run>\train_log.jsonl -Tail 1`. Check the first R_0 step line after ~22:30.

## Pull (after CHAIN_RECOVER.done)
`bash E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/pull_recft.sh` -> `E:/Projects/TanitAD-artifacts/v8f-recover/<run>/` (ckpt, summary, argv, selfcheck, logs, SHA256SUMS) + `recft_markers/`; verifies each ckpt sha256 against its marker. Then **remove `C:\Users\Admin\box2_ship\BOX2_BUSY.txt`** (this agent took it; nobody else releases it), and make sure `D:\tanitad\logs\telemetry.stop` was written (the chain does).

## Gate evidence (banked in `evidence/`)
R_0 and C0_0 gates PASS (`gate_token_*.json`: commit, argv sha256, data sha256 `37f4f9ee...`, init sha256). **Real TRAIN-bank deviation table (R_0, bank2 s0, n = 433,301 of 613,132 records):** dv (m/s) q05/q50/q95 = -2.74 / -0.32 / +1.74 (|dv| q95 3.12); dy (m) = -1.05 / +0.017 / +1.50 (|dy| q95 1.81); dpsi (rad) = -0.175 / -0.0043 / +0.134 (|dpsi| q95 0.266); share of logged v in [3, 8] m/s = 0.302 (in-band dv median -0.48). K-B3 window-list sha `e144adcd...` equal to DAG-OACC's. recover_probe pass, gt vs rejoin target max abs err 7.3e-6, rc 0.0.

## NEXT AGENT (T1-kin read, dev box; card sections 6-7)
1. Controls first (any failure REFUSES the read): KP1 (summary `train_sha256` / `train_window_list_sha256` = DAG-OACC's; `RUN_*.done` already checked), KR1 (R `recover_n/eligible` within [0.25, 0.35], C0/C0rep exactly 0), KR2 (R ckpt != C0 ckpt; argv audit C0 = R minus the lever), KR3 (`info.recover.table` in `gate_selfcheck.json`), KA1 (bit-identity rerun of one rollout).
2. T1-kin rollouts of R_s, C0_s, C0rep_s (s = 0, 1) with the existing tool, logged frames; then frame-freeze: `dagland/framefreeze/` tool with `--scene hold_now` for FZ-R_s, FZ-C0_s, FZ-Y_s (DAG-OACC_s), FZ-HLVL_s (`--histswap cv_logged`), controls KFZ1-4.
3. Estimator: paired episode-cluster bootstrap B = 2,000 seed 0, 141 episodes / 2,399 windows, 19 Read-A rows, four families (STRATEGIC reported PAUSED). Report PRIMARY (M1 R - C0 separated-better at both seeds and mean share closed >= 0.5), guards (ADE6 R - C0 <= +0.0604 m; no separated-worse non-crawl row), FLOOR (|R - C0| > |C0rep - C0| on M1 at each seed), Read A of R vs P10, and the frame-freeze label (ROBUST / PARTIAL / ARTEFACT). Name which variance question each CI answers (episodes / training run / inference).
4. Register the table-build summary (KR3), the ~42 min build cost, and the gate tokens with the arm; Rule Zero: every verdict carries its committed next arm (card section 8).
