# HANDOFF_LFREAD: the T1-kin read package for arms LF050 and LF075 is staged and tested; nothing is launched, committed or landed. The tool is not built yet (`motion_c/lf_build/PINS.txt` is absent), so the chain refuses at gate 0.

Package `E:/Projects/TanitAD-artifacts/dagland/motion_c/lf_read/` (staged, not in git). Card `a3_prereg/PREREG_LEVELFLOOR_A3.md` sha 977a785e. Scratch `F:/TanitAD-scratch/motion_c/lf_read/` (`tree/` is the tool agent's; `out/`, `smoke/`, `work/`, `pkgtest/` are this package's).

## MM launch sequence (git bash), after `lf_build/PINS.txt` (key `patched_tool`) exists
```
cd /e/Projects/TanitAD-artifacts/dagland/motion_c/lf_read
python -I code/make_pins.py                       # only if code or tests changed
bash lf_read_chain.sh --dry-run                   # 16 rows, C0 + LLR checkpoints, base arrays, card, patched_tool pin PRESENT
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\lf_read\lf_read_chain.cmd'}"
ls raw; tail -n 5 logs/chain.out                  # ONE command; RESUME = launch the .cmd again
```
ETA ESTIMATED 95-130 min (banked rollouts 235-270 s each; the floor adds a second head forward; 16 rollouts, then ~1 min reader). Markers `raw/LFRO.done|fail`, `raw/LFREAD.done|fail`; result `raw/LF_READ.json`, `RESULTS.md` (headline names any dose with D at both seeds as COUNTED DOMINANT plus the two-dose disclosure).

## Built
`code/lf_read.py` loads `llr_read.py` by sha-pinned path and reuses its P, G, F, D, LEAK rules; adds per-dose roles, instruments (bind overall / inside M1 ticks, cap, lift, v_f vs logged), reach control (F-UNREACHED reads no rule), the card's section 5 labels, tick-vs-card consistency controls. `code/lf_precheck.py`: gate 1 (passes on the real C0 and LLR checkpoints, `raw/LF_PRECHECK.json`) and a per-rollout floor-key check. Tests 21 green; mutations 25 of 25 RED, clean controls GREEN first (`logs/mutations_lf.log`); known answers ka-null 29/29 (share 0.0, rows exactly 0.0) and ka-llr 45/45 (LLR_READ.json rows reproduced).

## Readings for the MM to overrule
1. Bare arm flags: the card says `--smooth off`, but the banked RIC0 / C0 done markers carry `--smooth sg72`; I copied the banked flags (`BASE_SMOOTH`) so LF - RIC0 isolates F. F must sit after S in the tool.
2. M1 tick = logged route speed >= 5.2 and own speed < 5.2 (M1's pair thresholds); the card's crawl band is reported beside. Logged speed = first-second path length of the tick's `gt`.
3. "M1 moved" = RILF - RIC0 on M1 separated-better at >= 1 seed.
4. Brief said llr_read's `.cmd` is CRLF; it is LF. Mine is CRLF.

## Unverified
The real tool: the floor-key JSON layout (read tolerantly), `lf_*` dtypes and the no-floor encoding, and the pre-flights. The reader's production path ran only on synthetic out roots.
