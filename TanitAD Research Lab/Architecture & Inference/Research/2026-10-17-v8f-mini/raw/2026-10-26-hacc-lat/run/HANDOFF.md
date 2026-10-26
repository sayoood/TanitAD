# HANDOFF: PREREG_V8F_HACC_LAT pipeline (built, tested, LAUNCHED 2026-10-26 01:38 Berlin)

## State
One detached chain is running on the dev-box CPU (no GPU, zero training). Chain: `cmd.exe` pid **35648** -> `bash hlat_chain.sh` pid **67032**.
Log: `C:/Users/Admin/hlat/logs/chain.out` (per-step logs `logs/<tag>.out`). Launcher: `hlat_chain.cmd` (CRLF); re-running it resumes (existing run JSONs, ckpts and the fit JSON are skipped, pins re-asserted).
Markers in `chain.out`: `ZZHLAT-DONE <n>` (n should be 12) or `ZZHLAT-STOP <reason>` (also touches `logs/STOP`; the other lane stops at its next step).

## What runs (from `hlat/frozen/`, a pinned copy of `hlat/code/`; never edit either while it runs)
0 wait ZZT1K-DONE (already there) -> freeze + sha256 pins (`PINS.sha256`, `PINS_H.sha256`; re-asserted before AND after every run: v8f_t1_eval.py, v8f_mini.py, v8f_hlat.py, data/eval modules) -> inputs md5/sha.
1 suite of the frozen copy (test_v8f_t1, hacc_infer, hlat_gate, hlat, mini, oacc; dev run: 152 passed, 92 skipped, 0 failed) -> C-T / C-T2 smoke vs the ORIGINAL tool, s0 and s1, 48 windows.
2 per seed (2 lanes): bake CA1/CB1/CB2 -> **C-B2** (train subset 192) -> **C-A1**, **C-B1** (eval-141 m10, `--no-analyze`, vs `t1k_run/dump_HACC_s*_m10`) -> HACC train tick dump (m10, 141 train clips) -> **C-T2** -> `fit` (A lambda rule + **C-A2** + GT:=P regression; B v*) -> bake A, B -> eval-141: A m10, B m10, A m30, B m30.
3 `hlat_summary.py` -> `RUN_SUMMARY.json` (controls + fit + bakes only).

## Artifacts
- Run output (eval-141, tool format): `run/t1v8f_{A,B}_s{0,1}_m{10,30}.json` + `run/dump_*`; controls' runs `run/*CA1*`, `run/*CB1*`. ckpts `ckpt/hlat{A,B,CA1,CB1,CB2}_s{0,1}.pt` + `ckpt/BAKE_*.json` (sha256 each). Fits `fit/hlat_fit_M2b_s{0,1}.json`, `fit/weights_A_*.json`, `fit/gate_B_*.json`. Train ticks `dumps_train/ticks_HACC_s*_m10_train.npz`. Controls `controls/*.json`.
- Repo-bound files (land from `hlat/code/`, CRLF): `stack/tanitad/models/v8f_mini.py` (gate), `stack/scripts/v8f_t1_eval.py` (`--dump-ticks`, `--windows train`), NEW `stack/scripts/v8f_hlat.py`, NEW `stack/tests/test_v8f_hlat.py`, NEW `stack/tests/test_v8f_hlat_gate.py`. Helpers outside the tree: `ct_smoke.sh`, `ct_compare_ticks.py`, `hlat_chain.sh`, `hlat_summary.py`, `work/edit_*.py` (the edit scripts).
- The arm runs' JSON carry the OUTCOME rows (arm - P10, arm - HACC): not opened by me or by the chain.

## What each control must read
C-T: 25/25 files bit-identical (smoke s0 already did at 24 windows) ; tick-0 plan vs dump `ol` = 0.0 | C-T2: tau, P, v0 diff 0 (bar 1e-6), GT float32 bits equal | C-A1, C-B1: max|d| <= 1e-9 m on cl/ol/fl (expected ~0) | C-B2: cross max|plan-P| = 0 (bar 1e-6), along > 1e-3, all fired | C-A2: lambda=0 vs clipped closed form <= 1e-12 (dry run 5.6e-15 KKT) | regression GT:=P: w <= 1e-12 | reachability: gate battery + 9 mutants RED (in-process, no on-disk mutation).

## Deviations and declared interpretations (v8f_hlat.py docstring J1-J6)
- J1 lambda0 = S1/S2 at HBLEND's banked w, EVALUATED ON THE TICK SAMPLES. J3 tick samples whose clip ends inside the 6 s plan (gt_ok False) are dropped from fits.
- J4 arm B "within F/2": read literally two-sided, |ADE6(HACC-V) - ADE6(HACC)| <= 0.0604; the one-sided reading and exact F/2 = 0.06037 are written to `fit/*.json: arm_B.reading_sensitivity`. CHECK it: if `same_as_used` is false the clause decides v*.
- Tick GT = the LOGGED future in the EXECUTED frame, time-indexed (as written). Measured on 1,200 train windows: median ADE2 of the HACC plan vs that GT is 0.16 m at tick 0, 1.3 m at tick 1, 5.0 m at tick 2 (the executed car lags/stalls while the human drives on). The pre-registered lambda rule then fell to a fallback branch in the dry run (m*=0, v*=8 fallback). Not changed; train-side only, no eval number involved.
- `--no-analyze` for C-A1/C-B1 (dumps are what is compared). Pre-existing failures in other test files (tac queue x4, paired_families x1) are identical on the original export and are not in the chain's suite.
