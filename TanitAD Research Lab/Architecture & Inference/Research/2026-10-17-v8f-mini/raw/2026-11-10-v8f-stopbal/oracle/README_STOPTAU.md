# STOPTAU read: implementation notes and numbers (no verdict; the Master Mind writes it)

Card: `PREREAD_V8F_STOPTAU.md`, sha256 `e940a0a2...6cee`, re-hashed equal before any work; asserted at every run of `stoptau_read.py` and pinned in `test_stoptau.py::test_card_hash_unchanged`. Tier: T1-kin first tick, zero training, inference-time transform of T's own output; never driving performance. CPU, 2 threads per process, `CUDA_VISIBLE_DEVICES=-1`, `tanitad` loaded from `F:/TanitAD-scratch/stopbal_read/tree` (asserted). All numbers MEASURED (artifacts below).

## Files
* `stoptau_read.py`: stages `train`, `selfcheck`, `tau`, `run`, `analyse`. Pure logic (`choose_tau`, `fires_goal`, `fires_lon`, `tau_code`, `check_tau`, `outcome_lever`, `clips_disjoint`) needs no torch. The eval model pass is `stoporc_read.run_arm`, imported and not edited, with its two hooks pointed at this file's transforms; the cells, floors and gain / cost rules are STOPHARD's (`SO.paired`, `SO.is_gain`, `SH.is_cost`).
* `test_stoptau.py`: 24 tests, all pass (MEASURED, CPU): card hash, grid, planted probability vector gives the planted tau (and three mutant tau rules read other values), float32 edge (0.55 fires, one ulp under does not), tau 0.5 / argmax equal STOPHARD's `hard_code` bit for bit with red arms (other tau differs), lon tau is on p(STOP) not argmax, structural check with five red arms, no label in any signature, planted clip overlap flagged, real train-141 vs eval-141 disjoint, window rule equals `eval_grid_windows.json` on all 141 eval clips, outcome table.
* `TAU_STOPTAU.json`: per-arm training false-positive counts and rates at every grid value, the references, tau, fallback flags, true positives on stop-point windows.
* `RESULT_STOPTAU.json`: every cell, control, count, floor, flag and outcome label.

## Training windows (stage `train`)
* 141 WMACT clips (`theory/wmact/train_stream/clips141.json`, clips sha256 `e9e0e2c4...`), each pulled alone by sftp (read only, md5 checked against box 2's manifest), used, deleted. Peak pulled data on disk 20,904,597 B (one clip). Nothing started or written on box 2. The label join is the file the STOPBAL argv names as `--join-train` (`D:\tanitad\runs\v8f_mini_full\join_train.npz`, 34,696,488 B, pulled read only).
* Windows: the eval-141 rule, `t in range(3, n-30)` with `t % 4 == 3` (equal to `eval_grid_windows.json` on all 141 eval clips, checked), a_t = a_s = 0. 2,396 windows; has-and-present 2,382; GT stop-point windows (STOPDEC's `tlstop == STOP_POINT` stratum from the join_train labels) 114; non-stop-point 2,268.
* Batch build: the loop of `stoporc_read.run_arm`, re-written for a train store and join. **Selfcheck** (BAL-NULL_s0, 51 eval windows through this file's batch code): goal probability max abs diff vs the STOPDEC npz 9.7e-8, lon 1.2e-7; pass.
* Control 2: no training clip id is in eval-141 (asserted before any model pass, and in the tests).

## tau per arm (`TAU_STOPTAU.json`; no arm hit the 0.95 fallback)
| arm | tau goal | tau lon | reference (matching NULL, unhardened): goal FP / lon FP of 2,268 |
|---|---|---|---|
| BAL-NULL_s0 / s1 / rep | 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | own: 16, 15, 25 / 24, 35, 42 |
| BAL-G_s0 / s1 | 0.80 / 0.80 | (0.50 / 0.50) | NULL s0: 16 / 24; NULL s1: 15 / 35 |
| BAL-L_s0 / s1 | (0.55 / 0.55) | 0.75 / 0.75 | same |
Unhardened training false positives of the lever arms: BAL-G goal 50 (s0) / 92 (s1); BAL-L lon 84 / 136.

## Eval cells (tactical ADE, knots 11-30, m; tau variant minus OWN; [95% CI]; * separated; floor in m)
| arm (decision, tau) | fires stop-point / other (at tau; at 0.5 or argmax) | S: 117 stop-point windows | A: 2,351 windows |
|---|---|---|---|
| BAL-G_s0 (goal, 0.80) | 11 / 11 ; 26 / 62 | -0.0088 [-0.0205, -0.0014]*, floor 0.0221: not a gain | -0.0005 [-0.0011, -0.0001]*, floor 0.00056: not a cost |
| BAL-G_s1 (goal, 0.80) | 14 / 22 ; 34 / 113 | +0.0040 [-0.0001, +0.0090], not separated | +0.0006 [0.0000, +0.0013]*, floor 0.00056: cost |
| BAL-L_s0 (lon, 0.75) | 8 / 12 ; 34 / 103 | -0.0040 [-0.0098, +0.0008], not separated | -0.0000 [-0.0005, +0.0006], not separated |
| BAL-L_s1 (lon, 0.75) | 18 / 25 ; 46 / 160 | -0.0030 [-0.0075, +0.0010], not separated | +0.0007 [-0.0000, +0.0015], not separated |
Floors (|delta(NULL_rep) - delta(NULL_s0)|, each NULL at its own tau): goal S 0.0221, A 0.00056; lon S 0.0133, A 0.0016. The other families (FDE, speed and acceleration MAE, heading and yaw-rate MAE, both bands) are in the JSON per arm, variant and set. TACTICAL family: n = 0 (T's decision is the transformed input). STRATEGIC: PAUSED (PI item 25).

## Controls (all seven arms pass, `controls_ok` true)
1. Replay: tau 0.5 (goal) and argmax (lon) give code bit-equal to `stophard_read.hard_code` on every arm; the lon replay plans equal STOPHARD's HARD-LON transform plans window by window (max abs 0.0 m, 2,399 windows, run in the same pass); all 24 replay cells per arm (S and A, 12 cells, goal and lon) equal `RESULT_STOPHARD.json` exactly (max abs diff 0.0, 0 mismatches). An unreachable threshold gives code bit-equal OWN and plan diff 0.0 m. Red arm (data-free, in the tests): a replay at another tau reads a different code.
2. Disjoint clips: asserted (0 shared ids of 141).
3. Planted probability vector gives the planted tau (0.60 for reference 5; 0.50, 0.65, 0.80, and the 0.95 fallback flagged); tests include three mutant rules that read other taus.
Also pass: only the named entries change and only where the decision fires (every tau transform, both sets), unchanged-code windows have plan equal OWN (0.0 m), 117 / 2,234 window counts, own probabilities equal STOPDEC's (<= 1e-5), own plan equals the dump (<= 1e-4).

## Outcome labels computed by the code (card's table, rows tried in order)
* BAL-G (goal tau 0.80): S gain at neither seed, A cost at s1: label `TAU-TRADES` (rows whose rule holds: TAU-TRADES, TAU-NULL).
* BAL-L (lon tau 0.75): S gain at neither seed, A cost at neither: label `TAU-NULL`.
The card's table lets two rows hold at once for BAL-G; the code reports the first row in the card's order and lists both. Both rows carry the same next step in the card.

## Temporary data (rule of 2026-11-07)
Scratch `F:/TanitAD-scratch/stoptau_build/` (113,750,567 B: pulled `join_train.npz` 34,696,488 B, `train_windows.npz` 6,110,786 B sha256 `a6945d06...` (also in `TAU_STOPTAU.json`), seven per-arm eval npz 10,416,045 B each with sha256 in `RESULT_STOPTAU.json` `banked`, logs, empty clip cache). All re-creatable (`stoptau_read.py` stages; join from box 2); deleted at the end of the job. The 141 pulled clips were deleted one by one; the cache folder is empty.
