# PREREG_V8F_VOE1B: violation of expectation with a calibrated untrained-network reference (Master Mind, 2026-11-07, written after VOE1's INVALID control read and before any VOE1B code or statistic; no M2b manipulation row of VOE1 has been read by anyone)

## Why
VOE1 (`PREREG_V8F_VOE1.md` `555c5a38`, IMPL_CHOICES `848a2c8c`) is **INVALID** by its own controls 1 and 2, as committed. The measured causes (`theory/voe1/DIAG_CONTROLS_VOE1.md`, control rows only):
* **Both controls were family-wise without calibration.** Each required 0 of 12 cells separated. In the run's own 20 permutation draws, which contain no signal by construction, at least one cell separated in 7 of 20 draws (control 1) and 8 of 20 (control 2). A clean pipeline fails such a control about a third of the time.
* **One untrained network per seed is not a null distribution.** Untrained world models give pairwise accuracies far from 0.5 in BOTH directions: 0.355 to 0.511 on FREEZE and 0.380 to 0.550 on ACTION-SWAP, with the two seeds disagreeing (RI_s0 ACTION-SWAP 0.38 to 0.39, RI_s1 0.51 to 0.55). An untrained architecture alone can therefore look "surprised" or "anti-surprised". A trained model must be read against the spread of many untrained ones, not against one.
* The ACTION-SWAP donor was chosen by a large difference in mean action, so the donor's actions also differ in magnitude. Surprise can then follow action magnitude rather than action use. VOE1B adds a magnitude-matched swap as the deciding row.

⛔ `theory/voe1/RESULT_VOE1_INVALID_READ.md` and the M2b manipulation rows of `RESULT_VOE1.json` stay unread. VOE1B recomputes everything; nothing from VOE1's M2b rows enters this card or its implementation.

## Data, models, manipulations (VOE1's, unchanged unless stated)
* eval-141, 1,976 windows, 141 clips. M2b s0 / s1, zero training. t0, the level step map, the score (max over the 1.6 s after t0), the pairwise accuracy, the episode-cluster bootstrap B 2,000 and the two-part rule are VOE1's IMPL_CHOICES (`848a2c8c`), reused verbatim.
* Manipulations: TELEPORT, FREEZE, REVERSE, SPLICE and ACTION-SWAP as in VOE1, plus:
* **ACTION-SWAP-M (deciding for the action question):** the donor comes from another episode. Its mean |longitudinal action| and its mean |yaw rate| over the swapped transitions are each within ±10 % of the window's own. AND the sign of its mean longitudinal action or of its mean yaw rate is opposite to the window's, where that mean is at least 0.5 m/s² or 0.05 rad/s in size. Windows with no such donor are dropped from this row, with n reported. Fewer than 300 qualifying windows makes the row **UNDERPOWERED**.
* **Untrained reference:** 16 untrained world models (`torch.manual_seed` 100 to 115). Each keeps the trained M2b chassis and re-initialises every `wm.*` parameter, as VOE1's RI did. For every manipulation and level, this gives a reference set of 16 accuracies.

## Committed labels, per manipulation, level and model seed
* **NOTICES:** all three hold at both model seeds:
  * (a) accuracy two-part separated above 0.5;
  * (b) for TELEPORT, REVERSE and SPLICE only, separated above the Copy-Last null (paired, as VOE1);
  * (c) the accuracy is above the **maximum** of the 16 untrained accuracies for that manipulation and level.
* **INIT-LEVEL:** (a) and (b) hold, (c) fails. An untrained network of the same architecture does this too.
* **NULL-ONLY:** (a) holds, (b) fails.
* **BLIND:** (a) fails.

## Controls that must read known values (the run is INVALID if any fails)
1. **Real-vs-real, calibrated:** count the separated cells among the 6 M2b real-vs-real cells (2 seeds × 3 levels; the later copy 0.2 s later, as VOE1). The count must not exceed the 95th percentile of the same count over the R = 199 reference draws (each draw scored as if real). The untrained real-vs-real cells are reported beside.
2. **Untrained leave-one-out:** treat each untrained seed in turn as "the model" and the other 15 as its reference. Apply rule (a) at that one seed plus rule (c) against the 15. Over all (seed, manipulation, level) cells of FREEZE, ACTION-SWAP and ACTION-SWAP-M, at most 10 % may come out NOTICES. Under exchangeability the expected share is at most 1/16.
3. **Calibration** (as VOE1): 20 draws (seeds 1001 to 1020), at most 5 % of rows separated per family.
4. **Synthetic** (VOE1's test, kept): the planted toy gives NOTICES on FREEZE and TELEPORT, and the action-blind toy gives BLIND on ACTION-SWAP. **Added:** a toy family whose untrained members are biased in random directions must give INIT-LEVEL, not NOTICES, for a "trained" member drawn from the same family.

## What each outcome triggers (committed now)
* ACTION-SWAP-M **BLIND** or **INIT-LEVEL** at every level → the error does not use the action beyond what an untrained network does. Together with IMAG1's IA-3, the lever is **E7 DIVRS** plus a counterfactual-target term.
* FREEZE **BLIND** or **INIT-LEVEL** at every level → the predictor does not carry motion beyond its architecture: **MRATE** (carded).
* TELEPORT / SPLICE **NULL-ONLY** at every level → surprise is a frame-change detector; IMAG1's IA-2 decides before any arm.
* **NOTICES** on FREEZE or ACTION-SWAP-M at any level → the first positive "imagination is used" evidence on video. The per-level table and the surprise traces go into the paper.
* Event alignment (IA-6d) is VOE1's, unchanged, and reported beside with no bar.

## Implementation choices
The implementer writes `theory/voe1b/IMPL_CHOICES_VOE1B.md` (only the deltas from `848a2c8c`: the ACTION-SWAP-M donor rule and its n, the 16 seeds, the reference-count procedure for control 1, the leave-one-out procedure for control 2, the added synthetic test) and records its sha256 **before any statistic is computed**.

## Log
* 2026-11-07 15:30:33 frozen: prefix 5785 bytes, sha256 aa317fe3f046e9d0522701d4b35c9d029b1fd55a4a4590823c02ff60449493c3 (everything above this line). No VOE1B code or statistic existed at this time.
* 2026-11-07 15:54:27 MM readings, written before any VOE1B result was read (the run was still going): (1) ACTION-SWAP-M's sign rule is read strictly: both the window's and the donor's signed mean must reach 0.5 m/s^2 or 0.05 rad/s (530 windows, 104 episodes; the looser reading would give 545). (2) '16 untrained models' means 16 per trained chassis (32 in all), and each M2b seed is compared with the 16 on its own chassis. Both are as fixed in IMPL_CHOICES_VOE1B (f7c2d624, frozen 15:40:13). RATIFIED.
