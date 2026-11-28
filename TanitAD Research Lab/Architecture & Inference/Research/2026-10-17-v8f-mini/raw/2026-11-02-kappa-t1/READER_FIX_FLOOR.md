# READER_FIX_FLOOR: the extractor's floor guard, applied against the fresh baseline (H-KAPPA-T1-1)

Written 2026-11-02 at 16:19 (local, +0100; file mtime 16:19:10, before the patched reader's first run at 16:19:42), **before any X number of the 8 rollouts was read**. Card `PREREG_FROZEN.md`, sha256 `9564d568...046f72c` (not edited). Reader before this fix: `code/kappa_t1_read.py` sha256 `60b2c02d...` (16 hex shown). Tier T1-kin. Estimator of every interval in the read: paired episode-cluster bootstrap, B = 2,000, seed 0 (answers *another draw of episodes* only). Evidence class MEASURED (ours).

## 1. State of knowledge at the time of writing

* All 8 rollouts finished (`logs/chain.out`, `ZZKAP-ROLLOUTS-DONE` 16:16:06). The chain's read refused at the first extraction: `extract FEAS_DAGOACC_s0: windows / floor differ from CTRL's (same_windows=True, same_floor=False)` (`logs/read.out`).
* The reader computes the two known answers KA1 and KA2 for all four arm-seeds **before** it extracts, and returns code 3 if one fails; it reached the extractor, so by control flow both passed. No control magnitude, no row and no outcome word was printed or written (`raw/` holds only `GATE_KAPPA.json`; `raw/arrays/` is empty). That inference is re-read from `raw/KAPPA_T1.json` in this run, not taken from here.
* Known from the brief that launched this fix, not measured by me: a fresh 16-thread run's P10 floor `fl` differs from the banked CTRL floor by up to 1.71e-02 m, while two fresh runs are bit-identical (SMOOTH-T1 `READER_FIXES.md` fix 7 and `raw_a1/NONDET_BOUND.json`).

## 2. The defect

`code/extract_smooth.py` (SMOOTH-T1, read-only import; the landed `extract_arrays.py` plus path arguments) refuses unless the run's floor is within `--floor-guard-m` of the **banked** CTRL dump's floor; the default is the card-era 1e-2 m. The banked rollouts ran in an earlier session; a fresh 16-thread session differs from them by thread-order float noise larger than 1e-2 m on the floor. The guard was written to catch a wrong grid or a wrong floor (metres), not float noise, and it compares the wrong pair for a card whose Y is a fresh same-session rerun.

## 3. The fix (reader and its call of the extractor; no operator, row, bar, arm or rule changes)

1. `extract()` in `code/kappa_t1_read.py` passes `--floor-guard-m 1.0` to the unchanged extractor: the gross-mismatch guard SMOOTH-T1 AMENDMENT_1 used (`a1_nondet_bound.py`, `GROSS_FLOOR_GUARD_M`). The extractor's exact window and grid identity check (`clip_index`, `ws`, `eid`, `g`, `v0` against the banked CTRL dump, 2,399 windows / 141 episodes) is unchanged.
2. **The same-session check is made against the fresh Y inside the reader, before any extraction** (new control `KF`): per arm-seed, (a) the windows of X equal the windows of the fresh Y exactly, (b) X's floor `fl` equals the fresh Y's floor to <= 1e-4 m (this is the clause KA2 already carries; KF restates it with the window identity), (c) the fresh Y's windows equal the banked CTRL's exactly, (d) the deviation of the fresh floor from the banked floor is **measured and reported** and must be <= 1.0 m (the gross guard). Any failure of (a)-(c) or (d) REFUSES the read.
3. X - P10 and Y - P10 keep the **banked P10 side**, as the card and SMOOTH-T1 do: the extractor's default `--floor banked` builds P10's per-window values from the banked floor array and the banked CTRL context. Two controls tie that to the banked record, both before any row is computed: `KB1` (the `b` arrays of every extracted X and Y equal the landed `gapdecomp/raw/arrays/{CTRL,SOUP}_s{seed}.npz` `b` arrays exactly, NaN-aware, on the 17 rows that are not S-POS; S-POS's `b` is the CTRV of the run's own ticks and is a within-run contrast) and `K3` (Y banked - P10, computed from those landed arrays, reproduces `SOUP_READ.json` (pinned sha256 `dca10c07...`) on all 76 arm-seed rows to the SMOOTH-T1 tolerance 6e-5). Both are SMOOTH-T1's controls, re-stated here.

## 4. Does it change a control of this card?

**No.** The card defines Y as the fresh S rerun (same tool file, same 16 threads, one run at a time), compares only runs made here or in `out_m16`, and states KA1 (Y equals `out_m16/SMOOTH_*`, <= 1e-4 m), KA2 (plan equals the box of P + SG(plan_raw - P); floor of X equals Y's) and KA3 (pytest). None is touched: KA1 and KA2 run first with the unchanged tolerance 1e-4 and REFUSE on a miss. The guard value of the extractor was not a card parameter. So this is an extractor fix of the same class as SMOOTH-T1 fix 7, **not an amendment**. KF, KB1 and K3 are added identity checks that can only REFUSE; none can change an outcome word.

**What SMOOTH-T1 did, stated exactly.** AMENDMENT_1 relaxed the extractor guard to 1.0 m against the banked CTRL floor, and compared X's floor with the fresh Y' beside (K5'), not as the extractor's own check. This fix does the same relaxation and, in addition, makes the fresh-Y comparison a gating control (KF) rather than a report line.

## 5. Limits, stated before the data

* The rows X - P10 and Y - P10 use the banked P10 side; the fresh floor differs from it by the measured KF(d) deviation (millimetres to centimetres on a few ticks). That is the SMOOTH-T1 reference choice, kept so the two cards share one P10. X - Y does not use P10 at all.
* The planner is deterministic and each arm is one training run per seed: intervals answer *another draw of episodes* only (`H-ESTIM-SEED-1` is not addressed).
* No class, threshold, operator, row or arm changes after data.

## 6. Result of the fix (appended after the read; sections 1 to 5 are as written at 16:19)

* Reader after the fix: `code/kappa_t1_read.py`, sha256 `d7cdcc6bc306e8b6f53b1d97b6814c55188dae395e5acfff29a4c8ac039d67a9`. The pre-fix reader is kept as `code/kappa_t1_read.pre_floorfix.py` (sha256 `60b2c02d...`).
* The patched reader ran at 16:19:43 (`logs/read2.out`, rc 0). All five controls PASS: KA1 and KA2 (every deviation exactly 0.0), KF, K3 (76 / 76 rows), KB1 (8 / 8 runs); see `RESULTS.md` section 1.
* Measured KF(d): the fresh floor differs from the banked CTRL floor by a maximum of 1.711e-02 m (mean 3.416e-05 m), identical in all four arm-seeds; the fresh X floor equals the fresh Y floor exactly. This reproduces the figure in the brief (1.71e-02 m) and is above the extractor's card-era 1e-2 m guard, which is why it refused.
* The window and grid identity (2,399 windows / 141 episodes) held exactly against both the banked CTRL and the fresh Y.
