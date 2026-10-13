# V-TACkr (L2 training replicate, seed 1) descriptive read, 2026-10-13 ~13:05Z. POST-HOC; L2 is scored at I-0.

Thor batch-8 dumps (md5 above), eval139, paired episode-cluster bootstrap, sampler seeds 0 / 1. The L2 effect at
training seed 1 is V-TACkr − V-R8r; at seed 0 it is V-TACk − V-R8 (`raw/vtack/`).

* **Replicates at both training seeds (separated on both sampler seeds, same sign), all WORSE:** `speed_mae_0_2s`
  (+0.031 / +0.027 at seed 1; +0.030 / +0.042 at seed 0) and the constraint MAEs lat / lon / spd (+0.029 / +0.059 /
  +0.070 at seed 1; +0.058 / +0.061 / +0.092 at seed 0).
* **Route (turn direction, heading-15, turn cross): no gain at either seed** (seed 1 all ns; seed 0 worse).
* **Does NOT replicate:** `off_drivable` (seed 0 +0.03 worse, seed 1 ns); tactical accuracy (`lat3_f1_v9` seed 1
  +0.090 better, seed 0 ns); map IoUs (mixed signs).
* ⚠ **The premise reading flips with the training seed.** `gs_trunk_proj_tac_v6` readings at steps 500 / 2000 /
  2500: V-TACkr −0.0001 / −0.0021 / **+0.0386** → last-50 % median **0.0183 (≥ 0.01, MET)**; V-TACk −0.0023 / −0.0022 /
  +0.0107 → **0.0042 (< 0.01, UNMET)**. B-PREM names V-TACk (seed 0), so as registered L2 reads PREMISE-UNMET. But
  the median is over **two** readings per arm, and the two seeds straddle the threshold. The premise instrument is
  too coarse to carry the L2 verdict by itself. This is reported, not used to re-score anything.
* `p_lat` argmax: V-TACkr 992 / 32 / 88 over classes 0 / 6 / 7; V-R8r 995 / 32 / 85; **V0r 1,061 / 1 / 50** (nearly
  but not fully degenerate; V0 at seed 0 is 1,112 / 0 / 0).
