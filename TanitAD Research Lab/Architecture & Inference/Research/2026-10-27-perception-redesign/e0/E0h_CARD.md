# E0h card -- map-quality dose-response (REFCV9_DESIGN §4.1; sets MAP-2's absolute value from the planner)

**Question.** How precise must the planner's map be before MAPPATH's cross-fitted soft cost converts it into fewer sidewalk plans?
**Frozen design (before any run was read).** MAPSEL's banked progress-matched fan (`F:/refcv9_mapsel/trace_s0_mappath.npz`, 4,754 windows) and MAPPATH's code **unchanged** (`refcv9_mapedge/mapsoft_path.py`, its seals checked by its own `main()`).
Only the planner's map is replaced: every GT sidewalk cell flagged (recall 1) + square FALSE stamps (1.0 m) on SEEN non-sidewalk cells inside the MAP-2 corridor (|y| < 3 m) of each band 0-5 / 5-10 / 10-20 / 20-40 / 40-100 m, added until the pooled corridor
precision TP/(TP+FP) of the band equals p (`nearfield_prec.py`'s definition, so p IS the MAP-2 quantity). p ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}. pdrv = drivable-or-unjudged minus stamps; code = GT with stamps = sidewalk. GT is still the scorer's truth.
**Readout.** S6c sidewalk rate of `MAPSOFT-G-XF` vs BASE (paired episode-cluster bootstrap, B 2000, seed 0, over EPISODES: *another draw of episodes only*; the stamp placement is a separate draw, not resampled; one training seed, one inference seed; T1 open loop, not driving performance).
`p_half` = smallest p with rate ≤ BASE/2 (interpolated); `p_MAP1` = same for the MAP-1 bar 0.098.
**Controls (a failure voids the readout).** K-P1: the p = 1 run (no stamps = the GT map) recovers ≥ 80 % of the banked GT-T1 gain (BASE 0.1792 → GT-T1 0.0219). K-ALL: a constant map (psw = pdrv = 255) reads the BASE rate exactly (4 dp). K-CAL: realized per-band precision within 0.02 of p.
Unit tests (`test_e0h.py`, `test_e0h_read.py`, 7 PASS): p = 1 plane is exactly the GT sidewalk plane; the all-flag map is constant; on synthetic GT with a known 4,800-cell sidewalk the stamp loop hits FP = 4,800 / 1,200 (p = 0.5 / 0.8) literally; no false flag on a GT sidewalk cell or outside corridor × band; `p_half` interpolation = 0.8 on a literal curve.
**Descriptive, not a bar:** stamp-size sensitivity at p = 0.7 (0.5 m and 2.0 m stamps).
**Disclosed limits.** (1) Random false stamps are a MODEL of own-map false alarms, not refcv8's actual ones (those sit at x 0-4 m on drivable cells, 63 % of the 0-5 m flags). (2) Recall is 1 everywhere, so this isolates precision; refcv8's recall is 0.34-0.62. (3) 68 of 4,754 windows have no GT: the synthetic map has no sidewalk there. (4) The soft-cost fit includes the banked human-likeness constraint (H ≤ H0), so a perfect map may sit above GT-T1's hard filter; that is why K-P1 is a share, not an equality.

**Result (MEASURED, ours; read 2026-11-03).** Raw: `raw/E0H_RESULT.json` (sha256 `474e8caadc65628cc76af3271c9242b528a948c41b4e1eaf2c682475f72c8460`) and `raw/E0H_RESULT.md` (sha256 `50459272085be57cb78a29f62e4525c1173e978e20d8ca906aa79e5f6dc6b8de`), written by `e0h_read.py` from the 19 run JSONs (copied from `F:/r9e0/e0h/read4/`; runs in `F:/r9e0/e0h/runs`). `VOID: false`.
**Tier: T1 self-action open loop on logged eval139 frames (not driving performance); S6c sidewalk rate, n = 3,198 windows over 136 episodes in every paired cell; ONE training seed, ONE inference seed.** Estimator: paired episode-cluster bootstrap, B 2000, seed 0, over EPISODES, which answers "would another draw of episodes say this?" and nothing else. ⚠️ **The false-flag PLACEMENT draw is a separate source of variance that NO interval below contains.** It is read directly instead: p 0.97 / 0.98 / 0.99 each have THREE placement seeds (0 = the default seed `20261102` per `SYNTH_RECORD.json`, then 1 and 2); every other p has one.

**Controls (a failure would void the readout; all PASS).** K-P1: the p = 1 run reads 0.0313; BASE 0.1792 to the banked GT-T1 0.0219 is the gain, and the p = 1 run recovers 0.940 of it (bar >= 0.80). K-ALL: the constant map reads 0.1792 = BASE exactly, paired delta +0.0000 [+0.0000, +0.0000]. K-CAL: realized corridor precision within 0.0002 of p in every band of every dose run (bar 0.02).

BASE (native plan) S6c sidewalk rate = **0.1792**; MAP-1 bar <= 0.098; halving = BASE / 2 = 0.0896. Rate = `MAPSOFT-G-XF` sidewalk rate; delta = paired difference vs BASE.

| run | p | placement seed | stamp m | sidewalk rate | delta vs BASE [95 % CI] | <= 0.098 | <= BASE/2 |
|---|---|---|---|---|---|---|---|
| constant map (K-ALL) | - | - | - | 0.1792 | +0.0000 [+0.0000, +0.0000] | no | no |
| p0.50 | 0.50 | 0 | 1.0 | 0.1614 | -0.0178 [-0.0277, -0.0096] | no | no |
| p0.60 | 0.60 | 0 | 1.0 | 0.1588 | -0.0203 [-0.0299, -0.0119] | no | no |
| p0.70 | 0.70 | 0 | 1.0 | 0.1517 | -0.0275 [-0.0393, -0.0171] | no | no |
| p0.70_stamp0.5 | 0.70 | 0 | 0.5 | 0.1604 | -0.0188 [-0.0281, -0.0106] | no | no |
| p0.70_stamp2.0 | 0.70 | 0 | 2.0 | 0.1404 | -0.0388 [-0.0515, -0.0270] | no | no |
| p0.80 | 0.80 | 0 | 1.0 | 0.1473 | -0.0319 [-0.0438, -0.0213] | no | no |
| p0.90 | 0.90 | 0 | 1.0 | 0.1332 | -0.0460 [-0.0604, -0.0334] | no | no |
| p0.95 | 0.95 | 0 | 1.0 | 0.1270 | -0.0522 [-0.0667, -0.0394] | no | no |
| p0.97 | 0.97 | 0 | 1.0 | 0.0976 | -0.0816 [-0.1017, -0.0631] | yes | no |
| p0.97_seed1 | 0.97 | 1 | 1.0 | 0.1151 | -0.0641 [-0.0792, -0.0494] | no | no |
| p0.97_seed2 | 0.97 | 2 | 1.0 | 0.1116 | -0.0675 [-0.0837, -0.0529] | no | no |
| p0.98 | 0.98 | 0 | 1.0 | 0.1085 | -0.0707 [-0.0876, -0.0542] | no | no |
| p0.98_seed1 | 0.98 | 1 | 1.0 | 0.1110 | -0.0682 [-0.0846, -0.0523] | no | no |
| p0.98_seed2 | 0.98 | 2 | 1.0 | 0.1138 | -0.0654 [-0.0821, -0.0496] | no | no |
| p0.99 | 0.99 | 0 | 1.0 | 0.0732 | -0.1060 [-0.1309, -0.0831] | yes | yes |
| p0.99_seed1 | 0.99 | 1 | 1.0 | 0.0722 | -0.1069 [-0.1318, -0.0844] | yes | yes |
| p0.99_seed2 | 0.99 | 2 | 1.0 | 0.0929 | -0.0863 [-0.1071, -0.0664] | yes | no |
| GT map, no stamps (K-P1) | 1.00 | - | - | 0.0313 | -0.1479 [-0.1793, -0.1178] | yes | yes |

**Per-p placement-seed read (seeds 0 / 1 / 2) at the two thresholds:**

| p | rate, seed 0 / 1 / 2 | seeds <= 0.098 (MAP-1) | seeds <= BASE/2 | within-p spread (max - min) |
|---|---|---|---|---|
| 0.97 | 0.0976 / 0.1151 / 0.1116 | **1 of 3** | 0 of 3 | 0.0175 |
| 0.98 | 0.1085 / 0.1110 / 0.1138 | **0 of 3** | 0 of 3 | 0.0053 |
| 0.99 | 0.0732 / 0.0722 / 0.0929 | **3 of 3** | 2 of 3 | 0.0206 |
| 0.50 / 0.60 / 0.70 / 0.80 / 0.90 / 0.95 (one seed each) | 0.1614 / 0.1588 / 0.1517 / 0.1473 / 0.1332 / 0.1270 | 0 of 1 each | 0 of 1 each | not measured |

**Reading: MAP-1 (sidewalk <= 0.098) is met at p 0.99 on all 3 placement seeds and at 0.97 / 0.98 on 1 of 3 / 0 of 3 ⇒ the planner needs map precision ≈ 0.99** (at recall 1, random 1.0 m false stamps, MAPPATH's cross-fitted soft cost, this planner). The design's provisional MAP-2 value 0.85 was not run; the grid points either side of it, p 0.8 and p 0.9, read 0.1473 and 0.1332, both above 0.098 and above BASE/2. Every p <= 0.95 run is separated from BASE (paired CI excludes 0), and the single-seed points fall monotonically with p (0.1614 at p 0.5 to 0.1270 at p 0.95), but none gets near the bar; the last step is the one that reaches it. At p 0.99 the three placement seeds all sit below all three p 0.98 seeds (0.0929 max vs 0.1085 min; a rank statement on n = 3, not a test), while the seed-0 p 0.97 run (0.0976) sits below all three p 0.98 runs: the ORDER of single runs between p 0.95 and 0.98 is not resolved by this design (within-p spread up to 0.0175).

**NOT quotable: the reader's `p_half` (0.9858, bracket 0.98 - 0.99) and `p_MAP1` (0.9697, bracket 0.95 - 0.97) in `E0H_RESULT.json`.** `e0h_read.py` sorts every 1.0 m run as its own (p, rate) point and interpolates at the first point at or below the level, so with three seeds per p it crosses at the BEST seed of the upper p: `p_MAP1` rests on the single p 0.97 seed-0 run (0.0976; seeds 1 and 2 read 0.1151 / 0.1116, above 0.098), and `p_half` runs from the WORST p 0.98 seed (0.1138) to the BEST p 0.99 seed (0.0722) (hand-reproduced: 0.98583 and 0.96970). They interpolate across placement noise (within-p spread 0.0175 / 0.0053 / 0.0206 at p 0.97 / 0.98 / 0.99) and carry no interval. The same reader gave `p_half` 0.9695 (bracket 0.95 - 1.0) on the first 10 runs, before p 0.97 - 0.99 existed (`F:/r9e0/e0h/read/E0H_RESULT.json`), then 0.9854 / 0.9860 / 0.9858 as points and seeds were added (`read2` / `read3` / `read4`): not a stable estimand. **Quote the per-p seed counts above instead.**

**Descriptive, not a bar:** stamp-size sensitivity at p 0.7 (one placement seed each): 0.5 m 0.1604, 1.0 m 0.1517, 2.0 m 0.1404 (range 0.0200); the ≈ 0.99 above is for 1.0 m stamps and the stamp size was not varied at p 0.99.

**NOT established (beyond limits 1-4 above, which stand).** (a) Placement seeds exist at three p only; three seeds are a count, not an interval, and the p 0.5 - 0.95 points carry placement noise of an unknown size (0.005 - 0.021 at p 0.97 - 0.99). (b) The p 0.97 / 0.98 / 0.99 points and placement seeds 1 and 2 are NOT in the frozen design (its grid stops at 0.95; the tip's `run_e0h.sh` lists 10 runs): they were added after the first read, so the extension is post hoc; no control or bar was changed. (c) One training seed and one inference seed; the planner samples, so the inference-seed floor of this rig is not read here (`H-ESTIM-SEED-1`). (d) The fan is the whole 4,754-window eval139 grid (= E0i's DEV 2,397 + HOLD 2,357 windows), so the HOLD half is inside this read; it sets a design value and scores no bar. (e) The paired n (3,198 windows / 136 episodes) is the S6c readout's against the fan's 4,754 windows; the selection rule is not stated in this card or in the reader (UNVERIFIED here). (f) ≈ 0.99 is a precision demand at recall 1 with random stamps, not a calibrated target for the in-model map head: refcv8's recall is 0.34 - 0.62 and its corridor precision is the INHERITED reference in `REFCV9_DESIGN.md` §5.2 (MAP-2: "0.251; 0.733 / 0.716 / 0.744"). The decision to re-set MAP-2's 5-20 m value is the Master Mind's at sealing.
