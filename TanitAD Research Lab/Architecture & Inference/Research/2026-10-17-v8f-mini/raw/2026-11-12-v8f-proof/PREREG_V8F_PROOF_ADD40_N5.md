# PREREG_V8F_PROOF_ADD40_N5 : tool checks on the anchored arms of base 0, the form of control p3 under the pod's memory limit, and the run list of source `cand-imagkin`

Note to card `PREREG_V8F_PROOF_ADD40.md` (after notes N1 to N4). Frozen after the tool checks of sequences lft0k2, rlam0k1 and rlft0k1 were read (controls and diagnostics only), and before: any run of the lists of section 3 exists beyond the ones named there as existing; any unsafe rate, family value or position error of any run of the tool has been read. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames, the lead does not react; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25).

## 1. What is measured (MEASURED, base 0, tool checks that name nothing; `imagplan/read_plumb/PLUMB3_RL*_s0.json`, `PLUMB4_*_s0.json`, `imagplan/read_replay/REPLAY_RL*_s0.json`)

* Source `cand-imagkin` on LFT_s0, RLAm_s0, RLFT_s0: k1 (the known response) 0.0 m on every active tick; k2 (the level equals the plan's imagination of a `cand-imag` run at tick 0) 0.0 m. On LFT_s0 also p1 first part and p1b (equal `off` in 141 of 141 files) and k3 (constant gap: 0 changed ticks; share of tick-0 choices above d = 0: 0.099 without, 0.772 with the shift of -20 m).
* Control p5r (the tool's state path against the reader, matched windows, 2.0 s): RLAm_s0 0.638 against 0.644; RLFT_s0 -0.790 against -0.790. Met on both.
* Control p5b (source `cand-imag`, decel family): RLFT_s0 -0.427 [-0.504, -0.358] (bar below 0.25: met). RLAm_s0 +0.495 [+0.452, +0.542] (bar at least 0.5: NOT met, by 0.005; reported as written). With the level kept (imagined gap under the plan at 2.0 s: median 25.4 m beside a present gap of 26.5 m) the response does carry over to the model's own plans, at about half of the kinematic size; on LA_s0 (level lost) it was +0.185.
* Beside, the second diagnostic at tick 0 (the reader's offsets on five baselines, active windows, slope at 2.0 s): RLAm_s0 logged 0.703, logged acceleration with the plan's yaw 0.701, the plan's acceleration with logged yaw 0.684, the plan 0.659, the plan with both channels at their mean 0.724. RLFT_s0: -0.839, -0.846, -0.700, -0.705, -0.833. LFT_s0: -0.396, -0.389, -0.280, -0.282, -0.390. On all three checkpoints the plan's ACCELERATION sequence weakens the response (its spread is 2.0 m/s2 against 0.79 m/s2 of the logged one, correlation 0.29) and the mean form restores it; the yaw channel does not matter.
* A convention, stated because it sits near the bar: p5b divides the imagined gap difference by the candidate's executed progress difference, which is 2.2 m per m/s2 of d at 2.0 s (the step speeds of the decel family); the reader's expectation for the same action offset is 2.0 m. A checkpoint with a reader slope of 1 reads 0.91 on p5b. The bar is not changed.

## 2. Control p3 under the pod's memory limit

The noise run copies the tokens of every window; the run IPknx_LFT_s0 was killed by the container's memory limit (50 GB) beside two trainers at 2026-11-15T23:28:17Z, the trainers survived. From here every sequence of the tool runs under a memory guard (`pod_imagplan_seq5.sh` and later forms). Form of p3 while trainers run: the noise run and its reference run on the FIRST 600 windows of the grid (`--max-windows 600` on both), compared bit for bit. The full form (2,399 windows) is run once per source when the pod has 40 GB free and is reported beside it; if the full form cannot be run before a row is written, the row says so. For source `cand-imag` the full form passed on LFT_s0 (rows of the first sequences, note N2).

## 3. The run list of source `cand-imagkin` (note N4 section 3), per checkpoint, decel family, suffix x

Checkpoints: the control arm RLFT at bases 0 and 1 (the arm on which the names `IMAG-LEVEL-*` are read first), and RLAm at bases 0 and 1 beside it (an arm with the term; it names nothing until note ADD37_N4 confirms the low weight).

* `IPo` off (run without the family variable) | `IPk` cand-imagkin with the p4/p5 log | `IPkid1` set {0}, no fallback | `IPkid3` constant gap 1e9 m | `IPks` level shifted by -20 m | `IPc` cand-cv | `IPl` cand-label | `IPi` cand-imag with the p4/p5 log (beside).
* p3: `IPk` and `IPkn` with `--max-windows 600` (suffix m).
* Existing at the freeze: IPix and IPkx of RLAm_s0 and of RLFT_s0 (controls read, no outcome read).
* Order of reading, per arm with both bases present: first the controls (p1 first part, p1b, p3, k1 to k3, p5r, p7, then p2 = `cand-label` cuts U against `off`, p6 = the shift raises the position error over 6 s, p8); only if all pass, the names of note N4 section 3 (U of `cand-imagkin` against `off`, against `cand-cv`, against `cand-label`; the families against `off`). p2 and p6 are outcome reads of control runs; they are read before the named differences and reported whatever they show.

## 4. What follows (committed before the data)

* Section 4 of note N4 stands. In addition, for source `cand-imag`: p5b is read on arm X at the bases note ADD37_N4 names; if it is again not met by a margin inside its interval, the candidate actions in the mean form of the second diagnostic (acceleration and yaw of the plan replaced by their mean over the ten steps, the candidate's d subtracted from the mean) are the next form of that source, with p5b re-read on it under the same bars, in an own note before any run.
* If the names of `cand-imagkin` on RLFT read `IMAG-LEVEL-NO-CUT` or `-WORSE-THAN-EXTRAPOLATION`: reported as written; p6, k3 and the gap bias at 2.0 s say whether the level is the cause; next lever a calibration of the imagined level on training clips (own card).

## 5. Limits

Base 0 only for section 1; tool checks on a dose arm name nothing; the p3 subset form covers a quarter of the windows.
