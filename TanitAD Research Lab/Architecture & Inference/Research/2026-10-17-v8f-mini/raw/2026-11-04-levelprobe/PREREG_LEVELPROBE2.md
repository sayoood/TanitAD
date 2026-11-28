# PREREG_LEVELPROBE2: H-LEVELPROBE-2, is HISTSWAP-LF's per-tick HLVL closure carried by a single per-window target speed, or does it need the per-tick logged profile?

Written and hashed BEFORE any level file of this card is built and before any rollout of it (`FROZEN_SHA2.txt`). Decided by the Master Mind after the HLVL-T0 reading of H-LEVELPROBE-1 (`RESULTS.md`: HLVL-T0 mean share closed -0.386, HARMS).

## A. THE CARD (verbatim from the Master Mind; sections B and C add facts and change no rule)

H-LEVELPROBE-2 (zero training; ORACLE diagnostics; T1-kin, paired episode-cluster bootstrap B = 2,000, seed 0; HLVL-T0's 4 cells, comparators and reader). Question: is HISTSWAP-LF's per-tick HLVL closure carried by a single per-window target speed, or does it need the per-tick logged profile? Arms (all `--histswap cv_pred --level-file`, constant twist held for the whole rollout, levels from the logged pose track of each eval-141 window, all 2,399 windows):
- **MEAN6** = mean logged speed over (t0, t0 + 6 s].
- **FUT1** = logged speed at t0 + 1 s.
- **FUT3** = logged speed at t0 + 3 s.
Known answer: a level file holding the t0 logged speeds reproduces your HLVL-T0 rows bit for bit (that is your existing cv_pred == cv_t0 test, rerun on the real files).
Committed outcomes:
- **TARGET-SPEED SUPPORTED** iff MEAN6 closes >= 0.30 of the M1 excess at >= 3 of 4 cells. => the next lever is a PREDICTED mean-speed / goal input (vision + nav; admissible; not derived from the situation classifier), pre-registered separately.
- **PROFILE** iff MEAN6, FUT1 and FUT3 all have mean closure < 0.10 => no single per-window scalar carries the closure; the scalar target-speed family is closed for M1.
- **INTERMEDIATE** otherwise, reported with the FUT1 / FUT3 / MEAN6 horizon pattern.
- HARMS flag beside each arm: ADE6 (arm - comparator) > +0.0604 m at any cell, or any non-crawl row separated-worse. Report the four families (STRATEGIC reported for v8F) and ADE6 for each arm.
- Cannot claim: oracles are diagnostics, not deployable; one training run per seed (H-ESTIM-SEED-1); T1-kin cameras follow the logged path.

## B. FACTS ESTABLISHED FROM FILES BEFORE THE FREEZE (they fill the card's brackets; they change no rule)

**B1. The speed array and the step.** Every level is read from `base_log["p10f"][j][0]`, the LOGGED measured speed of the clip arrays that `ClipStore` builds from the logged native pose track (`sfresh_inputs/inputs/native_tracks.pt`) through the programme's one spelling; it is the array HLVL (`cv_logged`) re-reads at each tick and HLVL-T0 (`cv_t0`) reads at t0. One step is 0.2 s (`DT = D.DT = 0.2`, `D.DT_NATIVE = 0.1`, two native rows per step). Hence, for a window at step t0: **FUT1 = p10f[t0 + 5][0]**, **FUT3 = p10f[t0 + 15][0]**, **MEAN6 = mean of p10f[t0 + 1 .. t0 + 30][0]** (30 values), and the known-answer file **L_T0 = p10f[t0][0]**. The rollout grid only holds windows with a complete 6 s future (`gt_ok[t0]`), so the indices exist; a non-finite value refuses the build.
**B2. The level files.** `F:/TanitAD-scratch/levelprobe_scratch/levels/L_{T0,MEAN6,FUT1,FUT3}.npz`, each with `clip_index`, `ws`, `level` (float64), one row per window of `eval_grid_windows.json` in the grid's order (2,399 rows, 141 episodes), built by `code/build_levels.py`; their sha256 are written to `FROZEN_SHA2.txt` before the first rollout. A level is held constant at every tick (`cv_pred`), the yaw rate is the executed one, as in HLVL-T0.
**B3. Runs.** Tags `LP2KA_DAGOACC_s0` (L_T0, the known answer), then `M6_*` (MEAN6), `F3_*` (FUT3), `F1_*` (FUT1), each on the four cells DAG-OACC s0 / s1 and SOUP s0 / s1 with `--smooth sg72 --histswap cv_pred --level-file`, in the order known answer, MEAN6, FUT3, FUT1, one rollout at a time on CPU (16 threads). Known-answer test: `LP2KA_DAGOACC_s0`'s dump and ticks arrays equal `HT0_DAGOACC_s0`'s bit for bit (the mode string aside); a failure refuses the read.
**B4. Comparators, rows and reader.** Y = KAPPA-T1's `SREP_*` runs (reused, pins in `histswap_lf/FROZEN_SHA.txt`); KA1 is the existing `LPKA1_DAGOACC_s0` (same tool sha256). The reader `code/levelprobe2_read.py` is `levelprobe_read.py` with the arms replaced; it keeps every known answer of HISTSWAP-LF (KY, KA1, KC1-KC3, KF, K3, KB1) and the 19 rows by `gapdecomp.decompose`. "Closes >= 0.30 of the M1 excess at a cell" = that cell's share closed, -(X - Y) / (Y - P10) on the M1 crawl-share row (undefined = 0), >= 0.30. "Mean closure" = the mean of the four cells' shares. The outcomes are evaluated in the order listed; if both TARGET-SPEED SUPPORTED and PROFILE hold, both are reported. KC1 for an arm additionally requires that the network's input speed at every tick equals the level file's value for that window (<= 2e-4 m/s).
**B5. Reference numbers** (HLVL / HCV from `histswap_lf/raw/HISTSWAP_LF_T1.json`, HLVL-T0 from `raw/LEVELPROBE.json`) are read from those files by the reader and are context only.
**B6. Deadline.** All rollouts must finish by 05:00 on 2026-11-04 (the RP01 read takes the CPU from ~05:15); the chain starts no run after 04:45 and says so.
