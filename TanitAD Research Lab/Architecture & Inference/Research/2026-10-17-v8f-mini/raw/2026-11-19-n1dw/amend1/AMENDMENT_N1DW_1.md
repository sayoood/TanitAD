# PREREG_V8F_N1DW, Amendment 1: the probe's regression arm, and its tree seeds. Frozen before any full PROBE-N1D-FEAT result

**Status:** frozen by the Master Mind (MM) on 2026-11-19, while the full probe run was still extracting features. The pod log read `[4512/5736]`, and `out/result/` did not exist.

**Seen before freezing:** only the build agent's 400-window smoke, which was a pipeline check, and the two facts below.

**Unchanged:** every N1DW arm, bar, outcome and control, and every PROBE-N1D-FEAT feature set, probe setting and outcome row. Card: `PREREG_V8F_N1DW.md`, sha256 59f2b7f9.

## 1. The regression arm as committed cannot remove the signal it tests
* **Committed (section 10):** "F-TEXT with column c0 and the exit columns zeroed must read separated below F-TEXT". The arm's purpose is to prove the probe sees the text.
* **Why it cannot.** NAVRBD section 2 defines the channel to be silent off roundabouts: every column is 0 wherever c0 is 0.
  - Column c14 (`dist_valid`) and column c13 (`dist`) are therefore nonzero only where c0 = 1.
  - So c14 and c13 still carry the roundabout support after c0 and the exit columns are zeroed. The arm leaves a copy of the signal in place.
* **Seen on the 400-window smoke:** the committed arm read -0.0225 [-0.0625, +0.0200], not separated. The same arm with all 15 columns zeroed read about -0.18, separated.
* **Amended.** The binding regression arm is **F-TEXT with all 15 columns zeroed, which must read separated below F-TEXT** at both N1D seeds. The committed arm is still run and printed with its value, beside, labelled "as committed; cannot remove the signal (c13 and c14 carry c0's support)". If it does not separate, it is reported as written.
* **Root-cause class:** a deliberate-regression arm that removes a signal from some columns while a redundant copy stays in others. The design check that would have caught it: for every column the arm keeps, ask whether that column is a function of a removed one.

## 2. The tree probe has no seed variance
* The gradient-boosted tree, with RBJK's settings (no subsampling, no early stopping), makes no random draw. Probe seeds 0 and 1 give identical tree predictions. The logistic probe is also deterministic.
* So "each outcome must hold at both N1D seeds and both probe seeds" adds nothing beyond the two N1D seeds for the deciding tree probe.
* The rule is kept as written. The read prints "probe seeds identical for the tree (no random draw)". Probe variance is NOT measured by this card; training variance is carried by the two N1D seeds only.
* The probe seed still drives the SHUF control's permutation, where it does vary.

## 3. Other build facts (no change to the card)
* The extractor reproduces the banked N1D dump: the tick-0 plan differs by 0.0 m, and the labels agree on 400 of 400 smoke windows.
* The pod tree `/root/strat1/navrbd/tree` equals 0c3b374f blob by blob: 2,630 of 2,630 non-test blobs.
* scikit-learn 1.9.1 was installed into the probe's own target folder. The pod's torch was not touched.
* The probe code and the handoff are landed beside this amendment, under `../probe/`.
