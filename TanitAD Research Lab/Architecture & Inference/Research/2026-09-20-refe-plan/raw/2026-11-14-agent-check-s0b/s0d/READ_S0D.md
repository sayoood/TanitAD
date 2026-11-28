# S0-D read (R-sim: agent lists through NAVSIM's own scorer): S0D-NULL. The agent check closes at Stage 0.

Master Mind, 2026-11-14.
- **Freeze:** `FREEZE_S0D.md`, landed `e3fe96ac` before the run.
- **Run:** `run/` (`S0D_RESULT.md`, `s0d_result.json`, `s0d_flags.npz`, logs).
- **Evidence class:** MEASURED. Dev-box CPU, 4,525 s, peak RSS 772 MB.
- **Tier:** a navtrain offline readout against NAVSIM's own sub-scores of P6's banked proposals. It is not a driving number, and nothing was keyed.
- **Estimator:** paired log-cluster bootstrap (10,000 draws, seed 20260927) over the 13 held-out logs. It answers "would another draw of logs say this" only.

## Controls (all valid)
- **Tests:** `test_s0d.py` OK; `--mutations` 12 RED.
- **R-sim-true is an EXACT identity:**
  - flags = truth on every proposal (0 mismatches);
  - raw nc and ttc = the fit pack (0 mismatches);
  - per-set gain = the oracle;
  - G-FID 1.000 / 1.000.
- **C-SIM-NONE (dynamic agents removed):** 0 count violations and 0 subset violations against R-sim-true.

## Result (held-out covered sets, n = 1,343, 13 logs; oracle O_NC+TTC = +1.811 PDMS)
| arm | gain [95 % CI] | capture of O [95 % CI] | NC-only / TTC-only capture | G-FID top-5 precision / recall |
|---|---|---|---|---|
| R-sim-true (identity) | +1.811 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **R-sim-t0** (single frame, velocity 0) | **−2.406** [−4.021, −0.810] | −1.328 | −7.39 / −1.38 | 0.096 / 0.825 |
| R-sim-t0v (single frame, TRUE velocity) | −2.406 [−4.021, −0.810] | −1.328 | −7.14 / −1.38 | 0.096 / 0.825 |
| **R-sim-cv** (t0 pose + true velocity, constant) | **+0.524** [+0.147, +1.074] | **0.290** [0.073, 0.634] | 0.16 / 0.30 | 0.383 / 0.578 |
| C-SIM-NONE | +0.030 [0, +0.093] | 0.016 | 0.00 / 0.02 | 1.000 / 0.029 |

**Outcome (committed table): S0D-NULL.**
- c(R-sim-t0) is −1.33, against the 0.25 bar.
- c(R-sim-t0v) is −1.33, against 0.25.
- c(R-sim-cv) is 0.29, against the 0.50 bar for MOTION.

## What it says (MEASURED)
1. **A single-frame agent list is harmful even through NAVSIM's own at-fault logic.**
   - Frozen agents raise 37,529 TTC flags against 17,365 true ones, so most of P6's top 5 is vetoed wrongly: precision 0.096.
   - **R-sim-t0v equals R-sim-t0**, so the stopped-track rule is not the cause. The frozen GEOMETRY is: agents that will move out of the ego's path are treated as standing in it.
2. **Motion is necessary but not sufficient.**
   - Even the TRUE t0 velocity extrapolated at constant velocity recovers only 0.29 of the oracle (+0.52 PDMS).
   - The rest needs the true future: R-sim-true recovers 1.0 by construction, and the S0-B footprint readout on the true future recovers 0.43.
3. **Together with S0-B (0.43 on the true future, −1.48 on a single frame) and S0-C (fusion 0.08):** no readout tested here turns what a perception head can plausibly supply (a single frame, or a single frame plus velocity) into ≥ 0.25 of the oracle agent term.

## Next (Rule Zero)
- **The card's S0D-NULL row closes the agent check at Stage 0.** It names a hard-negative fine-tune of P6's NC/TTC heads as the last successor.
- **That successor has close MEASURED precedents, both noise-level:**
  - `H-SCORER1P6-1` and `H-REP-P6-1`: S1E, a pair loss on NC, DAC and TTC, gives +0.191 vs P6 [+0.012, +0.372], below its R1 threshold of 0.352;
  - S1Z, the zero-pick / TTC-0 twin loss: Z5 SWAMPED.
  - So it is ranked BELOW the levers with the largest measured effect and is not run next.
- **By measured effect, the next REFe lever stays P5-C:** tracker-aware Tc20s / P5-R labels. TF1 measured +7.64 PDMS for Tc20s over the raw teacher targets under LQR tracking, teacher side. Its launch kit is being finished and runs after P7.
- Behind P5-C comes the navtrain teacher cascade (TF-N7's REACHED row), which needs the RJ5 bulk labels now downloading.
- **A motion-conditioned agent head (R-sim-cv's 0.29, about +0.5 PDMS on navtrain held-out) is the only agent-side lever with a positive measured ceiling.** It is an architecture change (card Q-D) and goes to the PI as an option, not a recommendation.
