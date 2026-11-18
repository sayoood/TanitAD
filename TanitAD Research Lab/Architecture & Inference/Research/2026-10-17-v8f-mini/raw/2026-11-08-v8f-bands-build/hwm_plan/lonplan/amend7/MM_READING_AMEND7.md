# MM reading: AMEND_HWMPLAN_7 FIT read (2026-11-18 ~02:45Z)

Evidence class MEASURED (ours, zero training), tier T0. Estimator: paired clip-cluster percentile bootstrap, B 2,000, seed 0, over the 1,000 clips. It answers "another draw of clips" only. Card `AMEND_HWMPLAN_7.md` (sha256 95c97b39), frozen before data. Result `readout/G7_FIT.json`. Controls `controls/CONTROLS_FIT.json`: all green, regression arms RED.

## Outcome as committed: **R7-FIT-FAIL at both chassis** (not TRANS-ONLY, seeds agree)

| | s0 | s1 |
|---|---|---|
| (a) ΔR², MLP-A minus linear RA | +0.0005 [-0.0266, +0.0270] | -0.0104 [-0.0389, +0.0176] |
| (a) ΔR², seed set B | +0.0095 [-0.0163, +0.0348] | +0.0019 [-0.0221, +0.0261] |
| bar for (a): point above max(F_T, M_k) | 0.0238 | 0.0246 |
| (b) Δr_wm, MLP-A minus RA | **-0.0659 [-0.0773, -0.0550]** | **-0.0703 [-0.0806, -0.0602]** |
| (b) Δr_wm, seed set B | -0.0643 [-0.0761, -0.0527] | -0.0653 [-0.0752, -0.0558] |

* On the fit, the nonlinear (RY-shape) readout ties the linear one: both ΔR² CIs contain 0, and both points are far below M_k.
* On the contrast the pass uses, it is **worse**, separated at both chassis and both seed sets. Its r_wm is 0.789 / 0.799 against the linear RA's 0.855 / 0.869.
* ⇒ **The RY-shape readout is ELIMINATED as a lever at N = 1,000, at the FIT stage.** This is a result. The SCORE read does not run (the card runs it only on R7-FIT-PASS).

## Handed to ACTRSP Option B (card section 5.2)

1. **No better readout of this class is left unused by the linear `fixed_k` term.** The nonlinear one loses on the pass contrast.
2. **GAIN7 of RA5 (Option B's frozen readout) at step 0, on 1,000 clips:**
   - s0: 1.310 [1.262, 1.355];
   - s1: 1.291 [1.239, 1.339];
   - share of anchors below gain 0.25: 12.1 % / 14.3 %.

   This is consistent with Amendment 1's 1.244 / 1.202 on 141 clips (INHERITED, `amend1/out/kstep_R35_s{k}.json`). r_wm of RA5 held out on the 859 new clips is 0.870 [0.856, 0.882] / 0.871 [0.858, 0.883].
3. **RA6 against RA5.** GAIN7 of RA6 is 1.147 / 1.054. The card leaves a move of Option B's readout to RA6 as an MM ruling:
   * **Ruling: Option B keeps RA5. No amendment.**
   * Reason 1: RA5 is held out on the new clips and still reads r_wm 0.870 / 0.871. That equals the 1,000-clip RA's in-sample 0.855 / 0.869, so RA6 offers no measured advantage on the contrast the term uses.
   * Reason 2: C10' and both step-0 references of the Option B read side were measured with RA5 (C10' PASS, `optb/read/c10p/`).
   * Reason 3: a change would need an Option B amendment before the pilot, with nothing measured to buy.

## Next (Rule Zero)

* The card's next lever is ACTRSP Option B. It is built and committed (86da79cd, with the CUDA fix of its G-DVB reader), its G-SUITE is running, and the box-2 chain relaunches on a PASS.
* Under R7-FIT-FAIL the card names no further readout lever.
* Deletions by the chain (logged in `read.log`): `pairs/` (269,182,440 B) and `work/` (4,398,060 B). Both are re-creatable from `build/` and the banked inputs. The fit outputs in `fit/` stay and are landed with this reading.
