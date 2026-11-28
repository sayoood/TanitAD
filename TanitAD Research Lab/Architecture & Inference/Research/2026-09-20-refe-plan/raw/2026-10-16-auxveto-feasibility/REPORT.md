# Aux-raster (geoDAC) veto feasibility: inventory, a validated instrument, agreement with NAVSIM's DAC, cost

**Built by a Sonnet agent; verified by the Master Mind on 2026-10-16:**
- `MANIFEST.sha256` 25/25 by sha256;
- `test_geodac.py` reproduces `ZZTEST_PASS 39 14` on the MM's run;
- `agreement*.py` computes no pick, PDMS or veto outcome.

The agent's harness refused a separate report file, so the MM wrote this one from its hand-back.

## Inventory (`INVENTORY.md`, `INVENTORY.json`)
For the 1,058 C3 tokens (123 logs) everything exists:
- the u400 proposals `[1058,64,8,3]`;
- the NAVSIM v1.1 table `[1058,64,6]` and logits `[1058,64,7]`; B's pick is reproduced 1058/1058;
- GT rasters 1058/1058;
- held-out PREDICTED rasters for all 1,058 tokens, from the A2 probe head (10k updates, seed 0, IoU 0.675) and from its 2k / 400-update configurations.

Index by token string: row order is `table.npz`'s, not `tokens.json`'s.

**Hazards:**
- The probe family saved predictions, not head weights.
- `c3_full/head_pred_H2-s0.npz` is an untrained fixture.
- The A3R folder's `A2_RESULT.md` says "U10k", but its units are 400 updates.

RL-val has GT rasters, but no held-out predicted rasters.

## Instrument (`geodac.py`, `test_geodac.py`)
- 39 checks: analytic identities, plus all-drivable and all-non-drivable zero controls.
- 14 deliberate-regression mutants, all RED: axis flips and swap, resolution, sign, footprint, first/last step, heading.
- The sampler matches the repo's torch `sample_pred_sdf` to 2.4e-5 m.
- The footprints match NAVSIM's raw states to 1.7e-5 m.
- The predicted rasters sit in the GT frame: the identity reproduces A2's 0.675218, and a y-flip reads 0.32.

## Agreement with NAVSIM's DAC (`AGREEMENT.json`; 67,712 proposals pooled; log-cluster bootstrap over 123 logs, B = 2,000)
| instrument | agreement | κ | NAVSIM fails found | specificity |
|---|---|---|---|---|
| GT raster, raw poses (primary) | 87.6 % [85.9, 89.4] | 0.713 [0.672, 0.751] | 77.1 % | 92.8 % |
| exact NAVSIM polygons, raw states (diagnostic) | 88.3 % | 0.722 | 72.2 % | 96.3 % |
| GT raster on NAVSIM's tracked states (diagnostic) | 94.7 % [93.8, 95.6] | 0.885 [0.861, 0.904] | 98.1 % | 93.0 % |

**Diagnosis.**
- NAVSIM v1.1's own code reproduces the table DAC on 67,712 / 67,712 proposals.
- **The cap is NAVSIM's LQR tracker:**
  - it turns 6,241 raw-OK proposals into DAC = 0, which is 27.8 % of all failures;
  - it repairs 1,665 others;
  - 76 % of the primary instrument's disagreements are tracker-made.
- The raster-only error is a boundary band: all 3,148 false alarms on tracked states lie within 0.25 m of the boundary.
- The base rate of DAC = 0 is 33.1 %, because C3 is failure-enriched. Read κ, not raw agreement.

## Cost (`COST.md`)
- The C3 held-out predictions cost 0.
- A new probe-family head costs ~1.6 h on the dev-box GPU (5 fold units).
- The full navtest is blocked on features: context features exist only for C3, and producing the rest is ESTIMATED at ~7.3 h of GPU.

## Blockers named for the veto read, and how `eval/PREREG_AUXVETO.md` handles each
1. A pose-only check misses tracker-made failures ⇒ the tracked variant is reported only, with its admissibility flagged.
2. The predicted raster's boundary accuracy (near-boundary SDF MAE 2.24 m) ⇒ a fixed 0.25 m margin, a reported margin curve, and the head-quality curve.
3. One head seed ⇒ the CI is labelled as blind to it, and a seed-1 replicate is committed before any lever claim.
4. C3 is failure-enriched ⇒ a stratified navtest estimator (the Z census at 529/12,146).
5. Mask vs SDF disagree ⇒ the mask is read, through a distance transform.

**Single copy, regenerable:** the tracked-states dump `C:/Users/Admin/tmpauxv/tracked/` (175 MB, `tracked_dump.py`, ~6 min).
