# V-R8-noLW: the pre-registered readout (2026-10-14) -- CONFIRMED

V-R8-noLW = V-R8 with `--w-r8-listwise 0` (the X1 listwise loss off, so the selection CE is back; argv sha 5ca38d3c,
only `--out` differs from the pre-registered dc4c1fa0). Trained on Thor 04:23Z-~08:50Z (3,000 steps), Thor base eval
(batch 8), pulled and md5-verified (`V-R8-noLW_model_final.md5` = the run's own `model_final.pt.md5`), evaluated on
the dev box at batch 4 with `--save-cands`, oracle variants, then `nolw_readout.py` -- the readout code written
2026-10-12, BEFORE this arm existed (`PREREG_VR8_NOLW.md` + `_A1.md`). Pipeline: `noLW_post.sh`.

## Pre-registered verdict: CONFIRMED (`readout.txt`, `readout.json`)

GT-turn windows (n 107), `cross_abs_6s`, paired episode-cluster bootstrap, dev box batch 4 vs V-R8 batch 4:

| seed | noLW | V-R8 | d | CI | floor (V-R8r - V-R8, Thor) | R | P1 |
|---|---|---|---|---|---|---|---|
| 0 | 10.418 m | 14.984 m | -4.566 | [-8.035, -1.480] BETTER | 3.303 | 1.38 | pass |
| 1 | 10.159 m | 15.141 m | -4.982 | [-8.543, -1.644] BETTER | 2.949 | 1.69 | pass |

P2 (the candidate bank is not damaged: oraclebase cross on GT turns): d -0.103 [-0.499, +0.319] and -0.219
[-0.588, +0.140], ns on both seeds -> pass. **The X1 listwise loss caused the turn defect; removing it removes it.**

## Four families (Thor batch-8 base dumps, all 1,112 windows; F = max(|V0r - V0|, |V-R8r - V-R8|) per seed;
"beyond F" = separated AND |d| > F on BOTH seeds) -- descriptive, POST-HOC, T0 open-loop

**vs V-R8** (`noLW_vs_VR8_thor_describe.json`): LATERAL better beyond F -- cross 6 s -0.651 / -0.657 m, curvature
0-2 s, yaw-rate 0-2 s; turn direction unchanged (-0.019 / -0.028 ns). PLAN ADE -0.266 / -0.265 m. SAFETY off-drivable
-0.057 / -0.048. LONGITUDINAL progress error better; speed MAE 0-2 s WORSE +0.015 / +0.015. TACTICAL lat3 vs GT plan
-0.013 and constraint MAE (speed) +0.007, both just at F.

**vs V0** (`noLW_vs_V0_thor_describe.json`): LATERAL equal (turn direction -0.009 / 0.000, heading, cross, curvature,
yaw all ns). LONGITUDINAL better beyond F -- along-track 6 s -0.313 / -0.339 m, speed MAE 2-6 s -0.083 / -0.098,
progress error -0.013 / -0.015; `ceil_violation` (any overshoot) WORSE +0.009 / +0.009 (0.052 -> 0.061). TACTICAL lat3
F1 0.315 -> 0.747 (⚠ V0's p_lat predicts ONE class on 1,112 / 1,112, so this is a win over a degenerate head).
SAFETY off-drivable -0.025 (seed 0, beyond F) / -0.021 ns. PERCEPTION map lane IoU -0.043 (within F).

## Ceiling compliance at the CEIL-OBEY definition (SPEC_REFCV8 8.3 (a)(2)) -- INDICATOR, not the bar reading

Emitted plan max speed > fed finite ceiling + 0.5 m/s, EXEMPT when the fan's slowest candidate also exceeds it
(`fan_min_vmax`); all 1,112 finite-ceiling windows (the bar's S-ROUTE restriction is NOT applied here):

| arm | violations > 0.5 m/s | exempt | NON-EXEMPT (bar <= 0.01) |
|---|---|---|---|
| V0 | 56 / 56 | 42 / 43 | **0.0126 / 0.0117** |
| V-R8 | 62 / 60 | 53 / 54 | 0.0081 / 0.0054 |
| V-R8-noLW | 61 / 60 | 51 / 51 | **0.0090 / 0.0081** |

noLW's extra raw overshoots sit in EXEMPT windows (every candidate too fast): a bank-side matter, not the selector.

## What it means for the launch

The R8-RECIPE flags for the launch: V-R8's, with `--w-r8-listwise 0` (the noLW argv). It is equal to V0 laterally and
better longitudinally at this rung; the refcv8 hypothesis that tactical conditioning improves TURN DIRECTION is not
supported at the tiny rung (no arm moved it beyond F). Awaits the PI (PI_DECISION_QUEUE, 2026-10-14 noLW item).
