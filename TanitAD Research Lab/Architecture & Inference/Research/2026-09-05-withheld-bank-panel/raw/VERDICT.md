# H-EGO-LIT-4 — pre-registered outcome evaluation
_source: panel_report.json; RIG RUNG (tiny) on a NON-PARITY corpus — validates the DESIGN and the GATE, never a model claim (H-SCALE-2)_
_n_windows 640 from 40 val episodes; estimator paired_episode_cluster_bootstrap_

## OUTCOME IV (checked FIRST) — does the gate FAIL the deliberate regression?

A5_regress (`--ablate-frames`, an echo BY CONSTRUCTION): gate2b **ECHOING**, scene degradation +0.0000, gate raised **True**

✅ the gate FAILS the regression ⇒ the panel is VALID.
   verdict is ECHOING — the informative failure: wrong ego hurts, wrong scene does not.

## The run-to-run noise floor (added mid-panel; not in the original pre-registration)

A0b_replicate = A0's flags and A0's seed, run a second time. Its paired delta vs A0 is pure nondeterminism at the level the panel scores.

| metric | withheld regime | kept regime |
|---|---|---|
| LON_speed_mae_mps | +0.0577 [-0.2883, +0.3967] ns | +0.0827 [-0.0095, +0.1784] ns |
| LON_along_mae_m | +0.0548 [-0.3799, +0.4759] ns | +0.0955 [+0.0054, +0.1935] SEP |
| LON_accel_mae_mps2 | +0.1638 [+0.0222, +0.3019] SEP | +0.0458 [-0.0963, +0.1874] ns |
| LAT_cross_mae_m | -0.0531 [-0.1130, +0.0120] ns | -0.0435 [-0.0941, +0.0075] ns |
| LAT_heading_mae_deg | +0.1498 [-0.8324, +1.1545] ns | -0.6526 [-1.4830, +0.1463] ns |
| LAT_yaw_rate_mae_radps | +0.0759 [+0.0164, +0.1473] SEP | -0.0075 [-0.1351, +0.0988] ns |
| ade_m | +0.0180 [-0.4164, +0.4331] ns | +0.0502 [-0.0447, +0.1509] ns |

⚠️ Any arm's delta that is not clearly LARGER than the corresponding cell is not attributable to its lever, regardless of its own CI.

## The echo instrument, kept regime (the deployed regime)

| arm | bank | gate2b verdict | scene rel | ego rel | gate raised |
|---|---|---|---|---|---|
| A0_fixed | fixed | **ECHOING** | +0.0732 | +0.1286 | True |
| A0b_replicate | fixed | **ECHOING** | +0.1211 | +0.1169 | True |
| A1_pred | pred | **READS_BOTH** | +0.4619 | +0.1642 | True |
| A2_random | random | **ECHOING** | +0.0942 | +0.1282 | True |
| A3_drop25 | fixed | **ECHOING** | +0.2130 | +0.5745 | True |
| A4_none | none | **IGNORES_EGO** | +0.7591 | +0.0417 | True |
| A5_regress | fixed | **ECHOING** | +0.0000 | +2.5957 | True |

constant-predictor control (MUST read exactly 0.0000 / 0.0000): scene +0.0000, ego +0.0000

## The pre-registered outcomes

- **ADOPT `pred` for refcv5** — — no
    A1 gate2b=READS_BOTH; withheld speed -0.1655 [-0.4159, +0.0619] ns; withheld along -0.1952 [-0.5103, +0.0971] ns; A1−A2 speed -0.1399 [-0.2735, -0.0046] SEP; larger than replicate floor: False
- **REFUSE `pred` (A1 reads ECHOING ⇒ the loop echoes its own prior)** — — no
    A1 gate2b=READS_BOTH, scene rel +0.4619
- **BANK GEOMETRY IS NOT THE BINDING CONSTRAINT (A1 ≈ A2)** — — no
    A1−A2 withheld speed -0.1399 [-0.2735, -0.0046] SEP — a control that matches the arm means the gain is not information
- **RETIRE the per-window roll (A4 ≥ A0 on echo AND families)** — — no
    A4 gate2b=IGNORES_EGO; withheld speed -0.0785 [-0.3523, +0.1982] ns
- **PlanTF signature — keep ego_dropout 0.5 (A3 better families, lost separation)** — 🔥 FIRES
    A3 gate2b=ECHOING; kept speed -0.1463 [-0.2115, -0.0796] SEP
- **10 m/s IS A BIASED CHOICE (A2 ≫ A0 ⇒ re-run A0 at the marginal's mean)** — — no
    A2 withheld speed -0.0256 [-0.2583, +0.1814] ns; training marginal mean 5.7998 m/s vs the fixed 10.0
- **THE HARM IS ANY SPEED-BLIND BANK, NOT 10 m/s (A2 ≈ A0)** — 🔥 FIRES
    A2 withheld speed -0.0256 [-0.2583, +0.1814] ns

## Summary

outcomes fired: ['PlanTF signature — keep ego_dropout 0.5 (A3 better families, lost separation)', 'THE HARM IS ANY SPEED-BLIND BANK, NOT 10 m/s (A2 ≈ A0)']

_RIG scope: ~19 M params, 48 non-parity training episodes, 2,000 steps, one seed per arm. Validates a DESIGN and a GATE, never a model claim (H-SCALE-2); nothing here enters MODEL_REGISTRY.md._