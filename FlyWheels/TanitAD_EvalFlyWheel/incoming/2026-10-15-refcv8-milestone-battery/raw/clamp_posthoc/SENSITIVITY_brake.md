# SENSITIVITY_brake -- EXPLORATORY (not covered by the amendment; no verdict)

Tree `step30000ema_unc`, the landed function with only `brake_ms2` varied (landed = 2.5 m/s^2 = `sel.accel_max`). TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). ESTIMATOR: episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run. Cross-check (brake 2.5 reproduces the main read's paired ON - OFF ADE exactly): **True**.

| seed | brake (m/s^2) | binding rate (n) | still above ceiling after clamp (n) / max excess m/s | S2 ADE ON - OFF | S2 os - ha0_ext (ON) | S6 ADE ON - OFF | S6 os - ha0_ext (ON) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 5.30 % (252) | 0.00 % (0) / - | +0.1145 [+0.0527, +0.1902] **W** | +0.1137 [+0.0488, +0.1878] | +0.1910 [+0.0475, +0.3693] **W** | -0.6236 [-0.9840, -0.2667] **B** |
| 0 | 1.0 | 5.03 % (239) | 5.22 % (248) / 6.1229 | +0.0305 [+0.0150, +0.0506] **W** | +0.0297 [+0.0081, +0.0536] | +0.0654 [+0.0061, +0.1558] **W** | -0.7492 [-1.0784, -0.4312] **B** |
| 0 | 2.5 (landed) | 5.30 % (252) | 4.69 % (223) / 5.7479 | +0.0629 [+0.0316, +0.1017] **W** | +0.0621 [+0.0267, +0.1022] | +0.1275 [+0.0287, +0.2563] **W** | -0.6871 [-1.0261, -0.3526] **B** |
| 0 | 5.0 | 5.30 % (252) | 3.22 % (153) / 5.1229 | +0.0870 [+0.0425, +0.1412] **W** | +0.0862 [+0.0386, +0.1408] | +0.1589 [+0.0386, +0.3151] **W** | -0.6557 [-1.0064, -0.3100] **B** |
| 0 | 10.0 | 5.30 % (252) | 1.37 % (65) / 3.8729 | +0.1048 [+0.0493, +0.1709] **W** | +0.1040 [+0.0463, +0.1703] | +0.1789 [+0.0450, +0.3502] **W** | -0.6357 [-0.9888, -0.2812] **B** |
| 1 | 0.0 | 5.30 % (252) | 0.00 % (0) / - | +0.1135 [+0.0518, +0.1884] **W** | +0.1149 [+0.0496, +0.1901] | +0.1815 [+0.0402, +0.3555] **W** | -0.6087 [-0.9736, -0.2541] **B** |
| 1 | 1.0 | 4.99 % (237) | 5.22 % (248) / 6.1229 | +0.0297 [+0.0144, +0.0495] **W** | +0.0312 [+0.0097, +0.0552] | +0.0577 [-0.0003, +0.1481] | -0.7325 [-1.0666, -0.4139] **B** |
| 1 | 2.5 (landed) | 5.28 % (251) | 4.69 % (223) / 5.7479 | +0.0619 [+0.0309, +0.1003] **W** | +0.0634 [+0.0279, +0.1039] | +0.1183 [+0.0209, +0.2456] **W** | -0.6719 [-1.0117, -0.3391] **B** |
| 1 | 5.0 | 5.30 % (252) | 3.22 % (153) / 5.1229 | +0.0860 [+0.0421, +0.1396] **W** | +0.0874 [+0.0393, +0.1418] | +0.1495 [+0.0312, +0.3016] **W** | -0.6407 [-0.9904, -0.2993] **B** |
| 1 | 10.0 | 5.30 % (252) | 1.37 % (65) / 3.8729 | +0.1038 [+0.0485, +0.1697] **W** | +0.1052 [+0.0468, +0.1722] | +0.1694 [+0.0375, +0.3371] **W** | -0.6208 [-0.9794, -0.2724] **B** |

Reading: brake 0 is the STRICT clamp (the literal R1: the plan may never exceed the ceiling, even when v0 already does). For a positive brake the allowance is max(limit, v0 - brake * t_mid): a SMALLER brake leaves an over-ceiling ego more speed (looser), a LARGER brake forces it down to the limit sooner (stricter), and brake -> infinity converges to the strict case. 'still above ceiling' counts rows more than 0.01 m/s over the ceiling after the clamp (fp32 dust of a plan capped AT the ceiling is excluded; the strict > 0 count is in the JSON). The ADE column is a cost against the LOGGED ego, which itself exceeds the ceiling on every binding window (CLAMP_POSTHOC.md section 8); it does not say which point the programme should pick.
