FAIL (M4, M5)

# MAPSOFT-G (mechanical; PREREG_MAPSOFT_G sha256 1687688bfb2b)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed

| arm (S6c) | sidewalk | edge contact | lane cross |
|---|---|---|---|
| BASE | 0.1792 | 0.2855 | 0.2939 |
| HUMAN | 0.0125 | 0.1485 | 0.1941 |
| GT-T1 | 0.0166 | 0.2148 | 0.2721 |
| OWN-T1 | 0.1657 | 0.2708 | 0.2908 |
| MAPSOFT-G-XF | 0.1710 | 0.2811 | 0.2939 |
| DER-MAPSOFT-G-XF | 0.1714 | 0.2789 | 0.2927 |

M1 sidewalk vs BASE -0.0081 [-0.0135, -0.0034] -> True
M2 edge vs BASE -0.0044 [-0.0088, -0.0003] -> True
M3 no four-family row worse -> True 
M4 deranged not below BASE (sidewalk -0.0078 [-0.0125, -0.0032], edge -0.0066 [-0.0125, -0.0016]) -> False
M5 STOP guard -> False
S6c ADE vs BASE +0.0079 [-0.0142, +0.0316]

fitted (r0, lam_s, lam_e) per fold (own): {"0": [10, 0.5, 0.0], "1": [10, 0.25, 0.0], "2": [10, 0.25, 0.0], "3": [20, 0.25, 0.0], "4": [10, 0.5, 0.0]}
fitted per fold (deranged): {"0": [0, 0.5, 0.0], "1": [10, 0.5, 0.0], "2": [10, 0.25, 0.0], "3": [0, 0.5, 0.0], "4": [10, 0.5, 0.0]}
share of GT-T1's sidewalk gain: 0.04999999999999993

controls failed: none

NEXT: the trade-off costs driving at the fitted point: constrain ALL four families in the fit (multi-objective).

ZZMSOFTGENDZZ