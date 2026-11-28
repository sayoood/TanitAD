FAIL (M1, M2)

# MAPPATH (mechanical; PREREG_MAPPATH sha256 34a030eef961 + AMENDMENTS 1-2)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed

| arm (S6c) | sidewalk | edge contact | lane cross |
|---|---|---|---|
| BASE | 0.1792 | 0.2855 | 0.2939 |
| HUMAN | 0.0125 | 0.1485 | 0.1941 |
| GT-T1 | 0.0219 | 0.2217 | 0.2813 |
| OWN-T1 | 0.1710 | 0.2805 | 0.2980 |
| MAPSOFT-G-XF | 0.1792 | 0.2855 | 0.2939 |
| DER-MAPSOFT-G-XF | 0.1782 | 0.2864 | 0.2936 |

M1 sidewalk vs BASE +0.0000 [+0.0000, +0.0000] -> False
M2 edge vs BASE +0.0000 [+0.0000, +0.0000] -> False
M3 no four-family row worse -> True 
M4 deranged not below BASE (sidewalk -0.0009 [-0.0028, +0.0006], edge +0.0009 [+0.0000, +0.0025]) -> True
M5 STOP guard -> True
S6c ADE vs BASE +0.0000 [+0.0000, +0.0000]

fitted (r0, lam_s, lam_e) per fold (own): {"0": [0, 0.0, 0.0], "1": [0, 0.0, 0.0], "2": [0, 0.0, 0.0], "3": [0, 0.0, 0.0], "4": [0, 0.0, 0.0]}
fitted per fold (deranged): {"0": [0, 0.0, 0.0], "1": [0, 0.0, 0.0], "2": [0, 0.0, 0.0], "3": [0, 0.0, 0.0], "4": [2, 0.25, 0.0]}
share of GT-T1's sidewalk gain: 0.0

controls failed: none

NEXT: name the path-only ceiling (GT-T1 on the progress-matched fan, beside): if it is ALSO small the sidewalk defect is a SPEED/reach effect and the lever moves to the lon side; if it is large the own map's 0.905 AUC does not convert: the authorised map-only fine-tune.

ZZMAPPATHENDZZ