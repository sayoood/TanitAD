# GAPSPREAD10K verdict (card PREREG_V8F_PROOF_ADD134)

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m); setting K0: readout K, no pre-roll, rule every 5 steps. A kinematic rollout on recorded frames, on the model's own actions; not driving performance.

Controls: p0 to p7 all pass (controls file sha256 6db11234b0e3ca2d). This card has no map of arms to controls.

Estimator: two-level bootstrap over the four model seeds and the evaluation episodes, 2,000 draws, on the same windows for both arms of a pair. The interval answers what another draw of episodes and of these four model seeds would say, not what another training run would say; the four seeds are two pairs of training runs of one line.

## Names

- GAPSPREAD10K-K0-PLACED-NOT-SEPARATED-FROM-FLAT
- GAPSPREAD10K-K0-VS-CANDIDATE-CONTACT-NOT-SEPARATED
- GAPSPREAD10K-K0-VS-CANDIDATE-FALLING-BEHIND-NOT-SEPARATED
- GAPSPREAD10K-K0-VS-CANDIDATE-LOWERS-UNSAFE
- GAPSPREAD10K-K0-VS-FLAT-CONTACT-NOT-SEPARATED
- GAPSPREAD10K-K0-VS-FLAT-FALLING-BEHIND-NOT-SEPARATED
- GAPSPREAD10K-K0-VS-FLAT-UNSAFE-NOT-SEPARATED

## Setting K0 (no pre-roll)

Windows with a lead 642, episodes with a lead 56.

### Levels (contact C, unsafe U, falling behind F)

- SP90 pooled over four model seeds C 0.0004 U 0.0365 F 0.1651; per model seed {"C": [0.0, 0.0, 0.0, 0.001557632398753894], "F": [0.1573208722741433, 0.1853582554517134, 0.1542056074766355, 0.16355140186915887], "U": [0.03914590747330961, 0.037366548042704624, 0.03558718861209965, 0.033807829181494664]}
- A90 pooled over four model seeds C 0.0000 U 0.0365 F 0.1651; per model seed {"C": [0.0, 0.0, 0.0, 0.0], "F": [0.1573208722741433, 0.18847352024922118, 0.15109034267912771, 0.16355140186915887], "U": [0.03914590747330961, 0.03558718861209965, 0.03914590747330961, 0.03202846975088968]}
- GEc pooled over four model seeds C 0.0000 U 0.0480 F 0.1589; per model seed {"C": [0.0, 0.0, 0.0, 0.0], "F": [0.14797507788161993, 0.1822429906542056, 0.14641744548286603, 0.1588785046728972], "U": [0.051601423487544484, 0.04804270462633452, 0.046263345195729534, 0.046263345195729534]}
- RLc pooled over four model seeds C 0.0004 U 0.0369 F 0.1460; per model seed {"C": [0.0, 0.0, 0.001557632398753894, 0.0], "F": [0.14018691588785046, 0.16978193146417445, 0.13395638629283488, 0.14018691588785046], "U": [0.03914590747330961, 0.033807829181494664, 0.03914590747330961, 0.03558718861209965]}
- RR pooled over four model seeds C 0.0300 U 0.0703 F 0.1589; per model seed {"C": [0.03115264797507788, 0.028037383177570093, 0.028037383177570093, 0.03271028037383177], "F": [0.17445482866043613, 0.16199376947040497, 0.16199376947040497, 0.13707165109034267], "U": [0.07473309608540925, 0.07295373665480427, 0.06227758007117438, 0.0711743772241993]}

### Pairs (first minus second, with intervals)

- SP90-VS-A90: UNSAFE {"interval": [-0.005294390655682338, 0.005295028755632872], "per_seed": [0.0, 0.0017793594306049821, -0.0035587188612099642, 0.0017793594306049821], "pooled": 0.0}; CONTACT {"interval": [0.0, 0.0016584574634306033], "per_seed": [0.0, 0.0, 0.0, 0.001557632398753894], "pooled": 0.0003894080996884735}; FALLING-BEHIND {"interval": [-0.006602844644843676, 0.006358186431180217], "per_seed": [0.0, -0.003115264797507788, 0.003115264797507788, 0.0], "pooled": 0.0}
- SP90-VS-GEc: UNSAFE {"interval": [-0.019531031043481528, -0.0051415503434095264], "per_seed": [-0.012455516014234875, -0.010676156583629894, -0.010676156583629894, -0.012455516014234875], "pooled": -0.011565836298932385}; CONTACT {"interval": [0.0, 0.0016584574634306033], "per_seed": [0.0, 0.0, 0.0, 0.001557632398753894], "pooled": 0.0003894080996884735}; FALLING-BEHIND {"interval": [-0.0007734166350007021, 0.012885555861022406], "per_seed": [0.009345794392523364, 0.003115264797507788, 0.00778816199376947, 0.004672897196261682], "pooled": 0.006230529595015576}
- SP90-VS-RLc: UNSAFE {"interval": [-0.008371298985458565, 0.00792072236016048], "per_seed": [0.0, 0.0035587188612099642, -0.0035587188612099642, -0.0017793594306049821], "pooled": -0.00044483985765124553}; CONTACT {"interval": [-0.0015944540322076805, 0.0015554451652307433], "per_seed": [0.0, 0.0, -0.001557632398753894, 0.001557632398753894], "pooled": 0.0}; FALLING-BEHIND {"interval": [0.012356234348248632, 0.026377506551680602], "per_seed": [0.017133956386292833, 0.01557632398753894, 0.020249221183800622, 0.02336448598130841], "pooled": 0.0190809968847352}
- A90-VS-GEc: UNSAFE {"interval": [-0.017481538004779657, -0.0062612284981319], "per_seed": [-0.012455516014234875, -0.012455516014234875, -0.0071174377224199285, -0.014234875444839857], "pooled": -0.011565836298932385}; CONTACT {"interval": [0.0, 0.0], "per_seed": [0.0, 0.0, 0.0, 0.0], "pooled": 0.0}; FALLING-BEHIND {"interval": [0.001897977434707977, 0.011226520234455362], "per_seed": [0.009345794392523364, 0.006230529595015576, 0.004672897196261682, 0.004672897196261682], "pooled": 0.006230529595015576}

### Families (longitudinal, lateral, tactical: engine reads of each pair; strategic: grid at T1-kin, position error at 30, 50 and 80 m)

- SP90-VS-A90: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- SP90-VS-GEc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED

## Limits (card section 7)

T1-kin only, no re-perception: the lead moves as logged whatever the ego does; readout K with no pre-roll alone (the margin is a setting of readout K); four model seeds of one line, two pairs of training runs; one dose, one floor; the hypothesis was formed after the data of row 201 and the form of SP90 was fixed before it; the label form is a reference with label values and not a candidate.

This card has no map of arms to controls.
