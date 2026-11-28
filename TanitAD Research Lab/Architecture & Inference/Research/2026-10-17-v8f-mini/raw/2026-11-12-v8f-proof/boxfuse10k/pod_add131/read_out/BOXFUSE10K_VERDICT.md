# BOXFUSE10K verdict (card PREREG_V8F_PROOF_ADD131)

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m); K0: readout K, no pre-roll; CONV: pinned readout, no pre-roll. A kinematic rollout on recorded frames, on the model's own actions; not driving performance.

Controls: j0 to j9 all pass (controls file sha256 152f79d42fd7084f). This card has no map of arms to controls.

Estimator: two-level bootstrap over the two model seeds and the evaluation episodes, 2,000 draws, on the same windows for both arms of a pair. The interval answers what another draw of episodes and of these two model seeds would say, not what another training run, another detector or another fit of the locator would say.

## Names

- BOXFUSE10K-CONV-JEc-CONTACT-NOT-SEPARATED
- BOXFUSE10K-CONV-JEc-FALLING-BEHIND-NOT-SEPARATED
- BOXFUSE10K-CONV-JEc-JOINED-GAP-NOT-SEPARATED
- BOXFUSE10K-CONV-JEc-UNSAFE-NOT-SEPARATED
- BOXFUSE10K-K0-JA90-CONTACT-NOT-SEPARATED
- BOXFUSE10K-K0-JA90-FALLING-BEHIND-NOT-SEPARATED
- BOXFUSE10K-K0-JA90-JOINED-GAP-NOT-SEPARATED
- BOXFUSE10K-K0-JA90-UNSAFE-NOT-SEPARATED
- BOXFUSE10K-K0-JEc-CONTACT-NOT-SEPARATED
- BOXFUSE10K-K0-JEc-FALLING-BEHIND-NOT-SEPARATED
- BOXFUSE10K-K0-JEc-JOINED-GAP-NOT-SEPARATED
- BOXFUSE10K-K0-JEc-UNSAFE-NOT-SEPARATED

## Setting K0 (no pre-roll)

Windows with a lead 642, episodes with a lead 56.

### Levels (contact C, unsafe U, falling behind F)

- JEc pooled C 0.0008 U 0.0480 F 0.1597; per model seed {"C": [0.0, 0.001557632398753894], "F": [0.14174454828660435, 0.17757009345794392], "U": [0.0498220640569395, 0.046263345195729534]}
- JA90 pooled C 0.0000 U 0.0374 F 0.1667; per model seed {"C": [0.0, 0.0], "F": [0.1526479750778816, 0.1806853582554517], "U": [0.03558718861209965, 0.03914590747330961]}
- GEc pooled C 0.0000 U 0.0498 F 0.1651; per model seed {"C": [0.0, 0.0], "F": [0.14797507788161993, 0.1822429906542056], "U": [0.051601423487544484, 0.04804270462633452]}
- A90 pooled C 0.0000 U 0.0374 F 0.1729; per model seed {"C": [0.0, 0.0], "F": [0.1573208722741433, 0.18847352024922118], "U": [0.03914590747330961, 0.03558718861209965]}
- RLc pooled C 0.0000 U 0.0365 F 0.1550; per model seed {"C": [0.0, 0.0], "F": [0.14018691588785046, 0.16978193146417445], "U": [0.03914590747330961, 0.033807829181494664]}
- RR pooled C 0.0296 U 0.0738 F 0.1682; per model seed {"C": [0.03115264797507788, 0.028037383177570093], "F": [0.17445482866043613, 0.16199376947040497], "U": [0.07473309608540925, 0.07295373665480427]}

### Pairs (first minus second, with intervals)

- JEc-VS-GEc: UNSAFE {"interval": [-0.00604977302979522, 0.00262248677930273], "per_seed": [-0.0017793594306049821, -0.0017793594306049821], "pooled": -0.0017793594306049821}; CONTACT {"interval": [0.0, 0.003046506778929948], "per_seed": [0.0, 0.001557632398753894], "pooled": 0.000778816199376947}; FALLING-BEHIND {"interval": [-0.013343458784678096, 0.0014574455657729321], "per_seed": [-0.006230529595015576, -0.004672897196261682], "pooled": -0.005451713395638629}
- JA90-VS-A90: UNSAFE {"interval": [-0.009481020648950797, 0.008403404352242043], "per_seed": [-0.0035587188612099642, 0.0035587188612099642], "pooled": 0.0}; CONTACT {"interval": [0.0, 0.0], "per_seed": [0.0, 0.0], "pooled": 0.0}; FALLING-BEHIND {"interval": [-0.013525190104108568, 0.0007504133074619888], "per_seed": [-0.004672897196261682, -0.00778816199376947], "pooled": -0.006230529595015576}
- JEc-VS-RLc: UNSAFE {"interval": [0.000889233172373542, 0.0240825497310338], "per_seed": [0.010676156583629894, 0.012455516014234875], "pooled": 0.011565836298932385}; CONTACT {"interval": [0.0, 0.003046506778929948], "per_seed": [0.0, 0.001557632398753894], "pooled": 0.000778816199376947}; FALLING-BEHIND {"interval": [-0.00578574821507317, 0.014275434619702652], "per_seed": [0.001557632398753894, 0.00778816199376947], "pooled": 0.004672897196261682}
- JA90-VS-RLc: UNSAFE {"interval": [-0.011989837752847947, 0.016336310205228004], "per_seed": [-0.0035587188612099642, 0.005338078291814947], "pooled": 0.0008896797153024913}; CONTACT {"interval": [0.0, 0.0], "per_seed": [0.0, 0.0], "pooled": 0.0}; FALLING-BEHIND {"interval": [0.004106512507192908, 0.01975732019239118], "per_seed": [0.012461059190031152, 0.010903426791277258], "pooled": 0.011682242990654205}
- GEc-VS-RLc: UNSAFE {"interval": [0.0035931867477239157, 0.02497756791980356], "per_seed": [0.012455516014234875, 0.014234875444839857], "pooled": 0.013345195729537365}; CONTACT {"interval": [0.0, 0.0], "per_seed": [0.0, 0.0], "pooled": 0.0}; FALLING-BEHIND {"interval": [0.003026300219688494, 0.017977873068402154], "per_seed": [0.00778816199376947, 0.012461059190031152], "pooled": 0.010124610591900311}

### Families (longitudinal, lateral, tactical: engine reads of each pair; strategic: grid at T1-kin, position error at 30, 50 and 80 m)

- JEc-VS-GEc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- JA90-VS-A90: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- JEc-VS-RLc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- JA90-VS-RLc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "MIXED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- GEc-VS-RLc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "MIXED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED

## Setting CONV (no pre-roll)

Windows with a lead 642, episodes with a lead 56.

### Levels (contact C, unsafe U, falling behind F)

- JEc pooled C 0.0016 U 0.0489 F 0.2298; per model seed {"C": [0.001557632398753894, 0.001557632398753894], "F": [0.19314641744548286, 0.26635514018691586], "U": [0.060498220640569395, 0.037366548042704624]}
- GEc pooled C 0.0016 U 0.0569 F 0.2329; per model seed {"C": [0.001557632398753894, 0.001557632398753894], "F": [0.19314641744548286, 0.27258566978193144], "U": [0.07473309608540925, 0.03914590747330961]}
- RLc pooled C 0.0016 U 0.0320 F 0.2243; per model seed {"C": [0.0, 0.003115264797507788], "F": [0.19626168224299065, 0.2523364485981308], "U": [0.042704626334519574, 0.021352313167259787]}
- RR pooled C 0.0280 U 0.0783 F 0.2593; per model seed {"C": [0.03582554517133956, 0.020249221183800622], "F": [0.2383177570093458, 0.2803738317757009], "U": [0.08540925266903915, 0.0711743772241993]}

### Pairs (first minus second, with intervals)

- JEc-VS-GEc: UNSAFE {"interval": [-0.020856094129123676, 0.004083866977947941], "per_seed": [-0.014234875444839857, -0.0017793594306049821], "pooled": -0.00800711743772242}; CONTACT {"interval": [0.0, 0.0], "per_seed": [0.0, 0.0], "pooled": 0.0}; FALLING-BEHIND {"interval": [-0.0151473728364863, 0.008322799687157843], "per_seed": [0.0, -0.006230529595015576], "pooled": -0.003115264797507788}
- JEc-VS-RLc: UNSAFE {"interval": [0.004048967427414938, 0.03138760869252301], "per_seed": [0.017793594306049824, 0.01601423487544484], "pooled": 0.016903914590747332}; CONTACT {"interval": [-0.003154797191714759, 0.002877905467833524], "per_seed": [0.001557632398753894, -0.001557632398753894], "pooled": 0.0}; FALLING-BEHIND {"interval": [-0.012602986715579698, 0.022786726278727132], "per_seed": [-0.003115264797507788, 0.014018691588785047], "pooled": 0.005451713395638629}
- GEc-VS-RLc: UNSAFE {"interval": [0.007185366529885434, 0.046379534512878164], "per_seed": [0.03202846975088968, 0.017793594306049824], "pooled": 0.024911032028469754}; CONTACT {"interval": [-0.003154797191714759, 0.002877905467833524], "per_seed": [0.001557632398753894, -0.001557632398753894], "pooled": 0.0}; FALLING-BEHIND {"interval": [-0.015050302117484421, 0.031114263196778937], "per_seed": [-0.003115264797507788, 0.020249221183800622], "pooled": 0.008566978193146417}

### Families (longitudinal, lateral, tactical: engine reads of each pair; strategic: grid at T1-kin, position error at 30, 50 and 80 m)

- JEc-VS-GEc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- JEc-VS-RLc: {"LATERAL": "MIXED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED
- GEc-VS-RLc: {"LATERAL": "NOT-SEPARATED", "LONGITUDINAL": "NOT-SEPARATED", "TACTICAL": "NOT-SEPARATED"}; strategic NOT-SEPARATED

## Limits (card section 8)

Two model seeds; one detector with no training on our data; one fit of the locator in the rollouts (row 199 read three at T0); T1-kin reads recorded frames at the rollout's steps, there is no re-perception of what the rolled-out car would see; the rows the rule treats as a lead are those of the base file, so the presence input is unchanged by construction.

The T0 reading of the card (BOXFUSE10K-T0 names) is not made by this reader.

This card has no map of arms to controls.
