# PESS A1 read (LCB1ND)

- Card A1 sha256 `e2a4452c1b81858bd39a8f09b6fdc16c91dfcc117e3feaa90e9fcb809c5173f1`; base card `185de4067ede99ff9aca0c0f559e5a62f0989872aef07b8b7c4bd8657d42f380`
- Members (8): C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM
- Checks: {"ENS9_sha_equals_PESS_read": true, "LCB1_sha_equals_PESS_read": true, "RED_LCB1ND_plus_TTC_equals_LCB1_bits": true, "LCB1ND_differs_from_LCB1_picks": 1449}

| arm | PDMS | minus P6 | EP x100 minus P6 | zero-pick share | P2 vs ENS9 | M vs ENS9 | guards | adverse |
|---|---|---|---|---|---|---|---|---|
| ENS9 | 91.414 [90.609, 92.221] | +0.2174 [+0.0337, +0.4084] | -0.0821 [-0.2623, +0.1080] | 0.02956 | - | - | {'G1': True, 'G2': True, 'G3': True, 'G4': True} | [] |
| LCB1 | 91.493 [90.688, 92.294] | +0.2968 [+0.1140, +0.4918] | -0.3773 [-0.5572, -0.1797] | 0.02791 | +0.0794 [-0.0212, +0.1958] False | -0.0016 [-0.0029, -0.0006] True | {'G1': True, 'G2': True, 'G3': True, 'G4': True} | ['EP'] |
| LCB1ND | 91.438 [90.613, 92.253] | +0.2410 [+0.0458, +0.4463] | -0.2460 [-0.4181, -0.0618] | 0.02857 | +0.0236 [-0.0725, +0.1234] False | -0.0010 [-0.0021, +0.0000] False | {'G1': True, 'G2': True, 'G3': True, 'G4': True} | ['EP'] |

- P1 LCB1ND vs P6: +0.2410 [+0.0458, +0.4463], floor 0.4971, clears False
- E: EP x100 LCB1ND - LCB1 +0.1312 [+0.0605, +0.1991], clears True
- **Outcome: A1-2 EP RECOVERED, BELOW BAR** (matches ['A1-2']); goal status BELOW GOAL
- K ledger: {"K": 34, "keys": {"ENS9": "ENS9:0c2a575f6ef884fb51e42ebcb0e82872dcfc568fb8400e0146ded7ca3acb93ff:E3", "LCB1": "LCB1:ad04543143ac494e58b659ac7681bcca8cc40d22965aabacca644b761f991dd6:E3", "LCB1ND": "LCB1ND:fb05ee70e023c1df636f26bd399286547b50e3eaf76f0a0b1666a04670410f96:E3"}, "this_read_added": {"ENS9": false, "LCB1": false, "LCB1ND": false}}
- STRATEGIC: PAUSED (PI 2026-09-19, PI_DECISION_QUEUE item 25); not a pass, not omitted.
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927 (another draw of logs only; fixed members, deterministic inference). Tier: T1-class open loop, full navtest E3.
