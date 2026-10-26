OVERALL: PASS -- SHIP-R8 on H2CN_FS2_DEPLOY8 (AMENDMENT 4), tree step48852ema_pf

# SHIP_R8_H2CN_FS2_DEPLOY8

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), over EPISODES with the models FIXED: answers 'another draw of episodes' only -- not another training run (one training seed), not another inference run (seeds 0 and 1 reported separately)
one training seed; T1 open loop (not driving performance)

| clause | status | reading |
|---|---|---|
| (1) [AMENDMENT 4d] the replicate H2cn stage's sha256 == its REGISTERED_H2CN_FS{s}_STAGE.sha256 (re-hashed now); the registration precedes the first eval-side stage output; the fit used no eval139 / train4572 clip; the Amendment 4d seal == its file; the RETIME_DEPLOY8 identity holds (K-PERSURF descriptive: its reference is the seed-0 fit) | **PASS** | stage sha256 == its REGISTERED_H2CN_FS{s}_STAGE.sha256 (re-hashed now); registration precedes the first eval-side output; fresh-clip disjoint; the Amendment 4d seal == its file; RETIME_DEPLOY8 identity holds; B 2000, seeds 0 and 1 (K-PERSURF descriptive for a replicate) |
| (1b) FEED PARITY: on the tree's own roll artifacts, both seeds, every window: the fed token equals the v9 per-frame token, `nav_cmd_clip` equals the per-clip token, and `r8_nav` is non-zero exactly where `r8_nav_known`; fails or cannot be shown -> SHIP-R8 INCOMPLETE | **PASS** | unchanged (same rolls): the registered FEED_PARITY files hash to the registered verdict's record | both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: H2CN_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: H2CN_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of H2CN_DEPLOY8 is separated worse than refcv7-50,400 [arm label substituted; the registered reason string is hard-coded; tac_file_arm == H2CN_DEPLOY8 enforced by the base] |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence (scored arm)

* clause (2) H2CN_FS2_DEPLOY8 - ha0_ext_pose: s0: -0.0471 [-0.0593, -0.0353] SEP | s1: -0.0463 [-0.0583, -0.0349] SEP
* clause (3) H2CN_FS2_DEPLOY8 - ha0_ext_pose: s0: -1.1303 [-1.4331, -0.8401] SEP | s1: -1.1247 [-1.4238, -0.8398] SEP
* FZR-NR per family {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True (separated worse at any seed: [])

| TAC cell (arm - refcv7-50,400; worse+separated = CI upper < 0) | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0036 [-0.0009, +0.0082] |
| S2|TAC_traj_lat_correct|s1 | +0.0031 [-0.0010, +0.0074] |
| S2|TAC_traj_lon_correct|s0 | +0.0392 [+0.0247, +0.0554] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0327 [+0.0188, +0.0485] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0086 [-0.0030, +0.0201] |
| S6|TAC_traj_lat_correct|s1 | +0.0108 [-0.0000, +0.0219] |
| S6|TAC_traj_lon_correct|s0 | +0.0249 [-0.0060, +0.0542] |
| S6|TAC_traj_lon_correct|s1 | +0.0265 [-0.0041, +0.0554] |

## beside (never scored)

* H2CN_NEAR: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0471 [-0.0593, -0.0353] SEP | s1: -0.0463 [-0.0583, -0.0349] SEP | E3 s0: -0.8763 [-1.2037, -0.5477] SEP | s1: -0.8755 [-1.1948, -0.5519] SEP
* H2CN_PERSURFACE: (2) PASS; (3) PASS; (4) PASS; (5) INCOMPLETE | E1 s0: -0.0476 [-0.0598, -0.0359] SEP | s1: -0.0467 [-0.0588, -0.0353] SEP | E3 s0: -1.1239 [-1.4272, -0.8323] SEP | s1: -1.1182 [-1.4169, -0.8328] SEP
* RETIME_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0342 [-0.0474, -0.0216] SEP | s1: -0.0329 [-0.0452, -0.0206] SEP | E3 s0: -0.8682 [-1.1959, -0.5380] SEP | s1: -0.8676 [-1.1891, -0.5429] SEP
* running-maximum changed windows per (surface, seed): {'S2|s0': 199, 'S6|s0': 122, 'S2|s1': 199, 'S6|s1': 122}

NEXT (committed in the amendment): PASS -> H2cn ships as refcv8's stage (the PI's (a)); registry + GOALS_AND_CLAIMS updated; NavSim only with a separate go.

inputs sha256: {"forward": "e817eb86f1542eb307e10370bcf94b9991859052865ed8276b31d2704a3a310f", "tac_arm": "486dd92edc449778d1ec55ab1156d5e261e48f35260c5d2b783cb83f13aa3f45", "tac_retime_record": "9ddc802b23490d0b7cbf159a022938f9e1f2e9ad108c8b94600f688fa5904904", "trace_pins": null, "pr": "382faf77f5c3244df00dcc372c6a251909397dce88f78b9b4650f93d15e3101b", "pp": "a439adc3cf14e0422c68226cc4c8e308de0b78446dd0875e9e78808e50480038", "tf": "30138287283841149d7df678d049d91f209a89dac4f9e109397f9b2972a76f2d", "parity_s0": "26800b57847150cfc8126518fe90c09c1065c3192f6acc1f0f100de63522ca70", "parity_s1": "89a5a9f51085676457b440fd1c1e2373f42b6eee8162d6d71685eb91696efe8f", "deploy_sha": "44cf194c7fe91174e9c99871334035da7b85a25cb1cadd2b1c37b8db1ce27893", "frozen": "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80", "provenance": "a1645f88dc01c5e316579e2e22310e62a2426fc9159a8adde7120d424a5d7269", "gate": "7478b23366da249ab75471363b2e245c1fc0f8eebcc549bc828b5b6a5f64e5f2", "registered": "3e77cd12e2e584e8be5da469da9e5bb4e51e4eda02ec048c7d3470eb7310b7ae", "fd8": "939d3da1257a59ad935781d33bf5810e6ee213502aed40e09c7a885271c64aec", "tac_registered_RETIME_DEPLOY8": "32f17fb09ebffd35756845e938b48308ab7d93eded1e7292e5be28be59265ceb"}
