OVERALL: PASS -- SHIP-R8 on FACTORED_DEPLOY8 (AMENDMENT 5), tree step48852ema_pf

# SHIP_R8_FACTORED_DEPLOY8

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), over EPISODES with the models FIXED: answers 'another draw of episodes' only -- not another training run (one training seed), not another inference run (seeds 0 and 1 reported separately)
one training seed; T1 open loop (not driving performance)

| clause | status | reading |
|---|---|---|
| (1) [AMENDMENT 5] the registered table's sha256 checks (unchanged) AND the identity controls (P_ss := P_os == OSONLY bit for bit; RETIME_DEPLOY8 == registered) AND both input traces' sha256 + final + 4,754 windows + hook counters | **PASS** | registered table checks PASS; P_ss := P_os reproduces OSONLY bit for bit; RETIME_DEPLOY8 == registered; both traces final, 4,754 windows, all forced, none unarmed, batch 1, sha256 unchanged |
| (1b) [AMENDMENT 5] FEED PARITY: the os-only rolls use the per-frame feed (perframe_full), asserted from the rolls' own manifests; not shown -> INCOMPLETE | **PASS** | both seeds: the L1 os-only and the SELFSHARP rolls' own manifests read nav_feed perframe_full (roll pf ok, 4,754 windows) |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: FACTORED_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: FACTORED_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of FACTORED_DEPLOY8 is separated worse than refcv7-50,400 [arm label substituted; the registered reason string is hard-coded; tac_file_arm == FACTORED_DEPLOY8 enforced by the base] |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence (scored arm)

* clause (2) FACTORED_DEPLOY8 - ha0_ext_pose: s0: -0.0331 [-0.0453, -0.0214] SEP | s1: -0.0309 [-0.0429, -0.0199] SEP
* clause (3) FACTORED_DEPLOY8 - ha0_ext_pose: s0: -0.9883 [-1.2959, -0.6863] SEP | s1: -0.9816 [-1.2920, -0.6865] SEP
* FZR-NR per family {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True (separated worse at any seed: [])

| TAC cell (arm - refcv7-50,400; worse+separated = CI upper < 0) | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0031 [-0.0014, +0.0079] |
| S2|TAC_traj_lat_correct|s1 | +0.0038 [-0.0005, +0.0083] |
| S2|TAC_traj_lon_correct|s0 | +0.0257 [+0.0123, +0.0400] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0197 [+0.0056, +0.0341] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0065 [-0.0049, +0.0180] |
| S6|TAC_traj_lat_correct|s1 | +0.0080 [-0.0040, +0.0198] |
| S6|TAC_traj_lon_correct|s0 | +0.0188 [-0.0062, +0.0438] |
| S6|TAC_traj_lon_correct|s1 | +0.0194 [-0.0045, +0.0418] |

## beside (never scored)

* OSONLY_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0337 [-0.0467, -0.0208] SEP | s1: -0.0329 [-0.0452, -0.0206] SEP | E3 s0: -0.8793 [-1.2077, -0.5543] SEP | s1: -0.8676 [-1.1891, -0.5429] SEP
* RETIME_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0342 [-0.0474, -0.0216] SEP | s1: -0.0329 [-0.0452, -0.0206] SEP | E3 s0: -0.8682 [-1.1959, -0.5380] SEP | s1: -0.8676 [-1.1891, -0.5429] SEP

NEXT (committed in the amendment): PASS -> the factored decode is a refcv8 deliverable candidate; build the in-model implementation (two planner passes on one trunk forward) and its bit-equal control before anything ships; NavSim only with a separate go.

inputs sha256: {"forward": "b35f694ba0cf694aefe98fd044a803d1afac97572f000dcb51154b337a89b83d", "tac_arm": "2a2790fe4e74475847ada824d5712fa4b52a255db472dfa19d189279c3122de3", "tac_retime_record": "17ce618c528bba6d46d86559400912ee1f63945d3c2dcac3edb32966883a4158", "trace_pins": null, "pr": "382faf77f5c3244df00dcc372c6a251909397dce88f78b9b4650f93d15e3101b", "pp": "a439adc3cf14e0422c68226cc4c8e308de0b78446dd0875e9e78808e50480038", "tf": "30138287283841149d7df678d049d91f209a89dac4f9e109397f9b2972a76f2d", "parity_s0": "26800b57847150cfc8126518fe90c09c1065c3192f6acc1f0f100de63522ca70", "parity_s1": "89a5a9f51085676457b440fd1c1e2373f42b6eee8162d6d71685eb91696efe8f", "deploy_sha": "44cf194c7fe91174e9c99871334035da7b85a25cb1cadd2b1c37b8db1ce27893", "frozen": "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80", "provenance": "a1645f88dc01c5e316579e2e22310e62a2426fc9159a8adde7120d424a5d7269", "gate": "7478b23366da249ab75471363b2e245c1fc0f8eebcc549bc828b5b6a5f64e5f2", "registered": "3e77cd12e2e584e8be5da469da9e5bb4e51e4eda02ec048c7d3470eb7310b7ae", "fd8": "939d3da1257a59ad935781d33bf5810e6ee213502aed40e09c7a885271c64aec", "tac_registered_RETIME_DEPLOY8": "32f17fb09ebffd35756845e938b48308ab7d93eded1e7292e5be28be59265ceb"}
