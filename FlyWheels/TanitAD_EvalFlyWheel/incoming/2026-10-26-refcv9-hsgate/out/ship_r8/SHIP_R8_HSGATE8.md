OVERALL: PASS -- SHIP-R8 on HSGATE8 (AMENDMENT HSG), tree step48852ema_pf

# SHIP_R8_HSGATE8

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), over EPISODES with the models FIXED: answers 'another draw of episodes' only -- not another training run (one training seed), not another inference run (seeds 0 and 1 reported separately)
one training seed; T1 open loop (not driving performance)

| clause | status | reading |
|---|---|---|
| (1) [PREREG_HSGATE] everything Amendment 4's clause 1 states (H2cn stage sha256 == registered; registration precedes the first eval-side output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the PREREG_HSGATE seal == its file (re-hashed now) AND the gate sha256 == REGISTERED_HSGATE.sha256 (re-hashed now) and its registration precedes the first eval-side output AND K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE AND the Amendment 5 input traces unchanged since the forward recorded them | **PASS** | Amendment 4's clause-1 conjunction holds on the --hsgate forward; PREREG_HSGATE seal == its file (re-hashed now) and == the forward's record; the gate sha256 == registered (re-hashed now); K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE all true; traces unchanged |
| (1b) [PREREG_HSGATE] FEED PARITY of BOTH halves: Amendment 4's (1b) (the registered FEED_PARITY of the battery rolls) AND Amendment 5's (1b) (the os-only and SELFSHARP rolls' own manifests perframe_full) | **PASS** | A4 (1b): PASS -- unchanged (same rolls): the registered FEED_PARITY files hash to the registered verdict's record | both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known || A5 (1b): PASS -- both seeds: the L1 os-only and the SELFSHARP rolls' own manifests read nav_feed perframe_full (roll pf ok, 4,754 windows) |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: HSGATE8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: HSGATE8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of HSGATE8 is separated worse than refcv7-50,400 [arm label substituted; the registered reason string is hard-coded; tac_file_arm == HSGATE8 enforced by the base] |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence (scored arm)

* clause (2) HSGATE8 - ha0_ext_pose: s0: -0.0444 [-0.0567, -0.0324] SEP | s1: -0.0439 [-0.0558, -0.0324] SEP
* clause (3) HSGATE8 - ha0_ext_pose: s0: -1.1932 [-1.4870, -0.9186] SEP | s1: -1.1924 [-1.4793, -0.9234] SEP
* FZR-NR per family {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True (separated worse at any seed: [])

| TAC cell (arm - refcv7-50,400; worse+separated = CI upper < 0) | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0038 [-0.0005, +0.0086] |
| S2|TAC_traj_lat_correct|s1 | +0.0036 [-0.0010, +0.0083] |
| S2|TAC_traj_lon_correct|s0 | +0.0308 [+0.0121, +0.0491] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0250 [+0.0069, +0.0430] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0105 [-0.0015, +0.0222] |
| S6|TAC_traj_lat_correct|s1 | +0.0102 [-0.0016, +0.0215] |
| S6|TAC_traj_lon_correct|s0 | +0.0471 [+0.0206, +0.0746] SEP |
| S6|TAC_traj_lon_correct|s1 | +0.0483 [+0.0212, +0.0756] SEP |

## beside (never scored)

* H2CN_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0471 [-0.0585, -0.0363] SEP | s1: -0.0464 [-0.0576, -0.0360] SEP | E3 s0: -1.1975 [-1.4942, -0.9168] SEP | s1: -1.1909 [-1.4800, -0.9163] SEP
* FACTORED_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0331 [-0.0453, -0.0214] SEP | s1: -0.0309 [-0.0429, -0.0199] SEP | E3 s0: -0.9883 [-1.2959, -0.6863] SEP | s1: -0.9816 [-1.2920, -0.6865] SEP
* RETIME_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0342 [-0.0474, -0.0216] SEP | s1: -0.0329 [-0.0452, -0.0206] SEP | E3 s0: -0.8682 [-1.1959, -0.5380] SEP | s1: -0.8676 [-1.1891, -0.5429] SEP
* running-maximum changed windows per (surface, seed): {'S2|s0': 143, 'S6|s0': 91, 'S2|s1': 143, 'S6|s1': 91}

NEXT (committed in the amendment): this verdict is bar G3 of PREREG_HSGATE (SHIP-R8 OVERALL PASS); the committed next step reads G1 + G2 + G3 + G4 together -> result/HSGATE_VERDICT.json (hsgate_bars.py)

## disclosed deviations (accepted by the Master Mind 2026-10-26 for P1-1..P1-6)

* D1 (confirmed) gate objective: mean_i BCEwithLogits + 0.5 * lambda * ||w||^2, bias unpenalised (the prereg fixed 'L2 on w' but not the scaling)
* D2 (confirmed) lambda CV: the 15 features are standardised on each CV TRAINING split (mean, std + 1e-9, l1_fit_score._std); the final fit standardises on all FIT rows
* D3 (confirmed) exact CV ties -> the larger lambda (no tie occurred: one configuration at the minimum, lambda 0.1)
* D4 (confirmed) a development smoke of hsgate_fit.py on the PARTIAL fresh trace (2,690 of 7,147 FIT rows) wrote ONLY to F:/refcv9_hsgate_scratch/smoke; the frozen gate is the full-trace fit
* D5 phase 1 read the LANDED FACTORED_DEPLOY8 package once (K-A5-PATH: the fit-side P_F code path reproduces its composed plan / staged positions bit for bit at seeds 0 and 1) -- booleans only, no feature / decision / rate / cell
* D6 (confirmed) G4 population: r = HSGATE8's F-rate on the S6 COMPLIANT windows of each inference seed (the windows the MD / FD cells use); the S2 F-rate is reported beside
* D7 the forward / verdict are COPIES of the 4c copies (ship_r8_hsgate_forward.py, ship_r8_hsgate_verdict.py; built by make_*.py, diffs beside); the originals are untouched; ship_r8_sub_tac.py is used unchanged
* D8 clause (1) / (1b) for HSGATE8 are conjunctions (Amendment 4's clause 1 + the gate's controls; A4 (1b) AND A5 (1b)) because HSGATE8 emits FACTORED_DEPLOY8's positions on gated windows
* D9 FACTORED_DEPLOY8 and H2CN_DEPLOY8 are RE-BUILT in the hsgate forward with their own registered code and asserted bit-equal to the LANDED packages (K_LANDED)
* D10 the FIRST --hsgate forward (07:46-07:55) WITHHELD on K_EVALFEAT ONLY: its 'extras file is the battery tree's dump_s{seed}' sub-check compared path STRINGS (str(Path) carries backslashes, the template slashes) and read False on a correct file; every other sub-check of K_EVALFEAT (key == K-LABSRC key, K-LABSRC pass, s0 extras sha256 == K-LABSRC's, all 4 joins PASS, all 4 next-window mutations RED by cross-check) and every other control read True.  Fixed to Path(...).resolve() equality (make_hsgate_forward.py), forward re-run unchanged otherwise; the withheld run's record is kept as out/forward/WITHHELD_H2CN_HSGATE_step48852ema_pf.FIRSTRUN.json

inputs sha256: {"forward": "0bfcb277f0630e699fd0db88a69f5f379b933db86bad8f680ef8e10939a05f8f", "tac_arm": "4a0b9ea71a21c344d8c5444a7f6ee05b4b03ba04bb0f8b3a827c6b84fff71bd4", "tac_retime_record": "8fd12e9cf75054d06014cdc2b56f718c601a2e7ca5c3f0d733bfeae7dfa89b8b", "trace_pins": null, "pr": "382faf77f5c3244df00dcc372c6a251909397dce88f78b9b4650f93d15e3101b", "pp": "a439adc3cf14e0422c68226cc4c8e308de0b78446dd0875e9e78808e50480038", "tf": "30138287283841149d7df678d049d91f209a89dac4f9e109397f9b2972a76f2d", "parity_s0": "26800b57847150cfc8126518fe90c09c1065c3192f6acc1f0f100de63522ca70", "parity_s1": "89a5a9f51085676457b440fd1c1e2373f42b6eee8162d6d71685eb91696efe8f", "deploy_sha": "44cf194c7fe91174e9c99871334035da7b85a25cb1cadd2b1c37b8db1ce27893", "frozen": "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80", "provenance": "a1645f88dc01c5e316579e2e22310e62a2426fc9159a8adde7120d424a5d7269", "gate": "7478b23366da249ab75471363b2e245c1fc0f8eebcc549bc828b5b6a5f64e5f2", "registered": "3e77cd12e2e584e8be5da469da9e5bb4e51e4eda02ec048c7d3470eb7310b7ae", "fd8": "939d3da1257a59ad935781d33bf5810e6ee213502aed40e09c7a885271c64aec", "tac_registered_RETIME_DEPLOY8": "32f17fb09ebffd35756845e938b48308ab7d93eded1e7292e5be28be59265ceb"}
