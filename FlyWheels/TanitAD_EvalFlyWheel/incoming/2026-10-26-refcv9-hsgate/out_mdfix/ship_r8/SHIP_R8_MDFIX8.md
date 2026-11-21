OVERALL: PASS -- SHIP-R8 on MDFIX8 (AMENDMENT HSG), tree step48852ema_pf

# SHIP_R8_MDFIX8

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), over EPISODES with the models FIXED: answers 'another draw of episodes' only -- not another training run (one training seed), not another inference run (seeds 0 and 1 reported separately)
one training seed; T1 open loop (not driving performance)

| clause | status | reading |
|---|---|---|
| (1) [PREREG_HSGATE] everything Amendment 4's clause 1 states (H2cn stage sha256 == registered; registration precedes the first eval-side output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the PREREG_HSGATE seal == its file (re-hashed now) AND the gate sha256 == its line in REGISTERED_MDFIX.sha256 (re-hashed now) AND the PREREG_HSGATE_MDFIX seal == its file and its registration precedes the first eval-side output AND K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE AND the Amendment 5 input traces unchanged since the forward recorded them | **PASS** | Amendment 4's clause-1 conjunction holds on the --hsgate forward; PREREG_HSGATE seal == its file (re-hashed now) and == the forward's record; the gate sha256 == registered (re-hashed now); K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE all true; traces unchanged |
| (1b) [PREREG_HSGATE] FEED PARITY of BOTH halves: Amendment 4's (1b) (the registered FEED_PARITY of the battery rolls) AND Amendment 5's (1b) (the os-only and SELFSHARP rolls' own manifests perframe_full) | **PASS** | A4 (1b): PASS -- unchanged (same rolls): the registered FEED_PARITY files hash to the registered verdict's record | both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known || A5 (1b): PASS -- both seeds: the L1 os-only and the SELFSHARP rolls' own manifests read nav_feed perframe_full (roll pf ok, 4,754 windows) |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: MDFIX8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: MDFIX8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of MDFIX8 is separated worse than refcv7-50,400 [arm label substituted; the registered reason string is hard-coded; tac_file_arm == MDFIX8 enforced by the base] |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence (scored arm)

* clause (2) MDFIX8 - ha0_ext_pose: s0: -0.0473 [-0.0589, -0.0366] SEP | s1: -0.0468 [-0.0578, -0.0363] SEP
* clause (3) MDFIX8 - ha0_ext_pose: s0: -1.2044 [-1.4985, -0.9322] SEP | s1: -1.1989 [-1.4854, -0.9314] SEP
* FZR-NR per family {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True (separated worse at any seed: [])

| TAC cell (arm - refcv7-50,400; worse+separated = CI upper < 0) | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0036 [-0.0009, +0.0084] |
| S2|TAC_traj_lat_correct|s1 | +0.0036 [-0.0010, +0.0083] |
| S2|TAC_traj_lon_correct|s0 | +0.0315 [+0.0127, +0.0499] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0252 [+0.0072, +0.0429] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0095 [-0.0022, +0.0212] |
| S6|TAC_traj_lat_correct|s1 | +0.0111 [+0.0000, +0.0220] |
| S6|TAC_traj_lon_correct|s0 | +0.0421 [+0.0166, +0.0684] SEP |
| S6|TAC_traj_lon_correct|s1 | +0.0446 [+0.0166, +0.0714] SEP |

## beside (never scored)

* H2CN_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0471 [-0.0585, -0.0363] SEP | s1: -0.0464 [-0.0576, -0.0360] SEP | E3 s0: -1.1975 [-1.4942, -0.9168] SEP | s1: -1.1909 [-1.4800, -0.9163] SEP
* FACTORED_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0331 [-0.0453, -0.0214] SEP | s1: -0.0309 [-0.0429, -0.0199] SEP | E3 s0: -0.9883 [-1.2959, -0.6863] SEP | s1: -0.9816 [-1.2920, -0.6865] SEP
* RETIME_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0342 [-0.0474, -0.0216] SEP | s1: -0.0329 [-0.0452, -0.0206] SEP | E3 s0: -0.8682 [-1.1959, -0.5380] SEP | s1: -0.8676 [-1.1891, -0.5429] SEP
* running-maximum changed windows per (surface, seed): {'S2|s0': 143, 'S6|s0': 91, 'S2|s1': 143, 'S6|s1': 91}

NEXT (committed in the amendment): this verdict is bar G3 of PREREG_HSGATE_MDFIX (SHIP-R8 OVERALL PASS); the committed next step reads G1 + G2 + G3 + G4 together -> result/MDFIX_VERDICT.json (mdfix_bars.py)

## disclosed deviations (accepted by the Master Mind 2026-10-26 for P1-1..P1-6)

* M1 s1a E_FIT is read literally as ALL FIT episodes (297); the FIT episodes holding a GT-cruise row (279) give the same projected FD upper bound to 4 dp (+0.0042) and are reported beside (next/MDFIX_POWER_FIT.json)
* M2 s1a SE_FIT = std (ddof 1) of the paired bootstrap deltas drawn with taniteval.ci's OWN episode draws (B 2000, seed 0) -- the same resampling as the cells
* M3 tau scan: candidates are every distinct held-out logit plus +inf (no gating); FIT MD / FD minimised on INTEGER counts (no float ties); K-TAU re-derives tau and both counts by an independent brute-force scan
* M4 REGISTERED_MDFIX.sha256 is a multi-line pin file with '#' comment lines (line 2 carries the PREREG sha); every pin is read by its EXACT path line (mdfix_common.registered_sha) -- ship_r8_sub_common.read_seal would read '#' and refuse; all four pins are re-hashed at forward start
* M5 the forward / verdict are GENERATED copies of the HSGATE phase-2 copies (make_mdfix_forward.py, make_mdfix_verdict.py; diffs beside); the HSGATE copies and the ship_r8_sub originals are untouched; the CLI switch keeps the name --hsgate and the forward JSON keeps the key 'hsgate' (read by the copies); ship_r8_sub_tac.py is used unchanged
* M6 G1's 97.5 % interval is taniteval.ci.paired_episode_cluster_bootstrap(alpha=0.025) on the SAME windows / draws as pair_read_r8.paired_cell; control K-CI975 asserts the same direct call at alpha 0.05 reproduces paired_cell's interval exactly and that the 97.5 % interval contains the 95 % one
* M7 the HSGATE8 beside rows read the LANDED HSGATE phase-2 package (out/arms/h2cn_hsgate/HSGATE8, sha256 recorded), not a rebuild
* M8 (carried from HSGATE D1 / D8 / D9) gate objective mean BCE + 0.5 lambda ||w||^2, bias unpenalised; clause (1) / (1b) for MDFIX8 are conjunctions (A4 clause 1 + the gate's controls; A4 (1b) AND A5 (1b)); H2CN_DEPLOY8 / FACTORED_DEPLOY8 are rebuilt and asserted bit-equal to the LANDED packages (K_LANDED)

inputs sha256: {"forward": "10d8998f40c18b5044c0bf9738ad861649d4278ac1a9d14b3b22fe38391cd93a", "tac_arm": "70af3876883b89bdbfb9f2186fca6aa02ed7364b5ebda9b8c4231084140be72b", "tac_retime_record": "c0e7f246908716a5f93a9586d4634261b9c61ab5c0f81851e799148340d43ab6", "trace_pins": null, "pr": "382faf77f5c3244df00dcc372c6a251909397dce88f78b9b4650f93d15e3101b", "pp": "a439adc3cf14e0422c68226cc4c8e308de0b78446dd0875e9e78808e50480038", "tf": "30138287283841149d7df678d049d91f209a89dac4f9e109397f9b2972a76f2d", "parity_s0": "26800b57847150cfc8126518fe90c09c1065c3192f6acc1f0f100de63522ca70", "parity_s1": "89a5a9f51085676457b440fd1c1e2373f42b6eee8162d6d71685eb91696efe8f", "deploy_sha": "44cf194c7fe91174e9c99871334035da7b85a25cb1cadd2b1c37b8db1ce27893", "frozen": "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80", "provenance": "a1645f88dc01c5e316579e2e22310e62a2426fc9159a8adde7120d424a5d7269", "gate": "7478b23366da249ab75471363b2e245c1fc0f8eebcc549bc828b5b6a5f64e5f2", "registered": "3e77cd12e2e584e8be5da469da9e5bb4e51e4eda02ec048c7d3470eb7310b7ae", "fd8": "939d3da1257a59ad935781d33bf5810e6ee213502aed40e09c7a885271c64aec", "tac_registered_RETIME_DEPLOY8": "32f17fb09ebffd35756845e938b48308ab7d93eded1e7292e5be28be59265ceb"}
