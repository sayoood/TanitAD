OVERALL: PASS -- SHIP-R8 on H2CNL_DEPLOY8 (AMENDMENT 4c), tree step48852ema_pf

# SHIP_R8_H2CNL_DEPLOY8

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), over EPISODES with the models FIXED: answers 'another draw of episodes' only -- not another training run (one training seed), not another inference run (seeds 0 and 1 reported separately)
one training seed; T1 open loop (not driving performance)

| clause | status | reading |
|---|---|---|
| (1) [AMENDMENT 4c] everything Amendment 4's clause 1 states (H2cn stage sha256 == registered; registration precedes the first eval-side output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the Amendment 4c seal == its file (re-hashed now) AND the H2cn-L stage sha256 == REGISTERED_H2CNL_STAGE.sha256 (re-hashed now) and its registration precedes the first eval-side output AND the freeze controls (K-H2CN-REPRO, K-JOIN, K-FIT, K-RELOAD, K-DETERM, K-DISJ, K-MUT) AND K-JOIN on the eval rows | **PASS** | Amendment 4's clause-1 conjunction holds on the --h2cnl forward; the Amendment 4c seal == its file (re-hashed now) and == the forward's record; the H2CNL_DEPLOY8 stage sha256 == registered (re-hashed now); K_REG_H2CNL, K_FREEZE_H2CNL, K_JOIN_EVAL all true |
| (1b) FEED PARITY: on the tree's own roll artifacts, both seeds, every window: the fed token equals the v9 per-frame token, `nav_cmd_clip` equals the per-clip token, and `r8_nav` is non-zero exactly where `r8_nav_known`; fails or cannot be shown -> SHIP-R8 INCOMPLETE | **PASS** | unchanged (same rolls): the registered FEED_PARITY files hash to the registered verdict's record | both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: H2CNL_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: H2CNL_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of H2CNL_DEPLOY8 is separated worse than refcv7-50,400 [arm label substituted; the registered reason string is hard-coded; tac_file_arm == H2CNL_DEPLOY8 enforced by the base] |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence (scored arm)

* clause (2) H2CNL_DEPLOY8 - ha0_ext_pose: s0: -0.0504 [-0.0620, -0.0392] SEP | s1: -0.0496 [-0.0610, -0.0387] SEP
* clause (3) H2CNL_DEPLOY8 - ha0_ext_pose: s0: -1.1994 [-1.4940, -0.9164] SEP | s1: -1.1935 [-1.4835, -0.9155] SEP
* FZR-NR per family {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True (separated worse at any seed: [])

| TAC cell (arm - refcv7-50,400; worse+separated = CI upper < 0) | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0036 [-0.0009, +0.0084] |
| S2|TAC_traj_lat_correct|s1 | +0.0038 [-0.0005, +0.0082] |
| S2|TAC_traj_lon_correct|s0 | +0.0428 [+0.0278, +0.0594] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0363 [+0.0229, +0.0522] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0086 [-0.0022, +0.0194] |
| S6|TAC_traj_lat_correct|s1 | +0.0108 [-0.0003, +0.0218] |
| S6|TAC_traj_lon_correct|s0 | +0.0381 [+0.0097, +0.0672] SEP |
| S6|TAC_traj_lon_correct|s1 | +0.0397 [+0.0112, +0.0692] SEP |

## beside (never scored)

* H2CN_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0471 [-0.0585, -0.0363] SEP | s1: -0.0464 [-0.0576, -0.0360] SEP | E3 s0: -1.1975 [-1.4942, -0.9168] SEP | s1: -1.1909 [-1.4800, -0.9163] SEP
* H2CNL_MUT8: (2) PASS; (3) PASS; (4) PASS; (5) PASS; (1) PASS | E1 s0: -0.0482 [-0.0597, -0.0372] SEP | s1: -0.0474 [-0.0585, -0.0369] SEP | E3 s0: -1.1453 [-1.4443, -0.8663] SEP | s1: -1.1387 [-1.4282, -0.8647] SEP
* H2CN_NEAR: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0471 [-0.0585, -0.0363] SEP | s1: -0.0464 [-0.0576, -0.0360] SEP | E3 s0: -0.8757 [-1.2029, -0.5476] SEP | s1: -0.8749 [-1.1950, -0.5514] SEP
* H2CN_PERSURFACE: (2) PASS; (3) PASS; (4) PASS; (5) INCOMPLETE | E1 s0: -0.0473 [-0.0588, -0.0365] SEP | s1: -0.0466 [-0.0577, -0.0362] SEP | E3 s0: -1.1974 [-1.4936, -0.9172] SEP | s1: -1.1907 [-1.4793, -0.9159] SEP
* RETIME_DEPLOY8: (2) PASS; (3) PASS; (4) PASS; (5) PASS | E1 s0: -0.0342 [-0.0474, -0.0216] SEP | s1: -0.0329 [-0.0452, -0.0206] SEP | E3 s0: -0.8682 [-1.1959, -0.5380] SEP | s1: -0.8676 [-1.1891, -0.5429] SEP
* running-maximum changed windows per (surface, seed): {'S2|s0': 143, 'S6|s0': 91, 'S2|s1': 143, 'S6|s1': 91}

NEXT (committed in the amendment): this verdict is bar L3 of Amendment 4c (SHIP-R8 OVERALL PASS); the committed next step reads L1 + L2 + L3 together -> BARS_4C.json (bars_4c.py)

## disclosed deviations (accepted by the Master Mind 2026-10-26 for P1-1..P1-6)

* P1-1 mutation construction (not specified by the amendment): the 6 LON6-softmax columns of the FIT rows are permuted by a seeded (seed 0) cross-episode derangement (h2cnl_freeze.cross_episode_derangement): a bijection, every row receives a DIFFERENT episode's columns, column marginals preserved exactly
* P1-2 K-FIT for the L / MUT stages has no earlier PASSED fit to equal (Amendment 4 compared against H2CN_RESULT.json): replaced by literals -- n_fresh == the registered fit's, n_params == 68100 / 68358 (H2cn's + 6 x 128), wd in the grid, 1 <= epochs <= 200
* P1-3 K-H2CN-REPRO for the mutation stage is shown by base-array digest equality (X[:, :10], Z, H, eid == the arrays whose 6-column-dropped fit reproduced the registered H2cn stage bit-for-bit), not by a third identical refit
* P1-4 h2cnl_freeze.py carries a DEV-ONLY --smoke-dir mode (3 epochs, outputs outside the package); used once, outputs only in the agent scratch dir
* P1-5 README_SHIP_R8_SUB.md and ship_r8_sub_common.py were NOT edited (both pinned by MANIFEST_4B)
* P1-6 h2cnl_freeze.DIFF.txt is large because of the new controls; the fit change itself is one call (fit_surface: X := concat[X, softmax(lon6)])
* P2-7 the forward / verdict / reproduction are COPIES (ship_r8_4c_forward.py, ship_r8_4c_verdict.py, read_4c.py; diffs beside) because the originals are pinned by MANIFEST_4B; ship_r8_sub_tac.py is used unchanged
* P2-8 the eval-side K-JOIN adds two cross-checks beyond the amendment (extras v0 == ctx v0; extras nav_true.traj == plan8 bit for bit) and a mutation (join on the NEXT window) that must read RED by those cross-checks alone
* P2-9 RT.forward's transient work dir lives on F:/refcv9_4c_scratch (C: nearly full) and is removed at the end; every output is under out_4c/
* P2-10 read_4c.py normalises backslashes BEFORE the out/ token (read_4b did it last) and skips the npz meta_json `tool` field (the producer is the 4c copy)
* P2-11 the mutation stage H2CNL_MUT8 is fed the TRUE eval-side LON6 softmax (only its FIT rows were deranged)
* P2-12 L1 / L2 'both inference seeds' is read as a conjunction (seed 0 AND seed 1)
* P2-13 bars_4c.py's verdict WORDING was edited after its first run: the first BARS_4C.json read 'VOID ... fix the instrument', which misnamed an ATTRIBUTION-control failure (K-MUT-L1) as an instrument failure; the edit only separates the two cases and names the amendment's branches -- every cell, bar flag and control is the same mechanical computation (the first file was overwritten)

inputs sha256: {"forward": "3cff2381be643775f8d1c26c2f7ef9aad29a5561bfcdeffbe1c1b3652f51c74c", "tac_arm": "53605c833c0d4610b01b6ecab85396dc33956981d4e4f0daaae1ca38a5d9a932", "tac_retime_record": "6982ad7404210dd4795c37a7ec8e97dbf795f68eb633bee2768e40b6e0824cc3", "trace_pins": null, "pr": "382faf77f5c3244df00dcc372c6a251909397dce88f78b9b4650f93d15e3101b", "pp": "a439adc3cf14e0422c68226cc4c8e308de0b78446dd0875e9e78808e50480038", "tf": "30138287283841149d7df678d049d91f209a89dac4f9e109397f9b2972a76f2d", "parity_s0": "26800b57847150cfc8126518fe90c09c1065c3192f6acc1f0f100de63522ca70", "parity_s1": "89a5a9f51085676457b440fd1c1e2373f42b6eee8162d6d71685eb91696efe8f", "deploy_sha": "44cf194c7fe91174e9c99871334035da7b85a25cb1cadd2b1c37b8db1ce27893", "frozen": "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80", "provenance": "a1645f88dc01c5e316579e2e22310e62a2426fc9159a8adde7120d424a5d7269", "gate": "7478b23366da249ab75471363b2e245c1fc0f8eebcc549bc828b5b6a5f64e5f2", "registered": "3e77cd12e2e584e8be5da469da9e5bb4e51e4eda02ec048c7d3470eb7310b7ae", "fd8": "939d3da1257a59ad935781d33bf5810e6ee213502aed40e09c7a885271c64aec", "tac_registered_RETIME_DEPLOY8": "32f17fb09ebffd35756845e938b48308ab7d93eded1e7292e5be28be59265ceb"}
