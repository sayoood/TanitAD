# MDFIX_VERDICT -- FAIL (G1, G4)

## prereg s0 disclosures

* S0-a Forking paths: the MDFIX label was chosen AFTER the HSGATE8 eval139 read, on FIT-row arithmetic only (next/FDCAP_FRONTIER_FIT.json; next/DRAFT_NEXT_ARM_HSGATE_MDFIX.md). This is the SECOND eval139 read of the gate family; the multiplicity is paid in the bars (G1 read at 97.5 %).
* S0-b No MDFIX output on eval139 existed at sealing; the draft's FIT-side numbers are 5-fold episode CV on 7,147 FIT rows, one training seed, one fit, no interval.

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0, over EPISODES only (models fixed): answers 'another draw of episodes' only -- blind to training and inference variance; ONE refcv8 training seed; ONE gate fit; inference seeds 0 and 1 reported separately, never pooled

| bar | seed 0 | seed 1 | pass |
|---|---|---|---|
| G1 MD(MDF)-MD(H2CN) sep below @97.5 % | -0.0078 [-0.0191, +0.0000] (0.5258 vs 0.5336, n=639/79 ep) | -0.0125 [-0.0283, -0.0016] SEP (0.5196 vs 0.5321, n=639/79 ep) | False |
| (beside) G1 cell @95 % | -0.0078 [-0.0175, +0.0000] (0.5258 vs 0.5336, n=639/79 ep) | -0.0125 [-0.0260, -0.0030] SEP (0.5196 vs 0.5321, n=639/79 ep) | |
| G2 FD(MDF)-FD(H2CN) not sep above @95 % | +0.0016 [+0.0000, +0.0037] (0.0620 vs 0.0603, n=1856/135 ep) | +0.0011 [+0.0000, +0.0035] (0.0614 vs 0.0603, n=1856/135 ep) | True |
| G4a MD(MDF)-MD(MIX_r) | -0.0069 [-0.0164, +0.0008] (0.5258 vs 0.5327, n=639/79 ep) | -0.0112 [-0.0246, -0.0017] SEP (0.5196 vs 0.5308, n=639/79 ep) | False |
| G4b FD(MDF)-FD(MIX_r) | +0.0010 [-0.0006, +0.0031] (0.0620 vs 0.0610, n=1856/135 ep) | +0.0003 [-0.0010, +0.0026] (0.0614 vs 0.0612, n=1856/135 ep) | False |
| G3 SHIP-R8 OVERALL | PASS | | True |

F-rate r (S6 compliant): s0 0.0068 (22/3251), s1 0.0089 (29/3251); beside S2 compliant s0 0.0084, s1 0.0099; HSGATE8 S6 compliant s0 0.0877, s1 0.0883

| beside cell | seed 0 | seed 1 |
|---|---|---|
| MD_MDF_vs_HS | +0.0423 [+0.0165, +0.0731] SEP (0.5258 vs 0.4836, n=639/79 ep) | +0.0329 [+0.0104, +0.0576] SEP (0.5196 vs 0.4867, n=639/79 ep) |
| FD_MDF_vs_HS | -0.0119 [-0.0226, -0.0030] SEP (0.0620 vs 0.0738, n=1856/135 ep) | -0.0113 [-0.0220, -0.0028] SEP (0.0614 vs 0.0727, n=1856/135 ep) |
| MD_MDF_vs_FX | +0.1362 [+0.0767, +0.2006] SEP (0.5258 vs 0.3897, n=639/79 ep) | +0.1299 [+0.0706, +0.1948] SEP (0.5196 vs 0.3897, n=639/79 ep) |
| FD_MDF_vs_FX | -0.0884 [-0.1225, -0.0570] SEP (0.0620 vs 0.1503, n=1856/135 ep) | -0.0911 [-0.1251, -0.0606] SEP (0.0614 vs 0.1525, n=1856/135 ep) |
| MD_MDF_vs_RETIME | +0.0563 [-0.0184, +0.1382] (0.5258 vs 0.4695, n=639/79 ep) | +0.0485 [-0.0264, +0.1272] (0.5196 vs 0.4710, n=639/79 ep) |
| FD_MDF_vs_RETIME | -0.0749 [-0.1151, -0.0364] SEP (0.0620 vs 0.1369, n=1856/135 ep) | -0.0760 [-0.1180, -0.0370] SEP (0.0614 vs 0.1374, n=1856/135 ep) |
| MD_MIX_vs_H8 | -0.0010 [-0.0014, -0.0006] SEP (0.5327 vs 0.5336, n=639/79 ep) | -0.0013 [-0.0019, -0.0007] SEP (0.5308 vs 0.5321, n=639/79 ep) |
| FD_MIX_vs_H8 | +0.0006 [+0.0004, +0.0008] SEP (0.0610 vs 0.0603, n=1856/135 ep) | +0.0008 [+0.0005, +0.0011] SEP (0.0612 vs 0.0603, n=1856/135 ep) |

A4d fit-seed floor (beside every H2cn-relative cell): {"file": "C:\\Users\\Admin\\refcv8_pf\\ship_r8_sub_fs\\out_4d\\READ_4D.json", "sha256": "30ff1f9f5c7d59faa32d870c5dec09587d2655723394639e44c1e8a8dccdcd5e", "word": "REPLICATED", "R3_vs_H2CN_DEPLOY8_seed0": {"1": {"s0": {"ADE_S2_compliant": {"delta": 0.0001, "lo": -0.0023, "hi": 0.0027, "separated": false, "n_windows": 4159, "n_episodes": 138, "n_dropped_nonfinite": 0, "mean_new": 0.23883004143708042, "mean_ref": 0.23869576250182253, "separated_below_0": false, "separated_above_0": false}, "ADE_S6_compliant": {"delta": 0.0785, "lo": 0.0205, "hi": 0.1565, "separated": true, "n_windows": 3251, "n_episodes": 138, "n_dropped_nonfinite": 0, "mean_new": 2.2908213693741395, "mean_ref": 2.2123268279881523, "separated_below_0": false, "separated_above_0": true}, "ADE_S2_all_windows": {"delta": -0.0001, "lo": -0.0025, "hi": 0.0021, "separated": false, "n_windows": 4754, "n_episodes": 139, "n_dropped_nonfinite": 0, "mean_new": 0.24700122779623326, "mean_ref": 0.24714795198550635, "separated_below_0": false, "separated_above_0": false}, "ADE_S6_all_windows": {"delta": 0.0723, "lo": 0.0194, "hi": 0.1394, "separated": true, "n_windows": 3668, "n_episodes": 139, "n_dropped_nonfinite": 0, "mean_new": 2.3624566193982846, "mean_ref": 2.290190544462841, "separated_below_0": false, "separated_above_0": true}, "FD_S6_compliant": {"delta": -0.0011, "lo": -0.008, "hi": 0.0049, "separated": false, "n_windows": 1856, "n_episodes": 135, "n_dropped_nonfinite": 1812, "mean_new": 0.059267241379310345, "me

controls: {"K-FWD": true, "K-ALIGN": true, "K-PKG-GATE": true, "K-HS-LANDED": true, "K-DIAG": true, "K-ZERO": true, "K-MIX": true, "K-CI975": true}

NEXT: s5 G1 FAIL (G2 PASS): the FIT-side MD gain does not transfer at this F-rate; H2CN_DEPLOY8 stands; the refcv8 post-hoc progress line ENDS; the lever is refcv9's trained lon decision (PREREG_LONHEAD_C, in flight); no third relabel of this gate family is admissible on eval139

## disclosed deviations

* M1 s1a E_FIT is read literally as ALL FIT episodes (297); the FIT episodes holding a GT-cruise row (279) give the same projected FD upper bound to 4 dp (+0.0042) and are reported beside (next/MDFIX_POWER_FIT.json)
* M2 s1a SE_FIT = std (ddof 1) of the paired bootstrap deltas drawn with taniteval.ci's OWN episode draws (B 2000, seed 0) -- the same resampling as the cells
* M3 tau scan: candidates are every distinct held-out logit plus +inf (no gating); FIT MD / FD minimised on INTEGER counts (no float ties); K-TAU re-derives tau and both counts by an independent brute-force scan
* M4 REGISTERED_MDFIX.sha256 is a multi-line pin file with '#' comment lines (line 2 carries the PREREG sha); every pin is read by its EXACT path line (mdfix_common.registered_sha) -- ship_r8_sub_common.read_seal would read '#' and refuse; all four pins are re-hashed at forward start
* M5 the forward / verdict are GENERATED copies of the HSGATE phase-2 copies (make_mdfix_forward.py, make_mdfix_verdict.py; diffs beside); the HSGATE copies and the ship_r8_sub originals are untouched; the CLI switch keeps the name --hsgate and the forward JSON keeps the key 'hsgate' (read by the copies); ship_r8_sub_tac.py is used unchanged
* M6 G1's 97.5 % interval is taniteval.ci.paired_episode_cluster_bootstrap(alpha=0.025) on the SAME windows / draws as pair_read_r8.paired_cell; control K-CI975 asserts the same direct call at alpha 0.05 reproduces paired_cell's interval exactly and that the 97.5 % interval contains the 95 % one
* M7 the HSGATE8 beside rows read the LANDED HSGATE phase-2 package (out/arms/h2cn_hsgate/HSGATE8, sha256 recorded), not a rebuild
* M8 (carried from HSGATE D1 / D8 / D9) gate objective mean BCE + 0.5 lambda ||w||^2, bias unpenalised; clause (1) / (1b) for MDFIX8 are conjunctions (A4 clause 1 + the gate's controls; A4 (1b) AND A5 (1b)); H2CN_DEPLOY8 / FACTORED_DEPLOY8 are rebuilt and asserted bit-equal to the LANDED packages (K_LANDED)
