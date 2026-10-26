# HSGATE_VERDICT -- FAIL (G2)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap B 2000, bootstrap seed 0, over EPISODES only (models fixed): answers 'another draw of episodes' only; ONE refcv8 training seed; ONE gate fit; inference seeds 0 and 1 reported separately, never pooled

| bar | seed 0 | seed 1 | pass |
|---|---|---|---|
| G1 MD(HS)-MD(H2CN) sep below | -0.0501 [-0.0845, -0.0224] SEP (0.4836 vs 0.5336, n=639/79 ep) | -0.0454 [-0.0767, -0.0186] SEP (0.4867 vs 0.5321, n=639/79 ep) | True |
| G2 FD(HS)-FD(H2CN) not sep above | +0.0135 [+0.0047, +0.0243] SEP (0.0738 vs 0.0603, n=1856/135 ep) | +0.0124 [+0.0037, +0.0234] SEP (0.0727 vs 0.0603, n=1856/135 ep) | False |
| G4a MD(HS)-MD(MIX_r) | -0.0375 [-0.0688, -0.0105] SEP (0.4836 vs 0.5210, n=639/79 ep) | -0.0328 [-0.0617, -0.0078] SEP (0.4867 vs 0.5195, n=639/79 ep) | True |
| G4b FD(HS)-FD(MIX_r) | +0.0056 [-0.0027, +0.0155] (0.0738 vs 0.0682, n=1856/135 ep) | +0.0043 [-0.0040, +0.0140] (0.0727 vs 0.0685, n=1856/135 ep) | True |
| G3 SHIP-R8 OVERALL | PASS | | True |

F-rate (S6 compliant): s0 0.0877 (285/3251), s1 0.0883 (287/3251)

beside: S2 F-rate (compliant): s0 0.0875, s1 0.0880

controls: {"K-FWD": true, "K-ALIGN": true, "K-PKG-GATE": true, "K-DIAG": true, "K-ZERO": true, "K-MIX": true}

NEXT: operating-point trade: next = the same gate with its threshold chosen on FIT rows under an FD cap (FD_FIT(gate) <= FD_FIT(H2cn-OOF)), pre-registered first

## disclosed deviations

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
