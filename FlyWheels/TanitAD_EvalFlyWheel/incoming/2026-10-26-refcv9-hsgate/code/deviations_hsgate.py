"""deviations_hsgate.py -- the disclosed deviations / open choices of PREREG_HSGATE, CONFIRMED by the Master Mind in the PHASE 2 GO (2026-10-26); stated in
every phase-2 verdict (ship_r8_hsgate_verdict.py, hsgate_bars.py)."""
DEVIATIONS = [
    "D1 (confirmed) gate objective: mean_i BCEwithLogits + 0.5 * lambda * ||w||^2, bias unpenalised (the prereg fixed 'L2 on w' but not the scaling)",
    "D2 (confirmed) lambda CV: the 15 features are standardised on each CV TRAINING split (mean, std + 1e-9, l1_fit_score._std); the final fit standardises on all FIT rows",
    "D3 (confirmed) exact CV ties -> the larger lambda (no tie occurred: one configuration at the minimum, lambda 0.1)",
    "D4 (confirmed) a development smoke of hsgate_fit.py on the PARTIAL fresh trace (2,690 of 7,147 FIT rows) wrote ONLY to F:/refcv9_hsgate_scratch/smoke; "
    "the frozen gate is the full-trace fit",
    "D5 phase 1 read the LANDED FACTORED_DEPLOY8 package once (K-A5-PATH: the fit-side P_F code path reproduces its composed plan / staged positions "
    "bit for bit at seeds 0 and 1) -- booleans only, no feature / decision / rate / cell",
    "D6 (confirmed) G4 population: r = HSGATE8's F-rate on the S6 COMPLIANT windows of each inference seed (the windows the MD / FD cells use); the S2 F-rate is "
    "reported beside",
    "D7 the forward / verdict are COPIES of the 4c copies (ship_r8_hsgate_forward.py, ship_r8_hsgate_verdict.py; built by make_*.py, diffs beside); the "
    "originals are untouched; ship_r8_sub_tac.py is used unchanged",
    "D8 clause (1) / (1b) for HSGATE8 are conjunctions (Amendment 4's clause 1 + the gate's controls; A4 (1b) AND A5 (1b)) because HSGATE8 emits "
    "FACTORED_DEPLOY8's positions on gated windows",
    "D9 FACTORED_DEPLOY8 and H2CN_DEPLOY8 are RE-BUILT in the hsgate forward with their own registered code and asserted bit-equal to the LANDED packages (K_LANDED)",
    "D10 the FIRST --hsgate forward (07:46-07:55) WITHHELD on K_EVALFEAT ONLY: its 'extras file is the battery tree's dump_s{seed}' sub-check compared path "
    "STRINGS (str(Path) carries backslashes, the template slashes) and read False on a correct file; every other sub-check of K_EVALFEAT (key == K-LABSRC key, "
    "K-LABSRC pass, s0 extras sha256 == K-LABSRC's, all 4 joins PASS, all 4 next-window mutations RED by cross-check) and every other control read True.  "
    "Fixed to Path(...).resolve() equality (make_hsgate_forward.py), forward re-run unchanged otherwise; the withheld run's record is kept as "
    "out/forward/WITHHELD_H2CN_HSGATE_step48852ema_pf.FIRSTRUN.json",
]
