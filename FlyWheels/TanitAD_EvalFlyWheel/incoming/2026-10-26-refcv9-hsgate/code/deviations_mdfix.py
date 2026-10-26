"""deviations_mdfix.py -- disclosures + disclosed open choices of PREREG_HSGATE_MDFIX (545ff217...) PHASE B; stated in ship_r8_mdfix_verdict.py and mdfix_bars.py."""
DISCLOSURES_S0 = [
    "S0-a Forking paths: the MDFIX label was chosen AFTER the HSGATE8 eval139 read, on FIT-row arithmetic only (next/FDCAP_FRONTIER_FIT.json; "
    "next/DRAFT_NEXT_ARM_HSGATE_MDFIX.md). This is the SECOND eval139 read of the gate family; the multiplicity is paid in the bars (G1 read at 97.5 %).",
    "S0-b No MDFIX output on eval139 existed at sealing; the draft's FIT-side numbers are 5-fold episode CV on 7,147 FIT rows, one training seed, one fit, no interval.",
]
DEVIATIONS = [
    "M1 s1a E_FIT is read literally as ALL FIT episodes (297); the FIT episodes holding a GT-cruise row (279) give the same projected FD upper bound to 4 dp "
    "(+0.0042) and are reported beside (next/MDFIX_POWER_FIT.json)",
    "M2 s1a SE_FIT = std (ddof 1) of the paired bootstrap deltas drawn with taniteval.ci's OWN episode draws (B 2000, seed 0) -- the same resampling as the cells",
    "M3 tau scan: candidates are every distinct held-out logit plus +inf (no gating); FIT MD / FD minimised on INTEGER counts (no float ties); K-TAU re-derives tau "
    "and both counts by an independent brute-force scan",
    "M4 REGISTERED_MDFIX.sha256 is a multi-line pin file with '#' comment lines (line 2 carries the PREREG sha); every pin is read by its EXACT path line "
    "(mdfix_common.registered_sha) -- ship_r8_sub_common.read_seal would read '#' and refuse; all four pins are re-hashed at forward start",
    "M5 the forward / verdict are GENERATED copies of the HSGATE phase-2 copies (make_mdfix_forward.py, make_mdfix_verdict.py; diffs beside); the HSGATE copies and "
    "the ship_r8_sub originals are untouched; the CLI switch keeps the name --hsgate and the forward JSON keeps the key 'hsgate' (read by the copies); "
    "ship_r8_sub_tac.py is used unchanged",
    "M6 G1's 97.5 % interval is taniteval.ci.paired_episode_cluster_bootstrap(alpha=0.025) on the SAME windows / draws as pair_read_r8.paired_cell; control "
    "K-CI975 asserts the same direct call at alpha 0.05 reproduces paired_cell's interval exactly and that the 97.5 % interval contains the 95 % one",
    "M7 the HSGATE8 beside rows read the LANDED HSGATE phase-2 package (out/arms/h2cn_hsgate/HSGATE8, sha256 recorded), not a rebuild",
    "M8 (carried from HSGATE D1 / D8 / D9) gate objective mean BCE + 0.5 lambda ||w||^2, bias unpenalised; clause (1) / (1b) for MDFIX8 are conjunctions (A4 clause 1 + "
    "the gate's controls; A4 (1b) AND A5 (1b)); H2CN_DEPLOY8 / FACTORED_DEPLOY8 are rebuilt and asserted bit-equal to the LANDED packages (K_LANDED)",
]
