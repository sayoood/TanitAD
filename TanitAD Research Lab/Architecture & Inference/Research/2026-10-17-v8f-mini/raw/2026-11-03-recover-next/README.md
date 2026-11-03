# recover_next — the RECOVER-FT card's committed next arms, argv pre-built (2026-11-03, before the read's verdict)

Card `recover_ft/PREREG_RECOVER_FT_FROZEN.md` section 8 names one next arm per outcome. The argv for every arm that needs
no new code is generated here from the shipped `R_{0,1}.train.txt` (same init, steps, data and flags; only the named lever
changes). None is launched: each needs a PI box-2 go, asked with the verdict.

| verdict | arm | argv (s0 / s1) | change vs R_s | box-2 cost (ESTIMATED) |
|---|---|---|---|---|
| NOT MOVED | R-APR | `RAPR_{0,1}.train.txt` | `--recover-dev-law apriori`, bank flags dropped (the trainer refuses a bank with the a-priori law) | ~0.65 h per run (no table build) |
| INSIDE-FLOOR / PARTIAL | R-rep | `RREP_{0,1}.train.txt` | `--seed 10+s` | ~1.3 h per run (table build ~42 min) |
| HARMS | dose | `RP01_{0,1}.train.txt` | `--recover-p 0.1` | ~1.3 h per run |
| HARMS | attribution | `RTL_{0,1}.train.txt` | `--recover-target logged`, `--recover-rejoin-steps` dropped (it does nothing under the logged target) | ~1.3 h per run |
| ARTEFACT / PARTIAL | training-time hold | not built | needs `--recover-hold-now` in `v8f_recover.py` + trainer + G-DVB entry + reach probe | build first |

At launch: copy the chosen argv into `recover_ft_launch/box2_files/argv/`, add the tags to `recft_chain.cmd`'s loop and to
`recft_check.py`'s tag set, re-run the box-2 gate (`--selfcheck-only`) per new tag, extend `pull_recft.sh`'s list.
The comparison arm for every row is the existing `C0_s` (and `C0rep_s` for the floor); no new control run is needed.
