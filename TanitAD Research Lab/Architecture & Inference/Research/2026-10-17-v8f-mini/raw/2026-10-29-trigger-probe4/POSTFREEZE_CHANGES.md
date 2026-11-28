# POSTFREEZE_CHANGES -- every edit made after `FROZEN_SHA.txt` (2026-10-29T04:48:11+01:00), before any real-data fit

None of these touches a design element of PREREG_FROZEN.md (architecture, training, arms, seeds, decision rule, controls, thresholds).

1. 2026-10-29 ~04:55, `p4_run.py` + `test_trigger_probe4.py`: added `p3_consistency` and its call in `stage_tables`.  PREREG_FROZEN.md section 1 states that the banked probe-3 G4+E / G8+E scores are re-scored here and checked against probe 3's
   own `resg_<pop>_linear.json` fold-mean AUC to 1e-9; the freeze-time code did not yet implement the sentence.  The check only RECORDS (`VERDICT.json` -> `detail.<pop>.input_check`, and a printed line); it does not alter any dAUC, control or verdict.
   New hashes: p4_run.py 528348531de6e732836c48d1c7ef3dd5c5abf300e0cf95037ef4704afd99368e, test_trigger_probe4.py 14e3559783314019fdc7ee0e49e08b97e5bae6885a1ad1571b88837223ccd9c7 (p4_lib.py, chain4.sh, chain4.cmd, launch_chain4.ps1, tp_lib.py unchanged).
