# PREREG_BRAKE_LOCUS -- AMENDMENT 2 (2026-10-19)  -- **POST-HOC: written AFTER run 1 of `brake_locus_r8.py`**

Run 1 (script sha256 prefix `7253adca216b`, original prereg + Amendment 1 sealed) produced `BRAKE_LOCUS.json` (sha256 prefix `f0a03c274dc5`; a copy is kept as `BRAKE_LOCUS_run1_before_amendment2.json`).  The rule, thresholds and the label in run 1 are unchanged by this file.  What follows was decided WITH sight of run 1's tables, so it is **exploratory / descriptive, not pre-registered**, and is reported in a block marked POST-HOC.  Seal: `PREREG_BRAKE_LOCUS_AMENDMENT2.sha256`; the tool now refuses unless all three seals match.

## P-1. CORRECTION: what `sel_idx_base` is (my wording in the original and in run 1's S4 caption was wrong)

Read from `refc_v3.py` lines 2637-2640: `out["traj_base"], out["sel_idx_base"] = out["traj"], out["sel_idx"]` -- `sel_idx_base` is the CORE's pick after its own ranking, which already includes the refcv8 factorised tactical term (`r8_sel`: beta_lon * log p(lon6) + constraint terms, `refc.py` `_rank`).  What moves it is the **E9 goal-selection graft** (`blended` / `e9_rank`, applied afterwards).  So `sel != sel_idx_base` = "the E9 goal graft changed the pick", NOT "the head grafts changed the pick".  Consequence: the head-coupled terms are INSIDE the base pick; on r8 false-brake windows the final pick equals the base pick in 85.6 % (seed 0), so the E9 graft is not the locus and the head-coupled ranking cannot be excluded by this column.  The S4 caption in the tool is corrected.

## P-2. POST-HOC probe: is the GT-nearest BASE anchor itself a braking anchor on the windows where it was selected?

On r8 in-bank windows with `sel_idx == a_star` (the selected anchor IS the GT-nearest base anchor), the class of the unrefined bank path (A2 rule).  On false-brake windows this is a lower bound on candidate-COVERAGE failures at the BASE-bank level (the human cruises, and the nearest anchor the bank offers brakes); on GT-cruise non-false-brake windows it is the control.  Selection-biased by construction (only windows where the model picked the nearest anchor), n small; reported with n and the episode-cluster CI; same for refcv7.  It does not measure the refined fan and does not settle (ii).

## P-3. Effect on the rule

None.  The HEAD-test verdict (FAIL, clause 2) and the label (UNDETERMINED) of run 1 stand; run 2 differs only by the corrected caption and the POST-HOC block.
