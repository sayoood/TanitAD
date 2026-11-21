# Addendum A1 to PREREG_LEVELHEAD_CL (875d8c19…): the head read-out check (written 2026-11-05 after the CL read, before any read-out data)

The CL read (`motion_c/cl_read/RESULTS.md`, T1-kin) lands on the card's section-5 branch **P fails, M1 not moved** (reindexed share
0.030 / 0.006, M1 RICL − RIC0 not separated at either seed; not LEAK-BORNE). The lever reached O (`CL_PRECHECK.json`:
e_level |weights| sum 0.297 / 0.386, not zero). The card's next step is the head read-out check; this addendum fixes its numbers and
the branch each outcome takes, before any read-out number exists.

## Measurements (zero training; CL_s0 and CL_s1 checkpoints; eval-141 windows at T0, i.e. the logged history and frames at the tick)
1. **Tracking:** v_hat (the head's output) against the logged speed at the tick: R² over all windows; MAE (m/s) in the 3 to 8 m/s band
   (the band of the crawl; R² is not used in the band because the band is selected on the target); MAE on the windows whose tick is
   an M1 crawl window of the C0 rollout if that set is recoverable from the banked dumps, else reported as not computed with the reason.
2. **O's sensitivity:** with everything else fixed, replace O's level input v_in = v_hat/10 by (v_hat + 3)/10 and by (v_hat − 3)/10;
   the change of the plan's mean speed over 0 to 3 s (m/s), mean over windows, per direction; and the norm of e_level's output relative
   to the norm of O's ego token, mean over windows.

## Committed outcomes (both seeds; a split is reported per seed and read as the weaker)
* **TRACKS** if R² >= 0.5 AND band MAE <= 1.5 m/s; **DOES-NOT-TRACK** if R² < 0.3 or band MAE > 2.5 m/s; else **PARTIAL**.
* **O-SENSITIVE** if the mean |plan-speed change| >= 0.5 m/s for the +3 m/s shift; **O-INSENSITIVE** if < 0.1 m/s; else **WEAK**.
* **TRACKS and O-INSENSITIVE or WEAK** ⇒ O learned to ignore a level it was handed, because in training the logged history already
  carries the speed exactly. Next arm **CLR** = RAPR-P01's argv (a-priori deviation law, p 0.1, rejoin target) + `--level-head on
  --w-level 0.2027`: inside the recovery recipe the history is perturbed, so the vision level is the informative input. Its own card
  (addendum A2) before any run.
* **TRACKS and O-SENSITIVE** ⇒ O uses the level and it still does not close the crawl in T1-kin: next is CLR as well (the level is
  supplied but O never trained on states where history and level disagree).
* **DOES-NOT-TRACK or PARTIAL** ⇒ the frozen single-frame field does not carry the afforded speed well enough at the crawl's band:
  next is the lead-aware head (lead presence and distance from the frame; CRAWL-LEAD: half the crawl sits behind a lead), its own card.
Estimator: per-window means with an episode-cluster bootstrap B = 2000 seed 0 for R², MAE and the sensitivity (another draw of
episodes only). No eval-141 number trains anything; this check chooses the next arm, as the card's section 5 commits.
