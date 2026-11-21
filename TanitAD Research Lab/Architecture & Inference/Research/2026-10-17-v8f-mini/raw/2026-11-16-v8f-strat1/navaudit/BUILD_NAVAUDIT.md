# BUILD_NAVAUDIT -- audit of the turns only v9 names, and a wide-announcement text nav

Evidence class: MEASURED (ours; banked T1-kin dumps, pinned `navtok_table.npz`, zero training, CPU). Tier: T1-kin tick 0. Results: `out/RESULT_NAVAUDIT.md` / `.json`. Pre-registration (written before scoring): `PREREG_NAVAUDIT.md`. Imports (never edits) the landed builder copy `F:/TanitAD-scratch/strat1/build/tree/stack/scripts/build_navtext_nav.py` (sha256 prefix `fdbe4d23`) and `navecho3` / `navecho2` / `navecho`.

## Reading

The bar is NOT met: the wide text (`d_ann = max(60 m, 8 s x v)`) names the correct turn on **124 of 197** windows (bar 160) and names the other side on **3** (bar 0). The cut is not the main cause.

Of the 75 windows only v9 names (19 episodes): ROUTE-LACKS-TURN 55, ANNOUNCED-LATE-CUT 11, OTHER 6, MATCH-FAILED 3, ANNOUNCED-LATE-ROW 0. The widened cut recovers 12 windows (0 lost; paired +0.061 [+0.027, +0.097]) and is the whole of the cut lever: with no cut at all the table holds the labelled side at NOW on 124 of 197. 50 of the 55 ROUTE-LACKS windows sit in 9 episodes whose text table holds no turn-sided row at any step of the clip: 6 roundabout clips (RB_* / ROTARY rows, mapped to none by the builder's rule 1, 43 windows) and 3 clips with CONTINUE rows only (7 windows). The 3 wrong-side windows are episode 139 (table says TURN_R at 50-56 m, label TURN_L). Nav-only out-of-fold macro-F1, wide minus pinned: ALL -0.013 [-0.039, +0.009], INF_ANY +0.004 [-0.017, +0.030] (not separated).

## Method as built
* Window set, labels, tick-0 nav: as navecho3 (2,399 dump windows, 197 TURN windows). Text nav at NOW = table row `off[ci] + ws`; K-FEED: it equals the N1 dump's nav_in.
* Classification: `navaudit.classify` (rules and `build_navtext_nav.py` lines in the module docstring and RESULT section 1). Later-row horizon 40 steps (0.2 s each, join `now_s` diff 0.2013 s).
* Wide variant: `build_variant` sets the builder's module constants `D_ANN_MIN`, `D_ANN_PER_MS` (`:35`) to 60 / 8 around `build_split`, nothing else. Proof: the same wrapper at 30 / 6 reproduces the pinned arrays exactly (train and eval, nav / nav_d / rowclamp); wide vs pinned differ in 14,437 train and 355 eval steps, exactly the set {flag 0, sided, countdown in (d_ann_30/6, d_ann_60/8]}, only in the one-hot, only FOLLOW to side; ch3, ch4, nav_d identical.
* No ego-future read: the builder reads the table, `now_s` and track rows up to row 2j (`:75-100`). Probe: 400 sampled steps, track rows after 2j replaced by garbage, 0 differ.
* Scoring: landed navecho3 `cue_state`, `turn_tables`, `share_ci`, `mean_pair`; classifier = navecho2 `crossfit_hgb` + `fam_extra.conf_pair`; B 2,000, seed 0.

## Tests (`test_navaudit.py`, 14 pass)
A mutation per rule goes RED: failed-rule without flag, without unusable; cut-rule without side check, cut counting inside-cut steps; row-later never firing / ignoring the later row; lacks never firing / swallowing the other side; diff proof catching a changed distance channel and a wrong step set; a builder that reads 10 track rows ahead fails the future-read probe.

## Next lever (no proposal to the PI: the bar is not met)
The missing cue is in the text source, not the cut: in 9 episodes the table never carries a turn side. The cheapest next measurement (zero training): (a) for the 6 roundabout episodes read what the table's RB_* / ROTARY rows carry (entry, exit number, no side) and test an `ROUNDABOUT` state against the v9 label on those 43 windows; a side cannot come from the text there, and a side built from the ego path would break the vision-only rule at inference; (b) for the 3 CONTINUE-only clips, compare the logged track's turn with the route text (the ego may leave the announced route). A channel for a roundabout state is an architecture change (nav_in has 5 channels), which is the PI's decision. The wide variant stays available as `navtext_nav_wide.npz` for any later arm; on the present evidence it adds 12 windows and 3 wrong-side namings.
