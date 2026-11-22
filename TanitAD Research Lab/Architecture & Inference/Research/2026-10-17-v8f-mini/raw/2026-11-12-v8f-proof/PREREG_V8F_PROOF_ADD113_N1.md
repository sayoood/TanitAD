# PREREG_V8F_PROOF_ADD113_N1 — note 1 to card ADD113 (CANDSET10K): the presence clause of control d2

Frozen before the queue start, before any rollout of the card exists and before the reader of the card exists. No value exists.

## What was found

The dry run of the queue on the pod printed the six commands, and the first line of the run log of three banked reference runs was read (arguments only, no value). The reference runs RR of the settings CONV and K0 (`I10RQ50Hb20`, `I10RQ50KF5Hb20`) were made by an earlier form of the queue whose log line carries no field for the pre-roll and none for the grid:

* `arm I10RQ50Hb20_s0 form RR replan steps 10 rule every 10 readout pinned kappa 0.2 model seed 0 reader seed 0 ...`
* `arm I10RQ50KF5Hb20_s0 form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 model seed 0 reader seed 0 ...`
* `arm I10RQ50KF5P60Hb20_s0 form RR replan steps 10 rule every 5 readout causal_g kappa 0.2 preroll 60 exec plan nav deadreckon grid grid_p60.json model seed 0 reader seed 0 ...`

Control d2 of the card asks for "readout, kappa, pre-roll, grid and model seed present in both". For CONV and K0 that clause would fail by construction, whatever the new runs do.

## The definition that holds

Control d2 reads:

* every field that both first lines carry is equal (form, replan steps, rule every, readout, kappa, pre-roll, exec, nav, grid, model seed, reader seed, checkpoint hash; the wrapper hash is left out and checked by d1);
* form, replan steps, rule every, readout, kappa, model seed and reader seed are present in both lines;
* where the reference line carries no pre-roll field, the new run's line has to read `preroll 0`; where it carries no grid field, the new run's line has to read `grid eval_grid_windows.json` (the pinned grid); where it carries no nav field, the new run's line has to read `nav deadreckon`. These are the defaults of the queue form that made those reference runs: a run with no pre-roll on the pinned grid;
* the rest of d2 (rule log, share above 0, q-map file from the dry run) is unchanged.

Nothing else of the card changes: settings, runs, measures, names, the other controls and the committed readings stand as frozen.

## Cost estimate, stated before the queue start

The pinned grid has 2,399 windows on 141 clips, the grid of the pre-roll of 30 steps 1,271 and that of 60 steps 284 (counted from the grid files). At the measured 10.5 min per rollout on 1,271 windows the six rollouts are ESTIMATED at about 1.5 pod-h, not 1.0 as the card says; the stop rule of 3 pod-h stands.
