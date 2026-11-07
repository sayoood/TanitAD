# YAX FINAL (full) 2026-11-07T16:00:52

Verdicts: {"X1": "REFUTED", "X2": "REFUTED"}  | selection: {"chosen_X1": "K1", "finished": "2026-11-07T03:02:54", "other_X2": "K2", "rule": "lower cache-VAL loss, relative gap < 1 % -> K1 (PREREG section 2)", "seconds": 1889.3, "val_final_loss": {"K1": 0.1856604619272824, "K2": 0.18566810005697712}}
Chain ended at X2 (recipe K2): REFUTED

Tier: NAVSIM v1.1 pseudo-simulation, open loop, NOT closed loop. Estimator: paired log-cluster bootstrap 10,000, seed 20260927.

Both recipe rungs are NOT PROVEN / REFUTED. Named levers, each with what unblocks it:
(i) ax-noise/dropout augmentation (keep ax, break its loop gain): CPU-only, needs a PI/MM go on a new PREREG;
(ii) P7 from scratch without the ax slot: GPU-days, PI decision;
(iii) larger cache (4,000 frames is the capacity limit): ~1.1 s/frame GPU encoder pass (about 1.2 h per 4,000), needs the GPU lock.
