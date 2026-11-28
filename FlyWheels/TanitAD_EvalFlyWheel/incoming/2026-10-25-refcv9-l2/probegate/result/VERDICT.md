FAIL (P1, P3, G)

P1=False  P2=True  P3=False  G=False
MD base 0.4695 -> probegate 0.4726, randgate 0.4773, oracle 0.3412
FD base 0.1395 -> probegate 0.1428, randgate 0.1584
ceiling share -0.024 [-0.238, 0.127]
G rows worse: LAT_cross_abs_final_m, LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid, TAC_traj_lon_correct  (ORACLE force degrades: LAT_curvature_mae_1pm_valid, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid)

estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)

NEXT LEVER: GATE INERT (P1 failed and the missed-decel point estimate did not improve): flipping the head's decel/cruise group at probe quality does not move the plan.  Next lever = stop gating post hoc; attribute the oracle gap per forced class (BRAKE_TO / CREEP / HOLD) at 0 GPU from the banked oracle trace, then train toward the probe's posterior (L3) only if one class carries the gap.
