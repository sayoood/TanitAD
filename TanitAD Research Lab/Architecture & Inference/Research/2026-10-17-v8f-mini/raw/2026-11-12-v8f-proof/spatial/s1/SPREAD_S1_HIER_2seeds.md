# SPREAD_S1_HIER_2seeds: 2 runs (HIER_s0, HIER_s1)
Tier T0: a world-model diagnostic, never driving performance. in-sample for the model (train probe clips). Min, max, range and sample std (ddof 1) of each point value over the runs. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
- gap R2 (state): min +0.4453, max +0.4556, range 0.0103, std 0.0073
- presence AUROC (state): min +0.7893, max +0.7897, range 0.0004, std 0.0003
- gap R2 (state minus ego): min +0.2581, max +0.2685, range 0.0103, std 0.0073
- presence AUROC (state minus ego): min +0.2302, max +0.2306, range 0.0004, std 0.0003
- change skill H: min +0.2629, max +0.2662, range 0.0033, std 0.0023
- change skill N: min +0.2519, max +0.2559, range 0.0040, std 0.0028
- change skill E: min +0.2911, max +0.2911, range 0.0000, std 0.0000
- H minus E: min -0.0282, max -0.0249, range 0.0033, std 0.0023
- H minus N: min +0.0070, max +0.0142, range 0.0073, std 0.0051
