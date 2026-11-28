# SPREAD_S1_HIER: 3 runs (HIER_s0, HIER_s1, HIER_s2)
Tier T0: a world-model diagnostic, never driving performance. in-sample for the model (train probe clips). Min, max, range and sample std (ddof 1) of each point value over the runs. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
- gap R2 (state): min +0.4106, max +0.4556, range 0.0450, std 0.0236
- presence AUROC (state): min +0.7874, max +0.7897, range 0.0023, std 0.0012
- gap R2 (state minus ego): min +0.2234, max +0.2685, range 0.0450, std 0.0236
- presence AUROC (state minus ego): min +0.2283, max +0.2306, range 0.0023, std 0.0012
- change skill H: min +0.2629, max +0.2678, range 0.0049, std 0.0025
- change skill N: min +0.2519, max +0.2776, range 0.0257, std 0.0138
- change skill E: min +0.2911, max +0.2911, range 0.0000, std 0.0000
- H minus E: min -0.0282, max -0.0233, range 0.0049, std 0.0025
- H minus N: min -0.0098, max +0.0142, range 0.0240, std 0.0123
