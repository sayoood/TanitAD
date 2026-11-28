# PAIR_S1: W_s0 minus HIER_s0
Tier T0: a world-model diagnostic, never driving performance. in-sample for the model (train probe clips). Paired episode-cluster bootstrap, clips as clusters (1200 clips), B 2000, seed 0, one draw. Frames with a gap 33114, pairs 24206. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A difference is separated only if it holds at both seeds AND is larger than the spread of the HIER seeds (`spread`); an interval over clips is blind to training variance.
- gap R2 (state): -0.0080 [-0.027, +0.010]
- presence AUROC (state): +0.0025 [-0.004, +0.009]
- change skill H: -0.0033 [-0.028, +0.021]
- change skill N: -0.0002 [-0.022, +0.022]
- change skill E: +0.0000 [-0.000, +0.000]
