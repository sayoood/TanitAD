# PAIR_S1: P_s1 minus G_s1
Tier T0: a world-model diagnostic, never driving performance. in-sample for the model (train probe clips). Paired episode-cluster bootstrap, clips as clusters (1200 clips), B 2000, seed 0, one draw. Frames with a gap 33114, pairs 24206. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A difference is separated only if it holds at both seeds AND is larger than the spread of the HIER seeds (`spread`); an interval over clips is blind to training variance.
- gap R2 (state): -0.0330 [-0.053, -0.014]
- presence AUROC (state): -0.0043 [-0.010, +0.001]
- change skill H: +0.0129 [-0.012, +0.040]
- change skill N: +0.0303 [+0.001, +0.062]
- change skill E: +0.0000 [+0.000, +0.000]
