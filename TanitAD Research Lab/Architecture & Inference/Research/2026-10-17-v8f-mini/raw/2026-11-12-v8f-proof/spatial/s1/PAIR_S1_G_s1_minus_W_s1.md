# PAIR_S1: G_s1 minus W_s1
Tier T0: a world-model diagnostic, never driving performance. in-sample for the model (train probe clips). Paired episode-cluster bootstrap, clips as clusters (1200 clips), B 2000, seed 0, one draw. Frames with a gap 33114, pairs 24206. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A difference is separated only if it holds at both seeds AND is larger than the spread of the HIER seeds (`spread`); an interval over clips is blind to training variance.
- gap R2 (state): +0.1159 [+0.087, +0.146]
- presence AUROC (state): +0.0205 [+0.012, +0.029]
- change skill H: -0.0402 [-0.075, -0.005]
- change skill N: -0.0666 [-0.109, -0.029]
- change skill E: +0.0000 [+0.000, +0.000]
