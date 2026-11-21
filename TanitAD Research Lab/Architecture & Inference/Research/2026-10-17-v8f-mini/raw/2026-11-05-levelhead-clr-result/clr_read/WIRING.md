# WIRING: the O-sensitivity of CLR and the wiring it fixes for LLR (A1's instrument on the CLR checkpoints; T0 readout, not driving performance)

**Decision: O-INSENSITIVE** -> BOTH heads' outputs ([sigmoid(logit), gap_hat] and the level) as one separate conditioning token of O (--aux-cond token, non-zero init)

Note: both seeds.

| seed | sensitivity (A1, signed) | +3 m/s mean signed change | mean-abs reading | label (mean-abs) | tracking | R2 | band MAE |
|---|---|---|---|---|---|---|---|
| s0 | O-INSENSITIVE | +0.0005 | 0.0006 | O-INSENSITIVE | PARTIAL | 0.780 | 1.902 |
| s1 | O-INSENSITIVE | +0.0021 | 0.0021 | O-INSENSITIVE | PARTIAL | 0.779 | 1.827 |
