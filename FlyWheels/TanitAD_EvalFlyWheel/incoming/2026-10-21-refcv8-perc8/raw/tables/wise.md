DEC = WiSE decoder-only blend (md5 dad4cb5b5fb316922b4e9a7e85cee8f8; build record: outside_set_is_theta8 = true), 297 windows (every 16th S2 window), unshifted clock, vs the EMA-48,852 pass A on the same (sha12, ws) keys.

| head | windows compared | bit-equal | differ | max abs diff (logit / xy / pair_err) |
|---|---|---|---|---|
| box3d slots | 281 | 281 | 0 | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| agent slots | 281 | 281 | 0 | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| 10 cm map rows | 293 | 293 (order-preserving embedding into A's table, bit-identical statistics) | 0 | |

Control verdict: **PASS** (box and map must read identically).