# LLR read-out: O's sensitivity to the token wiring (A2 card Reads; A1's measurement adapted to --aux-cond token)

Statistic: change of the plan's mean speed over 0..3 s (m/s), changed input minus baseline, per window, episode-cluster bootstrap B=2000 seed 0 over the full eval set. LEVEL labels = A1's committed rule on the +3 m/s shift; LEAD labels = A1's thresholds applied BY ANALOGY to the presence 0 -> 1 flip (the card commits none).

| input change | s0 mean signed [95% CI] | s0 mean abs | s1 mean signed [95% CI] | s1 mean abs |
|---|---|---|---|---|
| level: v_hat +3 m/s (token column 0 +0.3) | +0.0160 [+0.0147, +0.0174] | 0.0165 | +0.0049 [+0.0037, +0.0060] | 0.0091 |
| level: v_hat -3 m/s | -0.0152 [-0.0164, -0.0139] | 0.0156 | -0.0048 [-0.0059, -0.0036] | 0.0087 |
| lead: presence set to 1 (vs baseline) | +0.0017 [-0.0031, +0.0063] | 0.0283 | +0.0351 [+0.0309, +0.0393] | 0.0360 |
| lead: presence set to 0 (vs baseline) | -0.0016 [-0.0036, +0.0006] | 0.0113 | -0.0126 [-0.0150, -0.0103] | 0.0131 |
| lead: presence flip 1 minus 0 | +0.0033 [-0.0024, +0.0088] | 0.0394 | +0.0477 [+0.0437, +0.0520] | 0.0490 |
| beside: gap 30 m minus 10 m at presence 1 | -0.0402 [-0.0445, -0.0358] | 0.0409 | +0.0743 [+0.0667, +0.0824] | 0.0744 |
| beside: e_aux output zeroed | -0.0250 [-0.0314, -0.0187] | 0.0481 | -0.1080 [-0.1271, -0.0888] | 0.1309 |

| label | s0 | s1 |
|---|---|---|
| LEVEL (A1's rule, +3 m/s) | O-INSENSITIVE (mean-abs reading O-INSENSITIVE) | O-INSENSITIVE (mean-abs reading O-INSENSITIVE) |
| LEAD (by analogy, presence flip) | O-INSENSITIVE (mean-abs reading O-INSENSITIVE) | O-INSENSITIVE (mean-abs reading O-INSENSITIVE) |

||e_aux|| / ||ego token|| (mean of per-window ratios): s0 1.5338, s1 1.5699
e_aux |weight| column sums (level, presence, gap): s0 [73.104, 73.335, 75.372]; s1 [73.169, 71.968, 78.539]

Controls: ALL PASS
No wiring decision follows from this table: LLR is the last arm of the A2 card; the table is evidence for the write-up of why the token was or was not used.
