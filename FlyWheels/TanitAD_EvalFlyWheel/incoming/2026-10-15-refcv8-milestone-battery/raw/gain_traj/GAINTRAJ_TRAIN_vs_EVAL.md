# GAINTRAJ train vs eval -- EMA-30k flat g* (POST-HOC addendum, not registered; no role in the label)

g* [95 % episode-cluster CI] on the 139 TRAIN-clip rolls (registered PREREG section 6) beside the eval139 EMA-30k value on the same window set; the `all windows` eval row is added post hoc (the registered eval number is compliant-only). Difference = unpaired (two different clip sets).

| surface | window set | seed | train g* | eval g* | train - eval | n windows train / eval |
|---|---|---|---|---|---|---|
| S2 | compliant | 0 | 0.686 [0.603, 0.772] | 0.455 [0.377, 0.536] | +0.231 [+0.113, +0.350] sep | 4,211 / 4,159 |
| S2 | all | 0 | 0.496 [0.417, 0.578] | 0.384 [0.328, 0.449] | +0.112 [+0.011, +0.210] sep | 4,855 / 4,754 |
| S2 | compliant | 1 | 0.676 [0.595, 0.760] | 0.462 [0.386, 0.540] | +0.215 [+0.102, +0.327] sep | 4,211 / 4,159 |
| S2 | all | 1 | 0.501 [0.423, 0.584] | 0.379 [0.321, 0.440] | +0.122 [+0.026, +0.220] sep | 4,855 / 4,754 |
| S6 | compliant | 0 | 0.833 [0.736, 0.929] | 0.619 [0.531, 0.701] | +0.214 [+0.093, +0.339] sep | 3,301 / 3,251 |
| S6 | all | 0 | 0.781 [0.668, 0.903] | 0.600 [0.529, 0.666] | +0.181 [+0.051, +0.317] sep | 3,749 / 3,668 |
| S6 | compliant | 1 | 0.827 [0.734, 0.923] | 0.619 [0.531, 0.700] | +0.208 [+0.087, +0.333] sep | 3,301 / 3,251 |
| S6 | all | 1 | 0.782 [0.671, 0.903] | 0.597 [0.525, 0.663] | +0.185 [+0.053, +0.320] sep | 3,749 / 3,668 |

Control: the eval compliant rows reproduce the registered GAINTRAJ.json cells to 1e-12. Caveats: single training seed (the trees are checkpoints of ONE run); T1 open loop on logged eval139 frames; per-instant fits are in-sample per tree; eval episodes only; the interval answers 'another draw of EPISODES' only (not another training run, not another inference run); no extrapolation beyond step 30,000 (final = 48,852); train clips are in this run's training set.
