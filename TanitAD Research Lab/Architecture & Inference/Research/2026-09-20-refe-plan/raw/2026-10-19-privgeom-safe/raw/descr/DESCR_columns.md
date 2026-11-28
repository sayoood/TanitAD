# DESCRIPTIVE (post-hoc, no authority over any branch): which DAC-nosign columns carry the jump (seed 0, F stratum)

Reference (registered fits): F-AUC L 0.8322 [0.7929, 0.8722]; L+G(V), all 10 columns 0.9111 [0.8845, 0.9326]; Delta_F(L+G(V) - L) +0.0789 [+0.0562, +0.0984]; the primary L+G (11 columns incl. the sign) 0.9932 [0.9876, 0.9966].

## (a) standardised L+G(V) coefficients u (logit per training-set SD of the column; mean and SD over the 5 folds)

| column | mean u | sd over folds |
|---|---|---|
| `abs_clear_min` | -1.783 | 0.163 |
| `abs_clear_t00` | -0.008 | 0.095 |
| `abs_clear_t05` | -0.055 | 0.040 |
| `abs_clear_t10` | -0.126 | 0.057 |
| `abs_clear_t15` | -0.077 | 0.041 |
| `abs_clear_t20` | +0.161 | 0.060 |
| `abs_clear_t25` | +0.639 | 0.040 |
| `abs_clear_t30` | +1.116 | 0.097 |
| `abs_clear_t35` | +1.420 | 0.144 |
| `abs_clear_t40` | +0.521 | 0.098 |

## (b) drop-one: the 9-column arm without column j (drop effect = AUC_F(all 10) - AUC_F(without j), paired)

| dropped | F-AUC without j | Delta_F vs L | drop effect |
|---|---|---|---|
| `abs_clear_min` | 0.8598 [0.8231, 0.8937] | +0.0276 [+0.0130, +0.0430] | +0.0512 [+0.0368, +0.0637] |
| `abs_clear_t00` | 0.9111 [0.8845, 0.9326] | +0.0789 [+0.0561, +0.0985] | -0.0000 [-0.0001, +0.0001] |
| `abs_clear_t05` | 0.9111 [0.8845, 0.9326] | +0.0789 [+0.0562, +0.0985] | -0.0000 [-0.0001, +0.0000] |
| `abs_clear_t10` | 0.9112 [0.8846, 0.9327] | +0.0789 [+0.0562, +0.0985] | -0.0001 [-0.0002, +0.0001] |
| `abs_clear_t15` | 0.9114 [0.8848, 0.9328] | +0.0791 [+0.0564, +0.0987] | -0.0003 [-0.0004, -0.0001] |
| `abs_clear_t20` | 0.9109 [0.8844, 0.9324] | +0.0787 [+0.0559, +0.0985] | +0.0001 [-0.0003, +0.0005] |
| `abs_clear_t25` | 0.9139 [0.8875, 0.9349] | +0.0817 [+0.0586, +0.1011] | -0.0028 [-0.0050, -0.0007] |
| `abs_clear_t30` | 0.9081 [0.8776, 0.9323] | +0.0759 [+0.0539, +0.0935] | +0.0029 [-0.0007, +0.0078] |
| `abs_clear_t35` | 0.8946 [0.8656, 0.9190] | +0.0624 [+0.0419, +0.0816] | +0.0165 [+0.0107, +0.0222] |
| `abs_clear_t40` | 0.9105 [0.8844, 0.9315] | +0.0783 [+0.0559, +0.0977] | +0.0006 [-0.0029, +0.0035] |

## (c) single-column: L + column j only

| kept | F-AUC | Delta_F vs L | share of the all-10 Delta_F |
|---|---|---|---|
| `abs_clear_min` | 0.8364 [0.7986, 0.8753] | +0.0041 [+0.0021, +0.0069] | 0.05 |
| `abs_clear_t00` | 0.8320 [0.7922, 0.8722] | -0.0002 [-0.0010, +0.0004] | -0.00 |
| `abs_clear_t05` | 0.8321 [0.7924, 0.8723] | -0.0001 [-0.0009, +0.0006] | -0.00 |
| `abs_clear_t10` | 0.8322 [0.7926, 0.8722] | -0.0000 [-0.0008, +0.0006] | -0.00 |
| `abs_clear_t15` | 0.8334 [0.7943, 0.8730] | +0.0012 [-0.0005, +0.0027] | 0.02 |
| `abs_clear_t20` | 0.8372 [0.7983, 0.8756] | +0.0050 [+0.0005, +0.0092] | 0.06 |
| `abs_clear_t25` | 0.8455 [0.8095, 0.8807] | +0.0133 [+0.0022, +0.0274] | 0.17 |
| `abs_clear_t30` | 0.8556 [0.8181, 0.8900] | +0.0234 [+0.0098, +0.0385] | 0.30 |
| `abs_clear_t35` | 0.8557 [0.8187, 0.8934] | +0.0234 [+0.0102, +0.0372] | 0.30 |
| `abs_clear_t40` | 0.8205 [0.7791, 0.8641] | -0.0118 [-0.0248, +0.0025] | -0.15 |

## (d) raw, no fit: within-set F-AUC of the |column| alone (higher score = safer) and its mean by label

| column | AUC (larger = safer) | AUC (smaller = safer) | mean unsafe | mean safe |
|---|---|---|---|---|
| `abs_clear_min` | 0.4058 | 0.5942 | 0.819 | 1.672 |
| `abs_clear_t00` | 0.5000 | 0.5000 | 1.248 | 2.380 |
| `abs_clear_t05` | 0.5005 | 0.4995 | 1.190 | 2.333 |
| `abs_clear_t10` | 0.5609 | 0.4391 | 1.146 | 2.315 |
| `abs_clear_t15` | 0.6142 | 0.3858 | 1.083 | 2.322 |
| `abs_clear_t20` | 0.6831 | 0.3169 | 0.952 | 2.329 |
| `abs_clear_t25` | 0.7404 | 0.2596 | 0.778 | 2.332 |
| `abs_clear_t30` | 0.7675 | 0.2325 | 0.664 | 2.335 |
| `abs_clear_t35` | 0.7391 | 0.2609 | 0.686 | 2.331 |
| `abs_clear_t40` | 0.5925 | 0.4075 | 0.866 | 2.317 |

Evidence class: MEASURED, post-hoc, DESCRIPTIVE. Estimator: paired LOG-cluster bootstrap over 41 logs, B = 10000, seed 20260927 (the primary read's draws); it answers 'another draw of LOGS' only, not training variance (one seed here).
