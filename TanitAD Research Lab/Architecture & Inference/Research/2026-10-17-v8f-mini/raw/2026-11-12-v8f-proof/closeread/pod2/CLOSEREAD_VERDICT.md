# CLOSEREAD verdict (card ADD77, tier T0: a probe on cached tokens of logged frames)

All controls pass. Frame period 0.200 s. Episodes: map file.

| cell | r pooled | r 95 % | r per seed | MAE est | MAE zero | paired diff (est - 0) 95 % | name |
|---|---|---|---|---|---|---|---|
| L_H5 | +0.312 | [+0.200, +0.414] | +0.317, +0.306 | 1.945 | 2.027 | -0.082 [-0.173, +0.011] | CLOSEREAD2-L-H5-NO-CLOSING |
| N_H5 | +0.322 | [+0.205, +0.421] | +0.350, +0.292 | 1.925 | 2.027 | -0.102 [-0.191, -0.012] | CLOSEREAD2-N-H5-NO-CLOSING |
| L_H10 | +0.371 | [+0.276, +0.454] | +0.384, +0.359 | 1.885 | 2.038 | -0.153 [-0.266, -0.044] | CLOSEREAD2-L-H10-READS-CLOSING |
| N_H10 | +0.391 | [+0.288, +0.478] | +0.426, +0.356 | 1.874 | 2.038 | -0.164 [-0.280, -0.049] | CLOSEREAD2-N-H10-READS-CLOSING |
| L_H25 | +0.384 | [+0.283, +0.475] | +0.402, +0.367 | 1.876 | 2.080 | -0.204 [-0.363, -0.059] | CLOSEREAD2-L-H25-READS-CLOSING |
| N_H25 | +0.361 | [+0.257, +0.460] | +0.369, +0.353 | 1.902 | 2.080 | -0.178 [-0.324, -0.044] | CLOSEREAD2-N-H25-READS-CLOSING |

Beside (no names): the reader's own closing output and the two-point change of the gap read over H frames.

| H | row | r pooled | r per seed | MAE | MAE zero |
|---|---|---|---|---|---|
| H5 | reader_closing | +0.184 | +0.213, +0.159 | 2.142 | 2.027 |
| H5 | gap_two_point | +0.249 | +0.273, +0.223 | 3.314 | 2.027 |
| H10 | reader_closing | +0.179 | +0.214, +0.150 | 2.159 | 2.038 |
| H10 | gap_two_point | +0.267 | +0.291, +0.241 | 2.678 | 2.038 |
| H25 | reader_closing | +0.155 | +0.201, +0.117 | 2.248 | 2.080 |
| H25 | gap_two_point | +0.231 | +0.236, +0.226 | 2.250 | 2.080 |

**CLOSEREAD2-A-CLOSING-IS-READABLE** (4 of six cells read READS-CLOSING). six arm-history cells are read.

Families: this probe measures none of them. LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0).
