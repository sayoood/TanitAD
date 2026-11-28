#!/usr/bin/env python3
"""Find the masking convention under which the recomputed ha0 curvature MAE
reproduces the BANKED 0.040083. Until it does, the paired interval built on it is
INCONCLUSIVE, not decision-grade. ASCII only."""
import importlib
import sys

import numpy as np

import paired_curv as PC

BANKED_HA0 = 0.040083
BANKED_CL_WK7 = 0.033522

for gate in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
    PC.MIN_DS = gate
    importlib.reload  # no-op; module-level constant is read at call time
    h0, _ = PC.per_window_curv_mae(r"C:\Users\Admin\refav1_margin\p4out\dump_wk7", "ha0")
    cl, _ = PC.per_window_curv_mae(r"C:\Users\Admin\refav1_margin\p4out\dump_wk7", "cl")
    print("min_ds %5.2f  ha0 %.6f (banked %.6f, d=%+.6f)   cl %.6f (banked %.6f, d=%+.6f)  "
          "n_ha0=%d n_cl=%d"
          % (gate, np.nanmean(h0), BANKED_HA0, np.nanmean(h0) - BANKED_HA0,
             np.nanmean(cl), BANKED_CL_WK7, np.nanmean(cl) - BANKED_CL_WK7,
             int((~np.isnan(h0)).sum()), int((~np.isnan(cl)).sum())))
