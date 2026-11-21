#!/usr/bin/env python3
"""Regenerate the navtest head predictions of the three H1 arms (CPU) from the banked fp32 planner ctx and the arms' aux_drv.pt (400 updates).
Writes scratch npz (not part of the package: 10 MB each, bit-regenerable by this script).  H2 arms are not regenerated: H2-minus-H1 head fit is
<= 2e-4 IoU on both sets (S1_READ instrument/perception_navtest.json head_fit_pairs), so they carry no extra information about the shift."""
import sys, time
sys.path.insert(0, '.')
import shf_common as C
t0 = time.time()
for arm in ['H1-s0', 'H1-s1', 'H1-s2']:
    p = C.regenerate_navtest_pred(arm)
    print('DONE', arm, p, f'{time.time()-t0:.0f}s', flush=True)
print('ZZREGEN_OK', flush=True)
