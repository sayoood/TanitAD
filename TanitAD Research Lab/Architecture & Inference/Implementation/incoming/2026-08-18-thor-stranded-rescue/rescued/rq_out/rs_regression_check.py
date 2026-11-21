#!/usr/bin/env python3
"""Prove the STREAM-E renderer edits changed NOTHING for existing callers.

The banked panel6_chosen npz was produced by the PRE-EDIT gsplat_renderer.py. Re-render
the identical arm/frame through the CURRENT file and compare BIT-WISE. A `git diff`
showing an additive patch is an argument; this is evidence.
"""
import sys, numpy as np
sys.path.insert(0, '/home/nvidia/tanitad_cl/stack/experiments/alpasim-gsplat')
sys.path.insert(0, '/home/nvidia/nurec-gsplat')
from pathlib import Path
from render_quality import build_renderer, parse_arm
from rs_sweep import CONFIGS
SC = Path('/home/nvidia/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430')
BANK = '/home/nvidia/rq_out/panel6_chosen/render_AFTER_all4_cull95_sky03_f150.npz'
r, _ = build_renderer(SC, parse_arm(CONFIGS['chosen']), '/home/nvidia/nurec-gsplat')
f = 150
img, alpha, _ = r.render(r.gt_cam_to_nre(f), actor_time_us=float(r.frame_timestamps_us(f)[1]))
b = np.load(BANK)
same_img = bool(np.array_equal(img, b['render']))
same_a = bool(np.array_equal(alpha, b['alpha']))
print('banked   :', BANK)
print('img   bit-identical to the PRE-EDIT render :', same_img)
print('alpha bit-identical                        :', same_a)
if not same_img:
    d = np.abs(img.astype(int) - b['render'].astype(int))
    print('  max|diff|', d.max(), 'mean|diff|', d.mean(), 'frac differing', float((d.max(-1) > 0).mean()))
print('VERDICT:', 'NO REGRESSION' if (same_img and same_a) else 'RENDER CHANGED -- INVESTIGATE')
