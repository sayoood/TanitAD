"""Smoothness of the per-tick controls: 4-point subdivision (committed) vs linear midpoints (v8f_mini.upsample_prior's rule)."""
import json, sys
import numpy as np, torch
ROOT = 'C:/Users/Admin/v8fa1/tree'
sys.path.insert(0, f'{ROOT}/stack')
from tanitad.models import p10_prior as P
from tanitad.models import kinematic_prior as KP
S1 = f'{ROOT}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/raw'
m = P.load_frozen_p10(f'{ROOT}/stack/tanitad/configs/v8f_p10/p10_prior.pt')
man = torch.load('C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt', map_location='cpu', weights_only=False)
poses = [p.numpy().astype(np.float64) for p in man['poses']]; acts = [a.numpy().astype(np.float64) for a in man['actions']]
out = {}
for tag in ('S2', 'S6'):
    z = np.load(f'{S1}/arms_{tag}_s0.npz'); ci, ws = z['clip_index'].astype(int), z['ws'].astype(int)
    po = torch.from_numpy(np.stack([poses[c][t - 7:t + 1] for c, t in zip(ci, ws)])); ac = torch.from_numpy(np.stack([acts[c][t - 7:t + 1] for c, t in zip(ci, ws)]))
    kin3 = P.kin3_from_window(po, ac, 8); p30 = m(kin3); v0 = kin3[:, 0].float()
    res = {}
    for name in ('subdivision', 'linear'):
        if name == 'subdivision':
            ticks = KP.knots_to_ticks(p30.double()).float()
        else:
            knots = torch.cat([p30.new_zeros(p30.shape[0], 1, 2), p30], 1); mid = 0.5 * (knots[:, :-1] + knots[:, 1:])
            ticks = torch.stack([mid, knots[:, 1:]], 2).reshape(p30.shape[0], 60, 2)
        c = KP.controls_from_path(ticks, v0, kappa_cap=KP.P10_KAPPA_CAP)
        dk = (c[:, 1:, 1] - c[:, :-1, 1]).abs(); da = (c[:, 1:, 0] - c[:, :-1, 0]).abs()
        # the odd/even alternation a linear path puts into the controls (a kink at every knot): mean |c[odd] - c[even]| neighbours
        res[name] = {'kappa_total_variation_per_window': float(dk.sum(1).mean()), 'accel_total_variation_per_window': float(da.sum(1).mean()),
                     'kappa_abs_p99.9': float(np.percentile(c[..., 1].abs().numpy(), 99.9)), 'accel_abs_p99.9': float(np.percentile(c[..., 0].abs().numpy(), 99.9)),
                     'kappa_tv_median': float(dk.sum(1).median()), 'accel_tv_median': float(da.sum(1).median())}
    out[tag] = res; print(tag, json.dumps(res))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
