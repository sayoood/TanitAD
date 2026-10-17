"""MEASURE ADE(P10u, P10): P10's path re-rolled through rollout_unicycle from the measured v0, on REAL eval139
windows (the S1 package's own S2 / S6 window lists), with the controls the decoder path would derive.
Also: the same for the linear-midpoint interpolation and for the OPEN-loop inverse (controls), as controls."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, 'C:/Users/Admin/v8fa1/tree/stack')
from tanitad.models import p10_prior as P
from tanitad.models import kinematic_prior as KP
from tanitad.models.kinematic import rollout_unicycle, unicycle_controls_from_path

ROOT = 'C:/Users/Admin/v8fa1/tree'
S1 = f'{ROOT}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/raw'
P10F = f'{ROOT}/stack/tanitad/configs/v8f_p10/p10_prior.pt'
MAN = 'C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt'
W = 8
out_json = sys.argv[1] if len(sys.argv) > 1 else None

m = P.load_frozen_p10(P10F)
man = torch.load(MAN, map_location='cpu', weights_only=False)
poses = [p.numpy().astype(np.float64) for p in man['poses']]
acts = [a.numpy().astype(np.float64) for a in man['actions']]
clips = [str(c) for c in man['clip_id']]
print('eval139 manifest clips', len(clips))


def windows(tag):
    z = np.load(f'{S1}/arms_{tag}_s0.npz')
    return z['clip_index'].astype(int), z['ws'].astype(int), z


def roll_ticks(ctrl, v0):
    st = torch.zeros(ctrl.shape[0], 4)
    st[:, 3] = v0
    return rollout_unicycle(st, ctrl, dt=0.1)[..., :2]


def build_batch(ci, ws):
    po = np.stack([poses[c][t - W + 1:t + 1] for c, t in zip(ci, ws)])
    ac = np.stack([acts[c][t - W + 1:t + 1] for c, t in zip(ci, ws)])
    return torch.from_numpy(po), torch.from_numpy(ac)


res = {}
for tag in ('S2', 'S6'):
    ci, ws, z = windows(tag)
    ok = ws >= W - 1
    print(tag, 'windows', len(ws), 'with a full W=8 window', int(ok.sum()))
    ci, ws = ci[ok], ws[ok]
    row = {}
    for variant in ('dlg4', 'linear', 'open_loop'):
        ade6, ade2, kmax, capfrac, vfin = [], [], [], [], []
        t0 = time.time()
        for s in range(0, len(ws), 512):
            po, ac = build_batch(ci[s:s + 512], ws[s:s + 512])
            kin3 = P.kin3_from_window(po, ac, W)
            p30 = m(kin3)
            v0 = kin3[:, 0].to(torch.float32)
            if variant == 'linear':
                knots = torch.cat([p30.new_zeros(p30.shape[0], 1, 2), p30], 1)
                mid = 0.5 * (knots[:, :-1] + knots[:, 1:])
                ticks = torch.stack([mid, knots[:, 1:]], 2).reshape(p30.shape[0], 60, 2)
            else:
                ticks = KP.knots_to_ticks(p30.double()).float()
            if variant == 'open_loop':
                ctrl = unicycle_controls_from_path(ticks, dt=0.1)
            else:
                ctrl = KP.controls_from_path(ticks, v0, kappa_cap=KP.P10_KAPPA_CAP)
            rolled = roll_ticks(ctrl, v0)
            at_knots = rolled[:, 1::2]                       # ticks 2,4,..,60
            d = (at_knots - p30).norm(dim=-1)                # [B, 30]
            ade6.append(d.mean(1).numpy())
            ade2.append(d[:, :10].mean(1).numpy())
            kmax.append(ctrl[..., 1].abs().amax(1).numpy())
            capfrac.append((ctrl[..., 1].abs() >= KP.POSE_KAPPA_CAP - 1e-6).float().mean(1).numpy())
        ade6, ade2, kmax, capfrac = map(np.concatenate, (ade6, ade2, kmax, capfrac))
        row[variant] = {'n': int(len(ade6)), 'ade6_mean_m': float(ade6.mean()), 'ade6_median_m': float(np.median(ade6)),
                        'ade6_p95_m': float(np.percentile(ade6, 95)), 'ade6_max_m': float(ade6.max()),
                        'ade2_mean_m': float(ade2.mean()), 'frac_ade6_gt_0p05': float((ade6 > 0.05).mean()),
                        'kappa_abs_max_p99': float(np.percentile(kmax, 99)),
                        'windows_with_any_cap_bound_tick': float((capfrac > 0).mean()),
                        'seconds': time.time() - t0}
        print(tag, variant, json.dumps(row[variant]))
    res[tag] = row
if out_json:
    json.dump(res, open(out_json, 'w'), indent=1)
