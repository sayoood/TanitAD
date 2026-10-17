import sys, json
import numpy as np, torch
sys.path.insert(0, 'C:/Users/Admin/v8fa1/tree/stack')
from tanitad.models import p10_prior as P
from tanitad.models import kinematic_prior as KP
from tanitad.models.kinematic import rollout_unicycle
ROOT = 'C:/Users/Admin/v8fa1/tree'
S1 = f'{ROOT}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/raw'
P10F = f'{ROOT}/stack/tanitad/configs/v8f_p10/p10_prior.pt'
man = torch.load('C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt', map_location='cpu', weights_only=False)
poses = [p.numpy().astype(np.float64) for p in man['poses']]
acts = [a.numpy().astype(np.float64) for a in man['actions']]
m = P.load_frozen_p10(P10F)
W = 8
z = np.load(f'{S1}/arms_S2_s0.npz')
ci, ws = z['clip_index'].astype(int), z['ws'].astype(int)
po = torch.from_numpy(np.stack([poses[c][t - W + 1:t + 1] for c, t in zip(ci, ws)]))
ac = torch.from_numpy(np.stack([acts[c][t - W + 1:t + 1] for c, t in zip(ci, ws)]))
kin3 = P.kin3_from_window(po, ac, W)
p30 = m(kin3)
v0 = kin3[:, 0].float()
ticks = KP.knots_to_ticks(p30.double()).float()
st = torch.zeros(len(v0), 4); st[:, 3] = v0
rows = []
for cap in (0.3, 0.5, 1.0, 2.0, 1e9):
    for floor in (2.0, 1.0, 0.5):
        ctrl = KP.controls_from_path(ticks, v0, kappa_cap=cap, v_floor=floor)
        rolled = rollout_unicycle(st, ctrl, dt=0.1)[..., :2][:, 1::2]
        ade = (rolled - p30).norm(dim=-1).mean(1).numpy()
        r = {'cap': cap, 'floor': floor, 'mean': float(ade.mean()), 'median': float(np.median(ade)), 'p95': float(np.percentile(ade, 95)),
             'max': float(ade.max()), 'frac_gt_0.05': float((ade > 0.05).mean()),
             'kappa_abs_p99.9': float(np.percentile(ctrl[..., 1].abs().numpy(), 99.9)), 'a_abs_p99': float(np.percentile(ctrl[..., 0].abs().numpy(), 99))}
        rows.append(r); print(json.dumps(r))
json.dump(rows, open(sys.argv[1], 'w'), indent=1)
