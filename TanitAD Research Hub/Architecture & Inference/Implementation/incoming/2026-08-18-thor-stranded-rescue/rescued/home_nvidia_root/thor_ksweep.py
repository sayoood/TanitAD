'''Lever #4 quantified: what does SHORTENING the imagination roll actually buy?

The roll is the dominant stage (69ms of a 97ms tick) and is COMPUTE-bound, so its cost should be
linear in K. If it is, the latency half of the K question is settled by arithmetic and only the
ACCURACY half needs GPU time on a trained model + the four families.

⛔ THIS IS THE LATENCY HALF ONLY. K is an ARCHITECTURE parameter: the planner reads imagination at
horizons [5,10,15,20] and anchors live at 20 steps. Cutting K without a four-family measurement
would be trading a decision-quality unknown for milliseconds — exactly what the binding rule
forbids. This produces the cost curve so the PI can price that trade, not authorise it.
'''
import json, os, sys, time
sys.path.insert(0, os.path.expanduser('~/TanitAD/stack'))
sys.path.insert(0, os.path.expanduser('~/TanitAD/stack/scripts'))
import dataclasses
from types import SimpleNamespace
import torch
from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel
from train_flagship_v4 import resolve_v2_frames

DEV, H, W = 'cuda', 176, 624
cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection='cylindrical', v2_subframe='176x624', f_ref=None)
resolve_v2_frames(ns, cfg, label='ksweep')
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, 'tactical_pred', None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
model = WorldModel(cfg).to(DEV).eval()
Wn, C, A = cfg.predictor.window, cfg.encoder.in_channels, cfg.predictor.action_dim
frames = torch.randn(1, Wn, C, H, W, device=DEV)
ENC_BF16_MS = 27.8
out = {'device': torch.cuda.get_device_name(0), 'encoder_bf16_ms': ENC_BF16_MS,
       'imag_read_horizons': [5, 10, 15, 20]}

def bench(fn, warmup=4, iters=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    ts.sort(); return round(ts[len(ts)//2], 2)

with torch.no_grad():
    st0 = model.encode_window(frames); acts = torch.randn(1, Wn, A, device=DEV)
    rows = {}
    for K in (4, 8, 10, 12, 16, 20):
        def roll(k=K):
            ws = st0
            for _ in range(k):
                z = model.predictor(ws, acts)[1]
                ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
            return ws
        ms = bench(roll)
        rows[f'K{K}'] = {'roll_ms': ms, 'tick_ms': round(ENC_BF16_MS + ms, 2),
                         'ms_per_step': round(ms/K, 3),
                         'pct_of_100ms_budget': round(ENC_BF16_MS + ms, 1)}
    out['sweep'] = rows
    k20 = rows['K20']['roll_ms']
    out['linearity_check'] = {
        'ms_per_step_K4': rows['K4']['ms_per_step'],
        'ms_per_step_K20': rows['K20']['ms_per_step'],
        'verdict': 'LINEAR in K (compute-bound confirmed)' if abs(rows['K4']['ms_per_step']-rows['K20']['ms_per_step'])/rows['K20']['ms_per_step'] < 0.15 else 'NON-LINEAR — there is fixed overhead to attack'}
    out['savings_vs_K20'] = {k: round(k20 - v['roll_ms'], 2) for k, v in rows.items()}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(os.path.expanduser('~/thor_ksweep.json'),'w'), indent=1)
