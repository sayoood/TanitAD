'''The missing lever: bf16 on the PREDICTOR ROLL.

The encoder got 6.76x from bf16. The roll — now 71%% of the tick — is still fp32, and the
full-roll graph capture just proved it is COMPUTE-bound (one-graph capture bought 1.02%%, i.e.
launch overhead is already gone). Compute-bound + fp32 => precision is the untried lever.

G-P2: accuracy beside every speed delta, against the eager fp32 reference.
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

DEV, H, W, K = 'cuda', 176, 624, 20
cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection='cylindrical', v2_subframe='176x624', f_ref=None)
resolve_v2_frames(ns, cfg, label='bf16roll')
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, 'tactical_pred', None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
model = WorldModel(cfg).to(DEV).eval()
Wn, C, A = cfg.predictor.window, cfg.encoder.in_channels, cfg.predictor.action_dim
frames = torch.randn(1, Wn, C, H, W, device=DEV)
out = {'device': torch.cuda.get_device_name(0)}

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    ts.sort()
    return {'p50_ms': round(ts[len(ts)//2],2), 'p99_ms': round(ts[min(len(ts)-1,int(len(ts)*0.99))],2)}

with torch.no_grad():
    st0 = model.encode_window(frames); acts = torch.randn(1, Wn, A, device=DEV)
    def roll(dtype=None):
        ws = st0
        for _ in range(K):
            if dtype is None:
                z = model.predictor(ws, acts)[1]
            else:
                with torch.autocast('cuda', dtype=dtype):
                    z = model.predictor(ws, acts)[1]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws
    ref = roll().float().clone()
    out['roll_fp32'] = bench(lambda: roll())
    for name, dt in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        r = bench(lambda: roll(dt))
        o = roll(dt).float()
        rel = float((o-ref).norm()/ref.norm())
        out[f'roll_{name}'] = r
        out[f'roll_{name}_rel_err'] = round(rel, 6)
        out[f'roll_{name}_speedup_x'] = round(out['roll_fp32']['p50_ms']/r['p50_ms'], 2)
    best = min(out['roll_bf16']['p50_ms'], out['roll_fp16']['p50_ms'])
    out['projected_tick_ms'] = round(27.8 + best, 2)
    out['projected_speedup_vs_272ms_x'] = round(272.56/out['projected_tick_ms'], 2)
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(os.path.expanduser('~/thor_bf16_roll.json'),'w'), indent=1)
