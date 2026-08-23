"""LOAD-VERIFY: flagship v1 (speedjerk) + REF-C at the 256px SQUARE geometry,
against the raw val epcache that survives on pod2.

Never claims success from an exit code: strict-loads the real checkpoints,
builds the real window dataset over real episodes and runs a real forward.
"""
import glob
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, '/workspace/TanitAD/stack')
sys.path.insert(0, '/workspace/TanitAD/stack/scripts')

OUT = {}
EPC = '/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11'
V1 = '/workspace/v4gate30k/v1_speedjerk_ckpt.pt'
RC_BASE = '/workspace/models/refc-base-30k/ckpt.pt'
RC_XL = '/workspace/models/refc-xl-30k/ckpt.pt'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# code provenance for the modules this proof depends on
OUT['code_md5'] = {p: md5('/workspace/TanitAD/stack/' + p) for p in (
    'tanitad/data/calib.py', 'tanitad/models/fourbrain.py',
    'tanitad/config.py', 'tanitad/data/parity.py')}

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT['device'] = dev

# --------------------------------------------------------------- v1 geometry
t0 = time.time()
ck = torch.load(V1, map_location='cpu', weights_only=False)
sd = ck['model']
pos = sd['encoder.pos']
OUT['v1'] = {'ckpt': V1, 'step': int(ck.get('step', -1)),
             'encoder.pos': list(pos.shape),
             'action_dim': int(sd['predictor.act_emb.0.weight'].shape[1]),
             'load_s': round(time.time() - t0, 1)}

from eval_flagship_v4 import _eval_cfg, _plan, load_v1_from_ck   # noqa: E402
from tanitad.data.calib import CANONICAL_256                      # noqa: E402
from tanitad.data.mixing import load_episode                      # noqa: E402

world, grounding, step = load_v1_from_ck(ck, dev, frame=None)
cfg = _eval_cfg(None)
plan = _plan(cfg)
OUT['v1']['loaded_strict'] = True
OUT['v1']['state_dim'] = int(world.state_dim)
OUT['v1']['n_params_M'] = round(sum(p.numel() for p in world.parameters()) / 1e6, 2)
OUT['v1']['cfg_frame'] = CANONICAL_256.to_dict()
OUT['v1']['encoder_in_channels'] = int(cfg.encoder.in_channels)

# ------------------------------------------------------- real episodes + fwd
N_EP = 5
eps = [load_episode(os.path.join(EPC, 'ep_%05d.pt' % i), mmap=True)
       for i in range(N_EP)]
OUT['episodes'] = [{'i': i, 'frames': list(e.frames.shape),
                    'poses': list(e.poses.shape), 'actions': list(e.actions.shape),
                    'dtype': str(e.frames.dtype)} for i, e in enumerate(eps)]

from train_flagship4b import FlagshipWindowDataset                # noqa: E402

ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon,
                           maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
OUT['dataset'] = {'n_windows_5ep': len(ds), 'window': int(cfg.predictor.window),
                  'max_horizon': int(plan.max_horizon)}
stride8 = [i for i, (e, t) in enumerate(ds.index) if t % 8 == 0]
OUT['dataset']['n_stride8_windows_5ep'] = len(stride8)

b = torch.utils.data.default_collate([ds[i] for i in stride8[:4]])
OUT['batch_keys'] = {k: list(v.shape) for k, v in b.items()
                     if isinstance(v, torch.Tensor)}

with torch.no_grad():
    frames = b['frames'].to(dev).float() / 255.0
    enc = world.encoder(frames.flatten(0, 1)) if frames.ndim == 5 else world.encoder(frames)
OUT['forward'] = {'input': list(frames.shape), 'encoder_out': list(enc.shape),
                  'finite': bool(torch.isfinite(enc).all()),
                  'mean': float(enc.mean()), 'std': float(enc.std())}

# ------------------------------------------------------------------- REF-C
for name, p in (('refc_base', RC_BASE), ('refc_xl', RC_XL)):
    if not os.path.exists(p):
        OUT[name] = {'MISSING': p}
        continue
    c = torch.load(p, map_location='cpu', weights_only=False)
    cf = c.get('config') or c.get('cfg') or {}
    rec = {'ckpt': p, 'step': int(c.get('step', -1)),
           'keys': sorted(k for k in c.keys() if k != 'model')[:12]}
    if isinstance(cf, dict):
        rec['grid_shape'] = cf.get('grid_shape')
        rec['image_size'] = cf.get('image_size')
        rec['config_keys'] = sorted(cf.keys())[:25]
    else:
        rec['grid_shape'] = getattr(cf, 'grid_shape', None)
        rec['image_size'] = getattr(cf, 'image_size', None)
    m = c.get('model') or {}
    rec['n_tensors'] = len(m)
    for k in list(m)[:400]:
        if 'pos' in k and hasattr(m[k], 'shape') and len(m[k].shape) >= 2:
            rec.setdefault('pos_like', {})[k] = list(m[k].shape)
    OUT[name] = rec

print(json.dumps(OUT, indent=1, default=str))
