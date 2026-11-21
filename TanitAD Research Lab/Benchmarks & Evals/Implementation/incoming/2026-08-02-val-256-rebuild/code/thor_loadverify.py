"""THOR-SIDE LOAD-VERIFY of the relayed 256px-SQUARE val episodes.

Proves the transferred artifact is usable ON THOR: strict-loads flagship v1
(speedjerk) at CANONICAL_256, builds the real window dataset over the relayed
episodes and runs a real encoder forward on Thor's Blackwell GPU.
"""
import glob
import hashlib
import json
import os
import sys
import time

import torch

STACK = '/home/nvidia/TanitAD/stack'
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, 'scripts'))

OUT = {}
VAL = '/home/nvidia/valdata/physicalai-val-0c5f7dac3b11'
V1 = '/home/nvidia/models/flagship-v1-speedjerk/ckpt.pt'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


OUT['host'] = os.uname().nodename
OUT['torch'] = torch.__version__
OUT['cuda'] = torch.cuda.is_available()
if OUT['cuda']:
    OUT['gpu'] = torch.cuda.get_device_name(0)
    OUT['sm'] = list(torch.cuda.get_device_capability(0))
OUT['code_md5'] = {p: md5(os.path.join(STACK, p)) for p in (
    'tanitad/data/calib.py', 'tanitad/models/fourbrain.py',
    'tanitad/config.py', 'tanitad/data/parity.py')}

files = sorted(glob.glob(os.path.join(VAL, 'ep_*.pt')))
OUT['relayed_files'] = [{'f': os.path.basename(p), 'bytes': os.path.getsize(p),
                         'md5': md5(p)} for p in files]

# ------------------------------------------------------- parity guard on Thor
from tanitad.data import parity                                   # noqa: E402
try:
    OUT['parity'] = parity.assert_val_cache(VAL, label='thor-relay')
except SystemExit as e:
    OUT['parity'] = {'REFUSED': str(e)[:600]}

# --------------------------------------------------------------- load + fwd
from eval_flagship_v4 import _eval_cfg, _plan, load_v1_from_ck    # noqa: E402
from tanitad.data.calib import CANONICAL_256                      # noqa: E402
from tanitad.data.mixing import load_episode                      # noqa: E402
from train_flagship4b import FlagshipWindowDataset                # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
t0 = time.time()
ck = torch.load(V1, map_location='cpu', weights_only=False)
sd = ck['model']
OUT['v1'] = {'ckpt': V1, 'ckpt_md5': md5(V1), 'step': int(ck.get('step', -1)),
             'encoder.pos': list(sd['encoder.pos'].shape),
             'action_dim': int(sd['predictor.act_emb.0.weight'].shape[1]),
             'load_s': round(time.time() - t0, 1)}
world, grounding, step = load_v1_from_ck(ck, dev, frame=None)
cfg = _eval_cfg(None)
plan = _plan(cfg)
OUT['v1']['loaded_strict'] = True
OUT['v1']['state_dim'] = int(world.state_dim)
OUT['v1']['n_params_M'] = round(sum(p.numel() for p in world.parameters()) / 1e6, 2)
OUT['v1']['cfg_frame'] = CANONICAL_256.to_dict()

eps = [load_episode(p, mmap=True) for p in files]
OUT['episodes'] = [{'f': os.path.basename(p), 'frames': list(e.frames.shape),
                    'poses': list(e.poses.shape), 'dtype': str(e.frames.dtype)}
                   for p, e in zip(files, eps)]

ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon,
                           maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
stride8 = [i for i, (e, t) in enumerate(ds.index) if t % 8 == 0]
OUT['dataset'] = {'n_episodes': len(eps), 'n_windows': len(ds),
                  'n_stride8_windows': len(stride8),
                  'window': int(cfg.predictor.window),
                  'max_horizon': int(plan.max_horizon)}

b = torch.utils.data.default_collate([ds[i] for i in stride8[:4]])
t0 = time.time()
with torch.no_grad():
    frames = b['frames'].to(dev).float() / 255.0
    enc = world.encoder(frames.flatten(0, 1))
if dev == 'cuda':
    torch.cuda.synchronize()
OUT['forward'] = {'input': list(frames.shape), 'encoder_out': list(enc.shape),
                  'finite': bool(torch.isfinite(enc).all()),
                  'mean': float(enc.mean()), 'std': float(enc.std()),
                  'wall_s': round(time.time() - t0, 3)}

print(json.dumps(OUT, indent=1, default=str))
