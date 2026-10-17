"""Build the POSE-ONLY P10's shipped files in stack/tanitad/configs/v8f_p10/: the artifact (byte copy of the banked fit) and the
known-value fixture the gate and the tests read. Everything is derived from the banked fit (`raw/p10_pose/`); nothing is re-fit.

The fixture's `p10_banked` are the PREDICTIONS THE FIT SCRIPT WROTE (`p10_pose_eval_pred6.npy`: numpy float64 pose features ->
`fk.predict_p10`), i.e. an implementation that shares no code with the torch port the gate and the tests compare against it.
Windows: the same 565 eval141 windows as the steer fixture (every 4th with a full 8-frame window). Poses are stored in float64:
the pose track's yaw difference is a ~1e-3 rad subtraction, and a float32 round-trip of the yaw would exceed the 1e-4 m
tolerance at 6 s on fast windows.
"""
import hashlib
import json
import os
import shutil

import numpy as np
import torch

ROOT = 'C:/Users/Admin/v8fa1/tree'
PKG = f'{ROOT}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha'
RAW = f'{ROOT}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw'
FIT = f'{PKG}/raw/p10_pose'
OUT = f'{ROOT}/stack/tanitad/configs/v8f_p10'
os.makedirs(OUT, exist_ok=True)
shutil.copyfile(f'{FIT}/p10_pose_prior.pt', f'{OUT}/p10_pose_prior.pt')
sha = hashlib.sha256(open(f'{OUT}/p10_pose_prior.pt', 'rb').read()).hexdigest()
assert sha == json.load(open(f'{FIT}/p10_fit.json'))['artifact_sha256']

W = 8
z = np.load(f'{RAW}/eval141_kin.npz', allow_pickle=True)
g = json.load(open(f'{RAW}/eval_grid_windows.json'))
pred = np.load(f'{FIT}/p10_pose_eval_pred6.npy')
off = np.concatenate([[0], np.cumsum(z['n'])])
clips = z['clip'].tolist()
poses, banked, cidx, tsl = [], [], [], []
n = 0
for i, c in enumerate(clips):
    P = z['poses'][off[i]:off[i + 1]]
    ts = np.asarray(g['windows'][c])
    for j, t in enumerate(ts):
        k = 2 * int(t)
        if k >= W - 1 and (n % 4 == 0):
            poses.append(P[k - W + 1:k + 1].astype(np.float64))
            banked.append(pred[n])
            cidx.append(i)
            tsl.append(int(t))
        n += 1
assert n == len(pred) == 2399, (n, len(pred))
print('fixture windows', len(poses))

mt = torch.load('C:/Users/Admin/s1_floors_scratch/train4572_v2manifest.pt', map_location='cpu', weights_only=False)
fit_ids = [str(c) for c in mt['clip_id']]
me = torch.load('C:/Users/Admin/s1_floors_scratch/eval139_v2manifest.pt', map_location='cpu', weights_only=False)
e139 = [str(c) for c in me['clip_id']]
open_ids = [l for l in open(f'{FIT}/fit_clip_ids.txt', encoding='utf-8').read().split('\n') if l]
assert sorted(open_ids) == sorted(fit_ids)
np.savez_compressed(f'{OUT}/p10_pose_fixture.npz', poses=np.stack(poses), p10_banked=np.stack(banked).astype(np.float32),
                    clip_index=np.asarray(cidx), ts=np.asarray(tsl), eval141_clip_ids=np.asarray(clips),
                    fit_clip_ids=np.asarray(fit_ids), eval139_clip_ids=np.asarray(e139))
man_p = f'{OUT}/MANIFEST.json'
man = json.load(open(man_p))
man['p10_pose_prior.pt'] = {
    'sha256': sha, 'bytes': os.path.getsize(f'{OUT}/p10_pose_prior.pt'), 'units': 'pose_kappa',
    'source': 'TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha/raw/p10_pose/p10_pose_prior.pt',
    'fit': 'code/fit_p10_pose.py: ALL 4,572 train clips (train4572 view, manifest md5 77dafb50af0aedd1b940cf12c12518ef) by sha12, '
           'stride 8, 41,139 rows, pose-only kinematics (v0, ONE-frame a0, pose kappa0), seed 0, selection wd 1e-4 / 59 epochs'}
man['p10_pose_fixture.npz'] = {'windows': len(poses), 'window_len': W, 'poses_dtype': 'float64',
                               'source': 'eval141_kin.npz + raw/p10_pose/p10_pose_eval_pred6.npy (the fit script\'s own numpy-float64 '
                                         'predictions; every 4th window with a full 8-frame window)'}
json.dump(man, open(man_p, 'w'), indent=1)
print('p10 pose file sha256', sha, 'bytes', os.path.getsize(f'{OUT}/p10_pose_prior.pt'))
print('fit/eval141 overlap', len(set(fit_ids) & set(clips)), 'fit/eval139 overlap', len(set(fit_ids) & set(e139)))
