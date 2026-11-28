"""Build stack/tanitad/configs/v8f_p10/: the P10 artifact (byte copy) + the known-value fixture the gate and the tests
read. Everything here is derived from banked artifacts of the tip; nothing is re-fit."""
import hashlib, json, shutil, sys
import numpy as np, torch

ROOT = 'C:/Users/Admin/v8fa1/tree'
RAW = f'{ROOT}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw'
OUT = f'{ROOT}/stack/tanitad/configs/v8f_p10'
import os
os.makedirs(OUT, exist_ok=True)
shutil.copyfile(f'{RAW}/p10/p10_prior.pt', f'{OUT}/p10_prior.pt')
sha = hashlib.sha256(open(f'{OUT}/p10_prior.pt', 'rb').read()).hexdigest()

W = 8
z = np.load(f'{RAW}/eval141_kin.npz', allow_pickle=True)
g = json.load(open(f'{RAW}/eval_grid_windows.json'))
pred = np.load(f'{RAW}/p10/p10_eval_pred6.npy')
off = np.concatenate([[0], np.cumsum(z['n'])])
clips = z['clip'].tolist()
poses, acts, banked, cidx, tsl = [], [], [], [], []
n = 0
for i, c in enumerate(clips):
    P = z['poses'][off[i]:off[i + 1]]
    A = z['actions'][off[i]:off[i + 1]]
    ts = np.asarray(g['windows'][c])
    for j, t in enumerate(ts):
        k = 2 * int(t)
        if k >= W - 1 and (n % 4 == 0):
            poses.append(P[k - W + 1:k + 1]); acts.append(A[k - W + 1:k + 1])
            banked.append(pred[n]); cidx.append(i); tsl.append(int(t))
        n += 1
assert n == len(pred) == 2399, (n, len(pred))
print('fixture windows', len(poses))

# disjointness evidence: the fit's clips (first 600 of the train4572 manifest by sha12) vs eval141 / eval139
sha12 = lambda c: hashlib.sha256(str(c).encode()).hexdigest()[:12]
mt = torch.load('C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt', map_location='cpu', weights_only=False)
fit_ids = sorted((str(c) for c in mt['clip_id']), key=sha12)[:600]
me = torch.load('C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt', map_location='cpu', weights_only=False)
e139 = [str(c) for c in me['clip_id']]
np.savez_compressed(f'{OUT}/p10_fixture.npz', poses=np.stack(poses).astype(np.float32), actions=np.stack(acts).astype(np.float32),
                    p10_banked=np.stack(banked).astype(np.float32), clip_index=np.asarray(cidx), ts=np.asarray(tsl),
                    eval141_clip_ids=np.asarray(clips), fit_clip_ids=np.asarray(fit_ids), eval139_clip_ids=np.asarray(e139))
json.dump({'p10_prior.pt': {'sha256': sha, 'bytes': os.path.getsize(f'{OUT}/p10_prior.pt'),
                            'source': 'TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/p10/p10_prior.pt',
                            'fit': 'v8f_mini_fit_p10.py: first 600 train4572 clips by sha12, stride 8, 5,398 rows, steer unit, seed 0'},
           'p10_fixture.npz': {'windows': len(poses), 'window_len': W, 'source': 'eval141_kin.npz + p10_eval_pred6.npy (every 4th window with a full 8-frame window)'}},
          open(f'{OUT}/MANIFEST.json', 'w'), indent=1)
print('p10 file sha256', sha)
print('fit/eval141 overlap', len(set(fit_ids) & set(clips)), 'fit/eval139 overlap', len(set(fit_ids) & set(e139)))
