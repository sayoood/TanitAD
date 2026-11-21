"""Bank the val40 LEAD BLOCK as a standalone npz, row-aligned with the canonical 881 windows.

⭐ Why this artifact: once it exists, ANY banked per-window `pred` dump can be scored for
distance-keeping with zero label I/O, no gated-zip access and no episode cache — the expensive,
gated part of the pipeline is done once and travels as ~60 KB.
Row order = `window_last_indices` emission order over sorted(ep_*.pt) = every dump's row order.
"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval'), str(REPO/'stack'/'scripts')]
from tanitad.data.mixing import load_episode
from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD, lead_block,
                                   register_poses_to_time, window_last_indices)
from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw
VIEW, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sel = pd.read_parquet(DATA/'r0'/'phase0_selection.parquet'); sel['clip_id'] = sel['clip_id'].astype(str)
ids = sel['clip_id'].tolist(); chunk_of = dict(zip(sel['clip_id'], sel['chunk'].astype(int)))
def mem(z, c):
    for n in z.namelist():
        if n.endswith('.parquet') and n.split('/')[-1].startswith(c): return n
WP = np.array([0.5, 1.0, 1.5, 2.0])
L, LN, SP, ST, G0, EI, EP = [], [], [], [], [], [], []
for p in sorted(VIEW.glob('ep_*.pt')):
    ep = load_episode(p, mmap=False); poses = np.asarray(ep.poses, dtype=np.float64)
    last = window_last_indices(int(poses.shape[0]))
    pref = int(ep.episode_id).to_bytes(4, 'big').decode('ascii', 'replace')
    cid = [c for c in ids if c.startswith(pref)][0]; ch = chunk_of[cid]
    with zipfile.ZipFile(DATA/f'labels/egomotion/egomotion.chunk_{ch:04d}.zip') as ez:
        df = pd.read_parquet(io.BytesIO(ez.read(mem(ez, cid))))
    t = df['timestamp'].to_numpy(np.float64)/1e6; o = np.argsort(t)
    g = lambda c: df[c].to_numpy(np.float64)[o]
    ego = {'t': t[o], 'x': g('x'), 'y': g('y'),
           'yaw': np.unwrap(quaternion_yaw(g('qx'), g('qy'), g('qz'), g('qw'))),
           'v': np.hypot(g('vx'), g('vy'))}
    reg = register_poses_to_time(poses[:, :2], ego['t'], ego['x'], ego['y'])
    obs = None
    ozp = DATA/f'labels/obstacle.offline/obstacle.offline.chunk_{ch:04d}.zip'
    if ozp.exists():
        with zipfile.ZipFile(ozp) as oz:
            mo = mem(oz, cid)
            if mo:
                o2 = pd.read_parquet(io.BytesIO(oz.read(mo)))
                obs = {'t': o2['timestamp_us'].to_numpy(np.float64)/1e6,
                       'track': o2['track_id'].astype(str).to_numpy(object),
                       'center_x': o2['center_x'].to_numpy(np.float64),
                       'center_y': o2['center_y'].to_numpy(np.float64),
                       'size_x': o2['size_x'].to_numpy(np.float64),
                       'is_vehicle': o2['label_class'].astype(str).isin(VEHICLE_CLASSES).to_numpy()}
    b = lead_block(reg['t_s'][last], WP, obs, ego)
    L.append(b['leads']); LN.append(b['lead_lens']); SP.append(b['speeds'])
    ST.append(b['state'].astype(str)); G0.append(b['gap0_m'])
    EI.extend([p.stem]*last.size)
    EP.append({'file': p.name, 'n_windows': int(last.size), 'has_obstacle': obs is not None,
               'registration_residual_m': reg['residual_m']['median'], 'grid_dt_s': reg['grid_dt_s'],
               **{k: int(v) for k, v in b['counts'].items()}})
L = np.concatenate(L); LN = np.concatenate(LN); SP = np.concatenate(SP)
ST = np.concatenate(ST); G0 = np.concatenate(G0); EI = np.array(EI)
np.savez_compressed(OUT, leads=L, lead_lens=LN, speeds=SP, state=ST, gap0_m=G0, eid=EI,
                    ts_rel_s=WP, episodes_json=np.frombuffer(json.dumps(EP).encode(), np.uint8))
print(json.dumps({'n_windows': int(ST.size), 'n_episodes': len(EP),
                  'states': {k: int((ST == k).sum()) for k in (LEAD, NO_LEAD, NO_LABEL)},
                  'bytes': OUT.stat().st_size}))
