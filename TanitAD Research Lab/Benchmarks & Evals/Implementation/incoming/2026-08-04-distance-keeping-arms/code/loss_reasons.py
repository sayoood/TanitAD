import io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval'), str(REPO/'stack'/'scripts')]
from tanitad.data.mixing import load_episode
from taniteval.lead_source import window_last_indices, register_poses_to_time, lead_block, LEAD
from taniteval.lead_metrics import per_step_gap, LEAD_LAT_M
from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw
DATA = Path('C:/Users/Admin/tanitad-data/physicalai'); VIEW = Path(sys.argv[1])
sel = pd.read_parquet(DATA/'r0'/'phase0_selection.parquet'); sel['clip_id']=sel['clip_id'].astype(str)
ids = sel['clip_id'].tolist(); chunk_of = dict(zip(sel['clip_id'], sel['chunk'].astype(int)))
def mem(z,c):
    for n in z.namelist():
        if n.endswith('.parquet') and n.split('/')[-1].startswith(c): return n
WP = np.array([0.5,1.0,1.5,2.0]); L,LN,ST = [],[],[]
for p in sorted(VIEW.glob('ep_*.pt')):
    ep = load_episode(p, mmap=False); poses = np.asarray(ep.poses, dtype=np.float64)
    last = window_last_indices(int(poses.shape[0]))
    pref = int(ep.episode_id).to_bytes(4,'big').decode('ascii','replace')
    cid = [c for c in ids if c.startswith(pref)][0]; ch = chunk_of[cid]
    with zipfile.ZipFile(DATA/f'labels/egomotion/egomotion.chunk_{ch:04d}.zip') as ez:
        df = pd.read_parquet(io.BytesIO(ez.read(mem(ez,cid))))
    t = df['timestamp'].to_numpy(np.float64)/1e6; o = np.argsort(t)
    g = lambda c: df[c].to_numpy(np.float64)[o]
    ego = {'t':t[o],'x':g('x'),'y':g('y'),
           'yaw':np.unwrap(quaternion_yaw(g('qx'),g('qy'),g('qz'),g('qw'))),
           'v':np.hypot(g('vx'),g('vy'))}
    reg = register_poses_to_time(poses[:,:2], ego['t'], ego['x'], ego['y'])
    obs = None
    ozp = DATA/f'labels/obstacle.offline/obstacle.offline.chunk_{ch:04d}.zip'
    if ozp.exists():
        with zipfile.ZipFile(ozp) as oz:
            mo = mem(oz, cid)
            if mo:
                o2 = pd.read_parquet(io.BytesIO(oz.read(mo)))
                obs = {'t':o2['timestamp_us'].to_numpy(np.float64)/1e6,
                       'track':o2['track_id'].astype(str).to_numpy(object),
                       'center_x':o2['center_x'].to_numpy(np.float64),
                       'center_y':o2['center_y'].to_numpy(np.float64),
                       'size_x':o2['size_x'].to_numpy(np.float64),
                       'is_vehicle':o2['label_class'].astype(str).isin(VEHICLE_CLASSES).to_numpy()}
    b = lead_block(reg['t_s'][last], WP, obs, ego)
    L.append(b['leads']); LN.append(b['lead_lens']); ST.append(b['state'])
L = np.concatenate(L); LN = np.concatenate(LN); ST = np.concatenate(ST)
dumps = {a: torch.load(REPO/'taniteval/results'/f'windows_{a}.pt', map_location='cpu', weights_only=False)
         for a in ['refc-base-30k','flagship-30k']}
arms = {'refc-base-30k': dumps['refc-base-30k']['pred'].numpy().astype(np.float64),
        'flagship-30k': dumps['flagship-30k']['pred'].numpy().astype(np.float64),
        'cv': dumps['refc-base-30k']['cv'].numpy().astype(np.float64),
        'gt_oracle': dumps['refc-base-30k']['gt'].numpy().astype(np.float64)}
idx = np.flatnonzero(ST == LEAD)
rep = {}
for name, P in arms.items():
    kept = overshoot = outside = nolead_track = 0
    for i in idx:
        gap, lat, ok = per_step_gap(P[i], L[i], LN[i], lat_m=LEAD_LAT_M)
        fin = np.isfinite(gap) & np.isfinite(lat)
        if not fin.any(): nolead_track += 1; continue
        if ok.any(): kept += 1; continue
        if (gap[fin] < 0).any(): overshoot += 1
        else: outside += 1
    rep[name] = {'n_lead_windows': int(idx.size), 'kept': kept,
                 'lost_overshoot_gap_lt_0': overshoot,
                 'lost_outside_corridor_only': outside,
                 'lost_no_finite_lead_sample': nolead_track,
                 'keep_rate': round(kept/idx.size, 4)}
    print(name, json.dumps(rep[name]))
Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
