"""Paired episode-cluster CI on the OVERSHOOT indicator (predicted path passes the lead rear face)."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval')]
from taniteval.lead_metrics import per_step_gap, LEAD_LAT_M
from taniteval.ci import paired_episode_cluster_bootstrap, episode_cluster_bootstrap
z = np.load(sys.argv[1], allow_pickle=True)
L, LN, ST, EI = z['leads'], z['lead_lens'], z['state'].astype(str), z['eid'].astype(str)
d0 = torch.load(REPO/'taniteval/results/windows_refc-base-30k.pt', map_location='cpu', weights_only=False)
d1 = torch.load(REPO/'taniteval/results/windows_flagship-30k.pt', map_location='cpu', weights_only=False)
arms = {'refc-base-30k': d0['pred'].double().numpy(), 'flagship-30k': d1['pred'].double().numpy(),
        'cv': d0['cv'].double().numpy(), 'gt_oracle': d0['gt'].double().numpy()}
idx = np.flatnonzero(ST == 'LEAD'); eids = list(EI[idx])
ind = {}
for a, P in arms.items():
    v = np.zeros(idx.size)
    for j, i in enumerate(idx):
        gap, lat, ok = per_step_gap(P[i], L[i], LN[i], lat_m=LEAD_LAT_M)
        fin = np.isfinite(gap)
        v[j] = float(fin.any() and (gap[fin] < 0).any())      # passes the lead's rear face
    ind[a] = v
out = {'_what': 'OVERSHOOT = the arm\'s predicted path passes the lead vehicle\'s rear face at some '
                'horizon step. A rear-end contact precursor, computed over the 270 LEAD windows.',
       '_estimator': 'episode-cluster bootstrap; paired for deltas. B=2000, seed 0.',
       'n_lead_windows': int(idx.size), 'n_episodes': len(set(eids)), 'rate': {}, 'paired': {}}
for a in arms:
    out['rate'][a] = {'n': int(ind[a].sum()), **episode_cluster_bootstrap(ind[a], eids, n_boot=2000, seed=0)}
    r = out['rate'][a]; print(f"{a:<16} {r['n']:>3}/270  rate {r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]")
for a, b in [('flagship-30k', 'refc-base-30k'), ('flagship-30k', 'gt_oracle'),
             ('refc-base-30k', 'gt_oracle'), ('flagship-30k', 'cv'), ('refc-base-30k', 'cv')]:
    v = paired_episode_cluster_bootstrap(ind[a], ind[b], eids, n_boot=2000, seed=0)
    out['paired'][f'{a}_minus_{b}'] = v
    print(f"  {a}-{b}: {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] sep={v['separated']}")
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
