"""Paired ADE on ALL 881 windows and on the 270 LEAD windows only — so 'does distance-keeping
agree with ADE?' is answered on the SAME window subset, not across surfaces."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval')]
from taniteval.ci import paired_episode_cluster_bootstrap, episode_cluster_bootstrap
panel = json.load(open(sys.argv[1]))
ST = np.array(sum(([s]*0 for s in []), []))     # placeholder
# rebuild the state vector from the panel's per-episode counts is lossy; reload from the scorer out
lead = json.load(open(sys.argv[2]))
d0 = torch.load(REPO/'taniteval/results/windows_refc-base-30k.pt', map_location='cpu', weights_only=False)
d1 = torch.load(REPO/'taniteval/results/windows_flagship-30k.pt', map_location='cpu', weights_only=False)
gt = d0['gt'].double().numpy()
arms = {'refc-base-30k': d0['pred'].double().numpy(),
        'flagship-30k': d1['pred'].double().numpy(),
        'cv': d0['cv'].double().numpy()}
eid = [str(e) for e in d0['eid']]
# per-window ADE over the 4 waypoints (0.5..2.0 s) — the canonical ade_0_2s form
ade = {k: np.linalg.norm(v - gt, axis=-1).mean(1) for k, v in arms.items()}
st = np.load(sys.argv[3], allow_pickle=True)['state'].astype(str)
sub = {'all_881': np.ones(881, bool), 'LEAD_only': st == 'LEAD', 'NO_LEAD_only': st == 'NO_LEAD'}
out = {}
for sname, m in sub.items():
    e = [x for x, keep in zip(eid, m) if keep]
    blk = {'n': int(m.sum()), 'n_episodes': len(set(e)), 'point': {}, 'paired': {}}
    for a in arms:
        blk['point'][a] = episode_cluster_bootstrap(ade[a][m], e, n_boot=2000, seed=0)
    for a, b in [('refc-base-30k', 'flagship-30k'), ('refc-base-30k', 'cv'), ('flagship-30k', 'cv')]:
        blk['paired'][f'{a}_minus_{b}'] = paired_episode_cluster_bootstrap(
            ade[a][m], ade[b][m], e, n_boot=2000, seed=0)
    out[sname] = blk
    print(f"-- {sname}: n={blk['n']} eps={blk['n_episodes']}")
    for a in arms:
        p = blk['point'][a]; print(f"   ADE {a:<15} {p['mean']:.4f} [{p['lo']:.4f}, {p['hi']:.4f}]")
    for k, v in blk['paired'].items():
        print(f"   {k:<34} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] sep={v['separated']}")
Path(sys.argv[4]).write_text(json.dumps(out, indent=1, default=str))
