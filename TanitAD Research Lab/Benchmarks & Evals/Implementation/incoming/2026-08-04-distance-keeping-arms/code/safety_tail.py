"""Contact-precursor and close-following rates per arm, with BOTH denominators."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval')]
from taniteval.lead_metrics import distance_keeping
from taniteval.ci import paired_episode_cluster_bootstrap
z = np.load(sys.argv[1], allow_pickle=True)
L, LN, SP, ST, EI = z['leads'], z['lead_lens'], z['speeds'], z['state'].astype(str), z['eid'].astype(str)
d0 = torch.load(REPO/'taniteval/results/windows_refc-base-30k.pt', map_location='cpu', weights_only=False)
d1 = torch.load(REPO/'taniteval/results/windows_flagship-30k.pt', map_location='cpu', weights_only=False)
arms = {'refc-base-30k': d0['pred'].double().numpy(), 'flagship-30k': d1['pred'].double().numpy(),
        'cv': d0['cv'].double().numpy(), 'gt_oracle': d0['gt'].double().numpy()}
retention = json.load(open(sys.argv[2]))
n_lead = int((ST == 'LEAD').sum())
out = {'n_windows': int(ST.size), 'n_lead_windows': n_lead,
       'denominators': {'all_windows': int(ST.size), 'lead_windows': n_lead,
                        'note': 'rates are reported over BOTH; a rate over the arm-kept subset '
                                'alone is survivorship-selected and is never the headline'},
       'arms': {}}
per = {}
for a, P in arms.items():
    dk = distance_keeping(P, L, LN, SP, 0.5)
    hw = dk['headway_min_m']; per[a] = hw
    kept = int(np.isfinite(hw).sum())
    ov = retention[a]['lost_overshoot_gap_lt_0']
    out['arms'][a] = {
        'kept_windows': kept, 'keep_rate_over_lead': round(kept/n_lead, 4),
        'overshoot_gap_lt_0': ov, 'overshoot_rate_over_lead': round(ov/n_lead, 4),
        'lost_outside_corridor_only': retention[a]['lost_outside_corridor_only'],
        'min_headway_lt_2m': {'n': int((hw < 2).sum()),
                              'over_lead_windows': round(float((hw < 2).sum())/n_lead, 4),
                              'over_kept': round(float((hw < 2).sum())/max(kept, 1), 4)},
        'min_headway_lt_5m': {'n': int((hw < 5).sum()),
                              'over_lead_windows': round(float((hw < 5).sum())/n_lead, 4),
                              'over_kept': round(float((hw < 5).sum())/max(kept, 1), 4)},
        'contact_precursor_total': {
            'n': ov + int((hw < 2).sum()),
            'definition': 'the predicted path either passes the lead rear face (gap<0 at some '
                          'step) OR closes to under 2 m of it',
            'over_lead_windows': round((ov + int((hw < 2).sum()))/n_lead, 4)},
    }
# paired overshoot-indicator delta, episode-clustered
lead_idx = np.flatnonzero(ST == 'LEAD')
ind = {a: (~np.isfinite(per[a][lead_idx])).astype(float) for a in arms}   # lost the lead at all
out['paired_lost_lead_rate'] = {}
for a, b in [('flagship-30k', 'refc-base-30k'), ('flagship-30k', 'gt_oracle'),
             ('refc-base-30k', 'gt_oracle')]:
    out['paired_lost_lead_rate'][f'{a}_minus_{b}'] = paired_episode_cluster_bootstrap(
        ind[a], ind[b], list(EI[lead_idx]), n_boot=2000, seed=0)
print(json.dumps(out, indent=1))
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
