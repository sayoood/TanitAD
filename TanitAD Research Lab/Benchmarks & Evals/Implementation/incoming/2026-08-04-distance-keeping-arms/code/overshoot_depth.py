"""How DEEP is each arm's overshoot: first horizon step at which gap goes negative, and how many
admissible steps survive. Reconciles 'overshoot at some step' with 'lost the lead entirely'."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval')]
from taniteval.lead_metrics import per_step_gap, LEAD_LAT_M
from taniteval.ci import paired_episode_cluster_bootstrap
z = np.load(sys.argv[1], allow_pickle=True)
L, LN, ST, EI = z['leads'], z['lead_lens'], z['state'].astype(str), z['eid'].astype(str)
d0 = torch.load(REPO/'taniteval/results/windows_refc-base-30k.pt', map_location='cpu', weights_only=False)
d1 = torch.load(REPO/'taniteval/results/windows_flagship-30k.pt', map_location='cpu', weights_only=False)
arms = {'refc-base-30k': d0['pred'].double().numpy(), 'flagship-30k': d1['pred'].double().numpy(),
        'cv': d0['cv'].double().numpy(), 'gt_oracle': d0['gt'].double().numpy()}
idx = np.flatnonzero(ST == 'LEAD'); WP = [0.5, 1.0, 1.5, 2.0]
out = {'_what': 'overshoot depth over the 270 LEAD windows', 'n_lead_windows': int(idx.size), 'arms': {}}
depth = {}
for a, P in arms.items():
    first, nadm, ovr, deepest = [], [], 0, np.zeros(idx.size)
    for j, i in enumerate(idx):
        gap, lat, ok = per_step_gap(P[i], L[i], LN[i], lat_m=LEAD_LAT_M)
        fin = np.isfinite(gap)
        nadm.append(int(ok.sum()))
        neg = fin & (gap < 0)
        if neg.any():
            ovr += 1
            first.append(WP[int(np.flatnonzero(neg)[0])])
            deepest[j] = float(-gap[neg].min())     # how far past the rear face, metres
    fc = {f'{t}s': int(sum(1 for x in first if x == t)) for t in WP}
    out['arms'][a] = {'n_overshoot_windows': ovr,
                      'first_negative_step_hist': fc,
                      'mean_steps_in_corridor_over_lead': round(float(np.mean(nadm)), 3),
                      'n_zero_steps_in_corridor': int(sum(1 for x in nadm if x == 0)),
                      'max_overrun_past_rear_face_m': round(float(deepest.max()), 3),
                      'mean_overrun_when_overshooting_m': round(
                          float(deepest[deepest > 0].mean()) if (deepest > 0).any() else 0.0, 3)}
    depth[a] = deepest
    print(a, json.dumps(out['arms'][a]))
out['paired_mean_overrun_m'] = {}
for a, b in [('flagship-30k', 'refc-base-30k'), ('flagship-30k', 'gt_oracle'),
             ('refc-base-30k', 'gt_oracle')]:
    v = paired_episode_cluster_bootstrap(depth[a], depth[b], list(EI[idx]), n_boot=2000, seed=0)
    out['paired_mean_overrun_m'][f'{a}_minus_{b}'] = v
    print(f"  overrun {a}-{b}: {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] sep={v['separated']}")
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
