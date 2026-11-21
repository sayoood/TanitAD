"""⭐ The programme's FIRST cross-arm distance-keeping ranking.

Uses the banked `raw/val40_lead_block.npz`, so every arm is scored on the identical 270 LEAD
windows with ZERO label I/O and no gated-zip access. Each arm is admitted only after its `gt`
is bit-identical to the reference dump's — a dump on a different surface is REFUSED, not rescaled.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(r'G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD')
sys.path[:0] = [str(REPO/'stack'), str(REPO/'taniteval')]
from taniteval.lead_metrics import distance_keeping, paired_distance_keeping
from taniteval.ci import episode_cluster_bootstrap
z = np.load(sys.argv[1], allow_pickle=True)
L, LN, SP, ST, EI = z['leads'], z['lead_lens'], z['speeds'], z['state'].astype(str), z['eid'].astype(str)
ref = torch.load(REPO/'taniteval/results/windows_refc-base-30k.pt', map_location='cpu', weights_only=False)
GT = ref['gt']
n_lead = int((ST == 'LEAD').sum())
rows, refused = [], []
dk_cv = distance_keeping(ref['cv'].double().numpy(), L, LN, SP, 0.5)
dk_gt = distance_keeping(GT.double().numpy(), L, LN, SP, 0.5)
arms = {'GT_oracle': GT.double().numpy(), 'CV_floor': ref['cv'].double().numpy()}
for p in sorted((REPO/'taniteval'/'results').glob('windows_*.pt')):
    name = p.stem[len('windows_'):]
    d = torch.load(p, map_location='cpu', weights_only=False)
    if d['pred'].shape[0] != GT.shape[0]:
        refused.append({'arm': name, 'reason': f"{d['pred'].shape[0]} rows, not {GT.shape[0]}"}); continue
    if not torch.equal(d['gt'], GT):
        refused.append({'arm': name, 'reason': 'gt differs from the reference dump — different surface'}); continue
    arms[name] = d['pred'].double().numpy()
out = {'_what': 'cross-arm LONGITUDINAL distance-keeping on the canonical val40 270 LEAD windows',
       '_estimator': 'episode-cluster bootstrap (point), paired vs the CV floor. B=2000 seed 0.',
       '_admission': 'gt bit-identical to windows_refc-base-30k.pt; 881 rows',
       'n_lead_windows': n_lead, 'n_arms': len(arms), 'refused': refused, 'arms': {}}
for name, P in arms.items():
    dk = distance_keeping(P, L, LN, SP, 0.5)
    hw = dk['headway_min_m']; ok = np.isfinite(hw)
    ttc_b = episode_cluster_bootstrap(dk['min_ttc_s'][ok], list(EI[ok]), n_boot=2000, seed=0)
    pd_ = paired_distance_keeping(dk, dk_cv, EI, names=(name, 'CV'), n_boot=2000)['metrics']
    out['arms'][name] = {
        'n_kept': int(ok.sum()), 'keep_rate': round(float(ok.sum())/n_lead, 4),
        'mean_headway_min_m': dk.get('mean_headway_min_m'),
        'mean_time_gap_min_s': dk.get('mean_time_gap_min_s'), 'n_time_gap': dk.get('n_time_gap'),
        'mean_min_ttc_s': dk.get('mean_min_ttc_s'), 'n_closing': dk.get('n_closing'),
        'min_ttc_ci': {'lo': ttc_b['lo'], 'hi': ttc_b['hi']},
        'vs_cv_paired': {k: {'delta': v.get('delta'), 'lo': v.get('lo'), 'hi': v.get('hi'),
                             'separated': v.get('separated'), 'n_used': v.get('n_used')}
                         for k, v in pd_.items()}}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
o = out['arms']
print(f"{len(o)} arms scored, {len(refused)} refused, over {n_lead} LEAD windows\n")
print(f"{'arm':<32}{'kept':>6}{'headway':>10}{'tgap':>8}{'minTTC':>9}{'dTTC vs CV':>26}")
for name, v in sorted(o.items(), key=lambda kv: -(kv[1]['vs_cv_paired']['min_ttc_s']['delta'] or -99)):
    d = v['vs_cv_paired']['min_ttc_s']
    s = '' if d['delta'] is None else f"{d['delta']:+.3f} [{d['lo']:+.2f},{d['hi']:+.2f}]{'*' if d['separated'] else ''}"
    print(f"{name:<32}{v['n_kept']:>6}{v['mean_headway_min_m']:>10}{v['mean_time_gap_min_s']:>8}{v['mean_min_ttc_s']:>9}{s:>26}")
print("\nrefused:", json.dumps(refused))
