"""Prototype read of S_ADH / S_FALSE on the ACTSENS NULL_s3 windows (994 windows, 71 episodes), T0 oracle-goal plans: a CPU demonstration, not a result."""
import sys, json, numpy as np
sys.path.insert(0, 'F:/TanitAD-scratch/strategic'); sys.path.insert(0, 'F:/TanitAD-scratch/actsens/read/tree/taniteval')
import strategic_cells as S
from taniteval.ci import paired_episode_cluster_bootstrap as pb, episode_cluster_bootstrap as eb
T = np.load('F:/TanitAD-scratch/datascale/navtok/navtok_table.npz'); V = list(T['vocab'])
z = np.load('F:/TanitAD-scratch/actsens/read/out/NULL_s3/win/windows.npz', allow_pickle=True)
rows = np.load('F:/TanitAD-scratch/strategic/proto/actsens_rows.npy')
side_v, th_v = S.side_table(V)
cls = T['eval_cls_id'][rows]; d = T['eval_dist_m'][rows].astype(float); ok = T['eval_flag'][rows] == 0
side, th = side_v[cls], th_v[cls]
v0 = z['v0'].astype(float); eid = z['eid'].astype(str)
plans = {'hwm_oracle': np.load('F:/TanitAD-scratch/actsens/read/out/NULL_s3/win/plans_hwm_oracle_s0.npz')['plan_raw'],
         'flat_oracle': np.load('F:/TanitAD-scratch/actsens/read/out/NULL_s3/win/plans_flat_oracle_s0.npz')['plan_raw'],
         'logged': z['A'][:, :30].astype(float), 'hold': np.zeros((994, 30, 2))}
C = {}
for k, p in plans.items():
    if k == 'hold': p = p.copy()
    psi, xy = S.integrate(v0, p.astype(float)); C[k] = S.cells(psi, xy, side, th, d, ok)
out = {'n_windows': 994, 'n_episodes': int(len(set(eid))), 'note': 'T0 oracle-goal plans; goal is the true-future latent so adherence is a WM diagnostic'}
for k in plans:
    a = C[k]['s_adh']; f = C[k]['s_false']
    out[k] = {'s_adh_n': int(np.isfinite(a).sum()), 's_adh_rate': float(np.nanmean(a)) if np.isfinite(a).any() else None,
              's_false_n': int(np.isfinite(f).sum()), 's_false_rate': float(np.nanmean(f)) if np.isfinite(f).any() else None}
m = np.isfinite(C['hwm_oracle']['s_adh']) & np.isfinite(C['logged']['s_adh'])
out['n_adh_eps'] = int(len(set(eid[m]))); out['n_adh_both_scorable'] = int(m.sum())
if m.sum() > 5:
    r = pb(C['hwm_oracle']['s_adh'][m], C['logged']['s_adh'][m], eid[m], n_boot=2000, seed=0)
    out['paired_hwm_oracle_minus_logged_adh'] = {k: (float(r[k]) if k in ('delta', 'lo', 'hi') else r[k]) for k in ('delta', 'lo', 'hi', 'separated', 'n_windows', 'n_episodes')}
json.dump(out, open('F:/TanitAD-scratch/strategic/proto_actsens_read.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
