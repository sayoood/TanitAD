import numpy as np, json
from pathlib import Path
O=Path('F:/treedrift/out')
b=np.load('F:/l1c/bank/ztac_fresh.npz',allow_pickle=False)
fk={(int(c),int(t)):j for j,(c,t) in enumerate(zip(b['e_i'],b['t']))}
def load(n):
    z=np.load(O/n/'records.npz',allow_pickle=False); return z
def gt(n):
    g=[]
    for f in sorted((O/n/'dump').glob('ep*.npz')):
        z=np.load(f,allow_pickle=False); g.append(z['g'].astype(np.float64))
    return np.concatenate(g)
res={}
ref=load('fb_s0'); G=gt('fb_s0')
rows=np.array([fk[(int(c),int(t))] for c,t in zip(ref['e_i'],ref['t'])])
bank=b['traj_own'][rows].astype(np.float64)
def ade(traj4): return np.linalg.norm(traj4[:,:4,:]-G,axis=-1).mean(-1)
# NOTE: dump order = ep000 then ep001 = the records order (checked below)
ade_bank=ade(bank)
out={'n':70,'bank_mean_ade_2s_from_traj_own[:4]':float(ade_bank.mean())}
for n in ('fb_s0','tip_s0','fb_s1','bankpos_fb','bankpos_tip','bankpos1_fb','bankpos_c3766'):
    p=O/n/'records.npz'
    if not p.exists(): out[n]='MISSING'; continue
    z=load(n)
    assert (z['e_i']==ref['e_i']).all() and (z['t']==ref['t']).all()
    d=np.abs(z['traj'].astype(np.float64)-bank).max(axis=(1,2))
    a=ade(z['traj'].astype(np.float64))
    out[n]={'max_abs_vs_bank_m':float(d.max()),'n_differ_vs_bank':int((d>0).sum()),'n_gt_1e-5':int((d>1e-5).sum()),'n_gt_1e-4':int((d>1e-4).sum()),'mean_ade_2s':float(a.mean()),
            'paired_dADE_vs_bank_mean':float((a-ade_bank).mean()),'paired_dADE_sd':float((a-ade_bank).std(ddof=1))}
print(json.dumps(out,indent=1))
json.dump(out,open('F:/treedrift/BANKCMP.json','w'),indent=1)
