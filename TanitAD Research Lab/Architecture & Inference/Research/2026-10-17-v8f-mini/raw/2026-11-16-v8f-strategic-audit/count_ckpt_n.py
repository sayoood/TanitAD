import json, sys, numpy as np
sys.path.insert(0,'F:/TanitAD-scratch/strategic'); import strategic_cells as S
T=np.load('F:/TanitAD-scratch/datascale/navtok/navtok_table.npz'); V=list(T['vocab']); J=np.load('F:/TanitAD-scratch/datascale/navtok/raw/join_eval141.npz',allow_pickle=True)
g=json.load(open('F:/TanitAD-scratch/strategic/proto/eval_grid_windows.json')); ci={c:i for i,c in enumerate(T['eval_clip'].tolist())}
cs=[(c,t) for c in g['order'] for t in g['windows'][c]]; r=np.array([T['eval_off'][ci[c]]+t for c,t in cs]); epi=np.array([c for c,_ in cs])
sv,_=S.side_table(V); side=sv[T['eval_cls_id'][r]]; d=T['eval_dist_m'][r].astype(float); ok=T['eval_flag'][r]==0
rc_ok=J['rc_ok'][r]; has=J['has'][r]
out={'n_windows':len(cs),'text_ok':int(ok.sum()),'turn_LR':int((ok&(side!=0)).sum()),'eps_with_turn':len({e for e in epi[ok&(side!=0)]}),'rc_ok_rate':[float(rc_ok[:,m].mean()) for m in range(3)]}
for m,a in enumerate((30,50,80)):
    sel=ok&(side!=0)&np.isfinite(d)&(d+S.COMPLETE_M<=a)&rc_ok[:,m]&has
    out[f'ckpt{a}']={'n':int(sel.sum()),'eps':len(set(epi[sel])),'left':int((side[sel]==1).sum()),'right':int((side[sel]==-1).sum())}
print(json.dumps(out))
json.dump(out,open('F:/TanitAD-scratch/strategic/proto_eval141_counts.json','w'),indent=1)
