"""0-GPU diagnosis of K_HUMAN > 2 %: which FP5 point, which GT class, which episodes, and does a window-index shift of the GT raster lower the rate?"""
import sys, json, numpy as np, torch
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code'); sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_l2/code')
import mapsel_data as D, mapsel_lib as L, l2_common as C
torch.set_num_threads(4)
E=C.load_eval_pf(0); keys=list(zip(E['clip'].tolist(),E['ws'].tolist())); comp=np.asarray(E['compliant'],bool)
hp,valid=D.human_plans(keys); gt=D.GTReader()
sel=[i for i in range(len(keys)) if comp[i] and valid[i] and gt.has[keys[i][0]]][::4]
print('windows',len(sel))
def point_viol(codes, plan):
    g=L.Geom(torch.from_numpy(plan[None])); m=L.masks_gt(codes)
    ins=g.inside[0]                                   # [S,5]
    cc=torch.as_tensor(codes)[g.ix[0], g.iy[0]]       # [S,5]
    bad=ins & m['seen'][g.ix[0],g.iy[0]] & ~m['drv'][g.ix[0],g.iy[0]]
    return bad.numpy(), cc.numpy(), ins.numpy()
pt_names=['FL','RL','RR','FR','C']
cnt_pt=np.zeros(5); cls_hist=np.zeros(256); off=0; per={}
for n,i in enumerate(sel):
    ci,ws=keys[i]; codes=gt.codes(ci,ws)
    bad,cc,ins=point_viol(codes,hp[i])
    w_off=bad.any(); off+=int(w_off); per.setdefault(ci,[]).append(int(w_off))
    cnt_pt+= bad.any(0)
    for v in np.unique(cc[bad]): cls_hist[v]+= (cc[bad]==v).sum()
print('off rate (sample)',off/len(sel))
print('windows with a violation by point type', dict(zip(pt_names,(cnt_pt/len(sel)).round(4))))
print('violating-cell class histogram (share)', {int(k):round(float(v/cls_hist.sum()),3) for k,v in enumerate(cls_hist) if v>0})
pe={c:np.mean(v) for c,v in per.items()}; top=sorted(pe.items(), key=lambda x:-x[1])[:8]
print('episodes with off rate >= 0.5:', sum(1 for v in pe.values() if v>=0.5), 'of', len(pe), 'top', [(c,round(v,2),len(per[c])) for c,v in top])
# centre-only and rear-axle-only
cen=0; rear=0
for i in sel:
    ci,ws=keys[i]; codes=gt.codes(ci,ws); m=L.masks_gt(codes)
    g=L.Geom(torch.from_numpy(hp[i][None])); ix,iy,ins=g.ix[0,:,4],g.iy[0,:,4],g.inside[0,:,4]
    cen+= int((ins & m['seen'][ix,iy] & ~m['drv'][ix,iy]).any())
print('centre-only off rate', cen/len(sel))
# window-index shift of the GT raster
for sh in (-6,-3,-1,0,1,3,6):
    o=0; k=0
    for i in sel[::3]:
        ci,ws=keys[i]
        try: codes=gt.codes(ci,ws+sh)
        except Exception: continue
        bad,_,_=point_viol(codes,hp[i]); o+=int(bad.any()); k+=1
    print('GT window shift',sh,'off rate',round(o/max(k,1),4),'n',k)
