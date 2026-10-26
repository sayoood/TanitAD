import sys, numpy as np, torch
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code'); sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_l2/code')
import mapsel_data as D, mapsel_lib as L, l2_common as C
torch.set_num_threads(4)
E=C.load_eval_pf(0); keys=list(zip(E['clip'].tolist(),E['ws'].tolist())); comp=np.asarray(E['compliant'],bool)
hp,valid=D.human_plans(keys); gt=D.GTReader()
sel=[i for i in range(len(keys)) if comp[i] and valid[i] and gt.has[keys[i][0]]][::4]
res={'all_nondrv':0,'only_5_7':0,'only_7':0,'only_5':0,'only_0':0}; xr=np.zeros(5); xn=np.zeros(5); share0_win=[]
for i in sel:
    ci,ws=keys[i]; c=gt.codes(ci,ws); ct=torch.as_tensor(c)
    g=L.Geom(torch.from_numpy(hp[i][None])); ix,iy,ins=g.ix[0],g.iy[0],g.inside[0]
    cc=ct[ix,iy]
    seen=ins&(cc!=255)
    drv=torch.zeros_like(seen)
    for k in L.DRV_CODES: drv|=cc==k
    bad=seen&~drv
    res['all_nondrv']+=int(bad.any())
    res['only_5_7']+=int((seen&((cc==5)|(cc==7))).any()); res['only_7']+=int((seen&(cc==7)).any()); res['only_5']+=int((seen&(cc==5)).any()); res['only_0']+=int((seen&(cc==0)).any())
    # distance band of the nocls hits (x of the point, m)
    pts=L.footprint5(*L.path_samples(torch.from_numpy(hp[i][None]))[:2])[0]   # [S,5,2]
    x=pts[...,0].numpy(); b=np.digitize(x,[10,20,40,60]); h0=((cc==0)&seen).numpy()
    for k in range(5): xr[k]+=h0[b==k].sum(); xn[k]+=seen.numpy()[b==k].sum()
    share0_win.append(float(((cc==0)&seen).sum()/max(int(seen.sum()),1)))
n=len(sel)
print({k:round(v/n,4) for k,v in res.items()})
print('nocls share of SEEN human footprint samples by x band [<10,10-20,20-40,40-60,>60 m]:', (xr/np.maximum(xn,1)).round(3), 'n', xn.astype(int))
print('mean per-window nocls share', round(float(np.mean(share0_win)),4))
