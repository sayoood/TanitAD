"""The GT nocls holes on the HUMAN path: per-episode share of windows with a footprint check point on nocls (code 0); episodes with >= 0.5 named by sha12 ONLY (for the Data FlyWheel)."""
import sys, json, numpy as np, torch
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code'); sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_l2/code')
import mapsel_data as D, mapsel_lib as L, l2_common as C
torch.set_num_threads(2)
E=C.load_eval_pf(0); keys=list(zip(E['clip'].tolist(),E['ws'].tolist())); comp=np.asarray(E['compliant'],bool)
hp,valid=D.human_plans(keys); gt=D.GTReader(); uu,_=D.clip_uuids()
per={}; nearfield=[0,0]
for i in range(len(keys)):
    if not (comp[i] and valid[i] and gt.has[keys[i][0]]): continue
    ci,ws=keys[i]; c=torch.as_tensor(gt.codes(ci,ws))
    g=L.Geom(torch.from_numpy(hp[i][None])); cc=c[g.ix[0],g.iy[0]]; ins=g.inside[0]
    per.setdefault(ci,[]).append(int((ins&(cc==0)).any()))
rate={ci:float(np.mean(v)) for ci,v in per.items()}
bad=sorted([(D.sha12(uu[ci]),round(r,3),len(per[ci])) for ci,r in rate.items() if r>=0.5], key=lambda x:-x[1])
out={'episodes_ge_0.5':len(bad),'of':len(rate),'sha12_rate_nwindows':bad,'overall_share_windows':float(np.mean([x for v in per.values() for x in v]))}
print(json.dumps(out,indent=1)); json.dump(out,open('C:/Users/Admin/refcv8_pf/refcv9_mapsel/result/GT_NOCLS_HOLE_EPISODES.json','w'),indent=1)
