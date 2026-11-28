"""DESCRIPTIVE, post-hoc, NOT a gate: the human off-rate on the FULL S6-compliant GT population (n = 3198, the K_HUMAN population) under alternative 'non-drivable' class sets."""
import sys, json, numpy as np, torch
sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_mapsel/code'); sys.path.insert(0,'C:/Users/Admin/refcv8_pf/refcv9_l2/code')
import mapsel_data as D, mapsel_lib as L, l2_common as C
torch.set_num_threads(4)
E=C.load_eval_pf(0); keys=list(zip(E['clip'].tolist(),E['ws'].tolist())); comp=np.asarray(E['compliant'],bool)
hp,valid=D.human_plans(keys); gt=D.GTReader()
sel=[i for i in range(len(keys)) if comp[i] and valid[i] and gt.has[keys[i][0]]]
sets={'0,5,7 (as sealed)':(0,5,7),'5,7':(5,7),'0,7':(0,7),'7 only':(7,),'5 only':(5,),'0 only':(0,)}
cnt={k:0 for k in sets}; eid=[]; per={k:{} for k in sets}
for i in sel:
    ci,ws=keys[i]; c=torch.as_tensor(gt.codes(ci,ws))
    g=L.Geom(torch.from_numpy(hp[i][None])); cc=c[g.ix[0],g.iy[0]]; ins=g.inside[0]
    for k,cs in sets.items():
        m=torch.zeros_like(ins)
        for v in cs: m|=(cc==v)
        o=int((ins&m).any()); cnt[k]+=o; per[k].setdefault(int(E['eid'][i]),[]).append(o)
n=len(sel); out={'n':n}
for k in sets:
    pe=[np.mean(v) for v in per[k].values()]
    out[k]={'rate':round(cnt[k]/n,4),'episodes_ge_0.5':int(sum(1 for x in pe if x>=0.5)),'of':len(pe)}
print(json.dumps(out,indent=1))
json.dump(out,open('C:/Users/Admin/refcv8_pf/refcv9_mapsel/result/K_HUMAN_alt_class_sets_DESCRIPTIVE.json','w'),indent=1)
