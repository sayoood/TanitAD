"""Where does P6's own NC/TTC head fail at the pick? (navtrain fit pack, CPU, MEASURED on banked data)"""
import numpy as np, json
from scipy.stats import rankdata
d=np.load("F:/tanitad_scratch/p6fit/fit_pack.npz",allow_pickle=False)
L=d["logits"].astype(np.float64); P=d["pdm"].astype(np.float64)
sg=1/(1+np.exp(-L)); rule=sg[...,0]*sg[...,1]*(5*sg[...,2]+5*sg[...,3]+2*sg[...,4])/12
nc,dac,ttc,pd=P[...,0],P[...,1],P[...,3],P[...,6]
N=len(L); a=np.arange(N); pk=rule.argmax(1)
ok=(nc>0)&(ttc>0); pok=sg[...,0]*sg[...,3]
res={}
fail=~ok[a,pk]; res["n_sets"]=N; res["pick_agent_fail_share"]=float(fail.mean())
res["pick_nc0_share"]=float((nc[a,pk]==0).mean()); res["pick_ttc0_share"]=float((ttc[a,pk]==0).mean())
res["P6_own_p_NC*p_TTC_at_failing_pick: median"]=float(np.median(pok[a,pk][fail]))
res["share_failing_picks_with_own_p_ok>0.5 (confident misses)"]=float((pok[a,pk][fail]>0.5).mean())
res["share_failing_picks_with_own_p_ok>0.8"]=float((pok[a,pk][fail]>0.8).mean())
res["median own p_ok at passing pick"]=float(np.median(pok[a,pk][~fail]))
# top-5 within-set discrimination of agent failure, P6's own p_ok
top5=np.argsort(-rule,1)[:,:5]
S=np.take_along_axis(pok,top5,1); Y=np.take_along_axis(ok,top5,1)
m=(Y.any(1))&(~Y.all(1)); res["top5_sets_with_both_classes"]=int(m.sum())
def auc(s,y):
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
res["P6_own_top5_within_set_auc(agent-ok)"]=float(np.mean([auc(S[i],Y[i]) for i in np.where(m)[0]]))
res["best_alt_exists_when_pick_fails"]=float(((ok.any(1))&fail).sum()/max(fail.sum(),1))
res["rank_of_first_agent_ok_proposal_in_P6_order: median/90th"]=[float(np.median(r)) for r in [np.array([np.where(ok[i][np.argsort(-rule[i])])[0][0] for i in np.where(fail&ok.any(1))[0]])]]+[float(np.percentile(np.array([np.where(ok[i][np.argsort(-rule[i])])[0][0] for i in np.where(fail&ok.any(1))[0]]),90))]
# by log concentration
lg=d["log"]; u,inv=np.unique(lg,return_inverse=True); c=np.bincount(inv,weights=fail.astype(float)); n=np.bincount(inv)
res["agent_fail_pick_rate_across_logs: min/median/max (logs with >=20 sets)"]=[float(x) for x in np.percentile((c/n)[n>=20],[0,50,100])]
print(json.dumps(res,indent=1))
json.dump(res,open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_feas/p6_agent_blindspot.json","w"),indent=1)
