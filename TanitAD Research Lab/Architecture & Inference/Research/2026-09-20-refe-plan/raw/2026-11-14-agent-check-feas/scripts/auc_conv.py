"""P6 scorer AUCs for NC/TTC/DAC and a synthetic conversion curve (ESTIMATED) on the navtrain fit pack. CPU, navtrain only."""
import numpy as np, json
from math import erf, sqrt
d=np.load("F:/tanitad_scratch/p6fit/fit_pack.npz",allow_pickle=False)
L=d["logits"].astype(np.float64); P=d["pdm"].astype(np.float64); ho=d["is_heldout"]
sg=1/(1+np.exp(-L)); rule=sg[...,0]*sg[...,1]*(5*sg[...,2]+5*sg[...,3]+2*sg[...,4])/12
nc,dac,ttc,pd=P[...,0],P[...,1],P[...,3],P[...,6]
def auc(s,y):
    s=s.ravel(); y=y.ravel().astype(bool)
    n1=y.sum(); n0=(~y).sum()
    if n1==0 or n0==0: return float("nan")
    r=np.empty(len(s)); o=np.argsort(s,kind="mergesort"); r[o]=np.arange(1,len(s)+1)
    # tie-correct
    from scipy.stats import rankdata
    r=rankdata(s)
    return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
def within(s,y):
    a=[]
    for i in range(len(s)):
        yy=y[i].astype(bool)
        if yy.all() or (~yy).all(): continue
        a.append(auc(s[i],yy))
    return float(np.mean(a)), len(a)
out={"base_rates(share of proposals)":{"nc>0":float((nc>0).mean()),"nc==1":float((nc==1).mean()),"dac>0":float((dac>0).mean()),"ttc>0":float((ttc>0).mean())}}
for tag,m in [("heldout",ho),("all",np.ones(len(L),bool))]:
    r={}
    for nm,s,y in [("NC",sg[m][...,0],nc[m]>0),("DAC",sg[m][...,1],dac[m]>0),("TTC",sg[m][...,3],ttc[m]>0)]:
        r[nm]={"pooled_auc":auc(s,y),"within_set_auc_mean":within(s,y)[0],"n_sets_with_both_classes":within(s,y)[1]}
    out["P6_scorer_auc_"+tag]=r
# synthetic conversion curve: agent-ok = nc>0 & ttc>0 ; score = ok*delta + N(0,1) independent per proposal (optimistic)
ok=(nc>0)&(ttc>0)
base=100*pd[np.arange(len(L)),rule.argmax(1)].mean()
rng=np.random.default_rng(20261114)
from scipy.stats import norm
curve={}
for A in (0.6,0.7,0.8,0.9,0.95,0.99,1.0):
    delta=norm.ppf(min(A,0.999999))*sqrt(2) if A<1 else 12.0
    row={}
    for rep in range(1):
        sc=ok*delta+rng.standard_normal(ok.shape)
        q=1/(1+np.exp(-(sc-delta/2)*1.0))   # calibrated-ish posterior-like
        pk=(rule*q).argmax(1); soft=100*pd[np.arange(len(L)),pk].mean()-base
        hard=np.where(q>0.5,rule,-1); pk2=np.where((q>0.5).any(1),hard.argmax(1),rule.argmax(1)); hv=100*pd[np.arange(len(L)),pk2].mean()-base
    curve[str(A)]={"soft_gain_pdms":float(soft),"hard_veto_gain_pdms":float(hv)}
out["synthetic_conversion_curve (independent per-proposal noise = OPTIMISTIC, ESTIMATED, navtrain all 8,489 sets)"]=curve
out["base_pdms_pick"]=base
print(json.dumps(out,indent=1))
json.dump(out,open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_feas/auc_and_conversion.json","w"),indent=1)
