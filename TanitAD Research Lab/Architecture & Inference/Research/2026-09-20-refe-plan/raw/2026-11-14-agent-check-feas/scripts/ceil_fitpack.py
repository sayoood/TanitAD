"""Oracle ceilings with TRUE agent sub-scores on P6's navtrain fit pack (held-out + fit-extra), CPU, no navtest."""
import numpy as np, json
d=np.load("F:/tanitad_scratch/p6fit/fit_pack.npz",allow_pickle=False)
names=list(d["pdm_names"]); print(names, str(d["logit_space"]))
L=d["logits"].astype(np.float64); P=d["pdm"].astype(np.float64); ho=d["is_heldout"]; hp=d["has_pdm"]; log=d["log"]
print("sets",len(L),"heldout",ho.sum(),"has_pdm",hp.sum(),"logs",len(set(log)),"heldout logs",len(set(log[ho])))
print("ckpt_step uniq",np.unique(d["ckpt_step"]), "rank uniq",np.unique(d["rank"]))
ix={n:i for i,n in enumerate(names)}
print({n:(float(np.nanmin(P[...,i])),float(np.nanmax(P[...,i]))) for n,i in ix.items()})
sg=1/(1+np.exp(-L))  # P6 emits raw logits (read_scorer1p6.pick_logits: sigmoid); a first version of this script wrongly skipped the sigmoid
rule=sg[...,0]*sg[...,1]*(5*sg[...,2]+5*sg[...,3]+2*sg[...,4])/12
pick=rule.argmax(1)
# truth PDMS from sub-scores: columns assumed nc,dac,ep,ttc,c,(ddc?),pdms?
def col(n): return P[...,ix[n]]
res={}
def run(mask,tag):
    N=mask.sum(); I=np.arange(len(L))[mask]
    sub=P[mask]; r=rule[mask]; pk=r.argmax(1); a=np.arange(N)
    if "pdms" in ix: pd=sub[...,ix["pdms"]]
    else: pd=sub[...,0]*sub[...,1]*(5*sub[...,2]+5*sub[...,3]+2*sub[...,4])/12
    out={"n_sets":int(N),"pdms_pick":float(100*pd[a,pk].mean()),"pdms_best64":float(100*pd.max(1).mean())}
    nc=sub[...,ix["nc"]]; dac=sub[...,ix["dac"]]; ttc=sub[...,ix["ttc"]]
    out["nc0_at_pick_share"]=float((nc[a,pk]==0).mean()); out["dac0_at_pick_share"]=float((dac[a,pk]==0).mean()); out["ttc0_at_pick_share"]=float((ttc[a,pk]==0).mean())
    out["pdms0_at_pick_share"]=float((pd[a,pk]==0).mean())
    def veto(ok):
        rr=np.where(ok,r,-1.0); allbad=~ok.any(1); pk2=rr.argmax(1); pk2=np.where(allbad,pk,pk2); return float(100*pd[a,pk2].mean())
    out["O_NC (true NC>0 veto)"]=veto(nc>0)-out["pdms_pick"]
    out["O_DAC (true DAC>0 veto)"]=veto(dac>0)-out["pdms_pick"]
    out["O_TTC (true TTC>0 veto)"]=veto(ttc>0)-out["pdms_pick"]
    out["O_NC+TTC"]=veto((nc>0)&(ttc>0))-out["pdms_pick"]
    out["O_agent+DAC"]=veto((nc>0)&(ttc>0)&(dac>0))-out["pdms_pick"]
    out["O_NC_soft(NC==1)"]=veto(nc>=1)-out["pdms_pick"]
    # soft: multiply rule by TRUE nc, ttc factors
    for nm,f in [("soft_trueNC",nc),("soft_trueTTC",0.5+0.5*ttc),("soft_trueNC_TTC",nc*(0.5+0.5*ttc))]:
        pk3=(r*f+1e-9*r).argmax(1); out["O_"+nm]=float(100*pd[a,pk3].mean())-out["pdms_pick"]
    # fraction of tokens where an NC-clean alternative exists when pick NC==0
    z=(nc[a,pk]==0); out["nc0_pick_with_ncclean_alt"]=float(((nc>0).any(1)&z).sum()/max(z.sum(),1))
    out["share_ncclean_among_props(nc0 tokens)"]=float((nc[z]>0).mean()) if z.any() else None
    return out
for tag,m in [("heldout",ho&hp),("fit_extra",(~ho)&hp),("all",hp)]:
    res[tag]=run(m,tag)
print(json.dumps(res,indent=1))
json.dump(res,open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_feas/ceiling_fitpack.json","w"),indent=1)
