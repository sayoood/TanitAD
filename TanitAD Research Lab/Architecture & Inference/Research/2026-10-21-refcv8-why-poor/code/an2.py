import sys, numpy as np, json
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/evidence_B/"
sys.path.insert(0,S)
import tactical_conditioning as TC
LATN={0:"LANE_KEEP",1:"LC_L",2:"LC_R",3:"ABORT",4:"NUDGE_L",5:"NUDGE_R",6:"TURN_L",7:"TURN_R"}
LONN=TC.V7_LON_NAMES
def load(tag,seed): return dict(np.load(S+f"X_s{seed}_{tag}.npz",allow_pickle=True))
def boot_ci(v, eid, B=2000, seed=0):
    # episode-cluster bootstrap on mean (ratio of sums over valid windows)
    ok=np.isfinite(v); eids=np.unique(eid); rng=np.random.default_rng(seed)
    num=np.array([np.nansum(v[eid==e]) for e in eids]); den=np.array([ok[eid==e].sum() for e in eids],float)
    out=[]
    for _ in range(B):
        i=rng.integers(0,len(eids),len(eids)); out.append(num[i].sum()/max(den[i].sum(),1))
    return float(np.nansum(v)/ok.sum()), float(np.percentile(out,2.5)), float(np.percentile(out,97.5))
def pcb(a,b,eid,B=2000,seed=0):
    d=a-b; return boot_ci(d,eid,B,seed)
def head_report(pred,label,names,eid):
    m=label!=-100; p=pred[m]; l=label[m]; e=eid[m]
    cls=sorted(set(l.tolist())|set(p.tolist()))
    acc=(p==l).mean()
    maj=np.bincount(l).argmax(); const=(l==maj).mean()
    rows={}
    for c in cls:
        n=(l==c).sum(); npred=(p==c).sum(); tp=((p==c)&(l==c)).sum()
        rows[names.get(c,c) if isinstance(names,dict) else names[c]]=dict(n=int(n),n_pred=int(npred),recall=round(tp/n,4) if n else None,precision=round(tp/npred,4) if npred else None)
    rec=[r['recall'] for r in rows.values() if r['recall'] is not None and r['n']>0]
    bal=float(np.mean(rec))
    a=(p==l).astype(float); c0=(l==maj).astype(float)
    d=pcb(a,c0,e)
    return dict(n=int(m.sum()),acc=round(float(acc),4),const_class=int(maj),const_acc=round(float(const),4),head_minus_const=[round(x,4) for x in d],balanced_acc=round(bal,4),per_class=rows)
def lookup_echo(token,label,eid):
    """token->label lookup fit on the same windows (optimistic for the control): predict label by majority label given the fed token"""
    m=label!=-100; t=token[m]; l=label[m]
    pred=np.zeros_like(l)
    for v in np.unique(t):
        mm=t==v; pred[mm]=np.bincount(l[mm]).argmax()
    return round(float((pred==l).mean()),4)
if __name__=="__main__":
    for tag in ("step48852ema_pf","step50400_a6text"):
        for seed in ("0",):
            X=load(tag,seed); eid=X['eid']
            print("=====",tag,"seed",seed)
            print("nav_cmd counts",np.bincount(X['nav_cmd'].clip(0)))
            print("LAT:",json.dumps(head_report(X['lat_pred_nav_true'],X['lat_label'],LATN,eid)))
            print("LON:",json.dumps(head_report(X['lon_pred_nav_true'],X['lon_label'],{i:n for i,n in enumerate(LONN)},eid)))
            print("lat token-lookup echo acc",lookup_echo(X['nav_cmd'],X['lat_label'],eid),"; lon token-lookup",lookup_echo(X['nav_cmd'],X['lon_label'],eid))
            if 'lat_pred_nav_zero' in X:
                print("LAT navzero:",json.dumps(head_report(X['lat_pred_nav_zero'],X['lat_label'],LATN,eid))[:400])
                print("LON navzero acc/bal:",head_report(X['lon_pred_nav_zero'],X['lon_label'],{i:n for i,n in enumerate(LONN)},eid)['acc'],head_report(X['lon_pred_nav_zero'],X['lon_label'],{i:n for i,n in enumerate(LONN)},eid)['balanced_acc'])
