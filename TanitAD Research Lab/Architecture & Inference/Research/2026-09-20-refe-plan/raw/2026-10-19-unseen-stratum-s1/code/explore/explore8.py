import json, sys, sqlite3, collections, numpy as np, itertools
sys.path.insert(0,"C:/Users/Admin/par18/unseen_s1/code")
import unseen_token_rule as R, identity_check as IC
DB=IC.DB
known=IC.load_known()
logs=sorted(l for l in known if __import__('os').path.exists(f"{DB}/{l}.db"))
CH=["CAM_F0","CAM_L0","CAM_R0","CAM_B0","CAM_L1","CAM_L2","CAM_R1","CAM_R2"]
data=[]
for lg in logs:
    rec=R.read_log(f"{DB}/{lg}.db"); ts=rec["ts"]
    idx={t:i for i,t in enumerate(rec["tokens"])}
    true=collections.Counter(idx[t]%10 for t in known[lg]).most_common(1)[0][0]
    c=sqlite3.connect(f"file:{DB}/{lg}.db?mode=ro",uri=True)
    first={}
    firstim={}
    for ch in CH:
        im=np.array([r[0] for r in c.execute("select i.timestamp from image i join camera c on c.token=i.camera_token where c.channel=? order by i.timestamp",(ch,))])
        pm=R.paired_mask(ts,im) if len(im) else np.zeros(len(ts),bool)
        first[ch]=int(np.argmax(pm)) if pm.any() else 999
        firstim[ch]=(int(im[0]-ts[0])/1e3 if len(im) else None)
    c.close()
    data.append((lg,true,first,firstim,len(ts)))
# subset search
best=[]
for r in range(1,9):
    for S in itertools.combinations(CH,r):
        m=sum(1 for lg,t,f,_,n in data if max(f[c] for c in S)%10==t)
        best.append((m,S))
best.sort(key=lambda x:-x[0])
print(best[:8])
# minimal first image time offsets for different truths
for lg,t,f,fi,n in data[:3]+[d for d in data if d[0].endswith("00021_00143")]:
    print(lg[-28:],"true",t,"first_paired",{k[4:]:v for k,v in f.items()},"first image - ts0 (ms)",{k[4:]:(round(v,1) if v is not None else None) for k,v in fi.items()})
