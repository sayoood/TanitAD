import json, sys, sqlite3, collections, numpy as np, itertools, os
sys.path.insert(0,"C:/Users/Admin/par18/unseen_s1/code")
import unseen_token_rule as R
import identity_check as IC
DB="D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
known=IC.load_known()
logs=sorted(l for l in known if os.path.exists(f"{DB}/{l}.db"))
CH=["CAM_F0","CAM_L0","CAM_R0","CAM_B0","CAM_L1","CAM_L2","CAM_R1","CAM_R2"]
data=[]
for lg in logs:
    rec=R.read_log(f"{DB}/{lg}.db"); ts=rec["ts"]
    idx={t:i for i,t in enumerate(rec["tokens"])}
    cnt=collections.Counter(idx[t]%10 for t in known[lg]); true=cnt.most_common(1)[0][0]
    c=sqlite3.connect(f"file:{DB}/{lg}.db?mode=ro",uri=True)
    first={}; fim={}
    for ch in CH:
        im=np.array([r[0] for r in c.execute("select i.timestamp from image i join camera c on c.token=i.camera_token where c.channel=? order by i.timestamp",(ch,))])
        pm=R.paired_mask(ts,im) if len(im) else np.zeros(len(ts),bool)
        first[ch]=int(np.argmax(pm)) if pm.any() else 999
        fim[ch]=round(float(im[0]-ts[0])/1e3,1) if len(im) else None
    c.close()
    # global dt irregularities
    dt=np.diff(ts)/1e6
    data.append((lg,true,dict(cnt),first,fim,len(ts),float(dt.max()),int((dt>0.06).sum())))
pred=lambda f:max(f["CAM_F0"],f["CAM_L0"])%10
bad=[d for d in data if pred(d[3])!=d[1]]
print(len(bad),"bad of",len(data))
for lg,t,cnt,f,fim,n,dtm,ng in bad:
    print(lg[-34:],"true",t,"cnt",cnt,"pred",pred(f),"first",{k[4:]:v for k,v in f.items()},"first im-ts0",{k[4:]:v for k,v in fim.items()},"N",n,"dtmax",round(dtm,3),"ngaps",ng)
# subset search again on all 248
best=[]
for r in range(1,9):
    for S in itertools.combinations(CH,r):
        m=sum(1 for d in data if max(d[3][c] for c in S)%10==d[1])
        best.append((m,S))
best.sort(key=lambda x:-x[0]); print(best[:6])
# how many true phases are consistent with a pure 'first lidar_pc with image from ALL cams (at their own nearest) mod parity?'
