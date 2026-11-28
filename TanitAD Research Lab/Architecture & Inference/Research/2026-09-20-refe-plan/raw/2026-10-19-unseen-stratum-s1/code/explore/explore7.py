import json, sys, sqlite3, collections, numpy as np
sys.path.insert(0,"C:/Users/Admin/par18/unseen_s1/code")
import unseen_token_rule as R, identity_check as IC
DB=IC.DB
known=IC.load_known()
logs=sorted(l for l in known if __import__('os').path.exists(f"{DB}/{l}.db"))
CH=["CAM_F0","CAM_L0","CAM_R0","CAM_B0","CAM_L1","CAM_L2","CAM_R1","CAM_R2"]
res=collections.Counter(); rows=[]
for lg in logs:
    rec=R.read_log(f"{DB}/{lg}.db"); ts=rec["ts"]
    idx={t:i for i,t in enumerate(rec["tokens"])}
    true=collections.Counter(idx[t]%10 for t in known[lg]).most_common(1)[0][0]
    c=sqlite3.connect(f"file:{DB}/{lg}.db?mode=ro",uri=True)
    pm={}
    for ch in CH:
        im=np.array([r[0] for r in c.execute("select i.timestamp from image i join camera c on c.token=i.camera_token where c.channel=? order by i.timestamp",(ch,))])
        pm[ch]=R.paired_mask(ts,im) if len(im) else np.zeros(len(ts),bool)
    c.close()
    first={ch:(int(np.argmax(pm[ch])) if pm[ch].any() else None) for ch in CH}
    allm=np.all([pm[ch] for ch in CH],axis=0); a8=int(np.argmax(allm)) if allm.any() else None
    four=np.all([pm[ch] for ch in CH[:4]],axis=0); a4=int(np.argmax(four)) if four.any() else None
    hyp={"F0":first["CAM_F0"],"all8":a8,"all4":a4}
    for ch in CH[1:]: hyp[ch]=first[ch]
    for h,v in hyp.items():
        res[(h, v is not None and v%10==true)]+=1
    rows.append((lg,true,hyp))
for h in ["F0","all4","all8"]+CH[1:]:
    print(h,"match",res[(h,True)],"mismatch",res[(h,False)])
for lg,t,h in rows:
    if h["F0"]%10!=t: print("F0-miss",lg[-30:],"true",t,h)
