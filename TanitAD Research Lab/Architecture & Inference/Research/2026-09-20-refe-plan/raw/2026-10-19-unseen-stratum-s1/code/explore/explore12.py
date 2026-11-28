import json, sys, sqlite3, collections, numpy as np, os
sys.path.insert(0,"C:/Users/Admin/par18/unseen_s1/code")
import unseen_token_rule as R
import identity_check as IC
known=IC.load_known()
res=collections.Counter(); bad=[]
for D in ("D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval","D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"):
    for lg in sorted(l for l in known if os.path.exists(f"{D}/{l}.db")):
        rec=R.read_log(f"{D}/{lg}.db"); ts=rec["ts"]
        idx={t:i for i,t in enumerate(rec["tokens"])}
        true=collections.Counter(idx[t]%10 for t in known[lg]).most_common(1)[0][0]
        # H_A: current rule
        pa=R.first_paired(rec)
        # H_B: paired on F0 and L0, and ts_i >= first image ts of both (image available at or before the lidar_pc)
        pm=np.ones(len(ts),bool); av=np.ones(len(ts),bool)
        for ch in ("CAM_F0","CAM_L0"):
            pm&=R.paired_mask(ts,rec["img_ts"][ch]); av&=(ts>=rec["img_ts"][ch][0])
        hb=int(np.argmax(pm&av))
        # H_C: signed window: image within [ts-25ms, ts] (image before lidar) for both cams, first such i
        sg=np.ones(len(ts),bool)
        for ch in ("CAM_F0","CAM_L0"):
            im=rec["img_ts"][ch]; j=np.searchsorted(im,ts,side="right")-1  # last image <= ts
            ok=(j>=0)&((ts-im[np.clip(j,0,None)])<=25000)
            sg&=ok
        hc=int(np.argmax(sg)) if sg.any() else -1
        for name,v in (("A_pair_max",pa),("B_pair_and_available",hb),("C_signed_before_25ms",hc)):
            res[(name,v%10==true)]+=1
            if v%10!=true: bad.append((name,lg,true,v))
for n in ("A_pair_max","B_pair_and_available","C_signed_before_25ms"):
    print(n,"match",res[(n,True)],"mismatch",res[(n,False)])
for b in bad[:12]: print(b)
