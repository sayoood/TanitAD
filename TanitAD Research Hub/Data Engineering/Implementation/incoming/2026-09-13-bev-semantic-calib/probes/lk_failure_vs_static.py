import sys, numpy as np, cv2
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import flow_scale as FS, lag_scale as LS, run_real as RR

# Is the row-830+ flow collapse (a) content static in the image, or (b) the LK
# tracker failing at large displacement and returning ~0? Test: re-track the SAME
# rows with a much larger search (bigger window, more pyramid levels). Static
# content cannot start moving; a failing tracker can start succeeding.
recs=RR.load_records(RR.RUN); fdir=RR.RUN/"frames"
usable=[r for r in recs if r["complete"] and r["speed_ms"]>8.0 and (fdir/f"{int(r['frame']):06d}.jpg").exists()]
P=dict(RR.NOMINAL); P.update(height=1.427, fx=1713.0, lateral=-0.126, yaw=np.deg2rad(-6.80))
P["pitch"]=LS.pitch_for_horizon(P,465.0)

CONFIGS={
 "default (win 21, lvl 4)": dict(winSize=(21,21), maxLevel=4,
     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,0.01)),
 "wide    (win 51, lvl 7)": dict(winSize=(51,51), maxLevel=7,
     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,0.005)),
}
BANDS=[(660,760),(760,830),(830,900),(900,960),(960,1030)]
for name,lk in CONFIGS.items():
    got={b:[] for b in BANDS}; pred={b:[] for b in BANDS}; seed={b:0 for b in BANDS}; trk={b:0 for b in BANDS}
    for pi in np.linspace(0,len(usable)-1,25).astype(int):
        a=usable[pi]; f0=int(a["frame"])
        i0=cv2.imread(str(fdir/f"{f0:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
        p1=fdir/f"{f0+1:06d}.jpg"
        if i0 is None or not p1.exists(): continue
        i1=cv2.imread(str(p1),cv2.IMREAD_GRAYSCALE)
        H,W=i0.shape
        m=np.zeros((H,W),np.uint8); m[660:1030,:]=255
        pts=cv2.goodFeaturesToTrack(i0,maxCorners=900,qualityLevel=0.004,minDistance=8,mask=m,blockSize=7)
        if pts is None: continue
        q1,s1,_=cv2.calcOpticalFlowPyrLK(i0,i1,pts,None,**lk)
        q0,s0,_=cv2.calcOpticalFlowPyrLK(i1,i0,q1,None,**lk)
        ok=(s1.ravel()==1)&(s0.ravel()==1)&(np.linalg.norm((q0-pts).reshape(-1,2),axis=1)<1.0)
        P0=pts.reshape(-1,2); P1=q1.reshape(-1,2)
        t=np.asarray(a["t"],float); dt=1/29.922
        pose=(float(np.interp(dt,t,np.asarray(a["x"],float))),
              float(np.interp(dt,t,np.asarray(a["y"],float))),
              float(np.interp(dt,t,np.asarray(a["yaw"],float))))
        pr=FS.predict(P0,P,pose)
        for lo,hi in BANDS:
            b=(P0[:,1]>=lo)&(P0[:,1]<hi)
            seed[(lo,hi)]+=int(b.sum()); trk[(lo,hi)]+=int((b&ok).sum())
            sel=b&ok&np.isfinite(pr[:,1])
            got[(lo,hi)]+= list(P1[sel,1]-P0[sel,1]); pred[(lo,hi)]+= list(pr[sel,1]-P0[sel,1])
    print(f"\n{name}")
    print(f"  {'rows':>10} {'seeds':>7} {'tracked':>8} {'rate':>6}  {'measured dv':>12} {'predicted dv':>13}")
    for b in BANDS:
        g=np.array(got[b]); p=np.array(pred[b])
        mg=np.median(g) if len(g)>10 else np.nan
        mp=np.median(p) if len(p)>10 else np.nan
        print(f"  {b[0]:4d}-{b[1]:4d} {seed[b]:7d} {trk[b]:8d} {100*trk[b]/max(seed[b],1):5.0f}%"
              f"  {mg:11.1f} {mp:12.1f}")
