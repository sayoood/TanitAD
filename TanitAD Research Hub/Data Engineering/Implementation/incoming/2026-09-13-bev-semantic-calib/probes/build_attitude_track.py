import sys, json, numpy as np, cv2
S="/tmp/claude-0/-home-user-TanitAD/d367c501-690c-51f0-8672-b9ccb83dd3ca/scratchpad"
sys.path.insert(0,S); sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib/probes")
B="/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib"
sys.path.insert(0,B+"/probes"); sys.path.insert(0,B)
import lane_vp_wholeline as V, run_real as RR, bev_calib as BC
import attitude_tracklib as BT
recs=BT.recs
def lat40(f):
    r=recs[f]; px=np.asarray(r["x"],float); py=np.asarray(r["y"],float); m=px>0
    return abs(float(np.interp(40.,px[m],py[m]))) if m.sum()>2 else 9.
cand=[f for f,r in sorted(recs.items()) if r["complete"] and f in BT.have and r["speed_ms"]>8]
fitF=[f for i,f in enumerate(cand) if i%3!=1 and lat40(f)<0.3]     # 2/3 of frames
holdF=[f for f in cand[1::3] if lat40(f)<0.3]                      # SAME held-out set as v9
P0=V.params()
rows=[]
for F in fitF:
    img=cv2.imread(str(RR.RUN/"frames"/f"{F:06d}.jpg"),cv2.IMREAD_COLOR)
    rib=V.ribbon_uv(P0,recs[F],np.arange(7.,60.,1.0))
    fl,fr,_=V.lane_lines(img,rib)
    if not fl or not fr: continue
    (al,bl,nl),(ar,br,nr)=fl,fr
    if nl<6 or nr<3 or not (-3.5<bl<-0.6 and 0.6<br<3.5): continue
    vvp=(ar-al)/(bl-br); uvp=al+bl*vvp
    if not (440<vvp<520) or not (380<(ar+br*720)-(al+bl*720)<750): continue
    rows.append((recs[F]["t_session_s"], BT.yaw_for_u(uvp,463.0)))
A=np.array(rows); t=A[:,0]; ys=A[:,1]
print(f"fit samples: {len(A)} (v9 had 128)")
t_all=np.array(sorted(r["t_session_s"] for r in recs.values()))
tq=np.arange(np.floor(t_all.min()), np.ceil(t_all.max())+0.5, 0.5)
res={}
for half in (1.5,2.0,3.0):
    tr=BT.smooth(tq,t,ys,half=half)
    ang=[]
    for F in holdF:
        ts=recs[F]["t_session_s"]; a=BT.angle(F,float(np.interp(ts,tq,tr)),463.0)
        if a is not None: ang.append((ts,a))
    ang=np.array(ang); res[half]=(tr,ang)
    bins=[]
    for t0 in range(0,80,10):
        s=(ang[:,0]>=t0)&(ang[:,0]<t0+10)
        bins.append(f"{np.median(ang[s,1]):+.2f}" if s.sum()>=3 else "  n/a")
    v=ang[:,1]
    print(f"  window +-{half:.1f}s  held-out n {len(v)}  median {np.median(v):+.2f}  "
          f"robust sd {1.4826*np.median(np.abs(v-np.median(v))):.2f}  |  bins " + " ".join(bins))
best=min(res, key=lambda h: 1.4826*np.median(np.abs(res[h][1][:,1]-np.median(res[h][1][:,1])))
                            + abs(np.median(res[h][1][:,1])))
tr=res[best][0]
json.dump(dict(t_session_s=tq.tolist(), yaw_deg=[round(float(x),4) for x in tr],
               source=(f"lane vanishing point, whole-line Hough fits on gated STRAIGHT frames "
                       f"(n={len(A)}, 2 of every 3 frames; every 3rd held out for evaluation), "
                       f"running median +-{best} s then 1.5 s gaussian; YAW ONLY -- the horizon "
                       "track failed its paint-width check and is not applied; 2026-09-24")),
          open(S+"/attitude_track_v10.json","w"))
print(f"\n  CHOSEN window +-{best}s  yaw {tr.min():+.2f}..{tr.max():+.2f} deg -> attitude_track_v10.json")
