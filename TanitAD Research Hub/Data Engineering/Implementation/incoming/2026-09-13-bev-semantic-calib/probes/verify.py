import sys, numpy as np, json
sys.path.insert(0,"/home/user/TanitAD/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-09-13-bev-semantic-calib")
import trajrecon; sys.modules.setdefault('trajlib', sys.modules['trajrecon'])
import bev_calib as BC, lag_scale as LS, run_real as RR

recs=RR.load_records(RR.RUN)
anchors=RR.build_anchors(recs, RR.RUN, 10, 0.6, 0.1, 4000)
grid=BC.BevGrid(x_range=(5.0,40.0), y_range=(-10.0,10.0), cell=0.08)
print(f"anchors {len(anchors)}  (BEV AGREEMENT — an instrument that never saw the flow data)\n")

def P_of(f,h,hz,yaw=-7.01,lat=-0.12):
    P=dict(RR.NOMINAL); P.update(yaw=np.deg2rad(yaw), height=h, fx=f, lateral=lat)
    P["pitch"]=LS.pitch_for_horizon(P,hz); return P

arms=[
 ("SHIPPED (f 1478.3, h 1.17, horizon 464.4)",      P_of(1478.3,1.17,464.4,-7.007,0.25)),
 ("my earlier 'best' (f 1478.3, h 1.21, hz 523.4)", P_of(1478.3,1.21,523.4)),
 ("h 1.17 at the NEW f*h (f 2280, hz 465)",         P_of(2280.0,1.17,465.0)),
 ("** RENDERED: f 1438, h 1.70, horizon 465 **",    P_of(1438.0,1.70,465.0)),
 ("f 1525, h 1.75, horizon 437 (row-flow point)",   P_of(1525.0,1.75,437.4)),
 ("f 1300, h 1.72, horizon 454 (2-D flow point)",   P_of(1300.0,1.72,454.0)),
]
res=[]
for name,P in arms:
    s=RR.multi_agreement(anchors,P,grid); res.append((s,name))
    print(f"   {s:.6e}   {name}")
res.sort(reverse=True)
print(f"\n   BEST: {res[0][1]}")
base=[s for s,n in res if n.startswith("SHIPPED")][0]
print(f"   best vs shipped: {100*(res[0][0]/base-1):+.1f}%")
for s,n in res:
    print(f"     {100*(s/base-1):+7.1f}%  {n}")
