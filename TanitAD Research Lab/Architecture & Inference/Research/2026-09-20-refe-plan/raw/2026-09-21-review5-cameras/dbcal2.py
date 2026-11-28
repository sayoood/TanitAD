import sqlite3, glob, pickle, collections, math, random, json
import numpy as np
dbs = sorted(glob.glob('D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/**/*.db', recursive=True))
print(f"{len(dbs)} log DBs on disk")
def dec(b):
    if b is None: return None
    try: return np.asarray(pickle.loads(bytes(b)), dtype=np.float64)
    except Exception: return None
# 1) prove the decoder on one DB
c=sqlite3.connect(f"file:{dbs[0]}?mode=ro",uri=True)
rows=c.execute("SELECT channel,intrinsic,translation,rotation,width,height,distortion FROM camera").fetchall(); c.close()
print(f"\n--- {dbs[0].split('/')[-1]} ---")
for ch,K,T,R,w,h,D in sorted(rows):
    K=dec(K); T=dec(T); R=dec(R); D=dec(D)
    print(f" {ch:8s} {w}x{h}  fx={K[0][0]:9.3f} fy={K[1][1]:9.3f} cx={K[0][2]:8.3f} cy={K[1][2]:8.3f} "
          f" HFOV={2*math.degrees(math.atan(K[0][2]/K[0][0])):6.2f}  dist={np.round(D,4) if D is not None else None}")
print()
# 2) sweep
random.seed(0); sample = random.sample(dbs, 300)
K_by=collections.defaultdict(collections.Counter); E_by=collections.defaultdict(set); WH=collections.Counter(); n=0
for p in sample:
    try:
        c=sqlite3.connect(f"file:{p}?mode=ro",uri=True)
        rows=c.execute("SELECT channel,intrinsic,translation,rotation,width,height FROM camera").fetchall(); c.close()
    except Exception: continue
    n+=1
    for ch,K,T,R,w,h in rows:
        K=dec(K); T=dec(T); R=dec(R)
        if K is None or K.size<9: continue
        K=K.reshape(3,3); WH[(int(w),int(h))]+=1
        K_by[ch][(round(float(K[0,0]),3),round(float(K[1,1]),3),round(float(K[0,2]),3),round(float(K[1,2]),3))]+=1
        if T is not None and R is not None:
            E_by[ch].add((tuple(np.round(np.ravel(T)[:3],4)),tuple(np.round(np.ravel(R)[:4],4))))
print(f"swept {n} DBs.  image sizes: {dict(WH)}\n")
print(f"{'channel':9s} {'#K':>3s} {'#extrinsics':>12s}   distinct intrinsics (fx, fy, cx, cy)  [logs]  HFOV")
for ch in sorted(K_by):
    parts=[]
    for k,c2 in K_by[ch].most_common():
        parts.append(f"({k[0]:.1f},{k[1]:.1f},{k[2]:.1f},{k[3]:.1f})[{c2}] {2*math.degrees(math.atan(k[2]/k[0])):.1f}deg")
    print(f"{ch:9s} {len(K_by[ch]):3d} {len(E_by[ch]):12d}   " + " | ".join(parts))
print()
cal=json.load(open('D:/Projects/TanitAD/data/nuplan_cam_calib.json'))
print("banked nuplan_cam_calib.json vs the MODAL rig per channel:")
for ch in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0'):
    k=K_by[ch].most_common(1)[0][0]
    b=cal[ch]['K']
    same = abs(b[0][0]-k[0])<1e-3 and abs(b[1][1]-k[1])<1e-3 and abs(b[0][2]-k[2])<1e-3 and abs(b[1][2]-k[3])<1e-3
    print(f"  {ch}: banked ({b[0][0]},{b[1][1]},{b[0][2]},{b[1][2]})  modal {k}  match={same}  "
          f"modal share {K_by[ch].most_common(1)[0][1]}/{sum(K_by[ch].values())}")
