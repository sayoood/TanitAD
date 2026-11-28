import glob, json, math, os
import numpy as np, torch
from tanitad.data.physicalai import WHEELBASE
D = r"C:\Users\Admin\refav1_eval_slice\t1_dump"; EPS = r"C:\Users\Admin\refav1_eval_slice\eps"
man = json.load(open(os.path.join(D, "manifest.json")))
names = {int(e["file_index"]): e["name"] for e in man["episodes"]}
def integ(a, kap, v0, dt=0.2):
    x=y=yaw=0.0; v=float(v0); out=np.zeros((len(a),2))
    for k in range(len(a)):
        x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt; yaw+=v*float(kap[k])*dt
        v=max(0.0, v+float(a[k])*dt); out[k]=(x,y)
    return out
rows=[]
for fi,f in enumerate(sorted(glob.glob(os.path.join(D,"ep*.npz")))):
    z=np.load(f); o=torch.load(os.path.join(EPS,f"{names[fi]}.v2ep.pt"),map_location="cpu",weights_only=False)
    v=o["poses"][:,3].numpy().astype(float); st=o["actions"][:,0].numpy().astype(float)
    for i in range(z["g"].shape[0]):
        t=int(z["ws"][i]); v0=float(z["v0"][i]); g=z["g"][i].astype(float)
        a_h=(v[2*t]-v[2*t-2])/0.2; k_h=st[2*t-2]
        ha_ship=integ(np.full(10,a_h), np.full(10,k_h), v0)
        ha_fix =integ(np.full(10,a_h), np.full(10,math.tan(k_h)/WHEELBASE), v0)
        ha0    =integ(np.zeros(10), np.zeros(10), v0)
        exc=float(np.abs(g[:,1]).max())
        rows.append(dict(curved=exc>=0.3,
            ship=float(np.mean(np.abs(ha_ship[:,1]-g[:,1]))), fix=float(np.mean(np.abs(ha_fix[:,1]-g[:,1]))),
            ha0=float(np.mean(np.abs(ha0[:,1]-g[:,1]))),
            ship_l=float(np.mean(np.abs(ha_ship[:,0]-g[:,0]))), fix_l=float(np.mean(np.abs(ha_fix[:,0]-g[:,0]))),
            ha0_l=float(np.mean(np.abs(ha0[:,0]-g[:,0])))))
cur=[r for r in rows if r["curved"]]; sti=[r for r in rows if not r["curved"]]
for nm,key,keyl in (("ha shipped","ship","ship_l"),("ha REPAIRED","fix","fix_l"),("ha0 (const-v)","ha0","ha0_l")):
    print(f"{nm:14s} LATcurved {np.mean([r[key] for r in cur]):.4f}  LATstraight {np.mean([r[key] for r in sti]):.4f}  LONall {np.mean([r[keyl] for r in rows]):.4f}  (n {len(cur)}/{len(sti)})")
