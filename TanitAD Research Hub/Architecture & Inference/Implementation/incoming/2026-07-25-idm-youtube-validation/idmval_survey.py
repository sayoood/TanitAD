import torch, glob, numpy as np, math
def wrap(a): return a-(2*math.pi)*np.floor((a+math.pi)/(2*math.pi))
MAN={0:"keep",1:"accel",2:"decel",3:"stop",4:"left",5:"right",6:"laneL",7:"laneR"}
rows=[]
for p in sorted(glob.glob("/root/valdata/physicalai-val-0c5f7dac3b11/ep_*.pt")):
    d=torch.load(p,weights_only=False)
    po=d["poses"].numpy(); T=po.shape[0]; v=po[:,3]; yaw=po[:,2]
    yr=wrap(yaw[2:]-yaw[:-2])/0.2
    man=d.get("maneuvers")
    if man is not None:
        mv=man.numpy(); mu=np.bincount(mv[mv>=0],minlength=8); nneg=int((mv<0).sum())
    else:
        mu=np.zeros(8,dtype=int); nneg=0
    nz=("neg:%d "%nneg if nneg else "")+", ".join("%s:%d"%(MAN.get(i,i),mu[i]) for i in range(8) if mu[i]>0)
    rows.append((int(p.split("ep_")[1][:5]), int(d.get("episode_id",-1)), T,
                 float(v.min()), float(v.max()), float(v.mean()),
                 float(np.abs(yr).max()), float(yaw.max()-yaw.min()), nz))
    del d
hdr = "%3s %6s %4s %5s %5s %5s %5s %7s  maneuvers" % ("idx","epid","T","vmin","vmax","vmean","yr_mx","yawspan")
print(hdr)
for r in sorted(rows, key=lambda x:-(x[6])):
    print("%3d %6d %4d %5.1f %5.1f %5.1f %5.2f %7.2f  %s" % r)
