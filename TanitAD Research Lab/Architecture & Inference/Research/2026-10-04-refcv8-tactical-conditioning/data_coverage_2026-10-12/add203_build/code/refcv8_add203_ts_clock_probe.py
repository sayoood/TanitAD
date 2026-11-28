import json,sys,os,random
import numpy as np, pandas as pd
sys.path.insert(0,"C:/Users/Admin/tanitad-data/add203/repo/stack")
from tanitad.data.v2_dataset import stable_episode_id
from tanitad.data import pose_sync as PS
O="C:/Users/Admin/tanitad-data/add203/out/"; R="C:/Users/Admin/qland/landed_hold_datacov/raw/"
CAM="C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov/"
EGO="C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/"
clk={}
for l in open(O+"refcv8_clip_clock_sidecar.jsonl",encoding="utf-8"):
    if l.strip():
        r=json.loads(l); clk[r["sid"]]=r
ids=[l.strip() for l in open(R+"ADD_203_clip_ids.txt") if l.strip()]+[l.strip() for l in open(R+"refcv6_train_clips.txt") if l.strip()]+[l.strip() for l in open(R+"eval139_ids.txt") if l.strip()]
def ts_clock(cid):
    ts=pd.read_parquet(CAM+cid+".timestamps.parquet"); tf=ts[next(c for c in ts.columns if "time" in c.lower())].to_numpy(np.float64)
    eg=pd.read_parquet(EGO+cid+".parquet"); te=np.sort(eg["timestamp"].to_numpy(np.float64))
    t_query,fi,unit=PS.resample_grid(tf)
    g0=(t_query[0]-te[0])/unit
    d,dt,n=PS.grid_delta(tf)
    return g0,dt
rows=[]; err=[]; 
rnd=random.Random(0)
clocked=[c for c in ids if stable_episode_id(c) in clk]; unclocked=[c for c in ids if stable_episode_id(c) not in clk]
sample=clocked if len(sys.argv)>1 and sys.argv[1]=="all" else rnd.sample(clocked,800)
for c in sample:
    g0,dt=ts_clock(c); r=clk[stable_episode_id(c)]
    err.append((abs(g0-r["grid_start_s"]),abs(dt-r["dt_s"])))
e=np.array(err)
print('validation n=',len(e),'max|grid_start diff| s=',e[:,0].max(),'p99',np.quantile(e[:,0],0.99),'max|dt diff|',e[:,1].max())
out={}
for c in unclocked:
    g0,dt=ts_clock(c); out[stable_episode_id(c)]={"clip":c,"grid_start_s":g0,"dt_s":dt}
print('unclocked n',len(unclocked)); json.dump({str(k):v for k,v in out.items()},open(O+"ts_clock_unclocked.json","w"))
print('range g0 of unclocked',min(v["grid_start_s"] for v in out.values()),max(v["grid_start_s"] for v in out.values()))
