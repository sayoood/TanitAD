import json,sys,hashlib
sys.path.insert(0,"C:/Users/Admin/tanitad-data/add203/repo/stack")
from tanitad.data import clip_clock as cc
from tanitad.data.v2_dataset import stable_episode_id
O="C:/Users/Admin/tanitad-data/add203/out/"; R="C:/Users/Admin/qland/landed_hold_datacov/raw/"
add={stable_episode_id(l.strip()):l.strip() for l in open(R+"ADD_203_clip_ids.txt") if l.strip()}
un=json.load(open(O+"ts_clock_unclocked.json"))
sel={int(s):v for s,v in un.items() if int(s) in add}
print('unclocked ADD clips given a camera-timestamp clock:',len(sel))
lines=[l for l in open(O+"refcv8_clip_clock_sidecar.jsonl",encoding="utf-8") if l.strip()]
out=list(lines)
for s,v in sorted(sel.items()):
    out.append(json.dumps({"sid":s,"grid_start_s":v["grid_start_s"],"dt_s":v["dt_s"],"clock_src":"camera_timestamps"})+"\n")
p=O+"refcv8_clip_clock_sidecar_ts33.jsonl"
open(p,"w",encoding="utf-8",newline="\n").writelines(out)
meta=json.load(open(O+"refcv8_clip_clock_sidecar.jsonl.meta.json"))
meta["n_rows"]=len(out); meta["n_rows_camera_timestamp_route"]=len(sel)
meta["camera_timestamp_route"]={"rule":"grid_start_s = (t_query[0] - min(ego_log_ts))/1e6 with t_query = pose_sync.resample_grid(camera ts); dt_s = pose_sync.grid_delta dt",
 "validated_against":"all 4,653 position-inversion rows: max|d grid_start_s| 4.27e-07 s, max|d dt_s| 6.2e-09 s (ctl/ts_clock_probe.py)",
 "applied_to":"ONLY the 33 unclocked clips of the 203 ADD set; the 22 old-train and 3 eval139 unclocked clips are left unverified on purpose (refcv7/eval comparability)"}
json.dump(meta,open(p+".meta.json","w",encoding="utf-8"),indent=1)
t=cc.read_clip_clock_sidecar(p).table
print('reader OK, table size',len(t))
tr=[l.strip() for l in open(R+"refcv6_train_clips.txt") if l.strip()]
allids=tr+list(add.values())
unv=[c for c in allids if stable_episode_id(c) not in t]
print('train4572 unverified with ts33 sidecar:',len(unv),'=',round(len(unv)/4572*100,3),'%')
print('md5',hashlib.md5(open(p,'rb').read()).hexdigest(), len(out),'rows')
