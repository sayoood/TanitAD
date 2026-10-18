import sys, json, numpy as np
sys.path.insert(0, ".")
import u400_lib as L
runs = json.load(open("video_runs.json")); meta = json.load(open("video_meta.json"))
run = runs[0]
ld = L.LogData(run["log"])
pts = ld.lid_pose[[ld.lid_tok[t] for t in run["tokens"]]][:, :2]
print("ego xy range", pts.min(0), pts.max(0))
ld.load_map(pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max(), meta[run["tokens"][0]]["map"])
for k,(arr,tree) in ld.map.items():
    print(k, len(arr), None if len(arr)==0 else arr[0].bounds)
pose = L.Pose(ld.lid_pose[ld.lid_tok[run["tokens"][3]]])
print("pose", pose.T, pose.yaw)
for k in ld.map:
    print(k, len(L.view_clip(ld.map[k], pose, -10, 80, 30)))
