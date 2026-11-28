import sys, json, numpy as np, shapely
sys.path.insert(0, ".")
import u400_lib as L
runs = json.load(open("video_runs.json")); meta = json.load(open("video_meta.json"))
run = runs[0]
ld = L.LogData(run["log"])
pts = ld.lid_pose[[ld.lid_tok[t] for t in run["tokens"]]][:, :2]
ld.load_map(pts[:,0].min(), pts[:,1].min(), pts[:,0].max(), pts[:,1].max(), meta[run["tokens"][0]]["map"])
f0 = ld.img["CAM_F0"]
k = ld.nearest("CAM_F0", int(ld.lid_ts[ld.lid_tok[run["tokens"][3]]]))
pose = L.Pose(f0["pose"][k + 2]); cam = ld.cam["CAM_F0"]
print("cam t", cam.t, "R", np.round(cam.R, 3).tolist(), "fx", cam.fx, cam.cx, cam.cy)
print("ego pose z", pose.T[2], "Rm", np.round(pose.Rm,4).tolist())
polys = L.view_clip(ld.map["area"], pose, 2.5, 80.0, 40.0)
print(len(polys))
for p in polys[:8]:
    xy = np.asarray(p.exterior.coords)
    print("poly bounds", np.round(p.bounds,1), "area", round(p.area), "holes", len(p.interiors))
    G = np.c_[xy, np.zeros(len(xy))]
    Pb = G @ pose.Rz.T @ pose.Rm
    pc = cam.to_cam(Pb)
    uv, ok = cam.project(Pb, near=0.8)
    print("   z range", pc[:,2].min().round(1), pc[:,2].max().round(1), "uv x", uv[:,0].min().round(), uv[:,0].max().round(), "uv y", uv[:,1].min().round(), uv[:,1].max().round())
