"""S1d scope probe 5: do ANY of the 1,167 non-navtrain trainval logs have camera pixels on the dev box? (listing only)"""
import json, os, re
D = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/"
extra = set(json.load(open(D + "inventory_db.json"))["neither_trainval_logs"])
cam = {}
base = "D:/Projects/TanitAD/data/nuplan-camera"
for sub in ("mini", "scenarios4", "probe", "scenarios"):
    for root, _d, files in os.walk(f"{base}/{sub}"):
        for f in files:
            m = re.match(r"^(\d{4}\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-\d+_\d+_\d+)_CAM_", f)
            if m: cam.setdefault(m.group(1), set()).add(sub)
print("camera zips cover", len(cam), "distinct logs:", sorted({s for v in cam.values() for s in v}))
print("overlap with the 1,167 extra logs:", len(set(cam) & extra), sorted(set(cam) & extra)[:5])
# navtest frames dir (per-log/camera zips)
nt = "D:/Projects/TanitAD/data/refe_navtest/frames"
ntl = set()
for root, _d, files in os.walk(nt):
    for f in files:
        m = re.match(r"^(\d{4}\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-\d+_\d+_\d+)", f)
        if m: ntl.add(m.group(1))
print("refe_navtest/frames logs", len(ntl), "overlap with extra:", len(ntl & extra))
json.dump({"mini_etc_camera_logs": len(cam), "overlap_extra": sorted(set(cam) & extra), "navtest_frames_logs": len(ntl), "navtest_frames_overlap_extra": len(ntl & extra)}, open(D + "cam_overlap.json", "w"), indent=1)
