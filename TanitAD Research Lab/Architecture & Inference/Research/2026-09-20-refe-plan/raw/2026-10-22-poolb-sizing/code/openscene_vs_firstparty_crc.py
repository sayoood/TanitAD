#!/usr/bin/env python3
"""Is an OpenScene-archive JPEG byte-identical to the nuPlan first-party JPEG of the same name?  LOCAL ONLY, no data bytes read:
compares the ZIP central-directory (size, CRC32) of
  A = frames extracted from the OpenScene TEST archives   (D:/Projects/TanitAD/data/refe_navtest/frames/<log>_<CAM>.zip)
  B = frames extracted from the nuPlan first-party S3 mini camera zips (D:/Projects/TanitAD/data/nuplan-camera/{mini,scenarios4}/...)
for every log + camera present in both. Writes raw/openscene_vs_firstparty_crc.json.
(The mini/scenarios4 zips were written by fetch_front_camera.write_run_container: members named by the jpg hash, stored as written, CRC32 recomputed from the bytes read.)"""
import os, sys, json, zipfile, re
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
A = "D:/Projects/TanitAD/data/refe_navtest/frames/"
Bdirs = ["D:/Projects/TanitAD/data/nuplan-camera/mini/", "D:/Projects/TanitAD/data/nuplan-camera/scenarios4/"]
def cdmap(p):
    with zipfile.ZipFile(p) as z:
        return {os.path.basename(i.filename): (i.file_size, i.CRC) for i in z.infolist() if not i.is_dir()}
res = {"pairs": [], "n_logs_cams": 0, "n_common": 0, "n_equal": 0, "n_size_equal": 0}
afiles = {f for f in os.listdir(A) if f.endswith(".zip")}
for bd in Bdirs:
    for f in sorted(os.listdir(bd)):
        if not f.endswith(".zip"):
            continue
        # mini/ and scenarios4/ names: <log>_CAM_XX.zip (some scenarios4 dirs are subdirs: skip dirs)
        if f in afiles:
            a = cdmap(A + f); b = cdmap(bd + f)
            common = set(a) & set(b)
            eq = sum(1 for k in common if a[k] == b[k]); seq = sum(1 for k in common if a[k][0] == b[k][0])
            res["pairs"].append({"zip": f, "firstparty_dir": bd, "n_openscene": len(a), "n_firstparty": len(b), "n_common": len(common), "n_size_eq": seq, "n_size_crc_eq": eq})
            res["n_logs_cams"] += 1; res["n_common"] += len(common); res["n_equal"] += eq; res["n_size_equal"] += seq
json.dump(res, open(os.path.join(RAW, "openscene_vs_firstparty_crc.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "pairs"}), len(res["pairs"]), "zips")
for p in res["pairs"][:20]: print(p)
