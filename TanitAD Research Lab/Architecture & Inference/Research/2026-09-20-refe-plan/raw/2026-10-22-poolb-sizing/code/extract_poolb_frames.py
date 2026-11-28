#!/usr/bin/env python3
"""Stream the LOCAL openscene_metadata_trainval.tgz once (sha256 = HF LFS oid, prior probe raw/sha_check.json), keep the 60 pool-B pkl,
read each with the restricted unpickler and write a compact per-frame table: token, timestamp_us, has_route, driving_command,
the four P2/P6 cameras' jpg names (cams[CAM]['data_path']) and the four cameras' K + distortion. No network. numpy venv required.
Out: raw/poolB_frames.json.gz = {log: {pkl_sha256, n_frames, frames:[{i,token,ts,route,cmd,jpg:[F0,L0,R0,B0],K:[...]}]}}"""
import sys, os, json, gzip, tarfile, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pklio
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
TGZ = "D:/TanitAstraResearch/data/e024_corpus_20261010/openscene_metadata_trainval.tgz"
want = {r["log"] for r in json.load(open(os.path.join(RAW, "poolB_logs.json")))}
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
out = {}; t0 = time.time()
with tarfile.open(TGZ, mode="r|gz") as tf:
    for m in tf:
        if not m.isfile() or not m.name.endswith(".pkl"):
            continue
        log = m.name.rsplit("/", 1)[-1][:-4]
        if log not in want:
            continue
        data = tf.extractfile(m).read()
        pk = pklio.load_pkl_bytes(data)
        frames = []
        for i, f in enumerate(pk):
            cm = f["cams"]
            frames.append({"i": i, "token": f["token"], "ts": int(f["timestamp"]), "route": bool(f.get("roadblock_ids")), "cmd": [int(x) for x in f["driving_command"]],
                           "jpg": [cm[c]["data_path"] for c in CAMS],
                           "K": [[float(cm[c]["cam_intrinsic"][0][0]), float(cm[c]["cam_intrinsic"][1][1]), float(cm[c]["cam_intrinsic"][0][2]), float(cm[c]["cam_intrinsic"][1][2])] + [float(x) for x in cm[c]["distortion"]] for c in CAMS]})
        out[log] = {"pkl_sha256": hashlib.sha256(data).hexdigest(), "pkl_bytes": len(data), "n_frames": len(frames), "frames": frames}
        if len(out) == len(want):
            break
json.dump(out, gzip.open(os.path.join(RAW, "poolB_frames.json.gz"), "wt"))
print(len(out), "of", len(want), "pool-B pkl read in", round(time.time() - t0), "s")
