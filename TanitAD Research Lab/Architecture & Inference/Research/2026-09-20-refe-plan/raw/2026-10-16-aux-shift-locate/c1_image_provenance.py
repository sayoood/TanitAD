#!/usr/bin/env python3
"""H-d (input level, CPU, read-only): are the navtest camera frames the same KIND of file as the navtrain camera frames the trunk/head saw?
Compares JPEG container facts only (dimensions, file size, quality-table proxy, EXIF absence) between
  navtest: D:/Projects/TanitAD/data/refe_navtest/frames/<log>_<CAM>.zip   (the frames the planner path read for the 1,058 C3 tokens)
  navtrain: the first members of D:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz  (1 of the 32 train pixel shards; the only train pixels on this machine)
This can only EXCLUDE an input-pipeline difference (resolution / re-encode); it cannot find a photometric shift.  Not the trunk's ctx."""
import io, json, os, sys, tarfile, zipfile, glob, statistics
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C

def facts(b):
    im = Image.open(io.BytesIO(b))
    q = getattr(im, "quantization", None)
    qsum = int(sum(q[0])) if q else None          # sum of the luminance quantisation table: a re-encode at another quality changes it
    return dict(size=im.size, mode=im.mode, bytes=len(b), qsum=qsum, sub=getattr(im, "layer", None) and str(im.layer)[:40])

res = {"navtest": {}, "navtrain": {}}
c3 = C.c3()
logs = sorted(set(c3["token_log"].values()))
rng = np.random.default_rng(3)
for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
    rows = []
    for lg in rng.choice(logs, 12, replace=False):
        p = f"D:/Projects/TanitAD/data/refe_navtest/frames/{lg}_{cam}.zip"
        if not os.path.exists(p):
            continue
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg"))]
            for n in rng.choice(names, min(5, len(names)), replace=False):
                rows.append(facts(z.read(n)))
    res["navtest"][cam] = dict(n=len(rows), sizes=sorted(set(map(tuple, (r["size"] for r in rows)))), mode=sorted(set(r["mode"] for r in rows)),
                               bytes_mean=float(np.mean([r["bytes"] for r in rows])), bytes_sd=float(np.std([r["bytes"] for r in rows])), qsum=sorted(set(r["qsum"] for r in rows)))
    print("navtest", cam, res["navtest"][cam], flush=True)
# navtrain shard 25: stream the first members
got = {}
with tarfile.open("D:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz", "r|gz") as tf:
    for m in tf:
        if not m.isfile() or not m.name.lower().endswith((".jpg", ".jpeg")):
            continue
        cam = next((c for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0") if c in m.name), None)
        if cam is None:
            continue
        got.setdefault(cam, [])
        if len(got[cam]) < 60:
            got[cam].append(facts(tf.extractfile(m).read()))
        if all(len(got.get(c, [])) >= 60 for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")):
            break
for cam, rows in got.items():
    res["navtrain"][cam] = dict(n=len(rows), sizes=sorted(set(map(tuple, (r["size"] for r in rows)))), mode=sorted(set(r["mode"] for r in rows)),
                                bytes_mean=float(np.mean([r["bytes"] for r in rows])), bytes_sd=float(np.std([r["bytes"] for r in rows])), qsum=sorted(set(r["qsum"] for r in rows)))
    print("navtrain", cam, res["navtrain"][cam], flush=True)
C.jdump(res, os.path.join(C.HERE, "c1_image_provenance.json"))
print("DONE")
