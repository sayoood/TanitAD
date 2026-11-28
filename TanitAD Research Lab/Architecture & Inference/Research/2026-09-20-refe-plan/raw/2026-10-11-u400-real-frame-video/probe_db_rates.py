"""MEASURE (not assume) the real sensor rates from a nuPlan log DB: camera image rate per channel, lidar_pc rate, ego_pose rate, the
image-vs-ego_pose timestamp offset, and that every DB image filename is present in the raw camera zip.   python probe_db_rates.py <log>"""
import os
import sqlite3
import sys
import zipfile

import numpy as np

LOG = sys.argv[1]
DB = f"E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test/{LOG}.db"
con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
print("log:", LOG)
for ch in ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"):
    ts = np.array([r[0] for r in con.execute("select i.timestamp from image i join camera c on c.token=i.camera_token where c.channel=? order by i.timestamp", (ch,))], dtype=np.int64)
    dt = np.diff(ts) / 1e3
    print(f"{ch}: n={len(ts)} span={(ts[-1] - ts[0]) / 1e6:.3f} s rate={(len(ts) - 1) / ((ts[-1] - ts[0]) / 1e6):.6f} Hz  dt ms min/median/max/std = "
          f"{dt.min():.3f} / {np.median(dt):.3f} / {dt.max():.3f} / {dt.std():.3f}   n(dt>150 ms)={int((dt > 150).sum())}")
lt = np.array([x[0] for x in con.execute("select timestamp from lidar_pc order by timestamp")], dtype=np.int64)
dt = np.diff(lt) / 1e3
print(f"lidar_pc: n={len(lt)} rate={(len(lt) - 1) / ((lt[-1] - lt[0]) / 1e6):.6f} Hz  dt ms min/median/max = {dt.min():.3f} / {np.median(dt):.3f} / {dt.max():.3f}  (lidar_box rows attach to these)")
et = np.array([x[0] for x in con.execute("select timestamp from ego_pose order by timestamp")], dtype=np.int64)
print(f"ego_pose: n={len(et)} rate={(len(et) - 1) / ((et[-1] - et[0]) / 1e6):.3f} Hz")
d = np.array([(x[0] - x[1]) for x in con.execute("select i.timestamp, e.timestamp from image i join camera c on c.token=i.camera_token join ego_pose e on e.token=i.ego_pose_token where c.channel='CAM_F0'")])
print(f"image.timestamp - its own ego_pose.timestamp (CAM_F0), us: min {d.min()} max {d.max()}")
for ch, root in (("CAM_F0", "mini"), ("CAM_L0", "scenarios4"), ("CAM_R0", "scenarios4"), ("CAM_B0", "scenarios4")):
    zp = f"E:/Projects/TanitAD/data/nuplan-camera/{root}/{LOG}_{ch}.zip"
    if not os.path.exists(zp):
        print(f"{ch}: raw zip not available locally ({root})")
        continue
    bn = {os.path.basename(n) for n in zipfile.ZipFile(zp).namelist()}
    fn = [x[0] for x in con.execute("select i.filename_jpg from image i join camera c on c.token=i.camera_token where c.channel=?", (ch,))]
    print(f"{ch}: DB filenames present in raw zip: {sum(os.path.basename(f) in bn for f in fn)}/{len(fn)}  ({root})")
