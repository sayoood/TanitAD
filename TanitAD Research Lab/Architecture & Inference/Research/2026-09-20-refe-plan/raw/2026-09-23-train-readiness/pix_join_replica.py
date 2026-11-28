"""Fast replica of make_row's image choice on the one shard-25 log we hold, vs the tarball.

Replicates build_teacher_rollouts.make_row l.332-339 exactly: per camera, the image whose
timestamp is NEAREST the frame's lidar_pc timestamp, refused beyond 60 ms. The real producer is
run separately on the same log; this replica exists to get the answer in seconds and to break the
result down (which camera, what time offset) if the join is partial.
"""
import glob
import os
import sqlite3
import sys

import numpy as np
import yaml

LOG = open(sys.argv[1], encoding="utf-8").read().split()[0]
PKG = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
       "2026-09-20-refe-plan")
Y = yaml.safe_load(open(PKG + "/splits/navtrain.yaml", encoding="utf-8"))
tokens = set(Y.get("tokens") or [])
print(f"split tokens {len(tokens):,}")
dbs = [p for p in glob.glob("D:/Projects/TanitAD/data/nuplan/**/" + LOG + ".db", recursive=True)]
print("db:", dbs)
db = dbs[0]
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")

members = {}
with open("D:/Projects/TanitAD/data/openscene_probe/members.txt", encoding="utf-8") as f:
    for ln in f:
        p = ln.strip().split("/")
        if len(p) == 4 and p[1] == LOG and p[3].endswith(".jpg"):
            members.setdefault(p[2], set()).add(p[3])
print("tarball frames for this log per camera:", {k: len(v) for k, v in sorted(members.items())})

c = sqlite3.connect(db)
lp = c.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc ORDER BY timestamp").fetchall()
cam = {}
for ch in CAMS:
    ct = c.execute("SELECT token FROM camera WHERE channel=?", (ch,)).fetchone()[0]
    rows = c.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token=? "
                     "ORDER BY timestamp", (ct,)).fetchall()
    cam[ch] = (np.array([t for t, _ in rows], dtype=np.int64), [f for _, f in rows])
c.close()
nav = [(t, ts) for t, ts in lp if t in tokens]
print(f"lidar_pc rows {len(lp):,}; navtrain tokens in this log {len(nav)}")

hit = {ch: 0 for ch in CAMS}
all4 = 0
offs = {ch: [] for ch in CAMS}
miss = []
for tok, ts in nav:
    ok = 0
    for ch in CAMS:
        tss, fn = cam[ch]
        j = int(np.argmin(np.abs(tss - ts)))
        d = abs(int(tss[j]) - ts) / 1000.0
        base = os.path.basename(fn[j])
        if d <= 60.0 and base in members.get(ch, ()):
            hit[ch] += 1
            ok += 1
        else:
            # where IS the shipped image for this frame? nearest shipped timestamp, if any
            miss.append((tok, ch, round(d, 1), base))
        offs[ch].append(d)
    all4 += ok == 4
print("hits per camera:", hit, f"of {len(nav)}")
print(f"frames with all 4 cameras shipped: {all4}/{len(nav)}")
for ch in CAMS:
    o = np.array(offs[ch])
    print(f"  {ch}: |dt| to chosen image median {np.median(o):.1f} ms, max {o.max():.1f} ms")
if miss:
    print("first misses:", miss[:8])

# ⭐ the discriminating half: for the tarball's images of this log, which lidar frame is each
# NEAREST to, and is that frame a navtrain token? If the tarball ships images for frames we do
# not select (or vice versa), the association rule differs.
ship_ts = {}
for ch in CAMS:
    tss, fn = cam[ch]
    idx = {os.path.basename(f): t for t, f in zip(tss, fn)}
    ship_ts[ch] = sorted(idx[b] for b in members.get(ch, ()) if b in idx)
lp_ts = np.array([ts for _, ts in lp], dtype=np.int64)
navset = {ts for _, ts in nav}
for ch in CAMS:
    near_nav = sum(int(lp_ts[int(np.argmin(np.abs(lp_ts - t)))]) in navset for t in ship_ts[ch])
    print(f"  shipped {ch}: {len(ship_ts[ch])} images resolve in the DB; "
          f"{near_nav} are nearest to a navtrain frame")
if all4 == len(nav) and nav:
    print("ZZREPLICA_COMPLETEZZ")
else:
    print(f"ZZREPLICA_PARTIAL_{all4}_OF_{len(nav)}ZZ")
