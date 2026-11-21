"""Ensure OpenCV Haar cascades are available for the privacy (face+plate) blur
pass. opencv-python-headless 5.x stopped bundling the cascade XMLs. Fetch the
canonical files from the official OpenCV repo (passive XML data, not code) into
a local dir and verify each loads via cv2.CascadeClassifier. Prints a JSON report.
If the fetch fails (egress blocked), the caller must escalate rather than store
un-blurred footage.
"""
from __future__ import annotations
import json, os, urllib.request

DEST = "/workspace/tmp/yt_pilot/cascades"
os.makedirs(DEST, exist_ok=True)
# opencv 4.x tag has the stable cascade set (5.x master moved things around).
BASE = "https://raw.githubusercontent.com/opencv/opencv/4.10.0/data/haarcascades/"
FILES = ["haarcascade_frontalface_default.xml",
         "haarcascade_frontalface_alt2.xml",
         "haarcascade_profileface.xml",
         "haarcascade_russian_plate_number.xml"]

report = {"dest": DEST, "downloaded": {}, "loads": {}, "errors": []}

# show what cv2 thinks it has (to confirm the gap)
try:
    import cv2
    d = cv2.data.haarcascades
    report["cv2_haar_dir"] = d
    report["cv2_haar_dir_listing"] = sorted(os.listdir(d)) if os.path.isdir(d) else "MISSING"
except Exception as e:
    report["errors"].append(f"cv2 import: {type(e).__name__}: {e}")

for f in FILES:
    dst = os.path.join(DEST, f)
    try:
        if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
            req = urllib.request.Request(BASE + f, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as o:
                o.write(r.read())
        report["downloaded"][f] = os.path.getsize(dst)
    except Exception as e:
        report["errors"].append(f"download {f}: {type(e).__name__}: {e}")
        continue

# verify each loads
try:
    import cv2
    for f in FILES:
        dst = os.path.join(DEST, f)
        if os.path.exists(dst):
            c = cv2.CascadeClassifier(dst)
            report["loads"][f] = not c.empty()
except Exception as e:
    report["errors"].append(f"load: {type(e).__name__}: {e}")

print("GET_CASCADES_JSON_START")
print(json.dumps(report, indent=2))
print("GET_CASCADES_JSON_END")
