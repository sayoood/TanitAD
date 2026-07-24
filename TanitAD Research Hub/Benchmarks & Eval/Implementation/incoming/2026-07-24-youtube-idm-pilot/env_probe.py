"""Self-contained env probe for the YouTube-IDM pilot on pod3.
No tanitad import; just reports what tooling + network is available so we know
whether P2-P4 are feasible or must be escalated. Prints a JSON blob at the end.
"""
from __future__ import annotations
import json, shutil, socket, subprocess, sys, urllib.request, importlib.util

out = {}

# ---- python / torch / cuda ----
out["python"] = sys.version.split()[0]
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"

# ---- opencv + bundled Haar cascades (face + plate) ----
try:
    import cv2, os
    out["cv2"] = cv2.__version__
    cdir = getattr(cv2.data, "haarcascades", None)
    out["haar_dir"] = cdir
    cascades = {}
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_russian_plate_number.xml",
                 "haarcascade_profileface.xml"):
        p = os.path.join(cdir, name) if cdir else ""
        cascades[name] = bool(p and os.path.exists(p))
    out["haar_cascades"] = cascades
except Exception as e:
    out["cv2_error"] = f"{type(e).__name__}: {e}"

# ---- optional stronger detectors ----
for mod in ("av", "ultralytics", "mediapipe", "yt_dlp"):
    out[f"has_{mod}"] = importlib.util.find_spec(mod) is not None

# ---- ffmpeg / yt-dlp binaries ----
out["ffmpeg_bin"] = shutil.which("ffmpeg")
out["ytdlp_bin"] = shutil.which("yt-dlp")

# ---- network reachability (does NOT download; just connect + tiny GET) ----
def reach(host, port=443, timeout=8):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception as e:
        return f"{type(e).__name__}: {e}"

out["net_tcp"] = {h: reach(h) for h in
                  ("pypi.org", "www.youtube.com", "youtube.com",
                   "www.google.com", "huggingface.co")}

def http_head(url, timeout=10):
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"status": r.status}
    except Exception as e:
        return f"{type(e).__name__}: {e}"

out["http_head"] = {
    "youtube": http_head("https://www.youtube.com/"),
    "pypi": http_head("https://pypi.org/simple/yt-dlp/"),
}

print("ENV_PROBE_JSON_START")
print(json.dumps(out, indent=2))
print("ENV_PROBE_JSON_END")
