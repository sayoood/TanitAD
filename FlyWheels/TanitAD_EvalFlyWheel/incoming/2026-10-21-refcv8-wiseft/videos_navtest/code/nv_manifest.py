"""nv_manifest.py -- JSON manifest of every rendered artifact: path, size, md5, duration, n_frames, tokens covered.  Run after the renders."""
import hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path("C:/Users/Admin/refcv8_pf/navsim_video")
HOLD = Path(__file__).resolve().parent.parent
ff = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True).stdout.split("\n")[0].strip()
FFP = ff.replace("ffmpeg.exe", "ffprobe.exe")
wanted = json.load(open(ROOT / "work/wanted.json"))
logs = list(wanted)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe(p):
    r = subprocess.run([FFP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt:format=duration", "-of", "json", str(p)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    return {"codec": s["codec_name"], "width": s["width"], "height": s["height"], "fps": s["r_frame_rate"], "pix_fmt": s["pix_fmt"], "n_frames": int(s["nb_frames"]),
            "duration_s": round(float(j["format"]["duration"]), 2)}


rows = []
for arm in ("LEGAL", "RT_alt"):
    d = ROOT / arm
    for p in sorted(d.glob("*.mp4")) + sorted((d / "cards").glob("*.mp4")):
        toks, kind = [], "card"
        if p.name.startswith("navtest_REEL"):
            kind = "combined reel (title + 6 clips + summary, chapter markers)"
            toks = [t for l in logs for t in wanted[l]]
        elif p.name.startswith("navtest_"):
            kind = "clip (3.5 s intro card + scenes)"
            for l in logs:
                if l in p.name:
                    toks = wanted[l]
        r = {"path": str(p).replace("\\", "/"), "kind": kind, "ema_arm": "R8_LEGAL" if arm == "LEGAL" else "R8_RT", "bytes": p.stat().st_size, "md5": md5(p), **probe(p),
             "n_scenes_covered": len(toks), "tokens": toks}
        rows.append(r)
        print(p.name, r["duration_s"], r["bytes"], flush=True)
for p in sorted((HOLD / "checks").glob("*.png")) + sorted((HOLD / "data").glob("*")) + sorted((HOLD / "code").glob("*.py")):
    rows.append({"path": str(p).replace("\\", "/"), "kind": "staged support file", "bytes": p.stat().st_size, "md5": md5(p)})
man = {"generated_local": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "MP4s live OUTSIDE the hold (large): C:/Users/Admin/refcv8_pf/navsim_video/ -- one copy, dev box only", "files": rows}
json.dump(man, open(HOLD / "MANIFEST_videos_navtest.json", "w", encoding="utf-8"), indent=1)
print("manifest rows", len(rows))
