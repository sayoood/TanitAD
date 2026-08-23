#!/usr/bin/env python3
"""ADVERSARIAL: is 'mp4 605 vs rig 599, delta 6' real, generic, and NOT an OpenCV artefact?

Three INDEPENDENT sources per scene, so no two probes are the same location:
  A. data_info.json      -> dataset manifest num-frames for camera_front_wide_120fov
  B. rig_trajectories.json -> what our loader reads (n_frames)
  C. the mp4 itself      -> ffprobe packet count (libav) AND OpenCV metadata
"""
import json, subprocess, sys, os, glob

ROOT = "/home/nvidia/nurec_scenes/sample_set/26.04_release"
CAM = "camera_front_wide_120fov"
scenes = sorted(os.path.basename(p) for p in glob.glob(ROOT + "/*") if os.path.isdir(p))
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def ffprobe_packets(mp4):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets,nb_frames,r_frame_rate,duration,start_time",
             "-of", "json", mp4], capture_output=True, text=True, timeout=180)
        if out.returncode != 0:
            return {"err": out.stderr.strip()[:120]}
        s = json.loads(out.stdout)["streams"][0]
        return s
    except FileNotFoundError:
        return {"err": "no ffprobe"}
    except Exception as e:
        return {"err": repr(e)[:120]}

rows = []
for sc in scenes:
    d = os.path.join(ROOT, sc)
    mp4 = os.path.join(d, CAM + ".mp4")
    di = os.path.join(d, "data_info.json")
    rt = os.path.join(d, "rig_trajectories.json")
    if not (os.path.exists(mp4) and os.path.exists(di)):
        continue
    manifest = None
    try:
        j = json.load(open(di))
        manifest = j["shards"][0]["sensors"][CAM]["frame-range"]["num-frames"]
        toff = j["shards"][0]["sensors"][CAM]["frame-range"].get("sequence-frame-offset")
    except Exception as e:
        manifest, toff = f"ERR {e}", None
    rig_n = None
    if os.path.exists(rt):
        try:
            r = json.load(open(rt))
            # find the camera block, count timestamps
            def find(o, key):
                if isinstance(o, dict):
                    if key in o: return o[key]
                    for v in o.values():
                        f = find(v, key)
                        if f is not None: return f
                elif isinstance(o, list):
                    for v in o:
                        f = find(v, key)
                        if f is not None: return f
                return None
            cams = r.get("cameras") or r.get("camera") or r
            blk = None
            if isinstance(cams, dict) and CAM in cams:
                blk = cams[CAM]
            elif isinstance(cams, list):
                for c in cams:
                    if isinstance(c, dict) and c.get("name") == CAM:
                        blk = c
            if blk is not None:
                for k in ("frame_timestamps_us", "timestamps_us", "poses", "frames", "frame_timestamps"):
                    if k in blk:
                        rig_n = len(blk[k]); break
        except Exception as e:
            rig_n = f"ERR {type(e).__name__}"
    pk = ffprobe_packets(mp4)
    rows.append((sc[:8], manifest, toff, rig_n, pk.get("nb_read_packets"), pk.get("nb_frames"),
                 pk.get("r_frame_rate"), pk.get("start_time"), pk.get("duration"), pk.get("err", "")))
    if len(rows) >= limit:
        break

print("%-9s %8s %5s %7s %9s %8s %10s %8s %9s %s" % (
    "scene", "manifest", "foff", "rig_n", "ffpackets", "nb_frm", "fps", "start_t", "dur", "err"))
for r in rows:
    print("%-9s %8s %5s %7s %9s %8s %10s %8s %9s %s" % r)
delta = [(r[0], (int(r[4]) - int(r[1])) if (r[4] and isinstance(r[1], int)) else None) for r in rows]
print("\nffprobe_packets - manifest_num_frames:", delta)
