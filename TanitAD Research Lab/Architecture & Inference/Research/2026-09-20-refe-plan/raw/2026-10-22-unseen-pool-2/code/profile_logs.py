#!/usr/bin/env python3
"""Profile selected OpenScene trainval metadata logs from the LOCAL tgz (no network). Streams the tgz once; for each wanted member
runs a RESTRICTED unpickler (numpy array globals only, anything else raises: the pkl is third-party data and is never executed) and records
frame counts, timestamp regularity, route availability, anchor counts under NAVSIM's navtrain filter geometry (4 history frames incl. current,
10 future frames, 2 Hz), and which ground-truth fields exist.
Usage: profile_logs.py <tgz> <wanted.json> <out.json>     wanted.json = {"logs": [...]}"""
import sys, json, io, pickle, tarfile, time, hashlib
import numpy as np

class ArrayOnlyUnpickler(pickle.Unpickler):          # same whitelist as TanitAstra's tanit_astra.data.navsim_adapter.ArrayOnlyUnpickler
    def find_class(self, module, name):
        core = getattr(np, "_core", None) or np.core          # numpy >= 2 has np._core, numpy 1.x has np.core
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = core.multiarray._reconstruct
            allowed[prefix, "scalar"] = core.multiarray.scalar
        for prefix in ("numpy.core.numeric", "numpy._core.numeric"):
            allowed[prefix, "_frombuffer"] = core.numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported class: %s.%s" % (module, name))
        return allowed[module, name]

def profile(log):
    n = len(log)
    ts = [int(f.get("timestamp", -1)) for f in log]
    dts = np.diff(np.asarray(ts, dtype=np.int64)) / 1e6 if n > 1 else np.array([])
    regular = float(np.mean(np.abs(dts - 0.5) <= 0.075)) if len(dts) else 0.0
    route = [bool(f.get("roadblock_ids")) for f in log]
    anchors_all = list(range(3, n - 10)) if n >= 14 else []
    anchors_route = [j for j in anchors_all if route[j]]
    # strict: 14-frame window regular at 0.5 s
    anchors_reg = []
    for j in anchors_route:
        w = np.asarray(ts[j - 3:j + 11], dtype=np.float64)
        if np.allclose((w - w[3]) / 1e6, np.arange(-3, 11) * 0.5, atol=0.075):
            anchors_reg.append(j)
    f0 = log[0] if n else {}
    anns = f0.get("anns")
    return {
        "n_frames": n, "t0_us": ts[0] if n else None, "t1_us": ts[-1] if n else None,
        "duration_s": (ts[-1] - ts[0]) / 1e6 if n > 1 else 0.0, "frac_dt_0p5": round(regular, 4),
        "n_route_frames": int(sum(route)), "anchors_geom": len(anchors_all), "anchors_route": len(anchors_route), "anchors_regular": len(anchors_reg),
        "maps": sorted({str(f.get("map_location")) for f in log}),
        "frame_keys": sorted(f0.keys()) if isinstance(f0, dict) else None,
        "ann_keys": sorted(anns.keys()) if isinstance(anns, dict) else None,
        "n_gt_boxes_frame0": int(len(anns["gt_boxes"])) if isinstance(anns, dict) and "gt_boxes" in anns else None,
        "has_traffic_lights": "traffic_lights" in f0, "cam_keys": sorted(f0.get("cams", {}).keys()) if isinstance(f0.get("cams"), dict) else None,
        "has_ego_dynamic_state": "ego_dynamic_state" in f0, "has_driving_command": "driving_command" in f0,
        "first_cam_path": (f0.get("cams", {}).get("CAM_F0", {}) or {}).get("data_path") if isinstance(f0.get("cams"), dict) else None,
    }

if __name__ == "__main__":
    src, wanted_path, out = sys.argv[1:4]
    wanted = set(json.load(open(wanted_path))["logs"]); res = {}; t0 = time.time(); seen = 0
    with tarfile.open(src, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".pkl"):
                continue
            seg = m.name.rsplit("/", 1)[-1][:-4]
            if seg not in wanted:
                continue
            data = tf.extractfile(m).read()
            assert len(data) == m.size
            try:
                log = ArrayOnlyUnpickler(io.BytesIO(data)).load()
                p = profile(log); p["pkl_bytes"] = m.size; p["pkl_sha256"] = hashlib.sha256(data).hexdigest()
            except Exception as e:                      # fail loud: record the refusal, never guess
                p = {"error": repr(e)[:300], "pkl_bytes": m.size}
            res[seg] = p; seen += 1
            if seen % 20 == 0:
                print(seen, round(time.time() - t0), "s", flush=True)
    json.dump({"src": src, "wanted": len(wanted), "profiled": len(res), "seconds": round(time.time() - t0, 1), "logs": res}, open(out, "w"))
    print("done", len(res), "of", len(wanted), round(time.time() - t0, 1), "s")
