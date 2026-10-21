"""READ-ONLY. Per OpenScene trainval metadata pkl: frames, navtrain tokens (by the v1.1 filter), and whether each navtrain
token's t0 CAM_F0/L0/R0 jpg exists under a sensors root. Prints JSON."""
import json, os, pickle, sys
nav = json.load(open(sys.argv[1])); navtok = set(nav["tokens_sorted"]); rank = {l: i for i, l in enumerate(nav["log_names_sorted"])}
sensors = sys.argv[3]; out = {}
for d in sys.argv[2].split(","):
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pkl"):
            continue
        frames = pickle.load(open(os.path.join(d, f), "rb"))
        log = f[:-4]
        nt = [fr for fr in frames if fr["token"] in navtok]
        cams = {}
        for c in ("CAM_F0", "CAM_L0", "CAM_R0"):
            cams[c] = sum(1 for fr in nt if os.path.isfile(os.path.join(sensors, fr["cams"][c]["data_path"])))
        k0 = frames[0]
        out[log] = {"rank": rank.get(log), "n_frames": len(frames), "n_navtrain_tokens": len(nt),
                    "t0_present": cams, "log_name_field": k0.get("log_name"),
                    "frame_keys": sorted(k0.keys())[:40], "cam_keys": sorted(k0["cams"]["CAM_F0"].keys()),
                    "data_path_example": k0["cams"]["CAM_F0"]["data_path"]}
print(json.dumps(out, indent=0, default=str))
