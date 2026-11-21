"""Dev-box (CPU) path-mapped variants of arm L's canonical argv. ONLY path-valued flags and --device are mapped; every lever token is the canonical one."""
import json, sys, hashlib
SP, which = sys.argv[1], sys.argv[2]
W = "C:/Users/Admin/ar3_work"
V9 = "C:/Users/Admin/box2_ship/v9ship"
canon = json.load(open(SP + "/raw/argv_L_s0_canonical.json"))
MAP = {"--cache-dir": W + "/dc", "--v9-train": V9 + "/v9_labels_train.npz", "--lead-train": V9 + "/lead_train4572.npz",
       "--speed-train": V9 + "/refcv6_speed_max_v8_train.jsonl", "--join-train": W + "/dc/join_train.npz",
       "--p10": W + "/dc/inputs/p10_prior.pt", "--tracks": W + "/dc/inputs/native_tracks.pt", "--device": "cpu",
       "--out": W + "/runs/L_" + which}
if which == "fullshape":      # the exact lever set; frames-root = the 600 production frame files the dev box holds
    MAP["--frames-root"] = "C:/Users/Admin/tanitad-data/refav1-train600-256"
elif which == "live4":        # the 4 declared clips (AR-2's shape: --max-clips 4), frames built by the production builder for exactly those
    MAP["--frames-root"] = "C:/Users/Admin/box2_ship/v8floraar2_frames_10250735/train"
else:
    raise SystemExit("which?")
out = list(canon)
for k, v in MAP.items():
    out[out.index(k) + 1] = v
if which == "live4":
    out += ["--max-clips", "4"]
out += ["--selfcheck-only"]
json.dump(out, open(f"{SP}/raw/argv_L_s0_dev_{which}.json", "w"), indent=1)
print(" ".join(out)); print(hashlib.sha256(" ".join(out).encode()).hexdigest()[:16])
