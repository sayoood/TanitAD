#!/usr/bin/env python3
"""For a few LOCAL test-split logs: does a route-less log (empty roadblock_ids) still carry a driving_command / traffic lights / GT agents? Restricted unpickler, read-only."""
import sys, os, io, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from profile_logs import ArrayOnlyUnpickler
D = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
sel = sys.argv[1:]
out = {}
for l in sel:
    log = ArrayOnlyUnpickler(io.BytesIO(open(os.path.join(D, l + ".pkl"), "rb").read())).load()
    cmds = collections.Counter(tuple(int(x) for x in np.asarray(f["driving_command"]).ravel()) for f in log)
    rb = collections.Counter(len(f["roadblock_ids"]) if f["roadblock_ids"] is not None else -1 for f in log)
    tl = sum(1 for f in log if f.get("traffic_lights"))
    nb = [len(f["anns"]["gt_boxes"]) for f in log]
    out[l] = {"n": len(log), "driving_command_values": {str(k): v for k, v in cmds.most_common(6)}, "roadblock_len_hist": dict(rb.most_common(4)), "frames_with_traffic_lights": tl,
              "gt_boxes_per_frame_mean": round(float(np.mean(nb)), 1), "map": log[0]["map_location"], "frame0_roadblock_ids": list(log[0]["roadblock_ids"] or [])[:3]}
print(json.dumps(out, indent=1))
json.dump(out, open("raw/route_probe_test.json", "w"), indent=1)
