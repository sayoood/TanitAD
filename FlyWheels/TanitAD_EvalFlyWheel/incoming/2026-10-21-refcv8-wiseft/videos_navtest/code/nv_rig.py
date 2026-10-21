"""nv_rig.py -- per-token virtual-camera records (CAM_F0 mount in the NavSim lidar/ego frame + the stitch's virtual rotation M) for the selected
logs, computed with the refcv6/7/8 bridge's own `frames416.rig_record` from the OpenScene log pickles (read-only).  CPU only.
Asserts rig_key == the frame-bank index's rig_key for every token (positive pairing control)."""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True
os.environ["TANITAD_REPO"] = "C:/lgs/Lfb47a58376"
PKG = "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code"
sys.path.insert(0, PKG)
import boot8  # noqa: E402,F401  (puts the launch tree's R6/R7 code on sys.path)
import frames416 as F4  # noqa: E402
import numpy as np  # noqa: E402

LOGS_ROOT = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
W = "C:/Users/Admin/refcv8_pf/navsim_video/work"
sel = json.load(open(f"{W}/wanted.json"))
inp = json.load(__import__("gzip").open("C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/inputs/navtest_inputs.json.gz", "rt", encoding="utf-8"))["tokens"]
out = {}
for li, (log, toks) in enumerate(sel.items()):
    t0 = time.time()
    fr = F4.E2BF.load(os.path.join(LOGS_ROOT, f"{log}.pkl"))
    idx = {f["token"]: f for f in fr}
    meta = json.load(open(f"{W}/frames/frames_{li}.meta.json"))["tokens"]
    recs, nbad = {}, 0
    for m in meta:
        t = m["token"]
        cur = inp[t]["frame_tokens"][-1]
        rr = F4.rig_record(idx[cur]["cams"])
        if rr["rig_key"] != m["rig_key"]:
            nbad += 1
        recs[t] = {"rig_key": rr["rig_key"], "t_cam_lidar": rr["f0_sensor2lidar_translation"], "R_virtual_cam_to_lidar": rr["virtual_R_cam_to_lidar"]}
    out[log] = {"n": len(recs), "rig_key_mismatch_vs_bank_index": nbad, "records": recs}
    ts = np.array([r["t_cam_lidar"] for r in recs.values()]); 
    print(li, log, "n", len(recs), "rig_key mismatches", nbad, "t_cam mean", ts.mean(0).round(3), "t_cam max dev", np.abs(ts - ts.mean(0)).max().round(4), "%.1fs" % (time.time() - t0), flush=True)
    del fr, idx
json.dump(out, open(f"{W}/rigs.json", "w"))
