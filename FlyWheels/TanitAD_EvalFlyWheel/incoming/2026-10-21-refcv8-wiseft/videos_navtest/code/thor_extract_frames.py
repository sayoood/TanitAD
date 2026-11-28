"""thor_extract_frames.py -- RUNS ON THOR (python3 + numpy, CPU, read-only on the bank): copy the t0 frame (last of the 4 banked history rows) of
each wanted token out of the navtest 416x1024 frame bank into one npy per log under /home/nvidia/r8ns/navsim_video_tmp/.  Verifies every
token's 4-row block against the index's sha256[:16] (so a wrong row mapping or a torn shard fails loud).
usage: python3 thor_extract_frames.py wanted.json   (wanted.json = {"<log>": ["tok", ...], ...})"""
import hashlib, json, sys, os
import numpy as np

BANK = "/home/nvidia/r8ns/navtest/frame_bank"
OUT = "/home/nvidia/r8ns/navsim_video_tmp"
want = json.load(open(sys.argv[1]))
idx = json.load(open(f"{BANK}/index.json"))
assert (idx["frame"]["h"], idx["frame"]["w"], idx["frame"]["projection"]) == (416, 1024, "cylindrical"), idx["frame"]
assert abs(idx["frame"]["f_ref"] - 488.92398517830253) < 1e-9
os.makedirs(OUT, exist_ok=True)
mm_cache = {}
report = {}
for li, (log, toks) in enumerate(want.items()):
    arr = np.zeros((len(toks), 416, 1024, 3), np.uint8)
    meta = []
    n_ok = 0
    for i, t in enumerate(toks):
        r = idx["tokens"][t]
        assert r["log_name"] == log, (t, r["log_name"], log)
        f = r["file"]
        if f not in mm_cache:
            mm_cache[f] = np.load(f"{BANK}/{f}", mmap_mode="r")
        blk = np.ascontiguousarray(mm_cache[f][r["rows"]])
        assert blk.shape == (4, 416, 1024, 3) and blk.dtype == np.uint8, blk.shape
        sha = hashlib.sha256(blk.tobytes()).hexdigest()[:16]
        ok = (sha == r["sha256"])
        n_ok += int(ok)
        arr[i] = blk[-1]
        meta.append({"token": t, "file": f, "rows": r["rows"], "rig_key": r["rig_key"], "observed_frac": r["observed_frac"],
                     "sha16_block": sha, "index_sha16": r["sha256"], "sha_match": bool(ok), "t0_mean_px": float(blk[-1].mean())})
    np.save(f"{OUT}/frames_{li}.npy", arr)
    json.dump({"log": log, "tokens": meta}, open(f"{OUT}/frames_{li}.meta.json", "w"))
    report[log] = {"n": len(toks), "sha_match": n_ok, "mean_px": float(arr.mean())}
    print(li, log, report[log], flush=True)
json.dump(report, open(f"{OUT}/extract_report.json", "w"), indent=1)
