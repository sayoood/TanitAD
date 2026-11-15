"""Control C6 WINDOWS-EQUAL: an arm's 994 SCORE windows (eid, now, logged A, v0, donor) against R35's phase-1 windows of the same chassis: must be exactly equal. Beside: max |z| difference (the arm's
own adapter output; the adapter is frozen at M2b so 0.0 is expected, reported, not a bar). usage: c6_windows.py <arm_windows_dir> <chassis 0|1> <out.json>"""
import json, sys, os
import numpy as np
d, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
a = np.load(f"{d}/windows.npz", allow_pickle=False)
b = np.load(f"{os.environ['R35REF']}/s{k}/windows.npz", allow_pickle=False)
res = {"chassis": k, "n_arm": int(len(a["eid"])), "n_r35": int(len(b["eid"]))}
for key in ("eid", "now", "A", "v0", "donor"):
    res[f"equal_{key}"] = bool(a[key].shape == b[key].shape and (np.array_equal(a[key], b[key]) if key != "v0" else np.array_equal(a[key], b[key], equal_nan=True)))
res["z_max_abs_diff"] = float(np.abs(a["z"].astype(np.float64) - b["z"].astype(np.float64)).max()) if a["z"].shape == b["z"].shape else None
res["n_episodes"] = int(len(set(a["eid"].tolist())))
res["C6_PASS"] = bool(all(res[f"equal_{x}"] for x in ("eid", "now", "A", "v0")) and res["n_arm"] == 994 and res["n_episodes"] == 71)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if res["C6_PASS"] else 3)
