#!/usr/bin/env python3
"""E1 step 1: locate banked 20 Hz teacher rows and compute the sample-id overlap with the 1,018 (log, token) of d1_navtrain.json.
Scans every targets_rank0*.jsonl under the given roots (two places: the dev-box data dir and the F: archive), keyed by
(log_name, token) with sim_hz/step/rank/len(traj); the 'sample id' is (log, token), rank 0, step 0."""
import glob, json, os, sys, collections
d1 = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]
want = {(r["log"], r["token"]) for r in d1}
roots = sys.argv[3:]
files = []
for r in roots:
    files += glob.glob(os.path.join(r, "**", "targets_rank0*.jsonl"), recursive=True)
files = sorted(set(files))
summary = []
hz_keys = collections.defaultdict(set)         # sim_hz -> keys in want (rank 0, step 0)
for fp in files:
    c = collections.Counter(); ov = 0; n = 0
    for ln in open(fp, encoding="utf-8"):
        try: r = json.loads(ln)
        except json.JSONDecodeError: c["bad"] += 1; continue
        n += 1
        hz = int(r.get("sim_hz", 0)); c[f"hz{hz}"] += 1
        k = (r.get("log_name"), r.get("token"))
        if k in want and int(r.get("step", 0)) == 0 and int(r.get("rank", 0)) == 0:
            hz_keys[(hz, len(r.get("traj", [])))].add(k); ov += 1
    summary.append({"file": fp.replace("\\", "/"), "rows": n, "by_hz": dict(c), "overlap_rows_with_1018": ov})
res = {"n_want": len(want), "files": summary,
       "overlap_unique_by_hz_and_traj_len": {f"hz{h}_len{l}": len(v) for (h, l), v in sorted(hz_keys.items())}}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"n_want": res["n_want"], "overlap": res["overlap_unique_by_hz_and_traj_len"]}))
for s in summary: print(s["file"], s["rows"], s["by_hz"], s["overlap_rows_with_1018"])
