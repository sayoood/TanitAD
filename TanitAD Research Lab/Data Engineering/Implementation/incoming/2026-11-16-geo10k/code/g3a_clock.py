"""GEO10K step 3a: clip clock sidecar for the new clips (build_clip_clock_sidecar.fit_clock, unedited) + control vs the
existing sidecar (refcv8_clip_clock_sidecar_ts33.jsonl, built from the same inversion on the existing manifest)."""
import json, sys
import numpy as np
import torch
from multiprocessing import Pool
sys.path.insert(0, "/root/geo10k/code")
import geo_common as G
G.point_ego_lookup()
G.ES._index()
import build_clip_clock_sidecar as CC
from build_v9_labels import stable_episode_id


def one(a):
    cid, poses, ns = a
    log = CC._log(cid)
    if log is None:
        return cid, {"refused": "no log"}
    return cid, CC.fit_clock(poses, ns, log)


def run(manpath, outpath):
    m = torch.load(manpath, weights_only=False)
    jobs = [(c, p.double().numpy(), int(n)) for c, p, n in zip(m["clip_id"], m["poses"], m["n_stack"])]
    with Pool(24) as pool:
        res = pool.map(one, jobs, chunksize=20)
    rows, ref = [], {}
    for cid, f in res:
        if "refused" in f:
            ref[cid] = f["refused"]
            continue
        rows.append({"sid": stable_episode_id(cid), **f})
    with open(outpath, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows, ref


k2 = CC.self_test()
assert k2["ok"], k2
rows, ref = run(f"{G.OUT}/new10k_v2manifest.pt", f"{G.OUT}/clock_new10k.jsonl")
dts = np.array([r["dt_s"] for r in rows]); g0 = np.array([r["grid_start_s"] for r in rows])
print(f"[g3a] new10k clocked {len(rows)} refused {len(ref)}; dt median {np.median(dts):.7f} g0 median {np.median(g0):+.4f}", flush=True)
json.dump({"n_clocked": len(rows), "n_refused": len(ref), "refused_reasons": sorted(set(ref.values()))[:10], "refused_clips": sorted(ref)[:50]},
          open(f"{G.OUT}/clock_new10k_refused.json", "w"), indent=1)
crow, cref = run(f"{G.OUT}/ctl20_v2manifest.pt", f"{G.OUT}/clock_ctl20.jsonl")
ex = {}
for l in open(f"{G.IN}/refcv8_clip_clock_sidecar_ts33.jsonl"):
    r = json.loads(l)
    ex[int(r["sid"])] = r
diffs = []
for r in crow:
    e = ex.get(r["sid"])
    if e is None:
        diffs.append({"sid": r["sid"], "missing_in_existing": True})
        continue
    diffs.append({"sid": r["sid"], "d_g0": abs(r["grid_start_s"] - e["grid_start_s"]), "d_dt": abs(r["dt_s"] - e["dt_s"]),
                  "bitwise": bool(r["grid_start_s"] == e["grid_start_s"] and r["dt_s"] == e["dt_s"])})
rep = {"n": len(crow), "n_refused": len(cref), "rows": diffs, "n_bitwise": sum(d.get("bitwise", False) for d in diffs),
       "max_d_g0": max([d.get("d_g0", 0) for d in diffs] or [0]), "max_d_dt": max([d.get("d_dt", 0) for d in diffs] or [0])}
json.dump(rep, open(f"{G.OUT}/control_clock.json", "w"), indent=1)
print("[g3a] control clock: bitwise", rep["n_bitwise"], "/", rep["n"], "max d_g0", rep["max_d_g0"], "max d_dt", rep["max_d_dt"])
print("ZZ GEO10K-G3A DONE ZZ")
