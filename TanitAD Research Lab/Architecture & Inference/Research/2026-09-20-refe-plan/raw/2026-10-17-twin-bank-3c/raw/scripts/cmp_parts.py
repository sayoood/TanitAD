import json, sys, hashlib, numpy as np
sys.path[:0] = ["/workspace/ops/d7fix/pkg/refe", "/workspace/ops/d7fix/pkg/code"]
import online_labeller as OL
from pathlib import Path
A = "/workspace/ops/d7fix/mc_final"
B = "/workspace/data/refe_navtrain/metric_cache_twins_rejected"
C = "/workspace/data/refe_navtrain/metric_cache"
rows = [json.loads(l) for l in open("/workspace/ops/reroll_probe/bank_in/targets_rank1.jsonl")]
prod = {}
for l in open("/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl"):
    if '"lane_rank"' in l:
        r = json.loads(l)
        a = r.get("aug")
        if isinstance(a, dict) and a.get("kind") == "lane_rank":
            prod[(r["log_name"], r["token"], int(a["value"]))] = r
print("prod lane twins", len(prod))
res = {"n": 0, "in_prod": 0, "twin_exists_B": 0, "cl_equal": 0, "lane_ids_equal": 0, "log_exists_C": 0, "logcl_equal": 0, "twin_eq_logged_B": 0}
bad = []
for r in rows:
    k = (r["log_name"], r["token"], int(r["aug"]["value"]))
    res["n"] += 1
    res["in_prod"] += k in prod
    ta = OL.parts_path(A, k[0], k[1], {"kind": "lane_rank", "value": k[2]})
    tb = OL.parts_path(B, k[0], k[1], {"kind": "lane_rank", "value": k[2]})
    la = OL.parts_path(A, k[0], k[1], None)
    lc = OL.parts_path(C, k[0], k[1], None)
    if tb.is_file():
        res["twin_exists_B"] += 1
        pa, pb = OL.load_parts(ta), OL.load_parts(tb)
        ca = np.asarray(pa[3].linestring.coords); cb = np.asarray(pb[3].linestring.coords)
        eq = ca.shape == cb.shape and np.allclose(ca, cb, atol=1e-6)
        res["cl_equal"] += bool(eq)
        res["lane_ids_equal"] += list(pa[4]) == list(pb[4])
        if not eq: bad.append((k, ca.shape, cb.shape))
        pl = OL.load_parts(la)
        cl = np.asarray(pl[3].linestring.coords)
        res["twin_eq_logged_B"] += cl.shape == cb.shape and bool(np.allclose(cl, cb, atol=1e-6))
    if lc.is_file():
        res["log_exists_C"] += 1
        pl, pc = OL.load_parts(la), OL.load_parts(lc)
        cl = np.asarray(pl[3].linestring.coords); cc = np.asarray(pc[3].linestring.coords)
        res["logcl_equal"] += cl.shape == cc.shape and bool(np.allclose(cl, cc, atol=1e-6))
print(json.dumps(res)); print(bad[:5])
