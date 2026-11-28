import json, lzma, glob, os, sqlite3, collections
import numpy as np
P = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/lane_truth/"
g = np.load(P + "per_token_gap.npy"); print("gap shape", g.shape, "finite per col", (np.isfinite(g).sum(0) if g.ndim == 2 else np.isfinite(g).sum()))
if g.ndim == 2:
    for c in range(g.shape[1]):
        print(" col", c, "mean", round(float(np.nanmean(g[:, c])), 4), "n", int(np.isfinite(g[:, c]).sum()))
BP = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
LEAF = "center_line.CenterLine.info"
def dvec(d):
    return np.array([np.nan if p.get(LEAF) is None else float(p[LEAF]) for p in d["props"]])
bank = {}
for l in lzma.open(BP, "rt"):
    d = json.loads(l); bank[(tuple(d["key"]), d["ckpt_step"])] = d
rows = [json.loads(l) for l in open(P + "lane_labels/out_ctrl2_ph.jsonl", encoding="utf-8")]
ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
idx = {}
for f in glob.glob(ROOT + "/**/*.db", recursive=True):
    idx[os.path.basename(f)[:-3]] = f
print("dbs", len(idx))
def meta(log):
    f = idx.get(log)
    if not f:
        return ("NO_DB", "?")
    c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
    r = c.execute("select location, map_version, vehicle_name from log limit 1").fetchone(); c.close()
    return (r[0], r[1])
tab = collections.defaultdict(lambda: [0, 0, []])
perlog = collections.defaultdict(lambda: [0, 0])
for m in rows:
    k = (tuple(m["key"]), m["ckpt_step"]); v, b = dvec(m), dvec(bank[k])
    e = np.abs(v - b)
    exact = bool((e <= 1e-3).all())
    mt = meta(m["key"][0])
    t = tab[mt]; t[0 if exact else 1] += 1; t[2].append(float((e <= 1e-3).mean()))
    perlog[m["key"][0]][0 if exact else 1] += 1
    bm = bank[k]
print("banked row fields", [x for x in bank[k] if x != "props"], {x: bank[k][x] for x in bank[k] if x not in ("props", "key")})
for t, v in sorted(tab.items()):
    print(t, "exact sets", v[0], "differing sets", v[1], "mean frac<=1mm", round(float(np.mean(v[2])), 3))
print("logs mixed (both exact and differing)", sum(1 for a, b in perlog.values() if a and b), "of", len(perlog))
