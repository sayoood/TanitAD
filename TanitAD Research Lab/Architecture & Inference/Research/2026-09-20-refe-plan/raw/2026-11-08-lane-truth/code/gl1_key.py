import json, lzma, collections
import numpy as np
BP = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
LEAF = "center_line.CenterLine.info"
def dvec(d):
    return np.array([np.nan if p.get(LEAF) is None else float(p[LEAF]) for p in d["props"]])
rows = [json.loads(l) for l in lzma.open(BP, "rt")]
print("banked rows", len(rows), "sample key", rows[0]["key"], "ckpt", rows[0]["ckpt_step"])
full = collections.Counter((tuple(r["key"]), r["ckpt_step"]) for r in rows)
short = collections.Counter((r["key"][0], r["key"][1]) for r in rows)
print("unique full keys", len(full), "max rows per full", max(full.values()))
print("unique (log,token)", len(short), "max rows per (log,token)", max(short.values()), "tokens with >1 row", sum(1 for v in short.values() if v > 1))
print("distinct steps", sorted(collections.Counter(r["key"][2] for r in rows).items())[:8], "distinct ranks", sorted(set(r["key"][3] for r in rows)), "ckpts", sorted(set(r["ckpt_step"] for r in rows)))
last_short = {(r["key"][0], r["key"][1]): dvec(r) for r in rows}
by_full = {(tuple(r["key"]), r["ckpt_step"]): dvec(r) for r in rows}
mine = [json.loads(l) for l in open("F:/scratch/lane_truth_work/run/out_ctrl_ph/teacher_lane_r0.jsonl", encoding="utf-8")]
print("mine", len(mine), "sample", mine[0]["key"], mine[0].get("ckpt_step"))
def rep(name, dd):
    dd = np.concatenate(dd)
    print(f"{name}: n {dd.size} frac<=1mm {(dd <= 1e-3).mean():.4f} frac<=1e-6 {(dd <= 1e-6).mean():.4f} max {np.nanmax(dd):.4f}")
s, f, miss = [], [], 0
for m in mine:
    v = dvec(m)
    s.append(np.abs(v - last_short[(m["key"][0], m["key"][1])]))
    k = (tuple(m["key"]), m["ckpt_step"])
    if k in by_full:
        f.append(np.abs(v - by_full[k]))
    else:
        miss += 1
rep("(log,token) join, today's", s)
print("full-key misses", miss)
if f:
    rep("full key + ckpt join", f)
