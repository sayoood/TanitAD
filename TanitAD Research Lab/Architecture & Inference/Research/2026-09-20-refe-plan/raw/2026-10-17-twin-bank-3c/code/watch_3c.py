#!/usr/bin/env python3
"""Progress + yield-guard early warning for the 3-C full run (a script, not a model turn).   python3 watch_3c.py --out <run dir> --total 44409 [--started-epoch E]
Prints one JSON line: audited / total, per-K audited & kept, outcome histogram, rate, ETA, and `yield_warn` when the K=1 keep share so far (over twins whose
f(10 s) was measured, the PREREG guard's basis) is < 0.20 with >= 500 K=1 twins evaluated.  A warning is NOT the guard (that is evaluated by `merge` on the
complete audit) -- it only tells the Master Mind early that the build will stop."""
import argparse, glob, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reroll_bank_3c as B

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--total", type=int, default=44409); ap.add_argument("--started-epoch", type=float, default=None)
a = ap.parse_args()
n = shards = done = 0
reasons, per_k = {}, {}
for sd in sorted(glob.glob(str(Path(a.out) / "shards" / "s*"))):
    shards += 1; done += int((Path(sd) / "DONE").exists())
    for rec in B._read_jsonl_tolerant(Path(sd) / "audit.jsonl"):
        n += 1
        tag = "kept" if rec["drop_reason"] is None else rec["drop_reason"].split(":")[0]
        reasons[tag] = reasons.get(tag, 0) + 1
        e = per_k.setdefault(str(rec["k"]), {"audited": 0, "evaluated": 0, "kept": 0})
        e["audited"] += 1; e["evaluated"] += int(rec["f_at_s"] is not None); e["kept"] += int(tag == "kept")
out = {"now_utc": time.strftime("%FT%TZ", time.gmtime()), "audited": n, "total": a.total, "shards_seen": shards, "shards_done": done, "by_outcome": reasons, "per_k": per_k}
k1 = per_k.get("1")
if k1 and k1["evaluated"] >= 500:
    out["k1_keep_share_evaluated_so_far"] = round(k1["kept"] / k1["evaluated"], 4)
    if k1["kept"] / k1["evaluated"] < B.YIELD_GUARD_K1:
        out["yield_warn"] = "ZZ3C_YIELD_WARN K=1 keep share so far < 0.20: the PREREG guard will stop the build at merge; tell the Master Mind now"
if a.started_epoch and n:
    el = time.time() - a.started_epoch
    out["rate_per_s"] = round(n / el, 3); out["eta_h"] = round((a.total - n) / (n / el) / 3600, 2)
print(json.dumps(out))
