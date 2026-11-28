import json, glob, sys, statistics as st
for src in sys.argv[1:]:
    secs = {"RJ5": [], "R0": []}
    for fp in sorted(glob.glob(src + "/rj5_*.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "sec" in r and r.get("arm") in secs:
                secs[r["arm"]].append(r["sec"])
    for arm, v in secs.items():
        if v:
            v.sort()
            print(src, arm, "n", len(v), "mean %.2f" % (sum(v)/len(v)), "p50 %.2f" % v[len(v)//2], "p90 %.2f" % v[int(len(v)*0.9)], "sum_h %.2f" % (sum(v)/3600))
