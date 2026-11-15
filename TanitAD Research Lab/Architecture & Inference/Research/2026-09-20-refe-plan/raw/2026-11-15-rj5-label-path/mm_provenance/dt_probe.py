import json, gzip, sys, statistics, collections
ctl = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
st = {}
for r in ctl:
    st.setdefault(r["log_name"], []).append((r["bank_dmax"] or 0) > 1e-6)
cls = {lg: ("div" if any(v) else "exa") for lg, v in st.items()}
dt = collections.defaultdict(list); keys = collections.defaultdict(set); dev = collections.defaultdict(collections.Counter)
for l in open(sys.argv[2]):
    b = json.loads(l)
    lg = b["log_name"]
    if lg in cls:
        dt[lg].append(b.get("dt_ms") or 0); keys[lg] |= set(b.keys()); dev[lg][str(b.get("teacher_device"))] += 1
for c in ("exa", "div"):
    for lg in sorted(l for l in cls if cls[l] == c):
        v = dt[lg]
        print(c, lg[-30:], "n", len(v), "dt_ms median %.1f p10 %.1f p90 %.1f" % (statistics.median(v), sorted(v)[len(v) // 10], sorted(v)[9 * len(v) // 10]), dict(dev[lg]), len(keys[lg]))
