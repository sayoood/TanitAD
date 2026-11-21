import json, gzip, collections, sys
ctl = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
want = {(r["log_name"], r["token"]) for r in ctl}
bank = {}
for l in open(sys.argv[2]):
    b = json.loads(l)
    k = (b["log_name"], b["token"])
    if k in want and int(b.get("rank", 0)) == 0:
        bank[k] = {x: b.get(x) for x in ("goal", "origin_world", "ego", "command", "dt_ms", "source", "step")}
md = lambda a, b: max(abs(x - y) for x, y in zip(a, b))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in ctl:
    b = bank[(r["log_name"], r["token"])]
    d = by[(r["ch"], r["log_name"])]
    d["dm"].append(r["bank_dmax"] or 0.0); d["g"].append(md(r["goal"], b["goal"])); d["o"].append(md(r["origin_world"], b["origin_world"]))
    d["cmd"].append(b["command"]); d["src"].append(b["source"])
tot = collections.Counter()
for (ch, lg), d in sorted(by.items()):
    ex = all(x <= 1e-6 for x in d["dm"])
    gx = max(d["g"]); ox = max(d["o"])
    tot[(ch[-6:], "exact" if ex else "diverged", "goal_eq" if gx < 1e-4 else "goal_DIFF")] += 1
    print(ch[-6:], lg, "n", len(d["dm"]), "EXACT" if ex else "div  ", "dmax %.4f goal_maxdiff %.4g orig_maxdiff %.3g" % (max(d["dm"]), gx, ox), sorted(set(d["src"])), sorted(set(d["cmd"])))
for k, v in sorted(tot.items()): print("TOT", k, v)
