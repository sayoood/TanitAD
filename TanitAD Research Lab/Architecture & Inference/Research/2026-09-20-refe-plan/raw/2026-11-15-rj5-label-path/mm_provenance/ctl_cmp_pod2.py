import json, gzip, collections, sys
ctl = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
want = {(r["log_name"], r["token"]) for r in ctl}
bank = {}
for l in open(sys.argv[2]):
    b = json.loads(l)
    k = (b["log_name"], b["token"])
    if k in want and int(b.get("rank", 0)) == 0:
        bank[k] = b["origin_world"]
by = collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # n, n_orig_diff, n_traj_diff, n_traj_diff_given_orig_eq, which component differs
comp = collections.Counter()
for r in ctl:
    o = bank[(r["log_name"], r["token"])]
    d = [abs(a - b) for a, b in zip(r["origin_world"], o)]
    od = max(d) > 0
    for i, x in enumerate(d):
        if x > 0: comp[i] += 1
    td = (r["bank_dmax"] or 0) > 1e-6
    v = by[(r["ch"], r["log_name"])]
    v[0] += 1; v[1] += od; v[2] += td; v[3] += (td and not od)
print("components differing (0=x,1=y,2=yaw):", dict(comp))
agg = collections.Counter()
for (ch, lg), v in sorted(by.items()):
    agg["rows"] += v[0]; agg["orig_diff"] += v[1]; agg["traj_diff"] += v[2]; agg["traj_diff_orig_eq"] += v[3]
    if ch != "run_local" or v[2]:
        print(ch[-6:], lg[-28:], "n %4d  orig_diff %4d  traj_diff %4d  traj_diff_with_orig_equal %4d" % tuple(v[:4]))
print(dict(agg))
