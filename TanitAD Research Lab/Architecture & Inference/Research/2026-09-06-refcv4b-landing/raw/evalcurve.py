import json, sys, statistics as st

rows = []
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "eval_loss" in r:
            rows.append(r)

rows.sort(key=lambda r: r.get("step", 0))
cols = ["eval_loss", "eval_traj", "eval_lat", "eval_lon", "eval_lat_tac", "eval_lon_tac",
        "eval_cls", "eval_route", "eval_anchor_acc", "eval_goal_tac", "eval_goal2s_err_m",
        "eval_goal_str", "eval_sel_v3", "eval_law", "eval_goal_gate", "eval_goal_score_absmean"]
short = {c: c.replace("eval_", "")[:9] for c in cols}
print("n_eval_rows=%d  windows/eval=%s" % (len(rows), rows[-1].get("eval_windows")))
hdr = "%7s " % "step" + " ".join("%9s" % short[c] for c in cols)
print(hdr)
print("-" * len(hdr))
for r in rows:
    print("%7d " % r["step"] + " ".join(
        ("%9.4f" % r[c]) if isinstance(r.get(c), (int, float)) else "%9s" % "-" for c in cols))

print("\n=== BEST-SO-FAR (min) per column, and the step it happened ===")
for c in cols:
    vals = [(r[c], r["step"]) for r in rows if isinstance(r.get(c), (int, float))]
    if not vals:
        continue
    lo = min(vals); hi = max(vals)
    last = vals[-1]
    print("  %-22s min %9.4f @%6d | max %9.4f @%6d | last %9.4f @%6d | last-vs-min %+8.4f"
          % (c, lo[0], lo[1], hi[0], hi[1], last[0], last[1], last[0] - lo[0]))

print("\n=== LAST 5 vs PRIOR 5 (is it still improving?) ===")
for c in cols:
    v = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
    if len(v) < 10:
        continue
    a = st.median(v[-10:-5]); b = st.median(v[-5:])
    print("  %-22s prior5 %9.4f -> last5 %9.4f   delta %+8.4f  %s"
          % (c, a, b, b - a, "IMPROVING" if b < a else "WORSE/FLAT"))
