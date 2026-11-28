import json, sys, math, statistics as st

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

print("rows=%d" % len(rows))
if not rows:
    sys.exit(1)

keys = sorted({k for r in rows for k in r})
print("keys(%d): %s" % (len(keys), ", ".join(keys)))
print("step range: %s -> %s" % (rows[0].get("step"), rows[-1].get("step")))

# split into buckets of steps and report medians so noise does not decide
def bucket(rs, lo, hi):
    return [r for r in rs if lo <= r.get("step", -1) < hi]

edges = [0, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 32000]
track = ["loss", "traj", "cls", "law", "route", "lat", "lon", "lat_tac", "lon_tac",
         "anchor_acc", "goal_tac", "goal2s_err_m", "goal_str", "sel_v3",
         "goal_gate", "goal_score_absmean", "goal_gate_grad", "slot_valid_frac",
         "ego_keep_frac", "nav_injected", "ego_injected"]
track = [k for k in track if k in keys]

hdr = "%-9s %5s " % ("window", "n") + " ".join("%10s" % k[:10] for k in track)
print("\n" + hdr)
print("-" * len(hdr))
for i in range(len(edges) - 1):
    b = bucket(rows, edges[i], edges[i + 1])
    if not b:
        continue
    cells = []
    for k in track:
        vals = [r[k] for r in b if isinstance(r.get(k), (int, float))]
        cells.append("%10.4f" % st.median(vals) if vals else "%10s" % "-")
    print("%-9s %5d " % ("%d-%dk" % (edges[i] // 1000, edges[i + 1] // 1000), len(b)) + " ".join(cells))

# eval rows (if the trainer logs a separate eval line, it will have different keys)
evalish = [r for r in rows if any(k.startswith("eval") or k.startswith("val") for k in r)]
print("\neval-flavoured rows: %d" % len(evalish))
if evalish:
    ek = sorted({k for r in evalish for k in r})
    print("eval keys: %s" % ", ".join(ek))
    for r in evalish[-3:]:
        print(json.dumps({k: v for k, v in r.items()}, sort_keys=True)[:800])

# last 20 rows summary for the live tail
tail = rows[-20:]
print("\nlast-20 medians:")
for k in track:
    vals = [r[k] for r in tail if isinstance(r.get(k), (int, float))]
    if vals:
        print("  %-22s %10.5f   (min %.5f max %.5f)" % (k, st.median(vals), min(vals), max(vals)))
