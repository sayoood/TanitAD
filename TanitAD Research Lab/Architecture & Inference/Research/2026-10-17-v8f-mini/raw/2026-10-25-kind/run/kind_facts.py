"""PREREG_V8F_KIND: the committed reads and their intervals, printed from the chain's own kind_read.json (EXPLORATORY formatting only; the read decides)."""
import json
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/box2_ship/logs/v8f_kind/kind_read.json"
d = json.load(open(P, encoding="utf-8"))
print("outcome", d["decision"]["outcome"], "| A", [round(a, 3) for a in d["decision"]["A"]], "| per-seed row", d["decision"]["per_seed_row"])
print("speed side-read", json.dumps(d["speed_side_read"])[:400])
print("identity", {s: {k: (v["delta"], v["per_window_equal"]) for k, v in d["identity_control"][s].items()} for s in ("s0", "s1")})


def fmt(x):
    if not isinstance(x, dict):
        return str(x)
    sep = "sep" if x.get("separated") else "ns"
    return f"{x['delta']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] {sep}"


for s in ("s0", "s1"):
    r = d["rows"][s]
    print(f"\n== {s}: keys {list(r.keys())[:12]}")
    for key in ("gain", "necessity"):
        if key in r:
            for k, v in r[key].items():
                print(f"  {key:9s} {k:9s} {fmt(v)}")
    for key in ("gain_union", "union", "ade6"):
        if key in r:
            print(f"  {key}: {json.dumps(r[key])[:300]}")
st = d["strata"]["strata"]
for name, v in st.items():
    print(f"\n== stratum {name}: {v['n_windows']} windows / {v['n_episodes']} episodes")
    for s in ("s0", "s1"):
        g = v["seeds"][s]["gain"]
        print(f"  {s} " + " | ".join(f"{k} {fmt(x)}" for k, x in g.items()))
print("\ncontribution to pooled gain (m):", json.dumps(d["strata"]["contribution_to_pooled_gain_m"])[:1500])
b = d["beside"]
print("\nbeside keys", {s: list(b[s].keys())[:20] for s in ("s0", "s1")})
print("families_skipped", json.dumps(d["families_skipped"])[:600])
print("strategic_family", json.dumps(d["strategic_family"])[:300])
