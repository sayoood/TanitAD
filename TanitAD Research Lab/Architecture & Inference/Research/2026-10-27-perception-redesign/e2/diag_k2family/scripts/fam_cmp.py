"""Side-by-side four-family rows: K2M1N20-K0 (dt-fixed reread) vs K2DC-K0, seeds 0 and 1. Read-only.
Prints delta [lo, hi] sep, a_mean, b_mean and relative delta (delta / b_mean)."""
import json, sys

SRC = {
    ("K2M1N20", 0): "F:/refcv9_e3/k2diag/src/k2m1n20/reread_dtfix_read_families_s0.json",
    ("K2M1N20", 1): "F:/refcv9_e3/k2diag/src/k2m1n20/reread_dtfix_read_families_s1.json",
    ("K2DC", 0): "C:/Users/Admin/r9e2_read_K2DC/pull/read/families_s0.json",
    ("K2DC", 1): "C:/Users/Admin/r9e2_read_K2DC/pull/read/families_s1.json",
}
D = {k: json.load(open(p, encoding="utf-8")) for k, p in SRC.items()}
for k, d in D.items():
    print(k, d.get("schema"), "dt_s", d.get("dt_s"), "n", d.get("n_windows_paired"), "a", d["a"]["dir"], "b", d["b"]["dir"])
fams = list(D[("K2DC", 0)]["rows"].keys())
out = {}
for fam in fams:
    rows = D[("K2DC", 0)]["rows"][fam]
    if not isinstance(rows, dict):
        continue
    print("\n==", fam)
    for r in rows:
        line = "%-38s" % r
        for key in (("K2M1N20", 0), ("K2DC", 0), ("K2M1N20", 1), ("K2DC", 1)):
            x = D[key]["rows"].get(fam, {}).get(r)
            if not isinstance(x, dict) or x.get("delta") is None:
                line += " | %-34s" % "-"
                continue
            rel = (x["delta"] / x["b_mean"] * 100) if x.get("b_mean") else float("nan")
            sep = "SEP" if x.get("separated") else "   "
            worse = ""
            if x.get("separated"):
                worse = "W" if ((x["delta"] > 0) == (x.get("better") == "lower")) else "B"
            line += " | %+.4f [%+.4f,%+.4f] %s%s %+5.1f%%" % (x["delta"], x["lo"], x["hi"], sep, worse, rel)
            out.setdefault(fam, {}).setdefault(r, {})["%s_s%d" % key] = {
                "delta": x["delta"], "lo": x["lo"], "hi": x["hi"], "sep": x.get("separated"), "a": x.get("a_mean"),
                "b": x.get("b_mean"), "rel_pct": rel, "n": x.get("n"), "n_ep": x.get("n_episodes")}
        print(line)
print("\ncolumns: K2M1N20-K0 s0 | K2DC-K0 s0 | K2M1N20-K0 s1 | K2DC-K0 s1   (W = separated worse, B = separated better)")
json.dump(out, open("F:/refcv9_e3/k2diag/fam_cmp.json", "w"), indent=1)
