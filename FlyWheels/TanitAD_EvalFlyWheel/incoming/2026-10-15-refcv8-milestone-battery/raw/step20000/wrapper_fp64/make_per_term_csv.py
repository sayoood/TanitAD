"""Per-term table (F32 vs F64) from raw/arm_F32.json + raw/arm_F64.json -> per_term_wrapper_rel.csv (pure stdlib)."""
import csv, json
from pathlib import Path
H = Path(__file__).resolve().parent
a, b = (json.load(open(H / "raw" / f"arm_{x}.json", encoding="utf-8")) for x in ("F32", "F64"))
ta, tb = a["terms"], b["terms"]
keys = sorted(set(ta) | set(tb), key=lambda k: -ta.get(k, {}).get("wrapper", 0.0))
with open(H / "per_term_wrapper_rel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["term", "plain_F32", "wrapper_rel_F32", "floor_F32", "plain_F64", "wrapper_rel_F64", "floor_F64", "ratio_F32_over_F64", "plain_rel_F32_vs_F64", "in_both"])
    for k in keys:
        x, y = ta.get(k), tb.get(k)
        w.writerow([k, x and x["plain"], x and x["wrapper"], x and x["floor"], y and y["plain"], y and y["wrapper"], y and y["floor"],
                    (x["wrapper"] / y["wrapper"]) if (x and y and y["wrapper"] > 0) else "",
                    (abs(x["plain"] - y["plain"]) / max(abs(y["plain"]), 1e-12)) if (x and y) else "", bool(x and y)])
print(len(keys), "terms;", sum(1 for k in keys if k in ta and k in tb), "in both;",
      "F32 only:", [k for k in keys if k not in tb], "F64 only:", [k for k in keys if k not in ta])

rows = [float(r["plain_rel_F32_vs_F64"]) for r in csv.DictReader(open(H / "per_term_wrapper_rel.csv", encoding="utf-8")) if r["plain_rel_F32_vs_F64"] != ""]
print("plain value F32 vs F64 rel diff: n", len(rows), ">1e-2:", sum(r > 1e-2 for r in rows), ">1e-3:", sum(r > 1e-3 for r in rows), ">1e-5:", sum(r > 1e-5 for r in rows),
      "median %.2e" % sorted(rows)[len(rows) // 2])
