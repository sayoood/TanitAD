"""The A12 lever arm's early record as tables, beside the NEW-2 prereg record (read-only)."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
a12 = json.loads((D / "g_map_overfit_A15.EARLY_NONBINDING.json").read_text(encoding="utf-8"))
base = json.loads((D / "g_map_overfit.EARLY_NONBINDING.json").read_text(encoding="utf-8"))
cls = ["nocls", "drivable", "sidewalk", "lane", "crosswalk", "arrow", "edge", "hatched"]
f = lambda v: "n/a" if v is None else f"{v:.3f}"
print("binding", a12.get("binding"), "| launch_commit", a12.get("launch_commit"),
      "| near_lift_m", a12.get("near_lift_m"), "| gpu_shared_with", a12.get("gpu_shared_with"))
print("fingerprints", a12["fingerprints"])
print("same shared init as the NEW-2 MAIN:",
      a12["fingerprints"].get("branch_init_sha256_without_near")
      == base["fingerprints"]["branch_init_sha256"],
      "| same trunk init:", a12["fingerprints"]["trunk_init_sha256"]
      == base["fingerprints"]["trunk_init_sha256"])
print("\n| arm | " + " | ".join(cls) + " |")
print("|---|" + "---|" * len(cls))
rows = [("NEW-2 MAIN (prereg run)", base["results"]["healthy"])]
rows += [(f"A12 {k}", v) for k, v in a12["results"].items()]
for name, r in rows:
    fin = r["final"]
    print(f"| {name} | " + " | ".join(f"{f(fin['iou'][c])}/{f(fin['iou_raw'][c])}" for c in cls) + " |")
h = a12["results"]["healthy"]
print("\nA12 MAIN CE@1000/CE@0:", {c: round(h["final"]["ce_mean"][c] / h["step0"]["ce_mean"][c], 3)
                                 for c in cls})
print("A12 MAIN curve (declared) lane:", [round(x["iou"]["lane"], 3) for x in h["curve"]])
print("A12 MAIN curve (declared) edge:", [round(x["iou"]["edge"], 3) for x in h["curve"]])
v = a12["verdict"]
print("\nMAIN:", json.dumps(v["MAIN"]))
print("regression arms:", json.dumps(v["regression_arms"]))
print("controls reproduced:", v["controls_reproduced"], "| time guard:", v["time_guard"],
      "| G_MAP_OVERFIT:", v["G_MAP_OVERFIT"])
print("s/step, peak:", {k: (r["s_per_step"], r["peak_mem_GiB"]) for k, r in a12["results"].items()})
dg = (a12.get("informative_diagnostics") or {}).get("MAIN_final", {})
if "error" in dg:
    print("diag error:", dg["error"])
elif dg:
    print("\nA12 MAIN @1000 tolerance 0.2 m:", json.dumps(dg["tolerance_0.2m_band0_20"]))
    for c in ("lane", "edge"):
        print(f"  {c} by range:", {b: (x[c]["n"], None if x[c]["iou"] is None else round(x[c]["iou"], 3))
                                   for b, x in dg.items() if b != "tolerance_0.2m_band0_20"})
