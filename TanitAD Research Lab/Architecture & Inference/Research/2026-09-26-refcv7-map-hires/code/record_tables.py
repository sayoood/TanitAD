"""The early G-MAP-OVERFIT record as markdown tables (read-only over the banked JSONs)."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
a = json.loads((D / "g_map_overfit.EARLY_NONBINDING.json").read_text(encoding="utf-8"))
b = json.loads((D / "g_map_overfit_MAIN_long.INFORMATIVE.json").read_text(encoding="utf-8"))
o = json.loads((D / "grid_oracle_16frames.json").read_text(encoding="utf-8"))
cls = ["nocls", "drivable", "sidewalk", "lane", "crosswalk", "arrow", "edge", "hatched"]
bar = {c: (0.85 if c in ("nocls", "drivable", "sidewalk") else 0.5) for c in cls}
f = lambda v: "n/a" if v is None else f"{v:.3f}"
L = []
L.append("| arm | " + " | ".join(f"{c} (bar {bar[c]})" for c in cls) + " |")
L.append("|---|" + "---:|" * len(cls))
for arm in ("healthy", "s8_zeros", "lane_w0", "s8_detached"):
    r = a["results"][arm]
    fin, s0 = r["final"], r["step0"]
    L.append(f"| {arm} IoU declared / raw | " + " | ".join(
        f"{f(fin['iou'][c])} / {f(fin['iou_raw'][c])}" for c in cls) + " |")
    L.append(f"| {arm} CE@1000 / CE@0 | " + " | ".join(
        f(fin['ce_mean'][c] / s0['ce_mean'][c]) for c in cls) + " |")
L.append("| n scored (0-20 m) | " + " | ".join(str(a["results"]["healthy"]["final"]["n"][c]) for c in cls) + " |")
L.append("| MAIN_long @3,000 declared / raw | " + " | ".join(
    f"{f(b['result']['final']['iou'][c])} / {f(b['result']['final']['iou_raw'][c])}" for c in cls) + " |")
for cell in ("0.25m", "0.5m", "0.1m"):
    L.append(f"| GT-only oracle, {cell} class fractions | " + " | ".join(
        f(o["oracle"][cell]["iou"][c]) for c in cls) + " |")
print("\n".join(L))
v = a["verdict"]
print("\nMAIN verdict:", v["MAIN"]["verdict"], "| bars:", {k: x for k, x in v["MAIN"]["bars"].items()},
      "| ce_ratio_ok all:", all(v["MAIN"]["ce_ratio_ok"].values()),
      "| finite:", v["MAIN"]["loss_finite_every_step"], "| presence_short:", v["MAIN"]["presence_short"])
print("R1/R2:", json.dumps(v["regression_arms"]))
print("controls:", json.dumps({k: x.get("reproduced") for k, x in v["controls"].items()}),
      "| C1 n_scored", v["controls"]["C1_constant_drivable"]["n_scored"], "n_drivable",
      v["controls"]["C1_constant_drivable"]["n_drivable"], "iou",
      v["controls"]["C1_constant_drivable"]["expected_iou_drivable"])
print("time guard:", v["time_guard"], "| G_MAP_OVERFIT:", v["G_MAP_OVERFIT"])
print("s/step, peak GiB (informative; GPU shared):",
      {k: (a["results"][k]["s_per_step"], a["results"][k]["peak_mem_GiB"]) for k in a["results"]})
