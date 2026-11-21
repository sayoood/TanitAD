"""The early A18 MAIN record as tables (read-only): lane and edge every 500 steps (+ 2,700, where the
decay starts), the step-3,000 table (declared / raw) against the bars, MAIN's harness verdict, C1-C3,
and the 0.2 m tolerance P / R / F1 at 1,000 / 2,000 / 3,000.
Usage: a18_tables.py <g_map_overfit_A18_MAIN.EARLY_NONBINDING.json>"""
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
h = r["results"]["healthy"]
cur = {c["step"]: c for c in h["curve"]}
CLS = ["nocls", "drivable", "sidewalk", "lane", "crosswalk", "arrow", "edge", "hatched"]
BAR = {"nocls": .85, "drivable": .85, "sidewalk": .85}
f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
steps = [s for s in sorted(cur) if s % 500 == 0 or s == 2700]
print("step  | " + " | ".join(f"{s:>5}" for s in steps))
for c in ("lane", "edge"):
    print(f"{c:5s} | " + " | ".join(f(cur[s]["iou"][c]) for s in steps))
print("loss  | " + " | ".join(f(cur[s]["train_loss"]) for s in steps))
print("lr    | " + " | ".join(("-" if cur[s].get("lr") is None else f"{cur[s]['lr']:.1e}") for s in steps))
fin, v = h["final"], r["reading"]
print(f"\nstep 3,000 (declared / raw) -- MAIN {v['MAIN_verdict']}")
for c in CLS:
    b = BAR.get(c, .50)
    print(f"  {c:9s} bar {b:.2f}  {f(fin['iou'][c])} / {f(fin['iou_raw'][c])}  "
          f"{'PASS' if v['MAIN']['bars'][c] else 'FAIL'}  CE ratio ok {v['MAIN']['ce_ratio_ok'].get(c)}  "
          f"n {fin['n'][c]}")
print("presence short:", v["MAIN"]["presence_short"], "| loss finite:", v["MAIN"]["loss_finite_every_step"],
      "| C1-C3 reproduced:", v["controls_reproduced"], "| time guard:", v["time_guard"])
for st in ("1000", "2000", "3000"):
    d = (r.get("informative_diagnostics") or {}).get(st) or {}
    t = d.get("tolerance_0.2m_band0_20")
    if t:
        print(f"@{st} tol 0.2 m: edge P {t['edge']['P']:.3f} R {t['edge']['R']:.3f} F1 {t['edge']['F1']:.3f}"
              f" | lane P {t['lane']['P']:.3f} R {t['lane']['R']:.3f} F1 {t['lane']['F1']:.3f}")
print("s/step", h["s_per_step"], "| peak GiB", h["peak_mem_GiB"], "| wall_s", r["wall_s"],
      "| fingerprints = A15's:", all(r["fingerprints_equal_A15_record"].values()))
