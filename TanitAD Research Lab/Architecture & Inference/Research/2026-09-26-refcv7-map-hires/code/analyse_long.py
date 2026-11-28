"""MAIN_long: per-100-step curves (both rules) from the runner's log lines, first-crossing steps,
the range diagnostics and the 0.2 m tolerance F1 at 1,000 / 2,000 / 3,000, and a late-slope fit."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
rec = json.loads((D / "g_map_overfit_MAIN_long.INFORMATIVE.json").read_text(encoding="utf-8"))
rows = []
for ln in (D / "launch_long2.log").read_text(encoding="utf-8").splitlines():
    if ln.startswith("{") and '"arm": "MAIN_long"' in ln and '"eval_call"' in ln:
        rows.append(json.loads(ln))
rows = [r for r in rows if 1 <= r["eval_call"] <= 30]
assert [r["step"] for r in rows] == list(range(100, 3001, 100)), [r["step"] for r in rows]
bars = {"nocls": .85, "drivable": .85, "sidewalk": .85, "lane": .5, "crosswalk": .5,
        "arrow": .5, "edge": .5, "hatched": .5}
cls = ["nocls", "drivable", "sidewalk", "lane", "crosswalk", "arrow", "edge", "hatched"]
print("step | " + " | ".join(cls))
for r in rows:
    print(r["step"], "|", " | ".join(f"{r['iou'][c]:.3f}/{r['iou_raw'][c]:.3f}" for c in cls))
print("\nfirst crossing (declared rule, prior_corrected) and whether it STAYS above afterwards:")
for rule in ("iou", "iou_raw"):
    out = {}
    for c in cls:
        first = next((r["step"] for r in rows if r[rule][c] >= bars[c]), None)
        stays = None if first is None else all(r[rule][c] >= bars[c] for r in rows if r["step"] >= first)
        last_below = max((r["step"] for r in rows if r[rule][c] < bars[c]), default=None)
        out[c] = {"first": first, "stays_above_after_first": stays, "last_below": last_below}
    print(rule, json.dumps(out))


def slope(c, rule="iou", a=2100, b=3000):
    pts = [(r["step"], r[rule][c]) for r in rows if a <= r["step"] <= b]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    k = sxy / sxx
    ss_res = sum((y - (my + k * (x - mx))) ** 2 for x, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    return {"per_100_steps": round(100 * k, 4), "mean": round(my, 4), "n": n,
            "r2": round(1 - ss_res / ss_tot, 3) if ss_tot else None}


print("\nlate slopes (2,100-3,000, OLS, per 100 steps):")
for c in ("lane", "edge", "arrow", "crosswalk", "hatched"):
    print(" ", c, "declared", slope(c), "raw", slope(c, "iou_raw"))
print("  edge 1,100-2,000", slope("edge", "iou", 1100, 2000), "| lane 1,100-2,000", slope("lane", "iou", 1100, 2000))
print("\nrange diagnostics (declared rule; n = GT cells of the class in the bin; 0-20 m band):")
for st, dg in sorted(rec["range_diag"].items(), key=lambda kv: int(kv[0])):
    if "error" in dg:
        print(st, "ERROR", dg["error"])
        continue
    tol = dg.get("tolerance_0.2m_band0_20", {})
    for c in ("lane", "edge", "crosswalk", "arrow", "hatched"):
        cells = []
        for b, v in dg.items():
            if b == "tolerance_0.2m_band0_20":
                continue
            x = v[c]
            cells.append(f"{b}: n {x['n']} iou {'-' if x['iou'] is None else round(x['iou'], 3)}"
                         f"/{'-' if x['iou_raw'] is None else round(x['iou_raw'], 3)}")
        t = tol.get(c)
        ts = "" if t is None else (f" || tol0.2m P {t['P']:.3f} R {t['R']:.3f} F1 {t['F1']:.3f}"
                                   f" (iou {t['iou']:.3f})")
        print(f"  step {st} {c}: " + "; ".join(cells) + ts)
res = rec["result"]
print("\nfinal (3,000) summary from run_arm:", json.dumps({k: res["final"][k] for k in ("iou", "iou_raw")})[:800])
ce = {c: (res["final"]["ce_mean"][c] / res["step0"]["ce_mean"][c]) for c in cls}
print("CE final/step0:", {c: round(v, 3) for c, v in ce.items()})
print("s/step", res["s_per_step"], "peak_mem_GiB", res["peak_mem_GiB"], "finite", res["loss_finite_every_step"],
      "| record peak", rec["peak_cuda_max_memory_allocated_gib"], "| gpu_shared_with", rec.get("gpu_shared_with"),
      "| wall_s", rec["wall_s"])
print("fingerprints", rec["fingerprints"])
