"""N4-DAC Q3c: four-family paired CIs (frame executed_tracked, form R): baseline RJ5.R vs pipeline P1 / P2. STRATEGIC PAUSED (PI 2026-09-19)."""
import json, sys
sys.path.insert(0, ".")
import fam_ci as FC
A = json.load(open("fam/fam_base_RJ5R.json", encoding="utf-8"))
out = {}
for nm in ("P1_repairD", "P2_certA_repairD"):
    B = json.load(open(f"fam/fam_{nm}.json", encoding="utf-8"))
    fc = FC.compare(A, B)
    json.dump(fc, open(f"fam/fam_ci_baseRJ5R_vs_{nm}.json", "w", encoding="utf-8"), indent=1, default=str)
    rows = []
    for fam, mets in fc["frames"]["executed_tracked"]["families"].items():
        for k, v in mets.items():
            if isinstance(v, dict) and v.get("status") == "OK":
                rows.append((fam, k, v.get("A"), v.get("B"), v.get("diff_B_minus_A"), v.get("n_tokens"), v.get("n_logs"), v.get("separated_from_0")))
    out[nm] = rows
    sep = [r for r in rows if r[7]]
    print(nm, "metrics", len(rows), "separated", len(sep))
    for r in sep:
        print("  SEP", r)
json.dump(out, open("Q3c_fam_summary.json", "w", encoding="utf-8"), default=str)
# the metrics that decide the repair's safety: longitudinal speed/accel, lateral cross-track/heading/curvature/yaw, tactical
for nm, rows in out.items():
    print(nm)
    for r in rows:
        if any(s in r[1] for s in ("speed_bias", "speed_mae", "accel_mae", "cross", "heading", "curvature", "yaw", "progress_ratio")):
            d = r[4]
            print("  ", r[0], r[1], "A", r[2], "B", r[3], "diff", d)
