"""Markdown tables from an hist_analyze.py output dir. usage: python make_tables.py <analysis_dir> <label>"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); label = sys.argv[2]
ARMS = ["os_rep", "os_hold", "os_st", "os_nt"]
NAME = {"os": "A REAL", "os_rep": "A-rep (seed 1)", "os_hold": "B HOLD", "os_st": "C NAVSIM-ST", "os_nt": "D NAVSIM-NT"}
f = lambda c, k=4: ("%+.*f [%+.*f, %+.*f]" % (k, c["delta"], k, c["lo"], k, c["hi"])) + (" SEP" if c.get("separated") else "")
out = []
for surf in ("s6", "s2"):
    cp = json.load(open(d / f"cross_paired_{surf}.json", encoding="utf-8"))
    an = json.load(open(d / f"analysis_{surf}.json", encoding="utf-8"))
    nw, ne = cp["n_windows"], cp["n_episodes"]
    out.append(f"\n#### {label} {surf.upper()}: paired delta (arm - A REAL), episode-cluster bootstrap B=2000, n {nw} windows / {ne} episodes\n")
    cells = [("ADE", "ade_m", 3), ("ADE", "fde_m", 3), ("longitudinal", "LON_speed_mae_mps", 4), ("longitudinal", "LON_along_mae_m", 4),
             ("longitudinal", "LON_accel_mae_mps2", 4), ("lateral", "LAT_cross_mae_m", 4), ("lateral", "LAT_heading_mae_deg", 3),
             ("lateral", "LAT_yaw_rate_mae_radps_valid", 5), ("tactical", "TAC_traj_lat_correct", 4), ("tactical", "TAC_traj_lon_correct", 4)]
    out.append("| family | cell | REAL (mean) | " + " | ".join(NAME[a] for a in ARMS) + " |")
    out.append("|---|---|---|" + "---|" * len(ARMS))
    for fam, key, k in cells:
        base = cp["means"]["os"].get(key)
        row = [fam.upper(), key, ("%.4f" % base) if base is not None else "-"]
        for a in ARMS:
            c = cp["pairs"][f"{a}_minus_os"]["families"][fam].get(key)
            row.append(f(c, k) if c else "-")
        out.append("| " + " | ".join(row) + " |")
    out.append(f"| STRATEGIC | - | NOT APPLICABLE (strategic layer OFF, PI R5), n = 0 | | | | |")
lv = json.load(open(d / "lead_viol.json", encoding="utf-8"))
for surf in ("S6", "S2"):
    L = lv[surf]
    out.append(f"\n#### {label} LEAD-VIOL {surf} (lead windows {L['n_lead_windows']}; at-risk = human keeps >= 1.0 s; paired X - A REAL)\n")
    out.append("| arm | LEAD-VIOL share (all lead) | share_of_at_risk | n_viol/n_lead | progress ratio | paired at-risk delta |")
    out.append("|---|---|---|---|---|---|")
    for a in ["os", "os_rep", "os_hold", "os_st", "os_nt", "ha0_ext", "stop"]:
        r = L["arms"].get(a)
        if not r:
            continue
        pv = L["paired_vs_os"].get(a, {}).get("at_risk", {})
        out.append("| %s | %s | %s | %s/%s | %s | %s |" % (NAME.get(a, a), r.get("share"), r.get("share_of_at_risk"), r.get("n_viol"), r.get("n_lead"),
                   ("%.3f" % r["progress_ratio_mean"]) if r.get("progress_ratio_mean") is not None else "-", f(pv) if "delta" in pv else "-"))
v = json.load(open(d / "verdict_s6.json", encoding="utf-8"))
out.append(f"\n#### {label} PREREG s5 verdict (S6 primaries)\n")
for a, r in v.items():
    out.append(f"* **{NAME[a]}: {r['verdict']}** -- " + "; ".join(
        f"{x['metric'].split(' ')[0]} {x['delta']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}] (margin {x['margin']:.4f}, floor {x['seed_floor_abs']:.4f}) {x['class']}"
        for x in r["rows"] if "delta" in x))
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(out))
