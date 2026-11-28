"""r7base stage C (CPU only): paired os - ha0_ext cells (cross_paired, tip instruments) + a compact family table
and the control against the BANKED S2 seed-0 analysis. usage: python r7base_stageC.py <out_root>"""
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
BAT = Path(__file__).resolve().parent / "battery"
sys.path.insert(0, str(BAT))
sys.stdout.reconfigure(encoding="utf-8")
import r7_panel as P  # noqa: E402

BANK = Path("D:/refcv7_eval_kit/battery/step50400_a6text")
KEYS = {
    "LON": ["speed_mae_mps", "along_mae_m", "target_speed_acc.within_0.5_mps", "accel_mae_mps2"],
    "LAT": ["heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps", "cross_mae_m"],
}


def cell(ci, key):
    c = (ci.get("components") or {}).get(key)
    if not c:
        return None
    return {"mean": c["mean"], "lo": c["lo"], "hi": c["hi"], "n": c.get("n_windows"), "n_ep": c.get("n_episodes")}


rows = {}
for surf, pat_a, pat_p in (("S2", "analysis_s2_s{s}.json", "panel_s{s}"), ("S6", "analysis_s6_s{s}.json", "s6_s{s}")):
    for s in (0, 1):
        a = json.load(open(out_root / pat_a.format(s=s), encoding="utf-8"))
        r = {"n_windows": a["n_windows"], "n_episodes": a["n_episodes"], "arms": {}}
        for arm in ("os", "ha0_ext", "ha0", "stop"):
            if arm not in a["arms"]:
                continue
            ff = a["arms"][arm]["four_families"]
            lon, lat, tac = ff["longitudinal"], ff["lateral"], ff["tactical"]
            r["arms"][arm] = {
                "tier": a["arms"][arm].get("tier"),
                "LON": {k: cell(lon["ci"], k) for k in KEYS["LON"]},
                "LAT": {k: cell(lat["ci"], k) for k in KEYS["LAT"]},
                "LON_ego_progress": lon.get("ego_progress"),
                "LON_dk": {k: (lon.get("distance_keeping") or {}).get(k) for k in
                           ("status", "n", "n_windows", "mean_headway_min_m", "mean_time_gap_min_s", "n_time_gap",
                            "mean_min_ttc_s", "n_closing", "instants_s")},
                "TAC": {"status": tac.get("status"), "n": tac.get("n"),
                        "lateral_decision": tac.get("lateral_decision"),
                        "longitudinal_decision": tac.get("longitudinal_decision")},
                "STRAT": {k: ff["strategic"].get(k) for k in ("status", "reason", "n")},
            }
        pdir = str(out_root / pat_p.format(s=s)) if surf == "S6" else str(BANK / pat_p.format(s=s))
        cp = P.cross_paired(pdir, [("ha0_ext", "os", "os_minus_ha0ext")], n_boot=2000)
        json.dump(cp, open(out_root / f"cross_paired_{surf.lower()}_s{s}.json", "w", encoding="utf-8"), indent=1, default=str)
        blk = cp.get("os_minus_ha0ext") or cp
        r["paired_os_minus_ha0_ext"] = {}
        for fam in ("ade", "families", "paired", "cells"):
            pass
        r["paired_raw_keys"] = list(blk.keys())[:12] if isinstance(blk, dict) else None
        rows[f"{surf}_s{s}"] = r
        print(surf, s, "done", flush=True)
json.dump(rows, open(out_root / "family_rows.json", "w", encoding="utf-8"), indent=1, default=str)

# control: tip S2 seed-0 os vs the BANKED analysis_s0.json (same dump, older instruments / old 2 s lead block)
b = json.load(open(BANK / "analysis_s0.json", encoding="utf-8"))["arms"]["os"]["four_families"]
n = rows["S2_s0"]["arms"]["os"]
ctl = {}
for fam, ffk in (("LON", "longitudinal"), ("LAT", "lateral")):
    for k in KEYS[fam]:
        old = (b[ffk]["ci"].get("components") or {}).get(k)
        new = n[fam][k]
        ctl[k] = {"banked": old and old["mean"], "tip": new and new["mean"],
                  "abs_diff": (abs(old["mean"] - new["mean"]) if old and new else None)}
json.dump(ctl, open(out_root / "control_vs_banked_s2_s0.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(ctl))
print("ZZSTAGEC_DONE")
