"""S1 finalize: read every raw/*.json, evaluate the PREREG bar mechanically, emit raw/summary.json and tables.md.

The verdict is computed HERE from the committed bar and nowhere else:
  bar = `os - P10` ADE 0-6 s < 0, CI excluding 0, at BOTH inference seeds  (PREREG_HJ_STAGE0.md, H-V8F-FLOOR-1)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")


def J(name):
    p = os.path.join(RAW, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(r, dp=4):
    if r is None:
        return "n/a"
    s = " SEP" if r.get("separated") else ""
    return f"{r['delta']:+.{dp}f} [{r['lo']:+.{dp}f}, {r['hi']:+.{dp}f}]{s}"


F = {(h, s): J(f"floors_{h}_s{s}.json") for h in ("S2", "S6") for s in (0, 1)}
A = {(h, s): J(f"analysis_{h}_s{s}.json") for h in ("S2", "S6") for s in (0, 1)}
FB = {h: J(f"fallback_cv_{h}.json") for h in ("S2", "S6")}
CTL = J("control_refav1.json")
BANKED = {("S2", 0): "analysis_s0.json", ("S2", 1): "analysis_s1.json", ("S6", 0): "analysis_s6_s0.json", ("S6", 1): "analysis_s6_s1.json"}
BATT = "E:/refcv7_eval_kit/battery/step50400_a6text/"

S = {"bar": "os - P10 ADE 0-6 s < 0 with the CI excluding 0 at BOTH inference seeds (PREREG H-V8F-FLOOR-1)"}
# ---- the bar ------------------------------------------------------------------------------------------------
bar_rows, ok_all = {}, True
for s in (0, 1):
    r = F[("S6", s)]["paired_x_minus_y"]["os-P10"]["ade"]
    ok = (r["delta"] < 0) and (r["hi"] < 0)
    bar_rows[f"s{s}"] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "n_windows": r["n_windows"],
                         "n_episodes": r["n_episodes"], "bar_met": bool(ok)}
    ok_all &= ok
S["bar_S6_ade"] = bar_rows
S["verdict"] = "POS" if ok_all else "NEG"
# ---- the headline table ------------------------------------------------------------------------------------------
lines = ["| horizon | seed | pair (x - y) | ADE delta [95 % CI] | FDE delta [95 % CI] | n win / eps |", "|---|---|---|---|---|---|"]
for h in ("S6", "S2"):
    for s in (0, 1):
        f = F[(h, s)]
        for pr in ("os-P10", "os-damp50", "os-kd_x", "os-ha0_ext", "P10-kd_x", "P10-ha0_ext", "damp50-ha0_ext", "kd_x-ha0_ext",
                   "os-kd_x_coarse", "os-kd_x30_6s", "P10-P10"):
            r = f["paired_x_minus_y"].get(pr)
            if r:
                lines.append(f"| {h} | s{s} | {pr} | {fmt(r['ade'])} | {fmt(r['fde'])} | {r['ade']['n_windows']} / {r['ade']['n_episodes']} |")
S["table_paired"] = lines
lv = ["| horizon | seed | arm | mean ADE m | mean FDE m |", "|---|---|---|---|---|"]
for h in ("S6", "S2"):
    for s in (0, 1):
        for k, v in F[(h, s)]["means"].items():
            if "fde" in v:
                lv.append(f"| {h} | s{s} | {k} | {v['ade']:.4f} | {v['fde']:.4f} |")
S["table_levels"] = lv
ac = ["| horizon | arm | mean abs a (m/s^2) | zero-accel frac (<1e-3) |", "|---|---|---|---|"]
for h in ("S6", "S2"):
    for k, v in F[(h, 0)]["acts"].items():
        ac.append(f"| {h} | {k} | {v['mean_abs_a_mps2']:.4f} | {v['zero_accel_frac']:.4f} |")
    v1 = F[(h, 1)]["acts"]["os"]
    ac.append(f"| {h} | os (seed 1) | {v1['mean_abs_a_mps2']:.4f} | {v1['zero_accel_frac']:.4f} |")
S["table_acts"] = ac
# ---- controls ------------------------------------------------------------------------------------------------------------
ctl = {}
if CTL:
    ctl["refav1_grid_known_value"] = CTL["known_value"]
    ctl["fit_p10_vs_original_fit_mlp_kin"] = CTL["fit_p10_vs_original"]
    ctl["rebuild_vs_banked_eval"] = CTL["rebuild_vs_banked_eval"]
    ctl["kd_x_vs_original"] = CTL["kdx_vs_original_P_kdx"]
for h in ("S2", "S6"):
    for s in (0, 1):
        f = F[(h, s)]
        if not f:
            continue
        c = {"p10_minus_p10": f["paired_x_minus_y"]["P10-P10"]["ade"], "floors_vs_dump": f["controls"]}
        b = J("__none__")
        bp = os.path.join(BATT, BANKED[(h, s)])
        if os.path.exists(bp):
            ba = json.load(open(bp, encoding="utf-8"))
            bf = ba["refcv3"]["families_paired"]["paired_os_minus_ha0ext"]["families"]["ADE"]
            mine = (A[(h, s)] or {}).get("refcv3", {}).get("families_paired", {}).get("paired_os_minus_ha0ext", {}).get("families", {}).get("ADE")
            c["os_minus_ha0ext_banked_analysis"] = {k: {x: bf[k][x] for x in ("delta", "lo", "hi")} for k in ("ade_m", "fde_m")}
            c["os_minus_ha0ext_our_derived_dump_analysis"] = ({k: {x: mine[k][x] for x in ("delta", "lo", "hi")} for k in ("ade_m", "fde_m")} if mine else None)
            c["os_minus_ha0ext_our_direct_bootstrap"] = {k: {x: f["paired_x_minus_y"]["os-ha0_ext"][k][x] for x in ("delta", "lo", "hi")} for k in ("ade", "fde")}
        ctl[f"{h}_s{s}"] = c
S["controls"] = ctl
# ---- four families (battery tool) --------------------------------------------------------------------------------------------
fam = {}
for (h, s), a in A.items():
    if not a:
        continue
    fp = a["refcv3"]["families_paired"]
    block = {}
    for nm in ("paired_os_minus_P10", "paired_os_minus_damp50", "paired_os_minus_kd_x", "paired_os_minus_ha0ext",
               "paired_P10_minus_kd_x", "paired_P10_minus_ha0ext", "paired_damp50_minus_ha0ext", "paired_P10_minus_P10"):
        if nm in fp:
            rows = []
            for fm, d in fp[nm]["families"].items():
                for mk, r in d.items():
                    if isinstance(r, dict) and "delta" in r:
                        rows.append({"family": fm, "metric": mk, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                                     "separated": r["separated"], "n_windows": r["n_windows"], "n_dropped": r.get("n_dropped_nonfinite")})
            block[nm] = {"direction": fp[nm]["direction"], "rows": rows}
    dk = (a["refcv3"].get("distance_keeping") or {}).get("paired", {})
    for nm in ("paired_os_minus_P10", "paired_os_minus_damp50", "paired_os_minus_kd_x", "paired_os_minus_ha0ext"):
        if nm in dk:
            block.setdefault(nm, {})["distance_keeping"] = {m: {k: r.get(k) for k in ("delta", "lo", "hi", "separated", "n_used")}
                                                            for m, r in dk[nm]["metrics"].items()}
    levels = {}
    for arm, v in a["arms"].items():
        ff = v["four_families"]
        lo, la = ff["longitudinal"], ff["lateral"]
        levels[arm] = {"speed_mae_mps": lo.get("speed_mae_mps"), "target_speed_acc_0.5": (lo.get("target_speed_acc") or {}).get("within_0.5_mps"),
                       "along_mae_m": lo.get("along_mae_m"), "accel_mae_mps2": lo.get("accel_mae_mps2"),
                       "headway_min_m": (lo.get("distance_keeping") or {}).get("mean_headway_min_m"),
                       "time_gap_min_s": (lo.get("distance_keeping") or {}).get("mean_time_gap_min_s"),
                       "min_ttc_s": (lo.get("distance_keeping") or {}).get("mean_min_ttc_s"),
                       "heading_mae_deg": la.get("heading_mae_deg"), "curvature_mae_1pm": la.get("curvature_mae_1pm"),
                       "yaw_rate_mae_degps": la.get("yaw_rate_mae_degps"), "cross_mae_m": la.get("cross_mae_m")}
    fam[f"{h}_s{s}"] = {"paired": block, "levels": levels}
S["four_families"] = fam
S["strategic"] = "PAUSED (PI R6 ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25) -- not a pass, not omitted"
S["fusion_exploratory"] = {f"{h}_s{s}": F[(h, s)]["fusion"] for (h, s) in F if F[(h, s)]}
S["fallback_cv"] = {h: FB[h] for h in FB if FB[h]}
json.dump(S, open(os.path.join(RAW, "summary.json"), "w"), indent=1)
open(os.path.join(RAW, "tables.md"), "w", encoding="utf-8").write(
    "\n".join(["## paired", *lines, "", "## levels", *lv, "", "## acts", *ac]) + "\n")
print("VERDICT", S["verdict"], json.dumps(bar_rows))
print("\n".join(lines[:28]))
