import json
import sys
D = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/horizon_review/band_eval/"
R = json.load(open(D + "BAND_BASELINE.json"))
LAB = {"ade_m": ("ADE within band (m)", "PRIMARY"), "fde_m": ("FDE at band end (m)", "PRIMARY"),
       "speed_mae_mps": ("speed MAE (m/s)", "LONGITUDINAL"), "accel_mae_mps2": ("acceleration MAE (m/s2)", "LONGITUDINAL"),
       "along_mae_m": ("along-track MAE (m)", "LONGITUDINAL"),
       "dk_hw_abs_err_m": ("DK headway |arm - human| (m), lead windows", "LONGITUDINAL"), "dk_tg_abs_err_s": ("DK time gap |arm - human| (s), lead windows", "LONGITUDINAL"),
       "heading_mae_deg": ("heading MAE (deg)", "LATERAL"), "curvature_mae_1pm": ("curvature MAE (1/m)", "LATERAL"),
       "yaw_rate_mae_degps": ("yaw-rate MAE (deg/s)", "LATERAL"), "cross_mae_m": ("cross-track MAE (m)", "LATERAL")}
ORDER = ["ade_m", "fde_m", "speed_mae_mps", "accel_mae_mps2", "along_mae_m", "dk_hw_abs_err_m", "dk_tg_abs_err_s", "heading_mae_deg", "curvature_mae_1pm",
         "yaw_rate_mae_degps", "cross_mae_m"]


def f(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


def cell(v):
    return f"{f(v['mean'])} [{f(v['lo'])}, {f(v['hi'])}]" if v.get("status") == "OK" else "n/a"


def dcell(v):
    if v.get("status") != "OK":
        return "n/a", ""
    sg = "no" if not v["separated"] else ("yes, arm lower" if v["delta"] < 0 else "yes, arm higher")
    return f"{v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]", sg


out = []
for band in R["bands"]:
    t = R["bands"][band]["t_s"]
    out.append(f"\n### Band {band}  ({t[0]} to {t[1]} s)\n")
    for s in ("s0", "s1"):
        sd = R["seeds"][s][band]
        out.append(f"**DAG-OACC {s} vs P10**, T1-kin, MEASURED; point [95 % episode-cluster CI]; delta = arm minus P10 (paired). DK counts: {sd['dk_counts']['n_lead_windows']} lead windows.\n")
        out.append("| family | row | DAG-OACC | P10 | delta (paired) | CI excludes 0 | n windows / episodes (paired) |")
        out.append("|---|---|---|---|---|---|---|")
        for m in ORDER:
            r = sd["rows"][m]
            dc, sg = dcell(r["arm_minus_P10"])
            nn = r["arm_minus_P10"]
            out.append(f"| {LAB[m][1]} | {LAB[m][0]} | {cell(r['arm'])} | {cell(r['P10'])} | {dc} | {sg} | {nn.get('n_windows')} / {nn.get('n_episodes')} |")
        lv = sd["DK-VIOL"]
        if lv["arm"].get("status") == "OK":
            out.append(f"| LONGITUDINAL | DK-VIOL share of at-risk lead windows | {f(lv['arm']['share_of_at_risk'], 4)} ({lv['arm']['n_viol']}/{lv['arm']['n_at_risk']}) | "
                       f"{f(lv['P10']['share_of_at_risk'], 4)} ({lv['P10']['n_viol']}/{lv['P10']['n_at_risk']}) | not paired here | n/a | {lv['arm']['n_lead_windows']} lead windows |")
        out.append("")
out.append("\n### TACTICAL decision reads (existing trajectory-derived reads, full 0-6 s plan; not band-sliced)\n")
out.append("| seed | row | DAG-OACC | P10 | delta (paired) | CI excludes 0 | n |")
out.append("|---|---|---|---|---|---|---|")
for s in ("s0", "s1"):
    for nm, v in R["seeds"][s]["TACTICAL_decision_reads"]["rows"].items():
        dc, sg = dcell(v["arm_minus_P10"])
        out.append(f"| {s} | {nm} accuracy | {cell(v['arm'])} | {cell(v['P10'])} | {dc} | {sg} | {v['arm_minus_P10'].get('n_windows')} |")
open(D + "work/tables.md", "w", encoding="utf-8").write("\n".join(out))
print(len(out), "lines")
