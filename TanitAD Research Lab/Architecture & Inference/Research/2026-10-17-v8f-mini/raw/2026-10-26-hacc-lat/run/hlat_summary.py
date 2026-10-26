"""RUN_SUMMARY.json for the HLAT chain. Reads ONLY the controls' JSON, the fit JSON (train-side) and the bake reports, and checks that each expected eval run left its files. It never opens a t1v8f_*.json
of an arm run: those carry the outcome rows (arm - P10, arm - HACC on eval-141), which the Master Mind reads with the pre-registered reader."""
import hashlib
import json
import os
import sys

H = "C:/Users/Admin/hlat"
OUT = H + "/run"
res = {"prereg": "PREREG_V8F_HACC_LAT", "controls": {}, "runs": {}, "ckpts": {}, "fit": {}, "all_controls_pass": True}


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


for s in (0, 1):
    for nm, p in (("C-A1", f"{H}/controls/CA1_s{s}.json"), ("C-B1", f"{H}/controls/CB1_s{s}.json"), ("C-B2", f"{H}/controls/CB2_s{s}.json"), ("C-T2", f"{H}/controls/CT2_s{s}.json"),
                  ("C-T", f"{H}/controls/ct_s{s}/C-T.json"), ("C-T(dump-ticks)+tick0", f"{H}/controls/ct_s{s}/C-T_ticks.json")):
        d = load(p)
        res["controls"][f"{nm}_s{s}"] = None if d is None else {k: d[k] for k in d if k in ("pass", "max_abs_all", "cross_max_abs_m", "along_max_abs_m_power", "max_abs_m", "n_files", "n_windows_compared",
                                                                                         "tick0_plan_vs_dump_ol_max_abs_m", "dumps_identical_to_original", "tol_m", "bar_m", "n_windows_checked")}
        if d is None or not d.get("pass", False):
            res["all_controls_pass"] = False
    f = load(f"{H}/fit/hlat_fit_M2b_s{s}.json")
    if f is not None:
        res["fit"][f"s{s}"] = {"C-A2": f["controls"], "lambda0": f["lambda0"], "m_star": f["m_star"], "lam_star": f["lam_star"], "rule_branch": f["rule_branch"], "n_tick_samples_gt_ok": f["ticks"]["n_gt_ok"],
                               "by_tick": f["ticks"]["by_tick"], "v_star": f["arm_B"]["v_star"], "v_branch": f["arm_B"]["branch"], "v_reading_sensitivity": f["arm_B"]["reading_sensitivity"],
                               "mean_abs_w_star_minus_hacc": f["mean_abs_w_star_minus_hacc"], "seconds": f["seconds"]}
        if not (f["controls"]["C-A2_pass"] and f["controls"]["regression_pass"] and f["controls"]["kkt_pass"]):
            res["all_controls_pass"] = False
    else:
        res["all_controls_pass"] = False
    for arm in ("A", "B", "CA1", "CB1", "CB2"):
        b = load(f"{H}/ckpt/BAKE_{arm}_s{s}.json")
        if b is not None:
            res["ckpts"][f"{arm}_s{s}"] = {"path": b["out"], "sha256": b["sha256"], "bytes": b["bytes"], "gate_v_star": b["gate_v_star"], "new_keys": b["new_keys"]}
    for tag in (f"A_s{s}_m10", f"B_s{s}_m10", f"A_s{s}_m30", f"B_s{s}_m30", f"CA1_s{s}_m10", f"CB1_s{s}_m10"):
        j, d = f"{OUT}/t1v8f_{tag}.json", f"{OUT}/dump_{tag}"
        res["runs"][tag] = {"json": os.path.exists(j), "dump_dir": os.path.isdir(d), "n_episode_files": len([x for x in os.listdir(d) if x.startswith("ep")]) if os.path.isdir(d) else 0}
res["runs_complete"] = all(r["json"] and r["dump_dir"] and r["n_episode_files"] == 141 for r in res["runs"].values())
json.dump(res, open(H + "/RUN_SUMMARY.json", "w"), indent=1)
print(json.dumps({"all_controls_pass": res["all_controls_pass"], "runs_complete": res["runs_complete"]}))
sys.exit(0 if res["all_controls_pass"] and res["runs_complete"] else 2)
