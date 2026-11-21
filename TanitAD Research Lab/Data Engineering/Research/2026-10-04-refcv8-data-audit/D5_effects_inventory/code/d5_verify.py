"""D5 effects inventory -- independent re-read of the headline numbers from the RAW artifacts.

Purpose: every number in EFFECTS_INVENTORY.md that comes from a RESULT.md table is re-derived here
from the raw JSON / scorer files / label file it claims to come from, so the inventory does not
rest on a prose summary.  Read-only.  No GPU.  Prints NO clip ids (the label file is read, only
aggregates leave this process).

Controls (must read their known values, reported in raw/d5_verify.json):
  * C-MUT   the comparison function must REJECT a deliberately wrong expectation (mutation arm).
  * C-MD5   the label file's md5 must equal the md5 the launch config.json records for v7_labels.
  * C-N     the label file must hold exactly 4,572 records (the launch record's n_records).
  * C-UNREAD  count of files that could not be read must be 0 (a 0-hit claim about an unreadable
              file is not absence).
Run:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 python d5_verify.py
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("D:/Projects/TanitAD")
AI = REPO / "TanitAD Research Lab/Architecture & Inference/Research"
ROUTE = AI / "2026-10-04-refcv7-route-following/raw"
MAPBOX = AI / "2026-10-04-refcv7-map-box-diagnostics/raw"
NAV = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones"
LABELS = Path("D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz")
CONFIG = Path("D:/refcv7_eval_kit/ckpt/config.json")
OUT = Path(__file__).resolve().parent.parent / "raw" / "d5_verify.json"

rows = []
unreadable = []


def load(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        unreadable.append(f"{p}: {type(e).__name__}")
        return None


def close(got, want, tol):
    return got is not None and abs(got - want) <= tol


def check(name, got, want, tol, source):
    ok = close(got, want, tol)
    rows.append({"name": name, "got": got, "claimed": want, "tol": tol, "ok": bool(ok), "source": source})
    return ok


# ---------------------------------------------------------------- C-MUT (the check must be able to fail)
mut_rejects = not close(0.841, 0.5, 0.01)
rows.append({"name": "C-MUT comparison rejects a wrong expectation", "got": mut_rejects, "claimed": True, "tol": 0,
             "ok": bool(mut_rejects), "source": "self-test"})

# ---------------------------------------------------------------- route-following package (raw/route_analysis.json)
ra = load(ROUTE / "route_analysis.json")
if ra:
    s = "route-following/raw/route_analysis.json"
    r = ra["decision_eval_s0"]["rates_turn"]
    check("route: T tactical side-correct on GT-turn", r["T_tac_side"], 0.3925, 5e-4, s)
    check("route: C fan contains correct", r["C_fan_reach"], 1.0, 1e-9, s)
    check("route: K decoder pick direction-correct", r["K_core_pick"], 0.8411, 5e-4, s)
    check("route: E E9 pick direction-correct", r["E_e9_pick"], 0.8411, 5e-4, s)
    check("route: N nav side == GT", r["N_nav"], 0.4393, 5e-4, s)
    dc = ra["decision_eval_s0"]["drops"]["D_core (C -> K)"]
    check("route: D_core", dc["drop"], 0.1589, 5e-4, s)
    check("route: D_core CI lo", dc["ci95"][0], 0.0803, 5e-4, s)
    check("route: D_core CI hi", dc["ci95"][1], 0.2577, 5e-4, s)
    dc1 = ra["decision_eval_s1"]["drops"]["D_core (C -> K)"]
    check("route: D_core seed1", dc1["drop"], 0.150, 2e-3, s)
    a4 = ra["A4_bounds"]["eval_s0"]
    check("route: B3 all-window dADE", a4["B3"]["all"]["ade"]["delta_vs_V0"], -0.6061, 5e-4, s)
    check("route: B3 all CI lo", a4["B3"]["all"]["ade"]["delta_ci95"][0], -0.7276, 5e-4, s)
    check("route: B3 all CI hi", a4["B3"]["all"]["ade"]["delta_ci95"][1], -0.4932, 5e-4, s)
    check("route: B3 turn dADE", a4["B3"]["turn"]["ade"]["delta_vs_V0"], -0.8917, 5e-4, s)
    check("route: B1t turn dADE", a4["B1t"]["turn"]["ade"]["delta_vs_V0"], -0.5658, 5e-4, s)
    check("route: B1t all dADE", a4["B1t"]["all"]["ade"]["delta_vs_V0"], -0.0757, 5e-4, s)
    check("route: B2 turn dADE CI hi (includes 0)", a4["B2"]["turn"]["ade"]["delta_ci95"][1], 0.0280, 5e-4, s)
    check("route: ORACLE all dADE", a4["ORACLE"]["all"]["ade"]["delta_vs_V0"], -1.1748, 5e-4, s)
    check("route: V0 all-window ADE (800 classified)", a4["V0"]["all"]["ade"]["mean"], 2.0044, 5e-4, s)
    check("route: V0 turn ADE", a4["V0"]["turn"]["ade"]["mean"], 3.1861, 5e-4, s)
    check("route: ORACLE turn ADE", a4["ORACLE"]["turn"]["ade"]["mean"], 1.1907, 5e-4, s)
    w3 = ra["A1"]["levers_eval_s0"]["W3|0"]["all"]["ade"]
    check("route: W3 (E9 graft off) all-window dADE", w3["delta_vs_V0"], -0.0715, 5e-4, s)
    check("route: W3 CI hi", w3["delta_ci95"][1], -0.0154, 5e-4, s)
    v3 = ra["levers_eval_s0"]["V3|10"]["turn"]["dir_correct"]
    check("route: V3 (navc x10) turn dir-correct delta", v3["delta_vs_V0"], 0.0467, 5e-4, s)
    v0c = ra["levers_eval_s0"]["V0c|None"]["all"]["ade"]
    check("route: V0c (ceiling fix emulated) all-window dADE", v0c["delta_vs_V0"], 0.0863, 5e-4, s)
    check("route: V0c CI lo", v0c["delta_ci95"][0], 0.0179, 5e-4, s)
    sf = ra["seed_floor"]
    check("route: seed floor E9 pick same frac", sf["e9_pick_same_frac"], 0.911, 5e-4, s)
    check("route: seed floor median traj diff m", sf["traj_abs_diff_m_median"], 0.6388, 5e-4, s)
    check("route: ADE seed0", sf["ade_e9_s0"], 1.7229, 5e-4, s)
    check("route: ADE seed1", sf["ade_e9_s1"], 1.6947, 5e-4, s)
    check("route: navc gate", ra["gates"]["navc_gate"], 0.1625, 5e-4, s)
    pd = ra["posthoc_prior_diag_eval_s0"]["turn"]
    check("route: wrong-dir picks n", pd["n_pick_dir_wrong"], 17, 0.5, s)
    check("route: wrong-dir picks with prior's direction", pd["frac_wrong_picks_with_prior_dir"], 0.8235, 5e-4, s)
    check("route: prior direction-correct on turns", pd["prior_dir_correct_frac"], 0.5701, 5e-4, s)
    fa = ra["fan_eval_s0"]["all"]
    check("route: Spearman(E9, -ADE)", fa["rho_e9"]["mean"], 0.7527, 5e-4, s)
    check("route: Spearman(random)", fa["rho_random"]["mean"], -0.0047, 5e-4, s)
    check("route: top-8 holds L and R (all classified)", fa["e9_top8_LR"]["mean"], 0.27, 5e-4, s)
    v0 = ra["levers_eval_s0"]["V0|None"]
    check("route: V0 turn speed MAE 0-2s", v0["turn"]["speed_mae_0_2s"]["mean"], 0.312, 5e-4, s)
    check("route: V0 turn along-signed 6s", v0["turn"]["along_signed_6s"]["mean"], 3.272, 5e-4, s)
    check("route: V0 turn terminal heading err deg", v0["turn"]["term_heading_err_deg"]["mean"], 22.369, 5e-4, s)
    tt = ra["tactical_from_trajectory_eval_s0"]["V0|None"]
    check("route: tactical lon accuracy", tt["longitudinal_decision"]["accuracy"], 0.8273, 5e-4, s)
    check("route: tactical lon kappa", tt["longitudinal_decision"]["kappa"], 0.5395, 5e-4, s)
    check("route: tactical lat kappa", tt["lateral_decision"]["kappa"], 0.8229, 5e-4, s)

# ---------------------------------------------------------------- box NMS (raw/box_nms.json)
bn = load(ROUTE / "box_nms.json")
if bn:
    s = "route-following/raw/box_nms.json"
    e = bn["box3d"]["eval"]
    check("box: AP@2m no NMS", e["no_NMS"]["AP2m"], 0.2483, 5e-4, s)
    check("box: AP@2m centre NMS 2.5", e["centre_NMS(2.5)"]["AP2m"], 0.3494, 5e-4, s)
    check("box: mean boxes/obj @0.2589 no NMS", e["no_NMS"]["dups_at_0p2589"]["mean_slots_per_detected_object"], 2.1198, 5e-4, s)
    check("box: mean boxes/obj @0.2589 after NMS (the RESULT's 1.07)", e["centre_NMS(2.5)"]["dups_at_0p2589"]["mean_slots_per_detected_object"], 1.0673, 5e-4, s)
    # RESULT quotes the post-NMS duplicate count at the OLD gate; the shipped config re-fits the gate:
    check("box: mean boxes/obj at the RE-FITTED gate 0.2145 after NMS", e["centre_NMS(2.5)"]["dups_at_fit_gate"]["mean_slots_per_detected_object"], 1.1052, 5e-4, s)
    check("box: frac >=2 at the RE-FITTED gate after NMS", e["centre_NMS(2.5)"]["dups_at_fit_gate"]["frac_detected_objects_with_ge2"], 0.1038, 5e-4, s)
    check("box: F1 at re-fitted gate after NMS", e["centre_NMS(2.5)"]["at_fit_gate"]["f1"], 0.3922, 5e-4, s)
    check("box: F1 no NMS at its gate", e["no_NMS"]["at_fit_gate"]["f1"], 0.3034, 5e-4, s)
    ag = bn["agent"]["eval"]
    check("agent: AP@2m no NMS", ag["no_NMS"]["AP2m"], 0.1314, 5e-4, s)
    check("agent: AP@2m centre NMS 3.0", ag["centre_NMS(3.0)"]["AP2m"], 0.300, 1.5e-3, s)

# ---------------------------------------------------------------- map (raw/M_c.json)
mc = load(MAPBOX / "M_c.json")
if mc:
    s = "map-box/raw/M_c.json"
    ev = mc["eval"]
    for cls, pc, th in (("lane", 0.068, 0.164), ("crosswalk", 0.048, 0.105), ("arrow", 0.027, 0.059),
                        ("edge", 0.0, 0.042), ("hatched", 0.0, 0.062), ("drivable", 0.574, 0.576)):
        check(f"map: {cls} IoU declared rule (all bands)", ev["pc"][cls]["all"]["iou"], pc, 1.0e-3 if cls != "edge" else 1e-3, s)
        check(f"map: {cls} IoU TRAIN-fitted thresholds", ev["thr_phat"][cls]["all"]["iou"], th, 1e-3, s)
    check("map: lane paired thr-pc lo", mc["eval_paired_alt_minus_pc_thin"]["lane"]["thr_phat"]["ci95_alt_minus_pc_paired"][0], 0.086, 1e-3, s)
mb = load(MAPBOX / "M_b.json")
if mb:
    # structure differs per package version; record the keys instead of guessing
    rows.append({"name": "map: M_b.json keys (not asserted)", "got": list(mb.keys())[:8], "claimed": None, "tol": None, "ok": True,
                 "source": "map-box/raw/M_b.json"})
bb = load(MAPBOX / "B_box.json")
if bb:
    s = "map-box/raw/B_box.json"
    t3 = bb["box3d"]["eval"]["transforms"]["T3"]
    check("box: T3 conf_ratio (box3d)", t3.get("conf_ratio"), 0.973, 1e-3, s)
    t0 = bb["box3d"]["eval"]["transforms"]["T0"]
    check("box: T0 conf_ratio at the declared 0.5 gate (box3d)", t0.get("conf_ratio"), 0.008, 1e-3, s)

# ---------------------------------------------------------------- NavSim (step 5,000 / 30,000)
s5 = load(NAV / "step5000/summary_navtest.json")
if s5:
    s = "navsim/raw/milestones/step5000/summary_navtest.json"
    a = s5["arms"]
    check("navtest 5k: R7_A1 PDMS x100", a["R7_A1"]["PDMS"], 65.5976, 5e-4, s)
    check("navtest 5k: STOP PDMS x100", a["STOP"]["PDMS"], 61.8202, 5e-4, s)
    check("navtest 5k: R7_A1 DAC", a["R7_A1"]["DAC"], 84.5546, 5e-4, s)
    check("navtest 5k: R7_A1 NC", a["R7_A1"]["NC"], 88.0948, 5e-4, s)
    check("navtest 5k: A1-STOP", s5["pairs"]["R7_A1__minus__STOP"]["delta_x100"], 3.7774, 5e-4, s)
dn = load(NAV / "step5000/decomposition_navtest.json")
if dn:
    s = "navsim/raw/milestones/step5000/decomposition_navtest.json"
    sc = dn["ladder"]["single_term_ceiling"]
    check("navtest 5k: DAC single-term ceiling gain", sc["DAC"]["gain_x100"], 7.296, 5e-4, s)
    check("navtest 5k: TTC gain", sc["TTC"]["gain_x100"], 4.3241, 5e-4, s)
    check("navtest 5k: EP gain", sc["EP"]["gain_x100"], 4.1616, 5e-4, s)
    check("navtest 5k: NC gain", sc["NC"]["gain_x100"], 1.9299, 5e-4, s)
    check("navtest 5k: speed band 2-5 m/s minus STOP", dn["by_speed_band"]["2-5 m/s"]["minus_STOP_x100"], -11.5657, 5e-4, s)
    check("navtest 5k: RIGHT minus STOP", dn["by_command"]["RIGHT"]["minus_STOP_x100"], -3.7318, 5e-4, s)
for step, want in (("step5000", (0.1624, 0.3547, 0.1607)), ("step30000", (0.2269, 0.2596, 0.1564))):
    nh = load(NAV / f"{step}/summary_navhard.json")
    if nh:
        s = f"navsim/raw/milestones/{step}/summary_navhard.json"
        a = nh["arms"]
        check(f"navhard {step[4:]}: R7_A1 official two-stage EPDMS", a["R7_A1"]["official_two_stage_EPDMS"], want[0], 5e-4, s)
        check(f"navhard {step[4:]}: R7_A1 S2 DAC-zero rate", a["R7_A1"]["S2_multiplier_zero_rates"]["DAC"], want[1], 5e-4, s)
        check(f"navhard {step[4:]}: R7_A1 S2 NC-zero rate", a["R7_A1"]["S2_multiplier_zero_rates"]["NC"], want[2], 5e-4, s)
        check(f"navhard {step[4:]}: STOP official EPDMS", a["STOP_zero"]["official_two_stage_EPDMS"], 0.2985, 5e-4, s)
c30 = load(NAV / "step30000/scores_navtest/r7s30000_R7_A1/r7s30000_R7_A1.counts.json")
if c30:
    s = "navsim/raw/milestones/step30000/scores_navtest/r7s30000_R7_A1/r7s30000_R7_A1.counts.json"
    sm = c30["summary"]
    check("navtest 30k: R7_A1 PDMS (scorer's own summary, no interval yet)", sm["PDMS"] * 100, 71.8849, 5e-4, s)
    check("navtest 30k: DAC", sm["DAC"] * 100, 87.9796, 5e-4, s)
    check("navtest 30k: NC", sm["NC"] * 100, 91.2811, 5e-4, s)
    check("navtest 30k: EP", sm["EP"] * 100, 69.5362, 5e-4, s)
    check("navtest 30k: TTC", sm["TTC"] * 100, 80.8414, 5e-4, s)
    check("navtest 30k: tokens scored", c30.get("csv_token_rows"), 12146, 0.5, s)

# ---------------------------------------------------------------- label file + launch config (C-MD5, C-N, nav timing, VLM, lat_peak, vmax)
cfg = load(CONFIG)
label_facts = {}
try:
    md5 = hashlib.md5(open(LABELS, "rb").read()).hexdigest()
    recs = [json.loads(l) for l in gzip.open(LABELS, "rt", encoding="utf-8")]
    want_md5 = cfg["v7_labels"]["md5"] if cfg else None
    rows.append({"name": "C-MD5 label md5 == launch config v7_labels.md5", "got": md5, "claimed": want_md5, "tol": 0,
                 "ok": bool(want_md5 == md5), "source": "config.json vs label file"})
    check("C-N label file record count", len(recs), 4572, 0.5, "label file")
    check("labels: records with t0_s == 8.0", sum(r["t0_s"] == 8.0 for r in recs), 4572, 0.5, "label file")
    check("labels: records with tactical band [2,6]", sum(r["bands"]["tactical_s"] == [2.0, 6.0] for r in recs), 4572, 0.5, "label file")
    lr = [r for r in recs if r["nav_command"]["token"] in ("NAV_TURN_L", "NAV_TURN_R")]
    ts = np.array([r["nav_command"]["args"]["time_s"] for r in lr if r["nav_command"]["args"].get("time_s") is not None], float)
    check("labels: L/R nav records", len(lr), 1675, 0.5, "label file")
    check("labels: L/R median turn start s after anchor", float(np.median(ts)), 7.3, 0.06, "label file")
    check("labels: L/R share turn starts >6 s", float((ts > 6).mean()), 0.527, 1e-3, "label file")
    check("labels: L/R share turn starts >10 s", float((ts > 10).mean()), 0.420, 1e-3, "label file")
    check("labels: L/R p90 turn start s", float(np.percentile(ts, 90)), 26.2, 0.06, "label file")
    la = sum(1 for r in recs if r["alpamayo"]["lateral"].get("agree") is False)
    lo = sum(1 for r in recs if r["alpamayo"]["longitudinal"].get("agree") is False)
    check("labels: VLM-vs-geometry lateral disagree", la, 1537, 0.5, "label file")
    check("labels: VLM-vs-geometry longitudinal disagree", lo, 1655, 0.5, "label file")
    lp = np.array([abs(r["a_tac"]["lat_args"]["lat_peak_m"]) for r in recs], float)
    check("labels: lat_peak_m median", float(np.median(lp)), 19.67, 0.01, "label file")
    check("labels: lat_peak_m share > 5 m", float((lp > 5).mean()), 0.702, 1e-3, "label file")
    vb = [r["speed_max_input"]["v_max_bucket_kmh"] for r in recs]
    check("labels: clips fed a <=30 km/h ceiling", sum(1 for v in vb if v <= 30), 1741, 0.5, "label file")
    label_facts = {"n": len(recs), "md5": md5, "ceiling_le30_share": sum(1 for v in vb if v <= 30) / len(recs)}
    if cfg:
        census = cfg["tac_goal_stats"]["census"]
        small = sorted(k for k, v in census.items() if v["pos"] < 200)
        label_facts["goal_tokens_under_200_positive_records"] = small
        label_facts["n_goal_tokens_under_200"] = len(small)
        check("config: tokens under the 200-positive scoreability floor", len(small), cfg["tac_goal_stats"]["n_under_scoreability_floor"], 0.5, "config.json tac_goal_stats")
except Exception as e:  # noqa: BLE001
    unreadable.append(f"labels/config: {type(e).__name__}: {e}")

n_ok = sum(r["ok"] for r in rows)
summary = {"n_checks": len(rows), "n_ok": int(n_ok), "n_fail": int(len(rows) - n_ok), "unreadable_files": unreadable,
           "C_UNREAD_zero": len(unreadable) == 0, "label_facts": label_facts, "rows": rows}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"checks {len(rows)}  ok {n_ok}  fail {len(rows) - n_ok}  unreadable {len(unreadable)}")
for r in rows:
    if not r["ok"]:
        print("FAIL", r["name"], "got", r["got"], "claimed", r["claimed"], "tol", r["tol"], "|", r["source"])
for u in unreadable:
    print("UNREADABLE", u)
sys.exit(0 if (len(rows) - n_ok) == 0 and not unreadable else 1)
