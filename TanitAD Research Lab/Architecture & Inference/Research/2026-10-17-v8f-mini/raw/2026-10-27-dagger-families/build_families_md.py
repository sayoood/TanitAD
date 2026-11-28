#!/usr/bin/env python3
"""Builds FAMILIES_DAG.md and DAG_FAMILIES_ROWS.json from the raw outputs. Nothing is typed by hand: every number in a table is read from a JSON.

Inputs (all under E:/Projects/TanitAD-artifacts/dagfam/ unless absolute):
  DAG_FAMILIES.json         dagfam_families.py     (LONGITUDINAL / LATERAL / TACTICAL plan-derived / DISTANCE KEEPING, T0 and T1-kin; DAG-DAG0 and, at T0, DAG-M2b)
  PAIRFAM_T0_DAG-DAG0_s*.json, PAIRFAM_T0_DAG-M2b_s*.json   the unchanged programme CLI (TACTICAL heads + goal + STRATEGIC at T0)
  DAG_T1CELLS.json          dagfam_t1cells.py      (R-TLAT / R-TLON reproduction + DAG-ORC, per-class lat F1, paired S-POS at T1-kin)
  E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json   (the decisive read; context + the beside blocks it carries)
"""
import hashlib
import json
import os
import sys

D = "E:/Projects/TanitAD-artifacts/dagfam"
ART = "E:/Projects/TanitAD-artifacts"
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
F = J(D + "/DAG_FAMILIES.json")
TC = J(D + "/DAG_T1CELLS.json")
READ = J(ART + "/v8f-dagger-t1/read/DAG_READ.json")
PF = {(c, s): J(f"{D}/PAIRFAM_T0_{c}_s{s}.json") for c in ("DAG-DAG0", "DAG-M2b") for s in (0, 1)}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


TIERS = {"T0": "T0 (teacher-forced own-code eval-141; box 2's standard eval, ages cycle)",
         "T1": "T1-kin (self-action OPEN loop: plain causal bucket, nav dead-reckoned, scene logged, replan every 10 steps)"}
ROWS = []          # machine-readable: one dict per table cell


def add(contrast, family, metric, unit, tier, seed, d, lo, hi, sep, direction, mean_a, mean_b, n_w, n_e, src, note=""):
    """direction: 'lower' / 'higher' = which way is better for A - B; None = the instrument gives the metric no direction label."""
    verdict = ""
    if sep:
        if direction is None:
            verdict = "separated (no direction label)"
        else:
            better = (d < 0) if direction == "lower" else (d > 0)
            verdict = "separated, A better" if better else "separated, A WORSE"
    else:
        verdict = "not separated"
    ROWS.append({"contrast": contrast, "family": family, "metric": metric, "unit": unit, "tier": tier, "seed": seed, "delta": d, "lo": lo, "hi": hi, "separated": bool(sep),
                 "direction_better": direction, "verdict": verdict, "mean_a": mean_a, "mean_b": mean_b, "n_windows": n_w, "n_episodes": n_e, "source": src, "note": note})


# ------------------------------------------------------------------------------------------------------------------------ collect
def collect_families(contrast, tier, key_blk, seeds=(0, 1)):
    for s in seeds:
        blk = key_blk(s)
        for k, r in blk["families_paired_A_minus_B"].items():
            fam_nm = {"LONGITUDINAL": "LONGITUDINAL", "LATERAL": "LATERAL", "TACTICAL (plan-derived)": "TACTICAL"}[r["family"]]
            add(contrast, fam_nm, r["label"] + (" (plan-derived)" if fam_nm == "TACTICAL" else ""), r["unit"], tier, s, r["mean"], r["lo"], r["hi"], r["separated"],
                "lower" if r["lower_is_better"] else "higher", r["mean_a_on_pairs"], r["mean_b_on_pairs"], r["n_windows"], r["n_episodes"], f"DAG_FAMILIES.json:{tier}/{s}/{k}")
        dk = blk["distance_keeping"]["pair"]
        arms = blk["distance_keeping"]["arms"]
        la, lb = [x.strip() for x in dk["a_minus_b"].split(" - ")]
        for k, lab, unit in (("headway_min_m", "min headway", "m"), ("time_gap_min_s", "min time gap", "s"), ("min_ttc_s", "min TTC (censored at a 30 s cap)", "s")):
            r = dk["dk"][k]
            add(contrast, "LONGITUDINAL (distance keeping, K=30, lead windows)", lab, unit, tier, s, r["delta"], r["lo"], r["hi"], r["separated"], None,
                arms[la]["metrics"][k]["mean"], arms[lb]["metrics"][k]["mean"], r["n_used"], r["n_episodes"], f"DAG_FAMILIES.json:{tier}/{s}/distance_keeping/pair/dk/{k}",
                note="arm means are over each arm's own windows (n differs slightly from the paired n)")
        for k, lab, unit in (("headway_min_m", "absolute headway error vs the human", "m"), ("time_gap_min_s", "absolute time-gap error vs the human", "s"), ("min_ttc_s", "absolute min-TTC error vs the human", "s")):
            r = dk["abs_error_vs_human"][k]
            add(contrast, "LONGITUDINAL (distance keeping, K=30, lead windows)", lab, unit, tier, s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", None, None, r["n_used"], r["n_episodes"],
                f"DAG_FAMILIES.json:{tier}/{s}/distance_keeping/pair/abs_error_vs_human/{k}")
        for k, lab in (("at_risk", "LEAD-VIOL, at risk (violating share)"), ("all_lead_windows", "LEAD-VIOL, all lead windows")):
            r = dk["lead_viol"][k]
            va = arms[la]["lead_viol"]["share_of_at_risk" if k == "at_risk" else "share"]
            vb = arms[lb]["lead_viol"]["share_of_at_risk" if k == "at_risk" else "share"]
            add(contrast, "LONGITUDINAL (distance keeping, K=30, lead windows)", lab, "share", tier, s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", va, vb,
                r["n_windows"], r["n_episodes"], f"DAG_FAMILIES.json:{tier}/{s}/distance_keeping/pair/lead_viol/{k}",
                note=f"violations A/B among {r.get('n_at_risk', r.get('n_lead_windows'))}: {r['n_viol_a']}/{r['n_viol_b']}")
        r = dk["lead_viol"]["progress"]
        add(contrast, "LONGITUDINAL (distance keeping, K=30, lead windows)", "lead-progress ratio", "ratio", tier, s, r["delta"], r["lo"], r["hi"], r["separated"], None,
            r["lead_progress_ratio_a"], r["lead_progress_ratio_b"], r["n_windows"], r["n_episodes"], f"DAG_FAMILIES.json:{tier}/{s}/distance_keeping/pair/lead_viol/progress")


collect_families("DAG-DAG0", "T0", lambda s: F["T0"][f"s{s}"])
collect_families("DAG-DAG0", "T1", lambda s: F["T1"][f"s{s}"])
collect_families("DAG-M2b", "T0", lambda s: F["T0_vs_M2b"][f"s{s}"])
for c, key in (("DAG-DAG0", lambda s: F["T0"][f"s{s}"]), ("DAG-M2b", lambda s: F["T0_vs_M2b"][f"s{s}"])):
    for s in (0, 1):
        r = key(s)["ade6_paired_A_minus_B"]
        add(c, "ADE6 (reference row)", "ADE6, 6 s", "m", "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", r["mean_a"], r["mean_b"], r["n_windows"], r["n_episodes"],
            f"DAG_FAMILIES.json:{'T0' if c == 'DAG-DAG0' else 'T0_vs_M2b'}/{s}/ade6_paired_A_minus_B")
for s in (0, 1):
    r = F["T1"][f"s{s}"]["ade6_paired_A_minus_B"]
    add("DAG-DAG0", "ADE6 (reference row)", "ADE6, 6 s", "m", "T1", s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", r["mean_a"], r["mean_b"], r["n_windows"], r["n_episodes"],
        f"DAG_FAMILIES.json:T1/{s}/ade6_paired_A_minus_B")

# PAIRFAM (T0): TACTICAL heads + goal, STRATEGIC
for (c, s), P in PF.items():
    contrast = c
    src = f"PAIRFAM_T0_{c}_s{s}.json"
    th = P["tactical"]["heads"]
    for hd, lab in (("lateral", "lat head"), ("longitudinal", "lon head")):
        for m, mlab, d in (("macro_f1", "macro-F1", "higher"), ("accuracy", "accuracy", "higher")):
            r = th[hd][m]
            add(contrast, "TACTICAL", f"{lab} {mlab} (window argmax vs v9 label)", "", "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], d, r["a_point"], r["b_point"], r["n_windows"], r["n_episodes"],
                f"{src}:tactical/heads/{hd}/{m}", note=r["n_reason"])
    for m, mlab in (("macro_f1", "goal head macro-F1 (22 tokens)"), ("token_accuracy", "goal head micro token accuracy")):
        r = th["goal"][m]
        add(contrast, "TACTICAL", mlab, "", "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], "higher", r["a_point"], r["b_point"], r["n_windows"], r["n_episodes"], f"{src}:tactical/heads/goal/{m}",
            note=r["n_reason"])
    pd = P["tactical"]["plan_derived"]
    for k, lab in (("maneuver_5way_collapsed", "5-way collapsed manoeuvre accuracy (plan-derived)"), ("lateral_decision", "lateral decision accuracy (plan-derived, tool's own)"),
                   ("longitudinal_decision", "longitudinal decision accuracy (plan-derived, tool's own)")):
        r = pd[k]["accuracy"]
        add(contrast, "TACTICAL", lab, "", "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], "higher", r.get("a_point"), r.get("b_point"), r["n_windows"], r["n_episodes"], f"{src}:tactical/plan_derived/{k}/accuracy")
    gs = P["tactical"]["goal_setting"]
    for k, lab, unit, d in (("goal_point_error_m", "goal-point error (FDE6)", "m", "lower"), ("goal_bearing_mae_deg", "goal bearing MAE", "deg", "lower")):
        r = gs[k]
        add(contrast, "TACTICAL", lab, unit, "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], d, r.get("a_point"), r.get("b_point"), r["n_windows"], r["n_episodes"], f"{src}:tactical/goal_setting/{k}")
    for arc in ("30", "50", "80"):
        r = P["strategic"]["position_error_m"][arc]
        add(contrast, "STRATEGIC", f"S checkpoint position error at arc {arc} m", "m", "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", r["a_point"], r["b_point"], r["n_windows"], r["n_episodes"],
            f"{src}:strategic/position_error_m/{arc}", note=r["n_reason"])
        r = P["strategic"]["heading_error_deg"][arc]
        add(contrast, "STRATEGIC", f"S checkpoint heading error at arc {arc} m", "deg", "T0", s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", r.get("a_point"), r.get("b_point"), r["n_windows"], r["n_episodes"],
            f"{src}:strategic/heading_error_deg/{arc}")

# T1-kin cells: head F1 (R-TLAT/R-TLON), per-class lat F1, paired S-POS
for s in (0, 1):
    P = TC["per_seed"][f"s{s}"]
    for hd, lab in (("lat", "R-TLAT: lat macro-F1, ticks>=1, matched cells"), ("lon", "R-TLON: lon macro-F1, ticks>=1, matched cells")):
        for nm, contrast in (("DAG_minus_DAG0", "DAG-DAG0"), ("DAG_minus_ORC", "DAG-ORC")):
            r = P[f"R-T{hd.upper()}"][nm]
            add(contrast, "TACTICAL", lab, "", "T1", s, r["delta"], r["lo"], r["hi"], r["separated"], "higher", r["f1_a"], r["f1_b"], 2399, r["n_episodes"], f"DAG_T1CELLS.json:per_seed/s{s}/R-T{hd.upper()}/{nm}",
                note=f"matched cell pairs {r['n_cells_matched_pairs']}")
    for cname in ("TURN_L", "TURN_R", "LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R"):
        for blk, contrast in (("per_class_lat_F1_DAG_minus_DAG0_matched", "DAG-DAG0"), ("per_class_lat_F1_DAG_minus_ORC_matched", "DAG-ORC")):
            r = P[blk][cname]
            if "delta" not in r:
                continue
            add(contrast, "TACTICAL (per-class)", f"lat head F1 of {cname}, ticks>=1, matched cells", "", "T1", s, r["delta"], r["lo"], r["hi"], r["separated"], "higher", r["f1_a"], r["f1_b"], 2399, r["n_episodes"],
                f"DAG_T1CELLS.json:per_seed/s{s}/{blk}/{cname}", note=f"support {r['support_cells_matched']} matched cells in {r['episodes_with_support']} episodes; draws with the class absent {r['n_draws_class_absent']}")
    for arc in (30, 50, 80):
        for k, lab, unit in ((f"S-POS@{arc}", f"S checkpoint position error at arc {arc} m (ticks>=1, mean over scored ticks)", "m"), (f"S-HEAD@{arc}", f"S checkpoint heading error at arc {arc} m (ticks>=1)", "deg")):
            r = P["STRATEGIC_paired_DAG_minus_DAG0"][k]
            add("DAG-DAG0", "STRATEGIC", lab, unit, "T1", s, r["delta"], r["lo"], r["hi"], r["separated"], "lower", r["mean_a"], r["mean_b"], r["n_windows"], r["n_episodes"],
                f"DAG_T1CELLS.json:per_seed/s{s}/STRATEGIC_paired_DAG_minus_DAG0/{k}", note="windows scored in both runs at ticks>=1; computed from the T1-HEADS reader's own S error arrays (no pre-existing paired reader)")

json.dump(ROWS, open(D + "/DAG_FAMILIES_ROWS.json", "w", encoding="utf-8", newline="\n"), indent=1)


# ------------------------------------------------------------------------------------------------------------------------ render
def sg(x, dp=4):
    return "n/a" if x is None else f"{x:+.{dp}f}"


def lv(x, dp=4):
    return "-" if x is None else f"{x:.{dp}f}"


def dp_for(r):
    return 6 if abs(r["delta"]) < 1e-3 and r["unit"] in ("1/m",) else 4


def cell(r):
    dp = dp_for(r)
    return f"{sg(r['delta'], dp)} [{sg(r['lo'], dp)}, {sg(r['hi'], dp)}]"


def table(contrast, family_prefix, extra_cols=True):
    rows = [r for r in ROWS if r["contrast"] == contrast and r["family"].startswith(family_prefix)]
    metrics = []
    for r in rows:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
    out = ["| metric (A - B) | unit | tier | seed | A level | B level | A - B [95 % interval] | verdict | n windows / episodes |", "|---|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        for tier in ("T0", "T1"):
            for s in (0, 1):
                rr = [r for r in rows if r["metric"] == m and r["tier"] == tier and r["seed"] == s]
                if not rr:
                    continue
                r = rr[0]
                dp = dp_for(r)
                unit = r["unit"] if r["unit"] else "-"
                n = f"{r['n_windows']:,} / {r['n_episodes']}"
                if (tier == "T1" and r["family"].startswith("TACTICAL")) or m.startswith("LEAD-VIOL"):
                    n += f" ({r['note']})" if r["note"] else ""
                out.append(f"| {m} | {unit} | {'T1-kin' if tier == 'T1' else 'T0'} | s{s} | {lv(r['mean_a'], dp)} | {lv(r['mean_b'], dp)} | {cell(r)} | {r['verdict']} | {n} |")
    return chr(10).join(out)


def worse_list(contrast, fams=None):
    out = []
    for r in ROWS:
        if r["contrast"] != contrast or not r["separated"] or r["direction_better"] is None:
            continue
        better = (r["delta"] < 0) if r["direction_better"] == "lower" else (r["delta"] > 0)
        if not better:
            out.append(r)
    return out


def sentence_rows(rs):
    return "; ".join(f"{r['metric']} [{r['family']}] {'T1-kin' if r['tier'] == 'T1' else 'T0'} s{r['seed']} {cell(r)}" for r in rs) if rs else "none"


L = []
w = L.append
CTL = TC["controls"]
FC = F["controls"]

w("# FAMILIES_DAG: the pre-registered 'beside' rows of the DAgger arm (DAG vs DAG-0), four metric families at both tiers")
w("")
w("*Draft for the Master Mind. Nothing was staged, committed or pushed by it. Zero training, zero new rollouts: every number below is a re-reading of banked artifacts "
  "with the programme's own instruments.*")
w("")
w(f"* **Pre-registration:** `Project Steering/PREREG_V8F_DAGGER.md` section 5, *Beside (never decisive)*. The decisive read is `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json` "
  f"(OUTCOME **{READ['OUTCOME']['OUTCOME']}**, all {len(READ['controls_PASS'])} of its controls PASS: {all(READ['controls_PASS'].values())}); this document decides nothing and changes no outcome. DAG ran at p = 0.1 (`--dagger-p 0.1`, argv.json).")
w("* **Evidence class:** MEASURED (ours). Artifact paths are in the tables' `source` keys (`DAG_FAMILIES_ROWS.json`) and in section 9.")
w(f"* **Estimator of every interval:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = {F['n_boot']:,}, seed 0, over {F['n_episodes']} episodes, {F['n_windows']:,} windows (eval-141), "
  "A minus B per seed on the same windows. It answers only *would another draw of EPISODES say this?* The training-seed question is answered only by the two seeds (H-ESTIM-SEED-1); "
  "the planner is deterministic (PREREG_V8F_PGT), so no inference-seed question arises. A separated interval from one training seed is necessary, not sufficient.")
w("* **Tiers.** " + TIERS["T0"] + ". " + TIERS["T1"] + ". T1-kin is a **self-action OPEN loop**; it is never 'closed loop' and nothing here is a driving claim. "
  "The eval JSON stamps T0 as `T0-open-loop-own-held-outputs (trunk-probe grid)`; the paired-families tool stamps its output `T1-class open loop (own held outputs; no recorded future action)`. "
  "These are three labels for one open-loop scoring and I use the PREREG's names (the same unreconciled labelling as RESULT_V8F_ORECOVER section 3).")
w("* **STRATEGIC is NOT paused for v8F** (PI ruling 2026-10-17, R73): it is reported in section 2.4 and 4.")
w("* **Recipe:** `orc_families.py` of O-RECOVER (raw/2026-10-26-orecover-result/) re-parametrised (ORC -> DAG, ORC-0 -> DAG-0), run on the DAG chain's own tool tree "
  "`E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a` (the instruments `v8f_beta_eval.py`, `ci.py`, `lead_metrics.py`, `four_families.py`, `v8f_paired_families.py`, `v8f_dk_read.py` are byte-equal to "
  "O-RECOVER's 07dcb513 tree modulo CRLF, and to the mirror tip 7d29172b; hashes in section 9).")
w("* **Column key.** A is the first-named arm (for `DAG-DAG0`: DAG; for `DAG-ORC`: DAG; for `DAG-M2b`: DAG). 'A better' / 'A WORSE' is printed only where the metric has a direction (error: lower is "
  "better; agreement, F1: higher; LEAD-VIOL share: lower). Headway, time gap, TTC and lead-progress carry no direction label from the instrument (more distance kept is not automatically better).")
w("")

# ---- 0. headline
w("## 0. Headline, per family (computed from the tables)")
w("")
w("Separated rows where DAG is WORSE than the comparator (a direction-labelled metric, interval excluding 0, point on the worse side):")
w("")
for c in ("DAG-DAG0", "DAG-M2b", "DAG-ORC"):
    wl = worse_list(c)
    w(f"* **{c}:** " + sentence_rows(wl))
w("")
w("Context for the list: the `ADE6, 6 s` T0 s1 row (+0.0483 m) is the decisive read's R-T0 guard, which that read PASSED (bar: point <= +0.0604 m, F/2); it appears here only because its interval excludes 0. "
  "No row in the DAG - DAG-0 list is a guard of the decisive read other than that one, and none changes the OUTCOME (this document decides nothing). "
  "The T1-kin LEAD-VIOL and absolute-TTC rows are the ones that cut against the T1-kin gains: the lead follows its LOGGED future and does not react to the ego (an open-loop property of the rig, as RESULT_V8F_ORECOVER section 3 states for these runs), and the anti-echo block (1.1) does not admit either arm's longitudinal numbers as a capability claim.")
w("")


def fam_summary(contrast, fam_prefix):
    rs = [r for r in ROWS if r["contrast"] == contrast and r["family"].startswith(fam_prefix) and r["metric"] != "ADE6, 6 s"]
    sep = [r for r in rs if r["separated"]]
    good = [r for r in sep if r["direction_better"] is not None and ((r["delta"] < 0) == (r["direction_better"] == "lower"))]
    bad = [r for r in sep if r["direction_better"] is not None and r not in good]
    nolab = [r for r in sep if r["direction_better"] is None]
    return len(rs), len(sep), len(good), len(bad), len(nolab)


w("| contrast | family | cells (tier x seed x metric) | separated | separated, DAG better | separated, DAG WORSE | separated, no direction label |")
w("|---|---|---|---|---|---|---|")
for c in ("DAG-DAG0", "DAG-M2b"):
    for fam_prefix, lab in (("LONGITUDINAL", "LONGITUDINAL (incl. distance keeping)"), ("LATERAL", "LATERAL"), ("TACTICAL", "TACTICAL"), ("STRATEGIC", "STRATEGIC")):
        n, a, b, c_, d = fam_summary(c, fam_prefix)
        w(f"| {c} | {lab} | {n} | {a} | {b} | {c_} | {d} |")
w("")

# ---- 1. controls
w("## 1. Controls (each must hold; a failed control would make this document unusable)")
w("")
ok = lambda b: "PASS" if b else "**FAIL**"  # noqa: E731
ident_all = all(all(F[t][f"s{s}"]["identity_all_exactly_zero"].values()) for t in ("T0", "T1") for s in (0, 1)) and all(
    all(F["T0_vs_M2b"][f"s{s}"]["identity_all_exactly_zero"].values()) for s in (0, 1))
dk_ident = all(all(v[k]["delta"] == 0.0 and v[k]["lo"] == 0.0 and v[k]["hi"] == 0.0 for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s")) and v["lead_viol"]["delta"] == 0.0
               for blk in (F["T0"], F["T1"], F["T0_vs_M2b"]) for s in (0, 1) for v in [blk[f"s{s}"]["distance_keeping"]["identity_control"]])
rep_ok = all(FC["reproduces_reader_ade6"][f"s{s}"]["PASS"] for s in (0, 1))
grid_ok = all(all(v for k, v in c.items() if isinstance(v, bool)) and all(
    x["eid_equal_grid"] and x["ws_times2_equals_grid_frame"] and x["v0_max_abs_diff_vs_grid"] == 0.0 and x["g_max_abs_diff_vs_grid_gt_m"] < 1e-4 and x["n_windows"] == 2399
    for kk, x in c.items() if isinstance(x, dict)) for kk0, c in FC.items() if kk0.startswith("T1_grid"))
pf_ok = all(P["controls_ok"] is True and P["n_windows"] == 2399 and P["n_episodes"] == 141 and P["ages"] == "cycle" for P in PF.values())
# cross-instrument: the plan-derived decision agreement from v8f_paired_families.py equals family_windows' tac rows


def cross(contrast, blk):
    res = []
    for s in (0, 1):
        P = PF[(contrast, s)]
        for a, b in (("lateral_decision", "tac_lat_acc"), ("longitudinal_decision", "tac_lon_acc")):
            x, y = P["tactical"]["plan_derived"][a]["accuracy"], blk(s)["families_paired_A_minus_B"][b]
            res.append((x["delta"], x["lo"], x["hi"], x["separated"]) == (y["mean"], y["lo"], y["hi"], y["separated"]))
    return all(res), len(res)


x1, n1 = cross("DAG-DAG0", lambda s: F["T0"][f"s{s}"])
x2, n2 = cross("DAG-M2b", lambda s: F["T0_vs_M2b"][f"s{s}"])
hf = CTL["reproduces_reader_head_f1"]
hf_ok = all(v["PASS"] for v in hf.values())
cb = CTL["classboot_reproduces_standard_on_macro"]
cb_ok = all(v["PASS"] for v in cb.values())
idh = all(CTL["identity_head_f1"].values())
idc = all(CTL["identity_class_f1"].values())
ids_ = all(CTL["identity_spos"].values())
sp_rep = all(v["PASS"] for s in ("s0", "s1") for v in CTL["reproduces_reader_spos_vs_ctrv"][s].values())
cl_lv = all(CTL["class_levels_macro_equals_run_macro"].values())
x1ok = all(TC["per_seed"][f"s{s}"]["x1_rebuild"][a]["max_cl_m"] == 0.0 and TC["per_seed"][f"s{s}"]["x1_rebuild"][a]["max_pose_tick_f64"] == 0.0 for s in (0, 1) for a in ("DAG", "DAG0", "ORC"))
m2b_ok = all(F["T0_vs_M2b"][f"s{s}"]["m2b_ade6_mean_vs_registry_s17"]["equal_at_4dp"] for s in (0, 1))
lead_ok = F["inputs"]["lead_block_sha256"] == "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"
CONTROLS = [
    ("C1 identity, families", "every paired family row run on A against A reads exactly +0 [0, 0] (T0 and T1-kin DAG-DAG0, T0 DAG-M2b, both seeds, 9 rows each)", ident_all),
    ("C2 identity, distance keeping", "`read_dk`'s own identity control (headway, time gap, TTC, LEAD-VIOL) reads +0 [0, 0] in all 6 blocks", dk_ident),
    ("C3 reproduction of the reader's R-T0 / R-ADE", "this driver's ADE6 DAG - DAG0 equals `DAG_READ.json` rows R-T0 (T0) and R-ADE (T1-kin), delta, lo, hi, separated, both seeds: "
     + "; ".join(f"s{s} T0 {sg(FC['reproduces_reader_ade6'][f's{s}']['T0']['here']['delta'])}, T1-kin {sg(FC['reproduces_reader_ade6'][f's{s}']['T1']['here']['delta'])}" for s in (0, 1)), rep_ok),
    ("C4 T1-kin grid", "the dump's windows, episode ids, `2*ws` frames and `v0` equal the T0 grid's for DAG and DAG-0, both seeds; dump GT vs grid GT max difference "
     + f"{max(x['g_max_abs_diff_vs_grid_gt_m'] for s in (0, 1) for k, x in FC[f'T1_grid_s{s}'].items() if isinstance(x, dict)):.1e} m (float32 round trip)", grid_ok),
    ("C5 cross-instrument", f"the plan-derived lateral / longitudinal decision agreement from `v8f_paired_families.py` equals the `family_windows` rows to delta, lo, hi, separated ({n1} of {n1} for DAG-DAG0, {n2} of {n2} for DAG-M2b)", x1 and x2),
    ("C6 the paired-families CLI's own controls", "`controls_ok` (A-vs-A identity on 45 cells; each arm's point values reproduce its own eval JSON, 119 values) true for all 4 runs; 2,399 windows, 141 episodes, ages cycle", pf_ok),
    ("C7 reproduction of the reader's R-TLAT / R-TLON", "head macro-F1 DAG - DAG-0, ticks>=1, matched cells: delta, lo, hi, separated, matched-pair count and both F1 levels equal `DAG_READ.json` (4 rows)", hf_ok),
    ("C8 identity, head F1 / class F1 / S-POS", "(a, a) reads exactly +0 [0, 0] on the head-F1 rows (DAG-DAG0 and DAG-ORC), the per-class F1 routine, and the paired S-POS rows", idh and idc and ids_),
    ("C9 class-F1 bootstrap vs the standard estimator", "the routine used for per-class F1 reproduces `taniteval.ci` on macro-F1 exactly (delta, lo, hi, separated) for both heads and seeds", cb_ok),
    ("C10 per-class levels", "the mean of the per-class F1 levels equals the run's macro-F1 (the macro-F1 is the nan-mean of them)", cl_lv),
    ("C11 S-POS vs CTRV", "the per-run S - CTRV rows (ticks>=1) recomputed here equal `DAG_READ.json` beside.per_run_t1heads for DAG and DAG-0 (delta, lo, hi, separated; 12 rows)", sp_rep),
    ("C12 executed-track rebuild", "the rebuilt executed track equals the dumped `cl` (max 0.0 m) for DAG, DAG-0 and ORC, both seeds (the T1-HEADS X1 check)", x1ok),
    ("C13 M2b comparator", "the banked M2b eval's ADE6 mean equals MODEL_REGISTRY section 17 at 4 dp: "
     + ", ".join(f"s{s} {F['T0_vs_M2b'][f's{s}']['m2b_ade6_mean_vs_registry_s17']['here']:.4f} vs {F['T0_vs_M2b'][f's{s}']['m2b_ade6_mean_vs_registry_s17']['registry']}" for s in (0, 1)), m2b_ok),
    ("C14 inputs", f"lead block sha256 `{F['inputs']['lead_block_sha256'][:16]}...` is the registered one; grid `{F['inputs']['grid_sha256'][:12]}...`, kin `{F['inputs']['kin_sha256'][:12]}...`, join `{F['inputs']['join_sha256'][:12]}...`", lead_ok),
]
w("| control | what it asserts | status |")
w("|---|---|---|")
for nm, what, st in CONTROLS:
    w(f"| {nm} | {what} | {ok(st)} |")
ALL_OK = all(st for _, _, st in CONTROLS)
w("")
w(f"**All controls: {'PASS' if ALL_OK else 'FAIL'}.**")
w("")
w("*Not a control, disclosed:* `DAG_READ.json` itself was produced by `dag_read.py` (sha256 `" + READ["reader_sha256"][:16] + "...`) from the same artifacts; C3 and C7 tie this document to it.")
w("")

# ---- anti-echo
w("### 1.1 Anti-echo caveat on every LONGITUDINAL number (binding for how to read section 2.1)")
w("")
rows_ae = []
for lab, p in (("DAG s0", "t1v8f_DAG_s0_m10_causal_heads.json"), ("DAG-0 s0", "t1v8f_DAG0_s0_m10_causal_heads.json"), ("DAG s1", "t1v8f_DAG_s1_m10_causal_heads.json"), ("DAG-0 s1", "t1v8f_DAG0_s1_m10_causal_heads.json")):
    a = J(f"{ART}/v8f-dagger-t1/{p}")["arms"]["cl"]["four_families"]["longitudinal"]["anti_echo"]
    rows_ae.append((lab, "T1-kin", a["status"], a["longitudinal_claim_admissible"], a["flagged"], a["summary"], p))
for lab, p in (("DAG s0", "v8f-dagger/M2b_s0_dag/eval_M2b-dag_s0.json"), ("DAG-0 s0", "v8f-dagger/M2b_s0_dag0/eval_M2b-dag0_s0.json"), ("DAG s1", "v8f-dagger/M2b_s1_dag/eval_M2b-dag_s1.json"),
               ("DAG-0 s1", "v8f-dagger/M2b_s1_dag0/eval_M2b-dag0_s1.json")):
    a = J(f"{ART}/{p}")["raw"]["families"]["longitudinal"]["anti_echo"]
    rows_ae.append((lab, "T0", a["status"], None, None, "no ego speed at t0 in the window dict (condition undischarged)", p))
w("| run | tier | anti_echo.status | longitudinal_claim_admissible | flagged | summary | source |")
w("|---|---|---|---|---|---|---|")
for r in rows_ae:
    w(f"| {r[0]} | {r[1]} | {r[2]} | {r[3] if r[3] is not None else 'n/a'} | {r[4] if r[4] is not None else 'n/a'} | {r[5]} | `{r[6]}` |")
w("")
_m2b_ae = [J(f"{D}/ref/m2b/eval_M2b_s{s}.json")["raw"]["families"]["longitudinal"]["anti_echo"]["status"] for s in (0, 1)]
w(f"The M2b banked eval (`dagfam/ref/m2b/eval_M2b_s0.json`, `_s1.json`) carries the T0 anti-echo status {_m2b_ae[0]} / {_m2b_ae[1]}. A LONGITUDINAL contrast here describes what the lever changed on these windows; it is not a statement that either arm sets the right speed.")
w("")

# ---- 2. DAG - DAG0
w("## 2. DAG - DAG-0 (the decisive pair), per family, both tiers, both seeds")
w("")
w("A = DAG, B = DAG-0 (same init M2b, steps, data, LR path and seed; the lever is the route-paired own-state replacement at p = 0.1). 'level' columns are each arm's mean on the paired windows.")
w("")
w("### 2.0 ADE6 reference (the guard rows R-T0 and R-ADE of the decisive read, reproduced here: control C3)")
w("")
w("| row | tier | seed | DAG level | DAG-0 level | DAG - DAG-0 [95 % interval] | verdict | n windows / episodes |")
w("|---|---|---|---|---|---|---|---|")
for tier, lab in (("T0", "R-T0: T0 own-code ADE6"), ("T1", "R-ADE: T1-kin plain ADE6")):
    for s in (0, 1):
        r = [x for x in ROWS if x["contrast"] == "DAG-DAG0" and x["metric"] == "ADE6, 6 s" and x["tier"] == tier and x["seed"] == s][0]
        w(f"| {lab}, m | {'T1-kin' if tier == 'T1' else 'T0'} | s{s} | {lv(r['mean_a'])} | {lv(r['mean_b'])} | {cell(r)} | {r['verdict']} | {r['n_windows']:,} / {r['n_episodes']} |")
w("")
w("### 2.1 LONGITUDINAL")
w("")
w(table("DAG-DAG0", "LONGITUDINAL"))
w("")
w("### 2.2 LATERAL")
w("")
w(table("DAG-DAG0", "LATERAL"))
w("")
w("### 2.3 TACTICAL")
w("")
w("Plan-derived decisions are scored on both tiers (`family_windows`). Head and goal F1 at T0 come from the unchanged CLI `v8f_paired_families.py`; at T1-kin the head F1 rows are the reader's R-TLAT / R-TLON (re-derived, control C7) and the per-class rows of section 5. "
  "The goal-point rows are T0 only.")
w("")
w(table("DAG-DAG0", "TACTICAL"))
w("")
w("### 2.4 STRATEGIC (NOT paused)")
w("")
w("T0: S's checkpoint position / heading error at 30 / 50 / 80 m against the v9 route checkpoints (`v8f_paired_families.py`). T1-kin: the paired rows are built from the T1-HEADS reader's own S error arrays (control C11 ties them to the reader); "
  "the within-run S - CTRV (constant-turn-rate-and-velocity) contrast the reader carries is in section 2.5.")
w("")
w(table("DAG-DAG0", "STRATEGIC"))
w("")
w("### 2.5 STRATEGIC at T1-kin against CTRV, per run (the reader's unpaired view, recomputed; negative = S nearer the route checkpoint than CTRV)")
w("")
w("| run | arc | S - CTRV, ticks>=1 [95 % interval] | verdict | n windows |")
w("|---|---|---|---|---|")
for s in (0, 1):
    for lab in ("DAG", "DAG0"):
        for arc in (30, 50, 80):
            r = TC["per_seed"][f"s{s}"]["STRATEGIC_per_run_S_minus_CTRV"][lab][f"S-POS@{arc}"]
            v = "not separated" if not r["separated"] else ("separated, S nearer than CTRV" if r["delta"] < 0 else "separated, S farther than CTRV")
            w(f"| {'DAG-0' if lab == 'DAG0' else lab} s{s} | {arc} m | {sg(r['delta'])} [{sg(r['lo'])}, {sg(r['hi'])}] | {v} | {r['n_windows']:,} |")
w("")

# ---- 3. DAG - ORC
w("## 3. DAG - ORC on R-TLAT / R-TLON (T1-kin): on-policy states against the law")
w("")
w("⚠️ **The arms differ in more than the lever.** ORC is the O-RECOVER ego-perturbation arm (perturbation law, p = 0.3, trained on box 2 by another build); DAG is own-state replacement at p = 0.1. "
  "The two also differ in dose, in the fine-tune build and in what the replaced states are. This is the PREREG's 'scientific comparison, never decisive'. Same estimator, matched cells (scored in both runs), ticks >= 1.")
w("")
w("| row | seed | tier | DAG level | ORC level | DAG - ORC [95 % interval] | verdict | matched cell pairs | own-cells version beside |")
w("|---|---|---|---|---|---|---|---|---|")
for hd, lab in (("lat", "R-TLAT (lat macro-F1)"), ("lon", "R-TLON (lon macro-F1)")):
    for s in (0, 1):
        r = TC["per_seed"][f"s{s}"][f"R-T{hd.upper()}"]["DAG_minus_ORC"]
        v = "not separated" if not r["separated"] else ("separated, DAG better" if r["delta"] > 0 else "separated, DAG WORSE")
        o = r["own_cells_beside"]
        w(f"| {lab} | s{s} | T1-kin | {lv(r['f1_a'])} | {lv(r['f1_b'])} | {sg(r['delta'])} [{sg(r['lo'])}, {sg(r['hi'])}] | {v} | {r['n_cells_matched_pairs']:,} | {sg(o['delta'])} [{sg(o['lo'])}, {sg(o['hi'])}], {'separated' if o['separated'] else 'not separated'} |")
w("")
w("For reference, the same rows against DAG-0 (the decisive rows, reproduced here by control C7):")
w("")
w("| row | seed | DAG level | DAG-0 level | DAG - DAG-0 [95 % interval] | verdict | matched cell pairs |")
w("|---|---|---|---|---|---|---|")
for hd, lab in (("lat", "R-TLAT"), ("lon", "R-TLON")):
    for s in (0, 1):
        r = TC["per_seed"][f"s{s}"][f"R-T{hd.upper()}"]["DAG_minus_DAG0"]
        v = "not separated" if not r["separated"] else ("separated, DAG better" if r["delta"] > 0 else "separated, DAG WORSE")
        w(f"| {lab} | s{s} | {lv(r['f1_a'])} | {lv(r['f1_b'])} | {sg(r['delta'])} [{sg(r['lo'])}, {sg(r['hi'])}] | {v} | {r['n_cells_matched_pairs']:,} |")
w("")

# ---- 4. DAG - M2b
w("## 4. DAG - M2b at T0 (what 3,000 more steps plus own-state replacement did to the base)")
w("")
w("A = DAG, B = M2b (the init checkpoint; its banked box-2 standard eval `raw/full/eval/eval_M2b_s{0,1}.npz` at the mirror tip 7d29172b, the source of MODEL_REGISTRY section 17: ADE6 2.1753 / 2.0546). "
  "⚠️ DAG is M2b plus a 3,000-step fine-tune at lr 1e-4; **the fine-tune alone is DAG-0's job** (section 2). This contrast mixes the lever with 3,000 more steps and a changed LR schedule, and is T0 only "
  "(the M2b T1-kin dumps are not part of this package). Its sign across seeds is the information; each arm is one training seed.")
w("")
w("### 4.1 ADE6 (reference)")
w("")
w("| tier | seed | DAG level | M2b level | DAG - M2b [95 % interval] | verdict | n |")
w("|---|---|---|---|---|---|---|")
for s in (0, 1):
    r = [x for x in ROWS if x["contrast"] == "DAG-M2b" and x["metric"] == "ADE6, 6 s" and x["seed"] == s][0]
    w(f"| T0 | s{s} | {lv(r['mean_a'])} | {lv(r['mean_b'])} | {cell(r)} | {r['verdict']} | {r['n_windows']:,} / {r['n_episodes']} |")
w("")
for i, (fp, lab) in enumerate((("LONGITUDINAL", "4.2 LONGITUDINAL"), ("LATERAL", "4.3 LATERAL"), ("TACTICAL", "4.4 TACTICAL"), ("STRATEGIC", "4.5 STRATEGIC (NOT paused)"))):
    w(f"### {lab}")
    w("")
    w(table("DAG-M2b", fp))
    w("")

# ---- 5. per-class
w("## 5. Per-class lat-head F1 (TURN_L / TURN_R), T1-kin ticks >= 1")
w("")
w("The T1-HEADS drop sat in the lateral head. Per-class F1 = `taniteval.v8f_consistency.macro_f1(...)['per_class_f1']`, the quantity the macro-F1 is the nan-mean of, on the SAME encoded cells, the same matched-cell rule and slot layout as R-TLAT, "
  "and the same bootstrap draws (`ci._draws`, seed 0, B = 2,000; a draw in which the class has no labelled cell is excluded and counted). Class ids are v9 ids "
  "(`build_v9_labels.py:461`, LAT9): 0 LANE_KEEP, 1 TURN_L, 2 TURN_R, 3 LANE_CHANGE_L, 4 LANE_CHANGE_R. **Support is small for the turn classes, so read the n.**")
w("")
for blk, lab, ca, cb_ in (("per_class_lat_F1_DAG_minus_DAG0_matched", "5.1 DAG - DAG-0 (the decisive pair)", "DAG", "DAG-0"), ("per_class_lat_F1_DAG_minus_ORC_matched", "5.2 DAG - ORC (arms differ in more than the lever)", "DAG", "ORC")):
    w(f"### {lab}")
    w("")
    w(f"| class | seed | {ca} F1 | {cb_} F1 | {ca} - {cb_} [95 % interval] | verdict | support (matched cells) / episodes | draws with the class absent |")
    w("|---|---|---|---|---|---|---|---|")
    for cname in ("TURN_L", "TURN_R", "LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R"):
        for s in (0, 1):
            r = TC["per_seed"][f"s{s}"][blk][cname]
            if "delta" not in r:
                w(f"| {cname} | s{s} | - | - | not computed | {r['status']} | 0 | - |")
                continue
            v = "not separated" if not r["separated"] else ("separated, DAG better" if r["delta"] > 0 else "separated, DAG WORSE")
            w(f"| {cname} | s{s} | {lv(r['f1_a'])} | {lv(r['f1_b'])} | {sg(r['delta'])} [{sg(r['lo'])}, {sg(r['hi'])}] | {v} | {r['support_cells_matched']:,} / {r['episodes_with_support']} | {r['n_draws_class_absent']} |")
    w("")
w("### 5.3 Per-run per-class F1 levels (unpaired, each run's own scored cells): tick 0 (logged-state rollout step) against ticks >= 1 (the model's own executed states)")
w("")
w("| run | class | F1 tick 0 | support tick 0 | F1 ticks>=1 | support ticks>=1 |")
w("|---|---|---|---|---|---|")
for s in (0, 1):
    for lab in ("DAG", "DAG0", "ORC"):
        lvl = TC["per_seed"][f"s{s}"]["per_class_lat_F1_levels_unpaired"][lab]
        for ci_, cname in enumerate(TC["lat_class_names"]):
            if cname in ("LANE_CHANGE_L", "LANE_CHANGE_R"):
                continue
            a, b = lvl["tick0"], lvl["ticks_ge1"]
            w(f"| {'DAG-0' if lab == 'DAG0' else lab} s{s} | {cname} | {lv(a['per_class_f1'][ci_])} | {a['support'][ci_]} | {lv(b['per_class_f1'][ci_])} | {b['support'][ci_]} |")
w("")
w("(LANE_CHANGE_L / LANE_CHANGE_R have no labelled cell at either tick in this window set: the labelled lateral cells fall in 3 of the 5 classes. They are listed as 'not computed' in 5.1 and 5.2 and carry the same status in `DAG_T1CELLS.json`. Tick-0 supports 1,910 / 47 / 150 agree with the T0 paired-families tool's lateral-head `support` [1910, 47, 150, 0, 0] in `PAIRFAM_T0_DAG-DAG0_s0.json`.)")
w("")

# ---- 6. not computed
w("## 6. Not computed, with the reason and n")
w("")
w("| family | cell | tier | reason | n |")
w("|---|---|---|---|---|")
w("| TACTICAL | paired lat / lon head accuracy, goal-head F1, 5-way collapsed accuracy, goal-point error / bearing at T1-kin | T1-kin | `v8f_paired_families.py` needs an eval `run` dump (`pred`, `lat_logits`, `lon_logits`, `goal_prob`, `ckpt`, `ck_pos_err`, `ade6` per window) and cannot read the tick dumps; the head macro-F1 (R-TLAT / R-TLON) and per-class F1 are computed here from the ticks instead. The goal-head F1 exists UNPAIRED per run in `DAG_READ.json` beside.goal_macro_f1 (ticks>=1 and tick 0) | 0 (not attempted) |")
w("| TACTICAL | anchor selection; selected-vs-executed manoeuvre decisions | both | UNAVAILABLE BY DESIGN: a trajectory dump carries one path per window and no tactical selection (the tool's own statement) | 0 |")
w("| LONGITUDINAL | target-speed accuracy bands; ego-progress ratio | both | are in each eval JSON's `raw.families.longitudinal` as unpaired levels and were not paired here (nothing to pair at T1-kin: `arms.cl.four_families` carries them per run) | 0 (not attempted) |")
w("| LONGITUDINAL | anti-echo discharge | T0 | UNAVAILABLE in the eval JSON itself (no ego speed at t0 in the window dict); see 1.1 | 2,399 windows each |")
w("| LONGITUDINAL | distance keeping where no lead exists | both | only LEAD windows are scored (743 of 2,399); NO_LEAD 552, NOT_STRAIGHT 931, NO_LABEL 173 are counted and never scored as free flow | see section 2.1 n |")
w("| STRATEGIC | S checkpoint at T1-kin against the v9 route label on windows with no valid checkpoint or off-route ticks | T1-kin | windows or ticks with no label, or where S's pose is more than 3 m from the logged route, are not scored (the reader's rule); n per cell is in section 2.4 | see n |")
w("| per-class | LANE_CHANGE_L / LANE_CHANGE_R | T1-kin | no labelled cell in the matched ticks>=1 cells | 0 |")
w("| DAG - ORC | the four families other than R-TLAT / R-TLON | both | not asked by the PREREG (ORC is compared on the two head rows only) | 0 (not attempted) |")
w("| DAG - M2b | T1-kin | T1-kin | M2b has no T1-kin dump in this package; the comparison is T0 | 0 |")
w("| tick-0 F1, R76 crawl rows M1 / L2, T1 off-route fraction at tick 2, R-TLAT / R-TLON at section 2's tie rule | - | T1-kin | not requested of this package. The first three are carried per run (unpaired) in `DAG_READ.json` beside; the tie-rule re-score needs the bank matcher | 0 |")
w("")

# ---- 7. descriptive
w("## 7. What the families say, descriptively (computed from the tables; no interpretation)")
w("")


def digest(contrast, fam_prefix):
    """Per (metric, tier): the two seeds' separated verdicts, grouped. A = DAG."""
    rs = [r for r in ROWS if r["contrast"] == contrast and r["family"].startswith(fam_prefix) and r["metric"] != "ADE6, 6 s"]
    keys = []
    for r in rs:
        k = (r["metric"], r["tier"])
        if k not in keys:
            keys.append(k)
    groups = {"both seeds separated, same sign, DAG better": [], "both seeds separated, same sign, DAG WORSE": [], "both seeds separated, same sign, no direction label": [],
              "SEEDS DISAGREE (both separated, opposite signs)": [], "one seed separated, DAG better": [], "one seed separated, DAG WORSE": [], "one seed separated, no direction label": []}
    n_cells = 0
    for (m, tier) in keys:
        pair = {r["seed"]: r for r in rs if r["metric"] == m and r["tier"] == tier}
        if len(pair) < 2:
            continue
        n_cells += 1
        tg = lambda r: None if r["direction_better"] is None else ((r["delta"] < 0) == (r["direction_better"] == "lower"))  # noqa: E731
        nm = f"{m} ({'T1-kin' if tier == 'T1' else 'T0'}; " + ", ".join(f"s{sd} {sg(pair[sd]['delta'], dp_for(pair[sd]))}{'*' if pair[sd]['separated'] else ''}" for sd in (0, 1)) + ")"
        s0, s1_ = pair[0]["separated"], pair[1]["separated"]
        if s0 and s1_:
            if (pair[0]["delta"] > 0) != (pair[1]["delta"] > 0):
                groups["SEEDS DISAGREE (both separated, opposite signs)"].append(nm)
            else:
                t = tg(pair[0])
                groups["both seeds separated, same sign, " + ("no direction label" if t is None else ("DAG better" if t else "DAG WORSE"))].append(nm)
        elif s0 or s1_:
            r = pair[0] if s0 else pair[1]
            t = tg(r)
            groups["one seed separated, " + ("no direction label" if t is None else ("DAG better" if t else "DAG WORSE"))].append(nm)
    return n_cells, groups


for c, cl_ in (("DAG-DAG0", "DAG - DAG-0"), ("DAG-M2b", "DAG - M2b")):
    w(f"### {cl_}")
    w("")
    w("Digest by (metric, tier): the two seeds read together. A star marks the seed whose interval excludes 0. Cells with no separated seed are not listed. Counts are cells = metrics x tiers (two seeds each).")
    w("")
    for fp in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        n_cells, groups = digest(c, fp)
        w(f"* **{fp}** ({n_cells} metric-tier cells)")
        any_ = False
        for gname, items in groups.items():
            if items:
                any_ = True
                w(f"  * *{gname}* ({len(items)}): " + "; ".join(items))
        if not any_:
            w("  * no cell separated at either seed")
    w("")
w("### DAG - ORC")
w("")
for hd in ("lat", "lon"):
    w(f"* **R-T{hd.upper()}.** " + "; ".join(
        f"s{s} {sg(TC['per_seed'][f's{s}'][f'R-T{hd.upper()}']['DAG_minus_ORC']['delta'])} [{sg(TC['per_seed'][f's{s}'][f'R-T{hd.upper()}']['DAG_minus_ORC']['lo'])}, {sg(TC['per_seed'][f's{s}'][f'R-T{hd.upper()}']['DAG_minus_ORC']['hi'])}] "
        f"({'separated' if TC['per_seed'][f's{s}'][f'R-T{hd.upper()}']['DAG_minus_ORC']['separated'] else 'not separated'})" for s in (0, 1)))
w("")

# ---- 8. three variances
w("## 8. Which variance each interval answers")
w("")
w("1. **Another draw of EPISODES** (models held fixed): every interval here. That is all the bootstrap answers.")
w("2. **Another TRAINING run:** answered only by the two seeds. DAG-0 is the control for '3,000 more steps' and ORC-0 (banked by O-RECOVER, same argv as DAG-0, another build) is the replicate floor `RF` that the decisive bars used; "
  "there is no replicate of DAG itself at a given seed. A separated interval from one training seed is necessary, not sufficient (H-ESTIM-SEED-1). Where the two seeds disagree in sign on a cell, the table shows it.")
w("3. **Another INFERENCE run:** does not arise; the planner is deterministic (PREREG_V8F_PGT; DAG_READ.json estimator line).")
w("")

# ---- 9. manifest
w("## 9. Artifacts, provenance, estimator line")
w("")
w("```")
w(f"estimator : {F['estimator']}")
w(f"tier      : T0 = box 2 standard eval (ages cycle); T1-kin = dev-box CPU --limit causal --dump-ticks --dump-heads, replan-steps 10 (self-action OPEN loop)")
w(f"windows   : {F['n_windows']} windows, {F['n_episodes']} episodes (eval-141); B = {F['n_boot']}; seed 0")
w(f"tool tree : {F['tool_tree']}  (instruments byte-equal to the 07dcb513 tree used by O-RECOVER modulo CRLF)")
for k, v in F["inputs"].items():
    w(f"{k:26s}: {v}")
w("```")
w("")
w("Run artifacts consumed (paths from `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/runs.json`, sha256 of the T0 npz in `DAG_FAMILIES.json` `*/ids`):")
w("")
for s in (0, 1):
    ids = F["T0"][f"s{s}"]["ids"]
    w(f"* T0 s{s}: DAG `{ids['DAG']['path']}` (sha256 `{ids['DAG']['sha256'][:16]}...`); DAG-0 `{ids['DAG0']['path']}` (sha256 `{ids['DAG0']['sha256'][:16]}...`); M2b `{F['T0_vs_M2b'][f's{s}']['ids']['M2b']['path']}` (sha256 `{F['T0_vs_M2b'][f's{s}']['ids']['M2b']['sha256'][:16]}...`)")
    w(f"* T1-kin s{s}: DAG `{F['T1'][f's{s}']['ids']['DAG']['path']}`; DAG-0 `{F['T1'][f's{s}']['ids']['DAG0']['path']}`; ticks sha256 in `DAG_T1CELLS.json` inputs.ticks")
w("")
w("The package files and their roles are listed in `MANIFEST.txt` in this directory.")

open(D + "/FAMILIES_DAG.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("controls all pass:", ALL_OK, "| rows", len(ROWS), "| md bytes", os.path.getsize(D + "/FAMILIES_DAG.md"))
for nm, _, st in CONTROLS:
    print(" ", nm, ok(st))
