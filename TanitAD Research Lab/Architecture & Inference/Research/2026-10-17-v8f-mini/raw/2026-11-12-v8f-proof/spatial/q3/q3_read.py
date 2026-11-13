r"""q3_read.py -- the four metric families at T1-kin for arm pairs of SPATIAL stage 1 (card PREREG_V8F_PROOF_ADD12.md, question Q3).
CPU, zero training, no model pass.  Tier T1-kin (kinematic rollout on the model's own actions, logged frames): a diagnostic, never driving
performance.  Evidence class MEASURED once run on real dumps.

  python q3_read.py read --t1-root DIR[,DIR] --pairs G:HIER,W:HIER,G:W --seeds 0,1 --spread-arm HIER --spread-seeds 0,1,2,3,4,5 --out-dir DIR

Engine: every geometry, class and estimator function is imported unchanged from hierrep_read.py (the read tool of HIERREP, addendum 11):
load_dump, comps, kinematics, classify, episode_units, seed_arrays, boot_two_level (through read_cell), FAM_ROWS, THR, KEEP_STRAIGHT.

Reading choices (each the most literal reading of the card; none moves a threshold, a bar or an outcome rule):
  Q3-1  "family cell" = the ten rows FAM_ROWS on ALL windows, which is the family table the HIERREP read tool writes.  ADE6 on the critical,
        bulk and all windows and the tactical-band ADE are printed beside them and do not enter the name.
  Q3-2  "worse with the two-level interval excluding 0" = the paired window difference A minus B has its two-level interval (seeds, then
        episodes; B 2,000; rng 0) wholly above 0.  Every row is an error, so above 0 is worse.  "Better" = wholly below 0.
  Q3-3  "the six-seed spread" of a cell = the largest absolute difference of the arm-level cell mean between two baseline seeds (15 pairs
        with six seeds).  "Outside" = the absolute value of D is above that largest difference.  The median is printed beside it.
  Q3-4  The name is FINAL only with both seeds of the pair and all six baseline seeds.  With fewer it is written PRELIMINARY together with
        the name the rule would give on the seeds present; a preliminary name decides nothing.
  Q3-5  The family of a cell is the first part of its key (LONGITUDINAL, LATERAL, TACTICAL).  STRATEGIC: PAUSED (PI item 25).
Controls (first in the verdict): q1 window counts of the card of HIERREP; q2 an arm against itself reads exactly 0 in every cell;
  q3 the swapped pair reads exactly minus D; q4 a planted regression (three times the spread on the speed row, floor 0.05 m/s) must be
  named a cost by the same rule; q5 identical windows in every dump (else exit 3).
Option --same-sign applies the rule of PREREG_V8F_PROOF_ADD13 section 4 (DATAEFF): an effect also needs every seed on the side of D (else MIXED-SIGN).
Exit: 0 valid, 2 a control failed, 3 refused.
"""
import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TANITEVAL_ROOT", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/taniteval")
STAGEB = os.environ.get("HIERREP_STAGEB", "F:/TanitAD-scratch/v8proof/pod/stageB")
if STAGEB not in sys.path:
    sys.path.insert(0, STAGEB)
import numpy as np  # noqa: E402
import hierrep_read as RH  # noqa: E402

TIER = "T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance)"
CARD = "PREREG_V8F_PROOF_ADD12.md, section 4, Q3"
N_SPREAD_CARD = 6
EXTRA = (("ade6_m", "critical"), ("ade6_m", "bulk"), ("ade6_m", "all"), ("ade_tac_m", "all"))


def find_dump(roots, arm, s):
    for r in roots:
        d = RH.dump_dir(r, arm, s)
        if os.path.isdir(d):
            return d
    return None


def cell_rule(c, spread_max, same_sign=False):
    """Q3-2 and Q3-3.  -> (verdict, outside).  same_sign (rule of PREREG_V8F_PROOF_ADD13 section 4): an effect also needs every seed on the side of D."""
    D, lo, hi = c["D"], c["A_lo"], c["A_hi"]
    if not np.isfinite(D) or not np.isfinite(lo) or not np.isfinite(hi):
        return "NON-FINITE", False
    outside = bool(np.isfinite(spread_max) and abs(D) > spread_max)
    if same_sign and (lo > 0 or hi < 0):
        d = np.asarray(c.get("d_s", [D]), np.float64)
        if not (np.all(d > 0) or np.all(d < 0)):
            return "MIXED-SIGN", outside
    if lo > 0:
        return ("WORSE" if outside else "INSIDE-SPREAD"), outside
    if hi < 0:
        return ("BETTER" if outside else "INSIDE-SPREAD"), outside
    return "OPEN", outside


def q3_name(cells, final):
    cost = sorted({k.split(":")[0] for k, c in cells.items() if c["verdict"] == "WORSE"})
    gain = sorted({k.split(":")[0] for k, c in cells.items() if c["verdict"] == "BETTER"})
    name = "NO-COST" if not cost else " ".join("COST-" + f for f in cost)
    if gain:
        name += " " + " ".join("GAIN-" + f for f in gain)
    return {"name_by_rule": name, "final": bool(final), "name": name if final else "PRELIMINARY (" + name + ")",
            "cost_families": cost, "gain_families": gain}


def pair_cells(V, masks, inv, n_ep, A, B, seeds, n_boot, label_n, spread, same_sign=False):
    fam, extra = {}, {}
    for famname, key in RH.FAM_ROWS:
        sE, cE = RH.seed_arrays([V[(A, s)][key] for s in seeds], [V[(B, s)][key] for s in seeds], masks["all"], inv, n_ep)
        c = RH.read_cell(sE, cE, n_boot, label_n)
        sp = spread.get(key + "|all", {})
        c["spread_max"], c["spread_median"] = sp.get("max", float("nan")), sp.get("median", float("nan"))
        c["verdict"], c["outside_spread"] = cell_rule(c, c["spread_max"], same_sign)
        fam[famname + ":" + key] = c
    for key, m in EXTRA:
        sE, cE = RH.seed_arrays([V[(A, s)][key] for s in seeds], [V[(B, s)][key] for s in seeds], masks[m], inv, n_ep)
        c = RH.read_cell(sE, cE, n_boot, label_n)
        sp = spread.get(key + "|" + m, {})
        c["spread_max"], c["spread_median"] = sp.get("max", float("nan")), sp.get("median", float("nan"))
        c["verdict"], c["outside_spread"] = cell_rule(c, c["spread_max"], same_sign)
        extra[key + "|" + m] = c
    return fam, extra


def cmd_read(a):
    t0 = time.time()
    roots = [r for r in a.t1_root.split(",") if r]
    seeds = [int(x) for x in a.seeds.split(",")]
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    sp_seeds_req = [int(x) for x in a.spread_seeds.split(",")]
    res = {"card": a.card, "same_sign_rule": bool(a.same_sign), "tier": TIER, "evidence_class": "MEASURED (ours, T1-kin dumps)", "STRATEGIC": "PAUSED (PI item 25)",
           "estimator": "two-level bootstrap seeds then episodes (B %d, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged" % a.n_boot,
           "gaps": list(RH.GAPS), "controls": {}, "pairs": {}, "arm_levels": {}, "missing": []}
    need = []
    for A, B in pairs:
        for s in seeds:
            need += [(A, s), (B, s)]
    need += [(a.spread_arm, s) for s in sp_seeds_req]
    V, ref, where = {}, None, {}
    for arm, s in dict.fromkeys(need):
        d = find_dump(roots, arm, s)
        if d is None:
            res["missing"].append("%s_s%d" % (arm, s))
            continue
        try:
            D = RH.load_dump(d, a.expect_files)
        except ValueError as e:
            res["missing"].append("%s_s%d (%s)" % (arm, s, e))
            continue
        if ref is None:
            ref = D
        for k in ("g", "ws", "eid", "clip_index", "v0"):
            if not np.array_equal(D[k], ref[k]):
                print("REFUSED: windows differ between dumps: %s_s%d key %s" % (arm, s, k))
                return 3
        c = RH.comps(D["cl"], D["g"])
        c["_lon"], c["_lat"] = RH.classify(*RH.kinematics(D["cl"]))
        V[(arm, s)] = c
        where[(arm, s)] = d
    if ref is None:
        print("REFUSED: no dump found")
        return 3
    res["controls"]["q5_same_windows"] = {"PASS": True, "n_dumps": len(V)}
    uniq, inv = RH.episode_units(ref["eid"])
    n_ep = len(uniq)
    glon, glat = RH.classify(*RH.kinematics(ref["g"]))
    joint_g = glon * 3 + glat
    crit = joint_g != RH.KEEP_STRAIGHT
    masks = {"critical": crit, "bulk": ~crit, "all": np.ones(len(crit), bool)}
    for c in V.values():
        c["lon_err_rate"] = (c["_lon"] != glon).astype(float)
        c["lat_err_rate"] = (c["_lat"] != glat).astype(float)
        c["joint_err_rate"] = ((c["_lon"] * 3 + c["_lat"]) != joint_g).astype(float)
    n_c = {"n_windows": int(len(crit)), "n_episodes": int(n_ep), "n_crit": int(crit.sum()), "n_bulk": int((~crit).sum())}
    res["window_counts"] = n_c
    res["controls"]["q1_window_counts"] = {"PASS": bool(all(n_c[k] == v for k, v in RH.CARD_COUNTS.items())), "card": RH.CARD_COUNTS, "read": n_c}
    keys_all = [(k, "all") for _, k in RH.FAM_ROWS] + list(EXTRA)
    # ---- arm levels
    for (arm, s), c in sorted(V.items()):
        lv = {}
        for k, m in keys_all:
            x = c[k][masks[m]]
            lv[k + "|" + m] = float(np.nanmean(x))
        res["arm_levels"]["%s_s%d" % (arm, s)] = lv
    # ---- spread of the baseline seeds
    sp_seeds = [s for s in sp_seeds_req if (a.spread_arm, s) in V]
    spread = {}
    for k, m in keys_all:
        lv = [res["arm_levels"]["%s_s%d" % (a.spread_arm, s)][k + "|" + m] for s in sp_seeds]
        diffs = [abs(x - y) for x, y in itertools.combinations(lv, 2)]
        spread[k + "|" + m] = {"max": float(max(diffs)) if diffs else float("nan"), "median": float(np.median(diffs)) if diffs else float("nan"),
                               "n_pairs": len(diffs), "levels": lv}
    spread_complete = len(sp_seeds) >= N_SPREAD_CARD
    res["spread"] = {"arm": a.spread_arm, "seeds_present": sp_seeds, "seeds_requested": sp_seeds_req, "complete_six": bool(spread_complete), "cells": spread}
    # ---- pairs
    for A, B in pairs:
        ps = [s for s in seeds if (A, s) in V and (B, s) in V]
        tag = "%s_minus_%s" % (A, B)
        if not ps:
            res["pairs"][tag] = {"seeds_read": [], "note": "no seed with both arms"}
            continue
        fam, extra = pair_cells(V, masks, inv, n_ep, A, B, ps, a.n_boot, len(seeds), spread, a.same_sign)
        final = (len(ps) == len(seeds)) and spread_complete
        res["pairs"][tag] = {"seeds_read": ps, "seeds_requested": seeds, "families": fam, "beside": extra, "Q3": q3_name(fam, final)}
    # ---- controls q2, q3, q4 on the first pair with data
    first = next(((A, B) for A, B in pairs if res["pairs"]["%s_minus_%s" % (A, B)].get("seeds_read")), None)
    if first is not None:
        A, B = first
        ps = res["pairs"]["%s_minus_%s" % (A, B)]["seeds_read"]
        f_self, e_self = pair_cells(V, masks, inv, n_ep, A, A, ps, a.n_boot_control, len(seeds), spread)
        mx = max(abs(c["D"]) for c in list(f_self.values()) + list(e_self.values()))
        res["controls"]["q2_self_is_zero"] = {"PASS": bool(mx == 0.0), "max_abs_D": float(mx), "arm": A}
        f_sw, _ = pair_cells(V, masks, inv, n_ep, B, A, ps, a.n_boot_control, len(seeds), spread)
        f_fw = res["pairs"]["%s_minus_%s" % (A, B)]["families"]
        dev = max(abs(f_sw[k]["D"] + f_fw[k]["D"]) for k in f_fw)
        res["controls"]["q3_swap_is_minus"] = {"PASS": bool(dev < 1e-12), "max_abs_dev": float(dev)}
        key = "speed_mae_mps"
        smax = spread[key + "|all"]["max"]
        plant = max(3.0 * smax, 0.05) if np.isfinite(smax) else 0.05
        Vp = dict(V)
        for s in ps:
            cp = dict(V[(A, s)])
            cp[key] = V[(A, s)][key] + plant
            Vp[("__PLANT__", s)] = cp
        f_pl, _ = pair_cells(Vp, masks, inv, n_ep, "__PLANT__", A, ps, a.n_boot_control, len(seeds), spread)
        cpl = f_pl["LONGITUDINAL:" + key]
        named = q3_name(f_pl, True)
        ok = (cpl["verdict"] == "WORSE") and (abs(cpl["D"] - plant) < 1e-9) and (named["cost_families"] == ["LONGITUDINAL"]) and not named["gain_families"]
        res["controls"]["q4_planted_regression"] = {"PASS": bool(ok), "planted_mps": float(plant), "D_read": cpl["D"], "verdict": cpl["verdict"], "name": named["name_by_rule"]}
    flags = [v.get("PASS") for v in res["controls"].values() if isinstance(v, dict) and "PASS" in v]
    res["controls"]["ALL_PASS"] = bool(all(flags))
    res["dumps"] = {"%s_s%d" % k: v for k, v in sorted(where.items())}
    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "RESULT_Q3.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out_dir, "VERDICT_Q3.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(verdict(res))
    for tag, p in res["pairs"].items():
        if p.get("Q3"):
            print("%s: %s (seeds %s)" % (tag, p["Q3"]["name"], p["seeds_read"]))
    print("controls %s | spread seeds %s | missing %s" % ("PASS" if res["controls"]["ALL_PASS"] else "FAILED", sp_seeds, res["missing"]))
    return 0 if res["controls"]["ALL_PASS"] else 2


def _f(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else ("%+." + str(nd) + "f") % x


def verdict(res):
    L = ["# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1", "",
         "Card: " + res["card"] + ". Tier: " + res["tier"] + ". Evidence class: " + res["evidence_class"] + ".",
         "Estimator: " + res["estimator"] + ".", "STRATEGIC: " + res["STRATEGIC"] + ".", "", "## Controls (first)", ""]
    for k, v in res["controls"].items():
        if isinstance(v, dict):
            L.append("* %s: %s  %s" % (k, "PASS" if v.get("PASS") else "FAIL", json.dumps({x: y for x, y in v.items() if x != "PASS"})))
    L.append("* ALL_PASS: %s" % res["controls"]["ALL_PASS"])
    sp = res["spread"]
    L += ["", "## Run-to-run spread of the baseline", "",
          "Arm %s, seeds present %s of requested %s; complete for the card (six seeds): %s. A difference inside the spread is not an effect." % (
              sp["arm"], sp["seeds_present"], sp["seeds_requested"], sp["complete_six"])]
    if res["missing"]:
        L.append("Dumps not present: " + ", ".join(res["missing"]) + ".")
    for tag, p in res["pairs"].items():
        L += ["", "## " + tag.replace("_", " "), ""]
        if not p.get("seeds_read"):
            L.append("No seed with both arms.")
            continue
        q = p["Q3"]
        L.append("Seeds read %s of %s. **Q3: %s**%s" % (p["seeds_read"], p["seeds_requested"], q["name"],
                 "" if q["final"] else " (decides nothing: the name needs both seeds of the pair and six baseline seeds)"))
        L += ["", "| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |",
              "|---|---|---|---|---|---|"]
        for k, c in p["families"].items():
            L.append("| %s | %s | %s | [%s, %s] | %s (%s) | %s |" % (k, " / ".join(_f(x) for x in c["d_s"]), _f(c["D"]), _f(c["A_lo"]), _f(c["A_hi"]),
                     _f(c["spread_max"]).lstrip("+"), _f(c["spread_median"]).lstrip("+"), c["verdict"]))
        L += ["", "Beside (not in the name):", "", "| cell | per seed | D | two-level interval | spread max | rule |", "|---|---|---|---|---|---|"]
        for k, c in p["beside"].items():
            L.append("| %s | %s | %s | [%s, %s] | %s | %s |" % (k, " / ".join(_f(x) for x in c["d_s"]), _f(c["D"]), _f(c["A_lo"]), _f(c["A_hi"]),
                     _f(c["spread_max"]).lstrip("+"), c["verdict"]))
    L += ["", "## Arm levels (mean over windows)", "", "| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |", "|---|---|---|---|---|---|"]
    for arm, lv in res["arm_levels"].items():
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |" % (arm, lv["ade6_m|all"], lv["ade6_m|critical"], lv["speed_mae_mps|all"], lv["cross_mae_m|all"], lv["joint_err_rate|all"]))
    L += ["", "## Not measured by this read", ""] + ["* " + g for g in res["gaps"]]
    L += ["", "Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;",
          "a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--t1-root", required=True)
    r.add_argument("--pairs", default="G:HIER,W:HIER,G:W")
    r.add_argument("--seeds", default="0,1")
    r.add_argument("--spread-arm", default="HIER")
    r.add_argument("--spread-seeds", default="0,1,2,3,4,5")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--n-boot-control", type=int, default=300)
    r.add_argument("--expect-files", type=int, default=141)
    r.add_argument("--same-sign", action="store_true", help="rule of PREREG_V8F_PROOF_ADD13 section 4: an effect also needs every seed on the side of D")
    r.add_argument("--card", default=CARD)
    a = ap.parse_args(argv)
    return cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
