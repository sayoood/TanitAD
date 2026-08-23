#!/usr/bin/env python
"""Render the CORRECTED ladder tables from the re-read artifacts.

⛔ THIS FILE COMPUTES NOTHING ABOUT THE MODEL. It opens banked JSON and arranges
it. Every number it emits is traceable to one file and one field, and the JSON
it writes carries that provenance per table so a reader can re-derive any cell.

Inputs (all opened, never quoted from a headline — C91):
  A. `…/2026-08-17-latent-linear-ladder/raw/ll_*.json`      — INCUMBENT (pc6)
  B. `…/2026-08-18-k1-degeneracy-guard/raw/reread/llR_*.json` — REPAIRED (unpen)
  C. `…/2026-08-17-latent-linear-ladder/raw/ll_rep_*.json`   — REPAIRED (centred)
  D. `…/2026-08-18-ladder-corrected/raw/rep_guard/llrepG_*.json` — C + the guard

⚠️ B AND C ARE DIFFERENT REPAIR ROUTES AND ARE NEVER POOLED (C100). Every table
below is stamped with the route it came from; the two are placed side by side in
exactly one table, whose entire purpose is to show that they differ.
"""
from __future__ import annotations
import json
import pathlib

INC = pathlib.Path(
    "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Hub/"
    "Architecture & Inference/Implementation/incoming")
LL = INC / "2026-08-17-latent-linear-ladder" / "raw"
RR = INC / "2026-08-18-k1-degeneracy-guard" / "raw" / "reread"
GD = INC / "2026-08-18-k1-degeneracy-guard" / "raw"
OUTD = INC / "2026-08-18-ladder-corrected" / "raw"
REPG = OUTD / "rep_guard"

LADDER = ["ego_v0", "ego_accel", "ego_yawrate", "ego_curv", "n_agents_grid",
          "n_agents_all", "lead_present", "nearest_any", "lead_gap",
          "lead_closing", "lead_inv_ttc"]
# the four rungs on which the ladder's rung profile was read as independent
# confirmation of the 40:1 pooling bottleneck
POOLING_RUNGS = ["ego_yawrate", "ego_curv", "lead_closing", "lead_inv_ttc"]
ARMS = ["s11250", "nullmatched", "orcdir", "proxyv0", "s09000", "s09250",
        "s10000", "s02000", "egoorc_n0.1", "egoorc_n1", "egoorc_n3",
        "egoorc_n10", "tok11250", "tok11250null", "cells_tokwin"]
CKPT = ["s02000", "s09000", "s09250", "s10000", "s11250"]


def load(p):
    return json.loads(pathlib.Path(p).read_text("utf-8"))


def seed0(doc, tgt):
    return doc["targets"][tgt]["per_seed"]["0"]


def verdict(s):
    if not s["K1_separated"]:
        return "not-separated"
    return "PASS" if s["K1_delta"] < 0 else "FAIL-separated"


def main() -> int:
    inc = {a: load(LL / f"ll_{a}.json") for a in ARMS}
    rep = {a: load(RR / f"llR_{a}.json") for a in ARMS}
    out = {
        "_evidence_class": "MEASURED (ours; arrangement of banked JSON, no refit)",
        "eval_tier": "T0-DIAGNOSTIC",
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, n_boot 2000, 70 episode clusters",
        "forbidden": "overlapping_holdout_se",
        "routes": {"incumbent": "fit_mode pc6 (C92-defective)",
                   "repaired_A": "fit_mode unpen — ridge_fit(intercept_col=-1), the MODULE's repair",
                   "repaired_B": "fit_mode centred — the locally re-derived repair; NOT poolable with A"},
    }

    # ---- T1 the 87-FAIL inventory, recomputed from the artifacts ------------
    inv = {"die_at_repair": [], "killed_by_guard": [], "flip_to_PASS": [],
           "survive_both": [], "substantive": []}
    for a in ARMS:
        for t in LADDER:
            si, sr = seed0(inc[a], t), seed0(rep[a], t)
            if verdict(si) != "FAIL-separated":
                continue
            g = sr["k1_guard"]
            row = {"arm": a, "target": t, "old_K1": si["K1_delta"],
                   "new_K1": sr["K1_delta"], "K1B": g["K1B_delta"],
                   "K1B_ci": [g["K1B_lo"], g["K1B_hi"]],
                   "guard": g["guard_verdict"], "gt_sd": g["gt_sd"],
                   "K1B_rel_gt_sd": (abs(g["K1B_delta"]) / g["gt_sd"]) if g["gt_sd"] else None}
            v = verdict(sr)
            if v == "not-separated":
                inv["die_at_repair"].append(row)
            elif v == "PASS":
                inv["flip_to_PASS"].append(row)
            elif g["guard_verdict"] in ("CONSTANT-OFFSET-ONLY", "DEGENERATE-CONSTANT"):
                inv["killed_by_guard"].append(row)
            else:
                inv["survive_both"].append(row)
                if row["K1B_rel_gt_sd"] and row["K1B_rel_gt_sd"] >= 0.02:
                    inv["substantive"].append(row)
    out["T1_fail_inventory"] = {
        "banked_separated_FAILs": sum(len(v) for k, v in inv.items() if k != "substantive"),
        "counts": {k: len(v) for k, v in inv.items()},
        "route": "repaired_A (unpen)", **inv}

    # ---- T2 the corrected ladder, v6@11250 with every control --------------
    t2 = []
    for t in LADDER:
        r = {"target": t}
        for tag, arm in (("v6", "s11250"), ("null", "nullmatched"),
                         ("cv0", "proxyv0"), ("oracle", "orcdir")):
            s = seed0(rep[arm], t)
            g = s["k1_guard"]
            r[tag] = {"err": s["err"], "K1": s["K1_delta"],
                      "K1_ci": [s["K1_lo"], s["K1_hi"]],
                      "verdict": verdict(s), "K1B": g["K1B_delta"],
                      "K1B_ci": [g["K1B_lo"], g["K1B_hi"]],
                      "K1B_sep": g["K1B_separated"], "guard": g["guard_verdict"],
                      "K1B_rel_gt_sd": round(abs(g["K1B_delta"]) / g["gt_sd"], 4) if g["gt_sd"] else None,
                      "r": s["corr"], "r2": s["r2_ceiling"],
                      "r_wep": s["corr_within_ep"],
                      "r_pv0": s.get("corr_partial_v0"),
                      "alpha": s["alpha_chosen"], "pred_sd": s["pred_sd"]}
        r["unit"] = rep["s11250"]["targets"][t]["unit"]
        r["gt_sd"] = rep["s11250"]["targets"][t]["gt_sd"]
        r["n_eval"] = rep["s11250"]["targets"][t]["n_eval"]
        r["c_const"] = seed0(rep["s11250"], t)["c_const_value"]
        # ⛔ THE TRIVIAL-PROXY MARGIN, on every rung, in gt_sd units.
        r["v6_minus_cv0_K1B"] = round(r["v6"]["K1B"] - r["cv0"]["K1B"], 4)
        r["v6_minus_cv0_rel_gt_sd"] = round(
            (r["v6"]["K1B"] - r["cv0"]["K1B"]) / r["gt_sd"], 4) if r["gt_sd"] else None
        t2.append(r)
    out["T2_corrected_ladder_at_11250"] = {"route": "repaired_A (unpen)", "rows": t2}

    # ---- T3 the rung profile, incumbent vs repaired (the pooling question) --
    t3 = []
    for t in LADDER:
        si, sr = seed0(inc["s11250"], t), seed0(rep["s11250"], t)
        sn_i, sn_r = seed0(inc["nullmatched"], t), seed0(rep["nullmatched"], t)
        t3.append({"target": t,
                   "old_r2": si["r2_ceiling"], "new_r2": sr["r2_ceiling"],
                   "old_r": si["corr"], "new_r": sr["corr"],
                   "null_old_r2": sn_i["r2_ceiling"], "null_new_r2": sn_r["r2_ceiling"],
                   "old_alpha": si["alpha_chosen"], "new_alpha": sr["alpha_chosen"],
                   "old_verdict": verdict(si), "new_verdict": verdict(sr),
                   "guard": sr["k1_guard"]["guard_verdict"],
                   "pooling_confirmation_rung": t in POOLING_RUNGS})
    out["T3_rung_profile"] = {"route_new": "repaired_A (unpen)", "rows": t3}

    # ---- T4 checkpoint trajectory under the repair -------------------------
    out["T4_checkpoint_trajectory"] = {
        "route": "repaired_A (unpen)",
        "rows": [{"target": t, **{c: seed0(rep[c], t)["corr"] for c in CKPT},
                  **{c + "_r2": seed0(rep[c], t)["r2_ceiling"] for c in CKPT}}
                 for t in LADDER]}

    # ---- T5 the two repair routes, side by side ----------------------------
    t5 = []
    if (REPG / "llrepG_s11250.json").exists():
        repb = {a: load(REPG / f"llrepG_{a}.json") for a in
                ["s11250", "nullmatched", "orcdir", "proxyv0"]}
        out["T5_available"] = True
        for a in ["s11250", "nullmatched", "orcdir", "proxyv0"]:
            for t in LADDER:
                sa, sb = seed0(rep[a], t), seed0(repb[a], t)
                ga, gb = sa["k1_guard"], sb["k1_guard"]
                t5.append({"arm": a, "target": t,
                           "A_unpen_alpha": sa["alpha_chosen"],
                           "B_centred_alpha": sb["alpha_chosen"],
                           "alpha_differs": sa["alpha_chosen"] != sb["alpha_chosen"],
                           "A_K1": sa["K1_delta"], "B_K1": sb["K1_delta"],
                           "A_K1B": ga["K1B_delta"], "B_K1B": gb["K1B_delta"],
                           "A_verdict": verdict(sa), "B_verdict": verdict(sb),
                           "A_guard": ga["guard_verdict"], "B_guard": gb["guard_verdict"],
                           "verdict_differs": verdict(sa) != verdict(sb),
                           "guard_differs": ga["guard_verdict"] != gb["guard_verdict"],
                           "abs_K1_gap": round(abs(sa["K1_delta"] - sb["K1_delta"]), 6)})
        out["T5_two_routes"] = {
            "n_rows": len(t5),
            "alpha_choices_differing": sum(r["alpha_differs"] for r in t5),
            "verdicts_differing": sum(r["verdict_differs"] for r in t5),
            "guard_verdicts_differing": sum(r["guard_differs"] for r in t5),
            "max_abs_K1_gap": max((r["abs_K1_gap"] for r in t5), default=None),
            "rows": t5}
        # ---- T6 the guarded rep arms (route B), the last unre-read rows ----
        t6 = []
        for a in ["s11250", "nullmatched", "orcdir", "proxyv0"]:
            for t in LADDER:
                s = seed0(repb[a], t)
                g = s["k1_guard"]
                t6.append({"arm": a, "target": t, "K1": s["K1_delta"],
                           "K1_ci": [s["K1_lo"], s["K1_hi"]],
                           "verdict": verdict(s), "K1B": g["K1B_delta"],
                           "K1B_ci": [g["K1B_lo"], g["K1B_hi"]],
                           "K1B_sep": g["K1B_separated"],
                           "guard": g["guard_verdict"],
                           "quotable": s["K1_PASSES_GUARDED"],
                           "K1B_rel_gt_sd": round(abs(g["K1B_delta"]) / g["gt_sd"], 4) if g["gt_sd"] else None,
                           "r": s["corr"], "sd_ratio": g["sd_ratio"],
                           "seed_K1_range": repb[a]["targets"][t]["seed_K1_range"]})
        out["T6_rep_arms_guarded"] = {
            "route": "repaired_B (centred) — the route LATENT_LINEAR_LADDER §7 was rendered from",
            "seeds": "0,1,2 (the banked ll_rep_* seed set)",
            "counts": {"PASS": sum(r["verdict"] == "PASS" for r in t6),
                       "FAIL-separated": sum(r["verdict"] == "FAIL-separated" for r in t6),
                       "not-separated": sum(r["verdict"] == "not-separated" for r in t6),
                       "quotable_guarded_PASS": sum(bool(r["quotable"]) for r in t6),
                       "DEGENERATE-CONSTANT": sum(r["guard"] == "DEGENERATE-CONSTANT" for r in t6),
                       "CONSTANT-OFFSET-ONLY": sum(r["guard"] == "CONSTANT-OFFSET-ONLY" for r in t6),
                       "OK": sum(r["guard"] == "OK" for r in t6)},
            "rows": t6}
    else:
        out["T5_available"] = False

    (OUTD / "corrected_tables.json").write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps({"T1_counts": out["T1_fail_inventory"]["counts"],
                      "T1_total": out["T1_fail_inventory"]["banked_separated_FAILs"],
                      "T5_available": out["T5_available"],
                      "T5": {k: v for k, v in out.get("T5_two_routes", {}).items()
                             if k != "rows"},
                      "T6": out.get("T6_rep_arms_guarded", {}).get("counts")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
