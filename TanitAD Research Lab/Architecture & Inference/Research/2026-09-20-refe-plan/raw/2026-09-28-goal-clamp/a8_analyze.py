#!/usr/bin/env python3
"""SPEC_NAVTEST Amendment 8 -- the registered analysis. Validity gates (a)-(e) first; only then the statistic
(per-token PDMS(ON) - PDMS(OFF), the paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927)
read TWICE -- PRIMARY (every confirmation token, clustered by its log) and FRESH-LOG (the tokens of the logs that hold
no selection token); the four families of both arms; the registered decision. Writes result_a8.json and
RESULT_A8_GOAL_SANITISATION.md beside this file. Every number is read from an artifact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
sys.path.insert(0, str(PKG / "refe"))
import proxy_eval as PE  # noqa: E402

WORK = Path("D:/Projects/TanitAD/data/refe_proxy/goal_clamp/a8")
A8 = HERE / "a8"
OUT = HERE                        # a fixture test passes --out to a scratch dir: it must never write the real RESULT
FRESH = HERE / "amendment8_fresh_set.json"
SELFTEST = HERE / "a8_selftest_sibling.json"
ARMS = ("off", "on", "clamp150", "straight")
TESTED_PLANNER = "522a87ae2d57acc04442effec131e5085dc8d3595e5683c9b68554af2663a94f"
LON = ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_final_bias_m", "accel_mae_mps2")
LAT = ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "cross_final_mae_m")
TAC = ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed")


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def status(lb):
    lf = A8 / "score" / f"{lb}.log"
    if not lf.exists():
        return None
    return PE.status_of(lf.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    global WORK, A8, OUT, FRESH, SELFTEST
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--a8", default=str(A8))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fresh", default=str(FRESH))
    ap.add_argument("--selftest", default=str(SELFTEST))
    ap.add_argument("--tested-planner", default=None, help="fixture tests only: the planner sha the fixture expects")
    ap.add_argument("--landed-planner", default=str(PKG / "refe" / "planner.py"),
                    help="the planner.py whose bytes ran (default: the package's, i.e. the landed bytes)")
    a = ap.parse_args()
    WORK, A8, OUT, FRESH, SELFTEST = Path(a.work), Path(a.a8), Path(a.out), Path(a.fresh), Path(a.selftest)
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    import planner as PL
    fresh = jl(FRESH)
    census = jl(HERE / "route_cover_census_navtest_full.json")["rows"]
    tl = fresh["fresh_token_log"]
    new_logs = set(fresh["new_logs"])
    seams, recs, sts, csvs = {}, {}, {}, {}
    for arm in ARMS:
        lb = f"refe_a8_{arm}"
        p = WORK / f"{lb}.npz"
        if p.exists():
            z = np.load(p)
            seams[arm] = {str(t): z["poses"][i] for i, t in enumerate(z["token"])}
        recs[arm] = jl(A8 / f"{lb}.inputs.json") or jl(WORK / f"{lb}.inputs.json") or {}
        sts[arm] = status(lb)
        c = A8 / "score" / f"{lb}.csv"
        csvs[arm] = PE.read_csv(c) if c.exists() else {}
    res = {"amendment": "SPEC_NAVTEST Amendment 8 (registered 2026-09-28 06:32, SPEC blob 8ace80cc)",
           "tier": "NAVSIM v1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents (W3's stamp)",
           "n_fresh_registered": len(fresh["fresh_tokens"]), "arms_present": [a for a in ARMS if a in seams]}
    gates = {}
    # (a) the trigger fires on every confirmation token (census re-checked in the run)
    on = recs.get("on", {})
    trig = {t: bool((on[t].get("goal_diag") or {}).get("triggered")) for t in on}
    derr = [abs((on[t].get("goal_diag") or {}).get("ego_to_route_m", 1e9) - census[t]["ego_to_route_m"]) for t in on]
    gates["a"] = {"ok": bool(on) and all(trig.values()) and max(derr) <= 1e-6,
                  "rows_on": len(on), "triggered": sum(trig.values()), "max_abs_diff_vs_census_m": max(derr) if derr else None}
    # (b) the unit test with its mutation arms, on the bytes that ran
    stb = jl(SELFTEST) or {}
    landed = PE.sha256(Path(a.landed_planner))
    tested = a.tested_planner or TESTED_PLANNER
    gates["b"] = {"ok": not stb.get("failed", ["MISSING"]) and stb.get("planner_sha256") == tested == landed,
                  "failed": stb.get("failed"), "mutations": {k: v.get("target_red") for k, v in (stb.get("mutations") or {}).items()},
                  "landed_planner_sha256": landed}
    # (c) the OFF seam reproduces the pipeline's own pick and score on every token
    c_ok, c_det = False, {}
    pz = WORK / "refe_a8_off_props.npz"
    if pz.exists() and "off" in seams:
        z = np.load(pz)
        stub = type("RuleStub", (), {"rule": str(z["rule"]), "V1_W": PL.REFePlanner.V1_W, "PDM_W": PL.REFePlanner.PDM_W})()
        agg = [int(PL.REFePlanner.aggregate(stub, torch.from_numpy(z["logits"][i])[None])[0].argmax())
               for i in range(len(z["token"]))]
        pose_eq = [bool(np.array_equal(z["proposals"][i, int(z["pick"][i])], seams["off"][str(t)]))
                   for i, t in enumerate(z["token"])]
        rec_pick = [recs["off"].get(str(t), {}).get("pick") == int(z["pick"][i]) for i, t in enumerate(z["token"])]
        c1 = (sts.get("off") or {}).get("C1_max_abs_delta")
        c_det = {"executed_pose_equals_dumped_pick": f"{sum(pose_eq)}/{len(pose_eq)}",
                 "pick_equals_rule_argmax": f"{sum(int(a) == int(b) for a, b in zip(agg, z['pick']))}/{len(agg)}",
                 "recorded_pick_equals_dump": f"{sum(rec_pick)}/{len(rec_pick)}", "rule": str(z["rule"]),
                 "repair_last_heading": bool(z["repair_last_heading"]), "harness_C1_max_abs_delta": c1}
        c_ok = (all(pose_eq) and all(int(a) == int(b) for a, b in zip(agg, z["pick"])) and all(rec_pick)
                and c1 == 0.0 and str(z["rule"]) == "navsim_v1" and bool(z["repair_last_heading"]))
    gates["c"] = {"ok": c_ok, **c_det}
    # (d) every harness run PASSes with every token valid
    dd = {}
    for arm in ("off", "on"):
        s = sts.get(arm) or {}
        dd[arm] = {"status": s.get("status"), "csv_valid_rows": s.get("csv_valid_rows"), "seam_rows": len(seams.get(arm, {}))}
    gates["d"] = {"ok": all(v["status"] == "PASS" and v["csv_valid_rows"] == v["seam_rows"] > 0 for v in dd.values()), **dd}
    # (e) nothing but the goal differs between the arms' inputs
    off = recs.get("off", {})
    common = sorted(set(off) & set(on))
    ego_eq = sum(off[t]["ego"] == on[t]["ego"] for t in common)
    fr_eq = sum(off[t]["frames"] == on[t]["frames"] for t in common)
    goal_diff_trig = sum(off[t]["goal"] != on[t]["goal"] for t in common if trig.get(t))
    goal_eq_untrig = sum(off[t]["goal"] == on[t]["goal"] for t in common if not trig.get(t))
    n_trig = sum(1 for t in common if trig.get(t))
    gates["e"] = {"ok": bool(common) and ego_eq == fr_eq == len(common) and goal_diff_trig == n_trig
                  and goal_eq_untrig == len(common) - n_trig,
                  "tokens": len(common), "ego_identical": ego_eq, "frames_identical": fr_eq,
                  "goal_differs_on_triggered": f"{goal_diff_trig}/{n_trig}"}
    res["gates"] = gates
    gates_ok = all(g["ok"] for g in gates.values())
    # ---- the statistic (read only after the gates)
    toks = sorted(set(csvs.get("off", {})) & set(csvs.get("on", {})))
    sc = {arm: {t: 100.0 * float(csvs[arm][t]["score"]) for t in csvs[arm]} for arm in csvs if csvs[arm]}
    reads = {}
    for name, sel in (("PRIMARY", toks), ("FRESH-LOG", [t for t in toks if tl[t] in new_logs])):
        D = {t: sc["on"][t] - sc["off"][t] for t in sel}
        mu, lo, hi = PE.boot(D, tl) if D else (float("nan"),) * 3
        reads[name] = {"n_tokens": len(sel), "n_logs": len({tl[t] for t in sel}), "D_mean": mu, "ci95": [lo, hi],
                       "pdms_off": float(np.mean([sc["off"][t] for t in sel])) if sel else None,
                       "pdms_on": float(np.mean([sc["on"][t] for t in sel])) if sel else None}
    reads["estimator"] = "paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927"
    res["reads"] = reads
    # ---- families
    fam = {arm: jl(A8 / "families" / f"refe_a8_{arm}.json") for arm in ARMS}
    adverse = PE.fam_adverse(fam["off"], fam["on"]) if fam.get("off") and fam.get("on") else {"missing": True}
    res["families_adverse_on_vs_off"] = adverse
    # ---- the registered decision
    P, F = reads["PRIMARY"], reads["FRESH-LOG"]
    few = P["n_logs"] < 20 or F["n_logs"] < 20
    no_adv = not adverse.get("missing") and not adverse.get("adverse")
    if not gates_ok:
        verdict, why = "NOT PROVEN", ["gate(s) failed: " + ", ".join(g for g, v in gates.items() if not v["ok"])]
    elif few:
        verdict, why = "NOT PROVEN", [f"fewer than 20 logs in a read (PRIMARY {P['n_logs']}, FRESH-LOG {F['n_logs']})"]
    elif P["ci95"][0] > 0 and F["ci95"][0] > 0 and no_adv:
        verdict, why = "ADOPT", ["both lower bounds > 0 and no longitudinal / lateral component separates adversely"]
    elif P["ci95"][1] < 0:
        verdict, why = "REFUTED", ["PRIMARY upper bound < 0"]
    else:
        why = []
        if not P["ci95"][0] > 0:
            why.append("PRIMARY lower bound <= 0")
        if not F["ci95"][0] > 0:
            why.append("FRESH-LOG lower bound <= 0")
        if not no_adv:
            why.append("adverse family separation(s): " + ", ".join(f"{x['family']}.{x['metric']}"
                                                                    for x in adverse.get("adverse", [])))
        verdict = "NOT PROVEN"
    res["verdict"], res["verdict_reason"] = verdict, why
    # ---- reported, not gating
    rep = {}
    for arm in ("clamp150", "straight"):
        if sc.get(arm) and sc.get("off"):
            tt = sorted(set(sc[arm]) & set(sc["off"]))
            D = {t: sc[arm][t] - sc["off"][t] for t in tt}
            mu, lo, hi = PE.boot(D, tl)
            rep[arm] = {"n": len(tt), "pdms": float(np.mean([sc[arm][t] for t in tt])), "D_vs_off": mu, "ci95": [lo, hi],
                        "status": (sts.get(arm) or {}).get("status")}
    if toks:
        rep["subscore_deltas_on_minus_off_x100"] = {c: 100.0 * float(np.mean([float(csvs["on"][t][c]) - float(csvs["off"][t][c])
                                                                                for t in toks])) for c in PE.SUBSCORES}
    kinds: dict = {}
    for t in on:
        k = (on[t].get("goal_diag") or {}).get("fallback")
        kinds[k] = kinds.get(k, 0) + 1
    rep["fallback_kinds_on"] = kinds
    rep["goal_p2_m_median"] = {arm: float(np.median([np.hypot(*recs[arm][t]["goal"][2:4]) for t in recs[arm]]))
                               for arm in ARMS if recs.get(arm)}
    rep["selection53"] = jl(OUT / "a8_selection53.json")
    res["reported_not_gating"] = rep
    res["at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(res, open(OUT / "result_a8.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    # ---- the RESULT
    L = [f"# RESULT: SPEC_NAVTEST Amendment 8 (test-time goal sanitisation) -- **{verdict}**" +
         (f" ({'; '.join(why)})" if why else ""), "",
         f"Registered 2026-09-28 06:32 (SPEC blob `8ace80cc`); snapshot 015 (md5 d7c59f4f), Amendment 7's repair ON, "
         f"rule v1, both arms. {res['tier']}. Every number is read from `result_a8.json`.", "",
         "| gate | result | detail |", "|---|---|---|"]
    for g, v in gates.items():
        L.append(f"| ({g}) | {'PASS' if v['ok'] else 'FAIL'} | `{json.dumps({k: x for k, x in v.items() if k != 'ok'}, default=float)[:400]}` |")
    L += ["", "| read | tokens | logs | PDMS OFF | PDMS ON | D = ON - OFF | 95 % CI |", "|---|---|---|---|---|---|---|"]
    for name in ("PRIMARY", "FRESH-LOG"):
        r = reads[name]
        L.append(f"| {name} | {r['n_tokens']} | {r['n_logs']} | {r['pdms_off']:.3f} | {r['pdms_on']:.3f} | "
                 f"**{r['D_mean']:+.3f}** | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |" if r["n_tokens"] else
                 f"| {name} | 0 | 0 | - | - | - | - |")
    L += ["", f"Estimator: {reads['estimator']}.", "",
          "## Four families (families6 on both gating seams; strategic UNAVAILABLE in NAVSIM by design)", "",
          f"Adverse separations ON vs OFF (an ON interval entirely on the worse side of OFF's; longitudinal / lateral "
          f"components, GATING here): `{json.dumps({k: v for k, v in adverse.items() if k != 'tactical_no_interval'}, default=float)[:900]}`", "",
          "| seam | " + " | ".join(LON + LAT) + " |", "|---|" + "---|" * (len(LON) + len(LAT))]
    for arm in ARMS:
        f = ((fam.get(arm) or {}).get("families") or {}).get("refcv6", {})
        L.append(f"| refe_a8_{arm} | " + " | ".join(PE_fmt((f.get("longitudinal") or {}).get(m)) for m in LON) + " | " +
                 " | ".join(PE_fmt((f.get("lateral") or {}).get(m)) for m in LAT) + " |")
    L += ["", "| seam | " + " | ".join(TAC) + " | goal point error (m) | strategic |", "|---|" + "---|" * (len(TAC) + 2)]
    for arm in ARMS:
        f = ((fam.get(arm) or {}).get("families") or {}).get("refcv6", {})
        t_ = f.get("tactical") or {}
        L.append(f"| refe_a8_{arm} | " + " | ".join(f"{PE_fmt((t_.get(k) or {}).get('accuracy'))} / "
                                                    f"{PE_fmt((t_.get(k) or {}).get('kappa'))}" for k in TAC) +
                 f" | {PE_fmt((t_.get('goal_setting') or {}).get('goal_point_error_m'))} | "
                 f"{(f.get('strategic') or {}).get('status', 'MISSING')} |")
    L += ["", "## Reported, not gating", "", f"`{json.dumps({k: v for k, v in rep.items() if k != 'selection53'}, default=float)[:1500]}`", "",
          f"Selection tokens (exploratory, the 53 of the 1,123 the trigger fires on): "
          f"`{json.dumps(rep.get('selection53') or 'not computed', default=float)[:800]}`", "",
          f"Analysed {res['at_local']} (Europe/Berlin)."]
    (OUT / "RESULT_A8_GOAL_SANITISATION.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": verdict, "why": why, "reads": reads, "gates": {g: v["ok"] for g, v in gates.items()}},
                     default=float))
    print(f"ZZA8_VERDICT {verdict}")
    return 0


def PE_fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


if __name__ == "__main__":
    sys.exit(main())
