"""D6 P2 analysis -- the nav-withhold probe, exactly as pre-registered in SPEC_P1P2.md s5/s6.   tanitad venv.

    python d6_p2_analyze.py [--dir raw] [--synthetic]

Inputs : raw/p2_rescore_<arm>_s*.jsonl (d6_rescore.py on each arm's emitted plans: exact devkit re-score + route geometry), arms R7_NAVOFF / R7_NAVFOLLOW /
         R7_NAVFLIP on P2_all and R7_A1 (K7) on P2_K1; the banked A1 and A1_s1 frames (d6_scene_table), raw/d6_scene_geometry_step30000.csv, raw/geom_all.jsonl.
Outputs: raw/d6_p2.json.   Estimator: PAIRED log-cluster bootstrap (navhard logs, B = 2000, seed 0) of the per-token delta vs the banked A1.
Resolved effect (SPEC s6): |delta| >= 3.0 pp AND the 95 % CI excludes 0 AND |delta| > 2 x |banked A1 - A1_s1 delta on the same tokens|.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C
import d6_part2_anatomy as P

HERE = os.path.dirname(os.path.abspath(__file__))
RAWD = os.path.join(os.path.dirname(HERE), "raw")
ARMS = ("R7_NAVOFF", "R7_NAVFOLLOW", "R7_NAVFLIP")
T_RES = 0.03


def load_arm(d, arm):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"p2_rescore_{arm}_s*.jsonl"))):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except Exception:                                       # noqa: BLE001
                continue
            if r.get("status") == "OK":
                out[r["token"]] = r
    return out


def paired_boot(delta, logs, B=2000, seed=0):
    g = pd.DataFrame({"d": delta.values, "n": 1.0}, index=logs.values).groupby(level=0).sum()
    M = g.values
    pt = M[:, 0].sum() / M[:, 1].sum()
    rng = np.random.default_rng(seed)
    res = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(M), len(M))
        res[b] = M[i, 0].sum() / M[i, 1].sum()
    return float(pt), [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=RAWD)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    tag = "SYNTHETIC-TEST (not a result)" if a.synthetic else "MEASURED"
    inp, mapping = C.load_inputs()
    G = pd.read_csv(os.path.join(RAWD, "d6_scene_geometry_step30000.csv")).set_index("token")
    T = pd.read_csv(os.path.join(RAWD, "d6_scene_table_step30000.csv")).set_index("token")
    geom = {}
    for ln in open(os.path.join(RAWD, "geom_all.jsonl"), encoding="utf-8"):
        r = json.loads(ln)
        geom[r["token"]] = r
    P2_all = [l.strip() for l in open(os.path.join(RAWD, "spec_tokens_P2_all.txt")) if l.strip()]
    S_prem = set(l.strip() for l in open(os.path.join(RAWD, "spec_tokens_P2_S_prem.txt")) if l.strip())
    S_turn = set(l.strip() for l in open(os.path.join(RAWD, "spec_tokens_P2_S_turn.txt")) if l.strip())
    A1h = C.load_arm(30000, "R7_A1")
    out = {"tag": tag, "arms": {}}

    def metrics(rows):
        """rows: token -> rescore record. returns a frame of per-token metrics for the arm."""
        recs = []
        for t, r in rows.items():
            plan = np.asarray(r["plan_xyh"], dtype=np.float64)
            g = dict(geom[t])
            g["cl_plan_raw"] = {"MAIN": r["geom"]["cl_plan_raw"]["MAIN"]}
            f = P.plan_features(g, "MAIN", plan, float(A1h.loc[t, "v0"]))
            cmd = G.loc[t, "cmd"]
            yaw4 = float(plan[-1, 2])
            comp = (yaw4 >= P.T_HEAD) if cmd == "LEFT" else (yaw4 <= -P.T_HEAD) if cmd == "RIGHT" else np.nan
            recs.append({"token": t, "DAC0": int(r["resc"]["drivable_area_compliance"] == 0), "NC0": int(r["resc"]["no_at_fault_collisions"] == 0),
                         "cls": P.classify(f), "comp": comp, "yaw4": yaw4})
        return pd.DataFrame(recs).set_index("token")

    # banked A1 / A1_s1 on the same tokens
    base = pd.DataFrame({"DAC0": (T["A1_DAC"] == 0).astype(int), "NC0": (T["A1_NC"] == 0).astype(int), "cls": G["cls"], "yaw4": G["yaw4"]}).loc[P2_all]
    cmd = G.loc[P2_all, "cmd"]
    base["comp"] = np.where(cmd == "LEFT", base.yaw4 >= P.T_HEAD, np.where(cmd == "RIGHT", base.yaw4 <= -P.T_HEAD, np.nan)).astype(float)
    s1 = pd.DataFrame({"DAC0": (T["A1s1_DAC"] == 0).astype(int), "NC0": (T["A1s1_NC"] == 0).astype(int)}).loc[P2_all]
    logs = pd.Series({t: inp[t]["log_name"] for t in P2_all})
    strata = {"S_prem": [t for t in P2_all if t in S_prem], "S_turn": [t for t in P2_all if t in S_turn], "P2_all": P2_all}
    # K7: A1 re-run reproduces the banked plans
    k7 = load_arm(a.dir, "R7_A1")
    if k7:
        mx = [max(abs(r["repro"][k]) for k in r["repro"]) for r in k7.values()]
        out["K7"] = {"n": len(k7), "exact_subscore_reproduction_share": float(np.mean(np.asarray(mx) <= 1e-9)), "max_abs": float(np.max(mx))}
    for arm in ARMS:
        rows = load_arm(a.dir, arm)
        if not rows:
            continue
        M = metrics(rows)
        res = {"n_scored": len(M), "strata": {}}
        for sname, toks in strata.items():
            tt = [t for t in toks if t in M.index]
            if not tt:
                continue
            rs = {"n": len(tt)}
            for met in ("DAC0", "NC0"):
                d = M.loc[tt, met] - base.loc[tt, met]
                fl = s1.loc[tt, met] - base.loc[tt, met]
                pt, ci = paired_boot(d, logs[tt])
                fpt, fci = paired_boot(fl, logs[tt])
                resolved = bool(abs(pt) >= T_RES and (ci[0] > 0 or ci[1] < 0) and abs(pt) > 2 * abs(fpt))
                rs[met] = {"arm_rate": float(M.loc[tt, met].mean()), "A1_rate": float(base.loc[tt, met].mean()), "delta_pp": 100 * pt, "ci95_pp": [100 * ci[0], 100 * ci[1]],
                           "seed_floor_delta_pp": 100 * fpt, "RESOLVED": resolved}
            cm = [t for t in tt if not np.isnan(M.loc[t, "comp"])]
            if cm:
                dd = M.loc[cm, "comp"] - base.loc[cm, "comp"]
                pt, ci = paired_boot(dd, logs[cm])
                rs["compliance"] = {"arm": float(M.loc[cm, "comp"].mean()), "A1": float(base.loc[cm, "comp"].mean()), "delta_pp": 100 * pt, "ci95_pp": [100 * ci[0], 100 * ci[1]], "n": len(cm)}
            for c in ("ROUTE-FOLLOWING", "WRONG-SIDE", "OVER-STEER", "LATERAL-DRIFT", "ON-ROUTE"):
                rs.setdefault("class_share", {})[c] = {"arm": float((M.loc[tt, "cls"] == c).mean()), "A1": float((base.loc[tt, "cls"] == c).mean())}
            res["strata"][sname] = rs
        # K8 (NAVFLIP): the 4-s yaw sign flips on S_turn
        if arm == "R7_NAVFLIP":
            tt = [t for t in strata["S_turn"] if t in M.index]
            if tt:
                flip = (np.sign(M.loc[tt, "yaw4"]) != np.sign(base.loc[tt, "yaw4"])) & (np.abs(base.loc[tt, "yaw4"]) > 0.05)
                out["K8"] = {"n": len(tt), "yaw_sign_flipped_share": float(flip.mean()), "PASS": bool(flip.mean() >= 0.5)}
        out["arms"][arm] = res
    # ---------------------------------------------------------------- pre-registered reading
    rd = {}
    def res_ok(arm, stratum, met):
        try:
            return out["arms"][arm]["strata"][stratum][met]
        except KeyError:
            return None
    for arm in ("R7_NAVFOLLOW", "R7_NAVOFF"):
        m = res_ok(arm, "S_prem", "DAC0")
        if m:
            rd[f"{arm}_S_prem_DAC0"] = {"delta_pp": m["delta_pp"], "RESOLVED": m["RESOLVED"], "reading": ("O1: premature turn is nav-caused -> lever L2" if (m["RESOLVED"] and m["delta_pp"] < 0) else "not resolved")}
    m = res_ok("R7_NAVOFF", "S_turn", "DAC0")
    if m:
        rd["R7_NAVOFF_S_turn_DAC0"] = {"delta_pp": m["delta_pp"], "RESOLVED": m["RESOLVED"], "reading": ("O3: the command helps the turn; residual under-turn = execution deficit (L1)" if (m["RESOLVED"] and m["delta_pp"] > 0) else "not resolved")}
    k8 = out.get("K8")
    rd["K8"] = k8
    o1_fired = any(isinstance(v, dict) and v.get("RESOLVED") and "O1" in str(v.get("reading", "")) for v in rd.values())
    if k8 is not None and not k8["PASS"] and o1_fired:
        # POST-NUMBER DISCLOSURE: SPEC_P1P2 s6 does not say which row wins when K8 fails AND a resolved O1 drop exists; the table's two rows then CONTRADICT
        # (O4: "premature turn and under-turn CANNOT be attributed to nav" vs O1: "the premature turn is caused by the command"). The script reports the conflict, names no branch.
        rd["overall"] = ("REGISTERED RULES CONFLICT: K8 failed (O4: the nav input is not used on this surface -> no nav attribution) AND O1 fired (a resolved DAC0 drop on S_prem under NAVOFF/NAVFOLLOW -> "
                         "premature turn nav-caused). Both are reported; no single branch is named; the ruling belongs to the registrar.")
    elif k8 is not None and not k8["PASS"]:
        rd["overall"] = "O4: the nav input is not used on this surface (NAVFLIP does not flip the turn) -> the pathway is the defect; no nav attribution of premature/under-turn"
    else:
        flags = [v for k, v in rd.items() if isinstance(v, dict) and v.get("RESOLVED")]
        rd["overall"] = "see per-arm readings" if flags else "none resolved: negative for L2 on NavSim; keep L1 / L3"
    out["pre_registered_reading"] = rd
    dflt = lambda o: o.item() if hasattr(o, "item") else str(o)
    k7 = out.get("K7")
    if k7 is None or k7["exact_subscore_reproduction_share"] < 0.99:
        # SPEC s5: K7 (A1 re-run reproduces the banked sub-scores on >= 99 %) gates every P2 read; a failing / missing control -> INCONCLUSIVE, numbers withheld.
        json.dump({"WITHHELD_K7": k7, "arms": out.pop("arms"), "pre_registered_reading": out.pop("pre_registered_reading")},
                  open(os.path.join(a.dir, "d6_p2_WITHHELD_controls_failed.json"), "w", encoding="utf-8"), indent=1, default=dflt)
        out["pre_registered_reading"] = {"status": "INCONCLUSIVE (SPEC s5): K7 failed or missing", "K7": k7}
        json.dump(out, open(os.path.join(a.dir, "d6_p2.json"), "w", encoding="utf-8"), indent=1, default=dflt)
        print(json.dumps(out["pre_registered_reading"], indent=1, default=str))
        return
    json.dump(out, open(os.path.join(a.dir, "d6_p2.json"), "w", encoding="utf-8"), indent=1, default=dflt)
    print(json.dumps({"tag": tag, "K7": out.get("K7"), "K8": out.get("K8"), "reading": rd}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
