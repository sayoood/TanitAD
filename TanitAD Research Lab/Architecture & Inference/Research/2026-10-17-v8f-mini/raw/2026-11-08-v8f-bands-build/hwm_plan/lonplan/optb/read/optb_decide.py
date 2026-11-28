"""OPTB deciding-read driver (AMEND_OPTB section 3, deciding read; ACTRSP card 4.2 / 4.3 / 4.4). Tier T0 (oracle goal: a world-model diagnostic, never driving performance). CPU only, zero training.

It does NOT plan. The planner runs (`hwm_plan/run_hwm.py windows / plan`, phase 1's levers, arm `hwm_oracle`, inference seeds 0 and 1, the arm's own checkpoint and TRAIN latents) produce
`plans_hwm_oracle_s{0,1}.npz` per arm; this driver reads those plans exactly as `lonplan/build/contrast_l1full_strat.py` does (same `hwm_eval` reader from `lonplan/build/reader_src`, same
`band_metrics`, same `E.contrast` = paired episode-cluster bootstrap B 2,000 seed 0 over the 994 SCORE windows / 71 episodes). The existing scoring code is IMPORTED, not edited or copied.
New here (the contrast script has none of it): the lever-seed mean statistic Delta-bar_k (card 4.2 / READ_PLAN N3: the mean of the two lever-seed deltas bootstrapped as ONE paired statistic over the
same windows), Floor_k = max(F_T,k, F_I), BETTER / WORSE as card 4.2 defines them, and the mechanical outcome label of AMEND_OPTB section 3.
Families, per band, never pooled: LONGITUDINAL, LATERAL, TACTICAL (goal / anchor selection NOT COMPUTABLE, reason from hwm_eval.NOT_COMPUTED), STRATEGIC (reported in full on the 6 s band, structural n/a
items named by the reader, never a pass). Deciding quantities: band 0-2 s `acc_mae` (LONGITUDINAL) and `yaw_mae` (LATERAL).
Estimators and the question each answers: paired episode-cluster bootstrap = another draw of EPISODES (checkpoint, plans fixed); F_T,k = max - min of the three FT-NULL point means = another TRAINING run;
F_I = max |seed 1 - seed 0| over every arm and baseline present = another INFERENCE run.
Manifest JSON: {"windows": npz, "arms": {tag: {"s0": plans npz, "s1": plans npz|null}}, "baselines": {"R35_s0": {"s0":..,"s1":..}, "R35_s1": ...}, "mech": mech json|null,
                "controls": {"c2_c3_hold": true|false|null, "band_met_lost": {"0": bool|null, "1": bool|null}}, "pairs": optional override, "smoke": bool, "n_boot": 2000}
Usage:  python optb_decide.py --manifest M.json --out OUT.json [--n-boot 2000]
"""
import argparse, json, os, sys
sys.dont_write_bytecode = True

CODE_ROOT = os.environ.get("LONPLAN_CODE_ROOT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory")
READER = os.environ.get("LONPLAN_READER_SRC", "F:/TanitAD-scratch/actrsp/lonplan/build/reader_src")
LADDER_TREE = os.environ.get("WMACT_CODE_ROOT", "F:/TanitAD-scratch/ladder_tree")
Q = (("LONGITUDINAL", "acc_mae"), ("LATERAL", "yaw_mae"))                  # the deciding quantities, band 0-2 s
BAND = "0-2s"
PAIRS = {0: {"rsp": [("FTRSP_s0", "FTNULL_s0"), ("FTRSP_s2", "FTNULL_s2")], "nulls": ["FTNULL_s0", "FTNULL_s2", "FTNULL_s4"]},
         1: {"rsp": [("FTRSP_s1", "FTNULL_s1"), ("FTRSP_s3", "FTNULL_s3")], "nulls": ["FTNULL_s1", "FTNULL_s3", "FTNULL_s5"]}}
FAVOUR = ["OPTB-CEILING", "OPTB-NARROWS", "OPTB-MECH-ONLY", "OPTB-NULL", "OPTB-HARM"]           # most to least favourable
NEXT = {"OPTB-CEILING": "this WM replaces R35 for planning; read 2-6 s and 6 s at once; T1 read of HWM-PLAN phase 2 goal-head arms on it; term at levels 2 and 3 by amendment",
        "OPTB-NARROWS": "lambda 3.0, four runs, same pairs and floors, carded before any window is scored",
        "OPTB-MECH-ONLY": "response right, plan not improved: LON-PLAN, zero training, dev-box CPU, prior-regularised planner (--lam > 0, dose by an HWM-PLAN amendment), then the level-2 image constraint",
        "OPTB-NULL": "(i) term trained down (end rsp/L1 <= 0.5 x first, eval cycle gain >= 0.8): LAT-TOK and the adapter, PI decision. (ii) did not train down: lambda 3.0",
        "OPTB-HARM": "term stays off; lambda 0.3, four runs",
        "SEEDS-DISAGREE": "reported as such; the less favourable label's next step applies"}


# ------------------------------------------------------------------------------------------------------------- pure decision logic (tested without any data)
def better_worse(dbar, dj, floor):
    """card 4.2. dbar {delta, lo, hi}: the mean of the two lever-seed deltas as one paired statistic. dj: the two lever-seed point deltas. Positive = worse (lower is better).
    BETTER (this chassis): CI wholly below 0, |dbar| > Floor_k, both dj below 0. WORSE: CI wholly above 0 and dbar > Floor_k."""
    d = dbar["delta"]
    return {"better": bool(dbar["hi"] < 0 and abs(d) > floor and all(x < 0 for x in dj)), "worse": bool(dbar["lo"] > 0 and d > floor)}


def chassis_label(ci):
    """ci: {"dj": {q: [d1, d2]}, "dbar": {q: {...}}, "floor": {q: float}, "hold": {arm: {q: {"hi": float}}}, "mech": bool|None, "band_met_lost": bool|None, "c2_c3_hold": bool|None}.
    Order: HARM (WORSE on a deciding quantity, or BAND-MET lost) > SEEDS-DISAGREE (the two lever seeds differ in sign on a deciding quantity) > CEILING > NARROWS > MECH-ONLY > NULL."""
    bw = {q: better_worse(ci["dbar"][q], ci["dj"][q], ci["floor"][q]) for q in ci["dbar"]}
    qs = list(bw)
    out = {"bw": bw, "notes": []}
    if any(v["worse"] for v in bw.values()) or ci.get("band_met_lost"):
        out["label"] = "OPTB-HARM"
        return out
    if any((min(ci["dj"][q]) < 0 < max(ci["dj"][q])) for q in qs):
        out["label"] = "SEEDS-DISAGREE"
        out["notes"].append("the two lever seeds differ in sign on a deciding quantity")
        return out
    ceiling = any(all(h[q]["hi"] < 0 for h in ci["hold"].values()) for q in qs)     # HWM-ORACLE(arm) - HOLD, CI wholly below 0 on q, for every lever arm of this chassis; the other q is not WORSE (checked above)
    if ceiling:
        out["label"] = "OPTB-CEILING"
    elif any(v["better"] for v in bw.values()):
        out["label"] = "OPTB-NARROWS"
    elif ci.get("mech") is True:
        out["label"] = "OPTB-MECH-ONLY"
    elif ci.get("mech") is False:
        out["label"] = "OPTB-NULL"
        if ci.get("c2_c3_hold") is not True:
            out["notes"].append("NULL requires C2/C3 holding: UNVERIFIED here (controls.c2_c3_hold is not true)")
    else:
        out["label"] = "MECH-NOT-COMPUTABLE"
        out["notes"].append("the IA-3P-K mechanism read is missing: MECH-ONLY / NULL cannot be told apart")
    return out


def overall_label(per_chassis):
    """per_chassis {k: chassis_label dict}. Both chassis required. HARM at EITHER chassis is HARM (the card's own words); otherwise a label (or a SEEDS-DISAGREE) at one chassis only is SEEDS-DISAGREE and the
    less favourable label's next step applies."""
    if set(per_chassis) != {0, 1}:
        return {"label": "INCOMPLETE", "why": f"chassis present {sorted(per_chassis)}; both are required", "next": None}
    labs = {k: v["label"] for k, v in per_chassis.items()}
    if "OPTB-HARM" in labs.values():
        return {"label": "OPTB-HARM", "per_chassis": labs, "next": NEXT["OPTB-HARM"], "note": "HARM at either chassis (card wording); resolution of HARM-vs-SEEDS-DISAGREE is an MM ruling"}
    if len(set(labs.values())) == 1 and "SEEDS-DISAGREE" not in labs.values() and "MECH-NOT-COMPUTABLE" not in labs.values():
        L = labs[0]
        return {"label": L, "per_chassis": labs, "next": NEXT[L]}
    known = [l for l in labs.values() if l in FAVOUR]
    worst = max(known, key=FAVOUR.index) if known else None
    return {"label": "SEEDS-DISAGREE", "per_chassis": labs, "less_favourable": worst, "next": NEXT["SEEDS-DISAGREE"] + (": " + NEXT[worst] if worst else "")}


# ------------------------------------------------------------------------------------------------------------- scoring (the existing reader, imported)
def setup(windows):
    os.environ.setdefault("WMACT_CODE_ROOT", LADDER_TREE)
    for d in (os.path.join(LADDER_TREE, "stack"), os.path.join(LADDER_TREE, "taniteval"), os.path.join(CODE_ROOT, "hwm_plan"), os.path.join(CODE_ROOT, "wmact"), os.path.join(CODE_ROOT, "plantest2"),
              os.path.join(CODE_ROOT, "plantest"), READER):
        if d not in sys.path:
            sys.path.insert(0, d)
    import numpy as np
    import hwm_plan as H
    import hwm_data as D
    import hwm_eval as E
    import run_hwm as R
    assert os.path.dirname(os.path.abspath(E.__file__)) == os.path.abspath(READER), E.__file__
    ws = R.load_ws(os.path.dirname(windows) if windows.endswith(".npz") else windows)    # run_hwm.load_ws takes the DIRECTORY holding windows.npz
    ws.lead = D.lead_for(ws.eid, ws.now)
    ws.lead6 = D.lead_for(ws.eid, ws.now, D.LEAD_BLOCK6)
    ws.text = E.attach_text(ws)
    return np, H, E, ws


class Scorer:
    def __init__(self, windows, n_boot):
        self.np, self.H, self.E, self.ws = setup(windows)
        self.n_boot = n_boot
        self.cache = {}

    def met(self, spec):
        if spec not in self.cache:
            plan = self.H.hold_plan(self.ws.A[:, 0], self.H.N_TRAJ) if spec == "hold" else self.np.load(spec)["plan_raw"]
            self.cache[spec] = self.E.band_metrics(plan, self.ws.A, self.ws.v0, self.ws.lead, self.ws.lead6, self.ws.text)[0]
        return self.cache[spec]

    def con(self, x, y, fam, m):
        E = self.E
        return E.contrast(x, y, self.ws.eid, self.n_boot, E.DIRECTION.get(m, "lower") if fam == "STRATEGIC" else "lower")

    def mean_of(self, specs, band, m):
        a = self.np.stack([self.met(s)[band][m] for s in specs])
        return a.mean(0)                                                # NaN in either lever arm leaves NaN: the window is not scored (both finite, as E.contrast requires)

    def pm(self, spec, band, m):
        v = self.met(spec)[band][m]
        v = v[self.np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")


def run(man, n_boot):
    S = Scorer(man["windows"], n_boot)
    np, E = S.np, S.E
    arms, base = man["arms"], man.get("baselines", {})
    pairs = {int(k): v for k, v in man.get("pairs", {}).items()} or PAIRS
    chassis = [k for k in (0, 1) if k in pairs and all(a in arms for p in pairs[k]["rsp"] for a in p) and all(t in arms for t in pairs[k]["nulls"])]
    # inference floor: max |seed 1 - seed 0| over every arm and baseline that has both seeds
    allp = {**arms, **{k: v for k, v in base.items()}}
    seeds_pairs = {t: v for t, v in allp.items() if v.get("s1")}
    F_I = {}
    for fam, m in Q:
        diffs = [abs(S.pm(v["s1"], BAND, m) - S.pm(v["s0"], BAND, m)) for v in seeds_pairs.values()]
        F_I[m] = {"value": max(diffs) if diffs else float("nan"), "n_arms_with_both_seeds": len(diffs), "arms": sorted(seeds_pairs)}
    rows, per_c, floors = [], {}, {}
    for k in chassis:
        P = pairs[k]
        F_T = {}
        for fam, m in Q:
            pts = [S.pm(arms[t]["s0"], BAND, m) for t in P["nulls"]]
            F_T[m] = {"value": max(pts) - min(pts), "points": dict(zip(P["nulls"], pts))}
        floors[k] = {m: {"F_T": F_T[m], "F_I": F_I[m], "Floor": max(F_T[m]["value"], F_I[m]["value"] if F_I[m]["value"] == F_I[m]["value"] else 0.0)} for _, m in Q}
        ci = {"dj": {}, "dbar": {}, "floor": {}, "hold": {}, "mech": (man.get("mech_by_chassis") or {}).get(str(k)), "band_met_lost": (man.get("controls", {}).get("band_met_lost") or {}).get(str(k)),
              "c2_c3_hold": man.get("controls", {}).get("c2_c3_hold")}
        for fam, m in Q:
            dj = []
            for rsp, nul in P["rsp"]:
                r = S.con(S.met(arms[rsp]["s0"])[BAND][m], S.met(arms[nul]["s0"])[BAND][m], fam, m)
                dj.append(r["delta"])
                rows.append(dict(kind="deciding_pair", chassis=k, arm=rsp, vs=nul, family=fam, band=BAND, metric=m, **r))
            rs = [arms[a]["s0"] for a, _ in P["rsp"]]
            ns = [arms[b]["s0"] for _, b in P["rsp"]]
            r = S.con(S.mean_of(rs, BAND, m), S.mean_of(ns, BAND, m), fam, m)
            rows.append(dict(kind="deciding_dbar", chassis=k, arms=[a for a, _ in P["rsp"]], vs=[b for _, b in P["rsp"]], family=fam, band=BAND, metric=m, **r))
            ci["dj"][m], ci["dbar"][m], ci["floor"][m] = dj, r, floors[k][m]["Floor"]
            for rsp, _ in P["rsp"]:
                h = S.con(S.met(arms[rsp]["s0"])[BAND][m], S.met("hold")[BAND][m], fam, m)
                ci["hold"].setdefault(rsp, {})[m] = h
                rows.append(dict(kind="absolute_vs_hold", chassis=k, arm=rsp, family=fam, band=BAND, metric=m, **h))
        per_c[k] = ci
    # the full families grid, per band, never pooled: each lever arm vs its pair, vs HOLD, vs R35 (reference only)
    for k in chassis:
        for rsp, nul in pairs[k]["rsp"]:
            refs = [("pair", arms[nul]["s0"]), ("HOLD", "hold")] + ([("R35", base[f"R35_s{k}"]["s0"])] if f"R35_s{k}" in base else [])
            for fam, mets in E.FAMILIES.items():
                for b in E.BANDS:
                    for m in mets + (E.BESIDE if fam == "LATERAL" else ()):
                        for rn, rs_ in refs:
                            rows.append(dict(kind="grid", chassis=k, arm=rsp, ref=rn, family=fam, band=b, metric=m, reference_only=(rn == "R35"), **S.con(S.met(arms[rsp]["s0"])[b][m], S.met(rs_)[b][m], fam, m)))
    labels = {k: chassis_label(per_c[k]) for k in chassis}
    res = {"tier": "T0", "evidence": "MEASURED when run on real arm plans", "smoke": bool(man.get("smoke")), "n_windows": int(S.ws.n), "n_episodes": int(len(set(S.ws.eid))),
           "estimators": {"interval": "paired episode-cluster bootstrap (hwm_eval.contrast), B %d seed 0: another draw of EPISODES" % n_boot,
                          "F_T": "max - min of the three FT-NULL point means on the row: another TRAINING run", "F_I": "max |seed 1 - seed 0| over every arm and baseline with both seeds: another INFERENCE run",
                          "Floor_k": "max(F_T,k, F_I)"},
           "floors": floors, "decision_inputs": {str(k): {kk: vv for kk, vv in c.items()} for k, c in per_c.items()},
           "chassis_labels": {str(k): v for k, v in labels.items()}, "outcome": overall_label({k: v for k, v in labels.items()}),
           "tactical_not_computed": {"goal_setting": E.NOT_COMPUTED.get("TACTICAL.goal_setting")}, "not_computed": E.NOT_COMPUTED,
           "strategic_note": "REPORTED IN FULL on the 6 s band; structural n/a items (S_CKPT, 6-30 s intent, derived goals, detected info) are in `strategic` with reasons; never a pass; no outcome is decided on a STRATEGIC cell",
           "rows": rows}
    if chassis:
        k = chassis[0]
        a, b = pairs[k]["rsp"][0]
        res["strategic"] = E.strategic_block({"A": S.met(arms[a]["s0"]), "B": S.met(arms[b]["s0"]), "HOLD": S.met("hold")}, S.ws)
        res["tactical_kappa_first_pair"] = {r: {bd: {kk: v[bd][kk]["kappa"] for kk in ("lateral", "longitudinal", "maneuver_5way_collapsed")} for bd in E.BANDS}
                                            for r, v in E.tactical_blocks({"A": S.met(arms[a]["s0"]), "B": S.met(arms[b]["s0"]), "HOLD": S.met("hold")}).items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args()
    man = json.load(open(a.manifest, encoding="utf-8"))
    if os.path.exists(a.out):
        sys.exit(f"REFUSED: {a.out} exists")
    if man.get("mech"):
        mj = json.load(open(man["mech"], encoding="utf-8"))
        man["mech_by_chassis"] = {}
        for k in (0, 1):
            rs = [r for r in mj["rows"] if r["chassis"] == k]
            man["mech_by_chassis"][str(k)] = None if (mj["label"] == "NOT-COMPUTABLE" or not rs) else all(r.get("pass") for r in rs)
    res = run(man, a.n_boot or man.get("n_boot", 2000))
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"outcome": res["outcome"], "chassis_labels": res["chassis_labels"]}, default=float)[:1500])


if __name__ == "__main__":
    main()
