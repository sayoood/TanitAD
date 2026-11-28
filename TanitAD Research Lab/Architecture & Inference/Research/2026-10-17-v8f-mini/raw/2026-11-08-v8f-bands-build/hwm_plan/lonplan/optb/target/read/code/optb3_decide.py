"""OPTB3 deciding read (AMEND_OPTB3_TARGET sections 3-4; everything else is AMEND_OPTB). Tier T0 (oracle goal: a world-model diagnostic, never driving performance). CPU only, zero training.

Reuses the Amendment 2 scorer unchanged (optb_decide.Scorer: hwm_eval reader, band_metrics, E.contrast = paired episode-cluster bootstrap B 2,000 seed 0 over the 994 windows / 71 episodes).
It answers ANOTHER DRAW OF EPISODES only (plans and checkpoints fixed). F_T,k = max - min of the three FT-NULL point means = another TRAINING run. F_I = max |seed 1 - seed 0| over every arm
present (FTT, FTRSP, FTNULL, R35 baselines) = another INFERENCE run. Floor_k = max(F_T,k, F_I).
Per chassis k (one lever seed per chassis, so the statistic is the single paired delta, not a two-seed mean):
  FTT15_sk - FTNULL_sk and FTT125_sk - FTNULL_sk on band 0-2 s acc_mae (LONGITUDINAL) and yaw_mae (LATERAL); attribution FTT15_sk - FTT125_sk.
  BETTER: CI wholly below 0 and |delta| > Floor_k. WORSE: CI wholly above 0 and delta > Floor_k.
Mechanism: IA-3P-K gain (reads/ia3pk_<tag>.json). MECH-RAISED at chassis k: FTT15 gain above its FT-NULL pair's, CIs disjoint, gap above F_T,k of the IA-3P-K gain (max - min of the three FT-NULL point gains).
TARGET-READ (beside, no bar): frozen-probe cycle gain of the arm on the eval clips and its rsp/L1 trajectory from train_log.jsonl.
Labels: section 4 of the card (TGT-GAIN / TGT-GAIN-UNATTRIBUTED / TGT-MECH-ONLY / TGT-NULL (i)/(ii) / TGT-HARM / CHASSIS-DISAGREE), per chassis then combined.
Usage: python optb3_decide.py --root F:/TanitAD-scratch/optb3_read/out --out OUT.json [--n-boot 2000] [--alias FTT15_s0=FTRSP_s0,...] (alias = test/smoke only)
"""
import argparse, json, os, sys, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, "F:/TanitAD-scratch/actrsp/optb/read")
import optb_decide as OD                      # Amendment 2 scorer, imported not copied

A2_OUT = "F:/TanitAD-scratch/actrsp/optb/readchain/out"
A2_PLANS = "F:/TanitAD-scratch/actrsp/optb/plans"
PULLS = "C:/Users/Admin/t2v8f_pulls"
SCORE = "F:/TanitAD-scratch/actrsp/lonplan/score"
Q, BAND = OD.Q, OD.BAND
TARGET = {"FTT15": 1.5, "FTT125": 1.25}
NEW = {k: {"T15": f"FTT15_s{k}", "T125": f"FTT125_s{k}", "NULL": f"FTNULL_s{k}", "nulls": [f"FTNULL_s{k}", f"FTNULL_s{k + 2}", f"FTNULL_s{k + 4}"]} for k in (0, 1)}
FAVOUR = ["TGT-GAIN", "TGT-GAIN-UNATTRIBUTED", "TGT-MECH-ONLY", "TGT-NULL", "TGT-HARM"]          # most to least favourable
NEXT = {"TGT-GAIN": "g* = 1.75 at both chassis plus FTT15 at seeds 2 / 3 (four runs, PI go); this WM goes to the HWM-PLAN read",
        "TGT-GAIN-UNATTRIBUTED": "the term's presence, not the raised target, may carry it. Next: a target sweep 1.25 / 1.5 / 1.75 at a second seed (PI go)",
        "TGT-MECH-ONLY": "the response is raised and the plan did not improve: LON-PLAN prior-regularised planner (zero training, as OPTB-MECH-ONLY)",
        "TGT-NULL": "(i) the term reached g* on its own readout (TARGET-READ within 0.1 of g*) but IA-3P-K did not move: align the term's readout with the eval readout. (ii) otherwise: lambda 3.0 at g* = 1.5",
        "TGT-HARM": "the response-gain lever is closed on this chassis family; next: LON-PLAN (planner side)",
        "CHASSIS-DISAGREE": "reported as such; the less favourable label's next step applies"}


# ------------------------------------------------------------------------------------------------------------- pure decision logic (tested without data)
def bw(d, floor):
    """d {delta, lo, hi}; positive delta = worse (lower is better)."""
    return {"better": bool(d["hi"] < 0 and abs(d["delta"]) > floor), "worse": bool(d["lo"] > 0 and d["delta"] > floor)}


def apply_mm_rule_common_quantity(labels):
    """MM ruling 2026-11-19, frozen before any OPTB3 read data: section 4's 'On that quantity, FTT15 - FTT125 has its CI wholly below 0 at both chassis'
    means ONE deciding quantity common to both chassis. Two chassis labelled TGT-GAIN on different quantities read TGT-GAIN-UNATTRIBUTED."""
    if all(labels[k]["label"] == "TGT-GAIN" for k in (0, 1)):
        common_attr = set(labels[0].get("attributed_quantities", [])) & set(labels[1].get("attributed_quantities", []))
        if not common_attr:
            for k in (0, 1):
                labels[k] = dict(labels[k], label="TGT-GAIN-UNATTRIBUTED",
                                 notes=list(labels[k].get("notes", [])) + ["MM ruling: no deciding quantity is BETTER and attributed (FTT15 - FTT125 CI below 0) at BOTH chassis"])
    return labels


def chassis_label(c):
    """c: {"T15": {q: bw}, "attr": {q: {delta,lo,hi}}, "mech_raised": bool|None, "band_met_lost_T15": bool|None, "target_read_within_0p1": bool|None, "controls_hold": bool|None}."""
    out = {"notes": []}
    qs = list(c["T15"])
    if any(c["T15"][q]["worse"] for q in qs) or c.get("band_met_lost_T15"):
        out["label"] = "TGT-HARM"
        return out
    better = [q for q in qs if c["T15"][q]["better"]]
    attributed = [q for q in better if c["attr"][q]["hi"] < 0]
    out["better_quantities"], out["attributed_quantities"] = better, attributed
    if better:
        if attributed and c.get("mech_raised") is True:
            out["label"] = "TGT-GAIN"
        else:
            out["label"] = "TGT-GAIN-UNATTRIBUTED"
            if attributed and c.get("mech_raised") is not True:
                out["notes"].append("attributed on the quantity but MECH-RAISED does not hold: the card's GAIN row needs both; the card has no separate row, labelled by the card's second row")
        return out
    if c.get("mech_raised") is True:
        out["label"] = "TGT-MECH-ONLY"
    elif c.get("mech_raised") is False:
        out["label"] = "TGT-NULL"
        out["null_variant"] = "(i)" if c.get("target_read_within_0p1") is True else ("(ii)" if c.get("target_read_within_0p1") is False else "UNDETERMINED")
        if c.get("controls_hold") is not True:
            out["notes"].append("NULL requires the controls holding: UNVERIFIED here (controls_hold is not true)")
    else:
        out["label"] = "MECH-NOT-COMPUTABLE"
        out["notes"].append("the IA-3P-K mechanism read is missing")
    return out


def overall_label(per):
    if set(per) != {0, 1}:
        return {"label": "INCOMPLETE", "why": f"chassis present {sorted(per)}; both are required", "next": None}
    labs = {k: v["label"] for k, v in per.items()}
    if "TGT-HARM" in labs.values():
        return {"label": "TGT-HARM", "per_chassis": labs, "next": NEXT["TGT-HARM"], "note": "HARM at either chassis (card wording)"}
    if len(set(labs.values())) == 1 and labs[0] in FAVOUR:
        return {"label": labs[0], "per_chassis": labs, "next": NEXT[labs[0]], **({"null_variants": {k: v.get("null_variant") for k, v in per.items()}} if labs[0] == "TGT-NULL" else {})}
    known = [l for l in labs.values() if l in FAVOUR]
    worst = max(known, key=FAVOUR.index) if known else None
    return {"label": "CHASSIS-DISAGREE", "per_chassis": labs, "less_favourable": worst, "next": NEXT["CHASSIS-DISAGREE"] + (": " + NEXT[worst] if worst else "")}


def mech_raised(g_arm, g_null, floor):
    """g_*: {point, ci95:[lo,hi]}. Gap above 0, CIs disjoint (arm lo > null hi), gap above F_T,k."""
    gap = g_arm["point"] - g_null["point"]
    c = {"above": gap > 0, "ci_disjoint": g_arm["ci95"][0] > g_null["ci95"][1], "gap_above_floor": (floor is not None and gap > floor)}
    return {"gap": gap, "F_T_k": floor, **c, "pass": all(c.values())}


# ------------------------------------------------------------------------------------------------------------- paths and inputs
class Paths:
    def __init__(self, root, alias):
        self.root, self.alias = root, alias

    def real(self, t):
        return self.alias.get(t, t)

    def is_new(self, t):
        return t.startswith("FTT") and t not in self.alias

    def plans(self, t):
        return f"{self.root}/plans/{t}" if self.is_new(t) else f"{A2_PLANS}/{self.real(t)}"

    def out(self, t):
        return self.root if self.is_new(t) else A2_OUT

    def read(self, t):
        return f"{self.out(t)}/reads/ia3pk_{self.real(t)}.json"

    def evidence(self, t):
        return f"{self.out(t)}/evidence/{self.real(t)}.json"

    def bandmet(self, t):
        return f"{self.out(t)}/bandmet/{self.real(t)}.json"

    def bank(self, t):
        return f"{PULLS}/optb3_{t}" if self.is_new(t) else f"{PULLS}/optb_{self.real(t)}"


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def rsp_traj(log):
    v = []
    for ln in open(log, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            if "wm/rsp/L1" in r:
                v.append((r.get("step"), r["wm/rsp/L1"]))
    if not v:
        return None
    last = [x for _, x in v[-10:]]
    return {"n_logged": len(v), "first": {"step": v[0][0], "L1": v[0][1]}, "last": {"step": v[-1][0], "L1": v[-1][1]}, "mean_last_10": sum(last) / len(last)}


def sig(p):
    import numpy as np
    z = np.load(p, allow_pickle=False)
    return [hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in ("eid", "now")]


def build(a):
    P = Paths(a.root, dict(x.split("=") for x in a.alias.split(",")) if a.alias else {})
    new_tags = [NEW[k][g] for k in (0, 1) for g in ("T15", "T125")]
    null_tags = [t for k in (0, 1) for t in NEW[k]["nulls"]]
    rsp_tags = [f"FTRSP_s{i}" for i in range(4)]
    arms = {}
    for t in new_tags + null_tags + rsp_tags:
        d = P.plans(t)
        p0, p1 = f"{d}/plans_hwm_oracle_s0.npz", f"{d}/plans_hwm_oracle_s1.npz"
        if os.path.isfile(f"{d}/DONE") and os.path.isfile(p0) and os.path.isfile(p1):
            arms[t] = {"s0": p0, "s1": p1, "dir": d}
    base = {f"R35_s{k}": {"s0": f"{SCORE}/A0_s{k}/plans_hwm_oracle_s0.npz", "s1": f"{SCORE}/A0_s{k}/plans_hwm_oracle_s1.npz", "dir": f"{SCORE}/A0_s{k}"} for k in (0, 1)}
    return P, arms, base, new_tags


def run(a):
    P, arms, base, new_tags = build(a)
    missing = [t for t in new_tags + [x for k in (0, 1) for x in NEW[k]["nulls"]] if t not in arms]
    if missing:
        raise SystemExit(f"INCOMPLETE: plans missing for {missing}")
    windows = f"{SCORE}/A0_s0/windows.npz"
    ref = sig(windows)
    wid = {t: sig(v["dir"] + "/windows.npz") == ref for t, v in {**arms, **base}.items() if os.path.isfile(v["dir"] + "/windows.npz")}
    S = OD.Scorer(windows, a.n_boot)
    E = S.E
    allp = {**arms, **base}
    F_I = {}
    for fam, m in Q:
        d = [abs(S.pm(v["s1"], BAND, m) - S.pm(v["s0"], BAND, m)) for v in allp.values()]
        F_I[m] = {"value": max(d), "n_arms_with_both_seeds": len(d), "arms": sorted(allp)}
    rows, per, floors, mech_rows, target_read, traj, ctl = [], {}, {}, [], {}, {}, {}
    for k in (0, 1):
        N = NEW[k]
        F_T = {}
        for fam, m in Q:
            pts = [S.pm(arms[t]["s0"], BAND, m) for t in N["nulls"]]
            F_T[m] = {"value": max(pts) - min(pts), "points": dict(zip(N["nulls"], pts))}
        floors[k] = {m: {"F_T": F_T[m], "F_I": F_I[m], "Floor": max(F_T[m]["value"], F_I[m]["value"])} for _, m in Q}
        c = {"T15": {}, "T125": {}, "attr": {}, "floor": {}}
        for fam, m in Q:
            fl = floors[k][m]["Floor"]
            c["floor"][m] = fl
            for g in ("T15", "T125"):
                r = S.con(S.met(arms[N[g]]["s0"])[BAND][m], S.met(arms[N["NULL"]]["s0"])[BAND][m], fam, m)
                c[g][m] = bw(r, fl)
                rows.append(dict(kind=f"deciding_pair_{g}_vs_FTNULL", chassis=k, arm=N[g], vs=N["NULL"], family=fam, band=BAND, metric=m, **r, **c[g][m]))
            r = S.con(S.met(arms[N["T15"]]["s0"])[BAND][m], S.met(arms[N["T125"]]["s0"])[BAND][m], fam, m)
            c["attr"][m] = r
            rows.append(dict(kind="attribution_FTT15_minus_FTT125", chassis=k, arm=N["T15"], vs=N["T125"], family=fam, band=BAND, metric=m, **r))
            for g in ("T15", "T125"):
                rows.append(dict(kind="absolute_vs_hold", chassis=k, arm=N[g], family=fam, band=BAND, metric=m, **S.con(S.met(arms[N[g]]["s0"])[BAND][m], S.met("hold")[BAND][m], fam, m)))
        # mechanism
        R = {t: jl(P.read(t)) for t in N["nulls"] + [N["T15"], N["T125"]]}
        gp = [R[t]["ia3pk_gain_L1_2s_lon"]["point"] for t in N["nulls"]]
        FTm = max(gp) - min(gp)
        for g in ("T15", "T125"):
            mech_rows.append({"chassis": k, "arm": N[g], "vs": N["NULL"], "gain_arm": R[N[g]]["ia3pk_gain_L1_2s_lon"], "gain_null": R[N["NULL"]]["ia3pk_gain_L1_2s_lon"], **mech_raised(R[N[g]]["ia3pk_gain_L1_2s_lon"], R[N["NULL"]]["ia3pk_gain_L1_2s_lon"], FTm)})
        c["mech_raised"] = [r for r in mech_rows if r["arm"] == N["T15"]][0]["pass"]
        for g in ("T15", "T125"):
            tr = R[N[g]].get("target_read_cycle_gain_frozen_probe")
            gs = TARGET["FTT15" if g == "T15" else "FTT125"]
            target_read[N[g]] = {"g_star": gs, "cycle_gain_frozen_probe": tr, "within_0p1_of_g_star": (abs(tr["point"] - gs) <= 0.1) if tr else None, "note": "beside, no bar"}
            traj[N[g]] = rsp_traj(P.bank(N[g]) + "/train_log.jsonl")
        c["target_read_within_0p1"] = target_read[N["T15"]]["within_0p1_of_g_star"]
        # controls (computed from the banked files): C2 / C3 / C5 per arm, BAND-MET per arm
        for g in ("T15", "T125"):
            ev, bm = jl(P.evidence(N[g])), jl(P.bandmet(N[g]))
            ctl[N[g]] = {"c2_hold": ev.get("c2_hold"), "c3_hold": ev.get("c3_hold"), "c5_hold": ev.get("c5_hold"), "band_met_lost_levels": bm.get("lost_levels")}
        c["band_met_lost_T15"] = bool(ctl[N["T15"]]["band_met_lost_levels"])
        c["controls_hold"] = all(all(ctl[N[g]][x] is True for x in ("c2_hold", "c3_hold", "c5_hold")) for g in ("T15", "T125"))
        per[k] = c
    # the full families grid, per band, never pooled
    for k in (0, 1):
        N = NEW[k]
        for g in ("T15", "T125"):
            refs = [("pair_FTNULL", arms[N["NULL"]]["s0"]), ("HOLD", "hold"), ("R35", base[f"R35_s{k}"]["s0"])] + ([("FTT125", arms[N["T125"]]["s0"])] if g == "T15" else [])
            for fam, mets in E.FAMILIES.items():
                for b in E.BANDS:
                    for m in mets + (E.BESIDE if fam == "LATERAL" else ()):
                        for rn, rs_ in refs:
                            rows.append(dict(kind="grid", chassis=k, arm=N[g], ref=rn, family=fam, band=b, metric=m, reference_only=(rn == "R35"), **S.con(S.met(arms[N[g]]["s0"])[b][m], S.met(rs_)[b][m], fam, m)))
    labels = {k: chassis_label(per[k]) for k in (0, 1)}
    common = sorted(set(labels[0].get("better_quantities", [])) & set(labels[1].get("better_quantities", [])))
    lenient = {"chassis_labels": {str(k): v["label"] for k, v in labels.items()}, "outcome": overall_label(labels)}
    labels = apply_mm_rule_common_quantity(labels)
    res = {"tier": "T0", "evidence": "MEASURED when run on real arm plans", "n_windows": int(S.ws.n), "n_episodes": int(len(set(S.ws.eid))), "window_identity_eid_now": wid, "alias_test_only": a.alias or None,
           "estimators": {"interval": "paired episode-cluster bootstrap (hwm_eval.contrast), B %d seed 0: another draw of EPISODES only" % a.n_boot,
                          "F_T": "max - min of the three FT-NULL point means of the chassis: another TRAINING run", "F_I": "max |seed 1 - seed 0| over every arm present: another INFERENCE run", "Floor_k": "max(F_T,k, F_I)",
                          "mechanism_interval": "episode-cluster bootstrap over the 141 eval clips, B 2,000, seed 0 (ia3pk_read): another draw of EPISODES, checkpoint and probe form fixed"},
           "floors": {str(k): v for k, v in floors.items()}, "decision_inputs": {str(k): v for k, v in per.items()}, "mech_rows": mech_rows, "target_read": target_read, "rsp_L1_trajectory": traj, "controls": ctl,
           "chassis_labels": {str(k): v for k, v in labels.items()}, "outcome": overall_label(labels),
           "attribution_reading": {"label_uses": "MM_RULING_OPTB3_DECIDE.md (frozen before data): 'that quantity' = ONE deciding quantity common to both chassis; TGT-GAIN needs a quantity on which FTT15 is BETTER and FTT15 - FTT125 has its CI wholly below 0 at BOTH chassis, else TGT-GAIN-UNATTRIBUTED",
                                   "common_better_quantities_both_chassis": common,
                                   "common_attributed_quantities_both_chassis": sorted(set(labels[0].get("attributed_quantities", [])) & set(labels[1].get("attributed_quantities", []))),
                                   "per_chassis_reading_beside_decides_nothing": lenient},
           "ftt125_line": {str(k): per[k]["T125"] for k in (0, 1)},
           "tactical_not_computed": {"goal_setting": E.NOT_COMPUTED.get("TACTICAL.goal_setting")}, "not_computed": E.NOT_COMPUTED,
           "strategic_note": "REPORTED IN FULL on the 6 s band (see `strategic`), structural n/a items named with reasons; never a pass; no outcome is decided on a STRATEGIC cell", "rows": rows}
    a0 = NEW[0]
    res["strategic"] = E.strategic_block({"A": S.met(arms[a0["T15"]]["s0"]), "B": S.met(arms[a0["NULL"]]["s0"]), "HOLD": S.met("hold")}, S.ws)
    res["tactical_kappa_first_pair"] = {r: {bd: {kk: v[bd][kk]["kappa"] for kk in ("lateral", "longitudinal", "maneuver_5way_collapsed")} for bd in E.BANDS}
                                        for r, v in E.tactical_blocks({"A": S.met(arms[a0["T15"]]["s0"]), "B": S.met(arms[a0["NULL"]]["s0"]), "HOLD": S.met("hold")}).items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--alias", default=None)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"REFUSED: {a.out} exists")
    res = run(a)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"outcome": res["outcome"], "chassis_labels": res["chassis_labels"]}, default=float)[:2000])


if __name__ == "__main__":
    main()
