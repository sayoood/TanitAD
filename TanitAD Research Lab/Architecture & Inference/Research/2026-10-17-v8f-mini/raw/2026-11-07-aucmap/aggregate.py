#!/usr/bin/env python3
"""aggregate.py -- PREREG_AUCMAP.md sections 8-10 and 12 from raw/AUCMAP_RESULT.json.  CEILING-TYPE where synthetic / logged level; T1-kin.
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I aggregate.py   -> raw/AGG.json, raw/FOUR_FAMILIES.md"""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np
import aucmap as A

R = json.load(open(f"{A.RAW}/AUCMAP_RESULT.json"))["tasks"]
LV = [str(a) for a in A.AUC_LEVELS]
ADM, CEIL = list(A.ADMISSIBLE), list(A.CEILING)
OUT = {"a_star": {}, "reach": {}, "diag": {}, "anchor": {}, "near": {}, "noskill": {}}


def nm(kind, lv, sc, lab, perm=False):
    return f"{kind}_{lv}_{sc}_{lab}{'_perm' if perm else ''}"


# ---- A* per (lever, scope, label, corner) ---------------------------------------------------------------------------------------
for lv in ADM + CEIL:
    for sc in ("ANCHOR", "ALL"):
        r = R[nm("syn", lv, sc, "GO")]
        OUT["a_star"][f"{lv}|{sc}|GO"] = {"FULL": A.a_star(r, "rate_FULL"), "P": A.a_star(r, "rate_P"), "interval_FULL": A.interval(r, "rate_FULL"),
                                         "rate_FULL": [r["levels"][l]["rate_FULL"] for l in LV], "rate_P": [r["levels"][l]["rate_P"] for l in LV],
                                         "realized_auc_mean": [float(np.mean(r["levels"][l]["realized_auc"])) for l in LV]}
for lv in ADM:
    r = R[nm("syn", lv, "ANCHOR", "KEEP")]
    OUT["a_star"][f"{lv}|ANCHOR|KEEP"] = {"FULL": A.a_star(r, "rate_FULL"), "P": A.a_star(r, "rate_P"), "rate_FULL": [r["levels"][l]["rate_FULL"] for l in LV], "rate_P": [r["levels"][l]["rate_P"] for l in LV]}
fulls = [OUT["a_star"][f"{lv}|ANCHOR|GO"]["FULL"] for lv in ADM]
OUT["A_star_any"] = min([f for f in fulls if f is not None], default=None)
OUT["ceiling_family_reaches_at_1.0"] = {f"{lv}|{sc}": OUT["a_star"][f"{lv}|{sc}|GO"]["rate_FULL"][-1] for lv in CEIL for sc in ("ANCHOR", "ALL")}
OUT["admissible_reach_at_1.0"] = {f"{lv}|{sc}": OUT["a_star"][f"{lv}|{sc}|GO"]["rate_FULL"][-1] for lv in ADM for sc in ("ANCHOR", "ALL")}

# ---- which predicate binds at AUC 1.0 (from the stored per-draw, per-q predicate lists; GUARD is None where M1P was False) -----------
for key in [nm("syn", lv, sc, "GO") for lv in ADM + CEIL for sc in ("ANCHOR", "ALL")]:
    r = R[key]["levels"]["1.0"]
    sh = np.array(r["share_mean"])
    m1p = np.array([[g is not None for g in row] for row in r["GUARD_q"]])
    guard = np.array([[bool(g) if g is not None else False for g in row] for row in r["GUARD_q"]])
    m1full = np.array(r["M1FULL_q"], bool)
    OUT["diag"][key] = {"max_share_mean": float(sh.max()), "q_at_max": float(A.Q_GRID[int(sh.mean(0).argmax())]),
                        "draws_with_M1P_at_some_q": float(m1p.any(1).mean()), "draws_with_M1FULL_at_some_q": float(m1full.any(1).mean()),
                        "draws_with_M1P_and_GUARD_at_some_q": float((m1p & guard).any(1).mean()),
                        "draws_with_M1FULL_and_GUARD_at_some_q": float((m1full & guard).any(1).mean()),
                        "smallest_q_with_M1P_draw0": next((float(A.Q_GRID[j]) for j in range(21) if m1p[0, j]), None)}

# ---- shuffled-label arm vs the no-skill line (C5) -------------------------------------------------------------------------------
perm = R[nm("syn", "LF100", "ANCHOR", "GO", True)]["levels"]
ns = R[nm("syn", "LF100", "ANCHOR", "GO")]["levels"]["0.5"]
pp = np.array(perm["1.0"]["share_mean"]).mean(0)
nn = np.array(ns["share_mean"]).mean(0)
OUT["noskill"] = {"perm_realized_auc_vs_true_label_mean": float(np.mean(perm["1.0"]["realized_auc"])), "perm_realized_auc_range": [float(np.min(perm["1.0"]["realized_auc"])), float(np.max(perm["1.0"]["realized_auc"]))],
                  "max_abs_gap_perm_vs_nominal0.5_curve": float(np.abs(pp - nn).max()), "perm_share_at_q0.5": float(pp[10]), "nominal0.5_share_at_q0.5": float(nn[10]),
                  "rate_FULL_perm": [perm[l]["rate_FULL"] for l in LV], "ungated_share_q1": float(pp[-1])}

# ---- anchor curves and the NEAR test ----------------------------------------------------------------------------------------------
for lv in ADM:
    a = R[nm("anchor", lv, "ANCHOR", "GO")]
    cur = a["curve"]
    shs = np.array([[c["share_s"]["0"], c["share_s"]["1"]] for c in cur])
    ade = np.array([[c["ade6_s"]["0"], c["ade6_s"]["1"]] for c in cur])
    OUT["anchor"][lv] = {"n_scope": a["n_scope"], "realized_auc_card_v0lt8": a["realized_auc"], "auc_v0lt6_card": a["auc_v0lt6_card"], "auc_v0lt8_card": a["auc_v0lt8_card"], "auc_v0lt8_p1pgo": a["auc_v0lt8_p1pgo"],
                         "any_CORNER_FULL": any(c["CORNER_FULL"] for c in cur), "any_CORNER_P": any(c["CORNER_P"] for c in cur), "any_M1P": any(c["M1P"] for c in cur), "any_M1FULL": any(c["M1FULL"] for c in cur),
                         "max_share_mean": float(np.mean(shs, 1).max()), "q_at_max_share": float(a["q"][int(np.mean(shs, 1).argmax())]),
                         "share_q1": shs[-1].tolist(), "ade6_q1": ade[-1].tolist(), "smallest_q_M1P": next((a["q"][j] for j, c in enumerate(cur) if c["M1P"]), None),
                         "share_s_by_q": shs.tolist(), "ade6_s_by_q": ade.tolist(), "guard_by_q": [c["GUARD"] for c in cur], "m1p_by_q": [c["M1P"] for c in cur], "m1full_by_q": [c["M1FULL"] for c in cur]}
    # NEAR: nominal level closest to the anchor's realized AUC, and 0.785
    tgt = a["realized_auc"]
    lev_near = min(A.AUC_LEVELS, key=lambda x: abs(x - tgt))
    res = {}
    for tag, lev in (("closest", lev_near), ("0.785", 0.785)):
        rec = R[nm("syn", lv, "ANCHOR", "GO")]["levels"][str(lev)]
        S = np.array(rec["share_s"])      # (draws, q, seed)
        D = np.array(rec["ade6"])
        frac = []
        for s in (0, 1):
            ins = []
            for j in range(21):
                lo_s, hi_s = np.percentile(S[:, j, s], [2.5, 97.5]); lo_d, hi_d = np.percentile(D[:, j, s], [2.5, 97.5])
                ins.append(bool(lo_s <= shs[j, s] <= hi_s and lo_d <= ade[j, s] <= hi_d))
            frac.append(float(np.mean(ins)))
        res[tag] = {"level": lev, "frac_inside_per_seed": frac, "NEAR": bool(min(frac) >= 0.75)}
    OUT["near"][lv] = res

json.dump(OUT, open(f"{A.RAW}/AGG.json", "w", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---- four families at the best q (beside; decides nothing).  best q = smallest q whose point share_closed (mean of seeds) >= 0.5 on the AUC-1.0 oracle gate, ANCHOR scope; else q = 1.
A.init()
rig = A.Rig(levers=ADM + CEIL)
y = A.labels_from(rig.v0, rig.g)["GO"]
scope = rig.v0 < 8.0
fam = A.SR.FAMILY
lines = ["# FOUR_FAMILIES (AUCMAP, CEILING-TYPE: oracle gate AUC 1.0 draw 0, ANCHOR scope v0<8; tier T1-kin; STRATEGIC PAUSED (PI ITEM 25); rows are X (gated) minus R (RIC0), delta [lo, hi], seed 0 | seed 1; estimator: paired episode-cluster bootstrap B=2000)"]
OUT4 = {}
for lv in ADM + CEIL:
    sc = A.make_score(y, 1.0, 7, 0)
    qbest = 1.0
    for q in A.Q_GRID:
        e = A.evaluate(rig, lv, A.select_top(sc, scope, q))
        if e["share_mean"] >= 0.5:
            qbest = q
            break
    e = A.evaluate(rig, lv, A.select_top(sc, scope, qbest), full=True)
    OUT4[lv] = {"q": qbest, "share_s": e["share_s"], "ade6_s": e["ade6_s"], "M1P": e["M1P"], "M1FULL": e["M1FULL"], "GUARD": e["GUARD"], "active_separated_worse": {s: e["rows_xr"][s]["_active_separated_worse"] for s in A.SEEDS}}
    lines.append(f"\n## {lv} ({'ADMISSIBLE heads level' if lv in ADM else 'CEILING-LEVEL logged level'}), q = {qbest} (smallest q with point share_closed >= 0.5 at AUC 1.0; 1.0 if none), share_closed {e['share_s'][0]:.3f} | {e['share_s'][1]:.3f}, M1P {e['M1P']}, M1FULL {e['M1FULL']}, GUARD {e['GUARD']}")
    lines.append("| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse (direction-aware) s0/s1 |\n|---|---|---|---|---|")
    for i, lab in enumerate(A.GD.LABELS):
        f = fam[lab]
        r0, r1 = e["rows_xr"][0][lab], e["rows_xr"][1][lab]
        w0, w1 = A.SR.worse(A.GD, lab, r0), A.SR.worse(A.GD, lab, r1)
        tag = "STRATEGIC PAUSED (beside)" if f == "STRATEGIC" else f
        lines.append(f"| {tag} | {lab} | {r0['delta']:+.4f} [{r0['lo']:+.4f}, {r0['hi']:+.4f}] | {r1['delta']:+.4f} [{r1['lo']:+.4f}, {r1['hi']:+.4f}] | {w0}/{w1} |")
open(f"{A.RAW}/FOUR_FAMILIES.md", "w", newline="\n").write("\n".join(lines) + "\n")
OUT["four_families_q"] = OUT4
json.dump(OUT, open(f"{A.RAW}/AGG.json", "w", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: OUT[k] for k in ("A_star_any", "ceiling_family_reaches_at_1.0", "admissible_reach_at_1.0", "noskill", "near")}, indent=1, default=str))
print(json.dumps(OUT["anchor"] and {lv: {k: v for k, v in d.items() if not k.endswith("_by_q")} for lv, d in OUT["anchor"].items()}, indent=1, default=str))
print(json.dumps(OUT["four_families_q"], indent=1, default=str)[:3000])
