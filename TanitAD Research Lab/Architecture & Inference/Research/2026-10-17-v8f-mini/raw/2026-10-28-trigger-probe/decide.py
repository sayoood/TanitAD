#!/usr/bin/env python3
"""decide.py -- applies the committed decision rule (PREREG_FROZEN.md 7) and the controls (8) to the banked results.  Writes raw/results_pstop.json and raw/results_pmove.json.
Two readings, both reported: PRIMARY = the pooled out-of-fold AUC exactly as frozen; POST-FREEZE = the mean of per-fold AUCs (found necessary by control C2, see RESULTS.md).
CUDA_VISIBLE_DEVICES=-1  python -I decide.py"""
from __future__ import annotations

import json
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402
import run_probe as RP  # noqa: E402


def controls(res_v, metric):
    """C1 (shuffle: CI covers 0.5), C2 (donor-episode E: CI covers 0.5 and AUC <= true E), C3 (leak >= 0.99)."""
    g = (lambda a: a) if metric == "pooled" else (lambda a: a["foldmean"])
    A = res_v["arms"]
    out = {}
    for k in ("C1:shuffle:V+E", "C1:shuffle:E"):
        a = g(A[k])
        out[k] = {"auc": a["auc"], "lo": a["lo"], "hi": a["hi"], "pass": bool(a["lo"] <= 0.5 <= a["hi"])}
    a, e = g(A["C2:donor:E"]), g(A["E"])
    out["C2:donor:E"] = {"auc": a["auc"], "lo": a["lo"], "hi": a["hi"], "auc_true_E": e["auc"], "pass": bool(a["lo"] <= 0.5 <= a["hi"] and a["auc"] <= e["auc"]),
                         "ci_covers_0.5": bool(a["lo"] <= 0.5 <= a["hi"]), "hi_minus_0.5": a["hi"] - 0.5}
    for k in ("C3:leak:E", "C3:leak:V+E"):
        a = g(A[k])
        out[k] = {"auc": a["auc"], "pass": bool(a["auc"] >= 0.99)}
    out["all_pass"] = bool(all(v["pass"] for v in out.values()))
    return out


def rule(dL, dM, z_ge_v, z_auc, v_auc):
    """The committed rule.  dL / dM: paired dAUC(V+E - E) dicts (linear / MLP)."""
    info = dL["delta"] >= 0.05 and dL["lo"] > 0.0
    if info:
        return ("D1" if z_ge_v else "D2"), f"dAUC_lin {dL['delta']:.4f} CI [{dL['lo']:.4f},{dL['hi']:.4f}] >= 0.05 and excludes 0; AUC(Z+E) {z_auc:.4f} vs AUC(V+E) {v_auc:.4f} (threshold {v_auc - 0.02:.4f})"
    if dL["delta"] < 0.02 and dM["delta"] < 0.02:
        return "D3", f"dAUC_lin {dL['delta']:.4f}, dAUC_mlp {dM['delta']:.4f}, both < 0.02"
    if dM["delta"] >= 0.05 and dM["lo"] > 0.0:
        return "INCONCLUSIVE-NONLINEAR", f"linear dAUC {dL['delta']:.4f} < 0.05 but MLP dAUC {dM['delta']:.4f} CI [{dM['lo']:.4f},{dM['hi']:.4f}] clears"
    return "INCONCLUSIVE", f"dAUC_lin {dL['delta']:.4f} CI [{dL['lo']:.4f},{dL['hi']:.4f}], dAUC_mlp {dM['delta']:.4f}"


def main():
    ev = json.load(open(f"{RAW}/results_eval141.json")) if os.path.exists(f"{RAW}/results_eval141.json") else None
    for pop in ("pstop", "pmove"):
        d = RP.load_pop(pop)
        y, clip, outer = d["y"], d["clip"], d["outer"]
        counts = L.boot_counts(clip)
        rv = json.load(open(f"{RAW}/res_{pop}_V_linear.json"))
        rz = json.load(open(f"{RAW}/res_{pop}_Z_linear.json"))
        mv = json.load(open(f"{RAW}/res_{pop}_V_mlp.json"))
        mz = json.load(open(f"{RAW}/res_{pop}_Z_mlp.json"))
        ov, oz = np.load(f"{RAW}/oof_{pop}_V_linear.npz"), np.load(f"{RAW}/oof_{pop}_Z_linear.npz")
        mov, moz = np.load(f"{RAW}/oof_{pop}_V_mlp.npz"), np.load(f"{RAW}/oof_{pop}_Z_mlp.npz")
        assert np.array_equal(ov["y"], y) and np.array_equal(oz["y"], y)
        out = {"pop": pop, "n": int(len(y)), "n_go": int(y.sum()), "n_clips": int(len(np.unique(clip))), "evidence_class": "MEASURED (artifacts raw/res_*.json, raw/oof_*.npz)",
               "tier": "T0-class diagnostic on logged inputs (probe of representations, not driving performance)",
               "estimator": "episode(clip)-cluster bootstrap B=2000 seed 0: answers 'another draw of episodes', blind to training variance (H-ESTIM-SEED-1); Z comparison rests on ONE checkpoint",
               "arms_V_linear": rv["arms"], "arms_Z_linear": rz["arms"], "arms_V_mlp": mv["arms"], "arms_Z_mlp": mz["arms"],
               "paired_V_linear": rv["paired"], "paired_Z_linear": rz["paired"], "paired_V_mlp": mv.get("paired"), "paired_Z_mlp": mz.get("paired")}
        out["controls"] = {m: controls(rv, m) for m in ("pooled", "foldmean")}
        z_vs_v = {}
        for fam, (ovv, ozz) in (("linear", (ov, oz)), ("mlp", (mov, moz))):
            ko, kz = ("V+E", "Z+E")
            z_vs_v[fam] = {"pooled": L.dauc_ci(ozz[kz], ovv[ko], y, clip, counts=counts),
                           "foldmean": L.dauc_ci_fold(ozz[kz], ovv[ko], y, clip, outer, counts=counts)}
        out["Z+E_minus_V+E"] = z_vs_v
        ver = {}
        for metric in ("pooled", "foldmean"):
            sel = (lambda a: a) if metric == "pooled" else (lambda a: a["foldmean"])
            dL = rv["paired"]["V+E_minus_E"] if metric == "pooled" else rv["paired"]["V+E_minus_E"]["foldmean"]
            dM = mv["paired"]["V+E_minus_E"] if metric == "pooled" else mv["paired"]["V+E_minus_E"]["foldmean"]
            v_auc, z_auc = sel(rv["arms"]["V+E"])["auc"], sel(rz["arms"]["Z+E"])["auc"]
            vd, why = rule(dL, dM, z_auc >= v_auc - 0.02, z_auc, v_auc)
            ver[metric] = {"verdict": vd, "why": why, "controls_pass": out["controls"][metric]["all_pass"],
                           "formal_status": ("valid" if out["controls"][metric]["all_pass"] else "VOID under the frozen text: a control failed (see controls), no verdict may be drawn from this reading")}
            # caveats (cannot change a verdict): stronger ego baseline E_H; secondary label GO24; MLP reading of the Z comparison
            dEH = rv["paired"]["V+E_minus_E_H"] if metric == "pooled" else rv["paired"]["V+E_minus_E_H"]["foldmean"]
            dGO = rv["paired"]["GO24:V+E_minus_E"] if metric == "pooled" else rv["paired"]["GO24:V+E_minus_E"]["foldmean"]
            ver[metric]["caveats"] = {"dAUC(V+E - E_H)": {k: dEH[k] for k in ("delta", "lo", "hi")}, "dAUC_GO24(V+E - E)": {k: dGO[k] for k in ("delta", "lo", "hi")},
                                      "dAUC_mlp(V+E - E)": {k: dM[k] for k in ("delta", "lo", "hi")},
                                      "Z+E_minus_V+E_mlp": {k: z_vs_v["mlp"][metric][k] for k in ("delta", "lo", "hi")}}
        out["decision"] = ver
        if ev is not None:
            P = ev["pops"][pop]
            lin = lambda k: P["arms"][f"linear:{k}"]
            memo = lin("Z+E")["auc"] < lin("V+E")["auc"] - 0.02
            out["eval141_confirmation"] = {"n": P["n"], "n_go": P["n_go"], "n_episodes": P["n_episodes"],
                                           "linear": {k: {a: lin(k)[a] for a in ("auc", "lo", "hi")} for k in ("E", "V", "V+E", "Z", "Z+E")},
                                           "mlp": {k: {a: P["arms"][f"mlp:{k}"][a] for a in ("auc", "lo", "hi")} for k in ("E", "V", "V+E", "Z", "Z+E")},
                                           "paired_linear": {k: {a: v[a] for a in ("delta", "lo", "hi")} for k, v in P["paired"].items() if k.startswith("linear")},
                                           "memorisation_suspect (eval AUC(Z+E) < AUC(V+E) - 0.02, linear)": bool(memo), "subsets": P.get("subsets")}
        json.dump(out, open(f"{RAW}/results_{pop}.json", "w"), indent=1, default=float)
        print(pop, json.dumps({m: (v["verdict"], v["formal_status"][:5], v["why"]) for m, v in ver.items()}, indent=1))
        print(" controls pooled:", {k: (round(v["auc"], 4), v["pass"]) for k, v in out["controls"]["pooled"].items() if isinstance(v, dict)})
        print(" controls foldmean:", {k: (round(v["auc"], 4), v["pass"]) for k, v in out["controls"]["foldmean"].items() if isinstance(v, dict)})
        print(" Z+E - V+E:", {f: {m: round(z_vs_v[f][m]["delta"], 4) for m in z_vs_v[f]} for f in z_vs_v})


if __name__ == "__main__":
    main()
