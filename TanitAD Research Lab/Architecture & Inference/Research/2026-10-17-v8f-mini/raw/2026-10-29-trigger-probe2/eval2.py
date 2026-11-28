#!/usr/bin/env python3
"""eval2.py -- eval-141 confirmation, NO refit on eval (PREREG_FROZEN.md 9).  Probes are fitted ONCE on all 12,000 sampled TRAIN windows of the population (C / alpha by the inner CV, PCA on all of them) and applied unchanged
to the eval-141 windows of that population.  The comparators E, V+E, Z+E are RE-FITTED here (deterministic: same data, same inner CV) and their eval AUC is checked against the previous package's banked results_eval141.json (control EV0).
    python -I eval2.py --stage fit --pop pstop|pmove --fam linear|mlp [--budget-s 170]     banks raw/eval2_scores_<pop>_<fam>.npz  (resumable)
    python -I eval2.py --stage summ                                                         raw/results_eval2.json
Eval AUC is the single-fit AUC; CI = episode-cluster bootstrap by eval episode (B = 2000, seed 0); paired differences on the same resample.  CUDA_VISIBLE_DEVICES=-1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np

import common2 as C  # noqa: E402

L = C.L
ARMS_LIN = ("E", "V+E", "Z+E", "N5", "N5+E", "NL", "NL+E", "L+E", "Znn", "Znn+E", "Znl+E", "V4+E", "Vdiff+E")
ARMS_MLP = ("E", "V+E", "Z+E", "N5+E", "NL+E", "Znn+E", "Znl+E", "V4+E", "Vdiff+E")
PAIRS = (("N5+E", "E"), ("NL+E", "E"), ("Znn+E", "E"), ("Znn", "E"), ("Z+E", "Znn+E"), ("Znn+E", "V+E"), ("Znl+E", "E"), ("Z+E", "Znl+E"), ("V4+E", "V+E"), ("Vdiff+E", "V+E"), ("V4+E", "E"), ("Vdiff+E", "E"),
         ("V+E", "E"), ("Z+E", "V+E"), ("Z+E", "E"), ("Z+E", "NL+E"))


def pop_sel(ev, pop):
    return (ev["v0"] < 0.5) if pop == "pstop" else ((ev["v0"] >= 0.5) & (ev["v0"] < 6.0))


def proj(X, comp):
    return ((np.asarray(X, np.float32) - comp["mu"]) @ comp["comp"].T).astype(np.float64)


def eval_inputs(pop):
    """-> (train design blocks, eval design blocks, ye, eide, d): dicts name -> matrix, for every raw feature block."""
    ev = np.load(f"{C.OLDRAW}/eval141_features.npz", allow_pickle=False)
    sel = pop_sel(ev, pop)
    d = C.load_pop(pop)
    A = C.load_windows()
    Jt = C.load_join_npz(f"{C.OLDSCR}/join_train_rebuilt.npz")
    Je = C.load_join_npz(C.EVAL_JOIN)
    clips_tr = [str(c) for c in A["clips"]]
    navt, limt, _, _ = C.join_nav(Jt, clips_tr, d["clip"], d["t"])
    ix = {c: i for i, c in enumerate(Je["clip"].tolist())}
    ci = np.array([ix[str(n)] for n in ev["clip"][sel]])
    row = Je["off"][ci] + ev["t0"][sel]
    nave = Je["nav"][row].astype(np.float64)
    lime = np.concatenate([Je["limit"][ci], Je["limit_ok"][ci][:, None].astype(np.float32)], 1).astype(np.float64)
    tr, te = {"E": d["E"]}, {"E": ev["E"][sel].astype(np.float64)}
    tr["N5"], te["N5"] = navt, nave
    tr["NL"], te["NL"] = np.concatenate([navt, limt], 1), np.concatenate([nave, lime], 1)
    tr["L"], te["L"] = limt, lime
    pv, pz = np.load(f"{C.OLDRAW}/pca_V_{pop}.npz")["S"][5].astype(np.float64), np.load(f"{C.OLDRAW}/pca_Z_{pop}.npz")["S"][5].astype(np.float64)
    tr["V"], tr["Z"] = pv, pz
    te["V"] = proj(np.load(f"{C.OLDSCR}/V_eval141.npy")[sel], np.load(f"{C.OLDSCR}/pcacomp_V_{pop}_all.npz"))
    te["Z"] = proj(ev["Z"][sel], np.load(f"{C.OLDSCR}/pcacomp_Z_{pop}_all.npz"))
    for src in ("Znn", "Znl", "V4", "Vdiff"):
        pf = f"{C.RAW}/pca2_{src}_{pop}.npz"
        if not os.path.exists(pf):
            continue
        tr[src] = np.load(pf)["S"][5].astype(np.float64)
        comp = np.load(f"{C.SCR}/pcacomp2_{src}_{pop}_all.npz")
        if src in ("Znn", "Znl"):
            te[src] = proj(np.load(f"{C.SCR}/{src}_eval141.npy")[sel], comp)
        else:
            F4 = np.load(f"{C.SCR}/V4frames_eval141.npy")[sel].astype(np.float32)            # [n, 4, 40960]: t-3, t-2, t-1, t
            raw = F4.reshape(len(F4), -1) if src == "V4" else (F4[:, 3] - F4[:, 0])
            te[src] = proj(raw, comp)
    return tr, te, ev["go"][sel].astype(int), ev["eid"][sel], d, ev, sel


def base_of(name):
    return name[:-2] if name.endswith("+E") else name


def design(name, blocks, E):
    """'E' -> E; 'X+E' -> [X ++ E]; 'X' -> X."""
    if name == "E":
        return blocks["E"]
    return np.concatenate([blocks[name[:-2]], E], 1) if name.endswith("+E") else blocks[name]


def stage_fit(pop, fam, budget_s):
    t0 = time.time()
    C.require_ram(1500, "eval2")
    tr, te, ye, eide, d, ev, sel = eval_inputs(pop)
    names = ARMS_LIN if fam == "linear" else ARMS_MLP
    sp = f"{C.RAW}/eval2_scores_{pop}_{fam}.npz"
    sc = dict(np.load(sp)) if os.path.exists(sp) else {}
    hpf = f"{C.RAW}/eval2_hp_{pop}_{fam}.json"
    hps = json.load(open(hpf)) if os.path.exists(hpf) else {}
    for nm in names:
        if nm.replace("+", "_p_") in sc:
            continue
        if base_of(nm) not in tr or base_of(nm) not in te:
            print("SKIP (features not ready)", nm, flush=True)
            continue
        Xtr, Xte = design(nm, tr, tr["E"]), design(nm, te, te["E"])
        pred, hp, _ = L.fit_probe(Xtr, d["y"], d["inner"], fam)
        s = pred(Xte)
        sc[nm.replace("+", "_p_")] = s
        hps[nm] = hp
        np.savez(sp, **sc)
        json.dump(hps, open(hpf, "w"))
        print(f"eval2 {pop} {fam} {nm} hp {hp}  {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > budget_s:
            print("time box: banked, re-invoke to continue", flush=True)
            return False
    return True


def stage_summ():
    old = json.load(open(f"{C.OLDRAW}/results_eval141.json"))
    out = {"note": "eval-141, NO refit on eval; probes fitted once on the 12,000 TRAIN windows of the population; AUC = single-fit AUC, CI = episode-cluster bootstrap by eval episode (B=2000, seed 0), paired on the same resample", "pops": {}}
    for pop in C.POPS:
        ev = np.load(f"{C.OLDRAW}/eval141_features.npz", allow_pickle=False)
        sel = pop_sel(ev, pop)
        ye, eide = ev["go"][sel].astype(int), ev["eid"][sel]
        cnt = L.boot_counts(eide)
        P = {"n": int(sel.sum()), "n_go": int(ye.sum()), "n_episodes": int(len(np.unique(eide))), "families": {}}
        lm, sm = ev["launch"][sel], ev["stay"][sel]
        for fam in ("linear", "mlp"):
            sp = f"{C.RAW}/eval2_scores_{pop}_{fam}.npz"
            if not os.path.exists(sp):
                continue
            sc = {k.replace("_p_", "+"): v for k, v in np.load(sp).items()}
            F = {"arms": {}, "paired": {}, "ev0_refit_vs_banked": {}, "subsets": {}}
            for nm, s in sc.items():
                F["arms"][nm] = L.auc_ci(s, ye, eide, counts=cnt)
            for nm in ("E", "V+E", "Z+E"):
                if nm in sc:
                    F["ev0_refit_vs_banked"][nm] = {"refit": F["arms"][nm]["auc"], "banked": old["pops"][pop]["arms"][f"{fam}:{nm}"]["auc"], "abs_diff": abs(F["arms"][nm]["auc"] - old["pops"][pop]["arms"][f"{fam}:{nm}"]["auc"])}
            for a, b in PAIRS:
                if a in sc and b in sc:
                    F["paired"][f"{a}_minus_{b}"] = L.dauc_ci(sc[a], sc[b], ye, eide, counts=cnt)
            if fam == "linear":
                refs = {"E": ("V", "E"), "V+E": ("V", "V+E"), "Z+E": ("Z", "Z+E")}
                oofs = {"E": np.load(f"{C.OLDRAW}/oof_{pop}_V_linear.npz")["E"], "V+E": np.load(f"{C.OLDRAW}/oof_{pop}_V_linear.npz")["V+E"], "Z+E": np.load(f"{C.OLDRAW}/oof_{pop}_Z_linear.npz")["Z+E"]}
                for src, key in (("Znn", "Znn+E"), ("V4", "V4+E"), ("Vdiff", "Vdiff+E"), ("Znl", "Znl+E")):
                    f_ = f"{C.RAW}/oof2_{src}_{pop}_linear.npz"
                    if os.path.exists(f_) and key.replace(":", "__") in np.load(f_).files:
                        oofs[key] = np.load(f_)[key]
                for key in ("N5+E", "NL+E"):
                    f_ = f"{C.RAW}/oof2_nav_{pop}_linear.npz"
                    if os.path.exists(f_):
                        oofs[key] = np.load(f_)[key]
                for nm in ("E", "V+E", "Z+E", "N5+E", "NL+E", "Znn+E", "Znl+E", "V4+E", "Vdiff+E"):
                    if nm in sc and nm in oofs and lm.any() and sm.any():
                        pct = np.searchsorted(np.sort(oofs[nm]), sc[nm]) / len(oofs[nm])
                        m2 = lm | sm
                        F["subsets"][nm] = {"n_launch": int(lm.sum()), "n_stay": int(sm.sum()), "mean_percentile_launch": float(pct[lm].mean()), "mean_percentile_stay": float(pct[sm].mean()),
                                            "auc_launch_vs_stay": L.auc_ci(sc[nm][m2], lm[m2].astype(int), eide[m2])}
            P["families"][fam] = F
        out["pops"][pop] = P
    json.dump(out, open(f"{C.RAW}/results_eval2.json", "w"), indent=1, default=float)
    for pop, P in out["pops"].items():
        for fam, F in P["families"].items():
            print(pop, fam, "n", P["n"], "go", P["n_go"], "eps", P["n_episodes"])
            print("  ", {k: round(v["auc"], 3) for k, v in F["arms"].items()})
            for k, v in F["paired"].items():
                print("   ", k, f"{v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")
            print("   EV0", {k: round(v["abs_diff"], 6) for k, v in F["ev0_refit_vs_banked"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("fit", "summ"))
    ap.add_argument("--pop", choices=C.POPS)
    ap.add_argument("--fam", choices=("linear", "mlp"))
    ap.add_argument("--budget-s", type=float, default=170.0)
    a = ap.parse_args()
    if a.stage == "fit":
        ok = stage_fit(a.pop, a.fam, a.budget_s)
        print("complete" if ok else "INCOMPLETE")
    else:
        stage_summ()
