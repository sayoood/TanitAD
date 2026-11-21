#!/usr/bin/env python3
"""nav_arm.py -- POST-FREEZE DIAGNOSTIC (caveat-only; PREREG_FROZEN.md 7 says post-freeze arms cannot change a verdict).  Why: Z = h_O is a function of the frozen image features AND of the NON-IMAGE inputs the trunk also reads:
the nav token (J["nav"], 5 numbers: FOLLOW / TURN_L / TURN_R one-hot, d_next/100 clipped, args_valid -- derived from the ego's own future path: the v7.2 provenance says 'nav_command: ORACLE (ego-future)') and the speed limit
(J["limit"], 4 one-hot + limit_ok).  V+E has neither, so Z+E > V+E cannot by itself say the TRUNK reads the IMAGE better.  N = nav5 ++ limit4 ++ limit_ok at NOW.
Arms: N+E, V+N+E (V = PCA-64, fold-wise) on the SAME sampled windows / folds / bootstrap; paired against Z+E; eval-141 fit-on-TRAIN / apply.  Writes raw/results_nav_postfreeze.json.
CUDA_VISIBLE_DEVICES=-1  python -I nav_arm.py"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW, SCR = f"{HERE}/raw", f"{HERE}/scratch"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402
import run_probe as RP  # noqa: E402

TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
EVAL_JOIN = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"


def nav_of(J, names, ts):
    ix = {c: i for i, c in enumerate(J["clip"].tolist())}
    ci = np.array([ix[n] for n in names])
    row = J["off"][ci] + np.asarray(ts)
    return np.concatenate([J["nav"][row], J["limit"][ci], J["limit_ok"][ci][:, None].astype(np.float32)], 1).astype(np.float64)


def main():
    sys.path[:0] = [f"{TREE}/stack", f"{TREE}/taniteval"]
    from tanitad.data import v8f_mini_data as D
    t0 = time.time()
    Jt = D.load_join(f"{SCR}/join_train_rebuilt.npz")
    Je = D.load_join(EVAL_JOIN)
    ev = np.load(f"{RAW}/eval141_features.npz", allow_pickle=False)
    W = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    clips = [str(c) for c in W["clips"]]
    out = {"note": "POST-FREEZE, caveat-only", "pops": {}}
    for pop in ("pstop", "pmove"):
        d = RP.load_pop(pop)
        y, clip, outer, inner = d["y"], d["clip"], d["outer"], d["inner"]
        counts = L.boot_counts(clip)
        N = nav_of(Jt, [clips[c] for c in clip], d["t"])
        Sv = np.load(f"{RAW}/pca_V_{pop}.npz")["S"].astype(np.float64)
        arms = {"N+E": lambda f: np.concatenate([N, d["E"]], 1), "V+N+E": lambda f: np.concatenate([Sv[f], N, d["E"]], 1)}
        P = {"arms": {}, "paired": {}}
        oof = {}
        for fam in ("linear", "mlp"):
            for nm, fn in arms.items():
                o, hps = L.cv_oof(fn, y, outer, inner, fam)
                oof[(fam, nm)] = o
                P["arms"][f"{fam}:{nm}"] = {**L.auc_ci(o, y, clip, counts=counts), "foldmean": L.auc_ci_fold(o, y, clip, outer, counts=counts)}
                print(f"{pop} {fam} {nm} pooled {P['arms'][f'{fam}:{nm}']['auc']:.4f} foldmean {P['arms'][f'{fam}:{nm}']['foldmean']['auc']:.4f} {time.time() - t0:.0f}s", flush=True)
            ov = np.load(f"{RAW}/oof_{pop}_V_{fam}.npz")
            oz = np.load(f"{RAW}/oof_{pop}_Z_{fam}.npz")
            ref = {"E": ov["E"], "V+E": ov["V+E"], "Z+E": oz["Z+E"]}
            for a_, b_ in (("N+E", "E"), ("V+N+E", "V+E"), ("V+N+E", "N+E")):
                A_ = oof[(fam, a_)]
                B_ = oof[(fam, b_)] if (fam, b_) in oof else ref[b_]
                P["paired"][f"{fam}:{a_}_minus_{b_}"] = {**L.dauc_ci(A_, B_, y, clip, counts=counts), "foldmean": L.dauc_ci_fold(A_, B_, y, clip, outer, counts=counts)}
            for b_ in ("V+N+E", "N+E"):
                P["paired"][f"{fam}:Z+E_minus_{b_}"] = {**L.dauc_ci(ref["Z+E"], oof[(fam, b_)], y, clip, counts=counts),
                                                      "foldmean": L.dauc_ci_fold(ref["Z+E"], oof[(fam, b_)], y, clip, outer, counts=counts)}
        # eval-141: fit on all TRAIN, apply (linear family, the committed primary)
        sel = (ev["v0"] < 0.5) if pop == "pstop" else ((ev["v0"] >= 0.5) & (ev["v0"] < 6.0))
        Ne = nav_of(Je, ev["clip"][sel].tolist(), ev["t0"][sel])
        ye, eide = ev["go"][sel].astype(int), ev["eid"][sel]
        Svall = Sv[5]
        cv_ = np.load(f"{SCR}/pcacomp_V_{pop}_all.npz")
        Vf = np.load(f"{SCR}/V_eval141.npy").astype(np.float32)[sel]
        Sv_e = ((Vf - cv_["mu"]) @ cv_["comp"].T).astype(np.float64)
        cnt_e = L.boot_counts(eide)
        P["eval141"] = {"n": int(sel.sum()), "n_go": int(ye.sum())}
        sc = {}
        for nm, (Xt, Xe) in {"N+E": (np.concatenate([N, d["E"]], 1), np.concatenate([Ne, ev["E"][sel]], 1)),
                             "V+N+E": (np.concatenate([Svall, N, d["E"]], 1), np.concatenate([Sv_e, Ne, ev["E"][sel]], 1))}.items():
            pred, hp, _ = L.fit_probe(Xt, y, inner, "linear")
            sc[nm] = pred(Xe)
            P["eval141"][f"linear:{nm}"] = L.auc_ci(sc[nm], ye, eide, counts=cnt_e)
        er = json.load(open(f"{RAW}/results_eval141.json"))["pops"][pop]["arms"]
        P["eval141"]["linear:Z+E (banked)"] = {k: er["linear:Z+E"][k] for k in ("auc", "lo", "hi")}
        P["eval141"]["linear:V+E (banked)"] = {k: er["linear:V+E"][k] for k in ("auc", "lo", "hi")}
        P["eval141"]["linear:E (banked)"] = {k: er["linear:E"][k] for k in ("auc", "lo", "hi")}
        out["pops"][pop] = P
    json.dump(out, open(f"{RAW}/results_nav_postfreeze.json", "w"), indent=1, default=float)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
