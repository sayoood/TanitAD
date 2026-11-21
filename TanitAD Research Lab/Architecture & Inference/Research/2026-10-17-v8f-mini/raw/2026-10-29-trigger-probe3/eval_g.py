#!/usr/bin/env python3
"""eval_g.py -- PREREG 5.3 (eval-141 no-refit confirmation) and 5.2 (spatial readout map).
  stage extract : the 8x20 tokens of the 608 eval-141 windows (v0 < 6) -> F:/.../G8_eval141.npy ; alignment gate vs the eval cache's own pool4x10 and vs probe 1's V_eval141.npy
  stage linear  : probes fitted ONCE on all 12,000 TRAIN rows of the population (all-rows PCA = fold 5; C by inner CV), applied unchanged to the eval windows; weight map from the G8+E linear fit
  stage mlp     : the same with the MLP (resumable per arm)
Eval-141 is used for NO choice.  Episode bootstrap by eval episode.   CUDA_VISIBLE_DEVICES=-1  python -I eval_g.py --stage extract|linear|mlp [--pop pstop|pmove]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import g_lib as G  # noqa: E402
import numpy as np

L = G.L
ARMS = ("E", "G4", "G4+E", "G8", "G8+E")


def ev_load():
    return np.load(f"{G.OLDRAW}/eval141_features.npz", allow_pickle=False)


def stage_extract():
    import torch
    torch.set_num_threads(4)
    ev = ev_load()
    names, t0s = [str(c) for c in ev["clip"]], ev["t0"].astype(int)
    n = len(names)
    G8 = np.zeros((n, 160 * 1024), np.float16)
    G4c = np.zeros((n, 40 * 1024), np.float16)
    for i, (c, t) in enumerate(zip(names, t0s)):
        p = f"{G.EVAL_CACHE_CLIPS}/{c}.pt"
        assert os.path.exists(p), p
        o = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        assert o["pool8x20"].shape[1:] == (160, 1024) and t < o["pool8x20"].shape[0], (c, t, tuple(o["pool8x20"].shape))
        G8[i] = o["pool8x20"][t].to(torch.float16).reshape(-1).numpy()
        G4c[i] = o["pool4x10"][t].to(torch.float16).reshape(-1).numpy()
    g4 = G.g4_from_g8(G8.astype(np.float32).reshape(n, 160, 1024)).reshape(n, -1)
    V = np.load(f"{G.OLDSCR}/V_eval141.npy").astype(np.float32)                  # probe 1's eval V (what ClipStore gave the model)
    rel = lambda a, b: np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    r_cache, r_v = rel(g4, G4c.astype(np.float32)), rel(g4, V)
    q = lambda r: {"median": float(np.median(r)), "p99": float(np.percentile(r, 99)), "max": float(r.max()), "mean": float(r.mean())}
    out = {"n": n, "rel_l2_g4_vs_eval_cache_pool4x10": q(r_cache), "rel_l2_g4_vs_probe1_V_eval141": q(r_v)}
    ok = out["rel_l2_g4_vs_probe1_V_eval141"]["median"] <= 0.05 and out["rel_l2_g4_vs_probe1_V_eval141"]["max"] <= 0.5
    out["gate_pass"] = bool(ok)
    np.save(f"{G.SCR}/G8_eval141.npy", G8)
    json.dump(out, open(f"{G.RAW}/eval_g_extract.json", "w"), indent=1)
    print(json.dumps(out), flush=True)
    if not ok:
        raise SystemExit("eval alignment gate FAILED")


def pop_sel(ev, pop):
    return ev["v0"] < 0.5 if pop == "pstop" else (ev["v0"] >= 0.5) & (ev["v0"] < 6.0)


def build_xy(pop):
    ev = ev_load()
    sel = pop_sel(ev, pop)
    d = G.load_pop(pop)
    P = np.load(f"{G.RAW}/pcaG_{pop}.npz")
    G8e = np.load(f"{G.SCR}/G8_eval141.npy")[sel].astype(np.float32).reshape(-1, 160, 1024)
    pc8 = {"mu": P["mu8"][5], "comp": P["comp8"][5]}
    pc4 = {"mu": P["mu4"][5], "comp": P["comp4"][5]}
    Se8 = G.project_tokens(G8e, [pc8])[0].astype(np.float64)
    Se4 = G.project_tokens(G.g4_from_g8(G8e), [pc4])[0].astype(np.float64)
    S8 = np.load(f"{G.SCR}/S8_{pop}.npy", mmap_mode="r")
    S4 = np.load(f"{G.SCR}/S4_{pop}.npy", mmap_mode="r")
    S8a, S4a = np.asarray(S8[5], np.float64), np.asarray(S4[5], np.float64)
    Ee, Et = ev["E"][sel].astype(np.float64), d["E"]
    train = {"E": Et, "G4": S4a, "G4+E": np.concatenate([S4a, Et], 1), "G8": S8a, "G8+E": np.concatenate([S8a, Et], 1)}
    test = {"E": Ee, "G4": Se4, "G4+E": np.concatenate([Se4, Ee], 1), "G8": Se8, "G8+E": np.concatenate([Se8, Ee], 1)}
    return d, train, test, ev["go"][sel].astype(int), ev["eid"][sel], int(sel.sum())


def stage_fit(pop, fam):
    d, train, test, ye, eide, n = build_xy(pop)
    path = f"{G.RAW}/results_eval141_g_{pop}_{fam}.json"
    scp = f"{G.RAW}/eval141_g_scores_{pop}_{fam}.npz"
    res = json.load(open(path)) if os.path.exists(path) else {"pop": pop, "family": fam, "n": n, "n_go": int(ye.sum()), "n_episodes": int(len(np.unique(eide))), "arms": {}, "paired": {}}
    sc = dict(np.load(scp)) if os.path.exists(scp) else {}
    counts = L.boot_counts(eide)
    t0 = time.time()
    for arm in ARMS:
        if arm in sc:
            continue
        G.require_ram(3000, f"eval fit {pop} {fam} {arm}")
        pred, hp, tab = L.fit_probe(train[arm], d["y"], d["inner"], fam)
        sc[arm] = pred(test[arm])
        r = L.auc_ci(sc[arm], ye, eide, counts=counts)
        res["arms"][arm] = {**r, "hp": hp}
        np.savez(scp, **sc)
        json.dump(res, open(path, "w"), indent=1)
        print(f"eval141 {pop} {fam} {arm} auc {r['auc']:.4f} [{r['lo']:.4f},{r['hi']:.4f}] hp {hp} {time.time() - t0:.0f}s", flush=True)
        if fam == "linear" and arm == "G8+E":
            weight_map(pop, d, train, hp)
    if all(a in sc for a in ARMS):
        for a_, b_ in (("G8+E", "G4+E"), ("G4+E", "E"), ("G8+E", "E"), ("G8", "G4")):
            res["paired"][f"{a_}_minus_{b_}"] = L.dauc_ci(sc[a_], sc[b_], ye, eide, counts=counts)
        json.dump(res, open(path, "w"), indent=1)


def weight_map(pop, d, train, hp):
    """PREREG 5.2: the all-rows G8+E linear fit; per-token L2 norm of its 16 standardised coefficients (an 8x20 map); the same for G4+E (4x10) for comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = {}
    for arm, hw in (("G8+E", (8, 20)), ("G4+E", (4, 10))):
        X = train[arm]
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=hp if arm == "G8+E" else G.L.C_GRID[2], solver="lbfgs", max_iter=2000).fit(sc.transform(X), d["y"])
        nt = hw[0] * hw[1]
        w = m.coef_[0][:nt * G.K_TOK].reshape(nt, G.K_TOK)
        tok = np.linalg.norm(w, axis=1).reshape(hw)
        out[arm] = tok
        np.save(f"{G.RAW}/weightmap_{pop}_{arm.replace('+', '_')}.npy", tok)
    json.dump({"note": "G8+E C = the inner-CV choice of the all-rows fit; G4+E uses C=1e-2 (descriptive only)", "G8_C": hp}, open(f"{G.RAW}/weightmap_{pop}.json", "w"))
    print(f"weight map {pop} saved", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("extract", "linear", "mlp"))
    ap.add_argument("--pop", default="")
    a = ap.parse_args()
    if a.stage == "extract":
        G.require_ram(1500, "eval extract")
        stage_extract()
    else:
        for pop in ([a.pop] if a.pop else list(G.POPS)):
            stage_fit(pop, a.stage)
        print("stage done", a.stage, flush=True)
