"""wua_a1_tlin.py -- card ADD155 Amendment 4: linear temporal probe P-TLIN ([state t-4..t, EGO] vs [EGO], ridge) with the Amendment 3 permutation reference; one seed, all targets.

  python -I wua_a1_tlin.py --seed 0   -> C:/Users/Admin/wua_work/a1/TLIN_A1_s<seed>.json
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wua_a1 as A1   # noqa: E402
import wua_a1_perm as PM   # noqa: E402

TG = {"lead_gap": ["lead_gap"], "ag1_pos": ["ag1_x", "ag1_y"], "lead_closing": ["lead_closing"], "lead_rel_acc": ["lead_rel_acc"], "gap_change_2s": ["gap_change_2s"]}
REPS = (("L1", "L1"), ("L2", "L2"), ("L3", "L3"), ("TOK-MEAN", "tokmean"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    E = np.load(f"{A1.EXT}/WUA_RH_s{a.seed}.npz")
    Lb = np.load(A1.LAB)
    assert (E["clip_idx"] == Lb["clip_idx"]).all() and (E["t"] == Lb["t"]).all()
    names = list(Lb["names"]); Y = Lb["Y"]
    ci_all, t_all = E["clip_idx"], E["t"].astype(int)
    clips = list(E["clips"]); fclip = A1.fold_of(clips)
    v, acc, yr = E["v"], E["a"], E["yaw_rate"]
    H = A1.H
    keep = t_all >= H
    egoh = np.full((len(ci_all), 3 * (H + 1)), np.nan)
    for i in np.where(keep)[0]:
        egoh[i] = np.concatenate([v[i - H:i + 1], acc[i - H:i + 1], yr[i - H:i + 1]])
    keep &= np.isfinite(egoh).all(1)
    rows = np.where(keep)[0]
    assert (ci_all[rows - 4] == ci_all[rows]).all()
    ci, fo, EG = ci_all[rows], fclip[ci_all[rows]], egoh[rows]
    X = {r: np.concatenate([E[k][rows - s].astype(np.float64) for s in range(5)], 1) for r, k in REPS}
    lookup = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(ci, t_all[rows]))}
    W = A1.boot_w(len(clips))
    out = {"card": "ADD155", "amendment": 4, "seed": a.seed, "probe": "P-TLIN", "cells": {}}
    for tg, cols in TG.items():
        y0 = np.stack([Y[rows, names.index(c)] for c in cols], 1)
        cell = {r: {"draws": []} for r, _ in REPS}
        for d in [-1] + list(range(PM.R_DRAWS)):
            if d < 0:
                y = y0
            else:
                perm = PM.derange(len(clips), 100 + d)
                src = np.array([lookup.get((int(perm[c]), int(t)), -1) for c, t in zip(ci, t_all[rows])])
                y = np.full(y0.shape, np.nan); ok = src >= 0; y[ok] = y0[src[ok]]
            m = np.isfinite(y).all(1)
            pe = A1.oof_ridge(EG[m], y[m], ci[m], fo[m])
            for r, _ in REPS:
                pf = A1.oof_ridge(np.concatenate([X[r][m], EG[m]], 1), y[m], ci[m], fo[m])
                ok2 = np.isfinite(pf).all(1) & np.isfinite(pe).all(1)
                yy, pff, pee = y[m][ok2], pf[ok2], pe[ok2]
                if d < 0:
                    st = A1.r2_stats(yy, pff, pee, ci[m][ok2], W)
                    st.update({"n": int(ok2.sum()), "n_clips": int(len(np.unique(ci[m][ok2])))})
                    cell[r]["observed"] = st
                else:
                    sst = ((yy - yy.mean(0)) ** 2).sum()
                    cell[r]["draws"].append(float(((yy - pee) ** 2).sum() / sst - ((yy - pff) ** 2).sum() / sst))
            print(f"[tlin s{a.seed}] {tg} draw {d} {time.time() - t0:.0f}s", flush=True)
        for r, _ in REPS:
            c = cell[r]
            c["perm_max"] = max(c["draws"]); c["exceeds_all"] = bool(c["observed"]["delta"] > c["perm_max"])
            out["cells"][f"{r}|{tg}"] = c
    out["am4_sha256"] = hashlib.sha256(open("F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD155_AM4.md", "rb").read()).hexdigest()
    out["tool_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{A1.WORK}/a1/TLIN_A1_s{a.seed}.json", "w"), indent=1)
    print("WROTE", {k: (round(c["observed"]["delta"], 3), c["exceeds_all"]) for k, c in out["cells"].items()})


if __name__ == "__main__":
    main()
