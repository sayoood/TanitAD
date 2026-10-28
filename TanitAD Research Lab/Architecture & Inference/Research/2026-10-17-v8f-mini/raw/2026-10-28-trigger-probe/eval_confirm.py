#!/usr/bin/env python3
"""eval_confirm.py -- PREREG_FROZEN.md 5 "Confirmation (no refit)".  The probes are fitted ONCE on all sampled TRAIN windows (C / alpha by the inner CV over them, PCA on all of them) and applied unchanged
to eval-141's P-STOP / P-MOVE windows (crawl_diag.Env.window_set(): 2,399 windows, 141 episodes; v0 < 6 -> 608).  Eval-141 is used for NO choice.
Writes raw/eval141_features.npz and raw/results_eval141.json.   CUDA_VISIBLE_DEVICES=-1  python -I eval_confirm.py"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW, SCR = f"{HERE}/raw", f"{HERE}/scratch"
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402
import extract_z as EZ  # noqa: E402  (load_model, constants)

CD = "E:/Projects/TanitAD-artifacts/dagland/crawl_diag"
EVAL_CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache"
EVAL_JOIN_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"


def features():
    t0 = time.time()
    sys.path.insert(0, CD)
    spec = importlib.util.spec_from_file_location("crawl_diag", f"{CD}/crawl_diag.py")
    cd = importlib.util.module_from_spec(spec)
    sys.modules["crawl_diag"] = cd
    spec.loader.exec_module(cd)
    env = cd.Env()
    ws = env.window_set()
    D = env.tool().D
    sys.path.insert(0, f"{EZ.TREE}/taniteval")
    M, model, cfg = EZ.load_model()
    v0, g, eid, ci, t0s = ws["v0"], ws["g"], ws["eid"], ws["clip_index"], ws["ws"]
    sel = np.flatnonzero(v0 < 6.0)
    J = D.load_join(cd.EVAL_JOIN)
    tracks = D.load_native_tracks(cd.TRACKS)
    fk = D._fk()
    names = [str(J["clip"][int(ci[i])]) for i in sel]
    E8 = np.zeros((len(sel), 8))
    for j, i in enumerate(sel):
        kin3, _ = D.yaw_kin_from_track(tracks[names[j]])
        assert abs(kin3[int(t0s[i]), 0] - v0[i]) < 1e-6, ("eval v0 mismatch", i)
        E8[j] = fk.kin_features(kin3[int(t0s[i])][None].astype(np.float64))[0]
    # V (current frame, pool4x10) and Z (h_O) on the same windows
    _load = torch.load
    torch.load = lambda *aa, **kk: _load(*aa, **{**kk, "mmap": True, "weights_only": True})
    store = D.ClipStore(EVAL_CACHE, "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ds = D.V8FMiniWindows(store, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    cap = {}
    model.read_o.register_forward_hook(lambda m, i, o: cap.__setitem__("hO", o.flatten(1).detach().float()))
    Vf = np.zeros((len(sel), 40 * 1024), np.float16)
    Z = np.zeros((len(sel), cfg.n_q * cfg.d), np.float32)
    order = np.argsort(ci[sel], kind="stable")
    for a in range(0, len(sel), 32):
        blk = order[a:a + 32]
        its = [ds.item(int(ci[sel[b]]), int(t0s[sel[b]]), 0, 0) for b in blk]
        batch = D.collate(its)
        with torch.no_grad():
            model.forward_infer({k: batch[k] for k in M.INFERENCE_INPUT_KEYS})
        Z[blk] = cap["hO"].numpy()
        for k_, b in enumerate(blk):
            Vf[b] = batch["feats"][k_, -1].reshape(-1).numpy().astype(np.float16)           # the CURRENT frame (index hmax-1 = NOW) of the window the model itself read
    np.savez(f"{RAW}/eval141_features.npz", sel=sel, E=E8, Z=Z, v0=v0[sel], eid=eid[sel], clip=np.array(names), t0=t0s[sel], launch=ws["mask"][sel],
             stay=cd.stay_mask(v0, g)[sel], go=L_go(v0[sel], g[sel]))
    np.save(f"{SCR}/V_eval141.npy", Vf)
    print(f"eval features {len(sel)} windows {time.time() - t0:.0f}s", flush=True)


def L_go(v0, g):
    return L.go_label(v0, g, 29)


def main():
    if not os.path.exists(f"{RAW}/eval141_features.npz"):
        features()
    ev = np.load(f"{RAW}/eval141_features.npz", allow_pickle=False)
    Vf = np.load(f"{SCR}/V_eval141.npy").astype(np.float32)
    assert np.array_equal(ev["go"], ev["launch"]), "GO label != the registered launch mask on eval"
    out = {"n_eval_windows_v0_lt_6": int(len(ev["v0"])), "n_launch": int(ev["launch"].sum()), "n_stay": int(ev["stay"].sum()), "pops": {}}
    for pop, sel in (("pstop", ev["v0"] < 0.5), ("pmove", (ev["v0"] >= 0.5) & (ev["v0"] < 6.0))):
        d = None
        sys.path.insert(0, HERE)
        import run_probe as RP
        d = RP.load_pop(pop)
        y, inner = d["y"], d["inner"]
        Sv = np.load(f"{RAW}/pca_V_{pop}.npz")["S"][5].astype(np.float64)
        Sz = np.load(f"{RAW}/pca_Z_{pop}.npz")["S"][5].astype(np.float64)
        cv_ = np.load(f"{SCR}/pcacomp_V_{pop}_all.npz")
        cz_ = np.load(f"{SCR}/pcacomp_Z_{pop}_all.npz")
        Ee, ye, eide = ev["E"][sel], ev["go"][sel].astype(int), ev["eid"][sel]
        Sv_e = ((Vf[sel] - cv_["mu"]) @ cv_["comp"].T).astype(np.float64)
        Sz_e = ((ev["Z"][sel] - cz_["mu"]) @ cz_["comp"].T).astype(np.float64)
        train_X = {"E": d["E"], "V": Sv, "V+E": np.concatenate([Sv, d["E"]], 1), "Z": Sz, "Z+E": np.concatenate([Sz, d["E"]], 1)}
        eval_X = {"E": Ee.astype(np.float64), "V": Sv_e, "V+E": np.concatenate([Sv_e, Ee], 1), "Z": Sz_e, "Z+E": np.concatenate([Sz_e, Ee], 1)}
        counts = L.boot_counts(eide)
        P = {"n": int(sel.sum()), "n_go": int(ye.sum()), "n_episodes": int(len(np.unique(eide))), "arms": {}, "paired": {}}
        sc = {}
        for fam in ("linear", "mlp"):
            for arm in ("E", "V", "V+E", "Z", "Z+E"):
                pred, hp, tab = L.fit_probe(train_X[arm], y, inner, fam)
                s = pred(eval_X[arm])
                sc[(fam, arm)] = s
                r = L.auc_ci(s, ye, eide, counts=counts)
                P["arms"][f"{fam}:{arm}"] = {**r, "hp": hp}
                print(f"{pop} {fam} {arm} eval141 auc {r['auc']:.4f} [{r['lo']:.4f},{r['hi']:.4f}] hp {hp}", flush=True)
            for a_, b_ in (("V+E", "E"), ("Z+E", "V+E"), ("Z+E", "E"), ("V", "E")):
                P["paired"][f"{fam}:{a_}_minus_{b_}"] = L.dauc_ci(sc[(fam, a_)], sc[(fam, b_)], ye, eide, counts=counts)
            # subsets: percentile of each eval score within the TRAIN out-of-fold score distribution of the same arm (calibration-free)
            oofz = np.load(f"{RAW}/oof_{pop}_V_{'linear' if fam == 'linear' else 'mlp'}.npz")
            oofw = np.load(f"{RAW}/oof_{pop}_Z_{'linear' if fam == 'linear' else 'mlp'}.npz")
            P.setdefault("subsets", {})[fam] = {}
            lm, sm = ev["launch"][sel], ev["stay"][sel]
            for arm in ("E", "V+E", "Z+E"):
                ref = (oofw if arm.startswith("Z") else oofz)[arm.replace("+", "+").replace(":", "__")]
                pct = (np.searchsorted(np.sort(ref), sc[(fam, arm)]) / len(ref))
                dd = {"n_launch": int(lm.sum()), "n_stay": int(sm.sum()), "mean_percentile_launch": float(pct[lm].mean()) if lm.any() else None,
                      "mean_percentile_stay": float(pct[sm].mean()) if sm.any() else None}
                if lm.any() and sm.any():
                    m2 = lm | sm
                    dd["auc_launch_vs_stay"] = L.auc_ci(sc[(fam, arm)][m2], lm[m2].astype(int), eide[m2], n_boot=2000)
                P["subsets"][fam][arm] = dd
        out["pops"][pop] = P
    json.dump(out, open(f"{RAW}/results_eval141.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
