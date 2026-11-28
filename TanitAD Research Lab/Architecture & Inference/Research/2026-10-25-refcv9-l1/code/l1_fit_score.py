"""l1_fit_score.py -- refcv9 L1 (PREREG_L1.md): fit the progress heads on TRAIN, score them on eval139 along refcv8's own path.
CPU only.  Every hyper-parameter is chosen by 5-fold EPISODE cross-validation on the TRAIN rows; the eval rows are scored, never tuned on.

Inputs: the z_tac bank (l1_ztac_roll.py: ztac_eval_s{0,1}.npz, ztac_train_s0.npz), the banked battery tree step48852ema_pf (os plans,
panels, prior, GT), the A6 train139 tree (model-free target + features), the frozen per-frame RETIME table.
Outputs: <out>/L1_RESULT.json + <out>/RESULT_TABLES.md.   Arms / controls / bars: PREREG_L1.md sections 4-7 (names identical).

    CUDA_VISIBLE_DEVICES=-1 R8_LAUNCH_TREE=<tree> PYTHONPATH='<tree>/stack;<tree>/taniteval;<battery>/code' \
        python -X utf8 l1_fit_score.py --bank <dir> --out <dir> [--dryrun-synthetic]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "probe_repro"))
import along_track_localise as AL  # noqa: E402
import progress_headroom as PH  # noqa: E402
import progress_scalar as PS  # noqa: E402

RT = AL.RT
TREE = "step48852ema_pf"
SEEDS = (0, 1)
ARCH = Path("C:/Users/Admin/refcv8_eval_archive/battery")
PFPKG = Path("C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final")
FROZEN_PF = PFPKG / "raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json"
FROZEN_PF_SHA = "see PREREG_L1.md section 4 (recorded at run time)"
BANKED_TF = PFPKG / "pfpkg/raw/retime_trainfit/FORWARD_TRAINFIT_step48852ema_pf.json"
OSONLY_S0 = Path("C:/Users/Admin/refcv8_pf/r8oracle/run_48852ema_pf/dump_s0/refcv6_extras.npz")   # AMENDMENT 1 (K4 at seed 0)
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
KS = (4, 8, 16, 32, 64, 128, 256)
H1_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
MLP_WD = (1e-4, 1e-2, 1e-1)
MLP_MAX_EP = 200
NFOLD = 5
GMAX = 1.5


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- loading
def load_eval(tools, CR, seed, first, surf):
    ctx = CR.prepare_seed(TREE if not DRY else "step30000ema_unc", seed, tools, tools["rc"])
    S = RT.surface_inputs(tools, ctx, surf, first, TREE)
    rows = ctx["panels"][surf]["rows"]
    lim = ctx["lim"][rows]
    m = np.asarray(S["mask"], bool)
    d = {"O_all": np.asarray(S["X"]["O"]), "clip_all": np.asarray(S["clip"], np.int64), "ws_all": np.asarray(S["ws"], np.int64),
         "O": np.asarray(S["X"]["O"])[m], "HP": np.asarray(S["X"]["HP"])[m], "G": np.asarray(S["X"]["G"])[m],
         "X": PS.feats(S["dec"]["win"][m], lim[m]), "eid": np.asarray([str(e) for e in S["eid"]])[m],
         "clip": np.asarray(S["clip"], np.int64)[m], "ws": np.asarray(S["ws"], np.int64)[m], "v0": np.asarray(S["v0"], np.float64)[m],
         "dt": S["dt"], "n_all": int(len(m)), "dump": ctx["dump"]}
    del ctx
    return d


def load_train(tools, CR2, TF, surf, first):
    ctx = CR2.prepare_seed(TF.TRAIN_TREE, 0, tools, tools["rc"])
    S = TF.train_surface_inputs(tools, ctx, surf, first, TF.TRAIN_TREE)
    rows = ctx["panels"][surf]["rows"]
    lim = ctx["lim"][rows]
    m = np.asarray(S["mask"], bool)
    d = {"HP": np.asarray(S["X"]["HP"])[m], "G": np.asarray(S["X"]["G"])[m], "X": PS.feats(S["dec"]["win"][m], lim[m]),
         "eid": np.asarray([str(e) for e in S["eid"]])[m], "clip": np.asarray(S["clip"], np.int64)[m],
         "ws": np.asarray(S["ws"], np.int64)[m], "v0": np.asarray(S["v0"], np.float64)[m], "dt": S["dt"]}
    del ctx
    return d


def load_bank(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    W = int(meta["window_W"])
    ws = z["t"].astype(np.int64) + W - 1
    idx = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(z["e_i"], ws))}
    if len(idx) != len(ws):
        raise SystemExit(f"[l1] {path}: duplicate (e_i, ws) keys in the bank")
    return {"z": z["z_tac"].astype(np.float64), "traj": z["traj_own"].astype(np.float64), "e_i": z["e_i"].astype(np.int64), "ws": ws,
            "v0_in": z["v0_in"], "idx": idx, "meta": meta}


def join(bank, clip, ws, what="z"):
    j = np.array([bank["idx"].get((int(c), int(w)), -1) for c, w in zip(clip, ws)])
    if (j < 0).any():
        raise SystemExit(f"[l1] {int((j < 0).sum())} rows have no bank entry")
    return bank[what][j], j


# ------------------------------------------------------------------------------------------------- models
def _std(X):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return mu, sd


def ridge(Z, Y, lam, pen):
    R = lam * np.diag(pen.astype(np.float64)) + 1e-10 * np.eye(len(pen))
    return np.linalg.solve(Z.T @ Z + R, Z.T @ Y)


def design(X, Zl, st):
    """[1 | standardized scalars (9) | standardized PCA scores (k)] -- st carries every train-fit transform."""
    xs = (X[:, 1:] - st["xmu"]) / st["xsd"]
    cols = [np.ones((len(X), 1)), xs]
    if Zl is not None and st.get("k", 0) > 0:
        zs = (Zl - st["zmu"]) / st["zsd"]
        sc = zs @ st["V"][:, : st["k"]]
        cols.append((sc - st["smu"]) / st["ssd"])
    return np.concatenate(cols, 1)


def fit_transform(X, Zl, k):
    st = {"k": int(k)}
    st["xmu"], st["xsd"] = _std(X[:, 1:])
    if Zl is not None and k > 0:
        st["zmu"], st["zsd"] = _std(Zl)
        zs = (Zl - st["zmu"]) / st["zsd"]
        _u, _s, vt = np.linalg.svd(zs - zs.mean(0), full_matrices=False)
        st["V"] = vt.T[:, : max(KS)]
        sc = zs @ st["V"][:, :k]
        st["smu"], st["ssd"] = _std(sc)
    return st


def head_fit(X, Zl, H, k, lam):
    st = fit_transform(X, Zl, k)
    D = design(X, Zl, st)
    pen = np.ones(D.shape[1])
    pen[0] = 0.0
    st["W"] = ridge(D, H, lam, pen)
    st["lam"] = float(lam)
    return st


def head_pred(st, X, Zl):
    return design(X, Zl, st) @ st["W"]


def folds_of(eid, seed):
    u = np.unique(eid)
    rng = np.random.default_rng(seed)
    f = dict(zip(u, rng.permutation(len(u)) % NFOLD))
    return np.asarray([f[e] for e in eid])


def cv_head(X, Zl, H, eid, seed, ks):
    f = folds_of(eid, seed)
    sse = {}
    oof = {}
    for k in ks:
        for j in range(NFOLD):
            tr, te = f != j, f == j
            st = fit_transform(X[tr], None if Zl is None else Zl[tr], k)
            Dtr, Dte = design(X[tr], None if Zl is None else Zl[tr], st), design(X[te], None if Zl is None else Zl[te], st)
            pen = np.ones(Dtr.shape[1])
            pen[0] = 0.0
            for lam in LAMBDAS:
                W = ridge(Dtr, H[tr], lam, pen)
                P = Dte @ W
                sse[(k, lam)] = sse.get((k, lam), 0.0) + float(((P - H[te]) ** 2).sum())
                oof.setdefault((k, lam), np.zeros_like(H))[te] = P
    best = min(sse, key=sse.get)
    return best, {f"k{k}_lam{l}": v for (k, l), v in sse.items()}, oof[best]


def h1_fit(X, D, H, eid, seed):
    """(b): g_k(x) = a_k + b_k . x~ (x = v0, a0, omega0 standardized on train), fit by ridge LS of h_k on [d_k, d_k x~] (a unpenalized)."""
    xr = X[:, 1:4]
    mu, sd = _std(xr)
    xs = (xr - mu) / sd
    f = folds_of(eid, seed)

    def fit_k(k, lam, sel):
        Z = np.concatenate([D[sel, k:k + 1], D[sel, k:k + 1] * xs[sel]], 1)
        pen = np.array([0.0, 1.0, 1.0, 1.0])
        return ridge(Z, H[sel, k], lam, pen)

    best = {}
    for k in range(H.shape[1]):
        sc = {}
        for lam in H1_LAMBDAS:
            e = 0.0
            for j in range(NFOLD):
                tr, te = f != j, f == j
                w = fit_k(k, lam, tr)
                g = np.clip(w[0] + xs[te] @ w[1:], 0.0, GMAX)
                e += float(((g * D[te, k] - H[te, k]) ** 2).sum())
            sc[lam] = e
        lam = min(sc, key=sc.get)
        best[k] = {"lam": lam, "w": fit_k(k, lam, np.ones(len(H), bool)).tolist()}
    return {"mu": mu.tolist(), "sd": sd.tolist(), "per_instant": best}


def h1_gain(m, X):
    xs = (X[:, 1:4] - np.asarray(m["mu"])) / np.asarray(m["sd"])
    return np.stack([np.clip(m["per_instant"][k]["w"][0] + xs @ np.asarray(m["per_instant"][k]["w"][1:]), 0.0, GMAX)
                     for k in sorted(m["per_instant"])], 1)


def mlp_train(Xin, H, wd, epochs, seed, Xval=None, Hval=None):
    import torch
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xin.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, H.shape[1])).double()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    xt, ht = torch.from_numpy(Xin), torch.from_numpy(H)
    curve = []
    for _ep in range(epochs):
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = (net(xt[b]) - ht[b]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xval is not None:
            with torch.no_grad():
                curve.append(float((net(torch.from_numpy(Xval)) - torch.from_numpy(Hval)).abs().sum()))
    return net, curve


def mlp_inputs(X, Zl, st):
    xs = (X[:, 1:] - st["xmu"]) / st["xsd"]
    zs = (Zl - st["zmu"]) / st["zsd"]
    return np.concatenate([xs, zs], 1)


def h2n_fit(X, Zl, H, eid, seed):
    import torch
    f = folds_of(eid, seed)
    tot = {}
    for wd in MLP_WD:
        acc = np.zeros(MLP_MAX_EP)
        for j in range(NFOLD):
            tr, te = f != j, f == j
            st = {"xmu": X[tr, 1:].mean(0), "xsd": X[tr, 1:].std(0) + 1e-9, "zmu": Zl[tr].mean(0), "zsd": Zl[tr].std(0) + 1e-9}
            _n, curve = mlp_train(mlp_inputs(X[tr], Zl[tr], st), H[tr], wd, MLP_MAX_EP, seed, mlp_inputs(X[te], Zl[te], st), H[te])
            acc += np.asarray(curve)
        tot[wd] = acc
    best = min(((wd, int(np.argmin(a)) + 1, float(a.min())) for wd, a in tot.items()), key=lambda x: x[2])
    st = {"xmu": X[:, 1:].mean(0), "xsd": X[:, 1:].std(0) + 1e-9, "zmu": Zl.mean(0), "zsd": Zl.std(0) + 1e-9}
    net, _ = mlp_train(mlp_inputs(X, Zl, st), H, best[0], best[1], seed)
    n_par = int(sum(p.numel() for p in net.parameters()))

    def pred(Xe, Ze):
        with torch.no_grad():
            return net(torch.from_numpy(mlp_inputs(Xe, Ze, st))).numpy()
    return pred, {"wd": best[0], "epochs": best[1], "cv_l1_sum": best[2], "n_params": n_par}


# ------------------------------------------------------------------------------------------------- scoring
def comp(tools, P, G, dt):
    return tools["PR"].components(tools["ra"], P, G, dt)


def cell(tools, a, b, eid):
    c = tools["PR"].paired_cell(a, b, list(eid))
    return {k: c.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "status")}


def derange(eid):
    """eval episode e receives the z rows of the NEXT episode (sorted ids, cyclic), position-matched (index mod donor length)."""
    u = list(np.unique(eid))
    pos = {e: np.where(eid == e)[0] for e in u}
    out = np.empty(len(eid), np.int64)
    for i, e in enumerate(u):
        d = pos[u[(i + 1) % len(u)]]
        r = pos[e]
        out[r] = d[np.arange(len(r)) % len(d)]
    return out


def r2(P, H):
    return [float(1 - ((P[:, k] - H[:, k]) ** 2).sum() / ((H[:, k] - H[:, k].mean()) ** 2).sum()) for k in range(H.shape[1])]


def run_surface(tools, surf, ev, tr, banks, gains, keys_metrics):
    """fit every head on TRAIN rows of `surf`, score every arm on eval rows of `surf` at both seeds."""
    res = {"surface": surf, "fit": {}, "seeds": {}, "controls": {}}
    Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
    Ztr, jtr = join(banks["train"], tr["clip"], tr["ws"])
    v0_gap = float(np.nanmax(np.abs(banks["train"]["v0_in"][jtr] - tr["v0"])))
    res["controls"]["K5_train_join"] = {"n_rows": int(len(Htr)), "v0_bank_vs_panel_maxabs": v0_gap, "PASS": bool(v0_gap < 1e-3)}
    # ---- H2s: the scalar ridge (progress_scalar.py, unchanged procedure)
    lam_s, cvs = PS.cv_lambda(tr["X"], Htr, tr["eid"])
    m_s = PS.ridge_fit(tr["X"], Htr, lam_s)
    res["fit"]["H2s"] = {"lambda": lam_s, "train_R2": r2(PS.ridge_pred(m_s, tr["X"]), Htr), "n": int(len(Htr)),
                         "n_episodes": int(len(np.unique(tr["eid"]))), "d_in": int(tr["X"].shape[1] - 1)}
    # ---- CONST
    hc = Htr.mean(0)
    # ---- H2 (primary): ridge on scalars + PCA_k(z_tac); (k, lambda) by 5-fold episode CV on TRAIN
    (k2, l2), cv2, oof2 = cv_head(tr["X"], Ztr, Htr, tr["eid"], 0, KS)
    m2 = head_fit(tr["X"], Ztr, Htr, k2, l2)
    cgain = (oof2 * Htr).sum(0) / (oof2 * oof2).sum(0)
    res["fit"]["H2"] = {"k": int(k2), "lambda": float(l2), "train_R2": r2(head_pred(m2, tr["X"], Ztr), Htr), "oof_R2": r2(oof2, Htr),
                        "d_tac": int(Ztr.shape[1]), "n": int(len(Htr)), "n_params": int(m2["W"].size), "H2g_oof_gain": cgain.tolist(),
                        "cv_sse_best": min(cv2.values())}
    # ---- H2r: replicate (80 % of train EPISODES, numpy seed 1; folds seed 1)
    u = np.unique(tr["eid"])
    keep_e = set(np.random.default_rng(1).choice(u, size=int(round(0.8 * len(u))), replace=False))
    kr = np.array([e in keep_e for e in tr["eid"]])
    (kR, lR), _cvr, _ = cv_head(tr["X"][kr], Ztr[kr], Htr[kr], tr["eid"][kr], 1, KS)
    mR = head_fit(tr["X"][kr], Ztr[kr], Htr[kr], kR, lR)
    res["fit"]["H2r"] = {"k": int(kR), "lambda": float(lR), "n": int(kr.sum()), "n_episodes": len(keep_e)}
    # ---- H2n (secondary): the annex's MLP head, L1 loss
    t_m = time.time()
    p_n, info_n = h2n_fit(tr["X"], Ztr, Htr, tr["eid"], 0)
    res["fit"]["H2n"] = {**info_n, "wall_s": round(time.time() - t_m, 1)}
    # ---- H1 (b): linear-conditioned gain on refcv8's TRAIN os paths (bank, inference seed 0)
    Otr, _ = join(banks["train"], tr["clip"], tr["ws"], "traj")
    Otr = Otr[:, SLOTS[surf], :]
    Dtr = AL.arc(Otr) - AL.arc(tr["HP"])
    m1 = h1_fit(tr["X"], Dtr, Htr, tr["eid"], 0)
    res["fit"]["H1"] = m1
    res["fit"]["H0"] = {"gains": list(gains), "source": str(FROZEN_PF)}
    for s in SEEDS:
        E = ev[s]
        Z, _je = join(banks[f"eval{s}"], E["clip"], E["ws"])
        sp = AL.arc(E["HP"])
        Hev = AL.arc(E["G"]) - sp
        hh = {"H2s": PS.ridge_pred(m_s, E["X"]), "CONST": np.repeat(hc[None], len(sp), 0), "H2": head_pred(m2, E["X"], Z),
              "H2r": head_pred(mR, E["X"], Z), "H2n": p_n(E["X"], Z),
              "H2m": head_pred(m2, E["X"], Z[derange(E["eid"])]),
              "H2c": head_pred(m2, E["X"], np.repeat(m2["zmu"][None], len(sp), 0))}
        hh["H2g"] = hh["H2"] * cgain[None]
        d_os = AL.arc(E["O"]) - sp
        arms = {"OS": E["O"], "H0": RT.retime_apply(E["O"], E["HP"], np.asarray(gains, np.float64)),
                "H0_G1": RT.retime_apply(E["O"], E["HP"], np.ones(len(gains))),
                "H1": PH.at_progress(E["O"], sp + h1_gain(m1, E["X"]) * d_os),
                "ZERO_PRIOR": PH.at_progress(E["HP"], sp), "H2_PRIOR": PH.at_progress(E["HP"], sp + hh["H2"]),
                "H2s_PRIOR": PH.at_progress(E["HP"], sp + hh["H2s"])}
        for a, h in hh.items():
            arms[a] = PH.at_progress(E["O"], sp + h)
        if OUTDIR is not None:                    # AMENDMENT 2: bank every arm's plan so the braking-AUC read uses the identical arrays
            OUTDIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUTDIR / f"arms_{surf}_s{s}.npz", clip=E["clip"], ws=E["ws"], eid=E["eid"], v0=E["v0"], G=E["G"],
                                **{f"arm_{k}": np.asarray(v, np.float32) for k, v in arms.items()})
        C = {a: comp(tools, P, E["G"], E["dt"]) for a, P in arms.items()}
        c_hp = comp(tools, E["HP"], E["G"], E["dt"])
        e1 = {a: cell(tools, C[a]["ade_m"], c_hp["ade_m"], E["eid"]) for a in arms}
        pairs = {"H2-H0": ("H2", "H0"), "H2-H2s": ("H2", "H2s"), "H2-OS": ("H2", "OS"), "H0-OS": ("H0", "OS"), "H2m-H2s": ("H2m", "H2s"),
                 "H2m-H2": ("H2m", "H2"), "H2r-H0": ("H2r", "H0"), "H2n-H2s": ("H2n", "H2s"), "H2n-H0": ("H2n", "H0"),
                 "H1-H0": ("H1", "H0"), "H2s-H0": ("H2s", "H0"), "H2g-H0": ("H2g", "H0"), "H2c-H2s": ("H2c", "H2s"),
                 "CONST-H2s": ("CONST", "H2s"), "H2_PRIOR-H2s_PRIOR": ("H2_PRIOR", "H2s_PRIOR")}
        pc = {p: cell(tools, C[a]["ade_m"], C[b]["ade_m"], E["eid"]) for p, (a, b) in pairs.items()}
        fam = {p: {k: cell(tools, C[a][k], C[b][k], E["eid"]) for k in keys_metrics if k in C[a]}
               for p, (a, b) in (("H2-H0", ("H2", "H0")), ("H2-H2s", ("H2", "H2s")), ("H2-OS", ("H2", "OS")), ("H0-OS", ("H0", "OS")))}
        res["seeds"][str(s)] = {"n_windows": int(len(sp)), "n_episodes": int(len(np.unique(E["eid"]))), "E1_minus_ha0_ext_pose": e1,
                                "paired": pc, "four_families": fam, "eval_R2": {a: r2(h, Hev) for a, h in hh.items()},
                                "mean_ade": {a: float(np.mean(C[a]["ade_m"])) for a in arms}}
        res["controls"][f"s{s}"] = {
            "K2_g1_is_os_maxabs_m": float(np.abs(arms["H0_G1"].astype(np.float64) - E["O"].astype(np.float64)).max()),
            "K2_zero_prior_maxabs_m": float(np.abs(arms["ZERO_PRIOR"].astype(np.float64) - E["HP"].astype(np.float64)).max()),
            "K3_const_eval_R2": r2(hh["CONST"], Hev), "K3_PASS": bool(max(r2(hh["CONST"], Hev)) <= 1e-12)}
        res["controls"][f"s{s}"]["K2_PASS"] = bool(res["controls"][f"s{s}"]["K2_g1_is_os_maxabs_m"] == 0.0
                                                   and res["controls"][f"s{s}"]["K2_zero_prior_maxabs_m"] <= 1e-5)
        if banks[f"eval{s}"].get("traj") is not None:     # K4c: the bank's OWN plan at this surface's slots == the panel's os rows
            Ob, _ = join(banks[f"eval{s}"], E["clip"], E["ws"], "traj")
            res["controls"][f"s{s}"]["K4c_bank_slots_vs_panel_os_maxabs_m"] = float(np.abs(Ob[:, SLOTS[surf], :] - E["O"].astype(np.float64)).max())
    # inference-seed floor of the emitted plan (same rows at both seeds)
    a0 = comp(tools, ev[0]["O"], ev[0]["G"], ev[0]["dt"])["ade_m"]
    a1 = comp(tools, ev[1]["O"], ev[1]["G"], ev[1]["dt"])["ade_m"]
    res["inference_seed_floor_os"] = cell(tools, a0, a1, ev[0]["eid"])
    return res


SLOTS = {"S2": [0, 1, 2, 3], "S6": [1, 3, 4, 5, 6, 7]}


def main() -> int:
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dryrun-synthetic", action="store_true")
    ap.add_argument("--surfaces", default="S2,S6")
    a = ap.parse_args()
    DRY = bool(a.dryrun_synthetic)
    global OUTDIR
    OUTDIR = Path(a.out)
    t0 = time.time()
    import compliant_reread_r8 as CR
    import retime_trainfit_r8 as TF
    tools = RT.tools_load()
    keys = [m[1] for m in tools["PR"].METRICS]
    surfs = [x for x in a.surfaces.split(",") if x]
    out = {"tool": "l1_fit_score.py", "generated": time.strftime("%FT%T%z"), "script_sha256": sha256_file(__file__),
           "prereg_sha256": (sha256_file(ROOT / "PREREG_L1.md") if (ROOT / "PREREG_L1.md").exists() else None),
           "dryrun_synthetic": DRY, "tree": TREE if not DRY else "step30000ema_unc",
           "tier": "T1 self-action open loop on logged eval139 frames; arms are post-hoc emulations of the in-forward re-timing of refcv8's "
                   "post-cap emitted path (cap-binding windows' in-forward difference UNMEASURED)",
           "estimator": "pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0 (episodes only, models fixed)",
           "frozen_pf_table_sha256": sha256_file(FROZEN_PF), "surfaces": {}, "bank_controls": {}}
    ev = {surf: {} for surf in surfs}
    first = {}
    for s in SEEDS:
        for surf in surfs:
            ev[surf][s] = load_eval(tools, CR, s, first, surf)
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ftr = {}
    trs = {surf: load_train(tools, CR2, TF, surf, ftr) for surf in surfs}
    bank = Path(a.bank)
    if DRY:
        rng = np.random.default_rng(12345)
        allk = {}
        for s in SEEDS:
            E = ev["S2"][s]
            allk[s] = (E["clip_all"], E["ws_all"])
        d = 512
        banks = {}
        for name, (cl, wsv, O8) in {"eval0": (*allk[0], None), "eval1": (*allk[1], None)}.items():
            banks[name] = {"z": rng.standard_normal((len(cl), d)), "traj": None, "idx": {(int(c), int(w)): j for j, (c, w) in enumerate(zip(cl, wsv))}}
        trk = sorted({(int(c), int(w)) for surf in surfs for c, w in zip(trs[surf]["clip"], trs[surf]["ws"])})
        banks["train"] = {"z": rng.standard_normal((len(trk), d)), "idx": {k: j for j, k in enumerate(trk)},
                          "traj": np.zeros((len(trk), 8, 2)), "v0_in": np.full(len(trk), np.nan)}
        for surf in surfs:   # synthetic train 'os' = midway prior/human on 8 slots (only H1's input; dry-run only)
            for j, (c, w) in enumerate(zip(trs[surf]["clip"], trs[surf]["ws"])):
                jj = banks["train"]["idx"][(int(c), int(w))]
                banks["train"]["traj"][jj][SLOTS[surf]] = 0.5 * (trs[surf]["HP"][j] + trs[surf]["G"][j])
                banks["train"]["v0_in"][jj] = trs[surf]["v0"][j]
    else:
        banks = {"eval0": load_bank(bank / "ztac_eval_s0.npz"), "eval1": load_bank(bank / "ztac_eval_s1.npz"),
                 "train": load_bank(bank / "ztac_train_s0.npz")}
        # ---- bank controls: K4 (OWN == banked battery dump), K4b (OWN slots == panel os), K5 (alignment), K6 (seed dependence)
        for s in SEEDS:
            b = banks[f"eval{s}"]
            # AMENDMENT 1: seed 0's battery dump is the FULL arm-set draw -> K4 at s0 is judged against the os-only seed-0 child (r8oracle)
            ref = (OSONLY_S0 if s == 0 else ARCH / TREE / f"dump_s{s}" / "refcv6_extras.npz")
            ex = np.load(ref)
            kx = list(zip(ex["clip_index"].astype(int), ex["ws"].astype(int)))
            jb = np.array([b["idx"].get(k, -1) for k in kx])
            ok_align = bool((jb >= 0).all() and len(b["idx"]) == len(kx))
            dmax = float(np.abs(b["traj"][jb] - ex["nav_true.traj"].astype(np.float64)).max()) if ok_align else float("nan")
            E = ev["S2"][s]
            jp = np.array([b["idx"].get((int(c), int(w)), -1) for c, w in zip(E["clip_all"], E["ws_all"])])
            pmax = float(np.abs(b["traj"][jp][:, SLOTS["S2"]] - E["O_all"].astype(np.float64)).max()) if (jp >= 0).all() else float("nan")
            out["bank_controls"][f"s{s}"] = {"K5_eval_alignment_equal": ok_align, "n_bank": len(b["idx"]), "n_dump": len(kx),
                                             "K4_own_vs_dump_maxabs_m": dmax, "K4b_own_slots_vs_panel_os_maxabs_m": pmax,
                                             "K4_reference": str(ref), "K4b_binding": bool(s == 1),
                                             "PASS": bool(ok_align and dmax <= 1e-5 and (pmax <= 1e-5 or s == 0)), "bank_meta": b["meta"]}
        z0, z1 = banks["eval0"], banks["eval1"]
        common = sorted(set(z0["idx"]) & set(z1["idx"]))
        dz = np.abs(z0["z"][[z0["idx"][k] for k in common]] - z1["z"][[z1["idx"][k] for k in common]])
        out["bank_controls"]["K6_ztac_seed_dependence"] = {"n": len(common), "max_abs": float(dz.max()),
                                                           "max_abs_over_rms": float(dz.max() / np.sqrt((z0["z"] ** 2).mean()))}
        out["bank_controls"]["K6_ztac_seed_dependence"]["PASS"] = bool(out["bank_controls"]["K6_ztac_seed_dependence"]["max_abs_over_rms"] <= 1e-3)
        out["bank_controls"]["train_meta"] = banks["train"]["meta"]
    gains_all = json.loads(FROZEN_PF.read_text(encoding="utf-8"))["surfaces"]
    for surf in surfs:
        print(f"[l1] fitting + scoring {surf} ...", flush=True)
        out["surfaces"][surf] = run_surface(tools, surf, ev[surf], trs[surf], banks, gains_all[surf]["all"]["RETIME"], keys)
    # K7: H2s on S2 reproduces the 30k probe's fit exactly (model-free target, features and rows)
    if "S2" in out["surfaces"]:
        pb = json.loads((HERE / "probe_repro" / "PROGRESS_SCALAR.banked_30k.json").read_text(encoding="utf-8"))["fit"]
        f2 = out["surfaces"]["S2"]["fit"]["H2s"]
        out["controls_K7"] = {"lambda": [f2["lambda"], pb["lambda"]], "train_R2": [f2["train_R2"], pb["train_R2_per_instant"]],
                              "n": [f2["n"], pb["n_windows"]],
                              "PASS": bool(f2["lambda"] == pb["lambda"] and f2["n"] == pb["n_windows"]
                                           and max(abs(x - y) for x, y in zip(f2["train_R2"], pb["train_R2_per_instant"])) <= 1e-6)}
    # K1: H0 reproduces the banked per-surface RETIME_TRAINFIT cell on this tree; OS-pose reproduces too
    if not DRY:
        bt = json.loads(BANKED_TF.read_text(encoding="utf-8"))
        k1 = {}
        for surf in surfs:
            for s in SEEDS:
                try:
                    arms_b = bt["seeds"][str(s)]["surfaces"][surf]["sets"]["compliant"]["arms"]
                    want = arms_b["RETIME_TRAINFIT"]["minus_ha0_ext_pose"]["delta"]
                except Exception as e:  # noqa: BLE001
                    want = f"unreadable: {e!r}"
                got = out["surfaces"][surf]["seeds"][str(s)]["E1_minus_ha0_ext_pose"]["H0"]["delta"]
                k1[f"{surf}|s{s}"] = {"got": got, "banked": want, "PASS": bool(isinstance(want, float) and abs(got - want) <= 1e-4)}
        out["controls_K1"] = k1
    out["wall_s"] = round(time.time() - t0, 1)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "L1_RESULT.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[l1] WROTE {o / 'L1_RESULT.json'} wall {out['wall_s']}s ZZL1FITDONEZZ", flush=True)
    return 0


DRY = False
OUTDIR = None

if __name__ == "__main__":
    raise SystemExit(main())
