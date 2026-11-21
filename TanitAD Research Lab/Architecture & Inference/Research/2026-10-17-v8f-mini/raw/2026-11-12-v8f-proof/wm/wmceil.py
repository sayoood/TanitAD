"""H-V8F-PROOF-3 WMCEIL (card: F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF.md). T0 world-model diagnostic, dev-box CPU only.

Is the learned v8F-mini world model below a closed-form linear predictor on the SAME target latents?
Inputs (read-only): eval-141 level-1 latents + raw actions from the banked WMACT collection (latents.pt, built from the eval-141 cache),
the two M2b HACC checkpoints. Everything else is computed here. Pure statistics (ridge, ceilings) live in functions that need no torch/model,
so test_wmceil.py can run them on synthetic data.

Run:  PYTHONPATH=<tree>/stack  python wmceil.py [--smoke]        (CUDA_VISIBLE_DEVICES=-1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "E:/Projects/TanitAD-artifacts"
WMACT = f"{ART}/dagland/motion_c/theory/wmact"
LATENTS = f"{WMACT}/latents.pt"
BANKED = f"{WMACT}/RESULT_WMACT.json"
CKPT = {"M2b_s0": f"{ART}/v8f-mini-hacc/M2b_s0_hacc.pt", "M2b_s1": f"{ART}/v8f-mini-hacc/M2b_s1_hacc.pt"}
KS = {1: (1, 5, 10), 2: (1, 3), 3: (1, 3)}              # banked cells (level -> k), as wmact_lib.KS
STEP_S = {1: 0.2, 2: 0.4, 3: 0.8}                        # seconds per step at each level
HEADLINE = [(1, 1), (2, 1), (3, 1), (1, 10)]             # 0.2, 0.4, 0.8, 2.0 s
RHO_L1_LAGS = (1, 2, 4, 5, 10)                           # L1 lags in 0.2 s steps: 0.2 0.4 0.8 1.0 2.0 s
LAMS = np.logspace(-4, 2, 13)
B_BOOT, SEED, N_FOLDS = 2000, 0, 5
MARGIN = 0.01


# ================================================================================================ pure statistics (no torch)
def ar1_ceiling(rho):
    """Predictable fraction of the change z_k - z_0 for a stationary AR(1) feature with lag correlation rho: (1 - rho) / 2.
    Derivation: E(z_k - z_0)^2 = 2 s^2 (1 - rho); the best predictor rho * z_0 leaves s^2 (1 - rho^2); skill = 1 - (1 + rho) / 2."""
    return (1.0 - np.asarray(rho, dtype=np.float64)) / 2.0


def lag_corr(z0: np.ndarray, zk: np.ndarray) -> np.ndarray:
    """Per-dimension Pearson correlation of z0 [N, D] and zk [N, D] across windows."""
    a, b = z0 - z0.mean(0), zk - zk.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return (a * b).sum(0) / np.where(den > 0, den, np.nan)


def rho_summary(rho: np.ndarray) -> dict:
    r = rho[np.isfinite(rho)]
    return {"mean": float(r.mean()), "median": float(np.median(r)), "p10": float(np.percentile(r, 10)),
            "p90": float(np.percentile(r, 90)), "n_dims": int(r.size),
            "ceiling_of_mean_rho": float(ar1_ceiling(r.mean())), "ceiling_mean_over_dims": float(ar1_ceiling(r).mean())}


def _std(X, mu, sd):
    return (X - mu) / sd


class RidgePath:
    """Closed-form ridge with intercept on standardised features, all lambdas from one eigendecomposition. lambda is on the covariance scale
    (C = Xs'Xs / n), so it is comparable across n and feature counts."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.ym = Y.mean(0)
        Xs = _std(X, self.mu, self.sd)
        n = X.shape[0]
        C = Xs.T @ Xs / n
        self.lam_e, self.V = np.linalg.eigh(C)
        self.VB = self.V.T @ (Xs.T @ (Y - self.ym) / n)

    def predict(self, X: np.ndarray, lam: float) -> np.ndarray:
        Xs = _std(X, self.mu, self.sd)
        W = self.V @ (self.VB / (np.maximum(self.lam_e, 0.0) + lam)[:, None])
        return Xs @ W + self.ym


def _folds_of(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """fold id per sample, assigning whole groups (episodes) to folds with a fixed-seed permutation."""
    u = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(u))
    fold_of = np.empty(len(u), dtype=int)
    fold_of[perm] = np.arange(len(u)) % n_folds
    return fold_of[np.searchsorted(u, groups)]


def choose_lambda(X, Y, groups, lams=LAMS, n_inner=4, seed=1):
    """lambda chosen on an INNER split of the training episodes (never on the scored fold): the minimiser of the summed held-out squared error."""
    f = _folds_of(groups, n_inner, seed)
    sse = np.zeros(len(lams))
    for i in range(n_inner):
        tr, va = f != i, f == i
        if va.sum() == 0 or tr.sum() == 0:
            continue
        rp = RidgePath(X[tr], Y[tr])
        for j, lam in enumerate(lams):
            sse[j] += ((rp.predict(X[va], lam) - Y[va]) ** 2).sum()
    return float(lams[int(np.argmin(sse))])


def cross_fit(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, fold: np.ndarray, lams=LAMS):
    """Cross-fitted ridge predictions of Y from X: each outer fold is predicted by a model fitted on the other folds only, with lambda chosen on an
    inner episode split of those. Returns (pred [N, D], chosen lambda per fold)."""
    pred = np.zeros_like(Y, dtype=np.float64)
    lam_used = {}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        lam = choose_lambda(X[tr], Y[tr], groups[tr], lams)
        pred[te] = RidgePath(X[tr], Y[tr]).predict(X[te], lam)
        lam_used[int(f)] = lam
    return pred, lam_used


def sqerr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.float64) - b.astype(np.float64)) ** 2).sum(-1)


def skill_from_err(err: np.ndarray, copy: np.ndarray) -> float:
    return float(1.0 - err.sum() / copy.sum())


# ------------------------------------------------------------------------------------------------ bootstrap (the draws of wmact_lib / taniteval.ci)
def boot_weights(eid, n_boot=B_BOOT, seed=SEED):
    uniq = np.unique(np.asarray([str(e) for e in eid]))
    rng = np.random.default_rng(seed)
    n_ep = len(uniq)
    W = np.zeros((n_boot, n_ep))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n_ep, replace=True)
        W[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n_ep)
    return uniq, W


def ep_sum(v, eid, uniq):
    pos = np.searchsorted(uniq, np.asarray([str(e) for e in eid]))
    s = np.zeros(len(uniq))
    np.add.at(s, pos, np.asarray(v, np.float64))
    return s


def diff_row(es: dict, den: str, plus: str, minus: str, W) -> dict:
    """(sum err_plus - sum err_minus) / sum den, episode-cluster bootstrap. With plus = linear error and minus = learned error this is
    skill_learned - skill_linear."""
    p = float((es[plus].sum() - es[minus].sum()) / es[den].sum())
    b = (W @ es[plus] - W @ es[minus]) / (W @ es[den])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"point": p, "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}


# ================================================================================================ the WM side (torch)
def load_model(path):
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = model.load_state_dict(ck["model"], strict=False)
    assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}, (r.missing_keys, r.unexpected_keys)
    return model.eval(), cfg


def level_pack(wm, Z1, A, chunk: int = 512) -> dict:
    """VERBATIM the banked reader's rule (wmact_lib.level_pack): TRUE states z[l] = HJEPAStack.encode(z1, a) lifted frames, rollout pred[(l, k)] =
    HJEPAStack.rollout (context 1) from the start state under the level's actions (raw at level 1, macro embeddings of the same actions above).
    Extra here: the per-level action inputs, returned in ``act`` ([N, T_l - 1, d_a])."""
    import torch
    st = wm.stack
    true, pred, act = {}, {}, {}
    with torch.no_grad():
        for s0 in range(0, Z1.shape[0], chunk):
            sl = slice(s0, s0 + chunk)
            enc = st.encode(Z1[sl], A[sl])
            for l, ks in KS.items():
                H = max(ks)
                acts = A[sl][:, :H] if l == 1 else enc.act[l - 1][:, :H]
                roll = st.rollout(l, enc.z[l - 1][:, 0], acts)
                true.setdefault(l, []).append(enc.z[l - 1])
                act.setdefault(l, []).append(A[sl] if l == 1 else enc.act[l - 1])
                for k in ks:
                    pred.setdefault((l, k), []).append(roll[:, k - 1])
    cat = torch.cat
    return {"true": {l: cat(v).double().numpy() for l, v in true.items()}, "act": {l: cat(v).double().numpy() for l, v in act.items()},
            "pred": {k: cat(v).double().numpy() for k, v in pred.items()}}


# ================================================================================================ the read
def action_block(act_l: np.ndarray, k: int) -> np.ndarray:
    return act_l[:, :k].reshape(act_l.shape[0], -1)


def fold_ids(eid, seed=SEED):
    return _folds_of(np.asarray([str(e) for e in eid]), N_FOLDS, seed)


def compose_check(zt: np.ndarray, act1: np.ndarray, eid, fold, k: int, use_act: bool):
    """(d): per fold, a one-step (0.2 s) ridge fitted on ALL adjacent pairs of the other folds' windows, composed k times from z_0 with the true actions,
    against the direct k-step ridge's held-out error. Returns per-window composed squared error [N] and the one-step error at the start [N] (the
    window's first transition), both cross-fitted."""
    N, T, D = zt.shape
    comp = np.zeros(N)
    eps1 = np.zeros(N)
    eid = np.asarray([str(e) for e in eid])
    for f in np.unique(fold):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Zc, Zn = zt[tr, :-1].reshape(-1, D), zt[tr, 1:].reshape(-1, D)
        X = np.concatenate([Zc, act1[tr].reshape(-1, act1.shape[-1])], 1) if use_act else Zc
        grp = np.repeat(eid[tr], T - 1)
        lam = choose_lambda(X, Zn - Zc, grp, n_inner=3)
        rp = RidgePath(X, Zn - Zc)
        z = zt[te, 0].copy()
        for j in range(k):
            Xj = np.concatenate([z, act1[te, j]], 1) if use_act else z
            z = z + rp.predict(Xj, lam)
            if j == 0:
                eps1[te] = sqerr(z, zt[te, 1])
        comp[te] = sqerr(z, zt[te, k])
    return comp, eps1


def run(smoke: bool = False) -> dict:
    import torch
    import tanitad
    t0 = time.time()
    tdir = os.path.abspath(os.path.join(HERE, "tree")).replace("\\", "/").lower()
    tf = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    assert tf.startswith(tdir), f"tanitad imported from {tf}, not from {tdir}"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    L = torch.load(LATENTS, weights_only=False)
    ok = L["ok"].all(1).numpy()
    sel = np.flatnonzero(ok)
    eid = np.array(L["eid"])[sel]
    if smoke:
        keep = np.unique(eid)[:25]
        m = np.isin(eid, keep)
        sel, eid = sel[m], eid[m]
    uniq, Wt = boot_weights(eid)
    fold = fold_ids(eid)
    banked = json.load(open(BANKED))
    res = {"card": "PREREG_V8F_PROOF.md H-V8F-PROOF-3 WMCEIL", "tier": "T0 (teacher-forced, logged actions, true future frames; a world-model diagnostic, never driving performance)",
           "evidence_class": "MEASURED (this script on eval-141, M2b s0/s1)", "smoke": bool(smoke), "windows_used": int(len(sel)), "episodes_used": int(len(uniq)),
           "folds": {"n_folds": N_FOLDS, "seed": SEED, "episodes_per_fold": [int(len(np.unique(eid[fold == f]))) for f in range(N_FOLDS)],
                     "windows_per_fold": [int((fold == f).sum()) for f in range(N_FOLDS)]},
           "inputs": {"latents": LATENTS, "latents_sha256": hashlib.sha256(open(LATENTS, "rb").read()).hexdigest(), "banked": BANKED,
                      "checkpoints": CKPT, "tanitad_file": tf, "estimator": "episode-cluster bootstrap B 2,000 seed 0 over the episodes of the used windows; skill = 1 - sum err / sum C (ratio of sums)"},
           "seeds": {}}
    for tag, path in CKPT.items():
        model, cfg = load_model(path)
        wm = model.wm
        Z1 = L["Z1"][tag][sel].float()
        A_sc = wm.scale_actions(L["A_raw"][sel].float())[:, :-1]
        pk = level_pack(wm, Z1, A_sc)
        out = {"cells": {}, "rho_L1_lags": {}, "control_c": {}, "compose": {}}
        # (a) rho at the L1 lags (0.2 .. 2.0 s) on z1, beside the level-native cells below
        z1 = pk["true"][1]
        for g in RHO_L1_LAGS:
            r = lag_corr(z1[:, 0], z1[:, g])
            out["rho_L1_lags"][f"{g * 0.2:.1f}s"] = {"pooled": rho_summary(r), "per_fold_mean_rho": [float(np.nanmean(lag_corr(z1[fold == f, 0], z1[fold == f, g]))) for f in range(N_FOLDS)]}
        # cells: (a) rho + ceiling, (b) cross-fitted ridge, (c) learned WM by the banked rule
        for l, ks in KS.items():
            zt, at = pk["true"][l], pk["act"][l]
            for k in ks:
                key = f"L{l}/k{k}"
                z0, zk = zt[:, 0], zt[:, k]
                copy = sqerr(z0, zk)
                err_wm = sqerr(pk["pred"][(l, k)], zk)
                Y = zk - z0
                p_z, lam_z = cross_fit(z0, Y, eid, fold)
                p_a, lam_a = cross_fit(np.concatenate([z0, action_block(at, k)], 1), Y, eid, fold)
                err_z, err_a = sqerr(z0 + p_z, zk), sqerr(z0 + p_a, zk)
                es = {n: ep_sum(v, eid, uniq) for n, v in {"C": copy, "wm": err_wm, "lin_z": err_z, "lin_a": err_a}.items()}
                r = lag_corr(z0, zk)
                cell = {"horizon_s": round(STEP_S[l] * k, 2), "rho": rho_summary(r),
                        "rho_per_fold_mean": [float(np.nanmean(lag_corr(z0[fold == f], zk[fold == f]))) for f in range(N_FOLDS)],
                        "skill_learned": skill_from_err(err_wm, copy), "SKILL_lin_z": skill_from_err(err_z, copy), "SKILL_lin_za": skill_from_err(err_a, copy),
                        "skill_per_fold": {"learned": [skill_from_err(err_wm[fold == f], copy[fold == f]) for f in range(N_FOLDS)],
                                           "lin_za": [skill_from_err(err_a[fold == f], copy[fold == f]) for f in range(N_FOLDS)],
                                           "lin_z": [skill_from_err(err_z[fold == f], copy[fold == f]) for f in range(N_FOLDS)]},
                        "learned_minus_lin_za": diff_row(es, "C", "lin_a", "wm", Wt), "learned_minus_lin_z": diff_row(es, "C", "lin_z", "wm", Wt),
                        "lambda_chosen": {"z": lam_z, "za": lam_a}, "n_action_features": int(action_block(at, k).shape[1])}
                out["cells"][key] = cell
                ban = banked["rows"][tag][key]["SKILL"]["point"]
                out["control_c"][key] = {"reproduced": cell["skill_learned"], "banked": ban, "abs_diff": abs(cell["skill_learned"] - ban)}
        # (d) direct 2 s ridge vs the 10-fold composed 0.2 s ridge (level 1)
        zt1, a1 = pk["true"][1], pk["act"][1]
        for name, use_act in (("za", True), ("z", False)):
            comp, eps1 = compose_check(zt1, a1, eid, fold, 10, use_act)
            direct = out["cells"]["L1/k10"][f"skill_per_fold"]["lin_za" if use_act else "lin_z"]
            # direct squared error per fold from the cell's cross-fitted prediction
            Y10 = zt1[:, 10] - zt1[:, 0]
            Xd = np.concatenate([zt1[:, 0], action_block(a1, 10)], 1) if use_act else zt1[:, 0]
            pd_, _ = cross_fit(Xd, Y10, eid, fold)
            err_d = sqerr(zt1[:, 0] + pd_, zt1[:, 10])
            per_fold = []
            for f in range(N_FOLDS):
                m = fold == f
                e_d, e_c, e_1 = err_d[m].mean(), comp[m].mean(), eps1[m].mean()
                per_fold.append({"eps_k_direct": float(e_d), "composed": float(e_c), "eps_1": float(e_1), "ratio_eps_k_over_k_eps1": float(e_d / (10 * e_1)),
                                 "composed_above_direct": bool(e_c > e_d)})
            es = {n: ep_sum(v, eid, uniq) for n, v in {"d": err_d, "c": comp, "e1": eps1}.items()}
            out["compose"][name] = {"per_fold": per_fold, "ratio_pooled": float(err_d.mean() / (10 * eps1.mean())),
                                    "composed_minus_direct_over_direct": float((comp.sum() - err_d.sum()) / err_d.sum()),
                                    "jump_dominates": bool(err_d.mean() / (10 * eps1.mean()) < 1 and all(p["composed_above_direct"] for p in per_fold)),
                                    "ratio_boot": _ratio_boot(es, Wt)}
        res["seeds"][tag] = out
        print(tag, "done", round(time.time() - t0, 1), "s", flush=True)
    res["verdict"] = verdict(res)
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def _ratio_boot(es, W):
    b = (W @ es["d"]) / (10 * (W @ es["e1"]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def verdict(res: dict) -> dict:
    """The card's labels, headline horizons 0.2 / 0.4 / 0.8 / 2.0 s, comparator = the ridge on (z_t, the WM's action inputs).
    Card wording: BELOW = learned < SKILL_lin - 0.01 at two or more horizons, both seeds; AT-CEILING = |learned - SKILL_lin| <= 0.01 and
    SKILL_lin < 0.05 (read here, like BELOW, as 'at two or more horizons, both seeds': an implementation choice, stated in VERDICT_WMCEIL.md)."""
    ctrl_max = max(v["abs_diff"] for s in res["seeds"].values() for kk, v in s["control_c"].items() if tuple(int(x[1:]) for x in kk.split("/")) in HEADLINE)
    per_seed = {}
    for tag, s in res["seeds"].items():
        below = at = above = 0
        rows = {}
        for (l, k) in HEADLINE:
            c = s["cells"][f"L{l}/k{k}"]
            d = c["skill_learned"] - c["SKILL_lin_za"]
            b, a_c, a_u = d < -MARGIN, (abs(d) <= MARGIN and c["SKILL_lin_za"] < 0.05), d > MARGIN
            below, at, above = below + b, at + a_c, above + a_u
            rows[f"L{l}/k{k}"] = {"horizon_s": c["horizon_s"], "learned": c["skill_learned"], "SKILL_lin_za": c["SKILL_lin_za"], "SKILL_lin_z": c["SKILL_lin_z"], "diff": d,
                                  "label": "BELOW" if b else "AT-CEILING" if a_c else "ABOVE" if a_u else "within margin, SKILL_lin >= 0.05",
                                  "ci_separated": c["learned_minus_lin_za"]["separated"]}
        per_seed[tag] = {"n_below": below, "n_at": at, "n_above": above, "rows": rows,
                         "jump_dominates": {n: v["jump_dominates"] for n, v in s["compose"].items()}}
    allseeds = lambda key: all(v[key] >= 2 for v in per_seed.values())
    label = "WM-BELOW-LINEAR" if allseeds("n_below") else "WM-AT-CEILING" if allseeds("n_at") else "WM-ABOVE-LINEAR" if allseeds("n_above") else "UNCLASSIFIED (no card label holds at 2+ horizons at both seeds)"
    jump = all(v["jump_dominates"]["za"] for v in per_seed.values())
    nxt = {"WM-BELOW-LINEAR": "write the ridge solution into the predictor as its base (persistence + linear + learned residual) under its own card",
           "WM-AT-CEILING": "change the target (longer horizon, un-pooled tokens, whitened change), not the predictor"}.get(label, "outside the card's committed branches: report both values and decide with the PI")
    return {"label": label, "JUMP-DOMINATES": bool(jump), "control_c_max_abs_diff_headline": float(ctrl_max), "control_c_ok": bool(ctrl_max <= 0.01), "per_seed": per_seed, "next_lever_committed": nxt}


def write_outputs(res: dict, outdir: str, name: str = "WMCEIL"):
    json.dump(res, open(os.path.join(outdir, f"RESULT_{name}.json"), "w"), indent=1, default=float)
    v = res["verdict"]
    L = [f"# VERDICT_{name}: H-V8F-PROOF-3 (is the learned world model below the closed-form linear predictor?)", "",
         f"MEASURED, T0 (a world-model diagnostic, never driving performance). Raw: RESULT_{name}.json. eval-141, {res['windows_used']} windows, {res['episodes_used']} episodes, {N_FOLDS} folds by episode (seed {SEED}); episode-cluster bootstrap B 2,000.",
         f"Smoke run: {res['smoke']}", "",
         f"**Control (c): learned skill by the banked rule vs RESULT_WMACT.json, max abs diff over the four headline cells = {v['control_c_max_abs_diff_headline']:.6f}; {'OK (<= 0.01)' if v['control_c_ok'] else 'FAILED (> 0.01): the read STOPs'}.**",
         f"**Outcome: {v['label']}**   JUMP-DOMINATES: {v['JUMP-DOMINATES']}", f"Next lever (card): {v['next_lever_committed']}", ""]
    for tag, ps in v["per_seed"].items():
        L += [f"## {tag}", "| horizon | learned | SKILL_lin (z, a) | SKILL_lin (z) | learned - lin (z, a) | CI separated | label |", "|---|---|---|---|---|---|---|"]
        for key, r in ps["rows"].items():
            L.append(f"| {r['horizon_s']} s ({key}) | {r['learned']:+.4f} | {r['SKILL_lin_za']:+.4f} | {r['SKILL_lin_z']:+.4f} | {r['diff']:+.4f} | {r['ci_separated']} | {r['label']} |")
        s = res["seeds"][tag]
        L += ["", "rho of the target latents (per-dimension; mean / median / p10 / p90) and the AR(1) ceiling (1 - rho)/2:", "| cell | rho mean | median | p10 | p90 | ceiling of mean rho | measured SKILL_lin (z) |", "|---|---|---|---|---|---|---|"]
        for key, c in s["cells"].items():
            r = c["rho"]
            L.append(f"| {key} ({c['horizon_s']} s) | {r['mean']:.3f} | {r['median']:.3f} | {r['p10']:.3f} | {r['p90']:.3f} | {r['ceiling_of_mean_rho']:+.4f} | {c['SKILL_lin_z']:+.4f} |")
        L += ["", "(d) direct 2 s ridge vs 10-fold composed 0.2 s ridge (level 1), ratio = eps_k / (k eps_1), per fold:"]
        for nm, cp in s["compose"].items():
            L.append(f"* inputs {nm}: pooled ratio {cp['ratio_pooled']:.3f} (boot 95% [{cp['ratio_boot']['lo']:.3f}, {cp['ratio_boot']['hi']:.3f}]); per fold " + ", ".join(f"{p['ratio_eps_k_over_k_eps1']:.3f}" for p in cp["per_fold"]) +
                     f"; composed above direct on every fold: {all(p['composed_above_direct'] for p in cp['per_fold'])}; JUMP-DOMINATES {cp['jump_dominates']}")
        L.append("")
    L += ["Evidence class: MEASURED (this script; files above). Tier T0. STRATEGIC family: PAUSED (PI item 25), not part of this read.",
          "Implementation choices (card silent): cells = the banked cells (level-native steps 0.2 / 0.4 / 0.8 s; 2.0 s = L1 k10); start frame = window frame 0, windows with 16 real frames (as the banked reader); action inputs = WM-scaled raw actions at L1 and the stack's macro-action embeddings at L2 / L3 (as the WM gets them); lambda from an inner 4-fold episode split of the training folds; the one-step ridge in (d) uses all adjacent pairs of the training folds; AT-CEILING read at two or more horizons like BELOW."]
    open(os.path.join(outdir, f"VERDICT_{name}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    r = run(a.smoke)
    name = "WMCEIL" + ("_SMOKE" if a.smoke else "")
    write_outputs(r, a.out, name)
    ok = r["verdict"]["control_c_ok"]
    tag = f"{r['verdict']['label']} JUMP={r['verdict']['JUMP-DOMINATES']} ctrl_c_max={r['verdict']['control_c_max_abs_diff_headline']:.5f}"
    if a.smoke:
        print("smoke", tag)
    elif ok:
        open(os.path.join(a.out, "WMCEIL_DONE.txt"), "w").write(f"ZZ WMCEIL-DONE {tag} runtime_s={r['runtime_s']}\n")
    else:
        open(os.path.join(a.out, "WMCEIL_STOP.txt"), "w").write(f"ZZ WMCEIL-STOP control (c) did not reproduce the banked numbers within 0.01: max abs diff {r['verdict']['control_c_max_abs_diff_headline']:.5f}; see RESULT_WMCEIL.json\n")
    print(tag)
