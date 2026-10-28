#!/usr/bin/env python3
"""v8f_mini_eval_speedread.py -- the dev-box instrument of PREREG_V8F_SPEEDREAD (DRAFT): is the per-window speed information O needs PRESENT IN T'S OWN INPUT?

Zero training of the driving model, CPU only (CUDA_VISIBLE_DEVICES=-1). Named `v8f_mini_eval_*` so box 2's guard regex (`box2_ops/v8f_guard.sh`) would match it if it were ever run there.

THE QUESTION. KINDX: replacing only T's five SPEED cells by the GT cuts O's own-code ADE6 by 0.716 / 0.668 m (T0). SPEEDCAL: no affine map of T's OWN speed cells moves ADE6, not even one fitted
on eval (INFORMATION-LIMITED). WC: T's train constraint loss falls 18-25 % under an 8x weight while its eval error falls 2.4 %. So: is the right per-window speed in what T SEES (its scene tokens and
the ego-motion features it already gets), and T simply fails to extract it in a way that generalises -- or is it not there at all?

THE PROBE. A read-out of the five GT speed cells (physical m/s, at T's tick frame) from T's INPUT ONLY, FITTED ON TRAINING CLIPS ONLY (the 4,512 clips of X_s, the trainer's own builder and
--exclude 540:600 rule), SCORED on eval-141. Never fitted, tuned or selected on what it scores.
  input   = exactly the slice `V8FMini.run_t` reads: the `hist` = 4 scene frames ending at T's tick (last_t = F-1-a_t), `p10f` / `nav` at the tick, `limit`, `limit_ok`. S's checkpoints are NOT
            used (they are S's output, computed from the same input class). Keys read: READ_KEYS, a subset of the model's INFERENCE_INPUT_KEYS (checked).
  vector  = per-frame token mean of the 4 frames (4 x 1024) + T's tick frame pooled 4x10 -> 2x5 (10 x 1024) = 14,336 scene dims -> PCA 512 (fitted on every 16th TRAIN window, unsupervised)
            + the ego dims (p10f 8, nav 5, limit at its data width (4 on the real cache), limit_ok 1, frame_ok 4: 22); standardised with TRAIN statistics.
  LIN     = ridge per cell; lambda from LAMBDAS by grouped CV over TRAIN folds 1-4 (fold = sha256(clip) mod 5); refit on all TRAIN folds.
  MLP     = (512 + d_ego, 534 on the real cache) -> 256 -> 256 -> 5 (GELU, dropout 0.1, AdamW); early stopping on TRAIN fold 0 (held-out clips), fitted on folds 1-4. Hyper-parameters fixed here, never tuned.
  PRIMARY = whichever of LIN / MLP has the lower fold-0 (held-out TRAIN clips) RMSE, in train-SD units averaged over cells. Chosen in `fit`, BEFORE any eval batch is opened.
  floors  = CONST (the train mean per cell: skill exactly 0.0, the no-information value), EGO-only LIN / MLP (beside: does vision add?), PERM (the PRIMARY's class on permuted targets).

THE DECISION (read by `read`): at both seeds, ADE6 of X_s with T's five speed cells replaced IN THE CODE O READS by the PRIMARY read-out, minus X_s's own; the outcome table is
`decide()` (verbatim from the PREREG). T1-class open loop (X_s's own held S / T outputs, the eval's `--tac oracle --tac-groups speed` carrier, SPEEDCAL's tier). The `gt` mode is T0 (a ceiling).

SUBCOMMANDS
  extract  --cache ROOT --join J --tracks T --out DIR      the TRAIN windows (trainer's builder) -> raw scene memmap (fp16), ego, targets, window list (not resumable: ~20-30 min, re-run whole)
  fit      --train DIR --out DIR                            PCA, standardisers, LIN, MLP, EGO, CONST, PERM -> readout.npz + fit.json (the selection is recorded here)
  evals    --seed S --x CKPT --expect-sha H --readout R ... `v8f_mini_eval.py run` per mode (own, identity, gt, read, alt, const, perm), with the hook; resumable
  read     --eval-dir-s0 D0 --eval-dir-s1 D1 ...            controls, decision rows, outcome -> READ.json, OUTCOME.txt

PREREG_V8F_SPEEDSUB (DRAFT) adds two subcommands and leaves the four above, MODES and every SPEEDREAD code path as they were:
  subevals --seed S ... --expect-readout-sha H               `evals` for SUB_MODES (own, identity, gt, ego, egoshuf), tag prefix `speedsub`. `ego` = the BANKED `mlp_ego` read-out replaces the five
                                                             speed cells O reads (no re-fit, no tuning; it reads EGO_READ_KEYS only and never a pixel); `egoshuf` = the same cells shuffled across the
                                                             eval windows (a fixed permutation): the deliberate-regression arm
  subread  --fit-json F --eval-dir-s0 D0 --eval-dir-s1 D1 ... controls S1..S13, paired ADE6 rows, `decide_sub()` -> SUBREAD.json, OUTCOME.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

SCHEMA = "v8f-speedread-v1"
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
READ_KEYS = ("feats", "frame_ok", "p10f", "nav", "limit", "limit_ok", "a_t")
HIST = 4
POOL_TO = (2, 5)
PCA_K = 512
PCA_EVERY = 16
PCA_SEED = 20261027
N_FOLDS = 5
HOLDOUT_FOLD = 0
LAMBDAS = (1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
MLP_CFG = {"hidden": 256, "dropout": 0.1, "lr": 1e-3, "wd": 1e-2, "batch": 1024, "max_epochs": 40, "patience": 4, "seed": 0}
PERM_SEED = 20261028
EXCLUDE = (540, 600)
#: MEASURED from X_s0's own summary.json / selfcheck.json (box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo): the parity the train extraction must reproduce
TRAIN_PARITY = {"n_clips": 4512, "n_windows": 306566, "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"}
F_M = 0.1207
MODES = ("own", "identity", "gt", "read", "alt", "const", "perm")
# ---- PREREG_V8F_SPEEDSUB. MODES above is NOT changed: every SPEEDREAD flow reads it.
SUB_SCHEMA = "v8f-speedsub-v1"
SUB_PREFIX = "speedsub"
SUB_MODES = ("own", "identity", "gt", "ego", "egoshuf")
EGO_MODES = ("ego", "egoshuf")
MODE_READOUT = {"ego": "mlp_ego", "egoshuf": "mlp_ego"}                 # a mode that is not a read-out's own name reads this banked read-out
EGO_READ_KEYS = ("p10f", "nav", "limit", "limit_ok", "frame_ok", "a_t")  # what `ego` reads: no feats (pixels), no label, no pose, no steering
SUB_IDENTITY_EXTRAS = ("mlp_ego", "const")
F_SUB_PASS = F_M / 2.0                                                  # the PASS point-estimate bar of PREREG_V8F_SPEEDSUB: -F/2 (0.06035 m), computed, not rounded
SHUF_SEED = 20261029
TOL_OWN_BANKED = 1e-5                                                   # `own` vs SPEEDREAD's banked own, per window (m); a different thread count may move float32 sums by ~1e-7
TOL_GT_BANKED = 1e-4                                                    # `gt` vs SPEEDREAD's banked gt, per window (m): SPEEDREAD's own C6 tolerance
TOL_GT_DELTA_4DP = 5e-4                                                 # gt - own (delta, lo, hi) vs READ.json rows.<s>.gt (printed to 4 dp)
TIER_REG = "deliberate-regression arm (the read-out's cells shuffled across windows): a control, never a result"
TIER_ARM = "T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)"
TIER_T0 = "T0 (reads the eval GT): ceiling / control only, never a result"


def _np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ T's input: the slice run_t reads, and its fixed summary
def t_tick_inputs(batch, hist: int = HIST) -> dict:
    """Mirror of `V8FMini.run_t`'s reads: last_t = F - 1 - a_t; scene frames last_t-hist+1 .. last_t; p10f / nav at last_t. Reads READ_KEYS only."""
    feats = _np(batch["feats"])
    ok_all = _np(batch["frame_ok"]).astype(bool)
    B, Fn = feats.shape[0], feats.shape[1]
    a_t = _np(batch["a_t"]).astype(np.int64).reshape(B)
    last = (Fn - 1) - a_t
    pos = last[:, None] - (hist - 1 - np.arange(hist))[None]
    if (pos < 0).any():
        raise ValueError("a tick frame has fewer than `hist` frames in the batch")
    b = np.arange(B)
    return {"tok": feats[b[:, None], pos], "ok": ok_all[b[:, None], pos], "p10f": _np(batch["p10f"])[b, last].astype(np.float32),
            "nav": _np(batch["nav"])[b, last].astype(np.float32), "limit": _np(batch["limit"]).astype(np.float32),
            "limit_ok": _np(batch["limit_ok"]).astype(np.float32).reshape(B, 1)}


def scene_raw(tok, ok, grid) -> np.ndarray:
    """[B, hist, n_tok, d] -> [B, hist*d + 10*d]: the per-frame token mean (a padded frame reads 0) and T's tick frame pooled to 2x5."""
    gh, gw = grid
    B, H, N, d = tok.shape
    if N != gh * gw:
        raise ValueError(f"n_tok {N} != grid {gh}x{gw}")
    ph, pw = POOL_TO
    if gh % ph or gw % pw:
        raise ValueError(f"grid {gh}x{gw} does not pool to {ph}x{pw}")
    x = tok.astype(np.float32) * ok[:, :, None, None].astype(np.float32)
    means = x.mean(2)
    tick = x[:, -1].reshape(B, ph, gh // ph, pw, gw // pw, d).mean(axis=(2, 4))
    return np.concatenate([means.reshape(B, H * d), tick.reshape(B, ph * pw * d)], 1)


def ego_raw(inp: dict) -> np.ndarray:
    return np.concatenate([inp["p10f"], inp["nav"], inp["limit"], inp["limit_ok"], inp["ok"].astype(np.float32)], 1).astype(np.float32)


def t_tick_ego(batch, hist: int = HIST) -> dict:
    """The EGO-ONLY slice of `t_tick_inputs` (PREREG_V8F_SPEEDSUB): the same tick positions, reading EGO_READ_KEYS and nothing else -- never `feats`, a label, a pose or a steering channel.
    A batch without those other keys is enough, which is how the `ego` mode calls it. Equal to `t_tick_inputs`'s p10f / nav / limit / limit_ok / ok key for key (pinned by a test)."""
    ok_all = _np(batch["frame_ok"]).astype(bool)
    B, Fn = ok_all.shape[0], ok_all.shape[1]
    a_t = _np(batch["a_t"]).astype(np.int64).reshape(B)
    last = (Fn - 1) - a_t
    pos = last[:, None] - (hist - 1 - np.arange(hist))[None]
    if (pos < 0).any():
        raise ValueError("a tick frame has fewer than `hist` frames in the batch")
    b = np.arange(B)
    return {"ok": ok_all[b[:, None], pos], "p10f": _np(batch["p10f"])[b, last].astype(np.float32), "nav": _np(batch["nav"])[b, last].astype(np.float32),
            "limit": _np(batch["limit"]).astype(np.float32), "limit_ok": _np(batch["limit_ok"]).astype(np.float32).reshape(B, 1)}


def ego_vectors(batch, hist: int = HIST) -> np.ndarray:
    """[B, d_ego] float32: the ego block of `window_vectors`, built from EGO_READ_KEYS only."""
    return ego_raw(t_tick_ego({k: batch[k] for k in EGO_READ_KEYS}, hist))


def window_vectors(batch, grid=(4, 10), hist: int = HIST):
    """-> (scene [B, 14336], ego [B, d_ego]) float32: the read-out's whole view of a window (d_ego = 22 on the real cache)."""
    inp = t_tick_inputs(batch, hist)
    return scene_raw(inp["tok"], inp["ok"], grid), ego_raw(inp)


def speed_targets(batch, ci) -> tuple:
    """GT speed cells at T's tick frame (physical m/s) and their mask (a v9 row AND a finite value): the trainer's own `cons_absent=mask`."""
    cons = _np(batch["lab_t"]["cons"]).astype(np.float64)[:, ci]
    has = _np(batch["lab_t"]["has"]).astype(bool)
    return cons, np.isfinite(cons) & has[:, None]


# ------------------------------------------------------------------------------------------------ the fit (TRAIN only)
def fold_of(clips) -> np.ndarray:
    return np.array([int(hashlib.sha256(str(c).encode("utf-8")).hexdigest()[:8], 16) % N_FOLDS for c in clips], np.int64)


def fit_pca(S, k: int, seed: int = PCA_SEED):
    """Randomised PCA (2 power iterations) -> (mean [D], components [k, D]). Unsupervised: reads no label."""
    S = np.asarray(S, np.float32)
    mu = S.mean(0, dtype=np.float64).astype(np.float32)
    X = S - mu
    k = int(min(k, X.shape[0], X.shape[1]))
    k2 = int(min(k + 64, X.shape[0], X.shape[1]))
    Y = X @ np.random.default_rng(seed).standard_normal((X.shape[1], k2)).astype(np.float32)
    for _ in range(2):
        Y, _ = np.linalg.qr(Y)
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return mu, Vt[:k].astype(np.float32)


def standardiser(X):
    m = X.mean(0, dtype=np.float64)
    s = X.std(0, dtype=np.float64)
    return m.astype(np.float32), np.where(s > 1e-6, s, 1.0).astype(np.float32)


def ridge_solve(X, y, lam: float):
    """Ridge with an unpenalised intercept: (w [d], b)."""
    X = np.asarray(X, np.float64)
    xm, ym = X.mean(0), float(np.mean(y))
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (np.asarray(y, np.float64) - ym))
    return w, ym - float(xm @ w)


def ridge_cv(X, Y, M, folds, cv_folds, lambdas=LAMBDAS) -> dict:
    """Per cell: the lambda with the lowest mean CV RMSE over `cv_folds` (each held out in turn, the others fitted); ties -> the larger lambda."""
    out = {"lam": [], "cv_rmse": []}
    for c in range(Y.shape[1]):
        tab = []
        for lam in lambdas:
            errs = []
            for f in cv_folds:
                tr = M[:, c] & (folds != f) & np.isin(folds, cv_folds)
                te = M[:, c] & (folds == f)
                if tr.sum() < 2 or te.sum() < 1:
                    raise ValueError(f"cell {c}: fold {f} is empty")
                w, b = ridge_solve(X[tr], Y[tr, c], lam)
                errs.append(float(np.sqrt(np.mean((X[te] @ w + b - Y[te, c]) ** 2))))
            tab.append(float(np.mean(errs)))
        best = min(range(len(lambdas)), key=lambda i: (tab[i], -lambdas[i]))
        out["lam"].append(float(lambdas[best]))
        out["cv_rmse"].append(tab)
    return out


def ridge_fit(X, Y, M, lams) -> tuple:
    W = np.zeros((X.shape[1], Y.shape[1]))
    b = np.zeros(Y.shape[1])
    for c in range(Y.shape[1]):
        W[:, c], b[c] = ridge_solve(X[M[:, c]], Y[M[:, c], c], lams[c])
    return W, b


def mlp_build(d_in: int, n_out: int, cfg=MLP_CFG):
    import torch
    from torch import nn
    torch.manual_seed(int(cfg["seed"]))
    h = int(cfg["hidden"])
    return nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(h, h), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(h, n_out))


def mlp_fit(X, Y, M, tr_idx, ho_idx, cfg=MLP_CFG) -> dict:
    """Masked MSE on targets standardised by the fit rows; early stopping on the held-out TRAIN rows. -> {state, ym, ys, best_epoch, ho_rmse_sd}."""
    import torch
    net = mlp_build(X.shape[1], Y.shape[1], cfg)
    Yt = np.where(M, Y, np.nan)[tr_idx]
    ym, ys = np.nanmean(Yt, 0), np.nanstd(Yt, 0)
    ys = np.where(ys > 1e-6, ys, 1.0)
    Z = np.nan_to_num((np.where(M, Y, 0.0) - ym) / ys).astype(np.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    gen = np.random.default_rng(int(cfg["seed"]))
    Xh, Zh, Mh = (torch.from_numpy(np.ascontiguousarray(a[ho_idx])) for a in (X.astype(np.float32), Z, M.astype(np.float32)))

    def ho_rmse():
        net.eval()
        with torch.no_grad():
            e = ((net(Xh) - Zh) ** 2 * Mh).sum(0) / Mh.sum(0).clamp_min(1.0)
        net.train()
        return float(e.sqrt().mean())

    best, best_ep, best_state, bad = float("inf"), -1, None, 0
    for ep in range(int(cfg["max_epochs"])):
        order = gen.permutation(tr_idx)
        for i in range(0, len(order), int(cfg["batch"])):
            j = order[i:i + int(cfg["batch"])]
            xb, zb, mb = torch.from_numpy(X[j].astype(np.float32)), torch.from_numpy(Z[j]), torch.from_numpy(M[j].astype(np.float32))
            loss = ((net(xb) - zb) ** 2 * mb).sum() / mb.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        r = ho_rmse()
        if r < best - 1e-6:
            best, best_ep, bad = r, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    return {"state": best_state, "ym": ym, "ys": ys, "best_epoch": best_ep, "ho_rmse_sd": best, "cfg": {"hidden": int(cfg["hidden"]), "dropout": float(cfg["dropout"]), "seed": int(cfg["seed"])}}


def mlp_predict(fit: dict, X) -> np.ndarray:
    import torch
    net = mlp_build(X.shape[1], len(fit["ym"]), fit["cfg"])
    net.load_state_dict(fit["state"])
    net.eval()
    with torch.no_grad():
        z = net(torch.from_numpy(np.asarray(X, np.float32))).double().numpy()
    return z * fit["ys"] + fit["ym"]


def masked_rmse(P, Y, M) -> np.ndarray:
    return np.array([float(np.sqrt(np.mean((P[M[:, c], c] - Y[M[:, c], c]) ** 2))) if M[:, c].any() else float("nan") for c in range(Y.shape[1])])


def skill(P, Y, M, base) -> np.ndarray:
    """1 - MSE(P) / MSE(base) per cell, `base` = the TRAIN mean per cell: the CONST read-out reads exactly 0.0."""
    B = np.broadcast_to(np.asarray(base, np.float64), Y.shape)
    return 1.0 - masked_rmse(P, Y, M) ** 2 / masked_rmse(B, Y, M) ** 2


def permute_targets(Y, M, seed: int = PERM_SEED):
    """Each cell's present targets shuffled across windows (a fixed seed): the input-target link is broken, the marginal kept."""
    Yp = Y.copy()
    rng = np.random.default_rng(seed)
    for c in range(Y.shape[1]):
        i = np.flatnonzero(M[:, c])
        Yp[i, c] = Y[rng.permutation(i), c]
    return Yp


class ReadOut:
    """The fitted read-out as ONE object: raw (scene, ego) -> physical speed cells [B, 5]. kind in {lin, mlp, const, lin_ego, mlp_ego}."""

    def __init__(self, kind, pca_mu, pca_V, xm, xs, lin=None, mlp=None, const=None, ego_only=False, grid=(4, 10)):
        self.kind, self.pca_mu, self.pca_V, self.xm, self.xs = kind, pca_mu, pca_V, xm, xs
        self.lin, self.mlp, self.const, self.ego_only, self.grid = lin, mlp, const, ego_only, tuple(grid)

    def design(self, scene, ego) -> np.ndarray:
        if self.ego_only:
            X = np.asarray(ego, np.float32)
        else:
            X = np.concatenate([(np.asarray(scene, np.float32) - self.pca_mu) @ self.pca_V.T, np.asarray(ego, np.float32)], 1)
        return (X - self.xm) / self.xs

    def predict_raw(self, scene, ego) -> np.ndarray:
        if self.kind == "const":
            return np.broadcast_to(self.const, (len(ego), len(self.const))).astype(np.float64).copy()
        X = self.design(scene, ego)
        if self.lin is not None:
            return X.astype(np.float64) @ self.lin[0] + self.lin[1]
        return mlp_predict(self.mlp, X)

    def predict_batch(self, batch) -> np.ndarray:
        scene, ego = window_vectors(batch, self.grid)
        return self.predict_raw(scene, ego)

    def predict_ego_batch(self, batch) -> np.ndarray:
        """PREREG_V8F_SPEEDSUB: an EGO-ONLY read-out's physical speed cells [B, 5] from EGO_READ_KEYS alone. The scene is never opened, so a pixel cannot reach the prediction; the
        numbers equal `predict_batch`'s for an ego-only read-out (pinned by a test, and by control S7 on the real windows)."""
        if not self.ego_only:
            raise ValueError(f"read-out {self.kind!r} reads the scene: only an ego-only read-out has an ego-only path")
        return self.predict_raw(None, ego_vectors(batch))


_RO_ARRAYS = ("pca_mu", "pca_V", "xm", "xs", "const")


def save_readouts(ro: dict, path: str):
    """Plain tensors / lists only (loads with weights_only=True, no pickled class). The aliases `read` / `alt` are stored by NAME."""
    import torch
    st = {"schema": SCHEMA, "alias": {k: v.kind for k, v in ro.items() if k in ("read", "alt")}, "ro": {}}
    for k, r in ro.items():
        if k in ("read", "alt"):
            continue
        d = {"kind": r.kind, "ego_only": bool(r.ego_only), "grid": list(r.grid)}
        for nm in _RO_ARRAYS:
            v = getattr(r, nm)
            if v is not None:
                d[nm] = torch.as_tensor(np.asarray(v))
        if r.lin is not None:
            d["lin_W"], d["lin_b"] = torch.as_tensor(r.lin[0]), torch.as_tensor(r.lin[1])
        if r.mlp is not None:
            d["mlp_state"] = r.mlp["state"]
            d["mlp_ym"], d["mlp_ys"] = torch.as_tensor(r.mlp["ym"]), torch.as_tensor(r.mlp["ys"])
            d["mlp_best_epoch"] = int(r.mlp["best_epoch"])
            d["mlp_cfg"] = dict(r.mlp["cfg"])
        st["ro"][k] = d
    torch.save(st, path)


def load_readouts(path: str) -> dict:
    import torch
    st = torch.load(path, map_location="cpu", weights_only=True)
    ro = {}
    for k, d in st["ro"].items():
        g = {nm: (d[nm].numpy() if nm in d else None) for nm in _RO_ARRAYS}
        lin = (d["lin_W"].numpy(), d["lin_b"].numpy()) if "lin_W" in d else None
        mlp = {"state": d["mlp_state"], "ym": d["mlp_ym"].numpy(), "ys": d["mlp_ys"].numpy(), "best_epoch": d["mlp_best_epoch"], "cfg": d["mlp_cfg"]} if "mlp_state" in d else None
        ro[k] = ReadOut(d["kind"], g["pca_mu"], g["pca_V"], g["xm"], g["xs"], lin=lin, mlp=mlp, const=g["const"], ego_only=d["ego_only"], grid=d["grid"])
    for a, kind in st["alias"].items():
        ro[a] = ro[kind]
    return ro


def select_primary(ho_lin_sd, ho_mlp_sd) -> str:
    """PRIMARY = the lower held-out-TRAIN RMSE in train-SD units (mean over cells); a tie goes to the simpler LIN."""
    return "mlp" if float(np.mean(ho_mlp_sd)) < float(np.mean(ho_lin_sd)) else "lin"


def fit_all(scene_sub, scene_fn, ego, Y, M, clips, pca_k=PCA_K, mlp_cfg=MLP_CFG, log=print) -> dict:
    """The whole fit from TRAIN arrays. `scene_fn(rows)` returns raw scene rows (memmap-backed); `scene_sub` is the PCA subsample. Returns the read-outs and the fit record."""
    pca_mu, pca_V = fit_pca(scene_sub, pca_k)
    n = len(ego)
    Xs = np.zeros((n, pca_V.shape[0]), np.float32)
    for i in range(0, n, 4096):
        Xs[i:i + 4096] = (scene_fn(np.arange(i, min(n, i + 4096))) - pca_mu) @ pca_V.T
    X = np.concatenate([Xs, ego.astype(np.float32)], 1)
    xm, xs = standardiser(X)
    Xz = (X - xm) / xs
    em, es = standardiser(ego.astype(np.float32))
    Ez = (ego.astype(np.float32) - em) / es
    folds = fold_of(clips)
    cvf = [f for f in range(N_FOLDS) if f != HOLDOUT_FOLD]
    tr, ho = np.flatnonzero(folds != HOLDOUT_FOLD), np.flatnonzero(folds == HOLDOUT_FOLD)
    sd = np.array([np.std(Y[M[:, c], c]) for c in range(Y.shape[1])])
    rec = {"n_windows": int(n), "n_clips": int(len(set(map(str, clips)))), "d_scene_raw": int(scene_sub.shape[1]), "d_pca": int(pca_V.shape[0]), "d_ego": int(ego.shape[1]),
           "d_design": int(X.shape[1]), "n_present_per_cell": [int(v) for v in M.sum(0)], "folds": N_FOLDS, "holdout_fold": HOLDOUT_FOLD, "lambdas": list(LAMBDAS), "mlp_cfg": dict(mlp_cfg)}
    const = np.array([float(np.mean(Y[M[:, c], c])) for c in range(Y.shape[1])])
    outs = {}
    for name, D_ in (("lin", Xz), ("lin_ego", Ez)):
        cv = ridge_cv(D_, Y, M, folds, cvf)
        Mtr = M & (folds != HOLDOUT_FOLD)[:, None]
        Wh, bh = ridge_fit(D_, Y, Mtr, cv["lam"])
        ho_sd = masked_rmse(D_[ho] @ Wh + bh, Y[ho], M[ho]) / sd
        outs[name] = {"lin": ridge_fit(D_, Y, M, cv["lam"]), "ho_sd": ho_sd}
        rec[name] = {"lam": cv["lam"], "cv_rmse": cv["cv_rmse"], "holdout_rmse_sd": [float(v) for v in ho_sd]}
        log(f"[fit] {name}: lambda {cv['lam']}, held-out RMSE (SD) {np.round(ho_sd, 4).tolist()}")
    for name, D_ in (("mlp", Xz), ("mlp_ego", Ez)):
        m = mlp_fit(D_, Y, M, tr, ho, mlp_cfg)
        ho_sd = masked_rmse(mlp_predict(m, D_[ho]), Y[ho], M[ho]) / sd
        outs[name] = {"mlp": m, "ho_sd": ho_sd}
        rec[name] = {"best_epoch": m["best_epoch"], "holdout_rmse_sd": [float(v) for v in ho_sd]}
        log(f"[fit] {name}: best epoch {m['best_epoch']}, held-out RMSE (SD) {np.round(ho_sd, 4).tolist()}")
    primary = select_primary(outs["lin"]["ho_sd"], outs["mlp"]["ho_sd"])
    alt = "mlp" if primary == "lin" else "lin"
    Yp = permute_targets(Y, M)
    if primary == "lin":
        perm = {"lin": ridge_fit(Xz, Yp, M & (folds != HOLDOUT_FOLD)[:, None], rec["lin"]["lam"])}
    else:
        perm = {"mlp": mlp_fit(Xz, Yp, M, tr, ho, mlp_cfg)}
    Pp = Xz[ho] @ perm["lin"][0] + perm["lin"][1] if primary == "lin" else mlp_predict(perm["mlp"], Xz[ho])
    rec["perm_holdout_skill_max"] = float(np.nanmax(skill(Pp, Y[ho], M[ho], const)))
    rec.update({"primary": primary, "alt": alt, "const": [float(v) for v in const], "perm_class": primary, "perm_seed": PERM_SEED, "train_sd": [float(v) for v in sd]})
    base = dict(pca_mu=pca_mu, pca_V=pca_V, xm=xm, xs=xs)
    ro = {"lin": ReadOut("lin", lin=outs["lin"]["lin"], **base), "mlp": ReadOut("mlp", mlp=outs["mlp"]["mlp"], **base),
          "lin_ego": ReadOut("lin_ego", lin=outs["lin_ego"]["lin"], ego_only=True, pca_mu=None, pca_V=None, xm=em, xs=es),
          "mlp_ego": ReadOut("mlp_ego", mlp=outs["mlp_ego"]["mlp"], ego_only=True, pca_mu=None, pca_V=None, xm=em, xs=es),
          "const": ReadOut("const", const=const, **base), "perm": ReadOut("perm", lin=perm.get("lin"), mlp=perm.get("mlp"), **base)}
    ro["read"], ro["alt"] = ro[primary], ro[alt]
    return {"readouts": ro, "record": rec}


# ------------------------------------------------------------------------------------------------ the substitution (what O reads)
def phys_to_std(v, mu, sd) -> np.ndarray:
    return (np.asarray(v, np.float64) - np.asarray(mu, np.float64)) / np.asarray(sd, np.float64)


def substitute(own_code, new_std, idx):
    """T's code with ONLY the five speed cells replaced by `new_std` (standardised, as O reads them). Every other cell is T's own, untouched."""
    import torch
    out = own_code.clone()
    out[:, list(idx)] = torch.as_tensor(np.asarray(new_std), dtype=own_code.dtype, device=own_code.device)
    return out


def new_speed_std(mode: str, readout, batch, own_code, idx, mu, sd):
    """The standardised speed cells O reads in `mode`. identity = T's own cells (control 1); every read-out mode = the read-out's physical prediction, standardised by the CHECKPOINT's mu / sd."""
    if mode == "identity":
        return _np(own_code[:, list(idx)])
    if readout is None:
        raise ValueError(f"mode {mode!r} needs a read-out")
    if mode in EGO_MODES:                                                  # SPEEDSUB: the banked ego-only read-out, EGO_READ_KEYS only
        return phys_to_std(readout.predict_ego_batch(batch), mu, sd)
    return phys_to_std(readout.predict_batch(batch), mu, sd)


def make_hook_class(SA):
    """SpeedReadHook = SPEEDAVG's hook (install / restore / cells side channel) with the read-out as the source of the five cells. `gt` mode is SPEEDAVG's own `source_code`."""

    class SpeedReadHook(SA.SpeedAvgHook):
        def __init__(self, M, mode, idx, readout=None, extras=None, shuf=None):
            super().__init__(M, mode, None, idx)
            self.readout, self.mirror_ok = readout, None
            self.shuf, self.n_seen, self.shuf_dev = shuf, 0, 0.0              # `egoshuf` (SPEEDSUB): {"rows": [N, 5] physical, "perm": [N]}; the running window index and the alignment check
            self.extras = dict(extras or {})                                   # identity mode: every read-out's prediction on the SAME batches (the cell table, C-PATH)
            self.pred_rows = {k: [] for k in self.extras}

        def __call__(self, model, batch, code_override):
            import torch
            with torch.no_grad():
                own = model.forward_infer(batch)["code"]
                has = batch["lab_t"]["has"].bool()
                mu, sd = model.cons_mu[self.ci].double().cpu().numpy(), model.cons_sd[self.ci].double().cpu().numpy()
                if self.mode == "gt":
                    new = SA.replace_cells(own, SA.source_code("gt", own, None, model._code_gt(batch["lab_t"]).to(own.dtype), has), self.idx)
                elif self.mode == "egoshuf":
                    new = substitute(own, phys_to_std(self.shuffled_pred(batch, own.shape[0]), mu, sd), self.idx)
                else:
                    new = substitute(own, new_speed_std(self.mode, self.readout, batch, own, self.idx, mu, sd), self.idx)
                if self.mirror_ok is None:                                     # first batch: my tick slice == the model's own pad mask for T
                    Fn = batch["feats"].shape[1]
                    last_t = torch.full((own.shape[0],), Fn - 1, dtype=torch.long) - batch["a_t"].long()
                    _, pad = model._scene(model._encode(batch), batch["frame_ok"], last_t)
                    ok = t_tick_inputs(batch)["ok"]
                    self.mirror_ok = bool(np.array_equal(_np(pad), np.repeat(~ok, pad.shape[1] // ok.shape[1], axis=1))) and (
                        self.mode not in EGO_MODES or bool(np.array_equal(t_tick_ego(batch)["ok"], ok)))
                self.nonspeed_dev = max(self.nonspeed_dev, float((code_override[:, self.non] - new[:, self.non]).abs().max()))
                self.oracle_dev = max(self.oracle_dev, float((code_override - new).abs().max()))
                for k, v in (("own_std", own[:, self.idx]), ("partner_std", torch.full_like(own[:, self.idx], float("nan"))), ("used_std", new[:, self.idx]),
                             ("gt_phys", batch["lab_t"]["cons"][:, self.ci])):
                    self.rows[k].append(v.float().cpu().numpy())
                self.rows["has"].append(has.cpu().numpy())
                for k, r in self.extras.items():
                    self.pred_rows[k].append(r.predict_batch(batch))
                if self.mu is None:
                    self.mu, self.sd = mu.astype(np.float32), sd.astype(np.float32)
                self.n_calls += 1
                return self.orig(model, batch, new)

        def shuffled_pred(self, batch, B):
            """`egoshuf`: window i of the eval order reads the read-out's cells of window perm[i]. The on-the-fly read-out is compared with the stored row at the running index, so a hook
            that is not looking at the windows the source was made from is seen (`shuf_dev`)."""
            rows, perm = self.shuf["rows"], self.shuf["perm"]
            i0 = self.n_seen
            if i0 + B > len(perm):
                raise ValueError(f"the shuffle source holds {len(perm)} windows, the eval has produced {i0 + B}")
            live = self.readout.predict_ego_batch({k: batch[k] for k in EGO_READ_KEYS})
            self.shuf_dev = max(self.shuf_dev, float(np.max(np.abs(live - rows[i0:i0 + B]))))
            self.n_seen += B
            return rows[perm[i0:i0 + B]]

        def stats(self):
            s = super().stats()
            s["tick_mirror_ok"] = self.mirror_ok
            if self.shuf is not None:
                perm = np.asarray(self.shuf["perm"])
                s["shuffle"] = {"seed": SHUF_SEED, "n": int(len(perm)), "n_seen": int(self.n_seen), "n_fixed": int(np.sum(perm == np.arange(len(perm)))),
                                "perm_sha256": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest(), "alignment_max_abs_dev": self.shuf_dev}
            return s

        def cells(self):
            out = super().cells()
            for k, v in self.pred_rows.items():
                out[f"pred_{k}"] = np.concatenate(v)
            return out

    return SpeedReadHook


# ------------------------------------------------------------------------------------------------ the decision (verbatim from the PREREG)
def separated_better(r) -> bool:
    return r["hi"] < 0.0


def decide(rows: dict, F: float = F_M, controls_ok: bool = True) -> str:
    """rows[seed] = {"read": {delta, lo, hi}, "alt": {...}}. The PREREG's outcome table, first match wins."""
    if not controls_ok:
        return "REFUSED"
    s = sorted(rows)
    if len(s) != 2:
        return "REFUSED"
    rd = [rows[k]["read"] for k in s]
    al = [rows[k]["alt"] for k in s]
    if all(r["delta"] <= -F and separated_better(r) for r in rd):
        return "INFO IN INPUT"
    if all(separated_better(r) for r in rd):
        return "PARTIAL"
    if not any(separated_better(r) for r in rd + al):
        return "NOT IN INPUT"
    return "MIXED"


NEXT = {"INFO IN INPUT": "Stage Z (the read-out as a forward_infer speed sub-head, zero training, T1-kin, both seeds), then the box-2 arm SPEEDHEAD under its launch gate (PREREG section 6)",
        "PARTIAL": "the box-2 arm SPEEDHEAD still runs (the only measured lever on this line); its bar is unchanged; CAPTURE is reported",
        "NOT IN INPUT": "perception stays the lever: PREREG_V8F_LORA on the A40 (D-V8F-SPEED-PERCEPTION, the PI's go); nothing on box 2 on this line",
        "MIXED": "Stage Z at T1-kin on both seeds decides: <= -F separated at both -> SPEEDHEAD; else perception (A40, PI go)",
        "REFUSED": "fix the failed control; no row is read"}


def controls(ev: dict, banked: dict | None, fitrec: dict, tol_gt: float = 1e-4) -> dict:
    """ev[seed][mode] = {eid, ade6, pred, hook}. Each control must read its KNOWN value before any decision row is read."""
    out = {}
    for s, a in ev.items():
        c = {}
        c["C1_identity_eq_own"] = float(np.max(np.abs(a["identity"]["ade6"] - a["own"]["ade6"]))) == 0.0 and bool(np.array_equal(a["identity"]["pred"], a["own"]["pred"]))
        c["C2_nonspeed_untouched"] = all(a[m]["hook"]["nonspeed_max_abs_dev"] == 0.0 for m in MODES if m != "own")
        c["C3_tick_mirror"] = all(bool(a[m]["hook"].get("tick_mirror_ok")) for m in MODES if m != "own")
        c["C4_eid_aligned"] = all(np.array_equal(a[m]["eid"], a["own"]["eid"]) for m in MODES)
        if banked and s in banked:
            c["C5_own_eq_banked_speedcal_own"] = float(np.max(np.abs(a["own"]["ade6"] - banked[s]["own"]))) == 0.0
            c["C6_gt_eq_banked_speedcal_gt"] = float(np.max(np.abs(a["gt"]["ade6"] - banked[s]["gt"]))) <= tol_gt
        cid = a["identity"].get("cells")
        c["C7_path_read_eq_identity_pred"] = c["C8_const_skill_zero"] = False             # absent = FAIL: a control that cannot run is a refusal
        if cid is not None and "cells" in a["read"]:                                               # C-PATH: what O read in `read` == the identity run's read-out prediction on the same windows
            mu, sd = np.asarray(cid["cons_mu"], np.float64), np.asarray(cid["cons_sd"], np.float64)
            want = phys_to_std(cid["pred_read"], mu, sd).astype(np.float32)
            c["C7_path_read_eq_identity_pred"] = float(np.max(np.abs(a["read"]["cells"]["used_std"] - want))) <= 1e-5
            c["C8_const_skill_zero"] = float(np.nanmax(np.abs(skill(cid["pred_const"], cid["gt_phys"].astype(np.float64), np.isfinite(cid["gt_phys"]) & cid["has"][:, None],
                                                                     fitrec["const"])))) == 0.0
        out[s] = c
    out["C9_perm_no_skill"] = fitrec.get("perm_holdout_skill_max", 1.0) <= 0.02
    out["C10_train_parity"] = bool(fitrec.get("train_parity_ok", False))
    out["C11_disjoint"] = bool(fitrec.get("eval_train_disjoint", False))
    return out


def all_pass(c: dict) -> bool:
    return all((all(v.values()) if isinstance(v, dict) else bool(v)) for v in c.values())


def read_rows(ev: dict, paired, n_boot: int = 2000) -> dict:
    rows = {}
    for s, a in ev.items():
        eid = a["own"]["eid"]
        r = {m: paired(a[m]["ade6"], a["own"]["ade6"], eid, n_boot) for m in ("read", "alt", "const", "perm", "gt")}
        r["capture"] = r["read"]["delta"] / r["gt"]["delta"] if r["gt"]["delta"] != 0 else float("nan")
        r["perm_not_separated_better"] = not separated_better(r["perm"])
        rows[s] = r
    return rows


def taniteval_paired():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.abspath(os.path.join(here, "..", "..", "taniteval"))
    if p not in sys.path:
        sys.path.insert(0, p)
    from taniteval.ci import paired_episode_cluster_bootstrap as pb

    def paired(a, b, eid, n_boot):
        o = pb(np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid), n_boot=n_boot, seed=0)
        return {"delta": float(o["delta"]), "lo": float(o["lo"]), "hi": float(o["hi"])}
    return paired


# ------------------------------------------------------------------------------------------------ CLI (the real-data paths; nothing here runs in the self-test except `--help`)
def _stack_imports():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, ".."), here):
        if os.path.abspath(p) not in sys.path:
            sys.path.insert(0, os.path.abspath(p))
    from tanitad.data import v8f_mini_data as D  # noqa: N812
    from tanitad.models import v8f_mini as M      # noqa: N812
    if not set(READ_KEYS) <= set(M.INFERENCE_INPUT_KEYS):
        raise SystemExit(f"REFUSED: the read-out reads {sorted(set(READ_KEYS) - set(M.INFERENCE_INPUT_KEYS))}, outside the model's inference inputs")
    return D, M


def train_clip_list(D, man) -> tuple:
    """The trainer's own rule (v8f_mini_train.build_data): drop sorted-by-sha12 [540:600], then select_train_clips(.., 1.0)."""
    clips_all = [c for c, _ in man]
    excluded = sorted(clips_all, key=D.sha12)[EXCLUDE[0]:EXCLUDE[1]]
    ex = set(excluded)
    return D.select_train_clips([c for c in clips_all if c not in ex], 1.0), excluded


def cmd_extract(a):
    import torch
    torch.set_num_threads(int(os.environ.get("SPEEDREAD_THREADS", "8")))
    D, M = _stack_imports()
    man = D.read_manifest(a.cache, "train")
    clips, excluded = train_clip_list(D, man)
    J = D.load_join(a.join)
    tracks = D.load_native_tracks(a.tracks)
    store = D.ClipStore(a.cache, "train", lru=4, grid=a.grid, kin_mode="yaw", tracks=tracks)
    wins = D.train_windows(store, J, clips, hist=HIST)
    parity = {"n_clips": len(clips), "n_windows": len(wins), "excluded_sha256": sha256_text("\n".join(excluded))}
    ok = parity == TRAIN_PARITY
    print(f"[extract] parity {parity} -> {'OK' if ok else 'MISMATCH'}")
    if not ok and not a.allow_parity_mismatch:
        raise SystemExit("REFUSED: the train window list does not reproduce X_s's (TRAIN_PARITY)")
    ds = D.V8FMiniWindows(store, J, wins, hist=HIST)
    ci = [M.CI[n] for n in SPEED_CELLS]
    n, grid = len(wins), (4, 10) if a.grid == "pool4x10" else (8, 20)
    os.makedirs(a.out, exist_ok=True)
    d_scene = HIST * 1024 + POOL_TO[0] * POOL_TO[1] * 1024
    S = np.lib.format.open_memmap(os.path.join(a.out, "scene_raw.npy"), mode="w+", dtype=np.float16, shape=(n, d_scene))
    ego, Y, Mk = None, np.full((n, 5), np.nan), np.zeros((n, 5), bool)      # ego is allocated at the first batch: its width follows the data (A1)
    win_arr = np.zeros((n, 4), np.int64)
    for i0 in range(0, n, a.batch):
        items = []
        for i in range(i0, min(n, i0 + a.batch)):
            c_i, t, _, _ = wins[i]
            a_t = i % 3
            items.append(ds.item(c_i, t, a_t, a_t + (i // 3) % 5))
            win_arr[i] = (c_i, t, items[-1]["a_t"], items[-1]["a_s"])
        bt = D.collate(items)
        sc, eg = window_vectors(bt, grid)
        if ego is None:
            ego = np.zeros((n, eg.shape[1]), np.float32)          # 22 on the real cache (`limit` is 4-wide); the draft hard-coded 23
        S[i0:i0 + len(items)] = sc.astype(np.float16)
        ego[i0:i0 + len(items)] = eg
        Y[i0:i0 + len(items)], Mk[i0:i0 + len(items)] = speed_targets(bt, ci)
        if (i0 // a.batch) % 100 == 0:
            print(f"[extract] {i0 + len(items)} / {n}", flush=True)
    S.flush()
    clip_of = np.array([str(J["clip"][w[0]]) for w in win_arr])
    np.savez(os.path.join(a.out, "train_arrays.npz"), ego=ego, y=Y, m=Mk, win=win_arr, clip=clip_of)
    rec = {"schema": SCHEMA, "parity": parity, "parity_ok": ok, "cache": a.cache, "grid": a.grid, "join_sha256": sha256_file(a.join), "tracks_sha256": sha256_file(a.tracks),
           "window_list_sha256": sha256_text(json.dumps(win_arr.tolist())), "speed_cells": list(SPEED_CELLS), "d_ego": int(ego.shape[1]), "ages": "cycle (a_t = i % 3, a_s = a_t + (i // 3) % 5), clamped by the dataset"}
    json.dump(rec, open(os.path.join(a.out, "extract.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


def cmd_fit(a):
    tr = np.load(os.path.join(a.train, "train_arrays.npz"))
    S = np.load(os.path.join(a.train, "scene_raw.npy"), mmap_mode="r")
    ext = json.load(open(os.path.join(a.train, "extract.json"), encoding="utf-8"))
    sub = np.asarray(S[::PCA_EVERY], np.float32)
    res = fit_all(sub, lambda rows: np.asarray(S[rows], np.float32), tr["ego"], tr["y"], tr["m"], tr["clip"])
    rec = res["record"]
    eval_clips = [json.loads(l)["clip"] for l in open(a.eval_manifest, encoding="utf-8") if l.strip()]
    rec["eval_train_disjoint"] = not (set(eval_clips) & set(map(str, tr["clip"])))
    rec["train_parity_ok"] = bool(ext["parity_ok"])
    rec["extract"] = ext
    os.makedirs(a.out, exist_ok=True)
    save_readouts(res["readouts"], os.path.join(a.out, "readout.pt"))
    rec["readout_sha256"] = sha256_file(os.path.join(a.out, "readout.pt"))
    json.dump(rec, open(os.path.join(a.out, "fit.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print(f"[fit] PRIMARY = {rec['primary']} (held-out TRAIN clips); written before any eval batch")
    return 0


def cmd_evals(a):
    import torch
    torch.set_num_threads(int(os.environ.get("SPEEDREAD_THREADS", "8")))
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import v8f_mini_eval_speedavg as SA  # noqa: E402
    if sha256_file(a.x) != a.expect_sha:
        raise SystemExit("REFUSED: checkpoint sha256 mismatch")
    if sha256_file(a.readout) != json.load(open(a.fit_json, encoding="utf-8"))["readout_sha256"]:
        raise SystemExit("REFUSED: the read-out is not the one fit.json recorded")
    exp_ro = getattr(a, "expect_readout_sha", None)
    if exp_ro and sha256_file(a.readout) != exp_ro:
        raise SystemExit("REFUSED: the read-out is not the one the PREREG pins (--expect-readout-sha)")
    sub = getattr(a, "flow", "read") == "sub"                           # PREREG_V8F_SPEEDSUB: `subevals` sets it; `evals` never does, so the SPEEDREAD flow below is as it was
    modes = SUB_MODES if sub else MODES
    prefix = SUB_PREFIX if sub else "speedread"
    ro = load_readouts(a.readout)
    Mm, E = SA._load_modules()                                           # noqa: N806
    idx = SA.speed_idx(Mm)
    hist = int(torch.load(a.x, map_location="cpu", weights_only=False)["cfg"].get("hist", HIST))
    if hist != HIST:
        raise SystemExit(f"REFUSED: X_s reads hist {hist}, the read-out was built for {HIST}")
    Hook = make_hook_class(SA)
    paths = SA.paths_of(a)
    os.makedirs(a.eval_dir, exist_ok=True)
    for mode in modes:
        tag = f"M2b-mixo_s{a.seed}-{prefix}-{mode}"
        js = os.path.join(a.eval_dir, f"eval_{tag}.json")
        if os.path.exists(js) and json.load(open(js, encoding="utf-8")).get("speedread", {}).get("done"):
            print(f"[skip] {tag}")
            continue
        argv = SA.e_argv("own" if mode == "own" else "gt", a.x, tag, paths, SA.extra_of(a) + ["--families"])
        keys = SUB_IDENTITY_EXTRAS if sub else ("read", "alt", "lin", "mlp", "lin_ego", "mlp_ego", "const", "perm")
        extras = {k: ro[k] for k in keys} if mode == "identity" else None
        shuf = shuffle_source(a.eval_dir, a.seed, prefix) if mode == "egoshuf" else None
        hook = None if mode == "own" else Hook(Mm, mode, idx, ro.get(MODE_READOUT.get(mode, mode)), extras, shuf).install()
        try:
            rc = E.main(argv)
        finally:
            if hook is not None:
                hook.restore()
        if rc not in (0, None):
            raise SystemExit(f"FAIL: eval {tag} rc {rc}")
        blk = {"schema": SCHEMA, "mode": mode, "seed": a.seed, "tier": TIER_T0 if mode == "gt" else TIER_ARM, "x_sha256": a.expect_sha, "readout_sha256": sha256_file(a.readout), "done": True,
               "hook": hook.stats() if hook else None}
        if sub:
            blk.update({"flow": SUB_SCHEMA, "readout_key": MODE_READOUT.get(mode, mode), "tier": TIER_REG if mode == "egoshuf" else blk["tier"]})
        if hook is not None:
            np.savez_compressed(os.path.join(a.eval_dir, f"speedread_cells_{tag}.npz"), **hook.cells())
        d = json.load(open(js, encoding="utf-8"))
        d["speedread"] = blk
        json.dump(d, open(js, "w", encoding="utf-8", newline="\n"), indent=1)
        print(f"[ok] {tag}")
    return 0


def cell_table(cid: dict) -> dict:
    """Descriptive (decides nothing): eval RMSE (m/s) and skill vs the TRAIN mean, per cell, for T's own cells and every read-out, from the identity run's dump."""
    mu, sd = np.asarray(cid["cons_mu"], np.float64), np.asarray(cid["cons_sd"], np.float64)
    Y = cid["gt_phys"].astype(np.float64)
    Mk = np.isfinite(Y) & cid["has"].astype(bool)[:, None]
    base = cid["pred_const"][0]
    preds = {"T_own": cid["own_std"].astype(np.float64) * sd + mu}
    preds.update({k[5:]: cid[k] for k in cid if k.startswith("pred_")})
    return {k: {"rmse": [float(v) for v in masked_rmse(P, Y, Mk)], "skill": [float(v) for v in skill(P, Y, Mk, base)]} for k, P in preds.items()} | {
        "n_present": [int(v) for v in Mk.sum(0)], "cells": list(SPEED_CELLS)}


def load_eval(eval_dir: str, seed: int, modes=MODES, prefix: str = "speedread") -> dict:
    out = {}
    for m in modes:
        tag = f"M2b-mixo_s{seed}-{prefix}-{m}"
        z = np.load(os.path.join(eval_dir, f"eval_{tag}.npz"))
        js = json.load(open(os.path.join(eval_dir, f"eval_{tag}.json"), encoding="utf-8"))
        d = {"eid": np.asarray(z["eid"]), "ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64), "hook": (js.get("speedread") or {}).get("hook") or {}}
        if prefix != "speedread":                                       # the SPEEDSUB flow also reads the read-out sha; the SPEEDREAD flow's dict is unchanged
            d["readout_sha256"] = (js.get("speedread") or {}).get("readout_sha256")
        cp = os.path.join(eval_dir, f"speedread_cells_{tag}.npz")
        if os.path.exists(cp):
            zc = np.load(cp)
            d["cells"] = {k: zc[k] for k in zc.files}
        out[m] = d
    return out


def cmd_read(a):
    fitrec = json.load(open(a.fit_json, encoding="utf-8"))
    ev = {0: load_eval(a.eval_dir_s0, 0), 1: load_eval(a.eval_dir_s1, 1)}
    banked = {}
    for s, d in ((0, a.speedcal_dir_s0), (1, a.speedcal_dir_s1)):
        if d:
            banked[s] = {m: np.asarray(np.load(os.path.join(d, f"eval_M2b-mixo_s{s}-speedcal-{m}.npz"))["ade6"], np.float64) for m in ("own", "gt")}
    c = controls(ev, banked, fitrec)
    ok = all_pass(c)
    rows = read_rows(ev, taniteval_paired(), a.n_boot) if ok else {}
    outcome = decide({s: rows[s] for s in rows}, F_M, ok)
    rep = {"schema": SCHEMA, "outcome": outcome, "next": NEXT[outcome], "F_m": F_M, "controls": c, "rows": rows, "tier": TIER_ARM, "tier_gt": TIER_T0,
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
           "primary_readout": fitrec.get("primary"), "cells": {s: cell_table(ev[s]["identity"]["cells"]) for s in ev if "cells" in ev[s]["identity"]}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "READ.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    open(os.path.join(a.out, "OUTCOME.txt"), "w", encoding="utf-8", newline="\n").write(f"{outcome}\n{NEXT[outcome]}\n")
    print(f"ZZSPEEDREAD-DONE {outcome}")
    return 0 if outcome != "REFUSED" else 2


# ------------------------------------------------------------------------------------------------ PREREG_V8F_SPEEDSUB: the shuffle source, the controls, the rows and the decision
def shuffle_perm(n: int) -> np.ndarray:
    """The registered permutation of the eval windows for `egoshuf`: numpy default_rng(SHUF_SEED).permutation(n)."""
    return np.random.default_rng(SHUF_SEED).permutation(int(n))


def shuffle_source(eval_dir: str, seed: int, prefix: str = SUB_PREFIX) -> dict:
    """The cells `egoshuf` permutes: the identity run's `pred_mlp_ego` (physical m/s, eval window order). The identity run therefore comes first in SUB_MODES."""
    p = os.path.join(eval_dir, f"speedread_cells_M2b-mixo_s{seed}-{prefix}-identity.npz")
    if not os.path.exists(p):
        raise SystemExit(f"REFUSED: egoshuf needs the identity run's cells ({p}); run identity first")
    rows = np.load(p)["pred_mlp_ego"].astype(np.float64)
    return {"rows": rows, "perm": shuffle_perm(len(rows))}


def separated_worse(r) -> bool:
    return r["lo"] > 0.0


def controls_sub(ev: dict, banked: dict | None, fitrec: dict, expect_sha: str | None = None, tol_own: float = TOL_OWN_BANKED, tol_gt: float = TOL_GT_BANKED) -> dict:
    """ev[seed][mode] = {eid, ade6, pred, hook, cells?, readout_sha256}, mode in SUB_MODES. banked[seed] = SPEEDREAD's own / gt ADE6 and its identity run's pred_mlp_ego.
    Each control must read its KNOWN value before any row is read; a control that cannot run reads False (a refusal)."""
    out = {}
    for s, a in ev.items():
        c = {}
        n = len(a["own"]["eid"])
        b = (banked or {}).get(s)
        c["S1_identity_eq_own"] = float(np.max(np.abs(a["identity"]["ade6"] - a["own"]["ade6"]))) == 0.0 and bool(np.array_equal(a["identity"]["pred"], a["own"]["pred"]))
        c["S2_nonspeed_untouched"] = all(a[m]["hook"].get("nonspeed_max_abs_dev") == 0.0 for m in SUB_MODES if m != "own")
        c["S3_tick_mirror"] = all(bool(a[m]["hook"].get("tick_mirror_ok")) for m in SUB_MODES if m != "own")
        c["S4_eid_aligned"] = all(np.array_equal(a[m]["eid"], a["own"]["eid"]) for m in SUB_MODES)
        c["S5_own_eq_speedread_own"] = b is not None and float(np.max(np.abs(a["own"]["ade6"] - b["own"]))) <= tol_own
        c["S6_gt_eq_speedread_gt"] = b is not None and float(np.max(np.abs(a["gt"]["ade6"] - b["gt"]))) <= tol_gt
        cid, ce, cs = a["identity"].get("cells"), a["ego"].get("cells"), a["egoshuf"].get("cells")
        c["S7_ego_path_eq_identity_pred"] = c["S7b_identity_pred_eq_speedread_banked"] = c["S10_substitution_reaches_O"] = False
        if cid is not None and ce is not None and "pred_mlp_ego" in cid:
            mu, sd = np.asarray(cid["cons_mu"], np.float64), np.asarray(cid["cons_sd"], np.float64)
            want = phys_to_std(cid["pred_mlp_ego"], mu, sd).astype(np.float32)
            c["S7_ego_path_eq_identity_pred"] = float(np.max(np.abs(ce["used_std"] - want))) <= 1e-5
            if b is not None and "pred_mlp_ego" in b:
                c["S7b_identity_pred_eq_speedread_banked"] = cid["pred_mlp_ego"].shape == b["pred_mlp_ego"].shape and float(np.max(np.abs(cid["pred_mlp_ego"] - b["pred_mlp_ego"]))) <= 1e-5
            moved = np.any(ce["used_std"] != ce["own_std"], axis=1)
            c["S10_substitution_reaches_O"] = bool(moved.mean() >= 0.99) and (cs is not None and bool(np.mean(np.any(ce["used_std"] != cs["used_std"], axis=1)) >= 0.95))
        c["S8_readout_sha"] = expect_sha is not None and expect_sha == fitrec.get("readout_sha256") and all(a[m].get("readout_sha256") == expect_sha for m in SUB_MODES if m != "own")
        sh = a["egoshuf"]["hook"].get("shuffle") or {}
        c["S9_shuffle_registered_and_aligned"] = bool(sh) and sh["alignment_max_abs_dev"] <= 1e-6 and sh["n_seen"] == n and sh["n"] == n and sh["n_fixed"] <= max(3, 0.01 * n) and (
            sh["perm_sha256"] == hashlib.sha256(shuffle_perm(n).astype(np.int64).tobytes()).hexdigest())
        out[s] = c
    sel = {k: float(np.mean(fitrec[k]["holdout_rmse_sd"])) for k in ("lin", "mlp", "lin_ego", "mlp_ego") if k in fitrec}
    out["S11_train_parity"] = bool(fitrec.get("train_parity_ok", False))
    out["S12_disjoint"] = bool(fitrec.get("eval_train_disjoint", False))
    out["S13_selection_rule_picks_mlp_ego"] = len(sel) == 4 and min(sel, key=sel.get) == "mlp_ego"
    return out


def read_rows_sub(ev: dict, paired, n_boot: int = 2000) -> dict:
    rows = {}
    for s, a in ev.items():
        eid = a["own"]["eid"]
        r = {m: paired(a[m]["ade6"], a["own"]["ade6"], eid, n_boot) for m in ("ego", "egoshuf", "gt")}
        r["capture"] = r["ego"]["delta"] / r["gt"]["delta"] if r["gt"]["delta"] != 0 else float("nan")
        r["regression_arm_separated_worse"] = separated_worse(r["egoshuf"])
        rows[s] = r
    return rows


def decide_sub(rows: dict, F_pass: float = F_SUB_PASS, controls_ok: bool = True) -> str:
    """PREREG_V8F_SPEEDSUB's table; rows[seed] = {"ego": {delta, lo, hi}, "egoshuf": {...}}. The three outcomes are mutually exclusive; REFUSED comes first.
    REFUSED: a control failed, a seed is missing, or the deliberate-regression arm is not separated-worse at both seeds (the instrument would not see a harm).
    HARMS: ego separated-worse at either seed. PASS: ego separated-better AND delta <= -F_pass at both seeds. NULL: anything else."""
    if not controls_ok:
        return "REFUSED"
    s = sorted(rows)
    if s != [0, 1]:
        return "REFUSED"
    if not all(separated_worse(rows[k]["egoshuf"]) for k in s):
        return "REFUSED"
    ego = [rows[k]["ego"] for k in s]
    if any(separated_worse(r) for r in ego):
        return "HARMS"
    if all(separated_better(r) and r["delta"] <= -F_pass for r in ego):
        return "PASS"
    return "NULL"


def null_detail(rows: dict) -> str:
    """Descriptive only (the outcome is NULL either way)."""
    nb = sum(separated_better(rows[k]["ego"]) for k in sorted(rows))
    return {2: "NULL-SMALL (separated-better at both seeds, a point estimate above -F/2 at one or both)", 1: "NULL-ONE (separated-better at one seed only)"}.get(nb, "NULL-FLAT (separated-better at neither seed)")


NEXT_SUB = {"PASS": "T1-kin zero-training substitution of the banked ego read-out on the DAG-OACC / DAG-ORC base, both seeds, under its own PREREG (PREREG_V8F_SPEEDSUB section 6)",
            "HARMS": "PREREG_V8F_SPEEDCELL: zero-training per-cell attribution (GT in one cell at a time, ego in one cell at a time, five cells, both seeds) on the same instrument; nothing is fitted (section 7)",
            "NULL": "PREREG_V8F_SPEEDCELL: zero-training per-cell attribution (GT in one cell at a time, ego in one cell at a time, five cells, both seeds) on the same instrument; nothing is fitted (section 7)",
            "REFUSED": "fix the failed control (or the regression arm that did not read separated-worse); no row is read"}


def cmd_subevals(a):
    a.flow = "sub"
    return cmd_evals(a)


def cmd_subread(a):
    fitrec = json.load(open(a.fit_json, encoding="utf-8"))
    ev = {0: load_eval(a.eval_dir_s0, 0, SUB_MODES, SUB_PREFIX), 1: load_eval(a.eval_dir_s1, 1, SUB_MODES, SUB_PREFIX)}
    rj = json.load(open(a.speedread_read_json, encoding="utf-8"))
    banked = {}
    for s, d in ((0, a.speedread_dir_s0), (1, a.speedread_dir_s1)):
        b = {m: np.asarray(np.load(os.path.join(d, f"eval_M2b-mixo_s{s}-speedread-{m}.npz"))["ade6"], np.float64) for m in ("own", "gt")}
        b["pred_mlp_ego"] = np.load(os.path.join(d, f"speedread_cells_M2b-mixo_s{s}-speedread-identity.npz"))["pred_mlp_ego"].astype(np.float64)
        banked[s] = b
    c = controls_sub(ev, banked, fitrec, a.expect_readout_sha)
    rows = read_rows_sub(ev, taniteval_paired(), a.n_boot) if all_pass(c) else {}
    if rows:                                                           # S6b: the paired gt - own row reproduces SPEEDREAD's READ.json (delta, lo, hi) to 4 dp
        c["S6b_gt_delta_eq_speedread_read_json"] = {s: max(abs(rows[s]["gt"][k] - rj["rows"][str(s)]["gt"][k]) for k in ("delta", "lo", "hi")) <= TOL_GT_DELTA_4DP for s in rows}
    ok = all_pass(c)
    outcome = decide_sub(rows, F_SUB_PASS, ok)
    rep = {"schema": SUB_SCHEMA, "claim": "H-V8F-SPEEDSUB-1", "outcome": outcome, "detail": null_detail(rows) if outcome == "NULL" else None, "next": NEXT_SUB[outcome],
           "F_m": F_M, "F_pass_m": F_SUB_PASS, "controls": c, "rows": rows, "tier": TIER_ARM, "tier_gt": TIER_T0, "tier_regression": TIER_REG,
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
           "readout_sha256": a.expect_readout_sha, "selected_readout": "mlp_ego",
           "cells": {s: cell_table(ev[s]["identity"]["cells"]) for s in ev if "cells" in ev[s]["identity"]}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "SUBREAD.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    open(os.path.join(a.out, "OUTCOME.txt"), "w", encoding="utf-8", newline="\n").write(f"{outcome}\n{NEXT_SUB[outcome]}\n")
    print(f"ZZSPEEDSUB-DONE {outcome}")
    return 0 if outcome != "REFUSED" else 2


def _evals_args(p):
    """`evals`' arguments for `subevals` (the SPEEDREAD parser is left as it was) plus the pinned read-out sha."""
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--expect-sha", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--fit-json", required=True)
    p.add_argument("--expect-readout-sha", required=True, help="the read-out file's sha256 the PREREG pins")
    for k in ("--eval-dir", "--eval-cache", "--tracks", "--v9-eval", "--lead-eval", "--speed-eval", "--p10"):
        p.add_argument(k, required=True)
    for k in ("--eval-grid", "--eval-kin", "--eval-join"):
        p.add_argument(k, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--device", default="cpu")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--cache", required=True)
    p.add_argument("--join", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--grid", choices=("pool4x10", "pool8x20"), default="pool4x10")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--allow-parity-mismatch", action="store_true", help="SMOKE ONLY: the json records it and `fit` refuses to be read")
    p = sub.add_parser("fit")
    p.add_argument("--train", required=True)
    p.add_argument("--eval-manifest", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("evals")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--expect-sha", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--fit-json", required=True)
    for k in ("--eval-dir", "--eval-cache", "--tracks", "--v9-eval", "--lead-eval", "--speed-eval", "--p10"):
        p.add_argument(k, required=True)
    for k in ("--eval-grid", "--eval-kin", "--eval-join"):
        p.add_argument(k, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("read")
    p.add_argument("--fit-json", required=True)
    p.add_argument("--eval-dir-s0", required=True)
    p.add_argument("--eval-dir-s1", required=True)
    p.add_argument("--speedcal-dir-s0", required=True, help="SPEEDCAL's banked run/eval (own / gt npz): controls C5 / C6")
    p.add_argument("--speedcal-dir-s1", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("subevals")
    _evals_args(p)
    p = sub.add_parser("subread")
    p.add_argument("--fit-json", required=True)
    p.add_argument("--eval-dir-s0", required=True)
    p.add_argument("--eval-dir-s1", required=True)
    p.add_argument("--speedread-dir-s0", required=True, help="SPEEDREAD's banked eval dir (own / gt npz, identity cells): controls S5 / S6 / S7b")
    p.add_argument("--speedread-dir-s1", required=True)
    p.add_argument("--speedread-read-json", required=True, help="SPEEDREAD's READ.json: control S6b")
    p.add_argument("--expect-readout-sha", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", required=True)
    return ap


def main(argv=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "-1":
        raise SystemExit("REFUSED: CPU only -- set CUDA_VISIBLE_DEVICES=-1")
    a = build_parser().parse_args(argv)
    return {"extract": cmd_extract, "fit": cmd_fit, "evals": cmd_evals, "read": cmd_read, "subevals": cmd_subevals, "subread": cmd_subread}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
