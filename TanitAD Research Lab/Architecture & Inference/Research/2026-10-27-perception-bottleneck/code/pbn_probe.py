"""pbn_probe.py -- PERC-BOTTLENECK Q1 (PREREG_PERC_BOTTLENECK.md, sealed 5489cf56...): probes on the FROZEN refcv8 map-head
inputs vs the head itself, on the SAME eval cells. CPU only (torch, <= ~3 GB host RAM).

FIT  = cap_train (139 TRAIN clips), split clip-disjoint 80 % fit / 20 % VAL (hyper-parameters chosen on VAL ONLY).
SCORE = cap_eval (eval139 clips with a GT file). The score split is read once, after every choice is frozen.

    python -X utf8 pbn_probe.py --cap-train F:/perc_bottleneck/cap_train --cap-eval F:/perc_bottleneck/cap_eval \
        --out F:/perc_bottleneck/q1 --seal <PREREG sha256 file> --prereg <PREREG md>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EV_R = np.arange(0, 160, 2)                   # patch rows of the eval cells
EV_C = np.arange(108, 132, 2) - 104           # patch cols of the eval cells (|y| < 3 m)
X_M = (EV_R + 0.5) * 0.25                     # cell-centre x (m)
DRV = (1, 2, 3, 4, 6)
SW = 7
BINS = ((0, 5), (5, 10), (10, 20), (20, 40))
BANDS = {"A_0_20": (0, 20), "A0_0_5": (0, 5), "B_5_40": (5, 40)}
LOGW = None                                   # the head's frozen log class weights (read from capture.json)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_split(cap: Path, want_img: bool, img_keep: float, seed: int):
    """-> dict of per-cell arrays (eval cells only) + per-window patches for the conv probe."""
    meta = json.loads((cap / "capture.json").read_text(encoding="utf-8"))
    files = sorted(cap.glob("w_*.npz"))
    if len(files) != meta["n_written"]:
        raise SystemExit(f"[probe] {cap}: {len(files)} files vs capture.json n_written {meta['n_written']}")
    rng = np.random.default_rng(seed)
    D = {k: [] for k in ("enc", "lift", "img", "img_idx", "logit", "dec", "gt", "lv", "row", "col", "win", "clip")}
    P = {"enc": [], "gt": []}
    clips = []
    for wi, f in enumerate(files):
        z = np.load(f)
        s12 = f.name.split("_")[2]
        if s12 not in clips:
            clips.append(s12)
        ci = clips.index(s12)
        enc, lift, img = z["enc"], z["lift"], z["img"]
        if float(np.abs(enc[:, :8, :8].astype(np.float32)).mean()) == 0.0:
            raise SystemExit(f"[probe] all-zero enc in {f.name}: refusing a poisoned bank")
        P["enc"].append(enc)                                                  # [64, 160, 32] fp16
        gtp = z["gt"]
        P["gt"].append(gtp)
        e = enc[:, EV_R][:, :, EV_C].reshape(64, -1).T
        l_ = lift[:, EV_R][:, :, EV_C].reshape(64, -1).T
        lg = z["logit"][:, EV_R][:, :, EV_C].reshape(8, -1).T
        g = gtp[EV_R][:, EV_C].reshape(-1)
        n = g.size
        rr, cc = np.meshgrid(np.arange(len(EV_R)), np.arange(len(EV_C)), indexing="ij")
        D["enc"].append(e); D["lift"].append(l_); D["logit"].append(lg)
        D["dec"].append(z["dec"][EV_R][:, EV_C].reshape(-1)); D["gt"].append(g)
        D["lv"].append(z["lv"][EV_R][:, EV_C].reshape(-1)); D["row"].append(rr.reshape(-1)); D["col"].append(cc.reshape(-1))
        D["win"].append(np.full(n, wi, np.int32)); D["clip"].append(np.full(n, ci, np.int32))
        if want_img:
            im = img.reshape(img.shape[0], -1).T                                  # [960, 2048] fp16
            keep = np.nonzero((g != 255) & (rng.random(n) < img_keep))[0]
            D["img"].append(im[keep]); D["img_idx"].append(keep + wi * n)
    out = {k: np.concatenate(v) for k, v in D.items() if v}
    out["patch_enc"] = np.stack(P["enc"]); out["patch_gt"] = np.stack(P["gt"])
    out["clips"] = clips
    out["meta"] = meta
    return out


# ------------------------------------------------------------------------------------------------- metrics
def ap_score(score, pos):
    """Tie-aware all-point AP (precision/recall evaluated only at DISTINCT score thresholds; a constant score gives exactly
    the positive rate -- control C1)."""
    n = int(pos.sum())
    if n == 0:
        return float("nan")
    o = np.argsort(-score, kind="stable")
    s, p = score[o], pos[o].astype(np.float64)
    tp, fp = np.cumsum(p), np.cumsum(1 - p)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    rec, prec = tp[last] / n, tp[last] / (tp[last] + fp[last])
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def prec_at_recall(score, pos, r0):
    n = int(pos.sum())
    if n == 0 or not np.isfinite(r0):
        return float("nan")
    o = np.argsort(-score, kind="stable")
    s, p = score[o], pos[o].astype(np.float64)
    tp, fp = np.cumsum(p), np.cumsum(1 - p)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    rec, prec = tp[last] / n, tp[last] / (tp[last] + fp[last])
    ok = rec >= r0 - 1e-12
    return float(prec[ok].max()) if ok.any() else float("nan")


def auroc(score, pos):
    pos = pos.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(score)); o = np.argsort(score, kind="stable")
    s = score[o]; ranks = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks over ties
    b = np.r_[0, np.nonzero(np.diff(s))[0] + 1, len(s)]
    for i in range(len(b) - 1):
        ranks[b[i]:b[i + 1]] = ranks[b[i]:b[i + 1]].mean()
    r[o] = ranks
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------- probes
class Lin(nn.Module):
    def __init__(self, d):
        super().__init__(); self.f = nn.Linear(d, 8)

    def forward(self, x):
        return self.f(x)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 8))

    def forward(self, x):
        return self.f(x)


class Conv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(d, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                               nn.Conv2d(64, 8, 1))

    def forward(self, x):
        return self.f(x)


def standardiser(X):
    mu = X.astype(np.float32).mean(0)
    sd = X.astype(np.float32).std(0) + 1e-4
    return mu, sd


def fit_cell_probe(kind, Xf, yf, Xv, yv, vmask, wd, seed, epochs=8, bs=4096, lr=2e-3):
    """Train on (Xf, yf) [fp16 features, 8-class labels, 255 ignored]; per-epoch VAL score = sidewalk AP on vmask cells;
    returns (best model state, best val AP, best epoch, per-epoch curve)."""
    torch.manual_seed(seed)
    d = Xf.shape[1]
    m = (Lin if kind == "linear" else MLP)(d)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    idx = np.nonzero(yf != 255)[0]
    yt = torch.from_numpy(yf.astype(np.int64))
    best = (-1.0, None, -1)
    curve = []
    g = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        g.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = torch.from_numpy(Xf[b].astype(np.float32))
            loss = F.cross_entropy(m(x), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sv = predict_cells(m, Xv)
        a = ap_score(sv[vmask, SW], yv[vmask] == SW)
        curve.append(a)
        if a > best[0]:
            best = (a, {k: v.clone() for k, v in m.state_dict().items()}, ep)
    m.load_state_dict(best[1])
    return m, best[0], best[2], curve


@torch.no_grad()
def predict_cells(m, X, bs=16384):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(m(torch.from_numpy(X[i:i + bs].astype(np.float32))), -1).numpy())
    return np.concatenate(out)


def fit_conv_probe(Pf, Gf, Pv, Gv, vmask_fn, wd, seed, epochs=8, bs=8, lr=2e-3):
    torch.manual_seed(seed)
    m = Conv(Pf.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = (-1.0, None, -1)
    curve = []
    g = np.random.default_rng(seed)
    order = np.arange(len(Pf))
    for ep in range(epochs):
        m.train(); g.shuffle(order)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = torch.from_numpy(Pf[b].astype(np.float32)); y = torch.from_numpy(Gf[b].astype(np.int64))
            loss = F.cross_entropy(m(x), y, ignore_index=255)
            opt.zero_grad(); loss.backward(); opt.step()
        sv = predict_conv(m, Pv)
        vm, yv = vmask_fn()
        a = ap_score(sv[vm, SW], yv[vm] == SW)
        curve.append(a)
        if a > best[0]:
            best = (a, {k: v.clone() for k, v in m.state_dict().items()}, ep)
    m.load_state_dict(best[1])
    return m, best[0], best[2], curve


@torch.no_grad()
def predict_conv(m, P, bs=16):
    """-> eval-cell posteriors [n_win * 960, 8] in the cell order of load_split."""
    m.eval()
    out = []
    for i in range(0, len(P), bs):
        p = torch.softmax(m(torch.from_numpy(P[i:i + bs].astype(np.float32))), 1)[:, :, EV_R][:, :, :, EV_C]
        out.append(p.permute(0, 2, 3, 1).reshape(-1, 8).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-train", required=True)
    ap.add_argument("--cap-eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seal", required=True)
    ap.add_argument("--img-keep", type=float, default=0.25, help="share of FIT GT-seen cells kept for the 2,048-d img stage (RAM)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if open(a.seal, encoding="utf-8").read().split()[0] != sha256(a.prereg):
        raise SystemExit("[probe] PREREG does not match its seal: refusing to score")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    TR = load_split(Path(a.cap_train), True, a.img_keep, 0)
    global LOGW
    cw = np.array(TR["meta"]["class_weight"], np.float64)
    LOGW = np.log(cw)
    # ---- clip-disjoint FIT / VAL split of TRAIN (deterministic: every 5th clip by sha12 order -> VAL) ------------ #
    cl_sorted = sorted(TR["clips"])
    val_clips = {TR["clips"].index(c) for i, c in enumerate(cl_sorted) if i % 5 == 2}
    is_val = np.isin(TR["clip"], list(val_clips))
    xm_tr = X_M[TR["row"]]
    vmask_val = is_val & (TR["gt"] != 255) & TR["lv"] & (xm_tr < 20)          # VAL selection cells: band A, lift-valid
    fit = ~is_val
    rec = {"tool": "pbn_probe.py", "prereg_sha256": sha256(a.prereg), "n_train_clips": len(TR["clips"]),
           "n_val_clips": len(val_clips), "n_train_windows": int(len(TR["patch_enc"])), "fit_cells": int(fit.sum()),
           "fit_cells_seen": int((fit & (TR["gt"] != 255)).sum()), "val_sel_cells": int(vmask_val.sum())}
    # standardise on FIT
    stages = {}
    for k in ("enc", "lift"):
        mu, sd = standardiser(TR[k][fit])
        stages[k] = (mu, sd)
    img_rows_fit = np.isin(TR["img_idx"], np.nonzero(fit)[0])
    _ri = np.nonzero(img_rows_fit)[0]
    _acc = np.zeros(TR["img"].shape[1]); _acc2 = np.zeros(TR["img"].shape[1])
    for i in range(0, len(_ri), 20000):
        _x = TR["img"][_ri[i:i + 20000]].astype(np.float64); _acc += _x.sum(0); _acc2 += (_x ** 2).sum(0)
    mu_i = (_acc / len(_ri)).astype(np.float32)
    sd_i = (np.sqrt(np.maximum(_acc2 / len(_ri) - mu_i.astype(np.float64) ** 2, 0)) + 1e-4).astype(np.float32)
    stages["img"] = (mu_i, sd_i)

    def feat(Dd, k, rows=None):
        if k == "img":
            X = Dd["img"] if rows is None else Dd["img"][rows]
            outx = np.empty(X.shape, np.float16)
            for i in range(0, len(X), 20000):
                outx[i:i + 20000] = ((X[i:i + 20000].astype(np.float32) - mu_i) / sd_i).astype(np.float16)
            return outx
        mu, sd = stages[k]
        X = Dd[k] if rows is None else Dd[k][rows]
        return ((X.astype(np.float32) - mu) / sd).astype(np.float16)

    # img-stage cell bookkeeping: img rows map to cell indices TR["img_idx"]
    img_cell = TR["img_idx"]
    fit_img = img_rows_fit
    val_img = np.isin(img_cell, np.nonzero(is_val)[0])
    # ---- the probe zoo, fit on FIT, selected on VAL ----------------------------------------------------------- #
    WDS = (1e-4, 1e-3, 1e-2)
    zoo = {}
    plan = [("enc", "linear"), ("enc", "mlp"), ("lift", "linear"), ("lift", "mlp"), ("img", "linear"), ("img", "mlp"),
            ("enc+img", "linear"), ("enc+img", "mlp")]
    for st, kind in plan:
        tt = time.time()
        if st in ("enc", "lift"):
            Xf, yf = feat(TR, st, np.nonzero(fit)[0]), TR["gt"][fit]
            Xv, yv = feat(TR, st, np.nonzero(is_val)[0]), TR["gt"][is_val]
            vm = vmask_val[is_val]
        else:
            rf, rv = np.nonzero(fit_img)[0], np.nonzero(val_img)[0]
            ci_f, ci_v = img_cell[rf], img_cell[rv]
            parts_f = [feat(TR, "img", rf)]
            parts_v = [feat(TR, "img", rv)]
            if st == "enc+img":
                parts_f.insert(0, feat(TR, "enc", ci_f)); parts_v.insert(0, feat(TR, "enc", ci_v))
            Xf, Xv = np.concatenate(parts_f, 1), np.concatenate(parts_v, 1)
            yf, yv = TR["gt"][ci_f], TR["gt"][ci_v]
            vm = vmask_val[ci_v]
        res = []
        for wd in WDS:
            m, va, ep, curve = fit_cell_probe(kind, Xf, yf, Xv, yv, vm, wd, 0, a.epochs)
            res.append((va, wd, ep, curve, m))
        res.sort(key=lambda r: -r[0])
        va, wd, ep, curve, m = res[0]
        m1, va1, ep1, _ = fit_cell_probe(kind, Xf, yf, Xv, yv, vm, wd, 1, a.epochs)          # probe seed 1
        zoo[(st, kind)] = {"models": [m, m1], "wd": wd, "val_ap": [va, va1], "best_epoch": [ep, ep1], "d": int(Xf.shape[1]),
                           "n_fit": int((yf != 255).sum()), "grid": [(r[1], r[0]) for r in res], "curve_seed0": curve,
                           "fit_s": time.time() - tt}
        print(f"[probe] {st:8s} {kind:6s} d={Xf.shape[1]:5d} n_fit={int((yf != 255).sum()):7d} wd={wd:g} "
              f"val AP_sw(A,lv)={va:.4f}/{va1:.4f} ep={ep}/{ep1} {time.time() - tt:.0f}s", flush=True)
        if (st, kind) == ("enc", "mlp"):   # data-limited check: the same probe on HALF the FIT windows
            hw = np.unique(TR["win"][fit]); hw = hw[: len(hw) // 2]
            half = np.isin(TR["win"][fit], hw)
            mh, vah, _, _ = fit_cell_probe(kind, Xf[half], yf[half], Xv, yv, vm, wd, 0, a.epochs)
            zoo[(st, kind)]["half_fit_val_ap"] = vah
        del Xf, Xv
    # conv probe on the enc patch
    tt = time.time()
    win_tr = np.arange(len(TR["patch_enc"]))
    win_val = np.unique(TR["win"][is_val])
    win_fit = np.setdiff1d(win_tr, win_val)
    mu_e, sd_e = stages["enc"]

    def patch(P):
        return ((P.astype(np.float32) - mu_e[None, :, None, None]) / sd_e[None, :, None, None]).astype(np.float16)
    Pf, Gf = patch(TR["patch_enc"][win_fit]), TR["patch_gt"][win_fit]
    Pv = patch(TR["patch_enc"][win_val])
    cells_val = np.concatenate([np.nonzero(TR["win"] == w)[0] for w in win_val])

    def vfn():
        return vmask_val[cells_val], TR["gt"][cells_val]
    res = []
    for wd in WDS:
        m, va, ep, curve = fit_conv_probe(Pf, Gf, Pv, None, vfn, wd, 0, a.epochs)
        res.append((va, wd, ep, curve, m))
    res.sort(key=lambda r: -r[0])
    va, wd, ep, curve, m = res[0]
    m1, va1, ep1, _ = fit_conv_probe(Pf, Gf, Pv, None, vfn, wd, 1, a.epochs)
    zoo[("enc", "conv3x3")] = {"models": [m, m1], "wd": wd, "val_ap": [va, va1], "best_epoch": [ep, ep1], "d": 64,
                               "n_fit": int((Gf != 255).sum()), "grid": [(r[1], r[0]) for r in res], "curve_seed0": curve,
                               "fit_s": time.time() - tt}
    print(f"[probe] enc      conv   wd={wd:g} val AP_sw(A,lv)={va:.4f}/{va1:.4f} {time.time() - tt:.0f}s", flush=True)
    # controls fit on FIT: C1 constant, C2 positional prior
    yfit = TR["gt"][fit & (TR["gt"] != 255)]
    c1 = np.bincount(yfit, minlength=8)[:8] / len(yfit)
    c2 = np.zeros((len(EV_R), len(EV_C), 8))
    ff = fit & (TR["gt"] != 255)
    np.add.at(c2, (TR["row"][ff], TR["col"][ff], TR["gt"][ff]), 1)
    c2 = (c2 + 1e-3) / (c2 + 1e-3).sum(-1, keepdims=True)
    # val-AP of the head on VAL cells (descriptive: the head was TRAINED on these clips)
    del TR, Pf, Gf, Pv
    # ---- SCORE (eval139) -- read once, every choice frozen ------------------------------------------------------ #
    EV = load_split(Path(a.cap_eval), False, 1.0, 0)          # img is STREAMED per window below (host RAM)
    gt, lv, xm, clip = EV["gt"], EV["lv"], X_M[EV["row"]], EV["clip"]
    seen = gt != 255
    z = EV["logit"].astype(np.float64)
    s_head = np.exp(z - LOGW - (z - LOGW).max(1, keepdims=True)); s_head /= s_head.sum(1, keepdims=True)
    s_raw = np.exp(z - z.max(1, keepdims=True)); s_raw /= s_raw.sum(1, keepdims=True)
    scores = {"head": s_head, "head_rawsoftmax": s_raw, "C1_constant": np.broadcast_to(c1, (len(gt), 8)),
              "C2_posprior": c2[EV["row"], EV["col"]]}
    for (st, kind), Z in zoo.items():
        for si, m in enumerate(Z["models"]):
            if kind == "conv3x3":
                scores[f"{st}|{kind}|s{si}"] = predict_conv(m, patch(EV["patch_enc"]))
            elif st in ("enc", "lift"):
                scores[f"{st}|{kind}|s{si}"] = predict_cells(m, feat(EV, st))
    # img-based probes: stream the eval windows (each file read once), score every GT-seen cell, NaN elsewhere
    img_models = [(f"{st}|{kind}|s{si}", st, m) for (st, kind), Z in zoo.items() if st in ("img", "enc+img")
                  for si, m in enumerate(Z["models"])]
    for name, _, _ in img_models:
        scores[name] = np.full((len(gt), 8), np.nan, np.float32)
    NC = len(EV_R) * len(EV_C)
    files_ev = sorted(Path(a.cap_eval).glob("w_*.npz"))
    if len(files_ev) * NC != len(gt):
        raise SystemExit("[probe] eval cell bookkeeping: files x cells != loaded cells")
    n_img_rows = 0
    for wi, fpath in enumerate(files_ev):
        im = np.load(fpath)["img"]
        im = im.reshape(im.shape[0], -1).T
        cells = np.arange(wi * NC, (wi + 1) * NC)
        keep = np.nonzero(gt[cells] != 255)[0]
        if keep.size == 0:
            continue
        Xi = ((im[keep].astype(np.float32) - mu_i) / sd_i).astype(np.float16)
        Xe = feat(EV, "enc", cells[keep])
        for name, st, m in img_models:
            scores[name][cells[keep]] = predict_cells(m, np.concatenate([Xe, Xi], 1) if st == "enc+img" else Xi)
        n_img_rows += int(keep.size)
    if n_img_rows != int(seen.sum()):
        raise SystemExit("[probe] an eval GT-seen cell has no img score: the img stage would be scored on a subset")
    np.savez_compressed(out / "eval_cell_scores.npz", gt=gt, lv=lv, xm=xm, clip=clip, dec=EV["dec"],
                        **{"sw__" + k.replace("|", "__"): v[:, SW].astype(np.float32) for k, v in scores.items()},
                        **{"drv__" + k.replace("|", "__"): v[:, list(DRV)].sum(1).astype(np.float32) for k, v in scores.items()})
    # ---- metrics ------------------------------------------------------------------------------------------- #
    def cellset(lo, hi, stratum):
        m = seen & (xm >= lo) & (xm < hi)
        if stratum == "lv":
            m &= lv
        elif stratum == "nlv":
            m &= ~lv
        return m

    def head_op(m):
        flag = EV["dec"][m] == SW
        pos = gt[m] == SW
        tp = int((flag & pos).sum())
        return (tp / max(1, int(flag.sum()))), (tp / max(1, int(pos.sum()))), int(flag.sum())
    table = {}
    for bname, (lo, hi) in list(BANDS.items()) + [(f"bin_{lo}_{hi}", (lo, hi)) for lo, hi in BINS]:
        for strat in ("lv", "all", "nlv"):
            m = cellset(lo, hi, strat)
            if m.sum() == 0:
                continue
            ph, rh, nf = head_op(m)
            row = {"n_cells": int(m.sum()), "n_windows": int(len(np.unique(EV["win"][m]))), "n_clips": int(len(np.unique(clip[m]))),
                   "sw_rate": float((gt[m] == SW).mean()), "drv_rate": float(np.isin(gt[m], DRV).mean()),
                   "head_decision": {"precision": ph, "recall": rh, "n_flags": nf}}
            for k, s in scores.items():
                ssw = s[m, SW]
                sdr = s[m][:, list(DRV)].sum(1)
                row[k] = {"ap_sw": ap_score(ssw, gt[m] == SW), "auroc_sw": auroc(ssw, gt[m] == SW),
                          "p_at_head_recall_sw": prec_at_recall(ssw, gt[m] == SW, rh),
                          "ap_drv": ap_score(sdr, np.isin(gt[m], DRV)),
                          "acc8": float((s[m].argmax(1) == gt[m]).mean())}
            table[f"{bname}|{strat}"] = row
    # C4: the head's psw >= 0.5 precision per MAPSEL band, all GT-seen corridor cells
    c4 = {}
    for lo, hi, ref in ((0, 5, 0.251), (5, 10, 0.733), (10, 20, 0.716), (20, 40, 0.744)):
        m = cellset(lo, hi, "all")
        fl = s_head[m, SW] >= 0.5
        c4[f"{lo}_{hi}"] = {"mapsel": ref, "read": float((gt[m][fl] == SW).mean()) if fl.any() else None,
                            "flag_rate": float(fl.mean())}
    c4_pass = all(v["read"] is not None and abs(v["read"] - v["mapsel"]) <= 0.05 for v in c4.values())
    # C1: constant must read the positive rate exactly
    c1chk = {}
    for key in ("A_0_20|lv", "B_5_40|lv"):
        c1chk[key] = {"ap": table[key]["C1_constant"]["ap_sw"], "rate": table[key]["sw_rate"],
                      "auroc": table[key]["C1_constant"]["auroc_sw"]}
    c1_pass = all(abs(v["ap"] - v["rate"]) < 1e-6 and abs(v["auroc"] - 0.5) < 1e-9 for v in c1chk.values())
    # ---- best probe (chosen on VAL ap, NOT on eval) + paired bootstrap ------------------------------------- #
    probe_keys = [k for k in zoo]
    best_all = max(probe_keys, key=lambda k: max(zoo[k]["val_ap"]))
    best_img = max([k for k in probe_keys if k[0] == "img"], key=lambda k: max(zoo[k]["val_ap"]))
    best_enc = max([k for k in probe_keys if k[0] == "enc"], key=lambda k: max(zoo[k]["val_ap"]))

    def sk(k):
        Z = zoo[k]
        return f"{k[0]}|{k[1]}|s{int(np.argmax(Z['val_ap']))}"
    rng = np.random.default_rng(0)
    ncl = int(clip.max()) + 1
    by_c = [np.nonzero(clip == c)[0] for c in range(ncl)]
    comps = {"delta_best_minus_head|A_0_20|lv": ("A_0_20|lv", sk(best_all), "head"),
             "delta_best_minus_head|B_5_40|lv": ("B_5_40|lv", sk(best_all), "head"),
             "delta_img_minus_enc|A_0_20|lv": ("A_0_20|lv", sk(best_img), sk(best_enc)),
             "delta_img_minus_enc|B_5_40|lv": ("B_5_40|lv", sk(best_img), sk(best_enc)),
             "delta_best_minus_head|A0_0_5|all": ("A0_0_5|all", sk(best_all), "head"),
             "delta_C2_minus_head|A0_0_5|all": ("A0_0_5|all", "C2_posprior", "head")}
    bands_of = {"A_0_20|lv": cellset(0, 20, "lv"), "B_5_40|lv": cellset(5, 40, "lv"), "A0_0_5|all": cellset(0, 5, "all")}
    boot = {}
    for name, (key, ka, kb) in comps.items():
        m = bands_of[key]
        pa, pb, pos = scores[ka][:, SW], scores[kb][:, SW], gt == SW
        point = ap_score(pa[m], pos[m]) - ap_score(pb[m], pos[m])
        ds = []
        for _ in range(a.draws):
            idx = np.concatenate([by_c[c] for c in rng.integers(0, ncl, ncl)])
            idx = idx[m[idx]]
            ds.append(ap_score(pa[idx], pos[idx]) - ap_score(pb[idx], pos[idx]))
        lo_, hi_ = np.nanpercentile(ds, [2.5, 97.5])
        boot[name] = {"a": ka, "b": kb, "cells": key, "point": point, "ci95": [float(lo_), float(hi_)], "separated": bool(lo_ > 0 or hi_ < 0)}
        print(f"[probe] {name}: {point:+.4f} [{lo_:+.4f}, {hi_:+.4f}]", flush=True)
    # ---- the committed reading ----------------------------------------------------------------------------- #
    dA = boot["delta_best_minus_head|A_0_20|lv"]
    dB = boot["delta_best_minus_head|B_5_40|lv"]
    verdict = {}
    for nm, d in (("A_0_20", dA), ("B_5_40", dB)):
        verdict[nm] = "O1_HEAD_LIMITED" if (d["point"] >= 0.05 and d["ci95"][0] > 0) else "O2_FEATURE_LIMITED"
    loc = {}
    for nm in ("A_0_20", "B_5_40"):
        d = boot[f"delta_img_minus_enc|{nm}|lv"]
        loc[nm] = "LIFT_OR_BEV_ENCODER_LOSES_INFO" if (d["point"] >= 0.05 and d["ci95"][0] > 0) else "IMAGE_TRUNK_IS_THE_CEILING"
    a0 = cellset(0, 5, "all")
    flags_a0 = a0 & (EV["dec"] == SW)
    vis = {"A0_seen_cells": int(a0.sum()), "A0_share_lift_invalid": float((~lv[a0]).mean()),
           "A0_head_sw_flags": int(flags_a0.sum()), "A0_share_of_flags_on_lift_invalid": float((~lv[flags_a0]).mean()) if flags_a0.any() else None}
    vis["VISIBILITY_LIMITED"] = bool(vis["A0_share_of_flags_on_lift_invalid"] is not None and vis["A0_share_of_flags_on_lift_invalid"] > 0.5)
    dl = zoo[("enc", "mlp")]
    rec.update({
        "n_eval_windows": int(len(EV["patch_enc"])), "n_eval_clips": len(EV["clips"]),
        "controls": {"C1_constant": {"pass": c1_pass, "read": c1chk}, "C4_head_psw_vs_mapsel": {"pass": c4_pass, "read": c4},
                     "C3_n_d": {f"{k[0]}|{k[1]}": {"d": v["d"], "n_fit": v["n_fit"]} for k, v in zoo.items()}},
        "zoo": {f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != "models"} for k, v in zoo.items()},
        "best_probe_by_val": f"{best_all[0]}|{best_all[1]}", "best_img_by_val": f"{best_img[0]}|{best_img[1]}",
        "best_enc_by_val": f"{best_enc[0]}|{best_enc[1]}",
        "data_limited_check": {"enc_mlp_val_ap_full": max(dl["val_ap"][:1]), "enc_mlp_val_ap_half": dl.get("half_fit_val_ap"),
                               "gain_from_doubling": max(dl["val_ap"][:1]) - dl.get("half_fit_val_ap", float("nan")),
                               "DATA_LIMITED": bool(max(dl["val_ap"][:1]) - dl.get("half_fit_val_ap", 0) >= 0.02)},
        "bootstrap": boot, "verdict": verdict, "localisation": loc, "visibility": vis, "table": table,
        "estimator": f"paired episode-cluster bootstrap over eval clips, {a.draws} draws, seed 0; answers another draw of EPISODES; "
                     "probe seeds 0/1 reported in zoo; the head is ONE training seed",
        "wall_s": time.time() - t0})
    (out / "RESULT_Q1.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "localisation": loc, "visibility": vis, "controls_pass": [c1_pass, c4_pass],
                      "best": rec["best_probe_by_val"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
