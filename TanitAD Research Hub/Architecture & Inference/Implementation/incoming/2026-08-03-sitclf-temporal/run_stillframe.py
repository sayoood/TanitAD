"""STILL-FRAME vs REAL, paired on identical rows: is the situation classifier reading APPEARANCE?

THE QUESTION (coordinator's mid-flight correction, 2026-08-03)
--------------------------------------------------------------
A single 32x32 grayscale still frame was measured to read ego `speed` at 93 % of the 800 ms learned
latent. The situation labels are pure functions of the ego pose track. So a "vision-only" situation
classifier could be riding appearance -> speed -> label rather than perceiving the situation. This
runs the SAME arms on a substrate whose only difference is that the encoder's 3-frame stack carries
**no inter-frame motion** (`build_stillframe_substrate.py`).

WHAT IS COMPARED. Skill over each substrate's OWN permuted-feature null, so the two are on the same
scale despite being different feature spaces:

    recovery = (still_lift - still_null_lift) / (real_lift - real_null_lift)

  recovery ~ 1  =>  motion contributes nothing; the arm is reading APPEARANCE. The temporal story
                    is not what carries the signal, and the appearance-shortcut concern is LIVE.
  recovery ~ 0  =>  the arm's skill is motion-borne after all.

The paired episode-cluster bootstrap is valid here because both substrates share rows, labels, clip
ids and folds **by construction** — asserted below, not assumed.

usage:
  python run_stillframe.py --real <npz> --still <npz> --out results_stillframe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.eval.ap_ci import (ap_episode_cluster_bootstrap,        # noqa: E402
                                ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import (CausalSitHead, causal_window,       # noqa: E402
                                 clip_runs, cluster_folds,
                                 predict_sit_head, ridge_scores,
                                 width_for_param_budget)
from tanitad.eval.sitclf_deploy import precision_recall_at_budget    # noqa: E402

SEL_FRAC = 0.20
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
WIN_MAX = 8
ARMS = (("ridge_app16_w8", "ridge", 16, 8, None),
        ("ridge_app16_w1", "ridge", 16, 1, None),
        ("tf_app16_w8_d128", "tf", 16, 8, 417_028))


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def fit_pca(F, rows, r, device="cpu"):
    A = torch.from_numpy(np.asarray(F[rows], dtype=np.float32)).to(device)
    mu = A.mean(0, keepdim=True)
    A = A - mu
    _u, _s, v = torch.svd_lowrank(A, q=r + 16, niter=4)
    del A
    if device != "cpu":
        torch.cuda.empty_cache()
    return mu.cpu().numpy(), v[:, :r].cpu().numpy().astype(np.float32)


def project(F, mu, W):
    img = (np.asarray(F, dtype=np.float32) - mu) @ W
    img /= max(float(np.abs(img).mean()), 1e-6)
    return img


def mean_ap(y, s, v):
    vals = []
    for i in range(y.shape[1]):
        m = v[:, i]
        if m.sum() < 50 or y[m, i].sum() < 5:
            continue
        vals.append(average_precision(y[m, i], s[m, i]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--real", required=True)
    ap_.add_argument("--still", required=True)
    ap_.add_argument("--out", default="results_stillframe.json")
    ap_.add_argument("--n-boot", type=int, default=2000)
    ap_.add_argument("--device", default=None)
    a = ap_.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ZR, ZS = np.load(a.real), np.load(a.still)
    # ---- C-FID: the two substrates MUST share rows, labels, clips ------------
    for k in ("Y", "V", "clip_cluster", "t", "cache_tag"):
        if not np.array_equal(ZR[k], ZS[k]):
            raise SystemExit(f"C-FID FAILED: {k} differs between substrates — not pairable")
    if ZR["F"].shape != ZS["F"].shape:
        raise SystemExit("C-FID FAILED: feature shapes differ")
    same = float(np.mean(np.asarray(ZR["F"][:2000], np.float32)
                         == np.asarray(ZS["F"][:2000], np.float32)))
    log(f"C-FID OK: labels/clips/rows identical; feature arrays differ "
        f"(elementwise equal fraction on a 2000-row probe: {same:.4f})")
    if same > 0.99:
        raise SystemExit("the still-frame substrate is (nearly) identical to the real one — "
                         "the motion removal did not take effect")

    cc = ZR["clip_cluster"]
    sits = [str(s) for s in ZR["situations"]]
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)
    Yi = ZR["Y"].astype(np.int64)
    Y = ZR["Y"].astype(np.float32)
    _, hist_ok = causal_window(np.zeros((len(cc), 1), np.float32), st, en, WIN_MAX)
    V = ZR["V"].astype(bool) & hist_ok[:, None]
    Vf = V.astype(np.float32)
    log(f"{len(cc):,} rows, {len(st)} clips, common scorable {int(hist_ok.sum()):,}")

    rng = np.random.default_rng(0)
    perm_clip = rng.permutation(len(st))
    shuf_rows = np.arange(len(cc))
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1

    splits = []
    for f in (0, 1):
        te = folds == f
        tr = ~te
        tr_cl = np.unique(cc[tr])
        perm = np.random.default_rng(0 + f).permutation(tr_cl)
        sel_cl = set(int(x) for x in perm[:max(1, int(round(SEL_FRAC * len(tr_cl))))])
        is_sel = np.array([int(x) in sel_cl for x in cc])
        splits.append((tr & ~is_sel, tr & is_sel, te))

    scores = {}
    for sub_name, Z in (("real", ZR), ("still", ZS)):
        Fm = Z["F"]
        for arm, kind, r, win, budget in ARMS:
            for shuf in (False, True):
                key = f"{sub_name}::{arm}" + ("::NULL" if shuf else "")
                out = np.zeros_like(Y)
                d = None
                for f in (0, 1):
                    src = Fm[shuf_rows] if shuf else Fm
                    mu, W = fit_pca(src, np.flatnonzero(splits[f][0]), r, device=device)
                    X, _ = causal_window(project(src, mu, W), st, en, win)
                    fit, sel, te = splits[f]
                    if kind == "ridge":
                        best = (-np.inf, 1.0)
                        for lam in LAMBDAS:
                            m_ = mean_ap(Yi[sel],
                                         ridge_scores(X[fit], Y[fit], Vf[fit], X[sel], lam=lam),
                                         V[sel])
                            if np.isfinite(m_) and m_ > best[0]:
                                best = (m_, lam)
                        out[te] = ridge_scores(X[fit], Y[fit], Vf[fit], X[te], lam=best[1])
                    else:
                        if d is None:
                            d = width_for_param_budget(budget, r, win, 3)
                        torch.manual_seed(0)
                        m = CausalSitHead(r, win, d=d, n_out=3).to(device)
                        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
                        lossf = torch.nn.BCEWithLogitsLoss(
                            reduction="none", pos_weight=torch.full((3,), 20.0, device=device))
                        Xf = torch.from_numpy(np.ascontiguousarray(X[fit], np.float32))
                        Yf = torch.from_numpy(np.ascontiguousarray(Y[fit], np.float32))
                        Vff = torch.from_numpy(np.ascontiguousarray(Vf[fit], np.float32))
                        g = torch.Generator().manual_seed(0)
                        best = (-np.inf, None)
                        for _ep in range(8):
                            m.train()
                            pm = torch.randperm(Xf.shape[0], generator=g)
                            for b in range(0, Xf.shape[0], 1024):
                                jj = pm[b:b + 1024]
                                vb = Vff[jj].to(device)
                                if float(vb.sum()) == 0:
                                    continue
                                xb = Xf[jj].to(device).view(len(jj), win, r)
                                loss = (lossf(m(xb), Yf[jj].to(device)) * vb).sum() / vb.sum()
                                opt.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                                opt.step()
                            sc = mean_ap(Yi[sel], predict_sit_head(m, X[sel], in_dim=r,
                                                                   device=device), V[sel])
                            if np.isfinite(sc) and sc > best[0]:
                                best = (sc, predict_sit_head(m, X[te], in_dim=r, device=device))
                        out[te] = best[1]
                    del X
                scores[key] = out
                log(f"  {key}: done")

    res = {"_what": "STILL-FRAME control — does appearance alone carry the situation signal?",
           "_recovery": ("(still_lift - still_null) / (real_lift - real_null); ~1 means motion "
                         "contributes nothing and the arm is reading APPEARANCE"),
           "real_substrate": str(a.real), "still_substrate": str(a.still),
           "n_boot": a.n_boot, "per_situation": {}}

    for i, s in enumerate(sits):
        m = V[:, i]
        yv = Yi[m, i]
        eid = cc[m]
        ones = np.ones(yv.size, bool)
        row = {"n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
               "base_rate": round(float(yv.mean()), 6),
               "n_clusters_with_a_positive": int(len(np.unique(eid[yv > 0]))),
               "arms": {}}
        for arm, *_ in ARMS:
            rr = scores[f"real::{arm}"][m, i].astype(np.float64)
            ss = scores[f"still::{arm}"][m, i].astype(np.float64)
            rn = scores[f"real::{arm}::NULL"][m, i].astype(np.float64)
            sn = scores[f"still::{arm}::NULL"][m, i].astype(np.float64)
            lr_, ls_ = ap_lift(yv, rr), ap_lift(yv, ss)
            nr_, ns_ = ap_lift(yv, rn), ap_lift(yv, sn)
            skill_r, skill_s = lr_ - nr_, ls_ - ns_
            row["arms"][arm] = {
                "real": {"ap": round(average_precision(yv, rr), 5),
                         "ap_lift": ap_episode_cluster_bootstrap(yv, rr, eid,
                                                                 n_boot=a.n_boot, lift=True),
                         "own_null_ap_lift": round(nr_, 5),
                         "skill_over_null": round(skill_r, 5),
                         "op_5pct": precision_recall_at_budget(yv, rr, ones)},
                "still": {"ap": round(average_precision(yv, ss), 5),
                          "ap_lift": ap_episode_cluster_bootstrap(yv, ss, eid,
                                                                  n_boot=a.n_boot, lift=True),
                          "own_null_ap_lift": round(ns_, 5),
                          "skill_over_null": round(skill_s, 5),
                          "op_5pct": precision_recall_at_budget(yv, ss, ones)},
                "paired_still_minus_real": paired_ap_episode_cluster_bootstrap(
                    yv, ss, rr, eid, n_boot=a.n_boot, lift=True),
                "still_vs_its_own_null": paired_ap_episode_cluster_bootstrap(
                    yv, ss, sn, eid, n_boot=a.n_boot, lift=True),
                "RECOVERY_FRACTION": (round(float(skill_s / skill_r), 4)
                                      if abs(skill_r) > 1e-9 else None)}
            q = row["arms"][arm]
            p = q["paired_still_minus_real"]
            log(f"  {s} {arm}: real lift {lr_:.3f} (null {nr_:.3f}, skill {skill_r:+.3f}) | "
                f"still lift {ls_:.3f} (null {ns_:.3f}, skill {skill_s:+.3f}) | "
                f"RECOVERY {q['RECOVERY_FRACTION']} | still-real {p['delta']:+.3f} "
                f"[{p['lo']:+.3f},{p['hi']:+.3f}]{' SEP' if p['separated'] else ''}")
        res["per_situation"][s] = row
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
