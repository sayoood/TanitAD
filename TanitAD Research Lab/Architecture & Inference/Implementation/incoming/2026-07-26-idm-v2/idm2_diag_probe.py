"""IDM-v2 DIAGNOSIS (a) + (c): is the ceiling the FROZEN REPRESENTATION,
or monocular SCALE?

(a) LINEAR PROBE.  Closed-form ridge on the frozen latents, three receptive
    fields (centre frame / 9 frames / 17 frames).  Ridge lambda is chosen on an
    episode-disjoint INNER split of TRAIN only (no val leakage).  If a linear
    probe reaches the trained 2.9 M-param head, the head is not the bottleneck.

(c) SCALE.  For every channel we additionally report the metric after an
    ORACLE affine recalibration y ~ a*yhat + b fitted (i) globally, (ii) per
    DOMAIN, (iii) per CLIP.  A large per-clip/per-domain gain with a small
    global gain is the signature of monocular scale ambiguity (a multiplicative
    gain that the head cannot know); an additive-only gain is not.

    We also probe SCALE-FREE reparametrisations: curvature kappa = yaw_rate / v
    (rotation is observable from a single camera WITHOUT metric scale) and the
    cross-domain transfer PAI->comma / comma->PAI.

Writes /root/idm2/out/probe.json
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
import idm2_lib as L  # noqa: E402

K = 8                      # 17-frame superset; 9-frame is the inner slice
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False   # TF32 Gram -> non-PD Cholesky
torch.backends.cudnn.allow_tf32 = False


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
def _gram64(A, B, chunk=4096):
    """A[N,D] @ B[M,D].T accumulated in float64 (fp32 Gram over D=35k loses
    enough precision that Cholesky reports a non-PD matrix)."""
    out = torch.zeros(A.shape[0], B.shape[0], device=A.device, dtype=torch.float64)
    for i in range(0, A.shape[1], chunk):
        out += A[:, i:i + chunk].double() @ B[:, i:i + chunk].double().T
    return out


def ridge_fit_predict(Xtr, Ytr, Xev_list, lams):
    """Dual-form ridge.  Xtr [N,D] fp32 cuda (already standardised, no bias),
    Ytr [N,C] (already centred).  -> {lam: [pred on each Xev]}"""
    N = Xtr.shape[0]
    Kg = _gram64(Xtr, Xtr)
    Kg = 0.5 * (Kg + Kg.T)
    Kx = [_gram64(Xe, Xtr) for Xe in Xev_list]
    Yd = Ytr.double()
    I = torch.eye(N, device=Xtr.device, dtype=torch.float64)
    out = {}
    for lam in lams:
        A = Kg + lam * I
        Lc, info = torch.linalg.cholesky_ex(A)
        if int(info) != 0:
            alpha = torch.linalg.solve(A, Yd)
        else:
            alpha = torch.cholesky_solve(Yd, Lc)
        out[lam] = [(Kv @ alpha).float().cpu().numpy() for Kv in Kx]
        del A, Lc, alpha
    del Kg, Kx
    torch.cuda.empty_cache()
    return out


def standardise(Xtr, Xs):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-4)
    return (Xtr - mu) / sd, [(X - mu) / sd for X in Xs]


def affine_recal(pred, gt, groups, mode="affine"):
    """Oracle least-squares recalibration per group (groups=None -> global).

    mode="affine"  y ~ a*p + b   (2 dof)
    mode="scale"   y ~ a*p       (1 dof, PURELY MULTIPLICATIVE)
    mode="offset"  y ~ p + b     (1 dof, PURELY ADDITIVE)
    mode="mean"    y ~ b         (the "just know the clip mean" control -- this
                                  is what makes a per-clip affine look good for
                                  a channel whose variance is mostly BETWEEN clips)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    out = np.array(pred, copy=True)
    g = np.zeros(len(pred), dtype=object) if groups is None else np.asarray(groups)
    for u in np.unique(g):
        m = (g == u)
        if m.sum() < 5:
            continue
        p, y = pred[m], gt[m]
        if mode == "affine":
            A = np.stack([p, np.ones(m.sum())], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            out[m] = A @ coef
        elif mode == "scale":
            a = float((p * y).sum() / max((p * p).sum(), 1e-12))
            out[m] = a * p
        elif mode == "offset":
            out[m] = p + float((y - p).mean())
        elif mode == "mean":
            out[m] = float(y.mean())
    return out


# --------------------------------------------------------------------------- #
def main():
    tr_tags, va_tags = L.split_tags()
    log(f"train {len(tr_tags)} eps / val {len(va_tags)} eps")
    tr = L.build_set(tr_tags, k=K, stride=1)
    va = L.build_set(va_tags, k=K, stride=2)
    log(f"train windows {tuple(tr['Z'].shape)}  val {tuple(va['Z'].shape)}")

    # inner episode-disjoint split of TRAIN for lambda selection
    inner_val_eps = set(sorted(set(tr["eid"]))[::4])
    im = np.array([e in inner_val_eps for e in tr["eid"]])
    log(f"inner lambda-selection holdout: {im.sum()} windows / "
        f"{len(inner_val_eps)} eps")

    Str, Sva = tr["S"].numpy().astype(np.float64), va["S"].numpy().astype(np.float64)
    # derived scale-free target: curvature kappa = yaw_rate / v  (v floored)
    def kappa(S):
        return S[:, 1] / np.clip(S[:, 0], 3.0, None)
    TARGETS = list(L.SCALARS) + ["kappa", "log_speed"]

    def tgt(S, name):
        if name == "kappa":
            return kappa(S)
        if name == "log_speed":      # multiplicative error becomes ADDITIVE here
            return np.log(np.clip(S[:, 0], 0.5, None))
        return S[:, L.SCALARS.index(name)]

    Ytr_np = np.stack([tgt(Str, t) for t in TARGETS], 1)
    Yva_np = np.stack([tgt(Sva, t) for t in TARGETS], 1)

    Zt = tr["Z"].to(DEV)                       # [N,17,2048] fp16
    Zv = va["Z"].to(DEV)
    SLICES = {"center": slice(K, K + 1), "w9": slice(K - 4, K + 5),
              "w17": slice(0, 2 * K + 1)}
    LAMS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

    res = {"n_train_windows": int(tr["S"].shape[0]),
           "n_val_windows": int(va["S"].shape[0]),
           "n_train_eps": len(tr_tags), "n_val_eps": len(va_tags),
           "targets": TARGETS, "lams": LAMS, "probes": {}}

    for sname, sl in SLICES.items():
        Xt = Zt[:, sl].reshape(Zt.shape[0], -1).float()
        Xv = Zv[:, sl].reshape(Zv.shape[0], -1).float()
        Xt, (Xv,) = standardise(Xt, [Xv])
        Ymu = torch.tensor(Ytr_np[~im].mean(0), device=DEV, dtype=torch.float32)
        Yt = torch.tensor(Ytr_np, device=DEV, dtype=torch.float32)

        # --- lambda selection on the inner holdout (fit on inner-train only)
        t0 = time.time()
        pi = ridge_fit_predict(Xt[~im], Yt[~im] - Ymu, [Xt[im]], LAMS)
        best = {}
        for ti, tn in enumerate(TARGETS):
            sc = {lam: L.chan_metrics(pi[lam][0][:, ti] + Ymu[ti].item(),
                                      Ytr_np[im, ti])["r2"] for lam in LAMS}
            best[tn] = max(sc, key=sc.get)
        log(f"{sname}: inner lambda {best} ({time.time()-t0:.0f}s)")
        del pi
        torch.cuda.empty_cache()

        # --- refit on FULL train, predict val
        Ymu2 = torch.tensor(Ytr_np.mean(0), device=DEV, dtype=torch.float32)
        pf = ridge_fit_predict(Xt, Yt - Ymu2, [Xv], LAMS)
        entry = {"lambda_selected": best, "D": int(Xt.shape[1]), "per_target": {}}
        for ti, tn in enumerate(TARGETS):
            lam = best[tn]
            p = pf[lam][0][:, ti] + Ymu2[ti].item()
            g = Yva_np[:, ti]
            m = L.chan_metrics(p, g)
            m["boot_r2"] = L.boot_r2(p, g, va["eid"], n_boot=500)
            m["boot_medae"] = L.boot_mae(p, g, va["eid"], n_boot=500,
                                         reduce="median")
            # per-domain
            m["per_domain"] = {}
            for dom in ("pai", "cm"):
                dm = va["dom"] == dom
                m["per_domain"][dom] = L.chan_metrics(p[dm], g[dm])
            # (c) oracle recalibration -- WITH the multiplicative/additive
            # controls.  scale-only ~= affine  => the error is a GAIN
            # (monocular scale); offset-only ~= affine => it is a bias;
            # "mean" is the no-information control.
            m["recal"] = {}
            for grp_name, grp in (("global", None), ("per_domain", va["dom"]),
                                  ("per_clip", va["eid"])):
                for mode in ("affine", "scale", "offset", "mean"):
                    key = grp_name if mode == "affine" else f"{grp_name}_{mode}"
                    m["recal"][key] = L.chan_metrics(
                        affine_recal(p, g, grp, mode=mode), g)
            # lambda sweep on val (transparency only, NOT used for selection)
            m["val_r2_by_lambda"] = {
                str(l): L.chan_metrics(pf[l][0][:, ti] + Ymu2[ti].item(), g)["r2"]
                for l in LAMS}
            entry["per_target"][tn] = m
            print(f"  {sname:<7}{tn:<11} R2 {m['r2']:+.4f} rho {m['rho']:+.4f} "
                  f"MAE {m['mae']:.4f} nMedAE {m['nmedae']:.3f} | "
                  f"pai {m['per_domain']['pai']['r2']:+.3f} "
                  f"cm {m['per_domain']['cm']['r2']:+.3f} | recal glob "
                  f"{m['recal']['global']['r2']:+.3f} dom "
                  f"{m['recal']['per_domain']['r2']:+.3f} clip "
                  f"{m['recal']['per_clip']['r2']:+.3f} "
                  f"[clip scale-only {m['recal']['per_clip_scale']['r2']:+.3f} "
                  f"offset-only {m['recal']['per_clip_offset']['r2']:+.3f} "
                  f"mean-only {m['recal']['per_clip_mean']['r2']:+.3f}]",
                  flush=True)
        res["probes"][sname] = entry
        del pf, Xt, Xv
        torch.cuda.empty_cache()

    # ---- (c) CROSS-DOMAIN transfer with the w9 probe --------------------- #
    log("cross-domain transfer probes (w9)")
    sl = SLICES["w9"]
    Xt_all = Zt[:, sl].reshape(Zt.shape[0], -1).float()
    Xv_all = Zv[:, sl].reshape(Zv.shape[0], -1).float()
    cross = {}
    for src in ("pai", "cm"):
        mt = tr["dom"] == src
        Xt = Xt_all[torch.tensor(mt, device=DEV)]
        Xts, (Xvs,) = standardise(Xt, [Xv_all])
        Y = torch.tensor(Ytr_np[mt], device=DEV, dtype=torch.float32)
        Ymu = Y.mean(0)
        pf = ridge_fit_predict(Xts, Y - Ymu, [Xvs], [1e3, 1e4, 1e5])
        cross[src] = {}
        for lam in pf:
            for ti, tn in enumerate(TARGETS):
                p = pf[lam][0][:, ti] + Ymu[ti].item()
                for dom in ("pai", "cm"):
                    dm = va["dom"] == dom
                    mm = L.chan_metrics(p[dm], Yva_np[dm, ti])
                    mm["recal_per_clip_r2"] = L.chan_metrics(
                        affine_recal(p[dm], Yva_np[dm, ti], va["eid"][dm]),
                        Yva_np[dm, ti])["r2"]
                    cross[src][f"lam{lam:g}|{tn}|->{dom}"] = mm
        del pf, Xts, Xvs
        torch.cuda.empty_cache()
    res["cross_domain_w9"] = cross
    for src in cross:
        for tn in ("speed", "yaw_rate", "kappa"):
            for dom in ("pai", "cm"):
                k = f"lam10000|{tn}|->{dom}"
                if k in cross[src]:
                    v = cross[src][k]
                    print(f"  train {src} -> eval {dom:<4} {tn:<9} "
                          f"R2 {v['r2']:+.4f} MAE {v['mae']:.4f} "
                          f"recal/clip R2 {v['recal_per_clip_r2']:+.4f}",
                          flush=True)

    L.jdump(res, "/root/idm2/out/probe.json")


if __name__ == "__main__":
    main()
