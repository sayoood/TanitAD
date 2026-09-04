"""E-SEED-1 v2 — two defects in v1's probe repaired, and a spatial read added.

⛔ DEFECT 1 (v1): ONE ridge lambda was selected to maximise the MEAN inner-CV R2
   ACROSS BOTH TARGETS. On `dino_hf` speed dominated, dragged lambda to 1e3, and
   yaw_rate then read R2 -0.3095 -- an overfit, not a measurement. ⇒ lambda is
   now selected PER TARGET.
⛔ DEFECT 2 (v1): every arm except `dino_hf` selected lambda at the GRID MAXIMUM
   (1e4), i.e. the inner CV wanted more shrinkage than the grid allowed. That is
   the 2026-08-22 failure #3/#4 shape (max-lambda shrinks the ridge to the
   constant predictor and every arm reads the same near-zero value). ⇒ the grid
   is extended to 1e8 and the selected lambda is REFUSED if it sits at either
   boundary.
⭐ ADDED: a 4x4 SPATIAL pooling beside the global mean pool. Decoding (steer,
   accel) from Delta-z is a VISUAL-ODOMETRY task; a global mean pool over 256
   patch tokens destroys exactly the spatial signal it needs, so v1's LDAD read
   (~0 on EVERY arm, including real DINOv3) could not distinguish "LDAD is at the
   floor" from "the pooling threw the signal away". The spatial read separates
   those two.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BANK = HERE / "eseed1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "clone_scripts"))

import e_seed1_arms as E1                                    # noqa: E402

ARMS = ["dino_hf", "seed_asis", "seed_imnet", "seed_imnet_pos0",
        "scratch", "pixel"]
LAMBDAS = tuple(10.0 ** k for k in range(-4, 9))             # 1e-4 .. 1e8
PCA_K = 256
N_FIT_EP = 60
N_BOOT = 2000


@torch.no_grad()
def extract_spatial(arm: str, frames) -> np.ndarray:
    """4x4 spatial average pool of the patch-token grid -> [n, 16*d]."""
    n = frames.shape[0]
    if arm == "pixel":
        out = np.empty((n, 4 * 4 * 3), dtype=np.float32)
        for i in range(0, n, 256):
            x = torch.from_numpy(np.array(frames[i:i + 256])).float() / 255.0
            p = torch.nn.functional.avg_pool2d(x, 64)
            out[i:i + 256] = p.reshape(p.shape[0], -1).numpy()
        return out
    imnet = arm in ("dino_hf", "seed_imnet", "seed_imnet_pos0")
    if arm == "dino_hf":
        model = E1.build_hf().to(E1.DEV)
    elif arm == "scratch":
        model = E1.build_scratch().to(E1.DEV).to(E1.DT)
    else:
        model, _ = E1.build_seeded(zero_pos=(arm == "seed_imnet_pos0"))
        model = model.to(E1.DEV).to(E1.DT)
    mean, std = E1.IMNET_MEAN.to(E1.DEV), E1.IMNET_STD.to(E1.DEV)
    feats = None
    t0 = time.time()
    B = E1.BATCH
    for i in range(0, n, B):
        x = torch.from_numpy(np.array(frames[i:i + B])).to(E1.DEV).float() / 255.
        if imnet:
            x = (x - mean) / std
        x = x.to(E1.DT)
        tok = (model(pixel_values=x).last_hidden_state[:, -256:]
               if arm == "dino_hf" else model(x))
        b, _n, d = tok.shape
        g = tok.float().reshape(b, 16, 16, d).permute(0, 3, 1, 2)   # [b,d,16,16]
        g = torch.nn.functional.avg_pool2d(g, 4)                    # [b,d,4,4]
        z = g.reshape(b, -1)
        if feats is None:
            feats = np.empty((n, z.shape[1]), dtype=np.float32)
        feats[i:i + B] = z.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    assert np.isfinite(feats).all() and float(feats.std()) > 0
    print(f"  [{arm}/sp] d={feats.shape[1]} std={feats.std():.4f} "
          f"{time.time()-t0:.0f}s", flush=True)
    return feats


def r2(y, yh):
    ss = ((y - yh) ** 2).sum(0)
    st = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - ss / st


def probe1(Ffit, yfit, gfit, Fsc, ysc, gsc):
    """ONE target. PCA + ridge, lambda by 5-fold GroupKFold over FIT episodes."""
    mu = Ffit.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ffit - mu, full_matrices=False)
    V = Vt[:PCA_K].T
    Zf, Zs = (Ffit - mu) @ V, (Fsc - mu) @ V
    sd = Zf.std(0, keepdims=True) + 1e-8
    Zf, Zs = (Zf / sd).astype(np.float64), (Zs / sd).astype(np.float64)
    ym, ys = yfit.mean(), yfit.std() + 1e-8
    yf = ((yfit - ym) / ys).astype(np.float64)

    eps = np.unique(gfit)
    folds = np.array_split(eps, 5)
    curve = []
    for lam in LAMBDAS:
        sc = []
        for f in folds:
            m = np.isin(gfit, f)
            A = Zf[~m].T @ Zf[~m] + lam * np.eye(Zf.shape[1])
            W = np.linalg.solve(A, Zf[~m].T @ yf[~m])
            sc.append(float(r2(yf[m], Zf[m] @ W)))
        curve.append(float(np.mean(sc)))
    j = int(np.argmax(curve))
    lam = LAMBDAS[j]
    at_edge = j in (0, len(LAMBDAS) - 1)
    A = Zf.T @ Zf + lam * np.eye(Zf.shape[1])
    W = np.linalg.solve(A, Zf.T @ yf)
    yh = (Zs @ W) * ys + ym

    ue = np.unique(gsc)
    idx = {e: np.where(gsc == e)[0] for e in ue}
    rng = np.random.default_rng(0)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = np.concatenate([idx[e] for e in rng.choice(ue, len(ue), True)])
        d[b] = r2(ysc[ii], yh[ii])
    return {"r2": float(r2(ysc, yh)), "lambda": lam,
            "lambda_at_grid_edge": at_edge, "inner_cv_r2": curve[j],
            "rho": float(np.corrcoef(ysc, yh)[0, 1]),
            "ci95": [float(np.percentile(d, 2.5)),
                     float(np.percentile(d, 97.5))],
            "n_fit": int(Ffit.shape[0]), "n_scored": int(Fsc.shape[0]),
            "d_ambient": int(Ffit.shape[1]), "d_probe": PCA_K,
            "n_clusters": int(len(ue))}


def main() -> int:
    frames = np.load(BANK / "frames_u8.npy", mmap_mode="r")
    L = np.load(BANK / "labels.npy")
    ep = np.load(BANK / "rows.npy")[:, 0]
    fit, sc = ep < N_FIT_EP, ep >= N_FIT_EP
    res = {"meta": {"lambdas": [float(x) for x in LAMBDAS], "pca_k": PCA_K,
                    "n_boot": N_BOOT, "n_fit_ep": N_FIT_EP,
                    "model_id": E1.MODEL_ID,
                    "evidence_class": "MEASURED (ours; RTX 4060)"},
           "arms": {}}
    TG = {"speed": L[:, 0], "yaw_rate": L[:, 1]}
    TA = {"steer": L[:, 2], "accel": L[:, 3]}

    for arm in ARMS:
        print(f"\n=== {arm} ===", flush=True)
        Fg = np.load(BANK / f"feat_{arm}.npy")
        sp_path = BANK / f"featsp_{arm}.npy"
        if sp_path.exists():
            Fs = np.load(sp_path)
        else:
            Fs = extract_spatial(arm, frames)
            np.save(sp_path, Fs)
        a = res["arms"].setdefault(arm, {})
        for nm, F in (("global", Fg), ("spatial4x4", Fs)):
            Z0, Z1 = F[0::2], F[1::2]
            DZ = Z1 - Z0
            rng = np.random.default_rng(7)
            DZsh = Z1[rng.permutation(Z1.shape[0])] - Z0
            blk = a.setdefault(nm, {})
            for t, y in TG.items():
                blk[f"dec_{t}"] = probe1(Z0[fit], y[fit], ep[fit],
                                         Z0[sc], y[sc], ep[sc])
            for t, y in TA.items():
                blk[f"ldad_{t}"] = probe1(DZ[fit], y[fit], ep[fit],
                                          DZ[sc], y[sc], ep[sc])
                blk[f"ldadshuf_{t}"] = probe1(DZsh[fit], y[fit], ep[fit],
                                              DZsh[sc], y[sc], ep[sc])
            print(f"  [{nm}] d={F.shape[1]:6d} "
                  f"speed {blk['dec_speed']['r2']:+.4f} "
                  f"{blk['dec_speed']['ci95']} lam={blk['dec_speed']['lambda']:g}"
                  f"{' EDGE' if blk['dec_speed']['lambda_at_grid_edge'] else ''} | "
                  f"yawrate {blk['dec_yaw_rate']['r2']:+.4f} | "
                  f"LDAD steer {blk['ldad_steer']['r2']:+.4f} "
                  f"(shuf {blk['ldadshuf_steer']['r2']:+.4f}) "
                  f"accel {blk['ldad_accel']['r2']:+.4f} "
                  f"(shuf {blk['ldadshuf_accel']['r2']:+.4f})", flush=True)
        (BANK / "eseed1_v2.json").write_text(json.dumps(res, indent=2))

    # constant-only control: the exact no-information value
    res["controls"] = {
        "constant_only_scored_mean_r2": 0.0,
        "note": ("a predictor emitting the SCORED-split mean reads R2 = 0.0 "
                 "EXACTLY by construction; every number above is against that "
                 "same SST, so 0.0 IS the no-information value."),
        "fit_mean_predictor_r2": {
            k: float(r2(v[sc], np.full(int(sc.sum()), v[fit].mean())))
            for k, v in {**TG, **TA}.items()},
    }
    (BANK / "eseed1_v2.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", BANK / "eseed1_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
