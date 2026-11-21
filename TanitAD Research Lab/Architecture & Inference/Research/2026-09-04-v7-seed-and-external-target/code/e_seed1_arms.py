"""E-SEED-1 stage 2 — do the DINOv3 seed's TENSORS carry DINOv3's FUNCTION?

⛔ THE QUESTION. `stack/scripts/dinov3_seed_checkpoint.py` copies 292/292 DINOv3
block tensors into `ViTEncoder` and DECLARES two losses (positional table left at
random init because DINOv3 is RoPE-only; CLS/register tokens dropped). A THIRD is
undeclared and was found by source read 2026-09-03: the v6 encoder path feeds
`.float()/255` -- inputs in [0,1] -- while DINOv3's weights were trained on
IMAGENET-NORMALISED inputs (the normalisation exists in this repo ONLY inside
`O7Distill.target()`, train_v6_staged.py:970-972, which feeds the *teacher*).

Every one of those is a claim about TENSORS. None is a claim about the FUNCTION.
This panel measures the function.

ARMS (all see the identical frames; all bf16; all mean-pooled over patch tokens)
  dino_hf          real HF DINOv3ViTModel, ImageNet-normalised   <- the REFERENCE
  seed_asis        ViTEncoder+seed, input [0,1], pos at random init  <- WHAT THE
                                                     TRAINER WOULD ACTUALLY RUN
  seed_imnet       ViTEncoder+seed, ImageNet-normalised, random pos
  seed_imnet_pos0  ViTEncoder+seed, ImageNet-normalised, pos ZEROED
  scratch          ViTEncoder fresh random init, input [0,1]   <- DELIBERATE
                                                     REGRESSION / floor arm
  pixel            raw pixels avg-pooled to 16x16x3            <- RAW-INPUT FLOOR

CONTROLS (TanitAD_ValidateAIDesign S4)
  * constant-only  -> must read the no-information value EXACTLY (R2 = 0.0 for
                     the scored-split-mean predictor, by construction)
  * raw-input floor-> the `pixel` arm; a learned representation below it added
                     nothing
  * endpoint-shuffled Delta-z (H-LEAK-3) on every LDAD panel
  * n and d PRINTED; PCA basis and ridge lambda fitted on the FIT split ONLY,
    lambda chosen by GroupKFold over EPISODES inside FIT -- never on the scored
    split (the 2026-08-22 failure class)
  * INSTRUMENT VALIDITY: if `dino_hf` does not beat `scratch`, the panel is VOID.

ESTIMATOR: episode-cluster bootstrap (2000 draws) over the 30 SCORED episodes.
TIER: not a driving number. This is a representation/decodability diagnostic.
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
OUT = BANK
SEEDSCRIPTS = HERE / "clone_scripts"
sys.path.insert(0, str(SEEDSCRIPTS))

DINO_SNAP = Path(
    r"C:\Users\Admin\.cache\huggingface\hub"
    r"\models--facebook--dinov3-vitl16-pretrain-lvd1689m\snapshots"
    r"\ea8dc2863c51be0a264bab82070e3e8836b02d51")
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

DEV = "cuda"
DT = torch.bfloat16
BATCH = 16
N_FIT_EP = 60            # episodes 0..59 -> FIT (PCA + lambda live here)
N_BOOT = 2000
PCA_K = 256
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
def make_enc_cfg():
    from tanitad.config import EncoderConfig
    return EncoderConfig(in_channels=3, image_size=256, image_width=None,
                         patch_size=16, d_model=1024, depth=24, n_heads=16)


def build_seeded(zero_pos: bool):
    """ViTEncoder with the DINOv3 seed applied. Asserts the seed really landed."""
    import dinov3_seed_checkpoint as S
    from tanitad.models.encoder import ViTEncoder
    cfg = make_enc_cfg()
    state, hfcfg, prov = S.load_source(DINO_SNAP)
    state, _pref = S.strip_prefix(state)
    geo = S.infer_source_geometry(state, hfcfg)
    S.assert_target_geometry(geo, cfg)
    seed, report = S.build_seed(state, cfg, src_geo=geo)
    enc = ViTEncoder(cfg)
    res = enc.load_state_dict(seed, strict=False)
    missing = set(res.missing_keys)
    # ⛔ CONTENT ASSERTION: the only thing allowed to stay at init is `pos`.
    assert missing == {"pos"}, f"unexpected missing keys: {sorted(missing)}"
    assert not res.unexpected_keys, res.unexpected_keys
    if zero_pos:
        with torch.no_grad():
            enc.pos.zero_()
    return enc.eval(), report


def build_scratch():
    from tanitad.models.encoder import ViTEncoder
    return ViTEncoder(make_enc_cfg()).eval()


def build_hf():
    import truststore
    truststore.inject_into_ssl()
    from transformers import DINOv3ViTModel
    m = DINOv3ViTModel.from_pretrained(MODEL_ID, dtype=DT,
                                       local_files_only=True).eval()
    return m


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(arm: str, frames: np.ndarray) -> np.ndarray:
    n = frames.shape[0]
    if arm == "pixel":
        # raw-input floor: 16x16 average pool of the 3-channel frame
        out = np.empty((n, 16 * 16 * 3), dtype=np.float32)
        for i in range(0, n, 256):
            x = torch.from_numpy(np.asarray(frames[i:i + 256])).float() / 255.0
            p = torch.nn.functional.avg_pool2d(x, 16)          # [b,3,16,16]
            out[i:i + 256] = p.reshape(p.shape[0], -1).numpy()
        return out

    imnet = arm in ("dino_hf", "seed_imnet", "seed_imnet_pos0")
    if arm == "dino_hf":
        model = build_hf().to(DEV)
        rep = {"hf": MODEL_ID}
    elif arm == "scratch":
        model = build_scratch().to(DEV).to(DT)
        rep = {"init": "random"}
    else:
        model, rep = build_seeded(zero_pos=(arm == "seed_imnet_pos0"))
        model = model.to(DEV).to(DT)

    mean, std = IMNET_MEAN.to(DEV), IMNET_STD.to(DEV)
    feats = None
    t0 = time.time()
    for i in range(0, n, BATCH):
        x = torch.from_numpy(np.asarray(frames[i:i + BATCH])).to(DEV).float() / 255.0
        if imnet:
            x = (x - mean) / std
        x = x.to(DT)
        if arm == "dino_hf":
            tok = model(pixel_values=x).last_hidden_state[:, -256:]
        else:
            tok = model(x)
        z = tok.float().mean(dim=1)                              # [b, d]
        if feats is None:
            feats = np.empty((n, z.shape[1]), dtype=np.float32)
        feats[i:i + BATCH] = z.cpu().numpy()
        if i % (BATCH * 60) == 0:
            print(f"    {arm} {i}/{n} {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    assert np.isfinite(feats).all(), f"{arm}: non-finite features"
    assert float(feats.std()) > 0, f"{arm}: constant features"
    print(f"  [{arm}] d={feats.shape[1]} mean={feats.mean():+.4f} "
          f"std={feats.std():.4f} {time.time()-t0:.0f}s  {rep if len(str(rep))<120 else ''}")
    return feats


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def participation(F: np.ndarray) -> float:
    """participation ratio on the SQUARED spectrum (sigma^2) -- C132: never
    effective_rank(sigma)."""
    X = F - F.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    ev = (s ** 2)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def fit_pca(Ffit: np.ndarray, k: int):
    mu = Ffit.mean(0, keepdims=True)
    X = Ffit - mu
    # economical: eigendecomposition of the covariance
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return mu, Vt[:k].T


def ridge_fit(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float64)
    return np.linalg.solve(A, X.T @ Y)


def r2(y, yh):
    ss = ((y - yh) ** 2).sum(0)
    st = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return 1.0 - ss / st


def probe(Ffit, Yfit, gfit, Fsc, Ysc, gsc, tag):
    """PCA + ridge; lambda by GroupKFold over EPISODES inside FIT only."""
    mu, V = fit_pca(Ffit, PCA_K)
    Zf = (Ffit - mu) @ V
    Zs = (Fsc - mu) @ V
    sd = Zf.std(0, keepdims=True) + 1e-8
    Zf, Zs = Zf / sd, Zs / sd
    ymu, ysd = Yfit.mean(0, keepdims=True), Yfit.std(0, keepdims=True) + 1e-8

    # --- lambda selection: 5 folds over FIT episodes, never on the scored set
    eps = np.unique(gfit)
    folds = np.array_split(eps, 5)
    best, best_lam = -np.inf, None
    for lam in LAMBDAS:
        sc = []
        for f in folds:
            m = np.isin(gfit, f)
            W = ridge_fit(Zf[~m].astype(np.float64),
                          ((Yfit[~m] - ymu) / ysd).astype(np.float64), lam)
            sc.append(r2((Yfit[m] - ymu) / ysd, Zf[m].astype(np.float64) @ W).mean())
        s = float(np.mean(sc))
        if s > best:
            best, best_lam = s, lam
    W = ridge_fit(Zf.astype(np.float64), ((Yfit - ymu) / ysd).astype(np.float64),
                  best_lam)
    Yh = (Zs.astype(np.float64) @ W) * ysd + ymu

    out = {"lambda": best_lam, "inner_cv_r2": best,
           "n_fit": int(Ffit.shape[0]), "n_scored": int(Fsc.shape[0]),
           "d_ambient": int(Ffit.shape[1]), "d_probe": PCA_K}
    out["r2"] = [float(x) for x in r2(Ysc, Yh)]
    out["rho"] = [float(np.corrcoef(Ysc[:, j], Yh[:, j])[0, 1])
                  for j in range(Ysc.shape[1])]
    # episode-cluster bootstrap over the SCORED episodes
    ue = np.unique(gsc)
    rng = np.random.default_rng(0)
    draws = np.empty((N_BOOT, Ysc.shape[1]), dtype=np.float64)
    idx_by_ep = {e: np.where(gsc == e)[0] for e in ue}
    for b in range(N_BOOT):
        pick = rng.choice(ue, size=len(ue), replace=True)
        ii = np.concatenate([idx_by_ep[e] for e in pick])
        draws[b] = r2(Ysc[ii], Yh[ii])
    out["r2_ci95"] = [[float(np.percentile(draws[:, j], 2.5)),
                       float(np.percentile(draws[:, j], 97.5))]
                      for j in range(Ysc.shape[1])]
    out["tag"] = tag
    return out


def main() -> int:
    frames = np.load(BANK / "frames_u8.npy", mmap_mode="r")
    L = np.load(BANK / "labels.npy")
    R = np.load(BANK / "rows.npy")
    ep = R[:, 0]
    n = L.shape[0]
    fit_m = ep < N_FIT_EP
    sc_m = ~fit_m
    print(f"[split] episodes fit={len(np.unique(ep[fit_m]))} "
          f"scored={len(np.unique(ep[sc_m]))} | rows fit={fit_m.sum()} "
          f"scored={sc_m.sum()}  (episode-disjoint by construction)")

    arms = ["dino_hf", "seed_asis", "seed_imnet", "seed_imnet_pos0",
            "scratch", "pixel"]
    res = {"arms": {}, "meta": {
        "n_pairs": int(n), "n_fit_ep": N_FIT_EP, "pca_k": PCA_K,
        "lambdas": list(LAMBDAS), "n_boot": N_BOOT, "dtype": str(DT),
        "model_id": MODEL_ID,
        "evidence_class": "MEASURED (ours; RTX 4060; local val epcache)",
    }}

    for arm in arms:
        print(f"\n=== ARM {arm} ===", flush=True)
        fpath = BANK / f"feat_{arm}.npy"
        if fpath.exists():
            F = np.load(fpath)
            print(f"  [cached] {F.shape}")
        else:
            F = extract(arm, frames)
            np.save(fpath, F)
        # pair layout: frame 2i = t, frame 2i+1 = t+1
        Z0, Z1 = F[0::2], F[1::2]
        DZ = Z1 - Z0
        a = res["arms"].setdefault(arm, {})
        # n>d required: computed over ALL pairs (n=2971) at d=1024, matched
        # across every ViT arm (same corpus, same episodes, same n, same d).
        a["participation_sigma2"] = participation(Z0)
        a["d"] = int(F.shape[1])
        a["n_participation"] = int(Z0.shape[0])

        # --- P-DEC: decodability of the DYNAMIC ego state from z_t
        Ydec = np.stack([L[:, 0], L[:, 1]], 1)         # v_mps, yaw_rate
        a["dec"] = probe(Z0[fit_m], Ydec[fit_m], ep[fit_m],
                         Z0[sc_m], Ydec[sc_m], ep[sc_m], "speed,yaw_rate")
        # --- P-LDAD (prereg R0): (steer, accel) from Delta z
        Yact = np.stack([L[:, 2], L[:, 3]], 1)         # steer, accel
        a["ldad"] = probe(DZ[fit_m], Yact[fit_m], ep[fit_m],
                          DZ[sc_m], Yact[sc_m], ep[sc_m], "steer,accel|dz")
        # --- endpoint-shuffled control (H-LEAK-3): dz' = z_{pi(t)+1} - z_t
        rng = np.random.default_rng(7)
        perm = rng.permutation(n)
        DZs = Z1[perm] - Z0
        a["ldad_endpoint_shuffled"] = probe(
            DZs[fit_m], Yact[fit_m], ep[fit_m],
            DZs[sc_m], Yact[sc_m], ep[sc_m], "steer,accel|dz_shuffled")
        print(f"  participation(sigma^2) = {a['participation_sigma2']:.3f} "
              f"(n={a['n_participation']}, d={a['d']})")
        print(f"  DEC  R2 speed={a['dec']['r2'][0]:+.4f} "
              f"yawrate={a['dec']['r2'][1]:+.4f}  lam={a['dec']['lambda']}")
        print(f"  LDAD R2 steer={a['ldad']['r2'][0]:+.4f} "
              f"accel={a['ldad']['r2'][1]:+.4f}  "
              f"(shuffled {a['ldad_endpoint_shuffled']['r2'][0]:+.4f} / "
              f"{a['ldad_endpoint_shuffled']['r2'][1]:+.4f})")
        (OUT / "eseed1_results.json").write_text(json.dumps(res, indent=2))

    # --- constant-only control: the no-information value, exactly -----------
    Ydec = np.stack([L[:, 0], L[:, 1]], 1)
    Yact = np.stack([L[:, 2], L[:, 3]], 1)
    res["controls"] = {
        "constant_only_scored_mean_r2": [0.0, 0.0],
        "constant_only_fitmean_r2_dec":
            [float(x) for x in r2(Ydec[sc_m],
                                  np.tile(Ydec[fit_m].mean(0), (sc_m.sum(), 1)))],
        "constant_only_fitmean_r2_ldad":
            [float(x) for x in r2(Yact[sc_m],
                                  np.tile(Yact[fit_m].mean(0), (sc_m.sum(), 1)))],
        "note": ("a predictor emitting the SCORED-split mean reads R2 = 0.0 "
                 "EXACTLY by construction; the FIT-mean predictor is the "
                 "realisable constant control and reads <= 0."),
    }
    (OUT / "eseed1_results.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT / "eseed1_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
