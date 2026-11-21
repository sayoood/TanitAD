"""E-SEED-2 — does the DINOv3 seed carry SCENE content, not just ego?

⛔ THE QUESTION E-SEED-1 COULD NOT ANSWER. E-SEED-1 scored `speed` and
`yaw_rate`. Both are EGO. E-DEC-17 MEASURED that a frozen RANDOM encoder carries
the BEST speed of three arms (+0.3552) and concluded, verbatim, that `ego
decodability does not evidence a learned representation` and `no ego number may
be cited as evidence that an objective worked`. So E-SEED-1's verdict -- however
large -- is a statement about the transfer's FIDELITY, not about scene content.

`n_agents` is the register's environment target (E-DEC-8/9/12/14), the one on
which every self-supervised arm of ours sat below a constant predictor while
frozen DINOv3 read +0.2754. THIS panel puts it on the seed.

ARMS (one variable each, all frozen, no training, one forward pass):
  dino_hf          published DINOv3 ViT-L/16, ImageNet-normalised  (reference)
  seed_asis        the seeded ViTEncoder EXACTLY as the trainer feeds it
                   (`.float()/255`, `pos` at its own random init)  <- v7f's trunk
  seed_imnet       the same seed + ImageNet normalisation          (the repair)
  seed_imnet_pos0  the same + `pos` zeroed                         (mechanism)
  scratch          random-init ViTEncoder      <- DELIBERATE-REGRESSION ARM
  pixel            4x4 pooled raw RGB          <- raw-input floor
  (constant-only)  reads R2 = 0.0 EXACTLY by construction

ESTIMATOR. 4-fold CLIP-DISJOINT out-of-fold prediction over the 24 clips, so
every row is scored by a model that never saw its clip; pooled R2 against the
scored-set mean; clip-cluster bootstrap (2,000 draws) over the 24 clips, and the
PAIRED bootstrap for arm contrasts. PCA basis and ridge lambda are fitted INSIDE
each outer fold's training clips only (lambda by an inner 4-fold GroupKFold over
those clips). Nothing is selected on a row it scores.

TIER: none. This is a representation/decodability diagnostic, never a driving
number.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BANK = HERE / "eseed2"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "clone_scripts"))

DINO_SNAP = Path(
    r"C:\Users\Admin\.cache\huggingface\hub"
    r"\models--facebook--dinov3-vitl16-pretrain-lvd1689m\snapshots"
    r"\ea8dc2863c51be0a264bab82070e3e8836b02d51")
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

DEV = "cuda"
DTYPE = torch.bfloat16
BATCH = 4
PCA_K = 256
N_BOOT = 2000
K_OUTER = 4
K_INNER = 4
LAMBDAS = tuple(10.0 ** k for k in range(-4, 9))
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GRID = (16, 40)              # 256x640 / patch 16
POOL = (4, 10)               # -> a 4x4 cell grid, matching E-SEED-1's read
ARMS = ["dino_hf", "seed_asis", "seed_imnet", "seed_imnet_pos0",
        "scratch", "pixel"]


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
def make_enc_cfg():
    from tanitad.config import EncoderConfig
    return EncoderConfig(in_channels=3, image_size=256, image_width=640,
                         patch_size=16, d_model=1024, depth=24, n_heads=16)


def build_seeded(zero_pos: bool):
    import dinov3_seed_checkpoint as S
    from tanitad.models.encoder import ViTEncoder
    cfg = make_enc_cfg()
    state, hfcfg, _prov = S.load_source(DINO_SNAP)
    state, _pref = S.strip_prefix(state)
    geo = S.infer_source_geometry(state, hfcfg)
    S.assert_target_geometry(geo, cfg)
    seed, report = S.build_seed(state, cfg, src_geo=geo)
    enc = ViTEncoder(cfg)
    res = enc.load_state_dict(seed, strict=False)
    assert set(res.missing_keys) == {"pos"}, sorted(res.missing_keys)
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
    return DINOv3ViTModel.from_pretrained(
        MODEL_ID, dtype=DTYPE, local_files_only=True).eval()


@torch.no_grad()
def extract(arm: str, frames) -> tuple[np.ndarray, np.ndarray]:
    """Return (global [n,1024], spatial [n,16*1024])."""
    n = frames.shape[0]
    gh, gw = GRID
    if arm == "pixel":
        G = np.empty((n, 3), dtype=np.float32)
        S_ = np.empty((n, 4 * 4 * 3), dtype=np.float32)
        for i in range(0, n, 64):
            x = torch.from_numpy(np.array(frames[i:i + 64])).float() / 255.
            G[i:i + 64] = x.mean(dim=(2, 3)).numpy()
            p = torch.nn.functional.avg_pool2d(x, (64, 160))
            S_[i:i + 64] = p.reshape(p.shape[0], -1).numpy()
        return G, S_
    imnet = arm in ("dino_hf", "seed_imnet", "seed_imnet_pos0")
    if arm == "dino_hf":
        model = build_hf().to(DEV)
    elif arm == "scratch":
        model = build_scratch().to(DEV).to(DTYPE)
    else:
        model, _ = build_seeded(zero_pos=(arm == "seed_imnet_pos0"))
        model = model.to(DEV).to(DTYPE)
    mean, std = IMNET_MEAN.to(DEV), IMNET_STD.to(DEV)
    G = np.empty((n, 1024), dtype=np.float32)
    S_ = np.empty((n, 16 * 1024), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, BATCH):
        x = torch.from_numpy(np.array(frames[i:i + BATCH])).to(DEV).float() / 255.
        if imnet:
            x = (x - mean) / std
        x = x.to(DTYPE)
        tok = (model(pixel_values=x).last_hidden_state[:, -(gh * gw):]
               if arm == "dino_hf" else model(x))
        assert tok.shape[1] == gh * gw, (arm, tuple(tok.shape))
        b, _t, d = tok.shape
        G[i:i + BATCH] = tok.float().mean(1).cpu().numpy()
        g = tok.float().reshape(b, gh, gw, d).permute(0, 3, 1, 2)
        g = torch.nn.functional.avg_pool2d(g, POOL)          # [b,d,4,4]
        S_[i:i + BATCH] = g.reshape(b, -1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    # ⛔ CONTENT ASSERTIONS, never the exit code.
    assert np.isfinite(G).all() and np.isfinite(S_).all()
    assert float(G.std()) > 0 and float(S_.std()) > 0
    print(f"  [{arm}] global d=1024 std={G.std():.4f} | spatial "
          f"d={S_.shape[1]} std={S_.std():.4f}  {time.time()-t0:.0f}s",
          flush=True)
    return G, S_


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def pca_basis(Xf: np.ndarray, k: int):
    """Exact PCA via the Gram trick (n < d here, so this is exact and fast)."""
    mu = Xf.mean(0, keepdims=True)
    Xc = (Xf - mu).astype(np.float64)
    k = min(k, Xc.shape[0] - 1, Xc.shape[1])
    Gm = Xc @ Xc.T
    w, U = np.linalg.eigh(Gm)
    idx = np.argsort(w)[::-1][:k]
    w, U = np.clip(w[idx], 1e-12, None), U[:, idx]
    V = Xc.T @ (U / np.sqrt(w))                 # [d, k], orthonormal columns
    return mu, V


def _r2(y, yh):
    return float(1.0 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ridge_w(Z, y, lam):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y)


def oof_predict(F, y, clip, k_outer=K_OUTER, seed=0):
    """Out-of-fold predictions; PCA + lambda fitted inside each fold's train."""
    clips = np.unique(clip)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(clips), k_outer)
    pred = np.full(len(y), np.nan)
    lams, edges = [], []
    for te in folds:
        m_te = np.isin(clip, te)
        m_tr = ~m_te
        mu, V = pca_basis(F[m_tr], PCA_K)
        Ztr = ((F[m_tr] - mu) @ V)
        Zte = ((F[m_te] - mu) @ V)
        sd = Ztr.std(0, keepdims=True) + 1e-8
        Ztr, Zte = Ztr / sd, Zte / sd
        ym, ys = y[m_tr].mean(), y[m_tr].std() + 1e-8
        ytr = (y[m_tr] - ym) / ys
        # inner lambda selection, clip-disjoint, on TRAIN clips only
        tr_clips = np.unique(clip[m_tr])
        inner = np.array_split(rng.permutation(tr_clips), K_INNER)
        ctr = clip[m_tr]
        curve = []
        for lam in LAMBDAS:
            sc = []
            for f in inner:
                mi = np.isin(ctr, f)
                W = ridge_w(Ztr[~mi], ytr[~mi], lam)
                sc.append(_r2(ytr[mi], Ztr[mi] @ W))
            curve.append(float(np.mean(sc)))
        j = int(np.argmax(curve))
        lams.append(LAMBDAS[j])
        edges.append(j in (0, len(LAMBDAS) - 1))
        W = ridge_w(Ztr, ytr, LAMBDAS[j])
        pred[m_te] = (Zte @ W) * ys + ym
    assert np.isfinite(pred).all()
    return pred, lams, edges


def boot_ci(y, pred, clip, n_boot=N_BOOT, seed=0):
    cl = np.unique(clip)
    idx = {c: np.where(clip == c)[0] for c in cl}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.concatenate([idx[c] for c in rng.choice(cl, len(cl), True)])
        d[b] = _r2(y[ii], pred[ii])
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def paired_ci(y, pa, pb, clip, n_boot=N_BOOT, seed=0):
    cl = np.unique(clip)
    idx = {c: np.where(clip == c)[0] for c in cl}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.concatenate([idx[c] for c in rng.choice(cl, len(cl), True)])
        d[b] = _r2(y[ii], pa[ii]) - _r2(y[ii], pb[ii])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta": _r2(y, pa) - _r2(y, pb), "ci95": [lo, hi],
            "separated": bool(lo > 0 or hi < 0)}


def main() -> int:
    frames = np.load(BANK / "frames_u8.npy", mmap_mode="r")
    L = np.load(BANK / "labels.npy")
    clip = np.load(BANK / "rows.npy")[:, 0]
    meta = json.loads((BANK / "bank_meta.json").read_text(encoding="utf-8"))
    targets = {"n_agents": L[:, 0], "speed": L[:, 1], "yaw_rate": L[:, 2]}

    res = {"meta": {**meta, "arms": ARMS, "pca_k": PCA_K, "n_boot": N_BOOT,
                    "k_outer": K_OUTER, "k_inner": K_INNER,
                    "lambdas": [float(x) for x in LAMBDAS],
                    "pool": list(POOL), "grid": list(GRID),
                    "model_id": MODEL_ID,
                    "estimator": ("4-fold clip-disjoint out-of-fold prediction; "
                                  "pooled R2; clip-cluster bootstrap over 24 "
                                  "clips, 2000 draws; paired bootstrap for "
                                  "contrasts"),
                    "evidence_class": "MEASURED (ours; dev-box RTX 4060)"},
           "arms": {}, "preds": {}}

    for arm in ARMS:
        gp, sp = BANK / f"feat_{arm}.npy", BANK / f"featsp_{arm}.npy"
        if gp.exists() and sp.exists():
            G, S_ = np.load(gp), np.load(sp)
            print(f"=== {arm} (cached)", flush=True)
        else:
            print(f"=== {arm}", flush=True)
            G, S_ = extract(arm, frames)
            np.save(gp, G)
            np.save(sp, S_)
        blk = res["arms"].setdefault(arm, {})
        for space, F in (("global", G), ("spatial4x4", S_)):
            sb = blk.setdefault(space, {})
            for tname, y in targets.items():
                y = y.astype(np.float64)
                pred, lams, edges = oof_predict(F, y, clip)
                r2v = _r2(y, pred)
                sb[tname] = {
                    "r2": r2v, "ci95": boot_ci(y, pred, clip),
                    "rho": float(np.corrcoef(y, pred)[0, 1]),
                    "lambdas": [float(x) for x in lams],
                    "lambda_at_grid_edge": bool(any(edges)),
                    "n": int(len(y)), "d_ambient": int(F.shape[1]),
                    "d_probe": PCA_K, "n_clusters": int(len(np.unique(clip)))}
                res["preds"].setdefault(f"{arm}::{space}::{tname}",
                                        pred.tolist())
            print(f"  [{space}] " + " | ".join(
                f"{t} {sb[t]['r2']:+.4f} [{sb[t]['ci95'][0]:+.3f},"
                f"{sb[t]['ci95'][1]:+.3f}]" for t in targets), flush=True)
        (BANK / "eseed2_panel.json").write_text(
            json.dumps({k: v for k, v in res.items() if k != "preds"},
                       indent=2), encoding="utf-8")

    # ---- contrasts that decide the question -------------------------------
    P = res["preds"]
    contrasts = {}
    pairs = [("seed_asis", "scratch", "is the seed AS WIRED better than random?"),
             ("seed_imnet", "scratch", "is the REPAIRED seed better than random?"),
             ("seed_imnet", "seed_asis", "does input normalisation help?"),
             ("dino_hf", "seed_imnet", "how much does the repaired seed lose?"),
             ("dino_hf", "scratch", "INSTRUMENT VALIDITY: can the probe see it?"),
             ("dino_hf", "pixel", "does DINOv3 beat the raw-input floor?"),
             ("seed_imnet", "pixel", "does the repaired seed beat raw input?"),
             ("seed_asis", "pixel", "does the seed as wired beat raw input?"),
             ("seed_imnet", "seed_imnet_pos0", "does the random `pos` help?")]
    for space in ("global", "spatial4x4"):
        for tname, y in targets.items():
            y = y.astype(np.float64)
            for a, b, q in pairs:
                pa = np.asarray(P[f"{a}::{space}::{tname}"])
                pb = np.asarray(P[f"{b}::{space}::{tname}"])
                contrasts[f"{space}::{tname}::{a}-{b}"] = {
                    "question": q, **paired_ci(y, pa, pb, clip)}
    res["contrasts"] = contrasts
    res["controls"] = {
        "constant_only_r2": 0.0,
        "note": ("a predictor emitting the scored-set mean reads R2 = 0.0 "
                 "EXACTLY; every R2 above is against that same SST, so 0.0 IS "
                 "the no-information value"),
        "deliberate_regression_arm": "scratch (random-init ViTEncoder)",
        "raw_input_floor_arm": "pixel"}
    out = {k: v for k, v in res.items() if k != "preds"}
    (BANK / "eseed2_panel.json").write_text(json.dumps(out, indent=2),
                                            encoding="utf-8")
    np.savez_compressed(BANK / "eseed2_preds.npz",
                        **{k: np.asarray(v) for k, v in P.items()},
                        clip=clip, labels=L)
    print("\nwrote", BANK / "eseed2_panel.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
