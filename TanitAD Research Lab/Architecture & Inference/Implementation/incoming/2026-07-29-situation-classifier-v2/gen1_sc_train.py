"""Situation classifier — STEP 3 (pod3, GPU): the head, the CV, every arm.

Implements `PRE_REGISTRATION.md` Sec 5. **The held-out side is never read here**: this script emits
no held-out metric at all, only raw scores for `sc_eval.py`.

⭐ The architecture lesson, designed in rather than re-learnt: H2 MEASURED that concatenating the
raw 2048-d frozen state onto a working ego head **destroyed** it (3.74x -> 1.59x base, separated ->
not separated) — a capacity/swamping signature. So vision enters **LOW-RANK** here (a PCA basis fit
on TRAIN rows only), and the raw-concatenation arm is kept as an explicit ABLATION so the claim is
re-tested rather than inherited.

Arms (PRE-REG Sec 5.2):
    head_img_ego         PCA_r(state) + ego            <- PRIMARY
    head_img             PCA_r(state)
    head_ego             ego only                      <- baseline (d)
    head_img_ego_concat  raw 2048-d + ego              <- the swamping ABLATION
    head_priv            the privileged construction summary  <- C-POS, the POWER CEILING
    head_img_shuf        PCA_r(state) permuted ACROSS CLIPS   <- C-NEG, the noise floor

usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack python3 sc_train.py \
      --bundle /workspace/sitclf/bundle --feats /workspace/sitclf/feats \
      --out /workspace/sitclf/run --epochs 15
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

SITS = ("lane_change", "roundabout", "intersection")
WIN = 8                       # 0.8 s causal window, PRE-REG Sec 5.2
EGO_SCALE = np.array([10.0, 2.0, 0.5], dtype=np.float32)     # v, alon_pre, omega_pre
# ⭐ AMENDMENT A1 (see SITUATION_CLASSIFIER.md): rank 16 is PRIMARY, not 64. The sibling
# situation-semantics stream MEASURED a monotone swamping dose-response on the frozen v1 state —
# ego alone 3.659x -> +k16 3.685x -> +k64 3.000x -> +k256 2.116x -> +k2048 1.59x, with 16 PCs
# carrying 97.0 % of the state variance (INHERITED, `2026-07-26-situation-semantics/`). Degradation
# begins at k=64. The r in {16, 64} ladder plus the raw-2048 concat arm reproduces that curve on
# THIS target, which is a free independent replication.
CONFIGS = [dict(pw=20, d=128, r=16), dict(pw=50, d=128, r=16), dict(pw=20, d=128, r=64)]
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
ARMS = ("head_img_ego", "head_img", "head_ego", "head_img_ego_concat",
        "head_priv", "head_img_shuf")
# ⭐ AMENDMENT A1, second half: the LOW-CAPACITY end of the ladder. The same sibling stream MEASURED
# a 2,049-parameter linear RIDGE probe SEPARATING on `NOT_T_seen` (+0.01592 [+0.00737, +0.04746])
# where H2's 2.17 M-parameter head did NOT (+0.00601, not separated) — i.e. on this representation
# the HEAD has been the bottleneck, not the features. These arms are closed-form (no optimiser can
# be blamed for a null) and read the window FLAT — the sibling also measured that temporally pooling
# the window before reading it costs ~3.8x of the effect, so nothing is averaged here.
RIDGE_ARMS = ("ridge_img_ego", "ridge_img", "ridge_ego", "ridge_img_shuf")


# ------------------------------------------------------------------------------------ the head
class SitHead(nn.Module):
    def __init__(self, in_dim, d=128, n_out=3, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(WIN, d))
        layer = nn.TransformerEncoderLayer(d, 4, d * 4, dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2)
        self.att = nn.Linear(d, 1)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, x):                       # x [B, WIN, in_dim]
        h = self.enc(self.inp(x) + self.pos)
        a = torch.softmax(self.att(h), 1)
        return self.mlp((h * a).sum(1))         # [B, 3] independent Bernoulli logits


# ------------------------------------------------------------------------------------ substrate
def load_substrate(bundle, feats_dir, device):
    meta = json.load(open(os.path.join(bundle, "sc_meta.json")))
    Z = np.load(os.path.join(bundle, "sc_labels.npz"))
    rows, F, E, P, Y, V, CK = [], [], [], [], [], [], []
    off = 0
    for m in meta:
        p = os.path.join(feats_dir, f"clip_{m['k']:05d}.npy")
        if not os.path.exists(p):
            continue
        # NB: NOT `mmap_mode="r"` — one open handle per clip blows the file-descriptor limit at
        # this n (OSError 24 at ~1,300 clips) and the arrays are concatenated into RAM anyway.
        f = np.load(p)
        T = f.shape[0]
        assert T == m["T"], f"C-FID: clip {m['k']} T {T} != {m['T']}"
        F.append(f)
        E.append(Z[f"c{m['k']}_ego"] / EGO_SCALE)
        P.append(Z[f"c{m['k']}_priv"])
        Y.append(np.stack([Z[f"c{m['k']}_y_{s}"] for s in SITS], 1))
        V.append(np.stack([Z[f"c{m['k']}_valid_{s}"] for s in SITS], 1))
        rows.append(dict(k=m["k"], side=m["side"], chunk=m["chunk"], off=off, T=T))
        CK.append(np.full(T, len(rows) - 1, np.int32))
        off += T
    S = dict(F=np.concatenate(F), E=np.concatenate(E).astype(np.float32),
             P=np.concatenate(P).astype(np.float32), Y=np.concatenate(Y),
             V=np.concatenate(V), clip=np.concatenate(CK), rows=rows)
    # a window needs WIN-1 causal steps INSIDE the same clip
    ok = np.zeros(len(S["clip"]), bool)
    for r in rows:
        ok[r["off"] + WIN - 1:r["off"] + r["T"]] = True
    S["win_ok"] = ok
    print(f"[data] {len(rows)} clips, {len(S['clip']):,} frames, feat dim {S['F'].shape[1]}",
          flush=True)
    return S


def fit_pca(X, r, n_sample=250_000, seed=0):
    """PCA basis on TRAIN rows ONLY (mean + top-r right singular vectors)."""
    g = np.random.default_rng(seed)
    idx = g.choice(len(X), size=min(n_sample, len(X)), replace=False)
    A = torch.from_numpy(np.asarray(X[np.sort(idx)], dtype=np.float32))
    mu = A.mean(0, keepdim=True)
    A = A - mu
    _u, _s, v = torch.svd_lowrank(A, q=min(r + 16, A.shape[1] - 1), niter=4)
    return mu.numpy(), v[:, :r].numpy().astype(np.float32)


def arm_features(S, arm, mu, W, shuf_perm=None):
    """-> a [n_frames, dim] float32 matrix; windows are gathered from it at batch time."""
    if arm == "head_ego":
        return S["E"]
    if arm == "head_priv":
        return S["P"]
    img = ((np.asarray(S["F"], dtype=np.float32) - mu) @ W)
    img /= max(float(np.abs(img).mean()), 1e-6)
    if arm == "head_img_ego_concat":
        # in-place standardisation: the naive form allocates three 2048-d float32 copies of the
        # whole corpus (~12 GB at this n) inside a 50 GB cgroup
        raw = np.asarray(S["F"], dtype=np.float32)
        raw -= raw.mean(0, keepdims=True)
        raw /= np.maximum(raw.std(0, keepdims=True), 1e-3)
        return np.concatenate([raw, S["E"]], 1)
    if arm == "head_img":
        return img
    if arm == "head_img_shuf":
        return img[shuf_perm]
    return np.concatenate([img, S["E"]], 1)


def to_device_bank(X, device, max_gb=8.0):
    """Keep the whole feature bank on the GPU when it fits and gather windows THERE.

    Gathering `[B, WIN, dim]` windows on the CPU and copying them per batch was the bottleneck
    (MEASURED in the smoke run: it projected the full grid to ~9 h). The bank is 36 MB at r=16 and
    3.9 GB for the raw-2048 concat arm, both of which fit on a 46 GB A40."""
    gb = X.nbytes / 1e9
    if gb <= max_gb and device != "cpu":
        return torch.from_numpy(X).to(device), True
    return torch.from_numpy(X), False


def run_fold(X, S, tr_idx, te_idx, cfg, epochs, device, seed=0, bank=None, batch=1024):
    """-> per-epoch held-in-fold scores [epochs, n_te, 3]."""
    torch.manual_seed(seed)
    model = SitHead(X.shape[1] if bank is None else bank[0].shape[1], d=cfg["d"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    pw = torch.full((3,), float(cfg["pw"]), device=device)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    Xt, on_gpu = bank if bank is not None else to_device_bank(X, device)
    dev = Xt.device
    Y = torch.from_numpy(S["Y"].astype(np.float32)).to(dev)
    V = torch.from_numpy(S["V"].astype(np.float32)).to(dev)
    tri = torch.from_numpy(tr_idx).to(dev)
    tei = torch.from_numpy(te_idx).to(dev)
    off = torch.arange(-(WIN - 1), 1, device=dev)
    out = np.empty((epochs, len(te_idx), 3), np.float32)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_idx), device=dev)
        for b in range(0, len(perm), batch):
            j = tri[perm[b:b + batch]]
            xb = Xt[(j[:, None] + off[None, :])].to(device, torch.float32)
            yb, vb = Y[j].to(device), V[j].to(device)
            if vb.sum() == 0:
                continue
            loss = (lossf(model(xb), yb) * vb).sum() / vb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            sc = []
            for b in range(0, len(te_idx), 8192):
                j = tei[b:b + 8192]
                sc.append(torch.sigmoid(
                    model(Xt[(j[:, None] + off[None, :])].to(device, torch.float32))).cpu())
            out[ep] = torch.cat(sc).numpy()
    return out, model


def window_flat(X, idx, device, chunk=200_000):
    """[n, WIN*dim] — the window read FLAT (never averaged; pooling costs ~3.8x, INHERITED)."""
    off = torch.arange(-(WIN - 1), 1)
    Xt = torch.from_numpy(X)
    out = []
    for b in range(0, len(idx), chunk):
        j = torch.from_numpy(idx[b:b + chunk])
        out.append(Xt[(j[:, None] + off[None, :])].reshape(len(j), -1))
    return torch.cat(out).to(device, torch.float32)


def ridge_fit_predict(Xtr, ytr, vtr, Xte, lam):
    """CLOSED-FORM ridge on +-1 targets, per situation. No optimiser -> no optimiser to blame.

    Returns [n_te, 3] scores. Standardisation uses TRAIN rows only."""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-3)
    A = torch.cat([(Xtr - mu) / sd, torch.ones(len(Xtr), 1, device=Xtr.device)], 1).double()
    B = torch.cat([(Xte - mu) / sd, torch.ones(len(Xte), 1, device=Xte.device)], 1).double()
    out = torch.zeros(len(Xte), 3, dtype=torch.float64, device=Xte.device)
    eye = torch.eye(A.shape[1], dtype=torch.float64, device=A.device)
    eye[-1, -1] = 0.0                                   # never penalise the intercept
    for i in range(3):
        m = vtr[:, i].bool()
        if m.sum() < 50:
            continue
        Am = A[m]
        t = torch.where(ytr[m, i].bool(), 1.0, -1.0).double()
        w = torch.linalg.solve(Am.T @ Am + lam * eye, Am.T @ t)
        out[:, i] = B @ w
    return out.float().cpu().numpy()


def ap(y, s):
    y = np.asarray(y, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-np.asarray(s, float), kind="mergesort")
    yt = y[o]
    tp = np.cumsum(yt)
    P = tp / np.maximum(np.arange(1, len(yt) + 1), 1e-12)
    R = tp / yt.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--bundle", required=True)
    ap_.add_argument("--feats", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--epochs", type=int, default=15)
    ap_.add_argument("--device", default="cuda")
    ap_.add_argument("--folds", type=int, default=5)
    a = ap_.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = load_substrate(a.bundle, a.feats, a.device)

    rows = S["rows"]
    side = np.array([rows[c]["side"] for c in S["clip"]])
    tr_mask = (side == "TRAIN") & S["win_ok"]
    te_mask = (side == "HELDOUT") & S["win_ok"]
    tr_idx = np.nonzero(tr_mask)[0]
    te_idx = np.nonzero(te_mask)[0]
    print(f"[split] TRAIN windows {len(tr_idx):,}  HELDOUT windows {len(te_idx):,}", flush=True)

    # ---- PCA on TRAIN ROWS ONLY (never the held-out side) ----
    mu, Wb = {}, {}
    for r in sorted({c["r"] for c in CONFIGS}):
        mu[r], Wb[r] = fit_pca(S["F"][tr_idx], r)
        print(f"[pca] r={r} basis fitted on {len(tr_idx):,} TRAIN rows", flush=True)

    # C-NEG: permute the image features ACROSS CLIPS (labels untouched)
    rng = np.random.default_rng(0)
    cl = S["clip"]
    order = rng.permutation(len(rows))
    remap = np.empty(len(rows), np.int64)
    remap[order] = np.arange(len(rows))
    shuf = np.arange(len(cl))
    starts = {i: rows[i]["off"] for i in range(len(rows))}
    for i, r in enumerate(rows):
        j = int(order[i])
        n = min(r["T"], rows[j]["T"])
        shuf[r["off"]:r["off"] + n] = starts[j] + np.arange(n)
        if r["T"] > n:
            shuf[r["off"] + n:r["off"] + r["T"]] = starts[j] + n - 1

    # ---- chunk-grouped folds inside TRAIN ----
    tr_chunks = sorted({rows[c]["chunk"] for c in S["clip"][tr_idx]})
    folds = {c: i % a.folds for i, c in enumerate(tr_chunks)}
    fold_of = np.array([folds[rows[c]["chunk"]] for c in S["clip"][tr_idx]])

    summary = {"configs": CONFIGS, "arms": list(ARMS), "epochs": a.epochs, "win": WIN,
               "n_train_windows": int(len(tr_idx)), "n_heldout_windows": int(len(te_idx)),
               "folds": {str(f): sorted(c for c in tr_chunks if folds[c] == f)
                         for f in range(a.folds)},
               "cv": {}, "selected": {}}
    scores = {}
    for arm in ARMS:
        best = None
        seen_cfg = set()
        for cfg in CONFIGS:
            # `head_ego` / `head_priv` have no image rank -> only the pos_weight distinguishes them
            key_cfg = (cfg["pw"], cfg["d"], cfg["r"] if arm not in ("head_ego", "head_priv") else 0)
            if key_cfg in seen_cfg:
                continue
            seen_cfg.add(key_cfg)
            X = arm_features(S, arm, mu[cfg["r"]], Wb[cfg["r"]], shuf)
            bank = to_device_bank(X, a.device)
            oof = np.full((a.epochs, len(tr_idx), 3), np.nan, np.float32)
            for f in range(a.folds):
                m = fold_of == f
                sc, _ = run_fold(X, S, tr_idx[~m], tr_idx[m], cfg, a.epochs, a.device, bank=bank)
                oof[:, m] = sc
            cvap = np.array([[ap(S["Y"][tr_idx][S["V"][tr_idx][:, i], i],
                                 oof[e][S["V"][tr_idx][:, i], i]) for i in range(3)]
                             for e in range(a.epochs)])
            key = f"{arm}|pw{cfg['pw']}|d{cfg['d']}|r{cfg['r']}"
            summary["cv"][key] = {SITS[i]: [round(float(v), 6) for v in cvap[:, i]]
                                  for i in range(3)}
            m_ = float(np.nanmean(cvap, 1).max())
            e_ = int(np.nanargmax(np.nanmean(cvap, 1)))
            print(f"[cv] {key}: best mean CV-AP {m_:.5f} @ epoch {e_+1} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if best is None or m_ > best[0]:
                best = (m_, cfg, e_, X, bank, oof[e_])
            else:
                del bank
        _m, cfg, e_, X, bank, oof_sel = best
        summary["selected"][arm] = {"cfg": cfg, "epoch": int(e_ + 1), "cv_ap_mean": round(_m, 6)}
        # ---- final: retrain on ALL of TRAIN for exactly the selected epochs, score HELDOUT ----
        sc, model = run_fold(X, S, tr_idx, te_idx, cfg, e_ + 1, a.device, bank=bank)
        scores[arm] = sc[-1]
        # theta* reads the GENUINELY out-of-fold TRAIN scores from the selected config/epoch —
        # which the CV already produced. (Re-training on all of TRAIN and scoring TRAIN would be
        # in-sample and would set the threshold on a distribution the head has memorised.)
        scores[arm + "__trainoof"] = oof_sel
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "arm": arm,
                    "in_dim": int(X.shape[1]), "win": WIN, "sits": SITS,
                    "epoch": int(e_ + 1), "ego_scale": EGO_SCALE.tolist()},
                   os.path.join(a.out, f"{arm}.pt"))
        del bank
        torch.cuda.empty_cache()
        print(f"[final] {arm} trained {e_+1} epochs, held-out scored ({time.time()-t0:.0f}s)",
              flush=True)

    # ------------------------------------------------------------------ the RIDGE ladder (A1)
    Ytr = torch.from_numpy(S["Y"][tr_idx].astype(np.float32)).to(a.device)
    Vtr = torch.from_numpy(S["V"][tr_idx].astype(np.float32)).to(a.device)
    R_PCA = 16
    for arm in RIDGE_ARMS:
        base = {"ridge_img_ego": "head_img_ego", "ridge_img": "head_img",
                "ridge_ego": "head_ego", "ridge_img_shuf": "head_img_shuf"}[arm]
        X = arm_features(S, base, mu[R_PCA], Wb[R_PCA], shuf)
        Ftr = window_flat(X, tr_idx, a.device)
        Fte = window_flat(X, te_idx, a.device)
        best = None
        for lam in LAMBDAS:
            oof = np.full((len(tr_idx), 3), np.nan, np.float32)
            for f in range(a.folds):
                m = fold_of == f
                mt = torch.from_numpy(~m).to(a.device)
                oof[m] = ridge_fit_predict(Ftr[mt], Ytr[mt], Vtr[mt],
                                           Ftr[torch.from_numpy(m).to(a.device)], lam)
            cv = [ap(S["Y"][tr_idx][S["V"][tr_idx][:, i], i], oof[S["V"][tr_idx][:, i], i])
                  for i in range(3)]
            summary["cv"][f"{arm}|lam{lam:g}|r{R_PCA}"] = {SITS[i]: round(float(cv[i]), 6)
                                                           for i in range(3)}
            m_ = float(np.nanmean(cv))
            if best is None or m_ > best[0]:
                best = (m_, lam, oof)
        _m, lam, oof = best
        summary["selected"][arm] = {"lam": lam, "r": R_PCA, "cv_ap_mean": round(_m, 6),
                                    "n_params": int(Ftr.shape[1] + 1)}
        scores[arm] = ridge_fit_predict(Ftr, Ytr, Vtr, Fte, lam)
        scores[arm + "__trainoof"] = oof
        print(f"[ridge] {arm}: lam={lam:g}, {Ftr.shape[1]+1} params, CV-AP {_m:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Ftr, Fte

    np.savez_compressed(os.path.join(a.out, "scores.npz"),
                        te_idx=te_idx, tr_idx=tr_idx,
                        clip=S["clip"], side=side,
                        Y=S["Y"], V=S["V"], E=S["E"], P=S["P"],
                        chunk=np.array([r["chunk"] for r in rows]),
                        k=np.array([r["k"] for r in rows]),
                        off=np.array([r["off"] for r in rows]),
                        T=np.array([r["T"] for r in rows]),
                        pca_r=np.array(sorted(mu)), **scores)
    np.savez_compressed(os.path.join(a.out, "pca.npz"),
                        **{f"mu{r}": mu[r] for r in mu}, **{f"W{r}": Wb[r] for r in Wb})
    summary["wallclock_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(a.out, "train_summary.json"), "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
