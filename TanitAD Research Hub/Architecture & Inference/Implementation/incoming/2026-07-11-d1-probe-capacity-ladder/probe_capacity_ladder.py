"""D1 probe-capacity ladder + isotropy linkage — is trajectory info LOST or just less LINEAR?

Extends the loop's `stack/scripts/d1_probe_capacity.py` (linear-ridge vs one MLP, two
checkpoints) into a *capacity ladder* (OLS -> ridge sweep -> MLP-1h -> MLP-2h) and, crucially,
measures the **latent isotropy** of the SAME latents in the SAME run so the mechanistic link
can be tested apples-to-apples on one checkpoint:

    LeJEPA/SIGReg theory (arXiv 2605.26379): linear + *orthogonal* identifiability enables
    optimal linear-space planning; the isotropic Gaussian is the UNIQUE bias/variance minimiser
    for OLS/ridge (2605.09241 Sub-JEPA, 2606.09646 layerwise-probing framing).
    => If the readout is anisotropic (measured iso_ratio_active=0.250 at step-6500,
       2026-07-10 orthogonality note), a LINEAR trajectory probe should underperform a
       nonlinear one. The size of that gap is the discriminator.

Pre-registered reading (single checkpoint):
  - gap := best_linear_ADE - best_MLP_ADE.
  - LARGE positive gap + LOW iso_active  -> info present but non-linearly organised; the
    anisotropy taxes the linear readout. D1's near-linear ADE readout underperforms =>
    remedy is isotropisation (subspace SIGReg, Sub-JEPA) / decode capacity, NOT "info lost".
  - SMALL gap                            -> trajectory info is linearly accessible here;
    a D1 regression is NOT a probe-capacity artefact => look elsewhere (coordinate frame /
    normalisation, or genuine drift). Falsifier for the "less-linear" hypothesis.

Standalone (pure torch, 0 new deps). Reuses only the stack WorldModel + episode loader.

Usage (local 4060 or pod):
  python probe_capacity_ladder.py \
      --ckpt C:/Users/Admin/tanitad-data/eval/ckpt_full.pt \
      --cache-dir C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
      --episodes 12 --out results/2026-07-11-probe_ladder_step6500.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

STEPS_PER_S = 10  # comma2k19 / cosmos loaders resample to 10 Hz


# --------------------------------------------------------------------------- geometry
def ego_frame(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame displacement into the ego heading at the window's last frame."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack(
        [dxy[..., 0] * c - dxy[..., 1] * s, dxy[..., 0] * s + dxy[..., 1] * c], dim=-1
    )


# --------------------------------------------------------------------------- isotropy
def isotropy_metrics(S: torch.Tensor, energy: float = 0.99) -> dict:
    """Covariance-spectrum isotropy of the readout latent (the space the probe reads).

    Returns active_k (energy knee), iso_ratio_active (geo/arith mean of top-k eigenvalues,
    ->1 = isotropic), cond_active, participation_ratio. Mirrors the 2026-07-10 orthogonality
    instrument so the two are cross-checkable.
    """
    Sc = S - S.mean(0, keepdim=True)
    cov = (Sc.T @ Sc) / max(1, Sc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)  # descending
    total = eig.sum().clamp_min(1e-12)
    csum = torch.cumsum(eig, 0) / total
    active_k = int((csum < energy).sum().item()) + 1
    active_k = max(1, min(active_k, eig.numel()))
    top = eig[:active_k].clamp_min(1e-12)
    geo = torch.exp(torch.log(top).mean())
    iso_active = float(geo / top.mean())
    cond_active = float(top[0] / top[-1])
    pr = float((eig.sum() ** 2) / (eig.pow(2).sum().clamp_min(1e-12)))
    return {
        "active_k": active_k,
        "iso_ratio_active": round(iso_active, 4),
        "cond_number_active": round(cond_active, 2),
        "participation_ratio": round(pr, 3),
        "top_eigs": [round(float(x), 3) for x in eig[: min(8, eig.numel())]],
    }


# --------------------------------------------------------------------------- probes
def _split(E: torch.Tensor):
    """Route-level split by episode-index parity (I3-style, deterministic, matches loop script)."""
    tr = E % 2 == 0
    return tr, ~tr


def fit_ridge(Xtr, Ytr, Xva, alpha: float):
    Xa = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1)
    W = torch.linalg.solve(
        Xa.T @ Xa + alpha * torch.eye(Xa.shape[1]), Xa.T @ Ytr
    )
    return torch.cat([Xva, torch.ones(Xva.shape[0], 1)], 1) @ W


def fit_mlp(Xtr, Ytr, Xva, hidden=(256,), epochs=60, seed=0):
    torch.manual_seed(seed)
    dims = [Xtr.shape[1], *hidden]
    layers: list[torch.nn.Module] = [torch.nn.LayerNorm(Xtr.shape[1])]
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.GELU()]
    layers += [torch.nn.Linear(dims[-1], 2)]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(Xtr.shape[0])
        for j in range(0, len(perm), 512):
            b = perm[j : j + 512]
            loss = (net(Xtr[b]) - Ytr[b]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(Xva)


def ade(pred, target):
    return round(float((pred - target).norm(dim=-1).mean()), 3)


def pca_project(Xtr: torch.Tensor, Xva: torch.Tensor, k: int):
    """Top-k PCA basis fit on TRAIN only (no val leakage); project both splits."""
    mu = Xtr.mean(0, keepdim=True)
    U, Sv, Vh = torch.linalg.svd(Xtr - mu, full_matrices=False)
    B = Vh[:k].T  # [D, k]
    return (Xtr - mu) @ B, (Xva - mu) @ B


def probe_ladder(S: torch.Tensor, Y: torch.Tensor, E: torch.Tensor, pca_k: int = 0) -> dict:
    """Fit the capacity ladder on (latent -> future displacement); return held-out ADE per rung.

    pca_k>0 first reduces the latent to its top-pca_k PCA subspace (fit on train only). This
    is the well-powered probe: with the 2048-dim raw latent and only ~200 train samples the
    problem is underdetermined (D>>N) and OLS overfits; reducing to the active_k subspace
    (from spectral-sizing) makes N>D and removes that confound.
    """
    tr, va = _split(E)
    Xtr, Ytr, Xva, Yva = S[tr], Y[tr], S[va], Y[va]
    if pca_k > 0:
        Xtr, Xva = pca_project(Xtr, Xva, min(pca_k, Xtr.shape[0] - 1, Xtr.shape[1]))
    rungs = {
        "linear_ols": fit_ridge(Xtr, Ytr, Xva, 1e-4),
        "ridge_a1": fit_ridge(Xtr, Ytr, Xva, 1.0),
        "ridge_a10": fit_ridge(Xtr, Ytr, Xva, 10.0),
        "ridge_a100": fit_ridge(Xtr, Ytr, Xva, 100.0),
        "mlp_256": fit_mlp(Xtr, Ytr, Xva, hidden=(256,)),
        "mlp_256x2": fit_mlp(Xtr, Ytr, Xva, hidden=(256, 256)),
    }
    out = {k: ade(p, Yva) for k, p in rungs.items()}
    lin_keys = ["linear_ols", "ridge_a1", "ridge_a10", "ridge_a100"]
    mlp_keys = ["mlp_256", "mlp_256x2"]
    best_lin = min(out[k] for k in lin_keys)
    best_mlp = min(out[k] for k in mlp_keys)
    out["best_linear"] = round(best_lin, 3)
    out["best_mlp"] = round(best_mlp, 3)
    out["gap_abs"] = round(best_lin - best_mlp, 3)
    out["gap_rel_pct"] = round(100.0 * (best_lin - best_mlp) / max(best_lin, 1e-9), 1)
    # zero-motion reference (predict no displacement): how much signal is there at all
    out["ref_zero_ade"] = round(float(Yva.norm(dim=-1).mean()), 3)
    out["n_train"] = int(tr.sum())
    out["n_val"] = int(va.sum())
    return out


# --------------------------------------------------------------------------- collection
@torch.no_grad()
def collect(world, episodes, device, window, horizons, stride=8, batch=8):
    """Encode sliding windows -> last-frame latent S; ego-frame displacement targets per horizon."""
    hmax = max(horizons)
    S: list[torch.Tensor] = []
    Y = {h: [] for h in horizons}
    E: list[int] = []
    for ei, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - hmax, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i : i + batch]
            fw = torch.stack([fr[t : t + window] for t in ch]).to(device)
            st = world.encode_window(fw)[:, -1].cpu()
            last = torch.tensor([t + window - 1 for t in ch])
            for h in horizons:
                dxy = ep.poses[last + h, :2] - ep.poses[last, :2]
                Y[h].append(ego_frame(dxy, ep.poses[last, 2]))
            S.append(st)
            E.extend([ei] * len(ch))
    return (
        torch.cat(S).float(),
        {h: torch.cat(v).float() for h, v in Y.items()},
        torch.tensor(E),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True, help="a *val* epcache dir with ep_*.pt")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 20])  # 1 s, 2 s @10 Hz
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tanitad.config import base250cam_config
    from tanitad.data.mixing import load_episode
    from tanitad.instruments.numerics import strict_numerics
    from tanitad.models.fourbrain import WorldModel

    eps_paths = sorted(Path(args.cache_dir).glob("ep_*.pt"))[: args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps_paths]

    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    world.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(args.device).eval()
    window = world.predictor.cfg.window

    with strict_numerics():
        S, Y, E = collect(world, episodes, args.device, window, args.horizons)

    report = {
        "exp": "d1-probe-capacity-ladder",
        "checkpoint_step": step,
        "n_episodes": len(episodes),
        "n_samples": int(S.shape[0]),
        "state_dim": int(S.shape[1]),
        "isotropy": isotropy_metrics(S),
        "ade_by_horizon_raw2048": {},
        "ade_by_horizon_pca_activek": {},
    }
    active_k = report["isotropy"]["active_k"]
    report["pca_k_used"] = active_k
    for h in args.horizons:
        key = f"h{h}_{h // STEPS_PER_S}s"
        report["ade_by_horizon_raw2048"][key] = probe_ladder(S, Y[h], E, pca_k=0)
        report["ade_by_horizon_pca_activek"][key] = probe_ladder(S, Y[h], E, pca_k=active_k)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("PROBE_LADDER_DONE", flush=True)


if __name__ == "__main__":
    main()
