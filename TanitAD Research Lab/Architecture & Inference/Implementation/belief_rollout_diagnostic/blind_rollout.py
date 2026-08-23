"""Blind K-step belief-rollout diagnostic on the TRAINED 1-step imagination field.

Backlog P0 #0b (UWM-JEPA 2605.25313 + "Biased Dreams" 2604.25416 motivated).

The H15 ImaginationField is trained ONE step only (``h15_loss``: predict the
NEXT frame's hidden-sector tokens from the current masked frame). The UWM-JEPA
note (2026-07-15 KB) flagged a risk: *our epistemic sigma may dissipate over the
operative K-step rollout* -> if it collapses with horizon the H11/D8 self-monitor
trigger silently dies where anticipation matters. "Biased Dreams" sharpens it:
latent transitions have **attractor behavior** -> blind rollouts drift toward
well-represented regions ("discrepancy masking"), so the belief collapses to a
common attractor and its uncertainty becomes untrustworthy.

This script MEASURES the blind-rollout behavior of the DEPLOYED trained field on
REAL comma2k19 latents (step-6500 base250cam ckpt). No training, no config change
(D-018 safe). It runs the belief forward K steps with NO re-observation and asks:

  Q-sigma       Does mean hidden-cell log-variance GROW with horizon (calibrated:
                "I know I am rolling blind") or COLLAPSE toward the visible level
                (false confidence)?  -> the H11/D8 dissipation falsifier.
  Q-retention   Does the rolled belief retain hidden-cell fidelity ABOVE the
                no-imagination persistence baseline and the chance floor across
                horizons, and does it beat freeze-at-1-step (justifying multi-step)?
  Q-attractor   Does the belief drift to a common attractor (inter-sample cosine of
                the imagined hidden tokens -> 1)?  -> the "Biased Dreams" prediction.
  Q-calib       At each horizon, does per-cell predicted variance CORRELATE with
                per-cell actual error (does sigma point at the right cells)?

Arms compared per horizon (hidden-cell cosine vs the TRUE future tokens):
  rollout       imagination fed back each step (multi-step belief)
  freeze1       imagination run once at k=1, then held constant (1-step field, static)
  advect_only   flow-warp fed back, refine blocks skipped (ablate the transformer)
  persistence   hold the masked-frame tokens (pure no-imagination / no-dynamics)
  chance        shuffled-cell cosine (floor)

Run:  <tanitad venv>/python blind_rollout.py --episodes 24 --windows 4 --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# stack root on the path
STACK = Path(__file__).resolve().parents[4] / "stack"
sys.path.insert(0, str(STACK))

from tanitad.config import base250cam_config          # noqa: E402
from tanitad.models.fourbrain import WorldModel         # noqa: E402
from tanitad.models.imagination import advect, sector_mask  # noqa: E402

CKPT = "C:/Users/Admin/tanitad-data/eval/ckpt_full.pt"
VAL = "C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f"


def load_model(device):
    cfg = base250cam_config()
    model = WorldModel(cfg).to(device).eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    return model, cfg, ck.get("step")


def _center(x):
    """Remove the per-sample DC token (mean over cells). Raw ViT tokens share a
    huge common component that saturates cosine at ~1 for everything (incl. the
    chance floor); centering exposes the discriminative spatial structure."""
    return x - x.mean(dim=1, keepdim=True)


def _hidden_cosine(pred, true, hidden_mask):
    """mean cosine over hidden cells, on CENTERED tokens. [B,N,D], [B,N] bool."""
    p = F.normalize(_center(pred), dim=-1)
    t = F.normalize(_center(true), dim=-1)
    cos = (p * t).sum(-1)                                   # [B,N]
    return float(cos[hidden_mask].mean())


def _rel_l2(pred, true, hidden_mask):
    """Relative L2 error on hidden cells: ||pred-true||^2 / ||true-mean_true||^2.
    Scale/DC-robust; 0 = perfect, 1 = as bad as predicting the mean token, >1 worse."""
    tc = _center(true)
    num = (pred - true).pow(2).sum(-1)                      # [B,N]
    den = tc.pow(2).sum(-1).clamp_min(1e-8)                 # energy about the mean
    return float((num[hidden_mask] / den[hidden_mask]).mean())


def _inter_sample_cos(pred, hidden_mask):
    """mean pairwise cosine of the per-sample MEAN hidden token, CENTERED
    (attractor test: -> 1 means every sample's belief drifts to one attractor)."""
    predc = _center(pred)
    B = predc.shape[0]
    vecs = []
    for i in range(B):
        h = predc[i][hidden_mask[i]]
        if h.numel() == 0:
            continue
        vecs.append(F.normalize(h.mean(0), dim=-1))
    if len(vecs) < 2:
        return float("nan")
    V = torch.stack(vecs)                                   # [B, D]
    G = V @ V.t()
    off = G[~torch.eye(len(vecs), dtype=torch.bool, device=G.device)]
    return float(off.mean())


def _calib_corr(err2, var, hidden_mask):
    """Pearson corr between per-cell actual error^2 and predicted variance."""
    e = err2[hidden_mask]
    v = var[hidden_mask]
    if e.numel() < 3:
        return float("nan")
    e = e - e.mean(); v = v - v.mean()
    denom = (e.norm() * v.norm()).clamp_min(1e-12)
    return float((e * v).sum() / denom)


@torch.no_grad()
def run(device="cuda", episodes=24, windows=4, K=8, stride=6, seed=0):
    torch.manual_seed(seed)
    model, cfg, step = load_model(device)
    grid = model.encoder.grid_hw
    files = sorted(Path(VAL).glob("ep_*.pt"))[:episodes]

    # ---- build a batch of (t .. t+K) real-frame windows across episodes ------- #
    frames0, futures = [], []          # frames0[i]=[9,256,256] at t ; futures[i]=[K,9,256,256]
    gen = torch.Generator().manual_seed(seed)
    for f in files:
        ep = torch.load(f, map_location="cpu", weights_only=False)
        fr = ep["frames_u8"]                                # [T,9,256,256] uint8
        T = fr.shape[0]
        span = K * stride
        if T <= span + 1:
            continue
        starts = torch.randint(0, T - span - 1, (windows,), generator=gen).tolist()
        for t in starts:
            frames0.append(fr[t].float() / 255.0)
            futures.append(torch.stack([fr[t + (k + 1) * stride].float() / 255.0
                                        for k in range(K)]))
    frames0 = torch.stack(frames0).to(device)               # [B,9,256,256]
    futures = torch.stack(futures).to(device)               # [B,K,9,256,256]
    B = frames0.shape[0]

    # ---- mask, encode belief_0 + true tokens per horizon --------------------- #
    masked, vis = sector_mask(frames0, grid, generator=None)
    hidden = vis < 0.5                                       # [B,N] bool
    belief0 = model.encode_tokens(masked)                   # [B,N,D]
    # true tokens at each future horizon (batch the encodes)
    flat = futures.reshape(B * K, *futures.shape[2:])
    true_tok = model.encode_tokens(flat).reshape(B, K, *belief0.shape[1:])  # [B,K,N,D]

    imag = model.imagination
    D = belief0.shape[-1]

    # rolling beliefs for each arm
    b_roll = belief0.clone()
    b_adv = belief0.clone()
    # freeze1 belief: run imagination once at k=1 then hold
    b_freeze = None
    persist = belief0.clone()

    rows = []
    for k in range(K):
        true_k = true_tok[:, k]                             # [B,N,D]

        # --- rollout arm: imagination fed back ---
        pred_roll, logvar = imag(b_roll, vis)               # [B,N,D], [B,N]
        # --- advect-only arm ---
        flow = imag.flow_head(b_adv)
        pred_adv = advect(b_adv, flow, grid)
        # --- freeze1 arm ---
        if k == 0:
            b_freeze = pred_roll.clone()
        pred_freeze = b_freeze

        err2 = (pred_roll - true_k).pow(2).mean(-1)         # [B,N]
        var = torch.exp(logvar)
        rel_roll = _rel_l2(pred_roll, true_k, hidden)
        rel_pers = _rel_l2(persist, true_k, hidden)

        rows.append({
            "horizon": k + 1,
            # centered-cosine fidelity (higher = better)
            "cos_rollout": _hidden_cosine(pred_roll, true_k, hidden),
            "cos_freeze1": _hidden_cosine(pred_freeze, true_k, hidden),
            "cos_advect_only": _hidden_cosine(pred_adv, true_k, hidden),
            "cos_persistence": _hidden_cosine(persist, true_k, hidden),
            "cos_chance": _hidden_cosine(
                pred_roll[:, torch.randperm(pred_roll.shape[1], device=device)],
                true_k, hidden),
            # relative-L2 error (lower = better; 1.0 = as bad as the mean token)
            "relL2_rollout": rel_roll,
            "relL2_persistence": rel_pers,
            "relL2_freeze1": _rel_l2(pred_freeze, true_k, hidden),
            # skill vs no-imagination baseline (<1 = imagination beats persistence)
            "skill_rollout_over_persistence": rel_roll / max(rel_pers, 1e-8),
            # epistemic sigma (log-variance) -- should GROW (up) if calibrated blind
            "sigma_hidden_logvar": float(logvar[hidden].mean()),
            "sigma_visible_logvar": float(logvar[~hidden].mean()),
            "calib_gap": float(logvar[hidden].mean() - logvar[~hidden].mean()),
            # attractor collapse (Biased Dreams): centered inter-sample cosine -> 1
            "attractor_inter_sample_cos": _inter_sample_cos(pred_roll, hidden),
            # does sigma point at the wrong cells? per-cell err^2 vs var correlation
            "calib_err_var_corr": _calib_corr(err2, var, hidden),
            # belief token energy about its mean (attractor-to-origin check)
            "belief_centered_energy": float(_center(pred_roll)[hidden].pow(2).sum(-1).mean()),
            "true_centered_energy": float(_center(true_k)[hidden].pow(2).sum(-1).mean()),
        })

        # feed back for next step
        b_roll = pred_roll
        b_adv = pred_adv

    out = {
        "ckpt_step": step, "device": device, "n_windows": B, "K": K,
        "stride_frames": stride, "grid_hw": grid, "d_model": D,
        "mask_prob": cfg.h15.mask_prob, "note": "1-step-trained field, blind rollout, real comma2k19 val",
        "rows": rows,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = run(a.device, a.episodes, a.windows, a.k, a.stride, a.seed)
    out["wallclock_s"] = round(time.time() - t0, 1)
    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    (res / f"2026-07-17-blind_rollout-seed{a.seed}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    # compact console table
    print(f"ckpt step {out['ckpt_step']}  windows {out['n_windows']}  "
          f"K {out['K']}  {out['wallclock_s']}s")
    hdr = ("h", "cos_roll", "cos_pers", "cos_chnc", "relL2_rl", "relL2_pr",
           "skill", "sig_hid", "sig_vis", "attr", "calibr", "b_energy")
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in out["rows"]:
        print("  ".join(f"{v:>9.3f}" if isinstance(v, float) else f"{v:>9}" for v in [
            r["horizon"], r["cos_rollout"], r["cos_persistence"], r["cos_chance"],
            r["relL2_rollout"], r["relL2_persistence"],
            r["skill_rollout_over_persistence"], r["sigma_hidden_logvar"],
            r["sigma_visible_logvar"], r["attractor_inter_sample_cos"],
            r["calib_err_var_corr"], r["belief_centered_energy"]]))


if __name__ == "__main__":
    main()
