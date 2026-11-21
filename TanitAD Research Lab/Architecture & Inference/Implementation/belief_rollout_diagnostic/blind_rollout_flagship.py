"""Blind K-step belief-rollout diagnostic on the OPERATIVE flagship-speed model.

Backlog P0.1 (fleet directive 2026-07-17): re-run the 2026-07-17 σ-dissipation /
attractor-collapse diagnostic (which used the PRE-RESET step-6500 base250cam ckpt)
on the OPERATIVE post-reset flagship — dropping the pre-reset caveat.

Falsifier (backlog P0.1): σ-dissipation reproduces on the speed flagship;
FALSIFIER = it does NOT reproduce → the speed+jerk recipe already fixed it
(report which ingredient).

Runs on the eval pod (A40). Model = WorldModel(flagship4b, action_dim=3), the
exact build taniteval.loaders uses for `flagship-speed`. Val = the pod's canonical
held-out PhysicalAI val (40 eps) — NOTE: different distribution from the 07-17
comma2k19 run; σ-dissipation is a model-internal blind-rollout property, but the
val delta is recorded for honesty (P8).

No training, no config change (D-018 safe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Pod paths first (eval pod), then a repo-relative fallback so the parity test can
# import this module locally (the tanitad package resolves from the worktree stack).
for _p in ("/root/TanitAD/stack", "/root/TanitAD/stack/scripts"):
    if Path(_p).exists():
        sys.path.insert(0, _p)
_LOCAL_STACK = Path(__file__).resolve().parents[4] / "stack"
if _LOCAL_STACK.exists():
    sys.path.insert(0, str(_LOCAL_STACK))
    sys.path.insert(0, str(_LOCAL_STACK / "scripts"))

from tanitad.config import flagship4b_config              # noqa: E402
from tanitad.models.fourbrain import WorldModel           # noqa: E402
from tanitad.models.imagination import advect, sector_mask  # noqa: E402
from tanitad.data.mixing import load_episode              # noqa: E402

CKPT = "/root/models/flagship-speed/ckpt.pt"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"


def load_model(device):
    cfg = flagship4b_config()
    # match taniteval.loaders: speed flagship is action_dim=3
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if getattr(cfg, "tactical_pred", None) is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    model = WorldModel(cfg).to(device).eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    return model, cfg, ck.get("step")


def _center(x):
    return x - x.mean(dim=1, keepdim=True)


def _hidden_cosine(pred, true, hidden_mask):
    p = F.normalize(_center(pred), dim=-1)
    t = F.normalize(_center(true), dim=-1)
    cos = (p * t).sum(-1)
    return float(cos[hidden_mask].mean())


def _rel_l2(pred, true, hidden_mask):
    tc = _center(true)
    num = (pred - true).pow(2).sum(-1)
    den = tc.pow(2).sum(-1).clamp_min(1e-8)
    return float((num[hidden_mask] / den[hidden_mask]).mean())


def _inter_sample_cos(pred, hidden_mask):
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
    V = torch.stack(vecs)
    G = V @ V.t()
    off = G[~torch.eye(len(vecs), dtype=torch.bool, device=G.device)]
    return float(off.mean())


def _calib_corr(err2, var, hidden_mask):
    e = err2[hidden_mask]
    v = var[hidden_mask]
    if e.numel() < 3:
        return float("nan")
    e = e - e.mean(); v = v - v.mean()
    denom = (e.norm() * v.norm()).clamp_min(1e-12)
    return float((e * v).sum() / denom)


@torch.no_grad()
def run(device="cuda", episodes=40, windows=8, K=8, stride=6, seed=0):
    torch.manual_seed(seed)
    model, cfg, step = load_model(device)
    grid = model.encoder.grid_hw
    files = sorted(Path(VAL).glob("ep_*.pt"))[:episodes]

    frames0, futures = [], []
    gen = torch.Generator().manual_seed(seed)
    for f in files:
        ep = load_episode(str(f), mmap=True)
        fr = ep.frames                                    # [T,9,256,256] uint8 (mmap)
        T = fr.shape[0]
        span = K * stride
        if T <= span + 1:
            continue
        starts = torch.randint(0, T - span - 1, (windows,), generator=gen).tolist()
        for t in starts:
            frames0.append(fr[t].float().div(255.0).clone())
            futures.append(torch.stack([fr[t + (k + 1) * stride].float().div(255.0).clone()
                                        for k in range(K)]))
    frames0 = torch.stack(frames0).to(device)
    futures = torch.stack(futures).to(device)
    B = frames0.shape[0]

    masked, vis = sector_mask(frames0, grid, generator=None)
    hidden = vis < 0.5
    belief0 = model.encode_tokens(masked)
    flat = futures.reshape(B * K, *futures.shape[2:])
    # encode true future tokens in sub-batches (A40 has room, but be safe)
    tt = []
    for i in range(0, flat.shape[0], 128):
        tt.append(model.encode_tokens(flat[i:i + 128]))
    true_tok = torch.cat(tt).reshape(B, K, *belief0.shape[1:])

    imag = model.imagination
    D = belief0.shape[-1]

    b_roll = belief0.clone()
    b_adv = belief0.clone()
    b_freeze = None
    persist = belief0.clone()

    rows = []
    for k in range(K):
        true_k = true_tok[:, k]

        pred_roll, logvar = imag(b_roll, vis)
        flow = imag.flow_head(b_adv)
        pred_adv = advect(b_adv, flow, grid)
        if k == 0:
            b_freeze = pred_roll.clone()
        pred_freeze = b_freeze

        err2 = (pred_roll - true_k).pow(2).mean(-1)
        var = torch.exp(logvar)
        rel_roll = _rel_l2(pred_roll, true_k, hidden)
        rel_pers = _rel_l2(persist, true_k, hidden)

        rows.append({
            "horizon": k + 1,
            "cos_rollout": _hidden_cosine(pred_roll, true_k, hidden),
            "cos_freeze1": _hidden_cosine(pred_freeze, true_k, hidden),
            "cos_advect_only": _hidden_cosine(pred_adv, true_k, hidden),
            "cos_persistence": _hidden_cosine(persist, true_k, hidden),
            "cos_chance": _hidden_cosine(
                pred_roll[:, torch.randperm(pred_roll.shape[1], device=device)],
                true_k, hidden),
            "relL2_rollout": rel_roll,
            "relL2_persistence": rel_pers,
            "relL2_freeze1": _rel_l2(pred_freeze, true_k, hidden),
            "skill_rollout_over_persistence": rel_roll / max(rel_pers, 1e-8),
            "sigma_hidden_logvar": float(logvar[hidden].mean()),
            "sigma_visible_logvar": float(logvar[~hidden].mean()),
            "calib_gap": float(logvar[hidden].mean() - logvar[~hidden].mean()),
            "attractor_inter_sample_cos": _inter_sample_cos(pred_roll, hidden),
            "calib_err_var_corr": _calib_corr(err2, var, hidden),
            "belief_centered_energy": float(_center(pred_roll)[hidden].pow(2).sum(-1).mean()),
            "true_centered_energy": float(_center(true_k)[hidden].pow(2).sum(-1).mean()),
        })

        b_roll = pred_roll
        b_adv = pred_adv

    mask_prob = getattr(getattr(cfg, "h15", None), "mask_prob", None)
    out = {
        "ckpt_step": step, "device": device, "n_windows": B, "K": K,
        "stride_frames": stride, "grid_hw": grid, "d_model": D,
        "mask_prob": mask_prob,
        "model": "flagship-speed (WorldModel flagship4b, action_dim=3)",
        "val": "physicalai-val-0c5f7dac3b11 (pod canonical held-out)",
        "note": "OPERATIVE post-reset flagship, 1-step-trained field, blind rollout",
        "rows": rows,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/taniteval/results")
    a = ap.parse_args()
    t0 = time.time()
    out = run(a.device, a.episodes, a.windows, a.k, a.stride, a.seed)
    out["wallclock_s"] = round(time.time() - t0, 1)
    res = Path(a.out)
    res.mkdir(exist_ok=True, parents=True)
    (res / f"2026-07-18-blind_rollout-flagship-speed-seed{a.seed}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"MODEL flagship-speed  ckpt step {out['ckpt_step']}  windows {out['n_windows']}  "
          f"K {out['K']}  {out['wallclock_s']}s")
    hdr = ("h", "cos_roll", "cos_frz1", "cos_pers", "cos_chnc", "relL2_rl",
           "skill", "sig_hid", "sig_vis", "attr", "b_energy")
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in out["rows"]:
        print("  ".join(f"{v:>9.3f}" if isinstance(v, float) else f"{v:>9}" for v in [
            r["horizon"], r["cos_rollout"], r["cos_freeze1"], r["cos_persistence"],
            r["cos_chance"], r["relL2_rollout"],
            r["skill_rollout_over_persistence"], r["sigma_hidden_logvar"],
            r["sigma_visible_logvar"], r["attractor_inter_sample_cos"],
            r["belief_centered_energy"]]))


if __name__ == "__main__":
    main()
