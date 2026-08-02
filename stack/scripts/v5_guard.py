"""v5 GUARD + the P2 instrument for anchor-head arms (deep review 2026-08-02).

Two jobs in one file, because they share every moving part:

1. **The pre-registered 5k guard** (``PREREG_v5_cheapest_guard.md``): does ``goal_dropout 0.5``
   collapse route conditioning the way v2corpus's ``nav-dropout 0.5`` demonstrably did
   (route_acc_nav 1.0 → 0.5351)? Decides the relaunch's goal_dropout BEFORE 3.5 more GPU-days.
2. **TAC/STR families for the v4/v5 head** — ``run_hierarchy`` supports only
   ``tactical_policy`` arms and SKIPs otherwise, so without this the binding rule's tactical and
   strategic families are unscoreable on v5 at all.

## What is measured, and why it is collapse-proof

**STRATEGIC — route-following sensitivity.** Forward the SAME windows twice: once with the
commanded route token, once with the commands PERMUTED across the batch. Classify each output
trajectory (net heading over the horizon) and score agreement with the token it was given.

    follow_true  ≈ follow_shuffled   ⇒ the head IGNORES the route token (the v2corpus collapse)
    follow_true  ≫ follow_shuffled   ⇒ route conditioning is live

⭐ The permutation null needs NO knowledge of the embedding's dropped row and cannot be gamed by
class imbalance — a straight-heavy corpus inflates both arms equally, and the DELTA is the signal.
(⚠️ This measures *command-following*, the collapse detector — not route *prediction* skill;
GATE_PROTOCOL §0.7's void rule is about the latter and does not apply here.)

**LONGITUDINAL-GOAL — vt sensitivity.** Same windows, ``vt_band`` forced low vs high: the mean
output speed must move. Zero movement = the tactical speed goal is decorative — on the arm whose
whole longitudinal thesis is that goal.

**TACTICAL — selection consistency κ.** Manoeuvre class of the SELECTED raw anchor vs of the
refined output trajectory (Cohen's κ). The anchor-head analogue of ``maneuver_vs_trajectory``:
declared choice vs executed path.

⚠️ **Geometry: score v5 on v5's OWN val cache** (the w120 176×624 cylindrical build). The
canonical 256 px ``val40cache`` is the WRONG imaging geometry for this encoder — scoring on it
would be a silent instrument failure, not a model result.

Usage (pod2):
    PYTHONPATH=/workspace/TanitAD/stack python3 scripts/v5_guard.py \
        --ckpt /workspace/v5_modelonly.pt \
        --config /workspace/experiments/flagship-v5-w120-30k/config.json \
        --val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
        --episodes 40 --out /workspace/v5_guard_5k.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

# net-heading threshold for left/straight/right — matches the v2.1 route derivation's
# spirit (its adaptive horizon reduces to a fixed one on 2 s windows); stated, not hidden.
ROUTE_DEG = 8.0
R_LEFT, R_STRAIGHT, R_RIGHT = 0, 1, 2


def classify_route(traj: torch.Tensor, deg: float = ROUTE_DEG) -> torch.Tensor:
    """traj [B, H, 2] ego-frame -> {0 left, 1 straight, 2 right} by net path heading."""
    d = traj[:, -1] - traj[:, 0]
    head = torch.rad2deg(torch.atan2(d[:, 1], d[:, 0]))
    out = torch.full((traj.shape[0],), R_STRAIGHT, dtype=torch.long)
    out[head > deg] = R_LEFT           # +y is LEFT in the ego frame
    out[head < -deg] = R_RIGHT
    return out


def cohens_kappa(a: torch.Tensor, b: torch.Tensor, k: int = 3) -> float:
    """Agreement corrected for chance; the tactical-consistency scalar."""
    n = a.numel()
    if n == 0:
        return float("nan")
    po = float((a == b).float().mean())
    pe = sum(float((a == c).float().mean()) * float((b == c).float().mean())
             for c in range(k))
    return round((po - pe) / (1 - pe + 1e-9), 4)


def mean_speed(traj: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """[B] mean speed along each output trajectory, m/s."""
    steps = traj[:, 1:] - traj[:, :-1]
    return steps.norm(dim=-1).mean(dim=1) / dt


@torch.no_grad()
def run_guard(head, world, ds, device, episodes: int, batch: int = 16,
              seed: int = 0) -> dict:
    from torch.utils.data import default_collate

    from scripts.train_flagship_v4 import _goal_inputs

    g = torch.Generator().manual_seed(seed)
    sel = [i for i, (e, t) in enumerate(ds.index) if e < episodes and t % 8 == 0]
    n_vt = getattr(head.cfg, "n_vt_bands", 23)
    lo_band, hi_band = 2, max(3, n_vt - 3)

    rows = {"ep": [], "cmd": [], "cls_true": [], "cls_shuf": [], "cmd_shuf": [],
            "cls_anchor": [], "cls_refined": [], "v_lo": [], "v_hi": []}
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        b = default_collate([ds[i] for i in idx])
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        v0 = b["pose_last"][:, 3].float()
        st = world.encode_window(b["frames"])
        goal = _goal_inputs(head.cfg, b, v0)
        bs = v0.shape[0]

        # --- STRATEGIC: true command vs permuted command --------------------------
        out_t = head(st, v0, **goal)
        perm = torch.randperm(bs, generator=g).to(device)
        goal_s = dict(goal)
        if "route" in goal_s:
            goal_s["route"] = goal_s["route"][perm]
            if "route_graded" in goal_s:
                goal_s["route_graded"] = goal_s["route_graded"][perm]
        out_s = head(st, v0, **goal_s)

        # --- LONGITUDINAL GOAL: vt low vs high ------------------------------------
        goal_lo = dict(goal); goal_hi = dict(goal)
        if "vt_band" in goal:
            goal_lo["vt_band"] = torch.full_like(goal["vt_band"], lo_band)
            goal_hi["vt_band"] = torch.full_like(goal["vt_band"], hi_band)
        out_lo = head(st, v0, **goal_lo)
        out_hi = head(st, v0, **goal_hi)

        # --- TACTICAL: selected raw anchor vs refined output ----------------------
        anchors = head.decoder.anchors                      # [M, H, 2]
        sel_anchor = anchors[out_t["sel_idx"]]

        rows["ep"].extend(int(ds.index[i][0]) for i in idx)
        rows["cmd"].extend(goal.get("route",
                           torch.full((bs,), R_STRAIGHT)).cpu().tolist())
        rows["cmd_shuf"].extend(goal_s.get("route",
                                torch.full((bs,), R_STRAIGHT)).cpu().tolist())
        rows["cls_true"].extend(classify_route(out_t["traj"].cpu()).tolist())
        rows["cls_shuf"].extend(classify_route(out_s["traj"].cpu()).tolist())
        rows["cls_anchor"].extend(classify_route(sel_anchor.cpu()).tolist())
        rows["cls_refined"].extend(classify_route(out_t["traj"].cpu()).tolist())
        rows["v_lo"].extend(mean_speed(out_lo["traj"].cpu()).tolist())
        rows["v_hi"].extend(mean_speed(out_hi["traj"].cpu()).tolist())

    T = {k: torch.tensor(v) for k, v in rows.items()}
    n = T["cmd"].numel()
    follow_true = float((T["cls_true"] == T["cmd"]).float().mean())
    follow_shuf = float((T["cls_shuf"] == T["cmd_shuf"]).float().mean())

    # paired episode-cluster bootstrap on the delta (the programme's estimator, B=2000)
    eps_ids = T["ep"].unique()
    d_true = (T["cls_true"] == T["cmd"]).float()
    d_shuf = (T["cls_shuf"] == T["cmd_shuf"]).float()
    deltas = []
    gb = torch.Generator().manual_seed(1)
    for _ in range(2000):
        pick = eps_ids[torch.randint(len(eps_ids), (len(eps_ids),), generator=gb)]
        m = torch.isin(T["ep"], pick)
        deltas.append(float(d_true[m].mean() - d_shuf[m].mean()))
    deltas = sorted(deltas)
    lo_ci, hi_ci = deltas[50], deltas[1949]

    dv = float((T["v_hi"] - T["v_lo"]).mean())
    kappa = cohens_kappa(T["cls_anchor"], T["cls_refined"])

    verdict = ("ROUTE_LIVE" if lo_ci > 0.0 else
               "ROUTE_COLLAPSED — the token is ignored (the v2corpus failure); "
               "lower goal_dropout to 0.25 or ramp it before relaunch")
    return {
        "n_windows": n, "n_episodes": int(len(eps_ids)),
        "strategic": {
            "follow_true": round(follow_true, 4),
            "follow_shuffled": round(follow_shuf, 4),
            "delta": round(follow_true - follow_shuf, 4),
            "delta_ci95": [round(lo_ci, 4), round(hi_ci, 4)],
            "estimator": "paired_episode_cluster_bootstrap_B2000",
            "cmd_distribution": {str(c): int((T["cmd"] == c).sum())
                                 for c in (0, 1, 2)},
            "note": "command-FOLLOWING sensitivity (collapse detector), not route "
                    "prediction skill; §0.7's void rule concerns the latter",
        },
        "longitudinal_goal": {
            "dv_high_minus_low_mps": round(dv, 4),
            "verdict": "VT_LIVE" if abs(dv) > 0.3 else
                       "VT_DEAD — the tactical speed goal does not move the plan",
        },
        "tactical": {"anchor_vs_refined_kappa": kappa,
                     "verdict": ("DECORATIVE" if not math.isnan(kappa) and kappa < 0.1
                                 else "WEAK" if kappa < 0.4 else "SUBSTANTIAL")},
        "verdict": verdict,
    }


def load_v5(ckpt: str, config: str, device: str):
    """Rebuild WM + head from the run's OWN config.json (the loaders.py pattern —
    a v2-family checkpoint is unloadable without its recorded config)."""
    import dataclasses

    from tanitad.config import flagship4b_config
    from tanitad.models.flagship_v4 import FlagshipV4Head, v4_config
    from tanitad.models.fourbrain import WorldModel

    rec = json.loads(Path(config).read_text())
    cfg = flagship4b_config()
    # geometry + trunk overrides recorded by the v4 trainer
    for k, v in (rec.get("trunk") or {}).items():
        if hasattr(cfg.encoder, k):
            object.__setattr__(cfg.encoder, k, v)
    world = WorldModel(cfg)
    hcfg = v4_config()
    for k, v in (rec.get("head_cfg") or {}).items():
        if k == "decoder":
            hcfg.decoder = dataclasses.replace(hcfg.decoder, **v)
        elif hasattr(hcfg, k):
            setattr(hcfg, k, tuple(v) if isinstance(v, list) else v)
    head = FlagshipV4Head(hcfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    world.load_state_dict(ck["model"])
    head.load_state_dict(ck["head"])
    print(f"[guard] loaded step={ck.get('step')}  head tokens: "
          f"imag={head.n_imag_tokens}", flush=True)
    return world.to(device).eval(), head.to(device).eval(), ck.get("step")


def main():
    ap = argparse.ArgumentParser("v5_guard")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from tanitad.data.mixing import load_episode

    from scripts.train_flagship_v4 import FlagshipV4Dataset, horizon_plan
    from tanitad.config import flagship4b_config

    world, head, step = load_v5(a.ckpt, a.config, device)
    cfg = flagship4b_config()
    plan = horizon_plan(cfg)
    files = sorted(Path(a.val_cache).glob("ep_*.pt"))[:a.episodes]
    assert files, f"no ep_*.pt under {a.val_cache}"
    eps = [load_episode(str(p), mmap=True) for p in files]
    ds = FlagshipV4Dataset(eps, window=head.cfg.window,
                           max_horizon=plan.max_horizon,
                           maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
    res = run_guard(head, world, ds, device, episodes=a.episodes)
    res["ckpt"], res["step"] = a.ckpt, step
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["strategic"], indent=1), flush=True)
    print(f"[guard] LON-goal: {res['longitudinal_goal']}", flush=True)
    print(f"[guard] TAC κ:    {res['tactical']}", flush=True)
    print(f"[guard] VERDICT:  {res['verdict']}", flush=True)


if __name__ == "__main__":
    main()
