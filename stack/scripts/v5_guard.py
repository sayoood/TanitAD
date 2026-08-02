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
              probes=None,
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

        # ⛔ COND-IMAGINATION FEED. A head trained with --cond-imagination REQUIRES imagined
        # latents and raises without them (flagship_v15.build_tokens). The guard previously
        # called head(...) bare, so it could not score ANY v5-line arm — the exact class of arm
        # it exists to guard. Built through the TRAINER'S OWN helper so the tokens the guard
        # feeds are identical to the ones the head was trained on; a re-implementation here
        # would silently measure a different model.
        goal.update(_imagination_inputs(world, head.cfg, b, st, probes))

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
    n_all = T["cmd"].numel()

    # ⛔ CLASS-VOCABULARY MASK. The label pipeline's route is 0..3, where
    # 3 == refb_labels.ROUTE_UNKNOWN ("unjudgeable window", ~20 % of the corpus) — see
    # v4_labels.py:264/364 and refb_labels.py:511-513,536. `classify_route` above emits ONLY
    # 0/1/2, so a cmd==3 window can NEVER be matched: scoring it counts a DETERMINISTIC MISS
    # against the model for a command the instrument cannot express.
    # MEASURED 2026-08-02: the 5k run published n_windows 881 with a 3-class histogram summing
    # to 727 — 154 windows (17.5 %) were silently deflating follow_true, follow_shuffled AND the
    # bootstrap. Restricting to the in-vocabulary windows raises the delta ~1.21x.
    # The correct handling is to EXCLUDE them, not to default them to straight (refb_labels.py:511
    # "NEVER DEFAULT TO STRAIGHT"), and to report the excluded count so it is never invisible again.
    jud = (T["cmd"] >= 0) & (T["cmd"] <= 2)
    n_unjudgeable = int((~jud).sum())
    T = {k: v[jud] for k, v in T.items()}
    n = T["cmd"].numel()
    if n == 0:
        raise RuntimeError("v5_guard: every window is route-UNKNOWN — nothing judgeable to score")
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
            # the histogram MUST sum to n_windows_scored, or a class is invisible (see the mask)
            "cmd_distribution": {str(c): int((T["cmd"] == c).sum())
                                 for c in (0, 1, 2)},
            "n_unjudgeable_excluded": n_unjudgeable,
            "n_collected": n_all,
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
    """Rebuild geometry + WM + head from the run's OWN recorded launch args.

    ⚠️ The guard goes through the TRAINER'S seams (``resolve_v2_frames``, which also
    mutates the encoder config to the sub-frame) rather than re-deriving geometry —
    a hand-rebuilt encoder at the cache frame would load weights that expect
    176×624 and silently score garbage. Same class as C66: verify the path the
    code actually takes, not a lookalike."""
    import dataclasses
    from types import SimpleNamespace

    from scripts.train_flagship_v4 import resolve_v2_frames
    from tanitad.config import flagship4b_config
    from tanitad.models.flagship_v4 import FlagshipV4Head, v4_config
    from tanitad.models.fourbrain import WorldModel

    rec = json.loads(Path(config).read_text())
    ra = rec.get("args") or {}
    assert isinstance(ra, dict) and ra, "config.json has no recorded launch args"
    ns = SimpleNamespace(**ra)

    cfg = flagship4b_config()
    cache_frame, train_frame = resolve_v2_frames(ns, cfg, label="v5_guard")
    # v5 trains the v1 speed-input trunk (action_dim 3) — mirror the trainer's own
    # block (train_flagship_v4.py:1288ff) or the STRICT load refuses on act_emb.
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

    world = WorldModel(cfg)
    hcfg = v4_config()
    for k, v in (rec.get("head_cfg") or {}).items():
        if k == "decoder":
            hcfg.decoder = dataclasses.replace(hcfg.decoder, **v)
        elif hasattr(hcfg, k):
            setattr(hcfg, k, tuple(v) if isinstance(v, list) else v)
    head = FlagshipV4Head(hcfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    world.load_state_dict(ck["model"])                 # STRICT — a mismatch is a finding
    head.load_state_dict(ck["head"])
    print(f"[guard] loaded step={ck.get('step')} frame={train_frame.height}x"
          f"{train_frame.width} imag_tokens={head.n_imag_tokens}", flush=True)
    return (world.to(device).eval(), head.to(device).eval(), ck.get("step"),
            ns, cfg, cache_frame, train_frame)


def main():
    ap = argparse.ArgumentParser("v5_guard")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", required=True)
    # ⛔ A cond-imagination arm CANNOT be scored without its OWN probe vocabulary.
    # It must be the run's cached probe_vocab.pt: a re-sampled vocabulary silently
    # changes what the imagination was asked, so the guard would measure a different
    # conditioning than training used.
    ap.add_argument("--probes", default=None,
                    help="probe_vocab.pt from the run dir (REQUIRED for --cond-imagination arms)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from flagship_v4_data import FlagshipV4Dataset
    from train_flagship_v4 import _imagination_inputs  # noqa: F401 (used in run_guard)

    from tanitad.data.v2_dataset import build_v2_providers
    from tanitad.train.flagship_losses import horizon_plan

    world, head, step, ns, cfg, cache_frame, train_frame = \
        load_v5(a.ckpt, a.config, device)

    # the run's own val corpus, at the run's own frame (v2 compressed, lazy)
    slice_frame = None if train_frame == cache_frame else train_frame
    eps = build_v2_providers([a.val_cache], lru_size=int(getattr(ns, "v2_lru", 8)),
                             frame=slice_frame, verbose=False)
    assert eps, f"no *.v2ep.pt under {a.val_cache}"
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    ds = FlagshipV4Dataset(eps[:a.episodes], window=head.cfg.window,
                           max_horizon=plan.max_horizon,
                           maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
    probes = None
    if getattr(head.cfg, "cond_imagination", False):
        if not a.probes:
            raise SystemExit(
                "this checkpoint was trained with cond_imagination=True, so the head "
                "REQUIRES imagined latents. Pass --probes <run_dir>/probe_vocab.pt. "
                "Scoring it without them is impossible, not optional.")
        probes = torch.load(a.probes, map_location=device, weights_only=False)
        if isinstance(probes, dict):
            probes = probes.get("probes", probes.get("vocab"))
        probes = probes.to(device).float()
        print(f"[guard] probe vocabulary {tuple(probes.shape)} from {a.probes}", flush=True)
    res = run_guard(head, world, ds, device, episodes=a.episodes, probes=probes)
    res["ckpt"], res["step"] = a.ckpt, step
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["strategic"], indent=1), flush=True)
    print(f"[guard] LON-goal: {res['longitudinal_goal']}", flush=True)
    print(f"[guard] TAC κ:    {res['tactical']}", flush=True)
    print(f"[guard] VERDICT:  {res['verdict']}", flush=True)


if __name__ == "__main__":
    main()
