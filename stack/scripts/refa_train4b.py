"""REF-A 4-brain trainer (D-030) — the frozen-DINO twin of the flagship.

Mirrors scripts/train_flagship4b.py MODULE-FOR-MODULE, calling the SAME shared
4-brain loss (tanitad.train.flagship_losses.flagship_loss) on states produced by
the frozen-DINO adapter instead of the from-scratch ViT + spatial readout. The
ONLY two differences from the flagship trainer are, by construction:
  (1) the ENCODER — frozen-DINO features -> trainable grid adapter, vs a
      from-scratch ViT encoder + spatial readout; and
  (2) the SIGReg target — SIGREG_VARIANT="pred_only" (predictor outputs only;
      frozen features need no anti-collapse on the embeddings), vs the flagship's
      "full_relaxed" (full latent + predictions, position-subspace relaxed).
Everything else — the intent-conditioned operative predictor, the tactical-
predictor dynamics, the trained tactical/strategic policy brains, the
hierarchical metric grounding, and the JEPA / maneuver-CE / route-CE / waypoint /
inv-dyn terms plus every logged component — is the SAME shared code the flagship
runs (via the SHARED flagship_loss / build_grounding / horizon_plan). This is the
machine-checkable claim tests/test_refa_flagship_parity.py pins.

Consumes the per-episode DINO feature files written by scripts/dino_precompute.py
({"feats_fp16" [T,256,768], "actions" [T,2], "poses" [T,4], ...}); the feature-
window dataset threads the SAME strategic/tactical pseudo-labels the flagship's
FlagshipWindowDataset emits (nav_cmd / nav_valid / route_target + maneuver_label,
derived by scripts/refb_labels), only the input is DINO features, not frames. H15
imagination stays OUT (encoder/token-grid specific — never part of the shared
loss). Grounding heads are saved under a SEPARATE ckpt key so a vanilla RefAModel
still loads ckpt["model"]. Standardizer stats are frozen (REF-A item 1) and reused
on resume; the trainable adapter gets a 10x-longer LR warmup + its own gradient-
norm monitor (item 4); adapter-output per-dim std is the collapse monitor.

Usage (pod3 — the definitive corrected-DINO-features grounded 4-brain REF-A run):
  python scripts/refa_train4b.py --data-root /opt/dino_feats \
     --out /workspace/experiments/refa4b-30k --steps 30000 \
     --batch-size 128 --n-tokens 256 --rollout-k 4
Smoke (CPU):
  python scripts/refa_train4b.py --data toy --config smoke --steps 4 \
     --batch-size 4 --n-tokens 16 --rollout-k 4 --out <dir> --log-every 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

# Make the local checkout authoritative when run as a script; prepend the stack
# root (for `tanitad`) and the scripts dir (for refb_labels / refa_train).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import refb_labels  # noqa: E402  (scripts/refb_labels.py — pseudo-labels)
from refa_train import (FeatureWindowDataset,  # noqa: E402
                        load_feature_episodes, param_groups)

from tanitad.config import refa4b_config, refa4b_smoke_config  # noqa: E402
from tanitad.refs.refa import RefAModel  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)
from tanitad.train.train_worldmodel import cosine_lr  # noqa: E402

# REF-A's SIGReg target: the predictor outputs ONLY. The frozen DINO features
# carry no anti-collapse burden (they cannot collapse — they are data), so REF-A
# constrains only the predictor's own outputs. This is the SAME shared
# flagship_loss with this one argument flipped from the flagship's
# "full_relaxed" — the machine-checkable difference the parity test pins.
SIGREG_VARIANT = "pred_only"


# --------------------------------------------------------------------------- #
# Dataset — DINO feature windows + the SAME flagship strategic/tactical labels  #
# --------------------------------------------------------------------------- #
class FlagshipFeatureWindowDataset(FeatureWindowDataset):
    """FeatureWindowDataset EXTENDED with the flagship's strategic/tactical
    pseudo-labels, so the shared 4-brain loss reads the SAME label fields the
    flagship's FlagshipWindowDataset emits — only the input is DINO features:

        nav_cmd       [] long  — refb_labels.nav_command at the last window pose
        nav_valid     [] bool  — False when < NAV_MIN_STEPS of future exist
        route_target  [] long  — the same 3-way derivation, aux-CE target
        maneuver_label [] long — kinematic maneuver class at ``maneuver_h``

    Derivations are the SAME refb_labels calls the flagship dataset uses (nav
    command from the full episode's future heading; maneuver from pose_last vs
    future_poses[maneuver_h-1]) — the label axis is source-agnostic. Poses come
    from the feature files (dino_precompute stores the odometry (x,y,yaw,v))."""

    def __init__(self, episodes: list[dict], window: int, max_horizon: int,
                 maneuver_h: int):
        super().__init__(episodes, window, max_horizon)
        assert maneuver_h <= max_horizon, (maneuver_h, max_horizon)
        self.maneuver_h = maneuver_h

    def __getitem__(self, i: int):
        item = super().__getitem__(i)          # feats/actions/future_*/pose_last
        e_i, t = self.index[i]
        poses = self.episodes[e_i]["poses"]
        cmd, valid = refb_labels.nav_command(poses, t + self.window - 1)
        item["nav_cmd"] = torch.tensor(cmd, dtype=torch.long)
        item["nav_valid"] = torch.tensor(valid)
        item["route_target"] = torch.tensor(refb_labels.route_target(cmd),
                                             dtype=torch.long)
        p_last, p1 = item["pose_last"], item["future_poses"][self.maneuver_h - 1]
        item["maneuver_label"] = refb_labels.classify_maneuver(
            p_last[2], p1[2], p_last[3], p1[3]).long()
        return item


# --------------------------------------------------------------------------- #
# Synthetic DINO-feature episodes (--data toy: CI / CPU dry run, no cache)      #
# --------------------------------------------------------------------------- #
def _toy_poses(T: int, dt: float = 0.1, v0: float = 8.0, yaw_rate: float = 0.0,
               accel: float = 0.0) -> Tensor:
    """Unicycle odometry (x, y, yaw, v) [T, 4] — the contract pose width, so the
    maneuver/nav label derivations and the metric-grounding targets exercise."""
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _toy_feature_episode(eid: int, T: int, n_tokens: int, d_dino: int,
                         yaw_rate: float, accel: float) -> dict:
    """One synthetic dino_precompute-shaped episode: random fp16 token grids +
    unicycle poses + (steer, accel) actions. Distinct yaw_rate/accel per episode
    so the pseudo-labels (lane_keep / turn_* / accelerate / brake_stop) vary."""
    g = torch.Generator().manual_seed(100 + eid)
    feats = torch.randn(T, n_tokens, d_dino, generator=g).half()
    poses = _toy_poses(T, yaw_rate=yaw_rate, accel=accel)
    actions = torch.tensor([[yaw_rate, accel]], dtype=torch.float32).repeat(T, 1)
    return {"feats_fp16": feats, "actions": actions, "poses": poses,
            "episode_id": eid}


def _toy_episodes(n: int, T: int, n_tokens: int, d_dino: int) -> list[dict]:
    variety = [(0.0, 0.0), (0.06, 0.0), (-0.06, 0.0), (0.0, -1.2), (0.0, 1.2)]
    return [_toy_feature_episode(i, T, n_tokens, d_dino, *variety[i % len(variety)])
            for i in range(n)]


def build_datasets(cfg, plan, args) -> tuple:
    """Train/val feature-window datasets carrying the flagship label fields.
    ``--data toy`` = procedural feature episodes (CI / no-cache dry run);
    ``--data feats`` = the dino_precompute *train*/*val* feature dirs."""
    w, mh, man_h = cfg.predictor.window, plan.max_horizon, plan.maneuver_h
    if args.data == "toy":
        # Long enough that early windows clear NAV_MIN_STEPS -> route CE fires.
        T = w + mh + refb_labels.NAV_MIN_STEPS + 24
        n = max(args.episodes or 6, 4)
        n_val = max(1, n // 5)
        tr = _toy_episodes(n - n_val, T, args.n_tokens, args.d_dino)
        va = _toy_episodes(n_val, T, args.n_tokens, args.d_dino)
    else:
        assert args.data_root, "--data-root required for --data feats"
        tr, _ = load_feature_episodes(args.data_root, "*train*", args.episodes)
        try:
            va, _ = load_feature_episodes(args.data_root, "*val*",
                                          min(args.episodes or 8, 8))
        except AssertionError:
            va = []
    ds_tr = FlagshipFeatureWindowDataset(tr, w, mh, man_h)
    ds_va = (FlagshipFeatureWindowDataset(va, w, mh, man_h) if va else None)
    return tr, ds_tr, ds_va


def _health(states: Tensor) -> dict:
    with torch.no_grad():
        flat = states.detach().float().reshape(-1, states.shape[-1])
        s = torch.linalg.svdvals(flat - flat.mean(0))
        p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
        erank = float(torch.exp(-(p * p.log()).sum()))
        dim_std = float(flat.std(0).mean())
    return {"erank": round(erank, 1), "dim_std": round(dim_std, 5)}


def _grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm() ** 2)
    return sq ** 0.5


# --------------------------------------------------------------------------- #
# Train                                                                        #
# --------------------------------------------------------------------------- #
def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    use_amp = (not args.no_amp) and device == "cuda"

    cfg = refa4b_smoke_config() if args.config == "smoke" else refa4b_config()
    if args.rollout_k is not None:
        cfg.train.rollout_k = args.rollout_k

    plan = horizon_plan(cfg, op_fwd_k=args.op_fwd_k, tac_fwd_k=args.tac_fwd_k,
                        str_fwd_k=args.str_fwd_k)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_eps, ds_train, ds_val = build_datasets(cfg, plan, args)
    assert len(ds_train) >= args.batch_size, \
        f"only {len(ds_train)} windows for batch {args.batch_size} — add episodes"
    dl = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                    drop_last=True, num_workers=args.workers,
                    persistent_workers=bool(args.workers))
    print(f"[refa4b] {len(train_eps)} episodes / {len(ds_train)} train windows, "
          f"window {cfg.predictor.window}, max_h {plan.max_horizon}, "
          f"needed_fut {plan.needed_fut}, batch {args.batch_size}x{args.accum}, "
          f"sigreg={SIGREG_VARIANT}", flush=True)

    # The 4-brain REF-A: the SHARED brains on the frozen-DINO grid adapter.
    model = RefAModel.from_stack_config(cfg, n_tokens=args.n_tokens,
                                        d_dino=args.d_dino).to(device)
    assert model.adapter_kind == "grid"
    grounding = build_grounding(model.state_dim, device=device)
    weights = LossWeights(
        pred=args.pred_weight, tacpred=args.tacpred_weight, roll=args.roll_weight,
        goal=args.goal_weight, wp=args.wp_weight, man=args.man_weight,
        route=args.route_weight, invdyn=args.invdyn_weight, fwd=args.fwd_weight,
        sigreg=cfg.loss.sigreg.weight, inv=cfg.loss.inv_dyn_weight)
    # Two named param groups (REF-A item 4): the trainable adapter gets a 10x-
    # longer LR warmup + its own gnorm monitor; the predictor/policy/grounding
    # params share the predictor warmup (grounding heads via extra_params).
    opt = torch.optim.AdamW(
        param_groups(model, args.lr, list(grounding.parameters())),
        lr=args.lr, betas=cfg.train.betas, weight_decay=args.weight_decay)
    warm = {"predictor": args.warmup, "adapter": args.warmup * 10}
    all_params = list(model.parameters()) + list(grounding.parameters())

    def param_table():
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {
            "operative": c(model.predictor) + c(model.inv_dyn),
            "tactical_pred": c(model.tactical_pred) if model.tactical_pred else 0,
            "tactical_policy": c(model.tactical_policy),
            "strategic_policy": c(model.strategic_policy),
            "encoder_adapter": c(model.standardizer) + c(model.adapter),
            "grounding_heads": c(grounding),
            "total_model": c(model),
            "total_trainable": sum(p.numel() for p in all_params
                                   if p.requires_grad),
        }

    # Resume (grounding heads travel under their own key; standardizer stats
    # ALWAYS come from the checkpoint when one exists — never recomputed, item 1).
    ckpt_path = out_dir / "ckpt.pt"
    step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        grounding.load_state_dict(ck["grounding"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] resuming at step {step} (stored standardizer stats "
              f"reused)", flush=True)
    else:
        t_fit = time.perf_counter()
        model.standardizer.fit(ep["feats_fp16"] for ep in train_eps)
        print(f"[refa4b] standardizer fitted ONCE over the train corpus "
              f"({time.perf_counter() - t_fit:.1f}s) — stats frozen", flush=True)

    ptab = param_table()
    (out_dir / "config.json").write_text(json.dumps({
        "arch": "refa-4b", "config": args.config, "cfg": cfg.to_json(),
        "encoder": "frozen-DINO adapter (grid)", "sigreg_variant": SIGREG_VARIANT,
        "n_tokens": args.n_tokens, "d_dino": args.d_dino, "data": args.data,
        "data_root": args.data_root,
        "horizon_plan": {"level_cfg": {k: [list(h), fk] for k, (h, fk)
                                       in plan.level_cfg.items()},
                         "goal_h": plan.goal_h, "maneuver_h": plan.maneuver_h,
                         "needed_fut": plan.needed_fut,
                         "max_horizon": plan.max_horizon},
        "weights": vars(weights), "pose_scale": args.pose_scale,
        "param_breakdown": ptab,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[init] trainable {ptab['total_trainable']/1e6:.2f}M "
          f"(model {ptab['total_model']/1e6:.2f}M + grounding "
          f"{ptab['grounding_heads']/1e6:.3f}M) | {json.dumps(ptab)}", flush=True)

    model.train()
    grounding.train()
    data_iter = iter(dl)
    accum = max(1, args.accum)
    t_data = t_step = 0.0
    logf = (out_dir / "train_log.jsonl").open("a")
    t0 = time.time()
    last_log: dict = {}

    def save_ckpt(s):
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(),
                    "grounding": grounding.state_dict(),
                    "opt": opt.state_dict(), "step": s}, tmp)
        tmp.replace(ckpt_path)
        print(f"[ckpt] saved at step {s} -> {ckpt_path}", flush=True)

    while step < args.steps:
        lrs = {}
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, args.steps, warm[pg["name"]], args.lr)
            lrs[pg["name"]] = pg["lr"]
        opt.zero_grad(set_to_none=True)
        t_s0 = time.perf_counter()
        log: dict = {}
        gn_adapter = 0.0
        for _micro in range(accum):
            t_d0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            t_data += time.perf_counter() - t_d0
            feats = batch["feats"].to(device)             # frozen (requires_grad F)
            fut_feats = batch["future_feats"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                states = model.encode_window(feats)                   # [B, W, S]
                fut_states = model.encode_window(fut_feats[:, plan.needed_fut])
                total, log, parts = flagship_loss(
                    model, grounding, batch, states, fut_states, plan, cfg,
                    weights=weights, sigreg_variant=SIGREG_VARIANT,
                    sigreg_free_dims=0, pose_scale=args.pose_scale,
                    fwd_step_weight=args.fwd_step_weight, device=device)
            (total / accum).backward()
            gn_adapter += _grad_norm(model.adapter.parameters()) / accum
        if step == 0:
            n_sig = args.batch_size * len(cfg.predictor.horizons)
            if n_sig < 256:
                print(f"WARNING: pred-only SIGReg sees {n_sig} samples/step "
                      f"(< 256 F-2 floor); raise --batch-size (>= "
                      f"{-(-256 // len(cfg.predictor.horizons))}).", flush=True)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % args.ckpt_every == 0:
            save_ckpt(step)

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                hs = model.encode_window(batch["feats"].to(device))
            row = {"step": step, "loss": float(total.item()),
                   "data_s": round(t_data, 1), "step_s": round(t_step, 1)}
            row.update(log)                       # every shared flagship column
            row.update(_health(hs))
            # REF-A stability monitors (item 4 + collapse watch).
            row["adapter_std"] = round(model.adapter_dim_std(hs), 5)
            row["gnorm_adapter"] = round(gn_adapter, 4)
            row["lr_pred"] = round(lrs["predictor"], 8)
            row["lr_adapter"] = round(lrs["adapter"], 8)
            t_data = t_step = 0.0
            last_log = row
            line = json.dumps(row)
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()
        step += 1

    save_ckpt(step - 1)
    logf.close()
    summary = {"done": True, "final_step": step - 1, "final": last_log,
               "wallclock_s": round(time.time() - t0, 1),
               "sigreg_variant": SIGREG_VARIANT, "param_breakdown": ptab,
               "out": str(ckpt_path)}
    # Light val row (REAL-only val dir / toy val split), if present.
    if ds_val is not None and len(ds_val) > 0:
        model.eval()
        grounding.eval()
        with torch.no_grad():
            vb = torch.utils.data.default_collate(
                [ds_val[i] for i in range(min(args.batch_size, len(ds_val)))])
            vstates = model.encode_window(vb["feats"].to(device))
            vfut = model.encode_window(
                vb["future_feats"][:, plan.needed_fut].to(device))
            _, vlog, _ = flagship_loss(
                model, grounding, vb, vstates, vfut, plan, cfg, weights=weights,
                sigreg_variant=SIGREG_VARIANT, sigreg_free_dims=0,
                pose_scale=args.pose_scale, fwd_step_weight=args.fwd_step_weight,
                device=device)
        summary["val"] = {k: vlog[k] for k in ("pred", "goal", "wp", "man",
                                               "route", "ground", "g_op_fwd_ade_m")
                          if k in vlog}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str),
                                          encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("done", "final_step",
                                              "wallclock_s")}), flush=True)
    print("REFA4B_DONE", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["feats", "toy"], default="feats",
                    help="feats = dino_precompute *train*/*val* feature dirs; "
                         "toy = procedural feature episodes (CI / no-cache)")
    ap.add_argument("--data-root", default=None,
                    help="dino_precompute output root (contains *train*/*val* "
                         "feature dirs + META.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", choices=["refa4b", "smoke"], default="refa4b")
    ap.add_argument("--n-tokens", type=int, default=256,
                    help="DINO token-grid count (256 for DINOv2-B/14 @224; must "
                         "be a square divisible by the readout grid)")
    ap.add_argument("--d-dino", type=int, default=768, help="DINO feature dim")
    ap.add_argument("--episodes", type=int, default=0,
                    help="max episodes per split (0 = all); toy: #episodes")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=128,
                    help=">=86 keeps pred-only SIGReg above the 256-sample floor "
                         "(len(horizons) preds per window)")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000,
                    help="predictor LR warmup steps; the adapter gets 10x (item 4)")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--rollout-k", type=int, default=None,
                    help="K-step recursive rollout (D-027 default 4; default cfg)")
    # per-level grounding rollout horizons (op fine / tac 2 s / str long)
    ap.add_argument("--op-fwd-k", type=int, default=4)
    ap.add_argument("--tac-fwd-k", type=int, default=16)
    ap.add_argument("--str-fwd-k", type=int, default=20)
    ap.add_argument("--fwd-step-weight", type=float, default=0.5)
    ap.add_argument("--pose-scale", type=float, default=10.0)
    # loss weights (identical defaults to the flagship trainer)
    ap.add_argument("--pred-weight", type=float, default=1.0)
    ap.add_argument("--tacpred-weight", type=float, default=0.5)
    ap.add_argument("--roll-weight", type=float, default=0.5)
    ap.add_argument("--goal-weight", type=float, default=0.5)
    ap.add_argument("--wp-weight", type=float, default=1.0)
    ap.add_argument("--man-weight", type=float, default=0.5)
    ap.add_argument("--route-weight", type=float, default=0.5)
    ap.add_argument("--invdyn-weight", type=float, default=2.0)
    ap.add_argument("--fwd-weight", type=float, default=1.0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
