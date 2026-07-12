"""REF-A stage 2: train adapter + shared predictor from precomputed DINO
features (REFERENCE_ARCHITECTURES.md, stability items 1-6).

Consumes the per-episode feature files written by scripts/dino_precompute.py
({"feats_fp16" [T,256,768], "actions" [T,2], ...}) — no images, no encoder in
the training loop (item 6). Windows are consecutive feature rows (the
precompute stores latest-frame features per timestep), same window/future
contract as tanitad.data._contract.EpisodeWindowDataset.

Losses: multi-horizon prediction in ADAPTER space (change-weighted, A4)
      + K-step recursive rollout (D-027: rollout_k=4 default)
      + inverse dynamics (A5)
      + SigReg fp32 on PREDICTOR OUTPUTS ONLY (item 3, >=256-samples floor)
      + metric-dynamics grounding (parity with flagship B1, --mode dynamics):
        lambda_invdyn * metric-inverse-dynamics + lambda_fwd * forward-metric-
        consistency (tanitad.models.metric_dynamics). The metric heads live
        OUTSIDE RefAModel (saved under separate ckpt keys) so a vanilla
        RefAModel still loads ckpt["model"] and eval/driving_diagnostic read
        the trained StepDisplacementReadout the same way the flagship does.
        Grads reach the ADAPTER + predictor; the frozen DINO features carry no
        grad (nothing to ground there — the inherent, correct main-vs-REF-A
        asymmetry: from-scratch encoder vs frozen-DINO features).
Param groups: adapter gets a 10x longer LR warmup than the predictor (item 4)
plus its own gradient-norm monitor row; the metric heads share the predictor
warmup.

Usage (pod2/pod3):
  python scripts/refa_train.py --data-root /opt/dino_feats \
      --out /workspace/experiments/refa-30k --steps 30000 \
      --adapter grid --rollout-k 4 --invdyn-weight 2.0 --fwd-weight 1.0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tanitad.config import PredictorConfig
from tanitad.models.metric_dynamics import (MetricInverseDynamics,
                                            StepDisplacementReadout,
                                            accumulate_se2, gt_ego_waypoints,
                                            gt_step_dposes, relative_ego_pose,
                                            wrap_angle)
from tanitad.models.predictor import change_weighted_mse
from tanitad.refs.refa import RefAModel, refa_predictor_config
from tanitad.train.train_worldmodel import cosine_lr

# Loss weights follow the main trainer's operating point (train_worldmodel):
# pred 1.0, rollout 0.5*pred, inv-dyn 0.5, SigReg lambda 0.1 (LeJEPA knob).
PRED_WEIGHT = 1.0
ROLL_WEIGHT = 0.5
INV_WEIGHT = 0.5
SIGREG_WEIGHT = 0.1
# Metric-dynamics grounding weights (flagship B1 parity; finetune_traj defaults).
INVDYN_WEIGHT = 2.0     # lambda_invdyn: metric-inverse-dynamics
FWD_WEIGHT = 1.0        # lambda_fwd: forward-metric-consistency (rollout accum)
POSE_SCALE = 10.0       # metre normalizer -> metric losses stay O(1) under clip
FWD_STEP_WEIGHT = 0.5   # per-step Δpose anchor weight inside lambda_fwd


def smoke_pred_config() -> PredictorConfig:
    """Tiny predictor for CI / CPU smoke runs (--smoke). Adapter space stays
    768-dim — only the predictor trunk shrinks."""
    return PredictorConfig(d_model=64, depth=2, n_heads=2, window=4,
                           horizons=(1, 2, 4), action_dim=2)


class FeatureWindowDataset(Dataset):
    """Windows over precomputed per-timestep DINO feature rows.

    A window is simply ``rows[t : t+W]`` (latest-frame features per timestep);
    ``future_feats``/``future_actions``/``future_poses`` follow the
    EpisodeWindowDataset contract: rows ``t+W .. t+W+max_horizon``, with
    ``pose_last = poses[t+W-1]``. Poses (odometry ego-pose (x,y,yaw,v) stored by
    dino_precompute.py alongside the feature grids) are threaded with the SAME
    window indexing as feats/actions, so the metric-dynamics grounding targets
    align byte-for-byte with the predicted latents. Episodes are the raw dicts
    written by dino_precompute.py (mmap-loadable)."""

    def __init__(self, episodes: list[dict], window: int, max_horizon: int):
        self.window, self.max_horizon = window, max_horizon
        self.episodes = episodes
        self.index: list[tuple[int, int]] = []
        for e_i, ep in enumerate(episodes):
            t_max = ep["feats_fp16"].shape[0] - window - max_horizon
            self.index.extend((e_i, t) for t in range(t_max))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w, h = self.window, self.max_horizon
        return {
            "feats": ep["feats_fp16"][t:t + w],                    # [W,N,D] fp16
            "actions": ep["actions"][t:t + w].float(),             # [W,2]
            "future_feats": ep["feats_fp16"][t + w:t + w + h],     # [H,N,D]
            "future_actions": ep["actions"][t + w:t + w + h].float(),
            # odometry ego-pose for the metric-dynamics grounding targets:
            "future_poses": ep["poses"][t + w:t + w + h].float(),  # [H,4]
            "pose_last": ep["poses"][t + w - 1].float(),           # [4]
        }


def load_feature_episodes(data_root: str, pattern: str,
                          n: int = 0) -> tuple[list[dict], Path]:
    """Load ep_*.pt feature files from the newest dir matching ``pattern``
    (dino_precompute names dirs `<cache-name>-<encoder-tag>`). mmap keeps the
    fp16 grids on disk; windows fault in only the rows they touch."""
    root = Path(data_root)
    dirs = sorted(d for d in root.glob(pattern) if d.is_dir())
    assert dirs, f"no feature dir matching {pattern} under {root}"
    files = sorted(dirs[-1].glob("ep_*.pt"))
    if n:
        files = files[:n]
    assert files, f"no ep_*.pt files in {dirs[-1]}"
    eps = [torch.load(f, map_location="cpu", weights_only=True, mmap=True)
           for f in files]
    return eps, dirs[-1]


def build_metric_heads(state_dim: int, device: str = "cpu",
                       hidden: int = 512) -> dict:
    """Metric-dynamics heads for the flagship-B1 grounding (--mode dynamics
    parity). Kept OUTSIDE RefAModel and saved under separate ckpt keys
    ('metric_invdyn', 'step_readout') — exactly mirroring
    finetune_traj.build_heads — so ckpt['model'] stays a vanilla RefAModel
    state dict and eval / driving_diagnostic load the trained
    StepDisplacementReadout by the same 'step_readout' key the flagship uses.

    Both heads take ``state_dim`` (the ADAPTER output width: 768 pool / 2048
    grid) — the metric grounding therefore shapes the ADAPTER + predictor; the
    frozen DINO features carry no grad (nothing to ground there, the correct
    main-vs-REF-A asymmetry)."""
    return {
        "metric_invdyn":
            MetricInverseDynamics(state_dim, hidden=hidden).to(device),
        "step_readout":
            StepDisplacementReadout(state_dim, hidden=hidden).to(device),
    }


def _rollout(model: RefAModel, states: Tensor, actions: Tensor,
             fut_states: Tensor, fut_actions: Tensor,
             K: int) -> tuple[Tensor, list[Tensor]]:
    """K-step recursive rollout (train_worldmodel._rollout_loss pattern,
    2512.24497 multistep-as-augmentation): feed the 1-step prediction back
    into the window. Also returns the per-step predictions so SigReg can see
    them (they ARE predictor outputs, item 3). Future rows are materialized
    densely here, so target index for step j is simply j-1."""
    win_s, win_a = states, actions
    loss = torch.zeros((), device=states.device)
    preds: list[Tensor] = []
    for j in range(1, K + 1):
        z_hat = model.predictor(win_s, win_a)[1]
        preds.append(z_hat)
        loss = loss + (z_hat - fut_states[:, j - 1]).pow(2).mean()
        if j == K:
            break
        a_next = (fut_actions[:, j - 1] if fut_actions is not None
                  else win_a[:, -1])
        win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
        win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    return loss / K, preds


def compute_losses(model: RefAModel, batch: dict, rollout_k: int,
                   device: str = "cpu", *, metric_heads: dict | None = None,
                   mid_horizons=None, invdyn_weight: float = INVDYN_WEIGHT,
                   fwd_weight: float = FWD_WEIGHT, pose_scale: float = POSE_SCALE,
                   fwd_step_weight: float = FWD_STEP_WEIGHT) -> dict:
    """One forward pass -> all loss components (tensors, differentiable).

    Targets are the adapter outputs of the FUTURE feature rows (predict in
    adapter space), encoded WITH gradients (A1: no stop-grad/EMA crutch) —
    the collapse monitor + inv-dyn + SigReg-on-preds carry the stability.

    When ``metric_heads`` is provided (``build_metric_heads``), the flagship-B1
    metric-dynamics grounding is ADDED on top of the SSL core (which stays
    byte-for-byte identical): ``invdyn_weight * metric-inverse-dynamics`` on
    REAL adapter-latent pairs + ``fwd_weight * forward-metric-consistency`` that
    REUSES the K-step rollout's predicted latents. With ``metric_heads=None``
    (the SSL-only path exercised by test_refa) the return dict and every loss
    are unchanged."""
    feats = batch["feats"].to(device)                    # requires_grad False
    actions = batch["actions"].to(device)
    fut_feats = batch["future_feats"].to(device)
    fut_actions = batch["future_actions"].to(device)

    states = model.encode_window(feats)                  # [B, W, S]
    fut_states = model.encode_window(fut_feats)          # [B, Hmax, S]
    z_t = states[:, -1]

    preds = model.predict(states, actions)
    horizons = model.pred_cfg.horizons
    loss_pred = torch.zeros((), device=states.device)
    for k in horizons:
        target = fut_states[:, k - 1]
        prev = z_t if k == 1 else fut_states[:, k - 2]
        if model.pred_cfg.change_weighted:
            loss_pred = loss_pred + change_weighted_mse(preds[k], target, prev)
        else:
            loss_pred = loss_pred + (preds[k] - target).pow(2).mean()
    loss_pred = loss_pred / len(horizons)

    loss_roll = torch.zeros((), device=states.device)
    roll_preds: list[Tensor] = []
    if rollout_k > 1:
        loss_roll, roll_preds = _rollout(model, states, actions, fut_states,
                                         fut_actions, rollout_k)

    a_hat = model.inv_dyn(states[:, -2], states[:, -1])
    loss_inv = (a_hat - actions[:, -2]).pow(2).mean()

    # SigReg on PREDICTOR OUTPUTS ONLY (item 3) — fp32 inside SigReg; the
    # rollout's intermediate predictions count (they are predictor outputs).
    z_pred_all = torch.cat([preds[k] for k in horizons] + roll_preds)
    loss_sig = model.sigreg(z_pred_all)

    loss = (PRED_WEIGHT * loss_pred + ROLL_WEIGHT * loss_roll
            + INV_WEIGHT * loss_inv + SIGREG_WEIGHT * loss_sig)
    out = {"pred": loss_pred, "roll": loss_roll, "inv": loss_inv,
           "sigreg": loss_sig, "n_sig": int(z_pred_all.shape[0]),
           "states": states}

    # ----- ADD: metric-dynamics grounding (flagship B1 parity) --------------
    # Only the encoder axis differs between the main model and REF-A, so the
    # grounding math mirrors finetune_traj.compute_losses --mode dynamics; the
    # only substitution is ENCODER->ADAPTER (grads reach the adapter + predictor
    # + step readout; the frozen features carry no grad). Metre errors are
    # divided by ``pose_scale`` so the lambda weights operate on O(1) quantities
    # and grad-clipping does not starve the SSL gradients (the heads still emit
    # raw metres — eval reads them directly).
    if metric_heads is not None:
        mid, step_ro = metric_heads["metric_invdyn"], metric_heads["step_readout"]
        mh = list(mid_horizons) if mid_horizons is not None else list(horizons)
        ps = pose_scale
        pose_last = batch["pose_last"].to(device).float()        # [B, 4]
        future_poses = batch["future_poses"].to(device).float()  # [B, Hmax, 4]

        # (a) metric inverse dynamics on REAL adapter pairs (x_t, x_{t+k}) ->
        #     odometry metric relative ego-pose (Δx, Δy, Δyaw). Grounds ADAPTER.
        loss_mid = torch.zeros((), device=states.device)
        metric_de = 0.0
        for kh in mh:
            dpose = mid(z_t, fut_states[:, kh - 1])
            tgt = relative_ego_pose(pose_last, future_poses[:, kh - 1])
            loss_mid = loss_mid \
                + ((dpose[..., :2] - tgt[..., :2]) / ps).pow(2).mean() \
                + wrap_angle(dpose[..., 2] - tgt[..., 2]).pow(2).mean()
            metric_de += float((dpose[..., :2] - tgt[..., :2]).detach()
                               .norm(dim=-1).mean())
        loss_mid = loss_mid / len(mh)
        metric_de /= len(mh)

        # (b) forward metric consistency: REUSE the K-step rollout PREDICTED
        #     latents (rolled under TRUE actions in _rollout), decode each
        #     transition's per-step Δpose via StepDisplacementReadout, accumulate
        #     SE(2), and L2 the trajectory vs the odometry ego-trajectory. The
        #     transition pairs are (z_t, roll_preds[0]), (roll_preds[0],
        #     roll_preds[1]), ... — identical to metric_dynamics.rollout_decode,
        #     but without re-rolling the predictor (pinned in test_refa_grounding).
        if rollout_k > 1 and roll_preds:
            prevs = [z_t] + roll_preds[:-1]
            step_dp = torch.stack(
                [step_ro(prevs[j], roll_preds[j])
                 for j in range(len(roll_preds))], dim=1)         # [B, K, 3]
            pred_wp = accumulate_se2(step_dp)                     # [B, K, 2]
            gt_wp = gt_ego_waypoints(pose_last, future_poses,
                                     range(1, rollout_k + 1))
            gt_step = gt_step_dposes(pose_last, future_poses, rollout_k)
            loss_acc = ((pred_wp - gt_wp) / ps).pow(2).mean()
            loss_step = ((step_dp[..., :2] - gt_step[..., :2]) / ps).pow(2).mean() \
                + wrap_angle(step_dp[..., 2] - gt_step[..., 2]).pow(2).mean()
            loss_fwd = loss_acc + fwd_step_weight * loss_step
            fwd_ade = float((pred_wp.detach() - gt_wp).norm(dim=-1).mean())
        else:
            loss_fwd = torch.zeros((), device=states.device)
            fwd_ade = 0.0

        loss = loss + invdyn_weight * loss_mid + fwd_weight * loss_fwd
        out.update({"metric_invdyn": loss_mid, "fwd": loss_fwd,
                    "metric_de": round(metric_de, 4),
                    "fwd_ade": round(fwd_ade, 4)})

    out["loss"] = loss
    return out


def param_groups(model: RefAModel, lr: float, extra_params=()) -> list[dict]:
    """Two named groups: 'adapter' (10x longer warmup, item 4) and
    'predictor' (predictor trunk + heads + inv-dyn + any metric-dynamics heads
    passed via ``extra_params`` — they share the predictor's warmup)."""
    adapter = [p for n, p in model.named_parameters()
               if n.startswith("adapter.")]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith("adapter.")]
    rest = rest + list(extra_params)
    assert adapter and rest
    return [{"params": rest, "lr": lr, "name": "predictor"},
            {"params": adapter, "lr": lr, "name": "adapter"}]


def _grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm() ** 2)
    return sq ** 0.5


def _save_ckpt(path: Path, model, opt, step: int, metric_heads=None) -> None:
    # atomic write: a kill mid-save must not corrupt the resume point. The
    # metric heads go under their OWN keys ('metric_invdyn', 'step_readout'),
    # so ckpt['model'] stays a vanilla RefAModel state dict (eval loads it
    # unchanged; the flagship rollout eval reads ckpt['step_readout']).
    tmp = path.with_suffix(".tmp")
    blob = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    if metric_heads is not None:
        for name, h in metric_heads.items():
            blob[name] = h.state_dict()
    torch.save(blob, tmp)
    tmp.replace(path)
    print(f"[ckpt] saved at step {step}", flush=True)


def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    pred_cfg = smoke_pred_config() if args.smoke else refa_predictor_config()
    model = RefAModel(pred_cfg, bottleneck=args.bottleneck,
                      adapter_kind=args.adapter).to(device)
    # Metric-dynamics grounding heads (flagship B1 parity), kept OUTSIDE the
    # model (saved separately) but optimized WITH it in the predictor group.
    metric_heads = build_metric_heads(model.state_dim, device)
    mid_horizons = list(pred_cfg.horizons)
    metric_params = [p for h in metric_heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(param_groups(model, args.lr, metric_params),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    ground_kw = dict(metric_heads=metric_heads, mid_horizons=mid_horizons,
                     invdyn_weight=args.invdyn_weight, fwd_weight=args.fwd_weight,
                     pose_scale=args.pose_scale,
                     fwd_step_weight=args.fwd_step_weight)

    max_h = max(max(pred_cfg.horizons), args.rollout_k)
    train_eps, train_dir = load_feature_episodes(args.data_root, "*train*",
                                                 args.episodes)
    ds = FeatureWindowDataset(train_eps, pred_cfg.window, max_h)
    assert len(ds) >= args.batch, \
        f"only {len(ds)} windows for batch {args.batch} — add episodes"
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)
    print(f"[refa] train: {len(train_eps)} episodes / {len(ds)} windows "
          f"from {train_dir}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"arch": "REF-A", "pred_cfg": dataclasses.asdict(pred_cfg),
         "args": vars(args),
         "loss_weights": {"pred": PRED_WEIGHT, "roll": ROLL_WEIGHT,
                          "inv": INV_WEIGHT, "sigreg": SIGREG_WEIGHT,
                          "invdyn": args.invdyn_weight, "fwd": args.fwd_weight,
                          "pose_scale": args.pose_scale,
                          "fwd_step_weight": args.fwd_step_weight,
                          "mid_horizons": mid_horizons}},
        indent=2, default=str), encoding="utf-8")

    # Interruptible-pod resume; standardizer stats ALWAYS come from the
    # checkpoint when one exists — never recomputed (item 1).
    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for name, h in metric_heads.items():
            if name in ck:
                h.load_state_dict(ck[name])
        step = int(ck["step"]) + 1
        print(f"[resume] checkpoint found — resuming at step {step} "
              f"(stored standardizer stats reused)", flush=True)
    else:
        t_fit = time.perf_counter()
        model.standardizer.fit(ep["feats_fp16"] for ep in train_eps)
        print(f"[refa] standardizer fitted ONCE over the train corpus "
              f"({time.perf_counter() - t_fit:.1f}s) — stats frozen",
              flush=True)

    warm = {"predictor": args.warmup, "adapter": args.warmup * 10}  # item 4
    data_iter = iter(dl)
    t_data = t_step = 0.0
    last_log: dict = {}
    while step < args.steps:
        lrs = {}
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, args.steps, warm[pg["name"]], args.lr)
            lrs[pg["name"]] = pg["lr"]
        t_s0 = time.perf_counter()
        t_d0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        t_data += time.perf_counter() - t_d0

        opt.zero_grad(set_to_none=True)
        out = compute_losses(model, batch, args.rollout_k, device, **ground_kw)
        if step == 0 and out["n_sig"] < 256:
            print(f"WARNING: SigReg sees only {out['n_sig']} samples/step — "
                  f"statistically starved below ~256 (F-2 rule); increase "
                  f"--batch.", flush=True)
        out["loss"].backward()
        gn_adapter = _grad_norm(model.adapter.parameters())   # pre-clip, item 4
        # clip model + metric heads together (they are jointly optimized).
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + metric_params, 1.0))
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % args.save_every == 0:
            _save_ckpt(ckpt_path, model, opt, step, metric_heads)

        if step % args.log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)  # noqa: E731
            last_log = {
                "step": step, "loss": sc(out["loss"]),
                "pred": sc(out["pred"]),
                "roll": sc(out["roll"]),
                "inv": sc(out["inv"]),
                "sigreg": sc(out["sigreg"]),
                # metric-dynamics grounding (flagship B1): loss components +
                # the interpretable metre diagnostics (metric_de / fwd_ade).
                "metric_invdyn": sc(out["metric_invdyn"]),
                "fwd": sc(out["fwd"]),
                "metric_de": out["metric_de"],
                "fwd_ade": out["fwd_ade"],
                # collapse monitor: adapter output per-dim std (trainable
                # targets — drifting to 0 = collapse-to-easy-targets)
                "adapter_std": round(model.adapter_dim_std(out["states"]), 5),
                "gnorm": round(gnorm, 4),
                "gnorm_adapter": round(gn_adapter, 4),
                "lr_pred": lrs["predictor"], "lr_adapter": lrs["adapter"],
                "data_s": round(t_data, 1), "step_s": round(t_step, 1),
            }
            t_data = t_step = 0.0
            print(json.dumps(last_log), flush=True)
        step += 1

    _save_ckpt(ckpt_path, model, opt, step - 1, metric_heads)  # final resume point
    metrics = {"final": last_log, "steps": step, "device": device,
               "rollout_k": args.rollout_k,
               "n_params_trainable": (sum(p.numel() for p in model.parameters()
                                          if p.requires_grad)
                                      + sum(p.numel() for p in metric_params))}
    # Light val row (REAL-only val dir from the precompute), if present.
    try:
        val_eps, _ = load_feature_episodes(args.data_root, "*val*",
                                           min(args.episodes or 8, 8))
        vds = FeatureWindowDataset(val_eps, pred_cfg.window, max_h)
        if len(vds) > 0:
            model.eval()
            for h in metric_heads.values():
                h.eval()
            with torch.no_grad():
                vb = torch.utils.data.default_collate(
                    [vds[i] for i in range(min(16, len(vds)))])
                vout = compute_losses(model, vb, args.rollout_k, device,
                                      **ground_kw)
            metrics["val"] = {
                "pred": round(float(vout["pred"]), 5),
                "roll": round(float(vout["roll"]), 5),
                "metric_de": vout["metric_de"], "fwd_ade": vout["fwd_ade"],
                "adapter_std": round(model.adapter_dim_std(vout["states"]), 5)}
    except AssertionError:
        pass                                        # no val feature dir
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"done": True, "steps": step,
                      "out": str(out_dir)}), flush=True)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="dino_precompute output root (contains *train*/*val* "
                         "feature dirs + META.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--rollout-k", type=int, default=4,
                    help="K-step recursive rollout (D-027 default 4; 1=off)")
    ap.add_argument("--batch", type=int, default=64,
                    help=">=64 keeps SigReg above the 256-sample floor "
                         "(3 horizons + 4 rollout preds per window)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500,
                    help="predictor LR warmup steps; adapter gets 10x (item 4)")
    # Metric-dynamics grounding (flagship B1 parity, --mode dynamics).
    ap.add_argument("--invdyn-weight", type=float, default=INVDYN_WEIGHT,
                    help="λ_invdyn: metric-inverse-dynamics weight (def 2.0)")
    ap.add_argument("--fwd-weight", type=float, default=FWD_WEIGHT,
                    help="λ_fwd: forward-metric-consistency weight (def 1.0); "
                         "reuses the K-step rollout latents (--rollout-k)")
    ap.add_argument("--pose-scale", type=float, default=POSE_SCALE,
                    help="metre normalizer for the metric losses (heads emit "
                         "raw metres; eval reads them directly)")
    ap.add_argument("--fwd-step-weight", type=float, default=FWD_STEP_WEIGHT,
                    help="weight of the per-step Δpose anchor within λ_fwd")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--bottleneck", action="store_true",
                    help="adapter GELU bottleneck variant (pool adapter only)")
    ap.add_argument("--adapter", choices=("pool", "grid"), default="pool",
                    help="pool = v1 mean-pool; grid = stage-2b spatial "
                         "readout (Sayed review 2026-07-11 — use for all "
                         "new comparison runs)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny predictor (CI/CPU smoke)")
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
