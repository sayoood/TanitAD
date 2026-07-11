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
      + SigReg fp32 on PREDICTOR OUTPUTS ONLY (item 3, >=256-samples floor).
Param groups: adapter gets a 10x longer LR warmup than the predictor (item 4)
plus its own gradient-norm monitor row.

Usage (pod2):
  python scripts/refa_train.py --data-root /opt/dino_feats \
      --out /workspace/experiments/refa-30k --steps 30000
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
from tanitad.models.predictor import change_weighted_mse
from tanitad.refs.refa import RefAModel, refa_predictor_config
from tanitad.train.train_worldmodel import cosine_lr

# Loss weights follow the main trainer's operating point (train_worldmodel):
# pred 1.0, rollout 0.5*pred, inv-dyn 0.5, SigReg lambda 0.1 (LeJEPA knob).
PRED_WEIGHT = 1.0
ROLL_WEIGHT = 0.5
INV_WEIGHT = 0.5
SIGREG_WEIGHT = 0.1


def smoke_pred_config() -> PredictorConfig:
    """Tiny predictor for CI / CPU smoke runs (--smoke). Adapter space stays
    768-dim — only the predictor trunk shrinks."""
    return PredictorConfig(d_model=64, depth=2, n_heads=2, window=4,
                           horizons=(1, 2, 4), action_dim=2)


class FeatureWindowDataset(Dataset):
    """Windows over precomputed per-timestep DINO feature rows.

    A window is simply ``rows[t : t+W]`` (latest-frame features per timestep);
    ``future_feats``/``future_actions`` follow the EpisodeWindowDataset
    contract: rows ``t+W .. t+W+max_horizon``. Episodes are the raw dicts
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
                   device: str = "cpu") -> dict:
    """One forward pass -> all loss components (tensors, differentiable).

    Targets are the adapter outputs of the FUTURE feature rows (predict in
    adapter space), encoded WITH gradients (A1: no stop-grad/EMA crutch) —
    the collapse monitor + inv-dyn + SigReg-on-preds carry the stability."""
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
    return {"loss": loss, "pred": loss_pred, "roll": loss_roll,
            "inv": loss_inv, "sigreg": loss_sig,
            "n_sig": int(z_pred_all.shape[0]), "states": states}


def param_groups(model: RefAModel, lr: float) -> list[dict]:
    """Two named groups: 'adapter' (10x longer warmup, item 4) and
    'predictor' (predictor trunk + heads + inv-dyn)."""
    adapter = [p for n, p in model.named_parameters()
               if n.startswith("adapter.")]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith("adapter.")]
    assert adapter and rest
    return [{"params": rest, "lr": lr, "name": "predictor"},
            {"params": adapter, "lr": lr, "name": "adapter"}]


def _grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm() ** 2)
    return sq ** 0.5


def _save_ckpt(path: Path, model, opt, step: int) -> None:
    # atomic write: a kill mid-save must not corrupt the resume point
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)
    print(f"[ckpt] saved at step {step}", flush=True)


def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    pred_cfg = smoke_pred_config() if args.smoke else refa_predictor_config()
    model = RefAModel(pred_cfg, bottleneck=args.bottleneck).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.lr), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.05)

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
                          "inv": INV_WEIGHT, "sigreg": SIGREG_WEIGHT}},
        indent=2, default=str), encoding="utf-8")

    # Interruptible-pod resume; standardizer stats ALWAYS come from the
    # checkpoint when one exists — never recomputed (item 1).
    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
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
        out = compute_losses(model, batch, args.rollout_k, device)
        if step == 0 and out["n_sig"] < 256:
            print(f"WARNING: SigReg sees only {out['n_sig']} samples/step — "
                  f"statistically starved below ~256 (F-2 rule); increase "
                  f"--batch.", flush=True)
        out["loss"].backward()
        gn_adapter = _grad_norm(model.adapter.parameters())   # pre-clip, item 4
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % args.save_every == 0:
            _save_ckpt(ckpt_path, model, opt, step)

        if step % args.log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)  # noqa: E731
            last_log = {
                "step": step, "loss": sc(out["loss"]),
                "pred": sc(out["pred"]),
                "roll": sc(out["roll"]),
                "inv": sc(out["inv"]),
                "sigreg": sc(out["sigreg"]),
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

    _save_ckpt(ckpt_path, model, opt, step - 1)     # final resume point
    metrics = {"final": last_log, "steps": step, "device": device,
               "rollout_k": args.rollout_k,
               "n_params_trainable": sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)}
    # Light val row (REAL-only val dir from the precompute), if present.
    try:
        val_eps, _ = load_feature_episodes(args.data_root, "*val*",
                                           min(args.episodes or 8, 8))
        vds = FeatureWindowDataset(val_eps, pred_cfg.window, max_h)
        if len(vds) > 0:
            model.eval()
            with torch.no_grad():
                vb = torch.utils.data.default_collate(
                    [vds[i] for i in range(min(16, len(vds)))])
                vout = compute_losses(model, vb, args.rollout_k, device)
            metrics["val"] = {
                "pred": round(float(vout["pred"]), 5),
                "roll": round(float(vout["roll"]), 5),
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
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--bottleneck", action="store_true",
                    help="adapter GELU bottleneck variant")
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
