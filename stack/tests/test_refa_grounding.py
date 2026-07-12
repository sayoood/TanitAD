"""REF-A metric-dynamics grounding tests (flagship B1 parity), CPU / synthetic.

Pins the required checks for wiring the action-conditioned metric-dynamics
grounding (scripts/finetune_traj.py --mode dynamics) into the REF-A trainer:

  (1) the metric-inverse-dynamics loss ALONE puts nonzero gradient on the
      ADAPTER params, while the frozen DINO feature inputs stay grad-free
      (requires_grad False, .grad None) — the grounding shapes the trainable
      adapter+predictor, never the frozen features (the inherent, correct
      main-vs-REF-A asymmetry: from-scratch encoder vs frozen-DINO features);
  (2) the forward-metric-consistency SE(2) accumulation reproduces the GT ego
      waypoints on a synthetic constant-velocity episode (geometry correct),
      AND the trainer's REUSE of the K-step _rollout latents is byte-identical
      to metric_dynamics.rollout_decode (no double predictor roll);
  (3) one grounded training step is finite across ALL components (SSL core +
      metric_invdyn + fwd) with grads reaching adapter AND predictor;
  (4) a vanilla RefAModel still loads ckpt["model"] from a grounded run — the
      metric heads are saved under their own separate keys.

Runs on synthetic dino_precompute feature files that carry poses (odometry
(x,y,yaw,v)), so the metric-Δpose targets are exercised end-to-end.
"""

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refa_train  # noqa: E402  (scripts/refa_train.py)

from tanitad.models.metric_dynamics import (StepDisplacementReadout,  # noqa: E402
                                            accumulate_se2, gt_ego_waypoints,
                                            gt_step_dposes, rollout_decode)
from tanitad.refs.refa import RefAModel  # noqa: E402

D, N_TOK, T_EP = 768, 8, 28


def _cv_poses(T: int, v: float = 3.0, dt: float = 1.0) -> torch.Tensor:
    """Constant-velocity straight-ahead odometry: x advances, y/yaw 0, speed v.

    Contract pose = (x, y, yaw, v) [T, 4] — the real REF-A precompute width."""
    t = torch.arange(T, dtype=torch.float32)
    return torch.stack([v * dt * t, torch.zeros(T), torch.zeros(T),
                        torch.full((T,), v)], dim=-1)


def _make_feature_root(tmp_path, n_train=4, n_val=1, T=T_EP, n_tok=N_TOK):
    """Synthetic dino_precompute output WITH poses (constant-velocity episodes,
    distinct speeds so the metric targets vary across episodes)."""
    torch.manual_seed(0)
    for split, n in (("train", n_train), ("val", n_val)):
        d = tmp_path / f"toy-{split}-cache-dinov2-b14"
        d.mkdir()
        for i in range(n):
            torch.save({"feats_fp16": torch.randn(T, n_tok, D).half(),
                        "actions": torch.randn(T, 2) * 0.1,
                        "poses": _cv_poses(T, v=2.0 + 0.5 * i),
                        "episode_id": i}, d / f"ep_{i:05d}.pt")
    return tmp_path


def _model(adapter_kind="pool", n_tokens=N_TOK) -> RefAModel:
    return RefAModel(refa_train.smoke_pred_config(), sigreg_slices=32,
                     adapter_kind=adapter_kind, n_tokens=n_tokens)


def _fit(model, root):
    eps, _ = refa_train.load_feature_episodes(str(root), "*train*")
    model.standardizer.fit(ep["feats_fp16"] for ep in eps)


def _batch(root, model, rollout_k=4, batch=4):
    max_h = max(max(model.pred_cfg.horizons), rollout_k)
    eps, _ = refa_train.load_feature_episodes(str(root), "*train*")
    ds = refa_train.FeatureWindowDataset(eps, model.pred_cfg.window, max_h)
    return default_collate([ds[i] for i in range(batch)])


def _gsum(params) -> float:
    return sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)


# --------------------------------------------------------------------------- #
# (1) metric-inverse-dynamics gradient reaches the ADAPTER; features grad-free #
# --------------------------------------------------------------------------- #
def test_metric_invdyn_grad_reaches_adapter_only(tmp_path):
    root = _make_feature_root(tmp_path)
    model = _model()
    _fit(model, root)
    heads = refa_train.build_metric_heads(model.state_dim)
    batch = _batch(root, model)

    out = refa_train.compute_losses(model, batch, rollout_k=4, metric_heads=heads)
    model.zero_grad()
    for h in heads.values():
        h.zero_grad()
    # backward the metric-inverse-dynamics component ALONE (not the total loss)
    out["metric_invdyn"].backward()

    assert _gsum(model.adapter.parameters()) > 0, \
        "metric-invdyn loss did not reach the adapter"
    assert _gsum(heads["metric_invdyn"].parameters()) > 0
    # frozen DINO features are grad-free end-to-end (nothing to ground there)
    assert batch["feats"].requires_grad is False and batch["feats"].grad is None
    assert batch["future_feats"].requires_grad is False
    assert batch["future_feats"].grad is None
    # metric-invdyn does NOT touch the predictor (it is a real-pair readout)
    assert _gsum(model.predictor.parameters()) == 0


def test_grid_adapter_grounding_grad(tmp_path):
    """Production uses --adapter grid: grounding must reach the grid-readout
    params too (state_dim = readout out_dim 2048)."""
    root = _make_feature_root(tmp_path, n_train=2, T=16, n_tok=256)
    model = _model(adapter_kind="grid", n_tokens=256)
    assert model.state_dim == 2048
    _fit(model, root)
    heads = refa_train.build_metric_heads(model.state_dim)
    batch = _batch(root, model, batch=4)

    out = refa_train.compute_losses(model, batch, rollout_k=4, metric_heads=heads)
    model.zero_grad()
    for h in heads.values():
        h.zero_grad()
    out["metric_invdyn"].backward()
    assert _gsum(model.adapter.parameters()) > 0, "grounding missed grid readout"


# --------------------------------------------------------------------------- #
# (2) SE(2) accumulation + the _rollout-reuse is identical to rollout_decode   #
# --------------------------------------------------------------------------- #
def test_forward_consistency_se2_matches_gt_constant_velocity():
    poses = _cv_poses(40, v=3.0)
    last = 5
    pose_last = poses[last:last + 1]                       # [1, 4]
    future = poses[last + 1:last + 1 + 8].unsqueeze(0)     # [1, 8, 4]
    gt_wp = gt_ego_waypoints(pose_last, future, range(1, 9))
    acc = accumulate_se2(gt_step_dposes(pose_last, future, 8))
    # accumulating the TRUE per-step Δposes reproduces the GT ego waypoints
    assert torch.allclose(acc, gt_wp, atol=1e-4), (acc - gt_wp).abs().max().item()
    # straight constant-velocity: ego waypoints are purely longitudinal (y ~ 0)
    assert acc[..., 1].abs().max() < 1e-4
    assert acc[0, -1, 0] > 0                               # advanced forward


def test_rollout_reuse_equals_rollout_decode(tmp_path):
    """The trainer decodes Δposes from the K-step _rollout's predicted latents
    instead of re-rolling the predictor; assert it is byte-identical to the
    flagship's metric_dynamics.rollout_decode."""
    root = _make_feature_root(tmp_path)
    model = _model()
    _fit(model, root)
    step_ro = refa_train.build_metric_heads(model.state_dim)["step_readout"]
    batch = _batch(root, model)
    K = 4
    with torch.no_grad():
        states = model.encode_window(batch["feats"])
        fut_states = model.encode_window(batch["future_feats"])
        _, roll_preds = refa_train._rollout(
            model, states, batch["actions"], fut_states,
            batch["future_actions"], K)
        prevs = [states[:, -1]] + roll_preds[:-1]
        step_dp = torch.stack([step_ro(prevs[j], roll_preds[j])
                               for j in range(K)], dim=1)
        wp_reuse = accumulate_se2(step_dp)
        wp_rd, step_rd = rollout_decode(model.predictor, states, batch["actions"],
                                        batch["future_actions"], step_ro, K)
    assert torch.allclose(wp_reuse, wp_rd, atol=1e-5)
    assert torch.allclose(step_dp, step_rd, atol=1e-5)


# --------------------------------------------------------------------------- #
# (3) one grounded training step: finite everywhere, grads reach adapter+pred  #
# --------------------------------------------------------------------------- #
def test_grounded_step_finite_and_grads(tmp_path):
    root = _make_feature_root(tmp_path)
    model = _model()
    _fit(model, root)
    heads = refa_train.build_metric_heads(model.state_dim)
    metric_params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(
        refa_train.param_groups(model, 1e-3, metric_params), lr=1e-3)
    batch = _batch(root, model)

    out = refa_train.compute_losses(model, batch, rollout_k=4, metric_heads=heads,
                                    invdyn_weight=2.0, fwd_weight=1.0)
    for key in ("loss", "pred", "roll", "inv", "sigreg", "metric_invdyn", "fwd"):
        assert torch.isfinite(out[key].detach()).all(), key
    assert math.isfinite(out["metric_de"]) and math.isfinite(out["fwd_ade"])
    assert out["fwd_ade"] > 0                     # real motion (CV episode)
    # SigReg pool is unchanged by grounding: 3 horizons + 4 rollout preds.
    assert out["n_sig"] == batch["feats"].shape[0] * 7

    opt.zero_grad()
    out["loss"].backward()
    assert _gsum(model.adapter.parameters()) > 0
    assert _gsum(model.predictor.parameters()) > 0
    assert _gsum(heads["metric_invdyn"].parameters()) > 0
    assert _gsum(heads["step_readout"].parameters()) > 0
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + metric_params, 1.0)
    opt.step()
    # frozen features remain grad-free after a full grounded step
    assert batch["feats"].grad is None and batch["future_feats"].grad is None


def test_grounding_disabled_matches_ssl_only(tmp_path):
    """metric_heads=None must leave the SSL core untouched (the existing
    test_refa contract): identical return keys, and the DETERMINISTIC SSL
    components (pred/roll/inv) bit-equal whether or not grounding is attached —
    grounding is strictly additive. (SigReg is intentionally stochastic: fresh
    random projections each call, so it is excluded from the equality check.)"""
    root = _make_feature_root(tmp_path)
    model = _model()
    _fit(model, root)
    batch = _batch(root, model)
    ssl = refa_train.compute_losses(model, batch, rollout_k=4)
    assert set(ssl) == {"pred", "roll", "inv", "sigreg", "n_sig", "states",
                        "loss"}
    grounded = refa_train.compute_losses(model, batch, rollout_k=4,
                                         metric_heads=refa_train.
                                         build_metric_heads(model.state_dim))
    for k in ("pred", "roll", "inv"):
        assert torch.equal(ssl[k], grounded[k]), k
    # grounded adds the metric terms (finite, and the readouts contribute > 0).
    mid, fwd = grounded["metric_invdyn"], grounded["fwd"]
    assert torch.isfinite(mid) and mid > 0
    assert torch.isfinite(fwd) and fwd > 0
    assert grounded["n_sig"] == ssl["n_sig"]      # SigReg pool unchanged


# --------------------------------------------------------------------------- #
# (4) a vanilla RefAModel loads a grounded ckpt (metric heads are separate)    #
# --------------------------------------------------------------------------- #
def test_vanilla_refamodel_loads_grounded_ckpt(tmp_path):
    root = _make_feature_root(tmp_path)
    out_dir = tmp_path / "run"
    argv = ["--data-root", str(root), "--out", str(out_dir), "--steps", "2",
            "--rollout-k", "4", "--batch", "4", "--lr", "1e-3",
            "--log-every", "1", "--device", "cpu", "--smoke",
            "--invdyn-weight", "2.0", "--fwd-weight", "1.0"]
    metrics = refa_train.main(argv)

    # the new metric columns are logged
    for k in ("metric_invdyn", "fwd", "metric_de", "fwd_ade"):
        assert k in metrics["final"] and math.isfinite(metrics["final"][k])

    ck = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    # metric heads saved under their OWN top-level keys, NOT inside ck["model"]
    assert "metric_invdyn" in ck and "step_readout" in ck
    assert not any(k.startswith(("metric_invdyn", "step_readout"))
                   for k in ck["model"])
    # a vanilla RefAModel loads ck["model"] strictly (no leaked head params)
    vanilla = RefAModel(refa_train.smoke_pred_config())
    vanilla.load_state_dict(ck["model"])
    # eval reads the trained StepDisplacementReadout the flagship way
    sr = StepDisplacementReadout(vanilla.state_dim)
    sr.load_state_dict(ck["step_readout"])
