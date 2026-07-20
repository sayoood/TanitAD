"""Self-describing checkpoint loading for the local eval/gate harnesses.

Speed-input checkpoints (``StackConfig.speed_input`` — the PROVEN v0-as-3rd-
operative-action-channel fix; flagship-speed 0.628 m vs nospeed 2.918 m) carry
``predictor.act_emb.0.weight`` of shape ``[d, 3]`` plus widened
``tactical_pred.act_emb`` / ``inv_dyn`` tensors. The committed eval harnesses
used to build the WorldModel from a stock config (``action_dim=2``) and
strict-load, which CRASHES on such ckpts (size mismatch) — and relaxing to
``strict=False`` would silently leave those tensors random-init (garbage
numbers). Fix (flagship-v2 correctness review, HIGH finding): make the
CHECKPOINT self-describing —

* :func:`ckpt_action_dim` infers the trained ``action_dim``, preferring the
  run's saved ``config.json`` next to the ckpt (``train_flagship4b`` writes
  ``speed_input`` / ``predictor.action_dim`` there) with the ``act_emb``
  weight shape as the authoritative fallback (it works for pod-side v1 ckpts
  that predate the config flag, and on any disagreement the weights win —
  a strict load must match the tensors, not the paperwork);
* :func:`adapt_config_action_dim` rebuilds the eval config via
  ``dataclasses.replace`` (the trainer's exact wiring path) BEFORE the model
  is built, so loading stays ``strict=True`` — the point is to build the
  RIGHT shape, never to relax strictness;
* :func:`append_speed_channel` appends v0 constant-expanded over the action
  window AND the future actions. v0 = ``pose_last[:, 3] / SPEED_SCALE`` — the
  t=0 (last-input-frame) speed ONLY, never a future speed (leakage-safe) —
  the exact ``tanitad.train.flagship_losses`` training contract, mirroring
  ``experiments/reset-speed4b/eval_grounded_rollout_4b_speed.py``.

Plain ``action_dim=2`` checkpoints take the byte-identical pre-change path:
the config is returned untouched and no channel is appended.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from torch import Tensor

# MUST match tanitad/train/flagship_losses.py: v0 = pose_last[:, 3] / 10.0
# (contract pinned by experiments/reset-speed4b/eval_grounded_rollout_4b_speed.py)
SPEED_SCALE = 10.0

_ACT_KEY = "predictor.act_emb.0.weight"


def state_dict_of(ck) -> dict:
    """The model state_dict of a raw or wrapped (``{"model": ...}``) ckpt."""
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def _action_dim_from_config_json(ckpt_path) -> int | None:
    """``predictor.action_dim`` (or ``speed_input``) from the run's saved
    ``config.json`` next to the ckpt. None when absent or unreadable — the
    caller falls back to the weight shape."""
    if ckpt_path is None:
        return None
    try:
        cj = Path(ckpt_path).parent / "config.json"
        if not cj.is_file():
            return None
        meta = json.loads(cj.read_text(encoding="utf-8"))
        cfg = meta.get("cfg", meta) if isinstance(meta, dict) else meta
        if isinstance(cfg, str):          # trainers embed StackConfig.to_json()
            cfg = json.loads(cfg)
        if not isinstance(cfg, dict):
            return None
        pred = cfg.get("predictor")
        if isinstance(pred, dict) and "action_dim" in pred:
            return int(pred["action_dim"])
        if "speed_input" in cfg:
            return 3 if cfg["speed_input"] else 2
    except Exception:
        return None
    return None


def ckpt_action_dim(ck, ckpt_path=None, default: int = 2) -> tuple[int, str]:
    """Infer the operative ``action_dim`` a checkpoint was trained with.

    Returns ``(action_dim, source)``; ``source`` is ``"config.json"``,
    ``"weights"`` or ``"default"``. The ``act_emb`` weight shape is
    authoritative: when ``config.json`` disagrees with the tensors (or is
    absent — e.g. the pod-side v1 ckpts), the weights win.
    """
    sd = state_dict_of(ck)
    w = sd.get(_ACT_KEY) if isinstance(sd, dict) else None
    from_w = int(w.shape[1]) if torch.is_tensor(w) and w.ndim == 2 else None
    from_cfg = _action_dim_from_config_json(ckpt_path)
    if from_cfg is not None and (from_w is None or from_w == from_cfg):
        return from_cfg, "config.json"
    if from_w is not None:
        return from_w, "weights"
    return default, "default"


def adapt_config_action_dim(cfg, action_dim: int):
    """``dataclasses.replace`` the operative predictor (+ ``tactical_pred``
    when the model has one) onto ``action_dim`` BEFORE the WorldModel is
    built — the trainer's exact wiring path (``train_flagship4b.py``), so
    ``act_emb`` / ``inv_dyn`` are created at the trained shape and loading
    stays strict. Returns ``cfg`` UNTOUCHED when the dims already match
    (byte-identical 2-ch path)."""
    if cfg.predictor.action_dim == action_dim:
        return cfg
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=action_dim)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred,
                                                action_dim=action_dim)
    if hasattr(cfg, "speed_input"):
        cfg.speed_input = (action_dim == 3)
    return cfg


def build_world_from_ckpt(cfg, ck, ckpt_path=None):
    """Build a :class:`~tanitad.models.fourbrain.WorldModel` at the ckpt's
    trained ``action_dim``, then STRICT-load it.

    Returns ``(world, speed_input, source)``. 2-channel checkpoints leave
    ``cfg`` untouched and load exactly as before this fix existed.
    """
    from tanitad.models.fourbrain import WorldModel
    action_dim, source = ckpt_action_dim(ck, ckpt_path)
    cfg = adapt_config_action_dim(cfg, action_dim)
    world = WorldModel(cfg)
    world.load_state_dict(state_dict_of(ck))       # strict=True — right shape
    speed_input = action_dim >= 3
    if speed_input:
        print(f"[ckpt-compat] speed-input ckpt detected (action_dim="
              f"{action_dim}, source={source}) — the constant v0 action "
              f"channel will be appended at eval", flush=True)
    return world, speed_input, source


def append_speed_channel(actions: Tensor, v0: Tensor) -> Tensor:
    """Append ``v0`` ``[B, 1]`` (already divided by :data:`SPEED_SCALE`)
    constant-expanded over the time axis as the extra action channel:
    ``[B, K, 2] -> [B, K, 3]``. Exactly the training-side expansion in
    ``flagship_losses`` — the same t=0 v0 for the window and every future
    step (never a future speed)."""
    return torch.cat(
        [actions, v0.unsqueeze(1).expand(-1, actions.shape[1], -1)], dim=-1)
