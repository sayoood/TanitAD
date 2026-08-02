# SPDX-License-Identifier: Apache-2.0
"""TanitAD driver for AlpaSim — our world model driving closed-loop.

Implements AlpaSim's ``BaseTrajectoryModel`` so the runtime can call our stack exactly the way it
calls Alpamayo/VaVAM/Transfuser. AlpaSim asks for a trajectory each tick; we answer with the
operative planner's waypoints.

⛔ THREE CONTRACT SEAMS THAT WILL SILENTLY CORRUPT A RUN IF GOT WRONG — each is handled explicitly:

1. **GEOMETRY.** Our encoder RAISES if the input raster does not match its declared frame (its
   positional embedding is sized for it). AlpaSim delivers whatever the camera config produces, so
   every frame is resized to the model's trained geometry (176x624 sub-frame at 117 deg for v5f,
   256x256 for the v1 line) before it reaches the encoder.

2. **THE SPEED CHANNEL.** v1-line models take v0 as a THIRD action channel, scaled by
   SPEED_SCALE = 10.0. That constant is a hard contract with the eval path; get it wrong and the
   checkpoint decodes garbage while still producing plausible-looking numbers.

3. **FRAME CONVENTION.** AlpaSim wants ``trajectory_xy`` in the RIG frame, x forward / y left, plus
   per-step headings. Our operative readout emits ego-frame displacements accumulated through
   SE(2) — the same convention — so no rotation is applied, and that is asserted rather than
   assumed.

⚠️ WINDOW WARM-UP: our predictor needs W past frames. AlpaSim starts with none, so the first
frames are self-padded (the current frame repeated). Those ticks are marked in the telemetry
because a warm-up trajectory is not a model result and must not enter a scored aggregate.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    _e = os.path.expanduser(_p)
    if _e not in sys.path:
        sys.path.insert(0, _e)

from .base import BaseTrajectoryModel, ModelPrediction, PredictionInput

SPEED_SCALE = 10.0          # hard contract with eval_grounded_rollout_4b_speed.py
DT = 0.1                    # 10 Hz


class TanitADModel(BaseTrajectoryModel):
    """Our 4-brain world model as an AlpaSim driver policy."""

    def __init__(self, world, step_readout, cfg, device, camera_ids,
                 frame_hw=(176, 624), horizon=20, speed_input=True):
        self.world = world
        self.step_readout = step_readout
        self.cfg = cfg
        self.device = device
        self.camera_ids = camera_ids
        self.H, self.W = frame_hw
        self.horizon = horizon
        self.speed_input = speed_input
        self.window = cfg.predictor.window
        self._hist: list[np.ndarray] = []
        self.n_ticks = 0
        self.n_warmup_ticks = 0

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, model_cfg: Any, device: torch.device,
                    camera_ids: list[str], **kw) -> "TanitADModel":
        import dataclasses
        from types import SimpleNamespace

        from tanitad.config import flagship4b_config
        from tanitad.models.fourbrain import WorldModel
        from tanitad.models.metric_dynamics import StepDisplacementReadout
        from train_flagship_v4 import resolve_v2_frames

        ckpt = getattr(model_cfg, "checkpoint_path", None) or model_cfg["checkpoint_path"]
        geom = getattr(model_cfg, "geometry", None) or model_cfg.get("geometry", "w120")

        cfg = flagship4b_config()
        if geom == "w120":                       # v5-line: wide FOV sub-frame
            ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                                 projection="cylindrical", v2_subframe="176x624",
                                 f_ref=None)
            _, tf = resolve_v2_frames(ns, cfg, label="alpasim_driver")
            hw = (tf.height, tf.width)
        else:                                    # v1-line: 256x256 pinhole
            hw = (cfg.encoder.image_size, cfg.encoder.image_size)

        cfg.speed_input = True
        cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
        if getattr(cfg, "tactical_pred", None) is not None:
            cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

        world = WorldModel(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        # ⛔ STRICT: a silently partial load produces plausible garbage. A mismatch is a finding.
        world.load_state_dict(ck["model"])
        world = world.to(device).eval()

        sr = StepDisplacementReadout(world.state_dim)
        if "step_readout" in ck:
            sr.load_state_dict(ck["step_readout"])
        elif "grounding" in ck:
            from tanitad.train.flagship_losses import build_grounding
            g = build_grounding(world.state_dim)
            g.load_state_dict(ck["grounding"])
            sr = g.step["op"]
        sr = sr.to(device).eval()
        print(f"[tanitad-driver] loaded {ckpt} geom={geom} frame={hw} "
              f"window={cfg.predictor.window}", flush=True)
        return cls(world, sr, cfg, device, camera_ids, frame_hw=hw)

    # ------------------------------------------------------------------ frames
    def _prep(self, img: np.ndarray) -> np.ndarray:
        """HWC uint8 RGB -> the model's trained raster, channel-first float."""
        x = self._resize_and_center_crop(img, self.H, self.W)
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def _window(self, frame: np.ndarray) -> tuple[torch.Tensor, bool]:
        """Maintain the W-frame history. Returns (tensor, is_warmup)."""
        self._hist.append(frame)
        if len(self._hist) > self.window:
            self._hist = self._hist[-self.window:]
        warm = len(self._hist) < self.window
        hist = ([self._hist[0]] * (self.window - len(self._hist))) + self._hist \
            if warm else self._hist
        w = np.stack(hist)                                  # [W, C, H, W]
        t = torch.from_numpy(w).to(self.device).float()
        C = self.cfg.encoder.in_channels
        if t.shape[1] < C:                                  # 3ch RGB -> 9ch stack
            t = t.repeat(1, C // t.shape[1] + 1, 1, 1)[:, :C]
        if t.max() > 1.5:
            t = t / 255.0
        return t.unsqueeze(0), warm

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, model_input: PredictionInput) -> ModelPrediction:
        cam = model_input.camera_images
        key = self.camera_ids[0] if self.camera_ids else next(iter(cam))
        frames = cam[key]
        img = frames[-1].image if isinstance(frames, (list, tuple)) else frames.image
        fw, warm = self._window(self._prep(img))
        self.n_ticks += 1
        self.n_warmup_ticks += int(warm)

        v0 = float(model_input.speed)
        states = self.world.encode_window(fw)

        # actions: (steer, accel, v0/SPEED_SCALE) — the v1 speed-channel contract
        a = torch.zeros(1, self.window, self.cfg.predictor.action_dim, device=self.device)
        if self.cfg.predictor.action_dim >= 3:
            a[..., 2] = v0 / SPEED_SCALE

        from tanitad.models.metric_dynamics import (accumulate_se2,
                                                    rollout_transitions)
        trans = rollout_transitions(self.world.predictor, states, a, self.horizon)
        dpose = torch.stack([self.step_readout(trans[j][0], trans[j][1])
                             for j in range(self.horizon)], dim=1)      # [1,T,3]
        wp = accumulate_se2(dpose)[0].float().cpu().numpy()             # [T,2] ego frame

        # AlpaSim wants rig-frame x-forward/y-left — the SAME convention our readout emits,
        # so no rotation. Headings come from the path tangent via the base helper.
        headings = self._compute_headings_from_trajectory(wp)
        note = "WARMUP — window self-padded, NOT a scored tick" if warm else None
        return ModelPrediction(trajectory_xy=wp.astype(np.float32),
                               headings=headings.astype(np.float32),
                               reasoning_text=note)
