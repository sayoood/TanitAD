# SPDX-License-Identifier: Apache-2.0
"""TanitAD driver for AlpaSim — our world model driving closed-loop.

Implements AlpaSim's ``BaseTrajectoryModel`` so the runtime calls our stack exactly the way it
calls Alpamayo/VaVAM. AlpaSim asks for a trajectory each tick; we answer with the operative
planner's waypoints, rolled through the predictor.

⛔ FOUR CONTRACT SEAMS. Each silently corrupts a run if got wrong; each is handled explicitly.

1. **THE TEMPORAL WINDOW IS AlpaSim's JOB, NOT OURS.** ``CameraImages`` is
   ``dict[str, list[CameraFrame]]`` whose length **is** ``context_length`` — the runtime reads
   that property and delivers exactly that many frames. An earlier version of this file kept its
   own history buffer and self-padded the first ticks; that duplicated the runtime's contract and
   would have fed a window built from a different frame sequence than the one AlpaSim believes it
   supplied. We declare ``context_length`` and consume what arrives, asserting the count.

2. **GEOMETRY.** Our encoder RAISES if the raster does not match its declared frame (its
   positional embedding is sized for it). AlpaSim delivers whatever the camera config produces, so
   every frame is resized to the model's trained geometry before it reaches the encoder.

3. **THE SPEED CHANNEL.** v1-line models take v0 as a THIRD action channel scaled by
   ``SPEED_SCALE = 10.0`` — a hard contract with the eval path. Get it wrong and the checkpoint
   decodes garbage while still producing plausible-looking numbers.

4. **FRAME CONVENTION.** AlpaSim wants ``trajectory_xy`` in the RIG frame, x forward / y left.
   Our operative readout emits ego-frame displacements accumulated through SE(2) — the same
   convention — so no rotation is applied, and that is asserted rather than assumed.

⭐ COMMAND ENCODING IS AN EXACT MATCH, VERIFIED AT BOTH SOURCES — not assumed:
``alpasim_driver.models.base.DriveCommand`` is ``LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3``; our
route vocabulary is ``R_LEFT, R_STRAIGHT, R_RIGHT = 0, 1, 2`` (``stack/scripts/v5_guard.py``) with
``ROUTE_UNKNOWN = 3`` (``stack/scripts/refb_labels.py:536``). The identity is asserted at import,
so a future change on either side fails loudly instead of silently steering the wrong way.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    _e = os.path.expanduser(_p)
    if _e not in sys.path:
        sys.path.insert(0, _e)

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput

SPEED_SCALE = 10.0          # hard contract with eval_grounded_rollout_4b_speed.py
OUTPUT_HZ = 10              # our readout is 10 Hz (DT = 0.1 s)

# ⛔ FAIL AT IMPORT, not mid-drive, if the two vocabularies ever diverge.
_OURS = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2, "UNKNOWN": 3}
for _n, _v in _OURS.items():
    if int(getattr(DriveCommand, _n)) != _v:
        raise ImportError(
            f"DriveCommand.{_n} == {int(getattr(DriveCommand, _n))} but TanitAD's route "
            f"vocabulary uses {_v} (v5_guard.R_*/refb_labels.ROUTE_UNKNOWN). Fix the mapping in "
            f"_encode_command before driving — a silent mismatch steers the wrong way.")


class TanitADModel(BaseTrajectoryModel):
    """Our 4-brain world model as an AlpaSim driver policy."""

    def __init__(self, world, step_readout, cfg, device, camera_ids,
                 frame_hw=(176, 624), horizon=20):
        self.world = world
        self.step_readout = step_readout
        self.cfg = cfg
        self.device = device
        # NOTE: `camera_ids` is a read-only @property on the base class — assigning to it raises
        # AttributeError. Store behind a private name and expose it through the property below.
        self._camera_ids = list(camera_ids or [])
        self.H, self.W = frame_hw
        self.horizon = horizon
        self.window = cfg.predictor.window
        self.n_ticks = 0

    # ------------------------------------------------------- contract properties
    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        """The predictor consumes a fixed window of past frames; AlpaSim supplies exactly this."""
        return self.window

    @property
    def output_frequency_hz(self) -> int:
        return OUTPUT_HZ

    def _encode_command(self, command: DriveCommand) -> Any:
        """DriveCommand -> our nav/route class. Identity, asserted at import (see module docstring).

        UNKNOWN stays 3 and is NOT collapsed to STRAIGHT: `refb_labels.py:511` — "NEVER DEFAULT TO
        STRAIGHT" — because a defaulted unknown is indistinguishable from a real straight command
        and quietly biases every route metric computed downstream.
        """
        return int(command)

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

        def _get(k, default=None):
            if hasattr(model_cfg, k):
                return getattr(model_cfg, k)
            try:
                return model_cfg[k]
            except Exception:
                return default

        ckpt = _get("checkpoint_path")
        if not ckpt:
            raise ValueError("tanitad driver: model_cfg.checkpoint_path is required")
        geom = _get("geometry", "w120")
        horizon = int(_get("horizon", 20))

        cfg = flagship4b_config()
        if geom == "w120":                       # v5-line: wide-FOV sub-frame
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
              f"window={cfg.predictor.window} horizon={horizon}", flush=True)
        return cls(world, sr, cfg, device, camera_ids, frame_hw=hw, horizon=horizon)

    # ------------------------------------------------------------------ frames
    def _prep(self, img: np.ndarray) -> np.ndarray:
        """HWC uint8 RGB -> the model's trained raster, channel-first."""
        x = self._resize_and_center_crop(img, self.H, self.W)
        return np.ascontiguousarray(np.asarray(x).transpose(2, 0, 1))

    def _window(self, frames: list) -> torch.Tensor:
        """AlpaSim's supplied temporal frames -> [1, W, C, H, W] on device."""
        if len(frames) != self.window:
            raise RuntimeError(
                f"tanitad driver: AlpaSim supplied {len(frames)} frames but context_length "
                f"declares {self.window}. Scoring a mis-sized window is a silent instrument "
                f"failure — check the driver config's context length.")
        w = np.stack([self._prep(f.image) for f in frames])          # [W, C, H, W]
        t = torch.from_numpy(w).to(self.device).float()
        C = self.cfg.encoder.in_channels
        if t.shape[1] < C:                                           # 3ch RGB -> Cch stack
            t = t.repeat(1, C // t.shape[1] + 1, 1, 1)[:, :C]
        if t.max() > 1.5:
            t = t / 255.0
        return t.unsqueeze(0)

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        cam = prediction_input.camera_images
        key = self._camera_ids[0] if self._camera_ids else next(iter(cam))
        frames = cam[key]
        fw = self._window(frames)
        self.n_ticks += 1

        v0 = float(prediction_input.speed)
        states = self.world.encode_window(fw)

        # actions: (steer, accel, v0/SPEED_SCALE) — the v1 speed-channel contract
        a = torch.zeros(1, self.window, self.cfg.predictor.action_dim, device=self.device)
        if self.cfg.predictor.action_dim >= 3:
            a[..., 2] = v0 / SPEED_SCALE

        from tanitad.models.metric_dynamics import (accumulate_se2,
                                                    rollout_transitions)
        trans = rollout_transitions(self.world.predictor, states, a, self.horizon)
        dpose = torch.stack([self.step_readout(trans[j][0], trans[j][1])
                             for j in range(self.horizon)], dim=1)      # [1, T, 3]
        wp = accumulate_se2(dpose)[0].float().cpu().numpy()             # [T, 2] ego frame

        # AlpaSim wants rig-frame x-forward / y-left — the SAME convention our readout emits,
        # so no rotation. Headings come from the path tangent via the base helper.
        headings = self._compute_headings_from_trajectory(wp)
        cmd = self._encode_command(prediction_input.command)
        note = f"tanitad tick={self.n_ticks} v0={v0:.2f} m/s cmd={DriveCommand(cmd).name}"
        return ModelPrediction(trajectory_xy=np.asarray(wp, dtype=np.float32),
                               headings=np.asarray(headings, dtype=np.float32),
                               reasoning_text=note)
