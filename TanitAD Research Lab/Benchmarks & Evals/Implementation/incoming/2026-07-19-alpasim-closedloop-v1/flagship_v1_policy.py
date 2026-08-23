# SPDX-License-Identifier: Apache-2.0
"""TanitAD flagship-v1 driver plug-in for NVIDIA AlpaSim.

STATUS: adapter STUB (Phase-3 deliverable). The obs->action mapping is filled
in as far as the AlpaSim `BaseTrajectoryModel` contract and our own inference
code allow; every place that needs the *live* simulator to confirm a detail (or
our own action-normalisation constants to be pinned) is marked `TODO(sim)` /
`TODO(ours)`. It has NOT been run end to end — the eval pod cannot launch the
AlpaSim renderer (see INTAKE.md: no container runtime, NRE ships only as the
`nvcr.io/nvidia/nre/nre-ga:26.04` image). It is validated for *syntax* only
(`python -m py_compile`). Wire + smoke it on a docker-capable renderer host.

------------------------------------------------------------------------------
How AlpaSim calls a driver (reverse-engineered from the shipped Transfuser/VaVAM
adapters + `src/driver/src/alpasim_driver/models/base.py` + `main.py`):

  * A driver is an `alpasim.models` entry point resolved by name (`driver=<name>`).
    Register this class:
        [project.entry-points."alpasim.models"]
        flagship_v1 = "alpasim_flagship.flagship_v1_policy:FlagshipV1Model"
    plus an `alpasim.configs` Hydra dir exposing `driver/flagship_v1.yaml`.

  * The driver SERVICE (`alpasim_driver.main`) owns the gRPC `EgodriverService`
    (egodriver.proto): it accumulates `submit_image_observation` /
    `submit_egomotion_observation` / `submit_route`, and on each `drive()` builds
    a `PredictionInput` and calls `model.predict_batch([...])`. It then serialises
    our `ModelPrediction.trajectory_xy` into a rig-frame `common.Trajectory`,
    spacing the waypoints at `1 / output_frequency_hz` seconds
    (see `_convert_prediction_to_alpasim_trajectory`).  => We implement the model,
    NOT the proto. Runs inside the driver container OR as a bare external python
    process (`driver_source=external_static`, `wizard.external_services.driver=
    ["<ip>:6789"]`) — the latter is how we run on a GPU host without rebuilding
    the image (see docs/MANUAL_DRIVER.md, which runs a driver as a plain script).

Observation the sim hands us (`PredictionInput`):
    camera_images : dict[cam_id -> list[CameraFrame]]   # CameraFrame=(ts_us, HWC uint8 RGB)
    command       : DriveCommand                         # LEFT=0 STRAIGHT=1 RIGHT=2 UNKNOWN=3
    speed         : float  (m/s)                          # <- our v0 speed channel
    acceleration  : float  (m/s^2)
    ego_pose_history : list[PoseAtTime]                  # rig-frame poses + dyn states
    inference_seed : int
    (NOTE: intrinsics are NOT passed to the model — the renderer + the driver's
     RectificationTargetConfig produce a pinhole frame at the configured FOV/res.)

Action the sim expects back (`ModelPrediction`):
    trajectory_xy : np.ndarray (T, 2)  x,y offsets, RIG frame (x fwd, y LEFT)
    headings      : np.ndarray (T,)    radians, rig frame
    reasoning_text: optional

Our flagship-v1 (from stack/tanitad, validated by
experiments/reset-speed4b/eval_grounded_rollout_4b_speed.py + config.py):
    encoder input = 3 RGB sub-frames channel-stacked (9ch) @100ms spacing, 256px
    operative predictor window W = cfg.predictor.window (8 for flagship4b)
    action_dim = 3 = (steer, accel, v0)   v0 = speed / SPEED_SCALE(=10.0)
    closed-loop plan = run_hierarchy(...)["waypoints"] from the TRAINED tactical
       policy at horizons {5,10,15,20}@10Hz = {0.5,1.0,1.5,2.0}s  => 2 Hz, 2 s.
    (The grounded-rollout script is a TEACHER-FORCED metric harness — it needs
     the TRUE future actions — so it is NOT the planner; the tactical hierarchy is.)
------------------------------------------------------------------------------
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image

# --- AlpaSim driver contract ------------------------------------------------
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig

# --- Our stack --------------------------------------------------------------
from tanitad.config import flagship4b_config
from tanitad.eval.ckpt_compat import (  # self-describing action_dim + v0 channel
    SPEED_SCALE,
    append_speed_channel,
    build_world_from_ckpt,
)
from tanitad.models.fourbrain import run_hierarchy

logger = logging.getLogger(__name__)

# Single front-wide camera — matches our front-wide training input.
FLAGSHIP_CAMERA = "camera_front_wide_120fov"

# Training-time encoder constants (config.py::base250cam_config / flagship4b_config).
_ENC_SUBFRAMES = 3            # 9ch encoder input = 3 RGB sub-frames stacked
_SUBFRAME_DT_S = 0.10        # 100 ms micro-stack spacing ([t-200,t-100,t])
_ENC_IMG_PX = 256            # encoder image_size
_WP_STEPS = (5, 10, 15, 20)  # tactical waypoint horizons @10Hz -> 0.5/1/1.5/2 s
_OUTPUT_HZ = 2               # 0.5 s spacing between returned waypoints

# ego waypoints -> AlpaSim rig frame. AlpaSim rig = x forward, y LEFT.
# Transfuser flips y (CARLA is y-right); our PhysicalAI-AV / NVIDIA rig convention
# is x-fwd/y-left already => identity. TODO(ours): CONFIRM the y sign against a
# known left-turn window (gt_ego_waypoints) before trusting closed-loop steering.
_Y_SIGN = +1.0


class FlagshipV1Model(BaseTrajectoryModel):
    """4-brain latent world model as an AlpaSim trajectory driver.

    Single front-wide camera in, 2 s ego-frame waypoint plan out, via the trained
    strategic->tactical hierarchy. Maintains an internal rolling raw-frame buffer
    so it is robust to however many frames the sim packs into each PredictionInput.
    """

    # ---- factory -----------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "FlagshipV1Model":
        if list(camera_ids) != [FLAGSHIP_CAMERA]:
            raise ValueError(
                f"flagship_v1 is single-front-camera; expected [{FLAGSHIP_CAMERA}], "
                f"got {camera_ids}"
            )
        return cls(checkpoint_path=model_cfg.checkpoint_path, device=torch.device(device))

    def __init__(self, checkpoint_path: str, device: torch.device):
        self._device = device
        # ckpt_compat rebuilds the config at the ckpt's trained action_dim (3 for
        # speed-input) and STRICT-loads — see stack/tanitad/eval/ckpt_compat.py.
        cfg = flagship4b_config()
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        world, speed_input, source = build_world_from_ckpt(cfg, ck, checkpoint_path)
        if not speed_input:
            logger.warning("ckpt reports action_dim<3 (source=%s) — v0 speed channel "
                           "will NOT be appended; expected the speed-input flagship", source)
        self._world = world.to(device).eval()
        self._speed_input = speed_input
        self._window = int(world.predictor.cfg.window)      # W (8 for flagship4b)
        if world.tactical_policy is None or world.strategic_policy is None:
            raise ValueError("flagship_v1 needs the trained 4-brain policy heads "
                             "(tactical+strategic) — load a full flagship4b ckpt")
        # rolling raw-RGB buffer: (timestamp_us, HWC uint8) newest-last.
        # Need enough history to build W nine-channel stacks at 100 ms spacing.
        self._buf: deque[tuple[int, np.ndarray]] = deque(maxlen=self._raw_frames_needed())
        logger.info("flagship_v1 loaded from %s (action_dim source=%s, window=%d)",
                    checkpoint_path, source, self._window)

    # ---- required properties ----------------------------------------------
    @property
    def camera_ids(self) -> list[str]:
        return [FLAGSHIP_CAMERA]

    @property
    def context_length(self) -> int:
        # Ask the sim for enough temporal frames to fill our window; we also keep
        # our own buffer so we are robust if fewer arrive.
        return self._raw_frames_needed()

    @property
    def output_frequency_hz(self) -> int:
        return _OUTPUT_HZ

    def _raw_frames_needed(self) -> int:
        # W predictor ticks (100 ms apart) each built from _ENC_SUBFRAMES sub-frames
        # (100 ms apart). Worst case they do not overlap: (W-1)+ _ENC_SUBFRAMES.
        # TODO(ours): confirm the exact tick/sub-frame stride+overlap against the
        # window-dataset builder (FeatureWindowDataset4B) — this is an upper bound.
        return (self._window - 1) + _ENC_SUBFRAMES

    # ---- command mapping ---------------------------------------------------
    def _encode_command(self, command: DriveCommand) -> int:
        """Map AlpaSim DriveCommand -> our strategic nav_cmd index.

        TODO(ours): align these indices with the nav_cmd vocabulary the strategic
        policy was TRAINED on (config.py: nav_cmd derivation). Placeholder mapping:
        follow/straight=0, left=1, right=2.
        """
        return {DriveCommand.STRAIGHT: 0, DriveCommand.LEFT: 1,
                DriveCommand.RIGHT: 2, DriveCommand.UNKNOWN: 0}[command]

    # ---- preprocessing -----------------------------------------------------
    @staticmethod
    def _to_encoder_px(img: np.ndarray) -> np.ndarray:
        """HWC uint8 RGB -> 256x256 (bilinear). TODO(ours): match the EXACT train
        transform (front-wide crop/resize + any grayscale/normalisation). The eval
        script feeds frames as float/255 with the dataset's own crop already applied."""
        if img.shape[:2] == (_ENC_IMG_PX, _ENC_IMG_PX):
            return img
        return np.asarray(Image.fromarray(img).resize((_ENC_IMG_PX, _ENC_IMG_PX),
                                                       Image.Resampling.BILINEAR))

    def _build_window(self) -> torch.Tensor:
        """Assemble the predictor window [1, W, 9, 256, 256] from the frame buffer.

        Each of the W ticks is a 9-channel stack of _ENC_SUBFRAMES RGB sub-frames at
        100 ms spacing. Uses the newest frames; pads by repeating the oldest at warm-up.
        TODO(sim): the buffer is keyed by CameraFrame.timestamp_us — resample to exact
        100 ms grid once real frame timing is known instead of assuming last-N ordering.
        """
        frames = list(self._buf)
        if not frames:
            raise ValueError("flagship_v1: no camera frames buffered yet")
        while len(frames) < self._raw_frames_needed():        # warm-up pad
            frames.insert(0, frames[0])
        imgs = [self._to_encoder_px(f[1]) for f in frames]     # each HWC uint8
        ticks = []
        for t in range(self._window):
            stack = imgs[t:t + _ENC_SUBFRAMES]                 # 3 x HWC
            nine = np.concatenate(stack, axis=2)               # H,W,9
            ticks.append(nine)
        arr = np.stack(ticks, axis=0).astype(np.float32) / 255.0   # [W,H,W',9]
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [W,9,H,W']
        return t.unsqueeze(0).to(self._device)                      # [1,W,9,H,W']

    def _past_action_window(self, inp: PredictionInput) -> torch.Tensor:
        """Reconstruct the past action window [1, W, 2] = (steer, accel).

        We do NOT get past control inputs from the sim — only ego kinematics. Two
        options: (a) our InverseDynamicsHead on the encoded window, or (b) derive
        from ego_pose_history (yaw-rate->steer, longitudinal accel->accel).
        TODO(ours): pin the derivation + the EXACT train-time action normalisation
        so the FiLM conditioning sees in-distribution values. Placeholder = zeros
        (planner still runs on the visual state window; conditioning is weak).
        """
        return torch.zeros(1, self._window, 2, device=self._device)

    # ---- inference ---------------------------------------------------------
    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        return self.predict_batch([prediction_input])[0]

    def predict_batch(self, prediction_inputs: list[PredictionInput]) -> list[ModelPrediction]:
        # NOTE: batch=1 per session assumed here (single ego). TODO(sim): stack for
        # multi-rollout batching (topology>1) once we hold per-session buffers.
        out: list[ModelPrediction] = []
        for inp in prediction_inputs:
            self._validate_cameras(inp.camera_images)
            for _cam, seq in inp.camera_images.items():        # ingest into buffer
                for fr in seq:
                    self._buf.append((int(fr.timestamp_us), np.asarray(fr.image)))

            with torch.no_grad():
                frames = self._build_window()                  # [1,W,9,256,256]
                states = self._world.encode_window(frames)     # [1,W,S]
                actions = self._past_action_window(inp)        # [1,W,2]
                if self._speed_input:
                    v0 = torch.tensor([[inp.speed / SPEED_SCALE]],
                                      dtype=actions.dtype, device=self._device)
                    actions = append_speed_channel(actions, v0)   # [1,W,3]
                nav = torch.tensor([self._encode_command(inp.command)],
                                   dtype=torch.long, device=self._device)
                hier = run_hierarchy(self._world, states, actions, nav_cmd=nav)
                wp = hier["waypoints"]                          # tactical waypoints

            # wp -> (T,2) ego-frame at horizons {5,10,15,20}. TODO(ours): confirm
            # wp's layout (already the 4 horizon waypoints [1,4,2]? or dense [1,K,2]
            # needing index_select(_WP_STEPS-1)?) against TacticalPolicy.forward.
            wp = wp.detach().float().cpu().numpy()
            wp = wp[0] if wp.ndim == 3 else wp                 # -> (T,2) or (K,2)
            if wp.shape[0] > len(_WP_STEPS):                   # dense -> pick horizons
                wp = wp[[s - 1 for s in _WP_STEPS], :]
            traj = np.empty_like(wp)
            traj[:, 0] = wp[:, 0]                              # x forward
            traj[:, 1] = _Y_SIGN * wp[:, 1]                    # y (see _Y_SIGN TODO)

            headings = self._compute_headings_from_trajectory(traj)
            out.append(ModelPrediction(trajectory_xy=traj.astype(np.float32),
                                       headings=headings.astype(np.float32)))
        return out
