"""MetaDrive -> TanitAD episode-contract adapter (WP2).

Produces the SAME episode contract as :mod:`tanitad.data.toy_driving` so every
downstream consumer (dataset windows, world-model training, the D1-D3 gate
runner) ports over unchanged:

    frames  [T, 1, H, W]  float32 in [0, 1]     (single-channel top-down BEV)
    actions [T, 2]        (steer rad, accel m/s^2), taken between t and t+1
    poses   [T, 4]        (x, y, yaw, v)
    episode_id : int

MetaDrive is an OPTIONAL dependency (``pip install -e .[sim]``). It is imported
lazily inside the rollout function only, so the rest of the stack -- and CI --
runs with zero simulator dependencies. The frame / pose / action conversion
helpers below are pure NumPy/Torch and are unit-tested WITHOUT a live
simulator; the live rollout is exercised by a test that skips when MetaDrive is
not importable (``pytest.importorskip``).

Setup status -- measured 2026-07-06, RTX 4060 / Python 3.13 / Windows
(full note: ``TanitAD Research Hub/Tools&DevEnv/Research/2026-07-06-metadrive-adoption-and-alpasim-verdict.md``):

  * PyPI ``metadrive-simulator`` (0.2.6.0) does NOT install on py3.13 -- it
    pins ``gym==0.19.0`` whose ``setup.py`` fails to build under modern
    setuptools (``extras_require`` type error). Verdict: NO-GO on py3.13.
  * The native blocker is cleared: ``panda3d`` 1.10.16 and ``gymnasium`` 1.3.0
    both ship cp313 wheels and install in <1 min on this machine.
  * GO path: install MetaDrive from source (GitHub main; gymnasium-based) in a
    SUPERVISED session (external-code install needs user trust), then drive the
    live path via :func:`generate_metadrive_episode`.

The contract semantics deliberately mirror ``toy_driving.generate_episode``:
poses come from the ego vehicle's physical state, ``accel`` is the finite
difference of speed (m/s^2), and ``steer`` is the applied steering in radians.
Because the frame is ego-centric top-down, every action moves every pixel --
the consequence-dominant regime (A8) that the world model needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Re-use the exact episode container so a MetaDrive episode is type-identical to
# a toy episode -- downstream code cannot tell them apart.
from tanitad.data.toy_driving import ToyEpisode
# The episode-contract assembly (accel finite-diff, contract assertion, windowed
# Dataset) lives in one shared home so every adapter stays contract-identical.
# Re-exported below so existing importers of these names keep working.
from tanitad.data._contract import (EpisodeWindowDataset, assemble_episode,
                                     assert_contract, finite_diff_accel,
                                     frame_change_fraction)


# --------------------------------------------------------------------------- #
# Pure conversion helpers (unit-tested without a live simulator)              #
# --------------------------------------------------------------------------- #
def bev_frame_from_rgb(rgb: np.ndarray, size: int = 64) -> Tensor:
    """MetaDrive top-down RGB frame -> single-channel BEV tensor [1, size, size].

    Accepts ``[H, W]``, ``[H, W, 3]`` or ``[H, W, 4]`` arrays of any dtype.
    Channels are averaged to grayscale, values are normalized to ``[0, 1]``
    (uint8 by /255, float by max), and the frame is bilinearly resized to
    ``size x size``. Output matches the toy contract's per-frame shape/range.
    """
    arr = np.asarray(rgb)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)  # RGB(A) -> gray
    elif arr.ndim == 2:
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"expected 2-D or 3-D frame, got shape {arr.shape}")

    if np.issubdtype(np.asarray(rgb).dtype, np.integer):
        arr = arr / 255.0
    else:
        m = float(arr.max())
        arr = arr / m if m > 1.0 else arr
    arr = np.clip(arr, 0.0, 1.0)

    t = torch.from_numpy(arr)[None, None]                    # [1, 1, H, W]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].to(torch.float32)                            # [1, size, size]


def pose_from_state(x: float, y: float, yaw: float, speed_mps: float) -> np.ndarray:
    """Pack an ego state into the contract pose vector ``(x, y, yaw, v)``."""
    return np.array([x, y, yaw, speed_mps], dtype=np.float32)


def kmh_to_ms(speed_kmh: float) -> float:
    """MetaDrive reports speed in km/h; the contract's ``v`` is m/s."""
    return float(speed_kmh) / 3.6


def steering_to_rad(norm_steering: float, max_steering_deg: float = 40.0) -> float:
    """MetaDrive steering is normalized in ``[-1, 1]``; the contract wants rad.

    ``max_steering_deg`` is the vehicle's ``config['max_steering']`` (default 40).
    """
    return math.radians(float(norm_steering) * float(max_steering_deg))


# --------------------------------------------------------------------------- #
# Episode assembly + windowed dataset come from the shared contract module.    #
# ``assemble_episode`` / ``finite_diff_accel`` / ``frame_change_fraction`` are  #
# imported (and re-exported) at the top of this file. The MetaDrive dataset is  #
# the generic windowed dataset -- kept as a named subclass so a MetaDrive       #
# episode set reads intention-clearly at the call site.                        #
# --------------------------------------------------------------------------- #
class MetaDriveDataset(EpisodeWindowDataset):
    """Windowed dataset over MetaDrive episodes (see :class:`EpisodeWindowDataset`).

    MetaDrive episodes are produced by driving a live simulator, but the window
    contract is byte-for-byte identical to the toy set, so a model trained on the
    toy set consumes these unchanged. Splits are EPISODE-level (I3).
    """


# --------------------------------------------------------------------------- #
# Live rollout (lazy import; validated in a SUPERVISED session)              #
# --------------------------------------------------------------------------- #
@dataclass
class MetaDriveEpisodeConfig:
    steps: int = 80
    size: int = 64
    dt: float = 0.1                 # MetaDrive default physics step is 0.1 s
    map: str = "S"                  # single Straight block (cheap, deterministic)
    traffic_density: float = 0.1
    max_steering_deg: float = 40.0  # vehicle config['max_steering']


def _make_env(episode_id: int, cfg: MetaDriveEpisodeConfig):
    """Create a headless MetaDrive env with the ego on IDM autopilot.

    Imported lazily so importing this module never requires MetaDrive.
    """
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy

    env = MetaDriveEnv(dict(
        use_render=False,
        image_observation=False,
        map=cfg.map,
        traffic_density=cfg.traffic_density,
        start_seed=int(episode_id),
        num_scenarios=1,
        agent_policy=IDMPolicy,       # ego drives itself; our step action is a no-op
        physics_world_step_size=cfg.dt,
        log_level=50,                 # silence
    ))
    return env


def _topdown_frame(env, size: int) -> Tensor:
    """Fetch the top-down RGB view and convert to the BEV contract frame.

    Uses MetaDrive's built-in top-down renderer. The exact ``render`` kwargs
    differ slightly across MetaDrive versions; this asks for a windowless
    top-down array and hands the raw pixels to :func:`bev_frame_from_rgb`. If a
    given version returns ``None`` (needs an explicit observation), the caller
    should switch to ``TopDownObservation`` -- flagged for the supervised run.
    """
    rgb = env.render(mode="topdown", window=False,
                     screen_size=(size, size), draw_target_vehicle_trajectory=False)
    if rgb is None:
        raise RuntimeError(
            "MetaDrive top-down render returned None for this version; wire "
            "TopDownObservation instead (see module docstring / STATE handoff).")
    return bev_frame_from_rgb(np.asarray(rgb), size=size)


def generate_metadrive_episode(episode_id: int,
                               cfg: MetaDriveEpisodeConfig | None = None
                               ) -> ToyEpisode:
    """Drive one IDM-autopilot MetaDrive episode into the TanitAD contract.

    Requires MetaDrive (``pip install -e .[sim]``). Produces a
    :class:`ToyEpisode` indistinguishable from a toy episode downstream.
    """
    cfg = cfg or MetaDriveEpisodeConfig()
    env = _make_env(episode_id, cfg)
    frames: list[Tensor] = []
    poses: list[np.ndarray] = []
    steer_rad: list[float] = []
    noop = np.array([0.0, 0.0], dtype=np.float32)  # IDMPolicy overrides this
    try:
        env.reset(seed=int(episode_id))
        for _ in range(cfg.steps):
            ego = env.agent
            frames.append(_topdown_frame(env, cfg.size))
            x, y = float(ego.position[0]), float(ego.position[1])
            yaw = float(getattr(ego, "heading_theta", 0.0))
            v = kmh_to_ms(getattr(ego, "speed_km_h", getattr(ego, "speed", 0.0)))
            poses.append(pose_from_state(x, y, yaw, v))
            steer_rad.append(steering_to_rad(float(getattr(ego, "steering", 0.0)),
                                             cfg.max_steering_deg))
            _obs, _r, terminated, truncated, _info = env.step(noop)
            if terminated or truncated:
                break
    finally:
        env.close()
    return assemble_episode(frames, poses, steer_rad, cfg.dt, episode_id)
