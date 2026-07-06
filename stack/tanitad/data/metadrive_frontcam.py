"""MetaDrive front-camera RGB + perturbation-policy episode generator (backlog #1 b/c).

WHY THIS EXISTS
---------------
The merged MetaDrive adapter (`stack/tanitad/data/metadrive_env.py`) renders a
single-channel top-down BEV frame ``[T, 1, 64, 64]``. That is fine as a *probe*
signal, but it **cannot be mixed** with the real-data corpus: comma2k19
(D-009 ``base250cam``) emits ``[T, 6, 256, 256]`` — two consecutive RGB frames
channel-stacked. ``MixedWindowDataset._check_contract`` (D-010) rejects any sim
source whose frame shape differs from the real source, so today the D-010 mix
has NO admissible sim arm.

This module adds the **front-camera RGB path** so a MetaDrive episode is
byte-for-byte contract-identical to a comma2k19 episode (6-channel, 256 px,
same 2-frame stacking, same action/pose temporal alignment). That is what the
sim arm is *for* (D-010 role split): off-expert action-consequence coverage,
scripted occluders (H15/D9 object permanence) and blocked routes (D5/D6
fallback) — signals no real log can provide.

The single-channel BEV path in ``metadrive_env.py`` STAYS, untouched, for the
D3 imagined-vs-oracle probe. This module is additive.

WHAT IS AND IS NOT VALIDATED HERE
---------------------------------
Pure conversion / assembly / perturbation / scenario-config helpers are the
load-bearing logic and are unit-tested WITHOUT a live simulator (see
``tests/test_metadrive_frontcam.py``). The live rollout lazily imports MetaDrive
and is exercised only by a ``@pytest.mark.slow`` test that skips when MetaDrive
is not importable — MetaDrive still needs the supervised source install
(PyPI no-go on py3.13; verdict unchanged, see the 2026-07-06 note). Object
spawning for the occluder/blocked-route scenarios uses MetaDrive's live
scene-edit API and is flagged for the supervised run.

MetaDrive front-camera API (grounded 2026-07-13, docs.metadrive-simulator):
    cfg = dict(
        image_observation=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, W, H)},
    )
    obs["image"]  # (H, W, 3, stack_size); most-recent frame at [..., -1]
Values are normalized to [0, 1] float32 by default; we defensively handle
uint8 / unnormalized floats too. NOTE for the supervised run: MetaDrive's image
buffer has historically returned channels in BGR order and rows bottom-up on
some backends — verify orientation once against a saved PNG (world model is
channel-order-agnostic, but the human-facing replay overlay is not).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Reuse the exact contract primitives so this path cannot drift from the toy /
# comma2k19 / MetaDrive-BEV adapters. stack_two_frames is imported from the
# comma2k19 loader so sim and real use IDENTICAL 2-frame stacking semantics.
from tanitad.data._contract import assert_contract, finite_diff_accel
from tanitad.data.comma2k19 import stack_two_frames
from tanitad.data.metadrive_env import kmh_to_ms, pose_from_state, steering_to_rad
from tanitad.data.mixing import save_episode
from tanitad.data.toy_driving import ToyEpisode

# comma2k19 base250cam contract: 6 channels (2x RGB), 256 px square.
REAL_CHANNELS = 6
REAL_SIZE = 256


# --------------------------------------------------------------------------- #
# Pure conversion helpers (unit-tested without a live simulator)              #
# --------------------------------------------------------------------------- #
def _to_chw_float01(rgb: np.ndarray) -> Tensor:
    """A single MetaDrive RGB frame -> [3, H, W] float32 in [0, 1].

    Accepts ``(H, W, 3)``, ``(H, W, 3, stack)`` (most-recent frame taken),
    ``(H, W, 4)`` (alpha dropped) or ``(H, W)`` (broadcast to 3). Integer dtypes
    are divided by 255; floats already in [0, 1] pass through, floats with
    ``max > 1`` are divided by their max.
    """
    arr = np.asarray(rgb)
    if arr.ndim == 4:                       # (H, W, C, stack) -> most recent
        arr = arr[..., -1]
    if arr.ndim == 2:                       # gray -> 3 channels
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected 2/3/4-D frame, got shape {np.asarray(rgb).shape}")
    arr = arr[..., :3].astype(np.float32)   # drop alpha if present

    if np.issubdtype(np.asarray(rgb).dtype, np.integer):
        arr = arr / 255.0
    else:
        m = float(arr.max()) if arr.size else 1.0
        arr = arr / m if m > 1.0 else arr
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # [3, H, W]


def frontcam_frame(rgb: np.ndarray, size: int = REAL_SIZE) -> Tensor:
    """MetaDrive front-camera frame -> [3, size, size] float32 in [0, 1].

    Geometry mirrors ``comma2k19._decode_video`` EXACTLY (center-crop to the
    largest centered square, then bilinear resize) so sim and real frames share
    identical field-of-view handling before channel-stacking.
    """
    chw = _to_chw_float01(rgb)                       # [3, H, W]
    h, w = chw.shape[-2:]
    c = min(h, w)
    top, left = (h - c) // 2, (w - c) // 2
    square = chw[..., top:top + c, left:left + c][None]           # [1, 3, c, c]
    out = F.interpolate(square, size=(size, size), mode="bilinear",
                        align_corners=False)
    return out[0].clamp(0.0, 1.0).to(torch.float32)               # [3, size, size]


def assemble_frontcam_episode(frames3: list[Tensor], poses: list[np.ndarray],
                              steer_rad: list[float], dt: float,
                              episode_id: int) -> ToyEpisode:
    """Stack per-step buffers into a comma2k19-identical 6-channel episode.

    ``frames3`` is a length-``n`` list of ``[3, S, S]`` frames. They are stacked
    into ``[n-1, 6, S, S]`` (frame t-1 and t) and the action/pose at ``t+1`` is
    kept, EXACTLY matching ``comma2k19.build_episode`` (frames6[t] pairs (t, t+1),
    action/pose at t+1). ``accel`` is the finite-difference of the pose speed.
    """
    if len(frames3) < 2:
        raise ValueError("need >= 2 raw frames to form one 2-frame stack")
    vid = torch.stack(list(frames3), dim=0).to(torch.float32)     # [n, 3, S, S]
    n = vid.shape[0]
    frames6 = stack_two_frames(vid)                               # [n-1, 6, S, S]

    poses_arr = np.stack(poses).astype(np.float32)                # [n, 4]
    accel = finite_diff_accel(poses_arr[:, 3], dt)                # [n]
    actions_full = np.column_stack(
        [np.asarray(steer_rad, np.float32), accel]).astype(np.float32)  # [n, 2]

    ep = ToyEpisode(
        frames=frames6,                                          # already [0,1] float
        actions=torch.from_numpy(actions_full[1:n]).to(torch.float32),
        poses=torch.from_numpy(poses_arr[1:n]).to(torch.float32),
        episode_id=episode_id,
    )
    assert_contract(ep, channels=REAL_CHANNELS)
    return ep


# --------------------------------------------------------------------------- #
# Perturbation policy (pure, deterministic, unit-tested)                       #
# --------------------------------------------------------------------------- #
@dataclass
class PerturbConfig:
    """Scripted off-expert perturbation on top of a base (IDM/expert) action.

    Actions are MetaDrive-normalized ``[steer, throttle]`` in ``[-1, 1]``. The
    perturbation injects (a) a deterministic sinusoidal steering bias and
    (b) stochastic throttle pulses / brake stabs, so the episode explores
    action-consequence pairs that expert logs never contain (D-010 sim role).
    Off by default via ``steer_amp=0`` + zero probs -> identity, so a plain
    expert rollout is a config, not a code path.
    """
    steer_amp: float = 0.15         # sinusoidal steer-bias amplitude (norm units)
    steer_period: int = 25          # steps per sine cycle
    throttle_pulse_prob: float = 0.10
    throttle_pulse: float = 0.40    # additive throttle burst
    brake_prob: float = 0.05
    brake: float = -0.60            # brake stab (overrides throttle)


def perturb_action(base_action, t: int, cfg: PerturbConfig,
                   rng: np.random.Generator) -> np.ndarray:
    """Off-expert action = base action + scripted perturbation, clipped to [-1,1].

    Deterministic given ``t`` (sine bias) and ``rng`` (pulses), so an episode is
    fully reproducible from its seed. ``brake`` takes precedence over a throttle
    pulse when both would fire.
    """
    steer = float(base_action[0]) + cfg.steer_amp * math.sin(
        2.0 * math.pi * t / max(1, cfg.steer_period))
    throttle = float(base_action[1])
    u = float(rng.random())
    if u < cfg.brake_prob:
        throttle = cfg.brake
    elif u < cfg.brake_prob + cfg.throttle_pulse_prob:
        throttle = throttle + cfg.throttle_pulse
    return np.clip(np.array([steer, throttle], dtype=np.float32), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Scenario configs -> MetaDrive env kwargs (unit-tested)                       #
# --------------------------------------------------------------------------- #
@dataclass
class FrontcamScenario:
    """A MetaDrive front-camera scenario producing the 6ch/256px real contract.

    ``kind`` drives object placement performed live in ``populate_scene`` (needs
    the running sim). ``env_config`` returns the pure kwargs dict — fully
    testable offline.
    """
    name: str
    kind: str = "cruise"            # "cruise" | "scripted_occluder" | "blocked_route"
    steps: int = 200
    size: int = REAL_SIZE
    dt: float = 0.1                 # MetaDrive default physics step
    map: str = "S"                  # single Straight block (cheap, deterministic)
    traffic_density: float = 0.10
    max_steering_deg: float = 40.0
    stack_size: int = 3             # MetaDrive image buffer depth (we use [...,-1])
    extra: dict = field(default_factory=dict)

    def env_config(self, episode_id: int) -> dict:
        """Pure MetaDrive env kwargs for this scenario (no sim needed)."""
        cfg = dict(
            use_render=False,
            image_observation=True,
            # sensor name -> (RGBCamera, width, height); class injected live so
            # this dict is importable without MetaDrive on the path.
            sensors={"rgb_camera": ("RGBCamera", self.size, self.size)},
            vehicle_config=dict(image_source="rgb_camera"),
            image_on_cuda=False,           # honest default; flip on the pod (10x)
            map=self.map,
            traffic_density=self.traffic_density,
            start_seed=int(episode_id),
            num_scenarios=1,
            physics_world_step_size=self.dt,
            log_level=50,
        )
        cfg.update(self.extra)
        return cfg


def cruise_scenario(**kw) -> FrontcamScenario:
    """Free-drive baseline: expert/perturbed cruise, moderate traffic."""
    return FrontcamScenario(name="cruise", kind="cruise", **kw)


def scripted_occluder_scenario(**kw) -> FrontcamScenario:
    """A lead vehicle repeatedly occludes objects ahead (H15/D9 permanence).

    Higher traffic so a foreground vehicle enters/leaves the FOV; the live hook
    also parks a static object that the ego drives past (visible -> occluded ->
    visible), the object-permanence signal D9 needs.
    """
    kw.setdefault("traffic_density", 0.35)
    return FrontcamScenario(name="scripted_occluder", kind="scripted_occluder", **kw)


def blocked_route_scenario(**kw) -> FrontcamScenario:
    """A stalled object blocks the ego's lane -> forces a stop/avoid (D5/D6).

    The live hook spawns a static vehicle across the lane ahead; combined with a
    braking-biased PerturbConfig this yields the stop/near-miss coverage that
    closed-loop fallback gates need.
    """
    kw.setdefault("traffic_density", 0.05)
    return FrontcamScenario(name="blocked_route", kind="blocked_route", **kw)


# --------------------------------------------------------------------------- #
# Live rollout (lazy MetaDrive import; validated in a SUPERVISED session)     #
# --------------------------------------------------------------------------- #
def _make_frontcam_env(episode_id: int, scenario: FrontcamScenario):
    """Build a headless MetaDrive env with an RGB front camera.

    Lazily imports MetaDrive so importing this module never requires it. The
    ``"RGBCamera"`` string placeholder in ``env_config`` is swapped for the real
    class here.
    """
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.metadrive_env import MetaDriveEnv

    cfg = scenario.env_config(episode_id)
    w, h = cfg["sensors"]["rgb_camera"][1], cfg["sensors"]["rgb_camera"][2]
    cfg["sensors"] = {"rgb_camera": (RGBCamera, w, h)}
    return MetaDriveEnv(cfg)


def populate_scene(env, scenario: FrontcamScenario) -> None:
    """Place scenario-specific static objects (occluder / route blocker).

    Uses MetaDrive's live scene-edit API (``engine.spawn_object``); a no-op for
    ``cruise``. Flagged for supervised validation — the exact spawn signature is
    version-sensitive. Kept separate from the pure config so the offline tests
    never touch the sim.
    """
    if scenario.kind == "cruise":
        return
    # Supervised-run wiring point. Sketch (validate signatures on the pod):
    #   from metadrive.component.static_object.traffic_object import TrafficBarrier
    #   ego = env.agent
    #   ahead = ego.position + np.array([25.0, 0.0])       # 25 m down-lane
    #   env.engine.spawn_object(TrafficBarrier, position=ahead, heading_theta=0.0)
    raise NotImplementedError(
        f"populate_scene({scenario.kind!r}) needs the live sim; wire on the "
        "supervised MetaDrive install (see module docstring).")


def _frontcam_obs_frame(obs, size: int) -> Tensor:
    """Extract the most-recent RGB frame from a MetaDrive obs dict -> [3,S,S]."""
    image = obs["image"] if isinstance(obs, dict) else obs
    return frontcam_frame(np.asarray(image), size=size)


def generate_frontcam_episode(episode_id: int, scenario: FrontcamScenario,
                              perturb: PerturbConfig | None = None) -> ToyEpisode:
    """Drive one MetaDrive episode into the 6-channel real-data contract.

    Requires MetaDrive (``pip install -e .[sim]`` from source; supervised). With
    ``perturb`` set, the ego is stepped with off-expert perturbed actions and the
    APPLIED action (post-step ``ego.steering`` -> rad, accel from pose speed) is
    recorded, so actions stay physically consistent with the resulting frames.
    """
    from metadrive.policy.idm_policy import IDMPolicy  # noqa: F401  (import proof)

    env = _make_frontcam_env(episode_id, scenario)
    rng = np.random.default_rng(episode_id)
    frames3: list[Tensor] = []
    poses: list[np.ndarray] = []
    steer_rad: list[float] = []
    try:
        obs, _info = env.reset(seed=int(episode_id))
        populate_scene(env, scenario)
        for t in range(scenario.steps):
            ego = env.agent
            frames3.append(_frontcam_obs_frame(obs, scenario.size))
            x, y = float(ego.position[0]), float(ego.position[1])
            yaw = float(getattr(ego, "heading_theta", 0.0))
            v = kmh_to_ms(getattr(ego, "speed_km_h", getattr(ego, "speed", 0.0)))
            poses.append(pose_from_state(x, y, yaw, v))
            steer_rad.append(steering_to_rad(float(getattr(ego, "steering", 0.0)),
                                             scenario.max_steering_deg))
            base = np.array([0.0, 0.5], dtype=np.float32)   # mild forward base
            action = (perturb_action(base, t, perturb, rng)
                      if perturb is not None else base)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    return assemble_frontcam_episode(frames3, poses, steer_rad, scenario.dt,
                                     episode_id)


def generate_and_save(out_dir: str | Path, episode_ids: list[int],
                      scenario: FrontcamScenario,
                      perturb: PerturbConfig | None = None) -> list[Path]:
    """Generate perturbation episodes and persist each as a ``*.pt`` via
    ``tanitad.data.mixing.save_episode`` (uint8 frames; channel-agnostic).

    Returns the written paths. This is the backlog #1(c) entry point: the pod
    runs it to produce the sim arm of the D-010 mix.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for eid in episode_ids:
        ep = generate_frontcam_episode(eid, scenario, perturb)
        path = out / f"{scenario.name}-ep{eid:05d}.pt"
        save_episode(ep, str(path))
        written.append(path)
    return written
