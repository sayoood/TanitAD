"""Central configuration for the TanitAD stack (Phase 0).

All defaults are the validated starting points from ALPS-4B / Phase 0 Plan §2.1.
Every training run must serialize its full config into the experiment record.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EncoderConfig:
    in_channels: int = 1          # 1 = BEV toy; 9 = camera (3-frame stack, D-015)
    image_size: int = 64
    patch_size: int = 8           # 8 for 64px toy -> 8x8 grid; 16 for 256px camera
    d_model: int = 128
    depth: int = 6
    n_heads: int = 4
    grad_checkpoint: bool = False  # recompute block activations (F-5 GPU-memory lever)
    # Batch-free norms only (I2 batch-consistency): never BatchNorm here.


@dataclass
class PredictorConfig:
    d_model: int = 128            # must match encoder d_model after readout proj
    depth: int = 4
    n_heads: int = 4
    window: int = 6               # W-frame causal history (validated 6-8)
    horizons: tuple[int, ...] = (1, 2, 4)  # multi-horizon prediction (MTP, H5)
    action_dim: int = 2           # (steer, accel) continuous, FiLM-conditioned
    residual: bool = True         # A4: residual/delta prediction
    change_weighted: bool = True  # A4: change-weighted latent loss


@dataclass
class SigRegConfig:
    n_slices: int = 512           # random 1-D projections per step (validated)
    beta: float = 1.0             # Epps-Pulley kernel bandwidth
    weight: float = 0.1           # lambda_sigreg (keep fixed; LeJEPA single knob)
    free_dims: int = 0            # >0: exempt the first `free_dims` state dims
                                  # (a fixed ego-motion subspace) from SIGReg so
                                  # anti-collapse and metric-position structure
                                  # stop cancelling (RECOVERY_PLAN §B.3). SIGReg
                                  # still applies to the complement. 0 = off.


@dataclass
class LossConfig:
    sigreg: SigRegConfig = field(default_factory=SigRegConfig)
    pred_weight: float = 1.0
    inv_dyn_weight: float = 0.5   # A5: inverse-dynamics grounding


@dataclass
class ReadoutConfig:
    grid: int = 4                 # spatial grid readout G x G (A7: never global-pool)
    d_readout: int = 32           # per-cell projection dim


@dataclass
class TacticalConfig:
    n_maneuvers: int = 9          # discrete maneuver vocabulary (3 steer x 3 accel)
    horizon: int = 4              # imagine-and-select lookahead (steps)


@dataclass
class TacticalPolicyConfig:
    """Trained tactical brain (D-030 recovery): maneuver policy + 2 s goal +
    intent token. Ports the validated REF-B rev2 ``TacticalHead`` (budget-
    proven) into the world model, EXTENDED with a target-latent goal head so
    the tactical GOAL is (2 s ego sub-waypoints, target latent). A causal
    transformer over the operative state window, FiLM-conditioned on the
    strategic context token; emits an intent token that FiLM-conditions the
    operative predictor — closing the hierarchy. Runs every ``cadence``
    operative ticks (N_tac). State-dim-agnostic: composes on any compact state
    (from-scratch ViT+readout OR frozen-DINO adapter) — shared by the flagship
    and REF-A.
    """
    d_model: int = 512
    depth: int = 6                # REF-B rev2 tactical depth
    n_heads: int = 8
    n_maneuvers: int = 5          # == len(refs.refb.MANEUVER_CLASSES) (test-pinned)
    waypoint_horizons: tuple[int, ...] = (5, 10, 15, 20)   # 2 s @ 10 Hz, ego frame
    d_intent: int = 256           # intent token dim = FiLM cond of the operative
    cadence: int = 5              # N_tac: recompute every 5 operative steps


@dataclass
class StrategicPolicyConfig:
    """Trained strategic brain (D-030 recovery): a real route-level transformer
    over the operative state window, FiLM-conditioned on the nav-command
    embedding. Ports the REF-B rev2 ``StrategicHead`` (d384 x 4) into the world
    model. Outputs a context token (FiLM cond of the tactical brain) + route-
    heading logits (route_left/straight/right) trained by an auxiliary CE. The
    non-parametric ``StrategicGraph`` is kept as AUXILIARY memory, not the
    brain. Runs every ``cadence`` operative ticks (N_str). State-dim-agnostic
    (shared by the flagship and REF-A). Owns its nav embedding so it composes
    self-contained on any model.
    """
    n_commands: int = 4           # == len(refs.refb.NAV_COMMANDS) (test-pinned)
    d_cmd: int = 128              # nav-command embedding = FiLM cond
    d_model: int = 384            # REF-B rev2 strategic width
    depth: int = 4
    n_heads: int = 6
    d_ctx: int = 256              # context token = FiLM cond of the tactical brain
    n_route: int = 3              # == len(refs.refb.ROUTE_CLASSES) (test-pinned)
    cadence: int = 20             # N_str: recompute every 20 operative steps


@dataclass
class H15Config:
    """Imagination in unobserved areas (H15) — Phase 0 scope per D-008.

    Mechanisms: (1) sector-masked imagination training — whole spatial sectors
    of the input are hidden and the model must maintain beliefs about them;
    (2) latent advection prior — hidden cells evolve by a learned flow field
    (object permanence by construction); (3) epistemic gating — per-cell
    log-variance, the uncertainty signal that later triggers H2 modality
    steering and fallback margins.
    """
    enabled: bool = True
    mask_prob: float = 0.5        # fraction of batches that get a masked sector
    weight: float = 0.5           # loss weight of the imagination NLL
    depth: int = 3                # refinement blocks over the advected prior
    observed_weight: float = 0.1  # small consistency weight on visible cells


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    batch_size: int = 64          # MICRO-batch (what the GPU holds at once)
    accum_steps: int = 1          # optimizer effective batch = batch_size * accum
    save_every: int = 500         # checkpoint cadence (interruptible-pod proof)
    steps: int = 2000
    warmup_steps: int = 100
    device: str = "auto"          # auto -> cuda if available
    seed: int = 0
    log_every: int = 50
    out_dir: str = "experiments/dev"
    rollout_k: int = 1            # >1: recursive K-step rollout loss (bake-off
                                  # lever, 2512.24497 multistep-as-augmentation)


@dataclass
class StackConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    # Parametric tactical predictor (maneuver-horizon dynamics). None = off.
    tactical_pred: PredictorConfig | None = None
    # Trained tactical/strategic policy brains (D-030). None = off (base model).
    tactical_policy: TacticalPolicyConfig | None = None
    strategic_policy: StrategicPolicyConfig | None = None
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    tactical: TacticalConfig = field(default_factory=TacticalConfig)
    h15: H15Config = field(default_factory=H15Config)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def smoke_config() -> StackConfig:
    """Tiny config that must train on CPU or the RTX 4060 in <2 min (CI smoke)."""
    cfg = StackConfig()
    cfg.encoder = EncoderConfig(in_channels=1, image_size=64, patch_size=8,
                                d_model=64, depth=2, n_heads=2)
    cfg.predictor = PredictorConfig(d_model=64, depth=2, n_heads=2, window=4,
                                    horizons=(1, 2), action_dim=2)
    cfg.h15.depth = 1
    cfg.loss.sigreg.n_slices = 64
    cfg.train.batch_size = 16
    cfg.train.steps = 30
    cfg.train.warmup_steps = 5
    return cfg


def base250_config() -> StackConfig:
    """TanitAD-4B-M — the Phase 0 main-track model (~250 M params, D-008).

    Component budget (measured via count_params at instantiation):
      encoder ViT d768 x 14 blocks .......... ~99 M   (operative perception)
      operative predictor d768 x 12 + FiLM .. ~103 M  (action-conditioned dynamics)
      tactical predictor d512 x 6 ........... ~23 M   (maneuver-horizon dynamics,
                                                       MoE upgrade lands in WP4)
      H15 imagination field d768 x 3 ........ ~22 M   (advection + refine + sigma)
      inverse dynamics + heads .............. ~3 M
      strategic VQ + graph .................. ~0.1 M  (non-parametric by design)
    Trains on A40 (48 GB). Fits the 4060 only at batch <= 8 for pipeline debug.
    Stage-A input: BEV/gray 128 px; Stage-B switches in_channels=6 (2-frame RGB),
    image_size=224 via CLI overrides — parameter count barely moves.
    """
    cfg = StackConfig()
    cfg.encoder = EncoderConfig(in_channels=1, image_size=128, patch_size=16,
                                d_model=768, depth=14, n_heads=12)
    cfg.predictor = PredictorConfig(d_model=768, depth=12, n_heads=12, window=8,
                                    horizons=(1, 2, 4), action_dim=2)
    cfg.tactical_pred = PredictorConfig(d_model=512, depth=6, n_heads=8, window=8,
                                        horizons=(8, 16), action_dim=2)
    cfg.readout = ReadoutConfig(grid=4, d_readout=128)
    cfg.h15 = H15Config(enabled=True, depth=3)
    cfg.train.lr = 3e-4
    cfg.train.batch_size = 64
    cfg.train.steps = 60_000
    cfg.train.warmup_steps = 2_000
    return cfg


def base250cam_config() -> StackConfig:
    """TanitAD-4B-M, real-camera variant (D-009 real-first; D-015 input spec).

    Input per step: 3 RGB frames at 100 ms spacing channel-stacked
    (9 channels, [t-200ms, t-100ms, t]) at 256 px -> 16x16 token grid, so
    acceleration/curvature are observable inside one encoder input; the
    predictor adds ~800 ms of action history over its 8-step window.
    PRIMARY Phase 0 training config (comma2k19 / PhysicalAI-AV front camera).
    """
    cfg = base250_config()
    cfg.encoder = EncoderConfig(in_channels=9, image_size=256, patch_size=16,
                                d_model=768, depth=14, n_heads=12)
    return cfg


def flagship4b_config() -> StackConfig:
    """TanitAD-4B FLAGSHIP (D-030 recovery) — the FULL trained-and-wired
    4-brain stack: operative predictor + tactical_pred dynamics + TRAINED
    tactical policy + TRAINED strategic transformer, hierarchical grounding at
    every level, SIGReg with the position-subspace relaxation.

    Budget: the two new policy brains (~31 M) are funded by rebalancing the
    shared trunk DOWN from base250cam (encoder 14 -> 11, operative predictor
    12 -> 10) so the 4-brain total stays budget-matched to ~261 M (the REF-B
    rebalance philosophy). Measured total ~260 M (test_flagship4b pins +-5 %).

    Per-brain conditioning flow: strategic ctx --(FiLM)--> tactical --(intent
    FiLM)--> operative predictor. The grounding heads live OUTSIDE the model
    (saved under separate ckpt keys) so a vanilla WorldModel still loads a 4b
    checkpoint. REF-A composes the IDENTICAL brains, differing only in the
    encoder (frozen DINO + adapter) and the SIGReg target (predictor-only).
    """
    cfg = base250cam_config()
    # Rebalance the shared trunk to fund the trained policy brains.
    cfg.encoder = dataclasses.replace(cfg.encoder, depth=12)      # 14 -> 12
    cfg.predictor = dataclasses.replace(cfg.predictor, depth=10)  # 12 -> 10
    cfg.tactical_policy = TacticalPolicyConfig()
    cfg.strategic_policy = StrategicPolicyConfig()
    cfg.loss.sigreg.free_dims = 64            # §B.3 ego-motion subspace relaxation
    return cfg


def flagship4b_smoke_config() -> StackConfig:
    """Tiny CPU flagship (CI smoke / tests / dry runs) — same 4-brain structure,
    same conditioning wiring and grounding, shrunk widths and horizons (all
    <= max_horizon 4). 1-channel 64 px episodes."""
    cfg = smoke_config()
    cfg.tactical_pred = PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                        horizons=(3, 4), action_dim=2)
    cfg.tactical_policy = TacticalPolicyConfig(
        d_model=32, depth=1, n_heads=2, waypoint_horizons=(2, 4), d_intent=16,
        cadence=2)
    cfg.strategic_policy = StrategicPolicyConfig(
        d_cmd=16, d_model=32, depth=1, n_heads=2, d_ctx=16, cadence=4)
    cfg.loss.sigreg.free_dims = 8             # state_dim 512 -> 8 free, 504 regd
    return cfg
