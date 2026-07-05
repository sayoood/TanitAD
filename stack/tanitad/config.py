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
    in_channels: int = 1          # 1 = BEV toy; 3/6 = camera (2-frame stack)
    image_size: int = 64
    patch_size: int = 8           # 8 for 64px toy -> 8x8 grid; 16 for 224px camera
    d_model: int = 128
    depth: int = 6
    n_heads: int = 4
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
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    batch_size: int = 64
    steps: int = 2000
    warmup_steps: int = 100
    device: str = "auto"          # auto -> cuda if available
    seed: int = 0
    log_every: int = 50
    out_dir: str = "experiments/dev"


@dataclass
class StackConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    tactical: TacticalConfig = field(default_factory=TacticalConfig)
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
    cfg.loss.sigreg.n_slices = 64
    cfg.train.batch_size = 16
    cfg.train.steps = 30
    cfg.train.warmup_steps = 5
    return cfg
