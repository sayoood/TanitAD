"""REF-A model: frozen-DINO features -> trainable adapter -> shared predictor.

Stage 2 of the REF-A pipeline (REFERENCE_ARCHITECTURES.md): stage 1
(`scripts/dino_precompute.py`) wrote per-episode fp16 token grids
[T, 256, 768]; this module trains ONLY from those files — no images, no
encoder in the loop (stability item 6: feature-cache training).

Stability assurance mapping (spec items 1–6):
  1. `FeatureStandardizer` — per-channel mean/var computed ONCE over the
     train corpus, stored as frozen buffers; a loaded checkpoint reuses the
     stored stats (refitting raises).
  2. No gradient can touch the encoder — by construction: features are data
     tensors loaded from disk (`requires_grad=False` end-to-end; pinned by
     tests/test_refa.py).
  3. SigReg only on PREDICTOR outputs (≥256-samples/step floor, F-2 rule) —
     wired in scripts/refa_train.py.
  4. Adapter LR warmup 10× longer than the predictor's (separate param
     groups + gradient-norm monitor rows) — wired in scripts/refa_train.py.
  5. I2 batch-consistency: LayerNorm only, no BatchNorm/dropout anywhere.
  6. Feature-cache training — this whole module.

The operative predictor is imported UNCHANGED from the main stack
(`tanitad.models.predictor.OperativePredictor`) at base250cam dims (d768),
so the comparison isolates the encoder axis. Prediction targets live in
ADAPTER space (adapter outputs of future timesteps); because the adapter is
trainable, collapse-to-easy-targets is the known failure mode — the trainer
logs adapter output per-dim std every log interval as the collapse monitor,
and the inverse-dynamics head + SigReg-on-predictions provide the
anti-collapse pressure (LeJEPA doctrine: no stop-grad/EMA crutch, A1).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from tanitad.config import PredictorConfig, base250cam_config
from tanitad.models.inverse_dynamics import InverseDynamicsHead
from tanitad.models.predictor import OperativePredictor
from tanitad.models.sigreg import SigReg


def refa_predictor_config() -> PredictorConfig:
    """The shared operative-predictor config at main-track dims (d768).

    Taken from base250cam so REF-A's predictor capacity is IDENTICAL to the
    main model's — the comparison isolates the encoder (REF-A design rule).
    """
    return base250cam_config().predictor


class FeatureStandardizer(nn.Module):
    """Per-channel standardization of frozen DINO features (spec item 1).

    mean/std are computed ONCE over the training corpus via :meth:`fit` and
    stored as buffers, so they (a) travel inside every checkpoint and (b) can
    never drift between train and eval. A second :meth:`fit` raises — a
    checkpoint-loaded standardizer therefore reuses its stored stats and can
    never silently recompute them.
    """

    def __init__(self, dim: int = 768, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, feature_tensors: Iterable[Tensor]) -> None:
        """Accumulate per-channel mean/var over an iterable of [..., D]
        tensors (fp16 ok; accumulation runs in float64). One-shot: raises if
        already fitted (frozen statistics — no train/eval drift possible)."""
        if bool(self.fitted):
            raise RuntimeError(
                "FeatureStandardizer already fitted — REF-A stats are frozen "
                "(spec item 1); build a new standardizer instead of refitting")
        d = self.mean.shape[0]
        total = torch.zeros(d, dtype=torch.float64)
        total_sq = torch.zeros(d, dtype=torch.float64)
        n = 0
        for x in feature_tensors:
            flat = x.reshape(-1, d).to(torch.float64)
            total += flat.sum(0)
            total_sq += flat.pow(2).sum(0)
            n += flat.shape[0]
        if n == 0:
            raise ValueError("standardizer fit got zero feature rows")
        mean = total / n
        var = (total_sq / n - mean.pow(2)).clamp_min(0.0)
        self.mean.copy_(mean.to(self.mean.dtype))
        self.std.copy_(var.sqrt().clamp_min(self.eps).to(self.std.dtype))
        self.fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        if not bool(self.fitted):
            raise RuntimeError(
                "FeatureStandardizer not fitted — call fit() over the train "
                "corpus or load a checkpoint that carries the stats")
        return (x.float() - self.mean) / self.std


class DinoAdapter(nn.Module):
    """Mean-pool the token grid -> LayerNorm -> Linear(d_in -> d_out).

    Deliberately small and simple (spec: 'this is where stability lives').
    ``bottleneck=True`` inserts a GELU + second Linear (the spec's optional
    2-layer variant). Accepts any leading shape [..., N_tokens, D].
    """

    def __init__(self, d_in: int = 768, d_out: int = 768,
                 bottleneck: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        if bottleneck:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        else:
            self.proj = nn.Linear(d_in, d_out)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.proj(self.norm(tokens.mean(dim=-2)))


class RefAModel(nn.Module):
    """Standardizer + adapter + the UNCHANGED shared operative predictor.

    encode / encode_window / predict mirror the WorldModel interface the
    trainer needs, with frozen-feature token grids replacing frames.
    """

    def __init__(self, pred_cfg: PredictorConfig | None = None,
                 d_dino: int = 768, state_dim: int = 768,
                 bottleneck: bool = False, sigreg_slices: int = 512,
                 sigreg_beta: float = 1.0):
        super().__init__()
        self.pred_cfg = pred_cfg if pred_cfg is not None \
            else refa_predictor_config()
        self.state_dim = state_dim
        self.standardizer = FeatureStandardizer(d_dino)
        self.adapter = DinoAdapter(d_dino, state_dim, bottleneck=bottleneck)
        # Imported, never copied: the comparison isolates the encoder axis.
        self.predictor = OperativePredictor(self.pred_cfg, state_dim)
        # A5 grounding — also the cheapest anti-collapse pressure on the
        # trainable adapter space (constant states cannot encode actions).
        self.inv_dyn = InverseDynamicsHead(
            state_dim, self.pred_cfg.action_dim,
            hidden=max(256, min(1024, state_dim // 2)))
        self.sigreg = SigReg(sigreg_slices, sigreg_beta)

    def encode(self, feats: Tensor) -> Tensor:
        """Token grid [..., N, D] (fp16 ok) -> adapter state [..., S]."""
        return self.adapter(self.standardizer(feats))

    def encode_window(self, feats: Tensor) -> Tensor:
        """[B, W, N, D] -> [B, W, S] (WorldModel.encode_window analog)."""
        return self.encode(feats)

    def predict(self, states: Tensor, actions: Tensor) -> dict[int, Tensor]:
        """Causal window -> imagined future adapter-states per horizon."""
        return self.predictor(states, actions)

    @torch.no_grad()
    def adapter_dim_std(self, states: Tensor) -> float:
        """Collapse monitor: mean per-dim std of adapter outputs (trainable
        targets — falling toward 0 means collapse-to-easy-targets)."""
        flat = states.detach().float().reshape(-1, states.shape[-1])
        return float(flat.std(0).mean())
