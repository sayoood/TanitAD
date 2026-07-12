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

from tanitad.config import (PredictorConfig, StrategicPolicyConfig,
                            TacticalPolicyConfig, base250cam_config)
from tanitad.models.fourbrain import (StrategicPolicy, TacticalPolicy,
                                      run_hierarchy)
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


class DinoGridAdapter(nn.Module):
    """Spatially-faithful adapter: standardized tokens -> SpatialGridReadout.

    Stage-2b fix (Sayed review 2026-07-11): the v1 mean-pool adapter destroys
    the spatial token layout that the DINO-WM lineage (DINO-WM, EponaV2,
    DINO-world) relies on — and that the main model keeps via its own
    SpatialGridReadout. This adapter reuses THAT class unchanged (grid=4,
    d_readout=128 -> out_dim 2048, mirroring the main stack's state geometry)
    so the REF-A comparison isolates the ENCODER, not the readout. Accepts any
    leading shape [..., N_tokens, D].
    """

    def __init__(self, n_tokens: int = 256, d_in: int = 768, grid: int = 4,
                 d_readout: int = 128):
        super().__init__()
        from tanitad.models.readout import SpatialGridReadout
        self.readout = SpatialGridReadout(n_tokens, d_in, grid=grid,
                                          d_readout=d_readout)
        self.out_dim = self.readout.out_dim

    def forward(self, tokens: Tensor) -> Tensor:
        lead = tokens.shape[:-2]
        flat = tokens.reshape(-1, *tokens.shape[-2:])
        return self.readout(flat).reshape(*lead, self.out_dim)


class RefAModel(nn.Module):
    """Standardizer + adapter + the UNCHANGED shared operative predictor, and
    (optionally, D-030) the SHARED trained tactical/strategic policy brains —
    the frozen-DINO twin of the flagship ``WorldModel``.

    encode / encode_window / predict / hierarchy mirror the WorldModel interface
    the trainer needs, with frozen-feature token grids replacing frames.
    ``adapter_kind``: "grid" (stage-2b default for new runs — spatially
    faithful, state_dim = readout out_dim 2048) or "pool" (v1 mean-pool,
    kept for loading pre-revision checkpoints).

    4-brain composition (``tactical_policy_cfg`` + ``strategic_policy_cfg``, a
    matched set): builds the SAME ``StrategicPolicy`` / ``TacticalPolicy`` /
    intent-conditioned ``OperativePredictor`` / tactical-predictor dynamics that
    ``WorldModel`` holds, from the SAME config objects, on the adapter STATE.
    The brains are state-dim-agnostic, so with the grid adapter matched to the
    flagship's readout geometry (``grid`` / ``grid_d_readout``) every shared
    brain is byte-for-byte the same shape — the frozen-DINO adapter (vs the
    from-scratch ViT) is then the ONLY model-axis difference. Both configs None
    (the default) reproduces the base REF-A exactly (a policy-less checkpoint is
    a strict subset — vanilla-load safe). See :func:`fourbrain.run_hierarchy`.
    """

    def __init__(self, pred_cfg: PredictorConfig | None = None,
                 d_dino: int = 768, state_dim: int = 768,
                 bottleneck: bool = False, sigreg_slices: int = 512,
                 sigreg_beta: float = 1.0, adapter_kind: str = "pool",
                 n_tokens: int = 256, *, grid: int = 4,
                 grid_d_readout: int = 128,
                 tactical_policy_cfg: TacticalPolicyConfig | None = None,
                 strategic_policy_cfg: StrategicPolicyConfig | None = None,
                 tactical_pred_cfg: PredictorConfig | None = None):
        super().__init__()
        assert adapter_kind in ("pool", "grid"), adapter_kind
        self.pred_cfg = pred_cfg if pred_cfg is not None \
            else refa_predictor_config()
        self.adapter_kind = adapter_kind
        self.standardizer = FeatureStandardizer(d_dino)
        if adapter_kind == "grid":
            self.adapter: nn.Module = DinoGridAdapter(
                n_tokens, d_dino, grid=grid, d_readout=grid_d_readout)
            state_dim = self.adapter.out_dim
        else:
            self.adapter = DinoAdapter(d_dino, state_dim,
                                       bottleneck=bottleneck)
        self.state_dim = state_dim
        # ---- 4-brain composition (D-030): the SHARED policy/predictor brains,
        # mirroring WorldModel EXACTLY. The trained policy brains are a matched
        # set (the hierarchy needs the strategic ctx to condition tactical):
        # enable both or neither.
        if (tactical_policy_cfg is None) != (strategic_policy_cfg is None):
            raise ValueError("tactical_policy and strategic_policy must be "
                             "enabled together (the 4-brain hierarchy)")
        self.tactical_policy_cfg = tactical_policy_cfg
        self.strategic_policy_cfg = strategic_policy_cfg
        intent_dim = (tactical_policy_cfg.d_intent
                      if tactical_policy_cfg is not None else None)
        # Operative predictor — intent-conditioned when the tactical brain is on
        # (identity start; ``intent_dim=None`` reproduces the base REF-A
        # predictor exactly, so a policy-less checkpoint is a strict subset).
        # Imported, never copied: the comparison isolates the encoder axis.
        self.predictor = OperativePredictor(self.pred_cfg, state_dim,
                                            intent_dim=intent_dim)
        # Tactical-predictor dynamics (maneuver-horizon JEPA) — the SAME family
        # and config the flagship uses; None = off (base REF-A).
        self.tactical_pred = (OperativePredictor(tactical_pred_cfg, state_dim)
                              if tactical_pred_cfg is not None else None)
        # Trained strategic transformer + tactical policy (SHARED classes) — the
        # state-dim-agnostic brains compose on the adapter state identically.
        self.strategic_policy = (
            StrategicPolicy(strategic_policy_cfg, state_dim, self.pred_cfg.window)
            if strategic_policy_cfg is not None else None)
        self.tactical_policy = (
            TacticalPolicy(tactical_policy_cfg, state_dim, self.pred_cfg.window,
                           d_cond=strategic_policy_cfg.d_ctx)
            if tactical_policy_cfg is not None else None)
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

    def predict(self, states: Tensor, actions: Tensor,
                intent: Tensor | None = None) -> dict[int, Tensor]:
        """Causal window -> imagined future adapter-states per horizon
        (optionally intent-conditioned by the tactical brain, 4-brain path)."""
        return self.predictor(states, actions, intent=intent)

    def hierarchy(self, states: Tensor, actions: Tensor,
                  nav_cmd: Tensor | None = None) -> dict:
        """Run the shared strategic->tactical->operative chain on adapter states
        (identical to ``WorldModel.hierarchy``; see
        :func:`fourbrain.run_hierarchy`). Needs the 4-brain policy configs."""
        return run_hierarchy(self, states, actions, nav_cmd)

    @classmethod
    def from_stack_config(cls, cfg, *, n_tokens: int, d_dino: int = 768,
                          adapter_kind: str = "grid",
                          sigreg_slices: int | None = None) -> "RefAModel":
        """Build the 4-brain REF-A that MIRRORS ``WorldModel(cfg)`` module-for-
        module: the SAME shared predictor / tactical_pred / tactical_policy /
        strategic_policy from the SAME ``StackConfig``, on the frozen-DINO grid
        adapter. The adapter reuses ``cfg.readout``'s grid geometry, so the
        adapter state_dim equals the flagship readout's out_dim and every shared
        brain is byte-for-byte the same shape. The ONLY model-axis difference
        from ``WorldModel(cfg)`` is the encoder (adapter vs from-scratch ViT).
        ``n_tokens`` is the DINO token-grid count (256 for DINOv2-B/14 @224)."""
        assert (cfg.tactical_policy is not None
                and cfg.strategic_policy is not None), \
            "from_stack_config builds the 4-brain REF-A — cfg needs both policies"
        return cls(
            pred_cfg=cfg.predictor, d_dino=d_dino, n_tokens=n_tokens,
            adapter_kind=adapter_kind, grid=cfg.readout.grid,
            grid_d_readout=cfg.readout.d_readout,
            sigreg_slices=(sigreg_slices if sigreg_slices is not None
                           else cfg.loss.sigreg.n_slices),
            sigreg_beta=cfg.loss.sigreg.beta,
            tactical_policy_cfg=cfg.tactical_policy,
            strategic_policy_cfg=cfg.strategic_policy,
            tactical_pred_cfg=cfg.tactical_pred)

    @torch.no_grad()
    def adapter_dim_std(self, states: Tensor) -> float:
        """Collapse monitor: mean per-dim std of adapter outputs (trainable
        targets — falling toward 0 means collapse-to-easy-targets)."""
        flat = states.detach().float().reshape(-1, states.shape[-1])
        return float(flat.std(0).mean())
