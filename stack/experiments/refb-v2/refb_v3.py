"""REF-B model: TanitAD-4B-E2E — hierarchical vision->action, NO world model.

The honest end-to-end opponent to the latent-world-model stack
(REFERENCE_ARCHITECTURES.md REF-B, upgraded 2026-07-11 by Sayed to the full
4-layer 4B structure): operative + tactical + strategic + fallback, budget-
matched to the main model's ~261 M total within +-2 % (pinned by
tests/test_refb.py against base250cam read programmatically).

Layer map (vs the main 4B stack; rev2 = Sayed's 2026-07-11 review):
  1. Encoder + readout : the SAME ViTEncoder/SpatialGridReadout classes,
     imported UNCHANGED, trained from scratch. The ~130 M freed by having no
     predictor/imagination goes into a DEEPER encoder (25 blocks vs 14) and a
     WIDER operative head (d768 x 6 causal blocks).
  2. Operative head (10-20 Hz): causal transformer over the encoded state
     window -> direct (steer, accel) regression + 0.5 s action sequence via
     DIRECT multi-horizon heads — NO autoregressive recursion (D3-
     decomposition finding). FiLM-conditioned on the tactical intent token
     (same FiLM class as the main predictor).
  3. Tactical head (1-2 Hz): maneuver-class distribution ("target behavior",
     pseudo-labeled by scripts/refb_labels.py) + 2 s waypoints at direct
     horizons; emits an intent token that conditions the operative head.
     d512 x 6 (rev2), FiLM-conditioned on the STRATEGIC CONTEXT TOKEN.
  4. Strategic head (rev2): a real d384 x 4 causal transformer over the
     shared state window, FiLM-conditioned on the nav-command embedding
     {follow, left, right, straight} (default `follow` when unlabeled; nav
     commands are DERIVED per window from 15-25 s of future heading by
     scripts/refb_labels.nav_command). Outputs (a) a context token [d_ctx]
     that conditions the tactical head, and (b) route-heading logits
     (route_left / route_straight / route_right) trained by an auxiliary CE
     on the same derivation — DIRECT supervision for the layer, not just
     trickle-down gradient. ~8.4 M params (the ~9 M-class rev2 module).
  5. Fallback (label-free, NO world model):
     (a) ConfidenceHead — predicts its own realized waypoint error; inputs
         and targets are DETACHED, so its gradient can never reach the
         encoder or the other heads (pinned by test).
     (b) FeatureOOD — diagonal Mahalanobis distance of the encoder state to
         running train-feature statistics, FROZEN after warmup (buffers, zero
         trainable params). Both are logged every step by the trainer.
     (The H9 rule-barrier envelope attaches at EVAL time — out of scope.)

What REF-B structurally CANNOT do (pre-registered, the point of the
reference): no imagination error signal (D8), no latent rollout (LOPS/SC-02),
no imagine-and-select (D4 tactical lift), no closure reasoning (SC-01).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from tanitad.config import EncoderConfig, ReadoutConfig, base250cam_config
from tanitad.models.encoder import ViTEncoder
from tanitad.models.inverse_dynamics import InverseDynamicsHead
from tanitad.models.predictor import CausalBlock
from tanitad.models.readout import SpatialGridReadout

# Strategic vocabulary — index 0 is the unlabeled default. Kept 4-wide for
# interface stability: `straight` is reserved for intersection topologies and
# is NOT emitted by the comma pseudo-derivation (only follow/left/right are;
# see scripts/refb_labels.nav_command).
NAV_COMMANDS = ("follow", "left", "right", "straight")

# Route-heading aux classes (rev2) — order pinned against refb_labels.py
# by tests/test_refb.py.
ROUTE_CLASSES = ("route_left", "route_straight", "route_right")

# Tactical "target behavior" vocabulary (pseudo-labels: scripts/refb_labels.py).
MANEUVER_CLASSES = ("lane_keep", "turn_left", "turn_right", "accelerate",
                    "brake_stop")


@dataclass
class OperativeHeadConfig:
    d_model: int = 768
    depth: int = 6
    n_heads: int = 12
    action_dim: int = 2
    # 0.5 s @ 10 Hz: direct heads for a_t .. a_{t+action_seq-1} (offset 0 is
    # the reactive (steer, accel) regression; the rest is the sequence).
    action_seq: int = 5


@dataclass
class TacticalHeadConfig:
    d_model: int = 512
    depth: int = 6                # rev2: 4 -> 6 (funded by encoder 27 -> 25)
    n_heads: int = 8
    n_maneuvers: int = len(MANEUVER_CLASSES)
    # 2 s @ 10 Hz in 0.5 s strides — direct heads, no recursion.
    waypoint_horizons: tuple[int, ...] = (5, 10, 15, 20)
    d_intent: int = 256           # intent token dim (FiLM cond of operative)


@dataclass
class StrategicConfig:
    n_commands: int = len(NAV_COMMANDS)
    d_cmd: int = 128              # nav-command embedding = FiLM cond (rev2)
    d_model: int = 384            # rev2: real transformer over the window
    depth: int = 4
    n_heads: int = 6
    d_ctx: int = 256              # context token = FiLM cond of tactical
    n_route: int = len(ROUTE_CLASSES)   # aux route-heading classes


@dataclass
class FallbackConfig:
    hidden: int = 512             # confidence-head MLP width


@dataclass
class RefBConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    window: int = 8               # shared causal state-window (main: window 8)
    operative: OperativeHeadConfig = field(default_factory=OperativeHeadConfig)
    tactical: TacticalHeadConfig = field(default_factory=TacticalHeadConfig)
    strategic: StrategicConfig = field(default_factory=StrategicConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    # Ego-dynamics additions (2026-07-14). Gated; default off == the
    # pre-2026-07-14 REF-B EXACTLY (no extra params, byte-identical
    # state_dict, so existing checkpoints resume). Both are switched on
    # together by the trainer's --speed-input flag.
    speed_input: bool = False      # feed v0 = pose_last[:,3] as proprioception
    aux_accel: bool = False        # aux longitudinal-accel head (+ logged r2)
    # --- refbpatch (2026-07-17) — all gated, default off == pre-patch REF-B
    # EXACTLY (no extra params, state_dict byte-identical, ckpts still load).
    # Turned on together by the trainer's --refbpatch flag.
    aux_yaw: bool = False          # aux yaw-rate head (fixes yaw-blind encoder)
    ego_dropout: float = 0.0       # p(drop v0) in training — breaks the shortcut
    path_dists: tuple = ()         # fixed-ARC-LENGTH path head (m); () == off
    # --- v2 architecture (2026-07-18, Sayed-approved). Gated; default off ==
    # pre-v2 REF-B EXACTLY (state_dict byte-identical, existing ckpts still
    # load). Turned on together by the trainer's --arch-v2 flag.
    # B2: feed yr0 (t=0 yaw-rate) ALONGSIDE v0 -> ego_emb widens to Linear(2,·).
    yaw_input: bool = False
    # B1: replace the unimodal fixed-DISTANCE path head with a MULTIMODAL
    # anchored (DiffusionDrive-style) DISTANCE-path decoder over path_dists
    # arc-lengths (the speed-invariant geometry space). Drops the fixed-TIME
    # wp_heads. Requires path_dists set (refbpatch).
    anchored_tactical: bool = False
    # FINAL (Sayed 2026-07-18): B1 = TIME-anchored (DiffusionDrive/VADv2-
    # faithful) over waypoint_horizons. "distance" (arc-length/Frenet) is kept
    # as a switch but is NOT the chosen direction.
    anchor_space: str = "time"     # "time" (waypoint_horizons) | "distance"
    anchor_n: int = 128            # FPS anchor-vocabulary size
    anchor_pool: int = 4096        # synthetic default-anchor pool the FPS draws
    anchor_d: int = 384            # anchored-decoder width (REF-C-base)
    anchor_layers: int = 4         # cross-attention layers
    anchor_heads: int = 8


def refb_config() -> RefBConfig:
    """TanitAD-4B-E2E at main-track scale — budget-matched to base250cam.

    The encoder trunk is base250cam's EncoderConfig read PROGRAMMATICALLY
    (same in_channels/image_size/patch/d_model/n_heads) with depth 25 instead
    of 14 — that is where the freed predictor+imagination budget (~130 M)
    goes, together with the d768 x 6 operative head. Rev2 rebalance (Sayed
    review): encoder 27 -> 25 funds the strategic transformer (d384 x 4) and
    the deeper tactical head (d512 x 4 -> 6).

    Measured budget (count_params at instantiation, vs main 262.8 M):
      encoder ViT d768 x 25 + readout ....... ~179.3 M
      operative head d768 x 6 + inv-dyn ..... ~51.7 M
      tactical head d512 x 6 ................ ~21.7 M
      strategic d384 x 4 + nav embedding .... ~8.4 M   (rev2 ~9 M-class module)
      fallback confidence head .............. ~1.4 M   (OOD = buffers, 0 params)
      total ................................. ~262.5 M  (-0.1 % vs main)
    """
    cfg = RefBConfig()
    main_enc = base250cam_config().encoder
    cfg.encoder = dataclasses.replace(main_enc, depth=25)
    cfg.readout = ReadoutConfig(grid=4, d_readout=128)      # state_dim 2048
    return cfg


def refb_smoke_config() -> RefBConfig:
    """Tiny CPU config (CI smoke / tests / dry runs) — same structure, same
    horizons/action-seq lengths, shrunk widths. Episodes: 1-channel 64 px."""
    cfg = RefBConfig()
    cfg.encoder = EncoderConfig(in_channels=1, image_size=64, patch_size=8,
                                d_model=64, depth=2, n_heads=2)
    cfg.readout = ReadoutConfig(grid=4, d_readout=32)       # state_dim 512
    cfg.window = 4
    cfg.operative = OperativeHeadConfig(d_model=64, depth=2, n_heads=2)
    cfg.tactical = TacticalHeadConfig(d_model=32, depth=1, n_heads=2,
                                      d_intent=16)
    cfg.strategic = StrategicConfig(d_cmd=16, d_model=32, depth=1, n_heads=2,
                                    d_ctx=16)
    cfg.fallback = FallbackConfig(hidden=32)
    return cfg


def default_distance_anchors(dists: tuple, n_anchors: int,
                             pool_size: int = 4096, seed: int = 0,
                             device: str = "cpu") -> Tensor:
    """B1 anchor vocabulary in DISTANCE space: FPS over a synthetic pool of
    constant-curvature arc-length paths (straight + left/right arcs of varying
    radius), each read at the ``dists`` arc-lengths -> [n_anchors, len(dists), 2].

    Distance-native (mirrors refc.default_anchors but in the speed-invariant
    arc-length space of refb_labels.path_targets, per the 2026-07-18 B1 space
    correction: geometry, not time). Deterministic given ``seed``; the trainer
    OVERRIDES it with FPS over real GT path_targets (build_distance_anchors)."""
    from tanitad.refs.refc import furthest_point_sample
    g = torch.Generator(device=device).manual_seed(seed)
    s = torch.tensor(dists, dtype=torch.float32, device=device)        # [S]
    # curvature kappa (1/m): heavy-tailed over +-0.25 (radius >= 4 m); roads are
    # mostly gentle but sharp turns are the modes that matter.
    kappa = (torch.rand(pool_size, generator=g, device=device) - 0.5)
    kappa = torch.sign(kappa) * kappa.abs().pow(1.5) * 0.5             # [M]
    small = kappa.abs() < 1e-4
    ksafe = torch.where(small, torch.ones_like(kappa), kappa)
    ks = kappa[:, None] * s[None, :]                                   # [M, S]
    x = torch.where(small[:, None], s[None, :].expand_as(ks),
                    torch.sin(ks) / ksafe[:, None])
    y = torch.where(small[:, None], torch.zeros_like(ks),
                    (1.0 - torch.cos(ks)) / ksafe[:, None])
    pool = torch.stack([x, y], dim=-1)                                # [M, S, 2]
    return furthest_point_sample(pool, n_anchors, seed=seed).contiguous()


class OperativeHead(nn.Module):
    """Causal transformer over the state window -> direct action heads.

    forward(states [B, W, S], intent [B, d_intent]) -> [B, K, A] where
    row 0 is the reactive (steer, accel) regression at t and rows 1..K-1 are
    the 0.5 s action sequence — one DIRECT linear head per offset, no
    autoregressive recursion (D3-decomposition finding). The tactical intent
    token conditions every block via FiLM (the main stack's mechanism,
    CausalBlock imported unchanged). Deliberately NOT fed past actions: BC on
    (state, past-action) invites the copycat shortcut.
    """

    def __init__(self, cfg: OperativeHeadConfig, state_dim: int, window: int,
                 d_intent: int):
        super().__init__()
        self.cfg, self.window = cfg, window
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CausalBlock(d, cfg.n_heads, cond_dim=d_intent)
            for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict(
            {str(k): nn.Linear(d, cfg.action_dim)
             for k in range(cfg.action_seq)})

    def forward(self, states: Tensor, intent: Tensor) -> Tensor:
        b, w, _ = states.shape
        assert w == self.window, f"window mismatch: {w} != {self.window}"
        x = self.in_proj(states) + self.pos[:, :w]
        cond = intent.unsqueeze(1).expand(-1, w, -1)          # [B, W, d_int]
        mask = torch.triu(torch.ones(w, w, device=states.device,
                                     dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h = self.norm(x[:, -1])                               # [B, d]
        return torch.stack([self.heads[str(k)](h)
                            for k in range(self.cfg.action_seq)], dim=1)


class TacticalHead(nn.Module):
    """Maneuver distribution + geometry decoder + intent token (1-2 Hz layer).

    forward(states [B, W, S], ctx [B, d_cond]) ->
        (maneuver_logits [B, M], geom {key: [B, 2]}, intent [B, d_intent],
         tac_h [B, d], extra dict)
    Base (pre-v2): ``geom`` is DIRECT per-horizon ego-frame TIME waypoints.
    v2 (``anchored``, B1 2026-07-18): ``geom`` is a MULTIMODAL anchored
    DISTANCE-path — REF-C's AnchoredDiffusionDecoder (classifier mode) over the
    path_dists ARC-LENGTHS (the speed-invariant geometry space) REPLACES the
    unimodal wp_heads AND the refbpatch fixed-distance path head. The decoder
    cross-attends the tactical window sequence and is conditioned on the
    strategic ctx; the maneuver-head logits reweight the per-anchor confidences
    (H19), scale-matched through a LayerNorm so the prior BIASES rather than
    overwrites the base confidences (the H26 intent->action swamping lesson).
    ``extra`` carries anchor_logits / anchor_traj / sel_idx (+ debug norms).
    The strategic CONTEXT TOKEN conditions every block via FiLM.
    """

    def __init__(self, cfg: TacticalHeadConfig, state_dim: int, window: int,
                 d_cond: int, anchored: bool = False,
                 anchor_space: str = "time", path_dists: tuple = (),
                 anchor_n: int = 128, anchor_pool: int = 4096,
                 anchor_d: int = 384, anchor_layers: int = 4,
                 anchor_heads: int = 8):
        super().__init__()
        self.cfg, self.window = cfg, window
        self.anchored = anchored
        self.anchor_space = anchor_space
        self.path_dists = tuple(path_dists)
        # The anchored decoder lives in ONE space; its keys are the time
        # horizons (DiffusionDrive/VADv2-faithful) or the fixed arc-lengths.
        self.anchor_keys = (tuple(cfg.waypoint_horizons)
                            if anchor_space == "time" else self.path_dists)
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CausalBlock(d, cfg.n_heads, cond_dim=d_cond)
            for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.maneuver_head = nn.Linear(d, cfg.n_maneuvers)
        self.intent_proj = nn.Linear(d, cfg.d_intent)
        if anchored:
            from tanitad.refs.refc import (AnchoredDiffusionDecoder,
                                           DecoderConfig, default_anchors)
            if anchor_space not in ("time", "distance"):
                raise ValueError(f"anchor_space must be time|distance, "
                                 f"got {anchor_space}")
            if anchor_space == "distance" and not self.path_dists:
                raise ValueError("distance-anchored tactical needs path_dists")
            n_steps = len(self.anchor_keys)
            if anchor_space == "time":     # DiffusionDrive/VADv2 time anchors
                anchors = default_anchors(self.anchor_keys, anchor_n,
                                          anchor_pool, seed=0)
            else:                          # arc-length (Frenet) distance anchors
                anchors = default_distance_anchors(self.path_dists, anchor_n,
                                                   anchor_pool, seed=0)
            dcfg = DecoderConfig(d=anchor_d, n_heads=anchor_heads,
                                 layers=anchor_layers, ff_mult=4,
                                 aux_hidden=anchor_d, diffusion_steps=2,
                                 noise_std=0.1)
            # KV = the tactical window sequence [B, W, d] (fed as a fake
            # [B, d, W, 1] "map"); condition m = strategic ctx [B, d_cond].
            # graft_maneuver=False: the H19 prior is applied EXTERNALLY below
            # with a LayerNorm (H26 scale-match), not inside the decoder.
            self.wp_decoder = AnchoredDiffusionDecoder(
                feat_dim=d, n_steps=n_steps, d_meas=d_cond, d_ctx=d_cond,
                tac_latent_dim=d, anchors=anchors, cfg=dcfg,
                hierarchy=False, graft_maneuver=False,
                graft_target_latent=False, grounded_selector=False,
                n_maneuvers=cfg.n_maneuvers)
            self.maneuver_to_anchor = nn.Linear(cfg.n_maneuvers, anchor_n,
                                                bias=False)
            self.anchor_prior_norm = nn.LayerNorm(anchor_n)
        else:
            self.wp_heads = nn.ModuleDict(
                {str(k): nn.Linear(d, 2) for k in cfg.waypoint_horizons})

    def forward(self, states: Tensor, ctx: Tensor):
        b, w, _ = states.shape
        assert w == self.window, f"window mismatch: {w} != {self.window}"
        x = self.in_proj(states) + self.pos[:, :w]
        cond = ctx.unsqueeze(1).expand(-1, w, -1)             # [B, W, d_cond]
        mask = torch.triu(torch.ones(w, w, device=states.device,
                                     dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h = self.norm(x[:, -1])
        man_logits = self.maneuver_head(h)
        intent = self.intent_proj(h)
        extra: dict = {}
        if self.anchored:
            fmap = x.transpose(1, 2).unsqueeze(-1)             # [B, d, W, 1]
            dec = self.wp_decoder(fmap, ctx, steps=0)         # m = ctx
            prior = self.anchor_prior_norm(
                self.maneuver_to_anchor(torch.log_softmax(man_logits, dim=-1)))
            anchor_logits = dec["anchor_logits"] + prior      # H19 (biased)
            idx = anchor_logits.argmax(dim=1)                 # [B]
            traj = dec["anchor_traj"][torch.arange(b, device=x.device), idx]
            geom = {k: traj[:, i] for i, k in enumerate(self.anchor_keys)}
            extra = {"anchor_logits": anchor_logits,
                     "anchor_traj": dec["anchor_traj"], "sel_idx": idx,
                     "geom_space": self.anchor_space,
                     "_dbg_conf_norm":
                         dec["anchor_logits"].norm(dim=1).mean().detach(),
                     "_dbg_prior_norm": prior.norm(dim=1).mean().detach()}
        else:
            geom = {k: self.wp_heads[str(k)](h)
                    for k in self.cfg.waypoint_horizons}
        return man_logits, geom, intent, h, extra


class StrategicHead(nn.Module):
    """Route-level layer (rev2): causal transformer over the state window,
    FiLM-conditioned on the nav-command embedding.

    forward(states [B, W, S], cmd [B, d_cmd]) ->
        (ctx [B, d_ctx], route_logits [B, n_route])
    `ctx` replaces the raw nav embedding as the FiLM cond of the tactical
    head; `route_logits` (route_left / route_straight / route_right over the
    15-25 s horizon) receive an auxiliary CE in the trainer so the layer gets
    DIRECT supervision. Same CausalBlock/FiLM mechanism as the other layers,
    imported unchanged.
    """

    def __init__(self, cfg: StrategicConfig, state_dim: int, window: int):
        super().__init__()
        self.cfg, self.window = cfg, window
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CausalBlock(d, cfg.n_heads, cond_dim=cfg.d_cmd)
            for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.ctx_proj = nn.Linear(d, cfg.d_ctx)
        self.route_head = nn.Linear(d, cfg.n_route)

    def forward(self, states: Tensor, cmd: Tensor) -> tuple[Tensor, Tensor]:
        b, w, _ = states.shape
        assert w == self.window, f"window mismatch: {w} != {self.window}"
        x = self.in_proj(states) + self.pos[:, :w]
        cond = cmd.unsqueeze(1).expand(-1, w, -1)             # [B, W, d_cmd]
        mask = torch.triu(torch.ones(w, w, device=states.device,
                                     dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h = self.norm(x[:, -1])
        return self.ctx_proj(h), self.route_head(h)


class ConfidenceHead(nn.Module):
    """Fallback (a): predicts the model's own realized waypoint error.

    Trained on the realized error of the tactical waypoints, fully DETACHED
    from the main loss: RefBModel feeds it detached inputs and the trainer
    detaches the error target, so no gradient can flow from this head into
    the encoder or any other head (pinned by tests/test_refb.py).
    """

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)                        # [B]


class FeatureOOD(nn.Module):
    """Fallback (b): diagonal Mahalanobis score to running train-feature stats.

    Accumulates per-dim mean/var of the encoder state over the warmup phase
    (float64 sums, stored as buffers so they travel in every checkpoint), then
    :meth:`freeze` pins them — the label-free OOD signal a no-world-model
    stack can still have. Zero trainable parameters by design; score() and
    update() run under no_grad and can never touch the training graph.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("sum", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("sum_sq", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, z: Tensor) -> None:
        """Accumulate stats from a batch of states [B, S] (no-op once frozen)."""
        if bool(self.frozen):
            return
        flat = z.detach().reshape(-1, self.sum.shape[0]).to(torch.float64)
        self.sum += flat.sum(0)
        self.sum_sq += flat.pow(2).sum(0)
        self.count += flat.shape[0]

    def freeze(self) -> None:
        """Pin the statistics (idempotent) — call at the end of warmup."""
        self.frozen.fill_(True)

    @torch.no_grad()
    def score(self, z: Tensor) -> Tensor:
        """Per-sample diagonal Mahalanobis distance [B] (0 until >=2 samples)."""
        if float(self.count) < 2:
            return torch.zeros(z.shape[0], device=z.device)
        mean = self.sum / self.count
        var = (self.sum_sq / self.count - mean.pow(2)).clamp_min(self.eps)
        d2 = ((z.detach().to(torch.float64) - mean).pow(2) / var).mean(-1)
        return d2.sqrt().to(z.dtype)


class RefBModel(nn.Module):
    """The 4-layer E2E stack: encoder -> tactical -> operative (+ strategic
    conditioning + label-free fallback + shared inverse-dynamics aux)."""

    def __init__(self, cfg: RefBConfig):
        super().__init__()
        self.cfg = cfg
        # Layer 0/1 trunk: SAME classes as the main model, imported unchanged.
        self.encoder = ViTEncoder(cfg.encoder)
        self.readout = SpatialGridReadout(
            self.encoder.n_tokens, cfg.encoder.d_model,
            grid=cfg.readout.grid, d_readout=cfg.readout.d_readout)
        self.state_dim = self.readout.out_dim
        # Strategic (rev2): nav-command embedding -> FiLM cond of a real
        # transformer over the window; its ctx token conditions tactical.
        self.nav_emb = nn.Embedding(cfg.strategic.n_commands,
                                    cfg.strategic.d_cmd)
        self.strategic = StrategicHead(cfg.strategic, self.state_dim,
                                       cfg.window)
        self.tactical = TacticalHead(
            cfg.tactical, self.state_dim, cfg.window, cfg.strategic.d_ctx,
            anchored=cfg.anchored_tactical, anchor_space=cfg.anchor_space,
            path_dists=cfg.path_dists, anchor_n=cfg.anchor_n,
            anchor_pool=cfg.anchor_pool, anchor_d=cfg.anchor_d,
            anchor_layers=cfg.anchor_layers, anchor_heads=cfg.anchor_heads)
        self.operative = OperativeHead(cfg.operative, self.state_dim,
                                       cfg.window, cfg.tactical.d_intent)
        # Same aux as the main model (A5 grounding), same sizing rule.
        self.inv_dyn = InverseDynamicsHead(
            self.state_dim, cfg.operative.action_dim,
            hidden=max(256, min(1024, self.state_dim // 2)))
        # Fallback: confidence (detached) + feature-OOD (buffers only).
        self.confidence = ConfidenceHead(
            self.state_dim + cfg.tactical.d_intent, cfg.fallback.hidden)
        self.ood = FeatureOOD(self.state_dim)
        # Ego-dynamics additions (gated; see RefBConfig). speed_emb embeds the
        # current ego speed v0 to the nav-command dim and is ADDED to the nav
        # embedding, so the proprioceptive signal conditions the strategic
        # layer (and everything it conditions). accel_head predicts
        # longitudinal accel from the last window state — an operative-level
        # grounding aux like inv_dyn (NOT detached). Both are None (and absent
        # from state_dict) when their flags are off, keeping an off-flag model
        # byte-identical to the pre-2026-07-14 REF-B.
        # B2 (yaw_input): widen ego proprioception to [v0, yr0] -> ego_emb
        # Linear(2, d_cmd) and LayerNorm the ADDED term so it BIASES the nav
        # embedding rather than overwriting it (the H26 intent->action swamping
        # lesson). Pre-v2 speed-only keeps speed_emb Linear(1,·). Exactly one is
        # built; both absent -> byte-identical to the no-ego model.
        if cfg.yaw_input:
            self.ego_emb = nn.Linear(2, cfg.strategic.d_cmd)
            self.ego_norm = nn.LayerNorm(cfg.strategic.d_cmd)
            self.speed_emb = None
        elif cfg.speed_input:
            self.speed_emb = nn.Linear(1, cfg.strategic.d_cmd)
            self.ego_emb = self.ego_norm = None
        else:
            self.speed_emb = self.ego_emb = self.ego_norm = None
        self.accel_head = (nn.Sequential(
            nn.Linear(self.state_dim, 128), nn.GELU(), nn.Linear(128, 1))
            if cfg.aux_accel else None)
        # refbpatch (gated). yaw_head: aux yaw-rate from the last window state
        # (same shape/role as accel_head) — forces the encoder to represent
        # ego-rotation (REF-B's states->yaw R2 was 0.11). path_heads: fixed-
        # ARC-LENGTH ego-frame waypoints off the tactical latent h — a speed-
        # invariant geometry signal (TF++ path/speed decouple) that does NOT
        # see v0, so it can't shortcut. Both None (absent from state_dict) when
        # off, keeping an off-flag model byte-identical to pre-patch REF-B.
        self.yaw_head = (nn.Sequential(
            nn.Linear(self.state_dim, 128), nn.GELU(), nn.Linear(128, 1))
            if cfg.aux_yaw else None)
        self.path_heads = (nn.ModuleDict(
            {str(d): nn.Linear(cfg.tactical.d_model, 2) for d in cfg.path_dists})
            if cfg.path_dists else None)

    # --- WorldModel-compatible encode surface -------------------------------
    def encode(self, frames: Tensor) -> Tensor:
        """frames [B, C, H, W] -> compact state [B, S]."""
        return self.readout(self.encoder(frames))

    def encode_window(self, frames: Tensor) -> Tensor:
        """frames [B, W, C, H, W'] -> states [B, W, S]."""
        b, w = frames.shape[:2]
        flat = frames.reshape(b * w, *frames.shape[2:])
        return self.encode(flat).reshape(b, w, -1)

    # ------------------------------------------------------------------------
    def forward(self, frames: Tensor, nav_cmd: Tensor | None = None,
                v0: Tensor | None = None, yr0: Tensor | None = None) -> dict:
        """frames [B, W, C, H, W'], nav_cmd [B] long (None -> `follow`),
        v0 [B] optional current ego speed (pose_last[:,3]); consumed only when
        the speed-input flag is on (self.speed_emb is not None).

        Returns dict: states [B, W, S], maneuver_logits [B, M],
        waypoints {k: [B, 2]}, intent [B, d_intent], action_seq [B, K, A],
        route_logits [B, n_route] (strategic aux), ctx [B, d_ctx],
        conf_pred [B] (detached input path — fallback (a)), and — when the
        aux-accel head is enabled — accel_pred [B] (longitudinal-accel aux).
        """
        states = self.encode_window(frames)
        if nav_cmd is None:                       # unlabeled -> follow (idx 0)
            nav_cmd = torch.zeros(states.shape[0], dtype=torch.long,
                                  device=states.device)
        cmd = self.nav_emb(nav_cmd)
        # Proprioceptive ego conditioning, ADDED to the nav embedding (leakage-
        # safe: t=0 measurements). ego_dropout zeros it per-sample in training so
        # the planner can't lean on kinematic extrapolation (must read the
        # scene); eval always passes the true value.
        if self.ego_emb is not None and v0 is not None:        # B2: [v0, yr0]
            yr = yr0 if yr0 is not None else torch.zeros_like(v0)
            ego = torch.stack([v0, yr], dim=-1).to(cmd.dtype)  # [B, 2]
            if self.training and self.cfg.ego_dropout > 0.0:
                keep = (torch.rand(ego.shape[0], 1, device=ego.device)
                        >= self.cfg.ego_dropout).to(ego.dtype)  # JOINT drop
                ego = ego * keep
            cmd = cmd + self.ego_norm(self.ego_emb(ego))       # H26 scale-match
        elif self.speed_emb is not None and v0 is not None:    # pre-v2 speed
            v0_in = v0.to(cmd.dtype).reshape(-1, 1)
            if self.training and self.cfg.ego_dropout > 0.0:
                keep = (torch.rand(v0_in.shape[0], 1, device=v0_in.device)
                        >= self.cfg.ego_dropout).to(v0_in.dtype)
                v0_in = v0_in * keep
            cmd = cmd + self.speed_emb(v0_in)
        ctx, route_logits = self.strategic(states, cmd)
        maneuver_logits, geom, intent, tac_h, tac_extra = \
            self.tactical(states, ctx)
        action_seq = self.operative(states, intent)
        conf_in = torch.cat([states[:, -1].detach(), intent.detach()], dim=-1)
        conf_pred = self.confidence(conf_in)
        out = {"states": states, "maneuver_logits": maneuver_logits,
               "intent": intent, "action_seq": action_seq,
               "route_logits": route_logits, "ctx": ctx,
               "conf_pred": conf_pred}
        # Anchored (B1): geom is the multimodal selected trajectory in its
        # space; time-anchored (FINAL) -> it IS the time waypoints. Non-anchored
        # -> unimodal time wp_heads. tac_extra (anchor_logits/anchor_traj/
        # sel_idx/dbg) is {} when not anchored.
        if self.tactical.anchored and self.tactical.anchor_space == "distance":
            out["mm_path"] = geom
        else:
            out["waypoints"] = geom
        out.update(tac_extra)
        if self.accel_head is not None:
            # Operative-level grounding aux (NOT detached): longitudinal accel
            # predicted from the last window state.
            out["accel_pred"] = self.accel_head(states[:, -1]).squeeze(-1)
        if self.yaw_head is not None:            # refbpatch: aux yaw-rate
            out["yaw_pred"] = self.yaw_head(states[:, -1]).squeeze(-1)
        if self.path_heads is not None:          # refbpatch: fixed-distance path
            out["path_waypoints"] = {
                d: self.path_heads[str(d)](tac_h) for d in self.cfg.path_dists}
        return out


def param_breakdown(model: RefBModel) -> dict[str, int]:
    """Per-layer trainable-parameter table (report + metrics.json row).

    inv_dyn is booked under `operative` (it is the operative-level grounding
    aux, same as the main stack); readout under `encoder`; `strategic` =
    transformer + nav embedding (rev2); the OOD monitor is buffers-only and
    contributes 0 to `fallback`. Full-config measurement (vs main 262.8 M):
    encoder ~179.3 M, operative ~51.7 M, tactical ~21.7 M, strategic ~8.4 M,
    fallback ~1.4 M, total ~262.5 M (-0.1 %).
    """
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    # Gated ego-dynamics modules (0 when their flags are off): speed_emb is
    # booked under `strategic` (added alongside nav_emb), accel_head under
    # `operative` (an operative-level grounding aux, like inv_dyn).
    speed = cnt(model.speed_emb) if model.speed_emb is not None else 0
    # B2: ego_emb (+ its LayerNorm) booked under strategic, like speed_emb.
    ego = ((cnt(model.ego_emb) + cnt(model.ego_norm))
           if getattr(model, "ego_emb", None) is not None else 0)
    accel = cnt(model.accel_head) if model.accel_head is not None else 0
    # refbpatch: yaw_head booked with operative auxes; path_heads with tactical.
    yaw = cnt(model.yaw_head) if model.yaw_head is not None else 0
    path = cnt(model.path_heads) if model.path_heads is not None else 0
    # B1: the anchored wp_decoder + maneuver_to_anchor + prior LayerNorm live
    # inside model.tactical, so cnt(model.tactical) already counts them.
    return {
        "encoder": cnt(model.encoder) + cnt(model.readout),
        "operative": cnt(model.operative) + cnt(model.inv_dyn) + accel + yaw,
        "tactical": cnt(model.tactical) + path,
        "strategic": cnt(model.strategic) + cnt(model.nav_emb) + speed + ego,
        "fallback": cnt(model.confidence) + cnt(model.ood),
        "total": cnt(model),
    }
