"""REF-C model: Anchored-Diffusion-C — a DiffusionDrive-style trajectory head.

REF-C was TCP-C (a two-branch GRU trajectory/control stack, arXiv 2206.08129).
This revision REPLACES the GRU trajectory + control branches with an ANCHORED
TRUNCATED-DIFFUSION trajectory decoder in the DiffusionDrive spirit (arXiv
2411.15139): a fixed vocabulary of trajectory ANCHORS whose queries cross-attend
the conv feature map, emitting a per-anchor confidence + a per-anchor offset, and
(optionally) refining the winning modes with a few truncated denoising steps. The
rest of the TCP-C stack is KEPT verbatim: the torchvision-free ResNet-34-style
encoder, the measurement encoder with per-sample ego-dropout, the LAW latent-
world-model aux, the strategic-ctx hierarchy graft, and the REF-C.1 target-speed
class head.

Why anchors + FPS (not k-means): comma2k19 is ~74 % straight, so k-means
collapses almost every centroid onto the straight mode and starves the turns.
Furthest-point sampling (FPS) deliberately SPREADS the vocabulary to cover the
rare sharp-curve / hard-brake trajectories — the modes that actually matter.

Decoder (``AnchoredDiffusionDecoder``), two inference modes off ONE weight set:
  - ``steps=0`` (classifier / 0-step floor): anchor queries cross-attend the
    8x8x512 map (d=256, 2-3 MHA layers, FiLM(condition)); emit per-anchor
    confidence + per-anchor [n_horizons, 2] offset; traj = selected anchor +
    offset. This is what the trainer optimises.
  - ``steps>0`` (truncated diffusion): the SAME offset head refines the anchor
    trajectories over a few timestep-embedded denoising passes around the
    anchors; ``steps=0`` reproduces the classifier byte-for-byte.

Graft seams (gated, zero-init / identity starts — byte-identical when off):
  - ``hierarchy`` (default True): a strategic-ctx GRU over the W window frames
    -> a d_ctx token added (zero-init) to the decoder CONDITION embedding.
  - ``graft_maneuver`` (default True, H19): the model's maneuver-head logits (or
    an external tactical brain's) reweight the anchor confidence PRIORS through a
    learned maneuver->anchor projection (LIVE from step 0 — the H19 coupling is
    the point; the zero-init discipline applies to the ctx / target-latent
    seams).
  - ``graft_target_latent`` (default False): a tactical GOAL latent [B, S] FiLMs
    (zero-init -> identity) the decoder condition. Off by default because it has
    no standalone supervision (it only activates when a real tactical brain
    feeds a ``target_latent``).
  - ``grounded_selector`` (default False, param-free): score decoded ego-frame
    endpoints by a progress/collision proxy and blend with the top-1 confidence.
  - ``graft_imagination`` (default False, H15): a belief field over the conv-map
    tokens — latent-advection prior (object permanence) + transformer refinement
    + per-cell epistemic log-variance gating a residual belief written back into
    the tokens the anchor decoder cross-attends. ON for REF-C-XL (the flagship-
    matched capacity control); the whole field sits in the trajectory-loss
    gradient path (no dead params). Absent from the state_dict when off.
  - LAW ``law_head`` is KEPT: gradients flow through the decoded trajectory.

Scale presets (SAME code + decoder algorithm, three sizes — the size cap was
lifted so the budget lands where the encoder has PROVEN value: Hydra-MDP went
86.6 -> 91.0 PDMS purely by swapping ResNet-34 -> V2-99, so the ENCODER is the
lever, and the deeper/wider trunk is data-appropriate for the full 2376-ep set):
  - ``refc_config`` -> REF-C-base ~110 M (primary): a widened/deepened ResNet
    trunk (~90 M encoder, 8x8xF map preserved) + a d=384 / 4-layer / 128-anchor
    decoder. The data-appropriate reference.
  - ``refc_xl_config`` -> REF-C-XL ~260 M: a much wider/deeper ResNet trunk
    (~180 M encoder, 8x8xF map preserved) + a d=512 / 6-layer / 256-anchor
    decoder + the gated H15 imagination field (~22 M). Same-capacity control vs
    the 261 M flagship (removes the "REF-C is worse because smaller" confound).
  - ``refc_smoke_config`` -> tiny CPU config (CI / tests / dry runs).

REF-C.1 (gated ``refc1``, default False): the trajectory targets become fixed-
DISTANCE path checkpoints at (2, 5, 10, 20) m (refb_labels.path_targets) and a
target-speed classification head is added (CE + expected-value decode).

Gated-flag discipline (REF-B convention): with a flag off the corresponding
module is NOT constructed — the model is byte-identical to one that never had the
feature (state_dict keys pinned by tests/test_refc.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Strategic vocabulary — order pinned against scripts/refb_labels.py indices
# by tests/test_refc.py (same 4-wide interface as tanitad.refs.refb).
NAV_COMMANDS = ("follow", "left", "right", "straight")

# Maneuver vocabulary width — order matches refb_labels (lane_keep, turn_left,
# turn_right, accelerate, brake_stop). The model emits [B, N_MANEUVERS] logits;
# the trainer supplies the kinematic pseudo-label target (refc.py must not import
# from scripts/, so the count is pinned here and cross-checked by the tests).
N_MANEUVERS = 5
N_ROUTE = 3                        # route-heading aux (left / straight / right)


# ============================================================================
# Anchor vocabulary — FPS over ego-frame future trajectories
# ============================================================================

def synth_anchor_pool(horizons: tuple[int, ...], pool_size: int = 4096,
                      seed: int = 0, dt: float = 0.1,
                      device: str = "cpu") -> Tensor:
    """Synthesize a pool of ego-frame trajectories via random unicycle rollouts.

    Each rollout samples (v0, yaw_rate, accel) uniformly, integrates a unicycle
    from the ego origin (heading +x), and reads the position at every horizon
    step. Because the ego starts at the origin heading +x, the world positions
    ARE the ego-frame waypoints. Returns [pool_size, len(horizons), 2].

    Used for the model's DEFAULT anchor set (so REF-C builds without a data file)
    and by build_refc_anchors.py for the CPU-smoke / no-data path. Explicit
    ``device`` keeps this real even under a ``torch.device("meta")`` build.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    m = pool_size
    v = (torch.rand(m, generator=g, device=device) * 30.0)          # 0..30 m/s
    yaw_rate = (torch.rand(m, generator=g, device=device) - 0.5) * 0.7   # +-0.35
    accel = (torch.rand(m, generator=g, device=device) - 0.5) * 6.0     # +-3
    x = torch.zeros(m, device=device)
    y = torch.zeros(m, device=device)
    yaw = torch.zeros(m, device=device)
    max_h = max(horizons)
    pos = torch.zeros(m, max_h + 1, 2, device=device)
    for t in range(1, max_h + 1):
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + yaw_rate * dt
        v = (v + accel * dt).clamp_min(0.0)
        pos[:, t, 0] = x
        pos[:, t, 1] = y
    return torch.stack([pos[:, h] for h in horizons], dim=1)         # [m, S, 2]


def furthest_point_sample(pool: Tensor, n: int, seed: int = 0) -> Tensor:
    """Furthest-point sample ``n`` anchors from ``pool`` [M, S, 2] -> [n, S, 2].

    Greedy FPS in flattened-L2 space: seed one point, then repeatedly add the
    pool point maximally far from the current set (min-distance criterion). This
    SPREADS the vocabulary over the trajectory manifold (covering the rare
    curves) rather than concentrating on the dense straight majority the way
    k-means would. Deterministic given ``seed``.
    """
    m = pool.shape[0]
    if n > m:
        raise ValueError(f"cannot FPS {n} anchors from a pool of {m}")
    flat = pool.reshape(m, -1)
    g = torch.Generator(device=flat.device).manual_seed(seed)
    first = int(torch.randint(m, (1,), generator=g, device=flat.device))
    chosen = [first]
    dist = ((flat - flat[first]) ** 2).sum(dim=-1)                   # [M]
    for _ in range(n - 1):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, ((flat - flat[nxt]) ** 2).sum(dim=-1))
    return pool[torch.tensor(chosen, device=pool.device)]


def default_anchors(horizons: tuple[int, ...], n_anchors: int,
                    pool_size: int = 4096, seed: int = 0,
                    device: str = "cpu") -> Tensor:
    """The model's built-in anchor vocabulary: FPS over a synthetic pool.

    Deterministic (fixed seed) so two independently-built models share anchors
    byte-for-byte. Overridden at train time by build_refc_anchors.py output via
    :meth:`AnchoredDiffusionDecoder.load_anchors`.
    """
    pool = synth_anchor_pool(horizons, pool_size, seed, device=device)
    return furthest_point_sample(pool, n_anchors, seed=seed).contiguous()


# ============================================================================
# Configs
# ============================================================================

@dataclass
class CNNEncoderConfig:
    """ResNet-34-style trunk (torchvision-free): stem /4, four stages /2 each
    -> stride 32; feat_dim = base_width * 8; grid = image_size // 32.

    The size cap was lifted to spend the budget on the encoder (the Hydra-MDP
    ResNet-34 -> V2-99 lever). REF-C-base WIDENS base_width to 88 (V2-99-class
    trunk, ~90 M) and REF-C-XL to 168 (~180 M); both KEEP the 8x8xF conv map the
    anchor decoder cross-attends (grid = 8 at 256 px; F = base_width * 8, and the
    decoder's feat_proj adapts to any F — the contract is [B, F, 8, 8], not a
    fixed 512). ``blocks`` is the per-stage BasicBlock depth (widened trunk keeps
    the deep-34 (3, 6, 16, 6) shape; XL deepens stage-3/4)."""
    in_channels: int = 9          # D-015 3-frame RGB stack (latest = [-3:])
    image_size: int = 256
    base_width: int = 88          # V2-99-class width (~90 M trunk); XL -> 168
    blocks: tuple[int, ...] = (3, 6, 16, 6)    # deep-34 (8x8xF map preserved)

    @property
    def feat_dim(self) -> int:
        return self.base_width * 8

    @property
    def grid(self) -> int:
        return self.image_size // 32


@dataclass
class MeasurementConfig:
    hidden: int = 128
    d_out: int = 128


@dataclass
class TrajectoryConfig:
    # 2 s @ 10 Hz in 0.5 s strides (the REF-B tactical horizons) — time-indexed
    # waypoints; under refc1 the SAME step slots are read as fixed-distance path
    # checkpoints (RefCConfig.path_dists). The anchor vocabulary lives in this
    # [len(horizons), 2] ego-frame trajectory space.
    horizons: tuple[int, ...] = (5, 10, 15, 20)


@dataclass
class AnchorConfig:
    n_anchors: int = 128          # FPS vocabulary size (base 128; XL 256; 20 smoke)
    pool_size: int = 4096         # synthetic pool the default anchors FPS over
    seed: int = 0


@dataclass
class DecoderConfig:
    d: int = 384                  # decoder width (anchor queries + cross-attn)
    n_heads: int = 8
    layers: int = 4               # cross-attention layers (base 4; XL 6)
    ff_mult: int = 4
    aux_hidden: int = 384         # maneuver-head hidden
    diffusion_steps: int = 2      # truncated-denoise steps (0 == classifier)
    noise_std: float = 0.1        # train-time truncated-diffusion noise (metres)


@dataclass
class LawConfig:
    hidden: int = 2048            # latent-world-model aux MLP width


@dataclass
class StrategicCtxConfig:
    hidden: int = 512             # ctx GRU width
    d_ctx: int = 64               # strategic token -> decoder condition seam


@dataclass
class ImaginationConfig:
    """H15 belief field over the conv-map tokens (graft_imagination). Sized on
    REF-C-XL to ~22 M as the flagship-matched capacity control."""
    d: int = 512                  # belief-field width (in/out project feat_dim<->d)
    depth: int = 6                # self-attention refinement blocks
    n_heads: int = 8
    ff_mult: int = 4              # refinement-block MLP ratio
    head_hidden: int = 1024       # flow / log-variance head hidden width


@dataclass
class RefCConfig:
    encoder: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    window: int = 8               # shared state window (main stack: 8)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    law: LawConfig = field(default_factory=LawConfig)
    strategic: StrategicCtxConfig = field(default_factory=StrategicCtxConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    speed_hidden: int = 256       # refc1 target-speed class head width
    ego_dropout: float = 0.5      # per-sample Bernoulli zero of v0 (training)
    hierarchy: bool = True        # strategic ctx -> decoder condition (graft)
    graft_maneuver: bool = True   # maneuver logits reweight anchor priors (H19)
    graft_target_latent: bool = False   # FiLM the condition on a goal latent
    grounded_selector: bool = False     # progress/collision proxy vs top-1 conf
    graft_imagination: bool = False     # H15 belief field over conv-map tokens
    tactical_latent_dim: int = 512      # external target_latent width (S)
    refc1: bool = False           # fixed-distance path + target-speed class
    path_dists: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)   # metres (refc1)
    speed_bins: int = 4           # refc1 target-speed classes over [0, max]
    speed_max: float = 30.0       # m/s


def refc_config() -> RefCConfig:
    """REF-C-base ~110 M (the primary, data-appropriate for the full 2376-ep
    set): a V2-99-class ResNet trunk (base_width 88 -> ~90 M encoder, 8x8xF map)
    + the d=384 / 4-layer / 128-anchor anchored-diffusion decoder + LAW aux +
    strategic-ctx hierarchy. The budget lands in the ENCODER on purpose — the
    Hydra-MDP ResNet-34 -> V2-99 lever (86.6 -> 91.0 PDMS). Imagination graft OFF.
    Measured by param_breakdown at instantiation; tests pin the 90-130 M band."""
    return RefCConfig()


def refc_small_config() -> RefCConfig:
    """REF-C-small ~28 M — the low end of the size-vs-data scaling study
    (small 28 M / base 104 M / XL 252 M on the IDENTICAL 2376-ep set: a ~9x
    capacity range read via the 5k/15k/20k/30k milestone gates, to see where
    bigger helps vs overfits on our data). A plain ResNet-34 trunk (base_width
    64, blocks (3,4,6,3) -> ~21 M encoder) + a lean d=256 / 3-layer / 64-anchor
    decoder; same anchored-diffusion algorithm + LAW + strategic-ctx as base/XL,
    imagination OFF. Tests pin the 22-35 M band."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=9, image_size=256, base_width=64,
                                   blocks=(3, 4, 6, 3))
    cfg.decoder = DecoderConfig(d=256, n_heads=4, layers=3, ff_mult=4,
                                aux_hidden=256, diffusion_steps=2, noise_std=0.1)
    cfg.anchors = AnchorConfig(n_anchors=64, pool_size=2048)
    cfg.strategic = StrategicCtxConfig(hidden=512, d_ctx=64)
    return cfg


def refc_xl_config() -> RefCConfig:
    """REF-C-XL ~260 M (flagship-matched capacity control — same-capacity vs the
    261 M flagship, removing the "REF-C is worse because smaller" confound). SAME
    refc.py code + decoder algorithm as base; only the widths/depths grow:
      encoder   base_width 124, blocks (3, 8, 20, 6) -> ~200 M ResNet-L trunk
                (wider AND deeper than base; 8x8xF map preserved, F = 992, the
                decoder feat_proj adapts).
      decoder   d=512, 6 cross-attn layers, 256 FPS anchors.
      grafts    H15 imagination field ON (~22 M belief field over the conv-map
                tokens) — the extra budget beyond the wider trunk/decoder.
    Tests pin the 230-280 M band with a full encoder/decoder/imagination split."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=9, image_size=256, base_width=124,
                                   blocks=(3, 8, 20, 6))
    cfg.decoder = DecoderConfig(d=512, n_heads=8, layers=6, ff_mult=4,
                                aux_hidden=512, diffusion_steps=2, noise_std=0.1)
    cfg.anchors = AnchorConfig(n_anchors=256, pool_size=4096)
    cfg.strategic = StrategicCtxConfig(hidden=768, d_ctx=96)
    cfg.imagination = ImaginationConfig(d=512, depth=6, n_heads=8, ff_mult=4,
                                        head_hidden=1024)
    cfg.graft_imagination = True
    return cfg


def refc_smoke_config() -> RefCConfig:
    """Tiny CPU config (CI smoke / tests / dry runs) — same structure, same
    horizons/path_dists, shrunk widths. Episodes: 1-channel 64 px (grid 2, 4
    cross-attention positions); 20-anchor vocabulary."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=1, image_size=64, base_width=8,
                                   blocks=(1, 1, 1, 1))
    cfg.window = 4
    cfg.measurement = MeasurementConfig(hidden=32, d_out=16)
    cfg.anchors = AnchorConfig(n_anchors=20, pool_size=256)
    cfg.decoder = DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                                aux_hidden=32, diffusion_steps=2, noise_std=0.1)
    cfg.law = LawConfig(hidden=32)
    cfg.strategic = StrategicCtxConfig(hidden=16, d_ctx=8)
    cfg.speed_hidden = 32
    cfg.tactical_latent_dim = 32
    return cfg


# ============================================================================
# Encoder (KEEP verbatim from TCP-C)
# ============================================================================

class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + BN, identity/1x1 shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.down: nn.Module | None = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x: Tensor) -> Tensor:
        idn = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + idn)


class ResNetEncoder(nn.Module):
    """Torchvision-free ResNet-34-style image encoder.

    Stem (7x7 /2 + maxpool /2) then four BasicBlock stages at widths
    (w, 2w, 4w, 8w), strides (1, 2, 2, 2) -> total stride 32. forward returns
    (fmap [B, 8w, g, g], pooled [B, 8w]) with g = image_size // 32 — for the
    full config the TCP-shaped [B, 512, 8, 8] + 512-d pooled vector.
    """

    def __init__(self, cfg: CNNEncoderConfig):
        super().__init__()
        if cfg.image_size % 32 != 0:
            raise ValueError(f"image_size must be divisible by 32, "
                             f"got {cfg.image_size}")
        w = cfg.base_width
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        widths = (w, 2 * w, 4 * w, 8 * w)
        stages, c_in = [], w
        for s_i, (c_out, depth) in enumerate(zip(widths, cfg.blocks)):
            blocks = [BasicBlock(c_in, c_out, stride=1 if s_i == 0 else 2)]
            blocks += [BasicBlock(c_out, c_out) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.feat_dim = cfg.feat_dim
        self.grid = cfg.grid

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x, x.mean(dim=(2, 3))                # fmap [B,F,g,g], pooled


class StrategicCtx(nn.Module):
    """Hierarchy graft: tiny GRU over the W pooled window features -> d_ctx
    token (KEEP). Now feeds the decoder CONDITION rather than the measurement."""

    def __init__(self, feat_dim: int, hidden: int, d_ctx: int):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_ctx)

    def forward(self, pooled_seq: Tensor) -> Tensor:   # [B, W, F] -> [B, d_ctx]
        _, h = self.gru(pooled_seq)
        return self.proj(h[-1])


# ============================================================================
# Anchored truncated-diffusion decoder
# ============================================================================

class FiLM(nn.Module):
    """Feature-wise linear modulation. ``zero_init`` starts it as identity (for
    the graft seams that must not perturb the base); the core condition FiLM in
    the decoder layers is LIVE (default init) so the measurement/nav/v0 condition
    steers the decoder from step 0 rather than waiting for the FiLM to train."""

    def __init__(self, cond_dim: int, d: int, zero_init: bool = True):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)
        if zero_init:
            nn.init.zeros_(self.to_scale_shift.weight)
            nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class CrossAttnLayer(nn.Module):
    """Anchor-query cross-attention block: cross-attend the conv map, then a
    FiLM(condition)-modulated MLP (pre-norm, residual)."""

    def __init__(self, d: int, n_heads: int, cond_dim: int, ff_mult: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(d)
        self.film = FiLM(cond_dim, d, zero_init=False)   # live core conditioning
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))

    def forward(self, q: Tensor, kv: Tensor, cond: Tensor) -> Tensor:
        h = self.norm_q(q)
        q = q + self.cross(h, kv, kv, need_weights=False)[0]
        q = q + self.mlp(self.film(self.norm_f(q), cond.unsqueeze(1)))
        return q


class AnchoredDiffusionDecoder(nn.Module):
    """DiffusionDrive-style anchored trajectory decoder (two modes, one weight
    set). See the module docstring for the classifier / truncated-diffusion
    split and the gated graft seams.

    forward -> {anchor_logits [B, N], anchor_traj [B, N, S, 2], offset (base)
    [B, N, S, 2], traj [B, S, 2] (selected), sel_idx [B]}.
    """

    def __init__(self, feat_dim: int, n_steps: int, d_meas: int, d_ctx: int,
                 tac_latent_dim: int, anchors: Tensor, cfg: DecoderConfig,
                 hierarchy: bool, graft_maneuver: bool,
                 graft_target_latent: bool, grounded_selector: bool,
                 n_maneuvers: int = N_MANEUVERS):
        super().__init__()
        self.cfg = cfg
        self.n_steps = n_steps
        self.grounded = grounded_selector
        # Anchor vocabulary — a persistent buffer (travels with the checkpoint).
        self.register_buffer("anchors", anchors)              # [N, S, 2]
        d = cfg.d
        self.feat_proj = nn.Linear(feat_dim, d)               # conv map -> KV
        self.traj_proj = nn.Linear(n_steps * 2, d)            # traj estimate -> Q
        self.cond_proj = nn.Linear(d_meas, d)                 # measurement -> cond
        self.time_embed = nn.Embedding(cfg.diffusion_steps + 1, d)  # 0..steps
        self.layers = nn.ModuleList(
            CrossAttnLayer(d, cfg.n_heads, d, cfg.ff_mult)
            for _ in range(cfg.layers))
        self.conf_head = nn.Linear(d, 1)                      # per-anchor conf
        self.offset_head = nn.Linear(d, n_steps * 2)          # per-anchor offset
        # Graft: strategic ctx -> condition (zero-init identity start).
        self.ctx_to_cond: nn.Linear | None = None
        if hierarchy:
            self.ctx_to_cond = nn.Linear(d_ctx, d)
            nn.init.zeros_(self.ctx_to_cond.weight)
            nn.init.zeros_(self.ctx_to_cond.bias)
        # Graft (H19): maneuver logits reweight anchor priors. LIVE from step 0
        # (default Linear init) — the coupling is the point of the seam.
        self.maneuver_to_anchor: nn.Linear | None = None
        if graft_maneuver:
            self.maneuver_to_anchor = nn.Linear(n_maneuvers, anchors.shape[0],
                                                bias=False)
        # Graft: FiLM the condition on a tactical goal latent (zero-init).
        self.tgt_proj: nn.Linear | None = None
        self.tgt_film: FiLM | None = None
        if graft_target_latent:
            self.tgt_proj = nn.Linear(tac_latent_dim, d)
            self.tgt_film = FiLM(d, d)

    def load_anchors(self, anchors: Tensor) -> None:
        """Install an externally-built anchor vocabulary (build_refc_anchors.py).
        Shape must match [N, n_steps, 2] of the constructed decoder."""
        if tuple(anchors.shape) != tuple(self.anchors.shape):
            raise ValueError(f"anchor shape {tuple(anchors.shape)} != decoder "
                             f"{tuple(self.anchors.shape)}")
        self.anchors.copy_(anchors.to(self.anchors.dtype))

    def _decode(self, kv: Tensor, cond: Tensor, x_est: Tensor,
                t_idx: int) -> tuple[Tensor, Tensor]:
        """One decoder pass: current trajectory estimate + timestep -> queries;
        cross-attend the map; emit (conf [B, N], offset [B, N, S, 2])."""
        b, n = x_est.shape[:2]
        q = self.traj_proj(x_est.reshape(b, n, -1))           # [B, N, d]
        q = q + self.time_embed.weight[t_idx][None, None]     # timestep bias
        for layer in self.layers:
            q = layer(q, kv, cond)
        conf = self.conf_head(q).squeeze(-1)                  # [B, N]
        offset = self.offset_head(q).reshape(b, n, self.n_steps, 2)
        return conf, offset

    @staticmethod
    def _grounded_score(x: Tensor) -> Tensor:
        """Param-free progress/collision proxy over decoded endpoints [B,N,S,2]:
        reward forward reach, penalise lateral excursion (no obstacle map yet)."""
        end = x[:, :, -1]                                     # [B, N, 2]
        return 0.1 * end[..., 0] - 0.1 * end[..., 1].abs()

    def forward(self, fmap: Tensor, m: Tensor, ctx: Tensor | None = None,
                maneuver_logits: Tensor | None = None,
                target_latent: Tensor | None = None,
                steps: int = 0) -> dict:
        b = fmap.shape[0]
        kv = self.feat_proj(fmap.flatten(2).transpose(1, 2))  # [B, P, d]
        cond = self.cond_proj(m)                              # [B, d]
        if self.ctx_to_cond is not None and ctx is not None:
            cond = cond + self.ctx_to_cond(ctx)
        if self.tgt_film is not None and target_latent is not None:
            cond = self.tgt_film(cond, self.tgt_proj(target_latent))

        anchors = self.anchors.to(fmap.dtype)                 # [N, S, 2]
        n = anchors.shape[0]
        x0 = anchors[None].expand(b, n, self.n_steps, 2)
        conf, offset = self._decode(kv, cond, x0, 0)          # classifier pass
        x = anchors[None] + offset                            # [B, N, S, 2]

        # H19: maneuver prior reweights the anchor confidences (log-space).
        if self.maneuver_to_anchor is not None and maneuver_logits is not None:
            conf = conf + self.maneuver_to_anchor(
                torch.log_softmax(maneuver_logits, dim=-1))

        # Truncated diffusion: refine the anchor trajectories a few steps. Noise
        # only in training (deterministic at eval so decoding is reproducible).
        for i in range(steps):
            t_idx = min(i + 1, self.cfg.diffusion_steps)
            noise = (torch.randn_like(x) * self.cfg.noise_std
                     if self.training else torch.zeros_like(x))
            x_in = x + noise
            _, off = self._decode(kv, cond, x_in, t_idx)
            x = x_in + off

        score = conf + self._grounded_score(x) if self.grounded else conf
        idx = score.argmax(dim=1)                             # [B] (detached)
        traj = x[torch.arange(b, device=x.device), idx]       # [B, S, 2]
        return {"anchor_logits": conf, "anchor_traj": x, "offset": offset,
                "traj": traj, "sel_idx": idx}


# ============================================================================
# H15 imagination field (gated graft — belief over the conv-map tokens)
# ============================================================================

def advect(tokens: Tensor, flow: Tensor, grid_hw: int) -> Tensor:
    """Semi-Lagrangian warp of a token grid by a per-cell flow (in cell units):
    value'(x) = value(x - v(x)). The latent-advection prior (object permanence —
    a latent behind an occluder keeps moving while unobserved). Self-contained
    port of tanitad.models.imagination.advect (refc.py stays torch-only)."""
    b, n, d = tokens.shape
    x = tokens.transpose(1, 2).reshape(b, d, grid_hw, grid_hw)
    ys, xs = torch.meshgrid(
        torch.arange(grid_hw, device=tokens.device, dtype=tokens.dtype),
        torch.arange(grid_hw, device=tokens.device, dtype=tokens.dtype),
        indexing="ij")
    base = torch.stack([xs, ys], dim=-1)                       # [g, g, 2]
    f = flow.reshape(b, grid_hw, grid_hw, 2)
    pos = base.unsqueeze(0) - f                                # sample source
    pos = 2.0 * pos / max(grid_hw - 1, 1) - 1.0               # -> [-1, 1]
    warped = F.grid_sample(x, pos, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return warped.flatten(2).transpose(1, 2)


class ImagBlock(nn.Module):
    """Pre-norm self-attention + FiLM-free MLP (belief-field refinement over the
    token grid — same shape as the encoder's ViT Block, kept local)."""

    def __init__(self, d: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ImaginationField(nn.Module):
    """H15 gated belief field over the [B, F, g, g] conv-map tokens.

    Latent-advection prior (object permanence; zero-init flow -> identity warp at
    start) -> transformer refinement -> a per-cell epistemic log-variance. The
    confidence sigmoid(-logvar) gates a residual belief written back into the
    tokens the anchor decoder cross-attends: refined = tokens + conf * out(z).
    Every parameter (in_proj, flow, blocks, norm, logvar, out_proj) sits in the
    trajectory-loss gradient path — no dead params. Gated: absent when off."""

    def __init__(self, feat_dim: int, grid_hw: int, cfg: ImaginationConfig):
        super().__init__()
        self.grid_hw = grid_hw
        d = cfg.d
        self.in_proj = nn.Linear(feat_dim, d)                 # conv tokens -> d
        self.flow_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 2))
        nn.init.zeros_(self.flow_head[-1].weight)             # identity advection
        nn.init.zeros_(self.flow_head[-1].bias)               # (needs grid >= 4:
        # at grid 2 every zero-flow sample lands on the normalized boundary where
        # grid_sample's position gradient is clamped to 0 — real configs use 8)
        self.blocks = nn.ModuleList(
            ImagBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.logvar_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 1))
        self.out_proj = nn.Linear(d, feat_dim)                # belief -> feat_dim

    def forward(self, fmap: Tensor) -> tuple[Tensor, Tensor]:
        """fmap [B, F, g, g] -> (refined [B, F, g, g], logvar [B, g*g])."""
        b, fdim, g, _ = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)              # [B, N, F]
        z = self.in_proj(tokens)                              # [B, N, d]
        z = advect(z, self.flow_head(z), self.grid_hw)        # object permanence
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        logvar = self.logvar_head(z).squeeze(-1).clamp(-10.0, 10.0)   # [B, N]
        conf = torch.sigmoid(-logvar).unsqueeze(-1)           # low var -> trust
        refined = tokens + conf * self.out_proj(z)            # residual belief
        return refined.transpose(1, 2).reshape(b, fdim, g, g), logvar


# ============================================================================
# Model
# ============================================================================

class RefCModel(nn.Module):
    """Anchored-Diffusion-C: ResNet encoder + anchored-diffusion trajectory
    decoder + LAW aux + maneuver/route aux heads + hierarchical conditioning."""

    def __init__(self, cfg: RefCConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ResNetEncoder(cfg.encoder)
        feat = self.encoder.feat_dim
        n_steps = len(cfg.trajectory.horizons)
        if cfg.refc1 and len(cfg.path_dists) != n_steps:
            raise ValueError(f"refc1 needs len(path_dists) == "
                             f"len(horizons): {len(cfg.path_dists)} != "
                             f"{n_steps}")
        # Hierarchy graft (gated): absent when off — byte-identical model.
        if cfg.hierarchy:
            self.strategic = StrategicCtx(feat, cfg.strategic.hidden,
                                          cfg.strategic.d_ctx)
        # Measurement encoder (KEEP): [v0, nav one-hot] with ego-dropout. The
        # strategic ctx now conditions the DECODER, not the measurement.
        d_meas_in = 1 + len(NAV_COMMANDS)
        self.measurement = nn.Sequential(
            nn.Linear(d_meas_in, cfg.measurement.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.measurement.hidden, cfg.measurement.d_out),
            nn.ReLU(inplace=True))
        # Anchored-diffusion trajectory decoder (replaces TCP traj+control).
        anchors = default_anchors(cfg.trajectory.horizons, cfg.anchors.n_anchors,
                                  cfg.anchors.pool_size, cfg.anchors.seed,
                                  device="cpu")
        self.decoder = AnchoredDiffusionDecoder(
            feat, n_steps, cfg.measurement.d_out, cfg.strategic.d_ctx,
            cfg.tactical_latent_dim, anchors, cfg.decoder,
            hierarchy=cfg.hierarchy, graft_maneuver=cfg.graft_maneuver,
            graft_target_latent=cfg.graft_target_latent,
            grounded_selector=cfg.grounded_selector)
        # H15 imagination graft (gated): belief field over the conv-map tokens,
        # refining the [B, F, 8, 8] map the decoder cross-attends. Absent when off.
        if cfg.graft_imagination:
            self.imagination = ImaginationField(feat, self.encoder.grid,
                                                cfg.imagination)
        # Aux heads (always present): the maneuver head feeds BOTH the maneuver
        # CE and the H19 anchor reweight; the route head is the strategic aux.
        self.maneuver_head = nn.Sequential(
            nn.Linear(feat, cfg.decoder.aux_hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.decoder.aux_hidden, N_MANEUVERS))
        self.route_head = nn.Linear(feat, N_ROUTE)
        # LAW aux (KEEP): decoded trajectory enters NON-detached — gradients flow.
        self.law_head = nn.Sequential(
            nn.Linear(feat + 2 * n_steps, cfg.law.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.law.hidden, feat))
        # REF-C.1 (gated): target-speed classification head (KEEP).
        if cfg.refc1:
            self.speed_cls = nn.Sequential(
                nn.Linear(feat + cfg.measurement.d_out, cfg.speed_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.speed_hidden, cfg.speed_bins))

    # --- encode surface -----------------------------------------------------
    def encode_pooled(self, frames: Tensor) -> Tensor:
        """frames [B, C, H, W] -> pooled latent [B, F] (LAW target path)."""
        return self.encoder(frames)[1]

    def _speed_bin_centers(self, device, dtype) -> Tensor:
        half = self.cfg.speed_max / (2 * self.cfg.speed_bins)
        return torch.linspace(half, self.cfg.speed_max - half,
                              self.cfg.speed_bins, device=device, dtype=dtype)

    # ------------------------------------------------------------------------
    def forward(self, frames: Tensor, nav_cmd: Tensor | None = None,
                v0: Tensor | None = None,
                maneuver_logits: Tensor | None = None,
                target_latent: Tensor | None = None, steps: int = 0) -> dict:
        """frames [B, W, C, H, W'], nav_cmd [B] long (None -> `follow`), v0 [B]
        current ego speed (None -> zeros; scaled /10 inside). ``maneuver_logits``
        / ``target_latent`` are OPTIONAL external tactical-brain seams (else the
        model's own maneuver head drives the H19 reweight and the target-latent
        FiLM stays inactive). ``steps`` selects the decoder mode: 0 = classifier
        (default), >0 = truncated diffusion.

        Returns dict: pooled [B, F], traj / wp_seq [B, n_steps, 2], waypoints
        {key: [B, 2]}, anchor_logits [B, N], anchor_traj [B, N, n_steps, 2],
        offset [B, N, n_steps, 2], sel_idx [B], maneuver_logits [B, 5],
        route_logits [B, 3], law_pred [B, F], measurement [B, d_m] (+ hierarchy:
        ctx [B, d_ctx]) (+ graft_imagination: imag_logvar [B, g*g]) (+ refc1:
        speed_logits, target_speed).
        """
        b, w = frames.shape[:2]
        if self.cfg.hierarchy:
            fmap_all, pooled_all = self.encoder(
                frames.reshape(b * w, *frames.shape[2:]))
            pooled_seq = pooled_all.reshape(b, w, -1)
            pooled = pooled_seq[:, -1]
            fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
            ctx = self.strategic(pooled_seq)
        else:                                    # last frame only (same values)
            fmap, pooled = self.encoder(frames[:, -1])
            ctx = None

        # H15 belief field refines the conv-map tokens before the decoder (gated).
        imag_logvar = None
        if self.cfg.graft_imagination:
            fmap, imag_logvar = self.imagination(fmap)

        if nav_cmd is None:                      # unlabeled -> follow (idx 0)
            nav_cmd = torch.zeros(b, dtype=torch.long, device=frames.device)
        nav = F.one_hot(nav_cmd, len(NAV_COMMANDS)).to(pooled.dtype)
        v = torch.zeros(b, 1, dtype=pooled.dtype, device=pooled.device) \
            if v0 is None else (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
        if self.training and self.cfg.ego_dropout > 0:
            keep = (torch.rand(b, 1, device=v.device)
                    >= self.cfg.ego_dropout).to(v.dtype)
            v = v * keep                         # per-sample Bernoulli zero
        m = self.measurement(torch.cat([v, nav], dim=-1))

        # Aux heads (image branch): maneuver logits also drive the H19 reweight.
        man_logits = self.maneuver_head(pooled)
        route_logits = self.route_head(pooled)
        reweight = maneuver_logits if maneuver_logits is not None else man_logits

        dec = self.decoder(fmap, m, ctx=ctx, maneuver_logits=reweight,
                           target_latent=target_latent, steps=steps)
        traj = dec["traj"]
        law_pred = self.law_head(torch.cat([pooled, traj.reshape(b, -1)],
                                           dim=-1))

        keys = self.cfg.path_dists if self.cfg.refc1 \
            else self.cfg.trajectory.horizons
        out = {"pooled": pooled, "traj": traj, "wp_seq": traj,
               "waypoints": {k: traj[:, i] for i, k in enumerate(keys)},
               "anchor_logits": dec["anchor_logits"],
               "anchor_traj": dec["anchor_traj"], "offset": dec["offset"],
               "sel_idx": dec["sel_idx"], "maneuver_logits": man_logits,
               "route_logits": route_logits, "law_pred": law_pred,
               "measurement": m}
        if ctx is not None:
            out["ctx"] = ctx
        if imag_logvar is not None:
            out["imag_logvar"] = imag_logvar         # H15 per-cell uncertainty
        if self.cfg.refc1:
            logits = self.speed_cls(torch.cat([pooled, m], dim=-1))
            centers = self._speed_bin_centers(logits.device, logits.dtype)
            out["speed_logits"] = logits
            out["target_speed"] = F.softmax(logits, dim=-1) @ centers
        return out


def param_breakdown(model: RefCModel) -> dict[str, int]:
    """Per-module trainable-parameter table (report + config.json row).

    ``encoder`` (the proven lever — where the lifted budget goes) vs ``decoder``
    (the anchored-diffusion decoder INCLUDING its gated graft submodules ctx->
    cond, maneuver->anchor, target-latent FiLM) is the split that shows where the
    budget went; ``imagination`` is the H15 belief field (0 unless
    graft_imagination); ``aux`` is the maneuver + route heads; ``strategic`` is 0
    when hierarchy=False; ``speed`` is 0 unless refc1. REF-C-base lands ~110 M
    (tests pin 90-130 M), REF-C-XL ~260 M (230-280 M)."""
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    return {
        "encoder": cnt(model.encoder),
        "measurement": cnt(model.measurement),
        "strategic": cnt(model.strategic) if model.cfg.hierarchy else 0,
        "decoder": cnt(model.decoder),
        "imagination": cnt(model.imagination) if model.cfg.graft_imagination
        else 0,
        "aux": cnt(model.maneuver_head) + cnt(model.route_head),
        "law": cnt(model.law_head),
        "speed": cnt(model.speed_cls) if model.cfg.refc1 else 0,
        "total": cnt(model),
    }
