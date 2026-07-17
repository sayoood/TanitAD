"""REF-C model: TCP-C — the TCP driving architecture, BC-adapted to TanitAD.

TCP ("Trajectory-guided Control Prediction", arXiv 2206.08129) is the CARLA-
leaderboard-proven two-branch driving stack: a trajectory branch (GRU rolling
out ego-frame waypoints) and a control branch (GRU predicting future action
pairs) coupled by TRAJECTORY-GUIDED ATTENTION over the conv feature map. This
module adapts it to our epcache BC data contract and grafts in the program's
hierarchical conditioning (REFERENCE_ARCHITECTURES REF-C, ~30 M params —
deliberately NOT budget-matched to the 261 M main stack: TCP's own scale is
the point of the reference).

Adaptations vs the paper (each pinned by tests/test_refc.py):
  1. Encoder: torchvision-free ResNet-34-STYLE CNN (torchvision is not in the
     venv — checked 2026-07-18), conv1 takes the 9-channel 3-frame RGB stack;
     outputs the 8x8 conv map [B, 512, 8, 8] (256 px / 32) + pooled vector.
  2. Measurement encoder: MLP over [v0 = pose_last[:, 3]/10 (per-sample
     Bernoulli ego-dropout p=0.5, model.training-gated — the anti-kinematic-
     shortcut lever), nav one-hot(4)]. NO target point: our data has no route
     planner; route intent enters ONLY as the nav command (+ ctx, item 6).
  3. Trajectory branch: GRUCell rolled 4 steps at horizons (5, 10, 15, 20);
     each step outputs a 2D delta added to the running ego-frame position
     (TCP style). Hidden init from [pooled_feat, measurement].
  4. Control branch: GRUCell predicting K=4 future (steer, accel) pairs; at
     step k an MLP over [traj_hidden_k, ctrl_hidden_k] emits softmax weights
     over the 64 spatial positions; the attended feature feeds the step's
     action head and (with the action) the next GRU input.
  5. LAW latent-world-model aux (arXiv 2406.08481 bolt-on, REPLACES TCP's
     Roach expert distillation — we have no privileged expert): an MLP
     predicts the pooled latent 0.5 s ahead from [pooled latent, predicted
     waypoints]; gradients FLOW through the waypoints (that is the point —
     the action proposal must be predictive of the future latent). The
     no_grad target is computed by the trainer via encode_pooled.
  6. Hierarchy graft (gated ``hierarchy``, default True): a tiny strategic
     ctx GRU over the pooled features of the W window frames -> d_ctx=64
     token concatenated into the measurement vector (the strategic-ctx ->
     measurement seam; tactical intent = the nav command already in m).
  7. REF-C.1 (gated ``refc1``, default False): the SAME trajectory GRU emits
     fixed-DISTANCE path checkpoints at (2, 5, 10, 20) m instead of time-
     indexed waypoints (targets: refb_labels.path_targets arc-length
     resample) + a target-speed classification head (4 bins over [0, 30]
     m/s, CE + expected-value decode) — decoupling path geometry from
     kinematics.

Gated-flag discipline (REF-B convention): with ``hierarchy=False`` /
``refc1=False`` the corresponding modules are NOT constructed — the model is
byte-identical to one that never had the feature (state_dict keys pinned by
tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Strategic vocabulary — order pinned against scripts/refb_labels.py indices
# by tests/test_refc.py (same 4-wide interface as tanitad.refs.refb).
NAV_COMMANDS = ("follow", "left", "right", "straight")


@dataclass
class CNNEncoderConfig:
    """ResNet-34-style trunk (torchvision-free): stem /4, four stages /2 each
    -> stride 32; feat_dim = base_width * 8; grid = image_size // 32."""
    in_channels: int = 9          # D-015 3-frame RGB stack (latest = [-3:])
    image_size: int = 256
    base_width: int = 64
    blocks: tuple[int, ...] = (3, 4, 6, 3)     # ResNet-34 stage depths

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
    hidden: int = 512
    # 2 s @ 10 Hz in 0.5 s strides (the REF-B tactical horizons) — time-
    # indexed waypoints; under refc1 the SAME 4 rollout steps are read as
    # fixed-distance path checkpoints (RefCConfig.path_dists).
    horizons: tuple[int, ...] = (5, 10, 15, 20)


@dataclass
class ControlConfig:
    hidden: int = 512
    k: int = 4                    # future action pairs a_{t+1..t+4}
    att_hidden: int = 1024        # trajectory-guided attention MLP width
    head_hidden: int = 1024       # per-step action head width


@dataclass
class LawConfig:
    hidden: int = 2048            # latent-world-model aux MLP width


@dataclass
class StrategicCtxConfig:
    hidden: int = 512             # ctx GRU width
    d_ctx: int = 64               # strategic token -> measurement seam


@dataclass
class RefCConfig:
    encoder: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    window: int = 8               # shared state window (main stack: 8)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    law: LawConfig = field(default_factory=LawConfig)
    strategic: StrategicCtxConfig = field(default_factory=StrategicCtxConfig)
    speed_hidden: int = 256       # speed-pred (and refc1 speed-class) width
    ego_dropout: float = 0.5      # per-sample Bernoulli zero of v0 (training)
    hierarchy: bool = True        # graft the strategic ctx token (item 6)
    refc1: bool = False           # fixed-distance path + target-speed class
    path_dists: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)   # metres (refc1)
    speed_bins: int = 4           # refc1 target-speed classes over [0, max]
    speed_max: float = 30.0       # m/s


def refc_config() -> RefCConfig:
    """TCP-C at reference scale (~30 M): ResNet-34-style 9-ch encoder
    (~20.8 M) + GRU branches/attention/LAW (~9 M). Measured by
    param_breakdown at instantiation; tests pin the 25-35 M band."""
    return RefCConfig()


def refc_smoke_config() -> RefCConfig:
    """Tiny CPU config (CI smoke / tests / dry runs) — same structure, same
    horizons/K/path_dists, shrunk widths. Episodes: 1-channel 64 px (grid 2,
    4 attention positions)."""
    cfg = RefCConfig()
    cfg.encoder = CNNEncoderConfig(in_channels=1, image_size=64, base_width=8,
                                   blocks=(1, 1, 1, 1))
    cfg.window = 4
    cfg.measurement = MeasurementConfig(hidden=32, d_out=16)
    cfg.trajectory = TrajectoryConfig(hidden=32)
    cfg.control = ControlConfig(hidden=32, att_hidden=32, head_hidden=32)
    cfg.law = LawConfig(hidden=32)
    cfg.strategic = StrategicCtxConfig(hidden=16, d_ctx=8)
    cfg.speed_hidden = 32
    return cfg


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


class TrajectoryBranch(nn.Module):
    """TCP trajectory branch: GRUCell rollout of 4 ego-frame waypoints.

    Hidden init = Linear([pooled, measurement]); each step feeds the running
    position (2D) to the GRU and adds the predicted delta (TCP's accumulate-
    the-offsets scheme; no target point — see module docstring). Returns
    (wp_seq [B, n_steps, 2], hiddens [B, n_steps, H]) — the hiddens drive the
    control branch's trajectory-guided attention.
    """

    def __init__(self, feat_dim: int, d_measure: int, hidden: int,
                 n_steps: int):
        super().__init__()
        self.n_steps = n_steps
        self.h0 = nn.Linear(feat_dim + d_measure, hidden)
        self.gru = nn.GRUCell(2, hidden)
        self.delta = nn.Linear(hidden, 2)

    def forward(self, pooled: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        h = self.h0(torch.cat([pooled, m], dim=-1))
        pos = torch.zeros(pooled.shape[0], 2, dtype=pooled.dtype,
                          device=pooled.device)
        wps, hiddens = [], []
        for _ in range(self.n_steps):
            h = self.gru(pos, h)
            pos = pos + self.delta(h)
            wps.append(pos)
            hiddens.append(h)
        return torch.stack(wps, dim=1), torch.stack(hiddens, dim=1)


class ControlBranch(nn.Module):
    """TCP control branch: GRUCell + trajectory-guided spatial attention.

    Step k: att_logits = MLP([traj_hidden_k, ctrl_hidden_k]) -> softmax over
    the g*g spatial positions of the conv map -> attended feature [B, F];
    action_k = MLP([ctrl_hidden_k, attended]); next GRU input =
    [attended, action_k]. Returns (actions [B, K, 2], att [B, K, g*g]).
    """

    def __init__(self, feat_dim: int, d_measure: int, traj_hidden: int,
                 cfg: ControlConfig, n_pos: int):
        super().__init__()
        self.k = cfg.k
        self.h0 = nn.Linear(feat_dim + d_measure, cfg.hidden)
        self.gru = nn.GRUCell(feat_dim + 2, cfg.hidden)
        self.att = nn.Sequential(
            nn.Linear(traj_hidden + cfg.hidden, cfg.att_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.att_hidden, n_pos))
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden + feat_dim, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_hidden, 2))

    def forward(self, fmap: Tensor, pooled: Tensor, m: Tensor,
                traj_hiddens: Tensor) -> tuple[Tensor, Tensor]:
        flat = fmap.flatten(2).transpose(1, 2)      # [B, n_pos, F]
        h = self.h0(torch.cat([pooled, m], dim=-1))
        actions, atts = [], []
        for k in range(self.k):
            th = traj_hiddens[:, min(k, traj_hiddens.shape[1] - 1)]
            att = F.softmax(self.att(torch.cat([th, h], dim=-1)), dim=-1)
            attended = (att.unsqueeze(-1) * flat).sum(dim=1)     # [B, F]
            a = self.head(torch.cat([h, attended], dim=-1))      # [B, 2]
            actions.append(a)
            atts.append(att)
            h = self.gru(torch.cat([attended, a], dim=-1), h)
        return torch.stack(actions, dim=1), torch.stack(atts, dim=1)


class StrategicCtx(nn.Module):
    """Hierarchy graft: tiny GRU over the W pooled window features -> d_ctx
    token, concatenated into the measurement vector (item 6)."""

    def __init__(self, feat_dim: int, hidden: int, d_ctx: int):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_ctx)

    def forward(self, pooled_seq: Tensor) -> Tensor:   # [B, W, F] -> [B, d_ctx]
        _, h = self.gru(pooled_seq)
        return self.proj(h[-1])


class RefCModel(nn.Module):
    """TCP-C: two-branch TCP stack + LAW aux + hierarchical conditioning."""

    def __init__(self, cfg: RefCConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ResNetEncoder(cfg.encoder)
        feat = self.encoder.feat_dim
        n_pos = self.encoder.grid ** 2
        n_steps = len(cfg.trajectory.horizons)
        if cfg.refc1 and len(cfg.path_dists) != n_steps:
            raise ValueError(f"refc1 needs len(path_dists) == "
                             f"len(horizons): {len(cfg.path_dists)} != "
                             f"{n_steps} (one rollout step per checkpoint)")
        # Hierarchy graft (gated): absent when off — byte-identical model.
        if cfg.hierarchy:
            self.strategic = StrategicCtx(feat, cfg.strategic.hidden,
                                          cfg.strategic.d_ctx)
        d_meas_in = 1 + len(NAV_COMMANDS) \
            + (cfg.strategic.d_ctx if cfg.hierarchy else 0)
        self.measurement = nn.Sequential(
            nn.Linear(d_meas_in, cfg.measurement.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.measurement.hidden, cfg.measurement.d_out),
            nn.ReLU(inplace=True))
        self.trajectory = TrajectoryBranch(feat, cfg.measurement.d_out,
                                           cfg.trajectory.hidden, n_steps)
        self.control = ControlBranch(feat, cfg.measurement.d_out,
                                     cfg.trajectory.hidden, cfg.control, n_pos)
        # LAW aux (item 5): waypoints enter NON-detached — gradients flow.
        self.law_head = nn.Sequential(
            nn.Linear(feat + 2 * n_steps, cfg.law.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.law.hidden, feat))
        # Speed pred (TCP: on the image branch, anti-shortcut placement).
        self.speed_head = nn.Sequential(
            nn.Linear(feat, cfg.speed_hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.speed_hidden, 1))
        # REF-C.1 (gated): target-speed classification head.
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
                v0: Tensor | None = None) -> dict:
        """frames [B, W, C, H, W'], nav_cmd [B] long (None -> `follow`),
        v0 [B] current ego speed in m/s (None -> zeros; scaled /10 inside).

        Returns dict: pooled [B, F], waypoints {key: [B, 2]} (key = horizon,
        or path dist under refc1), wp_seq [B, n_steps, 2], actions [B, K, 2],
        att [B, K, n_pos], law_pred [B, F], speed_pred [B], measurement
        [B, d_m] (+ hierarchy: ctx [B, d_ctx]) (+ refc1: speed_logits
        [B, bins], target_speed [B] expected-value decode).
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

        if nav_cmd is None:                      # unlabeled -> follow (idx 0)
            nav_cmd = torch.zeros(b, dtype=torch.long, device=frames.device)
        nav = F.one_hot(nav_cmd, len(NAV_COMMANDS)).to(pooled.dtype)
        v = torch.zeros(b, 1, dtype=pooled.dtype, device=pooled.device) \
            if v0 is None else (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
        if self.training and self.cfg.ego_dropout > 0:
            keep = (torch.rand(b, 1, device=v.device)
                    >= self.cfg.ego_dropout).to(v.dtype)
            v = v * keep                         # per-sample Bernoulli zero
        m_in = [v, nav] + ([ctx] if ctx is not None else [])
        m = self.measurement(torch.cat(m_in, dim=-1))

        wp_seq, traj_h = self.trajectory(pooled, m)
        actions, att = self.control(fmap, pooled, m, traj_h)
        law_pred = self.law_head(torch.cat([pooled, wp_seq.flatten(1)],
                                           dim=-1))
        speed_pred = self.speed_head(pooled).squeeze(-1)

        keys = self.cfg.path_dists if self.cfg.refc1 \
            else self.cfg.trajectory.horizons
        out = {"pooled": pooled, "wp_seq": wp_seq,
               "waypoints": {k: wp_seq[:, i] for i, k in enumerate(keys)},
               "actions": actions, "att": att, "law_pred": law_pred,
               "speed_pred": speed_pred, "measurement": m}
        if ctx is not None:
            out["ctx"] = ctx
        if self.cfg.refc1:
            logits = self.speed_cls(torch.cat([pooled, m], dim=-1))
            centers = self._speed_bin_centers(logits.device, logits.dtype)
            out["speed_logits"] = logits
            out["target_speed"] = F.softmax(logits, dim=-1) @ centers
        return out


def param_breakdown(model: RefCModel) -> dict[str, int]:
    """Per-module trainable-parameter table (report + config.json row).

    `speed` books the L1 speed head plus, under refc1, the target-speed
    classification head; `strategic` is 0 when hierarchy=False (module not
    constructed). Full-config measurement: encoder ~20.8 M, branches +
    attention + LAW ~9 M, total ~30 M (tests pin the 25-35 M band).
    """
    cnt = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    return {
        "encoder": cnt(model.encoder),
        "measurement": cnt(model.measurement),
        "strategic": cnt(model.strategic) if model.cfg.hierarchy else 0,
        "trajectory": cnt(model.trajectory),
        "control": cnt(model.control),
        "law": cnt(model.law_head),
        "speed": cnt(model.speed_head)
        + (cnt(model.speed_cls) if model.cfg.refc1 else 0),
        "total": cnt(model),
    }
