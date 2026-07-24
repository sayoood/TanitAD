"""OUR OWN rig-robust dynamics-estimation encoder (Sayed directive 2026-07-22:
"train our own encoder for the IDM; we need our own model to estimate dynamics").

WHY THIS EXISTS — the measured failure this attacks
----------------------------------------------------
Frozen / light-FT encoders do NOT survive a camera-rig change, and — the sharper,
freshest finding — MULTI-DOMAIN CO-TRAINING DOES NOT FIX IT EITHER. MEASURED
(`…/incoming/2026-07-22-idm-proof/`):
  * frozen our-trained encoder, cross-rig speed R² 0.930 -> -2.465 (`results.json`);
    f-theta canon a no-op (f_eff matched at 266), light-FT inert cross-domain
    0.406 -> 0.411 (`results_regate.json`);
  * co-training on a 2nd domain did NOT recover held-out-rig transfer — held-out
    rig-B light-FT speed R² **-1.61** (vs -1.65 single-domain); symmetric held-out
    comma **0.452** (vs 0.411). ⇒ **data-diversity REFUTED** (`results_multirig.json`).
PUBLISHED corroboration: V-JEPA2-AC (arXiv:2506.09985 §limitations) reports the same
camera-pose sensitivity at 1M+ h SSL. So the collapse is REPRESENTATIONAL — neither
scale nor diversity buys rig-invariance. It must be ENGINEERED IN, explicitly.

THE DISRUPTIVE CORE — GAIA-2-style explicit camera-parameter conditioning
-------------------------------------------------------------------------
GAIA-2 (Wayve, arXiv:2503.20523, verified 3-0) achieves rig-generalization by
computing SEPARATE learned embeddings for **intrinsics, extrinsics, and distortion**,
summing them into ONE unified camera encoding **injected at EACH transformer block**,
and credits generalization to *explicit conditioning + multi-rig training, NOT scale*
(verbatim: "We compute separate embeddings for intrinsics, extrinsics, and
distortion, which are then summed to form a unified camera encoding"). We port that
mechanism into our from-scratch driving encoder so the encoder can EXPLAIN geometry
instead of implicitly binding the ego-motion frame to one camera pose (the mechanism
V-JEPA2-AC lacks and our multi-rig cotrain failed to overcome). Zero-init per-block
injection => a plain ViT at start (flagship-v1 weights warm-start byte-identically).

THE OBJECTIVE — action/dynamics-predictive AND rig-robust latent (all cited)
----------------------------------------------------------------------------
On OUR trained substrate (trained ≫ frozen in-dist is MEASURED: flagship-v1 0.452 vs
REF-A frozen 2.13-2.92), a multi-task loss:
  - masked-latent prediction  (V-JEPA2 2506.09985; predictive temporal SSL beats
    pixel-reconstruction for action-recoverability — 2606.07687 / 2606.31232)
  - action-conditioned forward prediction (our flagship recipe; DINO-WM D5RNACOZEI)
  - SIGReg anti-collapse       (LeJEPA 2511.08544 — our `sigreg.py`)
  - supervised metric IDM      (VPT 2206.11795; DriveWAM 2605.28544 — `idm_head.py`)
  - metric-scale grounding     (odometry Δpose — our `metric_dynamics.py`; Cosmos-3
    shows SUPERVISED in-domain grounding beats monocular geometry/VO on metric ATE)

Deployable substrate = camera-conditioned encoder + readout + IDM head (sub-300M,
envelope-checked). Predictor / masked-predictor / grounding are TRAINING-time
auxiliaries (kept OUTSIDE the deployed module). Dependency-light (torch only);
CPU-unit-tested (`stack/tests/test_dynamics_encoder.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tanitad.config import EncoderConfig, PredictorConfig
from tanitad.models.encoder import ViTEncoder
from tanitad.models.metric_dynamics import MetricInverseDynamics, relative_ego_pose
from tanitad.models.predictor import OperativePredictor, change_weighted_mse
from tanitad.models.readout import SpatialGridReadout
from tanitad.models.sigreg import SigReg, position_relaxed

# --------------------------------------------------------------------------- #
# Camera-parameter featurisation — GAIA-2 (arXiv:2503.20523): SEPARATE learned #
# embeddings for INTRINSICS, EXTRINSICS and DISTORTION, summed into ONE camera  #
# encoding, injected at EACH transformer block. A per-parameter known/unknown   #
# mask makes "unknown intrinsics" (e.g. L2D ships none) an in-distribution      #
# input pattern rather than a silent default.                                   #
# --------------------------------------------------------------------------- #
CAM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intrinsics", ("f_eff", "cx", "cy")),        # GAIA-2: focal + principal point
    ("extrinsics", ("pitch", "height", "roll")),  # camera pose vs the road frame
    ("distortion", ("k1", "is_fisheye")),         # f-theta poly vs rectilinear
)
CAM_PARAM_NAMES: tuple[str, ...] = tuple(n for _g, ns in CAM_GROUPS for n in ns)
N_CAM = len(CAM_PARAM_NAMES)                       # 8
# Rough normalisers -> O(1). px over the 256 canvas; pitch/roll rad; height m.
_CAM_SCALE = torch.tensor([1 / 256, 1 / 256, 1 / 256, 1.0, 1 / 2.0, 1.0, 1.0, 1.0])
# Neutral defaults for UNKNOWN params; the mask (0) tells the net not to trust them.
_CAM_DEFAULT = torch.tensor([266.0, 128.0, 128.0, 0.0, 1.4, 0.0, 0.0, 0.0])


def normalize_cam_params(raw: Tensor, known: Tensor | None = None) -> Tensor:
    """raw [.., 8] physical camera params -> [.., 16] = (scaled 8 | known mask 8).

    ``known`` [.., 8] in {0,1}; None => all known. Unknown columns are replaced by
    the neutral default BEFORE scaling so a garbage/NaN raw value cannot leak in."""
    raw = raw.float()
    if known is None:
        known = torch.ones_like(raw)
    known = known.float()
    raw = torch.where(known > 0.5, raw, _CAM_DEFAULT.to(raw))
    return torch.cat([raw * _CAM_SCALE.to(raw), known], dim=-1)


class CameraEncoding(nn.Module):
    """GAIA-2 camera conditioning: per-group MLPs (intrinsics / extrinsics /
    distortion), each mapping (scaled params | known mask) for that group to
    ``d_model``, SUMMED into one unified camera encoding [.., d_model]."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.embeds = nn.ModuleList()
        self.slices: list[tuple[int, int]] = []
        off = 0
        for _name, ns in CAM_GROUPS:
            g = len(ns)
            self.slices.append((off, off + g))
            off += g
            self.embeds.append(nn.Sequential(
                nn.Linear(2 * g, hidden), nn.GELU(), nn.Linear(hidden, d_model)))

    def forward(self, cam16: Tensor) -> Tensor:
        scaled, known = cam16[..., :N_CAM], cam16[..., N_CAM:]
        enc = None
        for (lo, hi), emb in zip(self.slices, self.embeds):
            feat = torch.cat([scaled[..., lo:hi], known[..., lo:hi]], dim=-1)
            e = emb(feat)
            enc = e if enc is None else enc + e
        return enc


class CameraConditionedEncoder(nn.Module):
    """ViT + spatial-grid readout with GAIA-2 per-block camera conditioning.

    Composes the flagship ``ViTEncoder`` + ``SpatialGridReadout`` (so flagship-v1
    weights load into ``.enc`` / ``.readout`` for the warm-start branch) and, before
    EACH ViT block, ADDS a per-block projection of the unified camera encoding to the
    token stream (GAIA-2 "added to the input latents at each transformer block").
    Per-block projections are zero-init => identity at start (warm-start byte-
    identical). The readout still sees an unmodified square token grid, so
    ``state_dim`` matches the flagship (2048 for the launch config).
    """

    def __init__(self, enc_cfg: EncoderConfig, grid: int, d_readout: int,
                 cam_hidden: int = 128, cam_inject_zero_init: bool = True,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.enc = ViTEncoder(enc_cfg)
        self.readout = SpatialGridReadout(self.enc.n_tokens, enc_cfg.d_model,
                                          grid=grid, d_readout=d_readout)
        self.cam_enc = CameraEncoding(enc_cfg.d_model, hidden=cam_hidden)
        # one zero-init projection per block -> additive per-block injection.
        self.inject = nn.ModuleList(
            nn.Linear(enc_cfg.d_model, enc_cfg.d_model)
            for _ in range(len(self.enc.blocks)))
        for lin in self.inject:
            if cam_inject_zero_init:
                nn.init.zeros_(lin.weight)
            else:
                nn.init.trunc_normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        self.state_dim = self.readout.out_dim

    def _tokens(self, frames: Tensor, cam16: Tensor) -> Tensor:
        e = self.enc
        t = e.patch(frames).flatten(2).transpose(1, 2) + e.pos     # [B,N,D]
        cam = self.cam_enc(cam16)                                   # [B,D]
        ckpt = self.grad_checkpoint and self.training and t.requires_grad
        for inj, blk in zip(self.inject, e.blocks):
            t = t + inj(cam).unsqueeze(1)                           # GAIA-2 per-block
            t = (torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
                 if ckpt else blk(t))                               # F-5 memory lever
        return e.norm(t)

    def forward(self, frames: Tensor, cam16: Tensor) -> Tensor:
        """frames [B,C,H,W] (float 0..1), cam16 [B,16] -> z [B, state_dim]."""
        return self.readout(self._tokens(frames, cam16))

    def encode_window(self, frames: Tensor, cam16: Tensor) -> Tensor:
        """frames [B,W,C,H,W] uint8-or-float, cam16 [B,16] -> z [B,W,state_dim].
        One camera vector per clip (broadcast over the W frames)."""
        b, w, c, h, ww = frames.shape
        f = frames.reshape(b * w, c, h, ww).float()
        if f.max() > 1.5:                                          # accept uint8
            f = f / 255.0
        cam = cam16.unsqueeze(1).expand(b, w, cam16.shape[-1]).reshape(b * w, -1)
        z = self.readout(self._tokens(f, cam))
        return z.reshape(b, w, -1)


class MaskedLatentPredictor(nn.Module):
    """V-JEPA2-style masked-latent prediction over a window of latents.

    A small bidirectional transformer sees the VISIBLE window latents (masked
    positions replaced by a learned mask token + position embedding) and predicts
    the FULL latent sequence; the loss is scored on the masked positions only
    (arXiv:2506.09985). Predictive temporal SSL — 2606.07687 / 2606.31232 show it
    beats pixel-reconstruction (MAE/VideoMAE) for action-recoverability + robustness.
    """

    def __init__(self, state_dim: int, window: int, d_model: int = 256,
                 depth: int = 2, n_heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(state_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, state_dim)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        """z [B,W,S] (targets), mask [B,W] bool (True = hidden) -> pred [B,W,S]."""
        b, w, _ = z.shape
        x = self.in_proj(z)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, w, -1), x)
        x = x + self.pos[:, :w]
        return self.out(self.blocks(x))


@dataclass
class DynEncConfig:
    """Config for the dynamics-estimation encoder + its training auxiliaries."""
    # --- backbone (defaults = flagship-v1 camera encoder, sub-300M) ---
    in_channels: int = 9              # 3-frame stack, D-015 input contract
    image_size: int = 256
    patch_size: int = 16              # -> 16x16 = 256 tokens
    d_model: int = 768
    depth: int = 12                   # flagship4b encoder depth
    n_heads: int = 12
    grid: int = 4                     # readout grid -> state_dim = grid^2 * d_readout
    d_readout: int = 128              # -> state_dim 2048 (flagship parity)
    cam_hidden: int = 128
    cam_inject_zero_init: bool = True
    grad_checkpoint: bool = False     # F-5: trade compute for GPU memory (real run)
    # --- window / objective ---
    window: int = 9                   # 2k+1 non-causal IDM window (k=4)
    pred_window: int = 8              # action-conditioned forward history
    mask_ratio: float = 0.4           # masked-latent prediction
    # --- auxiliary head sizes ---
    idm_d_model: int = 256
    idm_depth: int = 3
    mlp_d_model: int = 256            # masked-latent predictor width
    mlp_depth: int = 2
    ground_hidden: int = 512
    # --- loss weights (rebalanced at launch; see LAUNCH_PLAN.md) ---
    w_idm: float = 1.0
    w_fwd: float = 1.0
    w_mask: float = 1.0
    w_sigreg: float = 0.1             # LeJEPA validated lambda
    w_ground: float = 1.0
    sigreg_slices: int = 512
    sigreg_free_dims: int = 0
    traj_scale: float = 10.0
    pose_scale: float = 10.0

    def enc_cfg(self) -> EncoderConfig:
        return EncoderConfig(in_channels=self.in_channels, image_size=self.image_size,
                             patch_size=self.patch_size, d_model=self.d_model,
                             depth=self.depth, n_heads=self.n_heads)


def dynamics_encoder_smoke_config() -> DynEncConfig:
    """Tiny CPU config (CI smoke / dry runs) — same structure, shrunk everywhere.
    32px 9-ch, patch 8 -> 4x4 grid, readout grid 2 x d_readout 8 -> state_dim 32."""
    return DynEncConfig(
        image_size=32, patch_size=8, d_model=48, depth=2, n_heads=4,
        grid=2, d_readout=8, cam_hidden=32, window=5, pred_window=4,
        idm_d_model=32, idm_depth=2, mlp_d_model=32, mlp_depth=1,
        ground_hidden=32, sigreg_slices=32)


class DynamicsEncoderModel(nn.Module):
    """The full trainable stack: camera-conditioned encoder + the four auxiliary
    objectives. ``deployable`` = encoder + readout + IDM head (what ships); the
    predictor / masked-predictor / grounding are training-time only.
    """

    def __init__(self, cfg: DynEncConfig):
        super().__init__()
        import sys
        from pathlib import Path
        # idm_head lives under scripts/ (dependency-light, shared with the IDM proof)
        sp = str(Path(__file__).resolve().parents[2] / "scripts")
        if sp not in sys.path:
            sys.path.insert(0, sp)
        import idm_head as ih                                       # noqa: E402
        self._ih = ih
        self.cfg = cfg
        self.encoder = CameraConditionedEncoder(
            cfg.enc_cfg(), grid=cfg.grid, d_readout=cfg.d_readout,
            cam_hidden=cfg.cam_hidden, cam_inject_zero_init=cfg.cam_inject_zero_init,
            grad_checkpoint=cfg.grad_checkpoint)
        S = self.encoder.state_dim
        self.state_dim = S
        # (a) supervised metric IDM head (the deployed dynamics readout)
        self.idm_head = ih.IDMHead(state_dim=S, d_model=cfg.idm_d_model,
                                   depth=cfg.idm_depth, window=cfg.window)
        # (b) action-conditioned forward predictor (dynamics-predictive objective)
        self.predictor = OperativePredictor(
            PredictorConfig(d_model=cfg.idm_d_model, depth=2, n_heads=4,
                            window=cfg.pred_window, horizons=(1,), action_dim=2),
            state_dim=S)
        # (c) masked-latent predictor (V-JEPA2-style SSL)
        self.masked = MaskedLatentPredictor(S, cfg.window, d_model=cfg.mlp_d_model,
                                            depth=cfg.mlp_depth)
        # (d) metric-scale grounding (odometry Δpose)
        self.invdyn = MetricInverseDynamics(S, hidden=cfg.ground_hidden)
        # (e) anti-collapse
        self.sigreg = SigReg(n_slices=cfg.sigreg_slices)
        # FIXED scalar standardiser (persistent buffers so resume restores it) —
        # a PER-BATCH fit blows up when a batch has near-constant steer/accel
        # (std->1e-6 -> huge standardised targets); driving-physical defaults are
        # stable, and the trainer refines them from a corpus sample at startup.
        self.register_buffer("std_mean", torch.tensor([8.0, 0.0, 0.0, 0.0]))
        self.register_buffer("std_std", torch.tensor([5.0, 0.15, 0.05, 1.0]))

    def set_standardizer(self, mean: Tensor, std: Tensor) -> None:
        self.std_mean.copy_(mean.to(self.std_mean))
        self.std_std.copy_(std.to(self.std_std).clamp_min(1e-3))

    # -- param accounting for the sub-300M envelope check -------------------- #
    def deployable_params(self) -> int:
        return (sum(p.numel() for p in self.encoder.parameters())
                + sum(p.numel() for p in self.idm_head.parameters()))

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def training_step(self, batch: dict) -> tuple[Tensor, dict]:
        """One combined-objective step on a multi-domain batch.

        batch keys: frames [B,W,C,H,W], cam [B,16], actions [B,W,2], poses [B,W,4],
        scal_tgt [B,4], traj_tgt [B,H,2], step_tgt [B,3]. Returns (loss, logs)."""
        ih, cfg = self._ih, self.cfg
        frames, cam = batch["frames"], batch["cam"]
        z = self.encoder.encode_window(frames, cam)                # [B,W,S]
        b, w, S = z.shape
        c = w // 2
        logs: dict[str, float] = {}
        total = z.new_zeros(())

        # (a) supervised metric IDM at the window center (FIXED standardiser —
        # never per-batch, which is unstable on near-constant-target batches)
        std = ih.Standardizer(self.std_mean, self.std_std)
        idm_out = self.idm_head(z)
        idm_l = ih.idm_loss(idm_out, batch["scal_tgt"], batch["traj_tgt"], std,
                            traj_scale=cfg.traj_scale)
        total = total + cfg.w_idm * idm_l["loss"]
        logs["idm"] = float(idm_l["loss"].detach())

        # (b) action-conditioned forward prediction: history [0:pred_window] ->
        #     z_{pred_window} (horizon 1). Target stop-grad (JEPA), SIGReg holds
        #     anti-collapse. change-weighted so static content cannot dominate.
        pw = cfg.pred_window
        z_hat = self.predictor(z[:, :pw], batch["actions"][:, :pw])[1]
        tgt = z[:, pw].detach()
        fwd_l = change_weighted_mse(z_hat, tgt, z[:, pw - 1].detach())
        total = total + cfg.w_fwd * fwd_l
        logs["fwd"] = float(fwd_l.detach())

        # (c) masked-latent prediction (V-JEPA2-style); score masked positions only
        mask = torch.rand(b, w, device=z.device) < cfg.mask_ratio
        mask[:, c] = False                    # keep the center visible (IDM site)
        pred = self.masked(z, mask)
        m = mask.unsqueeze(-1)
        denom = m.sum().clamp_min(1)
        mask_l = (((pred - z.detach()) ** 2) * m).sum() / (denom * S)
        total = total + cfg.w_mask * mask_l
        logs["mask"] = float(mask_l.detach())

        # (d) metric-scale grounding: (z_c, z_{c+1}) -> odometry Δpose
        dpose = self.invdyn(z[:, c], z[:, c + 1])
        st = batch["step_tgt"]
        ground_l = (((dpose[..., :2] - st[..., :2]) / cfg.pose_scale) ** 2).mean() \
            + (self._wrap(dpose[..., 2] - st[..., 2]) ** 2).mean()
        total = total + cfg.w_ground * ground_l
        logs["ground"] = float(ground_l.detach())

        # (e) SIGReg anti-collapse on the online latents
        sig_l = position_relaxed(self.sigreg, z.reshape(b * w, S),
                                 cfg.sigreg_free_dims)
        total = total + cfg.w_sigreg * sig_l
        logs["sigreg"] = float(sig_l.detach())

        logs["total"] = float(total.detach())
        return total, logs

    @staticmethod
    def _wrap(a: Tensor) -> Tensor:
        return (a + math.pi) % (2 * math.pi) - math.pi
