"""CPU smoke for OUR rig-robust dynamics-estimation encoder.

Guards the pipeline that the launch run will scale: the camera-conditioned encoder,
the multi-domain window dataset (domain mixing + geometry domain-randomisation),
and the combined objective (masked-latent + action-conditioned forward + SIGReg +
supervised metric IDM + odometry grounding). Fast (CPU, synthetic, tiny config).

Design: `…/incoming/2026-07-22-own-dynamics-encoder/DESIGN.md`. Model:
`tanitad.models.dynamics_encoder`. Trainer/smoke: `scripts/train_dynamics_encoder.py`.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_dynamics_encoder as T  # noqa: E402
from tanitad.models.dynamics_encoder import (  # noqa: E402
    CAM_PARAM_NAMES, CameraConditionedEncoder, DynEncConfig, DynamicsEncoderModel,
    dynamics_encoder_smoke_config, normalize_cam_params)


def test_smoke_pipeline_finite_differentiable_fits_and_mixes_domains():
    """The authoritative end-to-end proof (mirrors `--smoke`)."""
    rep = T.smoke()
    assert rep["PASS"]
    assert rep["batch_domains"] >= 2                      # multi-domain batch
    assert rep["grad_norm"] > 0                           # differentiable
    assert rep["fit_total"][1] < rep["fit_total"][0]      # combined loss fell
    assert rep["fit_idm"][1] < rep["fit_idm"][0]          # supervised IDM fell
    assert rep["deployable_params_M"] < 300               # sub-300M envelope
    assert rep["cam_conditioning_response"] > 1e-6        # camera FiLM is live


def test_zero_init_camera_conditioning_is_identity():
    """cam_film zero-init (default) => geometry has NO effect at start, so
    flagship-v1 encoder weights warm-start byte-identically."""
    cfg = dynamics_encoder_smoke_config()
    enc = CameraConditionedEncoder(cfg.enc_cfg(), grid=cfg.grid,
                                   d_readout=cfg.d_readout,
                                   cam_inject_zero_init=True).eval()
    frames = torch.rand(2, cfg.window, cfg.in_channels, cfg.image_size,
                        cfg.image_size)
    raw = torch.tensor([266.0, 128.0, 128.0, 0.0, 1.4, 0.0, 0.0, 1.0])
    cam_a = normalize_cam_params(raw, None).unsqueeze(0).expand(2, -1)
    cam_b = normalize_cam_params(raw.clone().index_put(
        (torch.tensor([3]),), torch.tensor([0.3])), None).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        za = enc.encode_window(frames, cam_a)
        zb = enc.encode_window(frames, cam_b)
    assert torch.allclose(za, zb, atol=1e-6), "zero-init FiLM must ignore geometry"


def test_launch_config_is_sub_300M_and_state_dim_2048():
    """The REAL (launch) config's deployable substrate stays in the sub-300M
    envelope and matches the flagship state_dim (2048) for warm-start parity."""
    model = DynamicsEncoderModel(DynEncConfig())
    assert model.state_dim == 2048
    assert model.deployable_params() < 300e6
    # encoder alone is the flagship-v1 camera ViT scale (~85M at d768 x 12)
    enc_params = sum(p.numel() for p in model.encoder.enc.parameters())
    assert 60e6 < enc_params < 120e6


def test_geometry_aug_keeps_frames_and_params_consistent():
    """The extrinsics domain-randomisation shifts frames AND updates the camera
    params together (the consistency that forces the encoder to USE geometry)."""
    gen = torch.Generator().manual_seed(0)
    frames = torch.rand(5, 9, 32, 32)
    raw = torch.tensor([266.0, 128.0, 128.0, 0.0, 1.4, 0.0, 0.0, 1.0])
    changed = False
    for _ in range(20):
        f2, cam2 = T.geom_augment(frames.clone(), raw, max_dv=4, gen=gen)
        if not torch.equal(f2, frames):
            # a vertical shift must move cy and pitch off their originals
            assert not torch.isclose(cam2[2], raw[2]) or not torch.isclose(
                cam2[3], raw[3])
            changed = True
    assert changed, "geometry aug never fired"


def test_multidomain_dataset_balances_and_labels_domains():
    cfg = dynamics_encoder_smoke_config()
    domains = T.build_smoke_domains(cfg)
    ds = T.MultiDomainWindowDataset(domains, k=cfg.window // 2, stride=2, seed=0)
    counts = ds.domain_window_counts()
    assert len([c for c in counts.values() if c > 0]) >= 3
    b = ds.sample_batch(16)
    assert b["frames"].shape[0] == 16
    assert b["cam"].shape == (16, 2 * len(CAM_PARAM_NAMES))   # scaled | known mask
    assert b["domain"].unique().numel() >= 2       # round-robin mixes domains
