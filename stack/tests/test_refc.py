"""REF-C tests (tanitad/refs/refc.py + scripts/refc_train.py +
scripts/build_refc_anchors.py).

Pins the Anchored-Diffusion-C spec (REF-C redesign: DiffusionDrive-style anchored
truncated-diffusion trajectory decoder replacing the TCP GRU traj/control
branches; LAW aux + strategic-ctx hierarchy KEPT):
(a) build REF-C-base ~110 M (90-130 M band) AND REF-C-XL ~260 M (230-280 M,
    flagship-matched) with a complete param_breakdown that sums exactly and
    exposes the encoder-vs-decoder(-vs-imagination) split; nav vocab consistent
    with refb_labels indices,
(b) FPS anchor vocabulary: spreads (covers the tails), deterministic; the model
    ships built-in default anchors and can load externally-built ones,
(c) forward shapes on the smoke config in BOTH decoder modes (classifier /
    truncated diffusion); classifier == steps=0; eval is deterministic,
(d) H19: anchor priors SHIFT when the maneuver logits change (and do NOT when
    the maneuver graft is off),
(e) all losses fire finite, the full loss reaches EVERY parameter, and the LAW
    loss alone reaches the trajectory decoder (gradients flow through the decoded
    trajectory — the point of the aux),
(f) ego-dropout is per-sample, training-gated: OFF in eval (v0 sensitivity),
    fully zeroing under p=1.0 in train mode,
(g) gated flags follow the byte-identical-when-off discipline: refc1 / hierarchy
    / maneuver / target-latent grafts are absent from the state_dict when off,
(h) refc1 variant: path-checkpoint keys, target-speed classification head +
    expected-value decode in [0, speed_max], losses finite,
(i) optional grafts run: target-latent FiLM + grounded selector; the H15
    imagination graft is gated (absent when off, own-keys-only when on), refines
    the decoder's conv-map tokens in both modes, and has no dead params at grid 8,
(j) refb_labels.path_targets arc-length resample (straight/arc/degenerate),
(k) trainer smoke: 2 steps + ckpt + bit-exact resume, a 1-step --refc1 run, and
    a build_refc_anchors -> --anchors round-trip.
CPU-only, synthetic data.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_refc_anchors  # noqa: E402  (scripts/build_refc_anchors.py)
import refb_labels  # noqa: E402  (scripts/refb_labels.py)
import refc_train  # noqa: E402  (scripts/refc_train.py)
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.data.mixing import save_episode  # noqa: E402
from tanitad.refs.refc import (ImaginationConfig,  # noqa: E402
                               ImaginationField, N_MANEUVERS, N_ROUTE,
                               NAV_COMMANDS, RefCModel, default_anchors,
                               furthest_point_sample, param_breakdown,
                               refc_config, refc_smoke_config, refc_xl_config,
                               synth_anchor_pool)

# ---------- synthetic kinematics (test_refb conventions) ----------------------


def _poses(T: int, dt: float = 0.1, v0: float = 8.0, yaw_rate: float = 0.0,
           accel: float = 0.0, yaw0: float = 0.0) -> torch.Tensor:
    """Unicycle rollout -> contract poses [T, 4] = (x, y, yaw, v)."""
    rows, x, y, yaw, v = [], 0.0, 0.0, yaw0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _drive_episode(T: int, eid: int, yaw_rate: float = 0.0,
                   accel: float = 0.0, v0: float = 8.0, size: int = 64,
                   dt: float = 0.1):
    """Kinematically consistent contract episode (1-channel, smoke-sized)."""
    g = torch.Generator().manual_seed(1000 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, dt=dt, v0=v0, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, dt, eid)


def _make_cached_root(tmp_path: Path, n_train: int = 3, n_val: int = 1,
                      T: int = 200) -> Path:
    """Synthetic cached-mode root: *train*/*val* dirs of ep_*.pt."""
    specs = [(0.0, 0.0), (0.06, 0.0), (0.0, -1.2)]  # keep / left / brake
    for split, n in (("train", n_train), ("val", n_val)):
        d = tmp_path / f"toy-{split}"
        d.mkdir()
        for i in range(n):
            yr, ac = specs[i % len(specs)]
            ep = _drive_episode(T, eid=i, yaw_rate=yr, accel=ac)
            save_episode(ep, str(d / f"ep_{i:05d}.pt"))
    return tmp_path


def _max_h(cfg) -> int:
    return max(max(cfg.trajectory.horizons), refc_train.LAW_AHEAD,
               refc_train.SPEED_AHEAD)


def _batch(root: Path, cfg, n: int = 4):
    eps, _ = refc_train.load_cached_episodes(str(root), "*train*")
    ds = refc_train.FailLoudWindowDataset(
        eps, window=cfg.window, max_horizon=_max_h(cfg),
        channels=cfg.encoder.in_channels)
    return torch.utils.data.default_collate([ds[i] for i in range(n)])


# ---------- (a) build, param count, vocab -------------------------------------

def test_param_count_and_breakdown():
    """REF-C-base builds (meta device: no weight memory); ~110 M params
    (90-130 M band — the size cap was lifted to spend the budget on the encoder,
    the proven Hydra-MDP lever); breakdown covers every parameter exactly and the
    encoder-vs-decoder split shows where the budget went."""
    with torch.device("meta"):
        model = RefCModel(refc_config())
    bd = param_breakdown(model)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[refc] base param_breakdown: {json.dumps(bd, indent=2)}")
    print(f"[refc] base total params: {bd['total']:,}  "
          f"(encoder {bd['encoder']:,} / decoder {bd['decoder']:,})")
    assert bd["total"] == n_total                   # breakdown covers all
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"]
    assert 90_000_000 < bd["total"] < 130_000_000, f"{bd['total']:,}"
    # Budget lands in the encoder (the proven lever): >55 M and the dominant share.
    assert bd["encoder"] > 55_000_000               # V2-99-class trunk
    assert bd["encoder"] > bd["decoder"]            # encoder is where the budget went
    assert bd["decoder"] > 0                         # anchored-diffusion decoder
    assert bd["aux"] > 0                             # maneuver + route heads
    assert bd["strategic"] > 0                       # hierarchy default ON
    assert bd["imagination"] == 0                    # H15 graft OFF in base
    # The anchor decoder cross-attends the [B, F, 8, 8] conv map (grid pinned;
    # F widened past 512, so the decoder's feat_proj adapts — F is not fixed).
    assert model.encoder.grid == 8
    assert model.decoder.feat_proj.in_features == model.encoder.feat_dim


def test_xl_param_count_and_breakdown():
    """REF-C-XL builds at ~260 M (230-280 M band) — the flagship-matched
    capacity control (same-capacity vs the 261 M flagship). SAME decoder
    algorithm as base; the budget grows in the encoder (~200 M, the proven
    lever), the d=512/6-layer decoder, and the gated H15 imagination field
    (~22 M). Breakdown sums exactly and exposes the encoder/decoder/imagination
    split."""
    with torch.device("meta"):
        model = RefCModel(refc_xl_config())
    bd = param_breakdown(model)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[refc] XL param_breakdown: {json.dumps(bd, indent=2)}")
    print(f"[refc] XL total params: {bd['total']:,}  (encoder {bd['encoder']:,} "
          f"/ decoder {bd['decoder']:,} / imagination {bd['imagination']:,})")
    assert bd["total"] == n_total                   # breakdown covers all
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"]
    assert 230_000_000 < bd["total"] < 280_000_000, f"{bd['total']:,}"
    # Encoder is the dominant, proven-lever chunk; decoder grew (d=512/6 layers);
    # the H15 imagination graft is ON at ~22 M.
    assert bd["encoder"] > 150_000_000              # ResNet-L-class trunk
    assert bd["encoder"] > bd["decoder"] > bd["imagination"] > 0
    assert 15_000_000 < bd["imagination"] < 28_000_000, f"{bd['imagination']:,}"
    assert bd["decoder"] > 15_000_000               # d=512, 6 cross-attn layers
    # XL keeps the [B, F, 8, 8] map contract (grid pinned; feat_proj adapts to F).
    assert model.encoder.grid == 8
    assert model.decoder.feat_proj.in_features == model.encoder.feat_dim
    # XL is a same-capacity control: within ~15% of the 261 M flagship.
    assert abs(bd["total"] - 261_000_000) < 40_000_000, f"{bd['total']:,}"


def test_nav_vocab_consistent_with_labels():
    assert NAV_COMMANDS[refb_labels.NAV_FOLLOW] == "follow"
    assert NAV_COMMANDS[refb_labels.NAV_LEFT] == "left"
    assert NAV_COMMANDS[refb_labels.NAV_RIGHT] == "right"
    assert NAV_COMMANDS[refb_labels.NAV_STRAIGHT] == "straight"  # reserved
    # Maneuver / route widths match refb_labels' vocabularies.
    assert N_MANEUVERS == 5 and N_ROUTE == 3
    assert refb_labels.BRAKE_STOP == N_MANEUVERS - 1


# ---------- (b) FPS anchor vocabulary -----------------------------------------

def test_fps_anchor_vocabulary():
    horizons = (5, 10, 15, 20)
    pool = synth_anchor_pool(horizons, pool_size=512, seed=0)
    assert pool.shape == (512, 4, 2)
    anchors = furthest_point_sample(pool, 64, seed=0)
    assert anchors.shape == (64, 4, 2)
    # FPS is deterministic and returns DISTINCT anchors (spreads the vocab).
    a2 = furthest_point_sample(pool, 64, seed=0)
    assert torch.equal(anchors, a2)
    flat = anchors.reshape(64, -1)
    pair_min = torch.cdist(flat, flat).fill_diagonal_(float("inf")).min()
    assert float(pair_min) > 0.0                     # no duplicate anchors
    # FPS covers a WIDER lateral spread than a random draw of the same size
    # (the point vs k-means on ~74%-straight data).
    g = torch.Generator().manual_seed(1)
    rand = pool[torch.randperm(pool.shape[0], generator=g)[:64]]
    assert float(anchors[:, -1, 1].abs().max()) >= \
        float(rand[:, -1, 1].abs().max()) - 1e-4
    # The model's built-in default anchors are deterministic.
    d1 = default_anchors(horizons, 32, pool_size=256, seed=0)
    d2 = default_anchors(horizons, 32, pool_size=256, seed=0)
    assert d1.shape == (32, 4, 2) and torch.equal(d1, d2)


# ---------- (c) forward shapes, both modes ------------------------------------

def test_forward_shapes_both_modes(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()
    batch = _batch(root, cfg)
    b = batch["frames"].shape[0]
    feat = model.encoder.feat_dim
    n = cfg.anchors.n_anchors
    n_steps = len(cfg.trajectory.horizons)
    with torch.no_grad():
        clf = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3], steps=0)          # classifier
        dif = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3], steps=2)          # diffusion
    for out in (clf, dif):
        assert out["pooled"].shape == (b, feat)
        assert out["traj"].shape == (b, n_steps, 2)
        assert torch.equal(out["traj"], out["wp_seq"])            # alias
        assert set(out["waypoints"]) == set(cfg.trajectory.horizons)
        for k in cfg.trajectory.horizons:
            assert out["waypoints"][k].shape == (b, 2)
        assert out["anchor_logits"].shape == (b, n)
        assert out["anchor_traj"].shape == (b, n, n_steps, 2)
        assert out["offset"].shape == (b, n, n_steps, 2)
        assert out["sel_idx"].shape == (b,)
        assert out["maneuver_logits"].shape == (b, N_MANEUVERS)
        assert out["route_logits"].shape == (b, N_ROUTE)
        assert out["law_pred"].shape == (b, feat)
        assert out["ctx"].shape == (b, cfg.strategic.d_ctx)       # hierarchy ON
        assert "speed_logits" not in out                         # refc1 OFF
        for v in ("pooled", "traj", "anchor_logits", "anchor_traj", "offset",
                  "law_pred", "maneuver_logits", "route_logits"):
            assert torch.isfinite(out[v]).all(), v
    # steps=0 is the classifier floor; diffusion refinement MOVES the trajectory.
    assert not torch.equal(clf["traj"], dif["traj"])
    # eval is deterministic in both modes (no ego-dropout, no diffusion noise).
    with torch.no_grad():
        assert torch.equal(dif["anchor_logits"],
                           model(batch["frames"], nav_cmd=batch["nav_cmd"],
                                 v0=batch["pose_last"][:, 3], steps=2)
                           ["anchor_logits"])
    # nav_cmd/v0 default paths (None -> follow / zeros) also run.
    with torch.no_grad():
        out2 = model(batch["frames"])
    assert out2["traj"].shape == (b, n_steps, 2)


# ---------- (d) H19: maneuver logits reweight anchor priors --------------------

def test_maneuver_reweights_anchor_priors(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()                        # graft_maneuver ON
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()
    frames = _batch(root, cfg)["frames"]
    b = frames.shape[0]
    lg_a = torch.zeros(b, N_MANEUVERS)
    lg_a[:, 0] = 6.0                                 # strongly favour lane_keep
    lg_b = torch.zeros(b, N_MANEUVERS)
    lg_b[:, 1] = 6.0                                 # strongly favour turn_left
    with torch.no_grad():
        oa = model(frames, maneuver_logits=lg_a, steps=0)
        ob = model(frames, maneuver_logits=lg_b, steps=0)
    assert not torch.allclose(oa["anchor_logits"], ob["anchor_logits"])
    # With the maneuver graft OFF the external logits cannot move the priors.
    cfg_off = refc_smoke_config()
    cfg_off.graft_maneuver = False
    torch.manual_seed(0)
    m_off = RefCModel(cfg_off).eval()
    with torch.no_grad():
        assert torch.equal(m_off(frames, maneuver_logits=lg_a, steps=0)
                           ["anchor_logits"],
                           m_off(frames, maneuver_logits=lg_b, steps=0)
                           ["anchor_logits"])


# ---------- (e) losses fire, finite, full + LAW-through-trajectory grads -------

def test_losses_finite_and_backward(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    torch.manual_seed(0)
    model = RefCModel(cfg)
    batch = _batch(root, cfg)

    # LAW-only backward reaches the TRAJECTORY DECODER: gradients flow through
    # the decoded trajectory into the anchor offset head (spec item — the point).
    out = refc_train.compute_losses(model, batch)
    out["law"].backward()
    g = model.decoder.offset_head.weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0, "LAW gradient did not reach the decoder"

    # Full loss (diffusion mode): every component finite, every parameter trained.
    model.zero_grad(set_to_none=True)
    out = refc_train.compute_losses(model, batch, mode="diffusion")
    for key in ("loss", "traj", "cls", "law", "route", "man", "speed_cls",
                "speed_mae", "anchor_acc", "man_acc", "nav_follow_frac"):
        assert torch.isfinite(out[key].detach()), key
    assert float(out["speed_cls"]) == 0.0          # refc1 OFF -> zero term
    out["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.step()


# ---------- (f) ego-dropout: per-sample, training-gated -----------------------

def test_ego_dropout_gating(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    torch.manual_seed(0)
    model = RefCModel(cfg)
    frames = _batch(root, cfg)["frames"]
    b = frames.shape[0]
    v_lo = torch.zeros(b)
    v_hi = torch.full((b,), 20.0)

    # EVAL: dropout off — deterministic (classifier), and v0 MUST matter.
    model.eval()
    with torch.no_grad():
        o1 = model(frames, v0=v_hi)
        o2 = model(frames, v0=v_hi)
        o3 = model(frames, v0=v_lo)
    assert torch.equal(o1["traj"], o2["traj"])              # deterministic
    assert not torch.equal(o1["traj"], o3["traj"])          # v0 is live

    # TRAIN, p=1.0: every sample's v0 is Bernoulli-zeroed — v0 cannot matter.
    model.cfg.ego_dropout = 1.0
    model.train()
    with torch.no_grad():
        t1 = model(frames, v0=v_hi)
        t2 = model(frames, v0=v_lo)
    assert torch.equal(t1["traj"], t2["traj"])

    # TRAIN, p=0.5: the mask is PER-SAMPLE (fixed seeds -> deterministic check
    # that two different mask draws move the output).
    model.cfg.ego_dropout = 0.5
    with torch.no_grad():
        torch.manual_seed(0)
        d1 = model(frames, v0=v_hi)
        torch.manual_seed(1)
        d2 = model(frames, v0=v_hi)
    assert not torch.equal(d1["traj"], d2["traj"])


# ---------- (g) gated flags: byte-identical-when-off --------------------------

def test_gated_flags_absent_when_off():
    # refc1 off (default): no speed_cls keys.
    keys = set(RefCModel(refc_smoke_config()).state_dict())
    assert not any(k.startswith("speed_cls") for k in keys)

    def build_keys(**flags):
        cfg = refc_smoke_config()
        for k, v in flags.items():
            setattr(cfg, k, v)
        return set(RefCModel(cfg).state_dict())

    base = build_keys(hierarchy=False, graft_maneuver=False,
                      graft_target_latent=False)
    # Each graft adds ONLY its own module keys — no other structural drift.
    assert build_keys(hierarchy=True, graft_maneuver=False,
                      graft_target_latent=False) - base == {
        k for k in build_keys(hierarchy=True, graft_maneuver=False,
                              graft_target_latent=False)
        if k.startswith("strategic") or k.startswith("decoder.ctx_to_cond")}
    assert build_keys(hierarchy=False, graft_maneuver=True,
                      graft_target_latent=False) - base == {
        "decoder.maneuver_to_anchor.weight"}
    tgt_extra = build_keys(hierarchy=False, graft_maneuver=False,
                           graft_target_latent=True) - base
    assert tgt_extra and all(k.startswith("decoder.tgt_") for k in tgt_extra)

    # Two all-grafts-off builds (same seed) are byte-identical (keys + values).
    torch.manual_seed(0)
    cfg = refc_smoke_config()
    cfg.hierarchy = cfg.graft_maneuver = False
    sd1 = RefCModel(cfg).state_dict()
    torch.manual_seed(0)
    sd2 = RefCModel(cfg).state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k


def test_hierarchy_off_forward(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    cfg.hierarchy = False
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()
    batch = _batch(root, cfg)
    with torch.no_grad():
        out = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3])
    assert "ctx" not in out
    assert torch.isfinite(out["traj"]).all()
    assert param_breakdown(model)["strategic"] == 0


# ---------- (h) refc1 variant -------------------------------------------------

def test_refc1_variant(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    cfg.refc1 = True
    torch.manual_seed(0)
    model = RefCModel(cfg)
    assert any(k.startswith("speed_cls") for k in model.state_dict())
    batch = _batch(root, cfg)
    model.eval()
    with torch.no_grad():
        out = model(batch["frames"], nav_cmd=batch["nav_cmd"],
                    v0=batch["pose_last"][:, 3])
    b = batch["frames"].shape[0]
    assert set(out["waypoints"]) == set(cfg.path_dists)   # path-checkpoint keys
    assert out["speed_logits"].shape == (b, cfg.speed_bins)
    assert out["target_speed"].shape == (b,)              # expected-value decode
    assert float(out["target_speed"].min()) >= 0.0
    assert float(out["target_speed"].max()) <= cfg.speed_max
    # Losses: speed-class CE fires and the full loss trains the cls head too.
    model.train()
    losses = refc_train.compute_losses(model, batch)
    for key in ("loss", "traj", "cls", "law", "route", "man", "speed_cls",
                "speed_mae"):
        assert torch.isfinite(losses[key].detach()), key
    assert float(losses["speed_cls"].detach()) > 0.0
    losses["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------- (i) optional grafts: target-latent FiLM + grounded selector -------

def test_target_latent_and_grounded_grafts(tmp_path):
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    cfg.graft_target_latent = True
    cfg.grounded_selector = True
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()
    assert any(k.startswith("decoder.tgt_") for k in model.state_dict())
    frames = _batch(root, cfg)["frames"]
    b = frames.shape[0]
    tl = torch.randn(b, cfg.tactical_latent_dim)
    with torch.no_grad():
        out = model(frames, target_latent=tl, steps=2)
        base = model(frames, steps=2)                # no target_latent
    assert out["traj"].shape == base["traj"].shape
    assert torch.isfinite(out["traj"]).all()
    # sel_idx is a valid anchor index (grounded selector active).
    assert int(out["sel_idx"].min()) >= 0
    assert int(out["sel_idx"].max()) < cfg.anchors.n_anchors


# ---------- (i2) H15 imagination graft ----------------------------------------

def test_imagination_graft(tmp_path):
    """H15 belief field: gated (absent when off; adds ONLY its own keys when on),
    refines the conv-map tokens the decoder cross-attends in BOTH modes, exports
    a per-cell uncertainty, and (at the real grid=8) every submodule param sits
    in the trajectory-loss gradient path — no dead params."""
    root = _make_cached_root(tmp_path)
    cfg = refc_smoke_config()
    cfg.graft_imagination = True
    cfg.imagination = ImaginationConfig(d=16, depth=2, n_heads=2, ff_mult=2,
                                        head_hidden=16)
    torch.manual_seed(0)
    model = RefCModel(cfg).eval()

    # Byte-identical-when-off: the graft adds ONLY imagination.* keys.
    base_keys = set(RefCModel(refc_smoke_config()).state_dict())
    on_keys = set(model.state_dict())
    assert not any(k.startswith("imagination") for k in base_keys)
    assert on_keys - base_keys == {k for k in on_keys
                                   if k.startswith("imagination")}
    assert on_keys - base_keys                       # non-empty (graft present)

    frames = _batch(root, cfg)["frames"]
    b = frames.shape[0]
    g = cfg.encoder.grid
    with torch.no_grad():
        clf = model(frames, steps=0)
        dif = model(frames, steps=2)
    for out in (clf, dif):
        assert out["traj"].shape == (b, len(cfg.trajectory.horizons), 2)
        assert out["imag_logvar"].shape == (b, g * g)      # per-cell uncertainty
        assert torch.isfinite(out["traj"]).all()
        assert torch.isfinite(out["imag_logvar"]).all()
    assert not torch.equal(clf["traj"], dif["traj"])       # refinement still moves

    # No dead params at the real grid=8: every imagination submodule trains.
    # (The grid=2 smoke lands zero-flow advection samples on the clamped
    # normalized boundary, so this uses grid=8.) flow_head's pre-output Linear
    # sees zero grad at init — the chain rule through the zero-init identity-
    # advection output layer blocks it — then comes alive once that layer moves,
    # so the honest guarantee is: finite grad at init, nonzero after one step.
    torch.manual_seed(0)
    field = ImaginationField(feat_dim=64, grid_hw=8,
                             cfg=ImaginationConfig(d=32, depth=2, n_heads=4,
                                                   ff_mult=2, head_hidden=32))
    opt = torch.optim.SGD(field.parameters(), lr=0.1)
    fmap = torch.randn(2, 64, 8, 8)
    refined, logvar = field(fmap)
    assert refined.shape == fmap.shape and logvar.shape == (2, 64)
    (refined.pow(2).mean() + logvar.pow(2).mean()).backward()
    for name, p in field.named_parameters():        # finite everywhere at init
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    opt.step()
    opt.zero_grad(set_to_none=True)
    refined, logvar = field(fmap)                   # after the identity layer moves
    (refined.pow(2).mean() + logvar.pow(2).mean()).backward()
    for name, p in field.named_parameters():        # every param now trains
        assert float(p.grad.abs().sum()) > 0.0, name


# ---------- (j) path_targets arc-length resample ------------------------------

def test_path_targets_straight_arc_and_degenerate():
    dists = (2.0, 5.0, 10.0, 20.0)
    # Straight at 8 m/s, nonzero world yaw: 20 steps = 16 m of path. In-range
    # checkpoints land exactly at (d, 0); beyond-path checkpoints CLAMP to the
    # final path point (pod/refbpatch training semantics — geometry targets
    # never invent path the ego didn't drive).
    poses = _poses(40, v0=8.0, yaw0=1.0)
    tgt = refb_labels.path_targets(poses[:1], poses[1:21].unsqueeze(0), dists)
    assert tgt.shape == (1, 4, 2)
    path_len = 16.0
    for j, d in enumerate(dists):
        expect = min(d, path_len)
        assert abs(float(tgt[0, j, 0]) - expect) < 1e-2, (d, expect)
        assert abs(float(tgt[0, j, 1])) < 1e-2, d
    # Left arc: lateral ego displacement positive (+y = left), and the far
    # checkpoint bends MORE than the near one.
    poses_l = _poses(40, v0=8.0, yaw_rate=0.3)
    tgt_l = refb_labels.path_targets(poses_l[:1], poses_l[1:21].unsqueeze(0),
                                     dists)
    assert float(tgt_l[0, -1, 1]) > float(tgt_l[0, 0, 1]) > 0.0
    # Speed-invariance: the SAME road (constant curvature = yaw_rate/v, so halve
    # both) at half speed -> same checkpoints (within resampling tolerance).
    poses_s = _poses(40, v0=4.0, yaw_rate=0.15)
    tgt_s = refb_labels.path_targets(poses_s[:1], poses_s[1:21].unsqueeze(0),
                                     (2.0, 5.0))
    assert torch.allclose(tgt_s[0], tgt_l[0, :2], atol=0.15)
    # Stationary (degenerate zero-length path): finite, clamps to the origin.
    poses_0 = _poses(40, v0=0.0)
    tgt_0 = refb_labels.path_targets(poses_0[:1], poses_0[1:21].unsqueeze(0),
                                     dists)
    assert torch.isfinite(tgt_0).all()
    assert float(tgt_0.abs().max()) < 1e-6


# ---------- (k) trainer smoke + resume + anchors round-trip -------------------

def test_trainer_run_ckpt_resume_and_refc1(tmp_path):
    root = _make_cached_root(tmp_path)
    out_dir = tmp_path / "run"
    argv = ["--data-root", str(root), "--out", str(out_dir), "--steps", "2",
            "--batch", "4", "--lr", "1e-3", "--episodes", "0",
            "--log-every", "1", "--device", "cpu", "--smoke"]
    metrics = refc_train.main(argv)
    assert metrics["final"]["step"] == 1
    for k in ("loss", "traj", "cls", "law", "route", "man", "nav_follow_frac"):
        assert np.isfinite(metrics["final"][k]), k
    assert "val" in metrics                        # val cache dir was found
    ckpt_path = out_dir / "ckpt.pt"
    assert ckpt_path.exists()
    conf = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert conf["arch"].startswith("REF-C")
    assert conf["param_breakdown"]["total"] == metrics["n_params_trainable"]
    assert conf["optimizer"]["kind"].startswith("Adam")

    # Two fresh loads reproduce outputs bit-exactly on a fixed input.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = refc_smoke_config()
    m1, m2 = RefCModel(cfg), RefCModel(cfg)
    m1.load_state_dict(ck["model"])
    m2.load_state_dict(ck["model"])
    m1.eval(), m2.eval()
    torch.manual_seed(123)
    fixed = torch.rand(2, cfg.window, 1, 64, 64)
    with torch.no_grad():
        o1, o2 = m1(fixed), m2(fixed)
    for key in ("traj", "anchor_logits", "law_pred", "ctx"):
        assert torch.equal(o1[key], o2[key]), key
        assert torch.isfinite(o1[key]).all(), key

    # Resume: rerun with more steps — picks up at step 2, finishes at 3.
    metrics2 = refc_train.main(argv[:5] + ["4"] + argv[6:])
    assert metrics2["final"]["step"] == 3
    ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ck2["step"] == 3

    # REF-C.1 end-to-end: 1 step through the trainer with --refc1 (classifier).
    out1 = tmp_path / "run-refc1"
    m3 = refc_train.main(["--data-root", str(root), "--out", str(out1),
                          "--steps", "1", "--batch", "4", "--lr", "1e-3",
                          "--log-every", "1", "--device", "cpu", "--smoke",
                          "--mode", "classifier", "--refc1"])
    for k in ("loss", "traj", "cls", "law", "speed_cls", "speed_mae"):
        assert np.isfinite(m3["final"][k]), k
    assert m3["final"]["speed_cls"] > 0.0
    conf1 = json.loads((out1 / "config.json").read_text(encoding="utf-8"))
    assert conf1["cfg"]["refc1"] is True

    # build_refc_anchors -> --anchors round-trip: the FPS vocabulary loads and
    # trains (the anchor buffer travels into the run's config/checkpoint).
    anc_path = tmp_path / "anchors.pt"
    build_refc_anchors.main(["--out", str(anc_path), "--smoke",
                             "--n-anchors", "20"])
    saved = torch.load(anc_path, map_location="cpu", weights_only=True)
    assert saved["anchors"].shape == (20, 4, 2) and saved["method"] == "fps"
    out2 = tmp_path / "run-anchors"
    m4 = refc_train.main(["--data-root", str(root), "--out", str(out2),
                          "--steps", "1", "--batch", "4", "--lr", "1e-3",
                          "--log-every", "1", "--device", "cpu", "--smoke",
                          "--anchors", str(anc_path)])
    assert np.isfinite(m4["final"]["loss"])
