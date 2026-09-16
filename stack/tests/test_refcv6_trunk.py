"""refcv6 §2 — the ImageNet trunk over frame history, and §2b ego history.

Each test names its kind: *fails without the feature* / *control that must read
a known value* / *deliberate regression*.

⭐ The load-bearing identities this file exists for:
  1. the trunk really carries **ImageNet** weights (a statistic, not "no
     exception") — `E-SEED-2`'s trap is a prior that was never loaded;
  2. the two feature maps are **16x40x256** and **8x20x512** at 256x640, read
     from ``timm``'s ``feature_info`` rather than hard-coded;
  3. the normalisation happens **exactly once**;
  4. the **9-channel stem inflation reproduces the 3-channel response**, which
     is what makes it a knockout of the INPUT and not a different network;
  5. the **K-frame trunk with identity fusion is bit-identical to K = 1**, so
     frame history is a removable graft;
  6. the DD parameter groups carry the intended learning rates;
  7. the ego-history encoder **cannot read the future** — proved by MUTATING
     the future half of the tensor, not by reading the code.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

timm = pytest.importorskip("timm")

from tanitad.models import ego_history as eh                     # noqa: E402
from tanitad.models import timm_trunk as tt                      # noqa: E402
from tanitad.refs import refc                                    # noqa: E402

#: ⭐ PI 2026-09-16: 256 x 1024 cylindrical for ALL future training (same
#: 120 deg field; f_ref 305.577 -> 488.92). 640 is kept as a PARAMETER rather
#: than deleted: the banked caches are still 640 until the data agent lands the
#: rebuild, and a suite that only ever saw ONE width cannot catch a hard-coded
#: grid any better than one that only ever saw the other.
HW = (256, 1024)
GEOMETRIES = [(256, 1024), (256, 640)]
#: ``(backbone, stride-16 channels, stride-32 channels)`` — the PI's PRIMARY
#: and the second comparison run. These counts are what timm reports and what
#: this suite refuses to let anyone write down.
BACKBONES = [("resnet34.a1_in1k", 256, 512),
             ("resnet101.a1_in1k", 1024, 2048)]


def _trunk(**kw):
    """⚠️ Defaults to **resnet34** even though the PI's primary is resnet101:
    most tests below assert the small backbone's exact shapes and parameter
    count, and the r101 arm is covered by the parameterised tests. Every test
    that cares names its backbone."""
    kw.setdefault("image_hw", HW)
    kw.setdefault("model_name", "resnet34.a1_in1k")
    return tt.TimmResNetTrunk(tt.TimmTrunkConfig(**kw)).eval()


# ---------------------------------------------------------------------------
# 1. The weights are REAL
# ---------------------------------------------------------------------------
def test_trunk_loads_REAL_imagenet_weights_not_just_no_exception():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. ``pretrained=True`` raises on a
    failed download but NOT on a silent fallback, and a randomly-initialised
    trunk that claims ImageNet would train, converge and mean nothing. The
    pinned statistic is ``|conv1.w|.sum() = 1279.85`` (MEASURED 2026-09-16 on
    the 87,278,522 B ``timm/resnet34.a1_in1k`` safetensors); a He init of the
    same shape reads ~875."""
    t = _trunk(frames=1)
    stem = dict(t.net.named_modules())["conv1"]
    got = float(stem.weight.detach().abs().sum())
    assert abs(got - tt.IMAGENET_CONV1_ABS_SUM) / tt.IMAGENET_CONV1_ABS_SUM < 0.02
    # ...and the same tensor's std is ~2x a He init's, which is the property
    # that makes the sum diagnostic rather than coincidental.
    he = (2.0 / (3 * 7 * 7)) ** 0.5
    assert float(stem.weight.detach().std()) > 1.6 * he


@pytest.mark.parametrize("name,_a,_b", BACKBONES)
def test_random_init_FAILS_the_imagenet_check(name, _a, _b):
    """⛔ DELIBERATE REGRESSION — the MUTATION that proves test 1 is alive.
    Re-introduce the defect (a trunk that never loaded the prior) and the
    verifier must refuse — for EVERY backbone the PI may launch."""
    net = timm.create_model(name, pretrained=False, features_only=True,
                            out_indices=(3, 4))
    with pytest.raises(RuntimeError, match="were NOT loaded"):
        tt._assert_pretrained_loaded(net, name)


@pytest.mark.parametrize("name,_a,_b", BACKBONES)
def test_every_candidate_the_PI_may_launch_is_VERIFIABLE(name, _a, _b):
    """⛔ CONTROL. `pretrained=True` must be PROVABLE for the arms that will
    actually run, not only for the one that happened to be the default."""
    assert name in tt.CANDIDATE_BACKBONES
    assert name in tt._PINNED_STATS
    stem = tt._find_stem(_trunk(frames=1, model_name=name).net)[1]
    got = float(stem.weight.detach().abs().sum())
    want = tt._PINNED_STATS[name]
    assert abs(got - want) / want < 0.02


# ---------------------------------------------------------------------------
# 2. The shapes, read from timm rather than written down
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hw", GEOMETRIES)
@pytest.mark.parametrize("name,c16,c32", BACKBONES)
def test_both_feature_maps_have_the_documented_shapes(hw, name, c16, c32):
    """FAILS WITHOUT THE FEATURE, and it is the test that catches a hard-coded
    40 / 20 / 640 / 160. The grids must be DERIVED — stride-16 is h//16 x w//16
    (PERCEPTION: an oracle on the stride-32 grid tops out at AP 0.3341 against
    0.4713 on stride-16), stride-32 is h//32 x w//32 (the planner tokens: 256
    at 1024 px, 160 at 640)."""
    h, w = hw
    t = _trunk(frames=1, model_name=name, image_hw=hw)
    s16, s32, pooled = t.forward_features(torch.rand(1, 3, h, w))
    assert tuple(s16.shape) == (1, c16, h // 16, w // 16)
    assert tuple(s32.shape) == (1, c32, h // 32, w // 32)
    assert tuple(pooled.shape) == (1, c32)
    assert (t.s16_dim, t.s32_dim) == (c16, c32)
    assert t.grid_shape == (h // 32, w // 32)
    assert t.s16_shape == (h // 16, w // 16)
    if hw == (256, 1024):          # the PI's numbers, spelled out once
        assert t.s16_shape == (16, 64) and t.grid_shape == (8, 32)


@pytest.mark.parametrize("name,c16,c32", BACKBONES)
def test_channel_counts_come_from_timm_feature_info_not_a_hardcode(name, c16,
                                                                   c32):
    """⛔ CONTROL. The PI chose resnet101 as the PRIMARY, so a hard-coded
    256/512 is the defect — and it would build `feat_proj` 1536 channels too
    narrow, far from where anyone would look."""
    t = _trunk(frames=1, model_name=name)
    assert [t.s16_dim, t.s32_dim] == list(t.net.feature_info.channels())
    assert list(t.net.feature_info.reduction()) == [16, 32]
    assert [t.s16_dim, t.s32_dim] == [c16, c32]
    assert tt.trunk_feature_channels(name) == (c16, c32)


def test_the_two_chosen_backbones_report_DIFFERENT_widths():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE — MEASURED 2026-09-16, and the
    PI's own figures. A seam that survived only because the two backbones
    agreed would not be a seam."""
    a = _trunk(frames=1, model_name="resnet34.a1_in1k")
    b = _trunk(frames=1, model_name="resnet101.a1_in1k")
    assert (a.s16_dim, a.s32_dim) == (256, 512)
    assert (b.s16_dim, b.s32_dim) == (1024, 2048)
    assert a.trunk_param_count() == 21_284_672
    assert b.trunk_param_count() == 42_500_160


@pytest.mark.parametrize("name,c16,c32", BACKBONES)
def test_the_CONFIG_feat_dim_follows_the_BACKBONE(name, c16, c32):
    """⛔ CONTROL, and the one a hard-coded 512 would fail. `feat_proj` is
    sized from `CNNEncoderConfig.feat_dim` BEFORE the trunk is built, so the
    config must be able to answer for resnet101 as well as resnet34."""
    cfg = refc.CNNEncoderConfig(in_channels=9, image_size=256,
                                image_width=1024, trunk="timm",
                                trunk_name=name)
    assert cfg.feat_dim == c32
    assert cfg.s16_dim == c16
    assert cfg.grid_shape == (8, 32)
    with pytest.raises(ValueError, match="only a stride-32 map"):
        refc.CNNEncoderConfig().s16_dim


# ---------------------------------------------------------------------------
# 3. Normalisation — exactly once
# ---------------------------------------------------------------------------
def test_imagenet_normalisation_is_applied_EXACTLY_ONCE():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. `E-SEED-2` measured that
    skipping the normalisation wastes the prior; its twin — a helpful caller
    normalising a second time — halves the contrast just as quietly."""
    t = _trunk(frames=1)
    x = torch.rand(1, 3, *HW)
    t.norm_calls = 0
    t.forward_features(x)
    assert t.norm_calls == 1
    t.forward_features(x, already_normalised=True)
    assert t.norm_calls == 1          # the explicit opt-out is the ONLY skip
    # and the transform is the ImageNet one, on a value we can compute by hand
    n = t.normalise(torch.zeros(1, 3, 4, 4))
    for c in range(3):
        want = -tt.IMAGENET_MEAN[c] / tt.IMAGENET_STD[c]
        assert abs(float(n[0, c, 0, 0]) - want) < 1e-6


def test_nine_channel_normalisation_TILES_rather_than_averages():
    """⛔ CONTROL. Each RGB triple of the stack is its own image and gets its
    own statistics. Averaging them would apply the red constant to a blue
    channel two frames along."""
    t = _trunk(frames=3, mode="inflate")
    n = t.normalise(torch.zeros(1, 9, 2, 2))
    for k in range(3):
        for c in range(3):
            want = -tt.IMAGENET_MEAN[c] / tt.IMAGENET_STD[c]
            assert abs(float(n[0, 3 * k + c, 0, 0]) - want) < 1e-6


# ---------------------------------------------------------------------------
# 4. The 9-channel stem inflation — the registered knockout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("img", ["grey", "random"])
def test_nine_channel_inflation_reproduces_three(img):
    """FAILS WITHOUT THE FEATURE, and it is the property that makes `inflate`
    a knockout of the INPUT. Weights repeated and DIVIDED BY 3 make the
    9-channel trunk an exact re-expression of the 3-channel one on a
    repeated-frame input:
      sum_{j<3} sum_c (w[:,c]/3) x[c] == sum_c w[:,c] x[c].
    ⚠️ Both models must be in EVAL mode: a forward in train mode updates the
    BatchNorm running statistics and the two trunks then differ for a reason
    that has nothing to do with the inflation."""
    x = (torch.full((1, 3, *HW), 0.5) if img == "grey"
         else torch.rand(1, 3, *HW))
    t3 = _trunk(frames=1)
    t9 = _trunk(frames=3, mode="inflate")
    with torch.no_grad():
        a16, a32, _ = t3.forward_features(x)
        b16, b32, _ = t9.forward_features(x.repeat(1, 3, 1, 1))
    for a, b in ((a16, b16), (a32, b32)):
        rel = float((a - b).abs().max()) / max(float(a.abs().max()), 1e-9)
        assert rel < 1e-5, f"{img}: relative deviation {rel}"


def test_inflation_WITHOUT_the_divide_does_NOT_reproduce():
    """⛔ DELIBERATE REGRESSION — the MUTATION that proves the test above is
    alive. Repeat the stem WITHOUT dividing and the response must diverge, or
    the previous test would pass for the wrong reason."""
    t3 = _trunk(frames=1)
    stem = dict(t3.net.named_modules())["conv1"]
    bad = nn.Conv2d(9, stem.out_channels, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        bad.weight.copy_(stem.weight.repeat(1, 3, 1, 1))   # ⛔ no / 3
    t9 = _trunk(frames=3, mode="inflate")
    t9.net.conv1 = bad
    x = torch.rand(1, 3, *HW)
    with torch.no_grad():
        a16, _, _ = t3.forward_features(x)
        b16, _, _ = t9.forward_features(x.repeat(1, 3, 1, 1))
    rel = float((a16 - b16).abs().max()) / float(a16.abs().max())
    assert rel > 0.1, f"the missing /3 was invisible (rel {rel})"


def test_inflate_stem_REFUSES_an_already_inflated_stem():
    """⛔ CONTROL. Inflating twice would divide the prior twice."""
    conv = nn.Conv2d(9, 8, 3)
    with pytest.raises(ValueError, match="3-channel ImageNet stem"):
        tt.inflate_stem_(conv, 9)


# ---------------------------------------------------------------------------
# 5. FRAME HISTORY (PI 2026-09-16) — shared weights, fusion after the trunk
# ---------------------------------------------------------------------------
def test_shared_mode_runs_K_passes_of_a_THREE_channel_stem():
    """FAILS WITHOUT THE FEATURE. The whole point of the K-pass construction
    is that the pretrained stem keeps seeing exactly 3 ImageNet channels, so
    the stem must NOT have been inflated."""
    t = _trunk(frames=3, mode="shared")
    assert dict(t.net.named_modules())["conv1"].in_channels == 3
    assert t.fuse16 is not None and t.fuse32 is not None
    s16, s32, _ = t.forward_features(torch.rand(1, 9, *HW))
    assert tuple(s16.shape) == (1, 256, 16, 64)
    assert tuple(s32.shape) == (1, 512, 8, 32)


@pytest.mark.parametrize("fuse", ["concat1x1", "attn"])
def test_identity_init_makes_frame_history_a_REMOVABLE_graft(fuse):
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. With the identity init a fresh
    K-frame trunk emits what the single-frame trunk emits — all the weight on
    the NEWEST frame. Without it, "3 frames beat 1" would be confounded with
    "a random 1x1 conv was inserted"."""
    x = torch.rand(2, 9, *HW)
    t1 = _trunk(frames=1)
    tk = _trunk(frames=3, mode="shared", fuse=fuse)
    with torch.no_grad():
        a16, a32, _ = t1.forward_features(x[:, -3:])
        b16, b32, _ = tk.forward_features(x)
    tol = 0.0 if fuse == "concat1x1" else 1e-4
    assert float((a16 - b16).abs().max()) <= tol
    assert float((a32 - b32).abs().max()) <= tol


def test_plain_init_fusion_does_NOT_reproduce_the_single_frame_trunk():
    """⛔ DELIBERATE REGRESSION — the MUTATION for the test above. Turn the
    identity init off and the equality must break."""
    x = torch.rand(2, 9, *HW)
    t1 = _trunk(frames=1)
    tk = _trunk(frames=3, mode="shared", fuse_identity_init=False)
    with torch.no_grad():
        a32, _ = t1(x[:, -3:])
        b32, _ = tk(x)
    assert float((a32 - b32).abs().max()) > 1e-3


def test_fusion_reads_the_NEWEST_frame_not_the_oldest():
    """⛔ CONTROL. REF-C stacks OLDEST -> NEWEST (`D-015`: latest = [-3:]).
    Perturbing the newest frame must move the identity-initialised output;
    perturbing the oldest must not."""
    tk = _trunk(frames=3, mode="shared")
    x = torch.rand(1, 9, *HW)
    old, new = x.clone(), x.clone()
    old[:, 0:3] = torch.rand(1, 3, *HW)
    new[:, 6:9] = torch.rand(1, 3, *HW)
    with torch.no_grad():
        base, _ = tk(x)
        a, _ = tk(old)
        b, _ = tk(new)
    assert float((base - a).abs().max()) == 0.0
    assert float((base - b).abs().max()) > 1e-3


def test_trunk_reports_its_parameter_counts():
    """FAILS WITHOUT THE FEATURE. The PI asked for the numbers; the module has
    to be able to state them. MEASURED 2026-09-16: backbone 21,284,672."""
    t = _trunk(frames=3, mode="shared")
    assert t.trunk_param_count() == 21_284_672
    assert t.param_count() > t.trunk_param_count()   # + the two fusions
    p = t.provenance()
    assert p["frames"] == 3 and p["mode"] == "shared"
    assert p["s16"] == [256, 16, 64] and p["s32"] == [512, 8, 32]


def test_build_timm_trunk_DERIVES_K_from_the_stack_width():
    """⛔ CONTROL. A trunk built for a different K than the corpus delivers
    must fail at BUILD time, not at the first forward."""
    assert tt.build_timm_trunk(in_channels=9).k == 3
    assert tt.build_timm_trunk(in_channels=3).k == 1
    with pytest.raises(ValueError, match="not 3 x K"):
        tt.build_timm_trunk(in_channels=7)


# ---------------------------------------------------------------------------
# 6. The encoder factory and the REF-C contract
# ---------------------------------------------------------------------------
def test_refc_encoder_factory_defaults_to_the_LEGACY_trunk():
    """⛔ CONTROL. `trunk='refc'` must return exactly the in-repo encoder, or
    every existing config silently changes model."""
    cfg = refc.CNNEncoderConfig(in_channels=9, image_size=64, base_width=8,
                                blocks=(1, 1, 1, 1))
    assert isinstance(refc.build_encoder(cfg), refc.ResNetEncoder)
    with pytest.raises(ValueError, match="not in"):
        refc.build_encoder(refc.CNNEncoderConfig(trunk="resnet34"))


def test_timm_trunk_honours_the_REFC_ENCODER_CONTRACT():
    """FAILS WITHOUT THE FEATURE: `(fmap, pooled)` at stride 32, with
    `feat_dim` and `grid_shape`, so it is a drop-in for `ResNetEncoder`."""
    cfg = refc.CNNEncoderConfig(in_channels=9, image_size=256,
                                image_width=1024, trunk="timm",
                                trunk_name="resnet34.a1_in1k")
    enc = refc.build_encoder(cfg).eval()
    fmap, pooled = enc(torch.rand(1, 9, *HW))
    assert tuple(fmap.shape) == (1, 512, 8, 32)
    assert tuple(pooled.shape) == (1, 512)
    assert enc.feat_dim == cfg.feat_dim == 512
    assert enc.grid_shape == cfg.grid_shape == (8, 32)
    # ...and the perception map is reachable without a second trunk pass
    assert tuple(enc.last_s16.shape) == (1, 256, 16, 64)


# ---------------------------------------------------------------------------
# 7. The DiffusionDrive optimiser (`rl_config.py:119-131`)
# ---------------------------------------------------------------------------
class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x):                                  # pragma: no cover
        return self.head(self.encoder(x))


def test_param_groups_carry_the_INTENDED_learning_rates():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. DD's `cfg_lr_mult` is 0.5 on
    the image encoder (`rl_config.py:124-131`) and `weight_decay` is 1e-4
    (`:120`) — NOT torch AdamW's 1e-2 default."""
    m = _Toy()
    g = tt.param_groups_dd(m, lr=6e-4)
    assert [x["name"] for x in g] == ["encoder", "head"]
    assert g[0]["lr"] == pytest.approx(3e-4)
    assert g[1]["lr"] == pytest.approx(6e-4)
    assert g[0]["weight_decay"] == g[1]["weight_decay"] == 1e-4
    # the partition is EXHAUSTIVE and DISJOINT — a bug here does not crash,
    # it trains part of the model at the wrong rate
    assert len(g[0]["params"]) == 4 and len(g[1]["params"]) == 2
    ids = [id(p) for grp in g for p in grp["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for _ in m.parameters())
    opt = torch.optim.AdamW(g, lr=6e-4)
    assert [pg["lr"] for pg in opt.param_groups] == [3e-4, 6e-4]


def test_param_groups_REFUSE_an_empty_encoder_group():
    """⛔ DELIBERATE REGRESSION. A misspelled prefix puts everything in the
    head group and the 0.5x applies to nothing — the DD recipe in name only.
    That must raise, not pass silently."""
    with pytest.raises(ValueError, match="no parameters under"):
        tt.param_groups_dd(_Toy(), lr=1e-4, encoder_attr="image_encoder")


def test_default_optimiser_is_BIT_IDENTICAL_to_plain_Adam():
    """⛔ CONTROL. `--opt adam` must be exactly the line it replaced, or every
    banked arm silently changes optimiser."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import refc_v3_train as T                                 # noqa: E402
    m = _Toy()
    args = type("A", (), {"opt": "adam", "lr": 1e-4})()
    got = T.build_optimizer(m, args)
    want = torch.optim.Adam(m.parameters(), lr=1e-4)
    assert type(got) is type(want)
    assert len(got.param_groups) == len(want.param_groups) == 1
    for k in ("lr", "weight_decay", "betas", "eps"):
        assert got.param_groups[0][k] == want.param_groups[0][k]


# ---------------------------------------------------------------------------
# 8. §2b — EGO HISTORY, and the admissibility boundary
# ---------------------------------------------------------------------------
def test_ego_encoder_CANNOT_READ_THE_FUTURE():
    """⛔⛔ THE ADMISSIBILITY TEST, AND IT IS A MUTATION. The binding PI ruling
    of 2026-09-02 makes measured PAST ego legal and ego FUTURE not. Mutating
    every step at or after `n_past` must leave the output BIT-identical; if a
    future index is ever read, this goes red."""
    torch.manual_seed(0)
    enc = eh.EgoHistoryEncoder(eh.EgoHistoryConfig(enable=True, out_dim=8,
                                                   zero_init_out=False)).eval()
    seq = torch.randn(3, 20, eh.EGO_CHANNELS)
    n_past = 8
    with torch.no_grad():
        a = enc(seq, n_past)
        poisoned = seq.clone()
        poisoned[:, n_past:] = 1e6           # ⛔ the future, made unmissable
        b = enc(poisoned, n_past)
    assert torch.equal(a, b)
    # ...and the PAST genuinely matters, or the test above is vacuous
    with torch.no_grad():
        moved = seq.clone()
        moved[:, n_past - 1] += 1.0
        assert not torch.equal(a, enc(moved, n_past))


def test_ego_encoder_reading_the_future_GOES_RED():
    """⛔ DELIBERATE REGRESSION — the MUTATION that proves the test above is
    alive. A subclass that forgets to slice must fail the same assertion."""
    class Leaky(eh.EgoHistoryEncoder):
        def forward(self, seq, n_past=None):               # ⛔ no slice
            return super().forward(seq, None)

    torch.manual_seed(0)
    enc = Leaky(eh.EgoHistoryConfig(enable=True, out_dim=8,
                                    zero_init_out=False)).eval()
    seq = torch.randn(2, 20, eh.EGO_CHANNELS)
    poisoned = seq.clone()
    poisoned[:, 8:] = 1e6
    with torch.no_grad():
        assert not torch.equal(enc(seq, 8), enc(poisoned, 8))


def test_ego_channels_are_BACKWARD_differences():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. A CENTRED difference at the
    last past step would read the first FUTURE sample. Step 0 gets a zero
    rate rather than borrowing step 1's."""
    poses = torch.zeros(1, 6, 4)
    poses[0, :, 3] = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0, 99.0])   # v
    poses[0, :, 2] = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0, 9.0])     # yaw
    ch = eh.ego_channels_from_poses(poses, n_past=4, dt=0.1)
    assert tuple(ch.shape) == (1, 4, 3)
    assert torch.allclose(ch[0, :, 0], torch.tensor([0.0, 1.0, 3.0, 6.0]))
    assert float(ch[0, 0, 1]) == 0.0                     # no predecessor
    assert torch.allclose(ch[0, 1:, 1], torch.tensor([10.0, 20.0, 30.0]))
    # the future entries (99.0 / 9.0) must not appear anywhere
    assert float(ch.abs().max()) < 100.0


def test_ego_yaw_rate_WRAPS_at_pi():
    """⛔ CONTROL. A heading crossing +-pi would otherwise read ~62.8 rad/s at
    dt 0.1 — an impossible number a network will happily fit."""
    import math
    poses = torch.zeros(1, 2, 4)
    poses[0, 0, 2] = math.pi - 0.05
    poses[0, 1, 2] = -math.pi + 0.05
    ch = eh.ego_channels_from_poses(poses, n_past=2, dt=0.1)
    assert abs(float(ch[0, 1, 2]) - 1.0) < 1e-4


def test_ego_history_is_a_REMOVABLE_graft_at_step_zero():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. `ego_to_cond` is zero-init, so
    a freshly built ego-history model emits exactly what the same model
    without the input emits. Any later difference is LEARNED."""
    from tanitad.models.ego_history import EgoHistoryConfig
    torch.manual_seed(3)
    base = refc.RefCModel(refc.refc_smoke_config()).eval()
    torch.manual_seed(3)
    cfg = refc.refc_smoke_config()
    cfg.ego_history = EgoHistoryConfig(enable=True, steps=cfg.window,
                                       out_dim=8, zero_init_out=False)
    with_eh = refc.RefCModel(cfg).eval()
    with_eh.load_state_dict(base.state_dict(), strict=False)
    h, w = cfg.encoder.image_hw()
    frames = torch.rand(2, cfg.window, cfg.encoder.in_channels, h, w)
    poses = torch.randn(2, cfg.window, 4)
    with torch.no_grad():
        a = base(frames, v0=torch.tensor([8.0, 12.0]))["traj"]
        b = with_eh(frames, v0=torch.tensor([8.0, 12.0]),
                    ego_poses=poses, ego_n_past=cfg.window)["traj"]
    assert torch.allclose(a, b, atol=1e-6)


def test_model_REFUSES_a_built_ego_encoder_with_no_window():
    """⛔ CONTROL. A stamped input that reaches nothing is the refcv5
    false-provenance defect. It must raise, not quietly skip."""
    from tanitad.models.ego_history import EgoHistoryConfig
    cfg = refc.refc_smoke_config()
    cfg.ego_history = EgoHistoryConfig(enable=True, steps=cfg.window,
                                       out_dim=8)
    m = refc.RefCModel(cfg).eval()
    h, w = cfg.encoder.image_hw()
    frames = torch.rand(1, cfg.window, cfg.encoder.in_channels, h, w)
    with pytest.raises(ValueError, match="no `ego_poses`"):
        m(frames, v0=torch.tensor([5.0]))


def test_set_ego_window_is_ONE_SHOT():
    """⛔ CONTROL. The one-shot channel exists because `refc_v3` cannot forward
    the kwarg. It must be CONSUMED, so a second forward cannot silently
    re-encode a previous batch's window into this batch's condition."""
    from tanitad.models.ego_history import EgoHistoryConfig
    cfg = refc.refc_smoke_config()
    cfg.ego_history = EgoHistoryConfig(enable=True, steps=cfg.window,
                                       out_dim=8)
    m = refc.RefCModel(cfg).eval()
    h, w = cfg.encoder.image_hw()
    frames = torch.rand(1, cfg.window, cfg.encoder.in_channels, h, w)
    m.set_ego_window(torch.randn(1, cfg.window, 4), cfg.window)
    with torch.no_grad():
        m(frames, v0=torch.tensor([5.0]))
    with pytest.raises(ValueError, match="no `ego_poses`"):
        m(frames, v0=torch.tensor([5.0]))


def test_ego_history_builds_NOTHING_by_default():
    """⛔ CONTROL. `cfg.ego_history is None` must construct no module and no
    projection, or the parameter count and the RNG draw order change for every
    existing arm."""
    m = refc.RefCModel(refc.refc_smoke_config())
    assert m.ego_hist is None
    assert m.decoder.ego_to_cond is None
