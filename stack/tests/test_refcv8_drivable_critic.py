"""refcv8 drivable-area critic (`--r8-critic-drivable`, MM 2026-10-04 after D6 P1': on 1,140 navhard DAC-zero scenes the
117-candidate fan holds a fully clean plan 89 % of the time but refcv7's imitation-only selector picks one 6.5 %).

The critic scores each candidate's NavSim-DAC footprint (nuPlan's ego box at every plan pose, `pdm_proxy` geometry):
- LABEL (training): the SAM3 10 cm target -- 1 iff no in-range SEEN corner lies on a non-drivable code;
- FEATURE (inference, vision only -> NavSim-legal): P(drivable) from the model's OWN 10 cm map head, DETACHED;
- its selection term `w * logsigmoid(logit)` with `w` ZERO-INIT: step 0 is unchanged.

Pinned:
1. KNOWN VALUES: a straight path in a 6 m corridor = 1; the same path shifted 3 m left (the box reaches y = 4.15 m) = 0;
   road markings count as drivable, the road edge / sidewalk / no-class do not; unseen or out-of-range corners carry no
   evidence (w = 0 when none is seen);
2. the feature from map logits: ~1 in the corridor, ~0 shifted, no gradient into the map head;
3. STEP-0 IDENTITY on the rig (zero-init w) and its DELIBERATE REGRESSION (w moved -> the scores move);
4. the BCE trains the critic, never the map head;
5. the regression arm `--r8-roll-targets map`: the label follows the ROLLED map (another window's);
6. argv rules, the effective-weight gate, G-LIVE, G-DVB.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

H, W = 600, 320                                       # 60 m ahead x +-16 m at 0.1 m (the /2 grid)


def corridor(half_m=3.0, code=1, fill=0):
    """[1, H, W] codes: `code` where |y| <= half_m, `fill` elsewhere; col 0 = y -16 m (RIGHT)."""
    c = torch.full((1, H, W), fill, dtype=torch.uint8)
    y = (torch.arange(W, dtype=torch.float32) + 0.5) * 0.1 - 16.0
    c[:, :, y.abs() <= half_m] = code
    return c


def straight(y=0.0, xs=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)):
    return torch.tensor([[[[x, y] for x in xs]]], dtype=torch.float32)          # [1, 1, 8, 2]


# =========================================================================== #
# 1. known values                                                              #
# =========================================================================== #
def test_KNOWN_VALUES_a_straight_corridor_is_1_and_a_3m_shift_into_non_drivable_is_0():
    m = corridor()
    y0, w0 = C.drivable_target(straight(0.0), m)
    y3, w3 = C.drivable_target(straight(3.0), m)
    assert (float(y0), float(w0)) == (1.0, 1.0)
    assert (float(y3), float(w3)) == (0.0, 1.0)                   # the box's left corners reach y = 3 + 1.1485 m


def test_markings_count_as_drivable_and_edge_sidewalk_nocls_do_not():
    for code, want in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (6, 1.0), (5, 0.0), (7, 0.0), (0, 0.0)):
        m = torch.full((1, H, W), code, dtype=torch.uint8)
        assert float(C.drivable_target(straight(0.0), m)[0]) == want, code


def test_unseen_and_out_of_range_corners_carry_no_evidence():
    y, w = C.drivable_target(straight(0.0), torch.full((1, H, W), 255, dtype=torch.uint8))
    assert float(w) == 0.0                                        # nothing seen -> no label
    far = straight(0.0, xs=(70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 99.0, 99.5))
    y, w = C.drivable_target(far, corridor(fill=0))
    assert float(w) == 0.0                                        # beyond the 60 m grid
    _y, _w = C.drivable_target(straight(0.0), corridor(), has_map=torch.tensor([False]))
    assert float(_w) == 0.0                                       # a window without a map target


def test_the_footprint_is_the_rear_axle_box_of_pdm_proxy():
    from tanitad.rl import pdm_proxy as pp
    c = C.footprint_corners(straight(0.0))[0, 0, 0]               # pose (5, 0), heading 0
    xs = sorted(float(v) for v in c[:, 0])
    ys = sorted(float(v) for v in c[:, 1])
    centre = 5.0 + pp.PROXY.rear_axle_to_center
    assert xs[0] == pytest.approx(centre - pp.PROXY.ego_length / 2, abs=1e-5)
    assert xs[-1] == pytest.approx(centre + pp.PROXY.ego_length / 2, abs=1e-5)
    assert ys[0] == pytest.approx(-pp.PROXY.ego_width / 2, abs=1e-5) and ys[-1] == pytest.approx(pp.PROXY.ego_width / 2,
                                                                                                abs=1e-5)


# =========================================================================== #
# 2. the feature from the map head                                             #
# =========================================================================== #
def _logits_from_codes(codes, sharp=20.0):
    lg = torch.full((codes.shape[0], 8, H, W), -sharp)
    lg.scatter_(1, codes.long().clamp_max(7)[:, None], sharp)
    return lg.requires_grad_()


def test_the_feature_reads_the_map_head_and_is_detached():
    lg = _logits_from_codes(corridor())
    f = C.drivable_feature(torch.cat([straight(0.0), straight(3.0)], 1), lg)
    assert f.shape == (1, 2, 3) and not f.requires_grad
    assert float(f[0, 0, 0]) > 0.99 and float(f[0, 1, 0]) < 0.01           # min P(drivable)
    assert float(f[0, 0, 2]) == 1.0                                           # every corner in range


# =========================================================================== #
# 3 + 4. the rig: step-0 identity, its regression, the BCE                      #
# =========================================================================== #
@pytest.fixture(scope="module")
def rigs():
    pytest.importorskip("timm")
    T = R.trainer()
    cfgA, mA = R.build(T, True, n_alloc=4)
    cfgB, mB = R.build(T, True, n_alloc=4, critic_drivable=True, w_drivable=1.0)
    new = R.copy_into(mA, mB)
    assert new and all(k.startswith("r8_drv.") for k in new), new
    return T, (cfgA, mA), (cfgB, mB)


def _with_map(m, codes):
    """Inject the 10 cm map logits where the core's perception output carries them (the rig has no map head)."""
    orig = m.core.forward

    def fwd(*a, **k):
        out = orig(*a, **k)
        out["perception"] = dict(out.get("perception") or {}, map_hires_logits=_logits_from_codes(codes.expand(
            out["anchor_traj"].shape[0], H, W)).detach())
        return out
    m.core.forward = fwd
    return orig


def _run(cfg, m, bt, train=False):
    orig = _with_map(m, corridor())
    try:
        return R.forward(cfg, m, bt, seed=5, train=train)
    finally:
        del m.core.forward


def test_STEP0_the_zero_init_critic_leaves_everything_unchanged_and_its_regression_goes_RED(rigs):
    T, (cA, mA), (cB, mB) = rigs
    bt = R.batch(cA, b=3, seed=2)
    a = _run(cA, mA, bt)
    b = _run(cB, mB, bt)
    for k in ("traj", "sel_idx", "anchor_traj", "sel_score_v3"):
        assert torch.equal(a[k], b[k]), k
    assert "r8_drv_logit" in b and b["r8_drv_feat"].shape[-1] == 3
    with torch.no_grad():
        mB.r8_drv.w.fill_(0.5)
    try:
        b2 = _run(cB, mB, bt)
        assert not torch.equal(a["sel_score_v3"], b2["sel_score_v3"])          # a live w moves the scores
    finally:
        with torch.no_grad():
            mB.r8_drv.w.zero_()


def test_the_BCE_trains_the_critic_through_r8_losses(rigs):
    T, (cA, mA), (cB, mB) = rigs
    bt = R.batch(cB, b=3, seed=3)
    b = 3
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    mB.train()
    mB._r8_rc_dropout, mB._r8_rc_noise, mB._r8_nav_args_dropout = 0.3, (2.0, 0.75), 0.5
    batch = {"map_fine": corridor().expand(b, H, W).clone(), "map_fine_label": torch.ones(b, dtype=torch.bool)}
    prep = RT.r8_before_forward(mB, batch, "cpu", traj, torch.ones(b, 8, dtype=torch.bool), bt["pose_hist"][:, -1],
                                torch.zeros(b, 60, 4), torch.ones(b, 60, dtype=torch.bool), bt["v0"])
    _with_map(mB, corridor())
    try:
        out = R.forward(cB, mB, bt, seed=5, train=True, **prep["fwd"])
    finally:
        del mB.core.forward
    mB.zero_grad()
    total, tele = RT.r8_losses(mB, out, prep, traj, torch.ones(b, 8, dtype=torch.bool))
    assert "r8_drv" in tele and tele["n_drv"] > 0
    ld, _ = C.drivable_bce(out["r8_drv_logit"], *C.drivable_target(out["anchor_traj"].detach(), batch["map_fine"]))
    ld.backward()
    assert float(mB.r8_drv.mlp[0].weight.grad.abs().sum()) > 0.0
    prep_none = dict(prep, drv_map=None)
    with pytest.raises(SystemExit, match="no 10 cm map target"):
        RT.r8_losses(mB, out, prep_none, traj, torch.ones(b, 8, dtype=torch.bool))


# =========================================================================== #
# 5. the map-roll regression arm                                               #
# =========================================================================== #
def test_under_the_map_roll_the_label_is_another_windows():
    left = corridor()
    right = torch.zeros((1, H, W), dtype=torch.uint8)             # a window whose map has NO drivable cell
    batch = {"map_fine": torch.cat([left, right]), "map_fine_label": torch.ones(2, dtype=torch.bool)}
    paths = torch.cat([straight(0.0), straight(0.0)], 0)          # [2, 1, 8, 2]
    y, _ = C.drivable_target(paths, batch["map_fine"])
    rolled = RT.roll_targets(batch, "map")
    yr, _ = C.drivable_target(paths, rolled["map_fine"])
    assert y.flatten().tolist() == [1.0, 0.0] and yr.flatten().tolist() == [0.0, 1.0]


# =========================================================================== #
# 6. argv, gates                                                               #
# =========================================================================== #
ON = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05"]


def _pin(argv):
    tr = R.trainer()
    args = tr.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=C.R8Config())
    tr._pin_refcv8(cfg, args)
    return cfg.refcv8


def test_the_argv_rules_and_the_gates():
    r = _pin(ON + ["--map-hires", "on", "--r8-critic-drivable", "--w-r8-drivable", "0.1"])
    assert r.critic_drivable is True and r.w_drivable == 0.1
    for argv, needle in ((ON + ["--r8-critic-drivable", "--w-r8-drivable", "0.1"], "--map-hires on"),
                         (ON + ["--map-hires", "on", "--r8-critic-drivable"], "never supervised"),
                         (ON + ["--w-r8-drivable", "0.1"], "no critic"),
                         (["--arm", "hier", "--out", "X", "--r8-critic-drivable"], "without --refcv8")):
        with pytest.raises(SystemExit) as e:
            _pin(argv)
        assert needle in str(e.value), str(e.value)
    tr = R.trainer()
    assert "w_r8_drivable" in tr.REFC_WEIGHT_GATES
    assert {a.dest: a.default for a in tr.build_parser()._actions}["w_r8_drivable"] == 0.0
    import launch_gate as LG
    assert "w_r8_drivable" in LG.LIVE_WEIGHT_RULES and LG.LIVE_WEIGHT_RULES["w_r8_drivable"][1] == ("r8_drv",)
    from tanitad.train import declared_vs_built as dvb
    assert dvb.REGISTRY["r8_critic_drivable"].kind == "built" and dvb.REGISTRY["w_r8_drivable"].kind == "loss"
